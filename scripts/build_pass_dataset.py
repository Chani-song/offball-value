#!/usr/bin/env python3
"""Real open-play passes from our Bundesliga tracking, for fitting pass models.

Written for the two candidate replacements of the solver's positions-only pass
model (2026-09-25): A, a physics arrival race with three fitted numbers, and
B1, 준현's own logistic refitted on his velocity feature set. Both are fitted
on these rows, so both see the same passes, labels and targets.

What each row holds, and why:

  target     the INTENDED target, from the ball's heading over the first 4
             frames of flight only (observed_passes.infer_intended_target).
             The actual receiver is never used to choose it. Using the
             reception point instead leaks the label -- a completed pass ends
             at a team-mate's feet by definition -- which is exactly what sank
             the 360 kinematic model on 2026-09-18.
  receiver   the team-mate that target was aimed at (same function).
  label      1 when the first player to control the ball is a team-mate,
             from tracking (observed_passes.resolve_pass_target), independent
             of the event file's own Evaluation.
  bodies     corner-origin positions (the solver's frame, 105 x 68) and
             velocities over the 0.4 s before the kick
             (pass_dynamics.estimate_frame_velocities), the same window the
             solver's start states use.
  ball_speed the ball's speed over the first 4 frames, for the physics
             model's ball-speed-versus-distance fit.

Kick frame. The event's frame is often not the kick: on one match the passer
stood a median 3.6 m from the ball there and within 1.5 m only 78 times in 193.
So the kick is re-found in tracking: within 1 s either side of the event
frame, the moments the ball goes from within 1.5 m of the passer (the
project's control distance) to 2 m or more (resolve_pass_target's release
distance); the one nearest the event frame is the kick, and its last
controlled frame is where the row is read. The 1 s window is a choice, not a
measurement; passes with no such moment are dropped and counted.

Selection as in scripts/train_kinematic_delivery.py: open-play passes with a
kick frame, a resolvable intended target and outcome, travelling at least
3 m. DFL-MAT-J03WN1 is excluded, as in every earlier pass-model script: a red
card at about 7 minutes leaves it outside the population.

Usage:
    python scripts/build_pass_dataset.py --output data/processed/pass_models/passes.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    infer_attacking_direction,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.observed_passes import infer_intended_target, resolve_pass_target
from offball_value.pass_dynamics import ArrivalModelConfig, estimate_frame_velocities

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
HISTORY_FRAMES = 10          # 0.4 s at 25 fps
FORWARD_FRAMES = 125         # 5 s, as train_kinematic_delivery.py
FORWARD_STRIDE = 2
BALL_SPEED_FRAMES = 4        # the frames infer_intended_target reads the heading from
SYNC_FRAMES = 25             # search 1 s either side of the event frame for the kick
CONTROL_M, RELEASE_M = 1.5, 2.0
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def match_ids(raw_dir: Path) -> list[str]:
    ids = sorted({p.name.split("_")[-1].removesuffix(".xml")
                  for p in raw_dir.glob("DFL_02_01_matchinformation_*.xml")})
    return [m for m in ids if m not in EXCLUDED_MATCHES]


def kick_frame(frames, event_fid: int, passer: str) -> int | None:
    """Last controlled frame of the release nearest the event frame, or None."""
    def gap(f):
        fr = frames.get(f)
        if fr is None or fr.ball is None or passer not in fr.players:
            return None
        p = fr.players[passer]
        return math.hypot(float(fr.ball.x) - float(p.x), float(fr.ball.y) - float(p.y))
    releases, last_controlled = [], None
    for f in range(event_fid - SYNC_FRAMES, event_fid + SYNC_FRAMES + 1):
        g = gap(f)
        if g is None:
            continue
        if g <= CONTROL_M:
            last_controlled = f
        elif g >= RELEASE_M and last_controlled is not None:
            releases.append(last_controlled)
            last_controlled = None
    if not releases:
        return None
    return min(releases, key=lambda f: abs(f - event_fid))


def collect(match_id: str, raw_dir: Path) -> list[dict]:
    config = ArrivalModelConfig()
    files = find_bundesliga_files(raw_dir, match_id)
    meta = load_bundesliga_match_metadata(files["matchinfo"])
    clock = load_bundesliga_frame_clock(files["positions"])
    events = load_bundesliga_events(files["events"], clock)
    passes = events[
        (events["event_type"] == "pass")
        & (events["from_open_play"] == True)  # noqa: E712
        & events["frame_id"].notna()
        & events["player_id"].notna() & events["team_id"].notna()
    ].copy()
    if passes.empty:
        return []
    needed = set()
    for fid in passes["frame_id"].astype(int):
        needed.update(range(fid - SYNC_FRAMES - HISTORY_FRAMES, fid + SYNC_FRAMES + BALL_SPEED_FRAMES + 1))
        needed.update(range(fid - SYNC_FRAMES, fid + SYNC_FRAMES + FORWARD_FRAMES + 1, FORWARD_STRIDE))
    frames = load_bundesliga_frames(files["positions"], needed)
    keepers = {k for t in (meta.home_team_id, meta.away_team_id)
               if (k := meta.goalkeeper_id(t)) is not None}

    rows, unsynced = [], 0
    for _, ev in passes.iterrows():
        event_fid = int(ev["frame_id"])
        fid = kick_frame(frames, event_fid, str(ev["player_id"]))
        if fid is None:
            unsynced += 1
            continue
        frame = frames.get(fid)
        later = frames.get(fid + BALL_SPEED_FRAMES)
        if frame is None or frame.ball is None or later is None or later.ball is None:
            continue
        passer, team = str(ev["player_id"]), str(ev["team_id"])
        if passer not in frame.players:
            continue
        window = [frames[f] for f in range(fid - HISTORY_FRAMES, fid + 1) if f in frames]
        if len(window) < 3:
            continue
        intended = infer_intended_target(frames, fid, passer, team)
        observed = resolve_pass_target(frames, fid, passer, team,
                                       forward_frames=FORWARD_FRAMES, stride=FORWARD_STRIDE)
        if intended is None or observed is None or observed.travel_distance_m < 3.0:
            continue
        if intended.receiver_id not in frame.players:
            continue
        try:
            velocities = estimate_frame_velocities(window, fid, config)
        except (ValueError, KeyError):
            continue

        def body(pid, p):
            v = velocities.get(pid)
            return [round(float(p.x) + HALF_LENGTH, 4), round(float(p.y) + HALF_WIDTH, 4),
                    round(float(v.vx) if v is not None else 0.0, 4),
                    round(float(v.vy) if v is not None else 0.0, 4)]

        ball0 = (float(frame.ball.x), float(frame.ball.y))
        ball_speed = math.hypot(float(later.ball.x) - ball0[0],
                                float(later.ball.y) - ball0[1]) / (BALL_SPEED_FRAMES / FPS)
        pb = body(passer, frame.players[passer])
        target = [intended.target_xy[0] + HALF_LENGTH, intended.target_xy[1] + HALF_WIDTH]
        rows.append({
            "match_id": match_id, "frame_id": fid,
            "label": 1 if observed.completed else 0,
            "attack_direction": int(infer_attacking_direction(frame, team, meta)),
            "passer": pb, "ball": [ball0[0] + HALF_LENGTH, ball0[1] + HALF_WIDTH],
            "receiver_id": intended.receiver_id,
            "receiver": body(intended.receiver_id, frame.players[intended.receiver_id]),
            "target": [round(target[0], 4), round(target[1], 4)],
            "opponents": [body(pid, p) + [int(pid in keepers)]
                          for pid, p in frame.players.items() if str(p.team_id) != team],
            "ball_speed": round(ball_speed, 3),
            "pass_distance": round(math.hypot(target[0] - pb[0], target[1] - pb[1]), 3),
            "travel_distance": round(observed.travel_distance_m, 3),
            "flight_time": round(observed.flight_time_s, 3),
            "actual_receiver_is_intended": observed.receiver_id == intended.receiver_id,
            "event_frame_id": event_fid,
        })
    print(f"  [{match_id}] 킥 순간을 못 찾아 제외 {unsynced}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    ids = match_ids(args.raw_dir)
    print("경기:", ids, flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for match_id, got in zip(ids, pool.map(collect, ids, [args.raw_dir] * len(ids))):
            rows.extend(got)
            done = sum(r["label"] for r in got)
            print(f"  [{match_id}] 패스 {len(got):,} · 성공 {done:,} ({done / max(len(got), 1):.3f})",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lab = sum(r["label"] for r in rows)
    print(f"합계 {len(rows):,} 패스 · 성공률 {lab / max(len(rows), 1):.4f} → {args.output}", flush=True)


if __name__ == "__main__":
    main()
