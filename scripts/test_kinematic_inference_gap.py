#!/usr/bin/env python3
"""Is x360kin's optimism a train/inference mismatch, and does closing it help?

x360kin sits above the observed completion rate in all seven length bands --
0.871 where real 30-40 m passes complete 0.676 -- while the same model is
within 0.005 of observed in every band on real StatsBomb passes. A model that
calibrates perfectly on its own data and overshoots on ours is being fed
something it did not train on.

The suspect is velocity. Training differenced two freeze frames a median
0.97 s apart, re-matching identities by proximity within side, and saw only
the players a camera captured. The pipeline hands it 25 fps tracking with real
identities and every player on the pitch: cleaner, and therefore unfamiliar.

This runs x360kin on real Bundesliga passes under four input regimes, from
what the pipeline does now to what the fit actually saw, and scores each
against the observed outcome. If the optimism is the mismatch, it should fall
away as the inputs move toward the training convention.

Usage:
    python scripts/test_kinematic_inference_gap.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "calibrate_hybrid_delivery", ROOT / "scripts" / "calibrate_hybrid_delivery.py"
)
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)

from offball_value.bundesliga import (  # noqa: E402
    FPS,
    find_bundesliga_files,
    infer_attacking_direction,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.kinematic_xpass import TrackedPlayer  # noqa: E402
from offball_value.observed_passes import (  # noqa: E402
    infer_intended_target,
    resolve_pass_target,
)
from offball_value.pass_dynamics import (  # noqa: E402
    ArrivalModelConfig,
    estimate_frame_velocities,
)
from offball_value.xpass import (  # noqa: E402
    load_xpass_model,
    predict_pass_success_360,
    predict_pass_success_360_kinematic,
)

TRAIN_GAP_S = 0.97          # median spacing between consecutive 360 freeze frames
VISIBLE_COUNT = 18          # median players in a freeze frame
HISTORY_FRAMES = 10
FORWARD_FRAMES = 75
FORWARD_STRIDE = 2
BANDS = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 200)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path,
                   default=Path("out/delivery_analysis/kinematic_inference_gap.json"))
    return p.parse_args()


def tracked_from(frame, velocities, team, carrier, keepers, ball,
                 limit=None):
    players = list(frame.players.items())
    if limit is not None:
        players.sort(key=lambda kv: math.hypot(kv[1].x - ball[0], kv[1].y - ball[1]))
        players = players[:limit]
    out = []
    for pid, p in players:
        v = velocities.get(pid)
        out.append(TrackedPlayer(
            float(p.x), float(p.y),
            float(v.vx) if v is not None else 0.0,
            float(v.vy) if v is not None else 0.0,
            teammate=str(p.team_id) == team,
            actor=pid == carrier,
            keeper=pid in keepers,
        ))
    return out


def gap_velocities(frame, past, gap_s, team_of):
    """Velocities as a 360 frame pair yields them: no ids, matched within side."""
    pools: dict[str, list] = {}
    for pid, p in past.players.items():
        pools.setdefault(str(p.team_id), []).append((float(p.x), float(p.y)))
    out = {}
    for pid, p in frame.players.items():
        pool = pools.get(str(p.team_id)) or []
        if not pool:
            out[pid] = (0.0, 0.0)
            continue
        j = min(range(len(pool)),
                key=lambda k: math.hypot(pool[k][0] - p.x, pool[k][1] - p.y))
        was = pool.pop(j)
        out[pid] = ((p.x - was[0]) / gap_s, (p.y - was[1]) / gap_s)
    del team_of
    return out


class _V:
    __slots__ = ("vx", "vy")

    def __init__(self, vx, vy):
        self.vx, self.vy = vx, vy


def main() -> None:
    args = parse_args()
    config = ArrivalModelConfig()
    kin_model = load_xpass_model(
        Path("data/processed/xpass_360_kinematic/xpass_360_kinematic_gbdt.joblib"))
    x360_model = load_xpass_model(
        Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"))
    back = int(round(TRAIN_GAP_S * FPS))

    rows = []
    for match_id in [m for m in _cal.list_bundesliga_match_ids(args.raw_dir)
                     if m not in _cal.EXCLUDED_MATCHES]:
        files = find_bundesliga_files(args.raw_dir, match_id)
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
            continue
        needed = set()
        for fid in passes["frame_id"].astype(int):
            needed.update(range(fid - max(HISTORY_FRAMES, back), fid + 1))
            needed.update(range(fid, fid + FORWARD_FRAMES + 1, FORWARD_STRIDE))
        frames = load_bundesliga_frames(files["positions"], needed)
        keepers = {k for t in (meta.home_team_id, meta.away_team_id)
                   if (k := meta.goalkeeper_id(t)) is not None}

        for _, ev in passes.iterrows():
            fid = int(ev["frame_id"])
            frame, past = frames.get(fid), frames.get(fid - back)
            if frame is None or past is None or frame.ball is None:
                continue
            passer, team = str(ev["player_id"]), str(ev["team_id"])
            if passer not in frame.players:
                continue
            window = [frames[f] for f in range(fid - HISTORY_FRAMES, fid + 1)
                      if f in frames]
            if len(window) < 3:
                continue
            intended = infer_intended_target(frames, fid, passer, team)
            observed = resolve_pass_target(frames, fid, passer, team,
                                           forward_frames=FORWARD_FRAMES,
                                           stride=FORWARD_STRIDE)
            if intended is None or observed is None or observed.travel_distance_m < 3.0:
                continue
            ball = (float(frame.ball.x), float(frame.ball.y))
            target = intended.target_xy
            direction = infer_attacking_direction(frame, team, meta)
            try:
                fast = estimate_frame_velocities(window, fid, config)
            except (ValueError, KeyError):
                continue
            slow_raw = gap_velocities(frame, past, TRAIN_GAP_S, None)
            slow = {pid: _V(v[0], v[1]) for pid, v in slow_raw.items()}

            arms = {}
            for name, vel, limit in (("25fps·전원", fast, None),
                                     ("0.97초·전원", slow, None),
                                     ("25fps·18명", fast, VISIBLE_COUNT),
                                     ("0.97초·18명", slow, VISIBLE_COUNT)):
                tracked = tracked_from(frame, vel, team, passer, keepers, ball, limit)
                try:
                    arms[name] = float(predict_pass_success_360_kinematic(
                        kin_model, ball, target, direction, tracked))
                except Exception:
                    arms = {}
                    break
            if not arms:
                continue
            opponents = [(float(p.x), float(p.y)) for pid, p in frame.players.items()
                         if str(p.team_id) != team and pid not in keepers]
            try:
                arms["xpass360 (참고)"] = float(predict_pass_success_360(
                    x360_model, ball, target, direction, opponents))
            except Exception:
                continue
            rows.append({"label": 1.0 if observed.completed else 0.0,
                         "distance_m": math.hypot(target[0] - ball[0],
                                                  target[1] - ball[1]),
                         **arms})
        print(f"  [{match_id}] 누적 {len(rows):,}", flush=True)
        del frames

    label = np.array([r["label"] for r in rows])
    distance = np.array([r["distance_m"] for r in rows])
    names = ["25fps·전원", "0.97초·전원", "25fps·18명", "0.97초·18명", "xpass360 (참고)"]

    print(f"\n{'='*92}")
    print("입력 방식별 x360kin 예측 — 실제 Bundesliga 패스")
    print("="*92)
    print(f"  패스 {len(rows):,}개 · 실제 성공률 {label.mean():.4f}\n")
    print(f"  {'입력 방식':<20}{'예측 평균':>11}{'예측 중앙':>11}{'Brier':>10}{'실제 대비':>11}")
    out = {"passes": len(rows), "completion": float(label.mean()), "arms": {}}
    for name in names:
        v = np.array([r[name] for r in rows])
        brier = float(np.mean((v - label) ** 2))
        out["arms"][name] = {"mean": float(v.mean()), "median": float(np.median(v)),
                             "brier": brier}
        print(f"  {name:<18}{v.mean():>11.3f}{np.median(v):>11.3f}{brier:>10.4f}"
              f"{v.mean()-label.mean():>+11.3f}")
    print(f"  {'(기저율 Brier)':<18}{'':>11}{'':>11}"
          f"{label.mean()*(1-label.mean()):>10.4f}")

    print(f"\n  길이별 예측 vs 실제")
    print(f"  {'길이(m)':<12}{'n':>7}{'실제':>8}" + "".join(f"{n[:11]:>12}" for n in names))
    out["bands"] = []
    for lo, hi in BANDS:
        m = (distance >= lo) & (distance < hi)
        if m.sum() < 30:
            continue
        line = f"  [{lo}, {hi})".ljust(12) + f"{m.sum():>7,}{label[m].mean():>8.3f}"
        rec = {"lo": lo, "hi": hi, "n": int(m.sum()), "observed": float(label[m].mean())}
        for name in names:
            v = np.array([r[name] for r in rows])[m]
            line += f"{v.mean():>12.3f}"
            rec[name] = float(v.mean())
        out["bands"].append(rec)
        print(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
