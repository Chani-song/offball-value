#!/usr/bin/env python3
"""Could a StatsBomb 360 sequence recover the arrival race after all?

The claim that xpass360 cannot see the race rests on its freeze frames being
static. They are not quite: consecutive frames sit a median 1.17 s apart and
40 % of them under 1 s, so a velocity could in principle be differenced out of
two of them. What the frames lack is IDENTITY -- a freeze-frame entry carries
only teammate / actor / keeper and a location -- so the same player has to be
re-found by proximity, and at 1.17 s a sprinter covers 8 m while defensive
lines sit 5-10 m apart. The matching should therefore fail hardest exactly
where the signal lives.

That is a measurable claim, and Bundesliga tracking can settle it: it has the
true velocities and the true identities. So degrade it into the StatsBomb
representation -- two snapshots a realistic gap apart, identities discarded,
only the players a camera would have seen -- recover velocities by nearest
neighbour within team, and compare the receiver_first that results against the
one computed from real 25 fps tracking.

Usage:
    python scripts/test_statsbomb_velocity_recovery.py \
        --output out/delivery_analysis/velocity_recovery.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.observed_passes import infer_intended_target, resolve_pass_target
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    estimate_frame_velocities,
    point_reception_estimate,
)

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
HISTORY_FRAMES = 10
FORWARD_FRAMES = 75
FORWARD_STRIDE = 2
GAPS_S = [0.4, 0.8, 1.2, 2.0]      # 1.2 s is StatsBomb's measured median
VISIBLE_COUNT = 18                 # median players per 360 freeze frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def degraded_velocities(now, past, ball_xy, gap_s, visible_count=VISIBLE_COUNT):
    """Velocities as a 360 sequence would yield them: no ids, camera-limited.

    Matching is greedy nearest-neighbour WITHIN team, which is the best a
    consumer of freeze frames could do -- the teammate flag is the one piece of
    identity StatsBomb does provide.
    """
    def visible(frame):
        players = sorted(
            frame.players.items(),
            key=lambda kv: math.hypot(kv[1].x - ball_xy[0], kv[1].y - ball_xy[1]),
        )
        return players[:visible_count]

    now_v, past_v = visible(now), visible(past)
    out: dict[str, VelocityEstimate] = {}
    by_team: dict[str, list] = {}
    for pid, st in past_v:
        by_team.setdefault(str(st.team_id), []).append((pid, st))

    for pid, st in now_v:
        pool = by_team.get(str(st.team_id)) or []
        if not pool:
            out[pid] = VelocityEstimate(0.0, 0.0, 0.0, 2, gap_s)
            continue
        j = min(range(len(pool)),
                key=lambda k: math.hypot(pool[k][1].x - st.x, pool[k][1].y - st.y))
        _matched_id, matched = pool.pop(j)
        vx = (st.x - matched.x) / gap_s
        vy = (st.y - matched.y) / gap_s
        out[pid] = VelocityEstimate(vx, vy, math.hypot(vx, vy), 2, gap_s)
    return out


def main() -> None:
    args = parse_args()
    config = ArrivalModelConfig()
    match_ids = [m for m in list_bundesliga_match_ids(args.raw_dir)
                 if m not in EXCLUDED_MATCHES]
    rows = []
    for match_id in match_ids:
        files = find_bundesliga_files(args.raw_dir, match_id)
        load_bundesliga_match_metadata(files["matchinfo"])
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
        back = max(int(round(g * FPS)) for g in GAPS_S)
        for fid in passes["frame_id"].astype(int):
            needed.update(range(fid - max(HISTORY_FRAMES, back), fid + 1))
            needed.update(range(fid, fid + FORWARD_FRAMES + 1, FORWARD_STRIDE))
        frames = load_bundesliga_frames(files["positions"], needed)

        for _, ev in passes.iterrows():
            fid = int(ev["frame_id"])
            frame = frames.get(fid)
            if frame is None or frame.ball is None:
                continue
            passer, team = str(ev["player_id"]), str(ev["team_id"])
            if passer not in frame.players:
                continue
            window = [frames[f] for f in range(fid - HISTORY_FRAMES, fid + 1) if f in frames]
            if len(window) < 3:
                continue
            intended = infer_intended_target(frames, fid, passer, team)
            observed = resolve_pass_target(frames, fid, passer, team,
                                           forward_frames=FORWARD_FRAMES,
                                           stride=FORWARD_STRIDE)
            if intended is None or observed is None or observed.travel_distance_m < 3.0:
                continue
            ball = (float(frame.ball.x), float(frame.ball.y))
            try:
                true_v = estimate_frame_velocities(window, fid, config)
                true_est = point_reception_estimate(
                    frame, true_v, intended.receiver_id, team, ball,
                    intended.target_xy, config, passer_id=passer)
            except (ValueError, KeyError):
                continue
            row = {"label": 1.0 if observed.completed else 0.0,
                   "true_rf": float(true_est.receiver_first_probability),
                   "true_mech": float(np.clip(true_est.receive_probability, 0.0, 1.0)),
                   "degraded": {}}
            for gap in GAPS_S:
                past = frames.get(fid - int(round(gap * FPS)))
                if past is None or intended.receiver_id not in past.players:
                    continue
                try:
                    dv = degraded_velocities(frame, past, ball, gap)
                    est = point_reception_estimate(
                        frame, dv, intended.receiver_id, team, ball,
                        intended.target_xy, config, passer_id=passer)
                except (ValueError, KeyError):
                    continue
                row["degraded"][gap] = {
                    "rf": float(est.receiver_first_probability),
                    "mech": float(np.clip(est.receive_probability, 0.0, 1.0))}
            if row["degraded"]:
                rows.append(row)
        print(f"  [{match_id}] 누적 {len(rows):,}", flush=True)
        del frames

    label = np.array([r["label"] for r in rows])
    true_rf = np.array([r["true_rf"] for r in rows])
    true_mech = np.array([r["true_mech"] for r in rows])

    def auc(pred, lab):
        pos, neg = pred[lab > 0.5], pred[lab <= 0.5]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(order.size, float)
        ranks[order] = np.arange(1, order.size + 1)
        return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                     / (pos.size * neg.size))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n{'='*84}")
    print("360 방식으로 망가뜨린 속도로 receiver_first 를 얼마나 복원하나")
    print("="*84)
    print(f"  패스 {len(rows):,}개 · 실제 성공률 {label.mean():.3f}")
    print(f"  25fps 진짜 속도 기준: receiver_first AUC {auc(true_rf, label):.3f}"
          f" · 역학 AUC {auc(true_mech, label):.3f}")
    print(f"\n  {'스냅샷 간격':<14}{'n':>8}{'rf 상관':>10}{'rf 순위상관':>12}"
          f"{'rf AUC':>9}{'역학 AUC':>10}")
    out = {"passes": len(rows), "true_rf_auc": auc(true_rf, label),
           "true_mech_auc": auc(true_mech, label), "gaps": {}}
    for gap in GAPS_S:
        sel = [r for r in rows if gap in r["degraded"]]
        if len(sel) < 50:
            continue
        t = np.array([r["true_rf"] for r in sel])
        d = np.array([r["degraded"][gap]["rf"] for r in sel])
        dm = np.array([r["degraded"][gap]["mech"] for r in sel])
        lb = np.array([r["label"] for r in sel])
        rec = {"n": len(sel), "pearson": float(np.corrcoef(t, d)[0, 1]),
               "spearman": spearman(t, d), "rf_auc": auc(d, lb),
               "mech_auc": auc(dm, lb)}
        out["gaps"][gap] = rec
        mark = "  <- StatsBomb 중앙값" if abs(gap - 1.2) < 1e-9 else ""
        print(f"  {gap:>5.1f} s{'':<7}{len(sel):>8,}{rec['pearson']:>10.3f}"
              f"{rec['spearman']:>12.3f}{rec['rf_auc']:>9.3f}{rec['mech_auc']:>10.3f}{mark}")
    print("\n  (AUC 0.5 = 무작위. 진짜 속도 대비 얼마나 떨어지는지가 핵심)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
