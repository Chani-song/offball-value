#!/usr/bin/env python3
"""What does Bundesliga tracking say about keeping the ball under pressure?

The carry (dribble) side of Q is priced by

    path_min_retention = min_t PROD_defenders logistic((arrival_t - 0.35) / 0.25)

flagged `uncalibrated_carry_retention`. Nobody has checked the constants, the
functional form, or the product-over-defenders assumption against data. This
does, from the same tracking the scenes are built on.

At every sampled instant with a determinate ball owner, it computes the model's
instantaneous retention exactly as _carry_retention_at_time does (same arrival
model, same goalkeeper exclusion, same logistic and constants) and records
whether the owning side actually still has the ball T seconds later. The two
put side by side give the calibration curve the carry model has never had.

WHAT THIS IS NOT: observed carries are selected -- players keep the ball where
they can, and nobody dribbles into the situations the option catalogue invents.
The same off-distribution caveat that applies to xPass on counterfactual
through balls applies here. This measures the model against play that happened,
which is a reference point, not a target.

Usage:
    python scripts/measure_carry_retention_truth.py \
        --raw-dir data/raw/bundesliga-integrated \
        --output out/delivery_analysis/carry_retention_truth.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    player_time_to_point,
)

ACTION_TIME_S = 0.35     # LocalGamePayoffConfig.carry_pressure_action_time_seconds
SIGMA_S = 0.25           # LocalGamePayoffConfig.carry_pressure_sigma_seconds
CONTROL_M = 1.5          # SettledPossessionPhaseConfig.control_distance_m
EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}   # red card at ~7 min, outside the population


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--stride", type=int, default=5, help="frames between samples (25 fps)")
    p.add_argument("--horizons", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def owner_of(frame, control_m: float):
    """(player_id, team_id) of the player in foot control, else (None, None)."""
    if frame is None or frame.ball is None or not frame.players:
        return None, None
    pid, state = min(
        frame.players.items(),
        key=lambda kv: math.hypot(kv[1].x - frame.ball.x, kv[1].y - frame.ball.y),
    )
    if math.hypot(state.x - frame.ball.x, state.y - frame.ball.y) <= control_m:
        return pid, state.team_id
    return None, None


def main() -> None:
    args = parse_args()
    arrival = ArrivalModelConfig()
    dt = args.stride / FPS

    rows = []           # (model_retention, nearest_arrival_s, n_close, {T: retained})
    n_frames = n_owned = 0
    match_ids = [m for m in list_bundesliga_match_ids(args.raw_dir)
                 if m not in EXCLUDED_MATCHES]
    if args.max_matches:
        match_ids = match_ids[: args.max_matches]
    print(f"경기 {len(match_ids)}개 (제외 {sorted(EXCLUDED_MATCHES)})", flush=True)

    for match_id in match_ids:
        files = find_bundesliga_files(args.raw_dir, match_id)
        meta = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        keepers = {
            k for t in (meta.home_team_id, meta.away_team_id)
            if (k := meta.goalkeeper_id(t)) is not None
        }
        targets = []
        for _section, (start_frame, _t) in clock.section_start.items():
            targets.extend(range(int(start_frame), int(start_frame) + 75_000, args.stride))
        frames = load_bundesliga_frames(files["positions"], targets)
        order = sorted(frames)
        index = {fid: i for i, fid in enumerate(order)}
        n_frames += len(order)

        for i, fid in enumerate(order):
            if i == 0:
                continue
            frame = frames[fid]
            prev = frames[order[i - 1]]
            if order[i - 1] != fid - args.stride:      # section boundary / gap
                continue
            pid, team = owner_of(frame, CONTROL_M)
            if pid is None:
                continue
            n_owned += 1
            ball = (float(frame.ball.x), float(frame.ball.y))

            retention = 1.0
            arrivals = []
            for oid, state in frame.players.items():
                if state.team_id == team or oid in keepers:
                    continue
                before = prev.players.get(oid)
                if before is None:
                    vx = vy = 0.0
                else:
                    vx = (state.x - before.x) / dt
                    vy = (state.y - before.y) / dt
                velocity = VelocityEstimate(vx, vy, math.hypot(vx, vy), 2, dt)
                t_arr = float(player_time_to_point(state, velocity, ball, arrival))
                arrivals.append(t_arr)
                retention *= logistic((t_arr - ACTION_TIME_S) / SIGMA_S)
            if not arrivals:
                continue

            labels = {}
            for horizon in args.horizons:
                ahead = fid + int(round(horizon * FPS / args.stride)) * args.stride
                j = index.get(ahead)
                if j is None:
                    continue
                pid2, team2 = owner_of(frames[order[j]], CONTROL_M)
                if pid2 is None:
                    continue                      # ball in flight: unlabelled
                labels[horizon] = (team2 == team, pid2 == pid)
            if labels:
                rows.append((retention, min(arrivals),
                             sum(a <= 1.0 for a in arrivals), labels))
        print(f"  [{match_id}] 프레임 {len(order):,} · 누적 표본 {len(rows):,}", flush=True)
        del frames

    print(f"\n샘플 간격 {dt:.2f}s · 프레임 {n_frames:,} · 소유 판정 {n_owned:,}"
          f" · 라벨 가능 {len(rows):,}")

    pred = np.array([r[0] for r in rows])
    near = np.array([r[1] for r in rows])
    out = {"stride": args.stride, "dt_s": dt, "frames": n_frames,
           "owned": n_owned, "labelled": len(rows)}

    for horizon in args.horizons:
        keep = [r for r in rows if horizon in r[3]]
        if not keep:
            continue
        p = np.array([r[0] for r in keep])
        a = np.array([r[1] for r in keep])
        team_kept = np.array([r[3][horizon][0] for r in keep], dtype=float)
        same_kept = np.array([r[3][horizon][1] for r in keep], dtype=float)
        print("\n" + "=" * 80)
        print(f"[T = {horizon:.1f}초 뒤]  n={len(keep):,}")
        print("=" * 80)
        print(f"  같은 팀이 여전히 소유: {team_kept.mean():.3f}"
              f"  ·  같은 선수가 여전히 소유: {same_kept.mean():.3f}")
        print(f"  모델 예측 평균 {p.mean():.3f} · 중앙 {np.median(p):.3f}")
        rec = {"n": len(keep), "team_retained": float(team_kept.mean()),
               "same_player_retained": float(same_kept.mean()),
               "model_mean": float(p.mean()), "curves": {}}

        # 'team kept' counts a carrier who passed out of trouble as a survival,
        # so it is an UPPER bound on what a continued carry would achieve;
        # 'same player kept' counts a deliberate pass as a failure, so it is a
        # LOWER bound. The model's target lies between them.
        print(f"\n  가장 가까운 수비수 도착시간별  (상한=팀유지, 하한=같은선수)")
        print(f"  {'구간(초)':<15}{'n':>8}{'상한':>8}{'하한':>8}{'빼앗김':>9}"
              f"{'패스로털어냄':>13}{'모델':>8}")
        band = []
        edges = [0, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 99]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (a >= lo) & (a < hi)
            if m.sum() < 50:
                continue
            hi_b, lo_b = float(team_kept[m].mean()), float(same_kept[m].mean())
            lost = 1.0 - hi_b
            passed = hi_b - lo_b
            prd = float(p[m].mean())
            band.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "upper_team_kept": hi_b, "lower_same_player": lo_b,
                         "dispossessed": lost, "passed_away": passed,
                         "predicted": prd})
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>8,}{hi_b:>8.3f}{lo_b:>8.3f}"
                  f"{lost:>9.3f}{passed:>13.3f}{prd:>8.3f}")
        rec["curves"]["by_nearest_arrival"] = band

        print(f"\n  압박이 셀수록 빨리 털어내는가 (패스로 털어낸 비율)")
        tight = a < 0.5
        free = a >= 1.5
        if tight.sum() >= 50 and free.sum() >= 50:
            print(f"    수비수 0.5초 내 도달 (n={tight.sum():,}): "
                  f"빼앗김 {1-team_kept[tight].mean():.3f} · "
                  f"패스로털어냄 {team_kept[tight].mean()-same_kept[tight].mean():.3f}")
            print(f"    수비수 1.5초+ 여유 (n={free.sum():,}): "
                  f"빼앗김 {1-team_kept[free].mean():.3f} · "
                  f"패스로털어냄 {team_kept[free].mean()-same_kept[free].mean():.3f}")
            rec["pressure_contrast"] = {
                "tight_n": int(tight.sum()),
                "tight_dispossessed": float(1 - team_kept[tight].mean()),
                "tight_passed": float(team_kept[tight].mean() - same_kept[tight].mean()),
                "free_n": int(free.sum()),
                "free_dispossessed": float(1 - team_kept[free].mean()),
                "free_passed": float(team_kept[free].mean() - same_kept[free].mean()),
            }

        print(f"\n  모델 예측 구간별 (캘리브레이션)")
        print(f"  {'예측 구간':<16}{'n':>9}{'실제 팀유지':>12}{'모델 평균':>11}{'차이':>9}")
        cal = []
        for lo, hi in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1)):
            m = (p >= lo) & (p < hi)
            if m.sum() < 50:
                continue
            obs, prd = float(team_kept[m].mean()), float(p[m].mean())
            cal.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "observed": obs, "predicted": prd})
            print(f"  [{lo:.1f}, {hi:.1f})      {m.sum():>9,}{obs:>12.3f}"
                  f"{prd:>11.3f}{prd-obs:>+9.3f}")
        rec["curves"]["calibration"] = cal
        out[f"horizon_{horizon}"] = rec

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
