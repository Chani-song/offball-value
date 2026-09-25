#!/usr/bin/env python3
"""Put a defender in the lane that was not there. Which model notices?

Every measurement so far has been observational: how far P moves as the
response catalogue moves a defender it was already tracking. That is a spread,
not an intervention, and it is measured on our own counterfactual cells --
which is exactly the off-distribution ground xpass360 has an excuse on.

This is the controlled version. Take REAL Bundesliga open-play passes, insert
one stationary defender on the lane at a chosen perpendicular offset, and
recompute the delivery under all three models with everything else held fixed.
Real passes, so no model is out of its distribution; a true intervention, so
the number is a causal response and not a correlation.

There is a benchmark for the magnitude. StatsBomb's own completion rates by
lane occupancy (354,889 passes) go 0.918 with no defender within 2 m of the
lane to 0.655 with one -- a gap of 0.263. That gap is observational and so
partly confounded (passes into occupied lanes differ in other ways), which
makes it an upper reference rather than a target. A model whose counterfactual
response is an order of magnitude smaller than it is not responding.

For context, xpass360's own training report records exactly this probe on 2000
open passes: predicted 0.963 before, dropping 0.097 (rebuilt frame) or 0.034
(hand-edited) when a defender is inserted.

Usage:
    python scripts/probe_counterfactual_defender.py \
        --output out/delivery_analysis/counterfactual_defender.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    BundesligaObjectState,
    find_bundesliga_files,
    infer_attacking_direction,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    estimate_frame_velocities,
    point_reception_estimate,
)
from offball_value.observed_passes import infer_intended_target, resolve_pass_target
from offball_value.xpass import (
    lane_features,
    load_xpass_model,
    predict_pass_success,
    predict_pass_success_360,
    to_statsbomb_coordinates,
)

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
HISTORY_FRAMES = 10
FORWARD_FRAMES = 75
FORWARD_STRIDE = 2
SYNTH_ID = "SYNTH-DEFENDER-0"
OFFSETS_M = [0.0, 1.0, 2.0, 3.0, 5.0]
MAX_PASSES_PER_MATCH = 400          # the probe is 6x the work of one estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--xpass-model", type=Path,
                   default=Path("data/processed/xpass_v0/xpass_hist_gbdt.joblib"))
    p.add_argument("--xpass-360-model", type=Path,
                   default=Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"))
    p.add_argument("--calibration", type=Path,
                   default=Path("out/delivery_analysis/hybrid_calibration.json"),
                   help="isotonic map fitted by calibrate_hybrid_delivery.py")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def hybrid_from(est, xgeom: float) -> float:
    interaction = float(np.clip(
        est.path_survival_probability * est.receiver_first_probability
        * est.secure_possession_probability * est.pressure_execution_probability,
        0.0, 1.0))
    return float(np.clip(xgeom * interaction, 0.0, 1.0))


def lane_point(ball, target, offset_m):
    """Midpoint of the lane, pushed `offset_m` perpendicular to it."""
    mx, my = (ball[0] + target[0]) / 2.0, (ball[1] + target[1]) / 2.0
    dx, dy = target[0] - ball[0], target[1] - ball[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return mx, my
    return mx + (-dy / length) * offset_m, my + (dx / length) * offset_m


def collect(match_id, raw_dir, xmodel, x360, config):
    files = find_bundesliga_files(raw_dir, match_id)
    meta = load_bundesliga_match_metadata(files["matchinfo"])
    clock = load_bundesliga_frame_clock(files["positions"])
    events = load_bundesliga_events(files["events"], clock)
    passes = events[
        (events["event_type"] == "pass")
        & (events["from_open_play"] == True)  # noqa: E712
        & (events["evaluation"].isin(["successfullyCompleted", "unsuccessful"]))
        & events["frame_id"].notna()
        & events["x_end"].notna() & events["player_id"].notna()
        & events["team_id"].notna()
    ].copy()
    if passes.empty:
        return []
    passes = passes.iloc[:: max(1, len(passes) // MAX_PASSES_PER_MATCH)]

    needed = set()
    for fid in passes["frame_id"].astype(int):
        needed.update(range(fid - HISTORY_FRAMES, fid + 1))
        needed.update(range(fid, fid + FORWARD_FRAMES + 1, FORWARD_STRIDE))
    frames = load_bundesliga_frames(files["positions"], needed)
    keepers = {k for t in (meta.home_team_id, meta.away_team_id)
               if (k := meta.goalkeeper_id(t)) is not None}

    rows = []
    for _, ev in passes.iterrows():
        fid = int(ev["frame_id"])
        frame = frames.get(fid)
        if frame is None or frame.ball is None:
            continue
        window = [frames[f] for f in range(fid - HISTORY_FRAMES, fid + 1) if f in frames]
        if len(window) < 3:
            continue
        passer, team = str(ev["player_id"]), str(ev["team_id"])
        if passer not in frame.players:
            continue
        defending = meta.away_team_id if team == meta.home_team_id else meta.home_team_id
        ball = (float(frame.ball.x), float(frame.ball.y))
        # The DFL event has no destination (X-Position == X-Source-Position on
        # 85 % of passes), and naming the actual receiver would leak the
        # outcome, so the aim comes from the ball's release heading alone.
        intended = infer_intended_target(frames, fid, passer, team)
        observed = resolve_pass_target(frames, fid, passer, team,
                                       forward_frames=FORWARD_FRAMES,
                                       stride=FORWARD_STRIDE)
        if intended is None or observed is None:
            continue
        target = intended.target_xy
        receiver = intended.receiver_id
        if math.hypot(target[0] - ball[0], target[1] - ball[1]) < 5.0:
            continue                                   # too short for a lane
        try:
            velocities = estimate_frame_velocities(window, fid, config)
            base_est = point_reception_estimate(
                frame, velocities, str(receiver), team, ball, target, config,
                passer_id=passer)
        except (ValueError, KeyError):
            continue
        direction = infer_attacking_direction(frame, team, meta)
        try:
            xgeom = float(predict_pass_success(xmodel, ball, target, direction))
        except Exception:
            continue
        opponents = [(float(p.x), float(p.y)) for pid, p in frame.players.items()
                     if p.team_id != team and pid not in keepers]
        base = {
            "mech": float(np.clip(base_est.receive_probability, 0.0, 1.0)),
            "hybrid": hybrid_from(base_est, xgeom),
            "x360": float(predict_pass_success_360(x360, ball, target, direction, opponents)),
            "xgeom": xgeom,
        }
        lf = lane_features(
            to_statsbomb_coordinates(ball, direction),
            to_statsbomb_coordinates(target, direction),
            [to_statsbomb_coordinates(xy, direction) for xy in opponents])
        row = {"match": match_id,
               "label": 1.0 if observed.completed else 0.0,
               "blockers_2m": float(lf[2]),
               "lane_min_perp": float(lf[1]),
               "base": base, "shift": {}}
        for offset in OFFSETS_M:
            px, py = lane_point(ball, target, offset)
            synth = BundesligaObjectState(SYNTH_ID, defending, px, py, 0.0, 0.0)
            frame2 = frame.with_player(SYNTH_ID, synth)
            vel2 = dict(velocities)
            vel2[SYNTH_ID] = VelocityEstimate(0.0, 0.0, 0.0, 2, config.history_seconds)
            try:
                est2 = point_reception_estimate(
                    frame2, vel2, str(receiver), team, ball, target, config,
                    passer_id=passer)
            except (ValueError, KeyError):
                continue
            row["shift"][offset] = {
                "mech": float(np.clip(est2.receive_probability, 0.0, 1.0)),
                "hybrid": hybrid_from(est2, xgeom),
                "x360": float(predict_pass_success_360(
                    x360, ball, target, direction, opponents + [(px, py)])),
            }
        if row["shift"]:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = ArrivalModelConfig()
    xmodel = load_xpass_model(args.xpass_model)
    x360 = load_xpass_model(args.xpass_360_model)
    match_ids = [m for m in list_bundesliga_match_ids(args.raw_dir)
                 if m not in EXCLUDED_MATCHES]
    rows = []
    for match_id in match_ids:
        got = collect(match_id, args.raw_dir, xmodel, x360, config)
        rows.extend(got)
        print(f"  [{match_id}] 패스 {len(got):,} · 누적 {len(rows):,}", flush=True)

    print(f"\n{'='*88}")
    print("경로 중앙에 정지 수비수 1명을 꽂았을 때 P 변화 (실제 Bundesliga 패스)")
    print("="*88)
    print(f"  패스 {len(rows):,}개 · 실제 성공률 "
          f"{np.mean([r['label'] for r in rows]):.3f}")
    names = [("mech", "역학 단독"), ("hybrid", "하이브리드"),
             ("x360", "xpass360"), ("xgeom", "xPass 기하(수비수 입력 없음)")]
    print(f"\n  기준선 (수비수 넣기 전)")
    for key, label in names:
        v = np.array([r["base"][key] for r in rows])
        print(f"    {label:<28} 중앙 {np.median(v):.3f} · 평균 {v.mean():.3f}")

    out = {"passes": len(rows), "offsets": OFFSETS_M, "drops": {}}
    print(f"\n  {'경로에서 수비수까지':<20}{'n':>8}", end="")
    for _k, label in names[:3]:
        print(f"{label:>14}", end="")
    print()
    for offset in OFFSETS_M:
        sel = [r for r in rows if offset in r["shift"]]
        if not sel:
            continue
        line = f"  {offset:>6.1f} m{'':<12}{len(sel):>8}"
        rec = {}
        for key, _label in names[:3]:
            base = np.array([r["base"][key] for r in sel])
            after = np.array([r["shift"][offset][key] for r in sel])
            d = float(np.mean(after - base))
            rec[key] = {"mean_drop": d,
                        "median_drop": float(np.median(after - base)),
                        "base_mean": float(base.mean())}
            line += f"{d:>+14.3f}"
        out["drops"][offset] = rec
        print(line)

    # Calibration is monotone, so it cannot reorder anything -- but it CAN
    # squash the size of a response. A fix for the level is worthless if it
    # flattens the defender sensitivity that was the reason to keep this model.
    if args.calibration and args.calibration.exists():
        cal = json.loads(args.calibration.read_text(encoding="utf-8"))
        cx = np.array(cal["isotonic_x"], dtype=float)
        cy = np.array(cal["isotonic_y"], dtype=float)

        def apply_cal(v):
            idx = np.clip(np.searchsorted(cx, v, side="right") - 1, 0, len(cx) - 1)
            return cy[idx]

        print(f"\n{'='*88}")
        print("캘리브레이션을 적용한 뒤에도 수비수 반응이 남는가")
        print("="*88)
        print(f"  {'경로에서 수비수까지':<22}{'하이브리드 raw':>16}{'하이브리드 보정후':>18}")
        out["calibrated_drops"] = {}
        for offset in OFFSETS_M:
            sel = [r for r in rows if offset in r["shift"]]
            if not sel:
                continue
            base = np.array([r["base"]["hybrid"] for r in sel])
            after = np.array([r["shift"][offset]["hybrid"] for r in sel])
            raw_d = float(np.mean(after - base))
            cal_d = float(np.mean(apply_cal(after) - apply_cal(base)))
            out["calibrated_drops"][offset] = {"raw": raw_d, "calibrated": cal_d}
            print(f"  {offset:>6.1f} m{'':<14}{raw_d:>+16.3f}{cal_d:>+18.3f}")
        b = np.array([r["base"]["hybrid"] for r in rows])
        a0 = np.array([r["shift"][0.0]["hybrid"] for r in rows if 0.0 in r["shift"]])
        b0 = np.array([r["base"]["hybrid"] for r in rows if 0.0 in r["shift"]])
        print(f"\n  보정 후 기준선 평균 {apply_cal(b).mean():.3f} (실제 성공률 "
              f"{np.mean([r['label'] for r in rows]):.3f})")
        print(f"  보정 후 0 m 개입:  {apply_cal(b0).mean():.3f} -> {apply_cal(a0).mean():.3f}"
              f"   (상대변화 {100*(apply_cal(a0).mean()-apply_cal(b0).mean())/max(apply_cal(b0).mean(),1e-9):+.1f}%)")

    print(f"\n{'='*88}")
    print("개입 전 레인이 비어 있던 패스만 (StatsBomb 0명->1명 과 직접 비교)")
    print("="*88)
    clean = [r for r in rows if r["blockers_2m"] < 1 and 0.0 in r["shift"]]
    print(f"  개입 전 경로 2m 안 수비수 0명인 패스 {len(clean):,}개 "
          f"({len(clean)/max(len(rows),1):.1%})")
    if clean:
        print(f"  {'':<22}{'개입 전':>10}{'개입 후':>10}{'변화':>10}{'상대':>10}")
        out["zero_blocker"] = {"n": len(clean)}
        for key, label in names[:3]:
            b = np.array([r["base"][key] for r in clean])
            a = np.array([r["shift"][0.0][key] for r in clean])
            rel = 100 * (a.mean() - b.mean()) / max(b.mean(), 1e-9)
            out["zero_blocker"][key] = {"before": float(b.mean()),
                                        "after": float(a.mean()),
                                        "drop": float(a.mean() - b.mean()),
                                        "relative_pct": float(rel)}
            print(f"  {label:<20}{b.mean():>10.3f}{a.mean():>10.3f}"
                  f"{a.mean()-b.mean():>+10.3f}{rel:>+9.1f}%")
        lab = np.mean([r["label"] for r in clean])
        out["zero_blocker"]["observed_completion"] = float(lab)
        print(f"\n  이 패스들의 실제 성공률 {lab:.3f}"
              f"   (StatsBomb 레인 0명 패스의 실제 성공률 0.918)")
        print(f"  StatsBomb 기준 목표:  0.918 -> 0.655  ({-0.263:+.3f}, {-28.6:+.1f}%)")
        print("  주의: 그 격차는 관측이라 다른 특징의 교란이 섞여 있어 상한 참조일 뿐")

    print(f"\n  참고: StatsBomb 관측 격차 (경로 2m 안 수비수 0명 -> 1명)  -0.263")
    print(f"        xpass360 학습 리포트의 자체 probe                  -0.097 / -0.034")

    print(f"\n  {'':<20}{'수비수 경로상(0m) 넣었을 때 P':>34}")
    for key, label in names[:3]:
        sel = [r for r in rows if 0.0 in r["shift"]]
        after = np.array([r["shift"][0.0][key] for r in sel])
        base = np.array([r["base"][key] for r in sel])
        print(f"    {label:<24} {base.mean():.3f} -> {after.mean():.3f}"
              f"   (상대변화 {100*(after.mean()-base.mean())/max(base.mean(),1e-9):+.1f}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
