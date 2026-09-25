#!/usr/bin/env python3
"""A delivery model that can see the arrival race AND is fitted to outcomes.

The investigation left the two candidates each holding half of what is needed:

    mechanistic chain   orders the arrival race well (rank corr +0.758 within
                        a static stratum, against xpass360's +0.260) but its
                        level is badly wrong (median 0.32 on passes that
                        complete 81 % of the time)
    xpass360            is calibrated to within 0.01 on real passes but, held
                        inside a static stratum, barely orders the race at all

Isotonic calibration cannot resolve this: it fixes the level while flattening
the very response that made the mechanistic chain worth keeping (-0.208 to
-0.077 on the counterfactual-defender probe).

So: give a learned model the kinematic quantities as FEATURES and fit it to
observed outcomes. Both properties come from the thing that can actually
supply them -- ordering from the physics, level from the data.

HONEST LIMIT, stated before any number is read: six matches. This is a
feasibility probe, not a shippable model. Held out by MATCH, never by pass, so
the reported numbers are at least not self-flattering; but with folds this
small the variance across them matters more than the mean, and both are printed.

Usage:
    python scripts/train_kinematic_delivery.py \
        --output out/delivery_analysis/kinematic_model.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    find_bundesliga_files,
    infer_attacking_direction,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.observed_passes import infer_intended_target, resolve_pass_target
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    estimate_frame_velocities,
    point_reception_estimate,
)
from offball_value.xpass import (
    lane_features,
    load_xpass_model,
    predict_pass_success,
    predict_pass_success_360,
    to_statsbomb_coordinates,
)

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
HISTORY_FRAMES = 10
FORWARD_FRAMES = 125          # 5 s: recovers more long balls than the 3 s window
FORWARD_STRIDE = 2

KINEMATIC = [
    "receiver_first", "path_survival", "secure", "pressure",
    "distance_execution", "pass_execution",
    "defender_margin_s", "path_margin_s",
    "ball_arrival_s", "receiver_arrival_s", "defender_arrival_s",
    "pass_distance_m",
]
STATIC = [
    "passer_nearest_opponent", "lane_min_perpendicular", "lane_blockers_2m",
    "lane_blockers_4m", "target_nearest_opponent", "visible_opponents",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--xpass-model", type=Path,
                   default=Path("data/processed/xpass_v0/xpass_hist_gbdt.joblib"))
    p.add_argument("--xpass-360-model", type=Path,
                   default=Path("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def collect(match_id, raw_dir, xmodel, x360, config):
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
            velocities = estimate_frame_velocities(window, fid, config)
            est = point_reception_estimate(
                frame, velocities, intended.receiver_id, team, ball,
                intended.target_xy, config, passer_id=passer)
        except (ValueError, KeyError):
            continue
        direction = infer_attacking_direction(frame, team, meta)
        opponents = [(float(p.x), float(p.y)) for pid, p in frame.players.items()
                     if p.team_id != team and pid not in keepers]
        try:
            xgeom = float(predict_pass_success(xmodel, ball, intended.target_xy, direction))
            p360 = float(predict_pass_success_360(
                x360, ball, intended.target_xy, direction, opponents))
        except Exception:
            continue
        lf = lane_features(
            to_statsbomb_coordinates(ball, direction),
            to_statsbomb_coordinates(intended.target_xy, direction),
            [to_statsbomb_coordinates(xy, direction) for xy in opponents])
        interaction = float(np.clip(
            est.path_survival_probability * est.receiver_first_probability
            * est.secure_possession_probability * est.pressure_execution_probability,
            0.0, 1.0))
        def finite(v, cap=20.0):
            v = float(v)
            return float(np.clip(v, -cap, cap)) if math.isfinite(v) else 0.0
        rows.append({
            "match": match_id,
            "label": 1.0 if observed.completed else 0.0,
            "mech": float(np.clip(est.receive_probability, 0.0, 1.0)),
            "hybrid": float(np.clip(xgeom * interaction, 0.0, 1.0)),
            "x360": p360,
            "receiver_first": float(est.receiver_first_probability),
            "path_survival": float(est.path_survival_probability),
            "secure": float(est.secure_possession_probability),
            "pressure": float(est.pressure_execution_probability),
            "distance_execution": float(est.distance_execution_probability),
            "pass_execution": float(est.pass_execution_probability),
            "defender_margin_s": finite(est.defender_time_margin_s),
            "path_margin_s": finite(est.path_time_margin_s),
            "ball_arrival_s": finite(est.ball_arrival_time_s),
            "receiver_arrival_s": finite(est.receiver_arrival_time_s),
            "defender_arrival_s": finite(est.nearest_defender_arrival_time_s),
            "pass_distance_m": finite(est.pass_distance_m, 80.0),
            "passer_nearest_opponent": float(lf[0]),
            "lane_min_perpendicular": float(lf[1]),
            "lane_blockers_2m": float(lf[2]),
            "lane_blockers_4m": float(lf[3]),
            "target_nearest_opponent": float(lf[4]),
            "visible_opponents": float(lf[5]),
        })
    return rows


def auc(pred, label):
    pos, neg = pred[label > 0.5], pred[label <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def brier(pred, label):
    return float(np.mean((pred - label) ** 2))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

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

    label = np.array([r["label"] for r in rows])
    match = np.array([r["match"] for r in rows])
    print(f"\n패스 {len(rows):,}개 · 실제 성공률 {label.mean():.4f} · 경기 {len(set(match))}개")

    feature_sets = {
        "운동학만": KINEMATIC,
        "정적만(xpass360 와 같은 feature)": STATIC,
        "운동학+정적": KINEMATIC + STATIC,
    }
    out = {"passes": len(rows), "completion": float(label.mean()), "folds": {}, "models": {}}

    print(f"\n{'='*84}")
    print("경기 단위 leave-one-out · 보류 폴드에서만 평가")
    print("="*84)
    print(f"  {'':<34}{'AUC':>9}{'Brier':>9}{'폴드별 AUC 표준편차':>20}")

    for key, label_name in (("mech", "역학 raw"), ("hybrid", "하이브리드 raw"),
                            ("x360", "xpass360")):
        v = np.array([r[key] for r in rows])
        per = [auc(v[match == m], label[match == m]) for m in sorted(set(match))]
        out["models"][label_name] = {"auc": auc(v, label), "brier": brier(v, label)}
        print(f"  {label_name:<32}{auc(v, label):>9.3f}{brier(v, label):>9.4f}"
              f"{np.nanstd(per):>20.3f}")

    for name, cols in feature_sets.items():
        X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
        pred = np.zeros(len(rows))
        per = []
        for held in sorted(set(match)):
            fit, use = match != held, match == held
            model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=150, learning_rate=0.06,
                min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            model.fit(X[fit], label[fit])
            pred[use] = model.predict_proba(X[use])[:, 1]
            per.append(auc(pred[use], label[use]))
        out["models"][f"학습: {name}"] = {"auc": auc(pred, label),
                                        "brier": brier(pred, label),
                                        "fold_auc": per}
        print(f"  {'학습: ' + name:<32}{auc(pred, label):>9.3f}"
              f"{brier(pred, label):>9.4f}{np.nanstd(per):>20.3f}")

    print(f"\n  기저율만 찍는 예측기의 Brier: {label.mean()*(1-label.mean()):.4f}")
    print("  (AUC 0.5 = 무작위. 폴드 표준편차가 크면 경기 6개로는 못 믿는다는 뜻)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
