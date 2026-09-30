#!/usr/bin/env python3
"""Fix the hybrid delivery model's LEVEL without throwing away its kinematics.

The swap to xpass360 was motivated by the hybrid's low central P (0.624 on
genuine pass cells against a real open-play completion rate near 0.83). The
investigation showed the kinematics are the part that works -- stratified on
static geometry, the mechanistic chain correlates +0.762 with receiver_first
where xpass360 manages +0.267 -- and that the level is the part that is wrong.

So: keep the inputs, fit the output. The hybrid's raw score is

    raw = xpass_geometry x path_survival x receiver_first
                         x secure_possession x pressure_execution

a product of five probabilities, which is systematically pinned low. This
recomputes that raw score on REAL Bundesliga passes with the same machinery
(point_reception_estimate, same ArrivalModelConfig), pairs it with the recorded
outcome, and fits a monotone (isotonic) calibration raw -> P(complete).

Isotonic is chosen deliberately: it cannot reorder anything. Whatever the
kinematic chain says is better stays better; only the numbers attached to the
ordering move. Held out by match, never by pass, so no match leaks into its
own calibration.

NOT fitted on the human beneficiary labels. Those stay an evaluation set.

CAVEAT: observed passes are selected -- players attempt the ones they think
work. This corrects the level on attempted passes; it does not conjure a
ground truth for the counterfactual through balls the option catalogue
invents. That limit belongs in any claim made from these numbers.

Usage:
    python scripts/calibrate_hybrid_delivery.py \
        --raw-dir data/raw/bundesliga-integrated \
        --output out/delivery_analysis/hybrid_calibration.json
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
    infer_attacking_direction,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    estimate_frame_velocities,
    point_reception_estimate,
)
from offball_value.observed_passes import infer_intended_target, resolve_pass_target
from offball_value.xpass import load_xpass_model, predict_pass_success

EXCLUDED_MATCHES = {"DFL-MAT-J03WN1"}
HISTORY_FRAMES = 10          # 0.4 s at 25 fps, matches ArrivalModelConfig.history_seconds
FORWARD_FRAMES = 75          # 3 s: long enough for any pass to resolve
FORWARD_STRIDE = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--xpass-model", type=Path,
                   default=Path("data/processed/xpass_v0/xpass_hist_gbdt.joblib"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (sorted x, fitted y) defining a step map."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order].astype(float), y[order].astype(float)
    level = ys.copy()
    weight = np.ones_like(level)
    i = 0
    stack_lo, stack_val, stack_w = [], [], []
    for i in range(len(level)):
        lo, val, w = i, level[i], weight[i]
        while stack_val and stack_val[-1] > val:
            plo, pval, pw = stack_lo.pop(), stack_val.pop(), stack_w.pop()
            val = (pval * pw + val * w) / (pw + w)
            w += pw
            lo = plo
        stack_lo.append(lo); stack_val.append(val); stack_w.append(w)
    fitted = np.empty_like(level)
    end = len(level)
    for lo, val in zip(reversed(stack_lo), reversed(stack_val)):
        fitted[lo:end] = val
        end = lo
    return xs, fitted


def apply_isotonic(xs: np.ndarray, ys: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(xs, query, side="right") - 1, 0, len(xs) - 1)
    return ys[idx]


def brier(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((pred - label) ** 2))


def collect(match_id: str, raw_dir: Path, xmodel, config: ArrivalModelConfig):
    files = find_bundesliga_files(raw_dir, match_id)
    meta = load_bundesliga_match_metadata(files["matchinfo"])
    clock = load_bundesliga_frame_clock(files["positions"])
    events = load_bundesliga_events(files["events"], clock)

    passes = events[
        (events["event_type"] == "pass")
        & (events["from_open_play"] == True)  # noqa: E712
        & (events["evaluation"].isin(["successfullyCompleted", "unsuccessful"]))
        & events["frame_id"].notna()
        & events["player_id"].notna() & events["team_id"].notna()
    ].copy()
    if passes.empty:
        return [], {}

    needed = set()
    for fid in passes["frame_id"].astype(int):
        needed.update(range(fid - HISTORY_FRAMES, fid + 1))
        needed.update(range(fid, fid + FORWARD_FRAMES + 1, FORWARD_STRIDE))
    frames = load_bundesliga_frames(files["positions"], needed)

    rows = []
    stats = {"events": len(passes), "unresolved": 0, "label_agree": 0,
             "label_seen": 0, "intended_is_actual": 0}
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

        # MODEL INPUT: who the pass was aimed at, from the ball's release
        # heading only -- never from who ended up with it.
        intended = infer_intended_target(frames, fid, passer, team)
        # LABEL: who actually brought it under control.
        observed = resolve_pass_target(
            frames, fid, passer, team,
            forward_frames=FORWARD_FRAMES, stride=FORWARD_STRIDE)
        if intended is None or observed is None or observed.travel_distance_m < 3.0:
            stats["unresolved"] += 1
            continue
        stats["intended_is_actual"] += int(intended.receiver_id == observed.receiver_id)
        stats["label_seen"] += 1
        event_completed = ev["evaluation"] == "successfullyCompleted"
        if observed.completed == bool(event_completed):
            stats["label_agree"] += 1

        ball = (float(frame.ball.x), float(frame.ball.y))
        try:
            velocities = estimate_frame_velocities(window, fid, config)
            est = point_reception_estimate(
                frame, velocities, intended.receiver_id, team, ball,
                intended.target_xy, config, passer_id=passer)
        except (ValueError, KeyError):
            continue
        direction = infer_attacking_direction(frame, team, meta)
        try:
            xgeom = float(predict_pass_success(xmodel, ball, intended.target_xy, direction))
        except Exception:
            continue
        interaction = float(np.clip(
            est.path_survival_probability * est.receiver_first_probability
            * est.secure_possession_probability * est.pressure_execution_probability,
            0.0, 1.0))
        rows.append({
            "match": match_id,
            "label": 1.0 if observed.completed else 0.0,
            "event_label": 1.0 if event_completed else 0.0,
            "hybrid_raw": float(np.clip(xgeom * interaction, 0.0, 1.0)),
            "mechanistic": float(np.clip(est.receive_probability, 0.0, 1.0)),
            "xpass_geometry": xgeom,
            "interaction": interaction,
            "receiver_first": float(est.receiver_first_probability),
            "defender_margin_s": float(est.defender_time_margin_s)
            if np.isfinite(est.defender_time_margin_s) else 99.0,
            "ball_time_s": float(est.ball_arrival_time_s),
            "receiver_time_s": float(est.receiver_arrival_time_s),
            "defender_arrival_s": float(est.nearest_defender_arrival_time_s)
            if np.isfinite(est.nearest_defender_arrival_time_s) else 99.0,
            "path_margin_s": float(est.path_time_margin_s)
            if np.isfinite(est.path_time_margin_s) else 99.0,
            "pressure_arrival_s": float(est.passer_pressure_arrival_time_s)
            if np.isfinite(est.passer_pressure_arrival_time_s) else 99.0,
            "pass_distance_m": float(est.pass_distance_m),
            "path_survival": float(est.path_survival_probability),
            "secure": float(est.secure_possession_probability),
            "pressure": float(est.pressure_execution_probability),
            "travel_m": observed.travel_distance_m,
            "aim_m": intended.along_m,
            "aim_perp_m": intended.perpendicular_m,
            "intended_is_actual": 1.0 if intended.receiver_id == observed.receiver_id else 0.0,
            "flight_s": observed.flight_time_s,
        })
    return rows, stats


def main() -> None:
    args = parse_args()
    config = ArrivalModelConfig()
    xmodel = load_xpass_model(args.xpass_model)
    match_ids = [m for m in list_bundesliga_match_ids(args.raw_dir)
                 if m not in EXCLUDED_MATCHES]
    print(f"{len(match_ids)} matches", flush=True)

    rows = []
    agg = {"events": 0, "unresolved": 0, "label_agree": 0, "label_seen": 0,
           "intended_is_actual": 0}
    for match_id in match_ids:
        got, st = collect(match_id, args.raw_dir, xmodel, config)
        rows.extend(got)
        for k in agg:
            agg[k] += st.get(k, 0)
        print(f"  [{match_id}] passes {len(got):,} · total {len(rows):,}", flush=True)
    print(f"\n  events {agg['events']:,} · destination not resolved from tracking {agg['unresolved']:,}"
          f" ({agg['unresolved']/max(agg['events'],1):.1%})")
    if agg["label_seen"]:
        print(f"  tracking label vs event Evaluation agreement: "
              f"{agg['label_agree']/agg['label_seen']:.1%}  (independent check)")
        print(f"  intended receiver == actual receiver: "
              f"{agg['intended_is_actual']/agg['label_seen']:.1%}"
              f"  (100%% means leakage, low means inference failure — should sit near the completion rate)")

    label = np.array([r["label"] for r in rows])
    raw = np.array([r["hybrid_raw"] for r in rows])
    mech = np.array([r["mechanistic"] for r in rows])
    matches = np.array([r["match"] for r in rows])
    inferred = np.zeros(len(rows), dtype=bool)

    print(f"\n{'='*84}")
    print("[1] Population and current level")
    print("="*84)
    print(f"  {len(rows):,} open-play passes · actual completion rate {label.mean():.4f}")
    ev_label = np.array([r["event_label"] for r in rows])
    print(f"  completion rate by event Evaluation {ev_label.mean():.4f} (for reference)")
    print(f"  median travel distance {np.median([r['travel_m'] for r in rows]):.1f} m · "
          f"median flight time {np.median([r['flight_s'] for r in rows]):.2f} s")
    print(f"\n  Medians of the hybrid's 4 terms")
    for k, nm in (("path_survival","path survival"),("receiver_first","receiver first"),
                  ("secure","secure"),("pressure","passer pressure"),("xpass_geometry","xPass geometry")):
        print(f"    {nm:<12}{np.median([r[k] for r in rows]):>8.3f}")
    print(f"\n  {'':<20}{'median':>9}{'mean':>9}{'Brier':>9}")
    for nm, v in (("hybrid raw", raw), ("mechanistic only", mech)):
        print(f"  {nm:<18}{np.median(v):>9.3f}{v.mean():>9.3f}{brier(v, label):>9.4f}")
    print(f"  {'actual':<18}{'':>9}{label.mean():>9.3f}")

    print(f"\n{'='*84}")
    print("[1b] Actual completion rate by arrival race — observed, not modelled")
    print("="*84)
    print("  margin = time for the defender to reach the target − time for the receiver to be able to take the ball")
    print("  (positive = receiver first, 0 = simultaneous, negative = defender first)\n")
    margin = np.array([r["defender_margin_s"] for r in rows])
    print(f"  {'margin (s)':<20}{'n':>8}{'actual rate':>13}"
          f"{'hybrid':>11}{'mech.':>9}")
    bands = [(-99, -1.0, "defender 1 s+ first"), (-1.0, -0.5, "-1.0 ~ -0.5"),
             (-0.5, -0.2, "-0.5 ~ -0.2"), (-0.2, 0.2, "near simultaneous"),
             (0.2, 0.5, "+0.2 ~ +0.5"), (0.5, 1.0, "+0.5 ~ +1.0"),
             (1.0, 98, "receiver 1 s+ first")]
    for lo, hi, name in bands:
        m = (margin >= lo) & (margin < hi)
        if m.sum() < 30:
            continue
        print(f"  {name:<20}{m.sum():>8,}{label[m].mean():>13.3f}"
              f"{raw[m].mean():>11.3f}{mech[m].mean():>9.3f}")
    print("\n  The key is whether the 'near simultaneous' row sits near 0.5.")

    print(f"\n{'='*84}")
    print("[2] Leave-one-match-out isotonic calibration")
    print("="*84)
    cal_raw = np.zeros_like(raw)
    cal_mech = np.zeros_like(mech)
    for held in sorted(set(matches)):
        fit = matches != held
        use = matches == held
        xs, ys = isotonic(raw[fit], label[fit])
        cal_raw[use] = apply_isotonic(xs, ys, raw[use])
        xs2, ys2 = isotonic(mech[fit], label[fit])
        cal_mech[use] = apply_isotonic(xs2, ys2, mech[use])
    print(f"  {'':<26}{'median':>9}{'mean':>9}{'Brier':>9}{'gain':>9}")
    b0 = brier(raw, label)
    print(f"  {'hybrid raw':<24}{np.median(raw):>9.3f}{raw.mean():>9.3f}{b0:>9.4f}")
    b1 = brier(cal_raw, label)
    print(f"  {'hybrid calibrated':<24}{np.median(cal_raw):>9.3f}{cal_raw.mean():>9.3f}"
          f"{b1:>9.4f}{(b0-b1)/b0:>8.1%}")
    b2 = brier(mech, label)
    b3 = brier(cal_mech, label)
    print(f"  {'mechanistic raw':<24}{np.median(mech):>9.3f}{mech.mean():>9.3f}{b2:>9.4f}")
    print(f"  {'mechanistic calibrated':<24}{np.median(cal_mech):>9.3f}{cal_mech.mean():>9.3f}"
          f"{b3:>9.4f}{(b2-b3)/b2:>8.1%}")

    print(f"\n{'='*84}")
    print("[3] Calibration map: what hybrid raw turns into")
    print("="*84)
    xs, ys = isotonic(raw, label)
    print(f"  {'raw band':<16}{'n':>9}{'actual':>9}{'calibrated':>15}")
    out_map = []
    for lo, hi in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1)):
        m = (raw >= lo) & (raw < hi)
        if m.sum() < 30:
            continue
        mapped = float(apply_isotonic(xs, ys, np.array([(lo+hi)/2]))[0])
        out_map.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "observed": float(label[m].mean()), "mapped": mapped})
        print(f"  [{lo:.1f}, {hi:.1f})       {m.sum():>9,}{label[m].mean():>9.3f}{mapped:>15.3f}")

    out = {
        "passes": len(rows), "completion_rate": float(label.mean()),

        "hybrid_raw": {"median": float(np.median(raw)), "mean": float(raw.mean()),
                       "brier": b0},
        "hybrid_calibrated": {"median": float(np.median(cal_raw)),
                              "mean": float(cal_raw.mean()), "brier": b1},
        "mechanistic_raw": {"median": float(np.median(mech)), "brier": b2},
        "mechanistic_calibrated": {"median": float(np.median(cal_mech)), "brier": b3},
        "calibration_map": out_map,
        "isotonic_x": xs[::max(1, len(xs)//400)].tolist(),
        "isotonic_y": ys[::max(1, len(ys)//400)].tolist(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
