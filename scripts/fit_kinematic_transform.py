#!/usr/bin/env python3
"""Fit the map from arrival race to completion probability, on observed passes.

The hybrid reads the right thing and converts it wrongly. Measured on real
Bundesliga passes, with arrival times computed from tracking:

    margin              observed   hybrid
    defender 1s+ first    0.629     0.001
    dead heat             0.730     0.127
    receiver 1s+ first    0.939     0.774

Its receiver_first term is logistic(margin / 0.35), which is 0.5 at a dead heat
by construction, and then three more hand-designed probabilities multiply it
down to 0.127. Twelve constants in ArrivalModelConfig set that shape and none
of them carries a stated basis in this repo.

xpass360 converts correctly -- within 0.005 of observed completion in every
length band on 355k real passes -- but reads a still frame, so it has no
arrival race to convert.

So fit the conversion where both halves are available: tracking gives the
kinematics, the tracking-derived outcome gives the label. Low capacity on
purpose, because 1,388 passes cannot support what StatsBomb's 200k can, and
what is being replaced is a handful of constants rather than a feature set.

Held out by match. The reported curve against margin is the point of the
exercise -- a better AUC that still misses the dead-heat rate has not fixed
what was broken.

Usage:
    python scripts/fit_kinematic_transform.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from offball_value.pass_dynamics import ArrivalModelConfig  # noqa: E402
from offball_value.xpass import load_xpass_model  # noqa: E402

FEATURES = [
    "defender_margin_s",      # the race the hybrid already computes
    "path_margin_s",          # can anyone cut the flight
    "pressure_arrival_s",     # is the passer being closed down
    "pass_distance_m",
    "ball_time_s",
    "receiver_time_s",
]
BANDS = [(-99, -1.0, "수비수 1초+ 먼저"), (-1.0, -0.5, "-1.0 ~ -0.5"),
         (-0.5, -0.2, "-0.5 ~ -0.2"), (-0.2, 0.2, "거의 동시"),
         (0.2, 0.5, "+0.2 ~ +0.5"), (0.5, 1.0, "+0.5 ~ +1.0"),
         (1.0, 98, "수신자 1초+ 먼저")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--output", type=Path,
                   default=Path("out/delivery_analysis/kinematic_transform.json"))
    return p.parse_args()


def auc(pred, label):
    pos, neg = pred[label > 0.5], pred[label <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    config = ArrivalModelConfig()
    xmodel = load_xpass_model(Path("data/processed/xpass_v0/xpass_hist_gbdt.joblib"))
    ids = [m for m in _cal.list_bundesliga_match_ids(args.raw_dir)
           if m not in _cal.EXCLUDED_MATCHES]
    rows = []
    for match_id in ids:
        got, _stats = _cal.collect(match_id, args.raw_dir, xmodel, config)
        rows.extend(got)
        print(f"  [{match_id}] 누적 {len(rows):,}", flush=True)

    label = np.array([r["label"] for r in rows])
    match = np.array([r["match"] for r in rows])
    margin = np.array([r["defender_margin_s"] for r in rows])
    X = np.column_stack([
        np.clip([r[f] for r in rows], -20.0, 20.0) for f in FEATURES
    ])
    hybrid = np.array([r["hybrid_raw"] for r in rows])
    mech = np.array([r["mechanistic"] for r in rows])

    print(f"\n패스 {len(rows):,}개 · 실제 성공률 {label.mean():.4f} · 경기 {len(set(match))}개")

    fitted = np.zeros(len(rows))
    for held in sorted(set(match)):
        fit, use = match != held, match == held
        scaler = StandardScaler().fit(X[fit])
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(scaler.transform(X[fit]), label[fit])
        fitted[use] = model.predict_proba(scaler.transform(X[use]))[:, 1]

    print(f"\n{'='*84}")
    print("경기 단위 leave-one-out")
    print("="*84)
    print(f"  {'':<26}{'AUC':>9}{'Brier':>10}")
    for name, v in (("하이브리드 raw", hybrid), ("역학 raw", mech),
                    ("운동학 변환 (적합)", fitted)):
        print(f"  {name:<24}{auc(v, label):>9.3f}{np.mean((v-label)**2):>10.4f}")
    print(f"  {'기저율만':<24}{'—':>9}{label.mean()*(1-label.mean()):>10.4f}")

    print(f"\n{'='*84}")
    print("맞춰야 할 곡선을 실제로 맞추나")
    print("="*84)
    print(f"  {'margin (초)':<20}{'n':>7}{'실제':>9}{'적합':>9}{'하이브리드':>11}")
    out = {"passes": len(rows), "bands": []}
    for lo, hi, name in BANDS:
        m = (margin >= lo) & (margin < hi)
        if m.sum() < 30:
            continue
        rec = {"band": name, "n": int(m.sum()), "observed": float(label[m].mean()),
               "fitted": float(fitted[m].mean()), "hybrid": float(hybrid[m].mean())}
        out["bands"].append(rec)
        print(f"  {name:<20}{m.sum():>7,}{rec['observed']:>9.3f}"
              f"{rec['fitted']:>9.3f}{rec['hybrid']:>11.3f}")
    err_fit = np.mean([abs(b["fitted"] - b["observed"]) for b in out["bands"]])
    err_hyb = np.mean([abs(b["hybrid"] - b["observed"]) for b in out["bands"]])
    print(f"\n  구간별 평균 절대오차   적합 {err_fit:.3f}  ·  하이브리드 {err_hyb:.3f}")
    out["mean_abs_error"] = {"fitted": err_fit, "hybrid": err_hyb}
    out["auc"] = {"hybrid": auc(hybrid, label), "mech": auc(mech, label),
                  "fitted": auc(fitted, label)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
