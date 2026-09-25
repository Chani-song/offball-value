#!/usr/bin/env python3
"""B1-SB: 준현's velocity logistic fitted on StatsBomb 360 club passes.

Fitted on scripts/build_statsbomb_pass_dataset.py rows and scored twice:

  StatsBomb   5-fold cross-validation grouped by match (the L2 penalty is
              chosen by the same grouped folds' log-loss)
  transfer    our Bundesliga tracking passes (scripts/build_pass_dataset.py),
              which the model never sees. This is the test that matters: the
              2026-09-18 StatsBomb model looked fine inside StatsBomb and
              over-predicted our passes by +0.10.

The features are fit_pass_candidates.b1_features -- the same code, and the
same expected_pass.pass_features, that builds B1 from our own passes -- so
the two differ only in the passes they are fitted on. Family is not
modelled: every pass is scored as ground.

The candidates' scores on the same Bundesliga passes (baseline, A, B1, all
out of match) are read from the prep job's report.json and printed alongside.

Usage:
    python scripts/fit_pass_b1sb.py --statsbomb data/processed/pass_models/statsbomb_passes.jsonl \
        --ours data/processed/pass_models/passes.jsonl --output data/processed/pass_models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from defensive_positioning.expected_pass import FEATURE_SETS, ExpectedPass
from fit_pass_candidates import (BASELINE, C_GRID, auc, b1_features, baseline_predictions, fit_logit,
                                 log_loss, probe, report, standardise)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--statsbomb", type=Path, required=True)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    sb = load(args.statsbomb)
    ours = load(args.ours)
    y = np.array([r["label"] for r in sb], dtype=float)
    groups = np.array([r["match_id"] for r in sb])
    dist = np.array([r["pass_distance"] for r in sb])
    print(f"StatsBomb 패스 {len(sb):,} · 성공률 {y.mean():.4f} · 경기 {len(set(groups))}", flush=True)
    x = b1_features(sb)
    assert x.shape[1] == len(FEATURE_SETS["velocity"])

    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    best = None
    for c in C_GRID:
        loss = 0.0
        for tr, te in folds:
            xa, xb, *_ = standardise(x[tr], x[te])
            loss += log_loss(fit_logit(xa, y[tr], c).predict_proba(xb)[:, 1], y[te]) * len(te)
        loss /= len(y)
        print(f"  C={c}: 경기 묶음 교차검증 logloss {loss:.4f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, c)
    c = best[1]
    oof = np.zeros(len(y))
    for tr, te in folds:
        xa, xb, *_ = standardise(x[tr], x[te])
        oof[te] = fit_logit(xa, y[tr], c).predict_proba(xb)[:, 1]

    xs, _, mean, scale, const = standardise(x, x)
    m = fit_logit(xs, y, c)
    w = m.coef_[0].copy()
    w[const] = 0.0
    model = ExpectedPass(mean, scale, w, float(m.intercept_[0]), {
        "kind": "fitted_logistic", "target_semantics": "intended_at_kick",
        "validation_status": "statsbomb_grouped_cv5_and_bundesliga_transfer",
        "feature_set": "velocity", "l2_C": c,
        "training": "StatsBomb 360 open-play passes, men's club competitions; velocities from the "
                    "previous freeze frame (mirrored when it belongs to the other team); targets "
                    "intended-at-kick from the pass direction only",
        "competitions": sorted({r["competition"] for r in sb}), "passes": len(sb),
        "matches": len(set(groups)), "passes_file": str(args.statsbomb)}, "velocity")
    path = args.output / "B1SB_all.json"
    model.save(path)
    ExpectedPass.load(path)          # must load with 준현's own loader, no proxy opt-in

    yo = np.array([r["label"] for r in ours], dtype=float)
    do = np.array([r["pass_distance"] for r in ours])
    p_transfer = model.probability(b1_features(ours))
    p_base = baseline_predictions(ours)
    rep = {"statsbomb_passes": len(sb), "statsbomb_completion": float(y.mean()), "C": c,
           "statsbomb_cv": report("B1-SB (StatsBomb, 경기 밖)", oof, y, dist),
           "transfer": [report("baseline", p_base, yo, do), report("B1-SB", p_transfer, yo, do)],
           "probe": probe({"baseline": ExpectedPass.load(BASELINE, allow_proxy=True), "B1-SB": model})}
    (args.output / "report_b1sb.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")

    cv = rep["statsbomb_cv"]
    print(f"\nStatsBomb 안 (경기 밖 5겹): AUC {cv['auc']:.3f} · Brier {cv['brier']:.4f} · 예측평균 {cv['mean_predicted']:.3f} (실제 {y.mean():.3f})")
    print(f"\n우리 분데스리가 패스 {len(ours):,}개 (성공률 {yo.mean():.4f}, 기저율 Brier {yo.mean() * (1 - yo.mean()):.4f}) — 전이 시험")
    print(f"  {'모델':<22}{'AUC':>7}{'Brier':>9}{'logloss':>9}{'예측평균':>9}")
    prep = args.output / "report.json"
    others = json.loads(prep.read_text()).get("out_of_match", []) if prep.exists() else []
    for r in [*others, rep["transfer"][1]]:
        tag = r["name"] + ("  (우리 패스, 경기 밖)" if r in others and r["name"] != "baseline" else "")
        print(f"  {tag:<22}{r['auc']:>7.3f}{r['brier']:>9.4f}{r['log_loss']:>9.4f}{r['mean_predicted']:>9.3f}")
    print("\n  길이 구간별 실제 / B1-SB 예측")
    for b in rep["transfer"][1]["bands"]:
        print(f"  {b['band']:<10}{b['n']:>5}  실제 {b['observed']:.3f} · 예측 {b['predicted']:.3f}")
    print("\n뒷공간 스루패스 시험 (러너 앞 8 m로 패스)")
    for row in rep["probe"]:
        print("  " + row["상황"] + " → " + " · ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "상황"))
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
