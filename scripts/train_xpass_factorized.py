#!/usr/bin/env python3
"""Factorized xPass: frozen geometry model × learned defender-influence term.

Tests the proposal P = f_geo(geometry) × g_def(defenders): freeze the
geometry-only xPass, then fit the defender term on the RESIDUAL via a
logistic offset model,

    P = sigmoid( logit(f_geo) + w · defender_features + b ),

so the defender features cannot share credit with geometry.  The pass/fail
criterion is the counterfactual lane-blocker probe: inserting a synthetic
blocker into an open lane must move the prediction by a physically plausible
amount (the monolithic 360 model managed only ~2 points against an observed
44-point gradient).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from offball_value.xpass import (
    FEATURE_NAMES_360,
    event_features_and_label,
    is_open_play_pass,
    lane_features,
)

DEFENDER_FEATURES = FEATURE_NAMES_360[len(FEATURE_NAMES_360) - 6 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/statsbomb-open-data/data"),
    )
    parser.add_argument(
        "--geometry-model",
        type=Path,
        default=Path("data/processed/xpass_v0/xpass_hist_gbdt.joblib"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/xpass_factorized"),
    )
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    return parser.parse_args()


def fit_logistic_offset(
    offset: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    """Full-batch gradient fit of sigmoid(offset + w·x + b)."""

    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = features / scale
    weights = np.zeros(features.shape[1])
    bias = 0.0
    n = len(labels)
    for _ in range(epochs):
        logits = offset + normalized @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = probabilities - labels
        weights -= learning_rate * (normalized.T @ error) / n
        bias -= learning_rate * float(error.mean())
    return weights / scale, bias


def main() -> None:
    args = parse_args()
    geometry_model = joblib.load(args.geometry_model)
    frames_dir = args.data_root / "three-sixty"
    events_dir = args.data_root / "events"

    base_train: list[np.ndarray] = []
    def_train: list[np.ndarray] = []
    y_train: list[int] = []
    base_test: list[np.ndarray] = []
    def_test: list[np.ndarray] = []
    y_test: list[int] = []
    skipped = 0
    matches = sorted(frames_dir.glob("*.json"))
    for index, frames_path in enumerate(matches):
        events_path = events_dir / frames_path.name
        if not events_path.exists():
            continue
        try:
            frames = {
                str(row["event_uuid"]): row["freeze_frame"]
                for row in json.loads(frames_path.read_text(encoding="utf-8"))
                if row.get("freeze_frame")
            }
            events = json.loads(events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        is_test = index % args.test_stride == 0
        for event in events:
            if not is_open_play_pass(event):
                continue
            freeze = frames.get(str(event.get("id")))
            if not freeze:
                continue
            base, label = event_features_and_label(event)
            opponents = [
                (float(p["location"][0]), float(p["location"][1]))
                for p in freeze
                if not p.get("teammate", False)
            ]
            start = (float(event["location"][0]), float(event["location"][1]))
            end_location = event["pass"]["end_location"]
            end = (float(end_location[0]), float(end_location[1]))
            defender = lane_features(start, end, opponents)
            if is_test:
                base_test.append(base)
                def_test.append(defender)
                y_test.append(label)
            else:
                base_train.append(base)
                def_train.append(defender)
                y_train.append(label)

    xb_train, xd_train = np.stack(base_train), np.stack(def_train)
    xb_test, xd_test = np.stack(base_test), np.stack(def_test)
    yt, yv = np.asarray(y_train), np.asarray(y_test)
    print(f"train {len(yt)} / test {len(yv)} (skipped files {skipped})")

    def geo_logit(x: np.ndarray) -> np.ndarray:
        p = np.clip(geometry_model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    offset_train = geo_logit(xb_train)
    offset_test = geo_logit(xb_test)

    weights, bias = fit_logistic_offset(
        offset_train, xd_train, yt, args.epochs, args.learning_rate
    )

    def predict(offset: np.ndarray, defenders: np.ndarray) -> np.ndarray:
        logits = offset + defenders @ weights + bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    probabilities = predict(offset_test, xd_test)
    auc = float(roc_auc_score(yv, probabilities))
    brier = float(brier_score_loss(yv, probabilities))
    geometry_auc = float(
        roc_auc_score(yv, 1 / (1 + np.exp(-offset_test)))
    )
    print(f"factorized: AUC {auc:.4f} (geometry alone {geometry_auc:.4f}) · Brier {brier:.4f}")
    print("defender weights:")
    for name, weight in zip(DEFENDER_FEATURES, weights):
        print(f"  {name:<24} {weight:+.4f}")

    # Counterfactual lane-blocker probe (same protocol as the monolithic 360).
    lane_index = list(DEFENDER_FEATURES).index("lane_min_perpendicular")
    open_mask = xd_test[:, lane_index] > 6.0
    sample = np.flatnonzero(open_mask)[:2000]
    before = predict(offset_test[sample], xd_test[sample])
    blocked = xd_test[sample].copy()
    blocked[:, lane_index] = 0.5
    blocked[:, list(DEFENDER_FEATURES).index("lane_blockers_2m")] += 1.0
    blocked[:, list(DEFENDER_FEATURES).index("lane_blockers_4m")] += 1.0
    after = predict(offset_test[sample], blocked)
    drop = float((before - after).mean())
    print(f"counterfactual lane-blocker drop: {drop:+.4f}  (통짜 360: +0.0195)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "factorized_report.json").write_text(
        json.dumps(
            {
                "defender_features": list(DEFENDER_FEATURES),
                "weights": [float(w) for w in weights],
                "bias": float(bias),
                "auc": auc,
                "geometry_only_auc": geometry_auc,
                "brier": brier,
                "counterfactual_lane_blocker_drop": drop,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
