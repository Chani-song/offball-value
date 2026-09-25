#!/usr/bin/env python3
"""Train the xPass v0 baseline on StatsBomb open-data open-play passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from offball_value.xpass import FEATURE_NAMES, iter_match_passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/raw/statsbomb-open-data/data/events"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/xpass_v0"),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.events_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no event files under {args.events_dir}")

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    test_stride = max(2, int(round(1.0 / max(args.test_fraction, 1e-9))))
    for index, path in enumerate(files):
        # Deterministic match-level split prevents within-match leakage.
        is_test = index % test_stride == 0
        for features, label in iter_match_passes(path):
            if is_test:
                test_x.append(features)
                test_y.append(label)
            else:
                train_x.append(features)
                train_y.append(label)
        if (index + 1) % 250 == 0:
            print(
                f"[{index + 1}/{len(files)}] passes so far: "
                f"train {len(train_x)}, test {len(test_x)}",
                flush=True,
            )

    x_train = np.stack(train_x)
    y_train = np.asarray(train_y)
    x_test = np.stack(test_x)
    y_test = np.asarray(test_y)
    print(
        f"dataset: train {len(y_train)} / test {len(y_test)} passes, "
        f"completion rate train {y_train.mean():.4f} test {y_test.mean():.4f}"
    )

    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.08,
            max_iter=400,
            early_stopping=True,
            random_state=0,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "feature_names": list(FEATURE_NAMES),
        "train_passes": int(len(y_train)),
        "test_passes": int(len(y_test)),
        "train_completion_rate": float(y_train.mean()),
        "test_completion_rate": float(y_test.mean()),
        "models": {},
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, probabilities))
        brier = float(brier_score_loss(y_test, probabilities))
        bins = np.linspace(0.0, 1.0, args.calibration_bins + 1)
        calibration = []
        for low, high in zip(bins, bins[1:]):
            mask = (probabilities >= low) & (probabilities < high)
            if mask.sum() == 0:
                continue
            calibration.append(
                {
                    "bin": [float(low), float(high)],
                    "count": int(mask.sum()),
                    "predicted": float(probabilities[mask].mean()),
                    "observed": float(y_test[mask].mean()),
                }
            )
        report["models"][name] = {  # type: ignore[index]
            "auc": auc,
            "brier": brier,
            "calibration": calibration,
        }
        joblib.dump(model, args.output_dir / f"xpass_{name}.joblib")
        print(f"{name}: AUC {auc:.4f} · Brier {brier:.4f}")

    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"saved models and report to {args.output_dir}")


if __name__ == "__main__":
    main()
