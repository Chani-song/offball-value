#!/usr/bin/env python3
"""Train the defender-aware xPass against the INTENDED pass target.

StatsBomb records ``pass.end_location`` as where the ball actually finished,
so for an incomplete pass that is the interception point, not where the passer
aimed. Training on it inverts the causality the audit needs: the blocker who
cut the ball out is recorded standing at the lane's END, so the model learns
"a defender near the target means failure" from the very defender who caused
it. That is presumably why the counterfactual probe - drop a blocker into an
open lane - moves the prediction by only about two points against an observed
forty-four point gradient, and why two different architectures stalled at the
same place.

Around 60% of failed passes name ``pass.recipient``, and the 360 freeze frame
carries that player's position. For those the intended target is known
exactly, not approximated: the lane becomes passer -> intended receiver and
the interceptor sits in the MIDDLE of it, which is the geometry a defender-
aware model has to learn from.

Failures with no recipient are handled by --no-recipient:
  drop     - leave them out (default; no invented geometry)
  observed - keep the recorded end location, as the original script did
  nearest  - aim at the teammate nearest the recorded end location

Completed passes are untouched: end_location is already the receiver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from offball_value.xpass import (
    FEATURE_NAMES_360,
    event_features_and_label,
    is_open_play_pass,
    lane_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/statsbomb-open-data/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/xpass_360_intent"),
    )
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument(
        "--no-recipient",
        choices=("drop", "observed", "nearest"),
        default="drop",
        help="What to do with failed passes that name no recipient.",
    )
    return parser.parse_args()


def intended_target(
    event, freeze, mode: str
) -> tuple[float, float] | None:
    """Where the passer was aiming, as opposed to where the ball stopped."""
    pass_data = event["pass"]
    end_location = pass_data["end_location"]
    observed = (float(end_location[0]), float(end_location[1]))
    if "outcome" not in pass_data:
        return observed  # completed: the end location IS the receiver
    recipient = pass_data.get("recipient") or {}
    recipient_id = recipient.get("id")
    if recipient_id is not None:
        for player in freeze:
            if not player.get("teammate", False):
                continue
            actor = player.get("player") or {}
            if actor.get("id") == recipient_id:
                location = player["location"]
                return (float(location[0]), float(location[1]))
    if mode == "observed":
        return observed
    if mode == "nearest":
        mates = [
            (float(p["location"][0]), float(p["location"][1]))
            for p in freeze
            if p.get("teammate", False)
        ]
        if not mates:
            return None
        return min(
            mates,
            key=lambda xy: (xy[0] - observed[0]) ** 2 + (xy[1] - observed[1]) ** 2,
        )
    return None


def main() -> None:
    args = parse_args()
    frames_dir = args.data_root / "three-sixty"
    events_dir = args.data_root / "events"
    matches = sorted(frames_dir.glob("*.json"))
    if not matches:
        raise SystemExit(f"no 360 files under {frames_dir}")

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    skipped_files = 0
    retargeted = 0
    dropped_no_recipient = 0
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
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            skipped_files += 1
            print(f"[skip] corrupt file for {frames_path.name}: {error}")
            continue
        is_test = index % args.test_stride == 0
        for event in events:
            if not is_open_play_pass(event):
                continue
            freeze = frames.get(str(event.get("id")))
            if not freeze:
                continue
            end = intended_target(event, freeze, args.no_recipient)
            if end is None:
                dropped_no_recipient += 1
                continue
            base, label = event_features_and_label(event)
            opponents = [
                (float(player["location"][0]), float(player["location"][1]))
                for player in freeze
                if not player.get("teammate", False)
            ]
            start = (float(event["location"][0]), float(event["location"][1]))
            if "outcome" in event["pass"]:
                retargeted += 1
            features = np.concatenate([base, lane_features(start, end, opponents)])
            if is_test:
                test_x.append(features)
                test_y.append(label)
            else:
                train_x.append(features)
                train_y.append(label)
        if (index + 1) % 50 == 0:
            print(
                f"[{index + 1}/{len(matches)}] passes: "
                f"train {len(train_x)}, test {len(test_x)}",
                flush=True,
            )

    x_train = np.stack(train_x)
    y_train = np.asarray(train_y)
    x_test = np.stack(test_x)
    y_test = np.asarray(test_y)
    print(
        f"dataset: train {len(y_train)} / test {len(y_test)} passes, "
        f"completion train {y_train.mean():.4f} test {y_test.mean():.4f} "
        f"(corrupt files skipped: {skipped_files})"
    )
    print(
        f"failed passes re-aimed at the intended target: {retargeted} · "
        f"dropped for having no recipient: {dropped_no_recipient} "
        f"(policy: {args.no_recipient})"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=500,
        early_stopping=True,
        random_state=0,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    brier = float(brier_score_loss(y_test, probabilities))
    print(f"hist_gbdt(360): AUC {auc:.4f} · Brier {brier:.4f}")

    # Lane-sensitivity probe: does the model react to a blocker in the lane?
    lane_index = FEATURE_NAMES_360.index("lane_min_perpendicular")
    probe_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 25.1)]
    probe = []
    for low, high in probe_bins:
        mask = (x_test[:, lane_index] >= low) & (x_test[:, lane_index] < high)
        if mask.sum() == 0:
            continue
        probe.append(
            {
                "lane_min_perp_bin": [low, high],
                "count": int(mask.sum()),
                "predicted_mean": float(probabilities[mask].mean()),
                "observed_mean": float(y_test[mask].mean()),
            }
        )
    # Counterfactual probe: insert a synthetic blocker at the lane midpoint
    # of unblocked test passes and measure the predicted drop — this is the
    # property the audit needs (a defender path must move P).
    unblocked = x_test[x_test[:, lane_index] > 6.0][:2000].copy()
    before = model.predict_proba(unblocked)[:, 1]
    blocked = unblocked.copy()
    blocked[:, lane_index] = 0.5
    blocked[:, FEATURE_NAMES_360.index("lane_blockers_2m")] += 1.0
    blocked[:, FEATURE_NAMES_360.index("lane_blockers_4m")] += 1.0
    after = model.predict_proba(blocked)[:, 1]
    counterfactual_drop = float((before - after).mean())
    print(f"counterfactual lane-blocker drop: {counterfactual_drop:+.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "xpass_360_hist_gbdt.joblib")
    (args.output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES_360),
                "train_passes": int(len(y_train)),
                "test_passes": int(len(y_test)),
                "auc": auc,
                "brier": brier,
                "lane_sensitivity": probe,
                "counterfactual_lane_blocker_drop": counterfactual_drop,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
