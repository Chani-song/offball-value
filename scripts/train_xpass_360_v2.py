#!/usr/bin/env python3
"""Defender-aware xPass, with the counterfactual probe fixed and two ablations.

THE PROBE WAS BROKEN. It claimed to "insert a synthetic blocker at the lane
midpoint" but actually edited three feature columns by hand — lane distance
and the two blocker counts — and left the rest at their open-lane values. The
model was therefore shown a pass whose lane is blocked at 0.5 m while the
passer's nearest opponent is still 7 m away and the visible-opponent count is
unchanged: a combination that cannot occur, least of all on a short pass. The
resulting 2 % drop was read as the model being insensitive, and that reading
propagated into the conclusion that the binding constraint is the
data-generating process rather than anything fixable.

Placing an actual point at the midpoint and recomputing every lane feature
from it gives +7.7 % on the same model and the same passes. The defender
interaction was always there; the measurement was not.

Two ablations ride along, both of which the reviewer identified:

--no-recipient  StatsBomb records end_location as where the ball STOPPED, so
                for an incomplete pass that is the interception point, and the
                blocker who caused it is recorded at the lane's END rather than
                in the middle of it. 360 freeze frames carry no player
                identity, so the intended receiver cannot be looked up; the
                only options are to leave it (observed) or approximate it by
                the teammate nearest the recorded end (nearest).

--drop-choice   height and under_pressure are not inputs we have. Asking "if
                the ball were played there, would it arrive" does not come with
                a decision about whether to loft it — that is part of what the
                passer chooses in response to the very defenders we are moving.
                Worse, height carries the difficulty signal directly (41 % of
                blocked passes are aerial against 2 % of open ones), so the
                model can route difficulty around the geometry. Dropping them
                forces the geometry to carry it.
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

CHOICE_FEATURES = ("height_ground", "height_low", "height_high", "under_pressure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/statsbomb-open-data/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument(
        "--no-recipient",
        choices=("observed", "nearest"),
        default="observed",
        help="Where a failed pass was aimed: leave the recorded end, or the "
        "teammate nearest it.",
    )
    parser.add_argument(
        "--drop-choice",
        action="store_true",
        help="Drop height and under_pressure — decisions, not inputs.",
    )
    return parser.parse_args()


def intended_target(event, freeze, mode: str) -> tuple[float, float]:
    end_location = event["pass"]["end_location"]
    observed = (float(end_location[0]), float(end_location[1]))
    if mode == "observed" or "outcome" not in event["pass"]:
        return observed
    mates = [
        (float(p["location"][0]), float(p["location"][1]))
        for p in freeze
        if p.get("teammate", False) and not p.get("actor", False)
    ]
    if not mates:
        return observed
    return min(
        mates,
        key=lambda xy: (xy[0] - observed[0]) ** 2 + (xy[1] - observed[1]) ** 2,
    )


def collect(args):
    frames_dir = args.data_root / "three-sixty"
    events_dir = args.data_root / "events"
    matches = sorted(frames_dir.glob("*.json"))
    if args.max_matches:
        matches = matches[: args.max_matches]
    if not matches:
        raise SystemExit(f"no 360 files under {frames_dir}")

    train, test, retargeted, skipped = [], [], 0, 0
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
        bucket = test if index % args.test_stride == 0 else train
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
            end = intended_target(event, freeze, args.no_recipient)
            if args.no_recipient == "nearest" and "outcome" in event["pass"]:
                retargeted += 1
            # The raw geometry is kept so the probe can rebuild features from
            # real coordinates rather than editing columns.
            bucket.append((base, start, end, opponents, label))
        if (index + 1) % 50 == 0:
            print(f"[{index + 1}/{len(matches)}] train {len(train)} test {len(test)}", flush=True)
    return train, test, retargeted, skipped


def assemble(rows, keep):
    x = np.stack(
        [np.concatenate([b, lane_features(s, e, o)])[keep] for b, s, e, o, _ in rows]
    )
    return x, np.asarray([r[-1] for r in rows])


def main() -> None:
    args = parse_args()
    train, test, retargeted, skipped = collect(args)

    names = list(FEATURE_NAMES_360)
    keep = np.array(
        [i for i, n in enumerate(names) if not (args.drop_choice and n in CHOICE_FEATURES)]
    )
    kept_names = [names[i] for i in keep]

    x_train, y_train = assemble(train, keep)
    x_test, y_test = assemble(test, keep)
    print(
        f"dataset: train {len(y_train)} / test {len(y_test)} · "
        f"completion {y_train.mean():.4f}/{y_test.mean():.4f} · "
        f"features {len(kept_names)} · retargeted {retargeted} · skipped {skipped}"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=500, early_stopping=True, random_state=0
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    brier = float(brier_score_loss(y_test, probabilities))
    print(f"AUC {auc:.4f} · Brier {brier:.4f}")

    # Counterfactual probe, done properly: put a defender at a real coordinate
    # beside the lane midpoint and recompute every lane feature from it, so
    # passer distance, target distance and the visible count all move together.
    lane_index = kept_names.index("lane_min_perpendicular")
    open_rows = [r for r in test if lane_features(r[1], r[2], r[3])[1] > 6.0][:2000]
    probe_result = {}
    if open_rows:
        x_open, _ = assemble(open_rows, keep)
        before = model.predict_proba(x_open)[:, 1]
        rebuilt = []
        for base, start, end, opponents, _ in open_rows:
            dx, dy = end[0] - start[0], end[1] - start[1]
            norm = float(np.hypot(dx, dy)) + 1e-9
            mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            blocker = (mid[0] - dy / norm * 0.5, mid[1] + dx / norm * 0.5)
            rebuilt.append(
                np.concatenate(
                    [base, lane_features(start, end, [*opponents, blocker])]
                )[keep]
            )
        after = model.predict_proba(np.stack(rebuilt))[:, 1]
        # The old, broken form, for comparison on the same rows.
        hand = x_open.copy()
        hand[:, lane_index] = 0.5
        for column in ("lane_blockers_2m", "lane_blockers_4m"):
            hand[:, kept_names.index(column)] += 1.0
        hand_after = model.predict_proba(hand)[:, 1]
        probe_result = {
            "open_passes": int(len(open_rows)),
            "predicted_before": float(before.mean()),
            "counterfactual_drop_rebuilt": float((before - after).mean()),
            "counterfactual_drop_hand_edited": float((before - hand_after).mean()),
        }
        print(
            f"counterfactual drop — rebuilt {probe_result['counterfactual_drop_rebuilt']:+.4f} · "
            f"hand-edited {probe_result['counterfactual_drop_hand_edited']:+.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "xpass_360_hist_gbdt.joblib")
    (args.output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "feature_names": kept_names,
                "no_recipient": args.no_recipient,
                "drop_choice": bool(args.drop_choice),
                "train_passes": int(len(y_train)),
                "test_passes": int(len(y_test)),
                "auc": auc,
                "brier": brier,
                "probe": probe_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
