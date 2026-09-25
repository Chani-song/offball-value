"""Expected-pass (xPass) features from StatsBomb open-data events.

The 2026-08-28 lab meeting decided to replace the mechanistic delivery term
with a learned pass-completion model.  This module keeps the feature
definition in one place so training and later in-audit inference cannot
drift apart.

Deliberate v0 scope: only features that are also computable for a
counterfactual pass target inside the local-game audit (geometry, chosen
pass height, a pressure flag).  Body part, technique, and 360 freeze-frame
features are excluded for now because the audit cannot supply them for
synthetic targets.  StatsBomb event coordinates are already normalized so
the acting team attacks toward x = 120.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np


PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_XY = (120.0, 40.0)

SET_PIECE_PASS_TYPES = {
    "Corner",
    "Free Kick",
    "Goal Kick",
    "Kick Off",
    "Throw-in",
}

HEIGHT_NAMES = ("Ground Pass", "Low Pass", "High Pass")

FEATURE_NAMES = (
    "start_x",
    "start_y_centered_abs",
    "end_x",
    "end_y_centered_abs",
    "length",
    "angle_cos",
    "angle_sin",
    "forward_progress",
    "lateral_shift",
    "start_goal_distance",
    "end_goal_distance",
    "height_ground",
    "height_low",
    "height_high",
    "under_pressure",
)


def is_open_play_pass(event: Mapping[str, object]) -> bool:
    """Keep regular open-play passes with usable coordinates."""

    if event.get("type", {}).get("name") != "Pass":  # type: ignore[union-attr]
        return False
    pass_data = event.get("pass", {})
    if not isinstance(pass_data, Mapping):
        return False
    pass_type = pass_data.get("type", {})
    if isinstance(pass_type, Mapping) and pass_type.get("name") in SET_PIECE_PASS_TYPES:
        return False
    location = event.get("location")
    end_location = pass_data.get("end_location")
    return (
        isinstance(location, Sequence)
        and len(location) >= 2
        and isinstance(end_location, Sequence)
        and len(end_location) >= 2
    )


def pass_features(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    height_name: str,
    under_pressure: bool,
) -> np.ndarray:
    """Feature vector shared by training and counterfactual inference."""

    start_x, start_y = float(start_xy[0]), float(start_xy[1])
    end_x, end_y = float(end_xy[0]), float(end_xy[1])
    dx, dy = end_x - start_x, end_y - start_y
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) if length > 1e-9 else 0.0
    return np.asarray(
        [
            start_x,
            abs(start_y - PITCH_WIDTH / 2.0),
            end_x,
            abs(end_y - PITCH_WIDTH / 2.0),
            length,
            math.cos(angle),
            math.sin(angle),
            dx,
            abs(dy),
            math.hypot(GOAL_XY[0] - start_x, GOAL_XY[1] - start_y),
            math.hypot(GOAL_XY[0] - end_x, GOAL_XY[1] - end_y),
            1.0 if height_name == "Ground Pass" else 0.0,
            1.0 if height_name == "Low Pass" else 0.0,
            1.0 if height_name == "High Pass" else 0.0,
            1.0 if under_pressure else 0.0,
        ],
        dtype=np.float64,
    )


def event_features_and_label(
    event: Mapping[str, object],
) -> tuple[np.ndarray, int]:
    """StatsBomb pass event -> (features, completed label)."""

    pass_data: Mapping[str, object] = event["pass"]  # type: ignore[assignment]
    location: Sequence[float] = event["location"]  # type: ignore[assignment]
    end_location: Sequence[float] = pass_data["end_location"]  # type: ignore[assignment]
    height = pass_data.get("height", {})
    height_name = (
        str(height.get("name", "Ground Pass"))
        if isinstance(height, Mapping)
        else "Ground Pass"
    )
    features = pass_features(
        (float(location[0]), float(location[1])),
        (float(end_location[0]), float(end_location[1])),
        height_name,
        bool(event.get("under_pressure", False)),
    )
    # StatsBomb convention: a completed pass has no `outcome` entry.
    label = 0 if "outcome" in pass_data else 1
    return features, label


def iter_match_passes(
    events_path: Path,
) -> Iterator[tuple[np.ndarray, int]]:
    events = json.loads(events_path.read_text(encoding="utf-8"))
    for event in events:
        if is_open_play_pass(event):
            yield event_features_and_label(event)


# ---------------------------------------------------------------------------
# 360 freeze-frame features (defender-aware model)
# ---------------------------------------------------------------------------

FEATURE_NAMES_360 = FEATURE_NAMES + (
    "passer_nearest_opponent",
    "lane_min_perpendicular",
    "lane_blockers_2m",
    "lane_blockers_4m",
    "target_nearest_opponent",
    "visible_opponents",
)


def lane_features(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    opponents: Sequence[tuple[float, float]],
    distance_cap: float = 25.0,
) -> np.ndarray:
    """Defender-interaction features for one pass lane.

    These are exactly the quantities a defender-aware xPass needs so a
    counterfactual defender path can move the predicted probability: pressure
    on the passer, blockers inside the lane corridor, and cover at the
    reception point.  Missing opponents (nobody visible) fall back to the
    distance cap, i.e. "no pressure".
    """

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    lane = end - start
    lane_length = float(np.hypot(*lane))
    passer_nearest = distance_cap
    lane_min_perpendicular = distance_cap
    blockers_2m = 0
    blockers_4m = 0
    target_nearest = distance_cap
    for opponent in opponents:
        position = np.asarray(opponent, dtype=np.float64)
        passer_nearest = min(passer_nearest, float(np.hypot(*(position - start))))
        target_nearest = min(target_nearest, float(np.hypot(*(position - end))))
        if lane_length > 1e-9:
            projection = float(np.dot(position - start, lane) / lane_length**2)
            if 0.0 < projection < 1.0:
                perpendicular = float(
                    np.hypot(*(position - (start + projection * lane)))
                )
                lane_min_perpendicular = min(lane_min_perpendicular, perpendicular)
                if perpendicular <= 2.0:
                    blockers_2m += 1
                if perpendicular <= 4.0:
                    blockers_4m += 1
    return np.asarray(
        [
            min(passer_nearest, distance_cap),
            min(lane_min_perpendicular, distance_cap),
            float(blockers_2m),
            float(blockers_4m),
            min(target_nearest, distance_cap),
            float(len(opponents)),
        ],
        dtype=np.float64,
    )


def pass_features_360(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    height_name: str,
    under_pressure: bool,
    opponents: Sequence[tuple[float, float]],
) -> np.ndarray:
    return np.concatenate(
        [
            pass_features(start_xy, end_xy, height_name, under_pressure),
            lane_features(start_xy, end_xy, opponents),
        ]
    )


# ---------------------------------------------------------------------------
# Inference inside the local-game audit
# ---------------------------------------------------------------------------

AUDIT_PITCH_LENGTH = 105.0
AUDIT_PITCH_WIDTH = 68.0

_MODEL_CACHE: dict[str, object] = {}


def to_statsbomb_coordinates(
    xy: tuple[float, float],
    attacking_direction: int,
) -> tuple[float, float]:
    """Centered IDSSE metres -> StatsBomb attack-left-to-right frame.

    The audit uses pitch-centered coordinates with a per-scene attacking
    direction; StatsBomb normalizes every acting team to attack toward
    x = 120.  The y mirror is irrelevant because the v0 features only use
    centered-|y| quantities.
    """

    direction = 1 if int(attacking_direction) >= 0 else -1
    x_along = direction * float(xy[0]) + AUDIT_PITCH_LENGTH / 2.0
    y_across = float(xy[1]) + AUDIT_PITCH_WIDTH / 2.0
    return (
        float(np.clip(x_along, 0.0, AUDIT_PITCH_LENGTH))
        * PITCH_LENGTH
        / AUDIT_PITCH_LENGTH,
        float(np.clip(y_across, 0.0, AUDIT_PITCH_WIDTH))
        * PITCH_WIDTH
        / AUDIT_PITCH_WIDTH,
    )


def load_xpass_model(model_path: str | Path):
    key = str(model_path)
    if key not in _MODEL_CACHE:
        import joblib

        model = joblib.load(model_path)
        # An ablated model was trained on a subset of FEATURE_NAMES_360; the
        # report written beside it says which, so inference can build the full
        # feature row as usual and then drop the columns the model never saw.
        report = Path(model_path).with_name("training_report.json")
        if report.exists():
            names = json.loads(report.read_text(encoding="utf-8")).get(
                "feature_names"
            )
            if names and list(names) != list(FEATURE_NAMES_360):
                index = {name: i for i, name in enumerate(FEATURE_NAMES_360)}
                model._offball_feature_columns = np.asarray(
                    [index[name] for name in names], dtype=int
                )
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def predict_pass_success_360(
    model,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    attacking_direction: int,
    opponents: Sequence[tuple[float, float]],
    height_name: str = "Ground Pass",
    under_pressure: bool = False,
) -> float:
    """Defender-aware xPass for one audit-frame pass candidate.

    Unlike the geometry-only model this one takes the defending side's
    positions, so a counterfactual defender path moves the prediction
    directly and the mechanistic interaction terms are not needed — which
    also removes the double-count the hybrid mode accepted as a v0 bias.

    ``opponents`` are audit-frame coordinates and are converted alongside the
    pass endpoints, so the caller passes raw pitch positions.
    """
    features = pass_features_360(
        to_statsbomb_coordinates(start_xy, attacking_direction),
        to_statsbomb_coordinates(end_xy, attacking_direction),
        height_name,
        under_pressure,
        [to_statsbomb_coordinates(xy, attacking_direction) for xy in opponents],
    )
    columns = getattr(model, "_offball_feature_columns", None)
    if columns is not None:
        features = features[columns]
    probability = model.predict_proba(features.reshape(1, -1))[0, 1]
    return float(np.clip(probability, 0.0, 1.0))


def predict_pass_success(
    model,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    attacking_direction: int,
    height_name: str = "Ground Pass",
    under_pressure: bool = False,
) -> float:
    """xPass probability for one audit-frame pass candidate."""

    features = pass_features(
        to_statsbomb_coordinates(start_xy, attacking_direction),
        to_statsbomb_coordinates(end_xy, attacking_direction),
        height_name,
        under_pressure,
    )
    probability = model.predict_proba(features.reshape(1, -1))[0, 1]
    return float(np.clip(probability, 0.0, 1.0))


def predict_pass_success_360_kinematic(
    model,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    attacking_direction: int,
    players,
    height_name: str = "Ground Pass",
    under_pressure: bool = False,
) -> float:
    """xpass360's features plus the arrival race, for one audit-frame candidate.

    ``players`` are ``kinematic_xpass.TrackedPlayer`` in AUDIT-FRAME metres
    (centred pitch), while the static half is converted to StatsBomb units as
    usual. Both are correct: the static features were fitted in pitch units and
    the kinematic ones only ever use distances and differences, which are
    metres on either convention.

    Feature order must match training exactly -- FEATURE_NAMES_360 then
    KINEMATIC_FEATURE_NAMES -- so both sides build it from the same module.
    """
    from .kinematic_xpass import kinematic_features

    static = pass_features_360(
        to_statsbomb_coordinates(start_xy, attacking_direction),
        to_statsbomb_coordinates(end_xy, attacking_direction),
        height_name,
        under_pressure,
        [
            to_statsbomb_coordinates((p.x, p.y), attacking_direction)
            for p in players
            if not p.teammate
        ],
    )
    kinematic = kinematic_features(list(players), start_xy, end_xy)
    features = np.concatenate([static, kinematic])
    columns = getattr(model, "_offball_feature_columns", None)
    if columns is not None:
        features = features[columns]
    probability = model.predict_proba(features.reshape(1, -1))[0, 1]
    return float(np.clip(probability, 0.0, 1.0))
