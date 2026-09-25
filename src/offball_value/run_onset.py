"""Retrospective detection of meaningful off-ball movement onsets.

An onset is the earliest kinematic departure from the runner's preceding
movement state, not the first crossing of an arbitrary absolute speed.  Future
samples are used only to confirm that the detected change persists; these
episodes are retrospective analysis labels and are never optimizer inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    BundesligaFrame,
    BundesligaMatchMeta,
)


@dataclass(frozen=True)
class RunOnsetConfig:
    smoothing_seconds: float = 0.20
    pre_window_seconds: float = 0.50
    post_confirmation_seconds: float = 0.80
    displacement_horizon_seconds: float = 1.00
    minimum_speed_gain_mps: float = 1.50
    minimum_direction_change_degrees: float = 30.0
    direction_minimum_speed_mps: float = 2.0
    minimum_post_displacement_m: float = 3.0
    acceleration_onset_threshold_mps2: float = 0.75
    deceleration_onset_threshold_mps2: float = 1.00
    turn_rate_onset_threshold_degrees_s: float = 35.0
    minimum_deceleration_drop_mps: float = 0.80
    minimum_reacceleration_mps: float = 0.60
    deceleration_turn_degrees: float = 60.0
    merge_seconds: float = 0.50
    controlled_history_seconds: float = 0.50
    future_possession_seconds: float = 1.00
    control_distance_m: float = 1.50
    future_control_distance_m: float = 2.50
    minimum_stable_control_fraction: float = 0.80
    minimum_future_known_fraction: float = 0.50
    minimum_future_same_team_fraction: float = 0.80
    required_player_count: int | None = 22
    require_ball_inside_pitch: bool = True
    pitch_boundary_tolerance_m: float = 0.20

    def validate(self) -> None:
        positive = {
            "smoothing_seconds": self.smoothing_seconds,
            "pre_window_seconds": self.pre_window_seconds,
            "post_confirmation_seconds": self.post_confirmation_seconds,
            "displacement_horizon_seconds": self.displacement_horizon_seconds,
            "minimum_speed_gain_mps": self.minimum_speed_gain_mps,
            "minimum_direction_change_degrees": self.minimum_direction_change_degrees,
            "direction_minimum_speed_mps": self.direction_minimum_speed_mps,
            "minimum_post_displacement_m": self.minimum_post_displacement_m,
            "acceleration_onset_threshold_mps2": self.acceleration_onset_threshold_mps2,
            "deceleration_onset_threshold_mps2": self.deceleration_onset_threshold_mps2,
            "turn_rate_onset_threshold_degrees_s": self.turn_rate_onset_threshold_degrees_s,
            "merge_seconds": self.merge_seconds,
            "controlled_history_seconds": self.controlled_history_seconds,
            "future_possession_seconds": self.future_possession_seconds,
            "control_distance_m": self.control_distance_m,
            "future_control_distance_m": self.future_control_distance_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Run-onset parameters must be positive: {invalid}")
        for name, value in (
            ("minimum_stable_control_fraction", self.minimum_stable_control_fraction),
            ("minimum_future_known_fraction", self.minimum_future_known_fraction),
            ("minimum_future_same_team_fraction", self.minimum_future_same_team_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.pitch_boundary_tolerance_m < 0.0:
            raise ValueError("pitch_boundary_tolerance_m must be non-negative")


@dataclass(frozen=True)
class KinematicRunOnset:
    frame_id: int
    labels: tuple[str, ...]
    x: float
    y: float
    onset_speed_mps: float
    pre_mean_speed_mps: float
    post_mean_speed_mps: float
    speed_gain_mps: float
    direction_change_degrees: float
    post_displacement_m: float
    peak_acceleration_mps2: float
    peak_deceleration_mps2: float
    confidence_score: float


@dataclass(frozen=True)
class RunOnsetCandidate:
    match_id: str
    frame_id: int
    player_id: str
    team_id: str
    ball_carrier_id: str
    labels: tuple[str, ...]
    x: float
    y: float
    onset_speed_mps: float
    pre_mean_speed_mps: float
    post_mean_speed_mps: float
    speed_gain_mps: float
    direction_change_degrees: float
    post_displacement_m: float
    peak_acceleration_mps2: float
    peak_deceleration_mps2: float
    confidence_score: float
    ball_distance_m: float
    stable_control_fraction: float
    future_known_fraction: float
    future_same_team_fraction: float

    @property
    def primary_type(self) -> str:
        if "deceleration_turn" in self.labels:
            return "deceleration_turn"
        if "direction_change" in self.labels:
            return "direction_change"
        return "acceleration"

    def as_record(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "frame_id": self.frame_id,
            "player_id": self.player_id,
            "team_id": self.team_id,
            "ball_carrier_id": self.ball_carrier_id,
            "onset_labels": "|".join(self.labels),
            "primary_type": self.primary_type,
            "x": self.x,
            "y": self.y,
            "onset_speed_mps": self.onset_speed_mps,
            "pre_mean_speed_mps": self.pre_mean_speed_mps,
            "post_mean_speed_mps": self.post_mean_speed_mps,
            "speed_gain_mps": self.speed_gain_mps,
            "direction_change_degrees": self.direction_change_degrees,
            "post_displacement_m": self.post_displacement_m,
            "peak_acceleration_mps2": self.peak_acceleration_mps2,
            "peak_deceleration_mps2": self.peak_deceleration_mps2,
            "confidence_score": self.confidence_score,
            "ball_distance_m": self.ball_distance_m,
            "stable_control_fraction": self.stable_control_fraction,
            "future_known_fraction": self.future_known_fraction,
            "future_same_team_fraction": self.future_same_team_fraction,
        }


def _odd_window(seconds: float) -> int:
    value = max(1, int(round(seconds * FPS)))
    return value if value % 2 == 1 else value + 1


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float, copy=True)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _angle_between_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-9 or second_norm < 1e-9:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def _rising_edges(mask: np.ndarray) -> list[int]:
    return [
        index
        for index in range(1, len(mask))
        if bool(mask[index]) and not bool(mask[index - 1])
    ]


def _motion_arrays(
    xs: np.ndarray,
    ys: np.ndarray,
    config: RunOnsetConfig,
) -> dict[str, np.ndarray]:
    window = _odd_window(config.smoothing_seconds)
    x = _smooth(xs, window)
    y = _smooth(ys, window)
    vx = np.gradient(x, 1.0 / FPS)
    vy = np.gradient(y, 1.0 / FPS)
    vx = _smooth(vx, window)
    vy = _smooth(vy, window)
    ax = _smooth(np.gradient(vx, 1.0 / FPS), window)
    ay = _smooth(np.gradient(vy, 1.0 / FPS), window)
    speed = np.hypot(vx, vy)
    tangent = np.divide(
        vx * ax + vy * ay,
        speed,
        out=np.zeros_like(speed),
        where=speed >= 0.25,
    )
    heading = np.unwrap(np.arctan2(vy, vx))
    turn_rate = _smooth(np.gradient(heading, 1.0 / FPS), window)
    turn_rate[speed < config.direction_minimum_speed_mps] = 0.0
    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "tangent_acceleration": tangent,
        "turn_rate": turn_rate,
    }


def motion_signal_records(
    frame_ids: Sequence[int],
    xs: Sequence[float],
    ys: Sequence[float],
    config: RunOnsetConfig = RunOnsetConfig(),
) -> tuple[dict[str, float], ...]:
    """Expose smoothed signals for audit plots using the detector convention."""

    config.validate()
    ids = np.asarray(frame_ids, dtype=int)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    if not (len(ids) == len(x_values) == len(y_values)):
        raise ValueError("track arrays must have equal lengths")
    arrays = _motion_arrays(x_values, y_values, config)
    return tuple(
        {
            "frame_id": int(ids[index]),
            "x": float(arrays["x"][index]),
            "y": float(arrays["y"][index]),
            "speed_mps": float(arrays["speed"][index]),
            "tangential_acceleration_mps2": float(
                arrays["tangent_acceleration"][index]
            ),
            "turn_rate_degrees_s": float(
                math.degrees(arrays["turn_rate"][index])
            ),
        }
        for index in range(len(ids))
    )


def _metrics_at(
    index: int,
    arrays: Mapping[str, np.ndarray],
    config: RunOnsetConfig,
) -> dict[str, float]:
    pre = int(round(config.pre_window_seconds * FPS))
    post = int(round(config.post_confirmation_seconds * FPS))
    displacement = int(round(config.displacement_horizon_seconds * FPS))
    pre_gap = max(1, int(round(0.08 * FPS)))
    post_start = max(1, int(round(0.30 * FPS)))
    pre_slice = slice(index - pre, index - pre_gap + 1)
    post_slice = slice(index + post_start, index + post + 1)
    pre_velocity = np.asarray(
        [
            float(np.mean(arrays["vx"][pre_slice])),
            float(np.mean(arrays["vy"][pre_slice])),
        ]
    )
    post_velocity = np.asarray(
        [
            float(np.mean(arrays["vx"][post_slice])),
            float(np.mean(arrays["vy"][post_slice])),
        ]
    )
    pre_speed = float(np.mean(arrays["speed"][pre_slice]))
    post_speed = float(np.mean(arrays["speed"][post_slice]))
    post_displacement = float(
        math.hypot(
            arrays["x"][index + displacement] - arrays["x"][index],
            arrays["y"][index + displacement] - arrays["y"][index],
        )
    )
    confirmation_slice = slice(index, index + post + 1)
    return {
        "pre_speed": pre_speed,
        "post_speed": post_speed,
        "speed_gain": post_speed - pre_speed,
        "direction_change": _angle_between_degrees(pre_velocity, post_velocity),
        "post_displacement": post_displacement,
        "peak_acceleration": float(
            np.max(arrays["tangent_acceleration"][confirmation_slice])
        ),
        "peak_deceleration": float(
            max(0.0, -np.min(arrays["tangent_acceleration"][confirmation_slice]))
        ),
    }


def _merge_onsets(
    onsets: Sequence[KinematicRunOnset],
    config: RunOnsetConfig,
) -> tuple[KinematicRunOnset, ...]:
    if not onsets:
        return ()
    merge_frames = int(round(config.merge_seconds * FPS))
    ordered = sorted(onsets, key=lambda onset: (onset.frame_id, onset.labels))
    groups: list[list[KinematicRunOnset]] = [[ordered[0]]]
    for onset in ordered[1:]:
        if onset.frame_id - groups[-1][-1].frame_id <= merge_frames:
            groups[-1].append(onset)
        else:
            groups.append([onset])
    merged = []
    for group in groups:
        earliest = min(group, key=lambda onset: onset.frame_id)
        labels = tuple(
            label
            for label in ("acceleration", "direction_change", "deceleration_turn")
            if any(label in onset.labels for onset in group)
        )
        strongest = max(group, key=lambda onset: onset.confidence_score)
        merged.append(
            KinematicRunOnset(
                frame_id=earliest.frame_id,
                labels=labels,
                x=earliest.x,
                y=earliest.y,
                onset_speed_mps=earliest.onset_speed_mps,
                pre_mean_speed_mps=earliest.pre_mean_speed_mps,
                post_mean_speed_mps=strongest.post_mean_speed_mps,
                speed_gain_mps=strongest.speed_gain_mps,
                direction_change_degrees=max(
                    onset.direction_change_degrees for onset in group
                ),
                post_displacement_m=strongest.post_displacement_m,
                peak_acceleration_mps2=max(
                    onset.peak_acceleration_mps2 for onset in group
                ),
                peak_deceleration_mps2=max(
                    onset.peak_deceleration_mps2 for onset in group
                ),
                confidence_score=max(onset.confidence_score for onset in group),
            )
        )
    return tuple(merged)


def _detect_contiguous_track(
    frame_ids: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    config: RunOnsetConfig,
) -> tuple[KinematicRunOnset, ...]:
    arrays = _motion_arrays(xs, ys, config)
    pre = int(round(config.pre_window_seconds * FPS))
    post = max(
        int(round(config.post_confirmation_seconds * FPS)),
        int(round(config.displacement_horizon_seconds * FPS)),
    )
    valid = np.zeros(len(frame_ids), dtype=bool)
    valid[pre + 1 : len(frame_ids) - post - 1] = True
    acceleration_mask = (
        arrays["tangent_acceleration"] >= config.acceleration_onset_threshold_mps2
    ) & valid
    direction_mask = (
        np.abs(np.degrees(arrays["turn_rate"]))
        >= config.turn_rate_onset_threshold_degrees_s
    ) & (arrays["speed"] >= config.direction_minimum_speed_mps) & valid
    deceleration_mask = (
        arrays["tangent_acceleration"] <= -config.deceleration_onset_threshold_mps2
    ) & valid

    raw: list[KinematicRunOnset] = []
    for label, edges in (
        ("acceleration", _rising_edges(acceleration_mask)),
        ("direction_change", _rising_edges(direction_mask)),
        ("deceleration_turn", _rising_edges(deceleration_mask)),
    ):
        for index in edges:
            metrics = _metrics_at(index, arrays, config)
            if metrics["post_displacement"] < config.minimum_post_displacement_m:
                continue
            if label == "acceleration":
                if metrics["speed_gain"] < config.minimum_speed_gain_mps:
                    continue
                confidence = min(
                    metrics["speed_gain"] / config.minimum_speed_gain_mps,
                    metrics["post_displacement"] / config.minimum_post_displacement_m,
                )
            elif label == "direction_change":
                if (
                    min(metrics["pre_speed"], metrics["post_speed"])
                    < config.direction_minimum_speed_mps
                    or metrics["direction_change"]
                    < config.minimum_direction_change_degrees
                ):
                    continue
                confidence = min(
                    metrics["direction_change"]
                    / config.minimum_direction_change_degrees,
                    metrics["post_displacement"] / config.minimum_post_displacement_m,
                )
            else:
                horizon = int(round(config.post_confirmation_seconds * FPS))
                speeds = arrays["speed"][index : index + horizon + 1]
                trough = int(np.argmin(speeds))
                speed_drop = metrics["pre_speed"] - float(speeds[trough])
                later = speeds[min(len(speeds) - 1, trough + 3) :]
                reacceleration = (
                    float(np.max(later)) - float(speeds[trough])
                    if len(later)
                    else 0.0
                )
                if (
                    speed_drop < config.minimum_deceleration_drop_mps
                    or reacceleration < config.minimum_reacceleration_mps
                    or metrics["direction_change"] < config.deceleration_turn_degrees
                ):
                    continue
                confidence = min(
                    speed_drop / config.minimum_deceleration_drop_mps,
                    reacceleration / config.minimum_reacceleration_mps,
                    metrics["direction_change"] / config.deceleration_turn_degrees,
                )
            raw.append(
                KinematicRunOnset(
                    frame_id=int(frame_ids[index]),
                    labels=(label,),
                    x=float(arrays["x"][index]),
                    y=float(arrays["y"][index]),
                    onset_speed_mps=float(arrays["speed"][index]),
                    pre_mean_speed_mps=metrics["pre_speed"],
                    post_mean_speed_mps=metrics["post_speed"],
                    speed_gain_mps=metrics["speed_gain"],
                    direction_change_degrees=metrics["direction_change"],
                    post_displacement_m=metrics["post_displacement"],
                    peak_acceleration_mps2=metrics["peak_acceleration"],
                    peak_deceleration_mps2=metrics["peak_deceleration"],
                    confidence_score=float(confidence),
                )
            )
    return _merge_onsets(raw, config)


def detect_kinematic_run_onsets(
    frame_ids: Sequence[int],
    xs: Sequence[float],
    ys: Sequence[float],
    config: RunOnsetConfig = RunOnsetConfig(),
) -> tuple[KinematicRunOnset, ...]:
    """Detect and merge acceleration, turn, and check-run onset signals."""

    config.validate()
    ids = np.asarray(frame_ids, dtype=int)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    if not (len(ids) == len(x_values) == len(y_values)):
        raise ValueError("track arrays must have equal lengths")
    if len(ids) == 0:
        return ()
    if not np.all(np.diff(ids) > 0):
        raise ValueError("frame_ids must be strictly increasing")
    boundaries = np.flatnonzero(np.diff(ids) != 1) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(ids)]
    detected: list[KinematicRunOnset] = []
    minimum_length = int(
        round(
            (
                config.pre_window_seconds
                + config.displacement_horizon_seconds
                + 0.25
            )
            * FPS
        )
    )
    for start, end in zip(starts, ends):
        if end - start < minimum_length:
            continue
        detected.extend(
            _detect_contiguous_track(
                ids[start:end], x_values[start:end], y_values[start:end], config
            )
        )
    return _merge_onsets(detected, config)


def _nearest_player(frame: BundesligaFrame) -> tuple[str | None, float | None]:
    if frame.ball is None or not frame.players:
        return None, None
    return min(
        (
            (
                player_id,
                math.hypot(player.x - frame.ball.x, player.y - frame.ball.y),
            )
            for player_id, player in frame.players.items()
        ),
        key=lambda item: item[1],
    )


def team_controls_ball(
    frame: BundesligaFrame,
    team_id: str,
    control_distance_m: float,
) -> bool:
    """Return whether the closest player to the ball belongs to ``team_id``.

    Merely finding one teammate inside the control radius is insufficient in a
    contested-ball situation: an opponent can still be closer.  The detector
    therefore assigns control to the nearest player first, then applies the
    distance threshold.
    """

    if control_distance_m <= 0.0:
        raise ValueError("control_distance_m must be positive")
    nearest_id, distance = _nearest_player(frame)
    return bool(
        nearest_id is not None
        and distance is not None
        and distance <= control_distance_m
        and frame.players[nearest_id].team_id == team_id
    )


def ball_is_inside_pitch(
    frame: BundesligaFrame,
    tolerance_m: float = 0.20,
) -> bool:
    """Return whether the ball lies within a tolerant tracked-pitch boundary."""

    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    return bool(
        frame.ball is not None
        and abs(frame.ball.x) <= FIELD_LENGTH / 2.0 + tolerance_m
        and abs(frame.ball.y) <= FIELD_WIDTH / 2.0 + tolerance_m
    )


def _context_candidate(
    onset: KinematicRunOnset,
    player_id: str,
    frames: Mapping[int, BundesligaFrame],
    metadata: BundesligaMatchMeta,
    config: RunOnsetConfig,
    expected_player_count: Callable[[int], int] | None = None,
) -> RunOnsetCandidate | None:
    frame = frames.get(onset.frame_id)
    if frame is None or player_id not in frame.players or frame.ball is None:
        return None
    if config.require_ball_inside_pitch and not ball_is_inside_pitch(
        frame,
        config.pitch_boundary_tolerance_m,
    ):
        return None
    # The gate asks "is every player who should be on the pitch tracked?", not
    # "are there 22 players?". After a dismissal the right answer is 21, and
    # requiring 22 discards the rest of that match: DFL-MAT-J03WN1 (red card at
    # 7.1 min) yielded 2 run onsets where peer matches yield 52-85. A dismissal
    # is football the population includes; missing tracking is not, and only the
    # second should reject the frame.
    required = config.required_player_count
    if required is not None and expected_player_count is not None:
        required = expected_player_count(onset.frame_id)
    if required is not None and len(frame.players) != required:
        return None
    carrier_id, carrier_distance = _nearest_player(frame)
    if (
        carrier_id is None
        or carrier_distance is None
        or carrier_distance > config.control_distance_m
        or carrier_id == player_id
    ):
        return None
    runner = frame.players[player_id]
    carrier = frame.players[carrier_id]
    if runner.team_id != carrier.team_id:
        return None
    if metadata.goalkeeper_id(runner.team_id) == player_id:
        return None
    runner_ball_distance = math.hypot(runner.x - frame.ball.x, runner.y - frame.ball.y)
    if runner_ball_distance <= config.control_distance_m:
        return None

    history_frames = int(round(config.controlled_history_seconds * FPS))
    history_ids = range(onset.frame_id - history_frames, onset.frame_id + 1)
    stable = 0
    available = 0
    for frame_id in history_ids:
        sample = frames.get(frame_id)
        if sample is None or sample.ball is None:
            continue
        available += 1
        if team_controls_ball(sample, runner.team_id, config.control_distance_m):
            stable += 1
    required = history_frames + 1
    stable_fraction = stable / required
    if available < required or stable_fraction < config.minimum_stable_control_fraction:
        return None

    future_frames = int(round(config.future_possession_seconds * FPS))
    known = 0
    same_team = 0
    available_future = 0
    for frame_id in range(onset.frame_id, onset.frame_id + future_frames + 1):
        sample = frames.get(frame_id)
        if sample is None or sample.ball is None:
            continue
        available_future += 1
        nearest_id, distance = _nearest_player(sample)
        if nearest_id is None or distance is None or distance > config.future_control_distance_m:
            continue
        known += 1
        if sample.players[nearest_id].team_id == runner.team_id:
            same_team += 1
    if available_future < future_frames + 1:
        return None
    known_fraction = known / available_future
    same_team_fraction = same_team / known if known else 0.0
    if (
        known_fraction < config.minimum_future_known_fraction
        or same_team_fraction < config.minimum_future_same_team_fraction
    ):
        return None

    return RunOnsetCandidate(
        match_id=frame.match_id,
        frame_id=onset.frame_id,
        player_id=player_id,
        team_id=runner.team_id,
        ball_carrier_id=carrier_id,
        labels=onset.labels,
        x=onset.x,
        y=onset.y,
        onset_speed_mps=onset.onset_speed_mps,
        pre_mean_speed_mps=onset.pre_mean_speed_mps,
        post_mean_speed_mps=onset.post_mean_speed_mps,
        speed_gain_mps=onset.speed_gain_mps,
        direction_change_degrees=onset.direction_change_degrees,
        post_displacement_m=onset.post_displacement_m,
        peak_acceleration_mps2=onset.peak_acceleration_mps2,
        peak_deceleration_mps2=onset.peak_deceleration_mps2,
        confidence_score=onset.confidence_score,
        ball_distance_m=float(runner_ball_distance),
        stable_control_fraction=float(stable_fraction),
        future_known_fraction=float(known_fraction),
        future_same_team_fraction=float(same_team_fraction),
    )


def detect_run_onsets(
    frames: Mapping[int, BundesligaFrame],
    metadata: BundesligaMatchMeta,
    search_frame_ids: Sequence[int] | set[int] | None = None,
    config: RunOnsetConfig = RunOnsetConfig(),
    expected_player_count: Callable[[int], int] | None = None,
) -> tuple[RunOnsetCandidate, ...]:
    """Detect off-ball onsets and enforce controlled-possession context."""

    config.validate()
    search = set(search_frame_ids) if search_frame_ids is not None else None
    player_tracks: dict[str, list[tuple[int, float, float]]] = {}
    for frame_id in sorted(frames):
        frame = frames[frame_id]
        for player_id, player in frame.players.items():
            player_tracks.setdefault(player_id, []).append((frame_id, player.x, player.y))

    candidates: list[RunOnsetCandidate] = []
    for player_id, track in player_tracks.items():
        onsets = detect_kinematic_run_onsets(
            [sample[0] for sample in track],
            [sample[1] for sample in track],
            [sample[2] for sample in track],
            config,
        )
        for onset in onsets:
            if search is not None and onset.frame_id not in search:
                continue
            candidate = _context_candidate(
                onset, player_id, frames, metadata, config, expected_player_count
            )
            if candidate is not None:
                candidates.append(candidate)
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.frame_id,
                candidate.player_id,
                candidate.labels,
            ),
        )
    )
