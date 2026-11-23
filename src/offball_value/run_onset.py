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
    # 2026-09-27 (Chani-scene trace): 12 of her 45 strong+medium scenes fail the
    # context gate because the ball is IN FLIGHT when the run starts -- nobody is
    # within control_distance_m of it (median 3.45 m). With this on, a frame with
    # nobody on the ball counts as controlled by the team that last had it, in
    # both the carrier gate and the stable-history check, exactly as stage 0's
    # segmentation keeps a spell open through unknown-control samples. The
    # candidate's carrier is then the last controller (the passer) -- or, when
    # the runner himself just passed (pass and move), the next team-mate to
    # control the ball inside the future window -- and the candidate is marked
    # ball_in_flight. The future-possession check treats an in-flight frame the
    # same way. Off by default: v0.5 unchanged.
    in_flight_control: bool = False
    # 2026-10-23: this gate reads ball control from horizontal distance alone,
    # while stage 0 and 1c take a ball above 1.5 m to be nobody's
    # (shot_context.controlled_team_at_frame), so a lofted pass counts here as
    # the player's it passes over. The pass-anchored onset of Chani's strong
    # shot_002 (J03WOY) failed the 1 s future check on it. Set, a ball above
    # this height is in flight in every check of the gate (carrier, history,
    # future). None: height ignored, v0.6 unchanged.
    max_control_ball_height_m: float | None = None
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
        if self.max_control_ball_height_m is not None and self.max_control_ball_height_m <= 0.0:
            raise ValueError("max_control_ball_height_m must be positive")


@dataclass(frozen=True)
class PassAnchorConfig:
    """A second candidate source beside the kinematic detector (2026-10-23).

    The detector needs a burst or a turn, so it misses a runner who is already
    at speed and drags a defender with him: in Chani's strong shot_002 (J03WOY)
    the runner held 3.3-3.8 m/s and the defender followed him. Here a pass or
    cross by the attacking team is the onset for a team-mate running at it, but
    only when a defender near him reacts, i.e. makes a detector onset within
    reaction_seconds. Without the reaction every pass makes a candidate of every
    fast team-mate, and on the 7 matches the low-effect controls grew as much as
    the medium scenes. The candidates then face the same 1b and 1c gates.
    """

    minimum_runner_speed_mps: float = 3.5
    reaction_radius_m: float = 10.0
    reaction_seconds: float = 1.0

    def validate(self) -> None:
        invalid = [name for name, value in self.__dict__.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"pass-anchor parameters must be positive: {invalid}")


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
    ball_in_flight: bool = False

    @property
    def primary_type(self) -> str:
        if "pass_anchor" in self.labels:
            return "pass_anchor"
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
            "ball_in_flight": self.ball_in_flight,
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


def _nearest_player(
    frame: BundesligaFrame,
    max_ball_height_m: float | None = None,
) -> tuple[str | None, float | None]:
    """The player nearest the ball and his distance to it. Above
    max_ball_height_m nobody can be on the ball, so the distance is inf: the
    ball is in flight, whoever it passes over."""
    if frame.ball is None or not frame.players:
        return None, None
    nearest = min(
        (
            (
                player_id,
                math.hypot(player.x - frame.ball.x, player.y - frame.ball.y),
            )
            for player_id, player in frame.players.items()
        ),
        key=lambda item: item[1],
    )
    if (
        max_ball_height_m is not None
        and frame.ball.z is not None
        and frame.ball.z > max_ball_height_m
    ):
        return nearest[0], math.inf
    return nearest


def team_controls_ball(
    frame: BundesligaFrame,
    team_id: str,
    control_distance_m: float,
    max_ball_height_m: float | None = None,
) -> bool:
    """Return whether the closest player to the ball belongs to ``team_id``.

    Merely finding one teammate inside the control radius is insufficient in a
    contested-ball situation: an opponent can still be closer.  The detector
    therefore assigns control to the nearest player first, then applies the
    distance threshold. A ball above ``max_ball_height_m`` is nobody's.
    """

    if control_distance_m <= 0.0:
        raise ValueError("control_distance_m must be positive")
    nearest_id, distance = _nearest_player(frame, max_ball_height_m)
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


def _last_controller(frames, frame_id: int, config: RunOnsetConfig):
    """(player_id, team_id) of the last player within control distance of the
    ball at or before frame_id, walking back through the loaded frames; None
    when the loaded window runs out first."""
    f = frame_id
    while f in frames:
        sample = frames[f]
        if sample.ball is not None:
            pid, dist = _nearest_player(sample, config.max_control_ball_height_m)
            if pid is not None and dist is not None and dist <= config.control_distance_m:
                return pid, sample.players[pid].team_id
        f -= 1
    return None


def _next_controller(frames, frame_id: int, team_id: str, config: RunOnsetConfig, horizon_frames: int):
    """(player_id) of the first player of `team_id` within control distance of the
    ball after frame_id, within horizon_frames; None if an opponent gets there
    first or nobody does."""
    for f in range(frame_id + 1, frame_id + horizon_frames + 1):
        sample = frames.get(f)
        if sample is None or sample.ball is None:
            continue
        pid, dist = _nearest_player(sample, config.max_control_ball_height_m)
        if pid is not None and dist is not None and dist <= config.control_distance_m:
            return pid if sample.players[pid].team_id == team_id else None
    return None


def _team_has_ball(frames, frame_id: int, team_id: str, config: RunOnsetConfig) -> bool:
    """team_controls_ball, or -- with in_flight_control -- the ball in flight
    from that team (nobody on it, and it was last on a team-mate's foot)."""
    sample = frames.get(frame_id)
    if sample is None or sample.ball is None:
        return False
    if team_controls_ball(
        sample, team_id, config.control_distance_m, config.max_control_ball_height_m
    ):
        return True
    if not config.in_flight_control:
        return False
    pid, dist = _nearest_player(sample, config.max_control_ball_height_m)
    if pid is not None and dist is not None and dist <= config.control_distance_m:
        return False                      # someone (the other side) has it
    last = _last_controller(frames, frame_id, config)
    return last is not None and last[1] == team_id


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
    carrier_id, carrier_distance = _nearest_player(frame, config.max_control_ball_height_m)
    if carrier_id is None or carrier_distance is None:
        return None
    ball_in_flight = False
    if carrier_distance > config.control_distance_m:
        if not config.in_flight_control:
            return None
        last = _last_controller(frames, onset.frame_id, config)
        if last is None or last[0] not in frame.players:
            return None
        carrier_id, ball_in_flight = last[0], True
        if carrier_id == player_id:
            # pass and move: the runner released the ball; his carrier is whoever
            # of his team takes it next, if a team-mate does within the window
            carrier_id = _next_controller(frames, onset.frame_id, frame.players[player_id].team_id,
                                          config, int(round(config.future_possession_seconds * FPS)))
            if carrier_id is None or carrier_id not in frame.players:
                return None
    if carrier_id == player_id:
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
        if _team_has_ball(frames, frame_id, runner.team_id, config):
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
        nearest_id, distance = _nearest_player(sample, config.max_control_ball_height_m)
        if nearest_id is None or distance is None or distance > config.future_control_distance_m:
            if config.in_flight_control:
                last = _last_controller(frames, frame_id, config)
                if last is not None and last[1] == runner.team_id:
                    known += 1          # in flight from a team-mate: still theirs
                    same_team += 1
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
        ball_in_flight=ball_in_flight,
    )


def _player_tracks(
    frames: Mapping[int, BundesligaFrame],
) -> dict[str, list[tuple[int, float, float]]]:
    """(frame id, x, y) of every player over the loaded frames, in frame order."""
    player_tracks: dict[str, list[tuple[int, float, float]]] = {}
    for frame_id in sorted(frames):
        frame = frames[frame_id]
        for player_id, player in frame.players.items():
            player_tracks.setdefault(player_id, []).append((frame_id, player.x, player.y))
    return player_tracks


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
    candidates: list[RunOnsetCandidate] = []
    for player_id, track in _player_tracks(frames).items():
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


@dataclass(frozen=True)
class PassAnchoredCandidate:
    candidate: RunOnsetCandidate
    passer_id: str
    # (defender id, his distance to the runner at the pass in m, his onset
    # frame) for every defender that reacted, nearest first
    reactions: tuple[tuple[str, float, int], ...]


def detect_pass_anchored_run_onsets(
    frames: Mapping[int, BundesligaFrame],
    metadata: BundesligaMatchMeta,
    anchors: Sequence[tuple[int, str, str]],
    config: RunOnsetConfig = RunOnsetConfig(),
    anchor_config: PassAnchorConfig = PassAnchorConfig(),
    expected_player_count: Callable[[int], int] | None = None,
) -> tuple[PassAnchoredCandidate, ...]:
    """Run onsets anchored on passes; see PassAnchorConfig.

    ``anchors`` holds (frame id, passer id, team id) per pass or cross. A
    team-mate of the passer, not a keeper, is a candidate at the pass frame when
    his detector speed there reaches minimum_runner_speed_mps and a defender, not
    a keeper, within reaction_radius_m of him makes a kinematic onset in
    [pass, pass + reaction_seconds]. The onset is measured at the pass frame as
    the detector measures its own (_metrics_at, so post_mean_speed_mps is the
    run speed the later cuts read), and it must pass _context_candidate. One
    candidate per pass frame and runner.
    """

    config.validate()
    anchor_config.validate()
    pre = int(round(config.pre_window_seconds * FPS))
    post = max(
        int(round(config.post_confirmation_seconds * FPS)),
        int(round(config.displacement_horizon_seconds * FPS)),
    )
    reaction_frames = int(round(anchor_config.reaction_seconds * FPS))
    keepers = {metadata.goalkeeper_id(team_id) for team_id in metadata.teams}
    segments: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}
    onset_frames: dict[str, np.ndarray] = {}
    for player_id, track in _player_tracks(frames).items():
        ids = np.asarray([sample[0] for sample in track], dtype=int)
        xs = np.asarray([sample[1] for sample in track], dtype=float)
        ys = np.asarray([sample[2] for sample in track], dtype=float)
        onset_frames[player_id] = np.asarray(
            [onset.frame_id for onset in detect_kinematic_run_onsets(ids, xs, ys, config)],
            dtype=int,
        )
        boundaries = np.flatnonzero(np.diff(ids) != 1) + 1
        segments[player_id] = [
            (ids[start:end], _motion_arrays(xs[start:end], ys[start:end], config))
            for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(ids)])
            if end - start > pre + post + 2
        ]

    def motion_at(player_id: str, frame_id: int):
        # the frame's index in its contiguous track, where the detector itself
        # could place an onset (so every _metrics_at window is inside the track)
        for ids, arrays in segments.get(player_id, ()):
            if ids[0] <= frame_id <= ids[-1]:
                index = int(frame_id - ids[0])
                return (index, arrays) if pre + 1 <= index < len(ids) - post - 1 else None
        return None

    found: dict[tuple[int, str], PassAnchoredCandidate] = {}
    for frame_id, passer_id, team_id in anchors:
        frame = frames.get(frame_id)
        if frame is None or passer_id not in frame.players:
            continue
        for runner_id, runner in frame.players.items():
            if (
                runner.team_id != team_id
                or runner_id == passer_id
                or runner_id in keepers
                or (frame_id, runner_id) in found
            ):
                continue
            motion = motion_at(runner_id, frame_id)
            if motion is None:
                continue
            index, arrays = motion
            speed = float(arrays["speed"][index])
            if not speed >= anchor_config.minimum_runner_speed_mps:
                continue
            reactions = []
            for defender_id, defender in frame.players.items():
                if defender.team_id == team_id or defender_id in keepers:
                    continue
                distance = math.hypot(defender.x - runner.x, defender.y - runner.y)
                if distance > anchor_config.reaction_radius_m:
                    continue
                onsets = onset_frames.get(defender_id, np.empty(0, dtype=int))
                hits = onsets[(onsets >= frame_id) & (onsets <= frame_id + reaction_frames)]
                if len(hits):
                    reactions.append((defender_id, float(distance), int(hits[0])))
            if not reactions:
                continue
            metrics = _metrics_at(index, arrays, config)
            onset = KinematicRunOnset(
                frame_id=frame_id,
                labels=("pass_anchor",),
                x=float(arrays["x"][index]),
                y=float(arrays["y"][index]),
                onset_speed_mps=speed,
                pre_mean_speed_mps=metrics["pre_speed"],
                post_mean_speed_mps=metrics["post_speed"],
                speed_gain_mps=metrics["speed_gain"],
                direction_change_degrees=metrics["direction_change"],
                post_displacement_m=metrics["post_displacement"],
                peak_acceleration_mps2=metrics["peak_acceleration"],
                peak_deceleration_mps2=metrics["peak_deceleration"],
                # how far past the speed threshold, as the detector's own
                # scores read against theirs
                confidence_score=speed / anchor_config.minimum_runner_speed_mps,
            )
            candidate = _context_candidate(
                onset, runner_id, frames, metadata, config, expected_player_count
            )
            if candidate is not None:
                found[(frame_id, runner_id)] = PassAnchoredCandidate(
                    candidate=candidate,
                    passer_id=passer_id,
                    reactions=tuple(sorted(reactions, key=lambda reaction: reaction[1])),
                )
    return tuple(found[key] for key in sorted(found))
