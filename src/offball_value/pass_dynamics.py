from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    BundesligaFrame,
    BundesligaObjectState,
)


@dataclass(frozen=True)
class VelocityEstimate:
    vx: float
    vy: float
    speed: float
    sample_count: int
    window_seconds: float


@dataclass(frozen=True)
class ArrivalModelConfig:
    history_seconds: float = 0.4
    velocity_estimator: str = "linear"
    pass_speed_mps: float = 14.0
    player_max_speed_mps: float = 9.0
    player_acceleration_mps2: float = 3.5
    reaction_time_s: float = 0.10
    max_turn_penalty_s: float = 0.40
    race_sigma_s: float = 0.35
    observed_speed_cap_mps: float = 11.0
    path_samples: int = 19
    path_start_fraction: float = 0.08
    path_end_fraction: float = 0.80
    path_race_sigma_s: float = 0.30
    secure_time_buffer_s: float = 0.0
    secure_sigma_s: float = 0.45
    pressure_action_time_s: float = 0.45
    pressure_sigma_s: float = 0.20
    pass_distance_scale_m: float = 45.0


@dataclass(frozen=True)
class ReceptionRegionConfig:
    resolution_m: float = 1.5
    local_radius_m: float = 4.0
    local_sigma_m: float = 2.2
    min_ahead_m: float = 5.0
    max_ahead_m: float = 14.0
    movement_horizon_s: float = 1.8
    movement_lateral_sigma_m: float = 3.0
    movement_longitudinal_sigma_m: float = 5.0
    local_weight: float = 0.35
    min_relative_prior: float = 0.01


@dataclass(frozen=True)
class PointReceptionEstimate:
    target_x: float
    target_y: float
    pass_distance_m: float
    ball_arrival_time_s: float
    receiver_arrival_time_s: float
    nearest_defender_id: str | None
    nearest_defender_arrival_time_s: float
    defender_time_margin_s: float
    receiver_first_probability: float
    path_suppression_defender_id: str | None
    path_suppression_fraction: float | None
    path_time_margin_s: float
    path_survival_probability: float
    secure_possession_probability: float
    passer_pressure_defender_id: str | None
    passer_pressure_arrival_time_s: float
    pressure_execution_probability: float
    distance_execution_probability: float
    pass_execution_probability: float
    receive_probability: float


@dataclass(frozen=True)
class KickFrameEstimate:
    event_frame_id: int
    kick_frame_id: int
    release_frame_id: int
    offset_frames: int
    passer_ball_distance_m: float
    release_ball_speed_mps: float
    target_alignment: float
    detected: bool


@dataclass(frozen=True)
class ReceptionRegionEstimate:
    receiver_id: str
    point_count: int
    expected_receive_probability: float
    expected_receive_value: float
    option_value: float
    peak_x: float
    peak_y: float
    peak_prior: float
    peak_receive_probability: float
    peak_receive_value: float
    peak_weighted_value: float
    points: np.ndarray
    intent_prior: np.ndarray
    receive_probability: np.ndarray
    receive_value: np.ndarray
    weighted_value: np.ndarray


def estimate_velocity(
    frames: Iterable[BundesligaFrame],
    player_id: str,
    end_frame_id: int,
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> VelocityEstimate:
    if config.velocity_estimator not in {"linear", "quadratic_endpoint"}:
        raise ValueError(
            "velocity_estimator must be linear or quadratic_endpoint"
        )
    window_frames = max(1, int(round(config.history_seconds * FPS)))
    samples = []
    for frame in frames:
        if frame.frame_id > end_frame_id or end_frame_id - frame.frame_id > window_frames:
            continue
        player = frame.players.get(player_id)
        if player is None:
            continue
        time_s = (frame.frame_id - end_frame_id) / FPS
        samples.append((time_s, player.x, player.y))

    samples.sort(key=lambda sample: sample[0])
    if len(samples) < 2:
        return VelocityEstimate(0.0, 0.0, 0.0, len(samples), config.history_seconds)

    times = np.asarray([sample[0] for sample in samples], dtype=float)
    xs = np.asarray([sample[1] for sample in samples], dtype=float)
    ys = np.asarray([sample[2] for sample in samples], dtype=float)
    if config.velocity_estimator == "quadratic_endpoint" and len(samples) >= 3:
        # Fit position(t)=a*t^2+v_end*t+c on causal history only.  Because
        # t=0 is the decision frame, the linear coefficient is the estimated
        # instantaneous velocity at the end of the window.  This is more
        # responsive to ongoing acceleration/deceleration than the window-
        # average slope while retaining a deterministic fallback below.
        design = np.column_stack(
            [times**2, times, np.ones(len(times), dtype=float)]
        )
        vx = float(np.linalg.lstsq(design, xs, rcond=None)[0][1])
        vy = float(np.linalg.lstsq(design, ys, rcond=None)[0][1])
    else:
        design = np.column_stack([times, np.ones(len(times), dtype=float)])
        vx = float(np.linalg.lstsq(design, xs, rcond=None)[0][0])
        vy = float(np.linalg.lstsq(design, ys, rcond=None)[0][0])
    speed = math.hypot(vx, vy)

    if speed > config.observed_speed_cap_mps:
        scale = config.observed_speed_cap_mps / speed
        vx *= scale
        vy *= scale
        speed = config.observed_speed_cap_mps

    observed_window = float(times[-1] - times[0])
    return VelocityEstimate(vx, vy, speed, len(samples), observed_window)


def estimate_frame_velocities(
    frames: Iterable[BundesligaFrame],
    end_frame_id: int,
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> dict[str, VelocityEstimate]:
    frame_list = list(frames)
    end_frame = next(
        (frame for frame in frame_list if frame.frame_id == end_frame_id),
        None,
    )
    if end_frame is None:
        raise ValueError(f"Frame {end_frame_id} is not available")
    return {
        player_id: estimate_velocity(frame_list, player_id, end_frame_id, config)
        for player_id in end_frame.players
    }


def detect_kick_frame(
    frames: Iterable[BundesligaFrame],
    passer_id: str,
    event_frame_id: int,
    control_distance_m: float = 1.2,
    release_distance_m: float = 1.5,
    release_speed_mps: float = 6.0,
    target_xy: tuple[float, float] | None = None,
    target_player_id: str | None = None,
    min_target_alignment: float = 0.5,
) -> KickFrameEstimate:
    ordered = sorted(
        (
            frame
            for frame in frames
            if frame.ball is not None and passer_id in frame.players
        ),
        key=lambda frame: frame.frame_id,
    )
    if not ordered:
        raise ValueError(f"No ball/passer frames are available before {event_frame_id}")

    last_controlled: BundesligaFrame | None = None
    candidates: list[tuple[BundesligaFrame, BundesligaFrame, float, float]] = []
    previous_ball: BundesligaObjectState | None = None
    previous_frame_id: int | None = None
    for frame in ordered:
        passer = frame.players[passer_id]
        ball = frame.ball
        distance = math.hypot(ball.x - passer.x, ball.y - passer.y)
        if distance <= control_distance_m:
            last_controlled = frame

        raw_speed = ball.speed / 3.6 if ball.speed is not None else None
        derived_speed = None
        if previous_ball is not None and previous_frame_id is not None:
            elapsed = (frame.frame_id - previous_frame_id) / FPS
            if elapsed > 0:
                derived_speed = math.hypot(
                    ball.x - previous_ball.x,
                    ball.y - previous_ball.y,
                ) / elapsed
        ball_speed = raw_speed if raw_speed is not None else (derived_speed or 0.0)

        if (
            last_controlled is not None
            and 0 < frame.frame_id - last_controlled.frame_id <= 5
            and distance >= release_distance_m
            and ball_speed >= release_speed_mps
            and (
                not candidates
                or candidates[-1][0].frame_id != last_controlled.frame_id
            )
        ):
            alignment = 1.0
            candidate_target_xy = target_xy
            if target_player_id is not None:
                target_player = last_controlled.players.get(target_player_id)
                if target_player is not None:
                    candidate_target_xy = (target_player.x, target_player.y)
            if candidate_target_xy is not None:
                release_vector = np.asarray(
                    [
                        frame.ball.x - last_controlled.ball.x,
                        frame.ball.y - last_controlled.ball.y,
                    ],
                    dtype=float,
                )
                target_vector = np.asarray(
                    [
                        candidate_target_xy[0] - last_controlled.ball.x,
                        candidate_target_xy[1] - last_controlled.ball.y,
                    ],
                    dtype=float,
                )
                denominator = float(
                    np.linalg.norm(release_vector) * np.linalg.norm(target_vector)
                )
                alignment = (
                    float(np.dot(release_vector, target_vector) / denominator)
                    if denominator > 1e-9
                    else -1.0
                )
            candidates.append(
                (last_controlled, frame, float(ball_speed), alignment)
            )

        previous_ball = ball
        previous_frame_id = frame.frame_id

    if candidates:
        aligned_candidates = [
            candidate
            for candidate in candidates
            if candidate[3] >= min_target_alignment
        ]
        selected_candidates = aligned_candidates if aligned_candidates else candidates
        controlled_frame, release_frame, speed, alignment = min(
            selected_candidates,
            key=lambda candidate: abs(candidate[1].frame_id - event_frame_id),
        )
        ball = controlled_frame.ball
        passer = controlled_frame.players[passer_id]
        controlled_distance = math.hypot(ball.x - passer.x, ball.y - passer.y)
        return KickFrameEstimate(
            event_frame_id=event_frame_id,
            kick_frame_id=controlled_frame.frame_id,
            release_frame_id=release_frame.frame_id,
            offset_frames=controlled_frame.frame_id - event_frame_id,
            passer_ball_distance_m=float(controlled_distance),
            release_ball_speed_mps=float(speed),
            target_alignment=float(alignment),
            detected=bool(alignment >= min_target_alignment),
        )

    fallback = min(ordered, key=lambda frame: abs(frame.frame_id - event_frame_id))
    ball = fallback.ball
    passer = fallback.players[passer_id]
    distance = math.hypot(ball.x - passer.x, ball.y - passer.y)
    speed = ball.speed / 3.6 if ball.speed is not None else 0.0
    return KickFrameEstimate(
        event_frame_id=event_frame_id,
        kick_frame_id=fallback.frame_id,
        release_frame_id=fallback.frame_id,
        offset_frames=fallback.frame_id - event_frame_id,
        passer_ball_distance_m=float(distance),
        release_ball_speed_mps=float(speed),
        target_alignment=float("nan"),
        detected=False,
    )


def player_time_to_point(
    player: BundesligaObjectState,
    velocity: VelocityEstimate,
    target_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> float:
    dx = float(target_xy[0] - player.x)
    dy = float(target_xy[1] - player.y)
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return 0.0

    ux = dx / distance
    uy = dy / distance
    speed = min(velocity.speed, config.player_max_speed_mps)
    if speed > 1e-9:
        direction_cosine = float(
            np.clip((velocity.vx * ux + velocity.vy * uy) / speed, -1.0, 1.0)
        )
        turn_angle = math.acos(direction_cosine)
        speed_toward_target = max(0.0, velocity.vx * ux + velocity.vy * uy)
    else:
        turn_angle = 0.0
        speed_toward_target = 0.0

    reaction_time = (
        config.reaction_time_s
        + config.max_turn_penalty_s * turn_angle / math.pi
    )
    remaining_distance = max(0.0, distance - speed_toward_target * reaction_time)
    initial_speed = min(speed_toward_target, config.player_max_speed_mps)
    acceleration = max(config.player_acceleration_mps2, 1e-9)
    acceleration_time = max(
        0.0,
        (config.player_max_speed_mps - initial_speed) / acceleration,
    )
    acceleration_distance = (
        initial_speed * acceleration_time
        + 0.5 * acceleration * acceleration_time**2
    )

    if remaining_distance <= acceleration_distance:
        travel_time = (
            -initial_speed
            + math.sqrt(initial_speed**2 + 2.0 * acceleration * remaining_distance)
        ) / acceleration
    else:
        travel_time = acceleration_time + (
            remaining_distance - acceleration_distance
        ) / config.player_max_speed_mps
    return float(reaction_time + travel_time)


def ball_time_to_point(
    ball_xy: tuple[float, float],
    target_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> tuple[float, float]:
    distance = math.hypot(target_xy[0] - ball_xy[0], target_xy[1] - ball_xy[1])
    speed = max(config.pass_speed_mps, 1e-9)
    return float(distance), float(distance / speed)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0))))


def path_survival_estimate(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    attacking_team_id: str,
    ball_xy: tuple[float, float],
    target_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> tuple[str | None, float | None, float, float]:
    defenders = [
        defender
        for defender in frame.players.values()
        if defender.team_id != attacking_team_id
    ]
    if not defenders:
        return None, None, float("inf"), 1.0

    sample_count = max(1, config.path_samples)
    start_fraction = float(np.clip(config.path_start_fraction, 0.0, 1.0))
    end_fraction = float(np.clip(config.path_end_fraction, start_fraction, 1.0))
    fractions = np.linspace(start_fraction, end_fraction, sample_count)
    zero_velocity = VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds)
    best: tuple[str, float, float] | None = None

    for fraction in fractions:
        point = (
            ball_xy[0] + fraction * (target_xy[0] - ball_xy[0]),
            ball_xy[1] + fraction * (target_xy[1] - ball_xy[1]),
        )
        _, ball_time = ball_time_to_point(ball_xy, point, config)
        for defender in defenders:
            defender_time = player_time_to_point(
                defender,
                velocities.get(defender.object_id, zero_velocity),
                point,
                config,
            )
            margin = defender_time - ball_time
            if best is None or margin < best[2]:
                best = defender.object_id, float(fraction), float(margin)

    if best is None:
        return None, None, float("inf"), 1.0
    defender_id, fraction, margin = best
    probability = _logistic(margin / max(config.path_race_sigma_s, 1e-9))
    return defender_id, fraction, margin, probability


def secure_possession_probability(
    defender_time_margin_s: float,
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> float:
    if not math.isfinite(defender_time_margin_s):
        return 1.0
    centered_margin = defender_time_margin_s - config.secure_time_buffer_s
    return _logistic(centered_margin / max(config.secure_sigma_s, 1e-9))


def pass_execution_estimate(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    passer_id: str | None,
    attacking_team_id: str,
    pass_distance_m: float,
    config: ArrivalModelConfig = ArrivalModelConfig(),
) -> tuple[str | None, float, float, float, float]:
    passer = frame.players.get(passer_id) if passer_id else None
    zero_velocity = VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds)
    pressure_defenders = [
        (
            defender.object_id,
            player_time_to_point(
                defender,
                velocities.get(defender.object_id, zero_velocity),
                (passer.x, passer.y),
                config,
            ),
        )
        for defender in frame.players.values()
        if passer is not None and defender.team_id != attacking_team_id
    ]
    if pressure_defenders:
        pressure_defender_id, pressure_time = min(
            pressure_defenders,
            key=lambda item: item[1],
        )
        pressure_probability = _logistic(
            (pressure_time - config.pressure_action_time_s)
            / max(config.pressure_sigma_s, 1e-9)
        )
    else:
        pressure_defender_id = None
        pressure_time = float("inf")
        pressure_probability = 1.0

    distance_scale = max(config.pass_distance_scale_m, 1e-9)
    distance_probability = math.exp(-((pass_distance_m / distance_scale) ** 2))
    execution_probability = pressure_probability * distance_probability
    return (
        pressure_defender_id,
        float(pressure_time),
        float(pressure_probability),
        float(distance_probability),
        float(execution_probability),
    )


def point_reception_estimate(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    receiver_id: str,
    attacking_team_id: str,
    ball_xy: tuple[float, float],
    target_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
    passer_id: str | None = None,
) -> PointReceptionEstimate:
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        raise ValueError(f"Receiver {receiver_id} is not present in frame {frame.frame_id}")

    zero_velocity = VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds)
    receiver_time = player_time_to_point(
        receiver,
        velocities.get(receiver_id, zero_velocity),
        target_xy,
        config,
    )
    defender_times = [
        (
            defender.object_id,
            player_time_to_point(
                defender,
                velocities.get(defender.object_id, zero_velocity),
                target_xy,
                config,
            ),
        )
        for defender in frame.players.values()
        if defender.team_id != attacking_team_id
    ]
    if defender_times:
        nearest_defender_id, defender_time = min(
            defender_times,
            key=lambda item: item[1],
        )
    else:
        nearest_defender_id = None
        defender_time = float("inf")

    pass_distance, ball_time = ball_time_to_point(ball_xy, target_xy, config)
    control_time = max(ball_time, receiver_time)
    margin = defender_time - control_time
    if math.isfinite(margin):
        receiver_first_probability = _logistic(
            margin / max(config.race_sigma_s, 1e-9)
        )
    else:
        receiver_first_probability = 1.0
    (
        path_defender_id,
        path_fraction,
        path_margin,
        path_survival_probability,
    ) = path_survival_estimate(
        frame,
        velocities,
        attacking_team_id,
        ball_xy,
        target_xy,
        config,
    )
    secure_probability = secure_possession_probability(margin, config)
    (
        pressure_defender_id,
        pressure_time,
        pressure_probability,
        distance_probability,
        execution_probability,
    ) = pass_execution_estimate(
        frame,
        velocities,
        passer_id,
        attacking_team_id,
        pass_distance,
        config,
    )
    receive_probability = (
        execution_probability
        * path_survival_probability
        * receiver_first_probability
        * secure_probability
    )

    return PointReceptionEstimate(
        target_x=float(target_xy[0]),
        target_y=float(target_xy[1]),
        pass_distance_m=pass_distance,
        ball_arrival_time_s=ball_time,
        receiver_arrival_time_s=receiver_time,
        nearest_defender_id=nearest_defender_id,
        nearest_defender_arrival_time_s=float(defender_time),
        defender_time_margin_s=float(margin),
        receiver_first_probability=float(receiver_first_probability),
        path_suppression_defender_id=path_defender_id,
        path_suppression_fraction=path_fraction,
        path_time_margin_s=float(path_margin),
        path_survival_probability=float(path_survival_probability),
        secure_possession_probability=float(secure_probability),
        passer_pressure_defender_id=pressure_defender_id,
        passer_pressure_arrival_time_s=float(pressure_time),
        pressure_execution_probability=float(pressure_probability),
        distance_execution_probability=float(distance_probability),
        pass_execution_probability=float(execution_probability),
        receive_probability=float(receive_probability),
    )


def receiver_target_region(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    receiver_id: str,
    attacking_direction: int,
    config: ArrivalModelConfig = ArrivalModelConfig(),
    region_config: ReceptionRegionConfig = ReceptionRegionConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        raise ValueError(f"Receiver {receiver_id} is not present in frame {frame.frame_id}")
    velocity = velocities.get(
        receiver_id,
        VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds),
    )
    if velocity.speed > 0.5:
        direction = np.asarray([velocity.vx, velocity.vy], dtype=float) / velocity.speed
    else:
        direction = np.asarray([float(attacking_direction), 0.0], dtype=float)
    lateral_direction = np.asarray([-direction[1], direction[0]], dtype=float)

    max_ahead = min(
        region_config.max_ahead_m,
        max(
            region_config.min_ahead_m,
            velocity.speed * region_config.movement_horizon_s,
        ),
    )
    resolution = max(region_config.resolution_m, 0.25)
    ahead_values = np.unique(
        np.append(
            np.arange(
                -region_config.local_radius_m,
                max_ahead + 0.5 * resolution,
                resolution,
            ),
            0.0,
        )
    )
    lateral_limit = max(
        region_config.local_radius_m,
        2.0 * region_config.movement_lateral_sigma_m,
    )
    lateral_values = np.unique(
        np.append(
            np.arange(
                -lateral_limit,
                lateral_limit + 0.5 * resolution,
                resolution,
            ),
            0.0,
        )
    )
    ahead, lateral = np.meshgrid(ahead_values, lateral_values)
    relative = (
        ahead.ravel()[:, None] * direction[None, :]
        + lateral.ravel()[:, None] * lateral_direction[None, :]
    )
    points = relative + np.asarray([receiver.x, receiver.y], dtype=float)

    half_length = FIELD_LENGTH / 2.0
    half_width = FIELD_WIDTH / 2.0
    inside = (
        (points[:, 0] >= -half_length)
        & (points[:, 0] <= half_length)
        & (points[:, 1] >= -half_width)
        & (points[:, 1] <= half_width)
    )
    points = points[inside]
    ahead_flat = ahead.ravel()[inside]
    lateral_flat = lateral.ravel()[inside]

    radial_distance = np.hypot(ahead_flat, lateral_flat)
    local_prior = np.exp(
        -0.5
        * (radial_distance / max(region_config.local_sigma_m, 1e-9)) ** 2
    ) * (radial_distance <= region_config.local_radius_m)
    movement_anchor = min(max_ahead, max(1.0, 0.8 * velocity.speed))
    movement_prior = (
        np.exp(
            -0.5
            * (
                (ahead_flat - movement_anchor)
                / max(region_config.movement_longitudinal_sigma_m, 1e-9)
            )
            ** 2
        )
        * np.exp(
            -0.5
            * (
                lateral_flat
                / max(region_config.movement_lateral_sigma_m, 1e-9)
            )
            ** 2
        )
        * (ahead_flat >= 0.0)
        * (ahead_flat <= max_ahead)
    )
    local_weight = float(np.clip(region_config.local_weight, 0.0, 1.0))
    prior = local_weight * local_prior + (1.0 - local_weight) * movement_prior
    max_prior = float(np.max(prior)) if len(prior) else 0.0
    keep = prior >= region_config.min_relative_prior * max(max_prior, 1e-12)
    points = points[keep]
    prior = prior[keep]
    total_prior = float(np.sum(prior))
    if total_prior <= 1e-12:
        raise ValueError(f"Receiver {receiver_id} has no valid target-region points")
    return points, prior / total_prior


def reception_region_estimate(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    receiver_id: str,
    passer_id: str | None,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
    region_config: ReceptionRegionConfig = ReceptionRegionConfig(),
    receive_value_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> ReceptionRegionEstimate:
    from .obso import is_offside_position, score_at_points

    points, intent_prior = receiver_target_region(
        frame,
        velocities,
        receiver_id,
        attacking_direction,
        config,
        region_config,
    )
    if receive_value_fn is None:
        receive_value = np.asarray(
            score_at_points(points, attacking_direction),
            dtype=float,
        )
    else:
        receive_value = np.asarray(
            receive_value_fn(points, attacking_direction),
            dtype=float,
        )
    if receive_value.shape != (len(points),):
        raise ValueError("receive_value_fn must return one value per target point")

    offside = is_offside_position(
        frame,
        receiver_id,
        attacking_team_id,
        ball_xy,
        attacking_direction,
    )
    if offside:
        receive_probability = np.zeros(len(points), dtype=float)
    else:
        receive_probability = np.asarray(
            [
                point_reception_estimate(
                    frame,
                    velocities,
                    receiver_id,
                    attacking_team_id,
                    ball_xy,
                    (float(point[0]), float(point[1])),
                    config,
                    passer_id=passer_id,
                ).receive_probability
                for point in points
            ],
            dtype=float,
        )

    weighted_value = intent_prior * receive_probability * receive_value
    peak_index = int(np.argmax(weighted_value))
    return ReceptionRegionEstimate(
        receiver_id=receiver_id,
        point_count=len(points),
        expected_receive_probability=float(
            np.sum(intent_prior * receive_probability)
        ),
        expected_receive_value=float(np.sum(intent_prior * receive_value)),
        option_value=float(np.sum(weighted_value)),
        peak_x=float(points[peak_index, 0]),
        peak_y=float(points[peak_index, 1]),
        peak_prior=float(intent_prior[peak_index]),
        peak_receive_probability=float(receive_probability[peak_index]),
        peak_receive_value=float(receive_value[peak_index]),
        peak_weighted_value=float(weighted_value[peak_index]),
        points=points,
        intent_prior=intent_prior,
        receive_probability=receive_probability,
        receive_value=receive_value,
        weighted_value=weighted_value,
    )
