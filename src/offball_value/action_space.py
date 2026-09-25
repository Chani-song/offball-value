"""Kinematically feasible short-horizon endpoint actions.

The action grid in this module is independent of the 50 by 32 OBSO grid.  A
grid endpoint is admissible only when an explicit accelerate-then-cruise
trajectory can reach it while respecting shared speed, acceleration, and pitch
bounds.  Current velocity is estimated from pre-decision tracking only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaObjectState,
)
from .pass_dynamics import ArrivalModelConfig, VelocityEstimate, estimate_velocity


@dataclass(frozen=True)
class EndpointActionConfig:
    horizon_seconds: float = 2.0
    grid_resolution_m: float = 1.0
    max_speed_mps: float = 9.0
    max_acceleration_mps2: float = 3.5
    max_deceleration_mps2: float = 3.5
    velocity_history_seconds: float = 0.4
    velocity_estimator: str = "linear"
    motion_model: str = "single_acceleration"
    brake_duration_step_s: float = 0.1
    deceleration_step_mps2: float = 0.5
    field_length_m: float = FIELD_LENGTH
    field_width_m: float = FIELD_WIDTH
    numerical_tolerance: float = 1e-8

    def validate(self) -> None:
        positive = {
            "horizon_seconds": self.horizon_seconds,
            "grid_resolution_m": self.grid_resolution_m,
            "max_speed_mps": self.max_speed_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "velocity_history_seconds": self.velocity_history_seconds,
            "brake_duration_step_s": self.brake_duration_step_s,
            "deceleration_step_mps2": self.deceleration_step_mps2,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Endpoint-action parameters must be positive: {invalid}")
        if self.velocity_estimator not in {"linear", "quadratic_endpoint"}:
            raise ValueError(
                "velocity_estimator must be linear or quadratic_endpoint"
            )
        if self.motion_model not in {
            "single_acceleration",
            "brake_turn_accelerate",
        }:
            raise ValueError(
                "motion_model must be single_acceleration or "
                "brake_turn_accelerate"
            )


@dataclass(frozen=True)
class EndpointMotion:
    feasible: bool
    acceleration_x_mps2: float
    acceleration_y_mps2: float
    acceleration_mps2: float
    acceleration_duration_s: float
    cruise_duration_s: float
    terminal_vx_mps: float
    terminal_vy_mps: float
    terminal_speed_mps: float
    failure_reason: str | None = None
    motion_model: str = "single_acceleration"
    brake_acceleration_x_mps2: float = 0.0
    brake_acceleration_y_mps2: float = 0.0
    brake_acceleration_mps2: float = 0.0
    brake_duration_s: float = 0.0


@dataclass(frozen=True)
class EndpointAction:
    player_id: str
    endpoint_x: float
    endpoint_y: float
    labels: tuple[str, ...]
    optimization_eligible: bool
    distance_from_start_m: float
    motion: EndpointMotion

    @property
    def action_id(self) -> str:
        return f"{self.player_id}:{self.endpoint_x:.4f}:{self.endpoint_y:.4f}"

    def as_record(self, match_id: str, frame_id: int) -> dict[str, object]:
        return {
            "match_id": match_id,
            "frame_id": frame_id,
            "player_id": self.player_id,
            "action_id": f"{match_id}:{frame_id}:{self.action_id}",
            "endpoint_x": self.endpoint_x,
            "endpoint_y": self.endpoint_y,
            "labels": "|".join(self.labels),
            "optimization_eligible": self.optimization_eligible,
            "kinematically_feasible": self.motion.feasible,
            "distance_from_start_m": self.distance_from_start_m,
            "acceleration_x_mps2": self.motion.acceleration_x_mps2,
            "acceleration_y_mps2": self.motion.acceleration_y_mps2,
            "acceleration_mps2": self.motion.acceleration_mps2,
            "acceleration_duration_s": self.motion.acceleration_duration_s,
            "cruise_duration_s": self.motion.cruise_duration_s,
            "terminal_vx_mps": self.motion.terminal_vx_mps,
            "terminal_vy_mps": self.motion.terminal_vy_mps,
            "terminal_speed_mps": self.motion.terminal_speed_mps,
            "failure_reason": self.motion.failure_reason,
            "motion_model": self.motion.motion_model,
            "brake_acceleration_x_mps2": (
                self.motion.brake_acceleration_x_mps2
            ),
            "brake_acceleration_y_mps2": (
                self.motion.brake_acceleration_y_mps2
            ),
            "brake_acceleration_mps2": self.motion.brake_acceleration_mps2,
            "brake_duration_s": self.motion.brake_duration_s,
        }


@dataclass(frozen=True)
class PlayerEndpointActionSet:
    player_id: str
    start_x: float
    start_y: float
    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float
    velocity_sample_count: int
    velocity_window_seconds: float
    actions: tuple[EndpointAction, ...]
    velocity_estimator: str = "linear"

    @property
    def optimization_actions(self) -> tuple[EndpointAction, ...]:
        return tuple(action for action in self.actions if action.optimization_eligible)


def _in_pitch(
    x: float,
    y: float,
    config: EndpointActionConfig,
) -> bool:
    tolerance = config.numerical_tolerance
    return (
        -config.field_length_m / 2.0 - tolerance
        <= x
        <= config.field_length_m / 2.0 + tolerance
        and -config.field_width_m / 2.0 - tolerance
        <= y
        <= config.field_width_m / 2.0 + tolerance
    )


def _trajectory_stays_in_pitch(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    acceleration_xy: tuple[float, float],
    acceleration_duration_s: float,
    endpoint_xy: tuple[float, float],
    config: EndpointActionConfig,
) -> bool:
    """Check the exact coordinate extrema of the parabolic acceleration phase."""

    if not _in_pitch(*start_xy, config) or not _in_pitch(*endpoint_xy, config):
        return False

    phase_end = []
    for position, velocity, acceleration in zip(
        start_xy,
        initial_velocity_xy,
        acceleration_xy,
    ):
        values = [
            position,
            position
            + velocity * acceleration_duration_s
            + 0.5 * acceleration * acceleration_duration_s**2,
        ]
        if abs(acceleration) > config.numerical_tolerance:
            turning_time = -velocity / acceleration
            if 0.0 < turning_time < acceleration_duration_s:
                values.append(
                    position
                    + velocity * turning_time
                    + 0.5 * acceleration * turning_time**2
                )
        phase_end.append((min(values), max(values)))

    x_bound = config.field_length_m / 2.0 + config.numerical_tolerance
    y_bound = config.field_width_m / 2.0 + config.numerical_tolerance
    return (
        phase_end[0][0] >= -x_bound
        and phase_end[0][1] <= x_bound
        and phase_end[1][0] >= -y_bound
        and phase_end[1][1] <= y_bound
    )


def _failed_motion(
    initial_velocity_xy: tuple[float, float],
    reason: str,
) -> EndpointMotion:
    speed = math.hypot(*initial_velocity_xy)
    return EndpointMotion(
        feasible=False,
        acceleration_x_mps2=math.nan,
        acceleration_y_mps2=math.nan,
        acceleration_mps2=math.nan,
        acceleration_duration_s=math.nan,
        cruise_duration_s=math.nan,
        terminal_vx_mps=math.nan,
        terminal_vy_mps=math.nan,
        terminal_speed_mps=speed,
        failure_reason=reason,
    )


def solve_endpoint_motion(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    config: EndpointActionConfig = EndpointActionConfig(),
) -> EndpointMotion:
    """Find a bounded accelerate-then-cruise trajectory to one endpoint.

    The player uses one constant two-dimensional acceleration for ``tau``
    seconds and then continues at the resulting velocity.  Among feasible
    solutions, the longest acceleration phase is used because it requires the
    smallest acceleration magnitude for the same endpoint, subject to the
    terminal speed cap.
    """

    config.validate()
    if not all(math.isfinite(value) for value in (*start_xy, *initial_velocity_xy, *endpoint_xy)):
        raise ValueError("Start, velocity, and endpoint coordinates must be finite")
    if not _in_pitch(*endpoint_xy, config):
        return _failed_motion(initial_velocity_xy, "endpoint_outside_pitch")

    vx, vy = initial_velocity_xy
    initial_speed = math.hypot(vx, vy)
    if initial_speed > config.max_speed_mps + config.numerical_tolerance:
        return _failed_motion(initial_velocity_xy, "initial_speed_above_cap")

    horizon = config.horizon_seconds
    delta_x = endpoint_xy[0] - start_xy[0] - vx * horizon
    delta_y = endpoint_xy[1] - start_xy[1] - vy * horizon
    delta_norm = math.hypot(delta_x, delta_y)

    if delta_norm <= config.numerical_tolerance:
        if not _trajectory_stays_in_pitch(
            start_xy,
            initial_velocity_xy,
            (0.0, 0.0),
            0.0,
            endpoint_xy,
            config,
        ):
            return _failed_motion(initial_velocity_xy, "trajectory_leaves_pitch")
        return EndpointMotion(
            feasible=True,
            acceleration_x_mps2=0.0,
            acceleration_y_mps2=0.0,
            acceleration_mps2=0.0,
            acceleration_duration_s=0.0,
            cruise_duration_s=horizon,
            terminal_vx_mps=vx,
            terminal_vy_mps=vy,
            terminal_speed_mps=initial_speed,
        )

    maximum_acceleration_displacement = (
        0.5 * config.max_acceleration_mps2 * horizon**2
    )
    if delta_norm > maximum_acceleration_displacement + config.numerical_tolerance:
        return _failed_motion(initial_velocity_xy, "acceleration_limit")

    radicand = max(
        0.0,
        horizon**2 - 2.0 * delta_norm / config.max_acceleration_mps2,
    )
    minimum_tau = horizon - math.sqrt(radicand)
    q_min = 1.0 / (horizon - 0.5 * minimum_tau)
    q_max = 2.0 / horizon

    # |v0 + q*delta| <= vmax.  Initial speed is already within the cap,
    # therefore the positive quadratic root is the relevant upper bound.
    quadratic_a = delta_norm**2
    quadratic_b = 2.0 * (vx * delta_x + vy * delta_y)
    quadratic_c = initial_speed**2 - config.max_speed_mps**2
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    if discriminant < -config.numerical_tolerance:
        return _failed_motion(initial_velocity_xy, "speed_limit")
    discriminant = max(0.0, discriminant)
    positive_root = (-quadratic_b + math.sqrt(discriminant)) / (2.0 * quadratic_a)
    selected_q = min(q_max, positive_root)
    if selected_q < q_min - config.numerical_tolerance or selected_q <= 0.0:
        return _failed_motion(initial_velocity_xy, "speed_limit")
    selected_q = max(selected_q, q_min)

    acceleration_duration = 2.0 * (horizon - 1.0 / selected_q)
    acceleration_duration = min(horizon, max(minimum_tau, acceleration_duration))
    denominator = acceleration_duration * (
        horizon - 0.5 * acceleration_duration
    )
    if denominator <= config.numerical_tolerance:
        return _failed_motion(initial_velocity_xy, "degenerate_trajectory")
    ax = delta_x / denominator
    ay = delta_y / denominator
    acceleration = math.hypot(ax, ay)
    terminal_vx = vx + ax * acceleration_duration
    terminal_vy = vy + ay * acceleration_duration
    terminal_speed = math.hypot(terminal_vx, terminal_vy)

    if acceleration > config.max_acceleration_mps2 + 1e-6:
        return _failed_motion(initial_velocity_xy, "acceleration_limit")
    if terminal_speed > config.max_speed_mps + 1e-6:
        return _failed_motion(initial_velocity_xy, "speed_limit")
    if not _trajectory_stays_in_pitch(
        start_xy,
        initial_velocity_xy,
        (ax, ay),
        acceleration_duration,
        endpoint_xy,
        config,
    ):
        return _failed_motion(initial_velocity_xy, "trajectory_leaves_pitch")

    return EndpointMotion(
        feasible=True,
        acceleration_x_mps2=float(ax),
        acceleration_y_mps2=float(ay),
        acceleration_mps2=float(acceleration),
        acceleration_duration_s=float(acceleration_duration),
        cruise_duration_s=float(horizon - acceleration_duration),
        terminal_vx_mps=float(terminal_vx),
        terminal_vy_mps=float(terminal_vy),
        terminal_speed_mps=float(terminal_speed),
    )


def _phase_extrema(
    position: float,
    velocity: float,
    acceleration: float,
    duration_s: float,
) -> tuple[float, float]:
    values = [
        position,
        position + velocity * duration_s + 0.5 * acceleration * duration_s**2,
    ]
    if abs(acceleration) > 1e-12:
        turning_time = -velocity / acceleration
        if 0.0 < turning_time < duration_s:
            values.append(
                position
                + velocity * turning_time
                + 0.5 * acceleration * turning_time**2
            )
    return min(values), max(values)


def _two_phase_stays_in_pitch(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    brake_acceleration_xy: tuple[float, float],
    brake_duration_s: float,
    turn_acceleration_xy: tuple[float, float],
    turn_duration_s: float,
    endpoint_xy: tuple[float, float],
    config: EndpointActionConfig,
) -> bool:
    if not _in_pitch(*start_xy, config) or not _in_pitch(*endpoint_xy, config):
        return False

    phase_one_end = tuple(
        position
        + velocity * brake_duration_s
        + 0.5 * acceleration * brake_duration_s**2
        for position, velocity, acceleration in zip(
            start_xy,
            initial_velocity_xy,
            brake_acceleration_xy,
        )
    )
    phase_one_velocity = tuple(
        velocity + acceleration * brake_duration_s
        for velocity, acceleration in zip(
            initial_velocity_xy,
            brake_acceleration_xy,
        )
    )
    bounds = []
    for position, velocity, brake_acceleration, phase_position, phase_velocity, turn_acceleration in zip(
        start_xy,
        initial_velocity_xy,
        brake_acceleration_xy,
        phase_one_end,
        phase_one_velocity,
        turn_acceleration_xy,
    ):
        first = _phase_extrema(
            position,
            velocity,
            brake_acceleration,
            brake_duration_s,
        )
        second = _phase_extrema(
            phase_position,
            phase_velocity,
            turn_acceleration,
            turn_duration_s,
        )
        bounds.append((min(first[0], second[0]), max(first[1], second[1])))

    x_bound = config.field_length_m / 2.0 + config.numerical_tolerance
    y_bound = config.field_width_m / 2.0 + config.numerical_tolerance
    return (
        bounds[0][0] >= -x_bound
        and bounds[0][1] <= x_bound
        and bounds[1][0] >= -y_bound
        and bounds[1][1] <= y_bound
    )


def _positive_grid_values(maximum: float, step: float) -> tuple[float, ...]:
    count = int(math.floor(maximum / step))
    values = [step * index for index in range(1, count + 1)]
    if not values or maximum - values[-1] > 1e-9:
        values.append(maximum)
    return tuple(float(value) for value in values if value > 0.0)


def solve_brake_turn_endpoint_motion(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    config: EndpointActionConfig,
) -> EndpointMotion:
    """Find a causal brake-then-turn/re-accelerate trajectory.

    The v0.1 single-acceleration solution is retained whenever it is feasible.
    Otherwise, phase one decelerates along the incoming velocity without
    reversing it.  Phase two applies one bounded two-dimensional acceleration
    for the remaining horizon.  Brake duration and magnitude are searched on
    deterministic grids so every accepted endpoint has an explicit witness
    trajectory.
    """

    config.validate()
    baseline = solve_endpoint_motion(
        start_xy,
        initial_velocity_xy,
        endpoint_xy,
        config,
    )
    if baseline.feasible:
        return baseline
    if baseline.failure_reason in {
        "endpoint_outside_pitch",
        "initial_speed_above_cap",
    }:
        return baseline

    vx, vy = initial_velocity_xy
    initial_speed = math.hypot(vx, vy)
    if initial_speed <= config.numerical_tolerance:
        return _failed_motion(initial_velocity_xy, "brake_turn_limit")
    ux, uy = vx / initial_speed, vy / initial_speed
    horizon = config.horizon_seconds
    duration_values = _positive_grid_values(
        horizon,
        config.brake_duration_step_s,
    )
    best: tuple[float, float, EndpointMotion] | None = None

    for brake_duration in duration_values:
        maximum_without_reversal = initial_speed / brake_duration
        maximum_deceleration = min(
            config.max_deceleration_mps2,
            maximum_without_reversal,
        )
        deceleration_values = set(
            _positive_grid_values(
                maximum_deceleration,
                config.deceleration_step_mps2,
            )
        )
        deceleration_values.add(float(maximum_deceleration))
        turn_duration = horizon - brake_duration
        for deceleration in sorted(deceleration_values):
            brake_ax = -ux * deceleration
            brake_ay = -uy * deceleration
            phase_x = (
                start_xy[0]
                + vx * brake_duration
                + 0.5 * brake_ax * brake_duration**2
            )
            phase_y = (
                start_xy[1]
                + vy * brake_duration
                + 0.5 * brake_ay * brake_duration**2
            )
            phase_vx = vx + brake_ax * brake_duration
            phase_vy = vy + brake_ay * brake_duration

            if turn_duration <= config.numerical_tolerance:
                if math.hypot(
                    endpoint_xy[0] - phase_x,
                    endpoint_xy[1] - phase_y,
                ) > config.numerical_tolerance:
                    continue
                turn_ax = 0.0
                turn_ay = 0.0
            else:
                denominator = 0.5 * turn_duration**2
                turn_ax = (
                    endpoint_xy[0] - phase_x - phase_vx * turn_duration
                ) / denominator
                turn_ay = (
                    endpoint_xy[1] - phase_y - phase_vy * turn_duration
                ) / denominator
            turn_acceleration = math.hypot(turn_ax, turn_ay)
            if turn_acceleration > config.max_acceleration_mps2 + 1e-6:
                continue
            terminal_vx = phase_vx + turn_ax * turn_duration
            terminal_vy = phase_vy + turn_ay * turn_duration
            terminal_speed = math.hypot(terminal_vx, terminal_vy)
            if terminal_speed > config.max_speed_mps + 1e-6:
                continue
            if not _two_phase_stays_in_pitch(
                start_xy,
                initial_velocity_xy,
                (brake_ax, brake_ay),
                brake_duration,
                (turn_ax, turn_ay),
                turn_duration,
                endpoint_xy,
                config,
            ):
                continue

            motion = EndpointMotion(
                feasible=True,
                acceleration_x_mps2=float(turn_ax),
                acceleration_y_mps2=float(turn_ay),
                acceleration_mps2=float(turn_acceleration),
                acceleration_duration_s=float(turn_duration),
                cruise_duration_s=0.0,
                terminal_vx_mps=float(terminal_vx),
                terminal_vy_mps=float(terminal_vy),
                terminal_speed_mps=float(terminal_speed),
                motion_model="brake_turn_accelerate",
                brake_acceleration_x_mps2=float(brake_ax),
                brake_acceleration_y_mps2=float(brake_ay),
                brake_acceleration_mps2=float(deceleration),
                brake_duration_s=float(brake_duration),
            )
            effort = (
                deceleration**2 * brake_duration
                + turn_acceleration**2 * turn_duration
            )
            rank = (effort, brake_duration)
            if best is None or rank < best[:2]:
                best = (rank[0], rank[1], motion)

    if best is not None:
        return best[2]
    return _failed_motion(initial_velocity_xy, "brake_turn_limit")


def motion_position_at_time(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    motion: EndpointMotion,
    time_s: float,
) -> tuple[float, float]:
    """Evaluate a feasible endpoint trajectory for plotting and tests."""

    if not motion.feasible:
        raise ValueError("Cannot evaluate an infeasible endpoint motion")
    if time_s < 0:
        raise ValueError("time_s must be non-negative")
    total_duration = (
        motion.brake_duration_s
        + motion.acceleration_duration_s
        + motion.cruise_duration_s
    )
    if time_s > total_duration + 1e-9:
        raise ValueError("time_s exceeds the endpoint trajectory horizon")
    if motion.motion_model == "brake_turn_accelerate":
        brake_time = min(time_s, motion.brake_duration_s)
        phase_x = (
            start_xy[0]
            + initial_velocity_xy[0] * brake_time
            + 0.5 * motion.brake_acceleration_x_mps2 * brake_time**2
        )
        phase_y = (
            start_xy[1]
            + initial_velocity_xy[1] * brake_time
            + 0.5 * motion.brake_acceleration_y_mps2 * brake_time**2
        )
        if time_s <= motion.brake_duration_s:
            return float(phase_x), float(phase_y)
        brake_vx = (
            initial_velocity_xy[0]
            + motion.brake_acceleration_x_mps2 * motion.brake_duration_s
        )
        brake_vy = (
            initial_velocity_xy[1]
            + motion.brake_acceleration_y_mps2 * motion.brake_duration_s
        )
        acceleration_time = min(
            time_s - motion.brake_duration_s,
            motion.acceleration_duration_s,
        )
        x = (
            phase_x
            + brake_vx * acceleration_time
            + 0.5 * motion.acceleration_x_mps2 * acceleration_time**2
        )
        y = (
            phase_y
            + brake_vy * acceleration_time
            + 0.5 * motion.acceleration_y_mps2 * acceleration_time**2
        )
        cruise = max(
            0.0,
            time_s - motion.brake_duration_s - motion.acceleration_duration_s,
        )
        return (
            float(x + motion.terminal_vx_mps * cruise),
            float(y + motion.terminal_vy_mps * cruise),
        )

    tau = min(time_s, motion.acceleration_duration_s)
    x = (
        start_xy[0]
        + initial_velocity_xy[0] * tau
        + 0.5 * motion.acceleration_x_mps2 * tau**2
    )
    y = (
        start_xy[1]
        + initial_velocity_xy[1] * tau
        + 0.5 * motion.acceleration_y_mps2 * tau**2
    )
    cruise = max(0.0, time_s - motion.acceleration_duration_s)
    return (
        float(x + motion.terminal_vx_mps * cruise),
        float(y + motion.terminal_vy_mps * cruise),
    )


def _cap_velocity(
    velocity: VelocityEstimate,
    max_speed_mps: float,
) -> VelocityEstimate:
    if velocity.speed <= max_speed_mps or velocity.speed <= 1e-12:
        return velocity
    scale = max_speed_mps / velocity.speed
    return replace(
        velocity,
        vx=velocity.vx * scale,
        vy=velocity.vy * scale,
        speed=max_speed_mps,
    )


def _grid_axis_values(
    center: float,
    radius: float,
    half_extent: float,
    resolution: float,
) -> list[float]:
    # Anchor the global grid at the negative pitch boundary.  With a 105 m
    # pitch and 1 m resolution, x therefore runs -52.5, -51.5, ..., 52.5.
    lower_index = max(0, int(math.ceil((center - radius + half_extent) / resolution)))
    upper_index = min(
        int(math.floor(2.0 * half_extent / resolution)),
        int(math.floor((center + radius + half_extent) / resolution)),
    )
    return [
        float(-half_extent + index * resolution)
        for index in range(lower_index, upper_index + 1)
    ]


def _build_action(
    player_id: str,
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    labels: tuple[str, ...],
    optimization_eligible: bool,
    config: EndpointActionConfig,
) -> EndpointAction:
    if config.motion_model == "brake_turn_accelerate":
        motion = solve_brake_turn_endpoint_motion(
            start_xy,
            initial_velocity_xy,
            endpoint_xy,
            config,
        )
    else:
        motion = solve_endpoint_motion(
            start_xy,
            initial_velocity_xy,
            endpoint_xy,
            config,
        )
    return EndpointAction(
        player_id=player_id,
        endpoint_x=float(endpoint_xy[0]),
        endpoint_y=float(endpoint_xy[1]),
        labels=labels,
        optimization_eligible=bool(optimization_eligible and motion.feasible),
        distance_from_start_m=float(
            math.hypot(endpoint_xy[0] - start_xy[0], endpoint_xy[1] - start_xy[1])
        ),
        motion=motion,
    )


def generate_player_endpoint_actions(
    player: BundesligaObjectState,
    velocity: VelocityEstimate,
    observed_endpoint_xy: tuple[float, float] | None = None,
    config: EndpointActionConfig = EndpointActionConfig(),
) -> PlayerEndpointActionSet:
    """Generate feasible global-grid actions plus exact reference endpoints."""

    config.validate()
    velocity = _cap_velocity(velocity, config.max_speed_mps)
    start = (player.x, player.y)
    initial_velocity = (velocity.vx, velocity.vy)
    horizon = config.horizon_seconds
    cv_endpoint = (
        player.x + velocity.vx * horizon,
        player.y + velocity.vy * horizon,
    )
    if config.motion_model == "brake_turn_accelerate":
        grid_center = start
        grid_radius = config.max_speed_mps * horizon
    else:
        grid_center = cv_endpoint
        grid_radius = 0.5 * config.max_acceleration_mps2 * horizon**2
    xs = _grid_axis_values(
        grid_center[0],
        grid_radius,
        config.field_length_m / 2.0,
        config.grid_resolution_m,
    )
    ys = _grid_axis_values(
        grid_center[1],
        grid_radius,
        config.field_width_m / 2.0,
        config.grid_resolution_m,
    )

    actions_by_coordinate: dict[tuple[float, float], EndpointAction] = {}

    def add_action(action: EndpointAction) -> None:
        key = (round(action.endpoint_x, 6), round(action.endpoint_y, 6))
        existing = actions_by_coordinate.get(key)
        if existing is None:
            actions_by_coordinate[key] = action
            return
        labels = tuple(dict.fromkeys((*existing.labels, *action.labels)))
        actions_by_coordinate[key] = replace(
            existing,
            labels=labels,
            optimization_eligible=(
                existing.optimization_eligible or action.optimization_eligible
            ),
        )

    for x in xs:
        for y in ys:
            action = _build_action(
                player.object_id,
                start,
                initial_velocity,
                (x, y),
                ("grid",),
                True,
                config,
            )
            if action.motion.feasible:
                add_action(action)

    # Exact hold and constant-velocity endpoints are action-independent
    # references.  They participate in optimization only when physically
    # feasible and inside the pitch.
    add_action(
        _build_action(
            player.object_id,
            start,
            initial_velocity,
            start,
            ("hold",),
            True,
            config,
        )
    )
    add_action(
        _build_action(
            player.object_id,
            start,
            initial_velocity,
            cv_endpoint,
            ("constant_velocity",),
            True,
            config,
        )
    )
    if observed_endpoint_xy is not None:
        # The observed future is a diagnostic reference, never a source of an
        # optimizer-only candidate unless it already coincides with a grid or
        # action-independent reference point.
        add_action(
            _build_action(
                player.object_id,
                start,
                initial_velocity,
                observed_endpoint_xy,
                ("observed",),
                False,
                config,
            )
        )

    actions = tuple(
        sorted(
            actions_by_coordinate.values(),
            key=lambda action: (
                not action.optimization_eligible,
                action.endpoint_x,
                action.endpoint_y,
            ),
        )
    )
    return PlayerEndpointActionSet(
        player_id=player.object_id,
        start_x=player.x,
        start_y=player.y,
        initial_vx_mps=velocity.vx,
        initial_vy_mps=velocity.vy,
        initial_speed_mps=velocity.speed,
        velocity_sample_count=velocity.sample_count,
        velocity_window_seconds=velocity.window_seconds,
        actions=actions,
        velocity_estimator=config.velocity_estimator,
    )


def generate_scene_attacker_endpoint_sets(
    frame: BundesligaFrame,
    history_frames: Iterable[BundesligaFrame],
    attacker_ids: Iterable[str],
    observed_horizon_frame: BundesligaFrame | None = None,
    config: EndpointActionConfig = EndpointActionConfig(),
) -> tuple[PlayerEndpointActionSet, ...]:
    """Generate action sets for the preselected eligible attackers in a scene."""

    ordered_history = tuple(sorted(history_frames, key=lambda item: item.frame_id))
    velocity_config = ArrivalModelConfig(
        history_seconds=config.velocity_history_seconds,
        velocity_estimator=config.velocity_estimator,
        player_max_speed_mps=config.max_speed_mps,
        player_acceleration_mps2=config.max_acceleration_mps2,
        observed_speed_cap_mps=config.max_speed_mps,
    )
    action_sets = []
    for player_id in attacker_ids:
        player = frame.players.get(player_id)
        if player is None:
            continue
        velocity = estimate_velocity(
            ordered_history,
            player_id,
            frame.frame_id,
            velocity_config,
        )
        observed_player = (
            observed_horizon_frame.players.get(player_id)
            if observed_horizon_frame is not None
            else None
        )
        observed_endpoint = (
            (observed_player.x, observed_player.y)
            if observed_player is not None
            else None
        )
        action_sets.append(
            generate_player_endpoint_actions(
                player,
                velocity,
                observed_endpoint,
                config,
            )
        )
    return tuple(action_sets)


def apply_endpoint_action(
    frame: BundesligaFrame,
    action: EndpointAction,
) -> BundesligaFrame:
    """Move exactly one player while preserving every other state field."""

    player = frame.players.get(action.player_id)
    if player is None:
        raise KeyError(f"Player {action.player_id} is not present in frame {frame.frame_id}")
    moved = replace(
        player,
        x=action.endpoint_x,
        y=action.endpoint_y,
        speed=action.motion.terminal_speed_mps * 3.6 if action.motion.feasible else player.speed,
    )
    return frame.with_player(action.player_id, moved)
