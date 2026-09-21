"""Multi-step attacker reachability with tangential and normal acceleration.

The model evolves ``(x, y, speed, heading)`` at 0.1-second intervals.  Its
controls are expressed in the player's moving Frenet frame:

* tangential acceleration changes speed;
* normal acceleration rotates the velocity vector and therefore produces a
  curved path while preserving speed when tangential acceleration is zero.

The Fernández influence ellipse remains a proposal/visual reference only.  It
is deliberately not used as the feasibility boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Iterable

import numpy as np

from .bundesliga import FIELD_LENGTH, FIELD_WIDTH, BundesligaObjectState
from .empirical_action_space import (
    CausalMotionState,
    EmpiricalEndpointAction,
    EmpiricalEndpointActionSet,
)
from .fernandez_influence import (
    FernandezInfluenceEllipse,
    fernandez_influence_ellipse,
)


@dataclass(frozen=True)
class SteeringReachabilityConfig:
    horizon_seconds: float = 2.0
    integration_step_seconds: float = 0.1
    grid_resolution_m: float = 1.0
    state_position_resolution_m: float = 1.0
    state_speed_resolution_mps: float = 0.5
    # At 9 m/s, maximum normal control turns about 3.8 degrees per 0.1 s.
    # Bins must be narrower than that increment or early curvature aliases
    # into the coast state before it can accumulate.
    state_heading_bins: int = 96
    control_direction_count: int = 16
    max_speed_mps: float = 9.0
    max_tangential_acceleration_mps2: float = 4.5
    max_tangential_deceleration_mps2: float = 6.0
    max_normal_acceleration_mps2: float = 6.0
    minimum_plant_cut_initial_speed_mps: float = 2.0
    max_plant_cut_deceleration_mps2: float = 9.0
    plant_cut_braking_fractions: tuple[float, ...] = (2.0 / 3.0, 5.0 / 6.0, 1.0)
    plant_cut_durations_s: tuple[float, ...] = (0.1, 0.2, 0.3)
    plant_cut_acceleration_fractions: tuple[float, ...] = (0.5, 1.0)
    plant_cut_direction_count: int = 16
    minimum_plant_cut_angle_degrees: float = 45.0
    stationary_speed_threshold_mps: float = 0.1
    curvature_speed_floor_mps: float = 0.5
    terminal_heading_bins: int = 8
    variants_per_endpoint: int = 8
    path_sample_seconds: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6, 2.0)
    empirical_endpoint_tolerance_m: float = 1.0
    empirical_heading_tolerance_degrees: float = 45.0
    minimum_heading_speed_mps: float = 0.5
    field_length_m: float = FIELD_LENGTH
    field_width_m: float = FIELD_WIDTH
    numerical_tolerance: float = 1e-8

    def validate(self) -> None:
        positive = {
            "horizon_seconds": self.horizon_seconds,
            "integration_step_seconds": self.integration_step_seconds,
            "grid_resolution_m": self.grid_resolution_m,
            "state_position_resolution_m": self.state_position_resolution_m,
            "state_speed_resolution_mps": self.state_speed_resolution_mps,
            "state_heading_bins": self.state_heading_bins,
            "control_direction_count": self.control_direction_count,
            "max_speed_mps": self.max_speed_mps,
            "max_tangential_acceleration_mps2": (
                self.max_tangential_acceleration_mps2
            ),
            "max_tangential_deceleration_mps2": (
                self.max_tangential_deceleration_mps2
            ),
            "max_normal_acceleration_mps2": self.max_normal_acceleration_mps2,
            "minimum_plant_cut_initial_speed_mps": (
                self.minimum_plant_cut_initial_speed_mps
            ),
            "max_plant_cut_deceleration_mps2": (
                self.max_plant_cut_deceleration_mps2
            ),
            "plant_cut_direction_count": self.plant_cut_direction_count,
            "minimum_plant_cut_angle_degrees": (
                self.minimum_plant_cut_angle_degrees
            ),
            "stationary_speed_threshold_mps": self.stationary_speed_threshold_mps,
            "curvature_speed_floor_mps": self.curvature_speed_floor_mps,
            "terminal_heading_bins": self.terminal_heading_bins,
            "variants_per_endpoint": self.variants_per_endpoint,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Steering reachability parameters must be positive: {invalid}")
        bounded_sequences = {
            "plant_cut_braking_fractions": self.plant_cut_braking_fractions,
            "plant_cut_acceleration_fractions": (
                self.plant_cut_acceleration_fractions
            ),
        }
        for name, values in bounded_sequences.items():
            if not values or any(value <= 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{name} must contain values in (0, 1]")
        if not self.plant_cut_durations_s or any(
            duration <= 0.0 for duration in self.plant_cut_durations_s
        ):
            raise ValueError("plant-cut durations must be positive")
        if self.minimum_plant_cut_angle_degrees >= 180.0:
            raise ValueError("minimum plant-cut angle must be below 180 degrees")
        steps = self.horizon_seconds / self.integration_step_seconds
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("horizon must be an integer number of integration steps")
        sample_steps = [time / self.integration_step_seconds for time in self.path_sample_seconds]
        if not sample_steps or any(
            time <= 0.0 or time > self.horizon_seconds
            for time in self.path_sample_seconds
        ):
            raise ValueError("path samples must lie inside the horizon")
        if any(abs(step - round(step)) > 1e-9 for step in sample_steps):
            raise ValueError("path samples must coincide with integration steps")
        if abs(self.path_sample_seconds[-1] - self.horizon_seconds) > 1e-9:
            raise ValueError("final path sample must equal the horizon")


@dataclass(frozen=True)
class SteeringStepResult:
    x: float
    y: float
    speed_mps: float
    heading_radians: float
    applied_tangential_acceleration_mps2: float
    applied_normal_acceleration_mps2: float


@dataclass(frozen=True)
class SteeringControlMotion:
    terminal_vx_mps: float
    terminal_vy_mps: float
    terminal_speed_mps: float
    terminal_heading_bin: int
    effort_m2ps3: float
    maximum_path_speed_mps: float
    maximum_tangential_acceleration_mps2: float
    maximum_tangential_deceleration_mps2: float
    maximum_normal_acceleration_mps2: float
    path_xy: tuple[tuple[float, float], ...]
    control_sequence: tuple[tuple[float, float], ...]
    maneuver_type: str = "continuous_steering"
    cut_angle_degrees: float | None = None
    cut_braking_mps2: float | None = None
    cut_stop_time_s: float | None = None
    cut_plant_duration_s: float | None = None
    cut_post_acceleration_mps2: float | None = None


@dataclass(frozen=True)
class SteeringEndpointAction:
    player_id: str
    endpoint_x: float
    endpoint_y: float
    endpoint_cell_x: float
    endpoint_cell_y: float
    grid_snap_distance_m: float
    motion: SteeringControlMotion
    fernandez_normalized_radius: float
    inside_fernandez_contour: bool
    empirically_supported: bool = False
    nearest_empirical_endpoint_distance_m: float = math.inf
    nearest_empirical_heading_difference_degrees: float = math.inf
    empirical_primitive_match_id: str | None = None
    empirical_primitive_player_id: str | None = None
    empirical_primitive_frame_id: int | None = None
    labels: tuple[str, ...] = ("grid", "steering_feasible")
    optimization_eligible: bool = True

    @property
    def action_id(self) -> str:
        cut_token = (
            f":a{self.motion.cut_angle_degrees:.1f}"
            if self.motion.cut_angle_degrees is not None
            else ""
        )
        return (
            f"{self.player_id}:cell:{self.endpoint_cell_x:.1f}:"
            f"{self.endpoint_cell_y:.1f}:h{self.motion.terminal_heading_bin}:"
            f"{self.motion.maneuver_type}{cut_token}"
        )

    def as_record(self, match_id: str, frame_id: int) -> dict[str, object]:
        return {
            "match_id": match_id,
            "frame_id": int(frame_id),
            "player_id": self.player_id,
            "action_id": f"{match_id}:{frame_id}:{self.action_id}",
            "endpoint_x": self.endpoint_x,
            "endpoint_y": self.endpoint_y,
            "endpoint_cell_x": self.endpoint_cell_x,
            "endpoint_cell_y": self.endpoint_cell_y,
            "grid_snap_distance_m": self.grid_snap_distance_m,
            "labels": "|".join(self.labels),
            "optimization_eligible": self.optimization_eligible,
            "kinematically_feasible": True,
            "dynamically_feasible": True,
            "empirically_supported": self.empirically_supported,
            "motion_model": self.motion.maneuver_type,
            "maneuver_type": self.motion.maneuver_type,
            "terminal_vx_mps": self.motion.terminal_vx_mps,
            "terminal_vy_mps": self.motion.terminal_vy_mps,
            "terminal_speed_mps": self.motion.terminal_speed_mps,
            "terminal_heading_bin": self.motion.terminal_heading_bin,
            "control_effort_m2ps3": self.motion.effort_m2ps3,
            "maximum_path_speed_mps": self.motion.maximum_path_speed_mps,
            "maximum_tangential_acceleration_mps2": (
                self.motion.maximum_tangential_acceleration_mps2
            ),
            "maximum_tangential_deceleration_mps2": (
                self.motion.maximum_tangential_deceleration_mps2
            ),
            "maximum_normal_acceleration_mps2": (
                self.motion.maximum_normal_acceleration_mps2
            ),
            "cut_angle_degrees": self.motion.cut_angle_degrees,
            "cut_braking_mps2": self.motion.cut_braking_mps2,
            "cut_stop_time_s": self.motion.cut_stop_time_s,
            "cut_plant_duration_s": self.motion.cut_plant_duration_s,
            "cut_post_acceleration_mps2": (
                self.motion.cut_post_acceleration_mps2
            ),
            "path_xy": json.dumps(self.motion.path_xy, separators=(",", ":")),
            "control_sequence": json.dumps(
                self.motion.control_sequence, separators=(",", ":")
            ),
            "fernandez_normalized_radius": self.fernandez_normalized_radius,
            "inside_fernandez_contour": self.inside_fernandez_contour,
            "nearest_empirical_endpoint_distance_m": (
                self.nearest_empirical_endpoint_distance_m
            ),
            "nearest_empirical_heading_difference_degrees": (
                self.nearest_empirical_heading_difference_degrees
            ),
            "empirical_primitive_match_id": self.empirical_primitive_match_id,
            "empirical_primitive_player_id": self.empirical_primitive_player_id,
            "empirical_primitive_frame_id": self.empirical_primitive_frame_id,
            "failure_reason": None,
        }


@dataclass(frozen=True)
class SteeringEndpointActionSet:
    player_id: str
    start_x: float
    start_y: float
    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float
    reference_heading_radians: float
    influence_ellipse: FernandezInfluenceEllipse
    proposal_endpoints: tuple[tuple[float, float], ...]
    actions: tuple[SteeringEndpointAction, ...]

    @property
    def optimization_actions(self) -> tuple[SteeringEndpointAction, ...]:
        return tuple(action for action in self.actions if action.optimization_eligible)


@dataclass(frozen=True)
class SteeringSupportDiagnostics:
    endpoint_supported: bool | None
    endpoint_heading_supported: bool | None
    nearest_endpoint_distance_m: float | None
    nearest_heading_difference_degrees: float | None


@dataclass(frozen=True)
class _LatticeState:
    x: float
    y: float
    speed_mps: float
    heading_radians: float
    effort_m2ps3: float
    maximum_path_speed_mps: float
    maximum_tangential_acceleration_mps2: float
    maximum_tangential_deceleration_mps2: float
    maximum_normal_acceleration_mps2: float
    path_xy: tuple[tuple[float, float], ...]
    control_sequence: tuple[tuple[float, float], ...]


def _wrap_heading(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def steering_step(
    x: float,
    y: float,
    speed_mps: float,
    heading_radians: float,
    tangential_acceleration_mps2: float,
    normal_acceleration_mps2: float,
    config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> SteeringStepResult:
    """Integrate one constant-Frenet-control step with midpoint kinematics."""

    dt = config.integration_step_seconds
    requested_speed = speed_mps + tangential_acceleration_mps2 * dt
    new_speed = min(config.max_speed_mps, max(0.0, requested_speed))
    applied_tangential = (new_speed - speed_mps) / dt
    midpoint_speed = 0.5 * (speed_mps + new_speed)
    if midpoint_speed <= config.numerical_tolerance:
        return SteeringStepResult(
            float(x),
            float(y),
            float(new_speed),
            _wrap_heading(heading_radians),
            float(applied_tangential),
            0.0,
        )
    angular_velocity = normal_acceleration_mps2 / max(
        midpoint_speed, config.curvature_speed_floor_mps
    )
    heading_change = angular_velocity * dt
    midpoint_heading = heading_radians + 0.5 * heading_change
    return SteeringStepResult(
        x=float(x + midpoint_speed * math.cos(midpoint_heading) * dt),
        y=float(y + midpoint_speed * math.sin(midpoint_heading) * dt),
        speed_mps=float(new_speed),
        heading_radians=_wrap_heading(heading_radians + heading_change),
        applied_tangential_acceleration_mps2=float(applied_tangential),
        applied_normal_acceleration_mps2=float(normal_acceleration_mps2),
    )


def _moving_controls(
    config: SteeringReachabilityConfig,
) -> tuple[tuple[float, float], ...]:
    """Boundary points of an anisotropic tangential/normal control ellipse."""

    controls = [(0.0, 0.0)]
    for index in range(config.control_direction_count):
        angle = 2.0 * math.pi * index / config.control_direction_count
        cosine = math.cos(angle)
        tangential_limit = (
            config.max_tangential_acceleration_mps2
            if cosine >= 0.0
            else config.max_tangential_deceleration_mps2
        )
        controls.append(
            (
                float(tangential_limit * cosine),
                float(config.max_normal_acceleration_mps2 * math.sin(angle)),
            )
        )
    return tuple(controls)


def _launch_steps(
    state: _LatticeState,
    config: SteeringReachabilityConfig,
) -> tuple[SteeringStepResult, ...]:
    """Allow a near-stationary player to accelerate in any absolute direction."""

    dt = config.integration_step_seconds
    acceleration = config.max_tangential_acceleration_mps2
    terminal_speed = acceleration * dt
    displacement = 0.5 * acceleration * dt**2
    results = [
        SteeringStepResult(
            state.x,
            state.y,
            0.0,
            state.heading_radians,
            0.0,
            0.0,
        )
    ]
    for index in range(config.control_direction_count):
        heading = 2.0 * math.pi * index / config.control_direction_count
        results.append(
            SteeringStepResult(
                x=float(state.x + displacement * math.cos(heading)),
                y=float(state.y + displacement * math.sin(heading)),
                speed_mps=float(terminal_speed),
                heading_radians=_wrap_heading(heading),
                applied_tangential_acceleration_mps2=float(acceleration),
                applied_normal_acceleration_mps2=0.0,
            )
        )
    return tuple(results)


def _inside_pitch(x: float, y: float, config: SteeringReachabilityConfig) -> bool:
    return (
        abs(x) <= config.field_length_m / 2.0 + config.numerical_tolerance
        and abs(y) <= config.field_width_m / 2.0 + config.numerical_tolerance
    )


def _heading_bin(angle: float, count: int) -> int:
    normalized = (angle + 2.0 * math.pi) % (2.0 * math.pi)
    return int(math.floor(normalized / (2.0 * math.pi / count))) % count


def _state_key(
    state: _LatticeState,
    start_xy: tuple[float, float],
    config: SteeringReachabilityConfig,
) -> tuple[int, int, int, int]:
    return (
        int(round((state.x - start_xy[0]) / config.state_position_resolution_m)),
        int(round((state.y - start_xy[1]) / config.state_position_resolution_m)),
        int(round(state.speed_mps / config.state_speed_resolution_mps)),
        _heading_bin(state.heading_radians, config.state_heading_bins),
    )


def _snap_axis(value: float, half_extent: float, resolution: float) -> float:
    index = round((value + half_extent) / resolution)
    maximum_index = round(2.0 * half_extent / resolution)
    index = min(maximum_index, max(0, index))
    return float(-half_extent + index * resolution)


def _snap_axis_values(
    lower: float,
    upper: float,
    half_extent: float,
    resolution: float,
) -> list[float]:
    lower_index = max(0, int(math.ceil((lower + half_extent) / resolution)))
    upper_index = min(
        int(math.floor(2.0 * half_extent / resolution)),
        int(math.floor((upper + half_extent) / resolution)),
    )
    return [
        float(-half_extent + index * resolution)
        for index in range(lower_index, upper_index + 1)
    ]


def _terminal_heading_bin(
    heading_radians: float,
    reference_heading: float,
    count: int,
) -> int:
    return _heading_bin(heading_radians - reference_heading, count)


def _heading_difference_degrees(
    first_xy: tuple[float, float],
    second_xy: tuple[float, float],
) -> float:
    difference = abs(
        (math.atan2(first_xy[1], first_xy[0])
         - math.atan2(second_xy[1], second_xy[0])
         + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )
    return float(math.degrees(difference))


def _advance_state(
    state: _LatticeState,
    result: SteeringStepResult,
    sample_path: bool,
    config: SteeringReachabilityConfig,
) -> _LatticeState:
    tangential = result.applied_tangential_acceleration_mps2
    normal = result.applied_normal_acceleration_mps2
    point = ((result.x, result.y),) if sample_path else ()
    return _LatticeState(
        x=result.x,
        y=result.y,
        speed_mps=result.speed_mps,
        heading_radians=result.heading_radians,
        effort_m2ps3=state.effort_m2ps3
        + (tangential**2 + normal**2) * config.integration_step_seconds,
        maximum_path_speed_mps=max(state.maximum_path_speed_mps, result.speed_mps),
        maximum_tangential_acceleration_mps2=max(
            state.maximum_tangential_acceleration_mps2, tangential
        ),
        maximum_tangential_deceleration_mps2=max(
            state.maximum_tangential_deceleration_mps2, -tangential
        ),
        maximum_normal_acceleration_mps2=max(
            state.maximum_normal_acceleration_mps2, abs(normal)
        ),
        path_xy=state.path_xy + point,
        control_sequence=state.control_sequence + ((tangential, normal),),
    )


def enumerate_steering_motions(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    reference_heading_radians: float,
    config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> tuple[_LatticeState, ...]:
    """Enumerate quantized reachable terminal states without future tracking."""

    config.validate()
    initial_speed = math.hypot(*initial_velocity_xy)
    if initial_speed > config.max_speed_mps and initial_speed > 1e-12:
        scale = config.max_speed_mps / initial_speed
        initial_velocity_xy = (
            initial_velocity_xy[0] * scale,
            initial_velocity_xy[1] * scale,
        )
        initial_speed = config.max_speed_mps
    initial_heading = (
        math.atan2(initial_velocity_xy[1], initial_velocity_xy[0])
        if initial_speed >= config.stationary_speed_threshold_mps
        else reference_heading_radians
    )
    initial = _LatticeState(
        x=float(start_xy[0]),
        y=float(start_xy[1]),
        speed_mps=float(initial_speed),
        heading_radians=_wrap_heading(initial_heading),
        effort_m2ps3=0.0,
        maximum_path_speed_mps=float(initial_speed),
        maximum_tangential_acceleration_mps2=0.0,
        maximum_tangential_deceleration_mps2=0.0,
        maximum_normal_acceleration_mps2=0.0,
        path_xy=(),
        control_sequence=(),
    )
    states = {_state_key(initial, start_xy, config): initial}
    moving_controls = _moving_controls(config)
    sample_steps = {
        int(round(time / config.integration_step_seconds))
        for time in config.path_sample_seconds
    }
    total_steps = int(round(config.horizon_seconds / config.integration_step_seconds))
    for step_index in range(1, total_steps + 1):
        successors: dict[tuple[int, int, int, int], _LatticeState] = {}
        sample_path = step_index in sample_steps
        for state in states.values():
            if state.speed_mps < config.stationary_speed_threshold_mps:
                results = _launch_steps(state, config)
            else:
                results = tuple(
                    steering_step(
                        state.x,
                        state.y,
                        state.speed_mps,
                        state.heading_radians,
                        tangential,
                        normal,
                        config,
                    )
                    for tangential, normal in moving_controls
                )
            for result in results:
                if not _inside_pitch(result.x, result.y, config):
                    continue
                candidate = _advance_state(state, result, sample_path, config)
                key = _state_key(candidate, start_xy, config)
                existing = successors.get(key)
                candidate_rank = (
                    candidate.effort_m2ps3,
                    -math.hypot(candidate.x - start_xy[0], candidate.y - start_xy[1]),
                    candidate.heading_radians,
                )
                if existing is None:
                    successors[key] = candidate
                else:
                    existing_rank = (
                        existing.effort_m2ps3,
                        -math.hypot(existing.x - start_xy[0], existing.y - start_xy[1]),
                        existing.heading_radians,
                    )
                    if candidate_rank < existing_rank:
                        successors[key] = candidate
        states = successors
    return tuple(states.values())


def _empirical_annotation(
    action: SteeringEndpointAction,
    empirical_actions: Iterable[EmpiricalEndpointAction],
    config: SteeringReachabilityConfig,
) -> SteeringEndpointAction:
    empirical_actions = tuple(empirical_actions)
    if not empirical_actions:
        return action
    ranked = []
    for empirical in empirical_actions:
        endpoint_distance = math.hypot(
            action.endpoint_x - empirical.endpoint_x,
            action.endpoint_y - empirical.endpoint_y,
        )
        if (
            action.motion.terminal_speed_mps < config.minimum_heading_speed_mps
            or empirical.terminal_speed_mps < config.minimum_heading_speed_mps
        ):
            heading_difference = 0.0
        else:
            heading_difference = _heading_difference_degrees(
                (action.motion.terminal_vx_mps, action.motion.terminal_vy_mps),
                (empirical.terminal_vx_mps, empirical.terminal_vy_mps),
            )
        rank = (
            endpoint_distance
            + heading_difference / max(config.empirical_heading_tolerance_degrees, 1e-9),
            endpoint_distance,
            heading_difference,
            empirical.action_id,
        )
        ranked.append((rank, empirical))
    ranked.sort(key=lambda item: item[0])
    rank, nearest = ranked[0]
    endpoint_distance = float(rank[1])
    heading_difference = float(rank[2])
    supported = (
        endpoint_distance <= config.empirical_endpoint_tolerance_m
        and heading_difference <= config.empirical_heading_tolerance_degrees
    )
    return replace(
        action,
        empirically_supported=bool(supported),
        nearest_empirical_endpoint_distance_m=endpoint_distance,
        nearest_empirical_heading_difference_degrees=heading_difference,
        empirical_primitive_match_id=nearest.primitive_match_id,
        empirical_primitive_player_id=nearest.primitive_player_id,
        empirical_primitive_frame_id=nearest.primitive_frame_id,
    )


def generate_steering_endpoint_actions(
    player: BundesligaObjectState,
    ball: BundesligaObjectState,
    state: CausalMotionState,
    empirical_action_set: EmpiricalEndpointActionSet | None = None,
    config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> SteeringEndpointActionSet:
    config.validate()
    speed = state.speed_mps
    if speed > config.max_speed_mps and speed > 1e-12:
        scale = config.max_speed_mps / speed
        initial_velocity = (state.vx_mps * scale, state.vy_mps * scale)
    else:
        initial_velocity = (state.vx_mps, state.vy_mps)
    initial_speed = math.hypot(*initial_velocity)
    reference_heading = (
        math.atan2(initial_velocity[1], initial_velocity[0])
        if initial_speed >= config.stationary_speed_threshold_mps
        else state.heading_radians
    )
    ellipse = fernandez_influence_ellipse(
        (player.x, player.y),
        initial_velocity,
        (ball.x, ball.y),
    )

    radius = config.max_speed_mps * config.horizon_seconds
    xs = _snap_axis_values(
        player.x - radius,
        player.x + radius,
        config.field_length_m / 2.0,
        config.grid_resolution_m,
    )
    ys = _snap_axis_values(
        player.y - radius,
        player.y + radius,
        config.field_width_m / 2.0,
        config.grid_resolution_m,
    )
    proposal = tuple(
        (x, y)
        for x in xs
        for y in ys
        if ellipse.normalized_radius(x, y) <= 1.0 + 1e-9
    )

    terminal_states = enumerate_steering_motions(
        (player.x, player.y),
        initial_velocity,
        reference_heading,
        config,
    )
    best_by_cell_heading: dict[
        tuple[float, float, int], tuple[tuple[float, float], _LatticeState]
    ] = {}
    for terminal in terminal_states:
        cell_x = _snap_axis(
            terminal.x, config.field_length_m / 2.0, config.grid_resolution_m
        )
        cell_y = _snap_axis(
            terminal.y, config.field_width_m / 2.0, config.grid_resolution_m
        )
        heading_bin = _terminal_heading_bin(
            terminal.heading_radians,
            reference_heading,
            config.terminal_heading_bins,
        )
        snap_distance = math.hypot(terminal.x - cell_x, terminal.y - cell_y)
        key = (cell_x, cell_y, heading_bin)
        rank = (terminal.effort_m2ps3, snap_distance)
        existing = best_by_cell_heading.get(key)
        if existing is None or rank < existing[0]:
            best_by_cell_heading[key] = (rank, terminal)

    empirical_actions = (
        empirical_action_set.optimization_actions
        if empirical_action_set is not None
        else ()
    )
    actions = []
    for (cell_x, cell_y, heading_bin), (_, terminal) in best_by_cell_heading.items():
        endpoint = (terminal.x, terminal.y)
        normalized_radius = ellipse.normalized_radius(*endpoint)
        motion = SteeringControlMotion(
            terminal_vx_mps=float(terminal.speed_mps * math.cos(terminal.heading_radians)),
            terminal_vy_mps=float(terminal.speed_mps * math.sin(terminal.heading_radians)),
            terminal_speed_mps=float(terminal.speed_mps),
            terminal_heading_bin=heading_bin,
            effort_m2ps3=float(terminal.effort_m2ps3),
            maximum_path_speed_mps=float(terminal.maximum_path_speed_mps),
            maximum_tangential_acceleration_mps2=float(
                terminal.maximum_tangential_acceleration_mps2
            ),
            maximum_tangential_deceleration_mps2=float(
                terminal.maximum_tangential_deceleration_mps2
            ),
            maximum_normal_acceleration_mps2=float(
                terminal.maximum_normal_acceleration_mps2
            ),
            path_xy=terminal.path_xy,
            control_sequence=terminal.control_sequence,
        )
        action = SteeringEndpointAction(
            player_id=player.object_id,
            endpoint_x=float(endpoint[0]),
            endpoint_y=float(endpoint[1]),
            endpoint_cell_x=float(cell_x),
            endpoint_cell_y=float(cell_y),
            grid_snap_distance_m=float(math.hypot(endpoint[0] - cell_x, endpoint[1] - cell_y)),
            motion=motion,
            fernandez_normalized_radius=float(normalized_radius),
            inside_fernandez_contour=normalized_radius <= 1.0 + 1e-9,
        )
        if empirical_actions:
            action = _empirical_annotation(action, empirical_actions, config)
        actions.append(action)
    actions.sort(
        key=lambda action: (
            action.endpoint_cell_x,
            action.endpoint_cell_y,
            action.motion.terminal_heading_bin,
        )
    )
    return SteeringEndpointActionSet(
        player_id=player.object_id,
        start_x=float(player.x),
        start_y=float(player.y),
        initial_vx_mps=float(initial_velocity[0]),
        initial_vy_mps=float(initial_velocity[1]),
        initial_speed_mps=float(initial_speed),
        reference_heading_radians=float(reference_heading),
        influence_ellipse=ellipse,
        proposal_endpoints=proposal,
        actions=tuple(actions),
    )


def _plant_cut_position_at_time(
    start_xy: tuple[float, float],
    initial_heading_radians: float,
    initial_speed_mps: float,
    braking_mps2: float,
    plant_duration_s: float,
    exit_heading_radians: float,
    post_acceleration_mps2: float,
    time_s: float,
) -> tuple[float, float]:
    stop_time = initial_speed_mps / braking_mps2
    braking_time = min(time_s, stop_time)
    braking_distance = (
        initial_speed_mps * braking_time
        - 0.5 * braking_mps2 * braking_time**2
    )
    x = start_xy[0] + braking_distance * math.cos(initial_heading_radians)
    y = start_xy[1] + braking_distance * math.sin(initial_heading_radians)
    post_start_time = stop_time + plant_duration_s
    if time_s <= post_start_time:
        return float(x), float(y)
    post_time = time_s - post_start_time
    post_distance = 0.5 * post_acceleration_mps2 * post_time**2
    return (
        float(x + post_distance * math.cos(exit_heading_radians)),
        float(y + post_distance * math.sin(exit_heading_radians)),
    )


def generate_plant_cut_endpoint_actions(
    action_set: SteeringEndpointActionSet,
    empirical_action_set: EmpiricalEndpointActionSet | None = None,
    config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> tuple[SteeringEndpointAction, ...]:
    """Generate stop/plant/reorient/reaccelerate actions.

    This is a separate maneuver class from continuous Frenet steering.  The
    player brakes to zero along the incoming heading, spends an explicit plant
    interval at the turn point, chooses a new heading, and accelerates again.
    The target-scene future is not used.
    """

    config.validate()
    if action_set.initial_speed_mps < config.minimum_plant_cut_initial_speed_mps:
        return ()
    start_xy = (action_set.start_x, action_set.start_y)
    reference_heading = action_set.reference_heading_radians
    empirical_actions = (
        empirical_action_set.optimization_actions
        if empirical_action_set is not None
        else ()
    )
    best_by_cell_heading: dict[
        tuple[float, float, int],
        tuple[tuple[float, float, float], SteeringEndpointAction],
    ] = {}
    total_steps = int(
        round(config.horizon_seconds / config.integration_step_seconds)
    )
    for braking_fraction in config.plant_cut_braking_fractions:
        braking = config.max_plant_cut_deceleration_mps2 * braking_fraction
        stop_time = action_set.initial_speed_mps / braking
        stop_distance = action_set.initial_speed_mps**2 / (2.0 * braking)
        stop_xy = (
            start_xy[0] + stop_distance * math.cos(reference_heading),
            start_xy[1] + stop_distance * math.sin(reference_heading),
        )
        if not _inside_pitch(*stop_xy, config):
            continue
        for plant_duration in config.plant_cut_durations_s:
            post_duration = config.horizon_seconds - stop_time - plant_duration
            if post_duration <= config.integration_step_seconds - 1e-9:
                continue
            for direction_index in range(config.plant_cut_direction_count):
                relative_angle = (
                    2.0
                    * math.pi
                    * direction_index
                    / config.plant_cut_direction_count
                )
                signed_angle = (
                    (relative_angle + math.pi) % (2.0 * math.pi) - math.pi
                )
                signed_angle_degrees = math.degrees(signed_angle)
                if abs(signed_angle_degrees) + 1e-9 < (
                    config.minimum_plant_cut_angle_degrees
                ):
                    continue
                if abs(signed_angle_degrees + 180.0) < 1e-9:
                    signed_angle_degrees = 180.0
                exit_heading = _wrap_heading(reference_heading + relative_angle)
                heading_bin = _terminal_heading_bin(
                    exit_heading,
                    reference_heading,
                    config.terminal_heading_bins,
                )
                for acceleration_fraction in (
                    config.plant_cut_acceleration_fractions
                ):
                    post_acceleration = (
                        config.max_tangential_acceleration_mps2
                        * acceleration_fraction
                    )
                    terminal_speed = min(
                        config.max_speed_mps,
                        post_acceleration * post_duration,
                    )
                    if terminal_speed >= config.max_speed_mps - 1e-9:
                        # The current 2 s settings do not hit this branch, but
                        # avoid an inconsistent constant-acceleration path if
                        # future parameter changes do.
                        continue
                    path = tuple(
                        _plant_cut_position_at_time(
                            start_xy,
                            reference_heading,
                            action_set.initial_speed_mps,
                            braking,
                            plant_duration,
                            exit_heading,
                            post_acceleration,
                            time_s,
                        )
                        for time_s in config.path_sample_seconds
                    )
                    if not all(_inside_pitch(x, y, config) for x, y in path):
                        continue
                    endpoint = path[-1]
                    cell_x = _snap_axis(
                        endpoint[0],
                        config.field_length_m / 2.0,
                        config.grid_resolution_m,
                    )
                    cell_y = _snap_axis(
                        endpoint[1],
                        config.field_width_m / 2.0,
                        config.grid_resolution_m,
                    )
                    snap_distance = math.hypot(
                        endpoint[0] - cell_x,
                        endpoint[1] - cell_y,
                    )
                    normalized_radius = action_set.influence_ellipse.normalized_radius(
                        *endpoint
                    )
                    control_sequence = []
                    for step_index in range(total_steps):
                        midpoint_time = (
                            step_index + 0.5
                        ) * config.integration_step_seconds
                        if midpoint_time < stop_time:
                            control_sequence.append((-braking, 0.0))
                        elif midpoint_time < stop_time + plant_duration:
                            control_sequence.append((0.0, 0.0))
                        else:
                            control_sequence.append((post_acceleration, 0.0))
                    effort = (
                        braking**2 * stop_time
                        + post_acceleration**2 * post_duration
                    )
                    motion = SteeringControlMotion(
                        terminal_vx_mps=float(
                            terminal_speed * math.cos(exit_heading)
                        ),
                        terminal_vy_mps=float(
                            terminal_speed * math.sin(exit_heading)
                        ),
                        terminal_speed_mps=float(terminal_speed),
                        terminal_heading_bin=heading_bin,
                        effort_m2ps3=float(effort),
                        maximum_path_speed_mps=float(action_set.initial_speed_mps),
                        maximum_tangential_acceleration_mps2=float(
                            post_acceleration
                        ),
                        maximum_tangential_deceleration_mps2=float(braking),
                        maximum_normal_acceleration_mps2=0.0,
                        path_xy=path,
                        control_sequence=tuple(control_sequence),
                        maneuver_type="plant_and_cut",
                        cut_angle_degrees=float(signed_angle_degrees),
                        cut_braking_mps2=float(braking),
                        cut_stop_time_s=float(stop_time),
                        cut_plant_duration_s=float(plant_duration),
                        cut_post_acceleration_mps2=float(post_acceleration),
                    )
                    action = SteeringEndpointAction(
                        player_id=action_set.player_id,
                        endpoint_x=float(endpoint[0]),
                        endpoint_y=float(endpoint[1]),
                        endpoint_cell_x=float(cell_x),
                        endpoint_cell_y=float(cell_y),
                        grid_snap_distance_m=float(snap_distance),
                        motion=motion,
                        fernandez_normalized_radius=float(normalized_radius),
                        inside_fernandez_contour=normalized_radius <= 1.0 + 1e-9,
                        labels=("grid", "plant_cut_feasible"),
                    )
                    if empirical_actions:
                        action = _empirical_annotation(
                            action,
                            empirical_actions,
                            config,
                        )
                    # Preserve exact cut directions inside the coarser eight
                    # terminal-heading bins. Otherwise a 90-degree plant can
                    # be replaced by a lower-effort 67.5/112.5-degree action
                    # landing in the same cell.
                    key = (cell_x, cell_y, direction_index)
                    rank = (effort, snap_distance, abs(signed_angle_degrees))
                    existing = best_by_cell_heading.get(key)
                    if existing is None or rank < existing[0]:
                        best_by_cell_heading[key] = (rank, action)
    actions = [item[1] for item in best_by_cell_heading.values()]
    actions.sort(
        key=lambda action: (
            action.endpoint_cell_x,
            action.endpoint_cell_y,
            action.motion.terminal_heading_bin,
        )
    )
    return tuple(actions)


def generate_hybrid_endpoint_actions(
    player: BundesligaObjectState,
    ball: BundesligaObjectState,
    state: CausalMotionState,
    empirical_action_set: EmpiricalEndpointActionSet | None = None,
    config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> SteeringEndpointActionSet:
    """Return the union of continuous steering and plant-and-cut actions."""

    continuous = generate_steering_endpoint_actions(
        player,
        ball,
        state,
        empirical_action_set,
        config,
    )
    cut_actions = generate_plant_cut_endpoint_actions(
        continuous,
        empirical_action_set,
        config,
    )
    actions = tuple(
        sorted(
            (*continuous.actions, *cut_actions),
            key=lambda action: (
                action.endpoint_cell_x,
                action.endpoint_cell_y,
                action.motion.terminal_heading_bin,
                action.motion.maneuver_type,
            ),
        )
    )
    return replace(continuous, actions=actions)


def observed_steering_support_diagnostics(
    action_set: SteeringEndpointActionSet,
    observed_endpoint_xy: tuple[float, float] | None,
    observed_terminal_velocity_xy: tuple[float, float] | None,
    endpoint_tolerance_m: float = 1.0,
    heading_tolerance_degrees: float = 45.0,
    minimum_heading_speed_mps: float = 0.5,
) -> SteeringSupportDiagnostics:
    if observed_endpoint_xy is None:
        return SteeringSupportDiagnostics(None, None, None, None)
    actions = action_set.optimization_actions
    if not actions:
        return SteeringSupportDiagnostics(False, False, math.inf, math.inf)
    distances = np.asarray(
        [
            math.hypot(
                action.endpoint_x - observed_endpoint_xy[0],
                action.endpoint_y - observed_endpoint_xy[1],
            )
            for action in actions
        ]
    )
    nearest_distance = float(np.min(distances))
    endpoint_supported = nearest_distance <= endpoint_tolerance_m
    if observed_terminal_velocity_xy is None or math.hypot(
        *observed_terminal_velocity_xy
    ) < minimum_heading_speed_mps:
        return SteeringSupportDiagnostics(
            endpoint_supported, endpoint_supported, nearest_distance, None
        )
    nearby = np.flatnonzero(distances <= endpoint_tolerance_m)
    if len(nearby) == 0:
        return SteeringSupportDiagnostics(False, False, nearest_distance, math.inf)
    differences = [
        _heading_difference_degrees(
            (
                actions[int(index)].motion.terminal_vx_mps,
                actions[int(index)].motion.terminal_vy_mps,
            ),
            observed_terminal_velocity_xy,
        )
        for index in nearby
    ]
    nearest_heading = float(min(differences))
    return SteeringSupportDiagnostics(
        endpoint_supported,
        nearest_heading <= heading_tolerance_degrees,
        nearest_distance,
        nearest_heading,
    )
