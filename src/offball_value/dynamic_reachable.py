"""Direction-complete short-horizon endpoint reachability.

v0.4 separates a Fernández 360-degree proposal layer, explicit bounded-control
feasibility, and empirical support.  The target-scene future is never used to
generate optimizer actions.
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
class DynamicReachabilityConfig:
    horizon_seconds: float = 2.0
    grid_resolution_m: float = 1.0
    control_switch_step_seconds: float = 0.1
    phase_one_direction_count: int = 16
    phase_one_acceleration_fractions: tuple[float, ...] = (0.5, 1.0)
    max_speed_mps: float = 9.0
    max_acceleration_mps2: float = 4.5
    max_deceleration_mps2: float = 6.0
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
            "grid_resolution_m": self.grid_resolution_m,
            "control_switch_step_seconds": self.control_switch_step_seconds,
            "phase_one_direction_count": self.phase_one_direction_count,
            "max_speed_mps": self.max_speed_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "terminal_heading_bins": self.terminal_heading_bins,
            "variants_per_endpoint": self.variants_per_endpoint,
            "empirical_endpoint_tolerance_m": self.empirical_endpoint_tolerance_m,
            "empirical_heading_tolerance_degrees": self.empirical_heading_tolerance_degrees,
            "minimum_heading_speed_mps": self.minimum_heading_speed_mps,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Dynamic reachability parameters must be positive: {invalid}")
        if not self.phase_one_acceleration_fractions or any(
            fraction <= 0.0 or fraction > 1.0
            for fraction in self.phase_one_acceleration_fractions
        ):
            raise ValueError("phase-one acceleration fractions must be in (0, 1]")
        if not self.path_sample_seconds or any(
            time_s <= 0.0 or time_s > self.horizon_seconds
            for time_s in self.path_sample_seconds
        ):
            raise ValueError("path samples must lie inside the horizon")
        if abs(self.path_sample_seconds[-1] - self.horizon_seconds) > 1e-9:
            raise ValueError("final path sample must equal the horizon")


@dataclass(frozen=True)
class DynamicControlMotion:
    phase_one_ax_mps2: float
    phase_one_ay_mps2: float
    phase_one_duration_s: float
    phase_two_ax_mps2: float
    phase_two_ay_mps2: float
    phase_two_duration_s: float
    phase_one_control: str
    terminal_vx_mps: float
    terminal_vy_mps: float
    terminal_speed_mps: float
    terminal_heading_bin: int
    effort_m2ps3: float
    maximum_path_speed_mps: float
    path_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DynamicEndpointAction:
    player_id: str
    endpoint_x: float
    endpoint_y: float
    motion: DynamicControlMotion
    fernandez_normalized_radius: float
    inside_fernandez_contour: bool
    empirically_supported: bool = False
    nearest_empirical_endpoint_distance_m: float = math.inf
    nearest_empirical_heading_difference_degrees: float = math.inf
    empirical_primitive_match_id: str | None = None
    empirical_primitive_player_id: str | None = None
    empirical_primitive_frame_id: int | None = None
    labels: tuple[str, ...] = ("grid", "dynamic_feasible")
    optimization_eligible: bool = True

    @property
    def action_id(self) -> str:
        return (
            f"{self.player_id}:{self.endpoint_x:.4f}:{self.endpoint_y:.4f}:"
            f"h{self.motion.terminal_heading_bin}:"
            f"t{self.motion.phase_one_duration_s:.2f}"
        )

    def as_record(self, match_id: str, frame_id: int) -> dict[str, object]:
        return {
            "match_id": match_id,
            "frame_id": int(frame_id),
            "player_id": self.player_id,
            "action_id": f"{match_id}:{frame_id}:{self.action_id}",
            "endpoint_x": self.endpoint_x,
            "endpoint_y": self.endpoint_y,
            "labels": "|".join(self.labels),
            "optimization_eligible": self.optimization_eligible,
            "kinematically_feasible": True,
            "dynamically_feasible": True,
            "empirically_supported": self.empirically_supported,
            "motion_model": "two_phase_bounded_control",
            "terminal_vx_mps": self.motion.terminal_vx_mps,
            "terminal_vy_mps": self.motion.terminal_vy_mps,
            "terminal_speed_mps": self.motion.terminal_speed_mps,
            "terminal_heading_bin": self.motion.terminal_heading_bin,
            "phase_one_ax_mps2": self.motion.phase_one_ax_mps2,
            "phase_one_ay_mps2": self.motion.phase_one_ay_mps2,
            "phase_one_acceleration_mps2": math.hypot(
                self.motion.phase_one_ax_mps2,
                self.motion.phase_one_ay_mps2,
            ),
            "phase_one_duration_s": self.motion.phase_one_duration_s,
            "phase_one_control": self.motion.phase_one_control,
            "phase_two_ax_mps2": self.motion.phase_two_ax_mps2,
            "phase_two_ay_mps2": self.motion.phase_two_ay_mps2,
            "phase_two_acceleration_mps2": math.hypot(
                self.motion.phase_two_ax_mps2,
                self.motion.phase_two_ay_mps2,
            ),
            "phase_two_duration_s": self.motion.phase_two_duration_s,
            "control_effort_m2ps3": self.motion.effort_m2ps3,
            "maximum_path_speed_mps": self.motion.maximum_path_speed_mps,
            "path_xy": json.dumps(self.motion.path_xy, separators=(",", ":")),
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
class DynamicEndpointActionSet:
    player_id: str
    start_x: float
    start_y: float
    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float
    influence_ellipse: FernandezInfluenceEllipse
    proposal_endpoints: tuple[tuple[float, float], ...]
    actions: tuple[DynamicEndpointAction, ...]

    @property
    def optimization_actions(self) -> tuple[DynamicEndpointAction, ...]:
        return tuple(action for action in self.actions if action.optimization_eligible)


@dataclass(frozen=True)
class DynamicSupportDiagnostics:
    endpoint_supported: bool | None
    endpoint_heading_supported: bool | None
    nearest_endpoint_distance_m: float | None
    nearest_heading_difference_degrees: float | None


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


def _phase_position_velocity(
    start_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    acceleration_xy: tuple[float, float],
    duration_s: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    position = tuple(
        start + velocity * duration_s + 0.5 * acceleration * duration_s**2
        for start, velocity, acceleration in zip(
            start_xy, velocity_xy, acceleration_xy
        )
    )
    velocity = tuple(
        value + acceleration * duration_s
        for value, acceleration in zip(velocity_xy, acceleration_xy)
    )
    return (float(position[0]), float(position[1])), (
        float(velocity[0]),
        float(velocity[1]),
    )


def _axis_extrema(
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


def _path_stays_in_pitch(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    first_acceleration_xy: tuple[float, float],
    first_duration_s: float,
    second_acceleration_xy: tuple[float, float],
    second_duration_s: float,
    config: DynamicReachabilityConfig,
) -> bool:
    phase_position, phase_velocity = _phase_position_velocity(
        start_xy,
        initial_velocity_xy,
        first_acceleration_xy,
        first_duration_s,
    )
    bounds = []
    for index in range(2):
        first = _axis_extrema(
            start_xy[index],
            initial_velocity_xy[index],
            first_acceleration_xy[index],
            first_duration_s,
        )
        second = _axis_extrema(
            phase_position[index],
            phase_velocity[index],
            second_acceleration_xy[index],
            second_duration_s,
        )
        bounds.append((min(first[0], second[0]), max(first[1], second[1])))
    return (
        bounds[0][0] >= -config.field_length_m / 2.0 - config.numerical_tolerance
        and bounds[0][1]
        <= config.field_length_m / 2.0 + config.numerical_tolerance
        and bounds[1][0]
        >= -config.field_width_m / 2.0 - config.numerical_tolerance
        and bounds[1][1]
        <= config.field_width_m / 2.0 + config.numerical_tolerance
    )


def _motion_position_at_time(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    first_acceleration_xy: tuple[float, float],
    first_duration_s: float,
    second_acceleration_xy: tuple[float, float],
    time_s: float,
) -> tuple[float, float]:
    if time_s <= first_duration_s:
        position, _ = _phase_position_velocity(
            start_xy,
            initial_velocity_xy,
            first_acceleration_xy,
            time_s,
        )
        return position
    phase_position, phase_velocity = _phase_position_velocity(
        start_xy,
        initial_velocity_xy,
        first_acceleration_xy,
        first_duration_s,
    )
    position, _ = _phase_position_velocity(
        phase_position,
        phase_velocity,
        second_acceleration_xy,
        time_s - first_duration_s,
    )
    return position


def _terminal_heading_bin(
    vx: float,
    vy: float,
    reference_heading: float,
    count: int,
) -> int:
    relative = (math.atan2(vy, vx) - reference_heading + 2.0 * math.pi) % (
        2.0 * math.pi
    )
    return int(math.floor(relative / (2.0 * math.pi / count))) % count


def _heading_difference_degrees(
    first_xy: tuple[float, float],
    second_xy: tuple[float, float],
) -> float:
    first_heading = math.atan2(first_xy[1], first_xy[0])
    second_heading = math.atan2(second_xy[1], second_xy[0])
    difference = abs(
        (first_heading - second_heading + math.pi) % (2.0 * math.pi) - math.pi
    )
    return float(math.degrees(difference))


def _phase_one_controls(
    initial_velocity_xy: tuple[float, float],
    reference_heading: float,
    config: DynamicReachabilityConfig,
) -> tuple[tuple[float, float, str, float | None], ...]:
    controls: list[tuple[float, float, str, float | None]] = [
        (0.0, 0.0, "coast", None)
    ]
    for fraction in config.phase_one_acceleration_fractions:
        magnitude = config.max_acceleration_mps2 * fraction
        for index in range(config.phase_one_direction_count):
            angle = reference_heading + 2.0 * math.pi * index / config.phase_one_direction_count
            controls.append(
                (
                    magnitude * math.cos(angle),
                    magnitude * math.sin(angle),
                    "general_acceleration",
                    None,
                )
            )
    speed = math.hypot(*initial_velocity_xy)
    if speed > config.numerical_tolerance:
        ux = initial_velocity_xy[0] / speed
        uy = initial_velocity_xy[1] / speed
        for magnitude in sorted(
            {
                config.max_acceleration_mps2,
                0.5 * (config.max_acceleration_mps2 + config.max_deceleration_mps2),
                config.max_deceleration_mps2,
            }
        ):
            controls.append(
                (-ux * magnitude, -uy * magnitude, "hard_brake", magnitude)
            )
    return tuple(controls)


def solve_dynamic_endpoint_motions(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    config: DynamicReachabilityConfig = DynamicReachabilityConfig(),
) -> tuple[DynamicControlMotion, ...]:
    """Find diverse exact-grid witnesses with one control switch.

    The switch may occur at any 0.1 s grid point. Phase-one acceleration is
    direction-complete; phase two is solved exactly for the requested endpoint.
    This is a bounded-control approximation, not an assertion that the
    Fernández proposal itself is physically reachable.
    """

    config.validate()
    initial_speed = math.hypot(*initial_velocity_xy)
    if initial_speed > config.max_speed_mps + config.numerical_tolerance:
        return ()
    reference_heading = (
        math.atan2(initial_velocity_xy[1], initial_velocity_xy[0])
        if initial_speed > config.numerical_tolerance
        else 0.0
    )
    controls = _phase_one_controls(
        initial_velocity_xy,
        reference_heading,
        config,
    )
    switch_count = int(
        math.floor(
            (config.horizon_seconds - config.control_switch_step_seconds)
            / config.control_switch_step_seconds
            + 1e-9
        )
    )
    switch_times = [
        index * config.control_switch_step_seconds
        for index in range(switch_count + 1)
    ]
    best_by_bin: dict[int, tuple[tuple[float, float, str], DynamicControlMotion]] = {}
    for first_duration in switch_times:
        second_duration = config.horizon_seconds - first_duration
        for first_ax, first_ay, control_kind, brake_magnitude in controls:
            if first_duration <= config.numerical_tolerance and control_kind != "coast":
                continue
            if control_kind == "hard_brake" and brake_magnitude is not None:
                if first_duration > initial_speed / brake_magnitude + 1e-9:
                    continue
            phase_position, phase_velocity = _phase_position_velocity(
                start_xy,
                initial_velocity_xy,
                (first_ax, first_ay),
                first_duration,
            )
            phase_speed = math.hypot(*phase_velocity)
            if phase_speed > config.max_speed_mps + config.numerical_tolerance:
                continue
            denominator = 0.5 * second_duration**2
            second_ax = (
                endpoint_xy[0]
                - phase_position[0]
                - phase_velocity[0] * second_duration
            ) / denominator
            second_ay = (
                endpoint_xy[1]
                - phase_position[1]
                - phase_velocity[1] * second_duration
            ) / denominator
            second_magnitude = math.hypot(second_ax, second_ay)
            if second_magnitude > config.max_acceleration_mps2 + 1e-7:
                continue
            terminal_vx = phase_velocity[0] + second_ax * second_duration
            terminal_vy = phase_velocity[1] + second_ay * second_duration
            terminal_speed = math.hypot(terminal_vx, terminal_vy)
            if terminal_speed > config.max_speed_mps + config.numerical_tolerance:
                continue
            if not _path_stays_in_pitch(
                start_xy,
                initial_velocity_xy,
                (first_ax, first_ay),
                first_duration,
                (second_ax, second_ay),
                second_duration,
                config,
            ):
                continue
            path = tuple(
                _motion_position_at_time(
                    start_xy,
                    initial_velocity_xy,
                    (first_ax, first_ay),
                    first_duration,
                    (second_ax, second_ay),
                    time_s,
                )
                for time_s in config.path_sample_seconds
            )
            if math.hypot(
                path[-1][0] - endpoint_xy[0],
                path[-1][1] - endpoint_xy[1],
            ) > 1e-6:
                continue
            heading_bin = _terminal_heading_bin(
                terminal_vx,
                terminal_vy,
                reference_heading,
                config.terminal_heading_bins,
            )
            first_magnitude = math.hypot(first_ax, first_ay)
            effort = (
                first_magnitude**2 * first_duration
                + second_magnitude**2 * second_duration
            )
            motion = DynamicControlMotion(
                phase_one_ax_mps2=float(first_ax),
                phase_one_ay_mps2=float(first_ay),
                phase_one_duration_s=float(first_duration),
                phase_two_ax_mps2=float(second_ax),
                phase_two_ay_mps2=float(second_ay),
                phase_two_duration_s=float(second_duration),
                phase_one_control=control_kind,
                terminal_vx_mps=float(terminal_vx),
                terminal_vy_mps=float(terminal_vy),
                terminal_speed_mps=float(terminal_speed),
                terminal_heading_bin=heading_bin,
                effort_m2ps3=float(effort),
                maximum_path_speed_mps=float(
                    max(initial_speed, phase_speed, terminal_speed)
                ),
                path_xy=path,
            )
            rank = (effort, first_duration, control_kind)
            existing = best_by_bin.get(heading_bin)
            if existing is None or rank < existing[0]:
                best_by_bin[heading_bin] = (rank, motion)

    if not best_by_bin:
        return ()
    candidates = [item[1] for item in best_by_bin.values()]
    candidates.sort(key=lambda motion: (motion.effort_m2ps3, motion.terminal_heading_bin))
    selected = [candidates.pop(0)]
    while candidates and len(selected) < config.variants_per_endpoint:
        selected_bins = [motion.terminal_heading_bin for motion in selected]

        def diversity_rank(motion: DynamicControlMotion) -> tuple[int, float, int]:
            circular_distance = min(
                min(
                    abs(motion.terminal_heading_bin - selected_bin),
                    config.terminal_heading_bins
                    - abs(motion.terminal_heading_bin - selected_bin),
                )
                for selected_bin in selected_bins
            )
            return (
                -circular_distance,
                motion.effort_m2ps3,
                motion.terminal_heading_bin,
            )

        candidates.sort(key=diversity_rank)
        selected.append(candidates.pop(0))
    return tuple(selected)


def _empirical_annotation(
    action: DynamicEndpointAction,
    empirical_actions: Iterable[EmpiricalEndpointAction],
    config: DynamicReachabilityConfig,
) -> DynamicEndpointAction:
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
                (
                    action.motion.terminal_vx_mps,
                    action.motion.terminal_vy_mps,
                ),
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


def generate_dynamic_endpoint_actions(
    player: BundesligaObjectState,
    ball: BundesligaObjectState,
    state: CausalMotionState,
    empirical_action_set: EmpiricalEndpointActionSet | None = None,
    config: DynamicReachabilityConfig = DynamicReachabilityConfig(),
) -> DynamicEndpointActionSet:
    config.validate()
    speed = state.speed_mps
    if speed > config.max_speed_mps and speed > 1e-12:
        scale = config.max_speed_mps / speed
        initial_velocity = (state.vx_mps * scale, state.vy_mps * scale)
    else:
        initial_velocity = (state.vx_mps, state.vy_mps)
    initial_speed = math.hypot(*initial_velocity)
    ellipse = fernandez_influence_ellipse(
        (player.x, player.y),
        initial_velocity,
        (ball.x, ball.y),
    )

    # The ellipse is a visual/proposal layer. Dynamic feasibility is searched
    # independently in a conservative maximum-speed disk so the paper's
    # influence contour never becomes a hard reachability boundary.
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
    proposal = []
    actions = []
    cv_endpoint = (
        player.x + initial_velocity[0] * config.horizon_seconds,
        player.y + initial_velocity[1] * config.horizon_seconds,
    )
    maximum_control_displacement = (
        0.5 * config.max_deceleration_mps2 * config.horizon_seconds**2
    )
    empirical_actions = (
        empirical_action_set.optimization_actions
        if empirical_action_set is not None
        else ()
    )
    for x in xs:
        for y in ys:
            distance_from_start = math.hypot(x - player.x, y - player.y)
            normalized_radius = ellipse.normalized_radius(x, y)
            if normalized_radius <= 1.0 + 1e-9:
                proposal.append((x, y))
            if distance_from_start > radius + 1e-9:
                continue
            if (
                math.hypot(x - cv_endpoint[0], y - cv_endpoint[1])
                > maximum_control_displacement + config.grid_resolution_m
            ):
                continue
            motions = solve_dynamic_endpoint_motions(
                (player.x, player.y),
                initial_velocity,
                (x, y),
                config,
            )
            for motion in motions:
                action = DynamicEndpointAction(
                    player_id=player.object_id,
                    endpoint_x=x,
                    endpoint_y=y,
                    motion=motion,
                    fernandez_normalized_radius=normalized_radius,
                    inside_fernandez_contour=normalized_radius <= 1.0 + 1e-9,
                )
                if empirical_actions:
                    action = _empirical_annotation(
                        action,
                        empirical_actions,
                        config,
                    )
                actions.append(action)
    actions.sort(
        key=lambda action: (
            action.endpoint_x,
            action.endpoint_y,
            action.motion.terminal_heading_bin,
        )
    )
    return DynamicEndpointActionSet(
        player_id=player.object_id,
        start_x=float(player.x),
        start_y=float(player.y),
        initial_vx_mps=float(initial_velocity[0]),
        initial_vy_mps=float(initial_velocity[1]),
        initial_speed_mps=float(initial_speed),
        influence_ellipse=ellipse,
        proposal_endpoints=tuple(proposal),
        actions=tuple(actions),
    )


def observed_dynamic_support_diagnostics(
    action_set: DynamicEndpointActionSet,
    observed_endpoint_xy: tuple[float, float] | None,
    observed_terminal_velocity_xy: tuple[float, float] | None,
    endpoint_tolerance_m: float = 1.0,
    heading_tolerance_degrees: float = 45.0,
    minimum_heading_speed_mps: float = 0.5,
) -> DynamicSupportDiagnostics:
    if observed_endpoint_xy is None:
        return DynamicSupportDiagnostics(None, None, None, None)
    actions = action_set.optimization_actions
    if not actions:
        return DynamicSupportDiagnostics(False, False, math.inf, math.inf)
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
        return DynamicSupportDiagnostics(
            endpoint_supported, endpoint_supported, nearest_distance, None
        )
    nearby = np.flatnonzero(distances <= endpoint_tolerance_m)
    if len(nearby) == 0:
        return DynamicSupportDiagnostics(False, False, nearest_distance, math.inf)
    differences = [
        _heading_difference_degrees(
            (
                actions[int(index)].motion.terminal_vx_mps,
                actions[int(index)].motion.terminal_vy_mps,
            ),
            observed_terminal_velocity_xy,
        )
        for index in nearby
        if actions[int(index)].motion.terminal_speed_mps
        >= minimum_heading_speed_mps
    ]
    nearest_heading = min(differences) if differences else math.inf
    return DynamicSupportDiagnostics(
        endpoint_supported,
        endpoint_supported and nearest_heading <= heading_tolerance_degrees,
        nearest_distance,
        float(nearest_heading),
    )
