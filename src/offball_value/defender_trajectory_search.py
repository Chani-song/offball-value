"""Target-agnostic feasible trajectory proposals for one responding defender.

The local-game prototype previously generated a response by assigning the
defender a tactical point (for example, 1.5 m goal-side of the runner) and
steering toward it.  That bakes a football interpretation into the motion
generator.  This module separates the layers:

1. enumerate paths that are physically reachable from the defender's causal
   onset state, without seeing any attacking option;
2. compress the reachable set by endpoint and terminal-velocity diversity;
3. let the downstream continuation-value evaluator decide which path suppresses
   the runner, another option, or the local worst case.

The paths remain generic-player physical proposals, not learned predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Sequence

import numpy as np

from .steering_reachable import (
    SteeringReachabilityConfig,
    enumerate_steering_motions,
    steering_step,
)


TimedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class DefenderTrajectorySearchConfig:
    """Physical and diversity settings for the response proposal library."""

    planning_horizon_seconds: float = 1.8
    integration_step_seconds: float = 0.1
    response_delay_seconds: float = 0.2
    maximum_speed_mps: float = 9.0
    maximum_acceleration_mps2: float = 4.5
    maximum_deceleration_mps2: float = 6.0
    maximum_normal_acceleration_mps2: float = 6.0
    state_position_resolution_m: float = 1.0
    state_speed_resolution_mps: float = 0.75
    state_heading_bins: int = 48
    control_direction_count: int = 12
    endpoint_grid_resolution_m: float = 1.5
    terminal_speed_resolution_mps: float = 1.5
    terminal_heading_bins: int = 12
    # Human review found that keying diversity on the terminal state alone
    # deletes mid-path shape families: a curving man-mark and a straight dash
    # to the same endpoint collapsed into one bucket, and the lower-effort
    # straight path always won.  The mid-path waypoint keeps both alive.
    midpoint_grid_resolution_m: float = 2.0
    maximum_representatives: int = 160
    # Review of the 61841 audit found endpoint-lattice paths generated past
    # the touchline (a physically meaningless proposal); candidates leaving
    # the pitch are discarded.
    pitch_half_length_m: float = 52.5
    pitch_half_width_m: float = 34.0

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name
            not in {
                "response_delay_seconds",
                "state_heading_bins",
                "control_direction_count",
                "terminal_heading_bins",
                "maximum_representatives",
            }
            # midpoint_grid_resolution_m participates in the positivity check
            # via the default branch below.
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("trajectory-search physical settings must be positive")
        if self.response_delay_seconds < 0.0:
            raise ValueError("response delay cannot be negative")
        counts = (
            self.state_heading_bins,
            self.control_direction_count,
            self.terminal_heading_bins,
            self.maximum_representatives,
        )
        if any(int(value) < 1 for value in counts):
            raise ValueError("trajectory-search counts must be positive")
        if self.response_delay_seconds >= self.planning_horizon_seconds:
            raise ValueError("response delay must be shorter than planning horizon")
        for value in (
            self.planning_horizon_seconds,
            self.response_delay_seconds,
        ):
            steps = value / self.integration_step_seconds
            if abs(steps - round(steps)) > 1e-9:
                raise ValueError("planning horizon and delay must align to the step")


def _midpoint_of(
    path_txy: Sequence[TimedPoint],
    config: "DefenderTrajectorySearchConfig",
) -> tuple[float, float]:
    """Waypoint halfway through the steered segment (shape signature)."""

    mid_time = (
        config.response_delay_seconds + config.planning_horizon_seconds
    ) / 2.0
    previous = path_txy[0]
    for row in path_txy:
        if row[0] >= mid_time:
            span = row[0] - previous[0]
            fraction = (mid_time - previous[0]) / span if span > 1e-9 else 0.0
            return (
                previous[1] + fraction * (row[1] - previous[1]),
                previous[2] + fraction * (row[2] - previous[2]),
            )
        previous = row
    return (path_txy[-1][1], path_txy[-1][2])


@dataclass(frozen=True)
class FeasibleDefenderTrajectory:
    """One target-free path and its physical diagnostics."""

    trajectory_id: str
    path_txy: tuple[TimedPoint, ...]
    effort_m2ps3: float
    terminal_speed_mps: float
    terminal_heading_radians: float
    maximum_path_speed_mps: float
    maximum_tangential_acceleration_mps2: float
    maximum_tangential_deceleration_mps2: float
    maximum_normal_acceleration_mps2: float
    endpoint_x: float
    endpoint_y: float
    source: str = "target_agnostic_reachable_lattice"


def _step_config(
    duration: float,
    config: DefenderTrajectorySearchConfig,
) -> SteeringReachabilityConfig:
    return SteeringReachabilityConfig(
        horizon_seconds=float(duration),
        integration_step_seconds=float(duration),
        max_speed_mps=config.maximum_speed_mps,
        max_tangential_acceleration_mps2=config.maximum_acceleration_mps2,
        max_tangential_deceleration_mps2=config.maximum_deceleration_mps2,
        max_normal_acceleration_mps2=config.maximum_normal_acceleration_mps2,
        path_sample_seconds=(float(duration),),
    )


def _delay_state(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    config: DefenderTrajectorySearchConfig,
) -> tuple[tuple[TimedPoint, ...], tuple[float, float, float, float]]:
    """Coast through the perception-response delay without control."""

    dt = config.integration_step_seconds
    x, y = map(float, start_xy)
    speed = min(config.maximum_speed_mps, math.hypot(*initial_velocity_xy))
    heading = (
        math.atan2(initial_velocity_xy[1], initial_velocity_xy[0])
        if math.hypot(*initial_velocity_xy) > 1e-9
        else 0.0
    )
    path: list[TimedPoint] = [(0.0, x, y)]
    delay_steps = int(round(config.response_delay_seconds / dt))
    step_config = _step_config(dt, config)
    for step_index in range(1, delay_steps + 1):
        result = steering_step(x, y, speed, heading, 0.0, 0.0, step_config)
        x, y = result.x, result.y
        speed, heading = result.speed_mps, result.heading_radians
        path.append((round(step_index * dt, 10), x, y))
    return tuple(path), (x, y, speed, heading)


def _extend_with_coast(
    path: Sequence[TimedPoint],
    terminal_speed: float,
    terminal_heading: float,
    full_horizon_seconds: float,
    config: DefenderTrajectorySearchConfig,
) -> tuple[TimedPoint, ...]:
    """Extend a planned response neutrally when the scene horizon is longer."""

    rows = list(path)
    time_s, x, y = rows[-1]
    dt = config.integration_step_seconds
    while time_s < full_horizon_seconds - 1e-9:
        duration = min(dt, full_horizon_seconds - time_s)
        result = steering_step(
            x,
            y,
            terminal_speed,
            terminal_heading,
            0.0,
            0.0,
            _step_config(duration, config),
        )
        time_s = float(min(full_horizon_seconds, time_s + duration))
        x, y = result.x, result.y
        terminal_speed, terminal_heading = (
            result.speed_mps,
            result.heading_radians,
        )
        rows.append((time_s, x, y))
    return tuple(rows)


def _heading_bin(angle: float, count: int) -> int:
    normalized = (float(angle) + 2.0 * math.pi) % (2.0 * math.pi)
    return int(math.floor(normalized / (2.0 * math.pi / count))) % count


def _bucket_candidates(
    candidates: Sequence[FeasibleDefenderTrajectory],
    start_xy: tuple[float, float],
    reference_heading: float,
    config: DefenderTrajectorySearchConfig,
) -> tuple[FeasibleDefenderTrajectory, ...]:
    selected: dict[tuple[int, ...], FeasibleDefenderTrajectory] = {}
    for candidate in candidates:
        mid_x, mid_y = _midpoint_of(candidate.path_txy, config)
        key = (
            int(round((candidate.endpoint_x - start_xy[0]) / config.endpoint_grid_resolution_m)),
            int(round((candidate.endpoint_y - start_xy[1]) / config.endpoint_grid_resolution_m)),
            int(round(candidate.terminal_speed_mps / config.terminal_speed_resolution_mps)),
            _heading_bin(
                candidate.terminal_heading_radians - reference_heading,
                config.terminal_heading_bins,
            ),
            int(round((mid_x - start_xy[0]) / config.midpoint_grid_resolution_m)),
            int(round((mid_y - start_xy[1]) / config.midpoint_grid_resolution_m)),
        )
        incumbent = selected.get(key)
        rank = (candidate.effort_m2ps3, candidate.trajectory_id)
        if incumbent is None or rank < (
            incumbent.effort_m2ps3,
            incumbent.trajectory_id,
        ):
            selected[key] = candidate
    return tuple(selected.values())


def _feature(
    candidate: FeasibleDefenderTrajectory,
    start_xy: tuple[float, float],
    config: DefenderTrajectorySearchConfig,
) -> np.ndarray:
    maximum_displacement = max(
        config.maximum_speed_mps * config.planning_horizon_seconds,
        1e-9,
    )
    mid_x, mid_y = _midpoint_of(candidate.path_txy, config)
    return np.asarray(
        [
            (candidate.endpoint_x - start_xy[0]) / maximum_displacement,
            (candidate.endpoint_y - start_xy[1]) / maximum_displacement,
            candidate.terminal_speed_mps
            * math.cos(candidate.terminal_heading_radians)
            / config.maximum_speed_mps,
            candidate.terminal_speed_mps
            * math.sin(candidate.terminal_heading_radians)
            / config.maximum_speed_mps,
            (mid_x - start_xy[0]) / maximum_displacement,
            (mid_y - start_xy[1]) / maximum_displacement,
        ],
        dtype=float,
    )


def _diverse_subset(
    candidates: Sequence[FeasibleDefenderTrajectory],
    start_xy: tuple[float, float],
    config: DefenderTrajectorySearchConfig,
) -> tuple[FeasibleDefenderTrajectory, ...]:
    """Deterministic farthest-point compression in endpoint/velocity space."""

    if len(candidates) <= config.maximum_representatives:
        return tuple(
            sorted(candidates, key=lambda row: (row.effort_m2ps3, row.trajectory_id))
        )
    rows = list(candidates)
    features = np.stack([_feature(row, start_xy, config) for row in rows])
    seed = min(
        range(len(rows)),
        key=lambda index: (rows[index].effort_m2ps3, rows[index].trajectory_id),
    )
    chosen = [seed]
    available = np.ones(len(rows), dtype=bool)
    available[seed] = False
    minimum_distance = np.sum((features - features[seed]) ** 2, axis=1)
    while len(chosen) < config.maximum_representatives:
        indices = np.flatnonzero(available)
        if not len(indices):
            break
        best = max(
            indices,
            key=lambda index: (
                float(minimum_distance[index]),
                -float(rows[index].effort_m2ps3),
                rows[index].trajectory_id,
            ),
        )
        chosen.append(int(best))
        available[best] = False
        distance = np.sum((features - features[best]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return tuple(rows[index] for index in chosen)


def generate_feasible_defender_trajectories(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    scene_horizon_seconds: float,
    config: DefenderTrajectorySearchConfig = DefenderTrajectorySearchConfig(),
) -> tuple[FeasibleDefenderTrajectory, ...]:
    """Generate target-free, delayed, physically bounded response proposals."""

    config.validate()
    scene_horizon_seconds = float(scene_horizon_seconds)
    if scene_horizon_seconds <= config.response_delay_seconds:
        raise ValueError("scene horizon must exceed response delay")
    planning_horizon = min(config.planning_horizon_seconds, scene_horizon_seconds)
    dt = config.integration_step_seconds
    planning_horizon = math.floor(planning_horizon / dt + 1e-9) * dt
    delayed_path, delayed_state = _delay_state(
        start_xy,
        initial_velocity_xy,
        config,
    )
    delayed_x, delayed_y, delayed_speed, delayed_heading = delayed_state
    remaining_horizon = planning_horizon - config.response_delay_seconds
    sample_times = tuple(
        round(index * dt, 10)
        for index in range(1, int(round(remaining_horizon / dt)) + 1)
    )
    reachability = SteeringReachabilityConfig(
        horizon_seconds=remaining_horizon,
        integration_step_seconds=dt,
        state_position_resolution_m=config.state_position_resolution_m,
        state_speed_resolution_mps=config.state_speed_resolution_mps,
        state_heading_bins=config.state_heading_bins,
        control_direction_count=config.control_direction_count,
        max_speed_mps=config.maximum_speed_mps,
        max_tangential_acceleration_mps2=config.maximum_acceleration_mps2,
        max_tangential_deceleration_mps2=config.maximum_deceleration_mps2,
        max_normal_acceleration_mps2=config.maximum_normal_acceleration_mps2,
        terminal_heading_bins=config.terminal_heading_bins,
        path_sample_seconds=sample_times,
    )
    delayed_velocity = (
        delayed_speed * math.cos(delayed_heading),
        delayed_speed * math.sin(delayed_heading),
    )
    terminal_states = enumerate_steering_motions(
        (delayed_x, delayed_y),
        delayed_velocity,
        delayed_heading,
        reachability,
    )
    candidates = []
    delay = config.response_delay_seconds
    for index, state in enumerate(terminal_states):
        planned = tuple(delayed_path) + tuple(
            (
                float(delay + sample_time),
                float(point[0]),
                float(point[1]),
            )
            for sample_time, point in zip(sample_times, state.path_xy)
        )
        full_path = _extend_with_coast(
            planned,
            state.speed_mps,
            state.heading_radians,
            scene_horizon_seconds,
            config,
        )
        candidates.append(
            FeasibleDefenderTrajectory(
                trajectory_id=f"feasible:{index:05d}",
                path_txy=full_path,
                effort_m2ps3=float(state.effort_m2ps3),
                terminal_speed_mps=float(state.speed_mps),
                terminal_heading_radians=float(state.heading_radians),
                maximum_path_speed_mps=float(state.maximum_path_speed_mps),
                maximum_tangential_acceleration_mps2=float(
                    state.maximum_tangential_acceleration_mps2
                ),
                maximum_tangential_deceleration_mps2=float(
                    state.maximum_tangential_deceleration_mps2
                ),
                maximum_normal_acceleration_mps2=float(
                    state.maximum_normal_acceleration_mps2
                ),
                endpoint_x=float(state.x),
                endpoint_y=float(state.y),
            )
        )
    # Clamp proposals to the pitch: a defender does not keep running past the
    # touchline, and near-line starts must not empty the proposal set (from
    # one metre out at speed, every physically bounded branch overshoots
    # briefly).  Clamping projects the overshoot onto the boundary, which can
    # only reduce implied speeds.
    def _clamped(candidate: FeasibleDefenderTrajectory) -> FeasibleDefenderTrajectory:
        clamped_path = tuple(
            (
                sample[0],
                float(
                    np.clip(
                        sample[1],
                        -config.pitch_half_length_m,
                        config.pitch_half_length_m,
                    )
                ),
                float(
                    np.clip(
                        sample[2],
                        -config.pitch_half_width_m,
                        config.pitch_half_width_m,
                    )
                ),
            )
            for sample in candidate.path_txy
        )
        if clamped_path == candidate.path_txy:
            return candidate
        return FeasibleDefenderTrajectory(
            **{
                **candidate.__dict__,
                "path_txy": clamped_path,
                "endpoint_x": clamped_path[-1][1],
                "endpoint_y": clamped_path[-1][2],
            }
        )

    candidates = [_clamped(candidate) for candidate in candidates]
    bucketed = _bucket_candidates(
        candidates,
        start_xy,
        delayed_heading,
        config,
    )
    selected = _diverse_subset(bucketed, start_xy, config)
    return tuple(
        FeasibleDefenderTrajectory(
            **{
                **row.__dict__,
                "trajectory_id": f"feasible:{index:03d}",
            }
        )
        for index, row in enumerate(selected, 1)
    )
