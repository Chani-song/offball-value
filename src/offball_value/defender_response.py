"""Feasible, action-aware defender responses for local counterfactuals.

The attacker commits to a candidate two-second path at the decision time.  A
defender first continues their pre-decision motion during a fixed response
latency, then uses the same steering and plant-and-cut limits as the attacker
for the remaining horizon.  This module only constructs feasible response
sets and relevant-defender shortlists; threat minimization is deliberately a
separate optimization layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaObjectState,
)
from .empirical_action_space import (
    CausalMotionState,
    EmpiricalPrimitiveConfig,
    causal_motion_state_from_frames,
)
from .steering_reachable import (
    SteeringEndpointAction,
    SteeringReachabilityConfig,
    generate_hybrid_endpoint_actions,
)


@dataclass(frozen=True)
class DefenderResponseConfig:
    horizon_seconds: float = 2.0
    response_delay_seconds: float = 0.2
    relevant_defender_count: int = 3
    include_goalkeeper: bool = False

    def validate(self) -> None:
        if self.horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive")
        if not 0.0 <= self.response_delay_seconds < self.horizon_seconds:
            raise ValueError("response delay must lie in [0, horizon)")
        if self.relevant_defender_count <= 0:
            raise ValueError("relevant_defender_count must be positive")


@dataclass(frozen=True)
class RelevantDefender:
    defender_id: str
    current_distance_m: float
    feasible_endpoint_distance_m: float
    minimum_feasible_path_distance_m: float
    minimum_feasible_path_time_s: float
    minimum_feasible_path_action_id: str
    current_rank: int
    endpoint_rank: int
    path_rank: int
    combined_rank_score: float
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DefenderResponseAction:
    defender_id: str
    endpoint_x: float
    endpoint_y: float
    full_path_xy: tuple[tuple[float, float], ...]
    response_path_times_s: tuple[float, ...]
    base_action: SteeringEndpointAction

    @property
    def action_id(self) -> str:
        return f"delayed:{self.defender_id}:{self.base_action.action_id}"


@dataclass(frozen=True)
class DefenderResponseActionSet:
    defender_id: str
    decision_x: float
    decision_y: float
    response_start_x: float
    response_start_y: float
    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float
    response_delay_seconds: float
    response_horizon_seconds: float
    actions: tuple[DefenderResponseAction, ...]

    @property
    def unique_endpoint_count(self) -> int:
        return len(
            {
                (
                    action.base_action.endpoint_cell_x,
                    action.base_action.endpoint_cell_y,
                )
                for action in self.actions
            }
        )


def _clip_position(x: float, y: float) -> tuple[float, float]:
    return (
        float(min(FIELD_LENGTH / 2.0, max(-FIELD_LENGTH / 2.0, x))),
        float(min(FIELD_WIDTH / 2.0, max(-FIELD_WIDTH / 2.0, y))),
    )


def _capped_velocity(
    state: CausalMotionState,
    maximum_speed_mps: float,
) -> tuple[float, float]:
    speed = math.hypot(state.vx_mps, state.vy_mps)
    if speed <= maximum_speed_mps or speed <= 1e-12:
        return float(state.vx_mps), float(state.vy_mps)
    scale = maximum_speed_mps / speed
    return float(state.vx_mps * scale), float(state.vy_mps * scale)


def _response_path_sample_times(
    response_horizon_seconds: float,
    integration_step_seconds: float,
) -> tuple[float, ...]:
    """Use 0.4 s samples plus an exact final sample for a shortened horizon."""

    values: list[float] = []
    value = 0.4
    while value < response_horizon_seconds - 1e-9:
        steps = round(value / integration_step_seconds)
        values.append(float(steps * integration_step_seconds))
        value += 0.4
    values.append(float(response_horizon_seconds))
    return tuple(dict.fromkeys(values))


def generate_defender_response_actions(
    frame: BundesligaFrame,
    history_frames: Iterable[BundesligaFrame],
    defender_id: str,
    config: DefenderResponseConfig = DefenderResponseConfig(),
    reachability_config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
) -> DefenderResponseActionSet:
    """Generate delayed feasible responses without using target-scene future."""

    config.validate()
    reachability_config.validate()
    if defender_id not in frame.players:
        raise KeyError(f"Defender {defender_id} is missing from the decision frame")
    if frame.ball is None:
        raise ValueError("A ball state is required for the influence reference")

    history = tuple(sorted(history_frames, key=lambda item: item.frame_id))
    motion_state = causal_motion_state_from_frames(
        history,
        defender_id,
        frame.frame_id,
        EmpiricalPrimitiveConfig(history_seconds=0.4),
    )
    vx, vy = _capped_velocity(motion_state, reachability_config.max_speed_mps)
    defender = frame.players[defender_id]
    response_start = _clip_position(
        defender.x + vx * config.response_delay_seconds,
        defender.y + vy * config.response_delay_seconds,
    )
    delayed_player = replace(
        defender,
        x=response_start[0],
        y=response_start[1],
        speed=math.hypot(vx, vy) * 3.6,
    )
    delayed_state = CausalMotionState(
        vx_mps=vx,
        vy_mps=vy,
        speed_mps=math.hypot(vx, vy),
        ax_mps2=0.0,
        ay_mps2=0.0,
        longitudinal_acceleration_mps2=0.0,
        heading_radians=(
            math.atan2(vy, vx)
            if math.hypot(vx, vy) >= reachability_config.stationary_speed_threshold_mps
            else motion_state.heading_radians
        ),
        sample_count=motion_state.sample_count,
        observed_window_seconds=motion_state.observed_window_seconds,
    )
    response_horizon = config.horizon_seconds - config.response_delay_seconds
    response_samples = _response_path_sample_times(
        response_horizon,
        reachability_config.integration_step_seconds,
    )
    delayed_reachability = replace(
        reachability_config,
        horizon_seconds=response_horizon,
        path_sample_seconds=response_samples,
    )
    raw = generate_hybrid_endpoint_actions(
        delayed_player,
        frame.ball,
        delayed_state,
        empirical_action_set=None,
        config=delayed_reachability,
    )
    action_times = (
        0.0,
        config.response_delay_seconds,
        *(
            config.response_delay_seconds + sample_time
            for sample_time in response_samples
        ),
    )
    actions = tuple(
        DefenderResponseAction(
            defender_id=defender_id,
            endpoint_x=action.endpoint_x,
            endpoint_y=action.endpoint_y,
            full_path_xy=(
                (float(defender.x), float(defender.y)),
                response_start,
                *action.motion.path_xy,
            ),
            response_path_times_s=tuple(float(value) for value in action_times),
            base_action=action,
        )
        for action in raw.optimization_actions
    )
    return DefenderResponseActionSet(
        defender_id=defender_id,
        decision_x=float(defender.x),
        decision_y=float(defender.y),
        response_start_x=response_start[0],
        response_start_y=response_start[1],
        initial_vx_mps=vx,
        initial_vy_mps=vy,
        initial_speed_mps=math.hypot(vx, vy),
        response_delay_seconds=config.response_delay_seconds,
        response_horizon_seconds=response_horizon,
        actions=actions,
    )


def _rank_by_value(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda player_id: (values[player_id], player_id))
    return {player_id: index + 1 for index, player_id in enumerate(ordered)}


def _interpolate_path(
    path: Sequence[tuple[float, float]],
    source_times_s: Sequence[float],
    target_times_s: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Linearly align one sampled path with another path's timestamps."""

    import numpy as np

    xs = np.asarray([point[0] for point in path], dtype=float)
    ys = np.asarray([point[1] for point in path], dtype=float)
    return tuple(
        (
            float(np.interp(time_s, source_times_s, xs)),
            float(np.interp(time_s, source_times_s, ys)),
        )
        for time_s in target_times_s
    )


def _feasible_response_distances(
    action_set: DefenderResponseActionSet,
    attacker_path_xy: Sequence[tuple[float, float]],
    attacker_path_times_s: Sequence[float],
) -> tuple[float, float, float, str]:
    """Return endpoint and synchronized path distance over feasible responses."""

    attacker_endpoint = attacker_path_xy[-1]
    endpoint_distance = min(
        math.hypot(
            action.endpoint_x - attacker_endpoint[0],
            action.endpoint_y - attacker_endpoint[1],
        )
        for action in action_set.actions
    )
    best_rank: tuple[float, float, str, float] | None = None
    for action in action_set.actions:
        aligned_attacker = _interpolate_path(
            attacker_path_xy,
            attacker_path_times_s,
            action.response_path_times_s,
        )
        distances = tuple(
            math.hypot(
                defender_point[0] - attacker_point[0],
                defender_point[1] - attacker_point[1],
            )
            for defender_point, attacker_point in zip(
                action.full_path_xy, aligned_attacker
            )
        )
        index = min(range(len(distances)), key=distances.__getitem__)
        rank = (
            float(distances[index]),
            float(action.base_action.motion.effort_m2ps3),
            action.action_id,
            float(action.response_path_times_s[index]),
        )
        if best_rank is None or rank[:3] < best_rank[:3]:
            best_rank = rank
    if best_rank is None:
        raise ValueError(f"Defender {action_set.defender_id} has no feasible responses")
    return endpoint_distance, best_rank[0], best_rank[3], best_rank[2]


def shortlist_relevant_defenders(
    frame: BundesligaFrame,
    history_frames: Iterable[BundesligaFrame],
    attacking_team_id: str,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    config: DefenderResponseConfig = DefenderResponseConfig(),
    reachability_config: SteeringReachabilityConfig = SteeringReachabilityConfig(),
    response_action_sets: Mapping[str, DefenderResponseActionSet] | None = None,
) -> tuple[RelevantDefender, ...]:
    """Shortlist defenders using feasible current, endpoint, and path roles.

    Every outfield defender receives a delayed bounded-control response set
    before screening.  The three anchors are the nearest current marker, the
    defender able to finish closest to the attacker endpoint, and the defender
    able to approach the attacker path most closely at the same time.  Duplicate
    anchors are filled by feasible path rank.  This prevents a defender who can
    turn into the running lane from being discarded by a constant-velocity
    prefilter.
    """

    config.validate()
    if attacker_id not in frame.players:
        raise KeyError(f"Attacker {attacker_id} is missing")
    path = tuple((float(point[0]), float(point[1])) for point in attacker_path_xy)
    if not path:
        raise ValueError("attacker_path_xy cannot be empty")
    if attacker_path_times_s is None:
        if len(path) == 1:
            times = (config.horizon_seconds,)
        else:
            step = config.horizon_seconds / (len(path) - 1)
            times = tuple(index * step for index in range(len(path)))
    else:
        times = tuple(float(value) for value in attacker_path_times_s)
        if len(times) != len(path):
            raise ValueError("attacker path points and times must align")

    excluded_goalkeepers = set(goalkeeper_ids) if not config.include_goalkeeper else set()
    defender_ids = tuple(
        player_id
        for player_id, player in frame.players.items()
        if player.team_id != attacking_team_id
        and player_id not in excluded_goalkeepers
    )
    if not defender_ids:
        return ()
    history = tuple(history_frames)
    attacker = frame.players[attacker_id]
    current_distances: dict[str, float] = {}
    endpoint_distances: dict[str, float] = {}
    path_distances: dict[str, float] = {}
    path_times: dict[str, float] = {}
    path_action_ids: dict[str, str] = {}
    supplied_sets = response_action_sets or {}
    for defender_id in defender_ids:
        defender = frame.players[defender_id]
        current_distances[defender_id] = math.hypot(
            defender.x - attacker.x,
            defender.y - attacker.y,
        )
        action_set = supplied_sets.get(defender_id)
        if action_set is None:
            action_set = generate_defender_response_actions(
                frame,
                history,
                defender_id,
                config,
                reachability_config,
            )
        endpoint, path_distance, path_time, action_id = _feasible_response_distances(
            action_set,
            path,
            times,
        )
        endpoint_distances[defender_id] = endpoint
        path_distances[defender_id] = path_distance
        path_times[defender_id] = path_time
        path_action_ids[defender_id] = action_id

    current_ranks = _rank_by_value(current_distances)
    endpoint_ranks = _rank_by_value(endpoint_distances)
    path_ranks = _rank_by_value(path_distances)
    reasons: dict[str, set[str]] = {defender_id: set() for defender_id in defender_ids}
    anchors = (
        (min(current_distances, key=current_distances.get), "nearest_now"),
        (min(endpoint_distances, key=endpoint_distances.get), "best_endpoint_cover"),
        (min(path_distances, key=path_distances.get), "best_path_interceptor"),
    )
    selected: list[str] = []
    for defender_id, reason in anchors:
        reasons[defender_id].add(reason)
        if defender_id not in selected and len(selected) < config.relevant_defender_count:
            selected.append(defender_id)
    combined = {
        defender_id: (
            current_ranks[defender_id]
            + endpoint_ranks[defender_id]
            + 2.0 * path_ranks[defender_id]
        )
        for defender_id in defender_ids
    }
    for defender_id in sorted(defender_ids, key=lambda item: (combined[item], item)):
        if len(selected) >= min(config.relevant_defender_count, len(defender_ids)):
            break
        if defender_id not in selected:
            reasons[defender_id].add("next_feasible_interceptor")
            selected.append(defender_id)
    result = [
        RelevantDefender(
            defender_id=defender_id,
            current_distance_m=float(current_distances[defender_id]),
            feasible_endpoint_distance_m=float(endpoint_distances[defender_id]),
            minimum_feasible_path_distance_m=float(path_distances[defender_id]),
            minimum_feasible_path_time_s=float(path_times[defender_id]),
            minimum_feasible_path_action_id=path_action_ids[defender_id],
            current_rank=current_ranks[defender_id],
            endpoint_rank=endpoint_ranks[defender_id],
            path_rank=path_ranks[defender_id],
            combined_rank_score=float(combined[defender_id]),
            selection_reasons=tuple(sorted(reasons[defender_id])),
        )
        for defender_id in selected
    ]
    return tuple(result)
