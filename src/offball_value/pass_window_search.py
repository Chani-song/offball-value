"""Defender response search over controlled pass-release instants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .action_value import nearest_attacker_region_partition
from .defender_best_response import (
    ObservedBackgroundSequence,
    build_local_counterfactual_state,
)
from .defender_response import DefenderResponseAction
from .pass_window_value import controlled_ball_owner
from .pass_window_value import (
    PassWindowTrace,
    evaluate_pass_release_snapshot,
    summarize_pass_window,
)
from .reference_obso import (
    ReferenceOBSOConfig,
    bounded_maximum_reference_obso,
    bounded_maximum_reference_obso_in_region,
    pitch_grid,
)


@dataclass(frozen=True)
class PassWindowResponseMinimum:
    objective: str
    value: float
    peak_time_s: float
    maximum_x: float
    maximum_y: float
    defender_id: str
    defender_action_id: str
    defender_effort_m2ps3: float

    @property
    def rank_key(self) -> tuple[float, float, str]:
        return self.value, self.defender_effort_m2ps3, self.defender_action_id


@dataclass(frozen=True)
class PassWindowResponseIncumbents:
    direct_cover: PassWindowResponseMinimum
    other_cover: PassWindowResponseMinimum
    global_best: PassWindowResponseMinimum
    evaluated_response_count: int = 0


@dataclass(frozen=True)
class TargetedPassWindowResponseIncumbents:
    """Minima for focal, named-beneficiary, and their defensive dilemma."""

    direct_cover: PassWindowResponseMinimum
    beneficiary_cover: PassWindowResponseMinimum
    dilemma_best: PassWindowResponseMinimum
    beneficiary_id: str
    evaluated_response_count: int = 0


def evaluate_pass_window_response(
    background: ObservedBackgroundSequence,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    required_ball_owner_id: str,
    action: DefenderResponseAction,
    control_distance_m: float = 1.5,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> PassWindowTrace:
    """Fully evaluate one feasible response across the release window."""

    points = []
    for state in background.states:
        frame, velocities = build_local_counterfactual_state(
            state,
            attacker_id,
            attacker_path_xy,
            attacker_path_times_s,
            defender_id=action.defender_id,
            defender_path_xy=action.full_path_xy,
            defender_path_times_s=action.response_path_times_s,
        )
        # t=0 is a shared observed baseline.  A future action must not leak
        # into it through the forward slope used to interpolate a path.
        if state.time_s == 0.0:
            velocities[attacker_id] = state.velocities[attacker_id]
            velocities[action.defender_id] = state.velocities[action.defender_id]
        points.append(
            evaluate_pass_release_snapshot(
                state.time_s,
                frame,
                velocities,
                attacking_team_id,
                attacker_id,
                attacking_direction,
                goalkeeper_ids,
                control_distance_m=control_distance_m,
                obso_config=obso_config,
            )
        )
    return summarize_pass_window(points, attacker_id, required_ball_owner_id)


def incumbents_from_response_traces(
    traces: Sequence[tuple[DefenderResponseAction, PassWindowTrace]],
) -> PassWindowResponseIncumbents:
    """Choose initial exact R, O, and team minima from feasible traces."""

    if not traces:
        raise ValueError("at least one seed response trace is required")

    def minimum(objective: str) -> PassWindowResponseMinimum:
        candidates = []
        for action, trace in traces:
            if objective == "R":
                point = max(
                    (item for item in trace.points if item.release_time_s > 0.0),
                    key=lambda item: (item.focal.value, -item.release_time_s),
                )
                option = point.focal
                value = trace.focal_peak
            elif objective == "O":
                point = max(
                    (
                        item
                        for item in trace.points
                        if item.release_time_s > 0.0
                        and item.best_other is not None
                    ),
                    key=lambda item: (item.best_other.value, -item.release_time_s),
                )
                option = point.best_other
                value = trace.other_peak
            else:
                point = max(
                    (item for item in trace.points if item.release_time_s > 0.0),
                    key=lambda item: (item.team_maximum, -item.release_time_s),
                )
                option = None
                value = trace.team_peak
            maximum_x = (
                option.event_x if option is not None else point.surface.maximum_position[0]
            )
            maximum_y = (
                option.event_y if option is not None else point.surface.maximum_position[1]
            )
            candidates.append(
                PassWindowResponseMinimum(
                    objective=objective,
                    value=value,
                    peak_time_s=point.release_time_s,
                    maximum_x=float(maximum_x),
                    maximum_y=float(maximum_y),
                    defender_id=action.defender_id,
                    defender_action_id=action.action_id,
                    defender_effort_m2ps3=action.base_action.motion.effort_m2ps3,
                )
            )
        return min(candidates, key=lambda item: item.rank_key)

    return PassWindowResponseIncumbents(
        direct_cover=minimum("R"),
        other_cover=minimum("O"),
        global_best=minimum("max(R,O)"),
    )


def _player_peak(
    trace: PassWindowTrace,
    player_id: str,
) -> tuple[float, float, float, float]:
    candidates = []
    for point in trace.points:
        if point.release_time_s <= 0.0:
            continue
        option = next(
            item for item in point.player_options if item.player_id == player_id
        )
        candidates.append((option.value, point.release_time_s, option.event_x, option.event_y))
    if not candidates:
        raise ValueError("trace contains no post-onset player option")
    value, time_s, x, y = max(candidates, key=lambda item: (item[0], -item[1]))
    return (
        float(value),
        float(time_s),
        float(x) if x is not None else float("nan"),
        float(y) if y is not None else float("nan"),
    )


def targeted_incumbents_from_response_traces(
    traces: Sequence[tuple[DefenderResponseAction, PassWindowTrace]],
    beneficiary_id: str,
) -> TargetedPassWindowResponseIncumbents:
    """Build exact seeds for a focal-versus-named-beneficiary dilemma."""

    if not traces:
        raise ValueError("at least one seed response trace is required")
    direct_candidates = []
    beneficiary_candidates = []
    dilemma_candidates = []
    for action, trace in traces:
        direct_peak = _player_peak(trace, trace.points[0].focal.player_id)
        beneficiary_peak = _player_peak(trace, beneficiary_id)
        direct = _candidate("R_focal", *direct_peak, action)
        beneficiary = _candidate("O_beneficiary", *beneficiary_peak, action)
        dilemma = _candidate(
            "max(R_focal,O_beneficiary)",
            max(direct.value, beneficiary.value),
            (
                direct.peak_time_s
                if direct.value >= beneficiary.value
                else beneficiary.peak_time_s
            ),
            (
                direct.maximum_x
                if direct.value >= beneficiary.value
                else beneficiary.maximum_x
            ),
            (
                direct.maximum_y
                if direct.value >= beneficiary.value
                else beneficiary.maximum_y
            ),
            action,
        )
        direct_candidates.append(direct)
        beneficiary_candidates.append(beneficiary)
        dilemma_candidates.append(dilemma)
    return TargetedPassWindowResponseIncumbents(
        direct_cover=min(direct_candidates, key=lambda item: item.rank_key),
        beneficiary_cover=min(
            beneficiary_candidates, key=lambda item: item.rank_key
        ),
        dilemma_best=min(dilemma_candidates, key=lambda item: item.rank_key),
        beneficiary_id=beneficiary_id,
    )


def _player_region_mask(partition, player_id: str):
    import numpy as np

    if player_id not in partition.eligible_player_ids:
        return np.zeros(partition.owners.shape, dtype=bool)
    return partition.owners == partition.eligible_player_ids.index(player_id)


def search_targeted_pass_window_response_actions(
    background: ObservedBackgroundSequence,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    required_ball_owner_id: str,
    actions: Sequence[DefenderResponseAction],
    incumbents: TargetedPassWindowResponseIncumbents,
    control_distance_m: float = 1.5,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    obso_batch_size: int = 8,
    progress_every: int = 0,
) -> TargetedPassWindowResponseIncumbents:
    """Search R, a named teammate option, and ``max(R, O_named)`` exactly."""

    if not actions:
        return incumbents
    xgrid, ygrid, _ = pitch_grid(obso_config)
    direct = incumbents.direct_cover
    beneficiary = incumbents.beneficiary_cover
    dilemma = incumbents.dilemma_best
    evaluated = incumbents.evaluated_response_count

    def regional(frame, velocities, mask, cutoff):
        return bounded_maximum_reference_obso_in_region(
            frame,
            attacking_team_id,
            attacking_direction,
            mask,
            rejection_cutoff=cutoff,
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            apply_offside=True,
            config=obso_config,
            batch_size=obso_batch_size,
        )

    for action_index, action in enumerate(actions, start=1):
        direct_peak = (0.0, 0.0, float("nan"), float("nan"))
        beneficiary_peak = (0.0, 0.0, float("nan"), float("nan"))
        dilemma_direct_peak = (0.0, 0.0, float("nan"), float("nan"))
        dilemma_beneficiary_peak = (0.0, 0.0, float("nan"), float("nan"))
        direct_alive = beneficiary_alive = dilemma_alive = True
        valid_count = 0
        for state in background.states:
            if state.time_s <= 0.0:
                continue
            owner_id, _, controlled = controlled_ball_owner(
                state.frame, attacking_team_id, control_distance_m
            )
            if not controlled or owner_id != required_ball_owner_id:
                continue
            valid_count += 1
            frame, velocities = build_local_counterfactual_state(
                state,
                attacker_id,
                attacker_path_xy,
                attacker_path_times_s,
                defender_id=action.defender_id,
                defender_path_xy=action.full_path_xy,
                defender_path_times_s=action.response_path_times_s,
            )
            partition = nearest_attacker_region_partition(
                frame,
                attacking_team_id,
                attacker_id,
                attacking_direction,
                xgrid,
                ygrid,
            )
            focal_mask = partition.focal_mask
            beneficiary_mask = _player_region_mask(
                partition, incumbents.beneficiary_id
            )

            focal_result = None
            if direct_alive:
                focal_result = regional(frame, velocities, focal_mask, direct.value)
                if not focal_result.exact:
                    direct_alive = False
                elif focal_result.maximum.value > direct_peak[0]:
                    m = focal_result.maximum
                    direct_peak = (m.value, state.time_s, m.x, m.y)
            beneficiary_result = None
            if beneficiary_alive:
                beneficiary_result = regional(
                    frame, velocities, beneficiary_mask, beneficiary.value
                )
                if not beneficiary_result.exact:
                    beneficiary_alive = False
                elif beneficiary_result.maximum.value > beneficiary_peak[0]:
                    m = beneficiary_result.maximum
                    beneficiary_peak = (m.value, state.time_s, m.x, m.y)

            if dilemma_alive:
                focal_dilemma = (
                    focal_result
                    if focal_result is not None and focal_result.exact
                    else regional(frame, velocities, focal_mask, dilemma.value)
                )
                if not focal_dilemma.exact:
                    dilemma_alive = False
                else:
                    m = focal_dilemma.maximum
                    if m.value > dilemma_direct_peak[0]:
                        dilemma_direct_peak = (m.value, state.time_s, m.x, m.y)
                if dilemma_alive:
                    beneficiary_dilemma = (
                        beneficiary_result
                        if beneficiary_result is not None and beneficiary_result.exact
                        else regional(
                            frame, velocities, beneficiary_mask, dilemma.value
                        )
                    )
                    if not beneficiary_dilemma.exact:
                        dilemma_alive = False
                    else:
                        m = beneficiary_dilemma.maximum
                        if m.value > dilemma_beneficiary_peak[0]:
                            dilemma_beneficiary_peak = (
                                m.value,
                                state.time_s,
                                m.x,
                                m.y,
                            )
            if not direct_alive and not beneficiary_alive and not dilemma_alive:
                break
        if valid_count == 0:
            raise ValueError("pass window contains no post-onset controlled release")
        if direct_alive:
            candidate = _candidate("R_focal", *direct_peak, action)
            if candidate.rank_key < direct.rank_key:
                direct = candidate
        if beneficiary_alive:
            candidate = _candidate("O_beneficiary", *beneficiary_peak, action)
            if candidate.rank_key < beneficiary.rank_key:
                beneficiary = candidate
        if dilemma_alive:
            peak = max(
                (dilemma_direct_peak, dilemma_beneficiary_peak),
                key=lambda item: item[0],
            )
            candidate = _candidate(
                "max(R_focal,O_beneficiary)", *peak, action
            )
            if candidate.rank_key < dilemma.rank_key:
                dilemma = candidate
        evaluated += 1
        if progress_every and action_index % progress_every == 0:
            print(
                f"targeted responses {action_index}/{len(actions)} · "
                f"R={direct.value:.6f} O_named={beneficiary.value:.6f} "
                f"dilemma={dilemma.value:.6f}",
                flush=True,
            )
    return TargetedPassWindowResponseIncumbents(
        direct_cover=direct,
        beneficiary_cover=beneficiary,
        dilemma_best=dilemma,
        beneficiary_id=incumbents.beneficiary_id,
        evaluated_response_count=evaluated,
    )


def _candidate(
    objective: str,
    value: float,
    time_s: float,
    x: float,
    y: float,
    action: DefenderResponseAction,
) -> PassWindowResponseMinimum:
    return PassWindowResponseMinimum(
        objective=objective,
        value=value,
        peak_time_s=time_s,
        maximum_x=x,
        maximum_y=y,
        defender_id=action.defender_id,
        defender_action_id=action.action_id,
        defender_effort_m2ps3=action.base_action.motion.effort_m2ps3,
    )


def search_pass_window_response_actions(
    background: ObservedBackgroundSequence,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    required_ball_owner_id: str,
    actions: Sequence[DefenderResponseAction],
    incumbents: PassWindowResponseIncumbents,
    control_distance_m: float = 1.5,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    obso_batch_size: int = 8,
    progress_every: int = 0,
) -> PassWindowResponseIncumbents:
    """Update exact R, O, and max(R,O) minima for one action collection.

    Each objective uses safe upper-bound rejection against its current
    incumbent. A response that is not rejected receives an exact maximum over
    every controlled release state, so pruning changes runtime only.
    """

    if not actions:
        return incumbents
    xgrid, ygrid, _ = pitch_grid(obso_config)
    direct = incumbents.direct_cover
    other = incumbents.other_cover
    global_best = incumbents.global_best
    evaluated = incumbents.evaluated_response_count
    for action_index, action in enumerate(actions, start=1):
        direct_peak: tuple[float, float, float, float] = (0.0, 0.0, float("nan"), float("nan"))
        other_peak: tuple[float, float, float, float] = (0.0, 0.0, float("nan"), float("nan"))
        global_peak: tuple[float, float, float, float] = (0.0, 0.0, float("nan"), float("nan"))
        direct_alive = True
        other_alive = True
        global_alive = True
        valid_count = 0
        for state in background.states:
            if state.time_s <= 0.0:
                continue
            owner_id, owner_distance, controlled = controlled_ball_owner(
                state.frame, attacking_team_id, control_distance_m
            )
            if not controlled or owner_id != required_ball_owner_id:
                continue
            valid_count += 1
            frame, velocities = build_local_counterfactual_state(
                state,
                attacker_id,
                attacker_path_xy,
                attacker_path_times_s,
                defender_id=action.defender_id,
                defender_path_xy=action.full_path_xy,
                defender_path_times_s=action.response_path_times_s,
            )
            partition = nearest_attacker_region_partition(
                frame,
                attacking_team_id,
                attacker_id,
                attacking_direction,
                xgrid,
                ygrid,
            )
            if global_alive:
                bounded = bounded_maximum_reference_obso(
                    frame,
                    attacking_team_id,
                    attacking_direction,
                    rejection_cutoff=global_best.value,
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                    config=obso_config,
                    batch_size=obso_batch_size,
                )
                maximum = bounded.maximum
                if not bounded.exact:
                    global_alive = False
                elif maximum.value > global_peak[0]:
                    global_peak = (maximum.value, state.time_s, maximum.x, maximum.y)
            if direct_alive:
                bounded = bounded_maximum_reference_obso_in_region(
                    frame,
                    attacking_team_id,
                    attacking_direction,
                    partition.focal_mask,
                    rejection_cutoff=direct.value,
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                    config=obso_config,
                    batch_size=obso_batch_size,
                )
                maximum = bounded.maximum
                if not bounded.exact:
                    direct_alive = False
                elif maximum.value > direct_peak[0]:
                    direct_peak = (maximum.value, state.time_s, maximum.x, maximum.y)
            if other_alive:
                bounded = bounded_maximum_reference_obso_in_region(
                    frame,
                    attacking_team_id,
                    attacking_direction,
                    partition.other_mask,
                    rejection_cutoff=other.value,
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                    config=obso_config,
                    batch_size=obso_batch_size,
                )
                maximum = bounded.maximum
                if not bounded.exact:
                    other_alive = False
                elif maximum.value > other_peak[0]:
                    other_peak = (maximum.value, state.time_s, maximum.x, maximum.y)
            if not direct_alive and not other_alive and not global_alive:
                break
        if valid_count == 0:
            raise ValueError("pass window contains no controlled release state")
        if direct_alive:
            candidate = _candidate("R", *direct_peak, action)
            if candidate.rank_key < direct.rank_key:
                direct = candidate
        if other_alive:
            candidate = _candidate("O", *other_peak, action)
            if candidate.rank_key < other.rank_key:
                other = candidate
        if global_alive:
            candidate = _candidate("max(R,O)", *global_peak, action)
            if candidate.rank_key < global_best.rank_key:
                global_best = candidate
        evaluated += 1
        if progress_every and action_index % progress_every == 0:
            print(
                f"pass-window responses {action_index}/{len(actions)} · "
                f"R={direct.value:.6f} O={other.value:.6f} "
                f"global={global_best.value:.6f}",
                flush=True,
            )
    return PassWindowResponseIncumbents(
        direct_cover=direct,
        other_cover=other,
        global_best=global_best,
        evaluated_response_count=evaluated,
    )
