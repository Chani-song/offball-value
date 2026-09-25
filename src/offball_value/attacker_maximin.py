"""Terminal attacker max--defender min search for local counterfactuals.

The module keeps candidate construction and threat evaluation separate.  An
attacker commits to a causally generated feasible two-second path.  At t+2,
one of the action-specific relevant defenders may follow any causally
generated delayed response.  The ball and all non-intervened players retain
their observed future in this localized retrospective experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .action_value import nearest_attacker_region_partition
from .defender_best_response import (
    ObservedBackgroundState,
    build_local_counterfactual_state,
)
from .defender_response import DefenderResponseActionSet
from .reference_obso import (
    ReferenceOBSOConfig,
    bounded_maximum_reference_obso_in_region,
    bounded_maximum_reference_obso,
    maximum_reference_obso,
    pitch_grid,
)


@dataclass(frozen=True)
class TerminalThreatResult:
    branch_id: str
    value: float
    maximum_x: float
    maximum_y: float
    evaluated_cell_count: int
    defender_id: str | None = None
    defender_action_id: str | None = None
    defender_effort_m2ps3: float = 0.0

    @property
    def rank_key(self) -> tuple[float, float, str]:
        return self.value, self.defender_effort_m2ps3, self.branch_id


@dataclass(frozen=True)
class TerminalBestResponseSearch:
    observed_background_reference: TerminalThreatResult
    best: TerminalThreatResult
    evaluated_response_count: int
    defender_action_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TerminalBranchMinimum:
    """A response selected to minimize one explanation branch."""

    branch: str
    value: float
    maximum_x: float
    maximum_y: float
    evaluated_cell_count: int
    defender_id: str
    defender_action_id: str
    defender_effort_m2ps3: float

    @property
    def rank_key(self) -> tuple[float, float, str]:
        return self.value, self.defender_effort_m2ps3, self.defender_action_id


@dataclass(frozen=True)
class TerminalBranchResponseSearch:
    """Exact minima for direct-runner R and other-team O branches."""

    direct_cover: TerminalBranchMinimum
    other_cover: TerminalBranchMinimum
    evaluated_response_count: int
    defender_action_counts: tuple[tuple[str, int], ...]


def evaluate_terminal_threat(
    background: ObservedBackgroundState,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    branch_id: str,
    defender_id: str | None = None,
    defender_path_xy: Sequence[Sequence[float]] | None = None,
    defender_path_times_s: Sequence[float] | None = None,
    defender_action_id: str | None = None,
    defender_effort_m2ps3: float = 0.0,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    obso_batch_size: int = 8,
) -> TerminalThreatResult:
    """Evaluate the exact maximum reference OBSO at the terminal state."""

    frame, velocities = build_local_counterfactual_state(
        background,
        attacker_id,
        attacker_path_xy,
        attacker_path_times_s,
        defender_id=defender_id,
        defender_path_xy=defender_path_xy,
        defender_path_times_s=defender_path_times_s,
    )
    maximum = maximum_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
        config=obso_config,
        batch_size=obso_batch_size,
    )
    return TerminalThreatResult(
        branch_id=branch_id,
        value=maximum.value,
        maximum_x=maximum.x,
        maximum_y=maximum.y,
        evaluated_cell_count=maximum.evaluated_cell_count,
        defender_id=defender_id,
        defender_action_id=defender_action_id,
        defender_effort_m2ps3=defender_effort_m2ps3,
    )


def search_terminal_best_response(
    background: ObservedBackgroundState,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    response_action_sets: Mapping[str, DefenderResponseActionSet],
    fixed_background: TerminalThreatResult | None = None,
    initial_feasible_response: TerminalThreatResult | None = None,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    obso_batch_size: int = 8,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> TerminalBestResponseSearch:
    """Minimize t+2 team threat over one-defender feasible responses.

    The observed local background is reported for interpretation but never
    enters the response candidate set.  Every optimizing response is produced
    from t=0 information and the frozen motion constraints.
    """

    fixed = fixed_background or evaluate_terminal_threat(
        background,
        attacker_id,
        attacker_path_xy,
        attacker_path_times_s,
        attacking_team_id,
        attacking_direction,
        goalkeeper_ids,
        branch_id="observed-background-reference",
        obso_config=obso_config,
        obso_batch_size=obso_batch_size,
    )
    best: TerminalThreatResult | None = initial_feasible_response
    initial_action_id = (
        initial_feasible_response.defender_action_id
        if initial_feasible_response is not None
        else None
    )
    evaluated = 0
    counts: list[tuple[str, int]] = []
    for defender_id, action_set in response_action_sets.items():
        counts.append((defender_id, len(action_set.actions)))
        for action in action_set.actions:
            if action.action_id == initial_action_id:
                continue
            frame, velocities = build_local_counterfactual_state(
                background,
                attacker_id,
                attacker_path_xy,
                attacker_path_times_s,
                defender_id=defender_id,
                defender_path_xy=action.full_path_xy,
                defender_path_times_s=action.response_path_times_s,
            )
            bounded = bounded_maximum_reference_obso(
                frame,
                attacking_team_id,
                attacking_direction,
                rejection_cutoff=(best.value if best is not None else 1.0),
                velocities=velocities,
                goalkeeper_ids=goalkeeper_ids,
                apply_offside=True,
                config=obso_config,
                batch_size=obso_batch_size,
            )
            maximum = bounded.maximum
            result = TerminalThreatResult(
                branch_id=action.action_id,
                value=maximum.value,
                maximum_x=maximum.x,
                maximum_y=maximum.y,
                evaluated_cell_count=maximum.evaluated_cell_count,
                defender_id=defender_id,
                defender_action_id=action.action_id,
                defender_effort_m2ps3=action.base_action.motion.effort_m2ps3,
            )
            evaluated += 1
            if bounded.exact and (best is None or result.rank_key < best.rank_key):
                best = result
            if progress_every > 0 and evaluated % progress_every == 0:
                message = (
                    f"terminal responses evaluated: {evaluated}; "
                    f"incumbent={best.value:.9f}"
                )
                if progress is None:
                    print(message, flush=True)
                else:
                    progress(message)
    if best is None:
        raise ValueError("at least one feasible defender response is required")
    return TerminalBestResponseSearch(
        observed_background_reference=fixed,
        best=best,
        evaluated_response_count=evaluated,
        defender_action_counts=tuple(counts),
    )


def search_terminal_branch_responses(
    background: ObservedBackgroundState,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    response_action_sets: Mapping[str, DefenderResponseActionSet],
    initial_direct_cover: TerminalBranchMinimum | None = None,
    initial_other_cover: TerminalBranchMinimum | None = None,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    obso_batch_size: int = 8,
    progress_every: int = 0,
    progress: Callable[[str], None] | None = None,
) -> TerminalBranchResponseSearch:
    """Find responses that separately minimize R and O.

    For every defender response, the terminal OBSO grid is partitioned by the
    nearest eligible attacker.  Regional safe upper-bound searches then find
    the exact minimum of the focal runner branch R and the other-team branch O.
    The original global response remains ``min max(R, O)`` and is intentionally
    searched by :func:`search_terminal_best_response`.
    """

    xgrid, ygrid, _ = pitch_grid(obso_config)
    direct = initial_direct_cover
    other = initial_other_cover
    direct_seed_id = direct.defender_action_id if direct is not None else None
    other_seed_id = other.defender_action_id if other is not None else None
    evaluated = 0
    counts: list[tuple[str, int]] = []
    for defender_id, action_set in response_action_sets.items():
        counts.append((defender_id, len(action_set.actions)))
        for action in action_set.actions:
            frame, velocities = build_local_counterfactual_state(
                background,
                attacker_id,
                attacker_path_xy,
                attacker_path_times_s,
                defender_id=defender_id,
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
            if action.action_id != direct_seed_id:
                bounded_direct = bounded_maximum_reference_obso_in_region(
                    frame,
                    attacking_team_id,
                    attacking_direction,
                    partition.focal_mask,
                    rejection_cutoff=(direct.value if direct is not None else 1.0),
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                    config=obso_config,
                    batch_size=obso_batch_size,
                )
                maximum = bounded_direct.maximum
                candidate = TerminalBranchMinimum(
                    branch="R",
                    value=maximum.value,
                    maximum_x=maximum.x,
                    maximum_y=maximum.y,
                    evaluated_cell_count=maximum.evaluated_cell_count,
                    defender_id=defender_id,
                    defender_action_id=action.action_id,
                    defender_effort_m2ps3=action.base_action.motion.effort_m2ps3,
                )
                if bounded_direct.exact and (
                    direct is None or candidate.rank_key < direct.rank_key
                ):
                    direct = candidate
            if action.action_id != other_seed_id:
                bounded_other = bounded_maximum_reference_obso_in_region(
                    frame,
                    attacking_team_id,
                    attacking_direction,
                    partition.other_mask,
                    rejection_cutoff=(other.value if other is not None else 1.0),
                    velocities=velocities,
                    goalkeeper_ids=goalkeeper_ids,
                    apply_offside=True,
                    config=obso_config,
                    batch_size=obso_batch_size,
                )
                maximum = bounded_other.maximum
                candidate = TerminalBranchMinimum(
                    branch="O",
                    value=maximum.value,
                    maximum_x=maximum.x,
                    maximum_y=maximum.y,
                    evaluated_cell_count=maximum.evaluated_cell_count,
                    defender_id=defender_id,
                    defender_action_id=action.action_id,
                    defender_effort_m2ps3=action.base_action.motion.effort_m2ps3,
                )
                if bounded_other.exact and (
                    other is None or candidate.rank_key < other.rank_key
                ):
                    other = candidate
            evaluated += 1
            if progress_every > 0 and evaluated % progress_every == 0:
                message = (
                    f"branch responses evaluated: {evaluated}; "
                    f"R={direct.value:.9f}; O={other.value:.9f}"
                )
                if progress is None:
                    print(message, flush=True)
                else:
                    progress(message)
    if direct is None or other is None:
        raise ValueError("at least one feasible defender response is required")
    return TerminalBranchResponseSearch(
        direct_cover=direct,
        other_cover=other,
        evaluated_response_count=evaluated,
        defender_action_counts=tuple(counts),
    )
