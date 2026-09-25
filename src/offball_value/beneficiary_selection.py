"""Transparent beneficiary selection for a local defensive-allocation game.

The beneficiary is not the teammate with the largest free-space fraction.  A
candidate must have an option that improves after run onset *and* that can be
meaningfully reduced by a feasible alternative defender response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BeneficiarySelectionConfig:
    """Pre-registered gates for the v0.2 diagnostic.

    The defaults deliberately impose no tuned effect-size threshold.  They
    reject non-positive effects and expose the raw ranking for human audit.
    Population-level thresholds are to be chosen after inspecting the score
    distribution, rather than tuned on the three demonstration scenes.
    """

    minimum_option_value: float = 0.0
    minimum_creation_gain: float = 0.0
    minimum_choice_effect: float = 0.0


@dataclass(frozen=True)
class BeneficiaryCandidateScore:
    player_id: str
    onset_value: float
    runner_cover_value: float
    best_cover_value: float
    creation_gain: float
    choice_effect: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[float, float, float, str]:
        """Sort eligible candidates by allocation effect, then creation."""

        return (
            -self.choice_effect,
            -self.creation_gain,
            -self.runner_cover_value,
            self.player_id,
        )


@dataclass(frozen=True)
class BeneficiarySelection:
    selected_player_id: str | None
    candidates: tuple[BeneficiaryCandidateScore, ...]


@dataclass(frozen=True)
class FollowBeneficiaryCandidateScore:
    """Pairwise cross-cost induced by a runner-follow response."""

    player_id: str
    follow_value: float
    best_cover_value: float
    follow_benefit: float
    runner_follow_value: float
    runner_at_candidate_cover: float
    runner_cost: float
    follow_benefit_fraction: float
    runner_cost_fraction: float
    dilemma_score: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[float, float, float, str]:
        return (
            -self.dilemma_score,
            -self.follow_benefit,
            -self.runner_cost,
            self.player_id,
        )


@dataclass(frozen=True)
class FollowBeneficiarySelection:
    selected_player_id: str | None
    candidates: tuple[FollowBeneficiaryCandidateScore, ...]


def select_beneficiary(
    onset_values: Mapping[str, float],
    runner_cover_values: Mapping[str, float],
    best_cover_values: Mapping[str, float],
    *,
    legal_option: Mapping[str, bool] | None = None,
    meaningful_attacking_option: Mapping[str, bool] | None = None,
    config: BeneficiarySelectionConfig = BeneficiarySelectionConfig(),
) -> BeneficiarySelection:
    """Rank teammates using onset gain and defender-allocation effect.

    ``runner_cover_values`` are measured under the response that minimizes the
    focal runner's direct threat. ``best_cover_values`` are the lowest values
    attainable for the same teammate option over the feasible defender action
    set.  A player is selected only when all configured gates are passed.
    """

    player_ids = set(onset_values) | set(runner_cover_values) | set(best_cover_values)
    if not player_ids:
        return BeneficiarySelection(selected_player_id=None, candidates=())
    missing = {
        player_id
        for player_id in player_ids
        if player_id not in onset_values
        or player_id not in runner_cover_values
        or player_id not in best_cover_values
    }
    if missing:
        raise ValueError(f"beneficiary inputs are incomplete for: {sorted(missing)}")

    rows = []
    for player_id in sorted(player_ids):
        onset = float(onset_values[player_id])
        runner_cover = float(runner_cover_values[player_id])
        best_cover = float(best_cover_values[player_id])
        creation = runner_cover - onset
        choice = runner_cover - best_cover
        reasons = []
        if legal_option is not None and not bool(legal_option.get(player_id, False)):
            reasons.append("no_legal_option")
        if meaningful_attacking_option is not None and not bool(
            meaningful_attacking_option.get(player_id, False)
        ):
            reasons.append("not_attacking_option")
        if runner_cover <= config.minimum_option_value:
            reasons.append("option_too_small")
        if creation <= config.minimum_creation_gain:
            reasons.append("no_positive_creation_gain")
        if choice <= config.minimum_choice_effect:
            reasons.append("no_positive_choice_effect")
        rows.append(
            BeneficiaryCandidateScore(
                player_id=player_id,
                onset_value=onset,
                runner_cover_value=runner_cover,
                best_cover_value=best_cover,
                creation_gain=creation,
                choice_effect=choice,
                eligible=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )

    ordered = tuple(sorted(rows, key=lambda item: (not item.eligible, item.rank_key)))
    selected = next((item.player_id for item in ordered if item.eligible), None)
    return BeneficiarySelection(selected_player_id=selected, candidates=ordered)


def select_follow_beneficiary(
    follow_values: Mapping[str, float],
    best_cover_values: Mapping[str, float],
    runner_follow_value: float,
    runner_at_candidate_cover: Mapping[str, float],
    *,
    legal_option: Mapping[str, bool] | None = None,
    meaningful_attacking_option: Mapping[str, bool] | None = None,
    numerical_tolerance: float = 1e-6,
    minimum_follow_benefit_fraction: float = 0.0,
    minimum_runner_cost_fraction: float = 0.0,
) -> FollowBeneficiarySelection:
    """Select who benefits when the defender follows the focal runner.

    A candidate must exhibit both sides of the allocation trade-off: its own
    option must fall under a candidate-cover response, and the focal runner's
    option must rise under that same response.  Temporal change from ``t=0``
    is intentionally absent from the selection rule.
    """

    player_ids = set(follow_values) | set(best_cover_values) | set(
        runner_at_candidate_cover
    )
    missing = {
        player_id
        for player_id in player_ids
        if player_id not in follow_values
        or player_id not in best_cover_values
        or player_id not in runner_at_candidate_cover
    }
    if missing:
        raise ValueError(f"follow-beneficiary inputs are incomplete for: {sorted(missing)}")
    runner_follow = float(runner_follow_value)
    rows = []
    for player_id in sorted(player_ids):
        follow = float(follow_values[player_id])
        cover = float(best_cover_values[player_id])
        runner_cover = float(runner_at_candidate_cover[player_id])
        follow_benefit = follow - cover
        runner_cost = runner_cover - runner_follow
        follow_fraction = follow_benefit / max(abs(follow), numerical_tolerance)
        runner_fraction = runner_cost / max(abs(runner_cover), numerical_tolerance)
        reasons = []
        if legal_option is not None and not bool(legal_option.get(player_id, False)):
            reasons.append("no_legal_option")
        if meaningful_attacking_option is not None and not bool(
            meaningful_attacking_option.get(player_id, False)
        ):
            reasons.append("not_attacking_option")
        if follow_benefit <= numerical_tolerance:
            reasons.append("no_follow_benefit")
        elif follow_fraction < minimum_follow_benefit_fraction:
            reasons.append("follow_benefit_too_small")
        if runner_cost <= numerical_tolerance:
            reasons.append("no_runner_cost")
        elif runner_fraction < minimum_runner_cost_fraction:
            reasons.append("runner_cost_too_small")
        rows.append(
            FollowBeneficiaryCandidateScore(
                player_id=player_id,
                follow_value=follow,
                best_cover_value=cover,
                follow_benefit=follow_benefit,
                runner_follow_value=runner_follow,
                runner_at_candidate_cover=runner_cover,
                runner_cost=runner_cost,
                follow_benefit_fraction=follow_fraction,
                runner_cost_fraction=runner_fraction,
                dilemma_score=min(follow_fraction, runner_fraction),
                eligible=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )
    ordered = tuple(sorted(rows, key=lambda item: (not item.eligible, item.rank_key)))
    selected = next((item.player_id for item in ordered if item.eligible), None)
    return FollowBeneficiarySelection(selected_player_id=selected, candidates=ordered)
