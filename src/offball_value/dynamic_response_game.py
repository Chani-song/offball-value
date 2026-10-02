"""Selection rules for a dynamic local defender-response cost matrix.

Rows are attacking options, columns are feasible defender trajectories, and a
smaller entry means that the defender controls that option better over time.
The matrix is deliberately agnostic to the value model: geometry costs can be
used for auditing now and replaced by calibrated threat later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DynamicGameSelection:
    option_minimum_indices: tuple[int, ...]
    local_minimax_index: int
    compromise_index: int
    pareto_indices: tuple[int, ...]
    normalized_regret: np.ndarray


def _validate(
    cost_matrix: np.ndarray,
    efforts: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(cost_matrix, dtype=float)
    effort = np.asarray(efforts, dtype=float)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[1] < 1:
        raise ValueError("cost_matrix must contain at least one option and response")
    if effort.shape != (costs.shape[1],):
        raise ValueError("efforts must contain one value per response")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(effort)):
        raise ValueError("costs and efforts must be finite")
    return costs, effort


def normalized_option_regret(cost_matrix: np.ndarray) -> np.ndarray:
    """Normalize each option from its best to worst feasible response."""

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[1] < 1:
        raise ValueError("cost_matrix must be two-dimensional and non-empty")
    if not np.all(np.isfinite(costs)):
        raise ValueError("cost_matrix must be finite")
    minimum = np.min(costs, axis=1, keepdims=True)
    spread = np.ptp(costs, axis=1, keepdims=True)
    return np.divide(
        costs - minimum,
        spread,
        out=np.zeros_like(costs),
        where=spread > 1e-12,
    )


def pareto_response_indices(cost_matrix: np.ndarray) -> tuple[int, ...]:
    """Return response columns not dominated on every option cost."""

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[1] < 1:
        raise ValueError("cost_matrix must be two-dimensional and non-empty")
    keep = []
    for candidate in range(costs.shape[1]):
        dominated = False
        for challenger in range(costs.shape[1]):
            if challenger == candidate:
                continue
            if np.all(costs[:, challenger] <= costs[:, candidate]) and np.any(
                costs[:, challenger] < costs[:, candidate]
            ):
                dominated = True
                break
        if not dominated:
            keep.append(candidate)
    return tuple(keep)


def select_dynamic_game_responses(
    cost_matrix: np.ndarray,
    efforts: Sequence[float],
) -> DynamicGameSelection:
    """Extract option minima, raw local minimax, and regret compromise."""

    costs, effort = _validate(cost_matrix, efforts)
    regret = normalized_option_regret(costs)
    option_minima = tuple(
        min(
            range(costs.shape[1]),
            key=lambda response: (
                costs[option, response],
                effort[response],
                response,
            ),
        )
        for option in range(costs.shape[0])
    )
    local_minimax = min(
        range(costs.shape[1]),
        key=lambda response: (
            float(np.max(costs[:, response])),
            float(np.sum(costs[:, response])),
            effort[response],
            response,
        ),
    )
    compromise = min(
        range(costs.shape[1]),
        key=lambda response: (
            float(np.max(regret[:, response])),
            float(np.sum(regret[:, response])),
            effort[response],
            response,
        ),
    )
    return DynamicGameSelection(
        option_minimum_indices=option_minima,
        local_minimax_index=int(local_minimax),
        compromise_index=int(compromise),
        pareto_indices=pareto_response_indices(costs),
        normalized_regret=regret,
    )
