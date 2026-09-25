"""Utilities for continuous two-option defensive trade-offs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairTradeoff:
    direct_action_index: int
    beneficiary_action_index: int
    compromise_action_index: int
    direct_minimum: float
    beneficiary_at_direct: float
    direct_at_beneficiary: float
    beneficiary_minimum: float
    direct_branch_gap: float
    beneficiary_branch_gap: float
    minimum_branch_gap: float
    normalized_compromise_regret: float
    screening_strength: float


def analyze_pair_tradeoff(
    direct_values: Sequence[float],
    beneficiary_values: Sequence[float],
) -> PairTradeoff:
    """Summarize branch gaps and the best continuous compromise.

    Both arrays are residual threats for the same ordered defensive actions;
    lower is better for the defender.  The normalized compromise regret is
    zero when one action attains both minima and approaches one when no action
    gets close to either branch optimum.  ``screening_strength`` requires both
    a two-way branch gap and the absence of an easy compromise.
    """

    direct = np.asarray(direct_values, dtype=float)
    beneficiary = np.asarray(beneficiary_values, dtype=float)
    if direct.ndim != 1 or beneficiary.ndim != 1 or direct.shape != beneficiary.shape:
        raise ValueError("threat arrays must be aligned one-dimensional vectors")
    if direct.size == 0 or not np.all(np.isfinite(direct)) or not np.all(
        np.isfinite(beneficiary)
    ):
        raise ValueError("threat arrays must be non-empty and finite")

    direct_index = int(np.argmin(direct))
    beneficiary_index = int(np.argmin(beneficiary))
    direct_gap = float(max(0.0, direct[beneficiary_index] - direct[direct_index]))
    beneficiary_gap = float(
        max(0.0, beneficiary[direct_index] - beneficiary[beneficiary_index])
    )
    direct_scale = max(direct_gap, 1e-12)
    beneficiary_scale = max(beneficiary_gap, 1e-12)
    normalized_regret = np.maximum(
        np.maximum(0.0, direct - direct[direct_index]) / direct_scale,
        np.maximum(0.0, beneficiary - beneficiary[beneficiary_index])
        / beneficiary_scale,
    )
    compromise_index = int(np.argmin(normalized_regret))
    compromise_regret = float(normalized_regret[compromise_index])
    minimum_gap = min(direct_gap, beneficiary_gap)
    return PairTradeoff(
        direct_action_index=direct_index,
        beneficiary_action_index=beneficiary_index,
        compromise_action_index=compromise_index,
        direct_minimum=float(direct[direct_index]),
        beneficiary_at_direct=float(beneficiary[direct_index]),
        direct_at_beneficiary=float(direct[beneficiary_index]),
        beneficiary_minimum=float(beneficiary[beneficiary_index]),
        direct_branch_gap=direct_gap,
        beneficiary_branch_gap=beneficiary_gap,
        minimum_branch_gap=minimum_gap,
        normalized_compromise_regret=compromise_regret,
        screening_strength=float(minimum_gap * compromise_regret),
    )
