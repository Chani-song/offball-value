"""Utilities for direct--derived defensive response maps.

The threat evaluator itself deliberately lives elsewhere.  This module only
defines the adaptive temporal window and summaries over an already evaluated
set of feasible defender responses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveHorizonConfig:
    """Temporal bounds for one observed off-ball action diagnostic."""

    minimum_seconds: float = 1.2
    maximum_seconds: float = 3.2
    evaluation_step_seconds: float = 0.4
    quantization_seconds: float = 0.1

    def validate(self) -> None:
        if self.minimum_seconds <= 0.0:
            raise ValueError("minimum_seconds must be positive")
        if self.maximum_seconds < self.minimum_seconds:
            raise ValueError("maximum_seconds must not be below minimum_seconds")
        if self.evaluation_step_seconds <= 0.0:
            raise ValueError("evaluation_step_seconds must be positive")
        if self.quantization_seconds <= 0.0:
            raise ValueError("quantization_seconds must be positive")


@dataclass(frozen=True)
class AdaptiveHorizon:
    seconds: float
    evaluation_times_s: tuple[float, ...]
    limiting_reason: str
    seconds_to_shot: float
    observed_termination_seconds: float | None


def evaluation_times_for_horizon(
    horizon_seconds: float,
    step_seconds: float = 0.4,
) -> tuple[float, ...]:
    """Return regular post-onset samples plus the exact terminal time."""

    if horizon_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("horizon and step must be positive")
    values: list[float] = []
    value = step_seconds
    while value < horizon_seconds - 1e-9:
        values.append(float(round(value, 10)))
        value += step_seconds
    values.append(float(round(horizon_seconds, 10)))
    return tuple(dict.fromkeys(values))


def choose_adaptive_horizon(
    seconds_to_shot: float,
    observed_termination_seconds: float | None = None,
    config: AdaptiveHorizonConfig = AdaptiveHorizonConfig(),
) -> AdaptiveHorizon:
    """Choose a causal observed-future window without forcing two seconds.

    The horizon ends at the earliest of the shot, an observed possession/run
    termination supplied by the caller, and the explicit diagnostic cap.  It
    is quantized downward so the response integrator never crosses that event.
    """

    config.validate()
    if seconds_to_shot <= 0.0 or not math.isfinite(seconds_to_shot):
        raise ValueError("seconds_to_shot must be finite and positive")
    limits = [(float(seconds_to_shot), "shot")]
    if observed_termination_seconds is not None:
        if observed_termination_seconds <= 0.0 or not math.isfinite(
            observed_termination_seconds
        ):
            raise ValueError("observed termination must be finite and positive")
        limits.append((float(observed_termination_seconds), "observed_termination"))
    limits.append((float(config.maximum_seconds), "diagnostic_cap"))
    raw, reason = min(limits, key=lambda item: (item[0], item[1]))
    quantum = config.quantization_seconds
    quantized = math.floor((raw + 1e-9) / quantum) * quantum
    seconds = max(config.minimum_seconds, quantized)
    # A genuinely earlier observed event must never be crossed just to satisfy
    # the preferred minimum. Callers should reject such a scene instead.
    if raw < config.minimum_seconds - 1e-9:
        raise ValueError("usable observed window is shorter than the minimum")
    return AdaptiveHorizon(
        seconds=float(round(seconds, 10)),
        evaluation_times_s=evaluation_times_for_horizon(
            seconds, config.evaluation_step_seconds
        ),
        limiting_reason=reason,
        seconds_to_shot=float(seconds_to_shot),
        observed_termination_seconds=(
            float(observed_termination_seconds)
            if observed_termination_seconds is not None
            else None
        ),
    )


def trapezoid_horizon_mean(
    values: Sequence[float],
    times_s: Sequence[float],
) -> float:
    """Average a threat trace over its sampled post-onset interval."""

    array = np.asarray(values, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if array.ndim != 1 or times.ndim != 1 or array.shape != times.shape:
        raise ValueError("values and times must be aligned one-dimensional arrays")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be non-empty and finite")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        if array.size == 1 and np.all(np.isfinite(times)):
            return float(array[0])
        raise ValueError("times must be finite and strictly increasing")
    if array.size == 1:
        return float(array[0])
    return float(np.trapezoid(array, times) / (times[-1] - times[0]))


def pareto_frontier_indices(
    direct_values: Sequence[float],
    derived_values: Sequence[float],
    tolerance: float = 1e-12,
) -> tuple[int, ...]:
    """Return non-dominated response indices for two defender-minimized costs."""

    direct = np.asarray(direct_values, dtype=float)
    derived = np.asarray(derived_values, dtype=float)
    if direct.ndim != 1 or derived.ndim != 1 or direct.shape != derived.shape:
        raise ValueError("direct and derived values must be aligned vectors")
    if direct.size == 0 or not np.all(np.isfinite(direct)) or not np.all(
        np.isfinite(derived)
    ):
        raise ValueError("cost vectors must be non-empty and finite")
    frontier = []
    for index in range(len(direct)):
        weakly_better = (direct <= direct[index] + tolerance) & (
            derived <= derived[index] + tolerance
        )
        strictly_better = (direct < direct[index] - tolerance) | (
            derived < derived[index] - tolerance
        )
        if not bool(np.any(weakly_better & strictly_better)):
            frontier.append(index)
    return tuple(frontier)

