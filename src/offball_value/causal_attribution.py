"""Causal labels for defender-conditioned off-ball option values.

The module keeps three quantities separate:

* absolute option threat under the observed runner action;
* the same option's threat under a neutral runner action;
* exposure caused by allocating a defender to the runner instead of the option.

This prevents a large pre-existing option from being called runner-created only
because a defender could suppress it under a different response.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionAttribution:
    actual_value: float
    neutral_value: float
    causal_gain: float
    causal_gain_fraction: float
    allocation_effect: float
    allocation_effect_fraction: float
    label: str


def attribute_option(
    actual_value: float,
    neutral_value: float,
    allocation_effect: float,
    allocation_reference_value: float,
    *,
    minimum_relative_effect: float = 0.05,
    creation_baseline_fraction: float = 0.25,
    numerical_tolerance: float = 1e-6,
) -> OptionAttribution:
    """Classify an option as created, amplified, exposed, or weak.

    ``actual_value`` and ``neutral_value`` must use the same absolute value
    contract. ``allocation_effect`` is the within-actual-run difference between
    runner-follow and option-cover responses.  Relative effects use the actual
    or follow value as their denominator so scale alone cannot create a label.
    """

    actual = float(actual_value)
    neutral = float(neutral_value)
    allocation = float(allocation_effect)
    causal = actual - neutral
    causal_fraction = causal / max(abs(actual), numerical_tolerance)
    allocation_fraction = allocation / max(
        abs(float(allocation_reference_value)), numerical_tolerance
    )
    causal_positive = (
        causal > numerical_tolerance
        and causal_fraction >= minimum_relative_effect
    )
    allocation_positive = (
        allocation > numerical_tolerance
        and allocation_fraction >= minimum_relative_effect
    )
    if causal_positive:
        baseline_fraction = neutral / max(abs(actual), numerical_tolerance)
        label = (
            "runner_created"
            if baseline_fraction <= creation_baseline_fraction
            else "runner_amplified"
        )
    elif allocation_positive:
        label = "pre_existing_exposed"
    elif causal < -numerical_tolerance and abs(causal_fraction) >= minimum_relative_effect:
        label = "runner_suppressed"
    else:
        label = "weak_or_unrelated"
    return OptionAttribution(
        actual_value=actual,
        neutral_value=neutral,
        causal_gain=causal,
        causal_gain_fraction=causal_fraction,
        allocation_effect=allocation,
        allocation_effect_fraction=allocation_fraction,
        label=label,
    )
