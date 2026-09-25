"""The dilemma asked in geometry alone, with no value model.

Measured on 2026-09-10: the value-based dilemma score is almost entirely a
danger detector. A control using no two-sided structure at all — the median
Q in the scene — reaches AUC 0.624 against the structural measure's 0.639,
and every transform that strips the danger level out collapses to 0.51-0.60.
Near the box every Q is high whether or not the defender is actually torn.

The confound is inherent to the quantity: the knee is an ABSOLUTE threat
level, while a dilemma is RELATIVE structure. So ask the structural question
directly and leave Q out of it:

    is there anywhere A can be, at any moment, that covers BOTH X and Y?

    reachable_miss(A, target, t) = 1 - coverage of the target's position at t
                                   by A standing where he could be at t
    dilemma = min over A's feasible paths, max over t of
              max(reachable_miss to X, reachable_miss to Y)

High means no path lets him hold both; low means he covers both comfortably.
Nothing in it refers to goal danger, so a midfield 2v1 and a penalty-box 2v1
score the same when the geometry is the same — which is what the reviewer
appears to be judging.

This uses the SAME ball-free coverage ellipse as R9 and the same feasible
defender trajectories the payoff pipeline already generates, so it adds no
new constant and no new model. Whether the runner's and beneficiary's own
positions come from observed tracking is inherited from the caller, matching
R9's validated choice.

UNTESTED at the time of writing. The value-based gap measure is being
re-measured first; this is the alternative if it fails.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .vacated_space import coverage_field


def _coverage_at(
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    point_xy: tuple[float, float],
) -> float:
    """Coverage of one point, delegating so the geometry cannot drift."""
    return float(
        coverage_field(
            player_xy,
            velocity_xy,
            np.array([float(point_xy[0])]),
            np.array([float(point_xy[1])]),
        )[0, 0]
    )


def path_miss(
    path_txy: Sequence[Sequence[float]],
    target_at: Callable[[float], tuple[float, float]],
    times: Sequence[float],
) -> float:
    """Worst uncovered-ness of one target along one defender path.

    Taking the MAXIMUM over time, not the mean: a defender who loses his man
    for a moment has lost him. The pass only has to arrive once.
    """
    rows = [(float(p[0]), float(p[1]), float(p[2])) for p in path_txy]

    def at(time_s: float) -> tuple[float, float]:
        previous = rows[0]
        for row in rows:
            if row[0] >= time_s:
                if row[0] == previous[0]:
                    return (row[1], row[2])
                fraction = (time_s - previous[0]) / (row[0] - previous[0])
                return (
                    previous[1] + fraction * (row[1] - previous[1]),
                    previous[2] + fraction * (row[2] - previous[2]),
                )
            previous = row
        return (rows[-1][1], rows[-1][2])

    worst = 0.0
    for time_s in times:
        now = at(time_s)
        before = at(max(0.0, time_s - 0.2))
        velocity = ((now[0] - before[0]) / 0.2, (now[1] - before[1]) / 0.2)
        worst = max(worst, 1.0 - _coverage_at(now, velocity, target_at(time_s)))
    return worst


def geometric_dilemma(
    paths: Sequence[Sequence[Sequence[float]]],
    runner_at: Callable[[float], tuple[float, float]],
    beneficiary_at: Callable[[float], tuple[float, float]],
    times: Sequence[float],
) -> dict[str, float]:
    """How badly A is stretched between two men, in geometry only.

    ``paths`` are the defender's feasible trajectories — the same ones the
    payoff pipeline generates. Returns the minimax over them plus the two
    single-target floors, so the caller can form the structural gap that the
    value version could not: holding both minus holding the harder one alone.
    """
    if not paths:
        return {}
    miss_runner = [path_miss(path, runner_at, times) for path in paths]
    miss_other = [path_miss(path, beneficiary_at, times) for path in paths]
    both = min(max(a, b) for a, b in zip(miss_runner, miss_other))
    floor_runner = min(miss_runner)
    floor_other = min(miss_other)
    return {
        # The dilemma: the best he can do against the PAIR.
        "geometric_knee": both,
        "floor_runner": floor_runner,
        "floor_beneficiary": floor_other,
        # Structural surplus: what the pair costs over the harder single man.
        # Both terms are pure geometry, so scene danger cancels exactly
        # rather than approximately.
        "geometric_gap": both - max(floor_runner, floor_other),
        "paths": float(len(paths)),
    }
