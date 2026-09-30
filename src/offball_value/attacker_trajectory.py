"""R8: give the attacker the same counterfactual treatment as the defender.

Everything else in this pipeline is already counterfactual. The defender is
simulated over thousands of physically feasible responses; the pass is
priced at every release time in the window and the best is taken. Only the
attacking options stayed pinned to the one path they happened to run.

The reviewer's objection (2026-09-08): a team-mate who misreads the run
scores zero occupation, so the run itself gets credited with nothing — but
"the beneficiary candidates moved stupidly" is a fact about the team-mate, not
about the run. The first attempt (R7) fixed that by asking only "could he
reach the space in time", which threw away the direction information that
made R6 work: with a 1.5 s window most nearby players can reach it, so the
nearest man wins again and R7 fell to 20/24.

His refinement, implemented here: keep the initial state — position AND
velocity — and search the trajectories reachable from it, then take the one
that best attacks the vacated space. A player already running into the
space pays nothing; a player running away must decelerate and turn first,
and the physics charges him for it. So intent survives while stupidity is
forgiven, which is exactly the asymmetry that was missing.

The trajectory lattice is the existing target-free generator used for
defenders (``generate_feasible_defender_trajectories``) — it takes a start
state and returns bounded, physically legal paths, with no notion of which
team is running.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .defender_trajectory_search import (
    DefenderTrajectorySearchConfig,
    generate_feasible_defender_trajectories,
)
from .vacated_space import coverage_field

# Attackers accelerate and turn like the outfielders they are; this is the
# same physical envelope the defender search uses, not a fitted quantity.
ATTACKER_SEARCH = DefenderTrajectorySearchConfig(
    planning_horizon_seconds=1.5,
    # An attacker reading the run does not owe a reaction delay: he is the
    # one initiating, so his options start immediately.
    response_delay_seconds=0.0,
    maximum_representatives=48,
)


def best_attacking_trajectory(
    start_xy: tuple[float, float],
    start_velocity: tuple[float, float],
    xs: np.ndarray,
    ys: np.ndarray,
    loss: np.ndarray,
    horizon_seconds: float = 1.5,
    sample_step_seconds: float = 0.25,
    config: DefenderTrajectorySearchConfig = ATTACKER_SEARCH,
) -> tuple[float, tuple]:
    """Best achievable occupation of the vacated region, and the path.

    Scores every feasible trajectory from this start state by the same
    time-averaged coverage of the opened space that R6 applies to the
    observed path, and returns the maximum. Ties break on the lower
    trajectory id inside the generator, so the result is deterministic.
    """
    total = float(loss.sum())
    if total <= 1e-12:
        return 0.0, ()
    candidates = generate_feasible_defender_trajectories(
        start_xy, start_velocity, horizon_seconds, config
    )
    if not candidates:
        return 0.0, ()

    steps = max(int(round(horizon_seconds / sample_step_seconds)), 1)
    best_score, best_path = -1.0, ()
    for candidate in candidates:
        path = candidate.path_txy
        score = 0.0
        for index in range(1, steps + 1):
            target = index * sample_step_seconds
            xy, velocity = _sample(path, target)
            score += float((loss * coverage_field(xy, velocity, xs, ys)).sum())
        score /= steps * total
        if score > best_score + 1e-12:
            best_score, best_path = score, path
    return best_score, best_path


def _sample(
    path: Sequence[tuple[float, float, float]], time_s: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Position and finite-difference velocity on a trajectory at ``time_s``."""

    def at(target: float) -> tuple[float, float]:
        previous = path[0]
        for row in path:
            if float(row[0]) >= target:
                if float(row[0]) == float(previous[0]):
                    return (float(row[1]), float(row[2]))
                fraction = (target - float(previous[0])) / (
                    float(row[0]) - float(previous[0])
                )
                return (
                    float(previous[1]) + fraction * (float(row[1]) - float(previous[1])),
                    float(previous[2]) + fraction * (float(row[2]) - float(previous[2])),
                )
            previous = row
        return (float(path[-1][1]), float(path[-1][2]))

    now = at(time_s)
    before = at(max(0.0, time_s - 0.2))
    return now, ((now[0] - before[0]) / 0.2, (now[1] - before[1]) / 0.2)


def rule_r8(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    option_values: Mapping[str, float],
    horizon_seconds: float = 1.5,
) -> tuple[str, dict[str, float]]:
    """Beneficiary = best ACHIEVABLE occupation of the vacated space x value."""
    del carrier_id
    from .vacated_space import _region_grid, _state_v, _state_xy

    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == attacking_team or player_id == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys
        ),
    )
    trail = np.zeros_like(before)
    for after_xy, after_v in samples:
        np.maximum(
            trail,
            np.clip(
                before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
                0.0,
                None,
            ),
            out=trail,
        )
    if float(trail.sum()) <= 1e-12:
        return "", {"_vacated_area": 0.0}

    achievable: dict[str, float] = {}
    for player_id, row in state.items():
        if str(row["team"]) != attacking_team or player_id == runner_id:
            continue
        if player_id not in option_values:
            continue
        score, _ = best_attacking_trajectory(
            _state_xy(row), _state_v(row), xs, ys, trail, horizon_seconds
        )
        achievable[player_id] = score

    combined = {
        option_id: achievable.get(option_id, 0.0) * float(value)
        for option_id, value in option_values.items()
        if option_id != runner_id
    }
    if not combined:
        return "", {"_vacated_area": float(trail.sum())}
    best = max(combined, key=lambda key: (combined[key], key))
    combined["_vacated_area"] = float(trail.sum())
    return best, combined
