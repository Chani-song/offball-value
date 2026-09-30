"""R6: the reviewer's vacated-space beneficiary rule, as a single entry point.

Arrived at with the reviewer on 2026-09-08. In his words the beneficiary of
an off-ball run is found like this:

  When a defender moves in reaction to an off-ball runner, his influence
  over part of the area he was covering shrinks. The beneficiary is
  whoever can best exploit it.

with three refinements he made while reading the rule's own misses:

1. The vacated region ACCUMULATES along the defender's whole reaction path
   ("like Hansel and Gretel dropping crumbs"), not at a single instant.
2. Both sides advance through TIME. Scoring attackers frozen at the onset
   instant cannot express "Iyoha moves away from that area and Klarer gets
   closer" — the observation that took the rule from 16 to 20.
3. Occupying space is not enough: the beneficiary must be able to RECEIVE
   there and be dangerous there. "Close to the ball, so a high chance of
   receiving the pass ... and facing the goal head-on." That is Q = P x G x A.

So R6 = (occupation of the vacated trail, over time) x Q.

Development-set scores (round 1, n=24, the reviewer's corrected labels):
occupation alone 20, Q alone 7, product 21, R1 (distance rule) 20. Neither
factor carries the other; the conjunction is the rule.

Coverage geometry is position+velocity only — the ball is excluded on the
reviewer's instruction, which also sidesteps the Fernández radius shrinking
the ball carrier's zone to its 4 m floor.

The one free constant is the reaction horizon (1.5 s), chosen with the
reviewer; scores decline gently to 18 at 3.0 s, so it is a modelling choice
rather than a plateau. NOT yet validated out of sample.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .vacated_space import rule_r5

HORIZON_SECONDS = 1.5
SAMPLE_STEP_SECONDS = 0.25
# R7 reachability: how fast a candidate could enter the opened space if he
# chose to. Ordinary outfield sprint physics, not fitted to labels.
REACH_MAX_SPEED_MPS = 8.0
REACH_ACCELERATION_MPS2 = 4.0


def rule_r6(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    attacker_at: Callable[[str, float], tuple[tuple[float, float], tuple[float, float]]],
    option_values: Mapping[str, float],
) -> tuple[str, dict[str, float]]:
    """Beneficiary = argmax over attackers of (vacated occupation) x value.

    ``option_values`` maps option id -> Q under the response in which the
    defender follows the runner: the threat that option actually carries if
    the defender commits. Options absent from it cannot be scored (they were
    never priced) and are skipped — a catalogue gap, not a rule decision.
    """
    _, detail = rule_r5(
        state, attacking_team, runner_id, carrier_id, defender_id, samples, attacker_at
    )
    occupation = {k: v for k, v in detail.items() if not k.startswith("_")}
    combined = {
        option_id: occupation.get(option_id, 0.0) * float(value)
        for option_id, value in option_values.items()
        if option_id != runner_id
    }
    if not combined:
        return "", {"_vacated_area": detail.get("_vacated_area", 0.0)}
    best = max(combined, key=lambda key: (combined[key], key))
    combined["_vacated_area"] = detail.get("_vacated_area", 0.0)
    return best, combined


def rule_r7(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    samples: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    option_values: Mapping[str, float],
    horizon_seconds: float = HORIZON_SECONDS,
) -> tuple[str, dict[str, float]]:
    """R7: beneficiary = who COULD exploit the opened space, times value.

    R6 scores attackers on where they actually went, so a team-mate who
    reads the run badly scores zero and the run itself is credited with
    nothing. The reviewer's objection (2026-09-08): "the beneficiary
    candidates moved stupidly ... if both of them were smart enough to make
    the movement that exploits it well, shouldn't the most threatening one
    then be the beneficiary?". The run created the chance; failing to take
    it is a separate fact.

    That also restores symmetry: the defender is already simulated at his
    best available response and Q is already counterfactual ("if the pass
    were made"), so freezing attackers to their observed paths was the odd
    one out.

    Reachability replaces observed occupation: for each candidate, the
    fraction of the vacated region he could occupy within the horizon,
    starting from his onset state with ordinary sprint physics. The
    velocity component toward each cell counts, so momentum still helps —
    but running the wrong way no longer disqualifies him outright.
    """
    del carrier_id
    from .vacated_space import (  # local import keeps the module import-light
        _region_grid,
        _state_v,
        _state_xy,
        coverage_field,
    )
    import numpy as np

    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == attacking_team or player_id == defender_id:
            continue
        np.maximum(
            others,
            coverage_field(_state_xy(row), _state_v(row), xs, ys),
            out=others,
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
    total = float(trail.sum())
    if total <= 1e-12:
        return "", {"_vacated_area": 0.0}

    reach: dict[str, float] = {}
    for player_id, row in state.items():
        if str(row["team"]) != attacking_team or player_id == runner_id:
            continue
        px, py = _state_xy(row)
        vx, vy = _state_v(row)
        dx = xs[None, :] - px
        dy = ys[:, None] - py
        distance = np.hypot(dx, dy)
        safe = np.maximum(distance, 1e-9)
        toward = np.clip(
            (dx * vx + dy * vy) / safe, 0.0, REACH_MAX_SPEED_MPS
        )
        to_top = np.maximum(REACH_MAX_SPEED_MPS - toward, 0.0) / REACH_ACCELERATION_MPS2
        during = toward * to_top + 0.5 * REACH_ACCELERATION_MPS2 * to_top**2
        times = np.where(
            distance <= during,
            (
                np.sqrt(
                    np.maximum(toward**2 + 2 * REACH_ACCELERATION_MPS2 * distance, 0.0)
                )
                - toward
            )
            / REACH_ACCELERATION_MPS2,
            to_top + (distance - during) / REACH_MAX_SPEED_MPS,
        )
        # Share of the opened space he could stand in before the window ends.
        reach[player_id] = float(
            (trail * (times <= horizon_seconds)).sum() / total
        )

    combined = {
        option_id: reach.get(option_id, 0.0) * float(value)
        for option_id, value in option_values.items()
        if option_id != runner_id
    }
    if not combined:
        return "", {"_vacated_area": total}
    best = max(combined, key=lambda key: (combined[key], key))
    combined["_vacated_area"] = total
    return best, combined
