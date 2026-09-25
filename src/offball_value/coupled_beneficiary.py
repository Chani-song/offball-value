"""R9: occupy the space AT the moment the ball can arrive there.

R6 multiplies two quantities computed on separate timelines:

- occupation: the area-weighted MEAN, over a fixed 1.5 s window, of how much
  of the vacated space the attacker covers;
- Q: the MAXIMUM, over release times up to the scene horizon (~2.8 s), of
  the threat if the ball were played to him.

Nothing forces those to refer to the same instant. A player can top the
occupation term at 0.5 s while his best Q sits at 2.9 s, by which time he
may have left the space entirely. The reviewer's objection (2026-09-09):

  "공간점유가 최적인 시점이랑 패스를 받을 수 있는 시점이 일치해야 하는건데,
   그게 따로 돌아간다는거지? 그러면 안되잖아."

R9 forms the product inside the time loop instead:

    score(i) = Σ_t |loss(t)| · occupation_i(t) · Q_i(t)  /  Σ_t |loss(t)|

so the rule asks one question — *is this player standing in the opened space
at a moment when a dangerous ball can arrive there?* — rather than two
unrelated ones. Q_i(t) is read from the option's own priced candidate grid,
so no new value computation is involved; the multiplication simply moved.

The reaction window is the scene's own horizon rather than 1.5 s, because a
clock you truncate at 1.5 s has almost no Q left to couple to: the coupled
rule scores 22 at a 1.5 s cut and 24 from 2.0 s on. That removes R6's one
free constant.

WHAT THE VALIDATION DOES AND DOES NOT SHOW
Round 1 (development, n=24): R6 23, R9 24. Round 2 (out of sample, n=21
after excluding the contaminated 57121): R6 18, R9 19; R6-only 0, R9-only 1,
McNemar p = 1.0. Both rounds are +1 with nothing lost, which is NOT evidence
of better accuracy at these sample sizes. R9 is adopted for structure — one
fewer free constant, one coherent question — and the accuracy claim is
explicitly not made. See docs/round2_coupled_clock_prediction_sealed.md.

Known limitation: the coupled gain appears only with observed attacker
trajectories over the full horizon. Substituting constant-velocity or frozen
motion removes it (+0, -1), and R9 reads about twice as much real future as
R6 did. The control that keeps this honest is that leaving the window long
while switching Q back to its decoupled form scores 23, not 24 — so the
extra future alone does not produce the difference.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .vacated_space import _region_grid, _state_v, _state_xy, coverage_field

SAMPLE_STEP_SECONDS = 0.25
# How far from an instant a priced pass still counts as arriving "then". Not
# a tuned quantity: the round-1 score is 24 for every value from 0.15 s to
# 1.5 s, so the rule is flat in it. A pass cannot be conjured at an arbitrary
# instant either — it has to be one the pricing stage found feasible — which
# is why this is a lookup window rather than an interpolation.
VALUE_WINDOW_SECONDS = 0.4


def value_at_times(
    candidate_grid: Sequence[Mapping[str, object]],
) -> list[tuple[float, float]]:
    """(arrival time, best Q priced for that arrival), ascending, from one cell."""
    best: dict[float, float] = {}
    for row in candidate_grid or ():
        time_s = round(float(row["event_time"]), 3)
        value = float(row["q"])
        if value > best.get(time_s, -1.0):
            best[time_s] = value
    return sorted(best.items())


def value_near(
    curve: Sequence[tuple[float, float]],
    time_s: float,
    window_s: float = VALUE_WINDOW_SECONDS,
) -> float:
    """Best Q the pricing stage found for a ball arriving around ``time_s``."""
    if not curve:
        return 0.0
    nearby = [value for t, value in curve if abs(t - time_s) <= window_s]
    if nearby:
        return max(nearby)
    nearest = min(curve, key=lambda item: abs(item[0] - time_s))
    return 0.0 if abs(nearest[0] - time_s) > 1.0 else nearest[1]


def rule_r9(
    state: Mapping[str, Mapping[str, object]],
    attacking_team: str,
    runner_id: str,
    carrier_id: str,
    defender_id: str,
    defender_at: Callable[[float], tuple[tuple[float, float], tuple[float, float]]],
    attacker_at: Callable[[str, float], tuple[tuple[float, float], tuple[float, float]]],
    value_curves: Mapping[str, Sequence[tuple[float, float]]],
    horizon_seconds: float,
    step_seconds: float = SAMPLE_STEP_SECONDS,
    value_window_s: float = VALUE_WINDOW_SECONDS,
) -> tuple[str, dict[str, float]]:
    """Beneficiary = argmax of space-and-threat evaluated at the same instant.

    ``defender_at(t)`` and ``attacker_at(player_id, t)`` both take ABSOLUTE
    seconds from the run onset — the coupling is meaningless if the two sides
    are indexed differently, which is exactly the off-by-one that inflated
    R6's first round-2 score by one scene.

    ``value_curves`` maps an option id to its (arrival time, Q) curve under
    the response in which the defender follows the runner. Options absent
    from it were never priced and are skipped: a catalogue gap, not a
    decision by the rule.
    """
    del carrier_id
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

    options = [option_id for option_id in sorted(value_curves) if option_id != runner_id]
    if not options:
        return "", {"_vacated_area": 0.0}

    totals = {option_id: 0.0 for option_id in options}
    vacated_total = 0.0
    time_s = step_seconds
    while time_s <= horizon_seconds + 1e-9:
        after_xy, after_v = defender_at(time_s)
        # Space the defender has given up at THIS instant. Instantaneous, as
        # in rule_r5 — the accumulation is what the sum over instants does.
        loss = np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        weight = float(loss.sum())
        if weight > 1e-12:
            vacated_total += weight
            for option_id in options:
                value = value_near(value_curves[option_id], time_s, value_window_s)
                if value <= 0.0:
                    continue
                xy, velocity = attacker_at(option_id, time_s)
                totals[option_id] += (
                    float((loss * coverage_field(xy, velocity, xs, ys)).sum()) * value
                )
        time_s = round(time_s + step_seconds, 6)

    if vacated_total <= 1e-12:
        return "", {"_vacated_area": 0.0}

    detail = {
        option_id: totals[option_id] / vacated_total for option_id in options
    }
    best = max(detail, key=lambda key: (detail[key], key))
    detail["_vacated_area"] = vacated_total
    return best, detail
