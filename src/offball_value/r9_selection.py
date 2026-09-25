"""Pick the beneficiary with R9, from inside the payoff build.

R9 scored 24/24 on round 1 and 19/21 on round 2; R1, which the pipeline has
been shipping as `derived_option_source`, scored 20 and 14.5/22. R9 was never
wired in because of an ordering problem rather than a disagreement: R1 needs
only positions, so it can be computed before anything is priced, while R9
needs Q_i(t) and Q does not exist until the responses have been priced.

That ordering is not actually a constraint here. The build prices every
option under every surviving response BEFORE it uses the beneficiary to form
the local pair, so by that point the curves R9 wants already exist. This
module reads them out of the priced responses and calls the same `rule_r9`
that scripts/score_coupled_beneficiary.py validated against the labels.

Two inputs have to come from the same places the scorer took them, or the
number this produces is not the number that was validated:

  value curves   the cells of `direct_best_response_id` -- the response in
                 which the defender follows the runner -- and each cell's
                 `candidate_grid`, which exists only under
                 --record-candidate-grid.
  defender path  the `target_conditioned_baseline` response, NOT the direct
                 best one. The scorer uses the baseline for coverage
                 geometry, and mixing the two silently changes the rule.

Both sides are indexed in absolute seconds from the onset. R9's whole claim
is that occupation and pass arrival are read at the same instant, so an
off-by-one between the two clocks is not a small error -- it is the one that
inflated R6's first round-2 score.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .assignment_rule import attacking_team_id, onset_state
from .coupled_beneficiary import rule_r9, value_at_times

VELOCITY_WINDOW_S = 0.2


def _path_lookup(rows: Sequence[tuple[float, float, float]]) -> Callable:
    """Interpolate a (t, x, y) path, and difference it for a velocity."""

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

    def sampled(time_s: float):
        now = at(time_s)
        before = at(max(0.0, time_s - VELOCITY_WINDOW_S))
        return now, (
            (now[0] - before[0]) / VELOCITY_WINDOW_S,
            (now[1] - before[1]) / VELOCITY_WINDOW_S,
        )

    return sampled


def _attacker_lookup(game: Mapping[str, object]) -> Callable:
    frames = sorted(
        (
            float(frame["relative_time_s"]),
            {str(p[0]): (float(p[2]), float(p[3])) for p in frame["players"]},
        )
        for frame in game["background_frames"]  # type: ignore[index]
    )

    def attacker_at(player_id: str, time_s: float):
        previous = current = frames[0]
        for frame in frames:
            current = frame
            if frame[0] >= time_s:
                break
            previous = frame
        if player_id not in current[1] or player_id not in previous[1]:
            fallback = (
                current[1].get(player_id) or previous[1].get(player_id) or (0.0, 0.0)
            )
            return fallback, (0.0, 0.0)
        if current[0] == previous[0]:
            return current[1][player_id], (0.0, 0.0)
        span = current[0] - previous[0]
        fraction = (time_s - previous[0]) / span
        start, end = previous[1][player_id], current[1][player_id]
        return (
            (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            ),
            ((end[0] - start[0]) / span, (end[1] - start[1]) / span),
        )

    return attacker_at


def select_r9_beneficiary(
    game: Mapping[str, object],
    defender_id: str,
    responses: Sequence[Mapping[str, object]],
    direct_best_response_id: object,
    runner_id: str,
) -> tuple[str | None, dict]:
    """(beneficiary option id, detail) or (None, {}) when R9 cannot be asked.

    Returns None rather than guessing whenever an input the validated scorer
    relied on is absent, so a silently different rule never ships under R9's
    name. The caller falls back to its configured default.
    """
    baseline = next(
        (r for r in responses if r.get("kind") == "target_conditioned_baseline"),
        None,
    )
    if baseline is None or not baseline.get("path_txy"):
        return None, {}
    direct = next(
        (r for r in responses
         if str(r.get("response_id")) == str(direct_best_response_id)),
        None,
    )
    if direct is None:
        return None, {}

    priced = {
        option_id: cell
        for option_id, cell in (direct.get("cells") or {}).items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and str(option_id) != str(runner_id)
    }
    curves = {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in priced.items()
    }
    curves = {k: v for k, v in curves.items() if v}
    if not curves:
        return None, {}

    state = onset_state(game)
    if defender_id not in state:
        return None, {}

    defender_at = _path_lookup(
        [(float(p[0]), float(p[1]), float(p[2])) for p in baseline["path_txy"]]
    )
    picked, detail = rule_r9(
        state,
        attacking_team_id(game),
        str(runner_id),
        str(game["carrier_id"]),
        str(defender_id),
        defender_at,
        _attacker_lookup(game),
        curves,
        float(game["horizon_seconds"]),
    )
    if not picked:
        return None, {}
    return str(picked), dict(detail)
