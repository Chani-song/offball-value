"""Is R9 flat in the constants nobody derived, or balanced on them?

R9's headline is that it removed R6's free constant (T = 1.5 s). That claim
is only worth anything if the constants that REMAIN are inert. Three were
inherited from the vacated-space module and have never been varied:

  coverage radius     10 m  — the Fernández ellipse size with the ball removed
  region half-width   25 m  — how far around the defender the grid extends
  sample step        0.25 s — how finely the reaction is sampled

None was chosen by fitting, but "not fitted" and "does not matter" are
different claims, and only the second one licenses calling R9 constant-free.

Round 1 only. A rule that holds across a plateau is a rule; a rule that
peaks at exactly the inherited value is a coincidence worth knowing about.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import value_at_times, value_near  # noqa: E402
from offball_value.vacated_space import (  # noqa: E402
    _state_v,
    _state_xy,
    coverage_field,
)

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)


def grid(defender_row, half_width_m: float, step_m: float):
    start = _state_xy(defender_row)
    return (
        np.arange(start[0] - half_width_m, start[0] + half_width_m + 1e-9, step_m),
        np.arange(start[1] - half_width_m, start[1] + half_width_m + 1e-9, step_m),
    )


def predict(game, *, radius_m, half_width_m, grid_step_m, sample_step_s):
    state, team, runner, defender_id, defender_at, attacker_at, curves, horizon = game
    xs, ys = grid(state[defender_id], half_width_m, grid_step_m)
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
            continue
        np.maximum(
            others,
            coverage_field(_state_xy(row), _state_v(row), xs, ys, radius_m),
            out=others,
        )
    before = np.maximum(
        others,
        coverage_field(
            _state_xy(state[defender_id]), _state_v(state[defender_id]), xs, ys, radius_m
        ),
    )

    totals = {option_id: 0.0 for option_id in curves}
    vacated = 0.0
    time_s = sample_step_s
    while time_s <= horizon + 1e-9:
        after_xy, after_v = defender_at(time_s)
        loss = np.clip(
            before
            - np.maximum(others, coverage_field(after_xy, after_v, xs, ys, radius_m)),
            0.0,
            None,
        )
        weight = float(loss.sum())
        if weight > 1e-12:
            vacated += weight
            for option_id, curve in curves.items():
                value = value_near(curve, time_s)
                if value <= 0.0:
                    continue
                xy, velocity = attacker_at(option_id, time_s)
                totals[option_id] += (
                    float((loss * coverage_field(xy, velocity, xs, ys, radius_m)).sum())
                    * value
                )
        time_s = round(time_s + sample_step_s, 6)

    if vacated <= 1e-12 or not totals:
        return ""
    return max(totals, key=lambda key: (totals[key], key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    options = parser.parse_args()

    labels = _coupled._scorer.load_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    games, answers = [], []
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        state = onset_state(scene)
        if defender_id not in state:
            continue
        defender = next(
            (
                row
                for row in scene["candidate_defenders"]
                if str(row["defender_id"]) == defender_id
            ),
            None,
        )
        if defender is None:
            continue
        response = next(
            (
                row
                for row in defender["responses"]
                if str(row["response_id"]) == str(defender.get("direct_best_response_id"))
            ),
            None,
        )
        if response is None:
            continue
        defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
        if defender_at is None:
            continue
        runner = str(scene["runner_id"])
        curves = {
            option_id: value_at_times(cell.get("candidate_grid") or [])
            for option_id, cell in response["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner
        }
        if not curves:
            continue
        games.append(
            (
                state,
                attacking_team_id(scene),
                runner,
                defender_id,
                defender_at,
                attacker_at,
                curves,
                float(scene["horizon_seconds"]),
            )
        )
        answers.append(answer)

    base = dict(radius_m=10.0, half_width_m=25.0, grid_step_m=1.0, sample_step_s=0.25)

    def score(**overrides):
        settings = {**base, **overrides}
        return sum(
            predict(game, **settings) == answer for game, answer in zip(games, answers)
        )

    print(f"라운드 1  n = {len(games)}   기준 설정 = {base}")
    print(f"기준 점수 {score()}\n")

    sweeps = [
        ("커버리지 반경 (m)", "radius_m", [6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 18.0]),
        ("격자 반폭 (m)", "half_width_m", [12.0, 18.0, 25.0, 32.0, 40.0]),
        ("격자 간격 (m)", "grid_step_m", [0.5, 1.0, 1.5, 2.0]),
        ("샘플 간격 (s)", "sample_step_s", [0.1, 0.2, 0.25, 0.4, 0.5]),
    ]
    for title, key, values in sweeps:
        print(title)
        for value in values:
            mark = "  <-- 현재" if abs(value - base[key]) < 1e-9 else ""
            print(f"  {value:>6}   {score(**{key: value})}{mark}", flush=True)
        print()


if __name__ == "__main__":
    main()
