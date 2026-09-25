"""Does R9 need the defender's PATH, or only the fact that he leaves?

Five different defender responses — his observed movement, a geometric
pursuit of the runner, the model's own best response, each paired with its
own Q or with another's — all pick the same beneficiary in all 24 round-1
games, even though the observed and pursuit paths separate by 4.4 m on
average and 9.0 m at the endpoint.

That is either robustness or a warning. If where the defender goes never
matters, the rule may not be measuring "the space he was dragged away from"
at all; it may just be measuring "who can receive dangerously near the zone
he started in". Those are different claims, and the project's causal story
depends on which one is true.

So this probe replaces the path with degenerate cases:

  vanish   his coverage is removed entirely from t=0 — he leaves, direction
           unspecified. If this also scores 24, the path carries nothing.
  frozen   he never moves — nothing is vacated at all. The floor.
  reverse  he moves directly AWAY from the runner, the same distance his
           pursuit would have covered. If this also scores 24, the rule is
           not about the runner either.

Round 1 only.
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
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    value_at_times,
    value_near,
)
from offball_value.vacated_space import (  # noqa: E402
    _region_grid,
    _state_v,
    _state_xy,
    coverage_field,
)

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)


def predict(state, team, runner, defender_id, mode, defender_at, attacker_at,
            curves, horizon):
    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
            continue
        np.maximum(
            others, coverage_field(_state_xy(row), _state_v(row), xs, ys), out=others
        )
    start_xy = _state_xy(state[defender_id])
    start_v = _state_v(state[defender_id])
    before = np.maximum(others, coverage_field(start_xy, start_v, xs, ys))

    totals = {option_id: 0.0 for option_id in curves}
    vacated = 0.0
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        if mode == "vanish":
            after = others
        elif mode == "frozen":
            after = np.maximum(others, coverage_field(start_xy, start_v, xs, ys))
        else:
            after_xy, after_v = defender_at(time_s)
            after = np.maximum(others, coverage_field(after_xy, after_v, xs, ys))
        loss = np.clip(before - after, 0.0, None)
        weight = float(loss.sum())
        if weight > 1e-12:
            vacated += weight
            for option_id, curve in curves.items():
                value = value_near(curve, time_s)
                if value <= 0.0:
                    continue
                xy, velocity = attacker_at(option_id, time_s)
                totals[option_id] += (
                    float((loss * coverage_field(xy, velocity, xs, ys)).sum()) * value
                )
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    if vacated <= 1e-12 or not totals:
        return ""
    return max(totals, key=lambda key: (totals[key], key))


def reversed_lookup(start_xy, start_v, runner_xy, pursuit):
    """Mirror the pursuit displacement about the defender's start.

    Same speed and same distance travelled, opposite direction — so the
    difference from the pursuit case is direction alone, not effort.
    """
    del start_v, runner_xy

    def at(time_s: float):
        (px, py), (vx, vy) = pursuit(time_s)
        return (
            (2 * start_xy[0] - px, 2 * start_xy[1] - py),
            (-vx, -vy),
        )

    return at


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

    modes = ["러너추적 (현행 R9)", "수비수 소멸", "수비수 정지", "러너 반대로"]
    hits = {mode: 0 for mode in modes}
    total = 0
    splits = []
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
        total += 1
        team = attacking_team_id(scene)
        horizon = float(scene["horizon_seconds"])
        start_xy = _state_xy(state[defender_id])
        away = reversed_lookup(start_xy, _state_v(state[defender_id]), None, defender_at)

        picks = {}
        for mode, mode_key, lookup in (
            (modes[0], "path", defender_at),
            (modes[1], "vanish", defender_at),
            (modes[2], "frozen", defender_at),
            (modes[3], "path", away),
        ):
            picks[mode] = predict(
                state, team, runner, defender_id, mode_key, lookup,
                attacker_at, curves, horizon,
            )
            hits[mode] += picks[mode] == answer
        names = {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]}
        splits.append((frame_id, str(defender["defender_name"]), answer, picks, names))

    print(f"n = {total}\n")
    for mode in modes:
        print(f"  {hits[mode]:>3}   {mode}")

    print("\n현행과 답이 갈리는 장면:")
    any_split = False
    for frame_id, name, answer, picks, names in splits:
        others = {m: p for m, p in picks.items() if p != picks[modes[0]]}
        if others:
            any_split = True
            detail = "  ".join(
                f"{m}={names.get(p, p or '없음')}" for m, p in others.items()
            )
            print(
                f"  {frame_id}/{name:<20} 정답={names.get(answer, answer):<18} "
                f"현행={names.get(picks[modes[0]], picks[modes[0]])}  |  {detail}"
            )
    if not any_split:
        print("  (없음 — 모든 변형이 모든 장면에서 같은 답)")


if __name__ == "__main__":
    main()
