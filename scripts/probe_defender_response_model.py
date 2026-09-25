"""Which defender response should the vacated space be computed under?

Two questions at once, both raised by reading the code rather than the
scores.

FIRST, the reviewer's own (2026-09-08): "공간 점유 계산할 때, 수비수인
Tanaka가 러너를 따라간다고 하고 계산하는거 맞지?" Yes — R9 uses the
`focus_runner` target-conditioned baseline, a purely geometric goal-side
pursuit of the runner. That is a counterfactual: it asks what space would
open IF the defender committed to the runner, not what actually opened. The
audits also carry the defender's OBSERVED path (`actual_reference`) and the
model's own preferred response, so the choice can be measured instead of
assumed.

SECOND, an inconsistency of exactly the kind that produced R9. The space
term is computed along the focus_runner path, but Q is read from the cells
of `direct_best_response_id` — a DIFFERENT response. So the rule already
prices "what is the threat if the defender plays his best response" against
"what space opens if he instead chases the runner". Pairing each path with
its own cells makes the question self-consistent.

Round 1 only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def find_response(defender, selector):
    """selector: 'focus_runner' | 'actual' | 'direct_best'."""
    if selector == "direct_best":
        wanted = str(defender.get("direct_best_response_id"))
        return next(
            (r for r in defender["responses"] if str(r["response_id"]) == wanted), None
        )
    kind = {
        "focus_runner": "target_conditioned_baseline",
        "actual": "actual_reference",
    }[selector]
    return next((r for r in defender["responses"] if r.get("kind") == kind), None)


def curves_of(response, runner):
    return {
        option_id: value_at_times(cell.get("candidate_grid") or [])
        for option_id, cell in response["cells"].items()
        if cell.get("legal") is not False
        and cell.get("q") is not None
        and option_id != runner
    }


def predict(state, team, runner, defender_id, defender_at, attacker_at, curves, horizon):
    xs, ys = _region_grid(state[defender_id])
    others = np.zeros((len(ys), len(xs)))
    for player_id, row in state.items():
        if str(row["team"]) == team or player_id == defender_id:
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
    totals = {option_id: 0.0 for option_id in curves}
    vacated = 0.0
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        after_xy, after_v = defender_at(time_s)
        loss = np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
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
                    float((loss * coverage_field(xy, velocity, xs, ys)).sum()) * value
                )
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    if vacated <= 1e-12 or not totals:
        return ""
    return max(totals, key=lambda key: (totals[key], key))


VARIANTS = [
    ("focus_runner", "direct_best", "현행 R9 (공간=러너추적, Q=최적응답)"),
    ("focus_runner", "focus_runner", "자기일관 반사실 (둘 다 러너추적)"),
    ("actual", "direct_best", "공간=실제움직임, Q=최적응답"),
    ("actual", "actual", "자기일관 서술 (둘 다 실제움직임)"),
    ("direct_best", "direct_best", "자기일관 모델최적 (둘 다 최적응답)"),
]


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

    hits = {label: 0 for _, _, label in VARIANTS}
    total = 0
    per_scene = []
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
        _, attacker_at = _coupled.absolute_lookups(scene, defender)
        if attacker_at is None:
            continue
        team = attacking_team_id(scene)
        runner = str(scene["runner_id"])
        horizon = float(scene["horizon_seconds"])

        row_result = {}
        usable = True
        for path_key, value_key, label in VARIANTS:
            path_response = find_response(defender, path_key)
            value_response = find_response(defender, value_key)
            if path_response is None or value_response is None:
                usable = False
                break
            defender_at = _coupled._path_lookup(
                [
                    (float(p[0]), float(p[1]), float(p[2]))
                    for p in path_response["path_txy"]
                ]
            )
            curves = curves_of(value_response, runner)
            if not curves:
                usable = False
                break
            row_result[label] = predict(
                state, team, runner, defender_id, defender_at, attacker_at,
                curves, horizon,
            )
        if not usable:
            continue

        total += 1
        for label, pick in row_result.items():
            hits[label] += pick == answer
        per_scene.append((frame_id, str(defender["defender_name"]), answer, row_result))

    print(f"라운드 1  n = {total}\n")
    for _, _, label in VARIANTS:
        print(f"  {hits[label]:>3}   {label}")

    print("\n현행과 갈리는 장면:")
    current = VARIANTS[0][2]
    for frame_id, name, answer, picks in per_scene:
        differing = {
            label: pick for label, pick in picks.items() if (pick == answer) != (picks[current] == answer)
        }
        if differing:
            marks = " ".join(
                f"{label.split('(')[0].strip()}={'O' if pick == answer else 'X'}"
                for label, pick in differing.items()
            )
            print(f"  {frame_id}/{name:<20} 현행={'O' if picks[current] == answer else 'X'}  {marks}")


if __name__ == "__main__":
    main()
