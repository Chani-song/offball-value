"""R10: does giving attackers trajectory freedom help now that the clock is long?

R8 asked "what is the BEST the attacker could have done from his onset
state" instead of "what did he do", and lost: 18/24 against R6's 23, with a
maximum of 20 at a 0.6 s horizon. The diagnosis was that a 1.5 s window is
long enough for nearly every nearby attacker to reach the opened space, so
once direction stops mattering the rule collapses back to nearest-man.

R9 changed the premise. The window is now the scene's full horizon (~2.8 s)
and the space term is paired with the threat available at the same instant,
so a trajectory is no longer scored on "can you get there" but on "can you
be there WHEN a dangerous ball can arrive". That is a different question
from the one R8 lost, which is why it is worth asking again.

    score(i) = max over feasible paths of
               Σ_t |loss(t)| · occupation_path(t) · Q_i(t) / Σ_t |loss(t)|

KNOWN INCOHERENCE, stated up front: Q_i(t) is priced for where the attacker
ACTUALLY went, so pairing it with a counterfactual path values a pass to a
destination the player no longer occupies. Re-pricing every candidate path
is the honest fix and is out of scope here; this probe therefore only asks
whether trajectory freedom on the SPACE term alone helps. R8 carried the
same flaw, so the comparison against it is at least like-for-like.

Round 1 only. Nothing here is scored on round 2 unless it wins first.
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

from offball_value.attacker_trajectory import _sample  # noqa: E402
from offball_value.coupled_beneficiary import (  # noqa: E402
    SAMPLE_STEP_SECONDS,
    value_at_times,
    value_near,
)
from offball_value.defender_trajectory_search import (  # noqa: E402
    DefenderTrajectorySearchConfig,
    generate_feasible_defender_trajectories,
)
from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
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


def scene_tensors(scene, defender_id, defender_at, horizon):
    """Per-instant vacated region and its area weight, computed once."""
    state = onset_state(scene)
    team = attacking_team_id(scene)
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
    times, losses, weights = [], [], []
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
            times.append(time_s)
            losses.append(loss)
            weights.append(weight)
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    return xs, ys, times, losses, weights


def coupled_score(xs, ys, times, losses, weights, positions, curve, window_s):
    """Σ w·occupation·Q / Σ w for one path already sampled at ``times``."""
    total = float(sum(weights))
    if total <= 1e-12:
        return 0.0
    running = 0.0
    for loss, weight, time_s, (xy, velocity) in zip(losses, weights, times, positions):
        value = value_near(curve, time_s, window_s)
        if value <= 0.0:
            continue
        running += (
            float((loss * coverage_field(xy, velocity, xs, ys)).sum()) * value
        )
    return running / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--representatives", type=int, default=48)
    parser.add_argument("--window", type=float, default=0.4)
    options = parser.parse_args()

    labels = _coupled._scorer.load_labels()
    scenes = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    hits = {"R9 관측 궤적": 0, "R10 최적 궤적": 0}
    total = 0
    rows = []
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
        horizon = float(scene["horizon_seconds"])
        curves = {
            option_id: value_at_times(cell.get("candidate_grid") or [])
            for option_id, cell in response["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner
        }
        if not curves:
            continue
        xs, ys, times, losses, weights = scene_tensors(
            scene, defender_id, defender_at, horizon
        )
        if not times:
            continue
        total += 1

        config = DefenderTrajectorySearchConfig(
            # The generator integrates at 0.1 s and rejects a horizon that is
            # not a whole number of steps; scene horizons are not (2.96 s).
            # Round UP so every sampled instant still falls on the path.
            planning_horizon_seconds=round(math.ceil(horizon * 10.0) / 10.0, 1),
            response_delay_seconds=0.0,
            maximum_representatives=options.representatives,
        )
        observed, achievable = {}, {}
        for option_id, curve in curves.items():
            observed[option_id] = coupled_score(
                xs, ys, times, losses, weights,
                [attacker_at(option_id, t) for t in times],
                curve, options.window,
            )
            row = state.get(option_id)
            if row is None:
                achievable[option_id] = observed[option_id]
                continue
            paths = generate_feasible_defender_trajectories(
                _state_xy(row), _state_v(row), horizon, config
            )
            best = 0.0
            for candidate in paths:
                best = max(
                    best,
                    coupled_score(
                        xs, ys, times, losses, weights,
                        [_sample(candidate.path_txy, t) for t in times],
                        curve, options.window,
                    ),
                )
            achievable[option_id] = best

        names = {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]}
        picks = {}
        for label, scores in (("R9 관측 궤적", observed), ("R10 최적 궤적", achievable)):
            pick = max(scores, key=lambda k: (scores[k], k)) if scores else ""
            picks[label] = pick
            hits[label] += pick == answer
        rows.append(
            (
                frame_id,
                str(defender["defender_name"]),
                names.get(answer, answer),
                {k: names.get(v, v) for k, v in picks.items()},
            )
        )
        print(
            f"  [{total}] {frame_id}/{str(defender['defender_name'])[:14]:<15} "
            f"R9 {'O' if picks['R9 관측 궤적'] == answer else 'X'}  "
            f"R10 {'O' if picks['R10 최적 궤적'] == answer else 'X'}",
            flush=True,
        )

    print(f"\n라운드 1  n = {total}")
    for label, score in hits.items():
        print(f"  {label:<16} {score}")
    print("\n서로 다른 장면:")
    for frame_id, name, answer, picks in rows:
        a = picks["R9 관측 궤적"] == answer
        b = picks["R10 최적 궤적"] == answer
        if a != b:
            print(
                f"  {frame_id}/{name:<18} 정답={answer:<20} "
                f"R9={picks['R9 관측 궤적']:<18} R10={picks['R10 최적 궤적']}"
            )


if __name__ == "__main__":
    main()
