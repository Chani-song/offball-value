"""Export the coupled rule's misses in the miss-review renderer's format.

Same job as the earlier R3/R5/R6 miss exports, but the candidate table now
shows what the coupled rule actually maximises: for each attacker, the
instant at which "standing in the opened space" and "a dangerous ball can
arrive here" line up best, and the two factors at that instant. That is the
column the reviewer needs in order to say whether the rule is wrong or the
scene is genuinely ambiguous.

Usage:
    python scripts/export_coupled_misses.py <audit_dir> [...] \
        --out misses.json [--frame 104717 --frame 13749]
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
    rule_r9,
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


def vacated_cells(state, team, defender_id, defender_at, horizon):
    """Grid cells the defender gives up, summed over the reaction, for display."""
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
    accumulated = np.zeros_like(before)
    time_s = SAMPLE_STEP_SECONDS
    while time_s <= horizon + 1e-9:
        after_xy, after_v = defender_at(time_s)
        accumulated += np.clip(
            before - np.maximum(others, coverage_field(after_xy, after_v, xs, ys)),
            0.0,
            None,
        )
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    peak = float(accumulated.max())
    if peak <= 1e-12:
        return []
    cells = []
    for row_index, y in enumerate(ys):
        for column, x in enumerate(xs):
            value = float(accumulated[row_index, column]) / peak
            if value > 0.06:
                cells.append({"x": round(float(x), 2), "y": round(float(y), 2), "v": round(value, 3)})
    return cells


def best_instant(state, team, defender_id, defender_at, attacker_at, curves, horizon):
    """Per option: the instant maximising occupation x Q, and both factors there."""
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
    peak: dict[str, tuple[float, float, float, float]] = {}
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
            for option_id, curve in curves.items():
                value = value_near(curve, time_s)
                xy, velocity = attacker_at(option_id, time_s)
                occupation = float(
                    (loss * coverage_field(xy, velocity, xs, ys)).sum() / weight
                )
                product = occupation * value
                if product > peak.get(option_id, (-1.0,))[0]:
                    peak[option_id] = (product, occupation, value, time_s)
        time_s = round(time_s + SAMPLE_STEP_SECONDS, 6)
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", action="append", default=[])
    options = parser.parse_args()

    labels = _coupled._scorer.load_labels()
    scenes: dict[tuple[str, str], dict] = {}
    for directory in options.audit_dirs:
        for scene in json.loads(
            (Path(directory) / "local_game_payoff_audits.json").read_text()
        ):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    payload = []
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

        team = attacking_team_id(scene)
        runner = str(scene["runner_id"])
        carrier = str(scene["carrier_id"])
        horizon = float(scene["horizon_seconds"])
        priced = {
            option_id: cell
            for option_id, cell in response["cells"].items()
            if cell.get("legal") is not False
            and cell.get("q") is not None
            and option_id != runner
        }
        curves = {
            option_id: value_at_times(cell.get("candidate_grid") or [])
            for option_id, cell in priced.items()
        }
        pick, _ = rule_r9(
            state, team, runner, carrier, defender_id,
            defender_at, attacker_at, curves, horizon,
        )
        if pick == answer:
            continue
        if options.frame and frame_id not in options.frame:
            continue

        names = {str(o["option_id"]): str(o["option_name"]) for o in defender["options"]}
        for row in scene["background_frames"][0]["players"]:
            names.setdefault(str(row[0]), str(row[4]))
        peak = best_instant(
            state, team, defender_id, defender_at, attacker_at, curves, horizon
        )
        candidates = []
        for option_id, (product, occupation, value, at_time) in sorted(
            peak.items(), key=lambda item: -item[1][0]
        ):
            cell = priced[option_id]
            candidates.append(
                {
                    "name": f"{names.get(option_id, option_id)}  ({at_time:.2f}초)",
                    "occ": f"{occupation:.3f}",
                    "P": (
                        f"{float(cell['delivery']):.2f}"
                        if cell.get("delivery") is not None
                        else None
                    ),
                    "G": (
                        f"{float(cell['goal']):.3f}"
                        if cell.get("goal") is not None
                        else None
                    ),
                    "score": f"{product:.4f}",
                    "is_answer": option_id == answer,
                    "is_pick": option_id == pick,
                    "is_carrier": option_id == carrier,
                }
            )

        payload.append(
            {
                "scene": f"{scene['match_label']} · {frame_id}",
                "direction": int(scene.get("attacking_direction") or 1),
                "attacking_team_id": team,
                "runner": str(scene["runner_name"]),
                "runner_id": runner,
                "carrier": str(scene["carrier_name"]),
                "carrier_id": carrier,
                "defender": str(defender["defender_name"]),
                "defender_id": defender_id,
                "answer": names.get(answer, answer),
                "answer_id": answer,
                "pick": names.get(pick, pick or "(없음)"),
                "pick_id": pick,
                "candidates": candidates,
                "cells": vacated_cells(state, team, defender_id, defender_at, horizon),
                "defender_path": [
                    [round(float(p[0]), 3), round(float(p[1]), 2), round(float(p[2]), 2)]
                    for p in next(
                        row["path_txy"]
                        for row in defender["responses"]
                        if row.get("kind") == "target_conditioned_baseline"
                    )
                ],
                "frames": [
                    {
                        "t": round(float(frame["relative_time_s"]), 3),
                        "players": frame["players"],
                        "ball": frame.get("ball"),
                    }
                    for frame in sorted(
                        scene["background_frames"],
                        key=lambda f: float(f["relative_time_s"]),
                    )
                ],
            }
        )

    options.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"오답 {len(payload)}개 -> {options.out}")
    for scene in payload:
        print(f"  {scene['scene']} · {scene['defender']}: 정답 {scene['answer']} / 규칙 {scene['pick']}")


if __name__ == "__main__":
    main()
