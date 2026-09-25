#!/usr/bin/env python3
"""Build a dynamic two-option local defender-response game for scene 1.

The two initial rows are the observed Klaus runner trajectory and the v0.5
counterfactual Kownacki carry trajectory.  Both are evaluated against the same
feasible defender paths using the moving goal-side marking geometry.  The
matrix wrapper is value-model agnostic so P/G/A can replace this diagnostic
cost later without changing the selection contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from evaluate_conditional_defender_game_v0_3 import (
    _reachability_config,
    _response_payload,
    _unique_endpoint_actions,
)
from evaluate_dynamic_carry_response_v0_6 import (
    _find_source_scene,
    _path_txy,
    _response_with_path,
    _summary_payload,
)
from offball_value.bundesliga import (
    FIELD_LENGTH,
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.dynamic_marking import (
    DynamicMarkingConfig,
    summarize_dynamic_marking,
)
from offball_value.dynamic_response_game import select_dynamic_game_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "data/processed/goal_side_accessibility_v0_5/"
            "goal_side_accessibility_v0_5.json"
        ),
    )
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=Path(
            "data/processed/direct_derived_response_map_v0_1/"
            "direct_derived_response_maps.json"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dynamic_local_game_v0_7"),
    )
    parser.add_argument("--goal-side-offset-m", type=float, default=1.5)
    parser.add_argument("--lookahead-seconds", type=float, default=0.4)
    parser.add_argument("--wrong-side-weight", type=float, default=2.0)
    return parser.parse_args()


def _mode_payload(
    key,
    label,
    selection_rule,
    response,
    option_paths,
    goal_xy,
    config,
    evaluation_times,
    cost_minimum,
    cost_spread,
    feasible_cost_matrix,
    pareto_indices,
):
    summaries = [
        summarize_dynamic_marking(
            option["path_txy"],
            _path_txy(response),
            goal_xy,
            config,
            evaluation_times,
        )
        for option in option_paths
    ]
    costs = np.asarray(
        [summary.mean_weighted_error_m for summary in summaries], dtype=float
    )
    regret = np.where(
        cost_spread > 1e-12,
        (costs - cost_minimum) / cost_spread,
        0.0,
    )
    response_index = response.get("index")
    ranks = []
    for option_index, cost in enumerate(costs):
        ranks.append(
            1 + int(np.sum(feasible_cost_matrix[option_index] < cost - 1e-9))
        )
    return {
        "key": key,
        "label": label,
        "selection_rule": selection_rule,
        "response": _response_with_path(response),
        "is_feasible_candidate": response_index is not None and int(response_index) >= 0,
        "is_pareto": (
            int(response_index) in pareto_indices
            if response_index is not None and int(response_index) >= 0
            else False
        ),
        "option_costs": [float(value) for value in costs],
        "option_regrets": [float(value) for value in regret],
        "option_ranks": ranks,
        "raw_worst_option_index": int(np.argmax(costs)),
        "regret_worst_option_index": int(np.argmax(regret)),
        "summaries": [_summary_payload(summary) for summary in summaries],
    }


def evaluate(args: argparse.Namespace) -> dict:
    result = json.loads(args.input_json.read_text(encoding="utf-8"))
    scenes = json.loads(args.scene_json.read_text(encoding="utf-8"))
    _find_source_scene(scenes, result)  # Fail early if the source scene changed.
    onset = int(result["onset_frame_id"])
    horizon_s = float(result["horizon_seconds"])
    files = find_bundesliga_files(args.data_dir, result["match_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(onset - 10, onset + int(round(horizon_s * FPS)) + 1)
    )
    history = tuple(frames[index] for index in range(onset - 10, onset + 1))
    decision = frames[onset]
    scenario = result["scenarios"]["actual_run"]
    config = DynamicMarkingConfig(
        goal_side_offset_m=float(args.goal_side_offset_m),
        lookahead_seconds=float(args.lookahead_seconds),
        response_delay_seconds=0.2,
        wrong_side_weight=float(args.wrong_side_weight),
    )
    goal_xy = (float(result["attacking_direction"]) * FIELD_LENGTH / 2.0, 0.0)
    runner_path = tuple(
        (float(time_s), float(x), float(y))
        for time_s, x, y in scenario["runner_path_txy"]
    )
    branches = []
    for branch in scenario["branches"]:
        owner_option = next(
            option for option in branch["options"] if option["is_ball_owner"]
        )
        carry_path = tuple(
            (float(time_s), float(x), float(y))
            for time_s, x, y in owner_option["carry_path_txy"]
        )
        option_paths = (
            {
                "option_id": "runner_direct",
                "player_id": result["runner_id"],
                "player_name": result["runner_name"],
                "label": f"{result['runner_name']} 직접 옵션",
                "option_type": "runner",
                "path_txy": runner_path,
            },
            {
                "option_id": "ball_owner_carry",
                "player_id": result["ball_owner_id"],
                "player_name": result["ball_owner_name"],
                "label": f"{result['ball_owner_name']} carry 옵션",
                "option_type": "carry",
                "path_txy": carry_path,
            },
        )
        evaluation_times = tuple(
            time_s for time_s, _, _ in carry_path if time_s >= 0.2 - 1e-9
        )
        generated = generate_defender_response_actions(
            decision,
            history,
            str(branch["defender_id"]),
            DefenderResponseConfig(
                horizon_seconds=horizon_s,
                response_delay_seconds=0.2,
            ),
            _reachability_config(),
        )
        responses = tuple(
            _response_payload(action, index)
            for index, action in enumerate(
                _unique_endpoint_actions(generated.actions)
            )
        )
        summaries = [[None for _ in responses] for _ in option_paths]
        costs = np.zeros((len(option_paths), len(responses)), dtype=float)
        for option_index, option in enumerate(option_paths):
            for response_index, response in enumerate(responses):
                summary = summarize_dynamic_marking(
                    option["path_txy"],
                    _path_txy(response),
                    goal_xy,
                    config,
                    evaluation_times,
                )
                summaries[option_index][response_index] = summary
                costs[option_index, response_index] = summary.mean_weighted_error_m
        efforts = tuple(float(response["effort_m2ps3"]) for response in responses)
        selected = select_dynamic_game_responses(costs, efforts)
        cost_minimum = np.min(costs, axis=1)
        cost_spread = np.ptp(costs, axis=1)
        response_by_index = {
            int(response["index"]): response for response in responses
        }
        modes = [
            _mode_payload(
                "actual",
                "Actual defense",
                "observed defender trajectory",
                branch["actual_response"],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "existing_runner",
                "기존 Runner 대응",
                "v0.5 goal-side runner-follow anchor",
                branch["runner_follow_response"],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "existing_endpoint",
                "기존 carry endpoint 억제",
                "v0.5 fixed carry endpoint minimum",
                owner_option["cover_response"],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "dynamic_runner",
                "Dynamic Runner 억제",
                "minimum full-trajectory runner marking cost",
                responses[selected.option_minimum_indices[0]],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "dynamic_carry",
                "Dynamic carry 억제",
                "minimum full-trajectory carry marking cost",
                responses[selected.option_minimum_indices[1]],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "dynamic_local_minimax",
                "Dynamic local minimax",
                "minimize the larger raw marking cost across the two local options",
                responses[selected.local_minimax_index],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
            _mode_payload(
                "dynamic_compromise",
                "Dynamic compromise",
                "minimize the larger normalized regret from each option-specific minimum",
                responses[selected.compromise_index],
                option_paths,
                goal_xy,
                config,
                evaluation_times,
                cost_minimum,
                cost_spread,
                costs,
                selected.pareto_indices,
            ),
        ]
        response_points = []
        pareto_set = set(selected.pareto_indices)
        for response_index, response in enumerate(responses):
            response_points.append(
                {
                    "response_index": int(response["index"]),
                    "runner_cost": float(costs[0, response_index]),
                    "carry_cost": float(costs[1, response_index]),
                    "runner_regret": float(
                        selected.normalized_regret[0, response_index]
                    ),
                    "carry_regret": float(
                        selected.normalized_regret[1, response_index]
                    ),
                    "effort_m2ps3": float(response["effort_m2ps3"]),
                    "is_pareto": response_index in pareto_set,
                }
            )
        branches.append(
            {
                "defender_id": branch["defender_id"],
                "defender_name": branch["defender_name"],
                "defender_position": branch["defender_position"],
                "feasible_response_count": len(responses),
                "pareto_response_count": len(selected.pareto_indices),
                "options": [
                    {
                        **{key: value for key, value in option.items() if key != "path_txy"},
                        "path_txy": [list(sample) for sample in option["path_txy"]],
                        "minimum_cost": float(cost_minimum[index]),
                        "maximum_cost": float(cost_minimum[index] + cost_spread[index]),
                    }
                    for index, option in enumerate(option_paths)
                ],
                "modes": modes,
                "response_points": response_points,
            }
        )
    return {
        "schema_version": "dynamic-local-game-v0.7",
        "status": "two-option dynamic geometry game; value model pending",
        "match_id": result["match_id"],
        "match_label": result["match_label"],
        "onset_frame_id": onset,
        "horizon_seconds": horizon_s,
        "attacking_team_id": result["attacking_team_id"],
        "attacking_direction": result["attacking_direction"],
        "goal_xy": list(goal_xy),
        "runner_id": result["runner_id"],
        "runner_name": result["runner_name"],
        "ball_owner_id": result["ball_owner_id"],
        "ball_owner_name": result["ball_owner_name"],
        "config": asdict(config),
        "branches": branches,
        "background_frames": result["background_frames"],
        "names": result["names"],
        "notes": [
            "Every response is evaluated against the same runner and carry trajectories over the full horizon.",
            "Dynamic local minimax uses raw geometry cost; compromise uses option-normalized regret.",
            "The higher-cost or higher-regret row is the attacker's current re-selection in this two-option diagnostic.",
            "Pass-lane, delivery, G, and post-reception A are intentionally not used yet.",
        ],
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dynamic_local_game_v0_7.json"
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "branch_count": len(result["branches"]),
                "config": result["config"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output: {output.resolve()}")
    for branch in result["branches"]:
        print(
            f"{branch['defender_name']}: {branch['feasible_response_count']} paths, "
            f"{branch['pareto_response_count']} Pareto"
        )
        for mode in branch["modes"][3:]:
            print(
                f"  {mode['label']}: runner={mode['option_costs'][0]:.2f}m "
                f"carry={mode['option_costs'][1]:.2f}m "
                f"regret=({mode['option_regrets'][0]:.2f},"
                f"{mode['option_regrets'][1]:.2f})"
            )


if __name__ == "__main__":
    main()
