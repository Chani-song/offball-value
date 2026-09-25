#!/usr/bin/env python3
"""Compare endpoint defense with full-trajectory goal-side carry defense.

This is a narrow scene-1 diagnostic.  It keeps the v0.5 Kownacki carry action
fixed, regenerates the same feasible response paths for each nominated
defender, and ranks them against a moving, look-ahead goal-side target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from evaluate_conditional_defender_game_v0_3 import (
    _reachability_config,
    _response_payload,
    _unique_endpoint_actions,
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
    MarkingPathCandidate,
    rank_dynamic_marking_paths,
    summarize_dynamic_marking,
)


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
        default=Path("data/processed/dynamic_carry_response_v0_6"),
    )
    parser.add_argument("--goal-side-offset-m", type=float, default=1.5)
    parser.add_argument("--lookahead-seconds", type=float, default=0.4)
    parser.add_argument("--wrong-side-weight", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _path_txy(response: dict) -> tuple[tuple[float, float, float], ...]:
    if response.get("path_txy"):
        return tuple(
            (float(time_s), float(x), float(y))
            for time_s, x, y in response["path_txy"]
        )
    return tuple(
        (float(time_s), float(point[0]), float(point[1]))
        for time_s, point in zip(response["times_s"], response["path_xy"])
    )


def _response_with_path(response: dict) -> dict:
    return {
        **response,
        "path_txy": [list(sample) for sample in _path_txy(response)],
    }


def _summary_payload(summary) -> dict:
    payload = asdict(summary)
    payload["samples"] = [asdict(sample) for sample in summary.samples]
    return payload


def _find_source_scene(scenes: list[dict], result: dict) -> dict:
    return next(
        scene
        for scene in scenes
        if str(scene["match_id"]) == str(result["match_id"])
        and int(scene["onset_frame_id"]) == int(result["onset_frame_id"])
        and str(scene["runner_id"]) == str(result["runner_id"])
    )


def _mode_payload(
    key: str,
    label: str,
    response: dict,
    actor_path,
    goal_xy,
    config,
    evaluation_times,
    rank_by_identifier,
) -> dict:
    summary = summarize_dynamic_marking(
        actor_path,
        _path_txy(response),
        goal_xy,
        config,
        evaluation_times,
    )
    identifier = str(response.get("index", key))
    return {
        "key": key,
        "label": label,
        "response": _response_with_path(response),
        "dynamic_rank": rank_by_identifier.get(identifier),
        "summary": _summary_payload(summary),
    }


def evaluate(args: argparse.Namespace) -> dict:
    result = json.loads(args.input_json.read_text(encoding="utf-8"))
    scenes = json.loads(args.scene_json.read_text(encoding="utf-8"))
    scene = _find_source_scene(scenes, result)
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
    output_branches = []
    for branch in scenario["branches"]:
        defender_id = str(branch["defender_id"])
        owner_option = next(
            option for option in branch["options"] if option["is_ball_owner"]
        )
        if owner_option["option_type"] != "forward_carry_space":
            raise ValueError("v0.6 diagnostic requires a forward-carry owner option")
        actor_path = tuple(
            (float(time_s), float(x), float(y))
            for time_s, x, y in owner_option["carry_path_txy"]
        )
        evaluation_times = tuple(
            time_s for time_s, _, _ in actor_path if time_s >= 0.2 - 1e-9
        )
        generated = generate_defender_response_actions(
            decision,
            history,
            defender_id,
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
        candidates = tuple(
            MarkingPathCandidate(
                identifier=str(response["index"]),
                path_txy=_path_txy(response),
                effort=float(response["effort_m2ps3"]),
            )
            for response in responses
        )
        ranking = rank_dynamic_marking_paths(
            actor_path,
            candidates,
            goal_xy,
            config,
            evaluation_times,
        )
        dynamic = ranking[0]
        response_by_index = {
            int(response["index"]): response for response in responses
        }
        dynamic_response = response_by_index[int(dynamic.candidate.identifier)]
        rank_by_identifier = {
            row.candidate.identifier: index
            for index, row in enumerate(ranking, start=1)
        }
        modes = [
            _mode_payload(
                "actual",
                "실제 수비",
                branch["actual_response"],
                actor_path,
                goal_xy,
                config,
                evaluation_times,
                rank_by_identifier,
            ),
            _mode_payload(
                "endpoint_cover",
                "기존 endpoint 억제",
                owner_option["cover_response"],
                actor_path,
                goal_xy,
                config,
                evaluation_times,
                rank_by_identifier,
            ),
            _mode_payload(
                "team_minimax",
                "기존 팀 minimax",
                branch["best_response"],
                actor_path,
                goal_xy,
                config,
                evaluation_times,
                rank_by_identifier,
            ),
            _mode_payload(
                "runner_follow",
                "기존 Runner 대응",
                branch["runner_follow_response"],
                actor_path,
                goal_xy,
                config,
                evaluation_times,
                rank_by_identifier,
            ),
            _mode_payload(
                "dynamic_goal_side",
                "새 dynamic goal-side 차단",
                dynamic_response,
                actor_path,
                goal_xy,
                config,
                evaluation_times,
                rank_by_identifier,
            ),
        ]
        output_branches.append(
            {
                "defender_id": defender_id,
                "defender_name": branch["defender_name"],
                "defender_position": branch["defender_position"],
                "feasible_response_count": len(responses),
                "carrier_option": {
                    "player_id": owner_option["player_id"],
                    "player_name": owner_option["player_name"],
                    "event_x": float(owner_option["event_x"]),
                    "event_y": float(owner_option["event_y"]),
                    "event_time_s": float(owner_option["peak_time_s"]),
                    "path_txy": [list(sample) for sample in actor_path],
                },
                "modes": modes,
                "top_dynamic_paths": [
                    {
                        "rank": rank,
                        "response_index": int(row.candidate.identifier),
                        "effort_m2ps3": float(row.candidate.effort),
                        "summary": _summary_payload(row.summary),
                    }
                    for rank, row in enumerate(ranking[: max(1, args.top_k)], start=1)
                ],
            }
        )
    return {
        "schema_version": "dynamic-carry-response-v0.6",
        "status": "moving-target geometry audit; not final threat model",
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
        "branches": output_branches,
        "background_frames": result["background_frames"],
        "names": result["names"],
        "notes": [
            "The attacker is a full carry trajectory, not one terminal point.",
            "At time t the target uses the carrier position at t + lookahead, then shifts goal-side by the configured offset.",
            "Lower weighted error is better; wrong-side distance is penalized twice by default.",
            "This audit changes response ranking only. It does not claim that the geometry cost is the final defensive value function.",
        ],
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dynamic_carry_response_v0_6.json"
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "branch_count": len(result["branches"]),
        "config": result["config"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")
    for branch in result["branches"]:
        print(f"{branch['defender_name']} ({branch['feasible_response_count']} paths)")
        for mode in branch["modes"]:
            summary = mode["summary"]
            rank = mode["dynamic_rank"]
            print(
                f"  {mode['label']}: rank={rank} "
                f"mean={summary['mean_weighted_error_m']:.2f}m "
                f"wrong={summary['wrong_side_fraction']:.0%}"
            )


if __name__ == "__main__":
    main()
