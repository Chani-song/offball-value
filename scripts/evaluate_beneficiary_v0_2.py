#!/usr/bin/env python3
"""Build a beneficiary v0.2 diagnostic from the v0.1 response landscape.

This script deliberately changes only beneficiary identification.  It reuses
the v0.1 runner-cover response and all of that defender's feasible endpoint
responses, then values legal receiver options with transparent pass reception
and geometric goal-danger components.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from offball_value.beneficiary_selection import (
    select_follow_beneficiary,
)
from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.defender_best_response import interpolate_path_state
from offball_value.obso import is_offside_position, score_at_points
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    ReceptionRegionConfig,
    VelocityEstimate,
    detect_kick_frame,
    estimate_frame_velocities,
    pass_execution_estimate,
    point_reception_estimate,
    reception_region_estimate,
)
from offball_value.pass_window_value import controlled_ball_owner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
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
        default=Path("data/processed/beneficiary_v0_2"),
    )
    parser.add_argument("--release-step-seconds", type=float, default=0.2)
    parser.add_argument("--control-distance-m", type=float, default=1.5)
    return parser.parse_args()


def _geometric_value(points, attacking_direction):
    return score_at_points(points, attacking_direction, epv_grid_path=None)


def _virtual_state(base_frame, base_velocities, scene, response, time_s):
    defender_id = scene["defender_id"]
    if math.isclose(time_s, 0.0):
        return base_frame, dict(base_velocities)
    x, y, vx, vy = interpolate_path_state(
        response["path_xy"], response["times_s"], time_s
    )
    defender = base_frame.players[defender_id]
    frame = base_frame.with_player(
        defender_id,
        replace(defender, x=float(x), y=float(y), speed=math.hypot(vx, vy) * 3.6),
    )
    velocities = dict(base_velocities)
    velocities[defender_id] = VelocityEstimate(
        float(vx), float(vy), float(math.hypot(vx, vy)), 0, 0.0
    )
    return frame, velocities


def _release_times(frame_map, onset, horizon_s, owner_id, step_s, control_distance):
    kick = detect_kick_frame(frame_map.values(), owner_id, onset)
    kick_time = (kick.kick_frame_id - onset) / FPS
    if kick.detected and 0.04 <= kick_time <= horizon_s + 1e-9:
        end_time = kick_time
        reason = "observed_kick"
    else:
        last_controlled = 0.0
        invalid_streak = 0
        end_frame = onset + int(round(horizon_s * FPS))
        for frame_id in range(onset + 1, end_frame + 1):
            controller, distance, controlled = controlled_ball_owner(
                frame_map[frame_id],
                frame_map[onset].players[owner_id].team_id,
                maximum_distance_m=control_distance,
            )
            if controlled and controller == owner_id:
                last_controlled = (frame_id - onset) / FPS
                invalid_streak = 0
            else:
                invalid_streak += 1
                if invalid_streak >= 3:
                    break
        end_time = min(horizon_s, last_controlled)
        reason = "same_owner_control_end"
    if end_time <= 0.0:
        raise ValueError("scene contains no post-onset controlled release instant")
    values = [0.0]
    cursor = step_s
    while cursor < end_time - 1e-9:
        values.append(round(cursor, 6))
        cursor += step_s
    values.append(round(end_time, 6))
    return tuple(sorted(set(values))), reason, kick


def _receiver_option(
    frame,
    velocities,
    player_id,
    owner_id,
    team_id,
    direction,
    arrival_config,
    region_config,
):
    ball_xy = (float(frame.ball.x), float(frame.ball.y))
    player = frame.players[player_id]
    if player_id == owner_id:
        pressure = pass_execution_estimate(
            frame,
            velocities,
            owner_id,
            team_id,
            0.0,
            arrival_config,
        )
        goal = float(_geometric_value([(player.x, player.y)], direction)[0])
        value = float(pressure[-1] * goal)
        return {
            "option_type": "carry_or_shot",
            "legal": True,
            "value": value,
            "event_x": float(player.x),
            "event_y": float(player.y),
            "receive_probability": float(pressure[-1]),
            "goal_value": goal,
            "path_survival_probability": 1.0,
            "receiver_first_probability": 1.0,
        }
    offside = is_offside_position(frame, player_id, team_id, ball_xy, direction)
    estimate = reception_region_estimate(
        frame,
        velocities,
        player_id,
        owner_id,
        team_id,
        direction,
        ball_xy,
        config=arrival_config,
        region_config=region_config,
        receive_value_fn=_geometric_value,
    )
    point_values = estimate.receive_probability * estimate.receive_value
    index = int(np.argmax(point_values))
    target = estimate.points[index]
    point = point_reception_estimate(
        frame,
        velocities,
        player_id,
        team_id,
        ball_xy,
        (float(target[0]), float(target[1])),
        arrival_config,
        passer_id=owner_id,
    )
    return {
        "option_type": "pass_receive",
        "legal": not offside,
        "value": 0.0 if offside else float(point_values[index]),
        "event_x": float(target[0]),
        "event_y": float(target[1]),
        "receive_probability": 0.0 if offside else float(point.receive_probability),
        "goal_value": float(estimate.receive_value[index]),
        "path_survival_probability": (
            0.0 if offside else float(point.path_survival_probability)
        ),
        "receiver_first_probability": (
            0.0 if offside else float(point.receiver_first_probability)
        ),
    }


def _fixed_event_value(
    frame,
    velocities,
    player_id,
    owner_id,
    team_id,
    direction,
    option_type,
    event_xy,
    arrival_config,
):
    goal = float(_geometric_value([event_xy], direction)[0])
    if option_type == "carry_or_shot":
        pressure = pass_execution_estimate(
            frame, velocities, owner_id, team_id, 0.0, arrival_config
        )
        return float(pressure[-1] * goal), {
            "receive_probability": float(pressure[-1]),
            "goal_value": goal,
            "path_survival_probability": 1.0,
            "receiver_first_probability": 1.0,
        }
    ball_xy = (float(frame.ball.x), float(frame.ball.y))
    if is_offside_position(frame, player_id, team_id, ball_xy, direction):
        return 0.0, {
            "receive_probability": 0.0,
            "goal_value": goal,
            "path_survival_probability": 0.0,
            "receiver_first_probability": 0.0,
        }
    point = point_reception_estimate(
        frame,
        velocities,
        player_id,
        team_id,
        ball_xy,
        event_xy,
        arrival_config,
        passer_id=owner_id,
    )
    return float(point.receive_probability * goal), {
        "receive_probability": float(point.receive_probability),
        "goal_value": goal,
        "path_survival_probability": float(point.path_survival_probability),
        "receiver_first_probability": float(point.receiver_first_probability),
    }


def _observed_player_xy(scene, player_id, time_s):
    frames = scene["background_frames"]
    if time_s <= frames[0]["time_s"]:
        player = next(item for item in frames[0]["players"] if item[0] == player_id)
        return float(player[2]), float(player[3])
    if time_s >= frames[-1]["time_s"]:
        player = next(item for item in frames[-1]["players"] if item[0] == player_id)
        return float(player[2]), float(player[3])
    upper = next(index for index, frame in enumerate(frames) if frame["time_s"] >= time_s)
    before, after = frames[upper - 1], frames[upper]
    first = next(item for item in before["players"] if item[0] == player_id)
    second = next(item for item in after["players"] if item[0] == player_id)
    fraction = (time_s - before["time_s"]) / (after["time_s"] - before["time_s"])
    return (
        float(first[2] + fraction * (second[2] - first[2])),
        float(first[3] + fraction * (second[3] - first[3])),
    )


def _select_runner_follow_response(scene, goal_side_offset_m=1.5):
    """Choose a feasible response that shadows the runner goal-side.

    The anchor balances physical goal-side tracking error with the existing
    runner-specific direct-threat score.  It is selected from the complete
    feasible response set rather than constructed as an unconstrained line.
    """

    direction = int(scene["attacking_direction"])
    rows = []
    for response in scene["responses"]:
        errors = []
        wrong_side = []
        for time_s in response["times_s"]:
            if time_s < 0.2 - 1e-9:
                continue
            runner = np.asarray(
                _observed_player_xy(scene, scene["runner_id"], float(time_s)),
                dtype=float,
            )
            goal = np.asarray((direction * 52.5, 0.0), dtype=float)
            to_goal = goal - runner
            norm = float(np.linalg.norm(to_goal))
            unit = to_goal / max(norm, 1e-9)
            target = runner + goal_side_offset_m * unit
            x, y, _, _ = interpolate_path_state(
                response["path_xy"], response["times_s"], float(time_s)
            )
            defender = np.asarray((x, y), dtype=float)
            errors.append(float(np.linalg.norm(defender - target)))
            wrong_side.append(max(0.0, -float(np.dot(defender - runner, unit))))
        tracking_error = float(np.mean(errors)) if errors else float("inf")
        wrong_side_error = float(np.mean(wrong_side)) if wrong_side else float("inf")
        rows.append(
            {
                "index": int(response["index"]),
                "tracking_error_m": tracking_error,
                "wrong_side_error_m": wrong_side_error,
                "direct_threat": float(response["direct_threat"]),
                "effort_m2ps3": float(response["effort_m2ps3"]),
            }
        )
    tracking = np.asarray(
        [row["tracking_error_m"] + 2.0 * row["wrong_side_error_m"] for row in rows]
    )
    direct = np.asarray([row["direct_threat"] for row in rows])
    tracking_norm = (tracking - np.min(tracking)) / max(float(np.ptp(tracking)), 1e-12)
    direct_norm = (direct - np.min(direct)) / max(float(np.ptp(direct)), 1e-12)
    for index, row in enumerate(rows):
        row["follow_anchor_score"] = float(0.5 * tracking_norm[index] + 0.5 * direct_norm[index])
    selected = min(
        rows,
        key=lambda row: (
            row["follow_anchor_score"],
            row["tracking_error_m"],
            row["effort_m2ps3"],
            row["index"],
        ),
    )
    return selected


def _mean_path_separation(first, second):
    times = sorted(set(first["times_s"]) | set(second["times_s"]))
    distances = []
    for time_s in times:
        ax, ay, _, _ = interpolate_path_state(first["path_xy"], first["times_s"], time_s)
        bx, by, _, _ = interpolate_path_state(second["path_xy"], second["times_s"], time_s)
        distances.append(math.hypot(ax - bx, ay - by))
    return float(np.mean(distances)) if distances else 0.0


def _evaluate_scene(scene, data_dir, release_step_seconds, control_distance_m):
    files = find_bundesliga_files(data_dir, scene["match_id"])
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    onset = int(scene["onset_frame_id"])
    horizon_s = float(scene["horizon_seconds"])
    final = onset + int(round(horizon_s * FPS))
    frame_map = load_bundesliga_frames(files["positions"], range(onset - 10, final + 1))
    decision = frame_map[onset]
    team_id = scene["attacking_team_id"]
    owner_id, owner_distance, controlled = controlled_ball_owner(decision, team_id)
    if not controlled or owner_id is None:
        raise ValueError(f"scene {scene['match_id']}:{onset} has no controlled owner")
    release_times, release_reason, kick = _release_times(
        frame_map,
        onset,
        horizon_s,
        owner_id,
        release_step_seconds,
        control_distance_m,
    )
    base_states = {}
    for time_s in release_times:
        frame_id = onset + int(round(time_s * FPS))
        base_states[time_s] = (
            frame_map[frame_id],
            estimate_frame_velocities(frame_map.values(), frame_id, ArrivalModelConfig()),
        )

    follow_anchor = _select_runner_follow_response(scene)
    response_index = int(follow_anchor["index"])
    runner_response = next(
        response for response in scene["responses"] if int(response["index"]) == response_index
    )
    goalkeeper_id = metadata.goalkeeper_id(team_id)
    teammate_ids = tuple(
        sorted(
            player_id
            for player_id, player in decision.players.items()
            if player.team_id == team_id
            and player_id not in {scene["runner_id"], goalkeeper_id}
        )
    )
    arrival_config = ArrivalModelConfig()
    region_config = ReceptionRegionConfig(resolution_m=2.0)

    onset_options = {}
    def peak_option(player_id, response):
        post = []
        for time_s in release_times[1:]:
            base_frame, base_velocities = base_states[time_s]
            frame, velocities = _virtual_state(
                base_frame, base_velocities, scene, response, time_s
            )
            option = _receiver_option(
                frame,
                velocities,
                player_id,
                owner_id,
                team_id,
                int(scene["attacking_direction"]),
                arrival_config,
                region_config,
            )
            post.append((option["value"], -time_s, time_s, option))
        peak = max(post, key=lambda item: (item[0], item[1]))
        return peak[3] | {"time_s": peak[2]}

    follow_options = {}
    for player_id in teammate_ids:
        base_frame, base_velocities = base_states[0.0]
        onset_options[player_id] = _receiver_option(
            base_frame,
            base_velocities,
            player_id,
            owner_id,
            team_id,
            int(scene["attacking_direction"]),
            arrival_config,
            region_config,
        )
        follow_options[player_id] = peak_option(player_id, runner_response)

    runner_follow_option = peak_option(scene["runner_id"], runner_response)

    best_cover_values = {}
    best_cover_indices = {}
    best_cover_components = {}
    for player_id in teammate_ids:
        exposed = follow_options[player_id]
        time_s = float(exposed["time_s"])
        base_frame, base_velocities = base_states[time_s]
        values = []
        components = []
        for response in scene["responses"]:
            frame, velocities = _virtual_state(
                base_frame, base_velocities, scene, response, time_s
            )
            value, detail = _fixed_event_value(
                frame,
                velocities,
                player_id,
                owner_id,
                team_id,
                int(scene["attacking_direction"]),
                exposed["option_type"],
                (float(exposed["event_x"]), float(exposed["event_y"])),
                arrival_config,
            )
            values.append(value)
            components.append(detail)
        index = min(
            range(len(values)),
            key=lambda item: (
                values[item],
                float(scene["responses"][item]["effort_m2ps3"]),
                item,
            ),
        )
        best_cover_values[player_id] = float(values[index])
        best_cover_indices[player_id] = int(index)
        best_cover_components[player_id] = components[index]

    runner_at_candidate_cover = {}
    runner_cover_option_details = {}
    for cover_index in sorted(set(best_cover_indices.values())):
        response = scene["responses"][cover_index]
        runner_cover_option_details[cover_index] = peak_option(
            scene["runner_id"], response
        )
    for player_id in teammate_ids:
        runner_at_candidate_cover[player_id] = float(
            runner_cover_option_details[best_cover_indices[player_id]]["value"]
        )

    selection = select_follow_beneficiary(
        {key: value["value"] for key, value in follow_options.items()},
        best_cover_values,
        float(runner_follow_option["value"]),
        runner_at_candidate_cover,
        legal_option={key: value["legal"] for key, value in follow_options.items()},
        meaningful_attacking_option={
            key: int(scene["attacking_direction"]) * float(value["event_x"]) > 0.0
            for key, value in follow_options.items()
        },
    )
    candidates = []
    for score in selection.candidates:
        player_id = score.player_id
        exposed = follow_options[player_id]
        cover_index = best_cover_indices[player_id]
        cover_response = scene["responses"][cover_index]
        follow_components = {
            key: exposed[key]
            for key in (
                "receive_probability",
                "goal_value",
                "path_survival_probability",
                "receiver_first_probability",
            )
        }
        cover_components = best_cover_components[player_id]
        reductions = {
            "pass_or_execution": follow_components["receive_probability"]
            - cover_components["receive_probability"],
            "pass_lane": follow_components["path_survival_probability"]
            - cover_components["path_survival_probability"],
            "receiver_race": follow_components["receiver_first_probability"]
            - cover_components["receiver_first_probability"],
        }
        mechanism = max(reductions, key=reductions.get)
        candidates.append(
            {
                **asdict(score),
                "player_name": metadata.players[player_id].short_name,
                "position": metadata.players[player_id].position,
                "option_type": exposed["option_type"],
                "peak_time_s": exposed["time_s"],
                "event_x": exposed["event_x"],
                "event_y": exposed["event_y"],
                "onset_value": float(onset_options[player_id]["value"]),
                "temporal_gain_from_v0": float(
                    exposed["value"] - onset_options[player_id]["value"]
                ),
                "attacking_half_event": (
                    int(scene["attacking_direction"]) * float(exposed["event_x"]) > 0.0
                ),
                "follow_components": follow_components,
                "best_cover_components": cover_components,
                "suppression_mechanism": mechanism,
                "component_reductions": reductions,
                "best_cover_response_index": cover_index,
                "best_cover_response": cover_response,
                "mean_response_separation_m": _mean_path_separation(
                    runner_response, cover_response
                ),
                "runner_candidate_cover_option": runner_cover_option_details[
                    cover_index
                ],
            }
        )

    return {
        "schema_version": "beneficiary-v0.2-scene-diagnostic",
        "match_id": scene["match_id"],
        "match_label": scene["match_label"],
        "onset_frame_id": onset,
        "runner_id": scene["runner_id"],
        "runner_name": scene["runner_name"],
        "defender_id": scene["defender_id"],
        "defender_name": scene["defender_name"],
        "attacking_team_id": team_id,
        "attacking_direction": scene["attacking_direction"],
        "ball_owner_id": owner_id,
        "ball_owner_name": metadata.players[owner_id].short_name,
        "ball_owner_distance_m": owner_distance,
        "release_times_s": list(release_times),
        "release_window_reason": release_reason,
        "kick_detected": kick.detected,
        "selected_beneficiary_id": selection.selected_player_id,
        "selected_beneficiary_name": (
            metadata.players[selection.selected_player_id].short_name
            if selection.selected_player_id is not None
            else None
        ),
        "runner_follow_response_index": response_index,
        "runner_follow_response": runner_response,
        "runner_follow_anchor": follow_anchor,
        "runner_follow_option": runner_follow_option,
        "actual_response": scene["actual_response"],
        "candidates": candidates,
        "background_frames": scene["background_frames"],
        "review": scene["review"],
        "response_count": len(scene["responses"]),
        "names": {
            player_id: player.short_name for player_id, player in metadata.players.items()
        },
        "notes": [
            "Runner-follow anchor is selected from all feasible responses using goal-side tracking error and runner direct threat.",
            "Receiver option = pass reception probability × geometric goal danger.",
            "Follow benefit compares the same exposed release time and target under runner-follow and candidate-cover responses.",
            "Runner cost recomputes the runner's best receiver option under each candidate-cover response.",
            "V0 temporal gain is display-only and never selects a beneficiary.",
            "Thresholds remain numerical epsilons; effect-size calibration is not yet applied.",
            "A named beneficiary event must be in the attacking half; playing position is never used as a filter.",
        ],
    }


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, scene in enumerate(scenes, start=1):
        print(
            f"[{index}/{len(scenes)}] {scene['match_id']}:{scene['onset_frame_id']} "
            f"{scene['runner_name']}",
            flush=True,
        )
        result = _evaluate_scene(
            scene,
            args.data_dir,
            args.release_step_seconds,
            args.control_distance_m,
        )
        results.append(result)
        top = result["selected_beneficiary_name"] or "none"
        print(
            f"  releases={result['release_times_s']} · beneficiary={top} · "
            f"{result['response_count']} responses",
            flush=True,
        )
    output = args.output_dir / "beneficiary_v0_2.json"
    output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "beneficiary-v0.2-diagnostic",
        "scene_count": len(results),
        "status": "human-audit before effect-size calibration",
        "selection_rule": "positive V0 gain gate, then maximum fixed-event defender choice effect",
        "arrival_model": asdict(ArrivalModelConfig()),
        "reception_region": asdict(ReceptionRegionConfig(resolution_m=2.0)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
