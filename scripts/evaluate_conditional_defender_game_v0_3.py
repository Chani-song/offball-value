#!/usr/bin/env python3
"""Evaluate defender-conditioned beneficiaries for the local off-ball game.

Unlike beneficiary v0.2, this diagnostic does not inherit one pre-selected
defender.  It first shortlists defenders using runner-only responsibility at
the decision frame, generates a separate feasible response set for each, and
then asks which teammate option benefits when that particular defender tracks
the runner goal-side.

The ball owner's option is also changed from a value at the observed position
to a bounded-control forward-carry option.  This is intentionally a transparent
diagnostic proxy rather than a calibrated possession-value model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from evaluate_beneficiary_v0_2 import (
    _fixed_event_value,
    _geometric_value,
    _mean_path_separation,
    _receiver_option,
    _release_times,
    _select_runner_follow_response,
    _virtual_state,
)
from offball_value.beneficiary_selection import select_follow_beneficiary
from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.defender_best_response import interpolate_path_state
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    ReceptionRegionConfig,
    VelocityEstimate,
    estimate_frame_velocities,
    player_time_to_point,
)
from offball_value.pass_window_value import controlled_ball_owner
from offball_value.steering_reachable import SteeringReachabilityConfig


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
        default=Path("data/processed/conditional_defender_game_v0_3"),
    )
    parser.add_argument("--relevant-defender-count", type=int, default=3)
    parser.add_argument("--release-step-seconds", type=float, default=0.2)
    parser.add_argument("--control-distance-m", type=float, default=1.5)
    parser.add_argument("--minimum-relative-cross-cost", type=float, default=0.05)
    parser.add_argument(
        "--scene-index",
        type=int,
        action="append",
        help="1-based scene index; repeat to evaluate a subset.",
    )
    return parser.parse_args()


def _reachability_config() -> SteeringReachabilityConfig:
    return SteeringReachabilityConfig(
        state_position_resolution_m=1.5,
        state_speed_resolution_mps=1.0,
        state_heading_bins=48,
        control_direction_count=12,
        plant_cut_direction_count=12,
        variants_per_endpoint=1,
    )


def _unique_endpoint_actions(actions):
    selected = {}
    for action in actions:
        key = (
            float(action.base_action.endpoint_cell_x),
            float(action.base_action.endpoint_cell_y),
        )
        rank = (float(action.base_action.motion.effort_m2ps3), action.action_id)
        incumbent = selected.get(key)
        if incumbent is None or rank < (
            float(incumbent.base_action.motion.effort_m2ps3),
            incumbent.action_id,
        ):
            selected[key] = action
    return tuple(selected[key] for key in sorted(selected))


def _response_payload(action, index):
    return {
        "index": int(index),
        "action_id": action.action_id,
        "endpoint_x": float(action.endpoint_x),
        "endpoint_y": float(action.endpoint_y),
        "times_s": [float(value) for value in action.response_path_times_s],
        "path_xy": [[float(x), float(y)] for x, y in action.full_path_xy],
        "effort_m2ps3": float(action.base_action.motion.effort_m2ps3),
        # The v0.3 follow anchor uses goal-side tracking only.  A constant
        # placeholder keeps the v0.2 helper's normalization neutral.
        "direct_threat": 0.0,
    }


def _actual_response(scene, defender_id):
    path = [
        [float(frame["time_s"]), float(player[2]), float(player[3])]
        for frame in scene["background_frames"]
        if frame["time_s"] >= 0.0
        for player in frame["players"]
        if player[0] == defender_id
    ]
    return {
        "path_txy": path,
        "times_s": [item[0] for item in path],
        "path_xy": [[item[1], item[2]] for item in path],
        "direct_threat": 0.0,
        "effort_m2ps3": 0.0,
        "index": -1,
    }


def _runner_only_defenders(scene, count):
    """Use only t=0 runner suppression responsibility to form the shortlist."""

    ordered = sorted(
        scene["candidate_defenders"],
        key=lambda row: (
            -float(row["direct_responsibility"]),
            row["defender_id"],
        ),
    )
    maximum = float(ordered[0]["direct_responsibility"]) if ordered else 0.0
    meaningful = [
        row
        for row in ordered
        if float(row["direct_responsibility"]) >= 0.10 * max(maximum, 1e-12)
    ]
    # The audit is comparative by construction: retain at least two runner-only
    # candidates when two outfield defenders are available, but never pad the
    # third slot with a virtually zero-responsibility player.
    minimum = min(2, len(ordered))
    selected = meaningful[: max(1, int(count))]
    if len(selected) < minimum:
        selected = ordered[:minimum]
    return selected


def _carry_point(action, time_s):
    # A zero response delay intentionally produces two identical t=0 samples
    # in DefenderResponseAction.  Collapse them before using the strict path
    # interpolator.
    samples = {}
    for sample_time, point in zip(
        action.response_path_times_s, action.full_path_xy
    ):
        samples[float(sample_time)] = point
    times = tuple(sorted(samples))
    path = tuple(samples[value] for value in times)
    x, y, vx, vy = interpolate_path_state(
        path, times, time_s
    )
    return float(x), float(y), float(vx), float(vy)


def _carry_path_until(action, time_s, step_s=0.2):
    times = [0.0]
    cursor = step_s
    while cursor < time_s - 1e-9:
        times.append(float(cursor))
        cursor += step_s
    times.append(float(time_s))
    return [
        [t, *_carry_point(action, t)[:2]]
        for t in times
    ]


def _carry_value(
    frame,
    velocities,
    owner_id,
    team_id,
    direction,
    event_xy,
    event_velocity,
    onset_goal_value,
    arrival_config,
):
    owner = frame.players[owner_id]
    event_frame = frame.with_player(
        owner_id,
        replace(
            owner,
            x=float(event_xy[0]),
            y=float(event_xy[1]),
            speed=math.hypot(*event_velocity) * 3.6,
        ),
    )
    if event_frame.ball is not None:
        event_frame = replace(
            event_frame,
            ball=replace(
                event_frame.ball,
                x=float(event_xy[0]),
                y=float(event_xy[1]),
            ),
        )
    event_velocities = dict(velocities)
    event_velocities[owner_id] = VelocityEstimate(
        float(event_velocity[0]),
        float(event_velocity[1]),
        float(math.hypot(*event_velocity)),
        0,
        0.0,
    )
    zero_velocity = VelocityEstimate(
        0.0, 0.0, 0.0, 0, arrival_config.history_seconds
    )
    defender_arrivals = sorted(
        (
            player_time_to_point(
                defender,
                event_velocities.get(defender.object_id, zero_velocity),
                (float(event_xy[0]), float(event_xy[1])),
                arrival_config,
            ),
            defender.object_id,
        )
        for defender in event_frame.players.values()
        if defender.team_id != team_id
    )
    individual_survival = [
        1.0
        / (
            1.0
            + math.exp(
                -(
                    arrival_time - arrival_config.pressure_action_time_s
                )
                / max(arrival_config.pressure_sigma_s, 1e-9)
            )
        )
        for arrival_time, _ in defender_arrivals
    ]
    endpoint_goal = float(_geometric_value([event_xy], direction)[0])
    goal_gain = max(0.0, endpoint_goal - onset_goal_value)
    # Unlike nearest-defender pressure, the product distinguishes one-cover
    # from two-cover states.  This is needed for the user's central-defender
    # allocation hypothesis: the closest marker may remain unchanged while a
    # second covering defender leaves with the runner.
    retention = float(np.prod(individual_survival))
    nearest = defender_arrivals[0] if defender_arrivals else (math.inf, None)
    support = defender_arrivals[1] if len(defender_arrivals) > 1 else (math.inf, None)
    return retention * goal_gain, {
        "receive_probability": retention,
        "goal_value": goal_gain,
        "path_survival_probability": retention,
        "receiver_first_probability": 1.0,
        "pressure_defender_id": nearest[1],
        "pressure_time_s": float(nearest[0]),
        "support_pressure_defender_id": support[1],
        "support_pressure_time_s": float(support[0]),
        "endpoint_goal_value": endpoint_goal,
        "onset_goal_value": float(onset_goal_value),
    }


def _evaluate_scene(
    scene,
    data_dir,
    relevant_defender_count,
    release_step_seconds,
    control_distance_m,
    minimum_relative_cross_cost,
):
    files = find_bundesliga_files(data_dir, scene["match_id"])
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    onset = int(scene["onset_frame_id"])
    horizon_s = float(scene["horizon_seconds"])
    final = onset + int(round(horizon_s * FPS))
    frame_map = load_bundesliga_frames(files["positions"], range(onset - 10, final + 1))
    history = tuple(frame_map[index] for index in range(onset - 10, onset + 1))
    decision = frame_map[onset]
    team_id = scene["attacking_team_id"]
    direction = int(scene["attacking_direction"])
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
    arrival_config = ArrivalModelConfig()
    region_config = ReceptionRegionConfig(resolution_m=2.0)
    base_states = {
        time_s: (
            frame_map[onset + int(round(time_s * FPS))],
            estimate_frame_velocities(
                frame_map.values(),
                onset + int(round(time_s * FPS)),
                arrival_config,
            ),
        )
        for time_s in release_times
    }
    goalkeeper_id = metadata.goalkeeper_id(team_id)
    teammate_ids = tuple(
        sorted(
            player_id
            for player_id, player in decision.players.items()
            if player.team_id == team_id
            and player_id not in {scene["runner_id"], goalkeeper_id}
        )
    )

    # A common ball-owner action set is held fixed across defender branches.
    owner_generated = generate_defender_response_actions(
        decision,
        history,
        owner_id,
        DefenderResponseConfig(
            horizon_seconds=horizon_s,
            response_delay_seconds=0.0,
        ),
        _reachability_config(),
    )
    owner_actions = _unique_endpoint_actions(owner_generated.actions)
    onset_goal = float(
        _geometric_value(
            [(decision.players[owner_id].x, decision.players[owner_id].y)],
            direction,
        )[0]
    )

    def peak_pass_option(branch_scene, player_id, response):
        post = []
        for time_s in release_times[1:]:
            base_frame, base_velocities = base_states[time_s]
            frame, velocities = _virtual_state(
                base_frame, base_velocities, branch_scene, response, time_s
            )
            option = _receiver_option(
                frame,
                velocities,
                player_id,
                owner_id,
                team_id,
                direction,
                arrival_config,
                region_config,
            )
            post.append((option["value"], -time_s, option | {"time_s": time_s}))
        return max(post, key=lambda item: (item[0], item[1]))[2]

    def peak_carry_option(branch_scene, response):
        best = None
        start_x = float(decision.players[owner_id].x)
        for time_s in release_times[1:]:
            base_frame, base_velocities = base_states[time_s]
            frame, velocities = _virtual_state(
                base_frame, base_velocities, branch_scene, response, time_s
            )
            for action_index, action in enumerate(owner_actions):
                x, y, vx, vy = _carry_point(action, time_s)
                if direction * (x - start_x) < 1.0:
                    continue
                value, detail = _carry_value(
                    frame,
                    velocities,
                    owner_id,
                    team_id,
                    direction,
                    (x, y),
                    (vx, vy),
                    onset_goal,
                    arrival_config,
                )
                rank = (value, -time_s, -float(action.base_action.motion.effort_m2ps3))
                if best is None or rank > best[0]:
                    best = (
                        rank,
                        {
                            "option_type": "forward_carry_space",
                            "legal": True,
                            "value": float(value),
                            "event_x": x,
                            "event_y": y,
                            "time_s": float(time_s),
                            "event_vx": vx,
                            "event_vy": vy,
                            "carry_action_index": int(action_index),
                            "carry_path_txy": _carry_path_until(action, time_s),
                            **detail,
                        },
                    )
        if best is None:
            return {
                "option_type": "forward_carry_space",
                "legal": False,
                "value": 0.0,
                "event_x": float(decision.players[owner_id].x),
                "event_y": float(decision.players[owner_id].y),
                "time_s": float(release_times[-1]),
                "event_vx": 0.0,
                "event_vy": 0.0,
                "carry_action_index": None,
                "carry_path_txy": [[0.0, float(decision.players[owner_id].x), float(decision.players[owner_id].y)]],
                "receive_probability": 0.0,
                "goal_value": 0.0,
                "path_survival_probability": 0.0,
                "receiver_first_probability": 1.0,
            }
        return best[1]

    defender_rows = _runner_only_defenders(scene, relevant_defender_count)
    branches = []
    for defender_order, defender_row in enumerate(defender_rows, start=1):
        defender_id = str(defender_row["defender_id"])
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
        actions = _unique_endpoint_actions(generated.actions)
        responses = [
            _response_payload(action, index) for index, action in enumerate(actions)
        ]
        actual = _actual_response(scene, defender_id)
        branch_scene = {
            **scene,
            "defender_id": defender_id,
            "defender_name": metadata.players[defender_id].short_name,
            "responses": responses,
            "actual_response": actual,
        }
        follow_anchor = _select_runner_follow_response(branch_scene)
        runner_response = responses[int(follow_anchor["index"])]

        follow_options = {}
        for player_id in teammate_ids:
            if player_id == owner_id:
                follow_options[player_id] = peak_carry_option(
                    branch_scene, runner_response
                )
            else:
                follow_options[player_id] = peak_pass_option(
                    branch_scene, player_id, runner_response
                )
        runner_follow_option = peak_pass_option(
            branch_scene, scene["runner_id"], runner_response
        )

        best_cover_values = {}
        best_cover_indices = {}
        best_cover_components = {}
        for player_id in teammate_ids:
            exposed = follow_options[player_id]
            time_s = float(exposed["time_s"])
            base_frame, base_velocities = base_states[time_s]
            values = []
            details = []
            for response in responses:
                frame, velocities = _virtual_state(
                    base_frame, base_velocities, branch_scene, response, time_s
                )
                if player_id == owner_id:
                    value, detail = _carry_value(
                        frame,
                        velocities,
                        owner_id,
                        team_id,
                        direction,
                        (float(exposed["event_x"]), float(exposed["event_y"])),
                        (float(exposed["event_vx"]), float(exposed["event_vy"])),
                        onset_goal,
                        arrival_config,
                    )
                else:
                    value, detail = _fixed_event_value(
                        frame,
                        velocities,
                        player_id,
                        owner_id,
                        team_id,
                        direction,
                        exposed["option_type"],
                        (float(exposed["event_x"]), float(exposed["event_y"])),
                        arrival_config,
                    )
                values.append(float(value))
                details.append(detail)
            cover_index = min(
                range(len(values)),
                key=lambda index: (
                    values[index],
                    float(responses[index]["effort_m2ps3"]),
                    index,
                ),
            )
            best_cover_values[player_id] = values[cover_index]
            best_cover_indices[player_id] = int(cover_index)
            best_cover_components[player_id] = details[cover_index]

        runner_cover_options = {}
        for cover_index in sorted(set(best_cover_indices.values())):
            runner_cover_options[cover_index] = peak_pass_option(
                branch_scene, scene["runner_id"], responses[cover_index]
            )
        runner_at_candidate_cover = {
            player_id: float(
                runner_cover_options[best_cover_indices[player_id]]["value"]
            )
            for player_id in teammate_ids
        }
        selection = select_follow_beneficiary(
            {key: value["value"] for key, value in follow_options.items()},
            best_cover_values,
            float(runner_follow_option["value"]),
            runner_at_candidate_cover,
            legal_option={key: value["legal"] for key, value in follow_options.items()},
            meaningful_attacking_option={
                key: direction * float(value["event_x"]) > 0.0
                for key, value in follow_options.items()
            },
            minimum_follow_benefit_fraction=minimum_relative_cross_cost,
            minimum_runner_cost_fraction=minimum_relative_cross_cost,
        )
        candidates = []
        for score in selection.candidates:
            player_id = score.player_id
            exposed = follow_options[player_id]
            cover_index = best_cover_indices[player_id]
            follow_components = {
                key: float(exposed[key])
                for key in (
                    "receive_probability",
                    "goal_value",
                    "path_survival_probability",
                    "receiver_first_probability",
                )
            }
            cover_components = {
                key: float(best_cover_components[player_id][key])
                for key in follow_components
            }
            reductions = {
                "execution_or_retention": follow_components["receive_probability"]
                - cover_components["receive_probability"],
                "path_survival": follow_components["path_survival_probability"]
                - cover_components["path_survival_probability"],
                "receiver_race": follow_components["receiver_first_probability"]
                - cover_components["receiver_first_probability"],
            }
            candidates.append(
                {
                    **asdict(score),
                    "player_name": metadata.players[player_id].short_name,
                    "position": metadata.players[player_id].position,
                    "option_type": exposed["option_type"],
                    "peak_time_s": float(exposed["time_s"]),
                    "event_x": float(exposed["event_x"]),
                    "event_y": float(exposed["event_y"]),
                    "carry_path_txy": exposed.get("carry_path_txy"),
                    "attacking_half_event": direction * float(exposed["event_x"]) > 0.0,
                    "follow_components": follow_components,
                    "best_cover_components": cover_components,
                    "suppression_mechanism": max(reductions, key=reductions.get),
                    "component_reductions": reductions,
                    "best_cover_response_index": cover_index,
                    "best_cover_response": responses[cover_index],
                    "mean_response_separation_m": _mean_path_separation(
                        runner_response, responses[cover_index]
                    ),
                    "runner_candidate_cover_option": runner_cover_options[cover_index],
                }
            )

        selected_name = (
            metadata.players[selection.selected_player_id].short_name
            if selection.selected_player_id is not None
            else None
        )
        branches.append(
            {
                "defender_rank": defender_order,
                "defender_id": defender_id,
                "defender_name": metadata.players[defender_id].short_name,
                "defender_position": metadata.players[defender_id].position,
                "runner_only_responsibility": float(
                    defender_row["direct_responsibility"]
                ),
                "response_count": len(responses),
                "runner_follow_response": runner_response,
                "runner_follow_anchor": follow_anchor,
                "runner_follow_option": runner_follow_option,
                "actual_response": actual,
                "selected_beneficiary_id": selection.selected_player_id,
                "selected_beneficiary_name": selected_name,
                "candidates": candidates,
            }
        )
        print(
            f"    {metadata.players[defender_id].short_name}: "
            f"{len(responses)} responses -> {selected_name or 'none'}",
            flush=True,
        )

    return {
        "schema_version": "conditional-defender-game-v0.3",
        "match_id": scene["match_id"],
        "match_label": scene["match_label"],
        "onset_frame_id": onset,
        "runner_id": scene["runner_id"],
        "runner_name": scene["runner_name"],
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "ball_owner_id": owner_id,
        "ball_owner_name": metadata.players[owner_id].short_name,
        "ball_owner_distance_m": float(owner_distance),
        "release_times_s": list(release_times),
        "release_window_reason": release_reason,
        "kick_detected": bool(kick.detected),
        "horizon_seconds": horizon_s,
        "minimum_relative_cross_cost": float(minimum_relative_cross_cost),
        "runner_only_shortlist": [
            {
                "defender_id": row["defender_id"],
                "defender_name": metadata.players[row["defender_id"]].short_name,
                "position": metadata.players[row["defender_id"]].position,
                "direct_responsibility": float(row["direct_responsibility"]),
            }
            for row in defender_rows
        ],
        "branches": branches,
        "owner_reachable_endpoint_count": len(owner_actions),
        "background_frames": scene["background_frames"],
        "review": scene["review"],
        "names": {
            player_id: player.short_name for player_id, player in metadata.players.items()
        },
        "notes": [
            "Relevant defenders are shortlisted using runner-only responsibility at t=0; no beneficiary value enters this step.",
            "Each defender receives an independent delayed bounded-control response set.",
            "Beneficiary is conditional on the responding defender, not a scene-global player label.",
            "Off-ball teammates use pass reception probability multiplied by geometric goal danger.",
            "The ball owner uses a forward bounded-control carry endpoint: endpoint retention probability multiplied by positive geometric goal-danger gain.",
            "A provisional 5% relative gate suppresses numerical beneficiary noise; it is not a population-calibrated final threshold.",
        ],
    }


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.input_json.read_text(encoding="utf-8"))
    if args.scene_index:
        requested = set(args.scene_index)
        scenes = [
            scene for index, scene in enumerate(scenes, start=1) if index in requested
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, scene in enumerate(scenes, start=1):
        print(
            f"[{index}/{len(scenes)}] {scene['runner_name']} "
            f"@ {scene['match_id']}:{scene['onset_frame_id']}",
            flush=True,
        )
        results.append(
            _evaluate_scene(
                scene,
                args.data_dir,
                args.relevant_defender_count,
                args.release_step_seconds,
                args.control_distance_m,
                args.minimum_relative_cross_cost,
            )
        )
    output = args.output_dir / "conditional_defender_game_v0_3.json"
    output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "conditional-defender-game-v0.3",
        "scene_count": len(results),
        "status": "human-audit diagnostic",
        "defender_shortlist": "top runner-only t=0 responsibility",
        "beneficiary_definition": "conditional on responding defender",
        "minimum_relative_cross_cost": args.minimum_relative_cross_cost,
        "ball_owner_option": "bounded-control forward carry space",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
