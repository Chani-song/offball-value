#!/usr/bin/env python3
"""Scene-level causal attribution diagnostic for the local off-ball game.

v0.4 corrects three issues exposed by human review of scene 1:

1. pass and carry use the same absolute ``success × endpoint danger`` contract;
2. progress gain is reported separately from absolute option threat;
3. an observed runner action is compared with a dynamically feasible neutral
   action before a teammate is called runner-created.

The script intentionally defaults to the first audited scene.  It is a
diagnostic contract test before population-scale extraction.
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
from evaluate_conditional_defender_game_v0_3 import (
    _actual_response,
    _carry_path_until,
    _carry_point,
    _reachability_config,
    _response_payload,
    _runner_only_defenders,
    _unique_endpoint_actions,
)
from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.causal_attribution import attribute_option
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
from offball_value.goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    influence_pitch_grid,
)
from offball_value.post_reception_value import (
    combine_delivery_and_accessibility,
    goal_side_accessibility,
)


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
        default=Path("data/processed/causal_attribution_v0_4"),
    )
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--relevant-defender-count", type=int, default=3)
    parser.add_argument("--release-step-seconds", type=float, default=0.2)
    parser.add_argument("--control-distance-m", type=float, default=1.5)
    parser.add_argument("--minimum-relative-effect", type=float, default=0.05)
    return parser.parse_args()


def _neutral_action(actions, runner, direction, initial_velocity):
    """Choose a feasible monotone braking/coast action, never a frozen player."""

    speed = math.hypot(*initial_velocity)
    heading = (
        (initial_velocity[0] / speed, initial_velocity[1] / speed)
        if speed > 1e-9
        else (float(direction), 0.0)
    )
    rows = []
    for index, action in enumerate(actions):
        dx = action.endpoint_x - runner.x
        dy = action.endpoint_y - runner.y
        displacement = math.hypot(dx, dy)
        progress = direction * dx
        path_longitudinal = []
        path_lateral = []
        for x, y in action.full_path_xy:
            path_dx, path_dy = x - runner.x, y - runner.y
            path_longitudinal.append(
                path_dx * heading[0] + path_dy * heading[1]
            )
            path_lateral.append(
                abs(-path_dx * heading[1] + path_dy * heading[0])
            )
        monotone = all(
            path_longitudinal[index] >= path_longitudinal[index - 1] - 0.15
            for index in range(1, len(path_longitudinal))
        )
        rows.append(
            {
                "displacement": displacement,
                "terminal_longitudinal": float(path_longitudinal[-1]),
                "maximum_lateral": float(max(path_lateral)),
                "progress": float(progress),
                "effort": float(action.base_action.motion.effort_m2ps3),
                "monotone": monotone,
                "index": index,
                "action": action,
            }
        )
    straight_brakes = [
        row
        for row in rows
        if row["terminal_longitudinal"] >= -1e-6
        and row["maximum_lateral"] <= 0.5
        and row["monotone"]
        and row["action"].base_action.motion.maneuver_type
        == "continuous_steering"
    ]
    if straight_brakes:
        selected = min(
            straight_brakes,
            key=lambda row: (
                row["terminal_longitudinal"],
                row["effort"],
                row["maximum_lateral"],
                row["index"],
            ),
        )
        selection_rule = (
            "minimum monotone forward travel among feasible continuous paths "
            "with <=0.5 m maximum lateral deviation"
        )
    else:
        minimum_displacement = min(row["displacement"] for row in rows)
        near_hold = [
            row
            for row in rows
            if row["displacement"] <= minimum_displacement + 0.75
        ]
        selected = min(
            near_hold,
            key=lambda row: (
                row["effort"],
                max(0.0, row["progress"]),
                abs(row["progress"]),
                row["index"],
            ),
        )
        selection_rule = (
            "fallback: within 0.75 m of minimum feasible displacement, then "
            "minimum effort"
        )
    action = selected["action"]
    return action, {
        "selection_rule": selection_rule,
        "endpoint_displacement_m": float(selected["displacement"]),
        "initial_heading_travel_m": float(selected["terminal_longitudinal"]),
        "maximum_initial_heading_lateral_m": float(
            selected["maximum_lateral"]
        ),
        "attacking_progress_m": float(
            direction * (action.endpoint_x - runner.x)
        ),
        "effort_m2ps3": float(action.base_action.motion.effort_m2ps3),
    }


def _action_as_response(action, index=0):
    payload = _response_payload(action, index)
    payload["path_txy"] = [
        [float(time_s), float(point[0]), float(point[1])]
        for time_s, point in zip(
            action.response_path_times_s, action.full_path_xy
        )
    ]
    return payload


def _scenario_state(base_frame, base_velocities, runner_id, runner_action, time_s):
    if runner_action is None or time_s <= 0.0:
        return base_frame, dict(base_velocities)
    x, y, vx, vy = _carry_point(runner_action, time_s)
    runner = base_frame.players[runner_id]
    frame = base_frame.with_player(
        runner_id,
        replace(runner, x=x, y=y, speed=math.hypot(vx, vy) * 3.6),
    )
    velocities = dict(base_velocities)
    velocities[runner_id] = VelocityEstimate(vx, vy, math.hypot(vx, vy), 0, 0.0)
    return frame, velocities


def _scenario_background(scene, runner_id, runner_action):
    if runner_action is None:
        return scene["background_frames"]
    output = []
    for source in scene["background_frames"]:
        frame = {**source, "players": [list(player) for player in source["players"]]}
        if frame["time_s"] >= 0.0:
            x, y, _, _ = _carry_point(runner_action, float(frame["time_s"]))
            for player in frame["players"]:
                if player[0] == runner_id:
                    player[2], player[3] = x, y
                    break
        output.append(frame)
    return output


def _carry_absolute_value(
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
    """Return absolute carry threat and a separately labelled progress gain."""

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
    arrivals = sorted(
        (
            float(
                player_time_to_point(
                    defender,
                    event_velocities.get(defender.object_id, zero_velocity),
                    (float(event_xy[0]), float(event_xy[1])),
                    arrival_config,
                )
            ),
            defender.object_id,
        )
        for defender in event_frame.players.values()
        if defender.team_id != team_id
    )
    survivals = [
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
        for arrival_time, _ in arrivals
    ]
    retention = float(np.prod(survivals))
    endpoint_goal = float(_geometric_value([event_xy], direction)[0])
    goal_gain = max(0.0, endpoint_goal - onset_goal_value)
    nearest = arrivals[0] if arrivals else (math.inf, None)
    support = arrivals[1] if len(arrivals) > 1 else (math.inf, None)
    return retention * endpoint_goal, {
        "success_probability": retention,
        "receive_probability": retention,
        "goal_value": endpoint_goal,
        "endpoint_goal_value": endpoint_goal,
        "onset_goal_value": float(onset_goal_value),
        "goal_gain": goal_gain,
        "progress_gain": retention * goal_gain,
        "path_survival_probability": retention,
        "receiver_first_probability": 1.0,
        "nearest_pressure_defender_id": nearest[1],
        "nearest_pressure_time_s": float(nearest[0]),
        "support_pressure_defender_id": support[1],
        "support_pressure_time_s": float(support[0]),
    }


def _evaluate_scene(scene, args):
    files = find_bundesliga_files(args.data_dir, scene["match_id"])
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    onset = int(scene["onset_frame_id"])
    horizon_s = float(scene["horizon_seconds"])
    final = onset + int(round(horizon_s * FPS))
    frame_map = load_bundesliga_frames(files["positions"], range(onset - 10, final + 1))
    history = tuple(frame_map[index] for index in range(onset - 10, onset + 1))
    decision = frame_map[onset]
    team_id = scene["attacking_team_id"]
    direction = int(scene["attacking_direction"])
    runner_id = scene["runner_id"]
    owner_id, owner_distance, controlled = controlled_ball_owner(decision, team_id)
    if not controlled or owner_id is None:
        raise ValueError("decision frame has no controlled ball owner")
    release_times, release_reason, kick = _release_times(
        frame_map,
        onset,
        horizon_s,
        owner_id,
        args.release_step_seconds,
        args.control_distance_m,
    )
    arrival_config = ArrivalModelConfig()
    region_config = ReceptionRegionConfig(resolution_m=2.0)
    value_contract = getattr(args, "value_contract", "endpoint_goal_danger")
    use_goal_side_accessibility = value_contract == "goal_side_accessibility"
    influence_config = GoalWeightedInfluenceConfig(
        grid_resolution_m=float(getattr(args, "influence_grid_resolution_m", 2.0))
    )
    if use_goal_side_accessibility:
        influence_xgrid, influence_ygrid = influence_pitch_grid(influence_config)
    else:
        influence_xgrid = influence_ygrid = None
    defending_team_ids = {
        player.team_id
        for player in decision.players.values()
        if player.team_id != team_id
    }
    defending_goalkeeper_ids = tuple(
        goalkeeper_id
        for defender_team_id in sorted(defending_team_ids)
        if (goalkeeper_id := metadata.goalkeeper_id(defender_team_id)) is not None
    )
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
    attacker_ids = tuple(
        sorted(
            player_id
            for player_id, player in decision.players.items()
            if player.team_id == team_id and player_id != goalkeeper_id
        )
    )

    # Generate the two attacker-side action sets from identical t=0 history.
    runner_generated = generate_defender_response_actions(
        decision,
        history,
        runner_id,
        DefenderResponseConfig(horizon_seconds=horizon_s, response_delay_seconds=0.0),
        _reachability_config(),
    )
    runner_actions = _unique_endpoint_actions(runner_generated.actions)
    neutral_action, neutral_diagnostics = _neutral_action(
        runner_actions,
        decision.players[runner_id],
        direction,
        (runner_generated.initial_vx_mps, runner_generated.initial_vy_mps),
    )
    neutral_response = _action_as_response(neutral_action)
    actual_runner_path = [
        [
            float(frame["time_s"]),
            float(next(p for p in frame["players"] if p[0] == runner_id)[2]),
            float(next(p for p in frame["players"] if p[0] == runner_id)[3]),
        ]
        for frame in scene["background_frames"]
        if frame["time_s"] >= 0.0
    ]
    actual_runner_endpoint = actual_runner_path[-1]
    neutral_diagnostics.update(
        {
            "endpoint_x": float(neutral_action.endpoint_x),
            "endpoint_y": float(neutral_action.endpoint_y),
            "observed_endpoint_x": float(actual_runner_endpoint[1]),
            "observed_endpoint_y": float(actual_runner_endpoint[2]),
            "observed_endpoint_displacement_m": math.hypot(
                actual_runner_endpoint[1] - decision.players[runner_id].x,
                actual_runner_endpoint[2] - decision.players[runner_id].y,
            ),
            "feasible_action_count": len(runner_actions),
        }
    )

    owner_generated = generate_defender_response_actions(
        decision,
        history,
        owner_id,
        DefenderResponseConfig(horizon_seconds=horizon_s, response_delay_seconds=0.0),
        _reachability_config(),
    )
    owner_actions = _unique_endpoint_actions(owner_generated.actions)
    onset_goal = float(
        _geometric_value(
            [(decision.players[owner_id].x, decision.players[owner_id].y)], direction
        )[0]
    )

    defender_rows = _runner_only_defenders(scene, args.relevant_defender_count)
    defender_response_sets = {}
    for row in defender_rows:
        defender_id = str(row["defender_id"])
        generated = generate_defender_response_actions(
            decision,
            history,
            defender_id,
            DefenderResponseConfig(
                horizon_seconds=horizon_s, response_delay_seconds=0.2
            ),
            _reachability_config(),
        )
        defender_response_sets[defender_id] = [
            _response_payload(action, index)
            for index, action in enumerate(_unique_endpoint_actions(generated.actions))
        ]

    def scenario_state(scenario, time_s):
        base_frame, base_velocities = base_states[time_s]
        return _scenario_state(
            base_frame,
            base_velocities,
            runner_id,
            neutral_action if scenario == "neutral" else None,
            time_s,
        )

    def apply_value_contract(
        frame,
        velocities,
        player_id,
        event_xy,
        endpoint_only_value,
        detail,
        event_velocity_xy=None,
    ):
        enriched = {
            **detail,
            "endpoint_only_value": float(endpoint_only_value),
            "value_contract": value_contract,
        }
        if not use_goal_side_accessibility:
            enriched.update(
                {
                    "post_reception_accessibility": 1.0,
                    "goal_side_covered_fraction": 0.0,
                }
            )
            return float(endpoint_only_value), enriched
        accessibility = goal_side_accessibility(
            frame,
            velocities,
            player_id,
            team_id,
            direction,
            (float(event_xy[0]), float(event_xy[1])),
            event_velocity_xy=event_velocity_xy,
            goalkeeper_ids=defending_goalkeeper_ids,
            config=influence_config,
            xgrid=influence_xgrid,
            ygrid=influence_ygrid,
        )
        delivery = float(
            detail.get("success_probability", detail["receive_probability"])
        )
        goal = float(detail.get("endpoint_goal_value", detail["goal_value"]))
        value = combine_delivery_and_accessibility(
            delivery,
            goal,
            accessibility.accessibility_score,
        )
        enriched.update(
            {
                "post_reception_accessibility": float(
                    accessibility.accessibility_score
                ),
                "goal_side_covered_fraction": float(
                    accessibility.covered_fraction
                ),
                "goal_side_intrinsic_value": float(
                    accessibility.intrinsic_value
                ),
                "goal_side_residual_value": float(accessibility.residual_value),
                "goal_side_residual_peak": float(accessibility.residual_peak),
                "goal_side_residual_centroid_x": float(
                    accessibility.residual_centroid_x
                ),
                "goal_side_residual_centroid_y": float(
                    accessibility.residual_centroid_y
                ),
            }
        )
        return value, enriched

    def build_catalogue(scenario):
        catalog = {}
        for player_id in attacker_ids:
            if player_id == owner_id:
                best = None
                start_x = float(decision.players[owner_id].x)
                for time_s in release_times[1:]:
                    frame, velocities = scenario_state(scenario, time_s)
                    for action_index, action in enumerate(owner_actions):
                        x, y, vx, vy = _carry_point(action, time_s)
                        if direction * (x - start_x) < 1.0:
                            continue
                        value, detail = _carry_absolute_value(
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
                        value, detail = apply_value_contract(
                            frame,
                            velocities,
                            owner_id,
                            (x, y),
                            value,
                            detail,
                            event_velocity_xy=(vx, vy),
                        )
                        rank = (
                            value,
                            detail["progress_gain"],
                            -time_s,
                            -float(action.base_action.motion.effort_m2ps3),
                        )
                        if best is None or rank > best[0]:
                            best = (
                                rank,
                                {
                                    "player_id": player_id,
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
                if best is not None:
                    catalog[player_id] = best[1]
                continue
            options = []
            for time_s in release_times[1:]:
                frame, velocities = scenario_state(scenario, time_s)
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
                option_value, option_detail = apply_value_contract(
                    frame,
                    velocities,
                    player_id,
                    (float(option["event_x"]), float(option["event_y"])),
                    float(option["value"]),
                    {
                        **option,
                        "success_probability": float(
                            option["receive_probability"]
                        ),
                        "endpoint_goal_value": float(option["goal_value"]),
                    },
                )
                progress = float(option["receive_probability"]) * max(
                    0.0, float(option["goal_value"]) - onset_goal
                )
                options.append(
                    (
                        option_value,
                        progress,
                        -time_s,
                        {
                            "player_id": player_id,
                            **option_detail,
                            "value": float(option_value),
                            "time_s": float(time_s),
                            "success_probability": float(
                                option["receive_probability"]
                            ),
                            "endpoint_goal_value": float(option["goal_value"]),
                            "onset_goal_value": onset_goal,
                            "goal_gain": max(
                                0.0, float(option["goal_value"]) - onset_goal
                            ),
                            "progress_gain": progress,
                        },
                    )
                )
            catalog[player_id] = max(options, key=lambda item: item[:3])[3]
        return catalog

    def score_fixed_option(scenario, branch_scene, response, option):
        time_s = float(option["time_s"])
        frame, velocities = scenario_state(scenario, time_s)
        frame, velocities = _virtual_state(
            frame, velocities, branch_scene, response, time_s
        )
        if option["option_type"] == "forward_carry_space":
            value, detail = _carry_absolute_value(
                frame,
                velocities,
                owner_id,
                team_id,
                direction,
                (float(option["event_x"]), float(option["event_y"])),
                (float(option["event_vx"]), float(option["event_vy"])),
                onset_goal,
                arrival_config,
            )
            return apply_value_contract(
                frame,
                velocities,
                owner_id,
                (float(option["event_x"]), float(option["event_y"])),
                value,
                detail,
                event_velocity_xy=(
                    float(option["event_vx"]),
                    float(option["event_vy"]),
                ),
            )
        value, detail = _fixed_event_value(
            frame,
            velocities,
            option["player_id"],
            owner_id,
            team_id,
            direction,
            option["option_type"],
            (float(option["event_x"]), float(option["event_y"])),
            arrival_config,
        )
        detail = {
            **detail,
            "success_probability": float(detail["receive_probability"]),
            "endpoint_goal_value": float(detail["goal_value"]),
            "onset_goal_value": onset_goal,
            "goal_gain": max(0.0, float(detail["goal_value"]) - onset_goal),
            "progress_gain": float(detail["receive_probability"])
            * max(0.0, float(detail["goal_value"]) - onset_goal),
        }
        return apply_value_contract(
            frame,
            velocities,
            option["player_id"],
            (float(option["event_x"]), float(option["event_y"])),
            float(value),
            detail,
        )

    scenario_results = {}
    for scenario, runner_action in (("actual_run", None), ("neutral", neutral_action)):
        catalog = build_catalogue(scenario)
        scenario_scene = {
            **scene,
            "background_frames": _scenario_background(
                scene, runner_id, runner_action
            ),
        }
        branches = []
        for row in defender_rows:
            defender_id = str(row["defender_id"])
            responses = defender_response_sets[defender_id]
            branch_scene = {
                **scenario_scene,
                "defender_id": defender_id,
                "defender_name": metadata.players[defender_id].short_name,
                "responses": responses,
            }
            follow_anchor = _select_runner_follow_response(branch_scene)
            follow_response = responses[int(follow_anchor["index"])]
            value_matrix = np.zeros((len(attacker_ids), len(responses)), dtype=float)
            detail_rows = {}
            for player_index, player_id in enumerate(attacker_ids):
                option = catalog[player_id]
                details = []
                for response_index, response in enumerate(responses):
                    value, detail = score_fixed_option(
                        scenario, branch_scene, response, option
                    )
                    value_matrix[player_index, response_index] = value
                    details.append(detail)
                detail_rows[player_id] = details
            meaningful = [
                index
                for index, player_id in enumerate(attacker_ids)
                if catalog[player_id]["legal"]
                and direction * float(catalog[player_id]["event_x"]) > 0.0
            ]
            team_values = np.max(value_matrix[meaningful], axis=0)
            team_totals = np.sum(value_matrix[meaningful], axis=0)
            best_response_index = min(
                range(len(responses)),
                key=lambda index: (
                    team_values[index],
                    team_totals[index] if use_goal_side_accessibility else 0.0,
                    float(responses[index]["effort_m2ps3"]),
                    index,
                ),
            )
            option_rows = []
            follow_index = int(follow_anchor["index"])
            for player_index, player_id in enumerate(attacker_ids):
                option = catalog[player_id]
                cover_index = min(
                    range(len(responses)),
                    key=lambda index: (
                        value_matrix[player_index, index],
                        float(responses[index]["effort_m2ps3"]),
                        index,
                    ),
                )
                follow_value = float(value_matrix[player_index, follow_index])
                cover_value = float(value_matrix[player_index, cover_index])
                follow_detail = detail_rows[player_id][follow_index]
                cover_detail = detail_rows[player_id][cover_index]
                option_rows.append(
                    {
                        "player_id": player_id,
                        "player_name": metadata.players[player_id].short_name,
                        "position": metadata.players[player_id].position,
                        "is_runner": player_id == runner_id,
                        "is_ball_owner": player_id == owner_id,
                        "option_type": option["option_type"],
                        "legal": bool(option["legal"]),
                        "event_x": float(option["event_x"]),
                        "event_y": float(option["event_y"]),
                        "peak_time_s": float(option["time_s"]),
                        "carry_path_txy": option.get("carry_path_txy"),
                        "catalogue_absolute_value": float(option["value"]),
                        "catalogue_progress_gain": float(option["progress_gain"]),
                        "follow_value": follow_value,
                        "cover_value": cover_value,
                        "allocation_effect": follow_value - cover_value,
                        "allocation_effect_fraction": (follow_value - cover_value)
                        / max(abs(follow_value), 1e-6),
                        "follow_components": follow_detail,
                        "cover_components": cover_detail,
                        "cover_response_index": int(cover_index),
                        "cover_response": responses[cover_index],
                        "best_response_value": float(
                            value_matrix[player_index, best_response_index]
                        ),
                        "best_response_components": detail_rows[player_id][
                            best_response_index
                        ],
                        "mean_follow_cover_separation_m": _mean_path_separation(
                            follow_response, responses[cover_index]
                        ),
                    }
                )
            branches.append(
                {
                    "defender_id": defender_id,
                    "defender_name": metadata.players[defender_id].short_name,
                    "defender_position": metadata.players[defender_id].position,
                    "runner_only_responsibility": float(
                        row["direct_responsibility"]
                    ),
                    "response_count": len(responses),
                    "runner_follow_anchor": follow_anchor,
                    "runner_follow_response": follow_response,
                    "best_response_index": int(best_response_index),
                    "best_response": responses[best_response_index],
                    "best_team_residual": float(team_values[best_response_index]),
                    "best_team_total": float(team_totals[best_response_index]),
                    "best_response_rule": (
                        "lexicographic min(max option Q, sum option Q, effort)"
                        if use_goal_side_accessibility
                        else "lexicographic min(max option Q, effort)"
                    ),
                    "best_team_option_id": attacker_ids[
                        meaningful[
                            int(np.argmax(value_matrix[meaningful, best_response_index]))
                        ]
                    ],
                    "actual_response": _actual_response(scene, defender_id),
                    "options": option_rows,
                }
            )
        global_branch_index = min(
            range(len(branches)),
            key=lambda index: (
                branches[index]["best_team_residual"],
                -branches[index]["runner_only_responsibility"],
                branches[index]["defender_id"],
            ),
        )
        scenario_results[scenario] = {
            "runner_path_txy": (
                actual_runner_path
                if scenario == "actual_run"
                else neutral_response["path_txy"]
            ),
            "catalogue": [
                {
                    **catalog[player_id],
                    "player_name": metadata.players[player_id].short_name,
                    "position": metadata.players[player_id].position,
                }
                for player_id in attacker_ids
            ],
            "branches": branches,
            "global_best_branch_index": int(global_branch_index),
            "global_best_defender_id": branches[global_branch_index]["defender_id"],
            "global_best_defender_name": branches[global_branch_index]["defender_name"],
            "global_best_team_residual": branches[global_branch_index][
                "best_team_residual"
            ],
        }

    # Attach causal labels within each responding-defender branch.  This is the
    # comparison needed to distinguish Ginczek's pre-existing allocation
    # exposure from an option genuinely added by Klaus's run.
    attribution_by_defender = []
    actual_by_defender = {
        branch["defender_id"]: branch
        for branch in scenario_results["actual_run"]["branches"]
    }
    neutral_by_defender = {
        branch["defender_id"]: branch
        for branch in scenario_results["neutral"]["branches"]
    }
    for defender_id in actual_by_defender:
        actual_branch = actual_by_defender[defender_id]
        neutral_branch = neutral_by_defender[defender_id]
        actual_options = {row["player_id"]: row for row in actual_branch["options"]}
        neutral_options = {row["player_id"]: row for row in neutral_branch["options"]}
        rows = []
        for player_id in attacker_ids:
            actual_option = actual_options[player_id]
            neutral_option = neutral_options[player_id]
            result = attribute_option(
                actual_option["best_response_value"],
                neutral_option["best_response_value"],
                actual_option["allocation_effect"],
                actual_option["follow_value"],
                minimum_relative_effect=args.minimum_relative_effect,
            )
            rows.append(
                {
                    **asdict(result),
                    "player_id": player_id,
                    "player_name": metadata.players[player_id].short_name,
                    "position": metadata.players[player_id].position,
                    "is_runner": player_id == runner_id,
                    "is_ball_owner": player_id == owner_id,
                    "option_type": actual_option["option_type"],
                    "actual_event_x": actual_option["event_x"],
                    "actual_event_y": actual_option["event_y"],
                    "actual_event_time_s": actual_option["peak_time_s"],
                    "neutral_event_x": neutral_option["event_x"],
                    "neutral_event_y": neutral_option["event_y"],
                    "neutral_event_time_s": neutral_option["peak_time_s"],
                }
            )
        rows.sort(
            key=lambda row: (
                row["is_runner"],
                -row["causal_gain"],
                -row["allocation_effect"],
                row["player_id"],
            )
        )
        attribution_by_defender.append(
            {
                "defender_id": defender_id,
                "defender_name": actual_branch["defender_name"],
                "options": rows,
            }
        )

    return {
        "schema_version": getattr(
            args, "schema_version", "causal-attribution-v0.4"
        ),
        "match_id": scene["match_id"],
        "match_label": scene["match_label"],
        "onset_frame_id": onset,
        "horizon_seconds": horizon_s,
        "runner_id": runner_id,
        "runner_name": scene["runner_name"],
        "ball_owner_id": owner_id,
        "ball_owner_name": metadata.players[owner_id].short_name,
        "ball_owner_distance_m": float(owner_distance),
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "release_times_s": list(release_times),
        "release_window_reason": release_reason,
        "kick_detected": bool(kick.detected),
        "minimum_relative_effect": float(args.minimum_relative_effect),
        "value_contract": value_contract,
        "influence": (
            {
                **asdict(influence_config),
                "defending_goalkeepers_excluded": list(
                    defending_goalkeeper_ids
                ),
            }
            if use_goal_side_accessibility
            else None
        ),
        "neutral_action": {
            **neutral_diagnostics,
            "path_txy": neutral_response["path_txy"],
        },
        "observed_runner_path_txy": actual_runner_path,
        "owner_reachable_endpoint_count": len(owner_actions),
        "scenarios": scenario_results,
        "attribution_by_defender": attribution_by_defender,
        "background_frames": scene["background_frames"],
        "review": scene["review"],
        "names": {
            player_id: player.short_name for player_id, player in metadata.players.items()
        },
        "notes": [
            (
                "Pass and carry values equal delivery probability multiplied by endpoint goal danger and post-reception goal-side accessibility."
                if use_goal_side_accessibility
                else "Pass and carry absolute values both equal success probability multiplied by endpoint goal danger."
            ),
            (
                "Post-reception accessibility is residual goal-weighted Fernandez influence after outfield-defender suppression; it is a score, not a calibrated shot probability."
                if use_goal_side_accessibility
                else "Post-reception accessibility is disabled in the v0.4 endpoint-only contract."
            ),
            "Progress gain is a separate success-weighted increase from the ball owner's onset goal danger.",
            "The neutral runner action is dynamically feasible and selected without target-scene future.",
            "Causal gain compares defender-specific minimax responses under actual and neutral runner actions.",
            "Allocation effect compares runner-follow and fixed-option-cover responses within the actual-run game.",
            "A pre-existing exposed label means allocation effect is material but runner causal gain is not.",
            "Only one defender is counterfactually moved; other players retain observed trajectories in this diagnostic.",
        ],
    }


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not 1 <= args.scene_index <= len(scenes):
        raise ValueError("scene-index is outside the available scene list")
    scene = scenes[args.scene_index - 1]
    print(
        f"scene {args.scene_index}: {scene['runner_name']} @ "
        f"{scene['match_id']}:{scene['onset_frame_id']}",
        flush=True,
    )
    result = _evaluate_scene(scene, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "causal_attribution_v0_4.json"
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "causal-attribution-v0.4",
        "scene_count": 1,
        "status": "scene-1 human-audit diagnostic",
        "absolute_value_contract": "success_probability × endpoint_goal_danger",
        "neutral_action": "minimum-displacement feasible low-effort action",
        "minimum_relative_effect": args.minimum_relative_effect,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
