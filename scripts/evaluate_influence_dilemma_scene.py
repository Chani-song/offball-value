#!/usr/bin/env python3
"""Evaluate one target-pair dilemma with goal-weighted player influence.

This is deliberately a scene diagnostic.  It fixes one defender (the current
runner marker), holds the observed background for all non-intervened players,
and compares every physically feasible response to one virtual runner path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
    build_local_counterfactual_state,
    interpolate_path_state,
    prepare_observed_background_sequence,
)
from offball_value.defender_response import (
    DefenderResponseConfig,
    generate_defender_response_actions,
)
from offball_value.fernandez_influence import fernandez_influence_surface
from offball_value.goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    goal_weighted_space_surface,
    influence_pitch_grid,
)
from offball_value.pass_dynamics import VelocityEstimate
from offball_value.steering_reachable import SteeringReachabilityConfig


TIMES = (0.2, 0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WOH")
    parser.add_argument("--frame-id", type=int, default=68836)
    parser.add_argument("--attack-rank", type=int, default=1)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/attacker_maximin_v0_1"),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _velocity(
    player_id: str, velocities: dict[str, VelocityEstimate]
) -> tuple[float, float]:
    velocity = velocities.get(player_id)
    return (
        (float(velocity.vx), float(velocity.vy))
        if velocity is not None
        else (0.0, 0.0)
    )


def _influence(frame, velocities, player_id, xgrid, ygrid, config):
    player = frame.players[player_id]
    return fernandez_influence_surface(
        xgrid,
        ygrid,
        (float(player.x), float(player.y)),
        _velocity(player_id, velocities),
        (float(frame.ball.x), float(frame.ball.y)),
        config.maximum_influence_speed_mps,
    )


def _mean(values: list[float]) -> float:
    times = np.asarray(TIMES, dtype=float)
    return float(np.trapezoid(np.asarray(values), times) / (times[-1] - times[0]))


def _summary(action, points):
    g_values = [point["runner_residual_fraction"] for point in points]
    k_values = [point["beneficiary_residual_fraction"] for point in points]
    g_mean = _mean(g_values)
    k_mean = _mean(k_values)
    return {
        "defender_action_id": action.action_id,
        "defender_path_times_s": list(action.response_path_times_s),
        "defender_path_xy": [list(point) for point in action.full_path_xy],
        "endpoint_x": float(action.endpoint_x),
        "endpoint_y": float(action.endpoint_y),
        "effort_m2ps3": float(action.base_action.motion.effort_m2ps3),
        "runner_horizon_mean": g_mean,
        "beneficiary_horizon_mean": k_mean,
        "pair_horizon_max": max(g_mean, k_mean),
        "runner_terminal_distance_m": points[-1]["runner_distance_m"],
        "beneficiary_terminal_distance_m": points[-1]["beneficiary_distance_m"],
        "runner_horizon_mean_distance_m": _mean(
            [point["runner_distance_m"] for point in points]
        ),
        "beneficiary_horizon_mean_distance_m": _mean(
            [point["beneficiary_distance_m"] for point in points]
        ),
        "points": points,
    }


def _score_prepared(
    prepared,
    defender_id,
    path_xy,
    path_times,
    config,
):
    points = []
    for state in prepared:
        if path_xy is None or path_times is None:
            virtual_frame = state["frame"]
            defender_sum = state["defender_sum"]
            defender = virtual_frame.players[defender_id]
            x, y = float(defender.x), float(defender.y)
        else:
            x, y, vx, vy = interpolate_path_state(
                path_xy, path_times, state["time_s"]
            )
            virtual_frame = state["frame"].with_player(
                defender_id,
                replace(
                    state["frame"].players[defender_id],
                    x=x,
                    y=y,
                    speed=math.hypot(vx, vy) * 3.6,
                ),
            )
            virtual_velocities = dict(state["velocities"])
            virtual_velocities[defender_id] = VelocityEstimate(
                vx, vy, math.hypot(vx, vy), 0, 0.0
            )
            virtual_defender = _influence(
                virtual_frame,
                virtual_velocities,
                defender_id,
                state["xgrid"],
                state["ygrid"],
                config,
            )
            defender_sum = (
                state["defender_sum"]
                - state["observed_selected_defender"]
                + virtual_defender
            )
        uncovered = np.exp(-config.defender_suppression_strength * defender_sum)
        row = {"time_s": state["time_s"]}
        for label in ("runner", "beneficiary"):
            intrinsic = state[f"{label}_intrinsic"]
            residual = intrinsic * uncovered
            row[f"{label}_residual_fraction"] = float(
                np.sum(residual) / max(float(np.sum(intrinsic)), 1e-12)
            )
            row[f"{label}_covered_fraction"] = float(
                1.0 - row[f"{label}_residual_fraction"]
            )
            target = virtual_frame.players[state[f"{label}_id"]]
            row[f"{label}_distance_m"] = float(
                math.hypot(x - target.x, y - target.y)
            )
        points.append(row)
    return points


def _prepare_influence_state(
    frame,
    velocities,
    time_s,
    runner_id,
    beneficiary_id,
    defender_id,
    attacking_team_id,
    attacking_direction,
    xgrid,
    ygrid,
    config,
):
    defender_ids = [
        player_id
        for player_id, player in frame.players.items()
        if player.team_id != attacking_team_id
    ]
    defender_surfaces = {
        player_id: _influence(frame, velocities, player_id, xgrid, ygrid, config)
        for player_id in defender_ids
    }
    state = {
        "time_s": float(time_s),
        "frame": frame,
        "velocities": velocities,
        "xgrid": xgrid,
        "ygrid": ygrid,
        "runner_id": runner_id,
        "beneficiary_id": beneficiary_id,
        "defender_sum": np.sum(list(defender_surfaces.values()), axis=0),
        "observed_selected_defender": defender_surfaces[defender_id],
    }
    for label, target_id in (
        ("runner", runner_id),
        ("beneficiary", beneficiary_id),
    ):
        target = frame.players[target_id]
        target_influence = _influence(
            frame, velocities, target_id, xgrid, ygrid, config
        )
        space_value = goal_weighted_space_surface(
            xgrid,
            ygrid,
            (float(target.x), float(target.y)),
            attacking_direction,
            config,
        )
        state[f"{label}_intrinsic"] = target_influence * space_value
    return state


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    directory = args.input_dir / match_id
    source = directory / f"frame_{args.frame_id}_attacker_maximin.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    attacks = sorted(
        payload["searched_attacks"],
        key=lambda item: item["best_response"]["value"],
        reverse=True,
    )
    attack = attacks[args.attack_rank - 1]
    targeted = attack["targeted_pass_window_dilemma"]
    runner_id = str(payload["attacker_id"])
    beneficiary_id = str(targeted["beneficiary_id"])
    # Keep the responsibility assignment coherent: the same incumbent marker
    # must choose between following the runner and protecting the other lane.
    defender_id = str(targeted["responses"]["direct_cover"]["selection"]["defender_id"])
    team_id = str(payload["attacking_team_id"])
    direction = int(payload["attacking_direction"])

    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frame_id = int(payload["frame_id"])
    frames = load_bundesliga_frames(
        files["positions"], range(frame_id - 10, frame_id + 51)
    )
    history = tuple(frames[index] for index in range(frame_id - 10, frame_id + 1))
    decision = frames[frame_id]
    background_config = DefenderBestResponseConfig(evaluation_times_s=TIMES)
    background = prepare_observed_background_sequence(
        frames, frame_id, background_config
    )
    reachability = SteeringReachabilityConfig(**payload["parameters"]["reachability"])
    response_config = DefenderResponseConfig(
        **payload["parameters"]["defender_response"]
    )
    action_set = generate_defender_response_actions(
        decision, history, defender_id, response_config, reachability
    )
    influence_config = GoalWeightedInfluenceConfig()
    xgrid, ygrid = influence_pitch_grid(influence_config)

    prepared = []
    actual_prepared = []
    for background_state in background.states:
        frame, velocities = build_local_counterfactual_state(
            background_state,
            runner_id,
            attack["attack_path_xy"],
            payload["attack_path_times_s"],
        )
        prepared.append(
            _prepare_influence_state(
                frame,
                velocities,
                background_state.time_s,
                runner_id,
                beneficiary_id,
                defender_id,
                team_id,
                direction,
                xgrid,
                ygrid,
                influence_config,
            )
        )
        # The observed card is a different world: both the runner and defender
        # retain their actual trajectories.  It is visual context, not another
        # defender response to the virtual attack.
        actual_prepared.append(
            _prepare_influence_state(
                background_state.frame,
                dict(background_state.velocities),
                background_state.time_s,
                runner_id,
                beneficiary_id,
                defender_id,
                team_id,
                direction,
                xgrid,
                ygrid,
                influence_config,
            )
        )

    evaluated = []
    for index, action in enumerate(action_set.actions, start=1):
        points = _score_prepared(
            prepared,
            defender_id,
            action.full_path_xy,
            action.response_path_times_s,
            influence_config,
        )
        evaluated.append(_summary(action, points))
        if index % args.progress_every == 0:
            print(f"evaluated {index}/{len(action_set.actions)}", flush=True)

    for rank, item in enumerate(
        sorted(
            evaluated,
            key=lambda value: (
                value["beneficiary_horizon_mean_distance_m"],
                value["beneficiary_terminal_distance_m"],
            ),
        ),
        start=1,
    ):
        item["beneficiary_mean_distance_rank"] = rank
    for rank, item in enumerate(
        sorted(
            evaluated,
            key=lambda value: (
                value["runner_horizon_mean_distance_m"],
                value["runner_terminal_distance_m"],
            ),
        ),
        start=1,
    ):
        item["runner_mean_distance_rank"] = rank

    direct = min(
        evaluated,
        key=lambda item: (
            item["runner_horizon_mean"],
            item["beneficiary_horizon_mean"],
            item["effort_m2ps3"],
        ),
    )
    beneficiary = min(
        evaluated,
        key=lambda item: (
            item["beneficiary_horizon_mean"],
            item["runner_horizon_mean"],
            item["effort_m2ps3"],
        ),
    )
    balanced = min(
        evaluated,
        key=lambda item: (
            item["pair_horizon_max"],
            item["runner_horizon_mean"] + item["beneficiary_horizon_mean"],
            item["effort_m2ps3"],
        ),
    )
    runner_span = max(
        beneficiary["runner_horizon_mean"] - direct["runner_horizon_mean"],
        1e-9,
    )
    beneficiary_span = max(
        direct["beneficiary_horizon_mean"]
        - beneficiary["beneficiary_horizon_mean"],
        1e-9,
    )
    compromise = min(
        evaluated,
        key=lambda item: (
            max(
                max(
                    0.0,
                    (item["runner_horizon_mean"] - direct["runner_horizon_mean"])
                    / runner_span,
                ),
                max(
                    0.0,
                    (
                        item["beneficiary_horizon_mean"]
                        - beneficiary["beneficiary_horizon_mean"]
                    )
                    / beneficiary_span,
                ),
            ),
            item["runner_horizon_mean"] + item["beneficiary_horizon_mean"],
            item["effort_m2ps3"],
        ),
    )

    actual_path = tuple(
        (frames[frame_id + int(round(time_s * 25))].players[defender_id].x,
         frames[frame_id + int(round(time_s * 25))].players[defender_id].y)
        for time_s in (0.0, *TIMES)
    )
    actual_points = _score_prepared(
        actual_prepared, defender_id, None, None, influence_config
    )
    actual = {
        "defender_action_id": "observed_future_for_audit_only",
        "defender_path_times_s": [0.0, *TIMES],
        "defender_path_xy": [list(point) for point in actual_path],
        "runner_horizon_mean": _mean(
            [point["runner_residual_fraction"] for point in actual_points]
        ),
        "beneficiary_horizon_mean": _mean(
            [point["beneficiary_residual_fraction"] for point in actual_points]
        ),
        "runner_terminal_distance_m": actual_points[-1]["runner_distance_m"],
        "beneficiary_terminal_distance_m": actual_points[-1][
            "beneficiary_distance_m"
        ],
        "runner_horizon_mean_distance_m": _mean(
            [point["runner_distance_m"] for point in actual_points]
        ),
        "beneficiary_horizon_mean_distance_m": _mean(
            [point["beneficiary_distance_m"] for point in actual_points]
        ),
        "points": actual_points,
    }
    actual["pair_horizon_max"] = max(
        actual["runner_horizon_mean"], actual["beneficiary_horizon_mean"]
    )

    result = {
        "schema_version": "influence-dilemma-scene-v0.1",
        "status": "diagnostic_not_final_threat_model",
        "match_id": match_id,
        "frame_id": frame_id,
        "runner_id": runner_id,
        "runner_name": metadata.players[runner_id].short_name,
        "beneficiary_id": beneficiary_id,
        "beneficiary_name": metadata.players[beneficiary_id].short_name,
        "defender_id": defender_id,
        "defender_name": metadata.players[defender_id].short_name,
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "attack_path_times_s": payload["attack_path_times_s"],
        "attack_path_xy": attack["attack_path_xy"],
        "evaluated_action_count": len(evaluated),
        "evaluation_times_s": list(TIMES),
        "responses": {
            "direct_cover": direct,
            "tradeoff_compromise": compromise,
            "beneficiary_cover": beneficiary,
            "balanced_cover": balanced,
            "observed_reference": actual,
        },
        "parameters": {
            "influence": asdict(influence_config),
            "reachability": payload["parameters"]["reachability"],
            "defender_response": payload["parameters"]["defender_response"],
        },
        "method_note": (
            "Fernandez player influence is preserved; goal-side/cone weights "
            "and exponential defensive suppression are explicit extensions."
        ),
    }
    output = directory / f"frame_{frame_id}_influence_dilemma.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")
    for key, item in result["responses"].items():
        print(
            f"{key:20s} G={item['runner_horizon_mean']:.4f} "
            f"K={item['beneficiary_horizon_mean']:.4f} "
            f"max={item['pair_horizon_max']:.4f}"
        )


if __name__ == "__main__":
    main()
