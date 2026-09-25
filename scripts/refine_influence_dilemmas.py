#!/usr/bin/env python3
"""Full-resolution refinement of coarse influence-dilemma candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.defender_best_response import (
    DefenderBestResponseConfig,
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
from offball_value.influence_dilemma import analyze_pair_tradeoff
from offball_value.pass_dynamics import VelocityEstimate
from offball_value.steering_reachable import SteeringReachabilityConfig


TIMES = (0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-csv",
        type=Path,
        default=Path("data/processed/influence_dilemma_screen_v0_1/pair_screen.csv"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/influence_dilemma_refined_v0_1"),
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--include",
        action="append",
        default=["DFL-MAT-J03WOH:68806:DFL-OBJ-002FXT"],
        help="Always refine MATCH:FRAME:BENEFICIARY in addition to the top rows.",
    )
    return parser.parse_args()


def _velocity(velocities, player_id):
    value = velocities.get(player_id)
    return (float(value.vx), float(value.vy)) if value is not None else (0.0, 0.0)


def _surface(frame, velocities, player_id, xgrid, ygrid, config):
    player = frame.players[player_id]
    return fernandez_influence_surface(
        xgrid,
        ygrid,
        (float(player.x), float(player.y)),
        _velocity(velocities, player_id),
        (float(frame.ball.x), float(frame.ball.y)),
        config.maximum_influence_speed_mps,
    )


def _mean(values):
    times = np.asarray(TIMES, dtype=float)
    return float(np.trapezoid(values, times) / (times[-1] - times[0]))


def _timed_path(frame_map, frame_id, player_id, dense=False):
    times = (
        tuple(index / FPS for index in range(int(2.0 * FPS) + 1))
        if dense
        else (0.0, *TIMES)
    )
    return tuple(
        (
            float(time_s),
            float(frame_map[frame_id + int(round(time_s * FPS))].players[player_id].x),
            float(frame_map[frame_id + int(round(time_s * FPS))].players[player_id].y),
        )
        for time_s in times
    )


def _action_payload(action):
    return {
        "action_id": action.action_id,
        "times_s": list(action.response_path_times_s),
        "path_xy": [list(point) for point in action.full_path_xy],
        "endpoint_x": float(action.endpoint_x),
        "endpoint_y": float(action.endpoint_y),
        "effort_m2ps3": float(action.base_action.motion.effort_m2ps3),
    }


def _prepare_scene(row, frame_map, influence_config):
    frame_id = int(row.frame_id)
    team_id = str(row.attacking_team_id)
    direction = int(row.attacking_direction)
    runner_id = str(row.runner_id)
    beneficiary_id = str(row.beneficiary_id)
    defender_id = str(row.defender_id)
    history = tuple(frame_map[index] for index in range(frame_id - 10, frame_id + 1))
    decision = frame_map[frame_id]
    actions = generate_defender_response_actions(
        decision,
        history,
        defender_id,
        DefenderResponseConfig(response_delay_seconds=0.2),
        SteeringReachabilityConfig(),
    ).actions
    background = prepare_observed_background_sequence(
        frame_map,
        frame_id,
        DefenderBestResponseConfig(evaluation_times_s=TIMES),
    )
    xgrid, ygrid = influence_pitch_grid(influence_config)
    target_ids = (runner_id, beneficiary_id)
    values = np.zeros((2, len(actions), len(TIMES)), dtype=float)
    observed_values = np.zeros((2, len(TIMES)), dtype=float)

    for time_index, observed in enumerate(background.states):
        frame = observed.frame
        velocities = dict(observed.velocities)
        defender_surfaces = {
            player_id: _surface(
                frame, velocities, player_id, xgrid, ygrid, influence_config
            )
            for player_id, player in frame.players.items()
            if player.team_id != team_id
        }
        fixed_defense = (
            np.sum(list(defender_surfaces.values()), axis=0)
            - defender_surfaces[defender_id]
        )
        intrinsic = []
        for target_id in target_ids:
            target = frame.players[target_id]
            intrinsic.append(
                _surface(
                    frame, velocities, target_id, xgrid, ygrid, influence_config
                )
                * goal_weighted_space_surface(
                    xgrid,
                    ygrid,
                    (float(target.x), float(target.y)),
                    direction,
                    influence_config,
                )
            )
        intrinsic = np.asarray(intrinsic)
        denominators = np.maximum(np.sum(intrinsic, axis=(1, 2)), 1e-12)
        observed_uncovered = np.exp(
            -influence_config.defender_suppression_strength
            * np.sum(list(defender_surfaces.values()), axis=0)
        )
        observed_values[:, time_index] = (
            np.sum(
                intrinsic * observed_uncovered[None, :, :], axis=(1, 2)
            )
            / denominators
        )
        for action_index, action in enumerate(actions):
            x, y, vx, vy = interpolate_path_state(
                action.full_path_xy,
                action.response_path_times_s,
                observed.time_s,
            )
            virtual_frame = frame.with_player(
                defender_id,
                replace(
                    frame.players[defender_id],
                    x=x,
                    y=y,
                    speed=math.hypot(vx, vy) * 3.6,
                ),
            )
            virtual_velocities = dict(velocities)
            virtual_velocities[defender_id] = VelocityEstimate(
                vx, vy, math.hypot(vx, vy), 0, 0.0
            )
            virtual_defender = _surface(
                virtual_frame,
                virtual_velocities,
                defender_id,
                xgrid,
                ygrid,
                influence_config,
            )
            uncovered = np.exp(
                -influence_config.defender_suppression_strength
                * (fixed_defense + virtual_defender)
            )
            values[:, action_index, time_index] = (
                np.sum(intrinsic * uncovered[None, :, :], axis=(1, 2))
                / denominators
            )

    horizon = np.asarray(
        [[_mean(values[t, a, :]) for a in range(len(actions))]
         for t in range(2)]
    )
    tradeoff = analyze_pair_tradeoff(horizon[0], horizon[1])
    selected = {
        "direct_cover": actions[tradeoff.direct_action_index],
        "compromise": actions[tradeoff.compromise_action_index],
        "beneficiary_cover": actions[tradeoff.beneficiary_action_index],
    }
    selected_payload = {}
    for key, action in selected.items():
        index = actions.index(action)
        item = _action_payload(action)
        item.update(
            {
                "runner_horizon_mean": float(horizon[0, index]),
                "beneficiary_horizon_mean": float(horizon[1, index]),
                "points": [
                    {
                        "time_s": float(time_s),
                        "runner_residual_fraction": float(values[0, index, time_index]),
                        "beneficiary_residual_fraction": float(
                            values[1, index, time_index]
                        ),
                    }
                    for time_index, time_s in enumerate(TIMES)
                ],
            }
        )
        selected_payload[key] = item
    selected_payload["observed_reference"] = {
        "action_id": "observed_defender_future_for_audit",
        "times_s": [point[0] for point in _timed_path(
            frame_map, frame_id, defender_id, dense=True
        )],
        "path_xy": [list(point[1:]) for point in _timed_path(
            frame_map, frame_id, defender_id, dense=True
        )],
        "runner_horizon_mean": _mean(observed_values[0]),
        "beneficiary_horizon_mean": _mean(observed_values[1]),
        "points": [
            {
                "time_s": float(time_s),
                "runner_residual_fraction": float(observed_values[0, time_index]),
                "beneficiary_residual_fraction": float(
                    observed_values[1, time_index]
                ),
            }
            for time_index, time_s in enumerate(TIMES)
        ],
    }
    separation = math.hypot(
        selected["direct_cover"].endpoint_x
        - selected["beneficiary_cover"].endpoint_x,
        selected["direct_cover"].endpoint_y
        - selected["beneficiary_cover"].endpoint_y,
    )
    return {
        "match_id": str(row.match_id),
        "frame_id": frame_id,
        "runner_id": runner_id,
        "runner_name": str(row.runner_name),
        "beneficiary_id": beneficiary_id,
        "beneficiary_name": str(row.beneficiary_name),
        "defender_id": defender_id,
        "defender_name": str(row.defender_name),
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "evaluated_action_count": len(actions),
        **asdict(tradeoff),
        "direct_terminal_separation_m": float(separation),
        "responses": selected_payload,
        "runner_actual_timed": _timed_path(frame_map, frame_id, runner_id, dense=True),
        "beneficiary_actual_timed": _timed_path(
            frame_map, frame_id, beneficiary_id, dense=True
        ),
        "defender_actual_timed": _timed_path(
            frame_map, frame_id, defender_id, dense=True
        ),
        "background_frames": [
            {
                "time_s": index / FPS,
                "players": [
                    [player_id, player.team_id, player.x, player.y]
                    for player_id, player in frame_map[frame_id + index].players.items()
                ],
                "ball": [
                    frame_map[frame_id + index].ball.x,
                    frame_map[frame_id + index].ball.y,
                ],
            }
            for index in range(int(2.0 * FPS) + 1)
        ],
    }


def main() -> None:
    args = parse_args()
    screen = pd.read_csv(args.screen_csv, dtype=str)
    numeric = [
        "frame_id",
        "screening_strength",
        "minimum_branch_gap",
        "normalized_compromise_regret",
        "attacking_direction",
    ]
    for column in numeric:
        screen[column] = pd.to_numeric(screen[column])
    selected = screen.sort_values(
        ["screening_strength", "minimum_branch_gap"], ascending=False
    ).head(args.top)
    include_rows = []
    for spec in args.include:
        match_id, frame_text, beneficiary_id = spec.split(":")
        match = screen[
            (screen.match_id == match_id)
            & (screen.frame_id == int(frame_text))
            & (screen.beneficiary_id == beneficiary_id)
        ]
        if match.empty:
            raise ValueError(f"included pair is absent from screen: {spec}")
        include_rows.append(match.iloc[[0]])
    if include_rows:
        selected = pd.concat([selected, *include_rows], ignore_index=True)
    selected = selected.drop_duplicates(
        ["match_id", "frame_id", "runner_id", "beneficiary_id", "defender_id"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = GoalWeightedInfluenceConfig(grid_resolution_m=1.0)
    results = []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        started = time.perf_counter()
        files = find_bundesliga_files(args.data_dir, row.match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        frame_id = int(row.frame_id)
        frame_map = load_bundesliga_frames(
            files["positions"], range(frame_id - 10, frame_id + 51)
        )
        result = _prepare_scene(row, frame_map, config)
        results.append(result)
        print(
            f"[{number}/{len(selected)}] frame {frame_id} "
            f"{row.runner_name}–{row.beneficiary_name} · "
            f"min gap {result['minimum_branch_gap']:.3f} · "
            f"sep {result['direct_terminal_separation_m']:.1f}m · "
            f"{time.perf_counter()-started:.1f}s",
            flush=True,
        )
    results.sort(key=lambda item: item["screening_strength"], reverse=True)
    output = args.output_dir / "refined_scenes.json"
    output.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    columns = [
        key
        for key in results[0]
        if key not in {"responses", "runner_actual_timed", "beneficiary_actual_timed",
                       "defender_actual_timed", "background_frames"}
    ]
    pd.DataFrame([{key: item[key] for key in columns} for item in results]).to_csv(
        args.output_dir / "refined_summary.csv", index=False
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "influence-dilemma-refined-v0.1",
                "status": "full-resolution_scene_diagnostic",
                "scene_count": len(results),
                "evaluation_times_s": TIMES,
                "influence": asdict(config),
                "reachability": asdict(SteeringReachabilityConfig()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
