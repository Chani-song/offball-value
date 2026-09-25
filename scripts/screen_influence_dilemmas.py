#!/usr/bin/env python3
"""Screen reviewed run onsets for two-way influence trade-offs.

The screening pass uses the observed runner trajectory, the nearest current
outfield defender, a coarse 2 m influence grid, and one low-effort trajectory
per reachable defender endpoint.  It ranks scenes for later exact refinement;
it is not the final max--min experiment.
"""

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
    infer_attacking_direction,
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
from offball_value.reference_obso import offside_attacker_ids
from offball_value.steering_reachable import SteeringReachabilityConfig


TIMES = (0.4, 0.8, 1.2, 1.6, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=Path(
            "data/processed/run_onset_v0_3/combined/qc_round2/"
            "cumulative_primary_settled.csv"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/influence_dilemma_screen_v0_1"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--grid-resolution", type=float, default=2.0)
    parser.add_argument("--response-delay", type=float, default=0.2)
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


def _representative_responses(actions):
    best = {}
    for action in actions:
        cell = (
            action.base_action.endpoint_cell_x,
            action.base_action.endpoint_cell_y,
        )
        incumbent = best.get(cell)
        key = (action.base_action.motion.effort_m2ps3, action.action_id)
        if incumbent is None or key < (
            incumbent.base_action.motion.effort_m2ps3,
            incumbent.action_id,
        ):
            best[cell] = action
    return tuple(sorted(best.values(), key=lambda action: action.action_id))


def _timed_path(frame_map, frame_id, player_id):
    return tuple(
        (
            float(time_s),
            float(frame_map[frame_id + int(round(time_s * FPS))].players[player_id].x),
            float(frame_map[frame_id + int(round(time_s * FPS))].players[player_id].y),
        )
        for time_s in (0.0, *TIMES)
    )


def _mean(values):
    times = np.asarray(TIMES, dtype=float)
    return float(np.trapezoid(values, times) / (times[-1] - times[0]))


def _screen_scene(
    row,
    frame_map,
    metadata,
    influence_config,
    reachability,
    response_delay_seconds,
):
    frame_id = int(row.frame_id)
    runner_id = str(row.player_id)
    decision = frame_map[frame_id]
    team_id = decision.players[runner_id].team_id
    direction = infer_attacking_direction(decision, team_id, metadata)
    goalkeeper_ids = {
        goalkeeper_id
        for one_team in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(one_team)) is not None
    }
    defenders = [
        (math.hypot(player.x - decision.players[runner_id].x,
                    player.y - decision.players[runner_id].y), player_id)
        for player_id, player in decision.players.items()
        if player.team_id != team_id and player_id not in goalkeeper_ids
    ]
    _, defender_id = min(defenders)
    onset_offside = offside_attacker_ids(
        decision,
        team_id,
        direction,
        (decision.ball.x, decision.ball.y),
    )
    attacking_goalkeeper = metadata.goalkeeper_id(team_id)
    beneficiary_ids = [
        player_id
        for player_id, player in decision.players.items()
        if player.team_id == team_id
        and player_id not in {runner_id, row.ball_carrier_id, attacking_goalkeeper}
        and player_id not in onset_offside
    ]
    if not beneficiary_ids:
        return [], None

    history = tuple(frame_map[index] for index in range(frame_id - 10, frame_id + 1))
    response_set = generate_defender_response_actions(
        decision,
        history,
        defender_id,
        DefenderResponseConfig(response_delay_seconds=response_delay_seconds),
        reachability,
    )
    actions = _representative_responses(response_set.actions)
    background = prepare_observed_background_sequence(
        frame_map,
        frame_id,
        DefenderBestResponseConfig(evaluation_times_s=TIMES),
    )
    xgrid, ygrid = influence_pitch_grid(influence_config)
    target_ids = [runner_id, *beneficiary_ids]
    values = np.zeros((len(target_ids), len(actions), len(TIMES)), dtype=float)

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
            influence = _surface(
                frame, velocities, target_id, xgrid, ygrid, influence_config
            )
            goal_weight = goal_weighted_space_surface(
                xgrid,
                ygrid,
                (float(target.x), float(target.y)),
                direction,
                influence_config,
            )
            intrinsic.append(influence * goal_weight)
        intrinsic = np.asarray(intrinsic)
        denominators = np.maximum(np.sum(intrinsic, axis=(1, 2)), 1e-12)

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
            virtual_defense = _surface(
                virtual_frame,
                virtual_velocities,
                defender_id,
                xgrid,
                ygrid,
                influence_config,
            )
            uncovered = np.exp(
                -influence_config.defender_suppression_strength
                * (fixed_defense + virtual_defense)
            )
            values[:, action_index, time_index] = (
                np.sum(intrinsic * uncovered[None, :, :], axis=(1, 2))
                / denominators
            )

    horizon_values = np.asarray(
        [[_mean(values[t, a, :]) for a in range(len(actions))]
         for t in range(len(target_ids))]
    )
    runner_values = horizon_values[0]
    results = []
    for target_index, beneficiary_id in enumerate(target_ids[1:], start=1):
        tradeoff = analyze_pair_tradeoff(
            runner_values, horizon_values[target_index]
        )
        direct_action = actions[tradeoff.direct_action_index]
        other_action = actions[tradeoff.beneficiary_action_index]
        compromise_action = actions[tradeoff.compromise_action_index]
        results.append(
            {
                "match_id": row.match_id,
                "frame_id": frame_id,
                "runner_id": runner_id,
                "runner_name": metadata.players[runner_id].short_name,
                "beneficiary_id": beneficiary_id,
                "beneficiary_name": metadata.players[beneficiary_id].short_name,
                "defender_id": defender_id,
                "defender_name": metadata.players[defender_id].short_name,
                "attacking_team_id": team_id,
                "attacking_direction": direction,
                "response_action_count": len(actions),
                **asdict(tradeoff),
                "direct_action_id": direct_action.action_id,
                "beneficiary_action_id": other_action.action_id,
                "compromise_action_id": compromise_action.action_id,
                "direct_terminal_separation_m": math.hypot(
                    direct_action.endpoint_x - other_action.endpoint_x,
                    direct_action.endpoint_y - other_action.endpoint_y,
                ),
            }
        )
    detail = {
        "match_id": row.match_id,
        "frame_id": frame_id,
        "runner_id": runner_id,
        "runner_name": metadata.players[runner_id].short_name,
        "defender_id": defender_id,
        "defender_name": metadata.players[defender_id].short_name,
        "attacking_team_id": team_id,
        "attacking_direction": direction,
        "runner_actual_timed": _timed_path(frame_map, frame_id, runner_id),
        "responses": {
            action.action_id: {
                "times_s": action.response_path_times_s,
                "path_xy": action.full_path_xy,
            }
            for action in actions
        },
    }
    return results, detail


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(args.cohort_csv, dtype={"match_id": str, "player_id": str})
    if args.limit > 0:
        cohort = cohort.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    influence_config = GoalWeightedInfluenceConfig(
        grid_resolution_m=args.grid_resolution
    )
    # Physical caps match the main v0.1 action space.  Only numerical state and
    # endpoint resolution are coarsened for screening.
    reachability = SteeringReachabilityConfig(
        grid_resolution_m=2.0,
        state_position_resolution_m=2.0,
        state_heading_bins=48,
        control_direction_count=12,
        variants_per_endpoint=3,
    )
    rows = []
    details = []
    total = len(cohort)
    done = 0
    for match_id, group in cohort.groupby("match_id", sort=True):
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        required_frames = {
            frame_id
            for onset in group.frame_id.astype(int)
            for frame_id in range(onset - 10, onset + int(2.0 * FPS) + 1)
        }
        frame_map = load_bundesliga_frames(
            files["positions"], required_frames
        )
        for row in group.itertuples(index=False):
            done += 1
            started = time.perf_counter()
            try:
                scene_rows, detail = _screen_scene(
                    row,
                    frame_map,
                    metadata,
                    influence_config,
                    reachability,
                    args.response_delay,
                )
                rows.extend(scene_rows)
                if detail is not None:
                    details.append(detail)
                status = f"{len(scene_rows)} pairs"
            except Exception as error:
                status = f"ERROR {type(error).__name__}: {error}"
            print(
                f"[{done}/{total}] {match_id} frame {int(row.frame_id)} "
                f"{status} · {time.perf_counter()-started:.1f}s",
                flush=True,
            )
            if rows:
                pd.DataFrame(rows).sort_values(
                    ["screening_strength", "minimum_branch_gap"],
                    ascending=False,
                ).to_csv(args.output_dir / "pair_screen.csv", index=False)
    table = pd.DataFrame(rows).sort_values(
        ["screening_strength", "minimum_branch_gap"], ascending=False
    )
    table.to_csv(args.output_dir / "pair_screen.csv", index=False)
    (args.output_dir / "scene_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "influence-dilemma-screen-v0.1",
        "status": "coarse_scene_screen_not_final_inference",
        "scene_count": total,
        "pair_count": len(table),
        "times_s": TIMES,
        "influence": asdict(influence_config),
        "reachability": asdict(reachability),
        "response_delay_seconds": args.response_delay,
        "notes": [
            "Observed runner path; nearest current outfield defender only.",
            "One minimum-effort trajectory per coarse 2 m endpoint.",
            "All shortlisted top scenes require full-resolution refinement.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
