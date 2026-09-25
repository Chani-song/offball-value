#!/usr/bin/env python3
"""Generate continuous-steering or hybrid plant-and-cut endpoint actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    find_bundesliga_files,
    load_bundesliga_frames,
    normalize_bundesliga_match_id,
)
from offball_value.empirical_action_space import (
    EmpiricalEndpointConfig,
    EmpiricalPrimitiveLibrary,
    causal_motion_state_from_frames,
    generate_empirical_endpoint_actions,
)
from offball_value.scene_extractor import SceneExtractorConfig
from offball_value.steering_reachable import (
    SteeringReachabilityConfig,
    generate_hybrid_endpoint_actions,
    generate_steering_endpoint_actions,
    observed_steering_support_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/processed/scene_extractor_v0_1"),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(
            "data/processed/empirical_movement_library_v0_1/"
            "leave_out_DFL-MAT-J03WMX/primitives.npz"
        ),
    )
    parser.add_argument(
        "--endpoint-review-csv",
        type=Path,
        default=None,
        help="Optional review CSV fixing the ordered frame/player set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_5_1"),
    )
    parser.add_argument("--scene-limit", type=int, default=12)
    parser.add_argument("--only-frame-id", type=int, default=None)
    parser.add_argument("--only-player-id", default=None)
    parser.add_argument("--max-speed", type=float, default=9.0)
    parser.add_argument("--max-tangential-acceleration", type=float, default=4.5)
    parser.add_argument("--max-tangential-deceleration", type=float, default=6.0)
    parser.add_argument(
        "--max-normal-acceleration",
        type=float,
        default=6.0,
        help=(
            "Provisional steering cap; approximately the leave-one-match-out "
            "99.9th percentile of the 2 s curvature-equivalent proxy."
        ),
    )
    parser.add_argument("--integration-step", type=float, default=0.1)
    parser.add_argument("--control-directions", type=int, default=16)
    parser.add_argument(
        "--state-heading-bins",
        type=int,
        default=96,
        help="3.75 degree bins prevent maximum high-speed steering from aliasing into coast.",
    )
    parser.add_argument("--state-position-resolution", type=float, default=1.0)
    parser.add_argument("--state-speed-resolution", type=float, default=0.5)
    parser.add_argument("--empirical-neighbors", type=int, default=2048)
    parser.add_argument(
        "--include-plant-cuts",
        action="store_true",
        help="Union calibrated stop/plant/reorient/reaccelerate maneuvers with steering.",
    )
    parser.add_argument("--max-plant-cut-deceleration", type=float, default=9.0)
    parser.add_argument("--minimum-plant-cut-speed", type=float, default=2.0)
    parser.add_argument("--plant-cut-directions", type=int, default=16)
    parser.add_argument("--minimum-plant-cut-angle", type=float, default=45.0)
    return parser.parse_args()


def _accepted_scenes(path: Path, limit: int) -> pd.DataFrame:
    scenes = pd.read_csv(path)
    result = scenes[scenes["accepted"].astype(str).str.lower() == "true"].copy()
    result = result.sort_values("frame_id")
    return result if limit < 0 else result.head(limit)


def _selected_keys(
    scenes: pd.DataFrame,
    review_csv: Path | None,
) -> list[tuple[int, str]]:
    eligible_by_frame = {
        int(row.frame_id): [
            player_id
            for player_id in str(row.eligible_attacker_ids).split("|")
            if player_id and player_id != "nan"
        ]
        for row in scenes.itertuples(index=False)
    }
    if review_csv is None:
        return [
            (frame_id, player_id)
            for frame_id, player_ids in eligible_by_frame.items()
            for player_id in player_ids
        ]
    reviews = pd.read_csv(review_csv)
    keys = [
        (int(row.frame_id), str(row.player_id))
        for row in reviews[["frame_id", "player_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    invalid = [
        key
        for key in keys
        if key[0] not in eligible_by_frame or key[1] not in eligible_by_frame[key[0]]
    ]
    if invalid:
        raise ValueError(f"Reviewed keys are not eligible attackers: {invalid}")
    return keys


def _terminal_velocity(frames, player_id: str) -> tuple[float, float] | None:
    samples = []
    for frame in frames:
        player = frame.players.get(player_id)
        if player is not None:
            samples.append((frame.frame_id / FPS, player.x, player.y))
    if len(samples) < 2:
        return None
    times = np.asarray([sample[0] for sample in samples], dtype=float)
    times -= times[-1]
    design = np.column_stack([times, np.ones(len(times))])
    weights = np.linalg.pinv(design)
    vx = float((weights @ np.asarray([sample[1] for sample in samples]))[0])
    vy = float((weights @ np.asarray([sample[2] for sample in samples]))[0])
    return vx, vy


def _reference_record(
    match_id: str,
    frame_id: int,
    player_id: str,
    label: str,
    endpoint: tuple[float, float],
    terminal_velocity: tuple[float, float] | None,
    supported: bool | None,
) -> dict[str, object]:
    vx, vy = terminal_velocity if terminal_velocity is not None else (math.nan, math.nan)
    return {
        "match_id": match_id,
        "frame_id": frame_id,
        "player_id": player_id,
        "action_id": f"{match_id}:{frame_id}:{player_id}:{label}",
        "endpoint_x": endpoint[0],
        "endpoint_y": endpoint[1],
        "labels": label,
        "optimization_eligible": False,
        "kinematically_feasible": None,
        "dynamically_feasible": None,
        "empirically_supported": supported,
        "motion_model": label,
        "terminal_vx_mps": vx,
        "terminal_vy_mps": vy,
        "terminal_speed_mps": math.hypot(vx, vy),
        "terminal_heading_bin": None,
        "path_xy": None,
        "failure_reason": None,
    }


def _proposal_record(
    match_id: str,
    frame_id: int,
    player_id: str,
    endpoint: tuple[float, float],
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "frame_id": frame_id,
        "player_id": player_id,
        "action_id": (
            f"{match_id}:{frame_id}:{player_id}:fernandez_proposal:"
            f"{endpoint[0]:.1f}:{endpoint[1]:.1f}"
        ),
        "endpoint_x": endpoint[0],
        "endpoint_y": endpoint[1],
        "labels": "fernandez_proposal",
        "optimization_eligible": False,
        "kinematically_feasible": False,
        "dynamically_feasible": False,
        "empirically_supported": False,
        "motion_model": "proposal_only",
        "terminal_vx_mps": None,
        "terminal_vy_mps": None,
        "terminal_speed_mps": None,
        "terminal_heading_bin": None,
        "path_xy": None,
        "failure_reason": "not_steering_action",
    }


def _sector_counts(action_set, heading: float) -> tuple[int, ...]:
    counts = [0] * 8
    seen = set()
    for action in action_set.optimization_actions:
        endpoint_cell = (action.endpoint_cell_x, action.endpoint_cell_y)
        if endpoint_cell in seen:
            continue
        seen.add(endpoint_cell)
        dx = action.endpoint_x - action_set.start_x
        dy = action.endpoint_y - action_set.start_y
        if math.hypot(dx, dy) < 0.5:
            continue
        relative = (math.atan2(dy, dx) - heading + 2.0 * math.pi) % (
            2.0 * math.pi
        )
        counts[int(math.floor(relative / (2.0 * math.pi / 8))) % 8] += 1
    return tuple(counts)


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scene_path = args.scene_dir / match_id / "scene_candidates.csv"
    scenes = _accepted_scenes(
        scene_path,
        -1 if args.endpoint_review_csv is not None else args.scene_limit,
    )
    keys = _selected_keys(scenes, args.endpoint_review_csv)
    if args.only_frame_id is not None:
        keys = [key for key in keys if key[0] == args.only_frame_id]
    if args.only_player_id is not None:
        keys = [key for key in keys if key[1] == args.only_player_id]
    if not keys:
        raise ValueError("No scene/player pairs selected")

    library = EmpiricalPrimitiveLibrary.load(args.library)
    library.assert_excludes_match(match_id)
    empirical_config = EmpiricalEndpointConfig(
        neighbor_count=args.empirical_neighbors,
        maximum_actions=1600,
        variants_per_endpoint=3,
        maximum_speed_mps=args.max_speed,
    )
    steering_config = SteeringReachabilityConfig(
        max_speed_mps=args.max_speed,
        max_tangential_acceleration_mps2=args.max_tangential_acceleration,
        max_tangential_deceleration_mps2=args.max_tangential_deceleration,
        max_normal_acceleration_mps2=args.max_normal_acceleration,
        integration_step_seconds=args.integration_step,
        control_direction_count=args.control_directions,
        state_heading_bins=args.state_heading_bins,
        state_position_resolution_m=args.state_position_resolution,
        state_speed_resolution_mps=args.state_speed_resolution,
        minimum_plant_cut_initial_speed_mps=args.minimum_plant_cut_speed,
        max_plant_cut_deceleration_mps2=args.max_plant_cut_deceleration,
        plant_cut_direction_count=args.plant_cut_directions,
        minimum_plant_cut_angle_degrees=args.minimum_plant_cut_angle,
    )
    scene_config = SceneExtractorConfig()
    files = find_bundesliga_files(args.data_dir, match_id)
    target_ids = set()
    for frame_id, _ in keys:
        target_ids.update(range(frame_id - scene_config.history_intervals, frame_id + 1))
        terminal_id = frame_id + scene_config.horizon_frames
        target_ids.update(range(terminal_id - 5, terminal_id + 1))
    frames = load_bundesliga_frames(files["positions"], target_ids)

    action_rows = []
    summary_rows = []
    for index, (frame_id, player_id) in enumerate(keys, start=1):
        frame = frames.get(frame_id)
        if frame is None or frame.ball is None or player_id not in frame.players:
            raise ValueError(f"Missing decision-frame tracking for {(frame_id, player_id)}")
        history = [
            frames[candidate_id]
            for candidate_id in range(frame_id - scene_config.history_intervals, frame_id + 1)
            if candidate_id in frames
        ]
        state = causal_motion_state_from_frames(history, player_id, frame_id, library.config)
        empirical_set = generate_empirical_endpoint_actions(
            frame.players[player_id],
            frame.ball,
            state,
            library,
            empirical_config,
        )
        action_generator = (
            generate_hybrid_endpoint_actions
            if args.include_plant_cuts
            else generate_steering_endpoint_actions
        )
        steering_set = action_generator(
            frame.players[player_id],
            frame.ball,
            state,
            empirical_set,
            steering_config,
        )
        common = {
            "start_x": steering_set.start_x,
            "start_y": steering_set.start_y,
            "initial_vx_mps": steering_set.initial_vx_mps,
            "initial_vy_mps": steering_set.initial_vy_mps,
            "initial_speed_mps": steering_set.initial_speed_mps,
            "initial_ax_mps2": state.ax_mps2,
            "initial_ay_mps2": state.ay_mps2,
            "initial_longitudinal_acceleration_mps2": state.longitudinal_acceleration_mps2,
            "velocity_sample_count": state.sample_count,
            "velocity_window_seconds": state.observed_window_seconds,
            "velocity_estimator": "quadratic_endpoint",
        }
        for endpoint in steering_set.proposal_endpoints:
            record = _proposal_record(match_id, frame_id, player_id, endpoint)
            record.update(common)
            action_rows.append(record)
        for action in steering_set.actions:
            record = action.as_record(match_id, frame_id)
            record.update(common)
            action_rows.append(record)

        terminal_id = frame_id + scene_config.horizon_frames
        terminal_frame = frames.get(terminal_id)
        observed_endpoint = None
        if terminal_frame is not None and player_id in terminal_frame.players:
            observed_player = terminal_frame.players[player_id]
            observed_endpoint = (observed_player.x, observed_player.y)
        observed_velocity = _terminal_velocity(
            [
                frames[candidate_id]
                for candidate_id in range(terminal_id - 5, terminal_id + 1)
                if candidate_id in frames
            ],
            player_id,
        )
        diagnostics = observed_steering_support_diagnostics(
            steering_set,
            observed_endpoint,
            observed_velocity,
        )
        cv_endpoint = (
            steering_set.start_x
            + steering_set.initial_vx_mps * steering_config.horizon_seconds,
            steering_set.start_y
            + steering_set.initial_vy_mps * steering_config.horizon_seconds,
        )
        if (
            abs(cv_endpoint[0]) <= FIELD_LENGTH / 2.0
            and abs(cv_endpoint[1]) <= FIELD_WIDTH / 2.0
        ):
            record = _reference_record(
                match_id,
                frame_id,
                player_id,
                "constant_velocity",
                cv_endpoint,
                (steering_set.initial_vx_mps, steering_set.initial_vy_mps),
                None,
            )
            record.update(common)
            action_rows.append(record)
        if observed_endpoint is not None:
            record = _reference_record(
                match_id,
                frame_id,
                player_id,
                "observed",
                observed_endpoint,
                observed_velocity,
                diagnostics.endpoint_heading_supported,
            )
            record.update(common)
            action_rows.append(record)

        actions = steering_set.optimization_actions
        continuous_actions = [
            action
            for action in actions
            if action.motion.maneuver_type == "continuous_steering"
        ]
        plant_cut_actions = [
            action
            for action in actions
            if action.motion.maneuver_type == "plant_and_cut"
        ]
        unique_endpoint_cells = {
            (action.endpoint_cell_x, action.endpoint_cell_y) for action in actions
        }
        inside_cells = {
            (action.endpoint_cell_x, action.endpoint_cell_y)
            for action in actions
            if action.inside_fernandez_contour
        }
        empirical_cells = {
            (action.endpoint_cell_x, action.endpoint_cell_y)
            for action in actions
            if action.empirically_supported
        }
        sectors = _sector_counts(steering_set, steering_set.reference_heading_radians)
        failure_reason = None
        if diagnostics.endpoint_supported is False:
            failure_reason = "endpoint_outside_steering_support"
        elif diagnostics.endpoint_heading_supported is False:
            failure_reason = "terminal_direction_outside_steering_support"
        summary_rows.append(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "player_id": player_id,
                **common,
                "optimization_action_count": len(actions),
                "continuous_steering_action_count": len(continuous_actions),
                "plant_cut_action_count": len(plant_cut_actions),
                "plant_cut_unique_endpoint_count": len(
                    {
                        (action.endpoint_cell_x, action.endpoint_cell_y)
                        for action in plant_cut_actions
                    }
                ),
                "grid_action_count": len(actions),
                "unique_endpoint_count": len(unique_endpoint_cells),
                "fernandez_proposal_count": len(steering_set.proposal_endpoints),
                "feasible_inside_fernandez_endpoint_count": len(inside_cells),
                "empirically_supported_endpoint_count": len(empirical_cells),
                "terminal_heading_bin_count": len(
                    {action.motion.terminal_heading_bin for action in actions}
                ),
                "spatial_sector_count": sum(count > 0 for count in sectors),
                "spatial_sector_counts": "|".join(str(count) for count in sectors),
                "observed_available": observed_endpoint is not None,
                "observed_feasible": diagnostics.endpoint_heading_supported,
                "observed_dynamic_endpoint_supported": diagnostics.endpoint_supported,
                "observed_endpoint_heading_supported": diagnostics.endpoint_heading_supported,
                "observed_failure_reason": failure_reason,
                "observed_distance_m": (
                    math.hypot(
                        observed_endpoint[0] - steering_set.start_x,
                        observed_endpoint[1] - steering_set.start_y,
                    )
                    if observed_endpoint is not None
                    else math.nan
                ),
                "nearest_action_to_observed_m": diagnostics.nearest_endpoint_distance_m,
                "nearest_terminal_heading_difference_degrees": (
                    diagnostics.nearest_heading_difference_degrees
                ),
                "library_size": empirical_set.library_size,
                "source_match_count": len(empirical_set.source_match_ids),
                "source_match_ids": "|".join(empirical_set.source_match_ids),
                **{
                    f"influence_{key}": value
                    for key, value in steering_set.influence_ellipse.as_record().items()
                },
            }
        )
        print(
            f"[{index}/{len(keys)}] frame {frame_id} {player_id}: "
            f"{len(unique_endpoint_cells)} reachable 1 m cells, "
            f"{len(actions)} endpoint-heading actions, "
            f"{len(plant_cut_actions)} plant cuts, "
            f"{sum(count > 0 for count in sectors)}/8 spatial sectors",
            flush=True,
        )

    actions = pd.DataFrame(action_rows)
    summary = pd.DataFrame(summary_rows)
    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / "attacker_endpoint_actions.csv"
    summary_path = output_dir / "attacker_endpoint_summary.csv"
    manifest_path = output_dir / "manifest.json"
    actions.to_csv(actions_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "match_id": match_id,
                "selected_key_count": len(keys),
                "steering_config": steering_config.__dict__,
                "action_space": (
                    "continuous_steering_union_plant_and_cut"
                    if args.include_plant_cuts
                    else "continuous_steering"
                ),
                "control_coordinates": "instantaneous_tangential_and_normal",
                "plant_cut_calibration": (
                    "training-only plant_cut_calibration_v0_1"
                    if args.include_plant_cuts
                    else None
                ),
                "grid_role": "1_m_terminal_cell_with_continuous_witness",
                "empirical_library": str(args.library),
                "empirical_source_match_ids": list(library.source_match_ids),
                "observed_future_usage": "audit_only",
                "fernandez_role": "proposal_and_visual_reference_not_feasibility",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    supported = summary["observed_endpoint_heading_supported"].dropna()
    print(f"action sets:                  {len(summary)}")
    print(f"steering actions:            {int(summary['optimization_action_count'].sum()):,}")
    if args.include_plant_cuts:
        print(
            "plant-cut actions:           "
            f"{int(summary['plant_cut_action_count'].sum()):,}"
        )
    print(f"median reachable cells:      {summary['unique_endpoint_count'].median():.1f}")
    print(f"endpoint+heading support:    {float(supported.mean()):.1%}")
    print(
        "all-eight-sector scenes:     "
        f"{int((summary['spatial_sector_count'] == 8).sum())}/{len(summary)}"
    )
    print(
        "target match in provenance:  "
        f"{bool((actions['empirical_primitive_match_id'] == match_id).any())}"
    )
    print(f"actions:                      {actions_path}")
    print(f"summary:                      {summary_path}")
    print(f"manifest:                     {manifest_path}")


if __name__ == "__main__":
    main()
