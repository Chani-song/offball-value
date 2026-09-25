#!/usr/bin/env python3
"""Generate v0.3 endpoints from a held-out empirical movement library."""

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
    observed_empirical_support_diagnostics,
)
from offball_value.scene_extractor import SceneExtractorConfig


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
        help="Optional prior review CSV fixing the frame/player audit set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_3"),
    )
    parser.add_argument("--scene-limit", type=int, default=12)
    parser.add_argument("--neighbor-count", type=int, default=2048)
    parser.add_argument("--maximum-actions", type=int, default=1600)
    parser.add_argument("--variants-per-endpoint", type=int, default=3)
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
        raise ValueError(f"Reviewed keys are not eligible accepted-scene attackers: {invalid}")
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


def _inside_pitch(x: float, y: float) -> bool:
    return abs(x) <= FIELD_LENGTH / 2.0 and abs(y) <= FIELD_WIDTH / 2.0


def _reference_record(
    match_id: str,
    frame_id: int,
    player_id: str,
    label: str,
    endpoint: tuple[float, float],
    terminal_velocity: tuple[float, float] | None,
    empirically_supported: bool | None,
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
        "empirically_supported": empirically_supported,
        "motion_model": label,
        "terminal_vx_mps": vx,
        "terminal_vy_mps": vy,
        "terminal_speed_mps": math.hypot(vx, vy),
        "terminal_heading_bin": None,
        "primitive_match_id": None,
        "primitive_player_id": None,
        "primitive_frame_id": None,
        "feature_distance": None,
        "endpoint_snap_distance_m": None,
        "fernandez_influence_score": None,
        "fernandez_normalized_radius": None,
        "path_xy": None,
        "failure_reason": None,
    }


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scenes_path = args.scene_dir / match_id / "scene_candidates.csv"
    # A review file is an explicit ordered selection and may contain accepted
    # scenes beyond the first N rows, so do not truncate before validating it.
    scenes = _accepted_scenes(
        scenes_path,
        -1 if args.endpoint_review_csv is not None else args.scene_limit,
    )
    keys = _selected_keys(scenes, args.endpoint_review_csv)
    if not keys:
        raise ValueError("No eligible attacker/scene pairs selected")

    library = EmpiricalPrimitiveLibrary.load(args.library)
    library.assert_excludes_match(match_id)
    action_config = EmpiricalEndpointConfig(
        neighbor_count=args.neighbor_count,
        maximum_actions=args.maximum_actions,
        variants_per_endpoint=args.variants_per_endpoint,
    )
    scene_config = SceneExtractorConfig()
    files = find_bundesliga_files(args.data_dir, match_id)

    target_ids = set()
    for frame_id, _ in keys:
        target_ids.update(
            range(frame_id - scene_config.history_intervals, frame_id + 1)
        )
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
            for candidate_id in range(
                frame_id - scene_config.history_intervals,
                frame_id + 1,
            )
            if candidate_id in frames
        ]
        state = causal_motion_state_from_frames(history, player_id, frame_id, library.config)
        action_set = generate_empirical_endpoint_actions(
            frame.players[player_id],
            frame.ball,
            state,
            library,
            action_config,
        )

        common = {
            "start_x": action_set.start_x,
            "start_y": action_set.start_y,
            "initial_vx_mps": state.vx_mps,
            "initial_vy_mps": state.vy_mps,
            "initial_speed_mps": state.speed_mps,
            "initial_ax_mps2": state.ax_mps2,
            "initial_ay_mps2": state.ay_mps2,
            "initial_longitudinal_acceleration_mps2": state.longitudinal_acceleration_mps2,
            "velocity_sample_count": state.sample_count,
            "velocity_window_seconds": state.observed_window_seconds,
            "velocity_estimator": "quadratic_endpoint",
        }
        for action in action_set.actions:
            record = action.as_record(match_id, frame_id)
            record.update(common)
            action_rows.append(record)

        terminal_id = frame_id + scene_config.horizon_frames
        terminal_frame = frames.get(terminal_id)
        observed_endpoint = None
        if terminal_frame is not None and player_id in terminal_frame.players:
            terminal_player = terminal_frame.players[player_id]
            observed_endpoint = (terminal_player.x, terminal_player.y)
        observed_velocity = _terminal_velocity(
            [
                frames[candidate_id]
                for candidate_id in range(terminal_id - 5, terminal_id + 1)
                if candidate_id in frames
            ],
            player_id,
        )
        diagnostics = observed_empirical_support_diagnostics(
            action_set,
            observed_endpoint,
            observed_velocity,
        )
        cv_endpoint = (
            action_set.start_x + state.vx_mps * library.config.horizon_seconds,
            action_set.start_y + state.vy_mps * library.config.horizon_seconds,
        )
        if _inside_pitch(*cv_endpoint):
            record = _reference_record(
                match_id,
                frame_id,
                player_id,
                "constant_velocity",
                cv_endpoint,
                (state.vx_mps, state.vy_mps),
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

        terminal_bins = {
            action.terminal_heading_bin for action in action_set.optimization_actions
        }
        endpoint_keys = {
            (action.endpoint_x, action.endpoint_y)
            for action in action_set.optimization_actions
        }
        multi_variant_endpoints = sum(
            1
            for endpoint in endpoint_keys
            if len(
                {
                    action.terminal_heading_bin
                    for action in action_set.optimization_actions
                    if (action.endpoint_x, action.endpoint_y) == endpoint
                }
            )
            > 1
        )
        failure_reason = None
        if diagnostics.endpoint_supported is False:
            failure_reason = "endpoint_outside_empirical_support"
        elif diagnostics.endpoint_heading_supported is False:
            failure_reason = "terminal_direction_outside_empirical_support"
        summary_rows.append(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "player_id": player_id,
                **common,
                "optimization_action_count": len(action_set.optimization_actions),
                "grid_action_count": len(action_set.optimization_actions),
                "unique_endpoint_count": len(endpoint_keys),
                "terminal_heading_bin_count": len(terminal_bins),
                "multi_variant_endpoint_count": multi_variant_endpoints,
                "observed_available": observed_endpoint is not None,
                "observed_feasible": diagnostics.endpoint_heading_supported,
                "observed_empirical_endpoint_supported": diagnostics.endpoint_supported,
                "observed_endpoint_heading_supported": diagnostics.endpoint_heading_supported,
                "observed_failure_reason": failure_reason,
                "observed_distance_m": (
                    math.hypot(
                        observed_endpoint[0] - action_set.start_x,
                        observed_endpoint[1] - action_set.start_y,
                    )
                    if observed_endpoint is not None
                    else math.nan
                ),
                "nearest_action_to_observed_m": diagnostics.nearest_endpoint_distance_m,
                "nearest_terminal_heading_difference_degrees": (
                    diagnostics.nearest_terminal_heading_difference_degrees
                ),
                "library_size": action_set.library_size,
                "neighbor_count": action_set.neighbor_count,
                "source_match_count": len(action_set.source_match_ids),
                "source_match_ids": "|".join(action_set.source_match_ids),
                **{
                    f"influence_{key}": value
                    for key, value in action_set.influence_ellipse.as_record().items()
                },
            }
        )
        print(
            f"[{index}/{len(keys)}] frame {frame_id} {player_id}: "
            f"{len(action_set.optimization_actions)} actions, "
            f"endpoint+heading supported={diagnostics.endpoint_heading_supported}",
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
                "library_path": str(args.library),
                "library_source_match_ids": list(library.source_match_ids),
                "selected_key_count": len(keys),
                "endpoint_config": action_config.__dict__,
                "observed_future_usage": "audit_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    supported = summary["observed_endpoint_heading_supported"].dropna()
    print(f"action sets:                 {len(summary)}")
    print(f"optimization actions:       {int(summary['optimization_action_count'].sum()):,}")
    print(f"endpoint+heading support:   {float(supported.mean()):.1%}")
    print(f"target match in provenance: {bool((actions['primitive_match_id'] == match_id).any())}")
    print(f"actions:                     {actions_path}")
    print(f"summary:                     {summary_path}")
    print(f"manifest:                    {manifest_path}")


if __name__ == "__main__":
    main()
