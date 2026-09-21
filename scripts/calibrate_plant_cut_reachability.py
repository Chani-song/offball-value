#!/usr/bin/env python3
"""Detect training-only V-cut-like primitives and calibrate plant-and-cut proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.empirical_action_space import EmpiricalPrimitiveLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(
            "data/processed/empirical_movement_library_v0_1/"
            "leave_out_DFL-MAT-J03WMX/primitives.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/empirical_movement_library_v0_1/"
            "leave_out_DFL-MAT-J03WMX/plant_cut_calibration_v0_1"
        ),
    )
    parser.add_argument("--minimum-initial-speed", type=float, default=6.0)
    parser.add_argument("--minimum-terminal-turn-degrees", type=float, default=90.0)
    parser.add_argument("--maximum-low-segment-speed", type=float, default=2.0)
    parser.add_argument("--minimum-forward-return", type=float, default=0.5)
    return parser.parse_args()


def _path_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    count = len(frame)
    forward = np.column_stack(
        [
            np.zeros(count),
            *[
                frame[f"path_{index}_forward_m"].to_numpy(dtype=float)
                for index in range(1, 6)
            ],
        ]
    )
    lateral = np.column_stack(
        [
            np.zeros(count),
            *[
                frame[f"path_{index}_lateral_m"].to_numpy(dtype=float)
                for index in range(1, 6)
            ],
        ]
    )
    return forward, lateral


def main() -> None:
    args = parse_args()
    library = EmpiricalPrimitiveLibrary.load(args.library)
    frame = library.frame.copy()
    forward, lateral = _path_arrays(frame)
    interval_seconds = 0.4
    segment_forward_velocity = np.diff(forward, axis=1) / interval_seconds
    segment_lateral_velocity = np.diff(lateral, axis=1) / interval_seconds
    segment_speed = np.hypot(segment_forward_velocity, segment_lateral_velocity)
    segment_heading_degrees = np.degrees(
        np.arctan2(segment_lateral_velocity, segment_forward_velocity)
    )
    segment_heading_change_degrees = np.abs(
        (np.diff(segment_heading_degrees, axis=1) + 180.0) % 360.0 - 180.0
    )

    initial_speed = frame["initial_speed_mps"].to_numpy(dtype=float)
    terminal_forward_velocity = frame["terminal_v_forward_mps"].to_numpy(
        dtype=float
    )
    terminal_lateral_velocity = frame["terminal_v_lateral_mps"].to_numpy(
        dtype=float
    )
    terminal_turn_degrees = np.abs(
        np.degrees(
            np.arctan2(terminal_lateral_velocity, terminal_forward_velocity)
        )
    )
    minimum_segment_index = np.argmin(segment_speed, axis=1)
    minimum_segment_speed = segment_speed[
        np.arange(len(frame)), minimum_segment_index
    ]
    minimum_speed_center_time = 0.2 + interval_seconds * minimum_segment_index
    peak_forward_index = np.argmax(forward, axis=1)
    peak_forward_displacement = np.max(forward, axis=1)
    forward_return_distance = (
        peak_forward_displacement
        - frame["endpoint_forward_m"].to_numpy(dtype=float)
    )

    direction_change_mask = (
        (initial_speed >= args.minimum_initial_speed)
        & (terminal_turn_degrees >= args.minimum_terminal_turn_degrees)
        & (minimum_segment_speed <= args.maximum_low_segment_speed)
    )
    plant_return_mask = (
        direction_change_mask
        & (peak_forward_index < forward.shape[1] - 1)
        & (peak_forward_displacement > 0.0)
        & (forward_return_distance >= args.minimum_forward_return)
    )

    stop_deceleration_proxy = np.full(len(frame), np.nan)
    valid_peak = peak_forward_displacement > 1e-9
    stop_deceleration_proxy[valid_peak] = (
        initial_speed[valid_peak] ** 2
        / (2.0 * peak_forward_displacement[valid_peak])
    )
    mean_braking_proxy = (
        initial_speed - minimum_segment_speed
    ) / minimum_speed_center_time
    post_cut_acceleration_proxy = (
        frame["terminal_speed_mps"].to_numpy(dtype=float)
        - minimum_segment_speed
    ) / np.maximum(2.0 - minimum_speed_center_time, 0.2)
    low_speed_duration_proxy = interval_seconds * np.sum(
        segment_speed <= args.maximum_low_segment_speed,
        axis=1,
    )

    derived = frame[
        ["match_id", "player_id", "game_section", "frame_id"]
    ].copy()
    derived["direction_change_like"] = direction_change_mask
    derived["plant_return_like"] = plant_return_mask
    derived["initial_speed_mps"] = initial_speed
    derived["terminal_turn_degrees"] = terminal_turn_degrees
    derived["minimum_segment_speed_mps"] = minimum_segment_speed
    derived["minimum_speed_center_time_s"] = minimum_speed_center_time
    derived["peak_forward_displacement_m"] = peak_forward_displacement
    derived["peak_forward_time_s"] = interval_seconds * peak_forward_index
    derived["forward_return_distance_m"] = forward_return_distance
    derived["endpoint_forward_m"] = frame["endpoint_forward_m"].to_numpy(dtype=float)
    derived["endpoint_lateral_m"] = frame["endpoint_lateral_m"].to_numpy(dtype=float)
    derived["mean_braking_proxy_mps2"] = mean_braking_proxy
    derived["stop_distance_deceleration_proxy_mps2"] = stop_deceleration_proxy
    derived["post_cut_acceleration_proxy_mps2"] = post_cut_acceleration_proxy
    derived["maximum_adjacent_segment_turn_degrees"] = np.max(
        segment_heading_change_degrees,
        axis=1,
    )
    derived["low_speed_duration_proxy_s"] = low_speed_duration_proxy
    derived["source_match_count"] = len(library.source_match_ids)
    derived["target_match_excluded"] = True

    selected = derived[direction_change_mask].copy()
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    metric_columns = [
        "initial_speed_mps",
        "terminal_turn_degrees",
        "minimum_segment_speed_mps",
        "minimum_speed_center_time_s",
        "peak_forward_displacement_m",
        "peak_forward_time_s",
        "forward_return_distance_m",
        "endpoint_forward_m",
        "endpoint_lateral_m",
        "mean_braking_proxy_mps2",
        "stop_distance_deceleration_proxy_mps2",
        "post_cut_acceleration_proxy_mps2",
        "maximum_adjacent_segment_turn_degrees",
        "low_speed_duration_proxy_s",
    ]
    summary_rows = []
    for subgroup_name, subgroup_mask in (
        ("direction_change_like", direction_change_mask),
        ("plant_return_like", plant_return_mask),
    ):
        subgroup = derived[subgroup_mask]
        for metric in metric_columns:
            values = subgroup[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            for quantile in quantiles:
                summary_rows.append(
                    {
                        "subgroup": subgroup_name,
                        "metric": metric,
                        "quantile": quantile,
                        "value": float(np.quantile(values, quantile)),
                        "sample_count": len(subgroup),
                        "source_match_count": subgroup["match_id"].nunique(),
                        "target_match_excluded": True,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    per_match = (
        derived.groupby("match_id", as_index=False)
        .agg(
            direction_change_like_count=("direction_change_like", "sum"),
            plant_return_like_count=("plant_return_like", "sum"),
        )
        .sort_values("match_id")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "plant_cut_candidates.csv"
    summary_path = args.output_dir / "plant_cut_calibration_summary.csv"
    per_match_path = args.output_dir / "plant_cut_counts_by_match.csv"
    manifest_path = args.output_dir / "manifest.json"
    selected.to_csv(candidates_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_match.to_csv(per_match_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "library": str(args.library),
                "source_match_ids": list(library.source_match_ids),
                "target_match_excluded": True,
                "detector": {
                    "minimum_initial_speed_mps": args.minimum_initial_speed,
                    "minimum_terminal_turn_degrees": (
                        args.minimum_terminal_turn_degrees
                    ),
                    "maximum_low_segment_speed_mps": (
                        args.maximum_low_segment_speed
                    ),
                    "minimum_forward_return_m": args.minimum_forward_return,
                    "path_sampling_seconds": list(
                        library.config.path_sample_seconds
                    ),
                },
                "direction_change_like_count": int(direction_change_mask.sum()),
                "plant_return_like_count": int(plant_return_mask.sum()),
                "recommended_common_parameters": {
                    "transient_cut_braking_cap_mps2": 9.0,
                    "post_cut_acceleration_cap_mps2": 4.5,
                    "plant_duration_sensitivity_s": [0.1, 0.2, 0.3],
                    "note": (
                        "braking rounded near plant-return stop-distance proxy "
                        "p99; plant duration is sensitivity-tested because the "
                        "0.4 s path sampling cannot identify foot-plant duration"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(per_match.to_string(index=False))
    print(f"direction-change-like: {int(direction_change_mask.sum())}")
    print(f"plant-return-like:     {int(plant_return_mask.sum())}")
    print(f"candidates:            {candidates_path}")
    print(f"summary:               {summary_path}")
    print(f"manifest:              {manifest_path}")


if __name__ == "__main__":
    main()
