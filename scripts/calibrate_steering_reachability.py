#!/usr/bin/env python3
"""Summarize training-only Frenet-control proxies for v0.5 calibration."""

from __future__ import annotations

import argparse
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
        "--output",
        type=Path,
        default=Path(
            "data/processed/empirical_movement_library_v0_1/"
            "leave_out_DFL-MAT-J03WMX/steering_acceleration_calibration.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = EmpiricalPrimitiveLibrary.load(args.library)
    frame = library.frame
    horizon = library.config.horizon_seconds
    initial_speed = frame["initial_speed_mps"].to_numpy(dtype=float)
    terminal_forward = frame["terminal_v_forward_mps"].to_numpy(dtype=float)
    terminal_lateral = frame["terminal_v_lateral_mps"].to_numpy(dtype=float)
    terminal_speed = np.hypot(terminal_forward, terminal_lateral)
    heading_change = np.abs(np.arctan2(terminal_lateral, terminal_forward))
    valid_heading = (initial_speed >= 1.0) & (terminal_speed >= 1.0)
    mean_tangential = (terminal_speed - initial_speed) / horizon
    curvature_equivalent_normal = (
        0.5 * (initial_speed + terminal_speed) * heading_change / horizon
    )
    causal_longitudinal = frame[
        "initial_longitudinal_acceleration_mps2"
    ].to_numpy(dtype=float)
    metrics = {
        "mean_tangential_acceleration_positive_mps2": mean_tangential[
            valid_heading & (mean_tangential >= 0.0)
        ],
        "mean_tangential_deceleration_mps2": -mean_tangential[
            valid_heading & (mean_tangential < 0.0)
        ],
        "curvature_equivalent_normal_acceleration_mps2": (
            curvature_equivalent_normal[valid_heading]
        ),
        "causal_longitudinal_acceleration_positive_mps2": causal_longitudinal[
            causal_longitudinal >= 0.0
        ],
        "causal_longitudinal_deceleration_mps2": -causal_longitudinal[
            causal_longitudinal < 0.0
        ],
    }
    quantiles = (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
    rows = []
    for metric, values in metrics.items():
        values = values[np.isfinite(values)]
        for quantile in quantiles:
            rows.append(
                {
                    "metric": metric,
                    "quantile": quantile,
                    "value_mps2": float(np.quantile(values, quantile)),
                    "sample_count": len(values),
                    "source_match_count": len(library.source_match_ids),
                    "target_match_excluded": True,
                    "interpretation": (
                        "two_second_average_proxy_not_instantaneous_biomechanical_limit"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
