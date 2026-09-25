#!/usr/bin/env python3
"""Summarize training-only acceleration proxies for v0.4 calibration."""

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
            "leave_out_DFL-MAT-J03WMX/dynamic_acceleration_calibration.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = EmpiricalPrimitiveLibrary.load(args.library)
    frame = library.frame
    horizon = library.config.horizon_seconds
    initial_speed = frame["initial_speed_mps"].to_numpy(dtype=float)
    endpoint_forward = frame["endpoint_forward_m"].to_numpy(dtype=float)
    endpoint_lateral = frame["endpoint_lateral_m"].to_numpy(dtype=float)
    terminal_forward = frame["terminal_v_forward_mps"].to_numpy(dtype=float)
    terminal_lateral = frame["terminal_v_lateral_mps"].to_numpy(dtype=float)
    metrics = {
        "constant_endpoint_equivalent_acceleration_mps2": np.hypot(
            2.0 * (endpoint_forward - initial_speed * horizon) / horizon**2,
            2.0 * endpoint_lateral / horizon**2,
        ),
        "mean_velocity_change_acceleration_mps2": np.hypot(
            terminal_forward - initial_speed,
            terminal_lateral,
        )
        / horizon,
        "causal_longitudinal_acceleration_abs_mps2": np.abs(
            frame["initial_longitudinal_acceleration_mps2"].to_numpy(dtype=float)
        ),
    }
    quantiles = (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995)
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
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
