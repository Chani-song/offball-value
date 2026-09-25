"""Convert the learned progression surface into an obso-compatible EPV grid.

Reads data/processed/progression_goal_danger_v0/surface.npz (30x20 bins on
the StatsBomb 120x80 pitch, attack toward x=120) and writes a CSV grid in
the score_at_points contract: rows = y across 68 m, cols = x across 105 m,
attack toward +x, loader-normalized by max.  The surface is symmetrized in
y (pitches are top-bottom symmetric; residual asymmetry is sampling noise)
and bilinearly upsampled to 1 m cells so per-metre depth gradients do not
staircase.

Usage:
    python scripts/build_progression_epv_grid.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SURFACE = Path("data/processed/progression_goal_danger_v0/surface.npz")
OUT = Path("data/processed/progression_goal_danger_v0/progression_epv_grid_v0.csv")

PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0
OUT_COLS, OUT_ROWS = 105, 68  # 1 m cells


def main() -> None:
    data = np.load(SURFACE)
    smooth = np.asarray(data["smooth"], dtype=float)  # (grid_x, grid_y)
    grid_x, grid_y = smooth.shape

    # Symmetrize across the pitch's long axis.
    smooth = 0.5 * (smooth + smooth[:, ::-1])

    # Bin centres in metres, centred pitch coordinates, attack toward +x.
    x_centres = (np.arange(grid_x) + 0.5) * (120.0 / grid_x) * (
        PITCH_LENGTH_M / 120.0
    ) - PITCH_LENGTH_M / 2.0
    y_centres = (np.arange(grid_y) + 0.5) * (80.0 / grid_y) * (
        PITCH_WIDTH_M / 80.0
    ) - PITCH_WIDTH_M / 2.0

    out_x = (np.arange(OUT_COLS) + 0.5) * (PITCH_LENGTH_M / OUT_COLS) - (
        PITCH_LENGTH_M / 2.0
    )
    out_y = (np.arange(OUT_ROWS) + 0.5) * (PITCH_WIDTH_M / OUT_ROWS) - (
        PITCH_WIDTH_M / 2.0
    )

    def interp_axis(values: np.ndarray, centres: np.ndarray, targets: np.ndarray, axis: int) -> np.ndarray:
        moved = np.moveaxis(values, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        out = np.empty((targets.size, flat.shape[1]), dtype=float)
        for column in range(flat.shape[1]):
            out[:, column] = np.interp(targets, centres, flat[:, column])
        return np.moveaxis(out.reshape((targets.size,) + moved.shape[1:]), 0, axis)

    fine = interp_axis(smooth, x_centres, out_x, axis=0)
    fine = interp_axis(fine, y_centres, out_y, axis=1)

    # score_at_points expects rows = y, cols = x.
    grid = fine.T
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT, grid, delimiter=",", fmt="%.6f")
    print(f"wrote {OUT} shape={grid.shape} max={grid.max():.5f} min={grid.min():.5f}")

    # Report the normalized central-lane depth profile the pipeline will see.
    normalized = grid / grid.max()
    row = OUT_ROWS // 2
    print("metres_from_goal  normalized_G")
    for metres in (35, 30, 25, 20, 15, 10, 5):
        col = min(OUT_COLS - 1, int((PITCH_LENGTH_M / 2.0 - metres) + PITCH_LENGTH_M / 2.0))
        print(f"{metres:>14}  {normalized[row, col]:.4f}")


if __name__ == "__main__":
    main()
