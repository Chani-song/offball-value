"""What a point is worth to take the ball at: the threat of the space it commands.

The value term in Q was a static grid lookup, G(z) -- the same number for every
candidate and blind to the defence. Removing it changed one answer in 42, and
it stayed unchanged when the prototype evaluated 2,600 points instead of seven,
so it was never really deciding anything.

The missing football is that taking the ball at z does not put you on a point,
it puts you in command of a region, and how far that region reaches toward goal
depends on who is standing between you and it. A centre-back in the lane does
not change the coordinate's static value at all, but it changes what you can do
from there.

So the value of z is the threat of the space you would command there:

    V(z) = SUM_w  influence(w | you at z) . uncovered(w) . xT(w)      (sum)
    V(z) = that / SUM_w influence(w | you at z) . uncovered(w)        (mean)

with xT from the PAUSA grid and `uncovered` the same exponential defender
suppression the accessibility term uses.

SUM OR MEAN IS NOT A FREE CHOICE -- dividing by the commanded area rewards
being penned in. A striker boxed into five square metres on the penalty spot
keeps only the most valuable cells on the pitch and scores a high mean; a
team-mate commanding the whole edge of the box scores a low one. The sum does
not invert like that, and influence decays on its own, so it needs no
normalising. Both are implemented so the choice can be measured rather than
argued.

Inherited, not endorsed: `suppression` defaults to the 1.25 that
goal_weighted_influence uses, which has no stated basis in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundesliga import FIELD_LENGTH, FIELD_WIDTH
from .obso import score_at_points
from .vacated_space import coverage_field

DEFAULT_STEP_M = 2.0
DEFAULT_SUPPRESSION = 1.25
# Influence is negligible well before this, so the per-point work runs on a
# window instead of the whole pitch: 1,890 cells down to a few hundred, which
# is the difference between hours and minutes over a full evaluation set.
WINDOW_HALF_M = 26.0


@dataclass(frozen=True)
class ValueField:
    """Per-time scaffolding: the grid, its threat, and defender suppression.

    Built once per instant and reused for every candidate and every point, so
    the per-point work is one influence field and a dot product.
    """

    xs: np.ndarray
    ys: np.ndarray
    weight: np.ndarray          # uncovered x xT, ready to dot with influence
    uncovered: np.ndarray
    cell_area: float


def build_value_field(
    defenders: list[tuple[tuple[float, float], tuple[float, float]]],
    attacking_direction: int,
    step_m: float = DEFAULT_STEP_M,
    suppression: float = DEFAULT_SUPPRESSION,
) -> ValueField:
    """Grid the pitch once: how open each cell is, times how much it is worth."""
    xs = np.arange(-FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0 + step_m, step_m)
    ys = np.arange(-FIELD_WIDTH / 2.0, FIELD_WIDTH / 2.0 + step_m, step_m)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    threat = np.asarray(score_at_points(points, attacking_direction), dtype=float)
    threat = threat.reshape(gx.shape)

    defender_sum = np.zeros_like(threat)
    for xy, velocity in defenders:
        defender_sum += coverage_field(xy, velocity, xs, ys)
    uncovered = np.exp(-suppression * defender_sum)
    return ValueField(
        xs=xs,
        ys=ys,
        weight=uncovered * threat,
        uncovered=uncovered,
        cell_area=step_m * step_m,
    )


def point_value(
    field: ValueField,
    at_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    mode: str = "sum",
    window_half_m: float = WINDOW_HALF_M,
) -> float:
    """Threat of the space a player would command taking the ball at ``at_xy``."""
    xi0, xi1 = np.searchsorted(
        field.xs, [at_xy[0] - window_half_m, at_xy[0] + window_half_m]
    )
    yi0, yi1 = np.searchsorted(
        field.ys, [at_xy[1] - window_half_m, at_xy[1] + window_half_m]
    )
    xi1 = max(xi1 + 1, xi0 + 2)
    yi1 = max(yi1 + 1, yi0 + 2)
    xs = field.xs[xi0:xi1]
    ys = field.ys[yi0:yi1]
    if xs.size < 2 or ys.size < 2:
        return 0.0
    influence = coverage_field(at_xy, velocity_xy, xs, ys)
    weight = field.weight[yi0:yi1, xi0:xi1]
    total = float(np.sum(influence * weight) * field.cell_area)
    if mode == "sum":
        return total
    commanded = float(
        np.sum(influence * field.uncovered[yi0:yi1, xi0:xi1]) * field.cell_area
    )
    return total / commanded if commanded > 1e-9 else 0.0
