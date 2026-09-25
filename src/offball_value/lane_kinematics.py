"""Kinematics of the passing lane, defined so they survive the move to inference.

The first kinematic xPass measured arrival times at the pass target, and it did
not transfer: trained on StatsBomb it predicted 0.90-0.92 on Bundesliga passes
that complete 0.81, worse in Brier than the base rate. Dropping the constant
features, the women's competitions and the international matches each moved it
by 0.02 or less, so none of those was the cause.

The cause is which point the features are anchored to. StatsBomb gives one
location per pass, `pass.end_location`, and that is where the ball ENDED. On a
completion it is a team-mate's feet, so `mate_time_to_target` is near zero as a
matter of arithmetic, not of football; on an interception the same holds for
the defender. A model fitted on that learns an identity that dissolves when the
target becomes a point the option catalogue invented.

The distinction that matters is not "does this feature see the outcome" -- the
static lane features see it too, since an interception truncates the lane --
but "is it near-deterministic given the outcome". Static lane occupancy is
merely correlated, and xpass360 calibrates to within 0.005 on it. Arrival time
at the endpoint is an identity, and it does not survive.

So these are anchored to the LANE rather than to its endpoint: whether
defenders are converging on the corridor the ball must travel down. The lane is
fixed by (start, end) in both settings, and a defender's velocity does not
depend on which point we chose to aim at, so the same number means the same
thing on a pass that was played and on one that was not.

In football terms this is the part of a passer's read that a still frame cannot
give: the lane looks open now, but is it closing?
"""

from __future__ import annotations

import math

import numpy as np

LANE_KINEMATIC_NAMES = (
    "lane_mean_closing_mps",       # defenders converging on the corridor
    "lane_nearest_closing_mps",    # the one already closest to it
    "lane_perp_rate_mps",          # how fast the tightest gap is shutting
    "mean_opp_speed_mps",          # how active the defence is at all
    "max_opp_speed_mps",
    "actor_speed_mps",             # the passer's own momentum
)

BESIDE_MARGIN_M = 2.0     # how far past either end still counts as beside the lane
CONSIDER_RADIUS_M = 20.0  # defenders further from the lane than this cannot reach it


def lane_kinematics(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    defenders,
    actor_velocity: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Lane-anchored kinematics, in LANE_KINEMATIC_NAMES order.

    ``defenders`` are ((x, y), (vx, vy)) in metres, goalkeepers already removed.
    Positive closing means moving toward the lane.
    """
    ax, ay = float(start_xy[0]), float(start_xy[1])
    bx, by = float(end_xy[0]), float(end_xy[1])
    length = math.hypot(bx - ax, by - ay)
    actor_speed = float(math.hypot(*actor_velocity))
    if length < 1e-6 or not defenders:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, actor_speed], dtype=float)

    ux, uy = (bx - ax) / length, (by - ay) / length
    nx, ny = -uy, ux                      # left normal to the lane

    closings, speeds = [], []
    nearest_perp = math.inf
    nearest_closing = 0.0
    nearest_rate = 0.0
    for (px, py), (vx, vy) in defenders:
        speeds.append(math.hypot(vx, vy))
        rx, ry = px - ax, py - ay
        along = rx * ux + ry * uy
        signed_perp = rx * nx + ry * ny
        perp = abs(signed_perp)
        if perp > CONSIDER_RADIUS_M:
            continue
        if along < -BESIDE_MARGIN_M or along > length + BESIDE_MARGIN_M:
            continue                      # behind the passer or past the target
        # Velocity across the lane, signed so positive is toward it.
        perp_velocity = vx * nx + vy * ny
        closing = -perp_velocity * (1.0 if signed_perp >= 0.0 else -1.0)
        closings.append(closing)
        if perp < nearest_perp:
            nearest_perp = perp
            nearest_closing = closing
            nearest_rate = -closing       # d(perp)/dt: negative means shutting

    return np.array(
        [
            float(np.mean(closings)) if closings else 0.0,
            nearest_closing,
            nearest_rate if math.isfinite(nearest_perp) else 0.0,
            float(np.mean(speeds)) if speeds else 0.0,
            float(np.max(speeds)) if speeds else 0.0,
            actor_speed,
        ],
        dtype=float,
    )
