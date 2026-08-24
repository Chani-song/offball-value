"""Fernández and Bornn (2018) player-influence reference contour.

This module implements the parametric ellipse from *Wide Open Spaces* as a
soft visualization reference.  It is not a fixed-horizon reachable set and
must not be used as the hard endpoint-feasibility boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FernandezInfluenceEllipse:
    center_x: float
    center_y: float
    major_radius_m: float
    minor_radius_m: float
    angle_degrees: float
    base_radius_m: float
    distance_to_ball_m: float
    speed_mps: float
    speed_ratio: float
    contour_sigma: float = 2.0

    def as_record(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    def normalized_radius(self, x: float, y: float) -> float:
        """Return elliptical radius (one is the displayed two-sigma contour)."""

        angle = math.radians(self.angle_degrees)
        dx = x - self.center_x
        dy = y - self.center_y
        along = dx * math.cos(angle) + dy * math.sin(angle)
        lateral = -dx * math.sin(angle) + dy * math.cos(angle)
        return float(
            math.sqrt(
                (along / max(self.major_radius_m, 1e-12)) ** 2
                + (lateral / max(self.minor_radius_m, 1e-12)) ** 2
            )
        )


def fernandez_influence_radius(distance_to_ball_m: float) -> float:
    """Return the paper's 4--10 m distance-to-ball radius transform."""

    if not math.isfinite(distance_to_ball_m) or distance_to_ball_m < 0.0:
        raise ValueError("distance_to_ball_m must be finite and non-negative")
    return float(min(4.0 + distance_to_ball_m**3 / 1120.0, 10.0))


def fernandez_influence_ellipse(
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    maximum_speed_mps: float = 13.0,
) -> FernandezInfluenceEllipse:
    """Construct the paper's velocity-oriented two-sigma contour.

    Equation 19 defines standard-deviation scales equal to half of the base
    radius, expanded/contracted by the squared speed ratio.  Returning the
    corresponding two-sigma contour makes the zero-speed semi-axes equal to
    the paper's plotted influence radius.  The Gaussian itself remains
    unbounded; this ellipse is a visual contour, not a feasibility cutoff.
    """

    values = (*player_xy, *velocity_xy, *ball_xy, maximum_speed_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("influence inputs must be finite")
    if maximum_speed_mps <= 0.0:
        raise ValueError("maximum_speed_mps must be positive")

    vx, vy = velocity_xy
    speed = math.hypot(vx, vy)
    distance_to_ball = math.hypot(
        player_xy[0] - ball_xy[0],
        player_xy[1] - ball_xy[1],
    )
    radius = fernandez_influence_radius(distance_to_ball)
    speed_ratio = min(1.0, (speed / maximum_speed_mps) ** 2)
    angle = math.degrees(math.atan2(vy, vx)) if speed > 1e-12 else 0.0
    return FernandezInfluenceEllipse(
        center_x=float(player_xy[0] + 0.5 * vx),
        center_y=float(player_xy[1] + 0.5 * vy),
        major_radius_m=float(radius * (1.0 + speed_ratio)),
        minor_radius_m=float(radius * (1.0 - speed_ratio)),
        angle_degrees=float(angle),
        base_radius_m=float(radius),
        distance_to_ball_m=float(distance_to_ball),
        speed_mps=float(speed),
        speed_ratio=float(speed_ratio),
    )


def fernandez_influence_surface(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    player_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    maximum_speed_mps: float = 13.0,
) -> np.ndarray:
    """Evaluate the normalized Fernández player-influence Gaussian.

    The covariance and half-second velocity shift follow the same construction
    as :func:`fernandez_influence_ellipse`.  Fernández and Bornn normalize the
    bivariate density by its value at the player's current position.  Because
    the mean is shifted in the velocity direction, that ratio can be slightly
    above one near the shifted mean; we clip it to the paper's stated [0, 1]
    influence range.

    Parameters
    ----------
    xgrid, ygrid:
        One-dimensional pitch coordinates.  The returned array has shape
        ``(len(ygrid), len(xgrid))``.
    """

    x_values = np.asarray(xgrid, dtype=float)
    y_values = np.asarray(ygrid, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("xgrid and ygrid must be one-dimensional")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("grid coordinates must be finite")

    ellipse = fernandez_influence_ellipse(
        player_xy,
        velocity_xy,
        ball_xy,
        maximum_speed_mps=maximum_speed_mps,
    )
    angle = math.radians(ellipse.angle_degrees)
    xx, yy = np.meshgrid(x_values, y_values)
    dx = xx - ellipse.center_x
    dy = yy - ellipse.center_y
    along = dx * math.cos(angle) + dy * math.sin(angle)
    lateral = -dx * math.sin(angle) + dy * math.cos(angle)

    # The ellipse radii are the displayed two-sigma semi-axes.
    sigma_along = max(ellipse.major_radius_m / ellipse.contour_sigma, 1e-6)
    sigma_lateral = max(ellipse.minor_radius_m / ellipse.contour_sigma, 1e-6)
    mahalanobis = (along / sigma_along) ** 2 + (lateral / sigma_lateral) ** 2

    player_dx = player_xy[0] - ellipse.center_x
    player_dy = player_xy[1] - ellipse.center_y
    player_along = player_dx * math.cos(angle) + player_dy * math.sin(angle)
    player_lateral = -player_dx * math.sin(angle) + player_dy * math.cos(angle)
    player_mahalanobis = (
        (player_along / sigma_along) ** 2
        + (player_lateral / sigma_lateral) ** 2
    )
    normalized = np.exp(-0.5 * (mahalanobis - player_mahalanobis))
    return np.clip(normalized, 0.0, 1.0)
