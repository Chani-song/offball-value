"""Post-reception goal-side accessibility built from player influence.

The delivery model answers whether the ball can reach an attacking option.
This module deliberately starts *after* that delivery: it moves the receiver
and ball to the candidate event point, then measures how much of the
receiver's goal-weighted Fernandez influence remains uncovered by outfield
defenders.

The returned accessibility is a transparent score in ``[0, 1]``.  It is not
calibrated as a probability of shooting or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .bundesliga import BundesligaFrame
from .goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    target_residual_influence,
)
from .pass_dynamics import VelocityEstimate


@dataclass(frozen=True)
class PostReceptionAccessibility:
    """Compact summary of usable goal-side space after an option succeeds."""

    accessibility_score: float
    covered_fraction: float
    intrinsic_value: float
    residual_value: float
    residual_peak: float
    residual_centroid_x: float
    residual_centroid_y: float


def post_reception_state(
    frame: BundesligaFrame,
    velocities: Mapping[str, VelocityEstimate],
    receiver_id: str,
    event_xy: tuple[float, float],
    event_velocity_xy: tuple[float, float] | None = None,
) -> tuple[BundesligaFrame, dict[str, VelocityEstimate]]:
    """Place the receiver and ball at the candidate reception/carry point."""

    if frame.ball is None:
        raise ValueError("post-reception state requires a ball")
    if receiver_id not in frame.players:
        raise KeyError(f"receiver {receiver_id} is missing")
    values = (*event_xy, *(event_velocity_xy or (0.0, 0.0)))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("event state must be finite")

    current_velocity = velocities.get(receiver_id)
    velocity_xy = (
        event_velocity_xy
        if event_velocity_xy is not None
        else (
            (float(current_velocity.vx), float(current_velocity.vy))
            if current_velocity is not None
            else (0.0, 0.0)
        )
    )
    speed = math.hypot(*velocity_xy)
    receiver = frame.players[receiver_id]
    event_frame = frame.with_player(
        receiver_id,
        replace(
            receiver,
            x=float(event_xy[0]),
            y=float(event_xy[1]),
            speed=speed * 3.6,
        ),
    )
    event_frame = replace(
        event_frame,
        ball=replace(
            frame.ball,
            x=float(event_xy[0]),
            y=float(event_xy[1]),
        ),
    )
    event_velocities = dict(velocities)
    event_velocities[receiver_id] = VelocityEstimate(
        float(velocity_xy[0]),
        float(velocity_xy[1]),
        float(speed),
        0,
        0.0,
    )
    return event_frame, event_velocities


def goal_side_accessibility(
    frame: BundesligaFrame,
    velocities: Mapping[str, VelocityEstimate],
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    event_xy: tuple[float, float],
    event_velocity_xy: tuple[float, float] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    config: GoalWeightedInfluenceConfig = GoalWeightedInfluenceConfig(),
    xgrid: np.ndarray | None = None,
    ygrid: np.ndarray | None = None,
) -> PostReceptionAccessibility:
    """Return residual goal-side influence after a successful option.

    Goalkeepers can be excluded because the geometric goal-danger term already
    represents the baseline difficulty of scoring from the location.  The
    accessibility term is then reserved for additional outfield pressure and
    goal-side coverage.
    """

    event_frame, event_velocities = post_reception_state(
        frame,
        velocities,
        receiver_id,
        event_xy,
        event_velocity_xy,
    )
    residual = target_residual_influence(
        event_frame,
        event_velocities,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        goalkeeper_ids=goalkeeper_ids,
        config=config,
        xgrid=xgrid,
        ygrid=ygrid,
    )
    accessibility = (
        float(
            np.clip(
                residual.residual_value / residual.intrinsic_value,
                0.0,
                1.0,
            )
        )
        if residual.intrinsic_value > 1e-12
        else 0.0
    )
    return PostReceptionAccessibility(
        accessibility_score=accessibility,
        covered_fraction=float(residual.covered_fraction),
        intrinsic_value=float(residual.intrinsic_value),
        residual_value=float(residual.residual_value),
        residual_peak=float(residual.residual_peak),
        residual_centroid_x=float(residual.residual_centroid_x),
        residual_centroid_y=float(residual.residual_centroid_y),
    )


def combine_delivery_and_accessibility(
    delivery_probability: float,
    endpoint_goal_danger: float,
    accessibility_score: float,
) -> float:
    """Apply the v0.5 value contract without calling it a probability."""

    values = (delivery_probability, endpoint_goal_danger, accessibility_score)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("value components must be finite")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("value components must lie in [0, 1]")
    return float(np.prod(values))
