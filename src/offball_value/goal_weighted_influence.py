"""Goal-weighted residual influence for target-specific defensive choices.

The player influence is the Fernández and Bornn (2018) Gaussian.  The
goal-side and goal-distance weights are an explicit research extension: they
make a defender valuable for controlling the space between an attacker and
the defended goal, rather than only for occupying one high-valued grid cell.
They must not be described as part of the original Fernández formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .bundesliga import FIELD_LENGTH, FIELD_WIDTH, BundesligaFrame
from .fernandez_influence import fernandez_influence_surface
from .pass_dynamics import VelocityEstimate


@dataclass(frozen=True)
class GoalWeightedInfluenceConfig:
    grid_resolution_m: float = 1.0
    maximum_influence_speed_mps: float = 13.0
    goal_distance_power: float = 1.5
    goal_side_softness_m: float = 1.5
    goal_side_floor: float = 0.20
    goal_cone_base_half_width_m: float = 2.5
    goal_cone_growth_per_m: float = 0.20
    goal_cone_floor: float = 0.20
    defender_suppression_strength: float = 1.25

    def validate(self) -> None:
        if self.grid_resolution_m <= 0.0:
            raise ValueError("grid_resolution_m must be positive")
        if self.maximum_influence_speed_mps <= 0.0:
            raise ValueError("maximum_influence_speed_mps must be positive")
        if self.goal_distance_power <= 0.0:
            raise ValueError("goal_distance_power must be positive")
        if self.goal_side_softness_m <= 0.0:
            raise ValueError("goal_side_softness_m must be positive")
        if not 0.0 <= self.goal_side_floor <= 1.0:
            raise ValueError("goal_side_floor must lie in [0, 1]")
        if self.goal_cone_base_half_width_m <= 0.0:
            raise ValueError("goal_cone_base_half_width_m must be positive")
        if self.goal_cone_growth_per_m < 0.0:
            raise ValueError("goal_cone_growth_per_m cannot be negative")
        if not 0.0 <= self.goal_cone_floor <= 1.0:
            raise ValueError("goal_cone_floor must lie in [0, 1]")
        if self.defender_suppression_strength <= 0.0:
            raise ValueError("defender_suppression_strength must be positive")


@dataclass(frozen=True)
class TargetResidualInfluence:
    target_id: str
    residual_value: float
    intrinsic_value: float
    covered_fraction: float
    residual_peak: float
    residual_centroid_x: float
    residual_centroid_y: float
    target_influence: np.ndarray
    defender_influence: np.ndarray
    space_value: np.ndarray
    residual_surface: np.ndarray


def influence_pitch_grid(
    config: GoalWeightedInfluenceConfig = GoalWeightedInfluenceConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return a regular cell-centre grid over the 105 x 68 m pitch."""

    config.validate()
    nx = int(math.ceil(FIELD_LENGTH / config.grid_resolution_m))
    ny = int(math.ceil(FIELD_WIDTH / config.grid_resolution_m))
    xgrid = np.linspace(
        -FIELD_LENGTH / 2.0 + FIELD_LENGTH / (2.0 * nx),
        FIELD_LENGTH / 2.0 - FIELD_LENGTH / (2.0 * nx),
        nx,
    )
    ygrid = np.linspace(
        -FIELD_WIDTH / 2.0 + FIELD_WIDTH / (2.0 * ny),
        FIELD_WIDTH / 2.0 - FIELD_WIDTH / (2.0 * ny),
        ny,
    )
    return xgrid, ygrid


def goal_weighted_space_surface(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    attacker_xy: tuple[float, float],
    attacking_direction: int,
    config: GoalWeightedInfluenceConfig = GoalWeightedInfluenceConfig(),
) -> np.ndarray:
    """Value goal-proximal, goal-side space around one attacker.

    ``attacking_direction`` is +1 for the right-hand goal and -1 for the
    left-hand goal.  A soft half-plane and cone emphasize the channel from the
    attacker toward goal.  Floors retain non-zero value outside that channel,
    allowing lateral support and cut-back space to remain visible.
    """

    config.validate()
    if attacking_direction not in (-1, 1):
        raise ValueError("attacking_direction must be -1 or +1")
    xx, yy = np.meshgrid(np.asarray(xgrid, dtype=float), np.asarray(ygrid, dtype=float))
    goal = np.asarray((attacking_direction * FIELD_LENGTH / 2.0, 0.0), dtype=float)
    attacker = np.asarray(attacker_xy, dtype=float)
    to_goal = goal - attacker
    goal_distance = float(np.linalg.norm(to_goal))
    if goal_distance <= 1e-9:
        unit = np.asarray((float(attacking_direction), 0.0))
    else:
        unit = to_goal / goal_distance
    dx = xx - attacker[0]
    dy = yy - attacker[1]
    along = dx * unit[0] + dy * unit[1]
    lateral = np.abs(-dx * unit[1] + dy * unit[0])

    side = 1.0 / (1.0 + np.exp(-along / config.goal_side_softness_m))
    side = config.goal_side_floor + (1.0 - config.goal_side_floor) * side
    half_width = (
        config.goal_cone_base_half_width_m
        + config.goal_cone_growth_per_m * np.clip(along, 0.0, goal_distance)
    )
    cone = np.exp(-0.5 * (lateral / half_width) ** 2)
    cone = config.goal_cone_floor + (1.0 - config.goal_cone_floor) * cone

    distance_to_goal = np.hypot(xx - goal[0], yy - goal[1])
    farthest = math.hypot(FIELD_LENGTH, FIELD_WIDTH / 2.0)
    proximity = np.clip(1.0 - distance_to_goal / farthest, 0.0, 1.0)
    return proximity**config.goal_distance_power * side * cone


def _velocity_xy(
    player_id: str,
    velocities: Mapping[str, VelocityEstimate],
) -> tuple[float, float]:
    velocity = velocities.get(player_id)
    if velocity is None:
        return 0.0, 0.0
    return float(velocity.vx), float(velocity.vy)


def target_residual_influence(
    frame: BundesligaFrame,
    velocities: Mapping[str, VelocityEstimate],
    target_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str] = (),
    config: GoalWeightedInfluenceConfig = GoalWeightedInfluenceConfig(),
    xgrid: np.ndarray | None = None,
    ygrid: np.ndarray | None = None,
) -> TargetResidualInfluence:
    """Integrate target influence that remains after all defenders' influence.

    Defensive influence is combined as a soft coverage field.  The exponential
    residual makes overlapping defenders matter: two defenders controlling the
    same lane suppress more of it than one.  This is a transparent v0.1
    operationalization, not a learned scoring-probability model.
    """

    config.validate()
    if frame.ball is None:
        raise ValueError("a ball state is required")
    if target_id not in frame.players:
        raise KeyError(f"target {target_id} is missing")
    target = frame.players[target_id]
    if target.team_id != attacking_team_id:
        raise ValueError("target must belong to the attacking team")
    if xgrid is None or ygrid is None:
        xgrid, ygrid = influence_pitch_grid(config)
    x_values = np.asarray(xgrid, dtype=float)
    y_values = np.asarray(ygrid, dtype=float)
    ball_xy = (float(frame.ball.x), float(frame.ball.y))

    target_surface = fernandez_influence_surface(
        x_values,
        y_values,
        (float(target.x), float(target.y)),
        _velocity_xy(target_id, velocities),
        ball_xy,
        maximum_speed_mps=config.maximum_influence_speed_mps,
    )
    defender_sum = np.zeros_like(target_surface)
    excluded = set(goalkeeper_ids)
    for player_id, player in frame.players.items():
        if player.team_id == attacking_team_id or player_id in excluded:
            continue
        defender_sum += fernandez_influence_surface(
            x_values,
            y_values,
            (float(player.x), float(player.y)),
            _velocity_xy(player_id, velocities),
            ball_xy,
            maximum_speed_mps=config.maximum_influence_speed_mps,
        )
    space_value = goal_weighted_space_surface(
        x_values,
        y_values,
        (float(target.x), float(target.y)),
        attacking_direction,
        config,
    )
    intrinsic = target_surface * space_value
    uncovered = np.exp(-config.defender_suppression_strength * defender_sum)
    residual = intrinsic * uncovered
    cell_area = (FIELD_LENGTH / len(x_values)) * (FIELD_WIDTH / len(y_values))
    intrinsic_value = float(np.sum(intrinsic) * cell_area)
    residual_value = float(np.sum(residual) * cell_area)
    covered_fraction = (
        float(np.clip(1.0 - residual_value / intrinsic_value, 0.0, 1.0))
        if intrinsic_value > 1e-12
        else 0.0
    )
    total = float(np.sum(residual))
    xx, yy = np.meshgrid(x_values, y_values)
    if total > 1e-12:
        centroid_x = float(np.sum(xx * residual) / total)
        centroid_y = float(np.sum(yy * residual) / total)
    else:
        centroid_x, centroid_y = float(target.x), float(target.y)
    return TargetResidualInfluence(
        target_id=target_id,
        residual_value=residual_value,
        intrinsic_value=intrinsic_value,
        covered_fraction=covered_fraction,
        residual_peak=float(np.max(residual)),
        residual_centroid_x=centroid_x,
        residual_centroid_y=centroid_y,
        target_influence=target_surface,
        defender_influence=defender_sum,
        space_value=space_value,
        residual_surface=residual,
    )
