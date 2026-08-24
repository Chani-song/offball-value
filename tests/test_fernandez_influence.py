from __future__ import annotations

import unittest

from offball_value.fernandez_influence import (
    fernandez_influence_ellipse,
    fernandez_influence_radius,
    fernandez_influence_surface,
)
from offball_value.goal_weighted_influence import goal_weighted_space_surface
from offball_value.goal_weighted_influence import (
    GoalWeightedInfluenceConfig,
    target_residual_influence,
)
from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.pass_dynamics import VelocityEstimate

import numpy as np


class FernandezInfluenceTest(unittest.TestCase):
    def test_radius_matches_paper_curve_and_cap(self) -> None:
        self.assertAlmostEqual(fernandez_influence_radius(0.0), 4.0)
        self.assertAlmostEqual(
            fernandez_influence_radius(15.0),
            4.0 + 15.0**3 / 1120.0,
        )
        self.assertEqual(fernandez_influence_radius(30.0), 10.0)

    def test_ellipse_shifts_and_stretches_along_velocity(self) -> None:
        ellipse = fernandez_influence_ellipse(
            (1.0, 2.0),
            (6.5, 0.0),
            (1.0, 2.0),
        )

        self.assertAlmostEqual(ellipse.center_x, 4.25)
        self.assertAlmostEqual(ellipse.center_y, 2.0)
        self.assertAlmostEqual(ellipse.angle_degrees, 0.0)
        self.assertGreater(ellipse.major_radius_m, ellipse.base_radius_m)
        self.assertLess(ellipse.minor_radius_m, ellipse.base_radius_m)

    def test_surface_is_normalized_and_velocity_oriented(self) -> None:
        grid = np.linspace(-8.0, 8.0, 65)
        influence = fernandez_influence_surface(
            grid,
            grid,
            (0.0, 0.0),
            (4.0, 0.0),
            (0.0, 0.0),
        )

        self.assertGreaterEqual(float(np.min(influence)), 0.0)
        self.assertLessEqual(float(np.max(influence)), 1.0)
        center = len(grid) // 2
        self.assertGreater(influence[center, center + 8], influence[center + 8, center])

    def test_goal_weight_values_goal_side_more_than_behind(self) -> None:
        xgrid = np.asarray([-10.0, 0.0, 10.0])
        ygrid = np.asarray([0.0])
        value = goal_weighted_space_surface(
            xgrid,
            ygrid,
            attacker_xy=(0.0, 0.0),
            attacking_direction=-1,
        )

        self.assertGreater(value[0, 0], value[0, 2])

    def test_goal_side_defender_reduces_residual_target_influence(self) -> None:
        target = BundesligaObjectState("a", "attack", 0.0, 0.0)
        ball = BundesligaObjectState("ball", "ball", -8.0, 0.0)
        near = BundesligaFrame(
            match_id="m",
            frame_id=1,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={
                "a": target,
                "d": BundesligaObjectState("d", "defend", 4.0, 0.0),
            },
            ball=ball,
        )
        far = near.with_player(
            "d", BundesligaObjectState("d", "defend", -25.0, 20.0)
        )
        velocities = {
            player_id: VelocityEstimate(0.0, 0.0, 0.0, 1, 0.4)
            for player_id in ("a", "d")
        }
        config = GoalWeightedInfluenceConfig(grid_resolution_m=2.0)

        near_value = target_residual_influence(
            near, velocities, "a", "attack", 1, config=config
        )
        far_value = target_residual_influence(
            far, velocities, "a", "attack", 1, config=config
        )

        self.assertLess(near_value.residual_value, far_value.residual_value)
        self.assertGreater(near_value.covered_fraction, far_value.covered_fraction)


if __name__ == "__main__":
    unittest.main()
