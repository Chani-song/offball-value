from __future__ import annotations

import unittest

from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.goal_weighted_influence import GoalWeightedInfluenceConfig
from offball_value.pass_dynamics import VelocityEstimate
from offball_value.post_reception_value import (
    combine_delivery_and_accessibility,
    goal_side_accessibility,
    post_reception_state,
)


class PostReceptionValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = BundesligaFrame(
            match_id="m",
            frame_id=1,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={
                "a": BundesligaObjectState("a", "attack", 0.0, 0.0),
                "d": BundesligaObjectState("d", "defend", 12.0, 0.0),
                "gk": BundesligaObjectState("gk", "defend", 50.0, 0.0),
            },
            ball=BundesligaObjectState("ball", "ball", -10.0, 0.0),
        )
        self.velocities = {
            player_id: VelocityEstimate(0.0, 0.0, 0.0, 1, 0.4)
            for player_id in self.frame.players
        }
        self.config = GoalWeightedInfluenceConfig(grid_resolution_m=2.0)

    def test_post_reception_state_moves_receiver_and_ball(self) -> None:
        frame, velocities = post_reception_state(
            self.frame,
            self.velocities,
            "a",
            (18.0, 4.0),
            (3.0, 0.0),
        )

        self.assertEqual((frame.players["a"].x, frame.players["a"].y), (18.0, 4.0))
        self.assertEqual((frame.ball.x, frame.ball.y), (18.0, 4.0))
        self.assertAlmostEqual(velocities["a"].speed, 3.0)

    def test_goal_side_defender_reduces_post_reception_accessibility(self) -> None:
        open_frame = self.frame.with_player(
            "d", BundesligaObjectState("d", "defend", -20.0, 20.0)
        )
        blocked_frame = self.frame.with_player(
            "d", BundesligaObjectState("d", "defend", 18.0, 0.0)
        )

        open_value = goal_side_accessibility(
            open_frame,
            self.velocities,
            "a",
            "attack",
            1,
            (10.0, 0.0),
            goalkeeper_ids=("gk",),
            config=self.config,
        )
        blocked_value = goal_side_accessibility(
            blocked_frame,
            self.velocities,
            "a",
            "attack",
            1,
            (10.0, 0.0),
            goalkeeper_ids=("gk",),
            config=self.config,
        )

        self.assertGreater(open_value.accessibility_score, blocked_value.accessibility_score)
        self.assertLess(open_value.covered_fraction, blocked_value.covered_fraction)

    def test_goalkeeper_exclusion_keeps_outfield_accessibility_separate(self) -> None:
        included = goal_side_accessibility(
            self.frame,
            self.velocities,
            "a",
            "attack",
            1,
            (20.0, 0.0),
            config=self.config,
        )
        excluded = goal_side_accessibility(
            self.frame,
            self.velocities,
            "a",
            "attack",
            1,
            (20.0, 0.0),
            goalkeeper_ids=("gk",),
            config=self.config,
        )

        self.assertGreater(excluded.accessibility_score, included.accessibility_score)

    def test_combined_contract_is_multiplicative_and_bounded(self) -> None:
        self.assertAlmostEqual(
            combine_delivery_and_accessibility(0.8, 0.5, 0.75),
            0.3,
        )
        with self.assertRaises(ValueError):
            combine_delivery_and_accessibility(1.1, 0.5, 0.75)


if __name__ == "__main__":
    unittest.main()
