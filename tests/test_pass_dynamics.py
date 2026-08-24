from __future__ import annotations

import math
import unittest

import numpy as np

from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    detect_kick_frame,
    estimate_velocity,
    pass_execution_estimate,
    path_survival_estimate,
    player_time_to_point,
    point_reception_estimate,
    receiver_target_region,
    reception_region_estimate,
    secure_possession_probability,
)


def frame(
    frame_id: int,
    players: dict[str, BundesligaObjectState],
    ball: BundesligaObjectState | None = None,
) -> BundesligaFrame:
    return BundesligaFrame(
        match_id="test",
        frame_id=frame_id,
        period=1,
        game_section="firstHalf",
        timestamp=None,
        players=players,
        ball=ball,
    )


class PassDynamicsTest(unittest.TestCase):
    def test_detect_kick_frame_uses_last_controlled_frame(self) -> None:
        passer = BundesligaObjectState("passer", "attack", 0.0, 0.0)
        frames = [
            frame(
                0,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 0.2, 0.0, speed=2.0),
            ),
            frame(
                1,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 0.4, 0.0, speed=4.0),
            ),
            frame(
                2,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 1.8, 0.0, speed=36.0),
            ),
            frame(
                3,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 2.2, 0.0, speed=32.0),
            ),
        ]

        kick = detect_kick_frame(frames, "passer", 3)

        self.assertEqual(kick.kick_frame_id, 1)
        self.assertEqual(kick.release_frame_id, 2)
        self.assertEqual(kick.offset_frames, -2)
        self.assertAlmostEqual(kick.release_ball_speed_mps, 36.0 / 3.6)
        self.assertEqual(kick.target_alignment, 1.0)
        self.assertTrue(kick.detected)

    def test_detect_kick_frame_can_find_release_after_event(self) -> None:
        passer = BundesligaObjectState("passer", "attack", 0.0, 0.0)
        frames = [
            frame(
                1,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 0.2, 0.0, speed=2.0),
            ),
            frame(
                2,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 0.4, 0.0, speed=4.0),
            ),
            frame(
                3,
                {"passer": passer},
                BundesligaObjectState("ball", "BALL", 1.8, 0.0, speed=36.0),
            ),
        ]

        kick = detect_kick_frame(frames, "passer", 1)

        self.assertEqual(kick.kick_frame_id, 2)
        self.assertEqual(kick.offset_frames, 1)
        self.assertTrue(kick.detected)

    def test_estimate_velocity_uses_only_pre_event_history(self) -> None:
        frames = []
        for frame_id in range(11):
            time_s = frame_id / 25.0
            players = {
                "runner": BundesligaObjectState(
                    object_id="runner",
                    team_id="attack",
                    x=2.0 * time_s,
                    y=-1.0 * time_s,
                )
            }
            frames.append(frame(frame_id, players))

        velocity = estimate_velocity(frames, "runner", 10)

        self.assertAlmostEqual(velocity.vx, 2.0, places=6)
        self.assertAlmostEqual(velocity.vy, -1.0, places=6)
        self.assertAlmostEqual(velocity.speed, math.sqrt(5.0), places=6)
        self.assertEqual(velocity.sample_count, 11)

    def test_quadratic_velocity_estimates_end_of_window_derivative(self) -> None:
        frames = []
        for frame_id in range(11):
            time_s = frame_id / 25.0
            players = {
                "runner": BundesligaObjectState(
                    object_id="runner",
                    team_id="attack",
                    x=4.0 * time_s - 2.0 * time_s**2,
                    y=0.0,
                )
            }
            frames.append(frame(frame_id, players))

        linear = estimate_velocity(frames, "runner", 10)
        endpoint = estimate_velocity(
            frames,
            "runner",
            10,
            ArrivalModelConfig(velocity_estimator="quadratic_endpoint"),
        )

        self.assertAlmostEqual(endpoint.vx, 2.4, places=6)
        self.assertGreater(linear.vx, endpoint.vx)
        self.assertEqual(endpoint.sample_count, 11)

    def test_current_direction_changes_time_to_target(self) -> None:
        player = BundesligaObjectState(
            object_id="runner",
            team_id="attack",
            x=0.0,
            y=0.0,
        )
        toward = VelocityEstimate(5.0, 0.0, 5.0, 11, 0.4)
        away = VelocityEstimate(-5.0, 0.0, 5.0, 11, 0.4)

        toward_time = player_time_to_point(player, toward, (10.0, 0.0))
        away_time = player_time_to_point(player, away, (10.0, 0.0))

        self.assertLess(toward_time, away_time)

    def test_defender_arriving_before_ball_reduces_probability(self) -> None:
        players = {
            "receiver": BundesligaObjectState("receiver", "attack", 10.0, 0.0),
            "near": BundesligaObjectState("near", "defense", 10.5, 0.0),
            "far": BundesligaObjectState("far", "defense", 25.0, 0.0),
        }
        velocities = {
            player_id: VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4)
            for player_id in players
        }

        estimate = point_reception_estimate(
            frame(10, players),
            velocities,
            "receiver",
            "attack",
            (0.0, 0.0),
            (10.0, 0.0),
            ArrivalModelConfig(pass_speed_mps=10.0),
        )

        self.assertEqual(estimate.nearest_defender_id, "near")
        self.assertLess(estimate.receiver_first_probability, 0.5)

    def test_defender_on_pass_path_reduces_path_survival(self) -> None:
        velocities = {
            "on_path": VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4),
            "off_path": VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4),
        }
        on_path_frame = frame(
            10,
            {
                "on_path": BundesligaObjectState("on_path", "defense", 5.0, 0.0),
            },
        )
        off_path_frame = frame(
            10,
            {
                "off_path": BundesligaObjectState("off_path", "defense", 5.0, 10.0),
            },
        )

        on_path = path_survival_estimate(
            on_path_frame,
            velocities,
            "attack",
            (0.0, 0.0),
            (10.0, 0.0),
        )
        off_path = path_survival_estimate(
            off_path_frame,
            velocities,
            "attack",
            (0.0, 0.0),
            (10.0, 0.0),
        )

        self.assertLess(on_path[3], off_path[3])

    def test_secure_possession_requires_time_before_pressure(self) -> None:
        contested = secure_possession_probability(0.0)
        open_receiver = secure_possession_probability(1.0)

        self.assertLess(contested, open_receiver)

    def test_passer_pressure_reduces_execution_probability(self) -> None:
        passer = BundesligaObjectState("passer", "attack", 0.0, 0.0)
        near_frame = frame(
            10,
            {
                "passer": passer,
                "near": BundesligaObjectState("near", "defense", 1.0, 0.0),
            },
        )
        far_frame = frame(
            10,
            {
                "passer": passer,
                "far": BundesligaObjectState("far", "defense", 15.0, 0.0),
            },
        )
        velocities = {
            "near": VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4),
            "far": VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4),
        }

        near = pass_execution_estimate(
            near_frame,
            velocities,
            "passer",
            "attack",
            15.0,
        )
        far = pass_execution_estimate(
            far_frame,
            velocities,
            "passer",
            "attack",
            15.0,
        )

        self.assertLess(near[4], far[4])

    def test_receiver_region_prior_is_normalized_and_velocity_aligned(self) -> None:
        players = {
            "receiver": BundesligaObjectState("receiver", "attack", 0.0, 0.0),
        }
        velocities = {
            "receiver": VelocityEstimate(6.0, 0.0, 6.0, 11, 0.4),
        }

        points, prior = receiver_target_region(
            frame(10, players),
            velocities,
            "receiver",
            attacking_direction=1,
        )

        self.assertAlmostEqual(float(np.sum(prior)), 1.0, places=9)
        self.assertGreater(float(np.max(points[:, 0])), 8.0)
        self.assertGreater(
            float(np.sum(prior[points[:, 0] > 0.0])),
            float(np.sum(prior[points[:, 0] < 0.0])),
        )

    def test_region_option_value_integrates_point_probabilities(self) -> None:
        players = {
            "passer": BundesligaObjectState("passer", "attack", -5.0, 0.0),
            "receiver": BundesligaObjectState("receiver", "attack", 0.0, 0.0),
        }
        velocities = {
            player_id: VelocityEstimate(0.0, 0.0, 0.0, 11, 0.4)
            for player_id in players
        }

        estimate = reception_region_estimate(
            frame(10, players),
            velocities,
            "receiver",
            "passer",
            "attack",
            1,
            (-5.0, 0.0),
            receive_value_fn=lambda points, _direction: np.ones(len(points)),
        )

        self.assertGreater(estimate.point_count, 1)
        self.assertGreater(estimate.option_value, 0.0)
        self.assertLessEqual(estimate.option_value, 1.0)
        self.assertAlmostEqual(
            estimate.option_value,
            estimate.expected_receive_probability,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
