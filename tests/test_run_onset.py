from __future__ import annotations

import unittest

import numpy as np

from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.run_onset import (
    RunOnsetConfig,
    ball_is_inside_pitch,
    detect_kinematic_run_onsets,
    motion_signal_records,
    team_controls_ball,
)


FPS = 25


def integrate_velocity(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cumsum(vx) / FPS, np.cumsum(vy) / FPS


class RunOnsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RunOnsetConfig(smoothing_seconds=0.12)

    def test_detects_acceleration_near_start_of_speed_change(self) -> None:
        frames = np.arange(130)
        vx = np.ones(130)
        vx[50:70] = np.linspace(1.0, 5.0, 20)
        vx[70:] = 5.0
        x, y = integrate_velocity(vx, np.zeros_like(vx))

        onsets = detect_kinematic_run_onsets(frames, x, y, self.config)
        acceleration = [onset for onset in onsets if "acceleration" in onset.labels]

        self.assertTrue(acceleration)
        self.assertLessEqual(abs(acceleration[0].frame_id - 50), 7)
        self.assertGreaterEqual(acceleration[0].speed_gain_mps, 1.5)

    def test_detects_direction_change_while_speed_is_preserved(self) -> None:
        frames = np.arange(150)
        angles = np.zeros(150)
        angles[55:76] = np.linspace(0.0, np.pi / 2.0, 21)
        angles[76:] = np.pi / 2.0
        vx = 4.0 * np.cos(angles)
        vy = 4.0 * np.sin(angles)
        x, y = integrate_velocity(vx, vy)

        onsets = detect_kinematic_run_onsets(frames, x, y, self.config)
        turns = [onset for onset in onsets if "direction_change" in onset.labels]

        self.assertTrue(turns)
        self.assertLessEqual(abs(turns[0].frame_id - 55), 8)
        self.assertGreaterEqual(turns[0].direction_change_degrees, 30.0)

    def test_constant_velocity_does_not_create_an_onset(self) -> None:
        frames = np.arange(120)
        x, y = integrate_velocity(np.full(120, 4.0), np.full(120, 1.0))

        self.assertEqual(
            detect_kinematic_run_onsets(frames, x, y, self.config),
            (),
        )

    def test_gaps_are_not_differentiated_across(self) -> None:
        frames = np.r_[np.arange(50), np.arange(80, 130)]
        x = np.r_[np.arange(50) / FPS, 30.0 + np.arange(50) / FPS]
        y = np.zeros_like(x)

        self.assertEqual(
            detect_kinematic_run_onsets(frames, x, y, self.config),
            (),
        )

    def test_signal_records_have_detector_fields(self) -> None:
        frames = np.arange(20)
        x, y = integrate_velocity(np.full(20, 2.0), np.zeros(20))
        records = motion_signal_records(frames, x, y, self.config)

        self.assertEqual(len(records), 20)
        self.assertEqual(records[10]["frame_id"], 10)
        self.assertAlmostEqual(records[10]["speed_mps"], 2.0, places=5)
        self.assertIn("turn_rate_degrees_s", records[10])

    def test_team_control_requires_teammate_to_be_closest_to_ball(self) -> None:
        frame = BundesligaFrame(
            match_id="match",
            frame_id=10,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={
                "attacker": BundesligaObjectState(
                    "attacker", "attack", 0.8, 0.0
                ),
                "defender": BundesligaObjectState(
                    "defender", "defence", 0.4, 0.0
                ),
            },
            ball=BundesligaObjectState("ball", "BALL", 0.0, 0.0),
        )

        self.assertFalse(team_controls_ball(frame, "attack", 1.5))
        self.assertTrue(team_controls_ball(frame, "defence", 1.5))

    def test_team_control_allows_the_controlling_teammate_to_change(self) -> None:
        frame = BundesligaFrame(
            match_id="match",
            frame_id=10,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={
                "first": BundesligaObjectState("first", "attack", 2.0, 0.0),
                "second": BundesligaObjectState("second", "attack", 0.5, 0.0),
                "defender": BundesligaObjectState(
                    "defender", "defence", 1.2, 0.0
                ),
            },
            ball=BundesligaObjectState("ball", "BALL", 0.0, 0.0),
        )

        self.assertTrue(team_controls_ball(frame, "attack", 1.5))

    def test_ball_inside_pitch_uses_tracking_tolerance(self) -> None:
        inside = BundesligaFrame(
            match_id="match",
            frame_id=10,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={},
            ball=BundesligaObjectState("ball", "BALL", 10.0, 34.1),
        )
        outside = BundesligaFrame(
            match_id="match",
            frame_id=11,
            period=1,
            game_section="firstHalf",
            timestamp=None,
            players={},
            ball=BundesligaObjectState("ball", "BALL", 10.0, 34.45),
        )

        self.assertTrue(ball_is_inside_pitch(inside))
        self.assertFalse(ball_is_inside_pitch(outside))


if __name__ == "__main__":
    unittest.main()
