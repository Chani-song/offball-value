from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from offball_value.background_rollout import (
    BackgroundRolloutConfig,
    EmpiricalBackgroundPredictor,
    fit_damping_lambda,
    predict_constant_velocity,
    predict_damped_constant_velocity,
)
from offball_value.empirical_action_space import (
    CausalMotionState,
    EmpiricalPrimitiveLibrary,
)


class BackgroundRolloutTest(unittest.TestCase):
    def test_fit_damping_recovers_synthetic_decay(self) -> None:
        damping = 1.0
        frame = primitive_frame(
            lambda time_s: 4.0 * (1.0 - math.exp(-damping * time_s)) / damping
        )
        fitted, _ = fit_damping_lambda(EmpiricalPrimitiveLibrary(frame))
        self.assertAlmostEqual(fitted, damping, places=2)

    def test_damped_velocity_moves_less_than_constant_velocity(self) -> None:
        state = motion_state(speed=4.0, heading=0.0)
        config = BackgroundRolloutConfig()
        constant = predict_constant_velocity(0.0, 0.0, state, 2.0, config)
        damped = predict_damped_constant_velocity(
            0.0,
            0.0,
            state,
            2.0,
            1.0,
            config,
        )
        self.assertLess(damped.x, constant.x)
        self.assertLess(damped.speed_mps, constant.speed_mps)

    def test_empirical_reference_rotates_local_path_into_world(self) -> None:
        library = EmpiricalPrimitiveLibrary(primitive_frame(lambda time_s: 3.0 * time_s))
        config = BackgroundRolloutConfig(empirical_neighbor_count=4)
        predictor = EmpiricalBackgroundPredictor(library, config)
        state = motion_state(speed=4.0, heading=math.pi / 2.0)

        predictions = predictor.predict_many(10.0, -5.0, state, (1.0,))

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertAlmostEqual(prediction.x, 10.0, places=6)
        self.assertAlmostEqual(prediction.y, -2.0, places=6)
        self.assertEqual(prediction.empirical_neighbor_count, 4)

    def test_constant_velocity_clips_pitch_boundary_and_normal_velocity(self) -> None:
        state = motion_state(speed=9.0, heading=0.0)
        prediction = predict_constant_velocity(51.0, 0.0, state, 2.0)
        self.assertTrue(prediction.boundary_clipped)
        self.assertAlmostEqual(prediction.x, 52.5)
        self.assertAlmostEqual(prediction.vx_mps, 0.0)


def motion_state(speed: float, heading: float) -> CausalMotionState:
    return CausalMotionState(
        vx_mps=speed * math.cos(heading),
        vy_mps=speed * math.sin(heading),
        speed_mps=speed,
        ax_mps2=0.0,
        ay_mps2=0.0,
        longitudinal_acceleration_mps2=0.0,
        heading_radians=heading,
        sample_count=11,
        observed_window_seconds=0.4,
    )


def primitive_frame(displacement) -> pd.DataFrame:
    rows = []
    sample_times = (0.4, 0.8, 1.2, 1.6, 2.0)
    for index in range(8):
        row = {
            "match_id": "train",
            "player_id": f"player-{index}",
            "team_id": "team",
            "game_section": "firstHalf",
            "frame_id": index * 50,
            "initial_speed_mps": 4.0,
            "initial_longitudinal_acceleration_mps2": 0.0,
            "endpoint_forward_m": displacement(2.0),
            "endpoint_lateral_m": 0.0,
            "terminal_v_forward_mps": 0.0,
            "terminal_v_lateral_mps": 0.0,
            "terminal_speed_mps": 0.0,
        }
        for path_index, time_s in enumerate(sample_times, start=1):
            row[f"path_{path_index}_forward_m"] = displacement(time_s)
            row[f"path_{path_index}_lateral_m"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
