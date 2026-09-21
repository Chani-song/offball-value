from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import BundesligaFrame, BundesligaObjectState
from offball_value.empirical_action_space import (
    EmpiricalEndpointConfig,
    EmpiricalPrimitiveConfig,
    EmpiricalPrimitiveLibrary,
    causal_motion_state_from_frames,
    extract_track_primitives,
    generate_empirical_endpoint_actions,
    observed_empirical_support_diagnostics,
)


class EmpiricalActionSpaceTest(unittest.TestCase):
    def test_extract_track_uses_causal_state_and_future_path(self) -> None:
        ids = list(range(0, 81))
        times = np.asarray(ids) / 25.0
        xs = 3.0 * times - 0.5 * times**2
        ys = np.zeros_like(times)
        config = EmpiricalPrimitiveConfig(sample_interval_seconds=10.0)

        result = extract_track_primitives(
            "train",
            "runner",
            "team",
            "firstHalf",
            ids,
            xs,
            ys,
            config,
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["initial_speed_mps"], 2.6, places=6)
        self.assertAlmostEqual(
            result.iloc[0]["initial_longitudinal_acceleration_mps2"],
            -1.0,
            places=6,
        )

    def test_library_round_trip_and_target_match_guard(self) -> None:
        frame = primitive_frame()
        library = EmpiricalPrimitiveLibrary(frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "library.npz"
            library.save(path)
            restored = EmpiricalPrimitiveLibrary.load(path)

        self.assertEqual(len(restored.frame), len(frame))
        self.assertEqual(restored.source_match_ids, ("train",))
        restored.assert_excludes_match("target")
        with self.assertRaises(ValueError):
            restored.assert_excludes_match("train")

    def test_extract_track_rejects_mid_horizon_tracking_teleport(self) -> None:
        ids = list(range(0, 81))
        times = np.asarray(ids) / 25.0
        xs = 2.0 * times
        xs[30] += 20.0
        result = extract_track_primitives(
            "train",
            "runner",
            "team",
            "firstHalf",
            ids,
            xs,
            np.zeros_like(times),
            EmpiricalPrimitiveConfig(sample_interval_seconds=10.0),
        )
        self.assertTrue(result.empty)

    def test_candidates_keep_terminal_heading_variants_without_target_future(self) -> None:
        library = EmpiricalPrimitiveLibrary(primitive_frame())
        frames = []
        for frame_id in range(11):
            frames.append(
                BundesligaFrame(
                    "target",
                    frame_id,
                    1,
                    "firstHalf",
                    None,
                    {
                        "runner": BundesligaObjectState(
                            "runner", "attack", frame_id * 0.04 * 3.0, 0.0
                        )
                    },
                    BundesligaObjectState("ball", "BALL", 0.0, 0.0),
                )
            )
        state = causal_motion_state_from_frames(frames, "runner", 10)
        action_set = generate_empirical_endpoint_actions(
            frames[-1].players["runner"],
            frames[-1].ball,
            state,
            library,
            EmpiricalEndpointConfig(neighbor_count=4, maximum_actions=100),
        )

        self.assertTrue(action_set.actions)
        self.assertNotIn("target", action_set.source_match_ids)
        endpoints = {}
        for action in action_set.actions:
            endpoints.setdefault((action.endpoint_x, action.endpoint_y), set()).add(
                action.terminal_heading_bin
            )
        self.assertTrue(any(len(bins) > 1 for bins in endpoints.values()))

        reference = action_set.actions[0]
        supported = observed_empirical_support_diagnostics(
            action_set,
            (reference.endpoint_x, reference.endpoint_y),
            (reference.terminal_vx_mps, reference.terminal_vy_mps),
        )
        self.assertTrue(supported.endpoint_supported)
        self.assertTrue(supported.endpoint_heading_supported)
        self.assertAlmostEqual(
            supported.nearest_terminal_heading_difference_degrees,
            0.0,
        )

        unsupported = observed_empirical_support_diagnostics(
            action_set,
            (50.0, 30.0),
            (1.0, 0.0),
        )
        self.assertFalse(unsupported.endpoint_supported)
        self.assertFalse(unsupported.endpoint_heading_supported)


def primitive_frame() -> pd.DataFrame:
    rows = []
    for index, (lateral, terminal_lateral) in enumerate(
        [(-1.0, -2.0), (1.0, 2.0), (-1.0, 2.0), (1.0, -2.0)]
    ):
        row = {
            "match_id": "train",
            "player_id": f"p{index}",
            "team_id": "team",
            "game_section": "firstHalf",
            "frame_id": index * 50,
            "initial_speed_mps": 3.0,
            "initial_longitudinal_acceleration_mps2": 0.0,
            "endpoint_forward_m": 4.0,
            "endpoint_lateral_m": lateral,
            "terminal_v_forward_mps": 1.0,
            "terminal_v_lateral_mps": terminal_lateral,
            "terminal_speed_mps": float(np.hypot(1.0, terminal_lateral)),
        }
        for path_index, fraction in enumerate((0.2, 0.4, 0.6, 0.8, 1.0), start=1):
            row[f"path_{path_index}_forward_m"] = 4.0 * fraction
            row[f"path_{path_index}_lateral_m"] = lateral * fraction
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
