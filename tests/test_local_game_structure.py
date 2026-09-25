from __future__ import annotations

import math
import unittest

from offball_value.local_game_structure import (
    LocalGameStructureConfig,
    build_structural_local_game,
    simulate_goal_side_response,
)
from offball_value.local_game_structure_audit import (
    render_structural_local_game_audit,
)


class LocalGameStructureTests(unittest.TestCase):
    def test_dynamic_response_is_finite_and_starts_from_current_state(self) -> None:
        actor = tuple((i / 10, 10 + i / 5, 2.0) for i in range(21))
        path = simulate_goal_side_response(
            (0.0, 0.0),
            (2.0, 0.0),
            actor,
            (52.5, 0.0),
            2.0,
        )

        self.assertEqual(path[0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(path[-1][0], 2.0)
        self.assertTrue(all(math.isfinite(value) for row in path for value in row))
        step_speeds = [
            math.hypot(after[1] - before[1], after[2] - before[2])
            / (after[0] - before[0])
            for before, after in zip(path, path[1:])
        ]
        self.assertLessEqual(max(step_speeds), 9.0 + 1e-6)

    def test_builds_three_defender_rows_and_keeps_carrier_visible(self) -> None:
        frames = []
        for index in range(-4, 17):
            time_s = index / 10
            players = [
                ["runner", "attack", 5 + 3 * time_s, 4.0, "Runner"],
                ["carrier", "attack", 0 + 2 * time_s, 0.0, "Carrier"],
                ["mate", "attack", 8 + time_s, -7.0, "Mate"],
                ["attack-gk", "attack", -45.0, 0.0, "Attack GK"],
                ["d1", "defend", 10 + time_s, 3.0, "Defender 1"],
                ["d2", "defend", 12.0, -1.0, "Defender 2"],
                ["d3", "defend", 13.0, 8.0, "Defender 3"],
                ["d4", "defend", 5.0, -12.0, "Defender 4"],
                ["defend-gk", "defend", 49.0, 0.0, "Defend GK"],
            ]
            frames.append(
                {
                    "relative_time_s": time_s,
                    "players": players,
                    "ball": [2 * time_s, 0.0],
                }
            )
        scene = {
            "match_id": "match",
            "match_label": "Attack vs Defend",
            "onset_frame_id": 100,
            "runner_id": "runner",
            "runner_name": "Runner",
            "carrier_id": "carrier",
            "carrier_name": "Carrier",
            "team_id": "attack",
            "attacking_direction": 1,
            "seconds_before_shot": 1.6,
            "frames": frames,
            "core_review": {"primary_defender": "d1", "derived_option": "carrier"},
        }

        game = build_structural_local_game(
            scene,
            LocalGameStructureConfig(
                maximum_horizon_seconds=1.6,
                minimum_horizon_seconds=1.0,
                # the (runner, defender) plausibility gate, added 2026-09-22, drops
                # the synthetic far defenders here; this test checks the row
                # structure before gating, so it switches the gate off
                pair_gate=False,
            ),
        )

        self.assertEqual(len(game["candidate_defenders"]), 3)
        self.assertIn("carrier", game["displayed_option_ids"])
        for defender in game["candidate_defenders"]:
            self.assertEqual(len(defender["cells"]), 2)
            for cell in defender["cells"]:
                self.assertGreaterEqual(cell["option_allocation_effect_fraction"], 0.0)
                self.assertGreaterEqual(cell["runner_cross_cost_fraction"], 0.0)

    def test_audit_html_identifies_structural_not_final_value(self) -> None:
        html = render_structural_local_game_audit([])
        self.assertIn("Defender × affected-option structure", html)
        self.assertIn("아직 <span class=\"formula\">P × G × A</span>", html)
        self.assertIn("response-controls", html)


if __name__ == "__main__":
    unittest.main()
