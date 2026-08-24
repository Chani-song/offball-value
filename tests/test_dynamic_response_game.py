import unittest

import numpy as np

from offball_value.dynamic_response_game import (
    normalized_option_regret,
    pareto_response_indices,
    select_dynamic_game_responses,
)


class DynamicResponseGameTest(unittest.TestCase):
    def test_option_minima_and_raw_local_minimax(self) -> None:
        costs = np.asarray(
            [
                [0.08, 0.18, 0.35],
                [0.45, 0.25, 0.12],
            ]
        )
        selected = select_dynamic_game_responses(costs, (3.0, 2.0, 1.0))
        self.assertEqual(selected.option_minimum_indices, (0, 2))
        self.assertEqual(selected.local_minimax_index, 1)

    def test_compromise_uses_relative_branch_regret(self) -> None:
        costs = np.asarray(
            [
                [2.0, 3.0, 5.0],
                [20.0, 21.0, 22.0],
            ]
        )
        selected = select_dynamic_game_responses(costs, (0.0, 0.0, 0.0))
        # Raw minimax follows the large second-option scale and selects column 0.
        self.assertEqual(selected.local_minimax_index, 0)
        # Relative regret also selects the shared minimum here.
        self.assertEqual(selected.compromise_index, 0)
        regret = normalized_option_regret(costs)
        np.testing.assert_allclose(regret[:, 0], (0.0, 0.0))

    def test_compromise_can_differ_from_raw_minimax(self) -> None:
        costs = np.asarray(
            [
                [1.0, 2.0, 5.0],
                [12.0, 11.0, 10.0],
            ]
        )
        selected = select_dynamic_game_responses(costs, (0.0, 0.0, 0.0))
        self.assertEqual(selected.local_minimax_index, 2)
        self.assertEqual(selected.compromise_index, 1)

    def test_pareto_excludes_jointly_worse_response(self) -> None:
        costs = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 3.0],
            ]
        )
        self.assertEqual(pareto_response_indices(costs), (0, 1))


if __name__ == "__main__":
    unittest.main()
