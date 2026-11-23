import unittest

from offball_value.dynamic_marking import (
    DynamicMarkingConfig,
    MarkingPathCandidate,
    interpolate_timed_point,
    marking_sample,
    moving_goal_side_target,
    prepare_path,
    rank_dynamic_marking_paths,
    summarize_dynamic_marking,
)


class DynamicMarkingTest(unittest.TestCase):
    def test_goal_side_target_is_between_actor_and_goal(self) -> None:
        self.assertEqual(
            moving_goal_side_target((2.0, 1.0), (12.0, 1.0), 2.0),
            (4.0, 1.0),
        )

    def test_wrong_side_defender_is_penalized(self) -> None:
        config = DynamicMarkingConfig(
            goal_side_offset_m=1.5,
            lookahead_seconds=0.0,
            response_delay_seconds=0.0,
            wrong_side_weight=2.0,
        )
        goal_side = marking_sample(
            0.0, (0.0, 0.0), (0.0, 0.0), (1.5, 0.0), (10.0, 0.0), config
        )
        wrong_side = marking_sample(
            0.0, (0.0, 0.0), (0.0, 0.0), (-1.5, 0.0), (10.0, 0.0), config
        )
        self.assertEqual(goal_side.wrong_side_distance_m, 0.0)
        self.assertAlmostEqual(goal_side.weighted_error_m, 0.0)
        self.assertGreater(wrong_side.wrong_side_distance_m, 0.0)
        self.assertGreater(wrong_side.weighted_error_m, goal_side.weighted_error_m)

    def test_lookahead_uses_future_attacker_position(self) -> None:
        config = DynamicMarkingConfig(
            goal_side_offset_m=1.0,
            lookahead_seconds=1.0,
            response_delay_seconds=0.0,
        )
        summary = summarize_dynamic_marking(
            ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 4.0, 0.0)),
            ((0.0, 3.0, 0.0), (1.0, 5.0, 0.0), (2.0, 5.0, 0.0)),
            (10.0, 0.0),
            config,
            (0.0, 1.0, 2.0),
        )
        self.assertAlmostEqual(summary.samples[0].anticipated_actor_x, 2.0)
        self.assertAlmostEqual(summary.samples[0].target_x, 3.0)
        self.assertAlmostEqual(summary.samples[0].weighted_error_m, 0.0)

    def test_dynamic_interceptor_beats_path_to_old_position(self) -> None:
        actor = ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 4.0, 0.0))
        old_position = MarkingPathCandidate(
            "old",
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        )
        interceptor = MarkingPathCandidate(
            "intercept",
            ((0.0, 1.0, 0.0), (1.0, 3.0, 0.0), (2.0, 5.0, 0.0)),
        )
        ranked = rank_dynamic_marking_paths(
            actor,
            (old_position, interceptor),
            (10.0, 0.0),
            DynamicMarkingConfig(
                goal_side_offset_m=1.0,
                lookahead_seconds=0.0,
                response_delay_seconds=0.0,
            ),
            (0.0, 1.0, 2.0),
        )
        self.assertEqual(ranked[0].candidate.identifier, "intercept")
        self.assertLess(
            ranked[0].summary.mean_weighted_error_m,
            ranked[1].summary.mean_weighted_error_m,
        )


class PreparedPathTest(unittest.TestCase):
    # unsorted, with a repeated time whose later sample must win
    RAW = ((0.4, 4.0, -1.0), (0.0, 0.0, 0.0), (0.2, 1.0, 2.0), (0.2, 3.0, 5.0), (1.0, 6.0, 1.5))

    def test_interpolates_exactly_like_the_raw_path(self) -> None:
        prepared = prepare_path(self.RAW)
        for time_s in (-1.0, 0.0, 0.05, 0.2, 0.31, 0.4, 0.99, 1.0, 3.0):
            self.assertEqual(
                interpolate_timed_point(prepared, time_s),
                interpolate_timed_point(self.RAW, time_s),
            )

    def test_holds_the_deduplicated_sorted_samples(self) -> None:
        prepared = prepare_path(self.RAW)
        self.assertEqual(
            tuple(prepared),
            ((0.0, 0.0, 0.0), (0.2, 3.0, 5.0), (0.4, 4.0, -1.0), (1.0, 6.0, 1.5)),
        )
        self.assertEqual(prepared.times.tolist(), [0.0, 0.2, 0.4, 1.0])
        self.assertIs(prepare_path(prepared), prepared)

    def test_rejects_non_finite_samples_like_interpolation_does(self) -> None:
        bad = ((0.0, 0.0, 0.0), (0.2, float("nan"), 1.0))
        with self.assertRaises(ValueError):
            interpolate_timed_point(bad, 0.1)
        with self.assertRaises(ValueError):
            prepare_path(bad)


if __name__ == "__main__":
    unittest.main()
