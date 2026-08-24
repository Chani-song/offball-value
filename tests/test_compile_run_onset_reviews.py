from __future__ import annotations

import unittest

import pandas as pd

from scripts.compile_run_onset_reviews import compile_reviews


class CompileRunOnsetReviewsTest(unittest.TestCase):
    def test_assigns_primary_secondary_and_excluded_cohorts(self) -> None:
        reviews = pd.DataFrame(
            [
                ["m", 1, "p1", "correct", "settled_attack", ""],
                ["m", 2, "p2", "too_early", "transition_support", ""],
                ["m", 3, "p3", "unclear", "unreviewed", "throw-in"],
            ],
            columns=[
                "match_id",
                "frame_id",
                "player_id",
                "onset_review",
                "context_review",
                "note",
            ],
        )
        candidates = pd.DataFrame(
            [["m", 1, "p1", 1.0], ["m", 2, "p2", 2.0], ["m", 3, "p3", 3.0]],
            columns=["match_id", "frame_id", "player_id", "confidence_score"],
        )

        result = compile_reviews(reviews, candidates)

        self.assertEqual(
            result["analysis_cohort"].tolist(),
            [
                "primary_settled",
                "secondary_transition_diagnostic",
                "exclude",
            ],
        )

    def test_rejects_unmatched_review_keys(self) -> None:
        reviews = pd.DataFrame(
            [["m", 1, "missing", "correct", "settled_attack", ""]],
            columns=[
                "match_id",
                "frame_id",
                "player_id",
                "onset_review",
                "context_review",
                "note",
            ],
        )
        candidates = pd.DataFrame(
            [["m", 1, "p1"]],
            columns=["match_id", "frame_id", "player_id"],
        )

        with self.assertRaises(ValueError):
            compile_reviews(reviews, candidates)


if __name__ == "__main__":
    unittest.main()
