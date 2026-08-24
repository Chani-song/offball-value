#!/usr/bin/env python3
"""Validate run-onset annotations and compile analysis cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = ["match_id", "frame_id", "player_id"]
REQUIRED_REVIEW_COLUMNS = KEY + ["onset_review", "context_review", "note"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews_csv", type=Path)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path(
            "data/processed/run_onset_v0_2/combined/run_onset_candidates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/run_onset_v0_3/combined/qc_round2"),
    )
    parser.add_argument(
        "--legacy-reviews-csv",
        type=Path,
        help="Optional v0.1 CSV whose 'correct' rows join the cumulative cohort.",
    )
    parser.add_argument(
        "--eligibility-candidates-csv",
        type=Path,
        default=Path(
            "data/processed/run_onset_v0_3/combined/run_onset_candidates.csv"
        ),
        help="Final detector candidates used to recheck cumulative eligibility.",
    )
    return parser.parse_args()


def assign_cohort(row: pd.Series) -> str:
    if (
        row["onset_review"] == "correct"
        and row["context_review"] == "settled_attack"
    ):
        return "primary_settled"
    if row["context_review"] in {"transition_support", "contested_possession"}:
        return "secondary_transition_diagnostic"
    return "exclude"


def compile_reviews(reviews: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_REVIEW_COLUMNS) - set(reviews.columns))
    if missing:
        raise ValueError(f"review CSV is missing columns: {missing}")
    if reviews.duplicated(KEY).any():
        raise ValueError("review CSV contains duplicate match/frame/player keys")
    if reviews["onset_review"].eq("unreviewed").any():
        raise ValueError("onset_review contains unreviewed rows")
    merged = reviews.merge(
        candidates,
        on=KEY,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "review rows missing from candidate set: "
            + str(unmatched[KEY].to_dict("records"))
        )
    merged = merged.drop(columns="_merge")
    merged.insert(6, "analysis_cohort", merged.apply(assign_cohort, axis=1))
    return merged


def main() -> None:
    args = parse_args()
    reviews = pd.read_csv(
        args.reviews_csv,
        dtype={"match_id": str, "player_id": str},
        keep_default_na=False,
    )
    candidates = pd.read_csv(
        args.candidates_csv,
        dtype={"match_id": str, "player_id": str},
    )
    merged = compile_reviews(reviews, candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all": merged,
        "primary_settled": merged[
            merged["analysis_cohort"] == "primary_settled"
        ],
        "secondary_transition_diagnostic": merged[
            merged["analysis_cohort"] == "secondary_transition_diagnostic"
        ],
        "excluded": merged[merged["analysis_cohort"] == "exclude"],
    }
    for name, frame in outputs.items():
        path = args.output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name}: {len(frame)} -> {path}")
    if args.legacy_reviews_csv is not None:
        legacy = pd.read_csv(
            args.legacy_reviews_csv,
            dtype={"match_id": str, "player_id": str},
            keep_default_na=False,
        )
        legacy_primary = legacy[legacy["review"] == "correct"][KEY].assign(
            qc_round="round1",
            source_label="correct_and_meaningful",
        )
        round2_primary = outputs["primary_settled"][KEY].assign(
            qc_round="round2",
            source_label="correct_settled_attack",
        )
        cumulative_keys = pd.concat(
            [legacy_primary, round2_primary],
            ignore_index=True,
        ).drop_duplicates(KEY)
        eligibility = pd.read_csv(
            args.eligibility_candidates_csv,
            dtype={"match_id": str, "player_id": str},
        )
        cumulative = cumulative_keys.merge(
            eligibility,
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
        path = args.output_dir / "cumulative_primary_settled.csv"
        cumulative.to_csv(path, index=False)
        print(f"cumulative_primary_settled: {len(cumulative)} -> {path}")


if __name__ == "__main__":
    main()
