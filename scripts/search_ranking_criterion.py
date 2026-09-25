#!/usr/bin/env python3
"""Is there any metric we already compute that ranks the expert's scenes up?

The shortlist is currently ordered by danger -- minimax_worst_q and friends --
and the scenes Chani marked strong sit in the bottom decile of exactly those.
Her medium scenes sit at the median. So the ordering is not weakly aligned
with her judgement; it is uncorrelated with it, and on the strong five it
points the wrong way.

Before concluding that we need a new quantity, this asks whether one of the
columns already in the ranking table separates her scenes, in either
direction. AUC is the measure: the probability that a randomly chosen triple
from one of her phases outranks a randomly chosen triple from elsewhere.

    0.50  the metric knows nothing about her judgement
    0.70  a usable signal
    0.30  a usable signal pointing the other way -- flip it

The expert set is phase membership, which is noisy: a phase holds several
triples and she was marking one moment. That noise pushes AUC toward 0.5, so
a value that survives it is real, and one near 0.5 is not proof of nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["minimax_worst_q", "second_best_q", "top_q", "option_spread",
        "vacated_xt", "vacated_mass", "vacated_mean_xt",
        "distance_to_runner_m", "n_options", "actual_minus_minimax_q"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path,
                   default=Path("data/processed/triple_ranking_r9_xt.csv"))
    p.add_argument("--runner-score", type=Path,
                   default=Path("data/processed/chani_runner_score.csv"))
    return p.parse_args()


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(a positive outranks a negative), ties counted as half."""
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over ties so a constant column scores 0.5, not 1.0
    values = np.concatenate([pos, neg])
    frame = pd.DataFrame({"v": values, "r": ranks})
    ranks = frame.groupby("v")["r"].transform("mean").to_numpy()
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    args = parse_args()
    rank = pd.read_csv(args.ranking)
    runner = pd.read_csv(args.runner_score)

    # Derived columns, each a different guess at what she might be seeing.
    with np.errstate(divide="ignore", invalid="ignore"):
        rank["spread_inv"] = -rank["option_spread"]
        rank["second_over_top"] = rank["second_best_q"] / rank["top_q"].replace(0, np.nan)
        rank["two_option_balance"] = -(rank["top_q"] - rank["second_best_q"]).abs()
        rank["regret"] = rank["actual_minus_minimax_q"]
        rank["xt_per_danger"] = rank["vacated_xt"] / rank["top_q"].replace(0, np.nan)
        rank["close_defender"] = -rank["distance_to_runner_m"]
    columns = BASE + ["spread_inv", "second_over_top", "two_option_balance",
                      "regret", "xt_per_danger", "close_defender"]

    for level in ("strong", "medium"):
        clips = runner[runner["effect"] == level]
        mask = np.zeros(len(rank), dtype=bool)
        for r in clips.itertuples():
            if pd.isna(getattr(r, "phase_start", None)):
                continue
            mask |= (
                (rank["match_id"] == r.match_id).to_numpy()
                & (rank["onset_frame_id"] >= float(r.phase_start)).to_numpy()
                & (rank["onset_frame_id"] <= float(r.phase_end)).to_numpy()
            )
        print(f"\n=== {level}: 전문가 국면 삼중항 {int(mask.sum())} / "
              f"전체 {len(rank)} ===")
        scored = []
        for column in columns:
            if column not in rank:
                continue
            values = rank[column].to_numpy(dtype=float)
            ok = np.isfinite(values)
            if ok.sum() < 10 or mask[ok].sum() < 3:
                continue
            scored.append((auc(values[ok], mask[ok]), column))
        scored.sort(key=lambda t: -abs(t[0] - 0.5))
        print(f"  {'지표':22} {'AUC':>7}   해석")
        for value, column in scored:
            if abs(value - 0.5) >= 0.10:
                note = "신호 있음" if value > 0.5 else "신호 있음 (뒤집어야)"
            elif abs(value - 0.5) >= 0.05:
                note = "약함"
            else:
                note = "무관"
            print(f"  {column:22} {value:7.3f}   {note}")


if __name__ == "__main__":
    main()
