#!/usr/bin/env python3
"""Does adopting stage 3's value stack change which scenes we would look at?

Two builds of the same 588 scenes differing only in the value stack. The
question is not which Q is larger -- the scales are different by construction
and comparing their levels means nothing. It is whether the ORDER changes,
because the order is what selects the scenes a human reviews and what stage 3
is then handed.

Reported per criterion: rank correlation over the triples both builds share,
and the overlap of their top 20 and top 5 %. A high correlation means the
adoption is cosmetic for our purposes and the argument for it is only that
stage 2 and stage 3 now speak one language. A low one means the shortlist we
have been reviewing was an artefact of the old scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["match_id", "onset_frame_id", "runner_id", "defender_id"]
CRITERIA = ["minimax_worst_q", "second_best_q", "top_q", "option_spread"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path,
                   default=Path("data/processed/triple_ranking.csv"))
    p.add_argument("--ssac", type=Path,
                   default=Path("data/processed/triple_ranking_ssac.csv"))
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    args = parse_args()
    native = pd.read_csv(args.native)
    ssac = pd.read_csv(args.ssac)
    print(f"native {len(native)} 삼중항 · ssac {len(ssac)} 삼중항")

    merged = native.merge(ssac, on=KEY, suffixes=("_native", "_ssac"))
    print(f"공통 삼중항 {len(merged)}개\n")
    if merged.empty:
        print("겹치는 삼중항이 없습니다 — 두 빌드가 다른 장면을 만들었습니다.")
        return

    print(f"{'기준':18} {'순위상관':>8} {'top20 겹침':>11} {'top5% 겹침':>11}")
    for column in CRITERIA:
        a = merged[f"{column}_native"].to_numpy(dtype=float)
        b = merged[f"{column}_ssac"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        rho = spearman(a[ok], b[ok])
        ascending = column == "option_spread"   # small spread is the dilemma
        order_a = np.argsort(a if ascending else -a)
        order_b = np.argsort(b if ascending else -b)
        n5 = max(1, int(round(0.05 * len(merged))))
        top20 = len(set(order_a[:args.top_n]) & set(order_b[:args.top_n]))
        top5 = len(set(order_a[:n5]) & set(order_b[:n5]))
        print(f"{column:18} {rho:8.3f} {top20:>6}/{args.top_n:<4} "
              f"{top5:>6}/{n5:<4}")

    print("\n수준 비교는 의미가 없지만, 척도가 얼마나 다른지는 참고가 됩니다:")
    for column in CRITERIA:
        a = merged[f"{column}_native"]
        b = merged[f"{column}_ssac"]
        print(f"  {column:18} native 중앙 {a.median():.4f} · ssac 중앙 {b.median():.4f}")


if __name__ == "__main__":
    main()
