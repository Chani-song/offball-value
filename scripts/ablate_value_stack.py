#!/usr/bin/env python3
"""Attribute the reordering to a factor, instead of to "the stack".

Adopting the stack moved 20 of the top 20. That number says the switch matters
but not what in it matters, and the switch is not three things -- it is four:

    adopted threat      replaces a static EPV grid lookup
    adopted completion  replaces the hybrid delivery model
    adopted tackle      replaces pass-shaped carry pricing
    accessibility DROPPED   -- not in the adopted code, a judgement made here

The fourth is the one to be suspicious of. `positional_threat` carries a
defender-room term, so keeping `goal_side_accessibility` as well would price
defender proximity twice; dropping it is defensible. But if dropping it is
what reorders the shortlist, then the reordering is this pipeline's doing and
says nothing about whose threat model is better.

Each ablation arm differs from native in exactly one factor, so its rank
correlation against native is that factor's contribution. Compared over the
scenes the arm actually built, not the full pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["match_id", "onset_frame_id", "runner_id", "defender_id"]
CRITERIA = ["minimax_worst_q", "second_best_q", "top_q", "option_spread"]
ARMS = {
    "ssac_threat": "채택 threat만",
    "ssac_pass": "채택 완료확률+태클만",
    "native_no_access": "접근성 항만 제거",
    "ssac": "전부 (실제 채택안)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path,
                   default=Path("data/processed/triple_ranking.csv"))
    p.add_argument("--arm-dir", type=Path,
                   default=Path("data/processed/ablation"))
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    args = parse_args()
    native = pd.read_csv(args.native)

    print(f"{'arm':22} {'기준':18} {'n':>6} {'순위상관':>9} {'top20 겹침':>11}")
    print("-" * 72)
    for arm, label in ARMS.items():
        path = args.arm_dir / f"triple_ranking_{arm}.csv"
        if not path.exists():
            print(f"{label:22} (없음: {path})")
            continue
        other = pd.read_csv(path)
        merged = native.merge(other, on=KEY, suffixes=("_n", "_a"))
        if merged.empty:
            print(f"{label:22} 겹치는 삼중항 없음")
            continue
        for i, column in enumerate(CRITERIA):
            a = merged[f"{column}_n"].to_numpy(dtype=float)
            b = merged[f"{column}_a"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            rho = spearman(a[ok], b[ok])
            ascending = column == "option_spread"
            oa = np.argsort(a if ascending else -a)
            ob = np.argsort(b if ascending else -b)
            k = min(args.top_n, len(merged))
            overlap = len(set(oa[:k]) & set(ob[:k]))
            name = label if i == 0 else ""
            print(f"{name:22} {column:18} {len(merged):6} {rho:9.3f} "
                  f"{overlap:>6}/{k:<4}")
        print()

    print("읽는 법: 순위상관이 native에 가까울수록 그 요인은 순서를 안 바꿉니다.")
    print("전부(ssac)가 크게 흔드는데 세 부분 arm이 전부 안 흔들면, 상호작용이")
    print("원인입니다 -- 요인 하나로는 설명이 안 된다는 뜻입니다.")


if __name__ == "__main__":
    main()
