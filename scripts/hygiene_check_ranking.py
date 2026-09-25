#!/usr/bin/env python3
"""Hygiene check: do the top-ranked triples land in Chani's strong/medium scenes?

Not a selection test. The criteria come from mechanism, and 88 labels from one
reviewer -- with a role shift between her reading and ours, and a stretch worked
at 20 seconds a scene -- cannot choose among them. What this can catch is a
criterion that behaves like noise, which would mean something is wrong.

The baseline is the honest one: if we keep the top K% of triples at random, what
share of Chani's strong/medium possessions do we still touch? A criterion has to
beat THAT, not beat zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import FPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path, default=Path("data/processed/triple_ranking.csv"))
    p.add_argument("--score", type=Path, default=Path("data/processed/chani_runner_score.csv"))
    p.add_argument("--window", type=float, default=5.0)
    p.add_argument("--criteria", nargs="+",
                   default=["second_best_q", "minimax_worst_q", "top_q"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tri = pd.read_csv(args.ranking)
    sc = pd.read_csv(args.score)
    sc = sc[sc["gap_s"].abs() <= args.window]

    # 각 삼중항이 찬의님 어느 라벨 국면에 속하는가
    tag = {}
    for _, r in sc.iterrows():
        m = (tri["match_id"] == r["match_id"]) & (tri["onset_frame_id"] >= r["phase_start"]) \
            & (tri["onset_frame_id"] <= r["phase_end"])
        for i in tri.index[m]:
            prev = tag.get(i)
            rank = {"strong": 3, "medium": 2, "low": 1}
            if prev is None or rank[r["effect"]] > rank[prev]:
                tag[i] = r["effect"]
    tri["chani"] = pd.Series(tag)
    print(f"삼중항 {len(tri)}개 중 찬의님 라벨 국면에 속한 것: {tri['chani'].notna().sum()}개")
    print(f"  등급별: {dict(tri['chani'].value_counts())}\n")

    # 목표: strong / s+m 국면을 '건드리는가' (그 국면의 삼중항이 하나라도 살아남는가)
    sc = sc.copy()
    sc["key"] = list(zip(sc["match_id"], sc["phase_start"], sc["phase_end"]))
    for _, subset, title in ((0, ("strong",), "strong"), (1, ("strong", "medium"), "strong+medium")):
        targets = sc[sc["effect"].isin(subset)]
        keys = set(targets["key"])
        if not keys:
            continue
        print("=" * 74)
        print(f"{title}: 라벨 국면 {len(keys)}개를 상위 K%가 덮는가")
        print("=" * 74)
        header = f"{'K%':>5}{'삼중항':>8}" + "".join(f"{c[:16]:>18}" for c in args.criteria) + f"{'무작위(평균)':>14}"
        print(header)
        rng = np.random.default_rng(args.seed)
        for k in (0.01, 0.05, 0.10, 0.25):
            n_keep = int(len(tri) * k)
            line = f"{k:>5.0%}{n_keep:>8}"
            for col in args.criteria:
                kept = tri.nlargest(n_keep, col)
                covered = set()
                for _, r in targets.iterrows():
                    m = (kept["match_id"] == r["match_id"]) & (kept["onset_frame_id"] >= r["phase_start"]) \
                        & (kept["onset_frame_id"] <= r["phase_end"])
                    if m.any():
                        covered.add(r["key"])
                line += f"{len(covered):>7}/{len(keys):<3}{len(covered)/len(keys):>7.0%}"
            # 무작위 기준선
            shares = []
            for _ in range(50):
                kept = tri.sample(n_keep, random_state=int(rng.integers(1 << 30)))
                covered = set()
                for _, r in targets.iterrows():
                    m = (kept["match_id"] == r["match_id"]) & (kept["onset_frame_id"] >= r["phase_start"]) \
                        & (kept["onset_frame_id"] <= r["phase_end"])
                    if m.any():
                        covered.add(r["key"])
                shares.append(len(covered) / len(keys))
            line += f"{np.mean(shares):>13.0%}"
            print(line)
        print()


if __name__ == "__main__":
    main()
