#!/usr/bin/env python3
"""The same AUC, but only on triples whose runner is the one she named.

Phase membership was too coarse to conclude from: a phase holds several
triples and she marked one moment, so the rest enter the positive class as
noise and drag every AUC toward 0.5. That is a sufficient explanation for
the medium set reading 0.47-0.54 on all sixteen columns, and it has to be
ruled out before saying the criteria are uninformative.

Here the positive class is triples where runner_id is one of the players she
named for that clip, resolved shirt number -> player_id through the match
metadata the same way score_against_chani2.py does it. Far fewer positives,
but they are the moments she was actually looking at.

Small n is the cost. The counts are printed with the AUCs; below about ten
positives an AUC is a description of those ten, not evidence about the rule.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offball_value.bundesliga import (  # noqa: E402
    find_bundesliga_files,
    load_bundesliga_match_metadata,
)

BASE = ["minimax_worst_q", "second_best_q", "top_q", "option_spread",
        "vacated_xt", "vacated_mass", "vacated_mean_xt",
        "distance_to_runner_m", "n_options", "actual_minus_minimax_q"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path,
                   default=Path("data/processed/triple_ranking_r9_xt.csv"))
    p.add_argument("--join", type=Path,
                   default=Path("data/processed/chani_annotation_join_v0_4.csv"))
    p.add_argument("--runner-score", type=Path,
                   default=Path("data/processed/chani_runner_score.csv"))
    p.add_argument("--raw-dir", type=Path,
                   default=Path("data/raw/bundesliga-integrated"))
    return p.parse_args()


def shirts(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [s for s in re.findall(r"\d+", value)]


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="stable")
    raw = np.empty(order.size, dtype=float)
    raw[order] = np.arange(1, order.size + 1)
    ranks = pd.DataFrame({"v": values, "r": raw}).groupby("v")["r"].transform("mean")
    ranks = ranks.to_numpy()
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main() -> None:
    args = parse_args()
    rank = pd.read_csv(args.ranking)
    join = pd.read_csv(args.join)
    runner = pd.read_csv(args.runner_score)
    merged = runner.merge(join[["clip_id", "offball_attackers", "shot_frame_id"]],
                          on="clip_id", how="left")

    meta_cache: dict[str, object] = {}

    def meta(match_id: str):
        if match_id not in meta_cache:
            files = find_bundesliga_files(args.raw_dir, match_id)
            meta_cache[match_id] = load_bundesliga_match_metadata(files["matchinfo"])
        return meta_cache[match_id]

    with np.errstate(divide="ignore", invalid="ignore"):
        rank["spread_inv"] = -rank["option_spread"]
        rank["second_over_top"] = rank["second_best_q"] / rank["top_q"].replace(0, np.nan)
        rank["regret"] = rank["actual_minus_minimax_q"]
        rank["close_defender"] = -rank["distance_to_runner_m"]
    columns = BASE + ["spread_inv", "second_over_top", "regret", "close_defender"]

    for level in ("strong", "medium", "low"):
        rows = merged[merged["effect"] == level]
        mask = np.zeros(len(rank), dtype=bool)
        clips_used = 0
        for r in rows.itertuples():
            if pd.isna(getattr(r, "phase_start", None)):
                continue
            try:
                m = meta(r.match_id)
            except Exception:
                continue
            by_shirt = {(p.team_id, str(p.shirt_number)): p.player_id
                        for p in m.players.values()}
            wanted = {pid for (_team, _s), pid in by_shirt.items()}
            named = {by_shirt.get((t, s))
                     for s in shirts(getattr(r, "offball_attackers", ""))
                     for t in {k[0] for k in by_shirt}}
            named = {p for p in named if p} & wanted
            if not named:
                continue
            hit = (
                (rank["match_id"] == r.match_id).to_numpy()
                & (rank["onset_frame_id"] >= float(r.phase_start)).to_numpy()
                & (rank["onset_frame_id"] <= float(r.phase_end)).to_numpy()
                & rank["runner_id"].isin(named).to_numpy()
            )
            if hit.any():
                clips_used += 1
            mask |= hit

        n = int(mask.sum())
        print(f"\n=== {level}: 러너가 일치하는 삼중항 {n}개 "
              f"(클립 {clips_used}/{len(rows)}) ===")
        if n < 3:
            print("  표본이 너무 작아 계산하지 않습니다.")
            continue
        if n < 10:
            print("  주의: 10개 미만이라 AUC는 이 표본의 서술일 뿐입니다.")
        scored = []
        for column in columns:
            values = rank[column].to_numpy(dtype=float)
            ok = np.isfinite(values)
            if mask[ok].sum() < 3:
                continue
            scored.append((auc(values[ok], mask[ok]), column))
        scored.sort(key=lambda t: -abs(t[0] - 0.5))
        for value, column in scored[:8]:
            if abs(value - 0.5) >= 0.15:
                note = "신호" if value > 0.5 else "신호 (뒤집어야)"
            elif abs(value - 0.5) >= 0.08:
                note = "약함"
            else:
                note = "무관"
            print(f"  {column:22} {value:7.3f}   {note}")


if __name__ == "__main__":
    main()
