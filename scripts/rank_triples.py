#!/usr/bin/env python3
"""Rank the (runner, defender, beneficiary) triples so only the top ones need looking at.

The pipeline emits ~1,900 triples over 7 matches -- one every 20 seconds of
football -- because stage 0 keeps every candidate and nothing ranks them. Chani
marks about one per possession. A threshold is needed, and it has to come from
the mechanism rather than from whatever correlates with the labels.

Four candidates, each with a reason to exist:

  minimax_worst_q   what the attack is worth when the defender plays his best
                    reply. High means no defensive choice holds it: the moment
                    is dangerous however he moves.
  second_best_q     the value of the attack's SECOND option under that reply.
                    A dilemma needs two options that both hurt; if the second is
                    worthless the defender simply covers the first and there is
                    no dilemma, however large the first is.
  option_spread     top q minus second q. The mirror of the above: SMALL spread
                    is the dilemma, large spread is an easy defensive read.
  regret            minimax_worst_q minus the best the defender could do knowing
                    the target. What uncertainty costs him -- the dilemma's
                    price, in the payoff's own units.

This reports each one's distribution, how many triples survive at various
thresholds, and how the surviving set overlaps Chani's strong / strong+medium.
It does NOT pick a winner: Chani's 88 labels are one reviewer's, the roles shift
between her reading and ours, and stage 3 (dilemma) belongs to Junhyun. Ranking
scenes for review is not the same as defining the dilemma.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path,
                   default=Path("/work/hdd/bbmr/kseo1/offball-out/v5_hybrid"))
    p.add_argument("--output", type=Path, default=Path("data/processed/triple_ranking.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    paths = sorted(args.build.glob("scene_*/local_game_payoff_audits.json"))
    print(f"빌드 파일 {len(paths)}개 읽는 중...", flush=True)
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for game in payload:
            runner = str(game["runner_id"])
            for d in game.get("candidate_defenders") or []:
                direct = next(
                    (r for r in d["responses"]
                     if str(r["response_id"]) == str(d.get("direct_best_response_id"))),
                    None,
                )
                if direct is None:
                    continue
                qs = sorted(
                    (float(c["q"]) for oid, c in direct["cells"].items()
                     if c.get("legal") is not False and c.get("q") is not None
                     and oid != runner),
                    reverse=True,
                )
                if not qs:
                    continue
                top1 = qs[0]
                top2 = qs[1] if len(qs) > 1 else 0.0
                mm = d.get("minimax_worst_q")
                rows.append({
                    "match_id": game["match_id"],
                    "onset_frame_id": game["onset_frame_id"],
                    "runner_id": runner,
                    "defender_id": str(d["defender_id"]),
                    "n_options": len(qs),
                    "top_q": top1,
                    "second_best_q": top2,
                    "option_spread": top1 - top2,
                    "minimax_worst_q": float(mm) if mm is not None else np.nan,
                    "actual_minus_minimax_q": d.get("actual_minus_minimax_q"),
                    "distance_to_runner_m": d.get("current_distance_to_runner_m"),
                })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\n삼중항 {len(frame)}개\n")

    print("=" * 74)
    print("각 기준의 분포")
    print("=" * 74)
    cols = ["minimax_worst_q", "second_best_q", "option_spread", "top_q"]
    print(frame[cols].describe(percentiles=[.5, .75, .9, .95, .99]).round(4).to_string())

    print("\n" + "=" * 74)
    print("임계값별 생존 수 (전체 대비)")
    print("=" * 74)
    n = len(frame)
    for col in ("minimax_worst_q", "second_best_q", "top_q"):
        print(f"\n{col} 이상:")
        for q in (0.50, 0.75, 0.90, 0.95, 0.99):
            thr = frame[col].quantile(q)
            k = int((frame[col] >= thr).sum())
            print(f"  상위 {1 - q:>5.0%} (≥{thr:.4f}) : {k:>5}개  경기당 {k / 7:>5.1f}개")
    print(f"\noption_spread 이하 (작을수록 딜레마):")
    for q in (0.50, 0.25, 0.10, 0.05, 0.01):
        thr = frame["option_spread"].quantile(q)
        k = int((frame["option_spread"] <= thr).sum())
        print(f"  하위 {q:>5.0%} (≤{thr:.4f}) : {k:>5}개  경기당 {k / 7:>5.1f}개")

    print("\n" + "=" * 74)
    print("기준들이 서로 다른 것을 고르는가 (상관)")
    print("=" * 74)
    print(frame[cols].corr().round(3).to_string())


if __name__ == "__main__":
    main()
