#!/usr/bin/env python3
"""How much of the argmax is P actually steering?

R9 ranks candidates by Q = P x G x A, so what decides the pick is not the level
of P but how far P spreads the candidates APART relative to how far G x A does.
Taking logs makes that additive,

    log Q = log P + log G + log A

so each factor's share of the variance in log Q across the candidates of one
response says how much it moves the ranking. Shares are covariance-based and
sum to 1 by construction, and they can go negative when a factor runs opposite
to the total.

This is the quantity the calibration experiment moved. Fitting P to observed
outcomes compresses it, its share falls, and G x A decides more of the pick --
which showed up as 19/21 dropping to 17/21.

Usage:
    python scripts/decompose_q_variance.py --label <name> --audits <dir> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--min-options", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shares = {"P": [], "G": [], "A": []}
    spreads = {"P": [], "G": [], "A": [], "Q": []}
    weights = []

    for directory in args.audits:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for scene in json.loads(path.read_text(encoding="utf-8")):
            for defender in scene["candidate_defenders"]:
                best = str(defender.get("direct_best_response_id"))
                response = next((r for r in defender["responses"]
                                 if str(r["response_id"]) == best), None)
                if response is None:
                    continue
                rows = [c for c in (response.get("cells") or {}).values()
                        if c.get("legal") is not False and c.get("q") is not None]
                if len(rows) < args.min_options:
                    continue
                p = np.log(np.maximum([float(c["delivery"]) for c in rows], FLOOR))
                g = np.log(np.maximum([float(c["goal"]) for c in rows], FLOOR))
                a = np.log(np.maximum([float(c["accessibility"]) for c in rows], FLOOR))
                q = p + g + a
                var_q = float(np.var(q))
                if var_q < 1e-12:
                    continue
                for key, v in (("P", p), ("G", g), ("A", a)):
                    shares[key].append(float(np.cov(v, q, bias=True)[0, 1] / var_q))
                    spreads[key].append(float(np.std(v)))
                spreads["Q"].append(float(np.std(q)))
                weights.append(len(rows))

    w = np.asarray(weights, dtype=float)
    print("=" * 76)
    print(f"log Q 분산 분해   {args.label}")
    print("=" * 76)
    print(f"  응답 {len(w):,}개 (옵션 {args.min_options}개 이상)\n")
    print(f"  {'':<10}{'분산 기여율':>14}{'log 표준편차 중앙':>20}")
    for key in ("P", "G", "A"):
        s = np.asarray(shares[key])
        sp = np.asarray(spreads[key])
        print(f"  {key:<10}{np.average(s, weights=w):>13.1%}{np.median(sp):>20.3f}")
    print(f"  {'(log Q)':<10}{'':>13} {np.median(spreads['Q']):>19.3f}")
    print("\n  기여율이 높을수록 그 항이 후보 순위를 더 많이 결정한다.")


if __name__ == "__main__":
    main()
