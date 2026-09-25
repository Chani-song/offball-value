#!/usr/bin/env python3
"""Is the delivery model wrong, or is its exponent?

Q = P x G x A gives every factor an exponent of 1, which nobody chose on
evidence. Today's result puts pressure on it: the raw hybrid spreads log P by
0.578 across candidates and scores 19/21, while the same model calibrated to
real outcomes spreads it by 0.215 and scores 17/21. The calibrated P is the
better probability and the worse input.

If that is a scaling problem rather than a modelling one, then raising the
CALIBRATED P to a power should restore the spread -- and the answers with it.
The ratio of the two spreads, 0.578 / 0.215 = 2.69, predicts where.

Recovering 19 near that alpha would say the delivery model was never the
thing to fix: P is doing duty as a weight, and the product's shape is what
decides how much weight it gets.

Alpha multiplies each grid row's q by P^(alpha-1), with the cell's own P as
the option's stand-in, since grid rows record only q. Approximate, hence a
diagnostic. And alpha is NOT to be chosen by which value scores best here --
that would be fitting the evaluation set. The question is only whether the
curve peaks where the spread ratio says it should.

Usage:
    python scripts/sweep_delivery_exponent.py --label <name> --audits <dir> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.assignment_rule import attacking_team_id, onset_state  # noqa: E402
from offball_value.coupled_beneficiary import rule_r9, value_at_times  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_coupled_beneficiary", ROOT / "scripts" / "score_coupled_beneficiary.py"
)
_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coupled)
_scorer = _coupled._scorer

ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.69, 3.0, 3.5, 4.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--label", type=str, default="")
    return p.parse_args()


def pick(scene, defender_id, alpha):
    state = onset_state(scene)
    if defender_id not in state:
        return None, None
    defender = next((r for r in scene["candidate_defenders"]
                     if str(r["defender_id"]) == defender_id), None)
    if defender is None:
        return None, None
    response = next((r for r in defender["responses"]
                     if str(r["response_id"]) == str(defender.get("direct_best_response_id"))),
                    None)
    if response is None:
        return None, None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None, None
    runner = str(scene["runner_id"])
    curves, spread = {}, []
    for oid, cell in response["cells"].items():
        if cell.get("legal") is False or cell.get("q") is None or oid == runner:
            continue
        p = max(float(cell.get("delivery") or 0.0), 1e-9)
        spread.append(np.log(p))
        scale = p ** (alpha - 1.0)
        grid = [{**row, "q": float(row["q"]) * scale}
                for row in (cell.get("candidate_grid") or [])]
        curves[oid] = value_at_times(grid)
    out, _ = rule_r9(state, attacking_team_id(scene), runner, str(scene["carrier_id"]),
                     defender_id, defender_at, attacker_at, curves,
                     float(scene["horizon_seconds"]))
    return out, (float(np.std(spread)) if len(spread) > 1 else None)


def main() -> None:
    args = parse_args()
    labels = _scorer.load_labels()
    scenes = {}
    for directory in args.audits:
        path = directory / "local_game_payoff_audits.json"
        if not path.exists():
            continue
        for scene in json.loads(path.read_text(encoding="utf-8")):
            scenes[(str(scene["match_id"]), str(scene["onset_frame_id"]))] = scene

    pairs = []
    for key, answer in sorted(labels.items()):
        match_id, frame_id, defender_id = key
        if frame_id == "57121":
            continue
        scene = scenes.get((match_id, frame_id))
        if scene is not None:
            pairs.append((scene, defender_id, answer))

    print("=" * 68)
    print(f"배달 지수 스윕   {args.label}")
    print("=" * 68)
    base_spread = [s for _, s in (pick(sc, d, 1.0) for sc, d, _ in pairs) if s]
    print(f"  n = {len(pairs)} · alpha=1 에서 장면 내 log P 표준편차 중앙 "
          f"{np.median(base_spread):.3f}\n")
    print(f"  {'alpha':>7}{'정답':>8}{'유효 log P 표준편차':>22}")
    for alpha in ALPHAS:
        hits = 0
        used = 0
        for scene, defender_id, answer in pairs:
            got, _ = pick(scene, defender_id, alpha)
            if got is None:
                continue
            used += 1
            hits += got == answer
        print(f"  {alpha:>7.2f}{hits:>8}{np.median(base_spread)*alpha:>22.3f}")
    print("\n  참고: 하이브리드 raw 의 log P 표준편차는 0.578, 19/21")


if __name__ == "__main__":
    main()
