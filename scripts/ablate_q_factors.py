#!/usr/bin/env python3
"""Do G and A change the beneficiary at all, or is the pick already P's?

The variance decomposition says log P carries 103.5 % of the spread in log Q
on the hybrid build, with G at -4.4 % and A at 0.9 %. If that is what it looks
like, then Q = P x G x A is not doing three things -- it is doing one, and the
other two factors are decoration on this task.

That is directly testable. Rescore R9 with factors switched off: drop G, drop
A, drop both. If the answers barely move, the product is not earning its shape
and the delivery-model question was never the whole story.

Switching a factor off means replacing it with 1 in the candidate grid, which
means dividing each grid row's q by that factor. The grid rows record only q,
so the cell's own G and A are used as the scene-level stand-in -- an
approximation, and the reason this is a diagnostic rather than a rebuild.

Usage:
    python scripts/ablate_q_factors.py --label <name> --audits <dir> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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

MODES = ("전부 (P x G x A)", "G 제거 (P x A)", "A 제거 (P x G)", "P 단독", "G x A 단독")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audits", type=Path, nargs="+", required=True)
    p.add_argument("--label", type=str, default="")
    return p.parse_args()


def pick(scene, defender_id, mode):
    state = onset_state(scene)
    if defender_id not in state:
        return None
    defender = next((r for r in scene["candidate_defenders"]
                     if str(r["defender_id"]) == defender_id), None)
    if defender is None:
        return None
    response = next((r for r in defender["responses"]
                     if str(r["response_id"]) == str(defender.get("direct_best_response_id"))),
                    None)
    if response is None:
        return None
    defender_at, attacker_at = _coupled.absolute_lookups(scene, defender)
    if defender_at is None:
        return None
    runner = str(scene["runner_id"])
    curves = {}
    for oid, cell in response["cells"].items():
        if cell.get("legal") is False or cell.get("q") is None or oid == runner:
            continue
        p = max(float(cell.get("delivery") or 0.0), 1e-9)
        g = max(float(cell.get("goal") or 0.0), 1e-9)
        a = max(float(cell.get("accessibility") or 0.0), 1e-9)
        if mode == MODES[0]:
            scale = 1.0
        elif mode == MODES[1]:
            scale = 1.0 / g
        elif mode == MODES[2]:
            scale = 1.0 / a
        elif mode == MODES[3]:
            scale = 1.0 / (g * a)
        else:
            scale = 1.0 / p
        grid = [{**row, "q": float(row["q"]) * scale}
                for row in (cell.get("candidate_grid") or [])]
        curves[oid] = value_at_times(grid)
    out, _ = rule_r9(state, attacking_team_id(scene), runner, str(scene["carrier_id"]),
                     defender_id, defender_at, attacker_at, curves,
                     float(scene["horizon_seconds"]))
    return out


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

    hits = {m: 0 for m in MODES}
    agree = {m: 0 for m in MODES}
    total = 0
    for (match_id, frame_id, defender_id), answer in sorted(labels.items()):
        if frame_id == "57121":
            continue
        scene = scenes.get((match_id, frame_id))
        if scene is None:
            continue
        picks = {m: pick(scene, defender_id, m) for m in MODES}
        if any(v is None for v in picks.values()):
            continue
        total += 1
        for m in MODES:
            hits[m] += picks[m] == answer
            agree[m] += picks[m] == picks[MODES[0]]

    print("=" * 72)
    print(f"Q 인자 제거 실험   {args.label}")
    print("=" * 72)
    print(f"  n = {total}\n")
    print(f"  {'':<22}{'정답':>7}{'원본과 같은 답':>16}")
    for m in MODES:
        print(f"  {m:<20}{hits[m]:>7}{agree[m]/max(total,1):>15.1%}")
    print("\n  '원본과 같은 답' 이 100% 에 가까우면 그 인자는 순위를 안 바꾼다는 뜻.")


if __name__ == "__main__":
    main()
