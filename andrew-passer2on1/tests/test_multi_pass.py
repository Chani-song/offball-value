"""Checks for passer2on1.multi_pass (2026-09-29): every pass candidate its own column, priced after the
defender's command.

  1 reduction   reaction 0: every pass column is the same against every defender command, the best legal
                one is the imported release, legality is the imported one, and the game has the imported value
  2 reactive    reaction 0.2 s: the game certifies, and it is worth at least the one-pass reactive game
                (the attack keeps that pass and gains the others)
  3 pipeline    solve_one end to end: solve, save, reload, re-certify, modal line

Run: PYTHONPATH=andrew-passer2on1:src python andrew-passer2on1/tests/test_multi_pass.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from defensive_positioning.markov import solve_markov_game
from defensive_positioning.models import GameConfig
from passer2on1.multi_markov import certificate_multi, solve_multi
from passer2on1.physics_pass import load_pass_model
from passer2on1.solve import game_from, solve_one

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/showcase_v1/figure_2v1/states_2v1_all.json"
MODEL = ROOT / "andrew/models/experimental_pass.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=3, step_seconds=0.6, physics_step=0.025)
TOL = 1e-12          # the same numbers computed in a different order: round-off, not a model threshold
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def record(code):
    return next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == code)


def main() -> None:
    model = load_pass_model(MODEL, True)
    print("1  reduction (S05)")
    rec = record("S05")
    plain = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew")
    multi = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.0, multi_pass=True)
    flat_rows = max(float(np.max(np.ptp(multi.pass_values[k], axis=-2))) for k in range(CONFIG.steps))
    check("every pass column the same against every defender command", flat_rows <= TOL, f"max spread {flat_rows:.1e}")
    same_legal = all(np.array_equal(multi.pass_legal[k].any(-1), plain.pass_index[k] >= 0) for k in range(CONFIG.steps))
    check("a legal pass exactly where the imported game has one", same_legal)
    best = max(float(np.max(np.abs(np.where(multi.pass_legal[k], multi.pass_values[k][..., 0, :], -np.inf).max(-1)
                                   .clip(min=0) * (plain.pass_index[k] >= 0) - plain.release[k])))
               for k in range(CONFIG.steps))
    check("best legal pass = imported release", best <= TOL, f"max diff {best:.1e}")
    a, b = solve_markov_game(plain), solve_multi(multi)
    check("same game value", abs(a.root_value - b.root_value) <= TOL, f"{a.root_value:.9f} vs {b.root_value:.9f}")
    gap = certificate_multi(b)["gap"]
    check("certifies", abs(gap) <= 2 * CONFIG.steps * CONFIG.solver_tolerance, f"gap {gap:.1e}")

    print("2  reactive (S34, S05)")
    for code in ("S34", "S05"):
        rec = record(code)
        one = solve_markov_game(game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2))
        many_game = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2, multi_pass=True)
        many = solve_multi(many_game)
        gap = certificate_multi(many)["gap"]
        check(f"{code}: certifies", abs(gap) <= 2 * CONFIG.steps * CONFIG.solver_tolerance, f"gap {gap:.1e}")
        check(f"{code}: worth at least the one-pass reactive game", many.root_value >= one.root_value - 1e-9,
              f"{many.root_value:.6f} vs {one.root_value:.6f}")
        root = many.attack_policy[0].reshape(-1)
        moves_n = len(CONFIG.directions) ** 2
        played = [(many_game.candidates[j][1], root[moves_n + j]) for j in np.flatnonzero(root[moves_n:] > 1e-6)]
        print(f"      {code} at 0 s: pass now {root[moves_n:].sum():.0%} "
              + ", ".join(f"{getattr(c, 'offset', '')} {c.family} {p:.0%}" for c, p in played))

    print("3  pipeline (S34)")
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("states", "policies"):
            (Path(tmp) / sub).mkdir()
        row = solve_one(record("S34"), MODEL, True, CONFIG, tmp, PHYSICS, "compass", "andrew",
                        pass_reaction_s=0.2, multi_pass=True)
        check("solve_one writes a multi-pass state", row["multi_pass"] and len(row["pass_candidates"]) == len(CONFIG.passes))
        check("reloaded policy certifies the same", abs(row["serialized_certificate"]["gap"] - row["certificate"]["gap"]) <= TOL)
        check("modal line lists the passes", "passes" in row["modal_line"][0])

    print("\nall multi-pass checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
