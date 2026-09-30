"""Checks for passer2on1.reactive_pass (2026-09-29): a pass priced after the defender's command.

  1 reduction   reaction 0: the pass column equals the imported release at every state and command
  2 same game   reaction 0 on a real record: same value and same policies as Background2on1Game
  3 reactive    reaction 0.2 s: the pass column differs by defender command somewhere, the game still
                certifies, and which pass is chosen and whether it is legal are unchanged

Run: PYTHONPATH=andrew-passer2on1:src python andrew-passer2on1/tests/test_reactive_pass.py
(OFFBALL_DATA_ROOT, default the working tree, holds data/processed and andrew/models.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from defensive_positioning.markov import certificate, solve_markov_game
from defensive_positioning.models import GameConfig
from passer2on1.physics_pass import load_pass_model
from passer2on1.solve import game_from

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/showcase_v1/figure_2v1/states_2v1_all.json"
MODEL = ROOT / "andrew/models/experimental_pass.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=3, step_seconds=0.6, physics_step=0.025)
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def record(code):
    return next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == code)


def main() -> None:
    model = load_pass_model(MODEL, True)
    rec = record("S05")
    print("1  reduction")
    plain = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew")
    zero = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.0)
    worst = max(float(np.max(np.abs(zero.release_by_command[k] - plain.release[k][..., None])))
                for k in range(CONFIG.steps))
    # the same numbers computed in a different order: equal to round-off, not bit for bit
    check("reaction 0: pass column = imported release, every state and command", worst <= 1e-12, f"max diff {worst:.1e}")

    print("2  same game")
    a, b = solve_markov_game(plain), solve_markov_game(zero)
    check("same root value", a.root_value == b.root_value, f"{a.root_value:.9f} vs {b.root_value:.9f}")
    same = all(np.array_equal(x, y) for k in range(CONFIG.steps)
               for x, y in ((a.attack_policy[k], b.attack_policy[k]), (a.defender_policy[k], b.defender_policy[k])))
    check("same policies at every instant", same)

    print("3  reactive")
    rec = record("S34")
    base = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew")
    react = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2)
    spread = max(float(np.max(np.ptp(react.release_by_command[k], axis=-1))) for k in range(CONFIG.steps))
    check("pass column differs by defender command somewhere", spread > 0, f"largest spread {spread:.4f}")
    check("chosen pass and its legality unchanged", all(np.array_equal(base.pass_index[k], react.pass_index[k])
                                                        for k in range(CONFIG.steps + 1)))
    check("horizon release unchanged", np.array_equal(base.release[-1], react.release[-1]))
    sol = solve_markov_game(react)
    gap = certificate(sol)["gap"]
    check("certifies", abs(gap) <= 2 * CONFIG.steps * CONFIG.solver_tolerance, f"gap {gap:.1e}")
    root = sol.attack_policy[0].reshape(-1)
    print(f"      S34 at 0 s: pass now {root[-1]:.0%} (imported pricing: 100%), value {sol.root_value:.4f}")

    print("\nall reactive-pass checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
