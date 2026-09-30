"""Checks for the 'xt' horizon of passer2on1.game (2026-09-29): no pass at the horizon, only keeping the ball.

  1 inside      every instant before the horizon is priced exactly as with the imported 'pass' horizon
  2 horizon     no legal release there (pass_index -1, release 0); keeping the ball = the imported retention
  3 solves      plain and multi-pass games: the horizon is worth what keeping pays, the certificate holds

A two-turn game (S05@0.6's first 1.2 s) so the imported horizon, priced for comparison, stays quick.

Run: PYTHONPATH=andrew-passer2on1:src python andrew-passer2on1/tests/test_terminal_xt.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from defensive_positioning.markov import solve_markov_game
from defensive_positioning.models import GameConfig
from passer2on1.multi_markov import certificate_multi, solve_multi
from passer2on1.physics_pass import load_pass_model
from passer2on1.solve import game_from

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", "/scratch/bbmr/kseo1/offball-value"))
STATES = ROOT / "data/processed/showcase_v1/figure_2v1/states_2v1_all.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=2, step_seconds=0.6, physics_step=0.025)
MODEL = load_pass_model(ROOT / "andrew/models/experimental_pass.json", True)
TOL = 1e-12          # the same numbers computed in a different order: round-off, not a model threshold
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def record(code, steps):
    """The record cut to `steps` turns: its background tracks keep steps + 1 instants."""
    rec = copy.deepcopy(next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == code))
    for key in ("positions", "velocities"):
        rec["background"][key] = rec["background"][key][:steps + 1]
    return rec


def main() -> None:
    rec = record("S05@0.6", CONFIG.steps)
    kw = dict(physics=PHYSICS, commands="compass", threat="andrew", background_tackles=True)
    old, new = (game_from(rec, MODEL, CONFIG, terminal=t, **kw) for t in ("pass", "xt"))
    last = CONFIG.steps

    print("1  inside")
    check("default horizon is the imported one", game_from(rec, MODEL, CONFIG, **kw).terminal == "pass")
    check("releases before the horizon unchanged",
          all(np.array_equal(old.release[k], new.release[k]) and np.array_equal(old.pass_index[k], new.pass_index[k])
              for k in range(last)))
    check("tackle survival unchanged", all(np.array_equal(a, b) for a, b in zip(old.survival, new.survival)))

    print("2  horizon")
    check("no legal release at the horizon", bool((new.pass_index[last] == -1).all() and (new.release[last] == 0).all()))
    check("the imported horizon had releases to remove", bool((old.pass_index[last] >= 0).any()))
    check("keeping the ball = the imported retention", np.array_equal(old.retained, new.retained))

    print("3  solves")
    plain = solve_markov_game(new)
    check("plain: horizon worth what keeping pays", bool(not plain.terminal_release.any())
          and np.array_equal(plain.value[last], new.retained))
    multi_game = game_from(rec, MODEL, CONFIG, pass_reaction_s=0.2, multi_pass=True, terminal="xt", **kw)
    multi = solve_multi(multi_game)
    bounds = certificate_multi(multi)
    check("multi-pass: certificate holds", abs(bounds["gap"]) <= 2 * CONFIG.steps * CONFIG.solver_tolerance,
          f"gap {bounds['gap']:.1e}")
    check("multi-pass: horizon worth what keeping pays", np.array_equal(multi.value[last], multi_game.retained))
    print(f"      value with the imported horizon {solve_markov_game(old).root_value:.4f}, "
          f"'xt' horizon {plain.root_value:.4f}, 'xt' multi-pass {multi.root_value:.4f}")

    print("\nall 'xt' horizon checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
