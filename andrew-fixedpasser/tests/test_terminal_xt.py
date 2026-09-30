"""Checks for the 'xt' horizon of fixedpasser.game (2026-09-29): no pass at the horizon; the ball stays with
the passer, worth its possession value with the better placed receiver as support.

  1 inside      every instant before the horizon is priced exactly as with the imported 'pass' horizon
  2 horizon     no legal release there (pass_index -1, release 0); keeping it = max over the two receivers of
                payoff.possession_value_all(passer, receiver, defence), recomputed here state by state
  3 solves      plain and multi-pass games: the horizon is worth what keeping pays, the certificate holds
  4 none        'none': the same with nothing for keeping it; certifies, never worth more than 'xt'

A two-turn game (S44's first 1.2 s) so the imported horizon, priced for comparison, stays quick.

Run: PYTHONPATH=andrew-fixedpasser:src python andrew-fixedpasser/tests/test_terminal_xt.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from defensive_positioning.markov import solve_markov_game
from defensive_positioning.models import GameConfig
from fixedpasser.multi_markov import certificate_multi, solve_multi
from fixedpasser.payoff import possession_value_all
from fixedpasser.physics_pass import load_pass_model
from fixedpasser.solve import game_from

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/showcase_v1/figure_3v1/states_3v1_all.json"
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
    """The record cut to `steps` turns: passer and background tracks keep steps + 1 instants."""
    rec = copy.deepcopy(next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == code))
    for part in ("passer", "background"):
        for key in ("positions", "velocities"):
            rec[part][key] = rec[part][key][:steps + 1]
    if rec.get("release_steps") is not None:
        rec["release_steps"] = rec["release_steps"][:steps + 1]
    return rec


def main() -> None:
    rec = record("S44", CONFIG.steps)
    kw = dict(physics=PHYSICS, commands="compass", threat="andrew")
    old, new = (game_from(rec, MODEL, CONFIG, terminal=t, **kw) for t in ("pass", "xt"))
    last = CONFIG.steps

    print("1  inside")
    check("default horizon is the imported one", game_from(rec, MODEL, CONFIG, **kw).terminal == "pass")
    check("releases before the horizon unchanged",
          all(np.array_equal(old.release[k], new.release[k]) and np.array_equal(old.pass_index[k], new.pass_index[k])
              for k in range(last)))

    print("2  horizon")
    check("no legal release at the horizon", bool((new.pass_index[last] == -1).all() and (new.release[last] == 0).all()))
    check("the imported horizon had releases to remove", bool((old.pass_index[last] >= 0).any()))
    check("the imported horizon kept nothing", bool((old.retained == 0).all()))
    rng = np.random.default_rng(20260929)
    worst = 0.0
    for flat in rng.choice(new.retained.size, size=500, replace=False):
        a, b, d = np.unravel_index(flat, new.retained.shape)
        defenders = np.vstack([new.defender[last].position[d][None], new.background[last]])
        want = max(float(possession_value_all(new.passer[last], layer[last].position[i], defenders, new.scenario))
                   for layer, i in ((new.carrier, a), (new.receiver, b)))
        worst = max(worst, abs(want - float(new.retained.flat[flat])))
    check("keeping it = passer's possession value, better receiver as support", worst <= TOL, f"max diff {worst:.1e}")

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

    print("4  'none' horizon")
    none = game_from(rec, MODEL, CONFIG, terminal="none", **kw)
    check("releases before the horizon unchanged",
          all(np.array_equal(old.release[k], none.release[k]) for k in range(last)))
    check("no legal release at the horizon, keeping it worth nothing",
          bool((none.pass_index[last] == -1).all() and (none.release[last] == 0).all() and (none.retained == 0).all()))
    none_multi = game_from(rec, MODEL, CONFIG, pass_reaction_s=0.2, multi_pass=True, terminal="none", **kw)
    solved = solve_multi(none_multi)
    gap = certificate_multi(solved)["gap"]
    check("multi-pass: certificate holds", abs(gap) <= 2 * CONFIG.steps * CONFIG.solver_tolerance, f"gap {gap:.1e}")
    check("multi-pass: never above the 'xt' game (keeping pays less)", solved.root_value <= multi.root_value + TOL,
          f"{solved.root_value:.4f} vs {multi.root_value:.4f}")

    print("\nall 'xt' horizon checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
