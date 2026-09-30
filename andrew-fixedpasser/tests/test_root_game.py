"""Checks for solve.root_game (2026-09-29): the opening payoff table saved with every solved state.

  1 shape       one row per defender command, one column per attack option, named, with legality
  2 consistent  the saved equilibrium on the saved table: the attack's mix guarantees the game value against
                every legal defender command, the defender's mix holds every legal attack column to it

A two-turn multi-pass game (S44's first 1.2 s) through solve_one end to end.

Run: PYTHONPATH=andrew-fixedpasser:src python andrew-fixedpasser/tests/test_root_game.py
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from defensive_positioning.models import GameConfig
from fixedpasser.solve import solve_one

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/showcase_v1/figure_3v1/states_3v1_all.json"
MODEL = ROOT / "andrew/models/experimental_pass.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=2, step_seconds=0.6, physics_step=0.025)
TOL = 1e-9           # the LP solver's own tolerance scale (config.solver_tolerance), not a model threshold
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def record(code, steps):
    rec = copy.deepcopy(next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == code))
    for part in ("passer", "background"):
        for key in ("positions", "velocities"):
            rec[part][key] = rec[part][key][:steps + 1]
    if rec.get("release_steps") is not None:
        rec["release_steps"] = rec["release_steps"][:steps + 1]
    return rec


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("states", "policies"):
            (Path(tmp) / sub).mkdir()
        row = solve_one(record("S44", CONFIG.steps), MODEL, True, CONFIG, tmp, PHYSICS, "compass", "andrew",
                        pass_reaction_s=0.2, multi_pass=True)
    game = row["root_game"]
    m = np.array([[np.nan if x is None else x for x in r] for r in game["matrix"]])
    rows, cols = np.array(game["defender_legal"]), np.array(game["attack_legal"])
    q, p = np.array(row["root_attack"]), np.array(row["root_defender"])

    print("1  shape")
    check("5 defender rows, one column per attack option", m.shape == (5, len(game["columns"])) == (5, len(q)), str(m.shape))
    check("25 joint moves then the passes", game["columns"][:25] == [f"move {c} {r}" for c in range(5) for r in range(5)]
          and len(game["columns"]) == 25 + len(row["pass_candidates"]))
    check("legal entries are numbers", bool(np.isfinite(m[np.ix_(rows, cols)]).all()))

    print("2  consistent")
    v = row["value"]
    sub = np.nan_to_num(m[np.ix_(rows, cols)])
    attack = float((sub @ q[cols]).min())
    defence = float((p[rows] @ sub).max())
    check("no weight on an illegal option", float(q[~cols].sum()) <= TOL and float(p[~rows].sum()) <= TOL)
    check("attack's mix guarantees the value", abs(attack - v) <= TOL, f"{attack:.9f} vs {v:.9f}")
    check("defender's mix holds every column to it", abs(defence - v) <= TOL, f"{defence:.9f} vs {v:.9f}")

    print("\nall root-game checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
