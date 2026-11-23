"""Checks for multi_markov.value_split (2026-10-24): the solved value split by whom it is earned through.

A real moment of the v0.6 shot-possession solve (data/processed/moments_v06_shotposs, its first 2v1 state),
cut to two 0.6 s turns so it solves in about a minute, with the abstract's flags (multi-pass, A-sym, pass
reaction 0.2 s, background tackles, agile P99.9 physics, run passes).

  1 adds up    runner + keep = the solved value at every state: exactly at the horizon, within the LP
               tolerance inside
  2 horizon    at the horizon the runner part is the best pass where the solver releases, keep the retained
               value where it does not; no part is negative anywhere
  3 root       the split root tables add up to the game's own opening table on its legal entries
  4 pipeline   solve_one with value_split gives the result it gives without, plus the split, whose root
               parts add up to the value

Run: PYTHONPATH=andrew-passer2on1:src python andrew-passer2on1/tests/test_value_split.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from defensive_positioning.models import GameConfig
from passer2on1.multi_markov import solve_multi, split_tables, value_split
from passer2on1.physics_pass import load_pass_model
from passer2on1.run_passes import passes_named
from passer2on1.solve import game_from, solve_one

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/moments_v06_shotposs/states_2v1.json"
MODEL = ROOT / "data/processed/pass_models_sym/Asym_all.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
STEPS = 2
CONFIG = GameConfig(steps=STEPS, step_seconds=0.6, physics_step=0.025, passes=passes_named("run"))
LP_TOL = 1e-8        # the parts are p . M . q, the value is the LP's: they agree to its tolerance
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def short(record):
    """The record cut to STEPS turns: background tracks to their first STEPS + 1 instants."""
    rec = json.loads(json.dumps(record))
    for key in ("positions", "velocities", "times_requested", "times_used"):
        rec["background"][key] = rec["background"][key][:STEPS + 1]
    return rec


def main() -> None:
    model = load_pass_model(MODEL, True)
    rec = short(json.loads(STATES.read_text())["states"][0])
    game = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2, multi_pass=True,
                     background_tackles=True)
    solution = solve_multi(game)
    split, max_error = value_split(game, solution)

    print("1  adds up")
    check("parts are runner and keep", list(split) == ["runner", "keep"], str(list(split)))
    worst = [float(np.max(np.abs(split["runner"][k] + split["keep"][k] - solution.value[k])))
             for k in range(STEPS + 1)]
    check("exactly at the horizon", worst[-1] == 0.0, f"{worst[-1]:.1e}")
    check("within the LP tolerance inside", max(worst) <= LP_TOL, f"per step {[f'{w:.1e}' for w in worst]}")
    check("max_error is that worst case", max_error == max(worst))

    print("2  horizon")
    released = solution.terminal_release
    check("runner = the best pass where released",
          np.array_equal(split["runner"][-1], np.where(released, game.release[-1], 0.0)))
    check("keep = the retained value where not", np.array_equal(split["keep"][-1], np.where(released, 0.0, game.retained)))
    check("both kinds of horizon occur", bool(released.any()) and bool((~released).any()),
          f"released at {released.mean():.0%} of horizon states")
    low = min(float(np.min(v)) for part in split.values() for v in part)
    check("no part negative", low >= -1e-12, f"min {low:.1e}")

    print("3  root")
    tables = split_tables(game, 0, np.array([0]), {n: v[1] for n, v in split.items()}, solution.schedule)
    rows, cols = (np.asarray(m)[0] for m in game.masks(0, np.array([0]), solution.schedule))
    full = np.asarray(game.matrices(0, np.array([0]), solution.value[1]))[0]
    legal = rows[:, None] & cols[None, :]
    diff = float(np.max(np.abs((tables["runner"][0] + tables["keep"][0] - full)[legal])))
    check("runner + keep = the opening table", diff <= LP_TOL, f"max diff {diff:.1e}")
    moves = len(CONFIG.directions) ** 2
    check("pass columns are all the runner's", float(np.abs(tables["keep"][0][:, moves:]).max()) == 0.0)

    print("4  pipeline")
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for flag in (False, True):
            folder = Path(tmp) / str(flag)
            for sub in ("states", "policies"):
                (folder / sub).mkdir(parents=True)
            out[flag] = solve_one({**rec, "index": 0}, MODEL, True, CONFIG, str(folder), PHYSICS, "compass",
                                  "andrew", 0.2, True, True, "pass", value_split=flag)
        skip = {"timing", "worker_peak_rss_mib", "value_split"}
        same = all(out[False][k] == out[True][k] for k in out[False] if k not in skip)
        check("same result with and without the split", same and set(out[True]) - set(out[False]) == {"value_split"})
        vs = out[True]["value_split"]
        total = sum(vs["root_value"].values())
        check("root parts add up to the value", abs(total - out[True]["value"]) <= LP_TOL,
              f"{vs['root_value']} vs {out[True]['value']:.6f}")
        print(f"      root value {out[True]['value']:.6f} = runner {vs['root_value']['runner']:.6f} "
              f"+ keep {vs['root_value']['keep']:.6f} · max error {vs['max_error']:.1e}")

    print("\nall value-split checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
