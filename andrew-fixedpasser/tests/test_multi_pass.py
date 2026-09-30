"""Checks for fixedpasser.multi_pass (2026-09-29): every pass to each receiver its own column, priced after
the defender's command.

  1 reduction   reaction 0 (S20): every pass column the same against every defender command, legal exactly
                where the imported game has a release, the best legal one = the imported release, same value
  2 gating      release_steps (S36@0.6: the ball rolls at 0.6 and 1.2 s): no legal pass at a closed instant
  3 reactive    reaction 0.2 s (S20): certifies; solve_one end to end (save, reload, re-certify, modal line)

Run: PYTHONPATH=andrew-fixedpasser:src python andrew-fixedpasser/tests/test_multi_pass.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from defensive_positioning.markov import solve_markov_game
from defensive_positioning.models import GameConfig
from fixedpasser.multi_markov import certificate_multi, solve_multi
from fixedpasser.physics_pass import load_pass_model
from fixedpasser.solve import game_from, solve_one

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", "/scratch/bbmr/kseo1/offball-value"))
STATES = ROOT / "data/processed/showcase_v1/figure_3v1/states_3v1_all.json"
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
    print("1  reduction (S20)")
    rec = record("S20")
    plain = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew")
    multi = game_from(rec, model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.0, multi_pass=True)
    spread = max(float(np.max(np.ptp(multi.pass_values[k], axis=-2))) for k in range(CONFIG.steps))
    check("every pass column the same against every defender command", spread <= TOL, f"max spread {spread:.1e}")
    check("a legal pass exactly where the imported game has one",
          all(np.array_equal(multi.pass_legal[k].any(-1), plain.pass_index[k] >= 0) for k in range(CONFIG.steps)))
    best = max(float(np.max(np.abs(np.where(multi.pass_legal[k], multi.pass_values[k][..., 0, :], -np.inf).max(-1)
                                   .clip(min=0) * (plain.pass_index[k] >= 0) - plain.release[k])))
               for k in range(CONFIG.steps))
    check("best legal pass = imported release", best <= TOL, f"max diff {best:.1e}")
    a, b = solve_markov_game(plain), solve_multi(multi)
    check("same game value", abs(a.root_value - b.root_value) <= TOL, f"{a.root_value:.9f} vs {b.root_value:.9f}")

    print("2  gating (S36@0.6)")
    gated = game_from(record("S36@0.6"), model, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2, multi_pass=True)
    steps = gated.release_steps
    check("release_steps read", steps == (False, False, True, False), str(steps))
    check("no legal pass at a closed instant", all(not gated.pass_legal[k].any() for k in range(CONFIG.steps) if not steps[k]))
    check("legal passes at the open instant", gated.pass_legal[2].any())

    print("3  reactive (S20)")
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("states", "policies"):
            (Path(tmp) / sub).mkdir()
        row = solve_one(rec, MODEL, True, CONFIG, tmp, PHYSICS, "compass", "andrew", pass_reaction_s=0.2, multi_pass=True)
        check("certifies", abs(row["certificate"]["gap"]) <= 2 * CONFIG.steps * CONFIG.solver_tolerance,
              f"gap {row['certificate']['gap']:.1e}")
        check("reloaded policy certifies the same", abs(row["serialized_certificate"]["gap"] - row["certificate"]["gap"]) <= TOL)
        check("one column per pass to each receiver", len(row["pass_candidates"]) == 2 * len(CONFIG.passes))
        m0 = row["modal_line"][0]
        print(f"      S20 at 0 s: value {row['value']:.4f}, pass now {m0['release_probability']:.0%}, defender "
              f"{[round(x, 2) for x in m0['defender_policy']]}, passes {[(row['pass_candidates'][p['candidate']], p['probability']) for p in m0['passes']]}")

    print("\nall 3v1 multi-pass checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
