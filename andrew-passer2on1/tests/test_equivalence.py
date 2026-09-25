#!/usr/bin/env python3
"""With no background defenders, this package must BE the imported game.

Four levels, each bit for bit (np.array_equal, not allclose):

  1  threat      positional_threat_all with one defender == value.positional_threat
  2  payoff      release_payoffs_with_background, empty background
                 == payoff.release_payoffs, on random kick states, real pass model
  3  game        Background2on1Game(empty) and markov.FiniteGame built from the
                 same real scene: release, pass_index, retained, survival, layers
  4  solution    both solved; root value equal to each other AND to the value the
                 original solver stored for that scene in stage3_carrier_3s

Level 4 is the end-to-end check: whatever the new code does, removing the
background gives back a number the untouched original already produced.

Run:  PYTHONPATH=andrew-passer2on1 .venv-delta/bin/python andrew-passer2on1/tests/test_equivalence.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.expected_pass import ExpectedPass
from defensive_positioning.markov import FiniteGame, certificate, solve_markov_game
from defensive_positioning.models import GameConfig
from defensive_positioning.payoff import release_payoffs, retention_payoff
from defensive_positioning.value import positional_threat

from passer2on1.game import Background2on1Game
from passer2on1.payoff import (positional_threat_all, release_payoffs_with_background,
                               retention_payoff_with_background)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "andrew/models/experimental_pass.json"
STATES = ROOT / "data/processed/stage3/carrier_beneficiary_states_v2.json"
SOLVED = Path("/work/hdd/bbmr/kseo1/offball-out/stage3_carrier_3s")
CONFIG = GameConfig(steps=3, step_seconds=1.0, physics_step=0.025)


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    rng = np.random.default_rng(20260923)
    model = ExpectedPass.load(MODEL, allow_proxy=True)
    record = json.loads(STATES.read_text())["states"][0]
    scenario = scenario_from(record["scenario"])

    print("1  threat")
    n = 5000
    ball = rng.uniform([0, 0], [105, 68], (n, 2))
    support = rng.uniform([0, 0], [105, 68], (n, 2))
    defender = rng.uniform([0, 0], [105, 68], (n, 2))
    a = positional_threat(ball, support, defender, scenario)
    b = positional_threat_all(ball, support, defender[:, None, :], scenario)
    check("one defender reproduces value.positional_threat", np.array_equal(a, b),
          f"max diff {np.abs(a - b).max():.1e}")

    print("2  payoff")
    carrier = rng.uniform([30, 5], [80, 63], (n, 2))
    receiver = rng.uniform([30, 5], [100, 63], (n, 2))
    cv, rv, dv = (rng.normal(0, 2, (n, 2)) for _ in range(3))
    empty = np.zeros((0, 2))
    a, la = release_payoffs(scenario, CONFIG, model, carrier, receiver, defender,
                            cv, rv, dv, return_legal=True)
    b, lb = release_payoffs_with_background(scenario, CONFIG, model, carrier, receiver, defender,
                                            cv, rv, dv, empty, empty, return_legal=True)
    check("release values, empty background", np.array_equal(a, b),
          f"max diff {np.abs(a - b).max():.1e}")
    check("release legality, empty background", np.array_equal(la, lb))
    a = retention_payoff(scenario, carrier, receiver, defender)
    b = retention_payoff_with_background(scenario, carrier, receiver, defender, empty)
    check("retention, empty background", np.array_equal(a, b))

    print("3  game  (one real scene, 3 turns x 1.0 s)")
    t = time.perf_counter()
    original = FiniteGame(scenario, model, CONFIG)
    ours = Background2on1Game(scenario, model, CONFIG,
                              background=np.zeros((CONFIG.steps + 1, 0, 2)))
    print(f"      built both in {time.perf_counter() - t:.0f}s")
    check("shapes", original.shapes == ours.shapes, str(ours.shapes))
    for role in ("carrier", "receiver", "defender"):
        same = all(np.array_equal(x.position, y.position) and np.array_equal(x.velocity, y.velocity)
                   for x, y in zip(getattr(original, role), getattr(ours, role)))
        check(f"{role} layers", same)
    check("survival", all(np.array_equal(x, y) for x, y in zip(original.survival, ours.survival)))
    check("release", all(np.array_equal(x, y) for x, y in zip(original.release, ours.release)))
    check("pass_index", all(np.array_equal(x, y) for x, y in zip(original.pass_index, ours.pass_index)))
    check("retained", np.array_equal(original.retained, ours.retained))

    print("4  solution")
    so, sb = solve_markov_game(original), solve_markov_game(ours)
    check("root value equal", so.root_value == sb.root_value, f"{sb.root_value!r}")
    cb = certificate(sb)
    check("certificate on ours", cb["gap"] <= 2 * CONFIG.steps * CONFIG.solver_tolerance,
          f"gap {cb['gap']:.1e}")
    stored = next(json.loads(line) for line in (SOLVED / "rows.jsonl").read_text().splitlines()
                  if json.loads(line)["index"] == record["index"])
    check("equals the original solver's stored stage3_carrier_3s value",
          stored["value"] == sb.root_value, f"stored {stored['value']!r}")
    print("\nall equivalence checks passed")


if __name__ == "__main__":
    main()
