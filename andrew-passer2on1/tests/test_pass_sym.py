#!/usr/bin/env python3
"""Does candidate A-sym hold the receiver and the defenders to one rule, and change nothing else?

  1  window      reach_window on its own cases: a standing player can wait
                 (latest +inf) and his earliest is reach_time's; one moving away
                 cannot be there before he comes back (latest -inf, back =
                 earliest = reach_time); one approaching slowly can stop and wait;
                 one approaching fast and close passes it and can't stop
                 (earliest <= latest < back)
  2  nesting     when every player can stop before every point, A-sym's margins
                 are A's exactly
  3  receiver    the S15 situation (runner at 9 m/s, target 4 m ahead, ball 1.1 s
                 away): A lets him pass the point at 0.44 s and wait; A-sym makes
                 him come back, so he takes the ball later and the margin drops.
                 A lead pass he meets in stride keeps A's margin.
  4  defender    the same rule for a defender: one who would run through the
                 target before the ball and can't stop no longer counts as there
                 at that time; one who can stop on it still does
  5  loading     an A-sym JSON loads as PhysicsRaceSymPass through load_pass_model

Run:  PYTHONPATH=andrew-passer2on1 python andrew-passer2on1/tests/test_pass_sym.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from passer2on1.physics_pass import (KIND, KIND_SYM, PhysicsRacePass, PhysicsRaceSymPass,
                                     load_pass_model, reach_time, reach_window)

PLAYER = {"reaction_s": 0.2, "speed_up": 4.5, "braking": 6.0, "plant_min_speed": 2.0,
          "plant_braking": 9.0, "plant_seconds": 0.1, "max_speed": 9.0}
BALL = {"intercept": 12.106, "slope": 0.217, "min": 7.313, "max": 25.302}


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def spec(kind):
    return {"kind": kind, "coef": [1.0, 1.0, 1.0], "ball_speed": BALL, "player": PLAYER}


def main() -> None:
    x = np.array([10.0, 0.0])
    print("1  window")
    e, l, b = reach_window([0.0, 0.0], [0.0, 0.0], x, PLAYER)
    check("standing: can wait, earliest = reach_time", np.isinf(l) and l > 0
          and np.isclose(e, reach_time([0.0, 0.0], [0.0, 0.0], x, PLAYER)), f"earliest {e:.2f}s")
    e, l, b = reach_window([5.0, 0.0], [-6.0, 0.0], x, PLAYER)
    check("moving away: only by coming back", np.isinf(l) and l < 0 and np.isclose(b, e)
          and np.isclose(e, reach_time([5.0, 0.0], [-6.0, 0.0], x, PLAYER)), f"back {b:.2f}s")
    e, l, b = reach_window([0.0, 0.0], [3.0, 0.0], x, PLAYER)
    check("approaching slowly: can stop and wait", np.isinf(l) and l > 0)
    e, l, b = reach_window([6.0, 0.0], [9.0, 0.0], x, PLAYER)
    check("approaching fast and close: passes it", e <= l < b, f"earliest {e:.2f} · latest {l:.2f} · back {b:.2f}")

    print("2  nesting")
    rng = np.random.default_rng(0)
    a, s = PhysicsRacePass(spec(KIND)), PhysicsRaceSymPass(spec(KIND_SYM))
    c = rng.uniform([20, 5], [60, 63], (300, 2))
    r = c + rng.uniform([-5, -20], [30, 20], (300, 2))
    t = r + rng.uniform(-6, 6, (300, 2))
    d = rng.uniform([20, 0], [100, 68], (300, 5, 2))
    ma = a.margins(c, r, d, np.zeros((300, 2)), np.zeros((300, 5, 2)), t)
    ms = s.margins(c, r, d, np.zeros((300, 2)), np.zeros((300, 5, 2)), t)
    check("everyone standing: A-sym margins = A margins",
          np.array_equal(ma[0], ms[0]) and np.array_equal(ma[1], ms[1]))

    print("3  receiver")
    carrier = np.array([50.0, 34.0])
    runner, run_v = np.array([64.0, 34.0]), np.array([9.0, 0.0])
    far = np.array([[90.0, 60.0]])                      # a defender out of the way
    near_t = runner + [4.0, 0.0]                        # the S15 pass: 4 m ahead
    ra, _ = a.margins(carrier, runner, far, run_v, np.zeros((1, 2)), near_t)
    rs, _ = s.margins(carrier, runner, far, run_v, np.zeros((1, 2)), near_t)
    check("pass to a point he runs through too early: margin drops", rs < ra, f"A {ra:.2f} → A-sym {rs:.2f}")
    lead_t = runner + [14.0, 0.0]                        # far enough to arrive with the ball
    ra, _ = a.margins(carrier, runner, far, run_v, np.zeros((1, 2)), lead_t)
    rs, _ = s.margins(carrier, runner, far, run_v, np.zeros((1, 2)), lead_t)
    check("lead pass he can meet or stop for: margin unchanged", np.isclose(ra, rs), f"{ra:.2f} = {rs:.2f}")

    print("4  defender")
    target = np.array([80.0, 34.0])
    runner = np.array([70.0, 34.0])
    fast = np.array([[74.0, 34.0]]), np.array([[9.0, 0.0]])   # runs through the target at once
    ra, _ = a.margins(carrier, runner, fast[0], np.zeros(2), fast[1], target)
    rs, _ = s.margins(carrier, runner, fast[0], np.zeros(2), fast[1], target)
    check("defender through the target before the ball, can't stop: counts as there later",
          rs > ra, f"A {ra:.2f} → A-sym {rs:.2f}")
    slow = np.array([[74.0, 34.0]]), np.array([[2.0, 0.0]])
    ra, _ = a.margins(carrier, runner, slow[0], np.zeros(2), slow[1], target)
    rs, _ = s.margins(carrier, runner, slow[0], np.zeros(2), slow[1], target)
    check("defender who can stop on it: unchanged", np.isclose(ra, rs), f"{ra:.2f} = {rs:.2f}")

    print("5  loading")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        path.write_text(json.dumps(spec(KIND_SYM)))
        check("A-sym JSON loads as PhysicsRaceSymPass", type(load_pass_model(path)) is PhysicsRaceSymPass)
        path.write_text(json.dumps(spec(KIND)))
        check("A JSON still loads as PhysicsRacePass", type(load_pass_model(path)) is PhysicsRacePass)
    print("\nall A-sym checks passed")


if __name__ == "__main__":
    main()
