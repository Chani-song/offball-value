#!/usr/bin/env python3
"""Does `release_steps` remove passes at the chosen instants and nothing else?

  1  default          no release_steps and all-True build the same game and
                      solve to the same value and policies, bit for bit.
  2  middle removed   (True, False, True): no legal release at instant 1, the
                      solver never releases there, the other instants' releases
                      are untouched, the certificate holds, and the attack is
                      worth no more than with every instant open.
  3  horizon removed  (True, True, False): no terminal release, the end of
                      the game is worth what keeping the ball is, nothing.
  4  rollouts         sampled plays under (True, False, True) never release at
                      instant 1.
  5  input            wrong length, numbers instead of booleans, or nothing
                      allowed are refused; numpy booleans are accepted; a
                      record's JSON `release_steps` reaches the game through
                      solve.game_from, and a record without one opens every
                      instant.

Structural checks, so two turns of one second as in test_game.py: the
properties hold for any horizon and the smaller game builds in seconds.

Run from the repository root:
      PYTHONPATH=andrew-fixedpasser:andrew-passer2on1 \\
      python andrew-fixedpasser/tests/test_release_steps.py
OFFBALL_DATA_ROOT points at the tree holding andrew/models and
data/processed/stage3 when that is not this repository.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.expected_pass import ExpectedPass
from defensive_positioning.markov import certificate, solve_markov_game
from defensive_positioning.models import GameConfig

from fixedpasser.game import FixedPasserGame
from fixedpasser.rollout import rollout
from fixedpasser.solve import game_from

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("OFFBALL_DATA_ROOT", ROOT))
MODEL = DATA / "andrew/models/experimental_pass.json"
STATES = DATA / "data/processed/stage3/fixedpasser_states_all.json"
SMALL = GameConfig(steps=2, step_seconds=1.0, physics_step=0.025)
TOL = 2 * SMALL.steps * SMALL.solver_tolerance


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def sliced(record, steps):
    """The record with its tracks cut to steps + 1 instants."""
    out = dict(record)
    out["passer"] = {k: np.asarray(v)[:steps + 1].tolist() for k, v in record["passer"].items()
                     if k in ("positions", "velocities")}
    bg = record["background"]
    out["background"] = {**bg, "positions": np.asarray(bg["positions"])[:steps + 1].tolist(),
                         "velocities": np.asarray(bg["velocities"])[:steps + 1].tolist()}
    return out


def build(record, model, config, release_steps=None):
    r = sliced(record, config.steps)
    return FixedPasserGame(scenario_from(r["scenario"]), model, config,
                           passer=np.asarray(r["passer"]["positions"]),
                           passer_velocity=np.asarray(r["passer"]["velocities"]),
                           background=np.asarray(r["background"]["positions"]),
                           background_velocity=np.asarray(r["background"]["velocities"]),
                           release_steps=release_steps)


def same_arrays(xs, ys):
    return len(xs) == len(ys) and all(np.array_equal(x, y) for x, y in zip(xs, ys))


def refused(make):
    try:
        make()
    except ValueError:
        return True
    return False


def main() -> None:
    model = ExpectedPass.load(MODEL, allow_proxy=True)
    records = json.loads(STATES.read_text())["states"]
    by_dir = {}
    for r in records:
        by_dir.setdefault(int(r["scenario"]["attack_direction"]), r)
    picks = [by_dir[1], by_dir[-1]]

    for r in picks:
        d = int(r["scenario"]["attack_direction"])
        print(f"direction {d:+d}  (index {r['index']})")

        print(" 1  default")
        g0, g1 = build(r, model, SMALL), build(r, model, SMALL, (True, True, True))
        s0, s1 = solve_markov_game(g0), solve_markov_game(g1)
        check("same release tables", same_arrays(g0.release, g1.release)
              and same_arrays(g0.pass_index, g1.pass_index))
        check("same value", s0.root_value == s1.root_value, f"{s0.root_value:.12f}")
        check("same policies", same_arrays(s0.attack_policy[:-1], s1.attack_policy[:-1])
              and same_arrays(s0.defender_policy[:-1], s1.defender_policy[:-1]))
        check("every instant open by default", g0.release_steps == (True, True, True))

        print(" 2  middle instant removed")
        g = build(r, model, SMALL, (True, False, True))
        s = solve_markov_game(g)
        check("instant 1 had legal releases before (the check is not empty)",
              bool(np.any(g0.pass_index[1] >= 0)), f"{int((g0.pass_index[1] >= 0).sum())} states")
        check("no legal release at instant 1", np.all(g.pass_index[1] == -1)
              and np.all(g.release[1] == 0.0))
        check("instants 0 and 2 untouched", np.array_equal(g.release[0], g0.release[0])
              and np.array_equal(g.release[2], g0.release[2])
              and np.array_equal(g.pass_index[0], g0.pass_index[0])
              and np.array_equal(g.pass_index[2], g0.pass_index[2]))
        opened = [bool(cm[:, -1].any()) for flat in g.chunks(1)
                  for _, cm in [g.masks(1, flat, s.schedule)]]
        check("release column masked at instant 1", not any(opened))
        check("solver never releases at instant 1",
              float(s.attack_policy[1][..., -1].max()) <= 1e-12)
        bounds = certificate(s)
        check("certificate", bounds["gap"] <= TOL, f"gap {bounds['gap']:.1e}")
        check("value no higher than with every instant open", s.root_value <= s0.root_value + 1e-12,
              f"{s.root_value:.6f} <= {s0.root_value:.6f}")

        print(" 3  horizon removed")
        g = build(r, model, SMALL, (True, True, False))
        s = solve_markov_game(g)
        check("no terminal release", not s.terminal_release.any())
        check("horizon worth nothing", np.all(s.value[-1] == 0.0))
        check("certificate", certificate(s)["gap"] <= TOL)
        open_plays = [rollout(s0, seed) for seed in range(20)]
        if any(p[-1]["event"] == "release" and len(p) - 1 == SMALL.steps for p in open_plays):
            # the open game really passes at the horizon, so closing it must cost the attack
            check("closing the horizon the open game passes at lowers the value",
                  s.root_value < s0.root_value, f"{s.root_value:.6f} < {s0.root_value:.6f}")

        print(" 4  rollouts")
        g = build(r, model, SMALL, (True, False, True))
        s = solve_markov_game(g)
        at = []
        for seed in range(20):
            play = rollout(s, seed)
            if play[-1]["event"] == "release":
                at.append(len(play) - 1)
        check("no release sampled at instant 1", 1 not in at, f"releases at instants {sorted(set(at))}")

    print(" 5  input")
    r = picks[0]
    check("two flags for three instants refused",
          refused(lambda: build(r, model, SMALL, (True, False))))
    check("instant numbers refused", refused(lambda: build(r, model, SMALL, (0, 1, 1))))
    check("nothing allowed refused", refused(lambda: build(r, model, SMALL, (False,) * 3)))
    g = build(r, model, SMALL, np.array([True, False, True]))
    check("numpy booleans accepted", g.release_steps == (True, False, True))
    record = sliced(r, SMALL.steps)
    record["release_steps"] = json.loads("[true, false, true]")
    check("record release_steps reaches the game",
          game_from(record, model, SMALL).release_steps == (True, False, True))
    record.pop("release_steps")
    check("record without release_steps opens every instant",
          game_from(record, model, SMALL).release_steps == (True, True, True))
    print("\nall release_steps checks passed")


if __name__ == "__main__":
    main()
