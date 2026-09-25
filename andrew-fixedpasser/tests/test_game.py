#!/usr/bin/env python3
"""Does the 3v1 game do what it claims?

  1  shared pricing   fixedpasser.payoff and passer2on1.payoff give identical
                      numbers on random kick states -- the two games must price
                      a pass the same way; the copy is not allowed to drift.
                      Same for background_tracks on a real scene.
  2  exactness        certificate gap within the solver's own tolerance.
  3  symmetry         swapping which receiver sits in which slot leaves the game
                      value unchanged: the two receivers are treated alike, so
                      only their positions should matter, never their slot.
  4  monotonicity     taking one receiver's passes away can only lower the
                      attack's value, and every release then goes to the one
                      left. Checked both ways.
  5  bookkeeping      tackling off (survival all ones), keeping the ball worth
                      nothing, release codes in [-1, 36), and the solver's
                      start positions equal to the tracking at onset.

Structural checks (3, 4) use two turns of one second: the properties hold for
any horizon and the smaller game builds in seconds. Exactness and positions use
the real three-turn game.

Run:  PYTHONPATH=andrew-fixedpasser:andrew-passer2on1 \\
      .venv-delta/bin/python andrew-fixedpasser/tests/test_game.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.expected_pass import ExpectedPass
from defensive_positioning.markov import certificate, solve_markov_game
from defensive_positioning.models import GameConfig

import fixedpasser.payoff as fp_pay
import fixedpasser.tracks as fp_tracks
import passer2on1.payoff as p2_pay
import passer2on1.tracks as p2_tracks
from fixedpasser.game import FixedPasserGame
from fixedpasser.rollout import rollout

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "andrew/models/experimental_pass.json"
STATES = ROOT / "data/processed/stage3/fixedpasser_states_all.json"
BUILD = Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac")
FULL = GameConfig(steps=3, step_seconds=1.0, physics_step=0.025)
SMALL = GameConfig(steps=2, step_seconds=1.0, physics_step=0.025)


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def build(record, model, config, *, swap=False, targets=("runner", "beneficiary")):
    steps = config.steps
    scen = dict(record["scenario"])
    if swap:
        scen["carrier"], scen["receiver"] = scen["receiver"], scen["carrier"]
        targets = tuple({"runner": "beneficiary", "beneficiary": "runner"}[t] for t in targets)
    passer = record["passer"]
    bg = record["background"]
    return FixedPasserGame(scenario_from(scen), model, config,
                           passer=np.asarray(passer["positions"])[:steps + 1],
                           passer_velocity=np.asarray(passer["velocities"])[:steps + 1],
                           background=np.asarray(bg["positions"])[:steps + 1],
                           background_velocity=np.asarray(bg["velocities"])[:steps + 1],
                           targets=targets)


def main() -> None:
    rng = np.random.default_rng(20260924)
    model = ExpectedPass.load(MODEL, allow_proxy=True)
    records = json.loads(STATES.read_text())["states"]
    # a scene attacking each way, so direction handling is exercised both times
    by_dir = {}
    for r in records:
        by_dir.setdefault(int(r["scenario"]["attack_direction"]), r)
    picks = [by_dir[1], by_dir[-1]]
    sc = scenario_from(picks[0]["scenario"])

    print("1  shared pricing with andrew-passer2on1")
    n = 4000
    carrier = rng.uniform([30, 5], [80, 63], (n, 2))
    receiver = rng.uniform([30, 5], [100, 63], (n, 2))
    defender = rng.uniform([30, 5], [100, 63], (n, 2))
    cv, rv, dv = (rng.normal(0, 2, (n, 2)) for _ in range(3))
    bg = rng.uniform([0, 0], [105, 68], (10, 2))
    bgv = rng.normal(0, 2, (10, 2))
    a, la = fp_pay.release_payoffs_with_background(sc, FULL, model, carrier, receiver, defender,
                                                   cv, rv, dv, bg, bgv, return_legal=True)
    b, lb = p2_pay.release_payoffs_with_background(sc, FULL, model, carrier, receiver, defender,
                                                   cv, rv, dv, bg, bgv, return_legal=True)
    check("release values identical", np.array_equal(a, b))
    check("release legality identical", np.array_equal(la, lb))
    prov = picks[0]["provenance"]
    payload = json.loads((BUILD / prov["scene_dir"] / "local_game_payoff_audits.json").read_text())[0]
    ids = {prov["runner_id"], prov["beneficiary_id"], prov["defender_id"], prov["carrier_id"]}
    t1 = fp_tracks.background_tracks(payload, ids, 3, 1.0)
    t2 = p2_tracks.background_tracks(payload, ids, 3, 1.0)
    check("background tracks identical", t1["ids"] == t2["ids"]
          and np.array_equal(t1["positions"], t2["positions"]))

    print("3  symmetry  (2 turns x 1.0 s, one scene each attacking direction)")
    for r in picks:
        t = time.perf_counter()
        g1, g2 = build(r, model, SMALL), build(r, model, SMALL, swap=True)
        v1, v2 = solve_markov_game(g1).root_value, solve_markov_game(g2).root_value
        check(f"direction {int(r['scenario']['attack_direction']):+d}: swapped slots, same value",
              abs(v1 - v2) <= 2 * SMALL.steps * SMALL.solver_tolerance,
              f"{v1:.12f} vs {v2:.12f}, {time.perf_counter() - t:.0f}s")

    print("4  monotonicity")
    for r in picks:
        d = int(r["scenario"]["attack_direction"])
        full = solve_markov_game(build(r, model, SMALL)).root_value
        for keep in ("runner", "beneficiary"):
            g = build(r, model, SMALL, targets=(keep,))
            v = solve_markov_game(g).root_value
            codes = np.concatenate([p.ravel() for p in g.pass_index])
            legal = codes[codes >= 0]
            only = np.all(legal < 18) if keep == "runner" else np.all(legal >= 18)
            check(f"direction {d:+d}: only {keep} -> value no higher", v <= full + 1e-12,
                  f"{v:.4f} <= {full:.4f}")
            check(f"direction {d:+d}: only {keep} -> every release goes to {keep}", bool(only))

    print("2,5  the real game  (3 turns x 1.0 s)")
    r = picks[0]
    t = time.perf_counter()
    g = build(r, model, FULL)
    print(f"      built in {time.perf_counter() - t:.0f}s, shapes {g.shapes}")
    sol = solve_markov_game(g)
    bounds = certificate(sol)
    check("certificate", bounds["gap"] <= 2 * FULL.steps * FULL.solver_tolerance,
          f"gap {bounds['gap']:.1e}, value {sol.root_value:.4f}")
    check("tackling off", all(np.all(s == 1.0) for s in g.survival))
    check("keeping the ball worth nothing", np.all(g.retained == 0.0))
    codes = np.concatenate([p.ravel() for p in g.pass_index])
    check("release codes in [-1, 36)", codes.min() >= -1 and codes.max() < 36,
          f"{int((codes >= 18).sum())} to the beneficiary, {int(((codes >= 0) & (codes < 18)).sum())} to the runner")
    frames = payload["background_frames"]
    onset = min(frames, key=lambda f: abs(f["frame_id"] - payload["onset_frame_id"]))
    real = {str(p[0]): np.array([p[2] + 52.5, p[3] + 34.0]) for p in onset["players"]}
    err = max(np.hypot(*(g.carrier[0].position[0] - real[prov["runner_id"]])),
              np.hypot(*(g.receiver[0].position[0] - real[prov["beneficiary_id"]])),
              np.hypot(*(g.defender[0].position[0] - real[prov["defender_id"]])),
              np.hypot(*(g.passer[0] - real[prov["carrier_id"]])))
    check("start positions equal the tracking", err < 1e-3, f"max {err:.1e} m")
    plays = [rollout(sol, s) for s in range(4)]
    ends = [p[-1]["event"] + (f"->{p[-1]['to']}" if p[-1]["event"] == "release" else "") for p in plays]
    check("rollouts decode their receiver", all(e in ("release->runner", "release->beneficiary",
                                                      "no_pass") for e in ends), ", ".join(ends))
    print("\nall 3v1 checks passed")


if __name__ == "__main__":
    main()
