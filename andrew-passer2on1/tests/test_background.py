#!/usr/bin/env python3
"""With background defenders present, do the three changes move the right way?

Constructed situations, each with a football expectation:

  1  offside   A runner past the controlled defender but not past the deepest
               outfield background defender is ONSIDE under the full rule, and
               OFFSIDE under the imported single-defender line.
  2  pass      A background defender standing on the receiver lowers the
               pass model's completion probability for a pass to his feet.
  3  threat    A background defender standing on the target lowers the threat
               of receiving there; one far away changes nothing.
  4  tracks    Background positions from a real scene: every non-strategic
               defender present, the goalkeeper included, positions equal to
               the tracking at t = 0 exactly, and interpolation between frames
               at t = 1.0.

Run:  PYTHONPATH=andrew-passer2on1 .venv-delta/bin/python andrew-passer2on1/tests/test_background.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.expected_pass import ExpectedPass
from defensive_positioning.models import GameConfig, PassChoice

from passer2on1.payoff import (offside_flags, positional_threat_all,
                               release_payoffs_with_background)
from passer2on1.tracks import background_tracks

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "andrew/models/experimental_pass.json"
STATES = ROOT / "data/processed/stage3/carrier_beneficiary_states_v2.json"
BUILD = Path("/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac")


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    model = ExpectedPass.load(MODEL, allow_proxy=True)
    record = json.loads(STATES.read_text())["states"][0]
    sc = scenario_from(record["scenario"])
    if sc.attack_direction != 1:
        # the constructed geometry below is written attacking +x
        record["scenario"]["attack_direction"] = 1
        sc = scenario_from(record["scenario"])

    print("1  offside  (attacking +x, goal at x = 105)")
    ball = np.array([[60.0, 34.0]])
    runner = np.array([[80.0, 30.0]])
    controlled = np.array([[75.0, 32.0]])           # runner is past him
    background = np.array([[85.0, 40.0], [100.0, 34.0]])   # deep centre-back, keeper
    everyone = np.concatenate([controlled[:, None, :], background[None]], axis=1)
    single = offside_flags(sc, runner, ball, controlled[:, None, :])
    full = offside_flags(sc, runner, ball, everyone)
    check("single-defender line flags him", bool(single[0]))
    check("full rule keeps him on (centre-back deeper)", not bool(full[0]))
    beyond = offside_flags(sc, np.array([[90.0, 30.0]]), ball, everyone)
    check("past the centre-back too -> offside", bool(beyond[0]))

    print("2  pass model")
    config = GameConfig(passes=(PassChoice((0.0, 0.0), "ground"),))
    carrier = np.array([[60.0, 34.0]])
    receiver = np.array([[72.0, 20.0]])
    # deep and wide: far from the receiver, and deeper than him so he is onside
    # against it alone AND against it plus the marker (second-last = marker, 72.5)
    far_defender = np.array([[100.0, 60.0]])
    zero = np.zeros((1, 2))
    alone = release_payoffs_with_background(sc, config, model, carrier, receiver, far_defender,
                                            zero, zero, zero, np.zeros((0, 2)), np.zeros((0, 2)))
    marked = release_payoffs_with_background(sc, config, model, carrier, receiver, far_defender,
                                             zero, zero, zero, np.array([[72.5, 20.5]]),
                                             np.zeros((1, 2)))
    check("a marker on the receiver lowers the pass value",
          float(marked[0, 0]) < float(alone[0, 0]),
          f"{float(alone[0, 0]):.3f} -> {float(marked[0, 0]):.3f}")

    print("3  threat")
    target = np.array([[90.0, 34.0]])
    support = np.array([[70.0, 34.0]])
    one = np.array([[[60.0, 5.0]]])
    near = np.array([[[60.0, 5.0], [91.0, 34.0]]])
    farther = np.array([[[60.0, 5.0], [10.0, 60.0]]])
    a = positional_threat_all(target, support, one, sc)
    b = positional_threat_all(target, support, near, sc)
    c = positional_threat_all(target, support, farther, sc)
    check("defender on the target lowers threat", float(b[0]) < float(a[0]),
          f"{float(a[0]):.3f} -> {float(b[0]):.3f}")
    check("defender far away changes nothing", float(c[0]) == float(a[0]))

    print("4  tracks from a real scene")
    prov = json.loads(STATES.read_text())["states"][0]["provenance"]
    payload = json.loads((BUILD / prov["scene_dir"] / "local_game_payoff_audits.json").read_text())[0]
    strategic = {prov["carrier_id"], prov["runner_id"], prov["defender_id"]}
    bg = background_tracks(payload, strategic, steps=3, step_seconds=1.0)
    frames = payload["background_frames"]
    onset = min(frames, key=lambda f: abs(f["frame_id"] - payload["onset_frame_id"]))
    att = str(payload["attacking_team_id"])
    defending = [str(p[0]) for p in onset["players"] if str(p[1]) != att]
    check("every defender but the controlled one, keeper included",
          sorted(bg["ids"]) == sorted(d for d in defending if d != prov["defender_id"]),
          f"{len(bg['ids'])} background, {len(bg['dropped'])} dropped")
    at0 = {str(p[0]): (p[2] + 52.5, p[3] + 34.0) for p in onset["players"]}
    err0 = max(np.hypot(*(bg["positions"][0, j] - np.array(at0[pid]))) for j, pid in enumerate(bg["ids"]))
    check("t = 0 equals the onset frame", err0 < 1e-9, f"max {err0:.1e} m")
    t0 = float(onset["relative_time_s"])
    lo = max((f for f in frames if float(f["relative_time_s"]) - t0 <= 1.0),
             key=lambda f: float(f["relative_time_s"]))
    hi = min((f for f in frames if float(f["relative_time_s"]) - t0 >= 1.0),
             key=lambda f: float(f["relative_time_s"]))
    pid = bg["ids"][0]
    plo = next(np.array([p[2] + 52.5, p[3] + 34.0]) for p in lo["players"] if str(p[0]) == pid)
    phi = next(np.array([p[2] + 52.5, p[3] + 34.0]) for p in hi["players"] if str(p[0]) == pid)
    between = (min(plo[0], phi[0]) - 1e-9 <= bg["positions"][1, 0, 0] <= max(plo[0], phi[0]) + 1e-9
               and min(plo[1], phi[1]) - 1e-9 <= bg["positions"][1, 0, 1] <= max(plo[1], phi[1]) + 1e-9)
    check("t = 1.0 lies between the frames either side", between,
          f"frames at {float(lo['relative_time_s']) - t0:.2f} / {float(hi['relative_time_s']) - t0:.2f} s")
    check("3.0 s instant taken from the last frame", abs(bg["times_used"][3] - 2.96) < 0.02,
          f"used {bg['times_used'][3]:.2f} s")
    print("\nall background checks passed")


if __name__ == "__main__":
    main()
