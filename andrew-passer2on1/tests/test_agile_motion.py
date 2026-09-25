#!/usr/bin/env python3
"""Does agile_motion move players the way it says?

  1  copies        the three copies (2v1, 3v1, review pages) are byte-identical
  2  reduction     with speed-up = braking = turning = 3.8 and no plant or
                   delay, the ellipse is the imported circle: advance and
                   build_layers reproduce the imported motion
  3  physics       hand-computed cases on a 0.6 s turn:
                     from rest, straight          2.7 m/s   (4.5 x 0.6)
                     6 m/s, stop                  2.4 m/s   (6.0 braking)
                     6 m/s, reverse               0.6 m/s   (plant: 9.0 braking)
                     4 m/s, 90 deg                curve, not plant
                     1.5 m/s, reverse             no plant  (under 2 m/s)
                     0.2 s delay                  onset velocity held, then 4.5
                   and on random states: speed never above the top speed, every
                   steering step inside the ellipse
  4  wiring        both games build their layers with the physics asked for,
                   physics None is the imported build_layers, and the games
                   record which physics they used

Run:  PYTHONPATH=andrew-passer2on1:andrew-fixedpasser:src \\
      .venv-delta/bin/python andrew-passer2on1/tests/test_agile_motion.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.models import GameConfig, PlayerState, Scenario
from defensive_positioning.motion import advance as imported_advance
from defensive_positioning.motion import build_layers as imported_build_layers

from passer2on1.agile_motion import AGILE, Physics, advance, build_layers, layers_for, steer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = GameConfig(steps=3, step_seconds=0.6, physics_step=0.025)
ISO = Physics("iso", 3.8, 3.8, 3.8, False, 2.0, 45.0, 9.0, 0.1, 0.0)


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def body(v, top=9.0, accel=4.5):
    return PlayerState((50.0, 34.0), tuple(v), maximum_speed=top, maximum_acceleration=accel)


SCENE = Scenario(carrier=body((0, 0)), receiver=body((0, 0)), defender=body((0, 0)),
                 attack_direction=1)
D = {d: i for i, d in enumerate(CONFIG.directions)}
FWD, BACK, SIDE, STAY = D[(1.0, 0.0)], D[(-1.0, 0.0)], D[(0.0, 1.0)], D[(0.0, 0.0)]


def end_velocity(v, command, hold=0.0):
    return advance((50.0, 34.0), v, body(v), command, SCENE, CONFIG, AGILE, hold)[1]


def main() -> None:
    print("1  copies")
    files = [ROOT / "andrew-passer2on1/passer2on1/agile_motion.py",
             ROOT / "andrew-fixedpasser/fixedpasser/agile_motion.py",
             ROOT / "src/offball_value/agile_motion.py"]
    check("three copies byte-identical", len({f.read_bytes() for f in files}) == 1)

    print("2  reduction to the imported circle")
    rng = np.random.default_rng(20260925)
    worst = 0.0
    for _ in range(500):
        v = rng.normal(0, 3, 2)
        top = max(9.0, float(np.linalg.norm(v)) + 0.1)
        pl = PlayerState((50.0, 34.0), tuple(v), maximum_speed=top, maximum_acceleration=3.8)
        c = int(rng.integers(len(CONFIG.directions)))
        a = advance((50.0, 34.0), v, pl, c, SCENE, CONFIG, ISO)
        b = imported_advance((50.0, 34.0), v, pl, c, SCENE, CONFIG)
        worst = max(worst, float(np.abs(a[2] - b[2]).max()), float(np.abs(a[1] - b[1]).max()))
    check("advance, 500 random states", worst < 1e-9, f"max diff {worst:.1e}")
    rec = json.loads((ROOT / "data/processed/stage3/passer2on1_states_agile06_final.json")
                     .read_text())["states"][0]
    sc = scenario_from(rec["scenario"])
    iso_player = PlayerState(sc.defender.position, sc.defender.velocity,
                             maximum_speed=sc.defender.maximum_speed, maximum_acceleration=3.8)
    ours = build_layers(iso_player, sc, CONFIG, ISO)
    theirs = imported_build_layers(iso_player, sc, CONFIG)
    same = all(len(a) == len(b) and np.abs(a.position - b.position).max() < 1e-9
               and np.array_equal(a.command, b.command) for a, b in zip(ours, theirs))
    check("build_layers on a real state", same, f"{[len(x) for x in ours]}")
    check("physics None is the imported function",
          all(np.array_equal(a.position, b.position) for a, b in
              zip(build_layers(sc.defender, sc, CONFIG, None),
                  imported_build_layers(sc.defender, sc, CONFIG))))

    print("3  physics")
    cases = [("from rest, straight", end_velocity((0, 0), FWD), (2.7, 0.0)),
             ("6 m/s, stop", end_velocity((6, 0), STAY), (2.4, 0.0)),
             ("6 m/s, reverse (plant)", end_velocity((6, 0), BACK), (0.6, 0.0)),
             ("1.5 m/s, reverse (no plant)", end_velocity((1.5, 0), BACK), (-1.575, 0.0))]
    for name, got, want in cases:
        check(name, np.allclose(got, want, atol=1e-9), f"{np.round(got, 4)} vs {want}")
    v = end_velocity((4, 0), SIDE)
    curve_only = Physics(**{**AGILE.as_dict(), "name": "curve", "plant_cut": False})
    v_curve = advance((50.0, 34.0), (4, 0), body((4, 0)), SIDE, SCENE, CONFIG, curve_only)[1]
    check("4 m/s, 90 deg: curve, not plant", np.allclose(v, v_curve) and v[0] > 2 and v[1] > 3,
          f"{np.round(v, 3)}")
    p, v, path = advance((50.0, 34.0), (3, 1), body((3, 1)), SIDE, SCENE, CONFIG, AGILE, 0.2)
    held = path[:9] - path[0]
    lin = np.outer(np.arange(9) * CONFIG.physics_step, (3, 1))
    check("0.2 s delay holds the onset velocity", np.abs(held - lin).max() < 1e-9)
    p0, v0, _ = advance((50.0, 34.0), (0, 0), body((0, 0)), FWD, SCENE, CONFIG, AGILE, 0.2)
    check("then speeds up at 4.5", np.allclose(v0, (1.8, 0.0), atol=1e-9), f"{np.round(v0, 4)}")
    fastest, outside = 0.0, 0.0
    dt = CONFIG.physics_step
    for _ in range(400):
        v = rng.normal(0, 4, 2)
        v = v if np.linalg.norm(v) < 8.9 else v * 8.9 / np.linalg.norm(v)
        c = int(rng.integers(len(CONFIG.directions)))
        _, ve, path = advance((52.5, 34.0), v, body(v), c, SCENE, CONFIG, AGILE)
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt
        fastest = max(fastest, float(steps.max()))
        desired = np.asarray(CONFIG.directions[c]) * 9.0
        w = steer(v, desired, AGILE, dt)
        dv, s = w - v, float(np.linalg.norm(v))
        if s > 1e-12:
            t = v / s
            a_t, a_n = float(dv @ t) / dt, float(np.linalg.norm(dv - (dv @ t) * t)) / dt
            cap = AGILE.speed_up if a_t >= 0 else AGILE.braking
            outside = max(outside, math.hypot(a_t / cap, a_n / AGILE.turning))
    check("speed never above 9.0", fastest <= 9.0 + 1e-9, f"max {fastest:.4f}")
    check("steering inside the ellipse", outside <= 1 + 1e-9, f"max {outside:.6f}")

    print("4  wiring")
    from passer2on1.game import Background2on1Game
    from fixedpasser.game import FixedPasserGame
    from defensive_positioning.expected_pass import ExpectedPass
    model = ExpectedPass.load(ROOT / "andrew/models/experimental_pass.json", allow_proxy=True)
    bg = rec["background"]
    g = Background2on1Game(sc, model, CONFIG, background=np.asarray(bg["positions"]),
                           background_velocity=np.asarray(bg["velocities"]), physics="agile")
    want = layers_for(sc, CONFIG, "agile")
    check("2v1 game uses agile layers", all(
        np.array_equal(a[k].position, b[k].position)
        for a, b in zip((g.carrier, g.receiver, g.defender), want) for k in range(4)))
    check("2v1 game records its physics", g.physics == AGILE.as_dict())
    r3 = json.loads((ROOT / "data/processed/stage3/fixedpasser_states_agile06_final.json")
                    .read_text())["states"][0]
    s3 = scenario_from(r3["scenario"])
    kw = dict(passer=np.asarray(r3["passer"]["positions"]),
              passer_velocity=np.asarray(r3["passer"]["velocities"]),
              background=np.asarray(r3["background"]["positions"]),
              background_velocity=np.asarray(r3["background"]["velocities"]))
    g3 = FixedPasserGame(s3, model, CONFIG, physics="agile", **kw)
    want3 = layers_for(s3, CONFIG, "agile")
    check("3v1 game uses agile layers, delay on the defender only", all(
        np.array_equal(a[k].position, b[k].position)
        for a, b in zip((g3.carrier, g3.receiver, g3.defender), want3) for k in range(4)))
    g0 = FixedPasserGame(s3, model, CONFIG, **kw)
    check("3v1 game without physics is the imported motion", all(
        np.array_equal(a[k].position, b[k].position)
        for a, b in zip((g0.carrier, g0.receiver, g0.defender),
                        [imported_build_layers(p, s3, CONFIG) for p in (s3.carrier, s3.receiver, s3.defender)])
        for k in range(4)) and g0.physics == {"name": "andrew"})
    print("\nall agile-motion checks passed")


if __name__ == "__main__":
    main()
