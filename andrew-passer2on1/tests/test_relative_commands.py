#!/usr/bin/env python3
"""Do the relative commands and run-aligned passes do what they claim?

  1  copies       agile_motion.py and run_passes.py byte-identical in their
                  three homes
  2  commands     on a real 3v1 scene whose runner sprints in behind:
                    - every body has 5 commands and command 0 is the stop
                    - the intercept point lies goal-side of the runner and on
                      his projected line; three turns of intercept close on
                      the runner's 1.8 s point and end goal-side of the start
                    - "continue" is the body's own heading; the stop is zero
                    - mirroring the scene (x -> 105 - x, direction flipped)
                      mirrors every layer exactly
                    - layers keep the 1 / 5 / 25 / 125 state counts
  3  run passes   18 candidates; along a diagonal run the "8 m ahead" target
                  sits 8 m along the velocity, not the axis; a standing
                  receiver gets the axis; the goalward targets point at the goal
  4  reduction    with compass commands, the imported passes and threat
                  "andrew", release_payoffs_with_background is unchanged (the
                  target helper falls back to PassChoice.target)
  5  threat       epv_time: a receiver free in behind a defender running the
                  other way is worth more than one marked at his feet, at the
                  same location; the andrew threat is direction-blind there

Run:  PYTHONPATH=andrew-passer2on1:andrew-fixedpasser:src \\
      .venv-delta/bin/python andrew-passer2on1/tests/test_relative_commands.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.expected_pass import ExpectedPass
from defensive_positioning.models import DEFAULT_PASSES, GameConfig

from passer2on1.agile_motion import (command_direction, command_names, intercept_point, layers_for,
                                     relative_commands)
from passer2on1.payoff import release_payoffs_with_background, threat_epv_time, possession_value_all
from passer2on1.run_passes import RUN_PASSES, RunPass, target_of

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("OFFBALL_DATA_ROOT", ROOT))   # states and Andrew's model live outside the repository
CFG = GameConfig(steps=3, step_seconds=0.6, physics_step=0.025)


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def mirrored(rec):
    m = json.loads(json.dumps(rec))
    for slot in ("carrier", "receiver", "defender"):
        m["scenario"][slot]["position"][0] = 105.0 - m["scenario"][slot]["position"][0]
        m["scenario"][slot]["velocity"][0] *= -1
    m["scenario"]["attack_direction"] = -rec["scenario"]["attack_direction"]
    if "passer" in m:
        m["passer"]["positions"] = [[105.0 - x, y] for x, y in m["passer"]["positions"]]
        m["passer"]["velocities"] = [[-vx, vy] for vx, vy in m["passer"]["velocities"]]
    return m


def main() -> None:
    print("1  copies")
    for name in ("agile_motion.py", "run_passes.py"):
        files = [ROOT / "andrew-passer2on1/passer2on1" / name, ROOT / "andrew-fixedpasser/fixedpasser" / name,
                 ROOT / "src/offball_value" / name]
        check(f"{name} byte-identical", len({f.read_bytes() for f in files}) == 1)

    print("2  commands")
    states = json.loads((DATA / "data/processed/stage3/fixedpasser_states_agile06_final2m.json").read_text())["states"]
    rec = max(states, key=lambda r: r["scenario"]["carrier"]["velocity"][0] * r["scenario"]["attack_direction"])
    sc = scenario_from(rec["scenario"])
    ad = sc.attack_direction
    cmds = relative_commands(rec, CFG, "3v1")
    check("5 commands per body, command 0 is the stop",
          all(len(c) == 5 and c[0].kind == "stop" for c in cmds.values()), str(command_names(cmds)))
    d, dv = np.asarray(sc.defender.position), np.asarray(sc.defender.velocity)
    r, rv = np.asarray(sc.carrier.position), np.asarray(sc.carrier.velocity)
    ip = intercept_point(d, 0, r, rv, CFG, 9.0, sc)
    along = rv / np.linalg.norm(rv)
    off_line = abs(along[0] * (ip - r)[1] - along[1] * (ip - r)[0])
    check("intercept point is goal-side of the runner and on his line",
          (ip[0] - r[0]) * ad > 0 and off_line < 0.5, f"runner x {r[0]:.1f}, point x {ip[0]:.1f}, off line {off_line:.2f} m")
    layers = layers_for(sc, CFG, "agile", cmds)
    # exact-identical states merge (the imported rule), so counts are at most 5^k
    counts = [[len(layers[i][k]) for k in range(4)] for i in range(3)]
    valid = all(layers[i][k].successor.max() < len(layers[i][k + 1]) and layers[i][k].successor.min() >= 0
                for i in range(3) for k in range(3))
    check("state counts at most 1/5/25/125 and successors valid",
          all(c[k] <= 5 ** k for c in counts for k in range(4)) and valid, str(counts))
    idx = 0
    for k in range(3):
        idx = int(layers[2][k].successor[idx, 1])          # command 1 of the defender = 러너 차단
    end = layers[2][3].position[idx]
    goal_point = r + rv * 1.8
    check("three turns of intercept close on the runner's 1.8 s point, goal-side of the start",
          np.linalg.norm(end - goal_point) < np.linalg.norm(d - goal_point) and (end[0] - d[0]) * ad > 0,
          f"{np.linalg.norm(d - goal_point):.1f} m -> {np.linalg.norm(end - goal_point):.1f} m")
    cont = next(c for c in cmds["carrier"] if c.kind == "continue")
    u = command_direction(cont, r, rv, 0, CFG, sc, 9.0)
    check("continue = own heading", np.allclose(u, rv / np.linalg.norm(rv)))
    check("stop = zero", np.allclose(command_direction(cmds["defender"][0], d, dv, 0, CFG, sc, 9.0), 0))
    m = mirrored(rec)
    lm = layers_for(scenario_from(m["scenario"]), CFG, "agile", relative_commands(m, CFG, "3v1"))
    worst = max(np.abs((105.0 - lm[i][k].position[:, 0]) - layers[i][k].position[:, 0]).max()
                for i in range(3) for k in range(4))
    check("mirrored scene gives mirrored layers", worst < 1e-9, f"max {worst:.1e} m")

    print("3  run passes")
    check("18 candidates, all ground", len(RUN_PASSES) == 18 and all(p.family == "ground" for p in RUN_PASSES))
    recv, vel = np.array([60.0, 30.0]), np.array([4.0, 3.0])
    ahead = next(p for p in RUN_PASSES if p.along == 8.0 and p.lateral == 0.0)
    t = target_of(ahead, recv, 1, vel, sc)
    check("8 m ahead lies along the run direction", np.allclose(t, recv + 8.0 * vel / 5.0), str(np.round(t, 2)))
    t0 = target_of(ahead, recv, 1, np.zeros(2), sc)
    check("standing receiver: along the attack axis", np.allclose(t0, recv + np.array([8.0, 0.0])))
    t_flip = target_of(ahead, recv, -1, np.zeros(2), sc)
    check("attacking the other way: the axis flips", np.allclose(t_flip, recv + np.array([-8.0, 0.0])))
    goalward = next(p for p in RUN_PASSES if p.goalward == 8.0)
    goal = np.array([105.0 if ad == 1 else 0.0, 34.0])
    tg = target_of(goalward, recv, ad, vel, sc)
    check("goalward target points at the goal", np.isclose(np.linalg.norm(tg - recv), 8.0)
          and np.linalg.norm(goal - tg) < np.linalg.norm(goal - recv))
    check("PassChoice still works through target_of",
          np.allclose(target_of(DEFAULT_PASSES[3], recv, 1, vel, sc), DEFAULT_PASSES[3].target(recv, 1)))

    print("4  reduction")
    model = ExpectedPass.load(DATA / "andrew/models/experimental_pass.json", allow_proxy=True)
    rng = np.random.default_rng(7)
    n = 300
    carrier = rng.uniform([30, 5], [80, 63], (n, 2))
    receiver = rng.uniform([30, 5], [100, 63], (n, 2))
    defender = rng.uniform([30, 5], [100, 63], (n, 2))
    cv, rv_, dv_ = (rng.normal(0, 2, (n, 2)) for _ in range(3))
    bg, bgv = rng.uniform([0, 0], [105, 68], (10, 2)), rng.normal(0, 2, (10, 2))
    a = release_payoffs_with_background(sc, CFG, model, carrier, receiver, defender, cv, rv_, dv_, bg, bgv)
    b = release_payoffs_with_background(sc, CFG, model, carrier, receiver, defender, cv, rv_, dv_, bg, bgv,
                                        threat="andrew")
    check("default call unchanged (explicit threat='andrew' identical)", np.array_equal(a, b))

    print("5  threat")
    target = np.array([80.0, 34.0]) if ad == 1 else np.array([25.0, 34.0])
    passer = np.array([60.0, 34.0]) if ad == 1 else np.array([45.0, 34.0])
    u = np.array([float(ad), 0.0])
    # a through ball: the receiver 8 m short of the target sprinting at it, a
    # defender 3 m behind him running the other way -- against a receiver standing
    # on the target with a defender 1 m goal-side of him
    free_in_behind = threat_epv_time(target, passer, (target - 11.0 * u)[None, :], (-3.0 * u)[None, :],
                                     target - 8.0 * u, 7.0 * u, sc)
    marked = threat_epv_time(target, passer, (target + 1.0 * u)[None, :], np.zeros((1, 2)), target, np.zeros(2), sc)
    check("epv_time: free in behind > marked at feet, same spot", free_in_behind > marked,
          f"{free_in_behind:.3f} vs {marked:.3f}")
    from passer2on1.payoff import epv_location
    # a standing marker and a receiver on the spot: the room reduces to 준현's 1 - exp(-d/9)
    for dist in (1.0, 3.0, 6.0, 12.0):
        got = threat_epv_time(target, passer, (target + dist * u)[None, :], np.zeros((1, 2)), target, np.zeros(2), sc)
        room = 1 - np.exp(-dist / 9.0)
        want = epv_location(target, sc) * (0.55 + 0.25 * room + 0.2 * epv_location(passer, sc)
                                           * (1 - np.exp(-np.linalg.norm(passer - (target + dist * u)) / 9.0)))
        check(f"standing marker at {dist:g} m reproduces the distance room", np.isclose(got, want, atol=1e-9),
              f"{got:.4f} vs {want:.4f}")
    # a third defender 10 m beside the passer fixes the support room in both cases
    side = passer + np.array([0.0, 10.0])
    a1 = possession_value_all(target, passer, np.stack([target - 3.0 * u, side]), sc)
    a2 = possession_value_all(target, passer, np.stack([target + 3.0 * u, side]), sc)
    check("andrew threat is direction-blind (3 m behind = 3 m in front)", np.isclose(a1, a2), f"{a1:.4f} vs {a2:.4f}")
    print("\nall relative-command checks passed")


if __name__ == "__main__":
    main()
