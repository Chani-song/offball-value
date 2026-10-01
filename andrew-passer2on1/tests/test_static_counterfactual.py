"""Checks for scripts/static_counterfactual.py (2026-09-30): holding one side to its real commands at every turn.

  1 held defender   the solver with the defender masked to his real command each turn = an independent dynamic
                    programme over the attack's choices against that defender path (max at every turn)
  2 held attack     the solver with the attack masked to its real columns = an independent programme over the
                    defender's commands against that attack path (min at every turn)
  3 order           held defender >= equilibrium >= held attack, and the static-best plan read off (1) is worth
                    no more than (1) once the defender answers it
  4 real commands   a player placed exactly on a command's end is read as that command

A two-turn game (S05@0.6's first 1.2 s), the real tracking for the held paths.

Run: PYTHONPATH=andrew-passer2on1:src:scripts python andrew-passer2on1/tests/test_static_counterfactual.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from defensive_positioning.models import DecisionSchedule, GameConfig
from offball_value.bundesliga import find_bundesliga_files, load_bundesliga_frames
from passer2on1.multi_markov import solve_multi
from passer2on1.physics_pass import load_pass_model
from passer2on1.solve import game_from
import static_counterfactual as sc

ROOT = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[2]))
STATES = ROOT / "data/processed/showcase_v1/figure_2v1/states_2v1_all.json"
PHYSICS = json.loads((ROOT / "data/processed/physics_limits/agile_p999_nodelay.json").read_text())
CONFIG = GameConfig(steps=2, step_seconds=0.6, physics_step=0.025)
MODEL = load_pass_model(ROOT / "andrew/models/experimental_pass.json", True)
TOL = 1e-9           # the LP solver's own tolerance scale (config.solver_tolerance), not a model threshold
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def terminal(game):
    release = (game.pass_index[-1] >= 0) & (game.release[-1] > game.retained)
    return np.where(release, game.release[-1], game.retained)


def dp_held_defender(game, d_cmds):
    """max over the attack's columns every turn, the defender on his held path -- no LP."""
    nodes = [0]
    for k, d in enumerate(d_cmds):
        nodes.append(int(game.defender[k].successor[nodes[k], d]))
    following = terminal(game)
    for k in range(game.config.steps - 1, -1, -1):
        C, R = np.meshgrid(np.arange(game.shapes[k][0]), np.arange(game.shapes[k][1]), indexing="ij")
        flat = np.ravel_multi_index((C.ravel(), R.ravel(), np.full(C.size, nodes[k])), game.shapes[k])
        M = game.matrices(k, flat, following)
        _, am = game.masks(k, flat, DecisionSchedule())
        vals = np.where(am, M[:, d_cmds[k], :], -np.inf).max(axis=1)
        now = np.full(game.shapes[k], np.nan)
        now.flat[flat] = vals
        following = now
    return float(following.flat[0])


def dp_held_attack(game, cols):
    """min over the defender's commands every turn, the attack on its held columns -- no LP."""
    A = len(game.config.directions)
    path = [(0, 0)]
    for k, a in enumerate(cols):
        if a >= A * A:
            break
        c, r = divmod(a, A)
        path.append((int(game.carrier[k].successor[path[k][0], c]), int(game.receiver[k].successor[path[k][1], r])))
    following = terminal(game)
    for k in range(min(len(cols), game.config.steps) - 1, -1, -1):
        D = np.arange(game.shapes[k][2])
        flat = np.ravel_multi_index((np.full(D.size, path[k][0]), np.full(D.size, path[k][1]), D), game.shapes[k])
        M = game.matrices(k, flat, following)
        dm, _ = game.masks(k, flat, DecisionSchedule())
        vals = np.where(dm, M[:, :, cols[k]], np.inf).min(axis=1)
        now = np.full(game.shapes[k], np.nan)
        now.flat[flat] = vals
        following = now
    return float(following.flat[0])


def main() -> None:
    rec = copy.deepcopy(next(r for r in json.loads(STATES.read_text())["states"] if r["provenance"]["code"] == "S05@0.6"))
    for key in ("positions", "velocities"):
        rec["background"][key] = rec["background"][key][:CONFIG.steps + 1]
    game = game_from(rec, MODEL, CONFIG, PHYSICS, "compass", "andrew", pass_reaction_s=0.2, multi_pass=True,
                     background_tackles=True)
    pv = rec["provenance"]
    start = int(pv["onset_frame_id"])
    files = find_bundesliga_files(ROOT / "data/raw/bundesliga-integrated", pv["match_id"])
    frames = load_bundesliga_frames(files["positions"], list(range(start - 10, start + 120)))
    ids = {"carrier": pv["carrier_id"], "receiver": pv["runner_id"], "defender": pv["defender_id"]}
    team = frames[start].players[ids["carrier"]].team_id
    attackers = {pid for pid, p in frames[start].players.items() if p.team_id == team}
    pkg = sc.package("2v1")

    V = solve_multi(game).root_value
    d_cmds, _ = sc.real_commands(game.defender, frames, start, ids["defender"], CONFIG.steps)
    print(f"  (held defender commands {d_cmds})")

    print("1  held defender")
    base = sc.held_masks(game, defender=d_cmds)
    held = solve_multi(game)
    game.masks = base
    S, S_dp = held.root_value, dp_held_defender(game, d_cmds)
    check("solver with the defender masked = the attack's own programme", abs(S - S_dp) <= TOL, f"{S:.9f} vs {S_dp:.9f}")
    plan = sc.plan_of(game, held, d_cmds)

    print("2  held attack")
    cols, passes, _ = sc.real_attack(game, pkg, frames, start, ids, attackers, "2v1", d_cmds)
    print(f"  (real attack columns {cols}, passes {passes})")
    sc.held_masks(game, attack=cols)
    A_ = solve_multi(game).root_value
    game.masks = base
    A_dp = dp_held_attack(game, cols)
    check("solver with the attack masked = the defender's own programme", abs(A_ - A_dp) <= TOL, f"{A_:.9f} vs {A_dp:.9f}")
    sc.held_masks(game, attack=plan)
    R = solve_multi(game).root_value
    game.masks = base
    R_dp = dp_held_attack(game, plan)
    check("the static-best plan, held: solver = programme", abs(R - R_dp) <= TOL, f"{R:.9f} vs {R_dp:.9f}")

    print("3  order")
    check("held defender >= equilibrium >= held attack", S >= V - TOL and V >= A_ - TOL, f"S {S:.4f} V {V:.4f} A {A_:.4f}")
    check("the static-best plan is worth no more once answered", R <= S + TOL, f"R {R:.4f} <= S {S:.4f}")

    print("4  real commands")
    layers = game.defender

    class P:                                          # a stand-in tracking frame: the player exactly on a command end
        def __init__(self, x, y):
            self.x, self.y = x, y

    class F:
        def __init__(self, xy):
            self.players = {"p": P(*xy)}
    ok = True
    for c in range(5):
        end = layers[1].position[int(layers[0].successor[0, c])]
        fake = {start + sc.STEP_FRAMES: F((end[0] - 52.5, end[1] - 34.0))}
        got, _ = sc.real_commands(layers, fake, start, "p", 1)
        ok &= got == [c] or float(np.linalg.norm(np.asarray(layers[1].position[int(layers[0].successor[0, got[0]])]) - end)) <= 1e-9
    check("a player on a command's end is read as that command (or an identical end)", bool(ok))

    print("\nall static-counterfactual checks passed" if not failures else f"\nFAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
