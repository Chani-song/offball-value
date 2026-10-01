#!/usr/bin/env python3
"""Static counterfactuals held for the whole window (2026-09-30, the user's design).

A static counterfactual holds the other side to what it really did and lets you do your best against that. The
evaluation's first version held the opponent only for the first 0.6 s and let both sides play the equilibrium
afterwards, which already gave the held side its best answers from 0.6 s on. Here the held side follows its
real movement at EVERY turn of the game (0 / 0.6 / 1.2 s), and the free side best-responds knowing it.

Each evaluation moment's game is built exactly as the evaluation run (jobs/solve_evaluation.sbatch) built it (same states, manifest settings,
pass model router, physics file) and then solved four times with the solver's own LP (solve_multi); holding a side
= leaving it one legal command per turn (its masks), so nothing else of the game changes:
  V  no constraint: the equilibrium value (must equal the evaluation run's saved value -- a check)
  S  the defender held to his real commands at every turn; the attack best-responds  (S >= V)
  R  the attack held to its static-best plan (the best response behind S, read off that solve along the held
     defender's path); the defender best-responds to it  (R <= S: how much of S survives a responsive defender)
  A  the attack held to what it really did (both attackers' real commands; a real pass at the turn it was played,
     as the candidate nearest where the ball was received); the defender best-responds  (A <= V)

Real commands: at every turn the command whose 0.6 s end (the solver's own movement, from the node the held player
has reached) is nearest the player's real position 0.6 s later -- the evaluation's rule, applied turn after turn.
A real pass (the passer on the ball at the turn's start, then the ball faster than TOP_SPEED, as build_eval_states)
ends the real attack; its candidate is the one whose target is nearest where the ball first slows to a player's
speed again (the reception), or where it is 1.2 s later if it never does.

Usage (one package on PYTHONPATH, like run.py; src and scripts for the tracking helpers):
    PYTHONPATH=andrew-passer2on1:src:scripts python scripts/static_counterfactual.py 2v1 \\
        --solved out/runs/eval_v1_2v1 --output data/processed/eval_v1/static_full --workers N
Then: python scripts/summarize_static.py data/processed/eval_v1/static_full
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FPS = 25
STEP_FRAMES = 15                   # 0.6 s
HALF_FRAMES = 4                    # ball speed over +-0.16 s (build_eval_states)
TOP_SPEED = 9.0                    # m/s, the states files' top player speed (build_eval_states)
RECEPTION_WINDOW = 60              # frames (2.4 s) to look for the reception after a pass starts
DATA = Path(os.environ.get("OFFBALL_DATA_ROOT", Path(__file__).resolve().parents[1]))   # the repository root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("2v1", "3v1"))
    p.add_argument("--solved", type=Path, required=True, help="the evaluation run (manifest, starting states, states)")
    p.add_argument("--indices", default=None)
    p.add_argument("--raw-dir", type=Path, default=DATA / "data/raw/bundesliga-integrated")
    p.add_argument("--physics-file", type=Path, default=DATA / "data/processed/physics_limits/agile_p999_nodelay.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def package(kind):
    name = "passer2on1" if kind == "2v1" else "fixedpasser"
    return {m: importlib.import_module(f"{name}.{m}") for m in ("solve", "multi_markov", "physics_pass", "payoff",
                                                                "run_passes")}


def pass_model_path(manifest, record, solved_model_dir):
    router = manifest["pass_model_router"]
    if not router:
        raise SystemExit("the evaluation run has no pass-model router in its manifest")
    name = router["by_match"].get(record["provenance"]["match_id"], router["default"])
    return solved_model_dir / name


# ---- tracking -------------------------------------------------------------------------------------------------

def ball_speed(frames, f):
    a, b = frames.get(f - HALF_FRAMES), frames.get(f + HALF_FRAMES)
    if a is None or b is None or a.ball is None or b.ball is None:
        return math.nan
    return math.hypot(b.ball.x - a.ball.x, b.ball.y - a.ball.y) / (2 * HALF_FRAMES / FPS)


def holder(frames, f, attackers):
    fr = frames.get(f)
    if fr is None or fr.ball is None or not ball_speed(frames, f) <= TOP_SPEED:
        return None
    return min((math.hypot(p.x - fr.ball.x, p.y - fr.ball.y), pid) for pid, p in fr.players.items() if pid in attackers)[1]


def real_pass(frames, start, attackers, passer_ok):
    """(pass seen in (start, start + 0.6 s], the ball's reception spot in corner coordinates)."""
    from build_stage3_states import corner
    if not passer_ok:
        return False, None
    kick = next((f for f in range(start + 1, start + STEP_FRAMES + 1) if ball_speed(frames, f) > TOP_SPEED), None)
    if kick is None:
        return False, None
    land = next((f for f in range(kick + 1, kick + RECEPTION_WINDOW) if ball_speed(frames, f) <= TOP_SPEED), None)
    fr = frames.get(land if land is not None else min(kick + 30, kick + RECEPTION_WINDOW - 1))
    return True, (None if fr is None or fr.ball is None else np.array(corner((fr.ball.x, fr.ball.y))))


# ---- the held sides --------------------------------------------------------------------------------------------

def real_commands(layers, frames, start, pid, steps):
    """Per turn: the command whose end is nearest the real position 0.6 s later, from the node reached so far."""
    from build_stage3_states import corner
    node, cmds, gaps = 0, [], []
    for k in range(steps):
        fr = frames.get(start + STEP_FRAMES * (k + 1))
        real = np.array(corner((fr.players[pid].x, fr.players[pid].y)))
        ends = [layers[k + 1].position[int(layers[k].successor[node, c])] for c in range(layers[k].successor.shape[1])]
        d = [float(np.linalg.norm(real - np.asarray(e))) for e in ends]
        c = int(np.argmin(d))
        cmds.append(c)
        gaps.append(round(sorted(d)[0], 3))
        node = int(layers[k].successor[node, c])
    return cmds, gaps


def held_masks(game, defender=None, attack=None):
    """Replace game.masks so the held side has one legal command per turn (its real one), where it is legal."""
    base = game.masks

    def masks(step, flat, schedule):
        dm, am = base(step, flat, schedule)
        dm, am = np.array(dm, copy=True), np.array(am, copy=True)
        if defender is not None and step < len(defender):
            keep = np.zeros_like(dm)
            keep[:, defender[step]] = dm[:, defender[step]]
            ok = keep.any(axis=1)
            dm[ok] = keep[ok]
        if attack is not None and step < len(attack) and attack[step] is not None:
            keep = np.zeros_like(am)
            keep[:, attack[step]] = am[:, attack[step]]
            ok = keep.any(axis=1)
            am[ok] = keep[ok]
        return dm, am
    game.masks = masks
    return base


def plan_of(game, solution, defender_cmds):
    """The attack's static-best plan: its likeliest column at each turn along the held defender's path."""
    actions = len(game.config.directions)
    moves_n = actions ** 2
    index, plan = (0, 0, 0), []
    for k in range(game.config.steps):
        a = int(np.asarray(solution.attack_policy[k][index]).argmax())
        plan.append(a)
        if a >= moves_n:
            break
        c, r = divmod(a, actions)
        index = (int(game.carrier[k].successor[index[0], c]), int(game.receiver[k].successor[index[1], r]),
                 int(game.defender[k].successor[index[2], defender_cmds[k]]))
    return plan


def real_attack(game, pkg, frames, start, ids, attackers, kind, defender_cmds):
    """The real attack per turn: joint move columns, or the real pass (then nothing after it). A pass's legality
    is read where the defender really was (his held path)."""
    steps, actions = game.config.steps, len(game.config.directions)
    c_cmds, c_gap = real_commands(game.carrier, frames, start, ids["carrier"], steps)
    r_cmds, r_gap = real_commands(game.receiver, frames, start, ids["receiver"], steps)
    cols, passes, node_c, node_r, node_d = [], [], 0, 0, 0
    sc = game.scenario
    for k in range(steps):
        f = start + STEP_FRAMES * k
        if kind == "2v1":
            passer_ok = holder(frames, f, attackers) == ids["carrier"]
        else:
            h = holder(frames, f, attackers)
            passer_ok = bool(game.release_steps[k]) and h not in (ids["carrier"], ids["receiver"])
        passed, spot = real_pass(frames, f, attackers, passer_ok)
        if passed and spot is not None:
            flat = int(np.ravel_multi_index((node_c, node_r, node_d), game.shapes[k]))
            legal = np.asarray(game.pass_legal[k]).reshape(-1, len(game.candidates))[flat]
            best, where = None, math.inf
            for j, (who, choice) in enumerate(game.candidates):
                if not legal[j]:
                    continue
                if kind == "2v1":                           # the one receiver is the runner, the receiver slot
                    layer, node = game.receiver, node_r
                else:                                         # 3v1: runner = carrier slot, beneficiary = receiver slot
                    layer, node = (game.carrier, node_c) if who == "runner" else (game.receiver, node_r)
                target = np.asarray(pkg["payoff"].target_of(choice, layer[k].position[node][None], sc.attack_direction,
                                                            layer[k].velocity[node][None], sc))[0]
                d = float(np.linalg.norm(target - spot))
                if d < where:
                    best, where = j, d
            if best is not None:
                cols.append(actions ** 2 + best)
                passes.append({"turn": k, "candidate": best, "target_gap_m": round(where, 2)})
                break
        cols.append(c_cmds[k] * actions + r_cmds[k])
        node_c = int(game.carrier[k].successor[node_c, c_cmds[k]])
        node_r = int(game.receiver[k].successor[node_r, r_cmds[k]])
        node_d = int(game.defender[k].successor[node_d, defender_cmds[k]])
    return cols, passes, {"carrier": c_gap, "receiver": r_gap}


# ---- one moment ------------------------------------------------------------------------------------------------

def one(kind, record, manifest, solved, physics_file, raw_dir, output):
    from defensive_positioning.models import GameConfig
    from offball_value.bundesliga import find_bundesliga_files, load_bundesliga_frames
    t0 = time.perf_counter()
    pkg = package(kind)
    physics = json.loads(Path(physics_file).read_text())
    cfg = manifest["config"]
    config = GameConfig(steps=cfg["steps"], step_seconds=cfg["step_seconds"], physics_step=cfg["physics_step"],
                        passes=pkg["run_passes"].passes_named(manifest["passes"]))
    model = pkg["physics_pass"].load_pass_model(pass_model_path(manifest, record, DATA / "data/processed/pass_models_sym"),
                                                True)
    extra = {"background_tackles": bool(manifest.get("background_tackles"))} if kind == "2v1" else {}
    game = pkg["solve"].game_from(record, model, config, physics, manifest["commands"], manifest["threat"],
                                  manifest["pass_reaction_s"], True, **extra)
    built = time.perf_counter()
    solve = pkg["multi_markov"].solve_multi

    pv = record["provenance"]
    ids = ({"carrier": pv["carrier_id"], "receiver": pv["runner_id"], "defender": pv["defender_id"]} if kind == "2v1"
           else {"carrier": pv["runner_id"], "receiver": pv["beneficiary_id"], "defender": pv["defender_id"]})
    start = int(pv["onset_frame_id"])
    files = find_bundesliga_files(Path(raw_dir), pv["match_id"])
    wanted = range(start - HALF_FRAMES, start + STEP_FRAMES * config.steps + RECEPTION_WINDOW + HALF_FRAMES + 1)
    frames = load_bundesliga_frames(files["positions"], list(wanted))
    team = frames[start].players[ids["carrier"]].team_id
    attackers = {pid for pid, p in frames[start].players.items() if p.team_id == team}

    V = solve(game).root_value
    d_cmds, d_gap = real_commands(game.defender, frames, start, ids["defender"], config.steps)
    base = held_masks(game, defender=d_cmds)
    held_def = solve(game)
    S = held_def.root_value
    plan = plan_of(game, held_def, d_cmds)
    game.masks = base
    held_masks(game, attack=plan)
    R = solve(game).root_value
    game.masks = base
    a_cols, a_passes, a_gap = real_attack(game, pkg, frames, start, ids, attackers, kind, d_cmds)
    held_masks(game, attack=a_cols)
    A = solve(game).root_value
    game.masks = base

    saved = json.loads((Path(solved) / "states" / f"state_{record['index']:03d}.json").read_text())["value"]
    row = {"code": pv["code"], "index": record["index"], "kind": kind, "V": V, "V_saved": saved, "S": S, "R": R,
           "A": A, "defender_real": d_cmds, "defender_gap_m": d_gap, "static_plan": plan, "attack_real": a_cols,
           "attack_real_passes": a_passes, "attack_gap_m": a_gap,
           "seconds": {"build": round(built - t0, 1), "total": round(time.perf_counter() - t0, 1)}}
    (Path(output) / f"{pv['code']}.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.solved / "manifest.json").read_text())
    records = json.loads((args.solved / "starting_states.json").read_text())["states"]
    if args.indices:
        wanted = {int(i) for i in args.indices.split(",")}
        records = [r for r in records if r["index"] in wanted]
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(one, args.kind, r, manifest, str(args.solved), str(args.physics_file),
                               str(args.raw_dir), str(args.output)): r["provenance"]["code"] for r in records}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
                print(f"{n}/{len(futures)} {row['code']}: V {row['V']:.4f} (saved {row['V_saved']:.4f}) S {row['S']:.4f} "
                      f"R {row['R']:.4f} A {row['A']:.4f} [{row['seconds']['total']:.0f} s]", flush=True)
            except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
                failed.append(futures[fut])
                print(f"FAILED {futures[fut]}: {type(exc).__name__}: {exc}", flush=True)
    print("failed:", failed)


if __name__ == "__main__":
    main()
