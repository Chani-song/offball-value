"""`exact_study.solve_one` for the 3v1 game.

Same solve, same certificate threshold, same save-and-reload check, same
record, so everything that reads an exact_study output reads this one. The
game is `FixedPasserGame` built from the record's passer and background
tracks, the rollouts are this package's (they decode which receiver a pass
went to), and no figure or clip is rendered.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from defensive_positioning.equilibrium_clips import scenario_from
from defensive_positioning.exact_study import load_policy, save_policy
from defensive_positioning.markov import certificate, solve_markov_game

from .agile_motion import command_names, relative_commands
from .game import FixedPasserGame
from .multi_markov import certificate_multi, modal_line_multi, solve_multi
from .multi_pass import MultiPassFixedGame
from .physics_pass import load_pass_model
from .rollout import rollout
from .run_passes import RunPass


def game_from(record, model, config, physics=None, commands=None, threat="andrew", pass_reaction_s=None,
              multi_pass=False, terminal="pass"):
    """`commands`: None / "compass" for the imported compass, "relative" for
    agile_motion.relative_commands built from the record. The record may carry
    `release_steps` (steps + 1 booleans, see game.py); without it every instant
    allows a release, as before. `multi_pass`: every pass candidate its own column, priced after
    pass_reaction_s (multi_pass.MultiPassFixedGame, 2026-09-29). `terminal`: the horizon's worth,
    "pass" (imported), "xt" or "none" (no pass there; see game.FixedPasserGame)."""
    cmd = relative_commands(record, config, "3v1") if commands == "relative" else None
    steps = config.steps
    passer, bg = record["passer"], record.get("background") or {}
    positions = np.asarray(bg.get("positions", np.zeros((steps + 1, 0, 2))), dtype=float)
    velocities = np.asarray(bg.get("velocities", np.zeros_like(positions)), dtype=float)
    if positions.size == 0:
        positions = np.zeros((steps + 1, 0, 2))
        velocities = np.zeros_like(positions)
    kwargs = dict(passer=np.asarray(passer["positions"], dtype=float),
                  passer_velocity=np.asarray(passer["velocities"], dtype=float),
                  background=positions, background_velocity=velocities,
                  physics=physics, commands=cmd, threat=threat, release_steps=record.get("release_steps"),
                  terminal=terminal)
    if multi_pass:
        return MultiPassFixedGame(scenario_from(record["scenario"]), model, config,
                                  reaction_s=pass_reaction_s or 0.0, **kwargs)
    if pass_reaction_s is not None:
        raise NotImplementedError("the one-pass reactive game is 2v1 only; use multi_pass")
    return FixedPasserGame(scenario_from(record["scenario"]), model, config, **kwargs)


def candidate_name(who, choice):
    """'runner:ground:[4.0, 0.0]' for an axis pass, 'runner:ground:along 8 lateral 0 goalward 0' for a run pass."""
    where = (list(choice.offset) if hasattr(choice, "offset")
             else f"along {choice.along:g} lateral {choice.lateral:g} goalward {choice.goalward:g}")
    return f"{who}:{choice.family}:{where}"


def root_game(game, solution, multi):
    """The opening decision's payoff table as the solver saw it (2026-09-29, for the evaluation metrics):
    [defender command, attack column] with the continuation already solved, which commands and columns were
    legal, and each column's name -- joint moves "move <carrier/runner cmd> <receiver/beneficiary cmd>" in the
    policy's order (carrier-major), then the passes (multi-pass) or the one best-pass "release" column."""
    flat = np.array([0])
    matrix = np.asarray(game.matrices(0, flat, solution.value[1]))[0]
    rows, cols = (np.asarray(m)[0] for m in game.masks(0, flat, solution.schedule))
    actions = len(game.config.directions)
    names = [f"move {c} {r}" for c in range(actions) for r in range(actions)]
    names += ([candidate_name(who, choice) for who, choice in game.candidates] if multi else ["release"])
    return dict(matrix=[[float(x) if np.isfinite(x) else None for x in row] for row in matrix],
                defender_legal=[bool(x) for x in rows], attack_legal=[bool(x) for x in cols], columns=names)


def solve_one(record, model_path, allow_proxy, config, output, physics=None, commands=None,
              threat="andrew", pass_reaction_s=None, multi_pass=False, terminal="pass"):
    start, cpu = time.perf_counter(), time.process_time()
    model = load_pass_model(model_path, allow_proxy)
    game = game_from(record, model, config, physics, commands, threat, pass_reaction_s, multi_pass, terminal)
    multi = isinstance(game, MultiPassFixedGame)
    solve, certify = (solve_multi, certificate_multi) if multi else (solve_markov_game, certificate)
    built = time.perf_counter()
    solution = solve(game)
    solved = time.perf_counter()
    bounds = certify(solution)
    if bounds["gap"] > 2 * config.steps * config.solver_tolerance or bounds["gap"] < -1e-10:
        raise RuntimeError(f"failed certificate: {bounds}")
    certified = time.perf_counter()
    core_seconds = time.process_time() - cpu
    name = f"state_{record['index']:03d}"
    output = Path(output)
    policy = output / "policies" / f"{name}.npz"
    save_policy(solution, policy)
    restored = load_policy(game, policy)
    restored_bounds = certify(restored)
    if abs(restored_bounds["gap"] - bounds["gap"]) > 1e-12:
        raise RuntimeError("serialized policy changed the certificate")
    codes = np.concatenate([p.ravel() for p in game.pass_index])
    result = dict(index=record["index"], stratum=record["stratum"],
                  scenario=record["scenario"], value=solution.root_value, physics=game.physics,
                  pass_model=Path(model_path).name, threat=game.threat,
                  release_steps=list(game.release_steps), pass_reaction_s=pass_reaction_s, multi_pass=multi,
                  terminal=game.terminal,
                  pass_candidates=[candidate_name(who, c) for who, c in game.candidates] if multi else None,
                  commands=command_names(game.commands) or "compass",
                  passes="run" if isinstance(config.passes[0], RunPass) else "andrew",
                  modal_line=(modal_line_multi if multi else modal_line)(game, solution),
                  certificate=bounds, serialized_certificate=restored_bounds,
                  max_local_gap=solution.max_local_gap,
                  joint_states_per_instant=[int(np.prod(shape)) for shape in game.shapes],
                  linear_programs=solution.linear_programs,
                  root_attack=solution.attack_policy[0].ravel().tolist(),
                  root_defender=solution.defender_policy[0].ravel().tolist(),
                  root_game=root_game(game, solution, multi),
                  mixed_states={role: sum(int(np.count_nonzero((p > 1e-9).sum(-1) > 1))
                                         for p in policies[:-1])
                                for role, policies in (("attack", solution.attack_policy),
                                                       ("defender", solution.defender_policy))},
                  timing=dict(build_seconds=built-start, solve_seconds=solved-built,
                              certificate_seconds=certified-solved, compute_seconds=certified-start,
                              cpu_seconds=core_seconds),
                  worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  policy=f"policies/{name}.npz", policy_bytes=policy.stat().st_size,
                  rollouts=[] if multi else [rollout(solution, seed) for seed in range(4)],
                  background_players=len((record.get("background") or {}).get("ids", [])),
                  best_release_to={"runner": int(((codes >= 0) & (codes < len(game.config.passes))).sum()),
                                   "beneficiary": int((codes >= len(game.config.passes)).sum())})
    result["timing"]["total_seconds_with_io_and_visualization"] = time.perf_counter()-start
    (output / "states" / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def modal_line(game, solution):
    """Both sides' likeliest actions from the root, with each side's action
    values at every decision visited (2026-09-27, so a mixed defender can be
    told apart from an indifferent one).

    At a visited state the payoff matrix is [defender command, attack column]
    (attack's payoff; the defender minimises). `defender_values[a]` is that
    payoff when the defender plays a against the attack's mixed policy;
    `carrier_slot_values[c]` / `receiver_slot_values[r]` are the best an
    attacker's command can do against the defender's mix, maximised over the
    other attacker; `release_value` is the pass column. A body whose values
    are all equal has nothing at stake at that decision.
    """
    cfg, actions = game.config, len(game.config.directions)
    index, out = (0, 0, 0), []
    for k in range(cfg.steps):
        flat = int(np.ravel_multi_index(index, game.shapes[k]))
        matrix = game.matrices(k, np.array([flat]), solution.value[k + 1])[0]
        dp = np.asarray(solution.defender_policy[k][index], dtype=float)
        ap = np.asarray(solution.attack_policy[k][index], dtype=float)
        defender_values = matrix @ ap
        attack_values = dp @ matrix
        moves = attack_values[:-1].reshape(actions, actions)
        a, d = int(ap.argmax()), int(dp.argmax())
        value = float(solution.value[k][index])
        # Every command in a mixed defender's support is worth the same against
        # the attack's mix (that is what makes it an equilibrium), so equal
        # values do not separate a dilemma from indifference. What does: the
        # attack's best reply to each PURE command. pure_loss = the least a
        # defender who must commit to one command gives up against a replying
        # attack, minus the mixed value. A true dilemma has it > 0; a defender
        # whose commands all lead to the same payoff has it = 0.
        pure_loss = float(matrix.max(axis=1).min() - value)
        rec = {"step": k, "time": float(cfg.times[k]), "index": [int(i) for i in index],
               "value": round(value, 6), "defender_pure_loss": round(max(pure_loss, 0.0), 6),
               "defender_policy": [round(float(x), 6) for x in dp],
               "defender_values": [round(float(x), 6) for x in defender_values],
               "carrier_slot_values": [round(float(x), 6) for x in moves.max(axis=1)],
               "receiver_slot_values": [round(float(x), 6) for x in moves.max(axis=0)],
               "release_value": round(float(attack_values[-1]), 6) if ap.size == actions**2 + 1 else None,
               "release_probability": round(float(ap[-1]), 6),
               "chosen_defender": d, "chosen_attack": a}
        out.append(rec)
        if a == actions**2:
            rec["event"] = "release"
            break
        c, r = divmod(a, actions)
        rec["event"] = "move"
        index = (int(game.carrier[k].successor[index[0], c]),
                 int(game.receiver[k].successor[index[1], r]),
                 int(game.defender[k].successor[index[2], d]))
    return out
