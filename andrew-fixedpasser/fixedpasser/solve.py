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

from .game import FixedPasserGame
from .physics_pass import load_pass_model
from .rollout import rollout


def game_from(record, model, config, physics=None):
    steps = config.steps
    passer, bg = record["passer"], record.get("background") or {}
    positions = np.asarray(bg.get("positions", np.zeros((steps + 1, 0, 2))), dtype=float)
    velocities = np.asarray(bg.get("velocities", np.zeros_like(positions)), dtype=float)
    if positions.size == 0:
        positions = np.zeros((steps + 1, 0, 2))
        velocities = np.zeros_like(positions)
    return FixedPasserGame(scenario_from(record["scenario"]), model, config,
                           passer=np.asarray(passer["positions"], dtype=float),
                           passer_velocity=np.asarray(passer["velocities"], dtype=float),
                           background=positions, background_velocity=velocities,
                           physics=physics)


def solve_one(record, model_path, allow_proxy, config, output, physics=None):
    start, cpu = time.perf_counter(), time.process_time()
    model = load_pass_model(model_path, allow_proxy)
    game = game_from(record, model, config, physics)
    built = time.perf_counter()
    solution = solve_markov_game(game)
    solved = time.perf_counter()
    bounds = certificate(solution)
    if bounds["gap"] > 2 * config.steps * config.solver_tolerance or bounds["gap"] < -1e-10:
        raise RuntimeError(f"failed certificate: {bounds}")
    certified = time.perf_counter()
    core_seconds = time.process_time() - cpu
    name = f"state_{record['index']:03d}"
    output = Path(output)
    policy = output / "policies" / f"{name}.npz"
    save_policy(solution, policy)
    restored = load_policy(game, policy)
    restored_bounds = certificate(restored)
    if abs(restored_bounds["gap"] - bounds["gap"]) > 1e-12:
        raise RuntimeError("serialized policy changed the certificate")
    codes = np.concatenate([p.ravel() for p in game.pass_index])
    result = dict(index=record["index"], stratum=record["stratum"],
                  scenario=record["scenario"], value=solution.root_value, physics=game.physics,
                  pass_model=Path(model_path).name,
                  certificate=bounds, serialized_certificate=restored_bounds,
                  max_local_gap=solution.max_local_gap,
                  joint_states_per_instant=[int(np.prod(shape)) for shape in game.shapes],
                  linear_programs=solution.linear_programs,
                  root_attack=solution.attack_policy[0].ravel().tolist(),
                  root_defender=solution.defender_policy[0].ravel().tolist(),
                  mixed_states={role: sum(int(np.count_nonzero((p > 1e-9).sum(-1) > 1))
                                         for p in policies[:-1])
                                for role, policies in (("attack", solution.attack_policy),
                                                       ("defender", solution.defender_policy))},
                  timing=dict(build_seconds=built-start, solve_seconds=solved-built,
                              certificate_seconds=certified-solved, compute_seconds=certified-start,
                              cpu_seconds=core_seconds),
                  worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  policy=f"policies/{name}.npz", policy_bytes=policy.stat().st_size,
                  rollouts=[rollout(solution, seed) for seed in range(4)],
                  background_players=len((record.get("background") or {}).get("ids", [])),
                  best_release_to={"runner": int(((codes >= 0) & (codes < len(game.config.passes))).sum()),
                                   "beneficiary": int((codes >= len(game.config.passes)).sum())})
    result["timing"]["total_seconds_with_io_and_visualization"] = time.perf_counter()-start
    (output / "states" / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
