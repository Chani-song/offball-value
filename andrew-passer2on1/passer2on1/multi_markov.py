"""Many passes in the stage game (2026-09-29): solver, certificate and modal line for games whose attack
has one column per pass candidate, not one column for the best pass.

The imported FiniteGame gives the attack actions**2 + 1 columns: the joint move, and one release -- the
pass that is best at that state. While a pass is priced at the moment it is played, every pass column is
the same against every defender command, the best one dominates the others, and keeping one loses
nothing. Once a pass is priced after the defender has committed to his command (see `committed_defence`),
a pass into one spot can be the right one against one defender command and a pass into another spot
against another: the defender cannot take them all away, and the attack mixes between them -- the
"cover one, the other is free" of the off-ball run, for passes. A game here says how many attack columns
it has (`attack_columns`: the moves, then one column per pass candidate) and builds its own `matrices`
and `masks`; the rest follows markov.solve_markov_game / markov.certificate line for line, with the
column count read from the game.

`committed_defence` is where each defender command has taken the controlled defender `reaction_s` after
the decision instant (the agile movement the layers use), and the background defenders carried on at
their velocity: what a pass played at that instant runs into before the defender reacts to it.

Kept byte-identical in andrew-passer2on1/passer2on1 and andrew-fixedpasser/fixedpasser.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from defensive_positioning.markov import MarkovSolution, _solve_batch
from defensive_positioning.models import DecisionSchedule

from .agile_motion import advance, resolve


def committed_defence(game, k, reaction_s, physics):
    """(position, velocity) [defender states at k, commands, 2] of the controlled defender reaction_s into
    each command, and (position, velocity) [m, 2] of the background carried on for reaction_s."""
    physics = resolve(physics)
    if physics is None:
        raise ValueError("committed defence needs the agile physics")
    if game.commands is not None:
        raise NotImplementedError("compass commands only: relative anchors sit on the decision grid")
    actions = len(game.config.directions)
    layer, sc = game.defender[k], game.scenario
    pos = np.repeat(np.asarray(layer.position, dtype=float)[:, None, :], actions, axis=1)
    vel = np.repeat(np.asarray(layer.velocity, dtype=float)[:, None, :], actions, axis=1)
    if reaction_s > 0:
        sub = dataclasses.replace(game.config, step_seconds=reaction_s)
        for s in range(len(layer.position)):
            for m in range(actions):
                p, v, _ = advance(layer.position[s], layer.velocity[s], sc.defender, m, sc, sub, physics)
                pos[s, m], vel[s, m] = p, v
    background = game.background[k] + reaction_s * game.background_velocity[k]
    return pos, vel, background, game.background_velocity[k]


def solve_multi(game, schedule: DecisionSchedule | None = None):
    """markov.solve_markov_game with `game.attack_columns` attack columns."""
    schedule = schedule or DecisionSchedule()
    steps, tolerance = game.config.steps, game.config.solver_tolerance
    for role in ("carrier", "receiver", "defender"):
        schedule.instants(role, steps)
    value, dp, ap = ([None]*(steps+1) for _ in range(3))
    terminal_release = (game.pass_index[-1] >= 0) & (game.release[-1] > game.retained)
    value[-1] = np.where(terminal_release, game.release[-1], game.retained)
    max_gap, programs = 0.0, 0
    actions, columns = len(game.config.directions), game.attack_columns
    for k in range(steps-1, -1, -1):
        shape = game.shapes[k]
        value[k] = np.empty(shape)
        dp[k], ap[k] = np.zeros(shape+(actions,)), np.zeros(shape+(columns,))
        for flat in game.chunks(k):
            matrices = game.matrices(k, flat, value[k+1])
            rm, cm = game.masks(k, flat, schedule)
            v, p, q, gap, count = _solve_batch(matrices, rm, cm, tolerance)
            value[k].flat[flat] = v
            dp[k].reshape(-1, actions)[flat] = p
            ap[k].reshape(-1, columns)[flat] = q
            max_gap, programs = max(max_gap, gap), programs+count
    return MarkovSolution(game, schedule, value, dp, ap, terminal_release, max_gap, programs)


def certificate_multi(solution):
    """markov.certificate with `game.attack_columns` attack columns."""
    game, schedule = solution.game, solution.schedule
    actions, columns = len(game.config.directions), game.attack_columns
    lower = np.where(solution.terminal_release, game.release[-1], game.retained)
    upper = np.maximum(game.release[-1], game.retained)
    for k in range(game.config.steps-1, -1, -1):
        next_lower, next_upper = np.empty(game.shapes[k]), np.empty(game.shapes[k])
        for flat in game.chunks(k):
            rm, cm = game.masks(k, flat, schedule)
            q = solution.attack_policy[k].reshape(-1, columns)[flat]
            p = solution.defender_policy[k].reshape(-1, actions)[flat]
            if (np.any(p < -1e-12) or np.any(q < -1e-12)
                    or not np.allclose(p.sum(axis=-1), 1, atol=1e-10, rtol=0)
                    or not np.allclose(q.sum(axis=-1), 1, atol=1e-10, rtol=0)
                    or np.any(p[~rm] > 1e-12) or np.any(q[~cm] > 1e-12)):
                raise ValueError("invalid or disallowed policy probability")
            defended = np.einsum("brc,bc->br", game.matrices(k, flat, lower), q)
            attacked = np.einsum("br,brc->bc", p, game.matrices(k, flat, upper))
            next_lower.flat[flat] = np.where(rm, defended, np.inf).min(axis=-1)
            next_upper.flat[flat] = np.where(cm, attacked, -np.inf).max(axis=-1)
        lower, upper = next_lower, next_upper
    low, high = float(lower[0, 0, 0]), float(upper[0, 0, 0])
    return {"lower": low, "upper": high, "gap": high-low}


def modal_line_multi(game, solution):
    """solve.modal_line for many pass columns: the same fields, the release now the sum of the pass
    columns, plus `passes` -- every pass played with some probability: candidate, probability, value."""
    cfg, actions = game.config, len(game.config.directions)
    moves_n = actions ** 2
    index, out = (0, 0, 0), []
    for k in range(cfg.steps):
        flat = int(np.ravel_multi_index(index, game.shapes[k]))
        matrix = game.matrices(k, np.array([flat]), solution.value[k + 1])[0]
        dp = np.asarray(solution.defender_policy[k][index], dtype=float)
        ap = np.asarray(solution.attack_policy[k][index], dtype=float)
        defender_values = matrix @ ap
        attack_values = dp @ matrix
        moves = attack_values[:moves_n].reshape(actions, actions)
        a, d = int(ap.argmax()), int(dp.argmax())
        value = float(solution.value[k][index])
        pure_loss = float(matrix.max(axis=1).min() - value)
        passes = [{"candidate": int(j), "probability": round(float(ap[moves_n + j]), 6),
                   "value": round(float(attack_values[moves_n + j]), 6)}
                  for j in np.flatnonzero(ap[moves_n:] > 1e-9)]
        rec = {"step": k, "time": float(cfg.times[k]), "index": [int(i) for i in index],
               "value": round(value, 6), "defender_pure_loss": round(max(pure_loss, 0.0), 6),
               "defender_policy": [round(float(x), 6) for x in dp],
               "defender_values": [round(float(x), 6) for x in defender_values],
               "carrier_slot_values": [round(float(x), 6) for x in moves.max(axis=1)],
               "receiver_slot_values": [round(float(x), 6) for x in moves.max(axis=0)],
               "release_value": round(float(attack_values[moves_n:].max()), 6),
               "release_probability": round(float(ap[moves_n:].sum()), 6),
               "passes": passes, "chosen_defender": d, "chosen_attack": a}
        out.append(rec)
        if a >= moves_n:
            rec["event"] = "release"
            break
        c, r = divmod(a, actions)
        rec["event"] = "move"
        index = (int(game.carrier[k].successor[index[0], c]),
                 int(game.receiver[k].successor[index[1], r]),
                 int(game.defender[k].successor[index[2], d]))
    return out


def move_columns(game, step, flat, following):
    """[state, defender command, joint move] -- markov.FiniteGame.matrices without its release column."""
    c, r, d = game.indices(step, flat)
    cs, rs, ds = (layers[step].successor[index] for layers, index in
                  ((game.carrier, c), (game.receiver, r), (game.defender, d)))
    onward = following[cs[:, :, None, None], rs[:, None, :, None], ds[:, None, None, :]]
    onward *= game.survival[step][c, d][:, :, None, :]
    actions = len(game.config.directions)
    return onward.transpose(0, 3, 1, 2).reshape(len(flat), actions, actions**2)
