"""Sample one equilibrium play of the 2v1 game (markov.rollout with run passes).

`markov.rollout` aims a release with `choice.target(receiver, direction)`,
which a RunPass cannot honour: it needs the receiver's velocity (and the goal
for a goalward pass). Same sampler, target through run_passes.target_of.
"""

from __future__ import annotations

import numpy as np

from .run_passes import target_of


def rollout(solution, seed=0):
    rng = np.random.default_rng(seed)
    game, index, records, tackled = solution.game, (0, 0, 0), [], False
    actions = len(game.config.directions)
    for k in range(game.config.steps + 1):
        c, r, d = index
        snapshot = {"time": float(game.config.times[k]),
                    "carrier": game.carrier[k].position[c].tolist(),
                    "receiver": game.receiver[k].position[r].tolist(),
                    "defender": game.defender[k].position[d].tolist(),
                    "carrier_velocity": game.carrier[k].velocity[c].tolist(),
                    "receiver_velocity": game.receiver[k].velocity[r].tolist(),
                    "defender_velocity": game.defender[k].velocity[d].tolist()}
        records.append(snapshot)
        if tackled:
            snapshot["event"] = "tackled_during_previous_interval"
            break
        if k == game.config.steps:
            release = bool(solution.terminal_release[index])
        else:
            attack = int(rng.choice(actions**2 + 1, p=solution.attack_policy[k][index]))
            release = attack == actions**2
        if release:
            choice = game.config.passes[int(game.pass_index[k][index])]
            target = target_of(choice, snapshot["receiver"], game.scenario.attack_direction,
                               snapshot["receiver_velocity"], game.scenario)
            snapshot.update(event="release", target=np.asarray(target).tolist(), family=choice.family)
            break
        if k == game.config.steps:
            snapshot["event"] = "retain"
            break
        defend = int(rng.choice(actions, p=solution.defender_policy[k][index]))
        carry, run = divmod(attack, actions)
        snapshot.update(event="move", commands={"carrier": carry, "receiver": run, "defender": defend})
        survival = float(game.survival[k][c, d, carry, defend])
        snapshot["interval_survival_probability"] = survival
        tackled = rng.random() > survival
        index = (int(game.carrier[k].successor[c, carry]),
                 int(game.receiver[k].successor[r, run]),
                 int(game.defender[k].successor[d, defend]))
    return records
