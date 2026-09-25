"""Sample one equilibrium play of the 3v1 game.

`markov.rollout` cannot be used: it reads a release as `config.passes[k]`
with k up to 17 and aims it at the `receiver` slot, whereas here k runs to 35
and half of those passes go to the runner in the `carrier` slot. This is the
same sampler with the receiver decoded and the bodies named.
"""

from __future__ import annotations

import numpy as np


def rollout(solution, seed=0):
    rng = np.random.default_rng(seed)
    game = solution.game
    index, records = (0, 0, 0), []
    actions = len(game.config.directions)
    steps = game.config.steps
    for k in range(steps + 1):
        a, b, d = index
        snapshot = {"time": float(game.config.times[k]),
                    "passer": game.passer[k].tolist(),
                    "runner": game.carrier[k].position[a].tolist(),
                    "beneficiary": game.receiver[k].position[b].tolist(),
                    "defender": game.defender[k].position[d].tolist(),
                    "passer_velocity": game.passer_velocity[k].tolist(),
                    "runner_velocity": game.carrier[k].velocity[a].tolist(),
                    "beneficiary_velocity": game.receiver[k].velocity[b].tolist(),
                    "defender_velocity": game.defender[k].velocity[d].tolist()}
        records.append(snapshot)
        if k == steps:
            release = bool(solution.terminal_release[index])
        else:
            attack = int(rng.choice(actions**2 + 1, p=solution.attack_policy[k][index]))
            release = attack == actions**2
        if release:
            who, choice = game.pass_target(k, index)
            snapshot.update(event="release", to=who, family=choice.family,
                            target=choice.target(snapshot[who],
                                                 game.scenario.attack_direction).tolist())
            break
        if k == steps:
            snapshot["event"] = "no_pass"     # no legal release worth more than nothing
            break
        defend = int(rng.choice(actions, p=solution.defender_policy[k][index]))
        run, support = divmod(attack, actions)        # carrier slot, receiver slot
        snapshot.update(event="move",
                        commands={"runner": run, "beneficiary": support, "defender": defend})
        index = (int(game.carrier[k].successor[a, run]),
                 int(game.receiver[k].successor[b, support]),
                 int(game.defender[k].successor[d, defend]))
    return records
