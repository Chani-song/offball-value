"""Passes the defender can read: the pass column priced against where each defender command takes him.

2026-09-29, the user's question -- why is a pass always 0 or 100 % while a dribble is mixed? In the
imported game a release at decision instant k is priced at the bodies' positions at k, so the pass
column of the stage matrix is the same against every defender command. A column that does not depend
on the defender is never worth mixing: the attack passes when the pass is worth more than the best it
can guarantee by moving, and never otherwise.

Here the defender's command at k is what he is doing when the ball is played: he has carried it out
for `reaction_s` before he reacts to the pass, and the pass is priced there -- the controlled defender
at his state `reaction_s` into that command (the agile movement, as the layers move him), the
background defenders carried on at their velocity for `reaction_s`. The passer still plays the pass
that looks best at k (the imported choice, `pass_index`) and its legality -- offside, on the pitch --
is still judged at k, the moment it is played; only its completion and threat see the defence later.
The pass column then differs by defender command, so passing and moving can be mixed.

The horizon's release (no defender decision after it) is priced as before. reaction_s = 0 reproduces
Background2on1Game exactly (tests/test_reactive_pass.py). Compass commands only for now: the relative
commands' anchors are laid out on the decision grid, not on a sub-step.

reaction_s has no measured source here: 0.2 s is the model's own reaction time
(agile_motion.AGILE.defender_delay_s, the reviewer's choice of 2026-09-25).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from defensive_positioning.expected_pass import FAMILIES

from .agile_motion import advance, resolve
from .game import Background2on1Game
from .payoff import defence, target_of, threat_of


class ReactivePassGame(Background2on1Game):

    def __init__(self, scenario, model, config=None, *, reaction_s: float, **kwargs):
        super().__init__(scenario, model, config, **kwargs)
        if self.commands is not None:
            raise NotImplementedError("reactive passes are built for compass commands only")
        physics = resolve(kwargs.get("physics"))
        if physics is None:
            raise ValueError("reactive passes need the agile physics")
        if reaction_s < 0:
            raise ValueError("reaction_s must be >= 0")
        self.reaction_s = float(reaction_s)
        cfg, sc = self.config, self.scenario
        actions = len(cfg.directions)
        sub = dataclasses.replace(cfg, step_seconds=self.reaction_s) if self.reaction_s > 0 else None
        self.release_by_command = []
        for k in range(cfg.steps):
            layer = self.defender[k]
            n = len(layer.position)
            pos = np.repeat(layer.position[:, None, :], actions, axis=1).astype(float)
            vel = np.repeat(layer.velocity[:, None, :], actions, axis=1).astype(float)
            if self.reaction_s > 0:
                for s in range(n):
                    for m in range(actions):
                        p, v, _ = advance(layer.position[s], layer.velocity[s], sc.defender, m, sc, sub, physics)
                        pos[s, m], vel[s, m] = p, v
            bg = self.background[k] + self.reaction_s * self.background_velocity[k]
            bgv = self.background_velocity[k]
            table = np.zeros(self.shapes[k] + (actions,))
            for flat in self.chunks(k):
                c, r, d = self.indices(k, flat)
                chosen = self.pass_index[k].flat[flat]
                cl, rl = self.carrier[k], self.receiver[k]
                for m in range(actions):
                    table.reshape(-1, actions)[flat, m] = self._price(
                        chosen, cl.position[c], rl.position[r], cl.velocity[c], rl.velocity[r],
                        pos[d, m], vel[d, m], bg, bgv)
            self.release_by_command.append(table)

    def _price(self, chosen, carrier, receiver, carrier_velocity, receiver_velocity,
               defender, defender_velocity, background, background_velocity):
        """completion x threat of each state's chosen pass (0 where there is none), the defence given."""
        cfg, sc = self.config, self.scenario
        defenders, velocities = defence(defender, defender_velocity, background, background_velocity)
        out = np.zeros(len(chosen))
        for i in np.unique(chosen[chosen >= 0]):
            rows = np.flatnonzero(chosen == i)
            choice = cfg.passes[int(i)]
            target = target_of(choice, receiver[rows], sc.attack_direction, receiver_velocity[rows], sc)
            completion = np.asarray(self.model.predict(
                carrier[rows], receiver[rows], defenders[rows], carrier_velocity[rows], receiver_velocity[rows],
                velocities[rows], target, FAMILIES.index(choice.family), sc.attack_direction,
                sc.pitch_length, sc.pitch_width))
            value = threat_of(self.threat, target, carrier[rows], defenders[rows], velocities[rows],
                              receiver[rows], receiver_velocity[rows], sc)
            out[rows] = completion * value
        return out

    def matrices(self, step, flat, following, release_values=None):
        matrix = super().matrices(step, flat, following, release_values)
        if release_values is None:
            matrix[..., -1] = self.release_by_command[step].reshape(-1, matrix.shape[1])[flat]
        return matrix
