"""The 3v1 (fixed passer) game with every pass candidate its own column, priced after the defender's
command (2026-09-29). See andrew-passer2on1/passer2on1/multi_pass.py for the idea; here the candidates
are each pass to each receiver -- (runner, pass) then (beneficiary, pass), the order of the imported
`pass_index` codes -- played from the passer's real position. Which receivers may be picked (`targets`)
and at which instants the passer may release (`release_steps`) gate legality as in the imported game;
legality is judged when the pass is played, its completion and threat after the defender's command.

reaction_s has no measured source here: 0.2 s is the model's own reaction time
(agile_motion.AGILE.defender_delay_s, the reviewer's choice of 2026-09-25).
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.expected_pass import FAMILIES

from .game import RECEIVERS, FixedPasserGame
from .multi_markov import committed_defence, move_columns
from .payoff import defence, release_payoffs_with_background, target_of, threat_of
from .physics_pass import PhysicsRacePass


class MultiPassFixedGame(FixedPasserGame):

    def __init__(self, scenario, model, config=None, *, reaction_s: float, **kwargs):
        super().__init__(scenario, model, config, **kwargs)
        if reaction_s < 0:
            raise ValueError("reaction_s must be >= 0")
        cfg, sc = self.config, self.scenario
        actions, n_pass = len(cfg.directions), len(cfg.passes)
        self.reaction_s = float(reaction_s)
        self.candidates = [(who, choice) for who in RECEIVERS for choice in cfg.passes]
        self.attack_columns = actions ** 2 + len(self.candidates)
        # A race pass model (A / A-sym) times everyone from the release: the committed defender then
        # races from where his command left him with no reaction of his own (physics_pass committed_s),
        # the others as fitted. A positions-only model sees the defence reaction_s on instead.
        self.timed = isinstance(model, PhysicsRacePass) and self.reaction_s > 0
        self.pass_values, self.pass_legal = [], []
        for k in range(cfg.steps):
            pos, vel, bg, bgv = committed_defence(self, k, self.reaction_s, kwargs.get("physics"))
            if self.timed:        # a race model moves everyone else from the release itself
                bg = self.background[k]
            values = np.zeros(self.shapes[k] + (actions, len(self.candidates)))
            legal = np.zeros(self.shapes[k] + (len(self.candidates),), dtype=bool)
            al, bl, dl = self.carrier[k], self.receiver[k], self.defender[k]
            for flat in self.chunks(k):
                a, b, d = self.indices(k, flat)
                n = len(flat)
                ball = np.broadcast_to(self.passer[k], (n, 2))
                ball_velocity = np.broadcast_to(self.passer_velocity[k], (n, 2))
                oks = []
                for who, layer, index in (("runner", al, a), ("beneficiary", bl, b)):
                    _, ok = release_payoffs_with_background(
                        sc, cfg, self.model, ball, layer.position[index], dl.position[d],
                        ball_velocity, layer.velocity[index], dl.velocity[d],
                        self.background[k], self.background_velocity[k], return_legal=True, threat=self.threat)
                    if who not in self.targets or not self.release_steps[k]:
                        ok = np.zeros_like(ok, dtype=bool)
                    oks.append(np.broadcast_to(ok, (n, n_pass)))
                legal.reshape(-1, len(self.candidates))[flat] = np.concatenate(oks, axis=-1)
                for m in range(actions):
                    values.reshape(-1, actions, len(self.candidates))[flat, m] = np.concatenate(
                        [self._priced(ball, layer.position[index], ball_velocity, layer.velocity[index],
                                      pos[d, m], vel[d, m], bg, bgv)
                         for layer, index in ((al, a), (bl, b))], axis=-1)
            self.pass_values.append(values)
            self.pass_legal.append(legal)

    def _priced(self, ball, receiver, ball_velocity, receiver_velocity, defender, defender_velocity,
                background, background_velocity):
        """completion x threat of every pass to this receiver [n, passes], the defence given, legality aside."""
        cfg, sc = self.config, self.scenario
        defenders, velocities = defence(defender, defender_velocity, background, background_velocity)
        out = np.empty((len(ball), len(cfg.passes)))
        for j, choice in enumerate(cfg.passes):
            target = target_of(choice, receiver, sc.attack_direction, receiver_velocity, sc)
            extra = {"committed_s": self.reaction_s} if self.timed else {}
            completion = np.asarray(self.model.predict(
                ball, receiver, defenders, ball_velocity, receiver_velocity, velocities, target,
                FAMILIES.index(choice.family), sc.attack_direction, sc.pitch_length, sc.pitch_width, **extra))
            value = threat_of(self.threat, target, ball, defenders, velocities, receiver, receiver_velocity, sc)
            out[:, j] = completion * value
        return out

    def matrices(self, step, flat, following, release_values=None):
        if release_values is not None:
            raise NotImplementedError("policy transfer is not built for many pass columns")
        actions = len(self.config.directions)
        matrix = np.empty((len(flat), actions, self.attack_columns))
        matrix[..., :actions ** 2] = move_columns(self, step, flat, following)
        matrix[..., actions ** 2:] = self.pass_values[step].reshape(-1, actions, len(self.candidates))[flat]
        return matrix

    def masks(self, step, flat, schedule):
        defender, attack = super().masks(step, flat, schedule)
        revise = step in schedule.instants("carrier", self.config.steps)
        passes = self.pass_legal[step].reshape(-1, len(self.candidates))[flat] & revise
        return defender, np.c_[attack[:, :-1], passes]
