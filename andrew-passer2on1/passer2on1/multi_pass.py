"""The 2v1 game with every pass candidate its own column, priced after the defender's command (2026-09-29).

Background2on1Game with the attack's release column replaced by one column per pass candidate
(config.passes: the meeting solver's axis offsets x ground / driven / lofted). A pass played at decision
instant k is priced -- completion x threat, the imported models -- against the defence where each
defender command has taken it `reaction_s` later (multi_markov.committed_defence): the defender commits
to a move, the ball is played, and he reacts to it only after that long. Whether a pass is legal -- on
the pitch, not offside -- is judged at k, when it is played, exactly as the imported pricing judges it.
The horizon's release (no defender decision after it) stays the imported best pass.

With reaction_s = 0 every pass column is the same against every defender command, the best one dominates
and the game is the imported one (tests/test_multi_pass.py). With reaction_s > 0 passes into different
spots answer different defender commands, so the attack can mix them -- and mix passing with moving.

reaction_s has no measured source here: 0.2 s is the model's own reaction time
(agile_motion.AGILE.defender_delay_s, the reviewer's choice of 2026-09-25).
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.expected_pass import FAMILIES

from .game import Background2on1Game
from .multi_markov import committed_defence, move_columns
from .payoff import defence, release_payoffs_with_background, target_of, threat_of
from .physics_pass import PhysicsRacePass


class MultiPassGame(Background2on1Game):

    def __init__(self, scenario, model, config=None, *, reaction_s: float, **kwargs):
        super().__init__(scenario, model, config, **kwargs)
        if reaction_s < 0:
            raise ValueError("reaction_s must be >= 0")
        cfg, sc = self.config, self.scenario
        actions, n_pass = len(cfg.directions), len(cfg.passes)
        self.reaction_s = float(reaction_s)
        self.candidates = [("runner", choice) for choice in cfg.passes]
        self.attack_columns = actions ** 2 + n_pass
        # A race pass model (A / A-sym) times everyone from the release: the committed defender then
        # races from where his command left him with no reaction of his own (physics_pass committed_s),
        # the others as fitted. A positions-only model sees the defence reaction_s on instead.
        self.timed = isinstance(model, PhysicsRacePass) and self.reaction_s > 0
        self.pass_values, self.pass_legal = [], []
        for k in range(cfg.steps):
            pos, vel, bg, bgv = committed_defence(self, k, self.reaction_s, kwargs.get("physics"))
            if self.timed:        # a race model moves everyone else from the release itself
                bg = self.background[k]
            values = np.zeros(self.shapes[k] + (actions, n_pass))
            legal = np.zeros(self.shapes[k] + (n_pass,), dtype=bool)
            cl, rl, dl = self.carrier[k], self.receiver[k], self.defender[k]
            for flat in self.chunks(k):
                c, r, d = self.indices(k, flat)
                _, ok = release_payoffs_with_background(
                    sc, cfg, self.model, cl.position[c], rl.position[r], dl.position[d],
                    cl.velocity[c], rl.velocity[r], dl.velocity[d],
                    self.background[k], self.background_velocity[k], return_legal=True, threat=self.threat)
                legal.reshape(-1, n_pass)[flat] = ok
                for m in range(actions):
                    values.reshape(-1, actions, n_pass)[flat, m] = self._priced(
                        cl.position[c], rl.position[r], cl.velocity[c], rl.velocity[r],
                        pos[d, m], vel[d, m], bg, bgv)
            self.pass_values.append(values)
            self.pass_legal.append(legal)

    def _priced(self, ball, receiver, ball_velocity, receiver_velocity, defender, defender_velocity,
                background, background_velocity):
        """completion x threat of every candidate [n, passes], the defence given, legality aside."""
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
        actions, n_pass = len(self.config.directions), len(self.candidates)
        matrix = np.empty((len(flat), actions, self.attack_columns))
        matrix[..., :actions ** 2] = move_columns(self, step, flat, following)
        matrix[..., actions ** 2:] = self.pass_values[step].reshape(-1, actions, n_pass)[flat]
        return matrix

    def masks(self, step, flat, schedule):
        defender, attack = super().masks(step, flat, schedule)
        revise = step in schedule.instants("carrier", self.config.steps)
        passes = self.pass_legal[step].reshape(-1, len(self.candidates))[flat] & revise
        return defender, np.c_[attack[:, :-1], passes]
