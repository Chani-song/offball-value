"""The imported 2v1 game, with background defenders in its terminal payoffs.

`Background2on1Game` subclasses `markov.FiniteGame` and replaces only the
constructor, and within it only the two calls that price a release and a
retention -- and, when `physics` is given, the movement layers
(agile_motion.py; physics None builds them with the imported function). Everything the solver, the certificate, the rollout and policy
serialisation touch -- the three strategic bodies' layers, `shapes`,
`survival`, `matrices`, `masks` -- is built exactly as the original builds it
or inherited unchanged.

Background defenders make no decision. Their position at decision step k is
where they actually were k * step_seconds after the run's onset, so they
enter each step's payoffs as a constant: the state space is the original's.
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.markov import FiniteGame
from defensive_positioning.models import GameConfig, Scenario
from defensive_positioning.motion import tackle_survival

from .agile_motion import describe, layers_for

from .payoff import release_payoffs_with_background, retention_payoff_with_background


class Background2on1Game(FiniteGame):
    """carrier (strategic, may be tackled) + receiver + controlled defender,
    priced against the whole defence.

    background:          [steps + 1, m, 2] positions of the other defenders,
                         corner origin, one row per decision instant. m may be 0.
    background_velocity: same shape. Not read by the positions-only pass model
                         in use; carried so a velocity model would get real values.
    """

    def __init__(self, scenario: Scenario, model, config: GameConfig | None = None, *,
                 background=None, background_velocity=None, keep_pass_tables=False,
                 value_function=None, physics=None):
        if value_function is not None:
            raise NotImplementedError("the background game prices with positional_threat_all")
        self.scenario, self.config, self.model = scenario, config or GameConfig(), model
        self.value_function = None
        config = self.config
        steps = config.steps
        background = (np.zeros((steps + 1, 0, 2)) if background is None
                      else np.asarray(background, dtype=float))
        if background.ndim != 3 or background.shape[0] != steps + 1 or background.shape[2] != 2:
            raise ValueError(f"background must be [steps + 1 = {steps + 1}, m, 2], "
                             f"got {background.shape}")
        background_velocity = (np.zeros_like(background) if background_velocity is None
                               else np.asarray(background_velocity, dtype=float))
        if background_velocity.shape != background.shape:
            raise ValueError("background_velocity must match background")
        if not (np.isfinite(background).all() and np.isfinite(background_velocity).all()):
            raise ValueError("background positions and velocities must be finite")
        self.background, self.background_velocity = background, background_velocity

        # ---- from here: markov.FiniteGame.__init__, unchanged except where marked ----
        actions = len(config.directions)
        upper = sum(actions ** (3*k) for k in range(config.steps + 1))
        if upper > config.max_joint_states:
            raise ValueError(f"up to {upper:,} joint states exceed max_joint_states="
                             f"{config.max_joint_states:,}; reduce steps/actions or explicitly raise it")
        # CHANGED: movement from agile_motion; physics None is the imported build_layers
        self.physics = describe(physics)
        self.carrier, self.receiver, self.defender = layers_for(scenario, config, physics)
        self.shapes = [tuple(len(layers[k]) for layers in
                             (self.carrier, self.receiver, self.defender))
                       for k in range(config.steps + 1)]
        self.survival = [tackle_survival(self.carrier[k], self.defender[k], config)
                         for k in range(config.steps)]
        self.release, self.pass_index = [], []
        self.pass_payoffs = [] if keep_pass_tables else None
        self.retained = None
        for k, shape in enumerate(self.shapes):
            best = np.empty(shape)
            chosen = np.empty(shape, dtype=np.int32)
            individual = np.empty(shape+(len(config.passes),)) if keep_pass_tables else None
            if k == config.steps:
                self.retained = np.empty(shape)
            for flat in self.chunks(k):
                c, r, d = self.indices(k, flat)
                cl, rl, dl = self.carrier[k], self.receiver[k], self.defender[k]
                # CHANGED: priced against the defence at instant k, not one defender
                table, legal = release_payoffs_with_background(
                    scenario, config, model, cl.position[c], rl.position[r], dl.position[d],
                    cl.velocity[c], rl.velocity[r], dl.velocity[d],
                    background[k], background_velocity[k], return_legal=True)
                scores = np.where(legal, table, -np.inf)
                if individual is not None:
                    individual.reshape(-1, len(config.passes))[flat] = table
                choice = scores.argmax(axis=-1)
                any_legal = legal.any(axis=-1)
                best.flat[flat] = np.where(any_legal, scores[np.arange(len(flat)), choice], 0.0)
                chosen.flat[flat] = np.where(any_legal, choice, -1)
                if k == config.steps:
                    # CHANGED: retention threat measured to the nearest defender
                    self.retained.flat[flat] = retention_payoff_with_background(
                        scenario, cl.position[c], rl.position[r], dl.position[d], background[k])
            self.release.append(best)
            self.pass_index.append(chosen)
            if individual is not None:
                self.pass_payoffs.append(individual)
