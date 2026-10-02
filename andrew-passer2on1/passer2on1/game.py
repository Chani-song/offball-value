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

`background_tackles` (2026-09-29, the user's call: the other defenders are NPCs -- no decisions, but where
they are matters to a dribble as it already does to a pass): the carrier can also lose the ball to them.
Each background defender walks his real track between decision instants (straight between the two
positions) and threatens the carrier's path with the imported tackle hazard -- the same rate, radius and
softness as the controlled defender's, nothing new; the hazards add, so the survivals multiply. Off by
default, which leaves every game as it was.
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.markov import FiniteGame
from defensive_positioning.models import GameConfig, Scenario
from defensive_positioning.motion import tackle_survival

from .agile_motion import describe, layers_for

from .payoff import release_payoffs_with_background, retention_payoff_with_background


def background_survival(carrier, start, end, config):
    """[carrier state, carry command]: the chance the carrier keeps the ball over one step against the
    background defenders, each walking straight from `start` to `end` [m, 2] -- motion.tackle_survival's
    hazard (config.tackle_rate / radius / softness) summed over them along the carrier's path."""
    steps_n = carrier.paths.shape[2]
    s = np.linspace(0.0, 1.0, steps_n)[:, None, None]
    walk = np.asarray(start, dtype=float)[None] + s * (np.asarray(end, dtype=float) - np.asarray(start, dtype=float))[None]
    gap = np.linalg.norm(carrier.paths[:, :, :, None, :] - walk[None, None], axis=-1)   # [C, cmd, t, m]
    z = (gap - config.tackle_radius) / config.tackle_softness
    hazard = (config.tackle_rate * 0.5 * (1.0 - np.tanh(z / 2.0))).sum(axis=-1)
    return np.exp(-np.trapezoid(hazard, dx=config.physics_step, axis=-1))


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
                 value_function=None, physics=None, commands=None, threat="andrew",
                 background_tackles=False, terminal="pass"):
        if value_function is not None:
            raise NotImplementedError("the background game prices with positional_threat_all")
        # terminal (2026-09-29): what the horizon is worth. "pass" is the imported rule -- the best pass
        # there or keeping the ball, whichever pays more, chosen after the defender's last move is known.
        # "xt": no pass at the horizon, only keeping it (retention_payoff_with_background), so every pass
        # in the game is played inside it, against a defender who chooses at the same time.
        if terminal not in ("pass", "xt"):
            raise ValueError(f"terminal must be 'pass' or 'xt', got {terminal!r}")
        if terminal == "xt" and keep_pass_tables:
            raise NotImplementedError("pass tables at an 'xt' horizon are not built")
        self.terminal = terminal
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
        self.commands, self.threat = commands, threat        # relative command sets by slot, or None
        self.carrier, self.receiver, self.defender = layers_for(scenario, config, physics, commands)
        self.shapes = [tuple(len(layers[k]) for layers in
                             (self.carrier, self.receiver, self.defender))
                       for k in range(config.steps + 1)]
        self.survival = [tackle_survival(self.carrier[k], self.defender[k], config)
                         for k in range(config.steps)]
        self.background_tackles = bool(background_tackles)
        if self.background_tackles and background.shape[1] > 0:
            self.survival = [s * background_survival(self.carrier[k], background[k], background[k + 1], config)
                             [:, None, :, None] for k, s in enumerate(self.survival)]
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
                if k == config.steps and self.terminal == "xt":
                    # no pass at the horizon: the "no legal release" the imported code writes
                    best.flat[flat], chosen.flat[flat] = 0.0, -1
                else:
                    # CHANGED: priced against the defence at instant k, not one defender
                    table, legal = release_payoffs_with_background(
                        scenario, config, model, cl.position[c], rl.position[r], dl.position[d],
                        cl.velocity[c], rl.velocity[r], dl.velocity[d],
                        background[k], background_velocity[k], return_legal=True, threat=threat)
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
                        scenario, cl.position[c], rl.position[r], dl.position[d], background[k],
                        threat=threat, carrier_velocity=cl.velocity[c], defender_velocity=dl.velocity[d],
                        background_velocity=background_velocity[k])
            self.release.append(best)
            self.pass_index.append(chosen)
            if individual is not None:
                self.pass_payoffs.append(individual)
