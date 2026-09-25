"""Three attackers against one defender, the passer scripted.

The imported solver builds its state from three moving bodies -- the
`carrier`, `receiver` and `defender` slots of `markov.FiniteGame` -- and never
asks who is in them. `matrices`, `masks`, `solve_markov_game`, `certificate`
and policy serialisation all run on those three slots. So the 3v1 game is the
imported one with the slots re-occupied:

    carrier slot   the off-ball RUNNER          strategic
    receiver slot  the BENEFICIARY              strategic
    defender slot  the controlled defender      strategic
    (outside)      the PASSER                   scripted: his real track
    (outside)      every other defender         scripted: their real tracks

The attack still has 5 x 5 joint moves (now runner x beneficiary) plus one
release, 26 columns; the state count is the imported one. What changes is in
the constructor only:

  release    at every state, passes from the passer's real position to the
             runner (18 choices) AND to the beneficiary (18 choices), each
             priced by the background-aware evaluator, the best of the 36
             kept. Keeping only the best is exact for the same reason the
             imported code gives: the release precedes the simultaneous move,
             so nothing the defender does next can change its value. Which
             receiver and which pass are recorded in `pass_index` as
             receiver * 18 + choice.
  survival   all ones. Tackling is off: a scripted passer cannot evade, so a
             tackle term would reward the defender for abandoning both
             receivers to camp on the ball -- and the reacting defender is,
             by the pair gate, not the man pressing the ball.
  retained   zero. There is no keeping it: the attack passes within the
             horizon, to whichever receiver it pays to, or scores nothing.

`targets` can remove one receiver's passes; the tests use it to check that
taking an option away never helps the attack.
"""

from __future__ import annotations

import numpy as np

from defensive_positioning.markov import FiniteGame
from defensive_positioning.models import GameConfig, Scenario
from .agile_motion import describe, layers_for

from .payoff import release_payoffs_with_background

RECEIVERS = ("runner", "beneficiary")


class FixedPasserGame(FiniteGame):
    """scenario: `Scenario(carrier=runner, receiver=beneficiary, defender=...)`.
    The slot names are the imported class's; the docstring above maps them.

    passer, passer_velocity:          [steps + 1, 2], corner origin
    background, background_velocity:  [steps + 1, m, 2], corner origin, m >= 0
    targets:                          which receivers the passer may pick
    """

    def __init__(self, scenario: Scenario, model, config: GameConfig | None = None, *,
                 passer, passer_velocity, background=None, background_velocity=None,
                 targets=RECEIVERS, physics=None):
        self.scenario, self.config, self.model = scenario, config or GameConfig(), model
        self.value_function = None
        config = self.config
        steps = config.steps
        passer = np.asarray(passer, dtype=float)
        passer_velocity = np.asarray(passer_velocity, dtype=float)
        if passer.shape != (steps + 1, 2) or passer_velocity.shape != (steps + 1, 2):
            raise ValueError(f"passer track must be [steps + 1 = {steps + 1}, 2]")
        background = (np.zeros((steps + 1, 0, 2)) if background is None
                      else np.asarray(background, dtype=float))
        if background.ndim != 3 or background.shape[0] != steps + 1 or background.shape[2] != 2:
            raise ValueError(f"background must be [steps + 1, m, 2], got {background.shape}")
        background_velocity = (np.zeros_like(background) if background_velocity is None
                               else np.asarray(background_velocity, dtype=float))
        if background_velocity.shape != background.shape:
            raise ValueError("background_velocity must match background")
        for name, array in (("passer", passer), ("passer_velocity", passer_velocity),
                            ("background", background),
                            ("background_velocity", background_velocity)):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        targets = tuple(targets)
        if not targets or any(t not in RECEIVERS for t in targets):
            raise ValueError(f"targets must be a nonempty subset of {RECEIVERS}")
        self.passer, self.passer_velocity = passer, passer_velocity
        self.background, self.background_velocity = background, background_velocity
        self.targets = targets

        actions = len(config.directions)
        upper = sum(actions ** (3*k) for k in range(config.steps + 1))
        if upper > config.max_joint_states:
            raise ValueError(f"up to {upper:,} joint states exceed max_joint_states="
                             f"{config.max_joint_states:,}; reduce steps/actions or explicitly raise it")
        # runner, beneficiary, defender; physics None is the imported build_layers
        self.physics = describe(physics)
        self.carrier, self.receiver, self.defender = layers_for(scenario, config, physics)
        self.shapes = [tuple(len(layers[k]) for layers in
                             (self.carrier, self.receiver, self.defender))
                       for k in range(config.steps + 1)]
        # tackling off; shape as markov.tackle_survival returns it
        self.survival = [np.ones((len(self.carrier[k]), len(self.defender[k]), actions, actions))
                         for k in range(config.steps)]
        passes = len(config.passes)
        self.release, self.pass_index = [], []
        self.pass_payoffs = None
        self.retained = None
        for k, shape in enumerate(self.shapes):
            best = np.empty(shape)
            chosen = np.empty(shape, dtype=np.int32)
            if k == config.steps:
                self.retained = np.zeros(shape)
            for flat in self.chunks(k):
                a, b, d = self.indices(k, flat)
                al, bl, dl = self.carrier[k], self.receiver[k], self.defender[k]
                n = len(flat)
                ball = np.broadcast_to(passer[k], (n, 2))
                ball_velocity = np.broadcast_to(passer_velocity[k], (n, 2))
                tables, legals = [], []
                for who, layer, index in (("runner", al, a), ("beneficiary", bl, b)):
                    table, legal = release_payoffs_with_background(
                        scenario, config, model, ball, layer.position[index], dl.position[d],
                        ball_velocity, layer.velocity[index], dl.velocity[d],
                        background[k], background_velocity[k], return_legal=True)
                    if who not in targets:
                        legal = np.zeros_like(legal, dtype=bool)
                    tables.append(table)
                    legals.append(legal)
                table = np.concatenate(tables, axis=-1)       # [n, 2 * passes]
                legal = np.concatenate(legals, axis=-1)
                scores = np.where(legal, table, -np.inf)
                choice = scores.argmax(axis=-1)
                any_legal = legal.any(axis=-1)
                best.flat[flat] = np.where(any_legal, scores[np.arange(n), choice], 0.0)
                chosen.flat[flat] = np.where(any_legal, choice, -1)
            self.release.append(best)
            self.pass_index.append(chosen)
        self._passes = passes

    def pass_target(self, step, index):
        """(receiver name, PassChoice) of the stored best release at a state."""
        code = int(self.pass_index[step][index])
        if code < 0:
            raise ValueError("no legal release at this state")
        return RECEIVERS[code // self._passes], self.config.passes[code % self._passes]
