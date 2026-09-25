"""Junhyun's value stack, adapted to this pipeline's frames and coordinates.

Stage 3 (the 2v1 equilibrium solver) prices possession with its own expected
threat and its own completion model. Stage 2 priced it with a static EPV grid
and a separately fitted delivery model, so the ranking that chose a scene and
the equilibrium that then solved it disagreed about what a position is worth.
This module is the other side of that seam: the same three pieces, callable
from here.

    threat_at_point   defensive_positioning.value.positional_threat
    PassCompletion    defensive_positioning.expected_pass.ExpectedPass
    carry_survival    defensive_positioning.motion.tackle_survival's hazard

TAKE ALL THREE OR NONE. Swapping the delivery model alone has already been
measured here: pass options rose 1.9x while carry options stayed at 1.0x, and
the two action classes stopped being comparable. The pieces are coherent with
each other, not with ours -- `config.value_stack` switches them together for
that reason, and `q_factors` folds the accessibility term away because
`positional_threat` already carries a defender-room term of its own.

COORDINATES. This repo puts the origin on the centre spot, x in [-52.5, 52.5]
and y in [-34, 34]. The imported code puts it on a corner, x in [0, 105] and
y in [0, 68]. Every entry point here takes ours and converts.

NOT CALIBRATED, AND THAT IS THE AUTHOR'S OWN WORDING. `value.py` says the
threat "is deliberately uncalibrated" and "should not be interpreted as an
observed scoring probability"; the pass model ships
`validation_status: experimental_proxy` with inferred receiver labels; the
three tackle constants are hand-set defaults. Adopting this buys one scale
across stages 2 and 3. It does not buy calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .bundesliga import FIELD_LENGTH, FIELD_WIDTH, BundesligaFrame

# Hand-set defaults from defensive_positioning.models.GameConfig. Named here so
# a sweep can move them without reaching into the other package's dataclass.
TACKLE_RATE = 2.2
TACKLE_RADIUS_M = 1.0
TACKLE_SOFTNESS_M = 0.45

GROUND, DRIVEN, LOFTED = 0, 1, 2


def _to_corner_origin(points: np.ndarray) -> np.ndarray:
    """Centre-spot origin -> corner origin, the imported code's convention."""
    points = np.asarray(points, dtype=float)
    return points + np.array([FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0])


@lru_cache(maxsize=4)
def _scenario(attacking_direction: int):
    """A Scenario carries only the pitch and the direction for these calls.

    The three players it also holds are required by its constructor and unused
    by `positional_threat`, which takes its points as arguments. They are put
    at the centre spot so the constructor's inside-the-pitch check passes.
    """
    from defensive_positioning.models import PlayerState, Scenario

    centre = PlayerState(position=(FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0),
                         velocity=(0.0, 0.0))
    return Scenario(carrier=centre, receiver=centre, defender=centre,
                    pitch_length=FIELD_LENGTH, pitch_width=FIELD_WIDTH,
                    attack_direction=int(attacking_direction))


@lru_cache(maxsize=4)
def _pass_model(path: str):
    from defensive_positioning.expected_pass import ExpectedPass

    return ExpectedPass.load(path, allow_proxy=True)


def threat_at_point(
    point: tuple[float, float],
    support_xy: tuple[float, float],
    defender_xy: tuple[float, float],
    attacking_direction: int,
) -> float:
    """Positional threat of arriving at `point`, in [0.2, 1].

    `support_xy` is the passer for a proposed reception and the receiver for
    retained possession, which is the imported module's own convention.
    """
    from defensive_positioning.value import positional_threat

    ball, support, defender = (
        _to_corner_origin(np.asarray(v, dtype=float))
        for v in (point, support_xy, defender_xy)
    )
    return float(positional_threat(ball, support, defender,
                                   _scenario(int(attacking_direction))))


def completion_probability(
    model_path: str,
    carrier_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    target_xy: tuple[float, float],
    defender_xys: Sequence[tuple[float, float]],
    attacking_direction: int,
    family: int = GROUND,
) -> float:
    """P(completion | kick-time positions, intended target, launch family).

    The model picks its own two defender roles -- nearest to the passer and
    nearest to the receiver -- out of whatever defenders it is handed, so pass
    the whole defending side rather than pre-selecting.
    """
    defenders = np.asarray(list(defender_xys), dtype=float)
    if defenders.ndim != 2 or defenders.shape[0] == 0:
        raise ValueError("supply at least one defender position")
    value = _pass_model(str(model_path)).predict(
        carrier=_to_corner_origin(np.asarray(carrier_xy, dtype=float)),
        receiver=_to_corner_origin(np.asarray(receiver_xy, dtype=float)),
        defenders=_to_corner_origin(defenders),
        carrier_velocity=None,
        receiver_velocity=None,
        defender_velocities=None,
        target=_to_corner_origin(np.asarray(target_xy, dtype=float)),
        family=int(family),
        attack_direction=int(attacking_direction),
    )
    return float(np.clip(value, 0.0, 1.0))


def carry_survival(
    carrier_path_txy: Sequence[tuple[float, float, float]],
    defender_path_txy: Sequence[tuple[float, float, float]],
    *,
    tackle_rate: float = TACKLE_RATE,
    tackle_radius_m: float = TACKLE_RADIUS_M,
    tackle_softness_m: float = TACKLE_SOFTNESS_M,
) -> float:
    """P(still has the ball) after carrying this path against this defender.

    The hazard of `motion.tackle_survival`: close is dangerous, and staying
    close is dangerous for as long as it lasts, because the hazard integrates
    over the path rather than being read off the final gap.

    Both paths are (t, x, y). They are resampled onto the union of their own
    time stamps, so the two need not be sampled alike, and the integral is
    taken on that clock.
    """
    carrier = np.asarray(list(carrier_path_txy), dtype=float)
    defender = np.asarray(list(defender_path_txy), dtype=float)
    if carrier.ndim != 2 or carrier.shape[0] < 2 or carrier.shape[1] != 3:
        return 1.0
    if defender.ndim != 2 or defender.shape[0] < 1 or defender.shape[1] != 3:
        return 1.0
    start = max(carrier[0, 0], defender[0, 0])
    stop = min(carrier[-1, 0], defender[-1, 0])
    if not np.isfinite([start, stop]).all() or stop <= start:
        return 1.0
    times = np.unique(np.concatenate([carrier[:, 0], defender[:, 0]]))
    times = times[(times >= start) & (times <= stop)]
    if times.size < 2:
        return 1.0
    gap = np.hypot(
        np.interp(times, carrier[:, 0], carrier[:, 1])
        - np.interp(times, defender[:, 0], defender[:, 1]),
        np.interp(times, carrier[:, 0], carrier[:, 2])
        - np.interp(times, defender[:, 0], defender[:, 2]),
    )
    z = (gap - tackle_radius_m) / tackle_softness_m
    hazard = tackle_rate * 0.5 * (1.0 - np.tanh(z / 2.0))
    return float(np.exp(-np.trapezoid(hazard, times)))


def defending_positions(
    frame: BundesligaFrame,
    attacking_team_id: str,
    exclude_ids: Iterable[str] = (),
) -> list[tuple[float, float]]:
    """Every tracked opponent in this frame, in our coordinates."""
    skip = set(exclude_ids)
    return [
        (float(state.x), float(state.y))
        for player_id, state in frame.players.items()
        if str(state.team_id) != str(attacking_team_id) and player_id not in skip
    ]


@dataclass(frozen=True)
class StackPaths:
    """Where the adopted artefacts live. Only the pass model is a file."""

    pass_model_path: str = "andrew/models/experimental_pass.json"

    def check(self) -> None:
        if not Path(self.pass_model_path).exists():
            raise ValueError(
                f"adopted pass model not found: {self.pass_model_path}. "
                "Clone the stage-3 repository into andrew/ and "
                "`pip install -e andrew/ --no-deps`."
            )
