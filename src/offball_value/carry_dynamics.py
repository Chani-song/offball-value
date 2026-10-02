"""What a carry to one point is worth, priced like a pass to one point.

The pass side has point_reception_estimate: give it a target and it answers
whether the ball gets there and is kept. The carry side had no equivalent. Its
value came from extending the carrier's OBSERVED velocity by 2-6 m and asking
only whether he is dispossessed along the way, which has two consequences that
measurement confirmed:

  - the carry cannot be priced anywhere except straight ahead, so "turn and
    drive into the space the defender left" is not expressible. The carrier is
    already heading within 45 deg of the vacated space 73 % of the time, but
    beyond 90 deg 9 % of the time, and those are unreachable by construction;
  - there is no arrival race at the destination. The pass asks whether the
    receiver beats his marker to the ball (receiver_first); the carry asks
    only whether someone can reach where the carrier ALREADY IS.

Meanwhile occupation credits him for vacated space within a 10 m soft radius
whether or not any priced carry could take him there.

This closes both, by mirroring the pass form exactly:

    pass   margin = defender_time - max(ball_time, receiver_time)
    carry  margin = defender_time - carrier_time          (he is the ball)

    P_carry = carrier_first x secure x path_retention

with carrier_first and secure from the same logistic and the same race sigma
the pass uses, so the two action classes finally answer one question on one
scale: does this player end up at this point with the ball.

CARRYING SPEED is the one quantity the pass form does not supply, and it is
not invented here. scripts/measure_carrying_speed.py measures it from tracking
as a ratio against the same side's off-ball players, and the caller passes it
in; the default below is a placeholder that must be replaced by that number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .bundesliga import BundesligaFrame
from .pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    player_time_to_point,
    secure_possession_probability,
)

# Measured, not assumed. scripts/measure_carry_speed_paired.py compares each
# player's own top speed in the second before he takes the ball with his own
# in the second after: 0.944 at +/-0.4 s, 0.937 at +/-1.0 s and +/-1.6 s on
# 5,059 / 3,306 / 1,937 pairs. The per-pair median climbs to 1.005 over the
# widest window, so most of even that gap is players decelerating to receive
# rather than a cost of carrying.
#
# An earlier attempt compared carriers against all off-ball team-mates and
# found carriers FASTER (99th percentile 7.72 against 6.46 m/s). That was
# measuring involvement -- the off-ball group includes everyone jogging back
# on the far side -- and it is why this constant is paired.
CARRY_SPEED_RATIO = 0.94

CARRY_PRESSURE_ACTION_TIME_S = 0.35
CARRY_PRESSURE_SIGMA_S = 0.25
PATH_SAMPLES = 7


@dataclass(frozen=True)
class CarryPointEstimate:
    target_x: float
    target_y: float
    carry_distance_m: float
    carrier_arrival_time_s: float
    nearest_defender_id: str | None
    nearest_defender_arrival_time_s: float
    defender_time_margin_s: float
    carrier_first_probability: float
    path_retention_probability: float
    weakest_path_time_s: float
    secure_possession_probability: float
    carry_probability: float


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def carry_point_estimate(
    frame: BundesligaFrame,
    velocities: Mapping[str, VelocityEstimate],
    carrier_id: str,
    attacking_team_id: str,
    target_xy: tuple[float, float],
    config: ArrivalModelConfig = ArrivalModelConfig(),
    goalkeeper_ids: Sequence[str] = (),
    carry_speed_ratio: float = CARRY_SPEED_RATIO,
) -> CarryPointEstimate | None:
    """Probability the carrier reaches ``target_xy`` still holding the ball."""
    carrier = frame.players.get(carrier_id)
    if carrier is None:
        return None
    carrier_velocity = velocities.get(
        carrier_id, VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds)
    )

    # Carrying is slower than a free sprint, so the arrival model runs with a
    # reduced cap. Scaling the config keeps one arrival model for both classes.
    carry_config = ArrivalModelConfig(
        **{
            **config.__dict__,
            "player_max_speed_mps": config.player_max_speed_mps * carry_speed_ratio,
            "player_acceleration_mps2": config.player_acceleration_mps2 * carry_speed_ratio,
        }
    )
    carrier_time = float(
        player_time_to_point(carrier, carrier_velocity, target_xy, carry_config)
    )

    excluded = set(goalkeeper_ids)
    defenders = [
        (
            float(player_time_to_point(player, velocities.get(
                player_id, VelocityEstimate(0.0, 0.0, 0.0, 0, config.history_seconds)),
                target_xy, config)),
            player_id,
        )
        for player_id, player in frame.players.items()
        if player.team_id != attacking_team_id and player_id not in excluded
    ]
    if defenders:
        defender_time, nearest_id = min(defenders)
    else:
        defender_time, nearest_id = math.inf, None

    margin = defender_time - carrier_time
    carrier_first = _logistic(margin / max(config.race_sigma_s, 1e-9)) \
        if math.isfinite(margin) else 1.0
    secure = float(secure_possession_probability(margin, config))

    # Retention along the straight line he would travel, as the existing carry
    # model does along the observed path: the weakest instant governs.
    start = (float(carrier.x), float(carrier.y))
    retention = 1.0
    weakest_time = math.inf
    for step in range(1, PATH_SAMPLES + 1):
        fraction = step / PATH_SAMPLES
        point = (
            start[0] + fraction * (target_xy[0] - start[0]),
            start[1] + fraction * (target_xy[1] - start[1]),
        )
        when = carrier_time * fraction
        product = 1.0
        for arrival, _pid in defenders:
            # Where a defender could be by the time the carrier is here.
            product *= _logistic(
                (arrival - when - CARRY_PRESSURE_ACTION_TIME_S)
                / CARRY_PRESSURE_SIGMA_S
            )
        del point
        if product < retention:
            retention = product
            weakest_time = when

    return CarryPointEstimate(
        target_x=float(target_xy[0]),
        target_y=float(target_xy[1]),
        carry_distance_m=float(math.hypot(target_xy[0] - start[0], target_xy[1] - start[1])),
        carrier_arrival_time_s=carrier_time,
        nearest_defender_id=nearest_id,
        nearest_defender_arrival_time_s=float(defender_time),
        defender_time_margin_s=float(margin),
        carrier_first_probability=float(carrier_first),
        path_retention_probability=float(retention),
        weakest_path_time_s=float(weakest_time),
        secure_possession_probability=secure,
        carry_probability=float(np.clip(carrier_first * secure * retention, 0.0, 1.0)),
    )
