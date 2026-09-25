"""Where a pass actually went, recovered from tracking rather than the event.

The DFL raw event carries only ONE position for a pass: `X-Position` equals
`X-Source-Position` on 85 % of open-play passes, so the event file does not say
where the ball ended up. Anything that reads `x_end` as a destination is really
reading the passer's own feet, which turns every pass into a zero-length one
whose "receiver" has to sprint back to the ball -- and collapses any kinematic
delivery model to zero.

So the destination is recovered from the positions file instead: follow the
ball from the kick until it comes under someone's control, and take that point,
that player and that flight time. The receiving player's team also gives an
outcome label that does not depend on the event file's own `Evaluation`, which
makes the two independent and lets one check the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bundesliga import FPS, BundesligaFrame


@dataclass(frozen=True)
class ObservedPass:
    kick_frame_id: int
    receive_frame_id: int
    flight_time_s: float
    target_xy: tuple[float, float]
    receiver_id: str
    receiver_team_id: str
    completed: bool
    travel_distance_m: float


def resolve_pass_target(
    frames: dict[int, BundesligaFrame],
    kick_frame_id: int,
    passer_id: str,
    passer_team_id: str,
    forward_frames: int = 75,
    stride: int = 2,
    release_distance_m: float = 2.0,
    control_distance_m: float = 1.5,
) -> ObservedPass | None:
    """Follow the ball from the kick to whoever first brings it under control.

    The ball must first LEAVE the passer (``release_distance_m``) so a player
    still dribbling is not mistaken for a reception, and the receiver is then
    the first player within ``control_distance_m`` of it. Returning None means
    the ball never resolved inside the window -- a long ball still travelling,
    a deflection, a stoppage -- and such passes are dropped rather than guessed.
    """
    start = frames.get(kick_frame_id)
    if start is None or start.ball is None:
        return None
    origin = (float(start.ball.x), float(start.ball.y))

    released = False
    for offset in range(stride, forward_frames + 1, stride):
        frame = frames.get(kick_frame_id + offset)
        if frame is None or frame.ball is None or not frame.players:
            continue
        ball = (float(frame.ball.x), float(frame.ball.y))
        passer = frame.players.get(passer_id)
        if not released:
            if passer is None:
                continue
            if math.hypot(ball[0] - passer.x, ball[1] - passer.y) >= release_distance_m:
                released = True
            continue

        nearest_id, nearest = min(
            frame.players.items(),
            key=lambda item: math.hypot(item[1].x - ball[0], item[1].y - ball[1]),
        )
        gap = math.hypot(nearest.x - ball[0], nearest.y - ball[1])
        if gap > control_distance_m or nearest_id == passer_id:
            continue
        return ObservedPass(
            kick_frame_id=kick_frame_id,
            receive_frame_id=frame.frame_id,
            flight_time_s=offset / FPS,
            target_xy=ball,
            receiver_id=str(nearest_id),
            receiver_team_id=str(nearest.team_id),
            completed=str(nearest.team_id) == str(passer_team_id),
            travel_distance_m=math.hypot(ball[0] - origin[0], ball[1] - origin[1]),
        )
    return None


@dataclass(frozen=True)
class IntendedPass:
    receiver_id: str
    target_xy: tuple[float, float]
    along_m: float
    perpendicular_m: float


def infer_intended_target(
    frames: dict[int, BundesligaFrame],
    kick_frame_id: int,
    passer_id: str,
    passer_team_id: str,
    velocity_frames: int = 4,
    minimum_along_m: float = 3.0,
    maximum_perpendicular_m: float = 8.0,
) -> IntendedPass | None:
    """Who the pass was aimed at, using only the ball's direction at release.

    The receiver must be chosen WITHOUT looking at who ended up with the ball.
    Naming the actual receiver leaks the outcome: on an interception that
    receiver is an opponent, and a delivery model handed an opponent as its
    receiver returns a low probability for arithmetic reasons rather than
    predictive ones -- which is exactly what produced a perfectly separated
    calibration curve on the first attempt.

    So the aim point comes from physics only. Take the ball's heading over the
    first frames of flight, project every team-mate onto that ray, and keep the
    one closest to it. The target is that projection -- ahead of a team-mate for
    a through ball, on his feet for a short one.
    """
    start = frames.get(kick_frame_id)
    later = frames.get(kick_frame_id + velocity_frames)
    if start is None or later is None or start.ball is None or later.ball is None:
        return None
    origin = (float(start.ball.x), float(start.ball.y))
    dx = float(later.ball.x) - origin[0]
    dy = float(later.ball.y) - origin[1]
    speed = math.hypot(dx, dy)
    if speed < 0.5:
        return None
    ux, uy = dx / speed, dy / speed

    best = None
    for player_id, player in start.players.items():
        if player_id == passer_id or str(player.team_id) != str(passer_team_id):
            continue
        rx, ry = float(player.x) - origin[0], float(player.y) - origin[1]
        along = rx * ux + ry * uy
        if along < minimum_along_m:
            continue
        perpendicular = abs(-uy * rx + ux * ry)
        if perpendicular > maximum_perpendicular_m:
            continue
        if best is None or perpendicular < best.perpendicular_m:
            best = IntendedPass(
                receiver_id=str(player_id),
                target_xy=(origin[0] + ux * along, origin[1] + uy * along),
                along_m=along,
                perpendicular_m=perpendicular,
            )
    return best
