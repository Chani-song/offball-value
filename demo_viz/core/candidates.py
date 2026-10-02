"""The candidate-pass fan, mirroring ``demo_viz/web/site/js/obso.js``.

A fixed geometric rule, stated once and shared by the browser and the video
renderer. It is deliberately not a model:

  * five directions, evenly spaced across a 90 degree sector centred on the
    way the team is attacking;
  * one fixed length, clipped to the pitch;
  * drawn from whoever is clearly on the ball, and from nobody otherwise.

No completion probability, no ranking, no expected value. The fan answers
"which way could a forward pass point from here", and the OBSO surface behind
it answers "what is over there" -- the two are shown together and left for the
viewer to read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How far a candidate ray runs, in metres. A plausible forward pass, not a
#: predicted or optimised distance.
CANDIDATE_DISTANCE_M = 25.0

#: Degrees either side of the attacking direction.
CANDIDATE_ANGLES_DEG = (-45.0, -22.5, 0.0, 22.5, 45.0)

#: A carrier has to be this close to the ball to count as holding it.
#:
#: This is a *drawing* threshold, and deliberately looser than the pipeline's
#: 1.5 m carrier gate: tracking records a player's torso while a dribbler
#: pushes the ball two or three metres ahead of it, so 1.5 m hides the fan on
#: frames where the ball is plainly at someone's feet. Measured over ten
#: scenes, loosening it to 2.5 m only moves coverage from 38 % to 43 % of
#: frames -- the ball really is in flight or loose the rest of the time -- but
#: it fixes the cases a viewer would call obviously wrong.
CARRIER_MAX_M = 2.5

#: ...and this much closer than the next player, or the ball is contested.
#: During a pass nobody is near the ball at all, and two players converging on a
#: loose ball are not carrying it; both cases give no carrier, and the caller
#: hides the fan rather than inventing a passer.
CARRIER_MARGIN_M = 0.5


@dataclass(frozen=True)
class CandidateFan:
    carrier_id: str
    origin: tuple[float, float]
    #: ``(degrees, (x, y))`` per ray, in pitch coordinates.
    rays: tuple[tuple[float, tuple[float, float]], ...]


def carrier_at(scene, index: int) -> str | None:
    """The player clearly on the ball at ``index``, or ``None``."""

    ball = scene.ball_xy[index]
    if not np.all(np.isfinite(ball)):
        return None

    best_id, best, runner_up = None, np.inf, np.inf
    for player_id, player in scene.players.items():
        xy = player.xy[index]
        if not np.all(np.isfinite(xy)):
            continue
        distance = float(np.hypot(xy[0] - ball[0], xy[1] - ball[1]))
        if distance < best:
            runner_up, best, best_id = best, distance, player_id
        elif distance < runner_up:
            runner_up = distance

    if best_id is None or best > CARRIER_MAX_M:
        return None
    if runner_up - best < CARRIER_MARGIN_M:
        return None
    if scene.players[best_id].side != "attack":
        return None
    return best_id


def _clip_to_pitch(x0, y0, x1, y1, length, width):
    half_l, half_w = length / 2.0, width / 2.0
    dx, dy = x1 - x0, y1 - y0
    t = 1.0
    if dx > 1e-9:
        t = min(t, (half_l - x0) / dx)
    if dx < -1e-9:
        t = min(t, (-half_l - x0) / dx)
    if dy > 1e-9:
        t = min(t, (half_w - y0) / dy)
    if dy < -1e-9:
        t = min(t, (-half_w - y0) / dy)
    t = float(np.clip(t, 0.0, 1.0))
    return x0 + dx * t, y0 + dy * t


def candidate_passes(scene, index: int) -> CandidateFan | None:
    """The five candidate directions at ``index``, or ``None`` with no carrier."""

    carrier_id = carrier_at(scene, index)
    if carrier_id is None:
        return None
    x0, y0 = (float(v) for v in scene.players[carrier_id].xy[index])
    heading = 0.0 if int(scene.attacking_direction) >= 0 else np.pi

    rays = []
    for degrees in CANDIDATE_ANGLES_DEG:
        angle = heading + np.radians(degrees)
        end = _clip_to_pitch(
            x0, y0,
            x0 + float(np.cos(angle)) * CANDIDATE_DISTANCE_M,
            y0 + float(np.sin(angle)) * CANDIDATE_DISTANCE_M,
            scene.pitch_length, scene.pitch_width,
        )
        rays.append((float(degrees), end))
    return CandidateFan(carrier_id=carrier_id, origin=(x0, y0), rays=tuple(rays))
