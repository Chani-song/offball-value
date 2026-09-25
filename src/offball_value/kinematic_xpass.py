"""Pass-success features that include the arrival race, shared by fit and use.

xpass360 reads a still frame, so it cannot express who reaches the ball first;
held inside a stratum of matched static geometry its correlation with
receiver_first is +0.267 where the mechanistic chain holds +0.762. But
consecutive 360 freeze frames sit a median 0.97 s apart, and degrading real
tracking to exactly that representation still recovers receiver_first at
correlation 0.934, so the race can be differenced out of a frame pair and
handed to the model as features. On 204,857 StatsBomb passes they are worth
+0.0039 AUC over the static set (fold sd 0.0015).

TRAIN AND INFERENCE MUST AGREE, so the feature construction lives here and
both callers import it. Duplicating it is how the two silently drift apart.

One asymmetry is deliberate and worth stating: at training time the velocities
come from a 0.97 s frame pair with identities re-matched by proximity, while
inside the pipeline they come from 25 fps tracking with real identities. The
pipeline therefore feeds CLEANER inputs than the fit ever saw. That direction
is the safe one -- the recovery test put the degraded and true quantities at
correlation 0.934 -- but it is still a shift, and it belongs in any claim made
from the resulting numbers.

Distances are metres throughout. Only differences and distances enter, so a
centred pitch and a corner-origin pitch give the same features as long as the
scale is right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINEMATIC_FEATURE_NAMES = (
    "def_time_to_target",
    "mate_time_to_target",
    "first_margin_s",
    "ball_time_s",
    "def_margin_vs_ball_s",
    "mate_margin_vs_ball_s",
    "lane_closing_mps",
    "mean_opp_speed",
    "max_opp_speed",
    "actor_speed",
)

PASS_SPEED_MPS = 14.0
MAX_SPEED_MPS = 9.0
ACCEL_MPS2 = 3.5
REACTION_S = 0.10
TURN_PENALTY_S = 0.40
LANE_CLOSING_RADIUS_M = 15.0
ABSENT_TIME_S = 20.0

_MODEL_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class TrackedPlayer:
    x: float
    y: float
    vx: float
    vy: float
    teammate: bool
    actor: bool = False
    keeper: bool = False


def time_to_point(player: TrackedPlayer, target_xy: tuple[float, float]) -> float:
    """Arrival time in metres/seconds, matching pass_dynamics.player_time_to_point."""
    dx = float(target_xy[0]) - player.x
    dy = float(target_xy[1]) - player.y
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return 0.0
    ux, uy = dx / distance, dy / distance
    speed = min(math.hypot(player.vx, player.vy), MAX_SPEED_MPS)
    if speed > 1e-9:
        toward = player.vx * ux + player.vy * uy
        cosine = float(np.clip(toward / speed, -1.0, 1.0))
        turn = math.acos(cosine)
        toward = max(0.0, toward)
    else:
        turn, toward = 0.0, 0.0
    reaction = REACTION_S + TURN_PENALTY_S * turn / math.pi
    remaining = max(0.0, distance - toward * reaction)
    initial = min(toward, MAX_SPEED_MPS)
    ramp_time = max(0.0, (MAX_SPEED_MPS - initial) / ACCEL_MPS2)
    ramp_distance = initial * ramp_time + 0.5 * ACCEL_MPS2 * ramp_time**2
    if remaining <= ramp_distance:
        travel = (
            -initial + math.sqrt(initial**2 + 2.0 * ACCEL_MPS2 * remaining)
        ) / ACCEL_MPS2
    else:
        travel = ramp_time + (remaining - ramp_distance) / MAX_SPEED_MPS
    return float(reaction + travel)


def kinematic_features(
    players: list[TrackedPlayer],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> np.ndarray:
    """The ten race features, in KINEMATIC_FEATURE_NAMES order."""
    ball_time = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]) / PASS_SPEED_MPS
    mid = ((start_xy[0] + end_xy[0]) / 2.0, (start_xy[1] + end_xy[1]) / 2.0)

    defender_times: list[float] = []
    mate_times: list[float] = []
    speeds: list[float] = []
    closings: list[float] = []
    actor_speed = 0.0

    for player in players:
        speed = math.hypot(player.vx, player.vy)
        if player.actor:
            actor_speed = speed
            continue
        if player.keeper:
            continue
        arrival = time_to_point(player, end_xy)
        if player.teammate:
            mate_times.append(arrival)
            continue
        defender_times.append(arrival)
        speeds.append(speed)
        dx, dy = mid[0] - player.x, mid[1] - player.y
        gap = math.hypot(dx, dy)
        if 1e-6 < gap < LANE_CLOSING_RADIUS_M:
            closings.append((player.vx * dx + player.vy * dy) / gap)

    defender = min(defender_times) if defender_times else ABSENT_TIME_S
    mate = min(mate_times) if mate_times else ABSENT_TIME_S
    return np.asarray(
        [
            defender,
            mate,
            defender - mate,
            ball_time,
            defender - ball_time,
            mate - ball_time,
            float(np.mean(closings)) if closings else 0.0,
            float(np.mean(speeds)) if speeds else 0.0,
            float(np.max(speeds)) if speeds else 0.0,
            actor_speed,
        ],
        dtype=float,
    )


def load_kinematic_model(model_path: str | Path):
    key = str(model_path)
    if key not in _MODEL_CACHE:
        import joblib

        _MODEL_CACHE[key] = joblib.load(model_path)
    return _MODEL_CACHE[key]
