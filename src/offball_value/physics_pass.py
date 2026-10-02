"""Candidate A: pass completion from the arrival race, with three fitted numbers.

The solver's pass model reads positions only, so a runner sprinting into the
space behind a defender who is running the other way prices the same as two
men standing still. This model reads the race instead.

  ball      travels the lane from the passer to the target at a speed that
            depends on the distance (v = intercept + slope * distance,
            least-squares on our passes' measured launch speeds, clipped to
            their 5th-95th percentile).
  players   reach a point by the solver's own agile motion
            (agile_motion.AGILE): they keep their velocity for the 0.2 s
            reaction, one moving away brakes first (a plant at 9.0 m/s^2 and
            0.1 s standing when running at 2 m/s or more, else 6.0), then they
            accelerate at 4.5 m/s^2 up to 9.0 m/s. The component of velocity
            toward the point is what counts; sideways speed is ignored, which
            is generous to everyone alike.
  margins   receiver: the earliest defender arrival at the target minus the
            time the receiver can take the ball there (the later of his own
            arrival and the ball's). Positive: nobody beats him to it.
            lane: the earliest any defender reaches any of 7 evenly spaced
            points on the lane minus the time the ball passes that point.
            Negative: someone can step into the lane first.
  P         logistic in the two margins (each clipped to +-3 s, where the race
            is decided either way): P = sigmoid(b0 + b1*receiver + b2*lane).
            b0, b1, b2 are fitted by maximum likelihood on our passes, which
            also fixes the level -- the uncalibrated mechanistic chain sat at a
            median 0.32 on passes completing 81 %.

Pass family (ground / driven / lofted) is not modelled: every family gets the
same number, so the solver's 18 passes behave as 6 targets. Everything is
distances and times, so the attacking direction does not enter.

Candidate A-sym (PhysicsRaceSymPass, 2026-09-28) holds the receiver and the
defenders to one rule. In A the receiver "arrives" when he runs through the
target and may then wait there for the ball -- a runner at 9 m/s who passes a
point 0.7 s before the ball is counted as standing on it -- while a defender who
has passed the point must stop and come back. A-sym asks of everyone whether he
can be AT the point when it matters: running through it in time, or getting
there and stopping to wait (reach_window). Same physics, same two margins, same
three fitted numbers; only who counts as there changes. When everyone can stop
before the points, A-sym and A give the same margins.

The interface is ExpectedPass.predict's, so the game calls it unchanged.
This file is kept byte-identical in andrew-passer2on1/passer2on1,
andrew-fixedpasser/fixedpasser and src/offball_value (where it is fitted).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KIND = "physics_race_logit"
KIND_SYM = "physics_race_sym_logit"
LANE_POINTS = 8          # the lane is cut in 8; the 7 interior points are checked
MARGIN_CLIP_S = 3.0


def _norm(v):
    return np.sqrt(np.sum(np.square(v), axis=-1))


def reach_time(position, velocity, point, player: dict):
    """Seconds for a player at `position` moving at `velocity` to reach `point`."""
    p, v, x = (np.asarray(a, dtype=float) for a in (position, velocity, point))
    tau = player["reaction_s"]
    q = p + v * tau
    gap = x - q
    dist = _norm(gap)
    speed = _norm(v)
    toward = np.sum(v * gap, axis=-1) / np.maximum(dist, 1e-9)
    away = toward < 0.0
    plant = away & (speed >= player["plant_min_speed"])
    braking = np.where(plant, player["plant_braking"], player["braking"])
    stop_s = np.where(away, speed / braking, 0.0) + np.where(plant, player["plant_seconds"], 0.0)
    shift = np.where(away[..., None], v * (speed / (2.0 * braking))[..., None], 0.0)
    dist = np.where(away, _norm(x - (q + shift)), dist)
    u = np.clip(np.where(away, 0.0, toward), 0.0, player["max_speed"])
    a, top = player["speed_up"], player["max_speed"]
    t_top = (top - u) / a
    d_top = u * t_top + 0.5 * a * t_top ** 2
    run = np.where(dist <= d_top, (-u + np.sqrt(u * u + 2.0 * a * dist)) / a,
                   t_top + (dist - d_top) / top)
    return tau + stop_s + run


def reach_window(position, velocity, point, player: dict):
    """When a player can be AT `point`, with reach_time's physics: (earliest, latest, back).

      earliest  reach_time: heading for the point at once, the moment he gets there
      latest    still approaching it and braking all the way -- with reach_time's own
                stop (the plant's deceleration from 2 m/s up, else the normal one):
                the last moment he gets there; +inf when he can stop before it and
                wait, -inf when after the reaction he is moving away from it
      back      overshoot, stop and come back -- reach_time's moving-away branch,
                for anyone (equal to earliest for a player moving away)
    He can be there at time t when earliest <= t <= latest, or t >= back.
    """
    p, v, x = (np.asarray(a, dtype=float) for a in (position, velocity, point))
    tau = player["reaction_s"]
    q = p + v * tau
    gap = x - q
    dist = _norm(gap)
    speed = _norm(v)
    toward = np.sum(v * gap, axis=-1) / np.maximum(dist, 1e-9)
    earliest = reach_time(p, v, x, player)
    plant = speed >= player["plant_min_speed"]
    braking = np.where(plant, player["plant_braking"], player["braking"])
    stop_s = speed / braking + np.where(plant, player["plant_seconds"], 0.0)
    back_dist = _norm(x - (q + v * (speed / (2.0 * braking))[..., None]))
    a, top = player["speed_up"], player["max_speed"]
    d_top = 0.5 * top * top / a
    back = tau + stop_s + np.where(back_dist <= d_top, np.sqrt(2.0 * back_dist / a),
                                   top / a + (back_dist - d_top) / top)
    u = np.clip(toward, 0.0, player["max_speed"])
    b = braking                         # the same stop as reach_time's moving-away branch
    can_stop = u * u <= 2.0 * b * dist
    late = tau + (u - np.sqrt(np.maximum(u * u - 2.0 * b * dist, 0.0))) / b
    latest = np.where(toward < 0.0, -np.inf, np.where(can_stop, np.inf, late))
    return earliest, latest, back


class PhysicsRacePass:
    kind = KIND

    def __init__(self, spec: dict):
        if spec.get("kind") != self.kind:
            raise ValueError(f"not a {self.kind} model")
        self.spec = spec
        self.coef = np.asarray(spec["coef"], dtype=float)
        if self.coef.shape != (3,) or not np.isfinite(self.coef).all():
            raise ValueError("coef must be three finite numbers")
        self.ball = spec["ball_speed"]
        self.player = spec["player"]
        self.metadata = spec.get("metadata", {})

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def ball_speed(self, distance):
        b = self.ball
        return np.clip(b["intercept"] + b["slope"] * np.asarray(distance, dtype=float),
                       b["min"], b["max"])

    def margins(self, carrier, receiver, defenders, receiver_velocity, defender_velocities,
                target, committed_s=None):
        """committed_s (2026-09-29, the multi-pass games): the first defender has spent that long after the
        release carrying out a command the game chose -- his position and velocity given are where it left
        him -- so he races from there with no reaction of his own, committed_s later. None: everyone keeps
        his velocity through the model's reaction, as fitted."""
        c, r, t = (np.asarray(a, dtype=float) for a in (carrier, receiver, target))
        d = np.asarray(defenders, dtype=float)
        dv = np.asarray(defender_velocities, dtype=float)
        rv = np.asarray(receiver_velocity, dtype=float)
        lead = np.broadcast_shapes(c.shape[:-1], r.shape[:-1], t.shape[:-1], rv.shape[:-1],
                                   d.shape[:-2], dv.shape[:-2])
        c, r, t, rv = (np.broadcast_to(a, lead + (2,)) for a in (c, r, t, rv))
        d = np.broadcast_to(d, lead + d.shape[-2:])
        dv = np.broadcast_to(dv, d.shape)
        length = _norm(t - c)
        speed = self.ball_speed(length)
        ball_at_target = length / speed
        take = self._take(r, rv, t, ball_at_target)
        there = self._there(d, dv, t[..., None, :], ball_at_target[..., None])
        if committed_s is not None:
            there[..., 0] = self._there(d[..., 0, :], dv[..., 0, :], t, ball_at_target,
                                        dict(self.player, reaction_s=0.0), committed_s)
        first_defender = there.min(axis=-1)
        receiver_margin = first_defender - take
        k = np.arange(1, LANE_POINTS) / LANE_POINTS                       # [K-1]
        points = c[..., None, :] + (t - c)[..., None, :] * k[:, None]     # [..., K-1, 2]
        ball_at = (length[..., None] * k) / speed[..., None]              # [..., K-1]
        lane_reach = self._there(d[..., :, None, :], dv[..., :, None, :],
                                 points[..., None, :, :], ball_at[..., None, :])  # [..., m, K-1]
        if committed_s is not None:
            lane_reach[..., 0, :] = self._there(d[..., 0, None, :], dv[..., 0, None, :], points, ball_at,
                                                dict(self.player, reaction_s=0.0), committed_s)
        lane_margin = (lane_reach - ball_at[..., None, :]).min(axis=(-2, -1))
        return receiver_margin, lane_margin

    def _take(self, receiver, receiver_velocity, target, ball_time):
        """When the receiver takes the ball: the later of his arrival and the ball's."""
        return np.maximum(reach_time(receiver, receiver_velocity, target, self.player), ball_time)

    def _there(self, position, velocity, point, ball_time, player=None, offset=0.0):
        """When a defender counts as at `point` (the ball gets there at `ball_time`); `offset` seconds are
        added to his clock (a committed start, see margins)."""
        return reach_time(position, velocity, point, player or self.player) + offset

    def probability_from_margins(self, receiver_margin, lane_margin):
        x1 = np.clip(receiver_margin, -MARGIN_CLIP_S, MARGIN_CLIP_S)
        x2 = np.clip(lane_margin, -MARGIN_CLIP_S, MARGIN_CLIP_S)
        z = self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2
        return 0.5 * (1.0 + np.tanh(z / 2.0))

    def predict(self, carrier, receiver, defenders, carrier_velocity, receiver_velocity,
                defender_velocities, target, family=0, attack_direction=1,
                pitch_length=105.0, pitch_width=68.0, committed_s=None):
        """ExpectedPass.predict's signature; family and direction do not enter."""
        m_r, m_l = self.margins(carrier, receiver, defenders, receiver_velocity,
                                defender_velocities, target, committed_s)
        return self.probability_from_margins(m_r, m_l)


class PhysicsRaceSymPass(PhysicsRacePass):
    """Candidate A-sym: a player counts as at a point only if he can be there when the
    ball is -- running through it by then, or stopping on it to wait (reach_window);
    one who passes it too early to stop must come back. For the receiver and the
    defenders alike."""
    kind = KIND_SYM

    def _take(self, receiver, receiver_velocity, target, ball_time):
        earliest, latest, back = reach_window(receiver, receiver_velocity, target, self.player)
        return np.maximum(np.where(latest >= ball_time, earliest, back), ball_time)

    def _there(self, position, velocity, point, ball_time, player=None, offset=0.0):
        earliest, latest, back = reach_window(position, velocity, point, player or self.player)
        earliest, latest, back = earliest + offset, latest + offset, back + offset
        return np.where(latest >= ball_time, earliest, back)


def load_pass_model(path, allow_proxy=False):
    """A candidate-A or A-sym JSON, or anything ExpectedPass.load accepts."""
    raw = json.loads(Path(path).read_text())
    if raw.get("kind") == KIND:
        return PhysicsRacePass(raw)
    if raw.get("kind") == KIND_SYM:
        return PhysicsRaceSymPass(raw)
    from defensive_positioning.expected_pass import ExpectedPass
    return ExpectedPass.load(path, allow_proxy=allow_proxy)
