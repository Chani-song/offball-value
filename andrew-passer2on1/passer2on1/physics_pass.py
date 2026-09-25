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

The interface is ExpectedPass.predict's, so the game calls it unchanged.
This file is kept byte-identical in andrew-passer2on1/passer2on1,
andrew-fixedpasser/fixedpasser and src/offball_value (where it is fitted).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KIND = "physics_race_logit"
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


class PhysicsRacePass:
    def __init__(self, spec: dict):
        if spec.get("kind") != KIND:
            raise ValueError(f"not a {KIND} model")
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
                target):
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
        take = np.maximum(reach_time(r, rv, t, self.player), ball_at_target)
        first_defender = reach_time(d, dv, t[..., None, :], self.player).min(axis=-1)
        receiver_margin = first_defender - take
        k = np.arange(1, LANE_POINTS) / LANE_POINTS                       # [K-1]
        points = c[..., None, :] + (t - c)[..., None, :] * k[:, None]     # [..., K-1, 2]
        ball_at = (length[..., None] * k) / speed[..., None]              # [..., K-1]
        lane_reach = reach_time(d[..., :, None, :], dv[..., :, None, :],
                                points[..., None, :, :], self.player)     # [..., m, K-1]
        lane_margin = (lane_reach - ball_at[..., None, :]).min(axis=(-2, -1))
        return receiver_margin, lane_margin

    def probability_from_margins(self, receiver_margin, lane_margin):
        x1 = np.clip(receiver_margin, -MARGIN_CLIP_S, MARGIN_CLIP_S)
        x2 = np.clip(lane_margin, -MARGIN_CLIP_S, MARGIN_CLIP_S)
        z = self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2
        return 0.5 * (1.0 + np.tanh(z / 2.0))

    def predict(self, carrier, receiver, defenders, carrier_velocity, receiver_velocity,
                defender_velocities, target, family=0, attack_direction=1,
                pitch_length=105.0, pitch_width=68.0):
        """ExpectedPass.predict's signature; family and direction do not enter."""
        m_r, m_l = self.margins(carrier, receiver, defenders, receiver_velocity,
                                defender_velocities, target)
        return self.probability_from_margins(m_r, m_l)


def load_pass_model(path, allow_proxy=False):
    """A candidate-A JSON, or anything ExpectedPass.load accepts."""
    raw = json.loads(Path(path).read_text())
    if raw.get("kind") == KIND:
        return PhysicsRacePass(raw)
    from defensive_positioning.expected_pass import ExpectedPass
    return ExpectedPass.load(path, allow_proxy=allow_proxy)
