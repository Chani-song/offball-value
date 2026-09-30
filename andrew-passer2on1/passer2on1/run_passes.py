"""Pass candidates aimed along the receiver's own run (2026-09-27).

The imported candidates sit around the receiver's current position on the
ATTACK AXIS: feet, 4 and 8 m ahead, 4 m behind, 4 m either side, each in three
families (ground / driven / lofted), 18 in all. A runner going diagonally in
behind has no candidate on his path, and the physics pass model (candidate A)
does not read the family, so the three families were the same pass priced
thrice.

These 18 are geometry instead, one family: 5 distances along the receiver's
run direction (-4, 0, 4, 8, 12 m) x 3 lateral offsets (-4, 0, 4 m), plus 4, 8
and 12 m from the receiver straight toward the goal. The run direction is his
velocity when he moves at 1 m/s or more, else the attack axis, so a standing
receiver gets the old axis-aligned set. Same count, same cost, and every
decoding of a pass index still uses len(config.passes).

The positions-only pass model still reads the family; every RunPass is
"ground" for it. This file is kept byte-identical in the three homes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUN_MIN_MPS = 1.0


@dataclass(frozen=True)
class RunPass:
    along: float = 0.0
    lateral: float = 0.0
    goalward: float = 0.0
    family: str = "ground"

    def target(self, receiver, direction: int, velocity=None, goal=None):
        """Aim point for a receiver at `receiver` moving at `velocity`; broadcastable."""
        r = np.asarray(receiver, dtype=float)
        if self.goalward:
            if goal is None:
                raise ValueError("a goalward pass needs the goal")
            to_goal = np.asarray(goal, dtype=float) - r
            norm = np.linalg.norm(to_goal, axis=-1, keepdims=True)
            return r + self.goalward * to_goal / np.maximum(norm, 1e-9)
        if velocity is None:
            u = np.broadcast_to(np.array([float(direction), 0.0]), r.shape)
        else:
            v = np.broadcast_to(np.asarray(velocity, dtype=float), r.shape)
            speed = np.linalg.norm(v, axis=-1, keepdims=True)
            axis = np.broadcast_to(np.array([float(direction), 0.0]), r.shape)
            u = np.where(speed >= RUN_MIN_MPS, v / np.maximum(speed, 1e-9), axis)
        n = np.stack([-u[..., 1], u[..., 0]], axis=-1)
        return r + self.along * u + self.lateral * n


RUN_PASSES = tuple(RunPass(along=a, lateral=l) for a in (-4.0, 0.0, 4.0, 8.0, 12.0)
                   for l in (-4.0, 0.0, 4.0)) + tuple(RunPass(goalward=g) for g in (4.0, 8.0, 12.0))


def goal_of(scenario):
    return np.array([scenario.pitch_length if scenario.attack_direction == 1 else 0.0,
                     scenario.pitch_width / 2.0])


def target_of(choice, receiver, direction: int, velocity=None, scenario=None):
    """The aim point of any pass choice, imported PassChoice or RunPass."""
    if isinstance(choice, RunPass):
        return choice.target(receiver, direction, velocity, goal_of(scenario) if scenario is not None else None)
    return choice.target(receiver, direction)


def passes_named(name: str):
    if name == "run":
        return RUN_PASSES
    if name == "andrew":
        from defensive_positioning.models import DEFAULT_PASSES
        return DEFAULT_PASSES
    raise ValueError(f"unknown pass set {name}")
