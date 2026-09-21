"""A per-frame influence cache, so changing a role is instant.

``goal_weighted_influence.target_residual_influence`` recomputes every
defender's influence surface for every target it is asked about. That is fine
for one annotated beneficiary but far too slow for an app where the user clicks
a different beneficiary or a different defender every second.

This module computes each player's Fernandez influence surface **once** per
sampled frame and keeps the defensive sum around, so a residual field for any
target is one multiply away, and swapping a defender for its no-response
baseline is one subtract-and-add.

The arithmetic is the repository's, unchanged:

    intrinsic = target_influence * goal_weighted_space_value(target)
    uncovered = exp(-k * sum_of_defender_influences)
    residual  = intrinsic * uncovered

``tests/test_demo_viz_app.py`` asserts this reproduces
``target_residual_influence`` to floating-point tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..config import ensure_repo_on_path
from ..quantities import _frame_at, _velocities, baseline_tracks
from ..scene import Scene

ensure_repo_on_path()

from offball_value.bundesliga import FIELD_LENGTH, FIELD_WIDTH  # noqa: E402
from offball_value.fernandez_influence import fernandez_influence_surface  # noqa: E402
from offball_value.goal_weighted_influence import (  # noqa: E402
    GoalWeightedInfluenceConfig,
    goal_weighted_space_surface,
    influence_pitch_grid,
)


@dataclass
class SpaceResult:
    """One residual field and its integral, for a chosen target."""

    field: np.ndarray                  # (ny, nx)
    value: float

    def __add__(self, other: "SpaceResult") -> "SpaceResult":
        return SpaceResult(np.maximum(self.field, other.field), self.value + other.value)


class InfluenceCache:
    """Sampled Fernandez influence for every player in one scene."""

    def __init__(
        self,
        scene: Scene,
        every: int = 5,
        grid_resolution_m: float = 2.0,
        config: GoalWeightedInfluenceConfig | None = None,
    ):
        self.scene = scene
        self.config = config or GoalWeightedInfluenceConfig(grid_resolution_m=grid_resolution_m)
        self.xgrid, self.ygrid = influence_pitch_grid(self.config)
        self.cell_area = (FIELD_LENGTH / len(self.xgrid)) * (FIELD_WIDTH / len(self.ygrid))

        step = max(1, int(every))
        indices = np.arange(0, scene.n_frames, step)
        if indices[-1] != scene.n_frames - 1:
            indices = np.append(indices, scene.n_frames - 1)
        self.indices = indices
        self.times = scene.times[indices]

        self.goalkeepers = tuple(p.player_id for p in scene.players.values() if p.is_goalkeeper)
        shape = (len(indices), len(self.ygrid), len(self.xgrid))
        self.influence: dict[str, np.ndarray] = {}
        self._defender_sum = np.zeros(shape)
        self._space_value: dict[str, np.ndarray] = {}
        self._baseline_influence: dict[tuple[str, int, str], np.ndarray] = {}

        for slot, index in enumerate(indices):
            frame = _frame_at(scene, int(index))
            velocities = _velocities(scene, int(index))
            if frame.ball is None:
                continue
            ball_xy = (float(frame.ball.x), float(frame.ball.y))
            for player_id, state in frame.players.items():
                surface = fernandez_influence_surface(
                    self.xgrid, self.ygrid,
                    (float(state.x), float(state.y)),
                    _velocity_of(velocities, player_id),
                    ball_xy,
                    maximum_speed_mps=self.config.maximum_influence_speed_mps,
                )
                store = self.influence.setdefault(player_id, np.zeros(shape))
                store[slot] = surface
                if state.team_id != scene.attacking_team_id and player_id not in self.goalkeepers:
                    self._defender_sum[slot] += surface
        self._ball_xy = np.array([scene.ball_xy[i] for i in indices])

    # -- lookups --------------------------------------------------------
    def slot_for(self, frame_index: int) -> int:
        """Nearest sampled slot for a scene frame index."""

        return int(np.argmin(np.abs(self.indices - int(frame_index))))

    def _space_value_for(self, target_id: str) -> np.ndarray:
        cached = self._space_value.get(target_id)
        if cached is not None:
            return cached
        scene = self.scene
        player = scene.players[target_id]
        values = np.zeros((len(self.indices), len(self.ygrid), len(self.xgrid)))
        for slot, index in enumerate(self.indices):
            xy = player.xy[int(index)]
            if not np.all(np.isfinite(xy)):
                continue
            values[slot] = goal_weighted_space_surface(
                self.xgrid, self.ygrid, (float(xy[0]), float(xy[1])),
                scene.attacking_direction, self.config,
            )
        self._space_value[target_id] = values
        return values

    def baseline_influence(
        self, defender_id: str, freeze_index: int, mode: str = "hold"
    ) -> np.ndarray:
        """Influence of a defender that stopped responding at ``freeze_index``."""

        key = (defender_id, int(freeze_index), mode)
        cached = self._baseline_influence.get(key)
        if cached is not None:
            return cached
        scene = self.scene
        tracks, velocities = baseline_tracks(
            scene, int(freeze_index), mode, defender_ids=(defender_id,)
        )
        track = tracks.get(defender_id)
        factual = self.influence.get(defender_id)
        surfaces = np.zeros((len(self.indices), len(self.ygrid), len(self.xgrid)))
        if track is not None:
            vx, vy = velocities.get(defender_id, (0.0, 0.0))
            for slot, index in enumerate(self.indices):
                # up to the freeze frame the baseline player *is* the real one,
                # so reuse the observed surface and keep the two curves identical
                if int(index) <= int(freeze_index) and factual is not None:
                    surfaces[slot] = factual[slot]
                    continue
                xy = track[int(index)]
                ball = self._ball_xy[slot]
                if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(ball))):
                    continue
                surfaces[slot] = fernandez_influence_surface(
                    self.xgrid, self.ygrid, (float(xy[0]), float(xy[1])), (vx, vy),
                    (float(ball[0]), float(ball[1])),
                    maximum_speed_mps=self.config.maximum_influence_speed_mps,
                )
        self._baseline_influence[key] = surfaces
        return surfaces

    # -- residual fields ------------------------------------------------
    def defender_sum(
        self,
        slot: int,
        swap: Sequence[tuple[str, int, str]] = (),
    ) -> np.ndarray:
        """Defensive coverage at ``slot``; ``swap`` replaces defenders by baselines."""

        total = self._defender_sum[slot].copy()
        for defender_id, freeze_index, mode in swap:
            factual = self.influence.get(defender_id)
            if factual is None:
                continue
            total -= factual[slot]
            total += self.baseline_influence(defender_id, freeze_index, mode)[slot]
        return np.clip(total, 0.0, None)

    def residual(
        self,
        target_id: str,
        slot: int,
        swap: Sequence[tuple[str, int, str]] = (),
    ) -> SpaceResult:
        """Goal-weighted space the target keeps after defensive coverage."""

        influence = self.influence.get(target_id)
        if influence is None:
            zeros = np.zeros((len(self.ygrid), len(self.xgrid)))
            return SpaceResult(zeros, 0.0)
        intrinsic = influence[slot] * self._space_value_for(target_id)[slot]
        uncovered = np.exp(-self.config.defender_suppression_strength * self.defender_sum(slot, swap))
        residual = intrinsic * uncovered
        return SpaceResult(residual, float(np.sum(residual) * self.cell_area))

    def residual_series(
        self,
        target_id: str,
        swap: Sequence[tuple[str, int, str]] = (),
    ) -> np.ndarray:
        """The scalar residual over every sampled frame."""

        return np.array([self.residual(target_id, slot, swap).value
                         for slot in range(len(self.indices))])

    def combined(
        self,
        target_ids: Iterable[str],
        slot: int,
        swap: Sequence[tuple[str, int, str]] = (),
    ) -> SpaceResult:
        """Several targets at once: fields combined by maximum, values summed."""

        results = [self.residual(target_id, slot, swap) for target_id in target_ids]
        if not results:
            zeros = np.zeros((len(self.ygrid), len(self.xgrid)))
            return SpaceResult(zeros, 0.0)
        total = results[0]
        for result in results[1:]:
            total = total + result
        return total


def _velocity_of(velocities, player_id: str) -> tuple[float, float]:
    estimate = velocities.get(player_id)
    if estimate is None:
        return 0.0, 0.0
    return float(estimate.vx), float(estimate.vy)
