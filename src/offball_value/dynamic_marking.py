"""Spatiotemporal marking diagnostics for a moving attacking option.

The endpoint-only diagnostics in the research prototype can select a defender
path that reaches a position after the attacker has already left it.  This
module instead compares defender and attacker paths at matched times.  It is a
geometry audit, not the final defensive value function.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


TimedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class DynamicMarkingConfig:
    """Configuration for a moving goal-side interception target."""

    goal_side_offset_m: float = 1.5
    lookahead_seconds: float = 0.4
    response_delay_seconds: float = 0.2
    wrong_side_weight: float = 2.0

    def validate(self) -> None:
        if self.goal_side_offset_m < 0.0:
            raise ValueError("goal_side_offset_m cannot be negative")
        if self.lookahead_seconds < 0.0:
            raise ValueError("lookahead_seconds cannot be negative")
        if self.response_delay_seconds < 0.0:
            raise ValueError("response_delay_seconds cannot be negative")
        if self.wrong_side_weight < 0.0:
            raise ValueError("wrong_side_weight cannot be negative")


@dataclass(frozen=True)
class MarkingSample:
    time_s: float
    actor_x: float
    actor_y: float
    anticipated_actor_x: float
    anticipated_actor_y: float
    defender_x: float
    defender_y: float
    target_x: float
    target_y: float
    actor_defender_distance_m: float
    target_error_m: float
    goal_side_projection_m: float
    wrong_side_distance_m: float
    weighted_error_m: float


@dataclass(frozen=True)
class DynamicMarkingSummary:
    mean_weighted_error_m: float
    peak_weighted_error_m: float
    terminal_weighted_error_m: float
    mean_target_error_m: float
    mean_actor_defender_distance_m: float
    mean_wrong_side_distance_m: float
    wrong_side_fraction: float
    sample_count: int
    samples: tuple[MarkingSample, ...]


@dataclass(frozen=True)
class MarkingPathCandidate:
    identifier: str
    path_txy: tuple[TimedPoint, ...]
    effort: float = 0.0


@dataclass(frozen=True)
class RankedMarkingPath:
    candidate: MarkingPathCandidate
    summary: DynamicMarkingSummary


def _deduplicate_path(path_txy: Sequence[TimedPoint]) -> tuple[TimedPoint, ...]:
    samples: dict[float, tuple[float, float]] = {}
    for time_s, x, y in path_txy:
        values = (float(time_s), float(x), float(y))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("path samples must be finite")
        samples[values[0]] = (values[1], values[2])
    if not samples:
        raise ValueError("path cannot be empty")
    return tuple((time_s, *samples[time_s]) for time_s in sorted(samples))


def interpolate_timed_point(
    path_txy: Sequence[TimedPoint],
    time_s: float,
) -> tuple[float, float]:
    """Linearly interpolate a path while clamping outside its time range."""

    path = _deduplicate_path(path_txy)
    time_s = float(time_s)
    if time_s <= path[0][0]:
        return path[0][1], path[0][2]
    if time_s >= path[-1][0]:
        return path[-1][1], path[-1][2]
    times = np.asarray([sample[0] for sample in path], dtype=float)
    upper = int(np.searchsorted(times, time_s, side="right"))
    before, after = path[upper - 1], path[upper]
    fraction = (time_s - before[0]) / max(after[0] - before[0], 1e-12)
    return (
        float(before[1] + fraction * (after[1] - before[1])),
        float(before[2] + fraction * (after[2] - before[2])),
    )


def moving_goal_side_target(
    actor_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    offset_m: float,
) -> tuple[float, float]:
    """Return a point ``offset_m`` beyond the actor toward the defended goal."""

    ax, ay = map(float, actor_xy)
    gx, gy = map(float, goal_xy)
    offset_m = float(offset_m)
    if not all(math.isfinite(value) for value in (ax, ay, gx, gy, offset_m)):
        raise ValueError("target inputs must be finite")
    if offset_m < 0.0:
        raise ValueError("offset_m cannot be negative")
    dx, dy = gx - ax, gy - ay
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        return ax, ay
    return ax + offset_m * dx / distance, ay + offset_m * dy / distance


def marking_sample(
    time_s: float,
    actor_xy: tuple[float, float],
    anticipated_actor_xy: tuple[float, float],
    defender_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    config: DynamicMarkingConfig = DynamicMarkingConfig(),
) -> MarkingSample:
    """Measure whether the defender controls the actor's future goal side."""

    config.validate()
    ax, ay = map(float, actor_xy)
    fx, fy = map(float, anticipated_actor_xy)
    dx, dy = map(float, defender_xy)
    gx, gy = map(float, goal_xy)
    target = moving_goal_side_target(
        (fx, fy), (gx, gy), config.goal_side_offset_m
    )
    goal_dx, goal_dy = gx - fx, gy - fy
    goal_distance = math.hypot(goal_dx, goal_dy)
    if goal_distance <= 1e-12:
        goal_unit = (0.0, 0.0)
    else:
        goal_unit = (goal_dx / goal_distance, goal_dy / goal_distance)
    projection = (dx - fx) * goal_unit[0] + (dy - fy) * goal_unit[1]
    wrong_side = max(0.0, -projection)
    target_error = math.hypot(dx - target[0], dy - target[1])
    weighted_error = target_error + config.wrong_side_weight * wrong_side
    return MarkingSample(
        time_s=float(time_s),
        actor_x=ax,
        actor_y=ay,
        anticipated_actor_x=fx,
        anticipated_actor_y=fy,
        defender_x=dx,
        defender_y=dy,
        target_x=float(target[0]),
        target_y=float(target[1]),
        actor_defender_distance_m=float(math.hypot(dx - ax, dy - ay)),
        target_error_m=float(target_error),
        goal_side_projection_m=float(projection),
        wrong_side_distance_m=float(wrong_side),
        weighted_error_m=float(weighted_error),
    )


def summarize_dynamic_marking(
    actor_path_txy: Sequence[TimedPoint],
    defender_path_txy: Sequence[TimedPoint],
    goal_xy: tuple[float, float],
    config: DynamicMarkingConfig = DynamicMarkingConfig(),
    evaluation_times_s: Iterable[float] | None = None,
) -> DynamicMarkingSummary:
    """Summarize one defender path against the full moving attacker path."""

    config.validate()
    actor_path = _deduplicate_path(actor_path_txy)
    defender_path = _deduplicate_path(defender_path_txy)
    horizon = min(actor_path[-1][0], defender_path[-1][0])
    if evaluation_times_s is None:
        times = sorted(
            {
                sample[0]
                for sample in actor_path
                if config.response_delay_seconds - 1e-9 <= sample[0] <= horizon + 1e-9
            }
            | {
                sample[0]
                for sample in defender_path
                if config.response_delay_seconds - 1e-9 <= sample[0] <= horizon + 1e-9
            }
        )
    else:
        times = sorted(
            {
                float(value)
                for value in evaluation_times_s
                if config.response_delay_seconds - 1e-9 <= float(value) <= horizon + 1e-9
            }
        )
    if not times:
        raise ValueError("no evaluation time remains after response delay")

    samples = []
    for time_s in times:
        actor_xy = interpolate_timed_point(actor_path, time_s)
        anticipated_time = min(horizon, time_s + config.lookahead_seconds)
        anticipated_xy = interpolate_timed_point(actor_path, anticipated_time)
        defender_xy = interpolate_timed_point(defender_path, time_s)
        samples.append(
            marking_sample(
                time_s,
                actor_xy,
                anticipated_xy,
                defender_xy,
                goal_xy,
                config,
            )
        )
    weighted = np.asarray([sample.weighted_error_m for sample in samples])
    target_errors = np.asarray([sample.target_error_m for sample in samples])
    distances = np.asarray(
        [sample.actor_defender_distance_m for sample in samples]
    )
    wrong = np.asarray([sample.wrong_side_distance_m for sample in samples])
    return DynamicMarkingSummary(
        mean_weighted_error_m=float(np.mean(weighted)),
        peak_weighted_error_m=float(np.max(weighted)),
        terminal_weighted_error_m=float(weighted[-1]),
        mean_target_error_m=float(np.mean(target_errors)),
        mean_actor_defender_distance_m=float(np.mean(distances)),
        mean_wrong_side_distance_m=float(np.mean(wrong)),
        wrong_side_fraction=float(np.mean(wrong > 1e-6)),
        sample_count=len(samples),
        samples=tuple(samples),
    )


def rank_dynamic_marking_paths(
    actor_path_txy: Sequence[TimedPoint],
    candidates: Sequence[MarkingPathCandidate],
    goal_xy: tuple[float, float],
    config: DynamicMarkingConfig = DynamicMarkingConfig(),
    evaluation_times_s: Iterable[float] | None = None,
) -> tuple[RankedMarkingPath, ...]:
    """Rank feasible paths by full-trajectory goal-side marking quality."""

    ranked = [
        RankedMarkingPath(
            candidate=candidate,
            summary=summarize_dynamic_marking(
                actor_path_txy,
                candidate.path_txy,
                goal_xy,
                config,
                evaluation_times_s,
            ),
        )
        for candidate in candidates
    ]
    ranked.sort(
        key=lambda row: (
            row.summary.mean_weighted_error_m,
            row.summary.peak_weighted_error_m,
            row.summary.terminal_weighted_error_m,
            float(row.candidate.effort),
            row.candidate.identifier,
        )
    )
    return tuple(ranked)
