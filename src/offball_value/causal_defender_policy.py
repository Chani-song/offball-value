"""Causal (observation-limited) defender policy — the v0.3 main line.

The 2026-08-28 lab meeting replaced the committed-plan oracle with a causal
information structure: at each replanning instant ``T`` the defender may use
observations up to ``T`` only, and its new plan takes effect after a
perception-reaction delay.  Between decisions the defender executes what it
already committed, so the produced trajectory is causal by construction.

Design choices (see docs/causal_defender_policy_v0_3_design.md):

- The defender plans against the WORST CASE over every displayed attacking
  option's physically feasible short-term branches — no clairvoyance and no
  overcommitment to the currently observed direction, which is also the
  structural guardrail against rewarding dribble feints that merely exploit
  the reaction delay.
- Non-focal players are extrapolated at constant velocity inside the
  defender's internal model.  Observed frames beyond ``T`` are never read.
- The internal objective is a cheap geometric threat proxy.  The resulting
  trajectory is priced afterwards by the ordinary retrospective evaluator,
  so policy heuristics never touch reported Q values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .defender_trajectory_search import (
    DefenderTrajectorySearchConfig,
    generate_feasible_defender_trajectories,
)
from .dynamic_marking import interpolate_timed_point
from .obso import score_at_points


TimedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class CausalPolicyConfig:
    """Information structure and internal-model settings."""

    replan_interval_seconds: float = 0.2
    reaction_delay_seconds: float = 0.2
    plan_horizon_seconds: float = 1.0
    candidate_representative_count: int = 24
    attacker_branch_directions: int = 8
    attacker_max_speed_mps: float = 9.0
    attacker_acceleration_mps2: float = 4.5
    coverage_radius_m: float = 2.0
    coverage_softness_m: float = 2.0
    observation_velocity_window_seconds: float = 0.3

    def validate(self) -> None:
        positive = (
            self.replan_interval_seconds,
            self.plan_horizon_seconds,
            self.attacker_max_speed_mps,
            self.attacker_acceleration_mps2,
            self.coverage_radius_m,
            self.coverage_softness_m,
            self.observation_velocity_window_seconds,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("causal-policy settings must be positive")
        if self.reaction_delay_seconds < 0.0:
            raise ValueError("reaction delay cannot be negative")
        if self.candidate_representative_count < 2:
            raise ValueError("need at least two candidate plans")
        if self.attacker_branch_directions < 3:
            raise ValueError("need at least three attacker branch directions")


def _observed_paths(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, tuple[TimedPoint, ...]]:
    rows: dict[str, list[TimedPoint]] = {}
    for frame in frames:
        time_s = float(frame["relative_time_s"])
        for player in frame["players"]:  # type: ignore[index]
            rows.setdefault(str(player[0]), []).append(
                (time_s, float(player[2]), float(player[3]))
            )
    return {
        player_id: tuple(sorted(samples))
        for player_id, samples in rows.items()
        if len(samples) >= 2
    }


def _observe(
    path: Sequence[TimedPoint],
    now: float,
    window: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Position and finite-difference velocity using samples at or before now."""

    clamped = min(float(now), float(path[-1][0]))
    position = interpolate_timed_point(path, clamped)
    earlier = interpolate_timed_point(path, max(float(path[0][0]), clamped - window))
    dt = clamped - max(float(path[0][0]), clamped - window)
    if dt <= 1e-9:
        return position, (0.0, 0.0)
    velocity = (
        (position[0] - earlier[0]) / dt,
        (position[1] - earlier[1]) / dt,
    )
    return position, velocity


def _reachable_distance(
    speed_toward: float,
    horizon: float,
    config: CausalPolicyConfig,
) -> float:
    speed = max(0.0, min(config.attacker_max_speed_mps, speed_toward))
    to_max = (config.attacker_max_speed_mps - speed) / config.attacker_acceleration_mps2
    if horizon <= to_max:
        return speed * horizon + 0.5 * config.attacker_acceleration_mps2 * horizon**2
    return (
        speed * to_max
        + 0.5 * config.attacker_acceleration_mps2 * to_max**2
        + config.attacker_max_speed_mps * (horizon - to_max)
    )


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def rollout_causal_defender_path(
    game: Mapping[str, object],
    defender_id: str,
    config: CausalPolicyConfig = CausalPolicyConfig(),
) -> tuple[TimedPoint, ...]:
    """Simulate the causal policy through the scene.

    Observed frames are read only at times up to each decision instant; the
    future of every other player enters the policy solely through constant
    velocity extrapolation of the last observation.
    """

    config.validate()
    frames: Sequence[Mapping[str, object]] = game["background_frames"]  # type: ignore[assignment]
    paths = _observed_paths(frames)
    horizon = float(game["horizon_seconds"])
    direction = int(game["attacking_direction"])
    attacking_team = str(game["attacking_team_id"])
    teams = {
        str(player[0]): str(player[1])
        for frame in frames[:1]
        for player in frame["players"]  # type: ignore[index]
    }
    option_ids = [
        str(option_id)
        for option_id in game["displayed_option_ids"]  # type: ignore[index]
    ]
    option_ids.append(str(game["runner_id"]))
    defender_path = paths[str(defender_id)]

    delay = config.reaction_delay_seconds
    interval = config.replan_interval_seconds

    # Reaction bootstrap: coast on the onset velocity until the first plan
    # can take effect.
    start_xy, start_velocity = _observe(
        defender_path, 0.0, config.observation_velocity_window_seconds
    )
    committed: list[TimedPoint] = [(0.0, start_xy[0], start_xy[1])]
    step = 0.1
    t = 0.0
    while t < delay - 1e-9:
        t = round(t + step, 10)
        committed.append(
            (
                min(t, horizon),
                start_xy[0] + start_velocity[0] * t,
                start_xy[1] + start_velocity[1] * t,
            )
        )
        if t >= horizon:
            return tuple(committed)
    committed_velocity = start_velocity

    decision_time = 0.0
    while committed[-1][0] < horizon - 1e-9:
        # State the defender will occupy when the new plan takes effect is
        # fully determined by what is already committed (causal).
        plan_start_time = committed[-1][0]
        plan_start_xy = (committed[-1][1], committed[-1][2])

        internal_horizon = min(
            config.plan_horizon_seconds, max(step, horizon - plan_start_time)
        )
        internal_horizon = math.floor(internal_horizon / step + 1e-9) * step
        internal_horizon = max(internal_horizon, 2 * step)
        search_config = DefenderTrajectorySearchConfig(
            planning_horizon_seconds=internal_horizon,
            response_delay_seconds=0.0,
            maximum_representatives=config.candidate_representative_count,
        )
        candidates = generate_feasible_defender_trajectories(
            plan_start_xy,
            committed_velocity,
            internal_horizon,
            search_config,
        )
        if not candidates:
            # Pinned against the boundary at speed the steered lattice can be
            # empty; hold position for one interval and re-plan.
            hold = min(interval, horizon - plan_start_time)
            committed.append(
                (
                    round(plan_start_time + hold, 10),
                    plan_start_xy[0],
                    plan_start_xy[1],
                )
            )
            committed_velocity = (0.0, 0.0)
            decision_time = round(decision_time + interval, 10)
            continue

        # Worst-case attacker branches from observations at the decision time.
        lookahead = (plan_start_time - decision_time) + internal_horizon
        branch_targets: list[tuple[float, float, float]] = []  # x, y, goal danger
        for option_id in option_ids:
            path = paths.get(option_id)
            if path is None:
                continue
            position, velocity = _observe(
                path, decision_time, config.observation_velocity_window_seconds
            )
            speed = math.hypot(*velocity)
            for branch in range(config.attacker_branch_directions):
                angle = 2.0 * math.pi * branch / config.attacker_branch_directions
                unit = (math.cos(angle), math.sin(angle))
                toward = max(0.0, velocity[0] * unit[0] + velocity[1] * unit[1])
                distance = _reachable_distance(toward, lookahead, config)
                target = (
                    position[0] + distance * unit[0],
                    position[1] + distance * unit[1],
                )
                danger = float(
                    score_at_points([target], direction, epv_grid_path=None)[0]
                )
                branch_targets.append((target[0], target[1], danger))
            if speed > 1e-6:
                # The observed heading is one of the branches only by
                # coincidence; include it explicitly so tracking the current
                # run is always represented.
                unit = (velocity[0] / speed, velocity[1] / speed)
                distance = _reachable_distance(speed, lookahead, config)
                target = (
                    position[0] + distance * unit[0],
                    position[1] + distance * unit[1],
                )
                danger = float(
                    score_at_points([target], direction, epv_grid_path=None)[0]
                )
                branch_targets.append((target[0], target[1], danger))

        # Team-mate defenders cover branches too (constant-velocity model).
        helper_positions: list[tuple[float, float]] = []
        for player_id, path in paths.items():
            if player_id == str(defender_id) or teams.get(player_id) == attacking_team:
                continue
            position, velocity = _observe(
                path, decision_time, config.observation_velocity_window_seconds
            )
            helper_positions.append(
                (
                    position[0] + velocity[0] * lookahead,
                    position[1] + velocity[1] * lookahead,
                )
            )

        def worst_threat(candidate) -> float:
            end_x, end_y = candidate.endpoint_x, candidate.endpoint_y
            worst = 0.0
            for target_x, target_y, danger in branch_targets:
                distance = math.hypot(end_x - target_x, end_y - target_y)
                for helper_x, helper_y in helper_positions:
                    distance = min(
                        distance,
                        math.hypot(helper_x - target_x, helper_y - target_y),
                    )
                exposure = _sigmoid(
                    (distance - config.coverage_radius_m)
                    / config.coverage_softness_m
                )
                worst = max(worst, danger * exposure)
            return worst

        best = min(
            candidates,
            key=lambda candidate: (
                worst_threat(candidate),
                candidate.effort_m2ps3,
                candidate.trajectory_id,
            ),
        )

        # Commit only the first replanning interval of the winning plan.
        commit_until = min(interval, internal_horizon, horizon - plan_start_time)
        if commit_until <= 1e-9:
            break
        for sample in best.path_txy:
            sample_time = float(sample[0])
            if 1e-9 < sample_time <= commit_until + 1e-9:
                committed.append(
                    (
                        round(plan_start_time + sample_time, 10),
                        float(sample[1]),
                        float(sample[2]),
                    )
                )
        if committed[-1][0] < plan_start_time + commit_until - 1e-9:
            # The final fragment is shorter than one integration step (scene
            # horizons need not align to the replanning grid); commit the
            # interpolated endpoint so the rollout always makes progress.
            fragment_xy = interpolate_timed_point(best.path_txy, commit_until)
            committed.append(
                (
                    round(plan_start_time + commit_until, 10),
                    float(fragment_xy[0]),
                    float(fragment_xy[1]),
                )
            )
        tail = interpolate_timed_point(best.path_txy, commit_until)
        just_before = interpolate_timed_point(
            best.path_txy, max(0.0, commit_until - step)
        )
        span = min(step, commit_until)
        committed_velocity = (
            (tail[0] - just_before[0]) / span,
            (tail[1] - just_before[1]) / span,
        ) if span > 1e-9 else committed_velocity
        decision_time = round(decision_time + interval, 10)

    return tuple(committed)
