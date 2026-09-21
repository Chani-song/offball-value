"""Structural defender-by-option screening for confirmed off-ball scenes.

This module deliberately stops before the final threat model.  It constructs
bounded, target-updating defender responses against observed attacker futures
and asks whether reallocating one defender from the runner to another option
creates a two-sided goal-side marking trade-off.

The observed future is used only for retrospective development-set auditing.
It must not be described as an online prediction or final counterfactual value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .dynamic_marking import (
    DynamicMarkingConfig,
    interpolate_timed_point,
    moving_goal_side_target,
    summarize_dynamic_marking,
)
from .steering_reachable import SteeringReachabilityConfig, steering_step


TimedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class LocalGameStructureConfig:
    """Pre-registered settings for the structural v0.1 audit."""

    maximum_horizon_seconds: float = 3.0
    minimum_horizon_seconds: float = 1.2
    integration_step_seconds: float = 0.1
    velocity_history_seconds: float = 0.4
    response_delay_seconds: float = 0.2
    lookahead_seconds: float = 0.4
    goal_side_offset_m: float = 1.5
    maximum_speed_mps: float = 9.0
    maximum_acceleration_mps2: float = 4.5
    maximum_deceleration_mps2: float = 6.0
    maximum_normal_acceleration_mps2: float = 6.0
    desired_velocity_time_constant_seconds: float = 0.35
    candidate_defender_count: int = 3
    displayed_option_count: int = 5

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name not in {"candidate_defender_count", "displayed_option_count"}
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("local-game structure parameters must be positive")
        if self.candidate_defender_count < 1 or self.displayed_option_count < 1:
            raise ValueError("candidate counts must be positive")
        if self.minimum_horizon_seconds > self.maximum_horizon_seconds:
            raise ValueError("minimum horizon cannot exceed maximum horizon")


def _frame_time(frame: Mapping[str, object]) -> float:
    return float(frame["relative_time_s"])


def _player_lookup(frame: Mapping[str, object]) -> dict[str, Sequence[object]]:
    return {str(player[0]): player for player in frame["players"]}  # type: ignore[index]


def _raw_observed_path(
    frames: Sequence[Mapping[str, object]], player_id: str
) -> tuple[TimedPoint, ...]:
    path = []
    for frame in frames:
        player = _player_lookup(frame).get(player_id)
        if player is not None:
            path.append((_frame_time(frame), float(player[2]), float(player[3])))
    if len(path) < 2:
        raise ValueError(f"player {player_id} does not have a usable observed path")
    return tuple(sorted(path))


def _sample_times(horizon_seconds: float, step_seconds: float) -> tuple[float, ...]:
    count = int(math.floor(horizon_seconds / step_seconds + 1e-9))
    values = [round(index * step_seconds, 10) for index in range(count + 1)]
    if horizon_seconds - values[-1] > 1e-8:
        values.append(float(horizon_seconds))
    return tuple(values)


def observed_future_path(
    frames: Sequence[Mapping[str, object]],
    player_id: str,
    horizon_seconds: float,
    step_seconds: float = 0.1,
) -> tuple[TimedPoint, ...]:
    """Resample one observed player future from onset to the audit horizon."""

    raw = _raw_observed_path(frames, player_id)
    return tuple(
        (time_s, *interpolate_timed_point(raw, time_s))
        for time_s in _sample_times(horizon_seconds, step_seconds)
    )


def estimate_onset_velocity(
    frames: Sequence[Mapping[str, object]],
    player_id: str,
    history_seconds: float = 0.4,
) -> tuple[float, float]:
    """Estimate causal onset velocity using only pre-onset observations."""

    raw = _raw_observed_path(frames, player_id)
    history = [row for row in raw if -history_seconds - 1e-9 <= row[0] <= 1e-9]
    if len(history) < 2:
        return 0.0, 0.0
    times = np.asarray([row[0] for row in history], dtype=float)
    design = np.column_stack([times, np.ones(len(times), dtype=float)])
    xs = np.asarray([row[1] for row in history], dtype=float)
    ys = np.asarray([row[2] for row in history], dtype=float)
    return (
        float(np.linalg.lstsq(design, xs, rcond=None)[0][0]),
        float(np.linalg.lstsq(design, ys, rcond=None)[0][0]),
    )


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def simulate_goal_side_response(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    actor_path_txy: Sequence[TimedPoint],
    goal_xy: tuple[float, float],
    horizon_seconds: float,
    config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> tuple[TimedPoint, ...]:
    """Generate a bounded pursuit path toward a moving goal-side target."""

    config.validate()
    dt = config.integration_step_seconds
    vx, vy = map(float, initial_velocity_xy)
    speed = min(config.maximum_speed_mps, math.hypot(vx, vy))
    if math.hypot(vx, vy) > 1e-9 and speed < math.hypot(vx, vy):
        scale = speed / math.hypot(vx, vy)
        vx *= scale
        vy *= scale
    actor_now = interpolate_timed_point(actor_path_txy, 0.0)
    heading = (
        math.atan2(vy, vx)
        if speed > 0.1
        else math.atan2(actor_now[1] - start_xy[1], actor_now[0] - start_xy[0])
    )
    x, y = map(float, start_xy)
    path: list[TimedPoint] = []
    steering_config = SteeringReachabilityConfig(
        horizon_seconds=max(dt, horizon_seconds),
        integration_step_seconds=dt,
        max_speed_mps=config.maximum_speed_mps,
        max_tangential_acceleration_mps2=config.maximum_acceleration_mps2,
        max_tangential_deceleration_mps2=config.maximum_deceleration_mps2,
        max_normal_acceleration_mps2=config.maximum_normal_acceleration_mps2,
        path_sample_seconds=(max(dt, horizon_seconds),),
    )
    times = _sample_times(horizon_seconds, dt)
    for index, time_s in enumerate(times):
        path.append((float(time_s), float(x), float(y)))
        if index == len(times) - 1:
            break
        step_duration = times[index + 1] - time_s
        if time_s + 1e-9 < config.response_delay_seconds:
            tangential = 0.0
            normal = 0.0
        else:
            anticipated_time = min(
                horizon_seconds, time_s + config.lookahead_seconds
            )
            anticipated_actor = interpolate_timed_point(
                actor_path_txy, anticipated_time
            )
            target_x, target_y = moving_goal_side_target(
                anticipated_actor, goal_xy, config.goal_side_offset_m
            )
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)
            desired_heading = math.atan2(dy, dx) if distance > 1e-9 else heading
            desired_speed = min(
                config.maximum_speed_mps,
                distance / max(config.desired_velocity_time_constant_seconds, dt),
            )
            if speed <= 0.1:
                heading = desired_heading
                tangential = config.maximum_acceleration_mps2
                normal = 0.0
            else:
                tangential = float(
                    np.clip(
                        (desired_speed - speed)
                        / config.desired_velocity_time_constant_seconds,
                        -config.maximum_deceleration_mps2,
                        config.maximum_acceleration_mps2,
                    )
                )
                angle_error = _wrap_angle(desired_heading - heading)
                requested_normal = angle_error * speed / max(dt, 1e-9)
                normal = float(
                    np.clip(
                        requested_normal,
                        -config.maximum_normal_acceleration_mps2,
                        config.maximum_normal_acceleration_mps2,
                    )
                )
        if abs(step_duration - dt) > 1e-8:
            local_config = SteeringReachabilityConfig(
                horizon_seconds=step_duration,
                integration_step_seconds=step_duration,
                max_speed_mps=config.maximum_speed_mps,
                max_tangential_acceleration_mps2=config.maximum_acceleration_mps2,
                max_tangential_deceleration_mps2=config.maximum_deceleration_mps2,
                max_normal_acceleration_mps2=config.maximum_normal_acceleration_mps2,
                path_sample_seconds=(step_duration,),
            )
        else:
            local_config = steering_config
        step = steering_step(x, y, speed, heading, tangential, normal, local_config)
        x, y = step.x, step.y
        speed, heading = step.speed_mps, step.heading_radians
    return tuple(path)


def _mean_marking_cost(
    actor_path: Sequence[TimedPoint],
    defender_path: Sequence[TimedPoint],
    goal_xy: tuple[float, float],
    config: LocalGameStructureConfig,
) -> float:
    marking = DynamicMarkingConfig(
        goal_side_offset_m=config.goal_side_offset_m,
        lookahead_seconds=config.lookahead_seconds,
        response_delay_seconds=config.response_delay_seconds,
    )
    times = _sample_times(
        min(actor_path[-1][0], defender_path[-1][0]),
        config.integration_step_seconds,
    )
    return float(
        summarize_dynamic_marking(
            actor_path,
            defender_path,
            goal_xy,
            marking,
            evaluation_times_s=times,
        ).mean_weighted_error_m
    )


def _onset_frame(scene: Mapping[str, object]) -> Mapping[str, object]:
    frames = scene["frames"]  # type: ignore[index]
    return min(frames, key=lambda frame: abs(_frame_time(frame)))


def _available_future_seconds(scene: Mapping[str, object]) -> float:
    return max(_frame_time(frame) for frame in scene["frames"])  # type: ignore[index]


def _display_name(player: Sequence[object]) -> str:
    return str(player[4]) if len(player) > 4 and player[4] else str(player[0])


def _goalkeeper_by_depth(
    player_ids: Sequence[str],
    players: Mapping[str, Sequence[object]],
    attacking_direction: int,
    *,
    attacking_team: bool,
) -> str | None:
    if not player_ids:
        return None
    # The attacking goalkeeper is nearest the team's own goal (minimum attack
    # progress); the defending goalkeeper is nearest the attacked goal.
    key = lambda player_id: attacking_direction * float(players[player_id][2])
    return min(player_ids, key=key) if attacking_team else max(player_ids, key=key)


def build_structural_local_game(
    scene: Mapping[str, object],
    config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> dict[str, object]:
    """Build defender rows and conditional affected-option cells for one scene."""

    config.validate()
    frames: Sequence[Mapping[str, object]] = scene["frames"]  # type: ignore[assignment]
    onset = _onset_frame(scene)
    players = _player_lookup(onset)
    runner_id = str(scene["runner_id"])
    carrier_id = str(scene["carrier_id"])
    attacking_team_id = str(scene["team_id"])
    attacking_direction = int(scene["attacking_direction"])
    horizon = min(
        config.maximum_horizon_seconds,
        float(scene["seconds_before_shot"]),
        _available_future_seconds(scene),
    )
    if horizon < config.minimum_horizon_seconds:
        raise ValueError(
            f"scene {scene['onset_frame_id']} has only {horizon:.2f}s of usable future"
        )
    goal_xy = (52.5 * attacking_direction, 0.0)

    attacking_ids = [
        player_id
        for player_id, player in players.items()
        if str(player[1]) == attacking_team_id
    ]
    defending_ids = [
        player_id
        for player_id, player in players.items()
        if str(player[1]) != attacking_team_id
    ]
    attacking_goalkeeper = _goalkeeper_by_depth(
        attacking_ids, players, attacking_direction, attacking_team=True
    )
    defending_goalkeeper = _goalkeeper_by_depth(
        defending_ids, players, attacking_direction, attacking_team=False
    )
    option_ids = [
        player_id
        for player_id in attacking_ids
        if player_id not in {runner_id, attacking_goalkeeper}
    ]
    outfield_defenders = [
        player_id for player_id in defending_ids if player_id != defending_goalkeeper
    ]
    runner_path = observed_future_path(
        frames, runner_id, horizon, config.integration_step_seconds
    )
    option_paths = {
        player_id: observed_future_path(
            frames, player_id, horizon, config.integration_step_seconds
        )
        for player_id in option_ids
    }

    defender_rows = []
    for defender_id in outfield_defenders:
        defender_start = players[defender_id]
        start_xy = (float(defender_start[2]), float(defender_start[3]))
        velocity = estimate_onset_velocity(
            frames, defender_id, config.velocity_history_seconds
        )
        actual_path = observed_future_path(
            frames, defender_id, horizon, config.integration_step_seconds
        )
        runner_response = simulate_goal_side_response(
            start_xy, velocity, runner_path, goal_xy, horizon, config
        )
        runner_actual_cost = _mean_marking_cost(
            runner_path, actual_path, goal_xy, config
        )
        runner_cover_cost = _mean_marking_cost(
            runner_path, runner_response, goal_xy, config
        )
        cells = []
        for option_id, option_path in option_paths.items():
            option_response = simulate_goal_side_response(
                start_xy, velocity, option_path, goal_xy, horizon, config
            )
            option_under_runner = _mean_marking_cost(
                option_path, runner_response, goal_xy, config
            )
            option_under_cover = _mean_marking_cost(
                option_path, option_response, goal_xy, config
            )
            runner_under_option = _mean_marking_cost(
                runner_path, option_response, goal_xy, config
            )
            option_release = max(0.0, option_under_runner - option_under_cover)
            runner_release = max(0.0, runner_under_option - runner_cover_cost)
            option_fraction = option_release / max(option_under_runner, 1e-6)
            runner_fraction = runner_release / max(runner_under_option, 1e-6)
            option_start = players[option_id]
            cells.append(
                {
                    "option_id": option_id,
                    "option_name": _display_name(option_start),
                    "option_type": (
                        "carrier_carry" if option_id == carrier_id else "teammate_option"
                    ),
                    "option_start_xy": [
                        float(option_start[2]),
                        float(option_start[3]),
                    ],
                    "option_path_txy": [list(row) for row in option_path],
                    "option_response_path_txy": [
                        list(row) for row in option_response
                    ],
                    "option_cost_under_runner_response_m": option_under_runner,
                    "option_best_cover_cost_m": option_under_cover,
                    "option_allocation_effect_m": option_release,
                    "option_allocation_effect_fraction": option_fraction,
                    "runner_cost_under_option_response_m": runner_under_option,
                    "runner_cross_cost_m": runner_release,
                    "runner_cross_cost_fraction": runner_fraction,
                    "structural_tradeoff_score": min(
                        option_fraction, runner_fraction
                    ),
                }
            )
        distance_to_runner = math.hypot(
            start_xy[0] - runner_path[0][1], start_xy[1] - runner_path[0][2]
        )
        defender_rows.append(
            {
                "defender_id": defender_id,
                "defender_name": _display_name(defender_start),
                "start_xy": list(start_xy),
                "current_distance_to_runner_m": distance_to_runner,
                "actual_runner_marking_cost_m": runner_actual_cost,
                "runner_response_cost_m": runner_cover_cost,
                "actual_to_runner_response_improvement_m": max(
                    0.0, runner_actual_cost - runner_cover_cost
                ),
                "actual_path_txy": [list(row) for row in actual_path],
                "runner_response_path_txy": [list(row) for row in runner_response],
                "cells": sorted(
                    cells,
                    key=lambda cell: (
                        -float(cell["structural_tradeoff_score"]),
                        -float(cell["option_allocation_effect_fraction"]),
                        str(cell["option_name"]),
                    ),
                ),
            }
        )
    defender_rows.sort(
        key=lambda row: (
            float(row["runner_response_cost_m"]),
            float(row["current_distance_to_runner_m"]),
            str(row["defender_name"]),
        )
    )
    selected_defenders = defender_rows[: config.candidate_defender_count]

    maximum_by_option: dict[str, float] = {}
    for row in selected_defenders:
        for cell in row["cells"]:  # type: ignore[index]
            option_id = str(cell["option_id"])
            maximum_by_option[option_id] = max(
                maximum_by_option.get(option_id, 0.0),
                float(cell["structural_tradeoff_score"]),
            )
    ranked_options = sorted(
        maximum_by_option,
        key=lambda option_id: (
            -maximum_by_option[option_id],
            option_id != carrier_id,
            str(players[option_id][4]),
        ),
    )
    displayed = ranked_options[: config.displayed_option_count]
    if carrier_id in option_ids and carrier_id not in displayed:
        displayed = ([carrier_id] + displayed)[: config.displayed_option_count]

    return {
        "schema_version": "structural-local-game-v0.1",
        "match_id": str(scene["match_id"]),
        "match_label": str(scene["match_label"]),
        "onset_frame_id": int(scene["onset_frame_id"]),
        "runner_id": runner_id,
        "runner_name": str(scene["runner_name"]),
        "carrier_id": carrier_id,
        "carrier_name": str(scene["carrier_name"]),
        "attacking_team_id": attacking_team_id,
        "attacking_direction": attacking_direction,
        "horizon_seconds": float(horizon),
        "goal_xy": list(goal_xy),
        "runner_path_txy": [list(row) for row in runner_path],
        "candidate_defenders": selected_defenders,
        "displayed_option_ids": displayed,
        "all_option_ids": ranked_options,
        "background_frames": list(frames),
        "human_review": dict(scene.get("core_review", {})),
        "notes": [
            "Observed attacker futures are used only for retrospective structural auditing.",
            "Matrix cells are goal-side geometry allocation effects, not P×G×A threat.",
            "Candidate defenders are the three players whose bounded responses best control the observed runner future.",
            "The carrier is retained in the displayed local option set even when not top-ranked.",
        ],
        "config": asdict(config),
    }


def build_structural_local_games(
    scenes: Sequence[Mapping[str, object]],
    config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> list[dict[str, object]]:
    """Build structural games in the human-confirmed scene order."""

    return [build_structural_local_game(scene, config) for scene in scenes]
