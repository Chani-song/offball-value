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
from .pair_plausibility import PairGateConfig, gate_pairs, nearest_to_ball


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
    desired_velocity_time_constant_seconds: float = 0.45
    heading_time_constant_seconds: float = 0.25
    target_velocity_sample_seconds: float = 0.2
    candidate_defender_count: int = 3
    # Defenders kept by pursuit cost on top of the goal-side ordering. Being
    # responsible for a runner (goal-side and close) and being able to reach
    # him are two different routes to facing the choice, and ranking by the
    # first alone drops a defender who stands further out but closes fast -
    # 151867's Bormuth is fourth on goal-side distance and second on pursuit
    # cost, and the reviewer had marked him as facing the dilemma. 0 disables.
    pursuit_union_count: int = 2
    # Plausibility gate on (runner, defender) pairs. Ranking by goal-side
    # marking distance answers "who is nearest the runner at onset", which
    # is not the same question as "who has to deal with this run": the run
    # itself decides that. A candidate survives if he is the first choice,
    # if the run passes within pair_gate_neighbour_delta_m of him of the
    # closest candidate's own closest approach, or if the run comes to him;
    # the latter two also require him to have stood within pair_gate_near_m
    # of the runner at onset, since a man further out is the covering line.
    # Nothing here reads the defender's own movement -- that is his response,
    # and the dilemmas worth finding include the ones where he froze.
    #
    # Scored against 188 candidate verdicts over 58 scenes: 93% agreement,
    # 84 of the 85 candidates the reviewer called responsible kept, and
    # 1,931 -> 1,060 candidates on the v7 build. See pair_plausibility.
    pair_gate: bool = True
    pair_gate_neighbour_delta_m: float = 3.0
    pair_gate_catch_margin_m: float = 3.0
    pair_gate_catch_end_m: float = 11.0
    pair_gate_near_m: float = 15.0
    # The defender nearest the ball is refused (unless ranked first) when the
    # run stays in front of him: his job is the carrier, and a run that does
    # not go past him never makes him choose. See pair_plausibility.
    pair_gate_drop_ball_nearest_in_front: bool = True
    displayed_option_count: int = 5
    # What the pursuing defender is allowed to know about where the actor is
    # going.  "kinematic" extrapolates the speed observed up to now and
    # re-plans every step; "clairvoyant" reads the actor's real future, which
    # is what this module did before and is kept only as an upper reference.
    pursuit_information: str = "kinematic"
    actor_velocity_history_seconds: float = 0.2
    # Human labels are EVALUATION DATA, not model input. Feeding a scene's
    # confirmed_defender_ids / confirmed_derived_ids back into the pipeline
    # makes the reviewer's own answer part of the answer, so the artifact can
    # no longer show whether the model stands on its own. Default False: the
    # pipeline ignores those manifest fields and the labels are carried
    # through untouched for side-by-side comparison only. Set True solely to
    # reproduce a historical pinned artifact.
    honor_human_pins: bool = False

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name
            not in {
                "candidate_defender_count",
                "pursuit_union_count",
                "displayed_option_count",
                "honor_human_pins",
                "pursuit_information",
                "pair_gate",
                "pair_gate_drop_ball_nearest_in_front",
            }
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("local-game structure parameters must be positive")
        if self.candidate_defender_count < 1 or self.displayed_option_count < 1:
            raise ValueError("candidate counts must be positive")
        if self.minimum_horizon_seconds > self.maximum_horizon_seconds:
            raise ValueError("minimum horizon cannot exceed maximum horizon")
        if self.pursuit_information not in {"kinematic", "clairvoyant"}:
            raise ValueError(
                "pursuit_information must be 'kinematic' or 'clairvoyant'"
            )


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


def simulate_goal_side_response_trace(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    actor_path_txy: Sequence[TimedPoint],
    goal_xy: tuple[float, float],
    horizon_seconds: float,
    config: LocalGameStructureConfig = LocalGameStructureConfig(),
    actor_initial_velocity_xy: tuple[float, float] | None = None,
) -> tuple[tuple[TimedPoint, ...], tuple[TimedPoint, ...]]:
    """Track a moving goal-side target with bounded, arrival-aware steering.

    The target at response time ``t`` is the goal-side point of the actor's
    anticipated position at ``t + lookahead``.  Unlike a pure point-pursuit
    controller, the desired defender velocity also contains the target's own
    velocity.  This prevents the defender from accelerating into the target,
    stopping there, and then curling past it while the target keeps moving.

    The returned target trace is deliberately part of the audit contract: a
    reviewer must be able to distinguish a bad tactical target from a bad
    physical tracking controller.
    """

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
    target_path: list[TimedPoint] = []
    steering_config = SteeringReachabilityConfig(
        horizon_seconds=max(dt, horizon_seconds),
        integration_step_seconds=dt,
        max_speed_mps=config.maximum_speed_mps,
        max_tangential_acceleration_mps2=config.maximum_acceleration_mps2,
        max_tangential_deceleration_mps2=config.maximum_deceleration_mps2,
        max_normal_acceleration_mps2=config.maximum_normal_acceleration_mps2,
        path_sample_seconds=(max(dt, horizon_seconds),),
    )
    def believe(now_s: float, ahead_s: float) -> tuple[float, float]:
        """Where the defender BELIEVES the actor will be ``lead_s`` from now.

        Reading the actor's real future makes the defender clairvoyant, and
        86% of the runs in this project are direction changes — precisely the
        event a real defender cannot see coming.  A marking cost computed that
        way answers "what could a defender who knew the future do", not "what
        could a defender do", so it is not the quantity the name claims.

        Under ``kinematic`` the defender extrapolates at the speed he has
        observed up to ``now_s`` and re-plans every integration step, so the
        prediction is wrong exactly when the runner cuts, and is corrected
        only after the cut becomes visible.  Marking error is still scored
        against the actor's REAL position: he acts on belief and is judged on
        reality.

        ``clairvoyant`` restores the previous behaviour as an upper reference.
        """
        if config.pursuit_information == "clairvoyant":
            return interpolate_timed_point(
                actor_path_txy, min(horizon_seconds, now_s + ahead_s)
            )
        here = interpolate_timed_point(actor_path_txy, now_s)
        window = min(config.actor_velocity_history_seconds, now_s)
        if window <= 1e-9:
            actor_vx, actor_vy = actor_initial_velocity_xy or (0.0, 0.0)
        else:
            earlier = interpolate_timed_point(actor_path_txy, now_s - window)
            actor_vx = (here[0] - earlier[0]) / window
            actor_vy = (here[1] - earlier[1]) / window
        actor_speed = math.hypot(actor_vx, actor_vy)
        if actor_speed > config.maximum_speed_mps:
            scale = config.maximum_speed_mps / actor_speed
            actor_vx *= scale
            actor_vy *= scale
        return (here[0] + actor_vx * ahead_s, here[1] + actor_vy * ahead_s)

    times = _sample_times(horizon_seconds, dt)
    for index, time_s in enumerate(times):
        path.append((float(time_s), float(x), float(y)))
        anticipated_actor = believe(time_s, config.lookahead_seconds)
        target_x, target_y = moving_goal_side_target(
            anticipated_actor, goal_xy, config.goal_side_offset_m
        )
        target_path.append((float(time_s), float(target_x), float(target_y)))
        if index == len(times) - 1:
            break
        step_duration = times[index + 1] - time_s
        if time_s + 1e-9 < config.response_delay_seconds:
            tangential = 0.0
            normal = 0.0
        else:
            velocity_sample_time = min(
                horizon_seconds,
                time_s + config.target_velocity_sample_seconds,
            )
            velocity_sample_duration = velocity_sample_time - time_s
            # The target's own velocity, as the defender believes it: the same
            # prediction advanced one sample, so the steering term cannot see
            # further ahead than the target point it is chasing.
            future_actor = believe(
                time_s, velocity_sample_duration + config.lookahead_seconds
            )
            future_target_x, future_target_y = moving_goal_side_target(
                future_actor, goal_xy, config.goal_side_offset_m
            )
            dx, dy = target_x - x, target_y - y
            if velocity_sample_duration > 1e-9:
                target_vx = (
                    future_target_x - target_x
                ) / velocity_sample_duration
                target_vy = (
                    future_target_y - target_y
                ) / velocity_sample_duration
            else:
                target_vx = 0.0
                target_vy = 0.0
            desired_vx = target_vx + (
                dx / config.desired_velocity_time_constant_seconds
            )
            desired_vy = target_vy + (
                dy / config.desired_velocity_time_constant_seconds
            )
            desired_speed_unclipped = math.hypot(desired_vx, desired_vy)
            desired_speed = min(
                config.maximum_speed_mps,
                desired_speed_unclipped,
            )
            desired_heading = (
                math.atan2(desired_vy, desired_vx)
                if desired_speed_unclipped > 1e-9
                else heading
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
                requested_normal = (
                    angle_error
                    * speed
                    / config.heading_time_constant_seconds
                )
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
    return tuple(path), tuple(target_path)


def simulate_goal_side_response(
    start_xy: tuple[float, float],
    initial_velocity_xy: tuple[float, float],
    actor_path_txy: Sequence[TimedPoint],
    goal_xy: tuple[float, float],
    horizon_seconds: float,
    config: LocalGameStructureConfig = LocalGameStructureConfig(),
) -> tuple[TimedPoint, ...]:
    """Generate the response path while preserving the original public API."""

    response_path, _ = simulate_goal_side_response_trace(
        start_xy,
        initial_velocity_xy,
        actor_path_txy,
        goal_xy,
        horizon_seconds,
        config,
    )
    return response_path


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

    # Onset velocities of the tracked actors, measured on PRE-onset frames
    # only, so a kinematic pursuer has a causal starting estimate at t=0.
    runner_onset_velocity = estimate_onset_velocity(
        frames, runner_id, config.velocity_history_seconds
    )
    option_onset_velocities = {
        player_id: estimate_onset_velocity(
            frames, player_id, config.velocity_history_seconds
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
        runner_response, runner_response_target = simulate_goal_side_response_trace(
            start_xy,
            velocity,
            runner_path,
            goal_xy,
            horizon,
            config,
            runner_onset_velocity,
        )
        runner_actual_cost = _mean_marking_cost(
            runner_path, actual_path, goal_xy, config
        )
        runner_cover_cost = _mean_marking_cost(
            runner_path, runner_response, goal_xy, config
        )
        cells = []
        for option_id, option_path in option_paths.items():
            option_response, option_response_target = simulate_goal_side_response_trace(
                start_xy,
                velocity,
                option_path,
                goal_xy,
                horizon,
                config,
                option_onset_velocities.get(option_id),
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
                    "option_response_target_path_txy": [
                        list(row) for row in option_response_target
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
        # How far this defender stands, at onset, from the position that would
        # actually be marking the runner: goal-side of him by the offset.  In
        # football the defender who has to decide about a run is the one who
        # was already responsible for that player, and responsibility is
        # proximity AND being on the goal side - a defender behind the runner
        # has already been beaten and it is no longer his call.
        #
        # This replaces the pursuit cost as the candidate ordering.  Pursuit
        # cost asks "if he chased, how well would he stick", which rewards a
        # quick defender fifteen metres away over the man actually marking,
        # and that is not how assignments work.  Measured on 11 scenes with
        # every outfield defender labelled: this ranks the reactor first in
        # 10 of 10 against 8 of 10 for pursuit cost, using only the onset
        # instant, while the pursuit cost keeps its own meaning downstream.
        goal_side_x, goal_side_y = moving_goal_side_target(
            (runner_path[0][1], runner_path[0][2]),
            goal_xy,
            config.goal_side_offset_m,
        )
        goal_side_marking_distance = math.hypot(
            start_xy[0] - goal_side_x, start_xy[1] - goal_side_y
        )
        defender_rows.append(
            {
                "defender_id": defender_id,
                "defender_name": _display_name(defender_start),
                "start_xy": list(start_xy),
                "current_distance_to_runner_m": distance_to_runner,
                "goal_side_marking_distance_m": goal_side_marking_distance,
                "actual_runner_marking_cost_m": runner_actual_cost,
                "runner_response_cost_m": runner_cover_cost,
                "actual_to_runner_response_improvement_m": max(
                    0.0, runner_actual_cost - runner_cover_cost
                ),
                "actual_path_txy": [list(row) for row in actual_path],
                "runner_response_path_txy": [list(row) for row in runner_response],
                "runner_response_target_path_txy": [
                    list(row) for row in runner_response_target
                ],
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
            float(row["goal_side_marking_distance_m"]),
            float(row["runner_response_cost_m"]),
            str(row["defender_name"]),
        )
    )
    confirmed_defender_ids = (
        [
            str(defender_id)
            for defender_id in scene.get("confirmed_defender_ids", [])
        ]
        if config.honor_human_pins
        else []
    )
    if confirmed_defender_ids:
        # Human review pinned the reacting defender(s) for this scene; the
        # runner-control ranking stays as a diagnostic order only.
        rows_by_id = {str(row["defender_id"]): row for row in defender_rows}
        missing = [
            defender_id
            for defender_id in confirmed_defender_ids
            if defender_id not in rows_by_id
        ]
        if missing:
            raise ValueError(
                f"confirmed defender ids not on the pitch: {missing}"
            )
        selected_defenders = [
            rows_by_id[defender_id] for defender_id in confirmed_defender_ids
        ]
    else:
        selected_defenders = defender_rows[: config.candidate_defender_count]
        # Union in the defenders who could reach the runner quickest, even if
        # they stand further from the marking position. See pursuit_union_count.
        if config.pursuit_union_count > 0:
            chosen = {str(row["defender_id"]) for row in selected_defenders}
            by_pursuit = sorted(
                defender_rows,
                key=lambda row: (
                    float(row["runner_response_cost_m"]),
                    str(row["defender_name"]),
                ),
            )
            for row in by_pursuit[: config.pursuit_union_count]:
                if str(row["defender_id"]) not in chosen:
                    selected_defenders = [*selected_defenders, row]
                    chosen.add(str(row["defender_id"]))

    gate_rows: list[dict[str, object]] = []
    if config.pair_gate and not confirmed_defender_ids:
        gate_config = PairGateConfig(
            neighbour_delta_m=config.pair_gate_neighbour_delta_m,
            catch_margin_m=config.pair_gate_catch_margin_m,
            catch_end_m=config.pair_gate_catch_end_m,
            near_m=config.pair_gate_near_m,
            drop_ball_nearest_in_front=config.pair_gate_drop_ball_nearest_in_front,
        )
        # Over every defender, keeper included, not just the candidates: the
        # man whose job is the ball may not be one of them.
        ball_nearest = nearest_to_ball(
            {
                str(pid): (float(players[pid][2]), float(players[pid][3]))
                for pid in defending_ids
                if pid in players
            },
            (float(onset["ball"][0]), float(onset["ball"][1])),
        )
        gate_rows = gate_pairs(
            runner_path,
            [
                (
                    str(row["defender_id"]),
                    (float(row["start_xy"][0]), float(row["start_xy"][1])),
                    rank,
                )
                for rank, row in enumerate(selected_defenders)
            ],
            gate_config,
            goal_xy=goal_xy,
            ball_nearest_id=ball_nearest,
        )
        verdict = {str(row["defender_id"]): row for row in gate_rows}
        for row in selected_defenders:
            call = verdict.get(str(row["defender_id"]))
            if call is None:
                continue
            row["pair_gate_kept"] = bool(call["kept"])
            row["pair_gate_reason"] = str(call["reason"])
            for name in ("path_min_m", "path_end_m", "catch_m", "delta_m",
                         "goal_side_m"):
                row[f"pair_gate_{name}"] = float(call[name])
            row["pair_gate_is_ball_nearest"] = bool(call["is_ball_nearest"])
    # The attacker's option set is a property of the attack, not of which
    # defenders we chose to model, so it is ranked over the candidates as
    # they stood BEFORE the gate. Keeping it fixed also means a build with
    # the gate is the pre-gate build minus some defender games, rather than
    # a different game with different options and different beneficiaries.
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

    # Only the survivors get a payoff game built, which is where the saving
    # is: the dropped candidates are the majority of the cost.
    if gate_rows:
        selected_defenders = [
            row for row in selected_defenders if row.get("pair_gate_kept", True)
        ]

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
        # Only surfaced to the payoff stage when pins are explicitly honoured;
        # otherwise the labels travel as `human_labels` for comparison only.
        "confirmed_derived_ids": (
            {
                str(key): str(value)
                for key, value in dict(
                    scene.get("confirmed_derived_ids", {})
                ).items()
            }
            if config.honor_human_pins
            else {}
        ),
        "human_labels": {
            "derived_by_defender": {
                str(key): str(value)
                for key, value in dict(
                    scene.get("confirmed_derived_ids", {})
                ).items()
            },
            "defender_ids": [
                str(defender_id)
                for defender_id in scene.get("confirmed_defender_ids", [])
            ],
            "note": "evaluation labels — never an input to the model",
        },
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
