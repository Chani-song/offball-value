"""Build horizon game states from endpoint actions without future leakage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
import math
from typing import Iterable, Mapping

from .action_space import (
    EndpointAction,
    EndpointActionConfig,
    PlayerEndpointActionSet,
)
from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    BundesligaFrame,
    BundesligaObjectState,
)
from .pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    estimate_frame_velocities,
)


@dataclass(frozen=True)
class CounterfactualState:
    frame: BundesligaFrame
    velocities: Mapping[str, VelocityEstimate]
    focal_player_id: str
    action_id: str
    ball_carrier_id: str
    boundary_clipped_player_ids: tuple[str, ...]
    used_infeasible_reference: bool


def _cap_velocity(
    velocity: VelocityEstimate,
    maximum_speed_mps: float,
) -> VelocityEstimate:
    speed = math.hypot(velocity.vx, velocity.vy)
    if speed <= maximum_speed_mps or speed <= 1e-12:
        return replace(velocity, speed=float(speed))
    scale = maximum_speed_mps / speed
    return replace(
        velocity,
        vx=float(velocity.vx * scale),
        vy=float(velocity.vy * scale),
        speed=float(maximum_speed_mps),
    )


def _clip_constant_velocity_endpoint(
    player: BundesligaObjectState,
    velocity: VelocityEstimate,
    horizon_seconds: float,
) -> tuple[BundesligaObjectState, VelocityEstimate, bool]:
    raw_x = player.x + velocity.vx * horizon_seconds
    raw_y = player.y + velocity.vy * horizon_seconds
    half_length = FIELD_LENGTH / 2.0
    half_width = FIELD_WIDTH / 2.0
    x = min(half_length, max(-half_length, raw_x))
    y = min(half_width, max(-half_width, raw_y))
    clipped_x = not math.isclose(x, raw_x, abs_tol=1e-9)
    clipped_y = not math.isclose(y, raw_y, abs_tol=1e-9)
    terminal_vx = 0.0 if clipped_x else velocity.vx
    terminal_vy = 0.0 if clipped_y else velocity.vy
    terminal_speed = math.hypot(terminal_vx, terminal_vy)
    propagated = replace(
        player,
        x=float(x),
        y=float(y),
        speed=float(terminal_speed * 3.6),
    )
    propagated_velocity = replace(
        velocity,
        vx=float(terminal_vx),
        vy=float(terminal_vy),
        speed=float(terminal_speed),
    )
    return propagated, propagated_velocity, bool(clipped_x or clipped_y)


def build_attacker_counterfactual_state(
    frame: BundesligaFrame,
    history_frames: Iterable[BundesligaFrame],
    ball_carrier_id: str,
    action_set: PlayerEndpointActionSet,
    action: EndpointAction,
    config: EndpointActionConfig = EndpointActionConfig(),
    allow_infeasible_reference: bool = False,
) -> CounterfactualState:
    """Propagate all players to the horizon, replacing one attacker action.

    Non-focal players follow capped constant velocity estimated only from the
    pre-decision history. The ball preserves its current offset from the
    propagated carrier. An infeasible named reference may be built only with
    an explicit opt-in; optimizer actions must always be feasible.
    """

    config.validate()
    if action.player_id != action_set.player_id:
        raise ValueError("Action and action set refer to different players")
    if action.player_id not in frame.players:
        raise KeyError(f"Focal player {action.player_id} is missing")
    if ball_carrier_id not in frame.players:
        raise KeyError(f"Ball carrier {ball_carrier_id} is missing")
    if not action.motion.feasible and not allow_infeasible_reference:
        raise ValueError(
            f"Action {action.action_id} is infeasible: {action.motion.failure_reason}"
        )
    if not action.motion.feasible and "grid" in action.labels:
        raise ValueError("An infeasible grid action cannot define a counterfactual state")

    ordered_history = tuple(sorted(history_frames, key=lambda item: item.frame_id))
    velocity_config = ArrivalModelConfig(
        history_seconds=config.velocity_history_seconds,
        player_max_speed_mps=config.max_speed_mps,
        player_acceleration_mps2=config.max_acceleration_mps2,
        observed_speed_cap_mps=config.max_speed_mps,
    )
    estimated = estimate_frame_velocities(
        ordered_history,
        frame.frame_id,
        velocity_config,
    )
    zero_velocity = VelocityEstimate(
        0.0,
        0.0,
        0.0,
        0,
        config.velocity_history_seconds,
    )
    propagated_players: dict[str, BundesligaObjectState] = {}
    propagated_velocities: dict[str, VelocityEstimate] = {}
    clipped_ids: list[str] = []
    for player_id, player in frame.players.items():
        velocity = _cap_velocity(
            estimated.get(player_id, zero_velocity),
            config.max_speed_mps,
        )
        propagated, terminal_velocity, clipped = _clip_constant_velocity_endpoint(
            player,
            velocity,
            config.horizon_seconds,
        )
        propagated_players[player_id] = propagated
        propagated_velocities[player_id] = terminal_velocity
        if clipped:
            clipped_ids.append(player_id)

    focal = frame.players[action.player_id]
    if action.motion.feasible:
        focal_velocity = VelocityEstimate(
            vx=float(action.motion.terminal_vx_mps),
            vy=float(action.motion.terminal_vy_mps),
            speed=float(action.motion.terminal_speed_mps),
            sample_count=action_set.velocity_sample_count,
            window_seconds=action_set.velocity_window_seconds,
        )
    else:
        # This branch is diagnostic-only for an explicitly requested hold or
        # observed reference that violates the shared physical limits.
        focal_velocity = zero_velocity
    propagated_players[action.player_id] = replace(
        focal,
        x=float(action.endpoint_x),
        y=float(action.endpoint_y),
        speed=float(focal_velocity.speed * 3.6),
    )
    propagated_velocities[action.player_id] = focal_velocity
    if action.player_id in clipped_ids:
        clipped_ids.remove(action.player_id)

    propagated_ball = frame.ball
    if frame.ball is not None:
        original_carrier = frame.players[ball_carrier_id]
        propagated_carrier = propagated_players[ball_carrier_id]
        offset_x = frame.ball.x - original_carrier.x
        offset_y = frame.ball.y - original_carrier.y
        ball_x = min(
            FIELD_LENGTH / 2.0,
            max(-FIELD_LENGTH / 2.0, propagated_carrier.x + offset_x),
        )
        ball_y = min(
            FIELD_WIDTH / 2.0,
            max(-FIELD_WIDTH / 2.0, propagated_carrier.y + offset_y),
        )
        carrier_velocity = propagated_velocities[ball_carrier_id]
        propagated_ball = replace(
            frame.ball,
            x=float(ball_x),
            y=float(ball_y),
            speed=float(carrier_velocity.speed * 3.6),
        )

    horizon_frames = int(round(config.horizon_seconds * FPS))
    propagated_frame = replace(
        frame,
        frame_id=frame.frame_id + horizon_frames,
        timestamp=(
            frame.timestamp + timedelta(seconds=config.horizon_seconds)
            if frame.timestamp is not None
            else None
        ),
        players=propagated_players,
        ball=propagated_ball,
    )
    return CounterfactualState(
        frame=propagated_frame,
        velocities=propagated_velocities,
        focal_player_id=action.player_id,
        action_id=action.action_id,
        ball_carrier_id=ball_carrier_id,
        boundary_clipped_player_ids=tuple(sorted(clipped_ids)),
        used_infeasible_reference=not action.motion.feasible,
    )
