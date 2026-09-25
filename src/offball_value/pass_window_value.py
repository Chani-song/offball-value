"""Pass-release-time values for an off-ball counterfactual.

The movement horizon and the valuation clock are deliberately separated. A
two-second path may be proposed, but a receiving option is legal and valuable
at the instant a controlled teammate can release the ball. Offside is frozen
at that release state; it is not recomputed at the end of the movement path or
at the projected reception time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .bundesliga import FPS, BundesligaFrame
from .defender_best_response import ObservedBackgroundSequence, ObservedBackgroundState
from .pass_dynamics import (
    ArrivalModelConfig,
    VelocityEstimate,
    estimate_frame_velocities,
)
from .reference_obso import (
    ReferenceOBSOConfig,
    ReferenceOBSOSurface,
    evaluate_reference_obso,
    offside_attacker_ids,
)


@dataclass(frozen=True)
class PlayerPassOption:
    player_id: str
    value: float
    event_x: float | None
    event_y: float | None
    pitch_control: float
    transition: float
    score: float
    player_event_distance_m: float | None
    onside_at_release: bool


@dataclass(frozen=True)
class PassReleaseSnapshot:
    release_time_s: float
    ball_owner_id: str | None
    owner_ball_distance_m: float
    controlled: bool
    team_maximum: float
    focal: PlayerPassOption
    best_other: PlayerPassOption | None
    player_options: tuple[PlayerPassOption, ...]
    surface: ReferenceOBSOSurface


@dataclass(frozen=True)
class PlayerPassWindowOption:
    player_id: str
    onset_value: float
    peak_value: float
    peak_time_s: float
    gain_from_onset: float
    peak_event_x: float | None
    peak_event_y: float | None
    onside_release_times_s: tuple[float, ...]


@dataclass(frozen=True)
class PassWindowTrace:
    points: tuple[PassReleaseSnapshot, ...]
    focal_peak: float
    focal_peak_time_s: float
    other_peak: float
    other_peak_time_s: float
    other_peak_player_id: str | None
    team_peak: float
    player_windows: tuple[PlayerPassWindowOption, ...]
    largest_teammate_gain_player_id: str | None
    largest_teammate_gain: float


def prepare_pass_release_background_sequence(
    frames: Mapping[int, BundesligaFrame] | Iterable[BundesligaFrame],
    decision_frame_id: int,
    release_times_s: Sequence[float],
    velocity_history_seconds: float = 0.4,
) -> ObservedBackgroundSequence:
    """Build causal observed states at candidate pass-release instants."""

    if not release_times_s:
        raise ValueError("release_times_s cannot be empty")
    if any(time_s < 0.0 for time_s in release_times_s):
        raise ValueError("release times cannot be negative")
    if tuple(sorted(set(release_times_s))) != tuple(release_times_s):
        raise ValueError("release times must be unique and increasing")
    frame_map = (
        dict(frames)
        if isinstance(frames, Mapping)
        else {frame.frame_id: frame for frame in frames}
    )
    velocity_config = ArrivalModelConfig(history_seconds=velocity_history_seconds)
    states: list[ObservedBackgroundState] = []
    for requested_time in release_times_s:
        target_id = decision_frame_id + int(round(requested_time * FPS))
        if target_id not in frame_map:
            raise KeyError(f"Observed release frame {target_id} is missing")
        states.append(
            ObservedBackgroundState(
                time_s=(target_id - decision_frame_id) / FPS,
                frame=frame_map[target_id],
                velocities=estimate_frame_velocities(
                    frame_map.values(), target_id, velocity_config
                ),
            )
        )
    return ObservedBackgroundSequence(
        decision_frame_id=decision_frame_id,
        states=tuple(states),
    )


def controlled_ball_owner(
    frame: BundesligaFrame,
    attacking_team_id: str,
    maximum_distance_m: float = 1.5,
) -> tuple[str | None, float, bool]:
    """Return the nearest attacking player and controlled-possession flag."""

    if frame.ball is None:
        return None, math.inf, False
    candidates = [
        (
            math.hypot(state.x - frame.ball.x, state.y - frame.ball.y),
            player_id,
        )
        for player_id, state in frame.players.items()
        if state.team_id == attacking_team_id
    ]
    if not candidates:
        return None, math.inf, False
    distance, player_id = min(candidates)
    return player_id, float(distance), distance <= maximum_distance_m


def per_attacker_pass_options(
    surface: ReferenceOBSOSurface,
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float] | None = None,
    offside_tolerance_m: float = 0.2,
) -> tuple[PlayerPassOption, ...]:
    """Attribute the release-state OBSO surface to eligible attackers.

    The team surface remains the frozen reference OBSO. Nearest-attacker cell
    attribution is an explanation wrapper that makes teammate options visible.
    Offside players receive an explicit zero-valued record rather than silently
    disappearing from a horizon trace.
    """

    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)
    attacker_ids = tuple(
        sorted(
            player_id
            for player_id, state in frame.players.items()
            if state.team_id == attacking_team_id
        )
    )
    offside = offside_attacker_ids(
        frame,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        offside_tolerance_m,
    )
    eligible_ids = tuple(player_id for player_id in attacker_ids if player_id not in offside)
    owner_grid = np.full(surface.obso.shape, -1, dtype=int)
    if eligible_ids:
        xx, yy = np.meshgrid(surface.xgrid, surface.ygrid)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        positions = np.asarray(
            [
                (frame.players[player_id].x, frame.players[player_id].y)
                for player_id in eligible_ids
            ],
            dtype=float,
        )
        owner_grid = np.argmin(
            np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2),
            axis=1,
        ).reshape(surface.obso.shape)

    results: list[PlayerPassOption] = []
    for player_id in attacker_ids:
        if player_id in offside:
            results.append(
                PlayerPassOption(
                    player_id=player_id,
                    value=0.0,
                    event_x=None,
                    event_y=None,
                    pitch_control=0.0,
                    transition=0.0,
                    score=0.0,
                    player_event_distance_m=None,
                    onside_at_release=False,
                )
            )
            continue
        player_index = eligible_ids.index(player_id)
        masked = np.where(owner_grid == player_index, surface.obso, -np.inf)
        row, column = np.unravel_index(int(np.argmax(masked)), masked.shape)
        x = float(surface.xgrid[column])
        y = float(surface.ygrid[row])
        player = frame.players[player_id]
        results.append(
            PlayerPassOption(
                player_id=player_id,
                value=float(masked[row, column]),
                event_x=x,
                event_y=y,
                pitch_control=float(surface.pitch_control[row, column]),
                transition=float(surface.transition[row, column]),
                score=float(surface.score[row, column]),
                player_event_distance_m=math.hypot(player.x - x, player.y - y),
                onside_at_release=True,
            )
        )
    return tuple(results)


def evaluate_pass_release_snapshot(
    release_time_s: float,
    frame: BundesligaFrame,
    velocities: Mapping[str, VelocityEstimate],
    attacking_team_id: str,
    focal_player_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    control_distance_m: float = 1.5,
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> PassReleaseSnapshot:
    """Evaluate all legal receiving options at one possible kick instant."""

    owner_id, owner_distance, controlled = controlled_ball_owner(
        frame, attacking_team_id, control_distance_m
    )
    surface = evaluate_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
        config=obso_config,
    )
    options = per_attacker_pass_options(
        surface,
        frame,
        attacking_team_id,
        attacking_direction,
    )
    by_id = {option.player_id: option for option in options}
    if focal_player_id not in by_id:
        raise KeyError(f"Focal attacker {focal_player_id} is missing")
    other = [option for option in options if option.player_id != focal_player_id]
    return PassReleaseSnapshot(
        release_time_s=float(release_time_s),
        ball_owner_id=owner_id,
        owner_ball_distance_m=owner_distance,
        controlled=controlled,
        team_maximum=surface.maximum,
        focal=by_id[focal_player_id],
        best_other=max(other, key=lambda item: (item.value, item.player_id)) if other else None,
        player_options=options,
        surface=surface,
    )


def summarize_pass_window(
    points: Sequence[PassReleaseSnapshot],
    focal_player_id: str,
    required_ball_owner_id: str | None = None,
) -> PassWindowTrace:
    """Aggregate controlled release snapshots after the run-onset baseline.

    The point at ``t=0`` is the agreed pre-movement reference ``V0``.  It is
    retained in ``points`` for display and for player-level onset gains, but it
    cannot distinguish defensive responses and therefore must not enter the
    response objective.  Peaks are taken over controlled release instants with
    ``t>0``.
    """

    valid = tuple(
        point
        for point in points
        if point.controlled
        and (
            required_ball_owner_id is None
            or point.ball_owner_id == required_ball_owner_id
        )
    )
    if not valid:
        raise ValueError("pass window contains no controlled release state")
    onset_points = tuple(
        point for point in valid if math.isclose(point.release_time_s, 0.0)
    )
    if not onset_points:
        raise ValueError("pass window must contain a controlled t=0 baseline")
    response_points = tuple(point for point in valid if point.release_time_s > 0.0)
    if not response_points:
        raise ValueError("pass window contains no post-onset release state")
    focal_point = max(
        response_points,
        key=lambda point: (point.focal.value, -point.release_time_s),
    )
    other_points = [
        point for point in response_points if point.best_other is not None
    ]
    other_point = (
        max(
            other_points,
            key=lambda point: (point.best_other.value, -point.release_time_s),
        )
        if other_points
        else None
    )
    player_ids = tuple(
        sorted({option.player_id for point in valid for option in point.player_options})
    )
    player_windows: list[PlayerPassWindowOption] = []
    for player_id in player_ids:
        timed = [
            (point, next(option for option in point.player_options if option.player_id == player_id))
            for point in valid
        ]
        onset = next(
            option
            for point, option in timed
            if math.isclose(point.release_time_s, 0.0)
        )
        post_onset = [
            (point, option)
            for point, option in timed
            if point.release_time_s > 0.0
        ]
        peak_point, peak = max(
            post_onset,
            key=lambda item: (item[1].value, -item[0].release_time_s),
        )
        player_windows.append(
            PlayerPassWindowOption(
                player_id=player_id,
                onset_value=onset.value,
                peak_value=peak.value,
                peak_time_s=peak_point.release_time_s,
                gain_from_onset=peak.value - onset.value,
                peak_event_x=peak.event_x,
                peak_event_y=peak.event_y,
                onside_release_times_s=tuple(
                    point.release_time_s
                    for point, option in timed
                    if option.onside_at_release
                ),
            )
        )
    teammates = [
        item for item in player_windows if item.player_id != focal_player_id
    ]
    gain_winner = (
        max(teammates, key=lambda item: (item.gain_from_onset, item.player_id))
        if teammates
        else None
    )
    return PassWindowTrace(
        points=valid,
        focal_peak=focal_point.focal.value,
        focal_peak_time_s=focal_point.release_time_s,
        other_peak=(other_point.best_other.value if other_point else 0.0),
        other_peak_time_s=(other_point.release_time_s if other_point else valid[0].release_time_s),
        other_peak_player_id=(
            other_point.best_other.player_id if other_point and other_point.best_other else None
        ),
        team_peak=max(point.team_maximum for point in response_points),
        player_windows=tuple(player_windows),
        largest_teammate_gain_player_id=(gain_winner.player_id if gain_winner else None),
        largest_teammate_gain=(gain_winner.gain_from_onset if gain_winner else 0.0),
    )
