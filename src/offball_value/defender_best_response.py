"""Threat-aware best responses to one counterfactual off-ball trajectory.

The v0.1 experiment is a *localized retrospective counterfactual*: the focal
attacker and one focal defender follow virtual feasible trajectories, while
the ball and all non-intervened players follow their observed future.  This
isolates the local attacker--defender interaction without pretending that a
constant-velocity rollout is a realistic full-team future model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .bundesliga import FPS, BundesligaFrame
from .defender_response import DefenderResponseAction
from .pass_dynamics import ArrivalModelConfig, VelocityEstimate, estimate_frame_velocities
from .reference_obso import (
    ReferenceOBSOConfig,
    ReferenceOBSOMaximum,
    maximum_reference_obso,
)


@dataclass(frozen=True)
class DefenderBestResponseConfig:
    """Frozen temporal definition of the v0.1 defender objective."""

    evaluation_times_s: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6, 2.0)
    velocity_history_seconds: float = 0.4
    obso_batch_size: int = 4

    def validate(self) -> None:
        if not self.evaluation_times_s:
            raise ValueError("evaluation_times_s cannot be empty")
        if any(time_s <= 0.0 for time_s in self.evaluation_times_s):
            raise ValueError("evaluation times must be positive")
        if tuple(sorted(self.evaluation_times_s)) != self.evaluation_times_s:
            raise ValueError("evaluation times must be strictly increasing")
        if len(set(self.evaluation_times_s)) != len(self.evaluation_times_s):
            raise ValueError("evaluation times must be unique")
        if self.velocity_history_seconds <= 0.0:
            raise ValueError("velocity_history_seconds must be positive")
        if self.obso_batch_size <= 0:
            raise ValueError("obso_batch_size must be positive")


@dataclass(frozen=True)
class ObservedBackgroundState:
    time_s: float
    frame: BundesligaFrame
    velocities: Mapping[str, VelocityEstimate]


@dataclass(frozen=True)
class ObservedBackgroundSequence:
    decision_frame_id: int
    states: tuple[ObservedBackgroundState, ...]


@dataclass(frozen=True)
class ThreatTracePoint:
    time_s: float
    maximum_obso: float
    maximum_x: float
    maximum_y: float
    evaluated_cell_count: int


@dataclass(frozen=True)
class TrajectoryThreatTrace:
    action_id: str
    points: tuple[ThreatTracePoint, ...]
    horizon_peak: float
    horizon_mean: float
    terminal: float
    effort_m2ps3: float

    @property
    def rank_key(self) -> tuple[float, float, float, float, str]:
        """Robust best response: lowest spike, then mean, terminal, effort."""

        return (
            self.horizon_peak,
            self.horizon_mean,
            self.terminal,
            self.effort_m2ps3,
            self.action_id,
        )


@dataclass(frozen=True)
class DefenderBestResponseSearch:
    best: TrajectoryThreatTrace
    terminal_traces: tuple[TrajectoryThreatTrace, ...]
    fully_evaluated_traces: tuple[TrajectoryThreatTrace, ...]

    @property
    def terminal_spread(self) -> float:
        values = [trace.terminal for trace in self.terminal_traces]
        return float(max(values) - min(values)) if values else float("nan")


def prepare_observed_background_sequence(
    frames: Mapping[int, BundesligaFrame] | Iterable[BundesligaFrame],
    decision_frame_id: int,
    config: DefenderBestResponseConfig = DefenderBestResponseConfig(),
) -> ObservedBackgroundSequence:
    """Build actual-future background states and causal local velocities."""

    config.validate()
    frame_map = (
        dict(frames)
        if isinstance(frames, Mapping)
        else {frame.frame_id: frame for frame in frames}
    )
    velocity_config = ArrivalModelConfig(
        history_seconds=config.velocity_history_seconds
    )
    states: list[ObservedBackgroundState] = []
    for time_s in config.evaluation_times_s:
        target_id = decision_frame_id + int(round(time_s * FPS))
        if target_id not in frame_map:
            raise KeyError(f"Observed background frame {target_id} is missing")
        velocities = estimate_frame_velocities(
            frame_map.values(), target_id, velocity_config
        )
        states.append(
            ObservedBackgroundState(
                time_s=float(time_s),
                frame=frame_map[target_id],
                velocities=velocities,
            )
        )
    return ObservedBackgroundSequence(
        decision_frame_id=decision_frame_id,
        states=tuple(states),
    )


def interpolate_path_state(
    path_xy: Sequence[Sequence[float]],
    path_times_s: Sequence[float],
    target_time_s: float,
) -> tuple[float, float, float, float]:
    """Linearly interpolate position and the local segment velocity."""

    if len(path_xy) != len(path_times_s) or len(path_xy) < 2:
        raise ValueError("path and timestamps must have the same length >= 2")
    times = np.asarray(path_times_s, dtype=float)
    points = np.asarray(path_xy, dtype=float)
    if points.shape != (len(times), 2) or not np.all(np.isfinite(points)):
        raise ValueError("path_xy must contain finite two-dimensional points")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("path timestamps must be strictly increasing")
    if target_time_s < times[0] - 1e-9 or target_time_s > times[-1] + 1e-9:
        raise ValueError("target time lies outside the sampled path")

    x = float(np.interp(target_time_s, times, points[:, 0]))
    y = float(np.interp(target_time_s, times, points[:, 1]))
    right = int(np.searchsorted(times, target_time_s, side="right"))
    if right <= 0:
        left, right = 0, 1
    elif right >= len(times):
        left, right = len(times) - 2, len(times) - 1
    else:
        left = right - 1
    duration = float(times[right] - times[left])
    vx = float((points[right, 0] - points[left, 0]) / duration)
    vy = float((points[right, 1] - points[left, 1]) / duration)
    return x, y, vx, vy


def _replace_virtual_player(
    frame: BundesligaFrame,
    velocities: dict[str, VelocityEstimate],
    player_id: str,
    state: tuple[float, float, float, float],
) -> BundesligaFrame:
    if player_id not in frame.players:
        raise KeyError(f"Virtual player {player_id} is absent at frame {frame.frame_id}")
    x, y, vx, vy = state
    speed = math.hypot(vx, vy)
    player = replace(frame.players[player_id], x=x, y=y, speed=speed * 3.6)
    velocities[player_id] = VelocityEstimate(vx, vy, speed, 0, 0.0)
    return frame.with_player(player_id, player)


def build_local_counterfactual_state(
    background: ObservedBackgroundState,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    defender_id: str | None = None,
    defender_path_xy: Sequence[Sequence[float]] | None = None,
    defender_path_times_s: Sequence[float] | None = None,
) -> tuple[BundesligaFrame, dict[str, VelocityEstimate]]:
    """Replace only the named virtual players in one observed-future frame."""

    frame = background.frame
    velocities = dict(background.velocities)
    attacker_state = interpolate_path_state(
        attacker_path_xy, attacker_path_times_s, background.time_s
    )
    frame = _replace_virtual_player(frame, velocities, attacker_id, attacker_state)
    if defender_id is not None:
        if defender_path_xy is None or defender_path_times_s is None:
            raise ValueError("a defender path and timestamps are required together")
        defender_state = interpolate_path_state(
            defender_path_xy, defender_path_times_s, background.time_s
        )
        frame = _replace_virtual_player(frame, velocities, defender_id, defender_state)
    return frame, velocities


def _summarize_trace(
    action_id: str,
    points: Sequence[ThreatTracePoint],
    effort_m2ps3: float,
) -> TrajectoryThreatTrace:
    if not points:
        raise ValueError("a threat trace must contain at least one point")
    values = np.asarray([point.maximum_obso for point in points], dtype=float)
    times = np.asarray([point.time_s for point in points], dtype=float)
    if len(points) == 1:
        mean = float(values[0])
    else:
        mean = float(np.trapezoid(values, times) / (times[-1] - times[0]))
    return TrajectoryThreatTrace(
        action_id=action_id,
        points=tuple(points),
        horizon_peak=float(np.max(values)),
        horizon_mean=mean,
        terminal=float(values[-1]),
        effort_m2ps3=float(effort_m2ps3),
    )


def evaluate_defender_trajectory(
    background: ObservedBackgroundSequence,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    action: DefenderResponseAction | None,
    defender_id: str | None = None,
    action_id: str = "observed-defender",
    config: DefenderBestResponseConfig = DefenderBestResponseConfig(),
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> TrajectoryThreatTrace:
    """Score one response, or the observed defender path when action is None."""

    config.validate()
    if action is not None:
        defender_id = action.defender_id
        defender_path = action.full_path_xy
        defender_times = action.response_path_times_s
        action_id = action.action_id
        effort = action.base_action.motion.effort_m2ps3
    else:
        defender_path = None
        defender_times = None
        effort = 0.0

    points: list[ThreatTracePoint] = []
    for state in background.states:
        frame, velocities = build_local_counterfactual_state(
            state,
            attacker_id,
            attacker_path_xy,
            attacker_path_times_s,
            defender_id=defender_id if action is not None else None,
            defender_path_xy=defender_path,
            defender_path_times_s=defender_times,
        )
        maximum: ReferenceOBSOMaximum = maximum_reference_obso(
            frame,
            attacking_team_id,
            attacking_direction,
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            apply_offside=True,
            config=obso_config,
            batch_size=config.obso_batch_size,
        )
        points.append(
            ThreatTracePoint(
                time_s=state.time_s,
                maximum_obso=maximum.value,
                maximum_x=maximum.x,
                maximum_y=maximum.y,
                evaluated_cell_count=maximum.evaluated_cell_count,
            )
        )
    return _summarize_trace(action_id, points, effort)


def select_best_response(
    traces: Sequence[TrajectoryThreatTrace],
) -> TrajectoryThreatTrace:
    """Select the feasible response minimizing worst post-decision threat."""

    if not traces:
        raise ValueError("at least one defender response trace is required")
    return min(traces, key=lambda trace: trace.rank_key)


def search_defender_best_response(
    background: ObservedBackgroundSequence,
    attacker_id: str,
    attacker_path_xy: Sequence[Sequence[float]],
    attacker_path_times_s: Sequence[float],
    attacking_team_id: str,
    attacking_direction: int,
    goalkeeper_ids: Sequence[str],
    actions: Sequence[DefenderResponseAction],
    config: DefenderBestResponseConfig = DefenderBestResponseConfig(),
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    progress_every: int = 0,
) -> DefenderBestResponseSearch:
    """Find the exact lexicographic best response with terminal lower bounds.

    Every feasible action is first evaluated at t+2.  Its terminal threat is
    a lower bound on its horizon peak.  Full traces are then evaluated from
    the smallest bound upward, stopping once no unevaluated action can improve
    on the incumbent peak.  Thus the staged search changes runtime, not the
    selected response.
    """

    if not actions:
        raise ValueError("at least one feasible defender action is required")
    terminal_background = ObservedBackgroundSequence(
        decision_frame_id=background.decision_frame_id,
        states=(background.states[-1],),
    )
    terminal: list[TrajectoryThreatTrace] = []
    for index, action in enumerate(actions, start=1):
        terminal.append(
            evaluate_defender_trajectory(
                terminal_background,
                attacker_id,
                attacker_path_xy,
                attacker_path_times_s,
                attacking_team_id,
                attacking_direction,
                goalkeeper_ids,
                action,
                config=config,
                obso_config=obso_config,
            )
        )
        if progress_every > 0 and index % progress_every == 0:
            print(f"terminal responses evaluated: {index}/{len(actions)}", flush=True)

    ordered = sorted(terminal, key=lambda trace: (trace.terminal, trace.action_id))
    by_id = {action.action_id: action for action in actions}
    full: list[TrajectoryThreatTrace] = []
    incumbent: TrajectoryThreatTrace | None = None
    for bound in ordered:
        if incumbent is not None and bound.terminal > incumbent.horizon_peak:
            break
        trace = evaluate_defender_trajectory(
            background,
            attacker_id,
            attacker_path_xy,
            attacker_path_times_s,
            attacking_team_id,
            attacking_direction,
            goalkeeper_ids,
            by_id[bound.action_id],
            config=config,
            obso_config=obso_config,
        )
        full.append(trace)
        if incumbent is None or trace.rank_key < incumbent.rank_key:
            incumbent = trace
    if incumbent is None:
        raise RuntimeError("best-response search did not evaluate any full trace")
    return DefenderBestResponseSearch(
        best=incumbent,
        terminal_traces=tuple(terminal),
        fully_evaluated_traces=tuple(full),
    )
