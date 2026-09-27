"""Reproducible public-reference OBSO for full tracking data.

This module deliberately lives beside, rather than inside, :mod:`obso`.  The
existing module is a useful heuristic prototype and contains runner-specific
extensions.  The evaluator here freezes the public C-OBSO/Friends-of-Tracking
convention so that every counterfactual search can use the same threat ruler.

It is not an exact numerical reproduction of Spearman (2018).  The original
paper used a fitted score curve and aerodynamic ball-flight model that are not
fully available with the open Bundesliga sample.  See
``docs/project2_research_design_v0_1.md`` for the provenance and deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
)


DEFAULT_SCORE_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "static" / "EPV_grid.csv"
)


@dataclass(frozen=True)
class ReferenceOBSOConfig:
    """Frozen parameters for the reproducible public OBSO convention.

    The motion and control defaults follow the public Friends-of-Tracking /
    C-OBSO implementation.  They are intentionally separate from the physical
    constraints later used to generate counterfactual action endpoints.
    """

    field_length_m: float = FIELD_LENGTH
    field_width_m: float = FIELD_WIDTH
    grid_cells_x: int = 50
    grid_cells_y: int = 32
    max_player_speed_mps: float = 5.0
    reaction_time_s: float = 0.7
    tti_sigma_s: float = 0.45
    lambda_att_hz: float = 4.3
    lambda_def_hz: float = 4.3
    goalkeeper_control_multiplier: float = 3.0
    average_ball_speed_mps: float = 15.0
    transition_sigma_m: float = 14.0
    integration_dt_s: float = 0.04
    max_integration_time_s: float = 10.0
    convergence_tolerance: float = 0.01
    time_to_control_veto: float = 3.0
    offside_tolerance_m: float = 0.2
    score_grid_path: Path = DEFAULT_SCORE_GRID_PATH

    def validate(self) -> None:
        positive = {
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
            "grid_cells_x": self.grid_cells_x,
            "grid_cells_y": self.grid_cells_y,
            "max_player_speed_mps": self.max_player_speed_mps,
            "tti_sigma_s": self.tti_sigma_s,
            "lambda_att_hz": self.lambda_att_hz,
            "lambda_def_hz": self.lambda_def_hz,
            "average_ball_speed_mps": self.average_ball_speed_mps,
            "transition_sigma_m": self.transition_sigma_m,
            "integration_dt_s": self.integration_dt_s,
            "max_integration_time_s": self.max_integration_time_s,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Reference OBSO parameters must be positive: {invalid}")
        if not 0 < self.convergence_tolerance < 1:
            raise ValueError("convergence_tolerance must lie between zero and one")


@dataclass(frozen=True)
class ReferenceOBSOSurface:
    """The fixed components and product for one complete game state."""

    xgrid: np.ndarray
    ygrid: np.ndarray
    pitch_control: np.ndarray
    transition: np.ndarray
    score: np.ndarray
    obso: np.ndarray

    @property
    def maximum(self) -> float:
        return float(np.nanmax(self.obso))

    @property
    def maximum_position(self) -> tuple[float, float]:
        row, col = np.unravel_index(int(np.nanargmax(self.obso)), self.obso.shape)
        return float(self.xgrid[col]), float(self.ygrid[row])


@dataclass(frozen=True)
class ReferenceOBSOMaximum:
    """Exact maximum without materializing the complete OBSO surface."""

    value: float
    x: float
    y: float
    evaluated_cell_count: int


@dataclass(frozen=True)
class ReferenceOBSOBoundedMaximum:
    """Exact maximum, or a certified rejection above a minimization cutoff."""

    maximum: ReferenceOBSOMaximum
    exact: bool


def pitch_grid(
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-centre x/y axes and flattened ``(x, y)`` points."""

    config.validate()
    dx = config.field_length_m / config.grid_cells_x
    dy = config.field_width_m / config.grid_cells_y
    xgrid = (
        np.arange(config.grid_cells_x, dtype=float) * dx
        - config.field_length_m / 2.0
        + dx / 2.0
    )
    ygrid = (
        np.arange(config.grid_cells_y, dtype=float) * dy
        - config.field_width_m / 2.0
        + dy / 2.0
    )
    xx, yy = np.meshgrid(xgrid, ygrid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return xgrid, ygrid, points


@lru_cache(maxsize=8)
def _load_peak_normalized_score_grid(path: str) -> np.ndarray:
    grid = np.loadtxt(path, delimiter=",")
    if grid.ndim != 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"Score grid must be a finite two-dimensional array: {path}")
    peak = float(np.max(grid))
    if peak <= 0:
        raise ValueError(f"Score grid must contain a positive value: {path}")
    result = np.asarray(grid / peak, dtype=float)
    result.setflags(write=False)
    return result


def score_surface(
    attacking_direction: int,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> np.ndarray:
    """Return the public peak-normalized score grid for one attack direction."""

    if attacking_direction not in (-1, 1):
        raise ValueError("attacking_direction must be -1 or 1")
    score = _load_peak_normalized_score_grid(str(config.score_grid_path))
    expected_shape = (config.grid_cells_y, config.grid_cells_x)
    if score.shape != expected_shape:
        raise ValueError(
            f"Score grid has shape {score.shape}; expected {expected_shape}. "
            "Resampling would change the frozen reference evaluator."
        )
    return np.fliplr(score).copy() if attacking_direction < 0 else score.copy()


def transition_surface(
    ball_xy: tuple[float, float],
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> np.ndarray:
    """Return the peak-normalized isotropic Gaussian transition surface."""

    _, _, points = pitch_grid(config)
    ball = np.asarray(ball_xy, dtype=float)
    if ball.shape != (2,) or not np.all(np.isfinite(ball)):
        raise ValueError("ball_xy must contain two finite coordinates")
    squared_distance = np.sum((points - ball) ** 2, axis=1)
    transition = np.exp(
        -squared_distance / (2.0 * config.transition_sigma_m**2)
    )
    return transition.reshape(config.grid_cells_y, config.grid_cells_x)


def _velocity_xy(
    velocities: Mapping[str, object] | None,
    player_id: str,
) -> np.ndarray:
    if velocities is None or player_id not in velocities:
        return np.zeros(2, dtype=float)

    value = velocities[player_id]
    if hasattr(value, "vx") and hasattr(value, "vy"):
        result = np.asarray([getattr(value, "vx"), getattr(value, "vy")], dtype=float)
    else:
        result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Velocity for {player_id} must contain two finite values")
    return result


def offside_attacker_ids(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    tolerance_m: float = 0.2,
) -> set[str]:
    """Identify attackers beyond the ball, halfway line, and second defender."""

    if attacking_direction not in (-1, 1):
        raise ValueError("attacking_direction must be -1 or 1")
    defenders = [
        state
        for state in frame.players.values()
        if state.team_id != attacking_team_id
    ]
    if len(defenders) < 2:
        return set()

    transformed_defender_x = sorted(
        (attacking_direction * state.x for state in defenders),
        reverse=True,
    )
    second_last = transformed_defender_x[1]
    line = max(second_last, attacking_direction * ball_xy[0], 0.0) + tolerance_m
    return {
        player_id
        for player_id, state in frame.players.items()
        if state.team_id == attacking_team_id
        and attacking_direction * state.x > line
    }


def _time_to_control(
    control_rate_hz: float,
    config: ReferenceOBSOConfig,
) -> float:
    return config.time_to_control_veto * math.log(10.0) * (
        math.sqrt(3.0) * config.tti_sigma_s / math.pi
        + 1.0 / control_rate_hz
    )


def _arrival_times(
    positions: np.ndarray,
    velocities: np.ndarray,
    target: np.ndarray,
    config: ReferenceOBSOConfig,
) -> np.ndarray:
    reaction_positions = positions + velocities * config.reaction_time_s
    distances = np.linalg.norm(target - reaction_positions, axis=1)
    return config.reaction_time_s + distances / config.max_player_speed_mps


def _logistic_arrival_probability(
    time_s: float,
    arrival_times: np.ndarray,
    config: ReferenceOBSOConfig,
) -> np.ndarray:
    scale = math.pi / (math.sqrt(3.0) * config.tti_sigma_s)
    logits = np.clip(scale * (time_s - arrival_times), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _pitch_control_at_target(
    target: np.ndarray,
    attacker_positions: np.ndarray,
    attacker_velocities: np.ndarray,
    defender_positions: np.ndarray,
    defender_velocities: np.ndarray,
    defender_control_rates: np.ndarray,
    ball_xy: np.ndarray,
    config: ReferenceOBSOConfig,
) -> float:
    if len(attacker_positions) == 0:
        return 0.0
    if len(defender_positions) == 0:
        return 1.0

    ball_travel_time = (
        float(np.linalg.norm(target - ball_xy)) / config.average_ball_speed_mps
    )
    attacker_arrivals = _arrival_times(
        attacker_positions,
        attacker_velocities,
        target,
        config,
    )
    defender_arrivals = _arrival_times(
        defender_positions,
        defender_velocities,
        target,
        config,
    )
    minimum_attacker = float(np.min(attacker_arrivals))
    minimum_defender = float(np.min(defender_arrivals))
    defender_time_to_control = _time_to_control(config.lambda_def_hz, config)

    if (
        minimum_attacker - max(ball_travel_time, minimum_defender)
        >= defender_time_to_control
    ):
        return 0.0
    if (
        minimum_defender - max(ball_travel_time, minimum_attacker)
        >= _time_to_control(config.lambda_att_hz, config)
    ):
        return 1.0

    attacker_keep = (
        attacker_arrivals - minimum_attacker
        < _time_to_control(config.lambda_att_hz, config)
    )
    defender_keep = defender_arrivals - minimum_defender < defender_time_to_control
    attacker_arrivals = attacker_arrivals[attacker_keep]
    defender_arrivals = defender_arrivals[defender_keep]
    defender_control_rates = defender_control_rates[defender_keep]

    p_attacker = 0.0
    p_defender = 0.0
    time_s = ball_travel_time
    end_time = ball_travel_time + config.max_integration_time_s
    while (
        1.0 - p_attacker - p_defender > config.convergence_tolerance
        and time_s <= end_time
    ):
        residual = max(0.0, 1.0 - p_attacker - p_defender)
        attacker_rate = config.lambda_att_hz * float(
            np.sum(
                _logistic_arrival_probability(
                    time_s,
                    attacker_arrivals,
                    config,
                )
            )
        )
        defender_rate = float(
            np.sum(
                defender_control_rates
                * _logistic_arrival_probability(
                    time_s,
                    defender_arrivals,
                    config,
                )
            )
        )

        # The public code uses forward Euler integration.  Scale only a rare
        # overshooting step so numerical error cannot create probabilities > 1.
        delta_attacker = residual * attacker_rate * config.integration_dt_s
        delta_defender = residual * defender_rate * config.integration_dt_s
        delta_total = delta_attacker + delta_defender
        if delta_total > residual and delta_total > 0:
            scale = residual / delta_total
            delta_attacker *= scale
            delta_defender *= scale
        p_attacker += delta_attacker
        p_defender += delta_defender
        time_s += config.integration_dt_s

    if p_attacker + p_defender <= 0:
        return 0.5
    return float(np.clip(p_attacker, 0.0, 1.0))


def pitch_control_at_points(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    points: Sequence[tuple[float, float]] | np.ndarray,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> np.ndarray:
    """Evaluate attacking-team PPCF at arbitrary pitch locations."""

    config.validate()
    targets = np.asarray(points, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, 2)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.all(np.isfinite(targets)):
        raise ValueError("points must be a finite array with shape (n, 2)")

    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)
    ball = np.asarray(ball_xy, dtype=float)

    excluded = (
        offside_attacker_ids(
            frame,
            attacking_team_id,
            attacking_direction,
            ball_xy,
            config.offside_tolerance_m,
        )
        if apply_offside
        else set()
    )
    attacker_items = [
        (player_id, state)
        for player_id, state in frame.players.items()
        if state.team_id == attacking_team_id and player_id not in excluded
    ]
    defender_items = [
        (player_id, state)
        for player_id, state in frame.players.items()
        if state.team_id != attacking_team_id
    ]

    attacker_positions = np.asarray(
        [(state.x, state.y) for _, state in attacker_items],
        dtype=float,
    ).reshape(-1, 2)
    defender_positions = np.asarray(
        [(state.x, state.y) for _, state in defender_items],
        dtype=float,
    ).reshape(-1, 2)
    attacker_velocities = np.asarray(
        [_velocity_xy(velocities, player_id) for player_id, _ in attacker_items],
        dtype=float,
    ).reshape(-1, 2)
    defender_velocities = np.asarray(
        [_velocity_xy(velocities, player_id) for player_id, _ in defender_items],
        dtype=float,
    ).reshape(-1, 2)
    goalkeeper_set = set(goalkeeper_ids)
    defender_control_rates = np.asarray(
        [
            config.lambda_def_hz
            * (config.goalkeeper_control_multiplier if player_id in goalkeeper_set else 1.0)
            for player_id, _ in defender_items
        ],
        dtype=float,
    )

    return np.asarray(
        [
            _pitch_control_at_target(
                target,
                attacker_positions,
                attacker_velocities,
                defender_positions,
                defender_velocities,
                defender_control_rates,
                ball,
                config,
            )
            for target in targets
        ],
        dtype=float,
    )


def evaluate_reference_obso(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> ReferenceOBSOSurface:
    """Compute the frozen 50 by 32 public-reference OBSO surface."""

    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)
    xgrid, ygrid, points = pitch_grid(config)
    control = pitch_control_at_points(
        frame=frame,
        attacking_team_id=attacking_team_id,
        attacking_direction=attacking_direction,
        points=points,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        ball_xy=ball_xy,
        apply_offside=apply_offside,
        config=config,
    ).reshape(config.grid_cells_y, config.grid_cells_x)
    transition = transition_surface(ball_xy, config)
    score = score_surface(attacking_direction, config)
    obso = control * transition * score
    return ReferenceOBSOSurface(
        xgrid=xgrid,
        ygrid=ygrid,
        pitch_control=control,
        transition=transition,
        score=score,
        obso=obso,
    )


def maximum_reference_obso(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    batch_size: int = 32,
) -> ReferenceOBSOMaximum:
    """Return the exact maximum OBSO value using a safe upper-bound search.

    OBSO at a cell is ``PPCF * transition * score`` and PPCF cannot exceed
    one.  Cells are therefore evaluated in descending order of
    ``transition * score``.  Once the largest unevaluated product is no
    greater than the best OBSO already observed, no remaining cell can win.
    This preserves the reference evaluator exactly while avoiding most of the
    1,600 pitch-control integrations during counterfactual search.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)

    _, _, points = pitch_grid(config)
    upper_bounds = (
        transition_surface(ball_xy, config) * score_surface(attacking_direction, config)
    ).ravel()
    order = np.argsort(upper_bounds)[::-1]
    best_value = -math.inf
    best_index = int(order[0])
    evaluated = 0

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        control = pitch_control_at_points(
            frame=frame,
            attacking_team_id=attacking_team_id,
            attacking_direction=attacking_direction,
            points=points[indices],
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            ball_xy=ball_xy,
            apply_offside=apply_offside,
            config=config,
        )
        values = control * upper_bounds[indices]
        local_index = int(np.argmax(values))
        if float(values[local_index]) > best_value:
            best_value = float(values[local_index])
            best_index = int(indices[local_index])
        evaluated += len(indices)

        next_start = start + len(indices)
        if next_start >= len(order) or upper_bounds[order[next_start]] <= best_value:
            break

    return ReferenceOBSOMaximum(
        value=float(best_value),
        x=float(points[best_index, 0]),
        y=float(points[best_index, 1]),
        evaluated_cell_count=evaluated,
    )


def bounded_maximum_reference_obso(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    rejection_cutoff: float,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    batch_size: int = 32,
) -> ReferenceOBSOBoundedMaximum:
    """Stop once a response is certified worse than a minimization incumbent.

    If ``exact`` is false, the returned value is a witnessed lower bound that
    is strictly above ``rejection_cutoff``; the true maximum can only be
    larger.  Otherwise the returned maximum is identical to
    :func:`maximum_reference_obso`.
    """

    if not math.isfinite(rejection_cutoff) or rejection_cutoff < 0.0:
        raise ValueError("rejection_cutoff must be finite and non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)

    _, _, points = pitch_grid(config)
    upper_bounds = (
        transition_surface(ball_xy, config) * score_surface(attacking_direction, config)
    ).ravel()
    order = np.argsort(upper_bounds)[::-1]
    best_value = -math.inf
    best_index = int(order[0])
    evaluated = 0

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        control = pitch_control_at_points(
            frame=frame,
            attacking_team_id=attacking_team_id,
            attacking_direction=attacking_direction,
            points=points[indices],
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            ball_xy=ball_xy,
            apply_offside=apply_offside,
            config=config,
        )
        values = control * upper_bounds[indices]
        local_index = int(np.argmax(values))
        if float(values[local_index]) > best_value:
            best_value = float(values[local_index])
            best_index = int(indices[local_index])
        evaluated += len(indices)

        # Strict inequality retains exact evaluation for primary-value ties,
        # allowing a secondary effort tie-break to remain deterministic.
        if best_value > rejection_cutoff:
            return ReferenceOBSOBoundedMaximum(
                maximum=ReferenceOBSOMaximum(
                    value=best_value,
                    x=float(points[best_index, 0]),
                    y=float(points[best_index, 1]),
                    evaluated_cell_count=evaluated,
                ),
                exact=False,
            )

        next_start = start + len(indices)
        if next_start >= len(order) or upper_bounds[order[next_start]] <= best_value:
            return ReferenceOBSOBoundedMaximum(
                maximum=ReferenceOBSOMaximum(
                    value=best_value,
                    x=float(points[best_index, 0]),
                    y=float(points[best_index, 1]),
                    evaluated_cell_count=evaluated,
                ),
                exact=True,
            )

    raise RuntimeError("bounded OBSO maximum search exhausted without a result")


def maximum_reference_obso_in_region(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    region_mask: np.ndarray,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    batch_size: int = 32,
) -> ReferenceOBSOMaximum:
    """Return the exact maximum OBSO over a caller-defined grid region."""

    return _region_maximum_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        region_mask,
        rejection_cutoff=None,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        ball_xy=ball_xy,
        apply_offside=apply_offside,
        config=config,
        batch_size=batch_size,
    ).maximum


def bounded_maximum_reference_obso_in_region(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    region_mask: np.ndarray,
    rejection_cutoff: float,
    velocities: Mapping[str, object] | None = None,
    goalkeeper_ids: Sequence[str] = (),
    ball_xy: tuple[float, float] | None = None,
    apply_offside: bool = True,
    config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
    batch_size: int = 32,
) -> ReferenceOBSOBoundedMaximum:
    """Return an exact regional maximum or a certified minimization rejection."""

    if not math.isfinite(rejection_cutoff) or rejection_cutoff < 0.0:
        raise ValueError("rejection_cutoff must be finite and non-negative")
    return _region_maximum_reference_obso(
        frame,
        attacking_team_id,
        attacking_direction,
        region_mask,
        rejection_cutoff=rejection_cutoff,
        velocities=velocities,
        goalkeeper_ids=goalkeeper_ids,
        ball_xy=ball_xy,
        apply_offside=apply_offside,
        config=config,
        batch_size=batch_size,
    )


def _region_maximum_reference_obso(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    region_mask: np.ndarray,
    rejection_cutoff: float | None,
    velocities: Mapping[str, object] | None,
    goalkeeper_ids: Sequence[str],
    ball_xy: tuple[float, float] | None,
    apply_offside: bool,
    config: ReferenceOBSOConfig,
    batch_size: int,
) -> ReferenceOBSOBoundedMaximum:
    """Shared safe upper-bound search for one side of an R/O partition."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)
    mask = np.asarray(region_mask, dtype=bool)
    expected = (config.grid_cells_y, config.grid_cells_x)
    if mask.shape != expected:
        raise ValueError(f"region_mask has shape {mask.shape}; expected {expected}")

    _, _, points = pitch_grid(config)
    eligible = np.flatnonzero(mask.ravel())
    if len(eligible) == 0:
        return ReferenceOBSOBoundedMaximum(
            maximum=ReferenceOBSOMaximum(
                value=0.0,
                x=float("nan"),
                y=float("nan"),
                evaluated_cell_count=0,
            ),
            exact=True,
        )
    upper_bounds = (
        transition_surface(ball_xy, config) * score_surface(attacking_direction, config)
    ).ravel()
    order = eligible[np.argsort(upper_bounds[eligible])[::-1]]
    best_value = -math.inf
    best_index = int(order[0])
    evaluated = 0

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        control = pitch_control_at_points(
            frame=frame,
            attacking_team_id=attacking_team_id,
            attacking_direction=attacking_direction,
            points=points[indices],
            velocities=velocities,
            goalkeeper_ids=goalkeeper_ids,
            ball_xy=ball_xy,
            apply_offside=apply_offside,
            config=config,
        )
        values = control * upper_bounds[indices]
        local_index = int(np.argmax(values))
        if float(values[local_index]) > best_value:
            best_value = float(values[local_index])
            best_index = int(indices[local_index])
        evaluated += len(indices)

        if rejection_cutoff is not None and best_value > rejection_cutoff:
            return ReferenceOBSOBoundedMaximum(
                maximum=ReferenceOBSOMaximum(
                    value=best_value,
                    x=float(points[best_index, 0]),
                    y=float(points[best_index, 1]),
                    evaluated_cell_count=evaluated,
                ),
                exact=False,
            )
        next_start = start + len(indices)
        if next_start >= len(order) or upper_bounds[order[next_start]] <= best_value:
            return ReferenceOBSOBoundedMaximum(
                maximum=ReferenceOBSOMaximum(
                    value=best_value,
                    x=float(points[best_index, 0]),
                    y=float(points[best_index, 1]),
                    evaluated_cell_count=evaluated,
                ),
                exact=True,
            )

    raise RuntimeError("regional OBSO maximum search exhausted without a result")
