"""Causal reference rollouts for non-intervened players.

These models predict a common background state from information available at
the decision frame.  Observed future tracking is deliberately outside this
module and is used only by the benchmark script as a held-out target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .bundesliga import FIELD_LENGTH, FIELD_WIDTH
from .empirical_action_space import CausalMotionState, EmpiricalPrimitiveLibrary


@dataclass(frozen=True)
class BackgroundRolloutConfig:
    horizons_seconds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    maximum_speed_mps: float = 9.0
    empirical_neighbor_count: int = 512
    empirical_speed_scale_mps: float = 1.5
    empirical_acceleration_scale_mps2: float = 2.0
    empirical_kernel_bandwidth: float = 1.0
    velocity_interval_seconds: float = 0.2
    field_length_m: float = FIELD_LENGTH
    field_width_m: float = FIELD_WIDTH

    def validate(self) -> None:
        positive = {
            "maximum_speed_mps": self.maximum_speed_mps,
            "empirical_neighbor_count": self.empirical_neighbor_count,
            "empirical_speed_scale_mps": self.empirical_speed_scale_mps,
            "empirical_acceleration_scale_mps2": (
                self.empirical_acceleration_scale_mps2
            ),
            "empirical_kernel_bandwidth": self.empirical_kernel_bandwidth,
            "velocity_interval_seconds": self.velocity_interval_seconds,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Background rollout parameters must be positive: {invalid}")
        if not self.horizons_seconds or any(
            horizon <= 0 for horizon in self.horizons_seconds
        ):
            raise ValueError("horizons_seconds must contain positive values")


@dataclass(frozen=True)
class BackgroundPrediction:
    model: str
    horizon_seconds: float
    x: float
    y: float
    vx_mps: float
    vy_mps: float
    speed_mps: float
    boundary_clipped: bool
    empirical_neighbor_count: int = 0
    empirical_mean_feature_distance: float | None = None
    fallback_model: str | None = None


def capped_velocity(
    vx_mps: float,
    vy_mps: float,
    maximum_speed_mps: float,
) -> tuple[float, float]:
    speed = math.hypot(vx_mps, vy_mps)
    if speed <= maximum_speed_mps or speed <= 1e-12:
        return float(vx_mps), float(vy_mps)
    scale = maximum_speed_mps / speed
    return float(vx_mps * scale), float(vy_mps * scale)


def _clip_prediction(
    model: str,
    horizon_seconds: float,
    x: float,
    y: float,
    vx_mps: float,
    vy_mps: float,
    config: BackgroundRolloutConfig,
    empirical_neighbor_count: int = 0,
    empirical_mean_feature_distance: float | None = None,
    fallback_model: str | None = None,
) -> BackgroundPrediction:
    half_length = config.field_length_m / 2.0
    half_width = config.field_width_m / 2.0
    clipped_x = min(half_length, max(-half_length, x))
    clipped_y = min(half_width, max(-half_width, y))
    x_hit = not math.isclose(clipped_x, x, abs_tol=1e-9)
    y_hit = not math.isclose(clipped_y, y, abs_tol=1e-9)
    terminal_vx = 0.0 if x_hit else vx_mps
    terminal_vy = 0.0 if y_hit else vy_mps
    terminal_vx, terminal_vy = capped_velocity(
        terminal_vx,
        terminal_vy,
        config.maximum_speed_mps,
    )
    return BackgroundPrediction(
        model=model,
        horizon_seconds=float(horizon_seconds),
        x=float(clipped_x),
        y=float(clipped_y),
        vx_mps=float(terminal_vx),
        vy_mps=float(terminal_vy),
        speed_mps=float(math.hypot(terminal_vx, terminal_vy)),
        boundary_clipped=bool(x_hit or y_hit),
        empirical_neighbor_count=int(empirical_neighbor_count),
        empirical_mean_feature_distance=empirical_mean_feature_distance,
        fallback_model=fallback_model,
    )


def predict_hold(
    start_x: float,
    start_y: float,
    horizon_seconds: float,
    config: BackgroundRolloutConfig = BackgroundRolloutConfig(),
    *,
    model: str = "hold",
    fallback_model: str | None = None,
) -> BackgroundPrediction:
    config.validate()
    return _clip_prediction(
        model,
        horizon_seconds,
        start_x,
        start_y,
        0.0,
        0.0,
        config,
        fallback_model=fallback_model,
    )


def predict_constant_velocity(
    start_x: float,
    start_y: float,
    state: CausalMotionState,
    horizon_seconds: float,
    config: BackgroundRolloutConfig = BackgroundRolloutConfig(),
) -> BackgroundPrediction:
    config.validate()
    vx, vy = capped_velocity(
        state.vx_mps,
        state.vy_mps,
        config.maximum_speed_mps,
    )
    return _clip_prediction(
        "constant_velocity",
        horizon_seconds,
        start_x + vx * horizon_seconds,
        start_y + vy * horizon_seconds,
        vx,
        vy,
        config,
    )


def damped_displacement_seconds(
    horizon_seconds: float,
    damping_lambda_per_second: float,
) -> float:
    if horizon_seconds < 0:
        raise ValueError("horizon_seconds cannot be negative")
    if damping_lambda_per_second < 0:
        raise ValueError("damping lambda cannot be negative")
    if damping_lambda_per_second <= 1e-12:
        return float(horizon_seconds)
    return float(
        -math.expm1(-damping_lambda_per_second * horizon_seconds)
        / damping_lambda_per_second
    )


def predict_damped_constant_velocity(
    start_x: float,
    start_y: float,
    state: CausalMotionState,
    horizon_seconds: float,
    damping_lambda_per_second: float,
    config: BackgroundRolloutConfig = BackgroundRolloutConfig(),
) -> BackgroundPrediction:
    config.validate()
    vx, vy = capped_velocity(
        state.vx_mps,
        state.vy_mps,
        config.maximum_speed_mps,
    )
    elapsed = damped_displacement_seconds(
        horizon_seconds,
        damping_lambda_per_second,
    )
    decay = math.exp(-damping_lambda_per_second * horizon_seconds)
    return _clip_prediction(
        "damped_constant_velocity",
        horizon_seconds,
        start_x + vx * elapsed,
        start_y + vy * elapsed,
        vx * decay,
        vy * decay,
        config,
    )


def fit_damping_lambda(
    library: EmpiricalPrimitiveLibrary,
    candidate_lambdas: Iterable[float] | None = None,
) -> tuple[float, np.ndarray]:
    """Fit one non-negative decay rate using only source-library primitives.

    The fit minimizes mean squared local displacement error over every stored
    path horizon.  Sufficient statistics keep the search memory efficient.
    """

    if candidate_lambdas is None:
        lambdas = np.linspace(0.0, 3.0, 301, dtype=float)
    else:
        lambdas = np.asarray(tuple(candidate_lambdas), dtype=float)
    if not len(lambdas) or np.any(lambdas < 0):
        raise ValueError("candidate lambdas must be a non-empty non-negative set")

    frame = library.frame
    speed = frame["initial_speed_mps"].to_numpy(dtype=float)
    constant = 0.0
    cross_by_time: list[float] = []
    speed_squared_sum = float(np.sum(speed**2))
    for index, _ in enumerate(library.config.path_sample_seconds, start=1):
        forward = frame[f"path_{index}_forward_m"].to_numpy(dtype=float)
        lateral = frame[f"path_{index}_lateral_m"].to_numpy(dtype=float)
        constant += float(np.sum(forward**2 + lateral**2))
        cross_by_time.append(float(np.sum(speed * forward)))

    sse = np.full(len(lambdas), constant, dtype=float)
    for time_s, cross in zip(
        library.config.path_sample_seconds,
        cross_by_time,
        strict=True,
    ):
        tau = np.asarray(
            [damped_displacement_seconds(time_s, value) for value in lambdas]
        )
        sse += -2.0 * tau * cross + tau**2 * speed_squared_sum
    best_index = int(np.argmin(sse))
    return float(lambdas[best_index]), sse / (
        len(frame) * len(library.config.path_sample_seconds)
    )


def _to_world(
    forward: float,
    lateral: float,
    heading_radians: float,
) -> tuple[float, float]:
    cosine = math.cos(heading_radians)
    sine = math.sin(heading_radians)
    return (
        float(forward * cosine - lateral * sine),
        float(forward * sine + lateral * cosine),
    )


class EmpiricalBackgroundPredictor:
    """Kernel-weighted reference trajectory from leave-one-match-out data."""

    def __init__(
        self,
        library: EmpiricalPrimitiveLibrary,
        config: BackgroundRolloutConfig = BackgroundRolloutConfig(),
    ) -> None:
        config.validate()
        if library.frame.empty:
            raise ValueError("Empirical primitive library is empty")
        self.library = library
        self.config = config
        frame = library.frame
        self._speed = frame["initial_speed_mps"].to_numpy(dtype=float)
        self._longitudinal_acceleration = frame[
            "initial_longitudinal_acceleration_mps2"
        ].to_numpy(dtype=float)
        path_forward_columns = [
            f"path_{index}_forward_m"
            for index in range(1, len(library.config.path_sample_seconds) + 1)
        ]
        path_lateral_columns = [
            f"path_{index}_lateral_m"
            for index in range(1, len(library.config.path_sample_seconds) + 1)
        ]
        self._path_forward = frame[path_forward_columns].to_numpy(dtype=float)
        self._path_lateral = frame[path_lateral_columns].to_numpy(dtype=float)
        self._path_times = np.asarray(
            (0.0, *library.config.path_sample_seconds),
            dtype=float,
        )

    def _neighbor_indices_and_weights(
        self,
        state: CausalMotionState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance = np.sqrt(
            (
                (self._speed - state.speed_mps)
                / self.config.empirical_speed_scale_mps
            )
            ** 2
            + (
                (
                    self._longitudinal_acceleration
                    - state.longitudinal_acceleration_mps2
                )
                / self.config.empirical_acceleration_scale_mps2
            )
            ** 2
        )
        count = min(self.config.empirical_neighbor_count, len(distance))
        if count == len(distance):
            indices = np.arange(len(distance))
        else:
            indices = np.argpartition(distance, count - 1)[:count]
        local_distance = distance[indices]
        bandwidth = self.config.empirical_kernel_bandwidth
        weights = np.exp(-0.5 * (local_distance / bandwidth) ** 2)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 1e-12:
            weights = np.full(len(indices), 1.0 / len(indices))
        else:
            weights /= total
        return indices, weights, local_distance

    def _weighted_local_position(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
        horizon_seconds: float,
    ) -> tuple[float, float]:
        if horizon_seconds < 0 or horizon_seconds > self._path_times[-1] + 1e-9:
            raise ValueError("Empirical prediction horizon lies outside the library")
        if horizon_seconds <= 1e-12:
            return 0.0, 0.0
        upper = int(np.searchsorted(self._path_times, horizon_seconds, side="left"))
        upper = min(max(1, upper), len(self._path_times) - 1)
        lower = upper - 1
        lower_time = self._path_times[lower]
        upper_time = self._path_times[upper]
        fraction = (horizon_seconds - lower_time) / (upper_time - lower_time)
        if lower == 0:
            lower_forward = np.zeros(len(indices), dtype=float)
            lower_lateral = np.zeros(len(indices), dtype=float)
        else:
            lower_forward = self._path_forward[indices, lower - 1]
            lower_lateral = self._path_lateral[indices, lower - 1]
        upper_forward = self._path_forward[indices, upper - 1]
        upper_lateral = self._path_lateral[indices, upper - 1]
        forward = lower_forward + fraction * (upper_forward - lower_forward)
        lateral = lower_lateral + fraction * (upper_lateral - lower_lateral)
        return float(np.sum(weights * forward)), float(np.sum(weights * lateral))

    def predict_many(
        self,
        start_x: float,
        start_y: float,
        state: CausalMotionState,
        horizons_seconds: Iterable[float] | None = None,
    ) -> tuple[BackgroundPrediction, ...]:
        horizons = tuple(
            self.config.horizons_seconds
            if horizons_seconds is None
            else horizons_seconds
        )
        indices, weights, distances = self._neighbor_indices_and_weights(state)
        results = []
        for horizon in horizons:
            forward, lateral = self._weighted_local_position(
                indices,
                weights,
                horizon,
            )
            previous_time = max(0.0, horizon - self.config.velocity_interval_seconds)
            previous_forward, previous_lateral = self._weighted_local_position(
                indices,
                weights,
                previous_time,
            )
            interval = max(horizon - previous_time, 1e-9)
            local_v_forward = (forward - previous_forward) / interval
            local_v_lateral = (lateral - previous_lateral) / interval
            dx, dy = _to_world(forward, lateral, state.heading_radians)
            vx, vy = _to_world(
                local_v_forward,
                local_v_lateral,
                state.heading_radians,
            )
            results.append(
                _clip_prediction(
                    "empirical_reference",
                    horizon,
                    start_x + dx,
                    start_y + dy,
                    vx,
                    vy,
                    self.config,
                    empirical_neighbor_count=len(indices),
                    empirical_mean_feature_distance=float(np.mean(distances)),
                )
            )
        return tuple(results)
