"""Leave-one-match-out empirical two-second movement primitives.

The empirical action space complements mechanistic endpoint solvers with
observed, pooled movement trajectories.  Candidate generation uses only the
decision frame and causal history.  Future tracking is stored in the training
library or used as a held-out diagnostic, never as a target-scene input to the
optimizer candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    FPS,
    BundesligaFrame,
    BundesligaMatchMeta,
    BundesligaObjectState,
)
from .fernandez_influence import (
    FernandezInfluenceEllipse,
    fernandez_influence_ellipse,
)


@dataclass(frozen=True)
class EmpiricalPrimitiveConfig:
    horizon_seconds: float = 2.0
    history_seconds: float = 0.4
    terminal_history_seconds: float = 0.2
    sample_interval_seconds: float = 2.0
    path_sample_seconds: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6, 2.0)
    minimum_heading_speed_mps: float = 0.25
    maximum_observed_speed_mps: float = 12.0

    def validate(self) -> None:
        positive = {
            "horizon_seconds": self.horizon_seconds,
            "history_seconds": self.history_seconds,
            "terminal_history_seconds": self.terminal_history_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "minimum_heading_speed_mps": self.minimum_heading_speed_mps,
            "maximum_observed_speed_mps": self.maximum_observed_speed_mps,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Empirical primitive parameters must be positive: {invalid}")
        if not self.path_sample_seconds:
            raise ValueError("path_sample_seconds cannot be empty")
        if any(
            time_s <= 0.0 or time_s > self.horizon_seconds
            for time_s in self.path_sample_seconds
        ):
            raise ValueError("path samples must lie inside the movement horizon")
        if abs(self.path_sample_seconds[-1] - self.horizon_seconds) > 1e-9:
            raise ValueError("the final path sample must equal the horizon")


@dataclass(frozen=True)
class CausalMotionState:
    vx_mps: float
    vy_mps: float
    speed_mps: float
    ax_mps2: float
    ay_mps2: float
    longitudinal_acceleration_mps2: float
    heading_radians: float
    sample_count: int
    observed_window_seconds: float


@dataclass(frozen=True)
class EmpiricalEndpointConfig:
    grid_resolution_m: float = 1.0
    neighbor_count: int = 2048
    speed_scale_mps: float = 1.5
    acceleration_scale_mps2: float = 2.0
    terminal_heading_bins: int = 8
    variants_per_endpoint: int = 3
    maximum_actions: int = 1600
    maximum_speed_mps: float = 9.0
    field_length_m: float = FIELD_LENGTH
    field_width_m: float = FIELD_WIDTH

    def validate(self) -> None:
        positive = {
            "grid_resolution_m": self.grid_resolution_m,
            "neighbor_count": self.neighbor_count,
            "speed_scale_mps": self.speed_scale_mps,
            "acceleration_scale_mps2": self.acceleration_scale_mps2,
            "terminal_heading_bins": self.terminal_heading_bins,
            "variants_per_endpoint": self.variants_per_endpoint,
            "maximum_actions": self.maximum_actions,
            "maximum_speed_mps": self.maximum_speed_mps,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Empirical endpoint parameters must be positive: {invalid}")


@dataclass(frozen=True)
class EmpiricalEndpointAction:
    player_id: str
    endpoint_x: float
    endpoint_y: float
    terminal_vx_mps: float
    terminal_vy_mps: float
    terminal_speed_mps: float
    primitive_match_id: str
    primitive_player_id: str
    primitive_frame_id: int
    terminal_heading_bin: int
    feature_distance: float
    endpoint_snap_distance_m: float
    fernandez_influence_score: float
    fernandez_normalized_radius: float
    path_xy: tuple[tuple[float, float], ...]
    labels: tuple[str, ...] = ("grid", "empirical")
    optimization_eligible: bool = True

    @property
    def action_id(self) -> str:
        return (
            f"{self.player_id}:{self.endpoint_x:.4f}:{self.endpoint_y:.4f}:"
            f"h{self.terminal_heading_bin}:"
            f"{self.primitive_match_id}:{self.primitive_frame_id}"
        )

    def as_record(self, match_id: str, frame_id: int) -> dict[str, object]:
        return {
            "match_id": match_id,
            "frame_id": int(frame_id),
            "player_id": self.player_id,
            "action_id": f"{match_id}:{frame_id}:{self.action_id}",
            "endpoint_x": self.endpoint_x,
            "endpoint_y": self.endpoint_y,
            "labels": "|".join(self.labels),
            "optimization_eligible": self.optimization_eligible,
            "kinematically_feasible": True,
            "empirically_supported": True,
            "motion_model": "empirical_primitive",
            "terminal_vx_mps": self.terminal_vx_mps,
            "terminal_vy_mps": self.terminal_vy_mps,
            "terminal_speed_mps": self.terminal_speed_mps,
            "terminal_heading_bin": self.terminal_heading_bin,
            "primitive_match_id": self.primitive_match_id,
            "primitive_player_id": self.primitive_player_id,
            "primitive_frame_id": self.primitive_frame_id,
            "feature_distance": self.feature_distance,
            "endpoint_snap_distance_m": self.endpoint_snap_distance_m,
            "fernandez_influence_score": self.fernandez_influence_score,
            "fernandez_normalized_radius": self.fernandez_normalized_radius,
            "path_xy": json.dumps(self.path_xy, separators=(",", ":")),
            "failure_reason": None,
        }


@dataclass(frozen=True)
class EmpiricalEndpointActionSet:
    player_id: str
    start_x: float
    start_y: float
    state: CausalMotionState
    influence_ellipse: FernandezInfluenceEllipse
    actions: tuple[EmpiricalEndpointAction, ...]
    library_size: int
    neighbor_count: int
    source_match_ids: tuple[str, ...]

    @property
    def optimization_actions(self) -> tuple[EmpiricalEndpointAction, ...]:
        return tuple(action for action in self.actions if action.optimization_eligible)


@dataclass(frozen=True)
class EmpiricalSupportDiagnostics:
    endpoint_supported: bool | None
    endpoint_heading_supported: bool | None
    nearest_endpoint_distance_m: float | None
    nearest_terminal_heading_difference_degrees: float | None


def _estimator_weights(
    sample_count: int,
    fps: float = FPS,
    quadratic: bool = True,
) -> np.ndarray:
    times = (np.arange(sample_count, dtype=float) - (sample_count - 1)) / fps
    if quadratic and sample_count >= 3:
        design = np.column_stack([times**2, times, np.ones(sample_count)])
    else:
        design = np.column_stack([times, np.ones(sample_count)])
    return np.linalg.pinv(design)


def _causal_state_from_arrays(
    xs: np.ndarray,
    ys: np.ndarray,
    config: EmpiricalPrimitiveConfig,
) -> CausalMotionState:
    sample_count = len(xs)
    if sample_count < 2 or len(ys) != sample_count:
        raise ValueError("causal state requires aligned position samples")
    quadratic = sample_count >= 3
    weights = _estimator_weights(sample_count, quadratic=quadratic)
    x_coefficients = weights @ xs
    y_coefficients = weights @ ys
    if quadratic:
        ax = float(2.0 * x_coefficients[0])
        ay = float(2.0 * y_coefficients[0])
        vx = float(x_coefficients[1])
        vy = float(y_coefficients[1])
    else:
        ax = 0.0
        ay = 0.0
        vx = float(x_coefficients[0])
        vy = float(y_coefficients[0])
    speed = math.hypot(vx, vy)
    if speed >= config.minimum_heading_speed_mps:
        heading = math.atan2(vy, vx)
    elif math.hypot(ax, ay) >= config.minimum_heading_speed_mps:
        heading = math.atan2(ay, ax)
    else:
        heading = 0.0
    longitudinal_acceleration = ax * math.cos(heading) + ay * math.sin(heading)
    return CausalMotionState(
        vx_mps=vx,
        vy_mps=vy,
        speed_mps=speed,
        ax_mps2=ax,
        ay_mps2=ay,
        longitudinal_acceleration_mps2=float(longitudinal_acceleration),
        heading_radians=float(heading),
        sample_count=sample_count,
        observed_window_seconds=float((sample_count - 1) / FPS),
    )


def causal_motion_state_from_frames(
    frames: Iterable[BundesligaFrame],
    player_id: str,
    end_frame_id: int,
    config: EmpiricalPrimitiveConfig = EmpiricalPrimitiveConfig(),
) -> CausalMotionState:
    config.validate()
    history_frames = int(round(config.history_seconds * FPS))
    samples = []
    for frame in frames:
        if frame.frame_id > end_frame_id or end_frame_id - frame.frame_id > history_frames:
            continue
        player = frame.players.get(player_id)
        if player is not None:
            samples.append((frame.frame_id, player.x, player.y))
    samples.sort(key=lambda item: item[0])
    if len(samples) < 2:
        return CausalMotionState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(samples), 0.0)
    return _causal_state_from_arrays(
        np.asarray([sample[1] for sample in samples], dtype=float),
        np.asarray([sample[2] for sample in samples], dtype=float),
        config,
    )


def _terminal_velocity(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[float, float, float]:
    weights = _estimator_weights(len(xs), quadratic=False)
    vx = float((weights @ xs)[0])
    vy = float((weights @ ys)[0])
    return vx, vy, math.hypot(vx, vy)


def _to_local(
    dx: float,
    dy: float,
    heading: float,
) -> tuple[float, float]:
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return (
        float(dx * cosine + dy * sine),
        float(-dx * sine + dy * cosine),
    )


def _to_world(
    forward: np.ndarray | float,
    lateral: np.ndarray | float,
    heading: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return (
        forward * cosine - lateral * sine,
        forward * sine + lateral * cosine,
    )


def extract_track_primitives(
    match_id: str,
    player_id: str,
    team_id: str,
    game_section: str,
    frame_ids: Sequence[int],
    xs: Sequence[float],
    ys: Sequence[float],
    config: EmpiricalPrimitiveConfig = EmpiricalPrimitiveConfig(),
) -> pd.DataFrame:
    """Extract deterministic non-overlapping primitives from one player track."""

    config.validate()
    ids = np.asarray(frame_ids, dtype=np.int64)
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    if not (len(ids) == len(x_values) == len(y_values)):
        raise ValueError("track arrays must have equal lengths")
    if len(ids) < 2:
        return pd.DataFrame()

    history_frames = int(round(config.history_seconds * FPS))
    terminal_frames = int(round(config.terminal_history_seconds * FPS))
    horizon_frames = int(round(config.horizon_seconds * FPS))
    interval_frames = int(round(config.sample_interval_seconds * FPS))
    path_offsets = tuple(int(round(time_s * FPS)) for time_s in config.path_sample_seconds)
    rows = []
    first_index = history_frames
    last_index = len(ids) - horizon_frames - 1
    for anchor_index in range(first_index, last_index + 1, interval_frames):
        terminal_index = anchor_index + horizon_frames
        if (
            ids[anchor_index] - ids[anchor_index - history_frames] != history_frames
            or ids[terminal_index] - ids[anchor_index] != horizon_frames
            or ids[terminal_index] - ids[terminal_index - terminal_frames] != terminal_frames
        ):
            continue
        # FrameSet tracks can contain discontinuities when a player leaves or
        # re-enters the observed area.  Initial/terminal regression alone does
        # not detect a one-frame teleport in the middle of a primitive.
        primitive_start_index = anchor_index - history_frames
        primitive_x = x_values[primitive_start_index : terminal_index + 1]
        primitive_y = y_values[primitive_start_index : terminal_index + 1]
        instantaneous_speeds = FPS * np.hypot(
            np.diff(primitive_x),
            np.diff(primitive_y),
        )
        if (
            not np.isfinite(primitive_x).all()
            or not np.isfinite(primitive_y).all()
            or not np.isfinite(instantaneous_speeds).all()
            or (
                len(instantaneous_speeds)
                and float(np.max(instantaneous_speeds))
                > config.maximum_observed_speed_mps
            )
        ):
            continue
        state = _causal_state_from_arrays(
            x_values[anchor_index - history_frames : anchor_index + 1],
            y_values[anchor_index - history_frames : anchor_index + 1],
            config,
        )
        terminal_vx, terminal_vy, terminal_speed = _terminal_velocity(
            x_values[terminal_index - terminal_frames : terminal_index + 1],
            y_values[terminal_index - terminal_frames : terminal_index + 1],
        )
        if (
            state.speed_mps > config.maximum_observed_speed_mps
            or terminal_speed > config.maximum_observed_speed_mps
        ):
            continue
        start_x = float(x_values[anchor_index])
        start_y = float(y_values[anchor_index])
        endpoint_forward, endpoint_lateral = _to_local(
            float(x_values[terminal_index] - start_x),
            float(y_values[terminal_index] - start_y),
            state.heading_radians,
        )
        terminal_forward, terminal_lateral = _to_local(
            terminal_vx,
            terminal_vy,
            state.heading_radians,
        )
        row: dict[str, object] = {
            "match_id": match_id,
            "player_id": player_id,
            "team_id": team_id,
            "game_section": game_section,
            "frame_id": int(ids[anchor_index]),
            "initial_speed_mps": state.speed_mps,
            "initial_longitudinal_acceleration_mps2": (
                state.longitudinal_acceleration_mps2
            ),
            "endpoint_forward_m": endpoint_forward,
            "endpoint_lateral_m": endpoint_lateral,
            "terminal_v_forward_mps": terminal_forward,
            "terminal_v_lateral_mps": terminal_lateral,
            "terminal_speed_mps": terminal_speed,
            "maximum_path_speed_mps": (
                float(np.max(instantaneous_speeds))
                if len(instantaneous_speeds)
                else 0.0
            ),
        }
        for index, offset in enumerate(path_offsets, start=1):
            path_forward, path_lateral = _to_local(
                float(x_values[anchor_index + offset] - start_x),
                float(y_values[anchor_index + offset] - start_y),
                state.heading_radians,
            )
            row[f"path_{index}_forward_m"] = path_forward
            row[f"path_{index}_lateral_m"] = path_lateral
        rows.append(row)
    return pd.DataFrame(rows)


def extract_match_primitives(
    positions_xml: str | Path,
    metadata: BundesligaMatchMeta,
    config: EmpiricalPrimitiveConfig = EmpiricalPrimitiveConfig(),
) -> pd.DataFrame:
    """Stream one Sportec XML match and extract outfield-player primitives."""

    config.validate()
    goalkeeper_ids = {
        player.player_id
        for player in metadata.players.values()
        if player.position == "TW"
    }
    chunks = []
    current: dict[str, str] | None = None
    collect = False
    frame_ids: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for event, elem in ET.iterparse(positions_xml, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current = dict(elem.attrib)
            player_id = current.get("PersonId", "")
            team_id = current.get("TeamId", "")
            collect = team_id.startswith("DFL-CLU-") and player_id not in goalkeeper_ids
            frame_ids = []
            xs = []
            ys = []
            continue
        if event == "end" and elem.tag == "Frame" and current is not None:
            if collect:
                frame_ids.append(int(elem.attrib["N"]))
                xs.append(float(elem.attrib["X"]))
                ys.append(float(elem.attrib["Y"]))
            elem.clear()
            continue
        if event == "end" and elem.tag == "FrameSet":
            if collect and current is not None:
                chunk = extract_track_primitives(
                    metadata.match_id,
                    current.get("PersonId", ""),
                    current.get("TeamId", ""),
                    current.get("GameSection", ""),
                    frame_ids,
                    xs,
                    ys,
                    config,
                )
                if not chunk.empty:
                    chunks.append(chunk)
            current = None
            collect = False
            elem.clear()
        elif event == "end":
            elem.clear()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


class EmpiricalPrimitiveLibrary:
    def __init__(
        self,
        frame: pd.DataFrame,
        config: EmpiricalPrimitiveConfig = EmpiricalPrimitiveConfig(),
    ) -> None:
        config.validate()
        required = {
            "match_id",
            "player_id",
            "frame_id",
            "initial_speed_mps",
            "initial_longitudinal_acceleration_mps2",
            "endpoint_forward_m",
            "endpoint_lateral_m",
            "terminal_v_forward_mps",
            "terminal_v_lateral_mps",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Empirical primitive library missing columns: {missing}")
        self.frame = frame.reset_index(drop=True)
        self.config = config

    @property
    def source_match_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame["match_id"].astype(str).unique()))

    def assert_excludes_match(self, match_id: str) -> None:
        if match_id in self.source_match_ids:
            raise ValueError(f"Target match leakage: {match_id} is in empirical library")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for column in self.frame.columns:
            values = self.frame[column].to_numpy()
            # NumPy stores pandas object columns as pickled arrays by default.
            # Convert our identifier columns to ordinary Unicode arrays so the
            # library remains portable and can be loaded with allow_pickle=False.
            if values.dtype.kind == "O":
                values = self.frame[column].astype(str).to_numpy(dtype=str)
            arrays[column] = values
        arrays["__config_json__"] = np.asarray(
            [json.dumps(self.config.__dict__, separators=(",", ":"))]
        )
        np.savez_compressed(target, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalPrimitiveLibrary":
        with np.load(path, allow_pickle=False) as data:
            columns = [name for name in data.files if name != "__config_json__"]
            frame = pd.DataFrame({column: data[column] for column in columns})
            raw_config = json.loads(str(data["__config_json__"][0]))
        if "path_sample_seconds" in raw_config:
            raw_config["path_sample_seconds"] = tuple(raw_config["path_sample_seconds"])
        return cls(frame, EmpiricalPrimitiveConfig(**raw_config))


def _snap_global_grid(
    value: np.ndarray,
    half_extent: float,
    resolution: float,
) -> np.ndarray:
    return (
        np.rint((value + half_extent) / resolution) * resolution - half_extent
    )


def _ellipse_score(
    x: float,
    y: float,
    ellipse: FernandezInfluenceEllipse,
) -> tuple[float, float]:
    dx = x - ellipse.center_x
    dy = y - ellipse.center_y
    angle = math.radians(ellipse.angle_degrees)
    along = dx * math.cos(angle) + dy * math.sin(angle)
    lateral = -dx * math.sin(angle) + dy * math.cos(angle)
    normalized_squared = (
        (along / max(ellipse.major_radius_m, 1e-9)) ** 2
        + (lateral / max(ellipse.minor_radius_m, 1e-9)) ** 2
    )
    return float(math.exp(-2.0 * normalized_squared)), float(math.sqrt(normalized_squared))


def _terminal_heading_bin(
    vx: float,
    vy: float,
    heading: float,
    bin_count: int,
) -> int:
    relative = (math.atan2(vy, vx) - heading + 2.0 * math.pi) % (2.0 * math.pi)
    return int(math.floor(relative / (2.0 * math.pi / bin_count))) % bin_count


def generate_empirical_endpoint_actions(
    player: BundesligaObjectState,
    ball: BundesligaObjectState,
    state: CausalMotionState,
    library: EmpiricalPrimitiveLibrary,
    config: EmpiricalEndpointConfig = EmpiricalEndpointConfig(),
) -> EmpiricalEndpointActionSet:
    """Generate grid actions from causally matched empirical primitives."""

    config.validate()
    frame = library.frame
    if frame.empty:
        raise ValueError("Empirical primitive library is empty")
    feature_distance = np.sqrt(
        (
            (frame["initial_speed_mps"].to_numpy(dtype=float) - state.speed_mps)
            / config.speed_scale_mps
        )
        ** 2
        + (
            (
                frame["initial_longitudinal_acceleration_mps2"].to_numpy(dtype=float)
                - state.longitudinal_acceleration_mps2
            )
            / config.acceleration_scale_mps2
        )
        ** 2
    )
    neighbor_count = min(config.neighbor_count, len(frame))
    if neighbor_count == len(frame):
        selected_indices = np.arange(len(frame))
    else:
        selected_indices = np.argpartition(feature_distance, neighbor_count - 1)[
            :neighbor_count
        ]
    selected_indices = selected_indices[
        np.argsort(feature_distance[selected_indices], kind="stable")
    ]
    selected = frame.iloc[selected_indices].reset_index(drop=True)
    selected_feature_distance = feature_distance[selected_indices]

    forward = selected["endpoint_forward_m"].to_numpy(dtype=float)
    lateral = selected["endpoint_lateral_m"].to_numpy(dtype=float)
    terminal_forward = selected["terminal_v_forward_mps"].to_numpy(dtype=float)
    terminal_lateral = selected["terminal_v_lateral_mps"].to_numpy(dtype=float)
    path_forward_columns = [
        f"path_{index}_forward_m"
        for index in range(1, len(library.config.path_sample_seconds) + 1)
    ]
    path_lateral_columns = [
        f"path_{index}_lateral_m"
        for index in range(1, len(library.config.path_sample_seconds) + 1)
    ]
    path_forward = selected[path_forward_columns].to_numpy(dtype=float)
    path_lateral = selected[path_lateral_columns].to_numpy(dtype=float)

    # Pooling is made left/right symmetric in the velocity-aligned coordinate
    # system.  The original and reflected samples retain distinct terminal
    # headings and trajectory primitives.
    forward = np.concatenate([forward, forward])
    lateral = np.concatenate([lateral, -lateral])
    terminal_forward = np.concatenate([terminal_forward, terminal_forward])
    terminal_lateral = np.concatenate([terminal_lateral, -terminal_lateral])
    path_forward = np.concatenate([path_forward, path_forward], axis=0)
    path_lateral = np.concatenate([path_lateral, -path_lateral], axis=0)
    selected_feature_distance = np.concatenate(
        [selected_feature_distance, selected_feature_distance]
    )
    source_rows = pd.concat([selected, selected], ignore_index=True)

    dx, dy = _to_world(forward, lateral, state.heading_radians)
    raw_x = player.x + np.asarray(dx)
    raw_y = player.y + np.asarray(dy)
    grid_x = _snap_global_grid(
        raw_x,
        config.field_length_m / 2.0,
        config.grid_resolution_m,
    )
    grid_y = _snap_global_grid(
        raw_y,
        config.field_width_m / 2.0,
        config.grid_resolution_m,
    )
    terminal_vx, terminal_vy = _to_world(
        terminal_forward,
        terminal_lateral,
        state.heading_radians,
    )
    terminal_vx = np.asarray(terminal_vx)
    terminal_vy = np.asarray(terminal_vy)
    terminal_speed = np.hypot(terminal_vx, terminal_vy)
    ellipse = fernandez_influence_ellipse(
        (player.x, player.y),
        (state.vx_mps, state.vy_mps),
        (ball.x, ball.y),
    )

    candidates: dict[tuple[float, float], list[tuple[float, EmpiricalEndpointAction]]] = {}
    path_fractions = np.asarray(library.config.path_sample_seconds) / library.config.horizon_seconds
    for index in range(len(grid_x)):
        x = float(grid_x[index])
        y = float(grid_y[index])
        if (
            x < -config.field_length_m / 2.0
            or x > config.field_length_m / 2.0
            or y < -config.field_width_m / 2.0
            or y > config.field_width_m / 2.0
            or terminal_speed[index] > config.maximum_speed_mps + 1e-9
        ):
            continue
        snap_distance = math.hypot(x - raw_x[index], y - raw_y[index])
        influence_score, normalized_radius = _ellipse_score(x, y, ellipse)
        path_dx, path_dy = _to_world(
            path_forward[index],
            path_lateral[index],
            state.heading_radians,
        )
        correction_x = x - raw_x[index]
        correction_y = y - raw_y[index]
        path_xy = tuple(
            (
                float(player.x + path_dx[path_index] + correction_x * fraction),
                float(player.y + path_dy[path_index] + correction_y * fraction),
            )
            for path_index, fraction in enumerate(path_fractions)
        )
        heading_bin = _terminal_heading_bin(
            float(terminal_vx[index]),
            float(terminal_vy[index]),
            state.heading_radians,
            config.terminal_heading_bins,
        )
        source = source_rows.iloc[index]
        action = EmpiricalEndpointAction(
            player_id=player.object_id,
            endpoint_x=x,
            endpoint_y=y,
            terminal_vx_mps=float(terminal_vx[index]),
            terminal_vy_mps=float(terminal_vy[index]),
            terminal_speed_mps=float(terminal_speed[index]),
            primitive_match_id=str(source["match_id"]),
            primitive_player_id=str(source["player_id"]),
            primitive_frame_id=int(source["frame_id"]),
            terminal_heading_bin=heading_bin,
            feature_distance=float(selected_feature_distance[index]),
            endpoint_snap_distance_m=float(snap_distance),
            fernandez_influence_score=influence_score,
            fernandez_normalized_radius=normalized_radius,
            path_xy=path_xy,
        )
        rank = (
            float(selected_feature_distance[index])
            + snap_distance
            + 0.15 * (1.0 - influence_score)
        )
        candidates.setdefault((x, y), []).append((rank, action))

    retained = []
    for endpoint_candidates in candidates.values():
        endpoint_candidates.sort(key=lambda item: (item[0], item[1].action_id))
        used_bins = set()
        for _, action in endpoint_candidates:
            if action.terminal_heading_bin in used_bins:
                continue
            retained.append(action)
            used_bins.add(action.terminal_heading_bin)
            if len(used_bins) >= config.variants_per_endpoint:
                break
    retained.sort(
        key=lambda action: (
            action.feature_distance
            + action.endpoint_snap_distance_m
            + 0.15 * (1.0 - action.fernandez_influence_score),
            action.endpoint_x,
            action.endpoint_y,
            action.terminal_heading_bin,
        )
    )
    retained = retained[: config.maximum_actions]
    return EmpiricalEndpointActionSet(
        player_id=player.object_id,
        start_x=float(player.x),
        start_y=float(player.y),
        state=state,
        influence_ellipse=ellipse,
        actions=tuple(retained),
        library_size=len(frame),
        neighbor_count=neighbor_count,
        source_match_ids=library.source_match_ids,
    )


def observed_empirical_support(
    action_set: EmpiricalEndpointActionSet,
    observed_endpoint_xy: tuple[float, float] | None,
    tolerance_m: float = 1.0,
) -> tuple[bool | None, float | None]:
    if observed_endpoint_xy is None:
        return None, None
    if not action_set.optimization_actions:
        return False, math.inf
    nearest = min(
        math.hypot(
            action.endpoint_x - observed_endpoint_xy[0],
            action.endpoint_y - observed_endpoint_xy[1],
        )
        for action in action_set.optimization_actions
    )
    return bool(nearest <= tolerance_m), float(nearest)


def observed_empirical_support_diagnostics(
    action_set: EmpiricalEndpointActionSet,
    observed_endpoint_xy: tuple[float, float] | None,
    observed_terminal_velocity_xy: tuple[float, float] | None,
    endpoint_tolerance_m: float = 1.0,
    heading_tolerance_degrees: float = 45.0,
    minimum_heading_speed_mps: float = 0.5,
) -> EmpiricalSupportDiagnostics:
    """Evaluate held-out endpoint and arrival-direction coverage.

    This is an audit diagnostic only.  The observed target-match future is not
    used while generating candidates.
    """

    if observed_endpoint_xy is None:
        return EmpiricalSupportDiagnostics(None, None, None, None)
    actions = action_set.optimization_actions
    if not actions:
        return EmpiricalSupportDiagnostics(False, False, math.inf, math.inf)
    distances = np.asarray(
        [
            math.hypot(
                action.endpoint_x - observed_endpoint_xy[0],
                action.endpoint_y - observed_endpoint_xy[1],
            )
            for action in actions
        ],
        dtype=float,
    )
    nearest_distance = float(np.min(distances))
    endpoint_supported = nearest_distance <= endpoint_tolerance_m
    if observed_terminal_velocity_xy is None:
        return EmpiricalSupportDiagnostics(
            endpoint_supported,
            endpoint_supported,
            nearest_distance,
            None,
        )
    observed_vx, observed_vy = observed_terminal_velocity_xy
    observed_speed = math.hypot(observed_vx, observed_vy)
    if observed_speed < minimum_heading_speed_mps:
        return EmpiricalSupportDiagnostics(
            endpoint_supported,
            endpoint_supported,
            nearest_distance,
            None,
        )
    nearby_indices = np.flatnonzero(distances <= endpoint_tolerance_m)
    if len(nearby_indices) == 0:
        return EmpiricalSupportDiagnostics(
            False,
            False,
            nearest_distance,
            math.inf,
        )

    observed_heading = math.atan2(observed_vy, observed_vx)
    differences = []
    for index in nearby_indices:
        action = actions[int(index)]
        if action.terminal_speed_mps < minimum_heading_speed_mps:
            continue
        action_heading = math.atan2(action.terminal_vy_mps, action.terminal_vx_mps)
        difference = abs(
            (action_heading - observed_heading + math.pi) % (2.0 * math.pi)
            - math.pi
        )
        differences.append(math.degrees(difference))
    nearest_heading = min(differences) if differences else math.inf
    return EmpiricalSupportDiagnostics(
        endpoint_supported,
        endpoint_supported and nearest_heading <= heading_tolerance_degrees,
        nearest_distance,
        float(nearest_heading),
    )
