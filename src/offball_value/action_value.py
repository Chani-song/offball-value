"""Fixed-reference-defense OBSO summaries for counterfactual actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .bundesliga import BundesligaFrame, BundesligaMatchMeta
from .counterfactual_state import CounterfactualState
from .reference_obso import (
    ReferenceOBSOConfig,
    ReferenceOBSOSurface,
    evaluate_reference_obso,
    offside_attacker_ids,
)


@dataclass(frozen=True)
class ThreatAggregationConfig:
    top_k_peaks: int = 5
    peak_suppression_radius_cells: float = 2.0
    logsumexp_temperature: float = 0.01

    def validate(self) -> None:
        if self.top_k_peaks <= 0:
            raise ValueError("top_k_peaks must be positive")
        if self.peak_suppression_radius_cells < 0:
            raise ValueError("peak_suppression_radius_cells cannot be negative")
        if self.logsumexp_temperature <= 0:
            raise ValueError("logsumexp_temperature must be positive")


@dataclass(frozen=True)
class ThreatAggregate:
    maximum: float
    maximum_x: float
    maximum_y: float
    peak_values: tuple[float, ...]
    top_k_peak_mean: float
    normalized_logsumexp: float


@dataclass(frozen=True)
class ActionThreatValue:
    action_id: str
    focal_player_id: str
    fixed_defense: ThreatAggregate
    focal_location_value: float
    focal_offside: bool
    highest_teammate_id: str | None
    highest_teammate_location_value: float
    boundary_clipped_player_ids: tuple[str, ...]
    surface: ReferenceOBSOSurface

    def as_record(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "focal_player_id": self.focal_player_id,
            "fixed_defense_maximum_obso": self.fixed_defense.maximum,
            "maximum_x": self.fixed_defense.maximum_x,
            "maximum_y": self.fixed_defense.maximum_y,
            "top_k_peak_mean": self.fixed_defense.top_k_peak_mean,
            "peak_values": "|".join(
                f"{value:.12g}" for value in self.fixed_defense.peak_values
            ),
            "normalized_logsumexp": self.fixed_defense.normalized_logsumexp,
            "focal_location_value": self.focal_location_value,
            "focal_offside": self.focal_offside,
            "highest_teammate_id": self.highest_teammate_id,
            "highest_teammate_location_value": (
                self.highest_teammate_location_value
            ),
            "boundary_clipped_player_ids": "|".join(
                self.boundary_clipped_player_ids
            ),
        }


@dataclass(frozen=True)
class NearestAttackerThreatDecomposition:
    """Explain a team surface without changing the OBSO evaluator.

    Every OBSO grid cell is attributed to its nearest eligible attacker.  The
    focal and non-focal maxima therefore form an exact partition of the
    original team-wide maximum.  This is an explanation layer, not a new
    threat model or optimization objective.
    """

    team_maximum: float
    focal_maximum: float
    focal_maximum_x: float | None
    focal_maximum_y: float | None
    focal_pitch_control: float
    focal_transition: float
    focal_score: float
    focal_owner_distance_m: float | None
    other_maximum: float
    other_maximum_x: float | None
    other_maximum_y: float | None
    other_player_id: str | None
    other_pitch_control: float
    other_transition: float
    other_score: float
    other_owner_distance_m: float | None
    focal_offside: bool

    @property
    def reconstructed_team_maximum(self) -> float:
        return max(self.focal_maximum, self.other_maximum)


@dataclass(frozen=True)
class NearestAttackerRegionPartition:
    """Fixed grid ownership used to compare defensive responses.

    Ownership is recomputed for every terminal counterfactual state, so an
    action-induced change to the offside line is respected.  The masks are an
    explanation partition only; they never enter the team-wide OBSO core.
    """

    eligible_player_ids: tuple[str, ...]
    owners: np.ndarray
    focal_mask: np.ndarray
    other_mask: np.ndarray
    focal_offside: bool


def nearest_attacker_region_partition(
    frame: BundesligaFrame,
    attacking_team_id: str,
    focal_player_id: str,
    attacking_direction: int,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    ball_xy: tuple[float, float] | None = None,
    offside_tolerance_m: float = 0.2,
) -> NearestAttackerRegionPartition:
    """Assign every OBSO cell to its nearest eligible attacker."""

    if focal_player_id not in frame.players:
        raise KeyError(f"Focal attacker {focal_player_id} is missing")
    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball")
        ball_xy = (frame.ball.x, frame.ball.y)
    offside = offside_attacker_ids(
        frame,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        offside_tolerance_m,
    )
    eligible_ids = tuple(
        sorted(
            player_id
            for player_id, player in frame.players.items()
            if player.team_id == attacking_team_id and player_id not in offside
        )
    )
    shape = (len(ygrid), len(xgrid))
    if not eligible_ids:
        owners = np.full(shape, -1, dtype=int)
        empty = np.zeros(shape, dtype=bool)
        return NearestAttackerRegionPartition(
            eligible_player_ids=(),
            owners=owners,
            focal_mask=empty.copy(),
            other_mask=empty.copy(),
            focal_offside=focal_player_id in offside,
        )

    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    attacker_positions = np.asarray(
        [
            (frame.players[player_id].x, frame.players[player_id].y)
            for player_id in eligible_ids
        ],
        dtype=float,
    )
    distances = np.linalg.norm(
        grid_points[:, None, :] - attacker_positions[None, :, :],
        axis=2,
    )
    owners = np.argmin(distances, axis=1).reshape(shape)
    if focal_player_id in eligible_ids:
        focal_index = eligible_ids.index(focal_player_id)
        focal_mask = owners == focal_index
    else:
        focal_mask = np.zeros(shape, dtype=bool)
    return NearestAttackerRegionPartition(
        eligible_player_ids=eligible_ids,
        owners=owners,
        focal_mask=focal_mask,
        other_mask=~focal_mask,
        focal_offside=focal_player_id in offside,
    )


def spatial_peak_values(
    values: np.ndarray,
    count: int,
    suppression_radius_cells: float,
) -> tuple[float, ...]:
    """Select high cells with non-maximum suppression in grid coordinates."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional surface")
    finite = np.isfinite(array)
    if not np.any(finite):
        return ()
    candidates = np.argwhere(finite)
    order = np.argsort(array[finite])[::-1]
    selected_coordinates: list[np.ndarray] = []
    selected_values: list[float] = []
    for ordered_index in order:
        coordinate = candidates[int(ordered_index)]
        if any(
            float(np.linalg.norm(coordinate - selected))
            <= suppression_radius_cells
            for selected in selected_coordinates
        ):
            continue
        selected_coordinates.append(coordinate)
        selected_values.append(float(array[tuple(coordinate)]))
        if len(selected_values) >= count:
            break
    return tuple(selected_values)


def normalized_logsumexp(values: np.ndarray, temperature: float) -> float:
    """Smooth maximum normalized by cell count, preserving surface scale."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    maximum = float(np.max(finite))
    scaled = np.exp((finite - maximum) / temperature)
    return float(maximum + temperature * math.log(float(np.mean(scaled))))


def aggregate_threat_surface(
    surface: ReferenceOBSOSurface,
    config: ThreatAggregationConfig = ThreatAggregationConfig(),
) -> ThreatAggregate:
    config.validate()
    peaks = spatial_peak_values(
        surface.obso,
        config.top_k_peaks,
        config.peak_suppression_radius_cells,
    )
    maximum_x, maximum_y = surface.maximum_position
    return ThreatAggregate(
        maximum=surface.maximum,
        maximum_x=maximum_x,
        maximum_y=maximum_y,
        peak_values=peaks,
        top_k_peak_mean=float(np.mean(peaks)) if peaks else float("nan"),
        normalized_logsumexp=normalized_logsumexp(
            surface.obso,
            config.logsumexp_temperature,
        ),
    )


def surface_value_at_position(
    surface: ReferenceOBSOSurface,
    position_xy: tuple[float, float],
) -> float:
    column = int(np.argmin(np.abs(surface.xgrid - position_xy[0])))
    row = int(np.argmin(np.abs(surface.ygrid - position_xy[1])))
    return float(surface.obso[row, column])


def decompose_threat_by_nearest_attacker(
    surface: ReferenceOBSOSurface,
    frame: BundesligaFrame,
    attacking_team_id: str,
    focal_player_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float] | None = None,
    offside_tolerance_m: float = 0.2,
) -> NearestAttackerThreatDecomposition:
    """Partition the fixed OBSO surface into focal and other team options.

    The allocation is deterministic: attacker ids are sorted before nearest-
    neighbour assignment, so an exact distance tie has a stable owner.  The
    returned maxima always reconstruct the unmodified team-wide maximum.
    """

    partition = nearest_attacker_region_partition(
        frame,
        attacking_team_id,
        focal_player_id,
        attacking_direction,
        surface.xgrid,
        surface.ygrid,
        ball_xy,
        offside_tolerance_m,
    )
    eligible_ids = partition.eligible_player_ids
    if not eligible_ids:
        return NearestAttackerThreatDecomposition(
            team_maximum=surface.maximum,
            focal_maximum=0.0,
            focal_maximum_x=None,
            focal_maximum_y=None,
            focal_pitch_control=0.0,
            focal_transition=0.0,
            focal_score=0.0,
            focal_owner_distance_m=None,
            other_maximum=0.0,
            other_maximum_x=None,
            other_maximum_y=None,
            other_player_id=None,
            other_pitch_control=0.0,
            other_transition=0.0,
            other_score=0.0,
            other_owner_distance_m=None,
            focal_offside=partition.focal_offside,
        )
    owners = partition.owners

    def attributed_maximum(
        owner_indices: set[int],
    ) -> tuple[float, float | None, float | None, int | None]:
        if not owner_indices:
            return 0.0, None, None, None
        mask = np.isin(owners, tuple(owner_indices))
        if not np.any(mask):
            return 0.0, None, None, None
        masked = np.where(mask, surface.obso, -np.inf)
        row, col = np.unravel_index(int(np.nanargmax(masked)), masked.shape)
        owner_index = int(owners[row, col])
        return (
            float(masked[row, col]),
            float(surface.xgrid[col]),
            float(surface.ygrid[row]),
            owner_index,
        )

    focal_indices = {eligible_ids.index(focal_player_id)} if np.any(partition.focal_mask) else set()
    other_indices = set(range(len(eligible_ids))) - focal_indices
    focal_value, focal_x, focal_y, _ = attributed_maximum(focal_indices)
    other_value, other_x, other_y, other_owner = attributed_maximum(other_indices)

    def components_at(
        x: float | None,
        y: float | None,
        owner_id: str | None,
    ) -> tuple[float, float, float, float | None]:
        if x is None or y is None:
            return 0.0, 0.0, 0.0, None
        column = int(np.argmin(np.abs(surface.xgrid - x)))
        row = int(np.argmin(np.abs(surface.ygrid - y)))
        owner_distance = (
            math.hypot(frame.players[owner_id].x - x, frame.players[owner_id].y - y)
            if owner_id is not None
            else None
        )
        return (
            float(surface.pitch_control[row, column]),
            float(surface.transition[row, column]),
            float(surface.score[row, column]),
            owner_distance,
        )

    focal_components = components_at(
        focal_x,
        focal_y,
        focal_player_id if focal_indices else None,
    )
    other_player_id = eligible_ids[other_owner] if other_owner is not None else None
    other_components = components_at(other_x, other_y, other_player_id)
    result = NearestAttackerThreatDecomposition(
        team_maximum=surface.maximum,
        focal_maximum=focal_value,
        focal_maximum_x=focal_x,
        focal_maximum_y=focal_y,
        focal_pitch_control=focal_components[0],
        focal_transition=focal_components[1],
        focal_score=focal_components[2],
        focal_owner_distance_m=focal_components[3],
        other_maximum=other_value,
        other_maximum_x=other_x,
        other_maximum_y=other_y,
        other_player_id=other_player_id,
        other_pitch_control=other_components[0],
        other_transition=other_components[1],
        other_score=other_components[2],
        other_owner_distance_m=other_components[3],
        focal_offside=partition.focal_offside,
    )
    if not math.isclose(
        result.reconstructed_team_maximum,
        result.team_maximum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Nearest-attacker attribution did not preserve team maximum")
    return result


def evaluate_counterfactual_threat(
    state: CounterfactualState,
    attacking_team_id: str,
    attacking_direction: int,
    metadata: BundesligaMatchMeta,
    aggregation_config: ThreatAggregationConfig = ThreatAggregationConfig(),
    obso_config: ReferenceOBSOConfig = ReferenceOBSOConfig(),
) -> ActionThreatValue:
    """Evaluate one propagated attacker action before optimizing a defender."""

    goalkeeper_ids: Sequence[str] = tuple(
        goalkeeper_id
        for team_id in metadata.teams
        if (goalkeeper_id := metadata.goalkeeper_id(team_id)) is not None
    )
    surface = evaluate_reference_obso(
        frame=state.frame,
        attacking_team_id=attacking_team_id,
        attacking_direction=attacking_direction,
        velocities=state.velocities,
        goalkeeper_ids=goalkeeper_ids,
        apply_offside=True,
        config=obso_config,
    )
    ball_xy = (
        (state.frame.ball.x, state.frame.ball.y)
        if state.frame.ball is not None
        else (0.0, 0.0)
    )
    offside = offside_attacker_ids(
        state.frame,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        obso_config.offside_tolerance_m,
    )
    focal = state.frame.players[state.focal_player_id]
    focal_value = (
        0.0
        if state.focal_player_id in offside
        else surface_value_at_position(surface, (focal.x, focal.y))
    )
    teammate_values = []
    for player_id, player in state.frame.players.items():
        if (
            player.team_id != attacking_team_id
            or player_id == state.focal_player_id
            or player_id in offside
        ):
            continue
        teammate_values.append(
            (
                surface_value_at_position(surface, (player.x, player.y)),
                player_id,
            )
        )
    if teammate_values:
        highest_value, highest_id = max(teammate_values)
    else:
        highest_value, highest_id = 0.0, None
    return ActionThreatValue(
        action_id=state.action_id,
        focal_player_id=state.focal_player_id,
        fixed_defense=aggregate_threat_surface(surface, aggregation_config),
        focal_location_value=float(focal_value),
        focal_offside=state.focal_player_id in offside,
        highest_teammate_id=highest_id,
        highest_teammate_location_value=float(highest_value),
        boundary_clipped_player_ids=state.boundary_clipped_player_ids,
        surface=surface,
    )
