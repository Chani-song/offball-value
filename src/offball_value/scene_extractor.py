"""Controlled-possession scene extraction for Bundesliga IDSSE tracking.

The extractor only defines the analysis cohort.  It does not use future match
outcomes, calculate OBSO, or rank off-ball actions.  Every failed criterion is
retained as an auditable rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Iterable, Mapping
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
    infer_attacking_direction,
    load_bundesliga_frames,
    parse_bundesliga_datetime,
)
from .reference_obso import offside_attacker_ids


SET_PIECE_EVENT_TYPES = frozenset({"pass", "cross"})
UNSTABLE_EVENT_TYPES = frozenset(
    {
        "tacklinggame",
        "ballclaiming",
        "otherballaction",
        "balldeflection",
        "foul",
        "offside",
        "refereeball",
        "penalty",
    }
)


@dataclass(frozen=True)
class SceneExtractorConfig:
    stable_control_seconds: float = 0.5
    control_distance_m: float = 1.5
    minimum_stable_fraction: float = 0.80
    sample_interval_seconds: float = 2.0
    horizon_seconds: float = 2.0
    set_piece_pre_seconds: float = 0.2
    set_piece_post_seconds: float = 3.0
    unstable_event_radius_seconds: float = 0.5
    required_player_count: int | None = 22
    require_eligible_attacker: bool = True

    @property
    def history_intervals(self) -> int:
        return int(math.ceil(self.stable_control_seconds * FPS))

    @property
    def sampling_frames(self) -> int:
        return max(1, int(round(self.sample_interval_seconds * FPS)))

    @property
    def horizon_frames(self) -> int:
        return max(1, int(round(self.horizon_seconds * FPS)))

    def validate(self) -> None:
        if self.stable_control_seconds <= 0:
            raise ValueError("stable_control_seconds must be positive")
        if self.control_distance_m <= 0:
            raise ValueError("control_distance_m must be positive")
        if not 0 < self.minimum_stable_fraction <= 1:
            raise ValueError("minimum_stable_fraction must lie in (0, 1]")
        if self.sample_interval_seconds <= 0 or self.horizon_seconds <= 0:
            raise ValueError("sample interval and horizon must be positive")


@dataclass(frozen=True)
class BallFrameIndexEntry:
    frame_id: int
    period: int
    game_section: str
    timestamp: datetime | None
    ball: BundesligaObjectState


@dataclass(frozen=True)
class SceneCandidate:
    match_id: str
    frame_id: int
    period: int
    game_section: str
    timestamp: datetime | None
    horizon_frame_id: int
    possession_team_id: str | None
    ball_carrier_id: str | None
    ball_carrier_distance_m: float | None
    stable_control_fraction: float
    stable_control_seconds_observed: float
    history_frames_available: int
    history_frames_required: int
    player_count: int
    ball_x: float | None
    ball_y: float | None
    ball_z: float | None
    attacking_direction: int | None
    eligible_attacker_ids: tuple[str, ...]
    offside_attacker_ids: tuple[str, ...]
    nearest_event_type: str | None
    nearest_event_frame_distance: int | None
    is_open_play: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "rejected"

    def as_record(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "frame_id": self.frame_id,
            "period": self.period,
            "game_section": self.game_section,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "horizon_frame_id": self.horizon_frame_id,
            "possession_team_id": self.possession_team_id,
            "ball_carrier_id": self.ball_carrier_id,
            "ball_carrier_distance_m": self.ball_carrier_distance_m,
            "stable_control_fraction": self.stable_control_fraction,
            "stable_control_seconds_observed": self.stable_control_seconds_observed,
            "history_frames_available": self.history_frames_available,
            "history_frames_required": self.history_frames_required,
            "player_count": self.player_count,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_z": self.ball_z,
            "attacking_direction": self.attacking_direction,
            "eligible_attacker_count": len(self.eligible_attacker_ids),
            "eligible_attacker_ids": "|".join(self.eligible_attacker_ids),
            "offside_attacker_count": len(self.offside_attacker_ids),
            "offside_attacker_ids": "|".join(self.offside_attacker_ids),
            "nearest_event_type": self.nearest_event_type,
            "nearest_event_frame_distance": self.nearest_event_frame_distance,
            "is_open_play": self.is_open_play,
            "accepted": self.accepted,
            "rejection_reasons": "|".join(self.rejection_reasons),
        }


@dataclass(frozen=True)
class SceneExtractionResult:
    candidates: tuple[SceneCandidate, ...]
    frames: Mapping[int, BundesligaFrame]
    history_frame_ids: Mapping[int, tuple[int, ...]]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(candidate.as_record() for candidate in self.candidates)


def load_bundesliga_ball_index(
    positions_xml: str | Path,
) -> dict[int, BallFrameIndexEntry]:
    """Load the lightweight ball trajectory used to build a 2 s frame cadence."""

    entries: dict[int, BallFrameIndexEntry] = {}
    current_frameset: dict[str, str] | None = None
    for event, elem in ET.iterparse(positions_xml, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_frameset = dict(elem.attrib)
            continue

        if event == "end" and elem.tag == "Frame" and current_frameset:
            if current_frameset.get("TeamId") == "BALL":
                frame_id = int(elem.attrib["N"])
                section = current_frameset.get("GameSection", "")
                entries[frame_id] = BallFrameIndexEntry(
                    frame_id=frame_id,
                    period=2 if section == "secondHalf" else 1,
                    game_section=section,
                    timestamp=parse_bundesliga_datetime(elem.attrib.get("T")),
                    ball=BundesligaObjectState(
                        object_id=current_frameset.get("PersonId", "BALL"),
                        team_id="BALL",
                        x=float(elem.attrib["X"]),
                        y=float(elem.attrib["Y"]),
                        z=_optional_float(elem.attrib.get("Z")),
                        speed=_optional_float(elem.attrib.get("S")),
                    ),
                )
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            current_frameset = None
            elem.clear()
        elif event == "end":
            elem.clear()
    return entries


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def candidate_frame_ids(
    ball_index: Mapping[int, BallFrameIndexEntry],
    config: SceneExtractorConfig = SceneExtractorConfig(),
) -> list[int]:
    """Build a deterministic period-anchored, non-overlapping frame cadence."""

    config.validate()
    by_section: dict[str, list[int]] = {}
    for frame_id, entry in ball_index.items():
        by_section.setdefault(entry.game_section, []).append(frame_id)

    candidates: list[int] = []
    for section in ("firstHalf", "secondHalf"):
        ids = sorted(by_section.get(section, []))
        if not ids:
            continue
        available = set(ids)
        first = ids[0] + config.history_intervals
        last = ids[-1] - config.horizon_frames
        for frame_id in range(first, last + 1, config.sampling_frames):
            history = range(frame_id - config.history_intervals, frame_id + 1)
            if all(history_id in available for history_id in history):
                candidates.append(frame_id)
    return candidates


def required_tracking_frame_ids(
    candidates: Iterable[int],
    config: SceneExtractorConfig = SceneExtractorConfig(),
) -> tuple[set[int], dict[int, tuple[int, ...]]]:
    target_ids: set[int] = set()
    histories: dict[int, tuple[int, ...]] = {}
    for frame_id in candidates:
        history = tuple(range(frame_id - config.history_intervals, frame_id + 1))
        histories[frame_id] = history
        target_ids.update(history)
    return target_ids, histories


def _nearest_player(
    frame: BundesligaFrame,
) -> tuple[str | None, float | None]:
    if frame.ball is None or not frame.players:
        return None, None
    player_id, distance = min(
        (
            (
                player_id,
                math.hypot(state.x - frame.ball.x, state.y - frame.ball.y),
            )
            for player_id, state in frame.players.items()
        ),
        key=lambda item: item[1],
    )
    return player_id, float(distance)


def _event_context(
    frame_id: int,
    events: pd.DataFrame,
    config: SceneExtractorConfig,
) -> tuple[str | None, int | None, bool, list[str]]:
    usable = events[events["frame_id"].notna()]
    if usable.empty:
        return None, None, True, []
    event_frames = usable["frame_id"].astype(int).to_numpy()
    nearest_position = int(np.argmin(np.abs(event_frames - frame_id)))
    nearest = usable.iloc[nearest_position]
    nearest_distance = int(abs(int(nearest["frame_id"]) - frame_id))
    reasons: list[str] = []

    set_piece_pre = int(round(config.set_piece_pre_seconds * FPS))
    set_piece_post = int(round(config.set_piece_post_seconds * FPS))
    false_open_play = usable[
        usable["event_type"].isin(SET_PIECE_EVENT_TYPES)
        & (usable["from_open_play"] == False)  # noqa: E712 - pandas scalar comparison
    ]
    for event_frame in false_open_play["frame_id"].astype(int):
        if event_frame - set_piece_pre <= frame_id <= event_frame + set_piece_post:
            reasons.append("set_piece_window")
            break

    unstable_radius = int(round(config.unstable_event_radius_seconds * FPS))
    unstable = usable[usable["event_type"].isin(UNSTABLE_EVENT_TYPES)]
    if np.any(np.abs(unstable["frame_id"].astype(int).to_numpy() - frame_id) <= unstable_radius):
        reasons.append("unstable_event_window")

    return (
        str(nearest["event_type"]) if pd.notna(nearest["event_type"]) else None,
        nearest_distance,
        not reasons,
        reasons,
    )


def evaluate_scene_candidate(
    frame_id: int,
    frames: Mapping[int, BundesligaFrame],
    history_frame_ids: Iterable[int],
    events: pd.DataFrame,
    metadata: BundesligaMatchMeta,
    config: SceneExtractorConfig = SceneExtractorConfig(),
) -> SceneCandidate:
    """Evaluate one timestamp using current and past information only."""

    config.validate()
    frame = frames.get(frame_id)
    history_ids = tuple(history_frame_ids)
    reasons: list[str] = []
    if frame is None:
        raise ValueError(f"Candidate frame {frame_id} was not loaded")

    carrier_id, carrier_distance = _nearest_player(frame)
    carrier_state = frame.players.get(carrier_id) if carrier_id else None
    possession_team_id = carrier_state.team_id if carrier_state else None
    if frame.ball is None:
        reasons.append("missing_ball")
    if carrier_id is None or carrier_distance is None:
        reasons.append("missing_ball_carrier")
    elif carrier_distance > config.control_distance_m:
        reasons.append("ball_carrier_too_far")

    if (
        config.required_player_count is not None
        and len(frame.players) != config.required_player_count
    ):
        reasons.append("incomplete_player_frame")

    controlled_samples = 0
    available_samples = 0
    for history_id in history_ids:
        history_frame = frames.get(history_id)
        if history_frame is None or history_frame.ball is None or carrier_id is None:
            continue
        player = history_frame.players.get(carrier_id)
        if player is None:
            continue
        available_samples += 1
        distance = math.hypot(
            player.x - history_frame.ball.x,
            player.y - history_frame.ball.y,
        )
        if distance <= config.control_distance_m:
            controlled_samples += 1

    required_samples = len(history_ids)
    stable_fraction = (
        controlled_samples / required_samples if required_samples else 0.0
    )
    stable_seconds = (
        max(0, controlled_samples - 1) / FPS if controlled_samples else 0.0
    )
    if available_samples < required_samples:
        reasons.append("incomplete_control_history")
    if stable_fraction < config.minimum_stable_fraction:
        reasons.append("unstable_control_history")

    nearest_event, event_distance, is_open_play, event_reasons = _event_context(
        frame_id,
        events,
        config,
    )
    reasons.extend(event_reasons)

    attacking_direction: int | None = None
    eligible: tuple[str, ...] = ()
    offside: tuple[str, ...] = ()
    if possession_team_id and frame.ball is not None:
        attacking_direction = infer_attacking_direction(
            frame,
            possession_team_id,
            metadata,
        )
        offside_set = offside_attacker_ids(
            frame,
            possession_team_id,
            attacking_direction,
            (frame.ball.x, frame.ball.y),
        )
        offside = tuple(sorted(offside_set))
        goalkeeper_id = metadata.goalkeeper_id(possession_team_id)
        eligible = tuple(
            sorted(
                player_id
                for player_id, state in frame.players.items()
                if state.team_id == possession_team_id
                and player_id != carrier_id
                and player_id != goalkeeper_id
                and player_id not in offside_set
            )
        )
    if config.require_eligible_attacker and not eligible:
        reasons.append("no_eligible_attacker")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return SceneCandidate(
        match_id=frame.match_id,
        frame_id=frame.frame_id,
        period=frame.period,
        game_section=frame.game_section,
        timestamp=frame.timestamp,
        horizon_frame_id=frame.frame_id + config.horizon_frames,
        possession_team_id=possession_team_id,
        ball_carrier_id=carrier_id,
        ball_carrier_distance_m=carrier_distance,
        stable_control_fraction=float(stable_fraction),
        stable_control_seconds_observed=float(stable_seconds),
        history_frames_available=available_samples,
        history_frames_required=required_samples,
        player_count=len(frame.players),
        ball_x=frame.ball.x if frame.ball else None,
        ball_y=frame.ball.y if frame.ball else None,
        ball_z=frame.ball.z if frame.ball else None,
        attacking_direction=attacking_direction,
        eligible_attacker_ids=eligible,
        offside_attacker_ids=offside,
        nearest_event_type=nearest_event,
        nearest_event_frame_distance=event_distance,
        is_open_play=is_open_play,
        accepted=not unique_reasons,
        rejection_reasons=unique_reasons,
    )


def extract_bundesliga_scenes(
    positions_xml: str | Path,
    events: pd.DataFrame,
    metadata: BundesligaMatchMeta,
    config: SceneExtractorConfig = SceneExtractorConfig(),
    limit_candidates: int | None = None,
) -> SceneExtractionResult:
    """Extract auditable 2 s controlled-possession candidates for one match."""

    ball_index = load_bundesliga_ball_index(positions_xml)
    candidates = candidate_frame_ids(ball_index, config)
    if limit_candidates is not None:
        candidates = candidates[: max(0, limit_candidates)]
    target_ids, histories = required_tracking_frame_ids(candidates, config)
    frames = load_bundesliga_frames(positions_xml, target_ids)
    evaluated = tuple(
        evaluate_scene_candidate(
            frame_id,
            frames,
            histories[frame_id],
            events,
            metadata,
            config,
        )
        for frame_id in candidates
        if frame_id in frames
    )
    return SceneExtractionResult(
        candidates=evaluated,
        frames=frames,
        history_frame_ids=histories,
    )


def pitch_dimensions() -> tuple[float, float]:
    """Expose the coordinate convention for audit renderers."""

    return FIELD_LENGTH, FIELD_WIDTH

