"""Stable-possession gating for run onsets inside shot-anchored phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .bundesliga import FPS, BundesligaFrame
from .run_onset import RunOnsetCandidate
from .shot_context import ShotContextConfig, controlled_team_at_frame


HARD_UNSTABLE_EVENT_TYPES = frozenset(
    {
        "tacklinggame",
        "ballclaiming",
        "balldeflection",
        "foul",
        "offside",
        "refereeball",
        "penalty",
    }
)
RESTART_EVENT_TYPES = frozenset({"pass", "cross"})


@dataclass(frozen=True)
class SettledPossessionConfig:
    history_seconds: float = 1.0
    control_distance_m: float = 1.5
    max_control_ball_height_m: float = 1.5
    minimum_team_control_fraction: float = 0.40
    minimum_known_control_fraction: float = 0.40
    minimum_same_team_known_fraction: float = 0.90
    unstable_event_radius_seconds: float = 0.75
    restart_post_seconds: float = 3.0
    minimum_seconds_before_shot: float = 2.0
    require_ball_in_opponent_half: bool = True

    @property
    def history_frames(self) -> int:
        return max(1, int(round(self.history_seconds * FPS)))

    def validate(self) -> None:
        positive = {
            "history_seconds": self.history_seconds,
            "control_distance_m": self.control_distance_m,
            "max_control_ball_height_m": self.max_control_ball_height_m,
            "unstable_event_radius_seconds": self.unstable_event_radius_seconds,
            "restart_post_seconds": self.restart_post_seconds,
            "minimum_seconds_before_shot": self.minimum_seconds_before_shot,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"settled-possession parameters must be positive: {invalid}")
        for name, value in (
            ("minimum_team_control_fraction", self.minimum_team_control_fraction),
            ("minimum_known_control_fraction", self.minimum_known_control_fraction),
            ("minimum_same_team_known_fraction", self.minimum_same_team_known_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class SettledPossessionAssessment:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    history_frames_required: int
    history_frames_available: int
    known_control_fraction: float
    team_control_fraction: float
    same_team_known_fraction: float
    opponent_control_frames: int
    nearest_unstable_event_type: str | None
    nearest_unstable_event_distance_frames: int | None
    seconds_before_shot: float
    ball_progress_m: float | None

    def as_record(self) -> dict[str, object]:
        return {
            "settled_possession": self.accepted,
            "settled_rejection_reasons": "|".join(self.rejection_reasons),
            "settled_history_frames_required": self.history_frames_required,
            "settled_history_frames_available": self.history_frames_available,
            "settled_known_control_fraction": self.known_control_fraction,
            "settled_team_control_fraction": self.team_control_fraction,
            "settled_same_team_known_fraction": self.same_team_known_fraction,
            "settled_opponent_control_frames": self.opponent_control_frames,
            "nearest_unstable_event_type": self.nearest_unstable_event_type,
            "nearest_unstable_event_distance_frames": self.nearest_unstable_event_distance_frames,
            "seconds_before_shot": self.seconds_before_shot,
            "ball_progress_m": self.ball_progress_m,
        }


def _event_rejections(
    onset_frame_id: int,
    events: pd.DataFrame,
    config: SettledPossessionConfig,
) -> tuple[list[str], str | None, int | None]:
    if events.empty or "frame_id" not in events:
        return [], None, None
    usable = events[events["frame_id"].notna()].copy()
    if usable.empty:
        return [], None, None
    usable["frame_id"] = usable["frame_id"].astype(int)
    usable["distance"] = (usable["frame_id"] - onset_frame_id).abs()
    nearest = usable.sort_values(["distance", "frame_id"]).iloc[0]
    nearest_type = str(nearest["event_type"]) if pd.notna(nearest.get("event_type")) else None
    nearest_distance = int(nearest["distance"])
    reasons: list[str] = []

    unstable_radius = int(round(config.unstable_event_radius_seconds * FPS))
    unstable = usable[
        usable["event_type"].astype(str).str.lower().isin(HARD_UNSTABLE_EVENT_TYPES)
        & (usable["distance"] <= unstable_radius)
    ]
    if not unstable.empty:
        reasons.append("unstable_event_window")

    restart_post = int(round(config.restart_post_seconds * FPS))
    if "from_open_play" in usable:
        restarts = usable[
            usable["event_type"].astype(str).str.lower().isin(RESTART_EVENT_TYPES)
            & (usable["from_open_play"] == False)  # noqa: E712
            & (usable["frame_id"] <= onset_frame_id)
            & (usable["frame_id"] >= onset_frame_id - restart_post)
        ]
        if not restarts.empty:
            reasons.append("recent_restart")
    return reasons, nearest_type, nearest_distance


def assess_settled_shot_context_onset(
    candidate: RunOnsetCandidate,
    *,
    frames: Mapping[int, BundesligaFrame],
    events: pd.DataFrame,
    phase_start_frame_id: int,
    shot_frame_id: int,
    attacking_direction: int,
    config: SettledPossessionConfig = SettledPossessionConfig(),
) -> SettledPossessionAssessment:
    """Assess whether a contextual run onset belongs to settled attacking play."""

    config.validate()
    reasons: list[str] = []
    if not phase_start_frame_id <= candidate.frame_id < shot_frame_id:
        reasons.append("outside_shot_attacking_phase")

    seconds_before_shot = (shot_frame_id - candidate.frame_id) / FPS
    if seconds_before_shot < config.minimum_seconds_before_shot:
        reasons.append("insufficient_pre_shot_horizon")

    frame = frames.get(candidate.frame_id)
    ball_progress: float | None = None
    if frame is None or frame.ball is None:
        reasons.append("missing_onset_ball")
    else:
        ball_progress = attacking_direction * float(frame.ball.x)
        if config.require_ball_in_opponent_half and ball_progress <= 0.0:
            reasons.append("ball_not_in_opponent_half")

    history_ids = range(candidate.frame_id - config.history_frames, candidate.frame_id + 1)
    history = [frames[frame_id] for frame_id in history_ids if frame_id in frames]
    required = config.history_frames + 1
    control_config = ShotContextConfig(
        control_distance_m=config.control_distance_m,
        max_control_ball_height_m=config.max_control_ball_height_m,
    )
    owners = [controlled_team_at_frame(sample, control_config) for sample in history]
    known = [owner for owner in owners if owner is not None]
    same_team = sum(owner == candidate.team_id for owner in owners)
    opponent = sum(owner is not None and owner != candidate.team_id for owner in owners)
    known_fraction = len(known) / required
    team_fraction = same_team / required
    same_team_known = same_team / len(known) if known else 0.0

    if len(history) < required:
        reasons.append("incomplete_settled_history")
    if known_fraction < config.minimum_known_control_fraction:
        reasons.append("insufficient_known_control_history")
    if team_fraction < config.minimum_team_control_fraction:
        reasons.append("insufficient_team_control_history")
    if same_team_known < config.minimum_same_team_known_fraction:
        reasons.append("mixed_recent_control")
    if opponent > 0:
        reasons.append("recent_opponent_control")

    event_reasons, nearest_type, nearest_distance = _event_rejections(
        candidate.frame_id, events, config
    )
    reasons.extend(event_reasons)
    unique_reasons = tuple(dict.fromkeys(reasons))
    return SettledPossessionAssessment(
        accepted=not unique_reasons,
        rejection_reasons=unique_reasons,
        history_frames_required=required,
        history_frames_available=len(history),
        known_control_fraction=float(known_fraction),
        team_control_fraction=float(team_fraction),
        same_team_known_fraction=float(same_team_known),
        opponent_control_frames=opponent,
        nearest_unstable_event_type=nearest_type,
        nearest_unstable_event_distance_frames=nearest_distance,
        seconds_before_shot=float(seconds_before_shot),
        ball_progress_m=ball_progress,
    )
