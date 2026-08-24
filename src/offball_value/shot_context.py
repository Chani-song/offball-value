"""Shot-anchored sampling of attacking contexts.

Shots are used only to retrieve attack-like match segments.  Nothing in this
module attributes the later shot to an earlier off-ball action, and shot
outcomes are not exposed as optimizer features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import pandas as pd

from .bundesliga import (
    FPS,
    BundesligaFrame,
    BundesligaMatchMeta,
    infer_attacking_direction,
)


@dataclass(frozen=True)
class ShotContextConfig:
    max_lookback_seconds: float = 60.0
    comparison_windows_seconds: tuple[float, ...] = (10.0, 15.0, 30.0, 60.0)
    control_distance_m: float = 1.5
    max_control_ball_height_m: float = 1.5
    opponent_control_confirmation_seconds: float = 0.32

    @property
    def max_lookback_frames(self) -> int:
        return int(round(self.max_lookback_seconds * FPS))

    @property
    def opponent_confirmation_samples(self) -> int:
        return max(1, int(round(self.opponent_control_confirmation_seconds * FPS)))

    def validate(self) -> None:
        if self.max_lookback_seconds <= 0.0:
            raise ValueError("max_lookback_seconds must be positive")
        if not self.comparison_windows_seconds:
            raise ValueError("comparison_windows_seconds must not be empty")
        if any(value <= 0.0 for value in self.comparison_windows_seconds):
            raise ValueError("comparison windows must be positive")
        if self.control_distance_m <= 0.0:
            raise ValueError("control_distance_m must be positive")
        if self.max_control_ball_height_m <= 0.0:
            raise ValueError("max_control_ball_height_m must be positive")
        if self.opponent_control_confirmation_seconds <= 0.0:
            raise ValueError("opponent_control_confirmation_seconds must be positive")


@dataclass(frozen=True)
class ShotAnchor:
    match_id: str
    event_id: str
    frame_id: int
    period: int
    team_id: str
    player_id: str | None
    xg: float | None
    x: float | None
    y: float | None
    from_open_play: bool
    build_up: str | None
    setup: str | None

    def as_record(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "period": self.period,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "xg": self.xg,
            "x": self.x,
            "y": self.y,
            "from_open_play": self.from_open_play,
            "build_up": self.build_up,
            "setup": self.setup,
        }


@dataclass(frozen=True)
class ShotAttackingPhase:
    shot: ShotAnchor
    available_start_frame_id: int
    possession_start_frame_id: int
    attacking_half_entry_frame_id: int
    shot_frame_id: int
    attacking_direction: int
    boundary_reason: str
    duration_seconds: float
    attacking_half_duration_seconds: float
    attacking_half_fraction: float
    attacking_team_control_fraction: float
    known_control_fraction: float
    window_start_frame_ids: tuple[tuple[float, int], ...]

    def as_record(self) -> dict[str, object]:
        record = self.shot.as_record()
        record.update(
            {
                "available_start_frame_id": self.available_start_frame_id,
                "possession_start_frame_id": self.possession_start_frame_id,
                "attacking_half_entry_frame_id": self.attacking_half_entry_frame_id,
                "shot_frame_id": self.shot_frame_id,
                "attacking_direction": self.attacking_direction,
                "boundary_reason": self.boundary_reason,
                "duration_seconds": self.duration_seconds,
                "attacking_half_duration_seconds": self.attacking_half_duration_seconds,
                "attacking_half_fraction": self.attacking_half_fraction,
                "attacking_team_control_fraction": self.attacking_team_control_fraction,
                "known_control_fraction": self.known_control_fraction,
            }
        )
        for seconds, frame_id in self.window_start_frame_ids:
            label = str(int(seconds)) if float(seconds).is_integer() else str(seconds).replace(".", "_")
            record[f"window_{label}s_start_frame_id"] = frame_id
        return record


def extract_shot_anchors(
    events: pd.DataFrame,
    *,
    open_play_only: bool = True,
) -> tuple[ShotAnchor, ...]:
    """Return deterministic shot anchors with provider actor metadata restored."""

    required = {"event_type", "frame_id", "team_id"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events are missing required columns: {sorted(missing)}")
    shots = events[events["event_type"].astype(str).str.lower() == "shotatgoal"].copy()
    shots = shots[shots["frame_id"].notna() & shots["team_id"].notna()]
    if open_play_only:
        shots = shots[shots["from_open_play"] == True]  # noqa: E712

    anchors: list[ShotAnchor] = []
    for _, row in shots.sort_values(["period", "frame_id", "event_id"]).iterrows():
        anchors.append(
            ShotAnchor(
                match_id=str(row.get("match_id") or ""),
                event_id=str(row.get("event_id") or ""),
                frame_id=int(row["frame_id"]),
                period=int(row.get("period") or (2 if int(row["frame_id"]) >= 100000 else 1)),
                team_id=str(row["team_id"]),
                player_id=str(row["player_id"]) if pd.notna(row.get("player_id")) else None,
                xg=float(row["xg"]) if pd.notna(row.get("xg")) else None,
                x=float(row["x_start"]) if pd.notna(row.get("x_start")) else None,
                y=float(row["y_start"]) if pd.notna(row.get("y_start")) else None,
                from_open_play=bool(row.get("from_open_play")),
                build_up=str(row["shot_build_up"]) if pd.notna(row.get("shot_build_up")) else None,
                setup=str(row["shot_setup"]) if pd.notna(row.get("shot_setup")) else None,
            )
        )
    return tuple(anchors)


def controlled_team_at_frame(
    frame: BundesligaFrame,
    config: ShotContextConfig = ShotContextConfig(),
) -> str | None:
    """Assign strict foot control to the nearest player, otherwise return None."""

    if frame.ball is None or not frame.players:
        return None
    if frame.ball.z is not None and frame.ball.z > config.max_control_ball_height_m:
        return None
    nearest = min(
        frame.players.values(),
        key=lambda player: math.hypot(player.x - frame.ball.x, player.y - frame.ball.y),
    )
    distance = math.hypot(nearest.x - frame.ball.x, nearest.y - frame.ball.y)
    return nearest.team_id if distance <= config.control_distance_m else None


def _possession_boundary(
    frames: Mapping[int, BundesligaFrame],
    ordered_ids: Sequence[int],
    shot_team_id: str,
    config: ShotContextConfig,
) -> tuple[int, str, dict[int, str | None]]:
    owners = {frame_id: controlled_team_at_frame(frames[frame_id], config) for frame_id in ordered_ids}
    known = [(frame_id, owners[frame_id]) for frame_id in ordered_ids if owners[frame_id] is not None]
    latest_attacking_control = next(
        (
            frame_id
            for frame_id, owner in reversed(known)
            if owner == shot_team_id
        ),
        None,
    )
    if latest_attacking_control is None:
        return ordered_ids[0], "no_confirmed_attacking_control", owners
    # A blocked shot or a loose ball can put a defender closer to the ball after
    # the last attacking foot-control sample.  The provider shot event is the
    # stronger possession signal at that boundary, so those later proximity
    # samples must not create a false opponent-possession break.
    known = [sample for sample in known if sample[0] <= latest_attacking_control]
    opponent_count = 0
    newest_opponent_frame: int | None = None
    for frame_id, owner in reversed(known):
        if owner == shot_team_id:
            opponent_count = 0
            newest_opponent_frame = None
            continue
        if opponent_count == 0:
            newest_opponent_frame = frame_id
        opponent_count += 1
        if opponent_count < config.opponent_confirmation_samples:
            continue
        if newest_opponent_frame is None:
            break
        next_attacking_control = next(
            (
                candidate_id
                for candidate_id, candidate_owner in known
                if candidate_id > newest_opponent_frame and candidate_owner == shot_team_id
            ),
            newest_opponent_frame + 1,
        )
        return next_attacking_control, "confirmed_opponent_control", owners
    return ordered_ids[0], "lookback_cap_or_period_start", owners


def _last_attacking_half_entry(
    frames: Mapping[int, BundesligaFrame],
    ordered_ids: Sequence[int],
    attacking_direction: int,
) -> int:
    first_id = ordered_ids[0]
    entry = first_id
    previous_attacking: bool | None = None
    for frame_id in ordered_ids:
        ball = frames[frame_id].ball
        if ball is None:
            continue
        attacking = attacking_direction * ball.x >= 0.0
        if attacking and previous_attacking is not True:
            entry = frame_id
        previous_attacking = attacking
    return entry


def infer_shot_attacking_phase(
    shot: ShotAnchor,
    frames: Mapping[int, BundesligaFrame],
    metadata: BundesligaMatchMeta,
    config: ShotContextConfig = ShotContextConfig(),
) -> ShotAttackingPhase:
    """Infer the shot-ending possession and its final attacking-half entry."""

    config.validate()
    shot_frame = frames.get(shot.frame_id)
    if shot_frame is None:
        raise ValueError(f"shot frame {shot.frame_id} is unavailable")
    period_ids = sorted(
        frame_id
        for frame_id, frame in frames.items()
        if frame.period == shot.period
        and shot.frame_id - config.max_lookback_frames <= frame_id <= shot.frame_id
    )
    if not period_ids:
        raise ValueError(f"no lookback frames are available for shot {shot.event_id}")

    possession_start, boundary_reason, owners = _possession_boundary(
        frames, period_ids, shot.team_id, config
    )
    possession_ids = [frame_id for frame_id in period_ids if frame_id >= possession_start]
    direction = infer_attacking_direction(shot_frame, shot.team_id, metadata)
    attacking_half_entry = _last_attacking_half_entry(frames, possession_ids, direction)

    ball_samples = [frames[frame_id].ball for frame_id in possession_ids]
    ball_samples = [ball for ball in ball_samples if ball is not None]
    attacking_samples = sum(direction * ball.x >= 0.0 for ball in ball_samples)
    known_control = [owners[frame_id] for frame_id in possession_ids if owners[frame_id] is not None]
    attacking_control = sum(owner == shot.team_id for owner in known_control)
    total = max(1, len(possession_ids))

    window_starts = tuple(
        (
            seconds,
            max(period_ids[0], shot.frame_id - int(round(seconds * FPS))),
        )
        for seconds in config.comparison_windows_seconds
    )
    return ShotAttackingPhase(
        shot=shot,
        available_start_frame_id=period_ids[0],
        possession_start_frame_id=possession_start,
        attacking_half_entry_frame_id=attacking_half_entry,
        shot_frame_id=shot.frame_id,
        attacking_direction=direction,
        boundary_reason=boundary_reason,
        duration_seconds=(shot.frame_id - possession_start) / FPS,
        attacking_half_duration_seconds=(shot.frame_id - attacking_half_entry) / FPS,
        attacking_half_fraction=attacking_samples / max(1, len(ball_samples)),
        attacking_team_control_fraction=attacking_control / max(1, len(known_control)),
        known_control_fraction=len(known_control) / total,
        window_start_frame_ids=window_starts,
    )
