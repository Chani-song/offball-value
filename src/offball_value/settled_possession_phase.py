"""Forward segmentation of settled attacking possessions over full matches.

The shot-anchored pipeline walks backwards from each shot to find the
possession that produced it (``shot_context._possession_boundary``).  This
module applies the same strict foot-control ownership model *forwards* over a
sampled whole-match timeline so that every settled possession — not only the
shot-ending ones — becomes an attacking phase with the same record schema as
``attacking_phases.csv``.

Anchor semantics: where the shot-anchored schema stores ``shot_frame_id``/
``frame_id``, a settled possession stores its possession-end frame (the last
confirmed own-team control sample), and ``xg`` is left missing.  Downstream
consumers that treat ``shot_frame_id`` as "phase end anchor" therefore keep
working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import pandas as pd

from .bundesliga import (
    FPS,
    BundesligaFrame,
    BundesligaMatchMeta,
    infer_attacking_direction,
)
from .shot_context import (
    ShotContextConfig,
    _last_attacking_half_entry,
    controlled_team_at_frame,
)


#: Event types after which play is stopped, so an ongoing possession spell
#: cannot continue across them.  Restarts themselves are provider ``pass`` /
#: ``cross`` events flagged ``FromOpenPlay=False`` (throw-ins, free kicks,
#: corners, goal kicks, kick-offs), mirroring the restart definition already
#: used by ``shot_context_onset.RESTART_EVENT_TYPES``.
STOPPAGE_EVENT_TYPES = frozenset(
    {
        "foul",
        "offside",
        "penalty",
        "refereeball",
        "finalwhistle",
    }
)
RESTART_EVENT_TYPES = frozenset({"pass", "cross"})


@dataclass(frozen=True)
class SettledPossessionPhaseConfig:
    """Configuration for forward settled-possession segmentation."""

    sample_stride_frames: int = 5
    minimum_duration_seconds: float = 6.0
    require_attacking_half_presence: bool = True
    control_distance_m: float = 1.5
    max_control_ball_height_m: float = 1.5
    opponent_control_confirmation_seconds: float = 0.32
    comparison_windows_seconds: tuple[float, ...] = (10.0, 15.0, 30.0, 60.0)

    @property
    def opponent_confirmation_samples(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    self.opponent_control_confirmation_seconds
                    * FPS
                    / self.sample_stride_frames
                )
            ),
        )

    @property
    def control_config(self) -> ShotContextConfig:
        return ShotContextConfig(
            control_distance_m=self.control_distance_m,
            max_control_ball_height_m=self.max_control_ball_height_m,
        )

    def validate(self) -> None:
        if self.sample_stride_frames < 1:
            raise ValueError("sample_stride_frames must be >= 1")
        if self.minimum_duration_seconds <= 0.0:
            raise ValueError("minimum_duration_seconds must be positive")
        if self.control_distance_m <= 0.0:
            raise ValueError("control_distance_m must be positive")
        if self.max_control_ball_height_m <= 0.0:
            raise ValueError("max_control_ball_height_m must be positive")
        if self.opponent_control_confirmation_seconds <= 0.0:
            raise ValueError("opponent_control_confirmation_seconds must be positive")
        if not self.comparison_windows_seconds or any(
            value <= 0.0 for value in self.comparison_windows_seconds
        ):
            raise ValueError("comparison windows must be positive and non-empty")


@dataclass(frozen=True)
class SettledPossessionPhase:
    """One settled possession expressed in the ``attacking_phases.csv`` schema."""

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
    possession_end_frame_id: int = 0
    phase_kind: str = "settled_possession"

    def as_record(self) -> dict[str, object]:
        record = {
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
        for seconds, frame_id in self.window_start_frame_ids:
            label = (
                str(int(seconds))
                if float(seconds).is_integer()
                else str(seconds).replace(".", "_")
            )
            record[f"window_{label}s_start_frame_id"] = frame_id
        record["possession_end_frame_id"] = self.possession_end_frame_id
        record["phase_kind"] = self.phase_kind
        return record


@dataclass
class _OpenSpell:
    team_id: str
    start_frame_id: int
    last_confirmed_frame_id: int
    opponent_run: list[int] = field(default_factory=list)


def stoppage_frame_ids(events: pd.DataFrame) -> tuple[int, ...]:
    """Return frames at which play is interrupted or restarted.

    Reuses the project's events control model: restarts are non-open-play
    ``pass``/``cross`` events, and the listed stoppage event types halt play
    directly.  A possession spell never spans one of these frames.
    """

    if events.empty or "frame_id" not in events or "event_type" not in events:
        return ()
    usable = events[events["frame_id"].notna()].copy()
    if usable.empty:
        return ()
    usable["frame_id"] = usable["frame_id"].astype(int)
    types = usable["event_type"].astype(str).str.lower()
    stoppage = types.isin(STOPPAGE_EVENT_TYPES)
    if "from_open_play" in usable:
        stoppage |= types.isin(RESTART_EVENT_TYPES) & (
            usable["from_open_play"] == False  # noqa: E712
        )
    return tuple(sorted(usable.loc[stoppage, "frame_id"].unique().tolist()))


def _controlling_player_id(
    frame: BundesligaFrame | None,
    team_id: str,
    control_distance_m: float,
) -> str | None:
    if frame is None or frame.ball is None or not frame.players:
        return None
    nearest_id, nearest = min(
        frame.players.items(),
        key=lambda item: math.hypot(
            item[1].x - frame.ball.x, item[1].y - frame.ball.y
        ),
    )
    distance = math.hypot(nearest.x - frame.ball.x, nearest.y - frame.ball.y)
    if distance <= control_distance_m and nearest.team_id == team_id:
        return nearest_id
    return None


def _close_spell(
    spell: _OpenSpell,
    boundary_reason: str,
    frames: Mapping[int, BundesligaFrame],
    metadata: BundesligaMatchMeta,
    sample_ids: Sequence[int],
    owners: Mapping[int, str | None],
    config: SettledPossessionPhaseConfig,
) -> SettledPossessionPhase | None:
    start = spell.start_frame_id
    end = spell.last_confirmed_frame_id
    duration = (end - start) / FPS
    if duration < config.minimum_duration_seconds:
        return None
    spell_ids = [frame_id for frame_id in sample_ids if start <= frame_id <= end]
    if not spell_ids:
        return None
    start_frame = frames[spell_ids[0]]
    direction = infer_attacking_direction(start_frame, spell.team_id, metadata)

    ball_samples = [
        frames[frame_id].ball
        for frame_id in spell_ids
        if frames[frame_id].ball is not None
    ]
    attacking_samples = sum(direction * ball.x >= 0.0 for ball in ball_samples)
    if config.require_attacking_half_presence and attacking_samples == 0:
        return None

    known = [
        owners[frame_id] for frame_id in spell_ids if owners[frame_id] is not None
    ]
    team_known = sum(owner == spell.team_id for owner in known)
    attacking_half_entry = _last_attacking_half_entry(frames, spell_ids, direction)

    end_frame = frames[end]
    ball = end_frame.ball
    window_starts = tuple(
        (seconds, max(start, end - int(round(seconds * FPS))))
        for seconds in config.comparison_windows_seconds
    )
    return SettledPossessionPhase(
        match_id=end_frame.match_id,
        event_id=f"settledposs-{end_frame.period}-{start}",
        frame_id=end,
        period=end_frame.period,
        team_id=spell.team_id,
        player_id=_controlling_player_id(
            end_frame, spell.team_id, config.control_distance_m
        ),
        xg=None,
        x=float(ball.x) if ball is not None else None,
        y=float(ball.y) if ball is not None else None,
        from_open_play=True,
        build_up=None,
        setup=None,
        available_start_frame_id=start,
        possession_start_frame_id=start,
        attacking_half_entry_frame_id=attacking_half_entry,
        shot_frame_id=end,
        attacking_direction=direction,
        boundary_reason=boundary_reason,
        duration_seconds=duration,
        attacking_half_duration_seconds=(end - attacking_half_entry) / FPS,
        attacking_half_fraction=attacking_samples / max(1, len(ball_samples)),
        attacking_team_control_fraction=team_known / max(1, len(known)),
        known_control_fraction=len(known) / max(1, len(spell_ids)),
        window_start_frame_ids=window_starts,
        possession_end_frame_id=end,
    )


def segment_settled_possession_phases(
    frames: Mapping[int, BundesligaFrame],
    events: pd.DataFrame,
    metadata: BundesligaMatchMeta,
    config: SettledPossessionPhaseConfig = SettledPossessionPhaseConfig(),
) -> tuple[SettledPossessionPhase, ...]:
    """Segment sampled whole-match frames into settled possession phases.

    ``frames`` should be sampled roughly every ``config.sample_stride_frames``
    frames.  A spell starts at the first confirmed own-team control sample,
    stays open through unknown-control samples, and ends when opponent control
    is confirmed (mirroring ``shot_context._possession_boundary``), when a
    stoppage or restart event occurs, or at the end of the period.  The
    possession end is trimmed back to the last confirmed own-team control
    sample so dead-ball tails are excluded.
    """

    config.validate()
    stoppages = stoppage_frame_ids(events)
    phases: list[SettledPossessionPhase] = []

    by_period: dict[int, list[int]] = {}
    for frame_id in sorted(frames):
        by_period.setdefault(frames[frame_id].period, []).append(frame_id)

    for period in sorted(by_period):
        sample_ids = by_period[period]
        owners = {
            frame_id: controlled_team_at_frame(frames[frame_id], config.control_config)
            for frame_id in sample_ids
        }
        spell: _OpenSpell | None = None
        stoppage_index = 0
        previous_id: int | None = None
        for frame_id in sample_ids:
            lower = previous_id if previous_id is not None else frame_id - 1
            interrupted = False
            while stoppage_index < len(stoppages) and stoppages[stoppage_index] <= frame_id:
                if stoppages[stoppage_index] > lower:
                    interrupted = True
                stoppage_index += 1
            if interrupted and spell is not None:
                phase = _close_spell(
                    spell, "stoppage_event", frames, metadata, sample_ids, owners, config
                )
                if phase is not None:
                    phases.append(phase)
                spell = None
            previous_id = frame_id

            owner = owners[frame_id]
            if spell is None:
                if owner is not None:
                    spell = _OpenSpell(owner, frame_id, frame_id)
                continue
            if owner is None:
                continue
            if owner == spell.team_id:
                spell.last_confirmed_frame_id = frame_id
                spell.opponent_run.clear()
                continue
            spell.opponent_run.append(frame_id)
            if len(spell.opponent_run) >= config.opponent_confirmation_samples:
                next_start = spell.opponent_run[0]
                phase = _close_spell(
                    spell,
                    "confirmed_opponent_control",
                    frames,
                    metadata,
                    sample_ids,
                    owners,
                    config,
                )
                if phase is not None:
                    phases.append(phase)
                spell = _OpenSpell(owner, next_start, frame_id)
        if spell is not None:
            phase = _close_spell(
                spell, "period_end", frames, metadata, sample_ids, owners, config
            )
            if phase is not None:
                phases.append(phase)
    return tuple(phases)
