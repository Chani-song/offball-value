from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .schema import EventRecord, FrameSnapshot, PlayerSnapshot


@dataclass(frozen=True)
class BallCarrier:
    player_id: str
    team: str
    distance_to_ball: float
    frame_id: int | None
    time_s: float | None


@dataclass(frozen=True)
class ControlledPossession:
    frame_index: int
    carrier: BallCarrier


@dataclass(frozen=True)
class CompletedPass:
    event_id: str | None
    team: str | None
    passer_id: str | None
    receiver_id: str
    pass_frame_id: int | None
    receive_frame_id: int | None
    pass_time_s: float | None
    receive_time_s: float | None
    frames_to_receive: int | None


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def default_ball_control_distance(frame: FrameSnapshot) -> float:
    """Return a rough foot-control radius for the frame coordinate system.

    Metrica sample tracking is normalized, while SkillCorner-style tracking is
    in meters. This heuristic keeps callers from hard-coding one scale.
    """
    xs: list[float] = []
    ys: list[float] = []
    if frame.ball is not None:
        xs.append(frame.ball.x)
        ys.append(frame.ball.y)
    for player in frame.players:
        xs.append(player.x)
        ys.append(player.y)

    if not xs or not ys:
        return 0.03

    max_abs_coordinate = max(abs(v) for v in xs + ys)
    return 0.03 if max_abs_coordinate <= 2.0 else 2.5


def _player_by_id(players: Iterable[PlayerSnapshot], player_id: str | None) -> PlayerSnapshot | None:
    if player_id is None:
        return None
    for player in players:
        if player.player_id == player_id:
            return player
    return None


def _carrier_from_player(frame: FrameSnapshot, player: PlayerSnapshot) -> BallCarrier | None:
    if frame.ball is None:
        return None
    return BallCarrier(
        player_id=player.player_id,
        team=player.team,
        distance_to_ball=euclidean_distance(player.x, player.y, frame.ball.x, frame.ball.y),
        frame_id=frame.frame_id,
        time_s=frame.time_s,
    )


def infer_ball_carrier(
    frame: FrameSnapshot,
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
    use_provider_possession: bool = True,
) -> BallCarrier | None:
    """Infer a foot-control carrier for one frame.

    The project convention is intentionally strict: a team is treated as in
    possession for state construction only when the ball is close enough to a
    player. Pass-flight frames are therefore not standalone possession states.
    """
    if frame.ball is None or not frame.players:
        return None

    if max_ball_height is not None and frame.ball.z is not None and frame.ball.z > max_ball_height:
        return None

    resolved_max_distance = max_distance
    if resolved_max_distance is None:
        resolved_max_distance = default_ball_control_distance(frame)

    if use_provider_possession:
        provider_player = _player_by_id(frame.players, frame.possession_player_id)
        if provider_player is not None:
            carrier = _carrier_from_player(frame, provider_player)
            if carrier is not None and carrier.distance_to_ball <= resolved_max_distance:
                return carrier

    nearest = min(
        frame.players,
        key=lambda player: euclidean_distance(player.x, player.y, frame.ball.x, frame.ball.y),
    )
    carrier = _carrier_from_player(frame, nearest)
    if carrier is None or carrier.distance_to_ball > resolved_max_distance:
        return None
    return carrier


def controlled_team(
    frame: FrameSnapshot,
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
) -> str | None:
    carrier = infer_ball_carrier(
        frame,
        max_distance=max_distance,
        max_ball_height=max_ball_height,
    )
    return carrier.team if carrier is not None else None


def is_controlled_by(
    frame: FrameSnapshot,
    team: str,
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
) -> bool:
    return controlled_team(
        frame,
        max_distance=max_distance,
        max_ball_height=max_ball_height,
    ) == team


def iter_controlled_possessions(
    frames: Sequence[FrameSnapshot],
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
) -> Iterable[ControlledPossession]:
    for idx, frame in enumerate(frames):
        carrier = infer_ball_carrier(
            frame,
            max_distance=max_distance,
            max_ball_height=max_ball_height,
        )
        if carrier is not None:
            yield ControlledPossession(frame_index=idx, carrier=carrier)


def next_controlled_possession(
    frames: Sequence[FrameSnapshot],
    start_index: int,
    team: str | None = None,
    receiver_id: str | None = None,
    exclude_player_id: str | None = None,
    max_frames: int | None = None,
    max_time_s: float | None = None,
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
) -> ControlledPossession | None:
    if start_index < 0:
        start_index = 0
    if start_index >= len(frames):
        return None

    start_time_s = frames[start_index].time_s
    stop_index = len(frames) if max_frames is None else min(len(frames), start_index + max_frames + 1)

    for idx in range(start_index, stop_index):
        frame = frames[idx]
        if (
            max_time_s is not None
            and start_time_s is not None
            and frame.time_s is not None
            and frame.time_s - start_time_s > max_time_s
        ):
            return None

        carrier = infer_ball_carrier(
            frame,
            max_distance=max_distance,
            max_ball_height=max_ball_height,
        )
        if carrier is None:
            continue
        if team is not None and carrier.team != team:
            continue
        if receiver_id is not None and carrier.player_id != receiver_id:
            continue
        if exclude_player_id is not None and carrier.player_id == exclude_player_id:
            continue
        return ControlledPossession(frame_index=idx, carrier=carrier)

    return None


def _is_pass_attempt(event: EventRecord) -> bool:
    event_type = (event.event_type or "").strip().lower()
    end_type = str(event.raw.get("end_type", "")).strip().lower()
    return (
        event_type == "pass"
        or end_type == "pass"
        or (event_type == "player_possession" and event.receiver_id is not None)
    )


def _frame_index_at_or_after_event(frames: Sequence[FrameSnapshot], event: EventRecord) -> int | None:
    for idx, frame in enumerate(frames):
        if event.period is not None and frame.period != event.period:
            continue
        if event.frame_start is not None and frame.frame_id is not None and frame.frame_id >= event.frame_start:
            return idx
        if (
            event.frame_start is None
            and event.time_s is not None
            and frame.time_s is not None
            and frame.time_s >= event.time_s
        ):
            return idx
    return None


def completed_passes_from_events(
    events: Iterable[EventRecord],
    frames: Sequence[FrameSnapshot],
    max_frames_after_event: int = 75,
    max_time_after_event_s: float = 5.0,
    max_distance: float | None = None,
    max_ball_height: float | None = 1.5,
) -> list[CompletedPass]:
    """Convert pass attempts into completed A->B transitions after reception.

    This uses future frames only to label the pass transition. Do not feed the
    resulting receiver/success information back into state features at pass time.
    """
    completed: list[CompletedPass] = []

    for event in events:
        if not _is_pass_attempt(event):
            continue

        start_idx = _frame_index_at_or_after_event(frames, event)
        if start_idx is None:
            continue

        receiver_id = event.receiver_id
        received = next_controlled_possession(
            frames,
            start_index=start_idx + 1,
            team=event.team,
            receiver_id=receiver_id,
            exclude_player_id=event.player_id if receiver_id is None else None,
            max_frames=max_frames_after_event,
            max_time_s=max_time_after_event_s,
            max_distance=max_distance,
            max_ball_height=max_ball_height,
        )
        if received is None:
            continue

        completed.append(
            CompletedPass(
                event_id=event.event_id,
                team=event.team,
                passer_id=event.player_id,
                receiver_id=receiver_id or received.carrier.player_id,
                pass_frame_id=event.frame_start,
                receive_frame_id=received.carrier.frame_id,
                pass_time_s=event.time_s,
                receive_time_s=received.carrier.time_s,
                frames_to_receive=received.frame_index - start_idx,
            )
        )

    return completed
