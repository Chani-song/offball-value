from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SourceName = Literal["metrica", "skillcorner", "statsbomb"]


@dataclass
class PlayerSnapshot:
    player_id: str
    team: str
    x: float
    y: float
    z: float | None = None
    role: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BallSnapshot:
    x: float
    y: float
    z: float | None = None
    is_detected: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameSnapshot:
    source: SourceName
    match_id: str
    period: int | None
    frame_id: int | None
    time_s: float | None
    ball: BallSnapshot | None
    players: list[PlayerSnapshot]
    possession_team: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    source: SourceName
    match_id: str
    event_id: str | None
    period: int | None
    time_s: float | None
    event_type: str | None
    player_id: str | None
    team: str | None
    x_start: float | None = None
    y_start: float | None = None
    x_end: float | None = None
    y_end: float | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def metrica_row_to_frame(
    row: dict[str, Any],
    match_id: str,
    team_prefixes: tuple[str, ...] = ("Home_", "Away_"),
) -> FrameSnapshot:
    players: list[PlayerSnapshot] = []

    for key, value in row.items():
        if not key.endswith("_x"):
            continue
        matched_prefix = None
        for prefix in team_prefixes:
            if key.startswith(prefix):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            continue

        y_key = key[:-2] + "_y"
        x = row.get(key)
        y = row.get(y_key)
        if x is None or y is None:
            continue

        player_id = key[:-2]  # e.g. Home_11
        team = matched_prefix[:-1]  # Home / Away
        players.append(
            PlayerSnapshot(
                player_id=player_id,
                team=team,
                x=float(x),
                y=float(y),
                raw={},
            )
        )

    ball = None
    if row.get("ball_x") is not None and row.get("ball_y") is not None:
        ball = BallSnapshot(
            x=float(row["ball_x"]),
            y=float(row["ball_y"]),
            raw={},
        )

    return FrameSnapshot(
        source="metrica",
        match_id=match_id,
        period=int(row["Period"]) if row.get("Period") is not None else None,
        frame_id=int(row["Frame"]) if row.get("Frame") is not None else None,
        time_s=float(row["Time [s]"]) if row.get("Time [s]") is not None else None,
        ball=ball,
        players=players,
        possession_team=None,
        raw={},
    )


def skillcorner_frame_to_snapshot(frame: dict[str, Any], match_id: str) -> FrameSnapshot:
    players: list[PlayerSnapshot] = []
    for p in frame.get("player_data", []):
        player_id = str(p.get("player_id", "unknown"))
        team = str(p.get("team_id", p.get("team", "unknown")))
        x = p.get("x")
        y = p.get("y")
        z = p.get("z")
        if x is None or y is None:
            continue

        players.append(
            PlayerSnapshot(
                player_id=player_id,
                team=team,
                x=float(x),
                y=float(y),
                z=float(z) if z is not None else None,
                role=str(p.get("player_role")) if p.get("player_role") is not None else None,
                raw=p,
            )
        )

    ball_obj = None
    ball = frame.get("ball_data")
    if isinstance(ball, dict) and ball.get("x") is not None and ball.get("y") is not None:
        ball_obj = BallSnapshot(
            x=float(ball["x"]),
            y=float(ball["y"]),
            z=float(ball["z"]) if ball.get("z") is not None else None,
            is_detected=bool(ball["is_detected"]) if ball.get("is_detected") is not None else None,
            raw=ball,
        )

    possession = frame.get("possession")
    possession_team = None
    if isinstance(possession, dict):
        for key in ["team_id", "team", "team_in_possession_id"]:
            if possession.get(key) is not None:
                possession_team = str(possession[key])
                break

    return FrameSnapshot(
        source="skillcorner",
        match_id=match_id,
        period=int(frame["period"]) if frame.get("period") is not None else None,
        frame_id=int(frame["frame"]) if frame.get("frame") is not None else None,
        time_s=float(frame["timestamp"]) if frame.get("timestamp") is not None else None,
        ball=ball_obj,
        players=players,
        possession_team=possession_team,
        raw=frame,
    )


def skillcorner_event_to_record(row: dict[str, Any], match_id: str) -> EventRecord:
    return EventRecord(
        source="skillcorner",
        match_id=match_id,
        event_id=str(row.get("event_id")) if row.get("event_id") is not None else None,
        period=int(row["period"]) if row.get("period") not in [None, ""] else None,
        time_s=float(row["time_start"]) if row.get("time_start") not in [None, ""] else None,
        event_type=str(row.get("event_type")) if row.get("event_type") is not None else None,
        player_id=str(row.get("player_id")) if row.get("player_id") not in [None, ""] else None,
        team=str(row.get("team_id")) if row.get("team_id") not in [None, ""] else None,
        x_start=float(row["x_start"]) if row.get("x_start") not in [None, ""] else None,
        y_start=float(row["y_start"]) if row.get("y_start") not in [None, ""] else None,
        x_end=float(row["x_end"]) if row.get("x_end") not in [None, ""] else None,
        y_end=float(row["y_end"]) if row.get("y_end") not in [None, ""] else None,
        frame_start=int(row["frame_start"]) if row.get("frame_start") not in [None, ""] else None,
        frame_end=int(row["frame_end"]) if row.get("frame_end") not in [None, ""] else None,
        raw=row,
    )


def statsbomb_event_to_record(event: dict[str, Any], match_id: str) -> EventRecord:
    loc = event.get("location") or [None, None]
    x_start = float(loc[0]) if len(loc) > 0 and loc[0] is not None else None
    y_start = float(loc[1]) if len(loc) > 1 and loc[1] is not None else None

    x_end = y_end = None
    if isinstance(event.get("pass"), dict):
        end_loc = event["pass"].get("end_location") or [None, None]
        x_end = float(end_loc[0]) if len(end_loc) > 0 and end_loc[0] is not None else None
        y_end = float(end_loc[1]) if len(end_loc) > 1 and end_loc[1] is not None else None
    elif isinstance(event.get("carry"), dict):
        end_loc = event["carry"].get("end_location") or [None, None]
        x_end = float(end_loc[0]) if len(end_loc) > 0 and end_loc[0] is not None else None
        y_end = float(end_loc[1]) if len(end_loc) > 1 and end_loc[1] is not None else None
    elif isinstance(event.get("shot"), dict):
        end_loc = event["shot"].get("end_location") or [None, None]
        x_end = float(end_loc[0]) if len(end_loc) > 0 and end_loc[0] is not None else None
        y_end = float(end_loc[1]) if len(end_loc) > 1 and end_loc[1] is not None else None

    return EventRecord(
        source="statsbomb",
        match_id=match_id,
        event_id=str(event.get("id")) if event.get("id") is not None else None,
        period=int(event["period"]) if event.get("period") is not None else None,
        time_s=None,
        event_type=str(event["type"]["name"]) if isinstance(event.get("type"), dict) else None,
        player_id=str(event["player"]["id"]) if isinstance(event.get("player"), dict) else None,
        team=str(event["team"]["id"]) if isinstance(event.get("team"), dict) else None,
        x_start=x_start,
        y_start=y_start,
        x_end=x_end,
        y_end=y_end,
        frame_start=None,
        frame_end=None,
        raw=event,
    )