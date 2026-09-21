from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    possession_player_id: str | None = None
    possession_group: str | None = None
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
    receiver_id: str | None = None
    receiver_team: str | None = None
    outcome: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _to_float(x: Any) -> float | None:
    if _is_missing(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x: Any) -> int | None:
    if _is_missing(x):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_str(x: Any) -> str | None:
    if _is_missing(x):
        return None
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _parse_time_to_seconds(x: Any) -> float | None:
    """
    Handles:
      - 12.34
      - "12.34"
      - "00:12.3"
      - "01:02:03.4"
    """
    if _is_missing(x):
        return None

    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()

    try:
        return float(s)
    except ValueError:
        pass

    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
            return float(mm) * 60.0 + float(ss)
        elif len(parts) == 3:
            hh, mm, ss = parts
            return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    except ValueError:
        return None

    return None


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
        x = _to_float(row.get(key))
        y = _to_float(row.get(y_key))
        if x is None or y is None:
            continue

        player_id = key[:-2]
        team = matched_prefix[:-1]
        players.append(
            PlayerSnapshot(
                player_id=player_id,
                team=team,
                x=x,
                y=y,
                raw={},
            )
        )

    ball = None
    ball_x = _to_float(row.get("ball_x"))
    ball_y = _to_float(row.get("ball_y"))
    if ball_x is not None and ball_y is not None:
        ball = BallSnapshot(
            x=ball_x,
            y=ball_y,
            raw={},
        )

    return FrameSnapshot(
        source="metrica",
        match_id=match_id,
        period=_to_int(row.get("Period")),
        frame_id=_to_int(row.get("Frame")),
        time_s=_to_float(row.get("Time [s]")),
        ball=ball,
        players=players,
        possession_team=None,
        raw={},
    )


def skillcorner_frame_to_snapshot(
    frame: dict[str, Any],
    match_id: str,
    player_team_by_id: dict[str, str] | None = None,
    player_role_by_id: dict[str, str] | None = None,
    possession_group_to_team: dict[str, str] | None = None,
) -> FrameSnapshot:
    players: list[PlayerSnapshot] = []
    player_team_by_id = player_team_by_id or {}
    player_role_by_id = player_role_by_id or {}

    for p in frame.get("player_data", []):
        x = p.get("x")
        y = p.get("y")

        # fallback for nested location-like structures
        if (x is None or y is None) and isinstance(p.get("coordinates"), dict):
            x = p["coordinates"].get("x")
            y = p["coordinates"].get("y")

        if x is None or y is None:
            continue

        player_id = str(
            p.get("player_id")
            or p.get("trackable_object")
            or p.get("id")
            or "unknown"
        )
        team = str(
            player_team_by_id.get(player_id)
            or p.get("team_id")
            or p.get("team")
            or p.get("team_shortname")
            or "unknown"
        )
        z = p.get("z")
        role = p.get("player_role") or player_role_by_id.get(player_id)

        players.append(
            PlayerSnapshot(
                player_id=player_id,
                team=team,
                x=float(x),
                y=float(y),
                z=float(z) if z is not None else None,
                role=str(role) if role is not None else None,
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
    possession_player_id = None
    possession_group = None
    if isinstance(possession, dict):
        possession_player_id = _to_str(possession.get("player_id"))
        possession_group = _to_str(possession.get("group"))
        for key in ["team_id", "team", "team_in_possession_id"]:
            if possession.get(key) is not None:
                possession_team = str(possession[key])
                break
        if possession_team is None and possession_group is not None:
            normalized_group = possession_group.strip().lower()
            if possession_group_to_team and normalized_group in possession_group_to_team:
                possession_team = possession_group_to_team[normalized_group]
            else:
                possession_team = possession_group

    return FrameSnapshot(
        source="skillcorner",
        match_id=match_id,
        period=_to_int(frame.get("period")),
        frame_id=_to_int(frame.get("frame")),
        time_s=_parse_time_to_seconds(frame.get("timestamp")),
        ball=ball_obj,
        players=players,
        possession_team=possession_team,
        possession_player_id=possession_player_id,
        possession_group=possession_group,
        raw=frame,
    )


def skillcorner_event_to_record(row: dict[str, Any], match_id: str) -> EventRecord:
    receiver_id = _to_str(row.get("player_targeted_id"))
    outcome = _to_str(row.get("pass_outcome")) or _to_str(row.get("end_type"))

    return EventRecord(
        source="skillcorner",
        match_id=match_id,
        event_id=_to_str(row.get("event_id")),
        period=_to_int(row.get("period")),
        time_s=_parse_time_to_seconds(row.get("time_start")),
        event_type=_to_str(row.get("event_type")),
        player_id=_to_str(row.get("player_id")),
        team=_to_str(row.get("team_id")),
        x_start=_to_float(row.get("x_start")),
        y_start=_to_float(row.get("y_start")),
        x_end=_to_float(row.get("x_end")),
        y_end=_to_float(row.get("y_end")),
        frame_start=_to_int(row.get("frame_start")),
        frame_end=_to_int(row.get("frame_end")),
        receiver_id=receiver_id,
        receiver_team=_to_str(row.get("team_id")) if receiver_id is not None else None,
        outcome=outcome,
        raw=row,
    )


def statsbomb_event_to_record(event: dict[str, Any], match_id: str) -> EventRecord:
    loc = event.get("location") or [None, None]
    x_start = float(loc[0]) if len(loc) > 0 and loc[0] is not None else None
    y_start = float(loc[1]) if len(loc) > 1 and loc[1] is not None else None

    x_end = y_end = None
    receiver_id = receiver_team = outcome = None
    if isinstance(event.get("pass"), dict):
        pass_payload = event["pass"]
        end_loc = pass_payload.get("end_location") or [None, None]
        x_end = float(end_loc[0]) if len(end_loc) > 0 and end_loc[0] is not None else None
        y_end = float(end_loc[1]) if len(end_loc) > 1 and end_loc[1] is not None else None
        if isinstance(pass_payload.get("recipient"), dict):
            receiver_id = _to_str(pass_payload["recipient"].get("id"))
            receiver_team = _to_str(event["team"].get("id")) if isinstance(event.get("team"), dict) else None
        if isinstance(pass_payload.get("outcome"), dict):
            outcome = _to_str(pass_payload["outcome"].get("name"))
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
        period=_to_int(event.get("period")),
        time_s=_parse_time_to_seconds(event.get("timestamp")),
        event_type=_to_str(event["type"].get("name")) if isinstance(event.get("type"), dict) else None,
        player_id=_to_str(event["player"].get("id")) if isinstance(event.get("player"), dict) else None,
        team=_to_str(event["team"].get("id")) if isinstance(event.get("team"), dict) else None,
        x_start=x_start,
        y_start=y_start,
        x_end=x_end,
        y_end=y_end,
        frame_start=None,
        frame_end=None,
        receiver_id=receiver_id,
        receiver_team=receiver_team,
        outcome=outcome,
        raw=event,
    )
