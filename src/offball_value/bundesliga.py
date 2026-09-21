from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import pandas as pd


FPS = 25.0
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


@dataclass(frozen=True)
class BundesligaTeamMeta:
    team_id: str
    name: str
    role: str


@dataclass(frozen=True)
class BundesligaPlayerMeta:
    player_id: str
    team_id: str
    short_name: str
    shirt_number: str | None = None
    position: str | None = None
    starting: bool | None = None


@dataclass(frozen=True)
class BundesligaMatchMeta:
    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    teams: dict[str, BundesligaTeamMeta]
    players: dict[str, BundesligaPlayerMeta]

    def team_role(self, team_id: str) -> str | None:
        team = self.teams.get(team_id)
        return team.role if team else None

    def goalkeeper_id(self, team_id: str) -> str | None:
        for player in self.players.values():
            if player.team_id == team_id and player.position == "TW":
                return player.player_id
        return None


@dataclass(frozen=True)
class BundesligaObjectState:
    object_id: str
    team_id: str
    x: float
    y: float
    z: float | None = None
    speed: float | None = None


@dataclass(frozen=True)
class BundesligaFrame:
    match_id: str
    frame_id: int
    period: int
    game_section: str
    timestamp: datetime | None
    players: dict[str, BundesligaObjectState]
    ball: BundesligaObjectState | None = None

    def with_player(self, player_id: str, state: BundesligaObjectState) -> "BundesligaFrame":
        players = dict(self.players)
        players[player_id] = state
        return replace(self, players=players)


@dataclass(frozen=True)
class BundesligaClock:
    section_start: dict[str, tuple[int, datetime]]

    def period_for_time(self, timestamp: datetime) -> int:
        second_half = self.section_start.get("secondHalf")
        if second_half is not None and timestamp >= second_half[1]:
            return 2
        return 1

    def frame_for_time(self, timestamp: datetime) -> int:
        section = "secondHalf" if self.period_for_time(timestamp) == 2 else "firstHalf"
        start_frame, start_time = self.section_start[section]
        seconds = (timestamp - start_time).total_seconds()
        return int(round(start_frame + seconds * FPS))


def normalize_bundesliga_match_id(match_id: str) -> str:
    match_id = match_id.strip()
    if match_id.startswith("DFL-MAT-"):
        return match_id
    return f"DFL-MAT-{match_id}"


def short_bundesliga_match_id(match_id: str) -> str:
    return normalize_bundesliga_match_id(match_id).removeprefix("DFL-MAT-")


def find_bundesliga_files(base_dir: str | Path, match_id: str) -> dict[str, Path]:
    base = Path(base_dir)
    normalized = normalize_bundesliga_match_id(match_id)
    files: dict[str, Path] = {}
    patterns = {
        "matchinfo": f"DFL_02_01_matchinformation_*_{normalized}.xml",
        "events": f"DFL_03_02_events_raw_*_{normalized}.xml",
        "positions": f"DFL_04_03_positions_raw_observed_*_{normalized}.xml",
    }
    for key, pattern in patterns.items():
        matches = sorted(base.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No Bundesliga {key} XML matching {pattern} under {base}")
        files[key] = matches[0]
    return files


def list_bundesliga_match_ids(base_dir: str | Path) -> list[str]:
    base = Path(base_dir)
    ids = []
    for path in sorted(base.glob("DFL_02_01_matchinformation_*_DFL-MAT-*.xml")):
        ids.append(path.stem.split("_")[-1])
    return ids


def parse_bundesliga_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _float_attr(elem: ET.Element, name: str) -> float | None:
    value = elem.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool_attr(elem: ET.Element | None, name: str) -> bool | None:
    if elem is None:
        return None
    value = elem.attrib.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _centered_x(value: float | None) -> float | None:
    return None if value is None else value - FIELD_LENGTH / 2.0


def _centered_y(value: float | None) -> float | None:
    return None if value is None else value - FIELD_WIDTH / 2.0


def load_bundesliga_match_metadata(matchinfo_xml: str | Path) -> BundesligaMatchMeta:
    root = ET.parse(matchinfo_xml).getroot()
    general = root.find(".//General")
    if general is None:
        raise ValueError(f"No <General> match metadata found in {matchinfo_xml}")

    teams: dict[str, BundesligaTeamMeta] = {}
    players: dict[str, BundesligaPlayerMeta] = {}

    for team_elem in root.findall(".//Team"):
        team_id = team_elem.attrib["TeamId"]
        teams[team_id] = BundesligaTeamMeta(
            team_id=team_id,
            name=team_elem.attrib.get("TeamName", team_id),
            role=team_elem.attrib.get("Role", ""),
        )
        for player_elem in team_elem.findall("./Players/Player"):
            player_id = player_elem.attrib["PersonId"]
            players[player_id] = BundesligaPlayerMeta(
                player_id=player_id,
                team_id=team_id,
                short_name=player_elem.attrib.get("Shortname", player_id),
                shirt_number=player_elem.attrib.get("ShirtNumber"),
                position=player_elem.attrib.get("PlayingPosition"),
                starting=player_elem.attrib.get("Starting") == "true",
            )

    return BundesligaMatchMeta(
        match_id=general.attrib["MatchId"],
        home_team_id=general.attrib["HomeTeamId"],
        away_team_id=general.attrib["GuestTeamId"],
        home_team_name=general.attrib.get("HomeTeamName", general.attrib["HomeTeamId"]),
        away_team_name=general.attrib.get("GuestTeamName", general.attrib["GuestTeamId"]),
        teams=teams,
        players=players,
    )


def load_bundesliga_frame_clock(positions_xml: str | Path) -> BundesligaClock:
    section_start: dict[str, tuple[int, datetime]] = {}
    current_frameset: dict[str, str] | None = None

    for event, elem in ET.iterparse(positions_xml, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_frameset = dict(elem.attrib)
            continue

        if event == "start" and elem.tag == "Frame" and current_frameset:
            if current_frameset.get("TeamId") == "BALL":
                section = current_frameset.get("GameSection")
                if section and section not in section_start:
                    timestamp = parse_bundesliga_datetime(elem.attrib.get("T"))
                    if timestamp is not None:
                        section_start[section] = (int(elem.attrib["N"]), timestamp)
                if {"firstHalf", "secondHalf"} <= set(section_start):
                    break

        if event == "end" and elem.tag == "FrameSet":
            current_frameset = None
            elem.clear()
        elif event == "end":
            elem.clear()

    if "firstHalf" not in section_start or "secondHalf" not in section_start:
        raise ValueError(f"Could not infer first/second-half frame clocks from {positions_xml}")

    return BundesligaClock(section_start=section_start)


def load_bundesliga_events(
    events_xml: str | Path,
    clock: BundesligaClock | None = None,
) -> pd.DataFrame:
    rows = []
    root = ET.parse(events_xml).getroot()

    for event_elem in root.findall("./Event"):
        event_time = parse_bundesliga_datetime(event_elem.attrib.get("EventTime"))
        frame_id = event_elem.attrib.get("CalculatedFrame")
        if frame_id is not None:
            frame_value = int(frame_id)
            period = 2 if frame_value >= 100000 else 1
        elif clock is not None and event_time is not None:
            frame_value = clock.frame_for_time(event_time)
            period = clock.period_for_time(event_time)
        else:
            frame_value = None
            period = None

        x_source = _float_attr(event_elem, "X-Source-Position")
        y_source = _float_attr(event_elem, "Y-Source-Position")
        x_position = _float_attr(event_elem, "X-Position")
        y_position = _float_attr(event_elem, "Y-Position")

        top_level_elem = next(iter(list(event_elem)), None)
        play_elem = event_elem.find(".//Play")
        action_elem = None
        if play_elem is not None:
            action_elem = next(iter(list(play_elem)), None)

        event_type = action_elem.tag if action_elem is not None else (
            play_elem.tag if play_elem is not None else (
                top_level_elem.tag if top_level_elem is not None else event_elem.tag
            )
        )
        actor_elem = play_elem if play_elem is not None else top_level_elem
        actor_attrib = actor_elem.attrib if actor_elem is not None else {}
        top_level_type = top_level_elem.tag.lower() if top_level_elem is not None else None

        from_open_play = _bool_attr(actor_elem, "FromOpenPlay")
        shot_build_up = actor_attrib.get("BuildUp")
        shot_setup = actor_attrib.get("TakerSetup")
        if from_open_play is None and event_type.lower() == "shotatgoal":
            open_play_tokens = f"{shot_build_up or ''} {shot_setup or ''}".lower()
            if "openplay" in open_play_tokens or shot_build_up in {"run", "lossOfPossession"}:
                from_open_play = True
            elif shot_build_up in {"cornerKick", "freeKick", "penaltyKick", "throwIn"}:
                from_open_play = False

        rows.append(
            {
                "match_id": event_elem.attrib.get("MatchId"),
                "event_id": event_elem.attrib.get("EventId"),
                "period": period,
                "frame_id": frame_value,
                "time": event_time,
                "event_type": event_type.lower() if event_type else None,
                "parent_event_type": top_level_type,
                "player_id": actor_attrib.get("Player"),
                "team_id": actor_attrib.get("Team"),
                "recipient_id": actor_attrib.get("Recipient"),
                "evaluation": actor_attrib.get("Evaluation"),
                "from_open_play": from_open_play,
                "ball_possession_phase": actor_attrib.get("BallPossessionPhase"),
                "xg": _float_attr(actor_elem, "xG") if actor_elem is not None else None,
                "shot_build_up": shot_build_up,
                "shot_setup": shot_setup,
                "after_free_kick": _bool_attr(actor_elem, "AfterFreeKick"),
                "x_start": _centered_x(x_source if x_source is not None else x_position),
                "y_start": _centered_y(y_source if y_source is not None else y_position),
                "x_end": _centered_x(x_position),
                "y_end": _centered_y(y_position),
            }
        )

    return pd.DataFrame(rows)


def load_bundesliga_frames(
    positions_xml: str | Path,
    target_frames: Iterable[int],
    include_referees: bool = False,
) -> dict[int, BundesligaFrame]:
    target_set = {int(frame) for frame in target_frames if frame is not None and int(frame) >= 0}
    frames: dict[int, BundesligaFrame] = {}
    if not target_set:
        return frames

    current_frameset: dict[str, str] | None = None

    for event, elem in ET.iterparse(positions_xml, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current_frameset = dict(elem.attrib)
            continue

        if event == "end" and elem.tag == "Frame" and current_frameset:
            frame_id = int(elem.attrib["N"])
            if frame_id in target_set:
                team_id = current_frameset.get("TeamId", "")
                person_id = current_frameset.get("PersonId", "")
                section = current_frameset.get("GameSection", "")
                frame = frames.get(frame_id)
                if frame is None:
                    frame = BundesligaFrame(
                        match_id=current_frameset.get("MatchId", ""),
                        frame_id=frame_id,
                        period=2 if section == "secondHalf" else 1,
                        game_section=section,
                        timestamp=parse_bundesliga_datetime(elem.attrib.get("T")),
                        players={},
                        ball=None,
                    )

                state = BundesligaObjectState(
                    object_id=person_id,
                    team_id=team_id,
                    x=float(elem.attrib["X"]),
                    y=float(elem.attrib["Y"]),
                    z=_float_attr(elem, "Z"),
                    speed=_float_attr(elem, "S"),
                )

                if team_id == "BALL":
                    frame = replace(frame, ball=state)
                elif include_referees or team_id.startswith("DFL-CLU-"):
                    frame = frame.with_player(person_id, state)
                frames[frame_id] = frame

            elem.clear()

        elif event == "end" and elem.tag == "FrameSet":
            current_frameset = None
            elem.clear()
        elif event == "end":
            elem.clear()

    return frames


def infer_attacking_direction(
    frame: BundesligaFrame,
    team_id: str,
    metadata: BundesligaMatchMeta,
) -> int:
    goalkeeper_id = metadata.goalkeeper_id(team_id)
    if goalkeeper_id and goalkeeper_id in frame.players:
        goalkeeper_x = frame.players[goalkeeper_id].x
        return 1 if goalkeeper_x < 0 else -1

    team_players = [p for p in frame.players.values() if p.team_id == team_id]
    if not team_players:
        return 1
    mean_x = sum(p.x for p in team_players) / len(team_players)
    return -1 if mean_x > 0 else 1
