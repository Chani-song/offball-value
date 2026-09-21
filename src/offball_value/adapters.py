from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .loaders import load_metrica_tracking
from .schema import (
    EventRecord,
    FrameSnapshot,
    metrica_row_to_frame,
    skillcorner_event_to_record,
    skillcorner_frame_to_snapshot,
    statsbomb_event_to_record,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
METRICA_DIR = RAW_DIR / "metrica-sample-data" / "data"
SKILLCORNER_DIR = RAW_DIR / "skillcorner-opendata" / "data"
STATSBOMB_DIR = RAW_DIR / "statsbomb-open-data" / "data"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_base(base_dir: str | Path | None, default: Path) -> Path:
    return Path(base_dir) if base_dir is not None else default


def _merge_metrica_tracking(base_dir: Path, game: int) -> pd.DataFrame:
    home = load_metrica_tracking(base_dir, game=game, team="Home")
    away = load_metrica_tracking(base_dir, game=game, team="Away")

    if len(home) != len(away):
        raise ValueError(f"Home/Away tracking lengths differ: {len(home)} vs {len(away)}")

    key_cols = [c for c in ["Frame", "Time [s]", "Period"] if c in home.columns]
    away_extra = away.drop(
        columns=[c for c in away.columns if c in key_cols or c.startswith("ball_")],
        errors="ignore",
    )
    return pd.concat([home.reset_index(drop=True), away_extra.reset_index(drop=True)], axis=1)


def load_metrica_frames(
    base_dir: str | Path | None = None,
    game: int = 1,
    match_id: str | None = None,
) -> list[FrameSnapshot]:
    """Load Metrica home/away tracking into normalized frame snapshots."""
    base = _resolve_base(base_dir, METRICA_DIR)
    resolved_match_id = match_id or f"metrica_game_{game}"
    merged = _merge_metrica_tracking(base, game=game)
    return [
        metrica_row_to_frame(row.to_dict(), match_id=resolved_match_id)
        for _, row in merged.iterrows()
    ]


def iter_metrica_frames(
    base_dir: str | Path | None = None,
    game: int = 1,
    match_id: str | None = None,
) -> Iterator[FrameSnapshot]:
    base = _resolve_base(base_dir, METRICA_DIR)
    resolved_match_id = match_id or f"metrica_game_{game}"
    merged = _merge_metrica_tracking(base, game=game)
    for _, row in merged.iterrows():
        yield metrica_row_to_frame(row.to_dict(), match_id=resolved_match_id)


def load_skillcorner_matches(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    base = _resolve_base(base_dir, SKILLCORNER_DIR)
    return _read_json(_require_file(base / "matches.json"))


def _default_skillcorner_match_id(base: Path) -> str:
    matches = load_skillcorner_matches(base)
    if not matches:
        raise ValueError(f"No SkillCorner matches found in {base / 'matches.json'}")
    return str(matches[0]["id"])


def _skillcorner_match_dir(base: Path, match_id: str | int | None) -> tuple[str, Path]:
    resolved_match_id = str(match_id) if match_id is not None else _default_skillcorner_match_id(base)
    return resolved_match_id, base / "matches" / resolved_match_id


def load_skillcorner_match_metadata(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    base = _resolve_base(base_dir, SKILLCORNER_DIR)
    resolved_match_id, match_dir = _skillcorner_match_dir(base, match_id)
    path = match_dir / f"{resolved_match_id}_match.json"
    if not path.exists():
        candidates = sorted(match_dir.glob("*_match.json"))
        if not candidates:
            raise FileNotFoundError(path)
        path = candidates[0]
    return _read_json(path)


def _skillcorner_frame_context(metadata: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    player_team_by_id: dict[str, str] = {}
    player_role_by_id: dict[str, str] = {}

    for player in metadata.get("players", []):
        team_id = player.get("team_id")
        if team_id is None:
            continue

        ids = [player.get("id"), player.get("trackable_object")]
        for player_id in ids:
            if player_id is not None:
                player_team_by_id[str(player_id)] = str(team_id)

        role = player.get("player_role")
        if isinstance(role, dict):
            role_name = role.get("acronym") or role.get("name")
            if role_name is not None:
                for player_id in ids:
                    if player_id is not None:
                        player_role_by_id[str(player_id)] = str(role_name)

    possession_group_to_team: dict[str, str] = {}
    home_team = metadata.get("home_team")
    if isinstance(home_team, dict) and home_team.get("id") is not None:
        possession_group_to_team["home team"] = str(home_team["id"])
    away_team = metadata.get("away_team")
    if isinstance(away_team, dict) and away_team.get("id") is not None:
        possession_group_to_team["away team"] = str(away_team["id"])

    return player_team_by_id, player_role_by_id, possession_group_to_team


def _skillcorner_tracking_path(match_dir: Path, match_id: str) -> Path:
    preferred = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    if preferred.exists():
        return preferred

    candidates = sorted(match_dir.glob("*tracking*.jsonl"))
    if not candidates:
        raise FileNotFoundError(preferred)
    return candidates[0]


def iter_skillcorner_frames(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> Iterator[FrameSnapshot]:
    base = _resolve_base(base_dir, SKILLCORNER_DIR)
    resolved_match_id, match_dir = _skillcorner_match_dir(base, match_id)
    path = _skillcorner_tracking_path(match_dir, resolved_match_id)
    metadata = load_skillcorner_match_metadata(match_id=resolved_match_id, base_dir=base)
    player_team_by_id, player_role_by_id, possession_group_to_team = _skillcorner_frame_context(metadata)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield skillcorner_frame_to_snapshot(
                    json.loads(line),
                    match_id=resolved_match_id,
                    player_team_by_id=player_team_by_id,
                    player_role_by_id=player_role_by_id,
                    possession_group_to_team=possession_group_to_team,
                )


def load_skillcorner_frames(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> list[FrameSnapshot]:
    return list(iter_skillcorner_frames(match_id=match_id, base_dir=base_dir))


def _skillcorner_dynamic_events_path(match_dir: Path, match_id: str) -> Path:
    preferred = match_dir / f"{match_id}_dynamic_events.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(match_dir.glob("*dynamic_events.csv"))
    if not candidates:
        raise FileNotFoundError(preferred)
    return candidates[0]


def iter_skillcorner_events(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> Iterator[EventRecord]:
    base = _resolve_base(base_dir, SKILLCORNER_DIR)
    resolved_match_id, match_dir = _skillcorner_match_dir(base, match_id)
    path = _skillcorner_dynamic_events_path(match_dir, resolved_match_id)
    events = pd.read_csv(path, low_memory=False)
    for row in events.to_dict(orient="records"):
        yield skillcorner_event_to_record(row, match_id=resolved_match_id)


def load_skillcorner_events(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> list[EventRecord]:
    return list(iter_skillcorner_events(match_id=match_id, base_dir=base_dir))


def load_skillcorner_phases_of_play(
    match_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    base = _resolve_base(base_dir, SKILLCORNER_DIR)
    resolved_match_id, match_dir = _skillcorner_match_dir(base, match_id)
    preferred = match_dir / f"{resolved_match_id}_phases_of_play.csv"
    if preferred.exists():
        return pd.read_csv(preferred)

    candidates = sorted(match_dir.glob("*phases_of_play.csv"))
    if not candidates:
        raise FileNotFoundError(preferred)
    return pd.read_csv(candidates[0])


def load_statsbomb_competitions(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    base = _resolve_base(base_dir, STATSBOMB_DIR)
    return _read_json(_require_file(base / "competitions.json"))


def load_statsbomb_matches(
    competition_id: str | int | None = None,
    season_id: str | int | None = None,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    base = _resolve_base(base_dir, STATSBOMB_DIR)
    matches_dir = _require_file(base / "matches")

    if competition_id is not None and season_id is not None:
        return _read_json(_require_file(matches_dir / str(competition_id) / f"{season_id}.json"))

    match_records: list[dict[str, Any]] = []
    for path in sorted(matches_dir.glob("*/*.json")):
        match_records.extend(_read_json(path))
    return match_records


def iter_statsbomb_events(
    match_id: str | int,
    base_dir: str | Path | None = None,
) -> Iterator[EventRecord]:
    base = _resolve_base(base_dir, STATSBOMB_DIR)
    path = _require_file(base / "events" / f"{match_id}.json")
    for event in _read_json(path):
        yield statsbomb_event_to_record(event, match_id=str(match_id))


def load_statsbomb_events(
    match_id: str | int,
    base_dir: str | Path | None = None,
) -> list[EventRecord]:
    return list(iter_statsbomb_events(match_id=match_id, base_dir=base_dir))


def load_statsbomb_lineups(
    match_id: str | int,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    base = _resolve_base(base_dir, STATSBOMB_DIR)
    return _read_json(_require_file(base / "lineups" / f"{match_id}.json"))


def load_statsbomb_freeze_frames(
    match_id: str | int,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    base = _resolve_base(base_dir, STATSBOMB_DIR)
    return _read_json(_require_file(base / "three-sixty" / f"{match_id}.json"))
