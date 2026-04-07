from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def find_metrica_game_dir(base_dir: Path, game: int = 1) -> Path:
    candidate = base_dir / f"Sample_Game_{game}"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find {candidate}")


def load_metrica_events(base_dir: str | Path, game: int = 1) -> pd.DataFrame:
    game_dir = find_metrica_game_dir(Path(base_dir), game)
    candidates = list(game_dir.glob("*EventsData*.csv")) + list(game_dir.glob("*RawEventsData*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No events CSV found under {game_dir}")
    return pd.read_csv(candidates[0])


def load_metrica_tracking(base_dir: str | Path, game: int = 1, team: str = "Home") -> pd.DataFrame:
    game_dir = find_metrica_game_dir(Path(base_dir), game)
    candidates = list(game_dir.glob(f"*TrackingData*{team}*.csv")) + list(game_dir.glob(f"*{team}*TrackingData*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {team} tracking CSV found under {game_dir}")

    path = candidates[0]

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        second = next(reader)
        third = next(reader)

    # Metrica raw tracking format:
    # row 1: team name section markers
    # row 2: jersey numbers
    # row 3: base columns (Frame, Time [s], Period, x, y, x, y, ..., ball_x, ball_y)
    jerseys = [x.strip() for x in second if x.strip() != ""]
    columns = third[:]

    player_columns = columns[3:-2]
    for i, jersey in enumerate(jerseys):
        if 2 * i + 1 >= len(player_columns):
            break
        player_columns[2 * i] = f"{team}_{jersey}_x"
        player_columns[2 * i + 1] = f"{team}_{jersey}_y"

    columns = columns[:3] + player_columns + ["ball_x", "ball_y"]

    data = pd.read_csv(path, names=columns, skiprows=3)

    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["Frame"]).reset_index(drop=True)

    if "Frame" in data.columns:
        data["Frame"] = data["Frame"].round().astype("Int64")
    if "Period" in data.columns:
        data["Period"] = pd.to_numeric(data["Period"], errors="coerce").round().astype("Int64")

    return data


def load_statsbomb_events(events_json: str | Path) -> pd.DataFrame:
    return pd.read_json(events_json)


def load_statsbomb_three_sixty(three_sixty_json: str | Path) -> pd.DataFrame:
    return pd.read_json(three_sixty_json)
