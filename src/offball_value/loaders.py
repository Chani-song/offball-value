from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


METRICA_GAME_DIRS = [
    "Sample_Game_1",
    "Sample_Game_2",
    "Sample_Game_3",
]


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
    candidates = list(game_dir.glob(f"*{team}*TrackingData*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {team} tracking CSV found under {game_dir}")

    path = candidates[0]
    # Metrica tracking files often have a 3-line header layout.
    header1 = pd.read_csv(path, nrows=0).columns.tolist()
    if len(header1) > 5 and header1[0] == "Frame":
        return pd.read_csv(path)

    return pd.read_csv(path, skiprows=2)



def load_statsbomb_events(events_json: str | Path) -> pd.DataFrame:
    return pd.read_json(events_json)



def load_statsbomb_three_sixty(three_sixty_json: str | Path) -> pd.DataFrame:
    return pd.read_json(three_sixty_json)
