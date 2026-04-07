from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offball_value.baseline import PlayerState, compute_toy_option_score, detect_candidate_run
from offball_value.loaders import load_metrica_tracking


DATA_DIR = ROOT / "data" / "raw" / "metrica-sample-data" / "data"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_player_ids(df: pd.DataFrame, prefix: str) -> List[str]:
    ids = []
    for col in df.columns:
        if col.startswith(prefix) and col.endswith("_x"):
            pid = col[:-2]
            if pid.lower().endswith("ball"):
                continue
            ids.append(pid)
    return sorted(ids)



def get_state(row: pd.Series, player_id: str) -> PlayerState | None:
    x_col = f"{player_id}_x"
    y_col = f"{player_id}_y"
    if x_col not in row or y_col not in row:
        return None
    x = row[x_col]
    y = row[y_col]
    if pd.isna(x) or pd.isna(y):
        return None
    return PlayerState(player_id=player_id, x=float(x), y=float(y))



def build_team_states(row: pd.Series, player_ids: List[str]) -> List[PlayerState]:
    states = []
    for pid in player_ids:
        s = get_state(row, pid)
        if s is not None:
            states.append(s)
    return states



def best_teammate_option(teammates: List[PlayerState], defenders: List[PlayerState], excluded_id: str | None = None):
    candidates = []
    for tm in teammates:
        if excluded_id is not None and tm.player_id == excluded_id:
            continue
        score = compute_toy_option_score(tm, defenders, attacking_direction=1)
        candidates.append((tm.player_id, score))
    if not candidates:
        return None, np.nan
    return max(candidates, key=lambda x: x[1])



def move_defender_toward_start(defenders_start: Dict[str, PlayerState], defenders_end: List[PlayerState], runner_start: PlayerState, runner_end: PlayerState) -> List[PlayerState]:
    if not defenders_end:
        return defenders_end

    # pick the defender closest to the runner at the end frame as a crude "attracted" defender
    closest = min(defenders_end, key=lambda d: (d.x - runner_end.x) ** 2 + (d.y - runner_end.y) ** 2)
    cf_defenders = deepcopy(defenders_end)
    for d in cf_defenders:
        if d.player_id == closest.player_id and d.player_id in defenders_start:
            d0 = defenders_start[d.player_id]
            # move partway back toward the start position, as a toy no-run counterfactual
            d.x = 0.7 * d0.x + 0.3 * d.x
            d.y = 0.7 * d0.y + 0.3 * d.y
    return cf_defenders



def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} does not exist. Run `python scripts/download_metrica.py` first."
        )

    home = load_metrica_tracking(DATA_DIR, game=1, team="Home")
    away = load_metrica_tracking(DATA_DIR, game=1, team="Away")

    key_cols = [c for c in ["Frame", "Time [s]", "Period"] if c in home.columns and c in away.columns]
    merged = home.merge(away, on=key_cols, suffixes=("", ""))

    home_ids = find_player_ids(merged, "Home_")
    away_ids = find_player_ids(merged, "Away_")

    horizon = 25  # about 1 second in 25 Hz data
    results = []

    for idx in range(0, len(merged) - horizon, horizon):
        row0 = merged.iloc[idx]
        row1 = merged.iloc[idx + horizon]

        home_start = {pid: get_state(row0, pid) for pid in home_ids}
        home_end = {pid: get_state(row1, pid) for pid in home_ids}
        away_start = {pid: get_state(row0, pid) for pid in away_ids}
        away_end = {pid: get_state(row1, pid) for pid in away_ids}

        defenders_start = {k: v for k, v in away_start.items() if v is not None}
        defenders_end = [v for v in away_end.values() if v is not None]
        teammates_end = [v for v in home_end.values() if v is not None]

        if len(defenders_end) < 3 or len(teammates_end) < 3:
            continue

        for pid in home_ids:
            s0 = home_start.get(pid)
            s1 = home_end.get(pid)
            if s0 is None or s1 is None:
                continue
            if not detect_candidate_run(s0.x, s0.y, s1.x, s1.y, min_distance=0.04):
                continue

            best_actual_id, best_actual_score = best_teammate_option(teammates_end, defenders_end, excluded_id=pid)
            if pd.isna(best_actual_score):
                continue

            cf_defenders = move_defender_toward_start(defenders_start, defenders_end, s0, s1)
            best_cf_id, best_cf_score = best_teammate_option(teammates_end, cf_defenders, excluded_id=pid)
            if pd.isna(best_cf_score):
                continue

            delta = float(best_actual_score - best_cf_score)
            movement = float(np.hypot(s1.x - s0.x, s1.y - s0.y))

            results.append(
                {
                    "frame_start": int(row0["Frame"]),
                    "frame_end": int(row1["Frame"]),
                    "time_start_s": float(row0["Time [s]"]),
                    "time_end_s": float(row1["Time [s]"]),
                    "runner_id": pid,
                    "runner_start_x": s0.x,
                    "runner_start_y": s0.y,
                    "runner_end_x": s1.x,
                    "runner_end_y": s1.y,
                    "runner_movement": movement,
                    "best_actual_receiver": best_actual_id,
                    "best_actual_score": best_actual_score,
                    "best_counterfactual_receiver": best_cf_id,
                    "best_counterfactual_score": best_cf_score,
                    "draft_offball_value": delta,
                }
            )

    out = pd.DataFrame(results)
    if out.empty:
        print("No candidate runs found with current thresholds.")
        return

    out = out.sort_values("draft_offball_value", ascending=False)
    out.to_csv(OUT_DIR / "metrica_draft_offball_value.csv", index=False)
    print(out.head(15).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {OUT_DIR / 'metrica_draft_offball_value.csv'}")


if __name__ == "__main__":
    main()
