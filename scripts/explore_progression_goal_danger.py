"""Prototype: data-driven progression goal danger vs the geometric G proxy.

Motivated by the 53833-B depth-flatness finding (standing at onset priced
above entering vacated space) and the Fernandez & Bornn (SSAC 2018)
discussion: their pitch value is fitted to defender occupancy and then
hand-normalised to recover a goal-ward gradient, so instead we estimate
progression value directly from outcomes:

    G_data(z) ~ P(this possession ends in a goal | on-ball event at z)

over StatsBomb open-data events.  This is the quantity our G proxies: how
dangerous it is for the attack to have the ball at z, shot-or-continue
averaged.  Output: a binned surface (with counts), a smoothed version, and
a comparison of the depth gradient against score_at_points along central
approach rays.

Caveats established by the v0.3.8 audit, to keep in mind when quoting it:
the estimate is conditional on the AVERAGE REAL DEFENCE at z (it is not a
defence-free quantity); a goal-scoring shot labels its own location, which
inflates the near-goal end (removing it moves the raw 25->17 m ratio
2.005 -> 1.596); and the coarse 30x20 histogram overstates the gradient
relative to the interpolated grid the pipeline consumes (1.713).

Usage:
    python scripts/explore_progression_goal_danger.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.obso import score_at_points

EVENTS_DIR = Path("data/raw/statsbomb-open-data/data/events")
OUT_DIR = Path("data/processed/progression_goal_danger_v0")

# StatsBomb pitch: 120 x 80 units, attack always toward x=120.
GRID_X, GRID_Y = 30, 20  # 4-unit bins
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0


def _possession_goal_events(events: list[dict]) -> list[tuple[float, float, int]]:
    """(x, y, possession-ends-in-goal) for every located on-ball event."""

    goal_possessions: set[tuple[int, str]] = set()
    for event in events:
        if event.get("type", {}).get("name") == "Shot" and event.get("shot", {}).get(
            "outcome", {}
        ).get("name") == "Goal":
            team = event.get("possession_team", {}).get("name", "")
            goal_possessions.add((int(event.get("possession", -1)), str(team)))
    rows: list[tuple[float, float, int]] = []
    for event in events:
        location = event.get("location")
        if not location or len(location) < 2:
            continue
        team = event.get("team", {}).get("name", "")
        possession_team = event.get("possession_team", {}).get("name", "")
        if team != possession_team:
            continue
        if event.get("type", {}).get("name") == "Shot" and event.get("shot", {}).get(
            "type", {}
        ).get("name") == "Penalty":
            continue
        key = (int(event.get("possession", -1)), str(possession_team))
        rows.append((float(location[0]), float(location[1]), int(key in goal_possessions)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="match-file cap (0 = all)")
    args = parser.parse_args()

    files = sorted(EVENTS_DIR.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    counts = np.zeros((GRID_X, GRID_Y), dtype=np.int64)
    goals = np.zeros((GRID_X, GRID_Y), dtype=np.int64)
    used = 0
    for index, path in enumerate(files):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for x, y, label in _possession_goal_events(events):
            i = min(GRID_X - 1, max(0, int(x / (120.0 / GRID_X))))
            j = min(GRID_Y - 1, max(0, int(y / (80.0 / GRID_Y))))
            counts[i, j] += 1
            goals[i, j] += label
        used += 1
        if index % 400 == 0:
            print(f"  parsed {index}/{len(files)} files", flush=True)

    rate = np.divide(goals, np.maximum(counts, 1), dtype=float)
    # Light neighbour smoothing to tame sparse bins.
    padded = np.pad(rate, 1, mode="edge")
    weight = np.pad(np.minimum(counts, 2000), 1, mode="edge").astype(float)
    smooth = np.zeros_like(rate)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            k = 2.0 if (di == 0 and dj == 0) else 1.0
            smooth += k * weight[1 + di : GRID_X + 1 + di, 1 + dj : GRID_Y + 1 + dj] * (
                padded[1 + di : GRID_X + 1 + di, 1 + dj : GRID_Y + 1 + dj]
            )
    norm = np.zeros_like(rate)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            k = 2.0 if (di == 0 and dj == 0) else 1.0
            norm += k * weight[1 + di : GRID_X + 1 + di, 1 + dj : GRID_Y + 1 + dj]
    smooth = np.divide(smooth, np.maximum(norm, 1e-9))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "surface.npz",
        counts=counts,
        goals=goals,
        rate=rate,
        smooth=smooth,
        grid_x=GRID_X,
        grid_y=GRID_Y,
    )
    print(f"parsed {used} matches; events binned: {int(counts.sum())}")

    # Depth-gradient comparison along the central approach, in metres from
    # goal, against the geometric proxy used by the pipeline today.
    def data_g(metres_from_goal: float) -> float:
        x_units = 120.0 - metres_from_goal * (120.0 / PITCH_LENGTH_M)
        i = min(GRID_X - 1, max(0, int(x_units / (120.0 / GRID_X))))
        j = GRID_Y // 2
        return float(smooth[i, j])

    def proxy_g(metres_from_goal: float) -> float:
        x = PITCH_LENGTH_M / 2.0 - metres_from_goal
        return float(score_at_points([(x, 0.0)], 1, epv_grid_path=None)[0])

    marks = [35.0, 30.0, 25.0, 20.0, 15.0, 10.0]
    print("\nmetres_from_goal  data_G      proxy_G     (central lane)")
    for m in marks:
        print(f"{m:>14.0f}  {data_g(m):.5f}    {proxy_g(m):.5f}")
    for a, b in ((30.0, 22.0), (28.0, 20.0), (25.0, 17.0)):
        ratio_data = data_g(b) / max(data_g(a), 1e-9)
        ratio_proxy = proxy_g(b) / max(proxy_g(a), 1e-9)
        print(
            f"gradient {a:.0f}m -> {b:.0f}m: data x{ratio_data:.2f} vs proxy x{ratio_proxy:.2f}"
        )


if __name__ == "__main__":
    main()
