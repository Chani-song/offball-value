from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def safe_read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_metrica() -> None:
    print_header("METRICA")

    base = RAW_DIR / "metrica-sample-data" / "data" / "Sample_Game_1"
    if not base.exists():
        print(f"[missing] {base}")
        return

    files = sorted(p.name for p in base.iterdir())
    print("files:")
    for name in files:
        print(" -", name)

    home_candidates = list(base.glob("*TrackingData*Home*.csv")) + list(base.glob("*Home*TrackingData*.csv"))
    away_candidates = list(base.glob("*TrackingData*Away*.csv")) + list(base.glob("*Away*TrackingData*.csv"))
    event_candidates = list(base.glob("*EventsData*.csv")) + list(base.glob("*RawEventsData*.csv"))

    print("home candidates:", [p.name for p in home_candidates])
    print("away candidates:", [p.name for p in away_candidates])
    print("event candidates:", [p.name for p in event_candidates])

    if not home_candidates:
        print("[warn] no home tracking file found")
        return

    home_csv = home_candidates[0]
    print(f"\npreview: {home_csv.name}")
    with open(home_csv, "r", encoding="utf-8") as f:
        for i in range(5):
            line = f.readline().rstrip("\n")
            print(f"[line {i+1}] {line}")

    if event_candidates:
        event_csv = event_candidates[0]
        print(f"\npreview event header: {event_csv.name}")
        with open(event_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                print("event header:", header[:20], "..." if len(header) > 20 else "")
            except StopIteration:
                print("[warn] empty event csv")


def inspect_skillcorner() -> None:
    print_header("SKILLCORNER")

    base = RAW_DIR / "skillcorner-opendata" / "data"
    if not base.exists():
        print(f"[missing] {base}")
        return

    matches_path = base / "matches.json"
    print("matches.json exists:", matches_path.exists())
    if not matches_path.exists():
        return

    matches = safe_read_json(matches_path)
    print("num matches:", len(matches))
    if not matches:
        return

    first_match = matches[0]
    print("first match keys:", list(first_match.keys()))
    first_id = first_match["id"]
    print("first match id:", first_id)

    match_dir = base / "matches" / str(first_id)
    if not match_dir.exists():
        print(f"[missing] {match_dir}")
        return

    print("match dir files:")
    for p in sorted(match_dir.iterdir()):
        print(" -", p.name)

    tracking_file = match_dir / f"{first_id}_tracking_extrapolated.jsonl"
    if tracking_file.exists():
        with open(tracking_file, "r", encoding="utf-8") as f:
            first_frame = json.loads(f.readline())
        print("\ntracking keys:", list(first_frame.keys()))
        if isinstance(first_frame.get("ball_data"), dict):
            print("ball keys:", list(first_frame["ball_data"].keys()))
        if first_frame.get("player_data"):
            sample_player = first_frame["player_data"][0]
            print("player sample keys:", list(sample_player.keys()))
    else:
        print("[warn] tracking file missing")

    dyn_file = match_dir / f"{first_id}_dynamic_events.csv"
    if dyn_file.exists():
        with open(dyn_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        print("\ndynamic_events header (first 40 cols):")
        print(header[:40], "..." if len(header) > 40 else "")
    else:
        print("[warn] dynamic_events.csv missing")

    phase_file = match_dir / f"{first_id}_phases_of_play.csv"
    if phase_file.exists():
        with open(phase_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        print("\nphases_of_play header:")
        print(header)
    else:
        print("[warn] phases_of_play.csv missing")


def inspect_statsbomb() -> None:
    print_header("STATSBOMB")

    base = RAW_DIR / "statsbomb-open-data" / "data"
    if not base.exists():
        print(f"[missing] {base}")
        return

    print("top-level:", [p.name for p in sorted(base.iterdir())])

    competitions_path = base / "competitions.json"
    if competitions_path.exists():
        competitions = safe_read_json(competitions_path)
        print("num competitions:", len(competitions))
        if competitions:
            print("competition sample keys:", list(competitions[0].keys()))
    else:
        print("[warn] competitions.json missing")

    events_dir = base / "events"
    if events_dir.exists():
        event_files = sorted(events_dir.glob("*.json"))
        print("num event files:", len(event_files))
        if event_files:
            event_file = event_files[0]
            events = safe_read_json(event_file)
            print("sample event file:", event_file.name)
            print("num events in file:", len(events))
            if events:
                print("event sample keys:", list(events[0].keys()))
    else:
        print("[warn] events dir missing")

    frames360_dir = base / "three-sixty"
    if frames360_dir.exists():
        files360 = sorted(frames360_dir.glob("*.json"))
        print("num 360 files:", len(files360))
        if files360:
            f360 = files360[0]
            frames = safe_read_json(f360)
            print("sample 360 file:", f360.name)
            print("num 360 rows:", len(frames))
            if frames:
                print("360 sample keys:", list(frames[0].keys()))
    else:
        print("[warn] three-sixty dir missing")


def main() -> None:
    inspect_metrica()
    inspect_skillcorner()
    inspect_statsbomb()


if __name__ == "__main__":
    sys.exit(main())