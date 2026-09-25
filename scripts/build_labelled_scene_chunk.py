#!/usr/bin/env python3
"""Build scene payloads for labelled onsets the 90-scene pool left out.

Of 34 human-labelled scenes only 18 are in the built pool, so every comparison
today rested on 21 pairs -- a set where a one-answer difference is noise and
no change can be told from another. The gap is not an accident: the extraction
deliberately drops onsets that already carry a review
(`excluded_reviewed`) so the labelling pool keeps widening onto fresh scenes.
Good for collecting labels, bad for using them, and nine of the missing onsets
are sitting in the candidates CSV with that flag set.

They need nothing re-detected, only the tracking frames attached. This reads
those rows and rebuilds the same payload the extractor writes, so the scenes
join the evaluation set rather than sitting beside it.

Human review columns are NOT carried into the payload. The labels are the
evaluation; they do not go into anything the model reads.

Usage:
    python scripts/build_labelled_scene_chunk.py \
        --frames 49266 122417 ... --output data/processed/.../labelled_recovered.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)

CANDIDATES = Path(
    "data/processed/settled_possession_run_onset_v0_1/shot_context_run_onsets.csv"
)
PRE_SECONDS = 2.0
POST_SECONDS = 3.0
STRIDE = 2          # 12.5 Hz, matching the extractor's animation-fps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/bundesliga-integrated"))
    p.add_argument("--candidates", type=Path, default=CANDIDATES)
    p.add_argument("--frames", type=str, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.candidates)
    wanted = {str(f) for f in args.frames}
    rows = table[table["frame_id"].astype(str).isin(wanted)]
    print(f"후보 CSV {len(table):,}행 · 요청 {len(wanted)} · 찾음 {len(rows)}")
    missing = wanted - set(rows["frame_id"].astype(str))
    if missing:
        print(f"  CSV 에 없는 프레임: {sorted(missing)}")

    scenes = []
    for match_id, group in rows.groupby("match_id"):
        files = find_bundesliga_files(args.raw_dir, str(match_id))
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        del clock
        targets = set()
        for _, row in group.iterrows():
            onset = int(row["frame_id"])
            targets.update(range(onset - int(PRE_SECONDS * FPS),
                                 onset + int(POST_SECONDS * FPS) + 1, STRIDE))
        frames = load_bundesliga_frames(files["positions"], targets)

        for _, row in group.iterrows():
            onset = int(row["frame_id"])
            samples = []
            for fid in range(onset - int(PRE_SECONDS * FPS),
                             onset + int(POST_SECONDS * FPS) + 1, STRIDE):
                frame = frames.get(fid)
                if frame is None:
                    continue
                players = [
                    [pid, str(p.team_id), round(float(p.x), 3), round(float(p.y), 3),
                     (metadata.players[pid].short_name
                      if pid in metadata.players else pid)]
                    for pid, p in frame.players.items()
                ]
                samples.append({
                    "frame_id": fid,
                    "relative_time_s": round((fid - onset) / FPS, 3),
                    "players": players,
                    "ball": ([round(float(frame.ball.x), 3),
                              round(float(frame.ball.y), 3)]
                             if frame.ball is not None else None),
                })
            if len(samples) < 10:
                print(f"  [skip] {onset}: 프레임 {len(samples)}개뿐")
                continue
            runner = metadata.players.get(str(row["player_id"]))
            carrier = metadata.players.get(str(row["ball_carrier_id"]))
            scenes.append({
                "match_id": str(row["match_id"]),
                "match_label": (f"{metadata.home_team_name} vs {metadata.away_team_name}"
                                " · settled-possession pool (라벨 회수분)"),
                "onset_frame_id": onset,
                "runner_id": str(row["player_id"]),
                "runner_name": runner.short_name if runner else str(row["player_id"]),
                "carrier_id": str(row["ball_carrier_id"]),
                "carrier_name": carrier.short_name if carrier else str(row["ball_carrier_id"]),
                "team_id": str(row["team_id"]),
                "attacking_direction": int(row["attacking_direction"]),
                "onset_labels": str(row["onset_labels"]).split("|"),
                "onset_speed_mps": float(row["onset_speed_mps"]),
                "pre_speed_mps": float(row["pre_mean_speed_mps"]),
                "post_speed_mps": float(row["post_mean_speed_mps"]),
                "settled_team_control_fraction": float(row["settled_team_control_fraction"]),
                "settled_same_team_known_fraction": float(row["settled_same_team_known_fraction"]),
                "ball_progress_m": float(row["ball_progress_m"]),
                "seconds_before_shot": float(row["seconds_before_shot"]),
                "shot_frame_id": int(row["shot_frame_id"]),
                "phase_human_review": str(row["phase_human_review"]),
                "frames": samples,
            })
            print(f"  [{match_id}] {onset} · 프레임 {len(samples)} · runner "
                  f"{scenes[-1]['runner_name']}", flush=True)
        del frames

    scenes.sort(key=lambda s: (s["match_id"], s["onset_frame_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenes, ensure_ascii=False))
    print(f"\n저장: {args.output} · 장면 {len(scenes)}개 · "
          f"{args.output.stat().st_size/1048576:.1f} MB")


if __name__ == "__main__":
    main()
