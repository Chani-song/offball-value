#!/usr/bin/env python3
"""Why does DFL-MAT-J03WN1 yield 2 contextual run onsets where peers yield 52-85?

Phase segmentation is healthy for that match (101 phases, normal durations,
attacking-half fractions and control fractions) and the tracking carries 21
players per frame, which is right for a match played 11v10. So the loss is
downstream of segmentation. This splits it: how many onsets detect_run_onsets
returns, and how many _nearest_phase then discards.

Read-only diagnostic. Writes nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.run_onset import RunOnsetConfig, detect_run_onsets
from offball_value.settled_possession_phase import (
    SettledPossessionPhaseConfig,
    segment_settled_possession_phases,
)

spec = importlib.util.spec_from_file_location(
    "ex", "scripts/extract_settled_possession_run_onsets.py"
)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

DATA = Path("data/raw/bundesliga-integrated")


def run(match_id: str) -> None:
    print(f"\n{'=' * 60}\n{match_id}\n{'=' * 60}", flush=True)
    files = find_bundesliga_files(DATA, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    clock = load_bundesliga_frame_clock(files["positions"])
    events = load_bundesliga_events(files["events"], clock)

    cfg = SettledPossessionPhaseConfig(sample_stride_frames=5, minimum_duration_seconds=6.0)
    sample_ids = range(0, ex.MAX_SEGMENTATION_FRAME_ID, 5)
    sampled = load_bundesliga_frames(files["positions"], sample_ids)
    phases = segment_settled_possession_phases(sampled, events, metadata, cfg)
    del sampled
    phase_frame = pd.DataFrame([p.as_record() for p in phases])
    print(f"phases: {len(phase_frame)}")

    ranges = ex._prepare_phase_ranges(phase_frame)
    target_ids, search_ids = ex._target_and_search_ids(ranges)
    print(f"target frames: {len(target_ids)}   search frames: {len(search_ids)}")

    frames = load_bundesliga_frames(files["positions"], target_ids)
    print(f"loaded frames: {len(frames)}")
    missing = [f for f in list(search_ids)[:2000] if f not in frames]
    print(f"search frames absent from loaded set (first 2000): {len(missing)}")

    detected = detect_run_onsets(
        frames, metadata, search_frame_ids=search_ids, config=RunOnsetConfig()
    )
    print(f"detect_run_onsets -> {len(detected)}")

    kept = sum(1 for c in detected if ex._nearest_phase(c, ranges) is not None)
    print(f"_nearest_phase kept -> {kept}   discarded -> {len(detected) - kept}")

    if detected:
        rec = pd.DataFrame([c.as_record() for c in detected])
        for col in ("onset_speed_mps", "speed_gain_mps", "post_displacement_m",
                    "direction_change_degrees", "confidence_score", "ball_distance_m"):
            if col in rec:
                print(f"  {col:26} 중앙값 {rec[col].median():.3f}")


def main() -> None:
    for match_id in ("DFL-MAT-J03WN1", "DFL-MAT-J03WMX"):
        run(match_id)


if __name__ == "__main__":
    main()
