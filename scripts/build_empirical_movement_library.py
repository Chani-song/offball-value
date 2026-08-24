#!/usr/bin/env python3
"""Build a leave-one-match-out library of observed two-second movements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.empirical_action_space import (
    EmpiricalPrimitiveConfig,
    EmpiricalPrimitiveLibrary,
    extract_match_primitives,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--target-match-id", default="J03WMX")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/empirical_movement_library_v0_1"),
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_match_id = normalize_bundesliga_match_id(args.target_match_id)
    source_match_ids = [
        match_id
        for match_id in list_bundesliga_match_ids(args.data_dir)
        if match_id != target_match_id
    ]
    if not source_match_ids:
        raise ValueError("No source matches remain after leave-one-match-out filtering")

    config = EmpiricalPrimitiveConfig(
        sample_interval_seconds=args.sample_interval_seconds
    )
    chunks = []
    for index, match_id in enumerate(source_match_ids, start=1):
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        chunk = extract_match_primitives(files["positions"], metadata, config)
        chunks.append(chunk)
        print(
            f"[{index}/{len(source_match_ids)}] {match_id}: "
            f"{len(chunk):,} primitives",
            flush=True,
        )

    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    library = EmpiricalPrimitiveLibrary(frame, config)
    library.assert_excludes_match(target_match_id)

    output_dir = args.output_dir / f"leave_out_{target_match_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / "primitives.npz"
    summary_path = output_dir / "primitive_summary.csv"
    manifest_path = output_dir / "manifest.json"
    library.save(library_path)

    summary = (
        frame.groupby("match_id", as_index=False)
        .agg(
            primitive_count=("frame_id", "size"),
            player_count=("player_id", "nunique"),
            median_initial_speed_mps=("initial_speed_mps", "median"),
            median_terminal_speed_mps=("terminal_speed_mps", "median"),
        )
        .sort_values("match_id")
    )
    summary.to_csv(summary_path, index=False)
    manifest = {
        "target_match_id": target_match_id,
        "source_match_ids": list(library.source_match_ids),
        "primitive_count": len(frame),
        "player_count": int(frame["player_id"].nunique()),
        "config": config.__dict__,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"total primitives: {len(frame):,}")
    print(f"source matches:   {len(library.source_match_ids)}")
    print(f"library:          {library_path}")
    print(f"summary:          {summary_path}")
    print(f"manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
