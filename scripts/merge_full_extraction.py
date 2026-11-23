#!/usr/bin/env python3
"""Merge the per-match run-onset extraction shards into one bundle + scene chunks.

The full extraction runs one Slurm task per match so the 7 matches load in
parallel rather than serially. Each task writes a complete v0_1-shaped bundle
for its own match; this concatenates them back into a single bundle and splits
the audit payload into fixed-size scene chunks that
render_local_game_payoff_audit_v0_1.py consumes via --scenes-json/--scene-index.

Chunk format is byte-identical in structure to v0_1's chunk{0,1,2}.json (a flat
list of scene payloads), so the renderer and every downstream scorer keep
working unchanged.

Usage:
    python scripts/merge_full_extraction.py \
        --root data/processed/settled_possession_run_onset_v0_2_full \
        --chunk-size 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CSV_FILES = (
    "attacking_phases.csv",
    "shot_context_run_onsets.csv",
    "audit_selection.csv",
    "match_summary.csv",
)
V0_2_FULL = "settled_possession_run_onset_v0_2_full"
# What the v0_2_full bundle changed; only true of that bundle, so it is written
# only under that version. Other bundles carry their real settings in
# per_match_manifests.
V0_2_FULL_NOTES = {
    "derived_from": "settled_possession_run_onset_v0_1",
    "changes_vs_v0_1": [
        "DFL-MAT-J03WN1 is INCLUDED (v0_1 excluded it). It is 11v10 for ~93% "
        "of the match after a ~7-minute red card, so it is outside the 11v11 "
        "population and MUST be filtered on match_id for any numeric "
        "comparison. It is kept because 13 of Chani's 88 labelled shot "
        "annotations (7 medium, 6 low, 0 strong) are in it.",
        "audit_per_match / audit_total caps lifted (v0_1 used 15 / 90). The "
        "90-scene pool was a labelling sample cap, not a model decision.",
        "Detection parameters are unchanged from v0_1 so scene definitions "
        "stay comparable with the existing 42-pair evaluation set.",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--version", default=V0_2_FULL,
                   help="version string for the merged manifest (default: the v0_2_full bundle)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shards = sorted(d for d in args.root.iterdir() if d.is_dir() and d.name.startswith("DFL-MAT-"))
    if not shards:
        raise SystemExit(f"no per-match shards under {args.root}")
    print(f"{len(shards)} shards: {[d.name for d in shards]}")

    missing = [d.name for d in shards if not (d / "audit_selection.csv").exists()]
    if missing:
        raise SystemExit(f"unfinished shards: {missing}")

    for name in CSV_FILES:
        frames = []
        for d in shards:
            path = d / name
            if not path.exists():
                print(f"  ! {d.name}/{name} missing, skipped")
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                # A match that yielded no audit rows writes a headerless empty
                # file (J03WN1: 101 phases but only 2 contextual onsets).
                print(f"  ! {d.name}/{name} empty (0 rows)")
                continue
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        sort_cols = [c for c in ("match_id", "frame_id") if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(args.root / name, index=False)
        print(f"{name}: {len(merged)} rows")

    payloads: list[dict] = []
    for d in shards:
        path = d / "shot_context_onset_audit.json"
        if not path.exists():
            continue
        payloads.extend(json.loads(path.read_text(encoding="utf-8")))
    payloads.sort(key=lambda s: (str(s.get("match_id")), int(s.get("onset_frame_id", 0))))
    (args.root / "shot_context_onset_audit.json").write_text(
        json.dumps(payloads, ensure_ascii=False), encoding="utf-8"
    )
    print(f"scene payloads: {len(payloads)}")

    size = args.chunk_size
    n_chunks = (len(payloads) + size - 1) // size
    for i in range(n_chunks):
        block = payloads[i * size : (i + 1) * size]
        (args.root / f"chunk{i}.json").write_text(
            json.dumps(block, ensure_ascii=False), encoding="utf-8"
        )
    print(f"{n_chunks} chunks (last has {len(payloads) - (n_chunks - 1) * size} scenes)")

    manifests = {}
    for d in shards:
        path = d / "manifest.json"
        if path.exists():
            manifests[d.name] = json.loads(path.read_text(encoding="utf-8"))
    (args.root / "manifest.json").write_text(
        json.dumps(
            {
                "version": args.version,
                **(V0_2_FULL_NOTES if args.version == V0_2_FULL else {}),
                "scene_count": len(payloads),
                "chunk_size": size,
                "chunk_count": n_chunks,
                "per_match_manifests": manifests,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(payloads)} scenes, {n_chunks} chunks -> {args.root}")
    print(f"render array size: 1-{len(payloads)}")


if __name__ == "__main__":
    main()
