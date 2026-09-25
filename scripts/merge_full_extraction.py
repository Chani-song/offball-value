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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shards = sorted(d for d in args.root.iterdir() if d.is_dir() and d.name.startswith("DFL-MAT-"))
    if not shards:
        raise SystemExit(f"no per-match shards under {args.root}")
    print(f"샤드 {len(shards)}개: {[d.name for d in shards]}")

    missing = [d.name for d in shards if not (d / "audit_selection.csv").exists()]
    if missing:
        raise SystemExit(f"미완료 샤드: {missing}")

    for name in CSV_FILES:
        frames = []
        for d in shards:
            path = d / name
            if not path.exists():
                print(f"  ! {d.name}/{name} 없음, 건너뜀")
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                # A match that yielded no audit rows writes a headerless empty
                # file (J03WN1: 101 phases but only 2 contextual onsets).
                print(f"  ! {d.name}/{name} 비어 있음 (0행)")
                continue
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        sort_cols = [c for c in ("match_id", "frame_id") if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(args.root / name, index=False)
        print(f"{name}: {len(merged)}행")

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
    print(f"장면 페이로드: {len(payloads)}개")

    size = args.chunk_size
    n_chunks = (len(payloads) + size - 1) // size
    for i in range(n_chunks):
        block = payloads[i * size : (i + 1) * size]
        (args.root / f"chunk{i}.json").write_text(
            json.dumps(block, ensure_ascii=False), encoding="utf-8"
        )
    print(f"청크 {n_chunks}개 (마지막 {len(payloads) - (n_chunks - 1) * size}장면)")

    manifests = {}
    for d in shards:
        path = d / "manifest.json"
        if path.exists():
            manifests[d.name] = json.loads(path.read_text(encoding="utf-8"))
    (args.root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "settled_possession_run_onset_v0_2_full",
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
    print(f"\n총 {len(payloads)}장면, 청크 {n_chunks}개 -> {args.root}")
    print(f"렌더 array 크기: 1-{len(payloads)}")


if __name__ == "__main__":
    main()
