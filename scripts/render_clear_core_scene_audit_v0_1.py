#!/usr/bin/env python3
"""Build the clear-only actual-motion scene audit from prior human reviews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from offball_value.clear_core_scene_audit import (
    render_clear_core_scene_audit,
    select_clear_core_scenes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "data/processed/shot_context_run_onset_v0_1/shot_context_onset_audit.json"
        ),
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path(
            "examples/research_audit/human_reviews/shot_context/"
            "shot_context_onset_v0_1_reviews.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/clear_core_scene_audit_v0_1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.audit_json.read_text(encoding="utf-8"))
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    selected = select_clear_core_scenes(scenes, reviews)
    if not selected:
        raise SystemExit("No correct + settled + clear scenes found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "clear_core_scenes.json"
    html_path = args.output_dir / "clear_core_scene_audit.html"
    manifest_path = args.output_dir / "manifest.json"
    json_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_clear_core_scene_audit(selected), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "clear-core-local-game-v0.1",
                "status": "human-confirmation-candidates-not-final-labels",
                "scene_count": len(selected),
                "selection": {
                    "onset_review": "correct",
                    "possession_review": "settled",
                    "interaction_review": "clear",
                },
                "source_audit": str(args.audit_json),
                "source_review": str(args.review_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"scenes: {len(selected)}")
    print(f"html: {html_path.resolve()}")


if __name__ == "__main__":
    main()
