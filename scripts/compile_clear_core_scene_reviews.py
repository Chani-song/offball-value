#!/usr/bin/env python3
"""Compile second-pass human reviews into the confirmed core scene set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from offball_value.clear_core_scene_audit import attach_confirmed_core_reviews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path("data/processed/clear_core_scene_audit_v0_1/clear_core_scenes.json"),
    )
    parser.add_argument(
        "--reviews-csv",
        type=Path,
        default=Path(
            "examples/research_audit/human_reviews/clear_core/"
            "clear_core_scene_reviews_human.csv"
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
    scenes = json.loads(args.scenes_json.read_text(encoding="utf-8"))
    with args.reviews_csv.open(encoding="utf-8-sig", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    reviewed, confirmed = attach_confirmed_core_reviews(scenes, reviews)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = args.output_dir / "reviewed_core_candidates.json"
    confirmed_path = args.output_dir / "confirmed_core_scenes.json"
    normalized_review_path = args.output_dir / "clear_core_scene_reviews_human.csv"
    reviewed_path.write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    confirmed_path.write_text(
        json.dumps(confirmed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "match_id",
        "onset_frame_id",
        "runner_id",
        "runner_name",
        "core_decision",
        "defender_visible",
        "derived_visible",
        "tradeoff_visible",
        "primary_defender",
        "derived_option",
        "note",
    ]
    with normalized_review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for scene in reviewed:
            writer.writerow(
                {
                    "match_id": scene["match_id"],
                    "onset_frame_id": scene["onset_frame_id"],
                    "runner_id": scene["runner_id"],
                    "runner_name": scene["runner_name"],
                    **scene["core_review"],
                }
            )

    anchor_key = ("DFL-MAT-J03WOH", 53844, "DFL-OBJ-0000F8")
    anchor_found = any(
        (scene["match_id"], int(scene["onset_frame_id"]), scene["runner_id"])
        == anchor_key
        for scene in confirmed
    )
    manifest_path = args.output_dir / "confirmed_core_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "confirmed-clear-core-local-game-v0.1",
                "status": "human-confirmed-development-scenes",
                "reviewed_scene_count": len(reviewed),
                "confirmed_scene_count": len(confirmed),
                "held_scene_count": sum(
                    scene["core_review"]["core_decision"] == "hold"
                    for scene in reviewed
                ),
                "rejected_scene_count": sum(
                    scene["core_review"]["core_decision"] == "reject"
                    for scene in reviewed
                ),
                "anchor_scene": {
                    "match_id": anchor_key[0],
                    "onset_frame_id": anchor_key[1],
                    "runner_id": anchor_key[2],
                    "runner_name": "F. Klaus",
                    "present": anchor_found,
                },
                "generalization_scene_count": len(confirmed) - int(anchor_found),
                "source_reviews": str(args.reviews_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"reviewed: {len(reviewed)}")
    print(f"confirmed: {len(confirmed)}")
    print(f"held: {len(reviewed) - len(confirmed)}")
    print(f"confirmed json: {confirmed_path.resolve()}")


if __name__ == "__main__":
    main()
