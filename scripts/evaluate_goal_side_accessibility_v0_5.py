#!/usr/bin/env python3
"""Evaluate scene 1 with delivery × goal danger × goal-side accessibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_causal_attribution_v0_4 import _evaluate_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "data/processed/direct_derived_response_map_v0_1/"
            "direct_derived_response_maps.json"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/bundesliga-integrated")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/goal_side_accessibility_v0_5"),
    )
    parser.add_argument("--scene-index", type=int, default=1)
    parser.add_argument("--relevant-defender-count", type=int, default=3)
    parser.add_argument("--release-step-seconds", type=float, default=0.2)
    parser.add_argument("--control-distance-m", type=float, default=1.5)
    parser.add_argument("--minimum-relative-effect", type=float, default=0.05)
    parser.add_argument("--influence-grid-resolution-m", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.value_contract = "goal_side_accessibility"
    args.schema_version = "goal-side-accessibility-v0.5"
    scenes = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not 1 <= args.scene_index <= len(scenes):
        raise ValueError("scene-index is outside the available scene list")
    scene = scenes[args.scene_index - 1]
    print(
        f"scene {args.scene_index}: {scene['runner_name']} @ "
        f"{scene['match_id']}:{scene['onset_frame_id']}",
        flush=True,
    )
    result = _evaluate_scene(scene, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "goal_side_accessibility_v0_5.json"
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "goal-side-accessibility-v0.5",
        "scene_count": 1,
        "status": "scene-1 human-audit diagnostic",
        "value_contract": (
            "delivery_probability × endpoint_goal_danger × "
            "post_reception_goal_side_accessibility"
        ),
        "team_response": "lexicographic min(max option Q, sum option Q, effort)",
        "influence_grid_resolution_m": args.influence_grid_resolution_m,
        "minimum_relative_effect": args.minimum_relative_effect,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
