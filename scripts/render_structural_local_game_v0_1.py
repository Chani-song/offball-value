#!/usr/bin/env python3
"""Build defender-by-option structural matrices for the confirmed core set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from offball_value.local_game_structure import build_structural_local_games
from offball_value.local_game_structure_audit import (
    render_structural_local_game_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes-json",
        type=Path,
        default=Path(
            "examples/research_audit/manifests/confirmed_core_scenes.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/structural_local_game_v0_1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.scenes_json.read_text(encoding="utf-8"))
    games = build_structural_local_games(scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "structural_local_games.json"
    html_path = args.output_dir / "structural_local_game_audit.html"
    summary_path = args.output_dir / "structural_local_game_summary.csv"
    json_path.write_text(
        json.dumps(games, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_structural_local_game_audit(games), encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "match_id",
                "onset_frame_id",
                "runner_name",
                "defender_name",
                "option_name",
                "option_allocation_effect_fraction",
                "runner_cross_cost_fraction",
                "structural_tradeoff_score",
            ],
        )
        writer.writeheader()
        for game in games:
            displayed = set(game["displayed_option_ids"])
            for defender in game["candidate_defenders"]:
                for cell in defender["cells"]:
                    if cell["option_id"] not in displayed:
                        continue
                    writer.writerow(
                        {
                            "match_id": game["match_id"],
                            "onset_frame_id": game["onset_frame_id"],
                            "runner_name": game["runner_name"],
                            "defender_name": defender["defender_name"],
                            "option_name": cell["option_name"],
                            "option_allocation_effect_fraction": cell[
                                "option_allocation_effect_fraction"
                            ],
                            "runner_cross_cost_fraction": cell[
                                "runner_cross_cost_fraction"
                            ],
                            "structural_tradeoff_score": cell[
                                "structural_tradeoff_score"
                            ],
                        }
                    )
    print(f"json: {json_path.resolve()}")
    print(f"html: {html_path.resolve()}")
    print(f"summary: {summary_path.resolve()}")
    print(f"scenes: {len(games)}")


if __name__ == "__main__":
    main()
