#!/usr/bin/env python3
"""Smoke-test fixed-reference-defense OBSO on a few attacker actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from offball_value.action_space import (
    EndpointAction,
    EndpointActionConfig,
    PlayerEndpointActionSet,
    generate_scene_attacker_endpoint_sets,
)
from offball_value.action_value import evaluate_counterfactual_threat
from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.counterfactual_state import build_attacker_counterfactual_state
from offball_value.scene_extractor import SceneExtractorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/processed/scene_extractor_v0_1"),
    )
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fixed_defense_smoke_v0_1"),
    )
    return parser.parse_args()


def _selected_actions(
    action_set: PlayerEndpointActionSet,
    attacking_direction: int,
) -> list[EndpointAction]:
    feasible = list(action_set.optimization_actions)
    grid = [action for action in feasible if "grid" in action.labels]
    references = [
        action
        for action in feasible
        if "hold" in action.labels or "constant_velocity" in action.labels
    ]
    if not grid:
        return references
    selectors = [
        max(grid, key=lambda action: attacking_direction * action.endpoint_x),
        min(grid, key=lambda action: attacking_direction * action.endpoint_x),
        max(grid, key=lambda action: action.endpoint_y),
        min(grid, key=lambda action: action.endpoint_y),
        max(grid, key=lambda action: action.distance_from_start_m),
    ]
    selected = []
    seen = set()
    for action in [*references, *selectors]:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        selected.append(action)
    return selected


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scene_path = args.scene_dir / match_id / "scene_candidates.csv"
    scenes = pd.read_csv(scene_path)
    scenes = scenes[scenes["accepted"].astype(str).str.lower() == "true"]
    if args.frame_id is not None:
        scenes = scenes[scenes["frame_id"] == args.frame_id]
    if scenes.empty:
        raise ValueError("No matching accepted scene")
    row = scenes.sort_values("frame_id").iloc[0]
    frame_id = int(row["frame_id"])
    scene_config = SceneExtractorConfig()
    action_config = EndpointActionConfig(
        horizon_seconds=scene_config.horizon_seconds
    )
    history_ids = tuple(
        range(frame_id - scene_config.history_intervals, frame_id + 1)
    )
    horizon_frame_id = frame_id + scene_config.horizon_frames
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frames = load_bundesliga_frames(
        files["positions"],
        {*history_ids, horizon_frame_id},
    )
    frame = frames[frame_id]
    history = [frames[history_id] for history_id in history_ids]
    attacker_ids = tuple(
        player_id
        for player_id in str(row["eligible_attacker_ids"]).split("|")
        if player_id and player_id != "nan"
    )
    action_sets = generate_scene_attacker_endpoint_sets(
        frame,
        history,
        attacker_ids,
        frames.get(horizon_frame_id),
        action_config,
    )
    # Prefer a runner for whom both named, t0-known references are feasible.
    action_set = next(
        (
            item
            for item in action_sets
            if any("hold" in action.labels for action in item.optimization_actions)
            and any(
                "constant_velocity" in action.labels
                for action in item.optimization_actions
            )
        ),
        action_sets[0],
    )
    attacking_direction = int(row["attacking_direction"])
    records = []
    for action in _selected_actions(action_set, attacking_direction):
        state = build_attacker_counterfactual_state(
            frame,
            history,
            str(row["ball_carrier_id"]),
            action_set,
            action,
            action_config,
        )
        started = time.perf_counter()
        value = evaluate_counterfactual_threat(
            state,
            str(row["possession_team_id"]),
            attacking_direction,
            metadata,
        )
        elapsed = time.perf_counter() - started
        record = value.as_record()
        record.update(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "player_id": action.player_id,
                "labels": "|".join(action.labels),
                "endpoint_x": action.endpoint_x,
                "endpoint_y": action.endpoint_y,
                "evaluation_seconds": elapsed,
            }
        )
        records.append(record)

    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_id}_fixed_defense_actions.csv"
    result = pd.DataFrame(records)
    result.to_csv(output_path, index=False)
    print(
        result[
            [
                "labels",
                "endpoint_x",
                "endpoint_y",
                "fixed_defense_maximum_obso",
                "focal_offside",
                "evaluation_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"mean evaluation seconds: {result['evaluation_seconds'].mean():.3f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
