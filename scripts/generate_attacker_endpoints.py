#!/usr/bin/env python3
"""Generate auditable 2 s attacker endpoint actions for extracted scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from offball_value.action_space import (
    EndpointActionConfig,
    generate_scene_attacker_endpoint_sets,
)
from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.endpoint_audit import (
    endpoint_actions_dataframe,
    endpoint_summary_dataframe,
    render_endpoint_audit_html,
    select_speed_stratified_action_sets,
    write_endpoint_outputs,
)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_1"),
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=12,
        help="Number of accepted scenes to process; omit by passing a negative value.",
    )
    parser.add_argument(
        "--frame-selection-csv",
        type=Path,
        default=None,
        help="Optional CSV whose frame_id order fixes the processed scene set.",
    )
    parser.add_argument("--audit-scene-count", type=int, default=6)
    parser.add_argument("--audit-players-per-scene", type=int, default=3)
    parser.add_argument("--max-speed", type=float, default=9.0)
    parser.add_argument("--max-acceleration", type=float, default=3.5)
    parser.add_argument(
        "--max-deceleration",
        type=float,
        default=None,
        help="Defaults to --max-acceleration for the v0.1-compatible model.",
    )
    parser.add_argument(
        "--velocity-estimator",
        choices=("linear", "quadratic_endpoint"),
        default="linear",
    )
    parser.add_argument(
        "--motion-model",
        choices=("single_acceleration", "brake_turn_accelerate"),
        default="single_acceleration",
    )
    parser.add_argument("--brake-duration-step", type=float, default=0.1)
    parser.add_argument("--deceleration-step", type=float, default=0.5)
    return parser.parse_args()


def _accepted_rows(
    path: Path,
    limit: int,
    frame_selection_csv: Path | None = None,
) -> pd.DataFrame:
    scenes = pd.read_csv(path)
    if "accepted" not in scenes:
        raise ValueError(f"No accepted column in {path}")
    accepted = scenes[scenes["accepted"].astype(str).str.lower() == "true"].copy()
    accepted = accepted.sort_values("frame_id")
    if frame_selection_csv is not None:
        selection = pd.read_csv(frame_selection_csv)
        ordered_ids = selection["frame_id"].astype(int).drop_duplicates().tolist()
        order = {frame_id: index for index, frame_id in enumerate(ordered_ids)}
        accepted = accepted[accepted["frame_id"].astype(int).isin(order)].copy()
        missing = sorted(set(ordered_ids) - set(accepted["frame_id"].astype(int)))
        if missing:
            raise ValueError(f"Selected frames are not accepted scenes: {missing}")
        accepted["_selection_order"] = accepted["frame_id"].astype(int).map(order)
        accepted = accepted.sort_values("_selection_order").drop(
            columns="_selection_order"
        )
    if limit >= 0:
        accepted = accepted.head(limit)
    return accepted


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scene_path = args.scene_dir / match_id / "scene_candidates.csv"
    scenes = _accepted_rows(
        scene_path,
        args.scene_limit,
        args.frame_selection_csv,
    )
    if scenes.empty:
        raise ValueError(f"No accepted scenes found in {scene_path}")

    scene_config = SceneExtractorConfig()
    action_config = EndpointActionConfig(
        horizon_seconds=scene_config.horizon_seconds,
        max_speed_mps=args.max_speed,
        max_acceleration_mps2=args.max_acceleration,
        max_deceleration_mps2=(
            args.max_acceleration
            if args.max_deceleration is None
            else args.max_deceleration
        ),
        velocity_estimator=args.velocity_estimator,
        motion_model=args.motion_model,
        brake_duration_step_s=args.brake_duration_step,
        deceleration_step_mps2=args.deceleration_step,
    )
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])

    target_frame_ids: set[int] = set()
    history_ids_by_frame: dict[int, tuple[int, ...]] = {}
    for frame_id in scenes["frame_id"].astype(int):
        history_ids = tuple(
            range(frame_id - scene_config.history_intervals, frame_id + 1)
        )
        history_ids_by_frame[frame_id] = history_ids
        target_frame_ids.update(history_ids)
        target_frame_ids.add(frame_id + scene_config.horizon_frames)
    frames = load_bundesliga_frames(files["positions"], target_frame_ids)

    action_frames = []
    summary_frames = []
    action_sets_by_frame = {}
    decision_frames = {}
    for row in scenes.itertuples(index=False):
        frame_id = int(row.frame_id)
        frame = frames.get(frame_id)
        if frame is None:
            continue
        history = [
            frames[history_id]
            for history_id in history_ids_by_frame[frame_id]
            if history_id in frames
        ]
        observed_horizon = frames.get(frame_id + scene_config.horizon_frames)
        attacker_ids = tuple(
            player_id
            for player_id in str(row.eligible_attacker_ids).split("|")
            if player_id and player_id != "nan"
        )
        action_sets = generate_scene_attacker_endpoint_sets(
            frame,
            history,
            attacker_ids,
            observed_horizon,
            action_config,
        )
        action_sets_by_frame[frame_id] = action_sets
        decision_frames[frame_id] = frame
        action_frames.append(
            endpoint_actions_dataframe(match_id, frame_id, action_sets)
        )
        summary_frames.append(
            endpoint_summary_dataframe(match_id, frame_id, action_sets)
        )

    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    selected = select_speed_stratified_action_sets(
        action_sets_by_frame,
        scene_count=args.audit_scene_count,
        players_per_scene=args.audit_players_per_scene,
    )
    html = render_endpoint_audit_html(
        decision_frames,
        selected,
        metadata,
        action_config,
    )
    output_dir = args.output_dir / match_id
    actions_path, summary_path, audit_path = write_endpoint_outputs(
        actions,
        summary,
        html,
        output_dir,
    )

    observed = summary[summary["observed_available"] == True]  # noqa: E712
    observed_feasible_rate = (
        float(observed["observed_feasible"].mean()) if not observed.empty else float("nan")
    )
    print(f"scenes:                    {len(action_sets_by_frame)}")
    print(f"attacker action sets:      {len(summary)}")
    print(f"optimization actions:      {int(summary['optimization_action_count'].sum())}")
    print(f"median actions/player:     {summary['optimization_action_count'].median():.1f}")
    print(f"observed feasible rate:    {observed_feasible_rate:.1%}")
    print(f"motion model:              {action_config.motion_model}")
    print(f"velocity estimator:        {action_config.velocity_estimator}")
    print(f"maximum deceleration:      {action_config.max_deceleration_mps2:.2f} m/s^2")
    print(f"actions:                   {actions_path}")
    print(f"summary:                   {summary_path}")
    print(f"audit:                     {audit_path}")


if __name__ == "__main__":
    main()
