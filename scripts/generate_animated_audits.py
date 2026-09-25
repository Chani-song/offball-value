#!/usr/bin/env python3
"""Generate interactive tracking animations for scene and endpoint QC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.animated_audit import (
    animation_frame_payload,
    render_tracking_animation_html,
)
from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.fernandez_influence import fernandez_influence_ellipse


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
        "--endpoint-dir",
        type=Path,
        default=Path("data/processed/attacker_endpoints_v0_1"),
    )
    parser.add_argument(
        "--baseline-endpoint-dir",
        type=Path,
        default=None,
        help="Optional v0.1 action directory to overlay for comparison.",
    )
    parser.add_argument(
        "--endpoint-review-csv",
        type=Path,
        default=None,
        help="Optional prior human-review CSV fixing the frame/player audit set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/animated_audit_v0_1"),
    )
    parser.add_argument("--scene-clips", type=int, default=16)
    parser.add_argument("--endpoint-clips", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--primary-label", default="v0.2 feasible endpoints")
    parser.add_argument("--baseline-label", default="v0.1 baseline endpoints")
    return parser.parse_args()


def _even_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0)
    if len(frame) <= count:
        return frame
    indices = np.linspace(0, len(frame) - 1, count).round().astype(int)
    return frame.iloc[np.unique(indices)]


def _select_scene_rows(
    candidates: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.DataFrame:
    accepted = candidates[candidates["accepted"].astype(str).str.lower() == "true"]
    rejected = candidates[candidates["accepted"].astype(str).str.lower() != "true"]
    accepted_count = count // 2
    selected = [_even_sample(accepted.sort_values("frame_id"), accepted_count)]
    reason_order = [
        "ball_carrier_too_far",
        "unstable_control_history",
        "set_piece_window",
        "unstable_event_window",
    ]
    used = set()
    rejected_rows = []
    for reason in reason_order:
        subset = rejected[
            rejected["rejection_reasons"].fillna("").str.contains(reason, regex=False)
        ]
        if subset.empty:
            continue
        row = subset.sample(1, random_state=seed + len(rejected_rows)).iloc[0]
        key = int(row["frame_id"])
        if key not in used:
            rejected_rows.append(row)
            used.add(key)
    remaining = count - accepted_count - len(rejected_rows)
    if remaining > 0:
        pool = rejected[~rejected["frame_id"].astype(int).isin(used)]
        if not pool.empty:
            extra = pool.sample(
                min(remaining, len(pool)),
                random_state=seed,
            )
            rejected_rows.extend(row for _, row in extra.iterrows())
    if rejected_rows:
        selected.append(pd.DataFrame(rejected_rows))
    return pd.concat(selected, ignore_index=True).head(count)


def _select_endpoint_rows(summary: pd.DataFrame, count: int) -> pd.DataFrame:
    uncovered = summary[
        summary["observed_available"]
        & ~summary["observed_feasible"].fillna(False)
    ].sort_values("nearest_action_to_observed_m", ascending=False)
    uncovered_count = min(len(uncovered), count // 2)
    selected = [uncovered.head(uncovered_count)]
    covered = summary[
        summary["observed_available"]
        & summary["observed_feasible"].fillna(False)
    ].sort_values("initial_speed_mps")
    selected.append(_even_sample(covered, count - uncovered_count))
    result = pd.concat(selected, ignore_index=True)
    return result.drop_duplicates(["frame_id", "player_id"]).head(count)


def _select_reviewed_endpoint_rows(
    summary: pd.DataFrame,
    review_path: Path,
    count: int,
) -> pd.DataFrame:
    reviews = pd.read_csv(review_path)
    keys = reviews[["frame_id", "player_id"]].drop_duplicates()
    selected = keys.merge(
        summary,
        on=["frame_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["initial_speed_mps"].isna().any():
        missing = selected[selected["initial_speed_mps"].isna()][
            ["frame_id", "player_id"]
        ].to_dict("records")
        raise ValueError(f"Reviewed endpoint rows missing from summary: {missing}")
    return selected.head(count)


def _frame_ids(decision_frame_id: int) -> list[int]:
    return list(range(decision_frame_id - 25, decision_frame_id + 51))


def _name(metadata, player_id: str | None) -> str:
    if not player_id:
        return "unknown"
    player = metadata.players.get(player_id)
    return player.short_name if player else player_id


def _representative_paths(
    group: pd.DataFrame,
    row,
    required_label: str | None = None,
    excluded_label: str | None = None,
) -> list[list[list[float]]]:
    if "path_xy" not in group.columns:
        return []
    eligible = group[
        group["labels"].fillna("").str.contains("grid", regex=False)
        & group["optimization_eligible"].fillna(False)
        & group["path_xy"].notna()
    ].copy()
    if required_label is not None:
        eligible = eligible[
            eligible["labels"].fillna("").str.contains(required_label, regex=False)
        ]
    if excluded_label is not None:
        eligible = eligible[
            ~eligible["labels"].fillna("").str.contains(excluded_label, regex=False)
        ]
    if eligible.empty:
        return []
    speed = math.hypot(float(row.initial_vx_mps), float(row.initial_vy_mps))
    if speed >= 0.25:
        heading = math.atan2(float(row.initial_vy_mps), float(row.initial_vx_mps))
    elif hasattr(row, "initial_ax_mps2") and math.hypot(
        float(row.initial_ax_mps2), float(row.initial_ay_mps2)
    ) >= 0.25:
        heading = math.atan2(float(row.initial_ay_mps2), float(row.initial_ax_mps2))
    else:
        heading = 0.0
    dx = eligible["endpoint_x"] - float(row.start_x)
    dy = eligible["endpoint_y"] - float(row.start_y)
    eligible["_distance"] = np.hypot(dx, dy)
    eligible["_spatial_sector"] = np.floor(
        (
            (
                np.arctan2(dy, dx)
                - heading
                + 2.0 * math.pi
            )
            % (2.0 * math.pi)
        )
        / (2.0 * math.pi / 8)
    ).astype(int)
    selected_indices = set()
    for _, sector in eligible.groupby("_spatial_sector"):
        selected_indices.add(int(sector["_distance"].idxmax()))
    if "terminal_heading_bin" in eligible.columns:
        effort_column = (
            "control_effort_m2ps3"
            if "control_effort_m2ps3" in eligible.columns
            else "_distance"
        )
        for _, terminal_group in eligible.dropna(
            subset=["terminal_heading_bin"]
        ).groupby("terminal_heading_bin"):
            selected_indices.add(int(terminal_group[effort_column].idxmin()))
    paths = []
    for index in sorted(selected_indices):
        try:
            points = json.loads(str(eligible.loc[index, "path_xy"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        paths.append(
            [
                [float(row.start_x), float(row.start_y)],
                *[[float(point[0]), float(point[1])] for point in points],
            ]
        )
    return paths


def main() -> None:
    args = parse_args()
    match_id = normalize_bundesliga_match_id(args.match_id)
    scene_path = args.scene_dir / match_id / "scene_candidates.csv"
    action_path = args.endpoint_dir / match_id / "attacker_endpoint_actions.csv"
    endpoint_summary_path = args.endpoint_dir / match_id / "attacker_endpoint_summary.csv"
    candidates = pd.read_csv(scene_path)
    actions = pd.read_csv(action_path)
    endpoint_summary = pd.read_csv(endpoint_summary_path)
    scene_rows = _select_scene_rows(candidates, args.scene_clips, args.seed)
    endpoint_rows = (
        _select_reviewed_endpoint_rows(
            endpoint_summary,
            args.endpoint_review_csv,
            args.endpoint_clips,
        )
        if args.endpoint_review_csv is not None
        else _select_endpoint_rows(endpoint_summary, args.endpoint_clips)
    )
    baseline_actions = None
    baseline_summary = None
    if args.baseline_endpoint_dir is not None:
        baseline_dir = args.baseline_endpoint_dir / match_id
        baseline_actions = pd.read_csv(
            baseline_dir / "attacker_endpoint_actions.csv"
        )
        baseline_summary = pd.read_csv(
            baseline_dir / "attacker_endpoint_summary.csv"
        ).set_index(["frame_id", "player_id"])

    target_ids = set()
    for frame_id in [
        *scene_rows["frame_id"].astype(int).tolist(),
        *endpoint_rows["frame_id"].astype(int).tolist(),
    ]:
        target_ids.update(_frame_ids(frame_id))
    files = find_bundesliga_files(args.data_dir, match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])
    frames = load_bundesliga_frames(files["positions"], target_ids)

    scene_payloads = []
    for row in scene_rows.itertuples(index=False):
        frame_id = int(row.frame_id)
        status = "accepted" if str(row.accepted).lower() == "true" else "rejected"
        scene_payloads.append(
            {
                "match_id": match_id,
                "decision_frame_id": frame_id,
                "title": f"frame {frame_id} · carrier {_name(metadata, row.ball_carrier_id)}",
                "details": (
                    f"distance={row.ball_carrier_distance_m:.2f}m · "
                    f"stable={row.stable_control_fraction:.0%} · "
                    f"event={row.nearest_event_type} · "
                    f"reasons={row.rejection_reasons if isinstance(row.rejection_reasons, str) else 'none'}"
                ),
                "status": status,
                "possession_team_id": row.possession_team_id,
                "carrier_id": row.ball_carrier_id,
                "focal_id": None,
                "frames": animation_frame_payload(
                    frames,
                    _frame_ids(frame_id),
                    frame_id,
                    metadata,
                ),
                "endpoints": [],
                "baseline_endpoints": [],
                "cv_endpoint": None,
                "observed_endpoint": None,
                "observed_feasible": None,
            }
        )

    scene_lookup = candidates.set_index("frame_id")
    endpoint_payloads = []
    for row in endpoint_rows.itertuples(index=False):
        frame_id = int(row.frame_id)
        player_id = str(row.player_id)
        group = actions[
            (actions["frame_id"] == frame_id)
            & (actions["player_id"] == player_id)
        ]
        continuous_mask = (
            group["labels"].fillna("").str.contains("grid", regex=False)
            & group["optimization_eligible"].fillna(False)
            & ~group["labels"].fillna("").str.contains(
                "plant_cut_feasible", regex=False
            )
        )
        plant_cut_mask = (
            group["labels"].fillna("").str.contains(
                "plant_cut_feasible", regex=False
            )
            & group["optimization_eligible"].fillna(False)
        )
        grid = group[
            continuous_mask
        ][["endpoint_x", "endpoint_y"]].values.tolist()
        plant_cut_grid = group[plant_cut_mask][
            ["endpoint_x", "endpoint_y"]
        ].values.tolist()
        vector_column_list = [
            "endpoint_x",
            "endpoint_y",
            "terminal_vx_mps",
            "terminal_vy_mps",
        ]
        vector_columns = set(vector_column_list)
        endpoint_vectors = (
            group[
                continuous_mask
            ][vector_column_list]
            .dropna()
            .values.tolist()
            if vector_columns <= set(group.columns)
            else []
        )
        plant_cut_vectors = (
            group[plant_cut_mask][vector_column_list].dropna().values.tolist()
            if vector_columns <= set(group.columns)
            else []
        )
        representative_paths = _representative_paths(
            group,
            row,
            excluded_label="plant_cut_feasible",
        )
        plant_cut_paths = _representative_paths(
            group,
            row,
            required_label="plant_cut_feasible",
        )
        cv = group[
            group["labels"].fillna("").str.contains(
                "constant_velocity", regex=False
            )
        ]
        observed = group[
            group["labels"].fillna("").str.contains("observed", regex=False)
        ]
        proposal_grid = group[
            group["labels"].fillna("").str.contains(
                "fernandez_proposal", regex=False
            )
        ][["endpoint_x", "endpoint_y"]].values.tolist()
        empirical_grid = (
            group[
                group["labels"].fillna("").str.contains("grid", regex=False)
                & group["optimization_eligible"].fillna(False)
                & group["empirically_supported"].fillna(False)
            ][["endpoint_x", "endpoint_y"]]
            .drop_duplicates()
            .values.tolist()
            if "empirically_supported" in group.columns
            else []
        )
        baseline_grid: list[list[float]] = []
        baseline_row = None
        if baseline_actions is not None and baseline_summary is not None:
            baseline_group = baseline_actions[
                (baseline_actions["frame_id"] == frame_id)
                & (baseline_actions["player_id"] == player_id)
            ]
            baseline_grid = baseline_group[
                baseline_group["labels"].fillna("").str.contains(
                    "grid", regex=False
                )
                & baseline_group["optimization_eligible"].fillna(False)
            ][["endpoint_x", "endpoint_y"]].values.tolist()
            if (frame_id, player_id) in baseline_summary.index:
                baseline_row = baseline_summary.loc[(frame_id, player_id)]
        scene = scene_lookup.loc[frame_id]
        decision_frame = frames.get(frame_id)
        influence = None
        if (
            decision_frame is not None
            and decision_frame.ball is not None
            and player_id in decision_frame.players
        ):
            player = decision_frame.players[player_id]
            influence = fernandez_influence_ellipse(
                (player.x, player.y),
                (float(row.initial_vx_mps), float(row.initial_vy_mps)),
                (decision_frame.ball.x, decision_frame.ball.y),
            ).as_record()
        baseline_status = (
            "n/a"
            if baseline_row is None
            else (
                "feasible"
                if bool(baseline_row["observed_feasible"])
                else "infeasible"
            )
        )
        estimator = getattr(row, "velocity_estimator", "unknown")
        empirical_mode = hasattr(row, "observed_endpoint_heading_supported")
        primary_supported = bool(
            getattr(
                row,
                "observed_endpoint_heading_supported",
                row.observed_feasible,
            )
        )
        status_subject = (
            "OBSERVED ENDPOINT+HEADING" if empirical_mode else "OBSERVED ENDPOINT"
        )
        status_word = (
            "SUPPORTED" if empirical_mode and primary_supported
            else "UNSUPPORTED" if empirical_mode
            else "FEASIBLE" if primary_supported
            else "INFEASIBLE"
        )
        observed_vector = None
        if not observed.empty and vector_columns <= set(observed.columns):
            observed_values = observed.iloc[0]
            if pd.notna(observed_values["terminal_vx_mps"]) and pd.notna(
                observed_values["terminal_vy_mps"]
            ):
                observed_vector = [
                    float(observed_values["endpoint_x"]),
                    float(observed_values["endpoint_y"]),
                    float(observed_values["terminal_vx_mps"]),
                    float(observed_values["terminal_vy_mps"]),
                ]
        if hasattr(row, "fernandez_proposal_count"):
            plant_cut_detail = (
                f" · plant cuts={int(row.plant_cut_action_count)}"
                if hasattr(row, "plant_cut_action_count")
                else ""
            )
            details = (
                f"initial speed={row.initial_speed_mps:.2f}m/s · "
                f"Fernández proposals={int(row.fernandez_proposal_count)} "
                f"(not automatically feasible) · "
                f"dynamic endpoints={int(row.unique_endpoint_count)} · "
                f"spatial sectors={int(row.spatial_sector_count)}/8 · "
                f"empirical-support endpoints="
                f"{int(row.empirically_supported_endpoint_count)} · "
                f"observed={'supported' if primary_supported else row.observed_failure_reason}"
                f"{plant_cut_detail}"
            )
        else:
            details = (
                f"initial speed={row.initial_speed_mps:.2f}m/s · "
                f"actions {args.baseline_label}={len(baseline_grid)}, "
                f"{args.primary_label}={int(row.optimization_action_count)} · "
                f"observed baseline={baseline_status}, "
                f"primary={'supported' if primary_supported else row.observed_failure_reason} · "
                f"velocity={estimator}"
            )
        endpoint_payloads.append(
            {
                "match_id": match_id,
                "decision_frame_id": frame_id,
                "title": f"frame {frame_id} · runner {_name(metadata, player_id)}",
                "details": details,
                "status": "accepted" if primary_supported else "rejected",
                "status_label": f"{status_subject} {status_word}",
                "possession_team_id": scene["possession_team_id"],
                "carrier_id": scene["ball_carrier_id"],
                "focal_id": player_id,
                "frames": animation_frame_payload(
                    frames,
                    _frame_ids(frame_id),
                    frame_id,
                    metadata,
                ),
                "endpoints": grid,
                "endpoint_vectors": endpoint_vectors,
                "plant_cut_endpoints": plant_cut_grid,
                "plant_cut_vectors": plant_cut_vectors,
                "proposal_endpoints": proposal_grid,
                "empirical_endpoints": empirical_grid,
                "representative_paths": representative_paths,
                "plant_cut_paths": plant_cut_paths,
                "baseline_endpoints": baseline_grid,
                "influence_ellipse": influence,
                "cv_endpoint": (
                    [float(cv.iloc[0]["endpoint_x"]), float(cv.iloc[0]["endpoint_y"])]
                    if not cv.empty
                    else None
                ),
                "observed_endpoint": (
                    [
                        float(observed.iloc[0]["endpoint_x"]),
                        float(observed.iloc[0]["endpoint_y"]),
                    ]
                    if not observed.empty
                    else None
                ),
                "observed_endpoint_vector": observed_vector,
                "observed_feasible": primary_supported,
            }
        )

    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_output = output_dir / "scene_animation_audit.html"
    endpoint_output = output_dir / "endpoint_animation_audit.html"
    scene_output.write_text(
        render_tracking_animation_html(
            scene_payloads,
            "scene",
            "Controlled-possession scene animation audit",
            "Actual tracking from one second before to two seconds after each decision candidate.",
        ),
        encoding="utf-8",
    )
    endpoint_output.write_text(
        render_tracking_animation_html(
            endpoint_payloads,
            "endpoint",
            "Attacker reachable-endpoint animation audit",
            (
                "Actual movement is replayed over the t0 action spaces; "
                f"{args.primary_label}, {args.baseline_label}, and Fernández "
                "influence are shown separately. Small green arrows encode "
                "the feasible terminal arrival direction at t+2."
            ),
            primary_endpoint_label=args.primary_label,
            baseline_endpoint_label=args.baseline_label,
        ),
        encoding="utf-8",
    )
    print(f"scene clips:    {len(scene_payloads)}")
    print(f"endpoint clips: {len(endpoint_payloads)}")
    print(f"scene audit:    {scene_output}")
    print(f"endpoint audit: {endpoint_output}")


if __name__ == "__main__":
    main()
