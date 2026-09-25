from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

from offball_value.bundesliga import (  # noqa: E402
    FIELD_LENGTH,
    BundesligaFrame,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.obso import (  # noqa: E402
    counterfactual_frame_for_runner,
    is_offside_position,
    receiver_option_value,
    receiver_post_reception_dangerous_space_value,
    receiver_post_reception_space_area,
)


DEFAULT_RESULTS = ROOT / "data" / "processed" / "offball_results.csv"
DEFAULT_OUT = (
    ROOT
    / "data"
    / "processed"
    / "visualizations"
    / "offball_half"
    / "benefits.csv"
)
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"


def parse_top_options(value: str | float | None) -> list[tuple[str, str, float]]:
    if value is None or pd.isna(value):
        return []
    out = []
    for item in str(value).split(";"):
        parts = item.split(":")
        if len(parts) != 3:
            continue
        try:
            out.append((parts[0], parts[1], float(parts[2])))
        except ValueError:
            continue
    return out


def lambda_key(value: float) -> str:
    return f"lambda_{f'{value:g}'.replace('-', 'm').replace('.', 'p')}"


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return distance(point, start), 0.0
    t = ((px - sx) * dx + (py - sy) * dy) / denom
    cx = sx + max(0.0, min(1.0, t)) * dx
    cy = sy + max(0.0, min(1.0, t)) * dy
    return distance(point, (cx, cy)), float(t)


def dangerous_space_norm(dangerous_space: float, space_radius: float) -> float:
    max_space_area = math.pi * space_radius**2 if space_radius > 0 else 1.0
    return max(0.0, min(1.0, dangerous_space / max_space_area))


def receiver_metrics(
    frame: BundesligaFrame,
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    adjusted_lambda: float,
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
) -> dict[str, float | bool]:
    if receiver_id not in frame.players:
        return {
            "offside": True,
            "obso": 0.0,
            "space_m2": 0.0,
            "dangerous_space": 0.0,
            "adjusted": 0.0,
        }
    offside = is_offside_position(frame, receiver_id, attacking_team_id, ball_xy, attacking_direction)
    if offside:
        return {
            "offside": True,
            "obso": 0.0,
            "space_m2": 0.0,
            "dangerous_space": 0.0,
            "adjusted": 0.0,
        }
    obso = receiver_option_value(
        frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
    )
    space = receiver_post_reception_space_area(
        frame,
        receiver_id,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
    )
    dangerous_space = receiver_post_reception_dangerous_space_value(
        frame,
        receiver_id,
        attacking_direction,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
    )
    adjusted = obso + adjusted_lambda * dangerous_space_norm(dangerous_space, space_radius)
    return {
        "offside": False,
        "obso": float(obso),
        "space_m2": float(space),
        "dangerous_space": float(dangerous_space),
        "adjusted": float(adjusted),
    }


def defensive_context(
    frame: BundesligaFrame,
    top_option_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    affected_defender_id: str | None,
) -> dict[str, float | int]:
    top_option = frame.players.get(top_option_id)
    if top_option is None:
        return {}
    top_xy = (top_option.x, top_option.y)
    goal_xy = (attacking_direction * FIELD_LENGTH / 2.0, 0.0)
    defenders = [
        player
        for player in frame.players.values()
        if player.team_id != attacking_team_id
    ]

    nearest = float("inf")
    pressure_sum = 0.0
    lane_pressure_sum = 0.0
    goal_side_blocker_sum = 0.0
    close_3m = 0
    close_5m = 0
    close_8m = 0
    affected_distance = float("nan")
    affected_pressure = 0.0
    affected_lane = 0.0
    affected_goal_side = 0.0

    for defender in defenders:
        defender_xy = (defender.x, defender.y)
        d_receiver = distance(defender_xy, top_xy)
        nearest = min(nearest, d_receiver)
        pressure = math.exp(-d_receiver / 5.0)
        pressure_sum += pressure
        close_3m += int(d_receiver <= 3.0)
        close_5m += int(d_receiver <= 5.0)
        close_8m += int(d_receiver <= 8.0)

        lane_distance, lane_projection = point_to_segment_distance(defender_xy, ball_xy, top_xy)
        lane_score = 0.0
        if 0.0 <= lane_projection <= 1.0:
            lane_score = math.exp(-0.5 * (lane_distance / 4.0) ** 2)
            lane_pressure_sum += lane_score

        goal_distance, goal_projection = point_to_segment_distance(defender_xy, top_xy, goal_xy)
        goal_score = 0.0
        if math.isfinite(goal_distance) and math.isfinite(goal_projection):
            line_score = math.exp(-0.5 * (goal_distance / 8.0) ** 2)
            if 0.0 <= goal_projection <= 1.0:
                front_weight = 1.0
            elif goal_projection < 0.0:
                front_weight = 0.25 * math.exp(goal_projection / 0.35)
            else:
                front_weight = 0.50 * math.exp(-(goal_projection - 1.0) / 0.35)
            goal_score = line_score * front_weight
            goal_side_blocker_sum += goal_score

        if affected_defender_id and defender.object_id == affected_defender_id:
            affected_distance = d_receiver
            affected_pressure = pressure
            affected_lane = lane_score
            affected_goal_side = goal_score

    return {
        "nearest_defender_distance_m": float(nearest),
        "receiver_pressure_sum": float(pressure_sum),
        "pass_lane_pressure_sum": float(lane_pressure_sum),
        "goal_side_blocker_sum": float(goal_side_blocker_sum),
        "close_defenders_3m": int(close_3m),
        "close_defenders_5m": int(close_5m),
        "close_defenders_8m": int(close_8m),
        "affected_distance_to_top_option_m": float(affected_distance),
        "affected_pressure_on_top_option": float(affected_pressure),
        "affected_pass_lane_score_to_top_option": float(affected_lane),
        "affected_goal_side_score_to_top_option": float(affected_goal_side),
    }


def gain(actual: dict, counterfactual: dict, key: str) -> float:
    return float(actual.get(key, 0.0)) - float(counterfactual.get(key, 0.0))


def reduction(actual: dict, counterfactual: dict, key: str) -> float:
    return float(counterfactual.get(key, 0.0)) - float(actual.get(key, 0.0))


def primary_benefit_label(row: dict[str, float | int | str | bool | None]) -> str:
    candidates = {
        "receiver_opened": max(
            0.0,
            float(row["nearest_defender_clearance_gain_m"]) / 3.0,
            float(row["receiver_pressure_reduction"]),
        ),
        "pass_lane_opened": max(0.0, float(row["pass_lane_pressure_reduction"])),
        "goal_side_blocker_removed": max(0.0, float(row["goal_side_blocker_reduction"])),
        "direct_defender_pulled": max(0.0, float(row["affected_removed_from_top_option_m"]) / 4.0),
        "dangerous_space_created": max(0.0, float(row["top_option_dangerous_space_gain"]) / 2.0),
        "obso_quality_gain": max(0.0, float(row["top_option_obso_gain"]) / 0.02),
    }
    label, score = max(candidates.items(), key=lambda item: item[1])
    if score <= 0.05:
        return "mixed_or_small"
    return label


def summarize(args: argparse.Namespace) -> pd.DataFrame:
    results = pd.read_csv(args.results)
    adjusted_key = f"draft_adjusted_option_offball_value_{lambda_key(args.adjusted_lambda)}"
    top_adjusted_key = f"actual_top_adjusted_options_{lambda_key(args.adjusted_lambda)}"
    rows = results.sort_values(adjusted_key, ascending=False).head(args.top_k).copy()
    rows.insert(0, "benefit_rank", range(1, len(rows) + 1))

    outputs = []
    for match_id, match_rows in rows.groupby("match_id", sort=False):
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        target_frames = set(match_rows["event_frame"].astype(int)) | set(match_rows["start_frame"].astype(int))
        frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)

        for _, row in match_rows.iterrows():
            start_frame = frames.get(int(row["start_frame"]))
            end_frame = frames.get(int(row["event_frame"]))
            if start_frame is None or end_frame is None:
                continue

            top_options = parse_top_options(row[top_adjusted_key])
            if not top_options:
                continue
            top_option_id, top_option_name, top_option_exported_adjusted = top_options[0]
            ball_xy = (float(row["ball_x"]), float(row["ball_y"]))
            attacking_team_id = str(row["team_id"])
            attacking_direction = int(row["attacking_direction"])
            affected_defender_id = (
                str(row["top_runner_affected_defender_id"])
                if "top_runner_affected_defender_id" in row and pd.notna(row["top_runner_affected_defender_id"])
                else None
            )

            cf_frame = counterfactual_frame_for_runner(
                start_frame,
                end_frame,
                str(row["runner_id"]),
                runner_alpha=args.runner_alpha,
                defender_alpha=args.defender_alpha,
                affected_defender_id=affected_defender_id,
                target_receiver_id=top_option_id,
                attacking_team_id=attacking_team_id,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                defender_selection="responsibility",
                fallback_to_selected_defender=False,
            )

            actual_metrics = receiver_metrics(
                end_frame,
                top_option_id,
                attacking_team_id,
                attacking_direction,
                ball_xy,
                args.adjusted_lambda,
                args.local_radius,
                args.local_samples,
                args.space_radius,
                args.space_samples,
                args.space_rings,
            )
            cf_metrics = receiver_metrics(
                cf_frame,
                top_option_id,
                attacking_team_id,
                attacking_direction,
                ball_xy,
                args.adjusted_lambda,
                args.local_radius,
                args.local_samples,
                args.space_radius,
                args.space_samples,
                args.space_rings,
            )
            actual_context = defensive_context(
                end_frame,
                top_option_id,
                attacking_team_id,
                attacking_direction,
                ball_xy,
                affected_defender_id,
            )
            cf_context = defensive_context(
                cf_frame,
                top_option_id,
                attacking_team_id,
                attacking_direction,
                ball_xy,
                affected_defender_id,
            )

            out: dict[str, float | int | str | bool | None] = {
                "rank": int(row["benefit_rank"]),
                "match_id": match_id,
                "event_frame": int(row["event_frame"]),
                "passer_name": row["passer_name"],
                "recipient_name": row["recipient_name"],
                "runner_name": row["runner_name"],
                "affected_defender_name": row.get("top_runner_affected_defender_name"),
                "top_option_name": top_option_name,
                "top_option_exported_adjusted": top_option_exported_adjusted,
                "team_topk_adjusted_gain": float(row[adjusted_key]),
                "top_option_adjusted_actual": actual_metrics["adjusted"],
                "top_option_adjusted_counterfactual": cf_metrics["adjusted"],
                "top_option_adjusted_gain": gain(actual_metrics, cf_metrics, "adjusted"),
                "top_option_obso_actual": actual_metrics["obso"],
                "top_option_obso_counterfactual": cf_metrics["obso"],
                "top_option_obso_gain": gain(actual_metrics, cf_metrics, "obso"),
                "top_option_space_actual_m2": actual_metrics["space_m2"],
                "top_option_space_counterfactual_m2": cf_metrics["space_m2"],
                "top_option_space_gain_m2": gain(actual_metrics, cf_metrics, "space_m2"),
                "top_option_dangerous_space_actual": actual_metrics["dangerous_space"],
                "top_option_dangerous_space_counterfactual": cf_metrics["dangerous_space"],
                "top_option_dangerous_space_gain": gain(actual_metrics, cf_metrics, "dangerous_space"),
                "top_option_was_offside_counterfactual": cf_metrics["offside"],
                "nearest_defender_distance_actual_m": actual_context["nearest_defender_distance_m"],
                "nearest_defender_distance_counterfactual_m": cf_context["nearest_defender_distance_m"],
                "nearest_defender_clearance_gain_m": gain(
                    actual_context,
                    cf_context,
                    "nearest_defender_distance_m",
                ),
                "close_defenders_3m_removed": reduction(actual_context, cf_context, "close_defenders_3m"),
                "close_defenders_5m_removed": reduction(actual_context, cf_context, "close_defenders_5m"),
                "close_defenders_8m_removed": reduction(actual_context, cf_context, "close_defenders_8m"),
                "receiver_pressure_actual": actual_context["receiver_pressure_sum"],
                "receiver_pressure_counterfactual": cf_context["receiver_pressure_sum"],
                "receiver_pressure_reduction": reduction(
                    actual_context,
                    cf_context,
                    "receiver_pressure_sum",
                ),
                "pass_lane_pressure_actual": actual_context["pass_lane_pressure_sum"],
                "pass_lane_pressure_counterfactual": cf_context["pass_lane_pressure_sum"],
                "pass_lane_pressure_reduction": reduction(
                    actual_context,
                    cf_context,
                    "pass_lane_pressure_sum",
                ),
                "goal_side_blocker_actual": actual_context["goal_side_blocker_sum"],
                "goal_side_blocker_counterfactual": cf_context["goal_side_blocker_sum"],
                "goal_side_blocker_reduction": reduction(
                    actual_context,
                    cf_context,
                    "goal_side_blocker_sum",
                ),
                "affected_distance_to_top_option_actual_m": actual_context[
                    "affected_distance_to_top_option_m"
                ],
                "affected_distance_to_top_option_counterfactual_m": cf_context[
                    "affected_distance_to_top_option_m"
                ],
                "affected_removed_from_top_option_m": gain(
                    actual_context,
                    cf_context,
                    "affected_distance_to_top_option_m",
                ),
                "affected_pressure_reduction_on_top_option": reduction(
                    actual_context,
                    cf_context,
                    "affected_pressure_on_top_option",
                ),
                "affected_pass_lane_score_reduction_to_top_option": reduction(
                    actual_context,
                    cf_context,
                    "affected_pass_lane_score_to_top_option",
                ),
                "affected_goal_side_score_reduction_to_top_option": reduction(
                    actual_context,
                    cf_context,
                    "affected_goal_side_score_to_top_option",
                ),
            }
            out["primary_benefit_type"] = primary_benefit_label(out)
            outputs.append(out)

    return pd.DataFrame(outputs).sort_values("rank").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize how each off-ball run benefits the top option.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--adjusted-lambda", type=float, default=0.5)
    parser.add_argument("--local-radius", type=float, default=3.0)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--space-radius", type=float, default=5.0)
    parser.add_argument("--space-samples", type=int, default=16)
    parser.add_argument("--space-rings", type=int, default=3)
    parser.add_argument("--runner-alpha", type=float, default=0.0)
    parser.add_argument("--defender-alpha", type=float, default=0.3)
    args = parser.parse_args()

    out = summarize(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    cols = [
        "rank",
        "runner_name",
        "affected_defender_name",
        "top_option_name",
        "primary_benefit_type",
        "team_topk_adjusted_gain",
        "top_option_adjusted_gain",
        "top_option_obso_gain",
        "top_option_space_gain_m2",
        "nearest_defender_clearance_gain_m",
        "close_defenders_5m_removed",
        "receiver_pressure_reduction",
        "pass_lane_pressure_reduction",
        "goal_side_blocker_reduction",
        "affected_removed_from_top_option_m",
    ]
    print(out[cols].head(args.top_k).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
