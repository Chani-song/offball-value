from __future__ import annotations

import argparse
from copy import copy
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offball_value.bundesliga import (
    BundesligaFrame,
    find_bundesliga_files,
    infer_attacking_direction,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
    short_bundesliga_match_id,
)
from offball_value.obso import (
    affected_defender_for_runner,
    counterfactual_frame_for_runner,
    defender_responsibilities_for_point,
    defender_responsibilities_for_receiver,
    is_offside_position,
    projected_runner_threat_point,
    receiver_defensive_suppression_attribution,
    receiver_option_value,
    receiver_post_reception_dangerous_space_value,
    receiver_post_reception_space_area,
    score_at_points,
)


DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def euclidean(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return float(math.hypot(a_x - b_x, a_y - b_y))


def point_to_segment_distance_xy(
    px: float,
    py: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    seg_x = end_x - start_x
    seg_y = end_y - start_y
    denom = seg_x * seg_x + seg_y * seg_y
    if denom <= 1e-9:
        return euclidean(px, py, start_x, start_y)
    t = max(0.0, min(1.0, ((px - start_x) * seg_x + (py - start_y) * seg_y) / denom))
    proj_x = start_x + t * seg_x
    proj_y = start_y + t * seg_y
    return euclidean(px, py, proj_x, proj_y)


def player_label(metadata, player_id: str | None) -> str | None:
    if player_id is None:
        return None
    player = metadata.players.get(player_id)
    if player is None:
        return player_id
    return player.short_name


def is_goalkeeper(metadata, player_id: str) -> bool:
    player = metadata.players.get(player_id)
    return player is not None and player.position == "TW"


def candidate_receiver_ids(
    frame: BundesligaFrame,
    metadata,
    attacking_team_id: str,
    excluded_ids: set[str],
    ball_xy: tuple[float, float],
    attacking_direction: int,
    min_receiver_ahead: float | None = None,
    min_receiver_score: float | None = None,
    max_receiver_goal_distance: float | None = None,
    filter_offside: bool = True,
) -> list[str]:
    candidates = []
    for player_id, player in frame.players.items():
        if player.team_id != attacking_team_id or player_id in excluded_ids or is_goalkeeper(metadata, player_id):
            continue
        if filter_offside and is_offside_position(
            frame,
            player_id,
            attacking_team_id,
            ball_xy,
            attacking_direction,
        ):
            continue
        if min_receiver_ahead is not None:
            ahead_m = attacking_direction * (player.x - ball_xy[0])
            if ahead_m < min_receiver_ahead:
                continue
        if min_receiver_score is not None:
            score = float(score_at_points([(player.x, player.y)], attacking_direction)[0])
            if score < min_receiver_score:
                continue
        if max_receiver_goal_distance is not None:
            goal_x = attacking_direction * 105.0 / 2.0
            goal_distance = math.hypot(player.x - goal_x, player.y)
            if goal_distance > max_receiver_goal_distance:
                continue
        candidates.append(player_id)
    return candidates


def topk_receiver_obso(
    frame: BundesligaFrame,
    metadata,
    attacking_team_id: str,
    attacking_direction: int,
    candidate_ids: list[str],
    ball_xy: tuple[float, float],
    k: int,
    local_radius: float,
    local_samples: int,
) -> tuple[float, list[tuple[str, float]]]:
    values = []
    for player_id in candidate_ids:
        value = receiver_option_value(
            frame,
            player_id,
            attacking_team_id,
            attacking_direction,
            ball_xy=ball_xy,
            local_radius=local_radius,
            local_samples=local_samples,
        )
        if not math.isnan(value):
            values.append((player_id, value))
    values.sort(key=lambda item: item[1], reverse=True)
    top = values[: max(1, k)]
    return float(sum(value for _, value in top)), top


def topk_receiver_post_space(
    frame: BundesligaFrame,
    candidate_ids: list[str],
    k: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
) -> tuple[float, list[tuple[str, float]]]:
    values = []
    for player_id in candidate_ids:
        value = receiver_post_reception_space_area(
            frame,
            player_id,
            space_radius=space_radius,
            space_samples=space_samples,
            space_rings=space_rings,
        )
        if not math.isnan(value):
            values.append((player_id, value))
    values.sort(key=lambda item: item[1], reverse=True)
    top = values[: max(1, k)]
    return float(sum(value for _, value in top)), top


def topk_receiver_dangerous_post_space(
    frame: BundesligaFrame,
    attacking_direction: int,
    candidate_ids: list[str],
    k: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
) -> tuple[float, list[tuple[str, float]]]:
    values = []
    for player_id in candidate_ids:
        value = receiver_post_reception_dangerous_space_value(
            frame,
            player_id,
            attacking_direction,
            space_radius=space_radius,
            space_samples=space_samples,
            space_rings=space_rings,
        )
        if not math.isnan(value):
            values.append((player_id, value))
    values.sort(key=lambda item: item[1], reverse=True)
    top = values[: max(1, k)]
    return float(sum(value for _, value in top)), top


def lambda_key(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{text}"


def parse_lambdas(value: str) -> list[float]:
    lambdas = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lambdas.append(float(item))
    return lambdas


def receiver_option_metrics(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    candidate_ids: list[str],
    ball_xy: tuple[float, float],
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
    adjusted_lambdas: list[float],
) -> list[dict[str, float | str]]:
    metrics: list[dict[str, float | str]] = []
    max_space_area = math.pi * space_radius**2 if space_radius > 0 else 1.0

    for player_id in candidate_ids:
        obso = receiver_option_value(
            frame,
            player_id,
            attacking_team_id,
            attacking_direction,
            ball_xy=ball_xy,
            local_radius=local_radius,
            local_samples=local_samples,
        )
        space = receiver_post_reception_space_area(
            frame,
            player_id,
            space_radius=space_radius,
            space_samples=space_samples,
            space_rings=space_rings,
        )
        dangerous_space = receiver_post_reception_dangerous_space_value(
            frame,
            player_id,
            attacking_direction,
            space_radius=space_radius,
            space_samples=space_samples,
            space_rings=space_rings,
        )
        if not (math.isfinite(obso) and math.isfinite(space) and math.isfinite(dangerous_space)):
            continue

        dangerous_space_norm = max(0.0, min(1.0, dangerous_space / max_space_area))
        row: dict[str, float | str] = {
            "player_id": player_id,
            "obso": obso,
            "post_space_m2": space,
            "dangerous_space": dangerous_space,
            "dangerous_space_norm": dangerous_space_norm,
        }
        for lam in adjusted_lambdas:
            row[f"adjusted_{lambda_key(lam)}"] = obso + lam * dangerous_space_norm
        metrics.append(row)

    return metrics


def topk_metric(
    metrics: list[dict[str, float | str]],
    metric_key: str,
    k: int,
) -> tuple[float, list[tuple[str, float]]]:
    values = [
        (str(item["player_id"]), float(item[metric_key]))
        for item in metrics
        if metric_key in item and math.isfinite(float(item[metric_key]))
    ]
    values.sort(key=lambda item: item[1], reverse=True)
    top = values[: max(1, k)]
    return float(sum(value for _, value in top)), top


def format_top_options(metadata, options: list[tuple[str, float]]) -> str:
    return ";".join(
        f"{player_id}:{player_label(metadata, player_id)}:{value:.6f}"
        for player_id, value in options
    )


def format_defender_responsibilities(metadata, responsibilities: list[dict[str, float | str]], limit: int = 5) -> str:
    return ";".join(
        (
            f"{item['defender_id']}:{player_label(metadata, str(item['defender_id']))}:"
            f"{float(item['responsibility']):.6f}:"
            f"{float(item['lane_score']):.6f}:"
            f"{float(item['pressure_score']):.6f}:"
            f"{float(item['goal_side_score']):.6f}"
        )
        for item in responsibilities[:limit]
    )


def format_defensive_suppression_attributions(
    metadata,
    attributions: list[dict[str, float | str]],
    limit: int = 5,
) -> str:
    return ";".join(
        (
            f"{item['defender_id']}:{player_label(metadata, str(item['defender_id']))}:"
            f"{float(item['allocated_contribution']):.6f}:"
            f"{float(item['raw_contribution']):.6f}:"
            f"{float(item['released_value']):.6f}:"
            f"{float(item['distance_to_receiver']):.2f}:"
            f"{float(item['distance_to_runner_path']):.2f}:"
            f"{float(item['distance_to_pass_lane']):.2f}"
        )
        for item in attributions[:limit]
    )


def format_defender_role_attributions(
    metadata,
    attributions: list[dict[str, float | str]],
    score_key: str,
    limit: int = 5,
) -> str:
    return ";".join(
        (
            f"{item['defender_id']}:{player_label(metadata, str(item['defender_id']))}:"
            f"{float(item.get(score_key, 0.0)):.6f}:"
            f"{float(item.get('raw_threat_contribution', 0.0)):.6f}:"
            f"{float(item.get('raw_contribution', 0.0)):.6f}:"
            f"{float(item.get('distance_to_receiver', float('nan'))):.2f}:"
            f"{float(item.get('goal_side_score', 0.0)):.6f}:"
            f"{float(item.get('pass_lane_score', 0.0)):.6f}:"
            f"{float(item.get('passer_pressure_score', 0.0)):.6f}"
        )
        for item in attributions[:limit]
    )


def runner_affected_suppression_attributions(
    attributions: list[dict[str, float | str]],
    nearest_n: int,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    attacking_team_id: str,
    ball_xy: tuple[float, float],
    attacking_direction: int,
    target_receiver_id: str | None,
    runner_id: str,
) -> list[dict[str, float | str]]:
    runner_start = start_frame.players.get(runner_id)
    runner_end = end_frame.players.get(runner_id)
    runner_threat_point = projected_runner_threat_point(start_frame, end_frame, runner_id)
    if runner_start is None or runner_end is None or runner_threat_point is None:
        return []

    runner_vec_x = runner_end.x - runner_start.x
    runner_vec_y = runner_end.y - runner_start.y
    runner_norm = euclidean(runner_start.x, runner_start.y, runner_end.x, runner_end.y)
    threat_x, threat_y = runner_threat_point
    responsibilities = defender_responsibilities_for_point(
        end_frame,
        attacking_team_id,
        runner_threat_point,
        ball_xy,
        attacking_direction,
    )
    responsibility_by_id = {
        str(item["defender_id"]): item
        for item in responsibilities
    }
    max_responsibility = max(
        (float(item.get("responsibility", 0.0)) for item in responsibilities),
        default=0.0,
    )

    direct_candidates = []
    for item in attributions:
        copied = dict(item)
        defender_id = str(copied["defender_id"])
        defender = end_frame.players.get(defender_id)
        defender_start = start_frame.players.get(defender_id)
        if defender is None or defender_start is None:
            continue

        distance_to_runner_end = float(copied.get("distance_to_receiver", float("nan")))
        distance_to_runner_threat = euclidean(defender.x, defender.y, threat_x, threat_y)
        distance_to_projected_path = point_to_segment_distance_xy(
            defender.x,
            defender.y,
            runner_start.x,
            runner_start.y,
            threat_x,
            threat_y,
        )
        if distance_to_runner_threat > 10.0 and distance_to_projected_path > 6.0:
            continue

        defender_vec_x = defender.x - defender_start.x
        defender_vec_y = defender.y - defender_start.y
        defender_norm = euclidean(defender_start.x, defender_start.y, defender.x, defender.y)
        follow_score = 0.0
        if runner_norm > 1e-6 and defender_norm > 1e-6:
            cosine = (runner_vec_x * defender_vec_x + runner_vec_y * defender_vec_y) / (
                runner_norm * defender_norm
            )
            follow_score = max(0.0, cosine) * min(1.0, defender_norm / max(runner_norm, 1e-6))

        start_threat_distance = euclidean(defender_start.x, defender_start.y, threat_x, threat_y)
        closing_score = 0.0
        if runner_norm > 1e-6:
            closing_score = max(
                0.0,
                min(1.0, (start_threat_distance - distance_to_runner_threat) / runner_norm),
            )

        responsibility = responsibility_by_id.get(defender_id, {})
        runner_responsibility = float(responsibility.get("responsibility", 0.0))
        runner_responsibility_score = (
            runner_responsibility / max_responsibility
            if max_responsibility > 1e-12
            else 0.0
        )
        goal_side_score = float(responsibility.get("goal_side_score", 0.0))
        pass_lane_score = float(responsibility.get("lane_score", copied.get("pass_lane_score", 0.0)))
        pressure_score = float(responsibility.get("pressure_score", 0.0))
        distance_to_threat_pass_lane = float(
            responsibility.get("distance_to_pass_lane", copied.get("distance_to_pass_lane", float("nan")))
        )

        target_distance = float("nan")
        target_option_bias_m = 0.0
        if target_receiver_id and target_receiver_id != runner_id:
            target = end_frame.players.get(target_receiver_id)
            if defender is not None and target is not None:
                target_distance = euclidean(defender.x, defender.y, target.x, target.y)
                if math.isfinite(distance_to_runner_end):
                    target_option_bias_m = max(0.0, distance_to_runner_end - target_distance)

        copied["distance_to_runner_end_m"] = float(distance_to_runner_end)
        copied["distance_to_runner_threat_point_m"] = float(distance_to_runner_threat)
        copied["distance_to_receiver"] = float(distance_to_runner_threat)
        copied["distance_to_runner_path"] = float(distance_to_projected_path)
        copied["distance_to_pass_lane"] = float(distance_to_threat_pass_lane)
        copied["pass_lane_score"] = float(pass_lane_score)
        copied["pressure_score"] = float(pressure_score)
        copied["goal_side_score"] = float(goal_side_score)
        copied["follow_score"] = float(follow_score)
        copied["runner_threat_point_x"] = float(threat_x)
        copied["runner_threat_point_y"] = float(threat_y)
        copied["runner_closing_score"] = float(closing_score)
        copied["runner_responsibility"] = float(runner_responsibility)
        copied["runner_responsibility_score"] = float(runner_responsibility_score)
        copied["distance_to_target_option"] = float(target_distance)
        copied["target_option_bias_m"] = float(target_option_bias_m)
        copied["target_assignment_score"] = float("nan")
        copied["target_biased_without_threat"] = False
        copied["lane_or_passer_biased"] = False
        copied["far_without_meaningful_threat"] = False
        direct_candidates.append(copied)

    if not direct_candidates:
        return []

    nearest = sorted(
        direct_candidates,
        key=lambda item: (
            min(
                float(item.get("distance_to_runner_threat_point_m", float("inf"))),
                float(item.get("distance_to_runner_path", float("inf"))) + 1.5,
            ),
            float(item.get("distance_to_runner_threat_point_m", float("inf"))),
        ),
    )[: max(1, nearest_n)]

    max_threat = max(float(item.get("raw_threat_contribution", 0.0)) for item in nearest)
    out = []
    for rank, copied in enumerate(nearest, start=1):
        threat = float(copied.get("raw_threat_contribution", 0.0))
        threat_norm = threat / max_threat if max_threat > 1e-12 else 0.0
        distance_to_runner_threat = float(copied.get("distance_to_runner_threat_point_m", float("inf")))
        distance_to_projected_path = float(copied.get("distance_to_runner_path", float("inf")))
        proximity_score = (
            math.exp(-distance_to_runner_threat / 6.0)
            if math.isfinite(distance_to_runner_threat)
            else 0.0
        )
        path_score = (
            math.exp(-0.5 * (distance_to_projected_path / 5.0) ** 2)
            if math.isfinite(distance_to_projected_path)
            else 0.0
        )
        follow_score = float(copied.get("follow_score", 0.0))
        closing_score = float(copied.get("runner_closing_score", 0.0))
        goal_side_score = float(copied.get("goal_side_score", 0.0))
        responsibility_score = float(copied.get("runner_responsibility_score", 0.0))
        assignment_gate = 0.65 + 0.35 * responsibility_score
        copied["runner_proximity_rank"] = rank
        copied["runner_assignment_gate"] = float(assignment_gate)
        copied["runner_assignment_plausible"] = True
        copied["runner_direct_score"] = assignment_gate * (
            0.28 * proximity_score
            + 0.22 * path_score
            + 0.15 * follow_score
            + 0.15 * closing_score
            + 0.12 * goal_side_score
            + 0.08 * threat_norm
        )
        out.append(copied)

    return sorted(
        out,
        key=lambda item: (
            float(item["runner_direct_score"]),
            float(item.get("raw_threat_contribution", 0.0)),
            float(item.get("runner_responsibility_score", 0.0)),
            float(item.get("goal_side_score", 0.0)),
            -float(item.get("distance_to_receiver", float("inf"))),
        ),
        reverse=True,
    )


def pass_lane_suppression_attributions(
    attributions: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    max_threat = max((float(item.get("raw_threat_contribution", 0.0)) for item in attributions), default=0.0)
    out = []
    for item in attributions:
        copied = dict(item)
        threat = float(copied.get("raw_threat_contribution", 0.0))
        threat_norm = threat / max_threat if max_threat > 1e-12 else 0.0
        lane_score = float(copied.get("pass_lane_score", 0.0))
        copied["pass_lane_suppressor_score"] = 0.75 * lane_score + 0.25 * threat_norm
        out.append(copied)
    return sorted(
        out,
        key=lambda item: (
            float(item["pass_lane_suppressor_score"]),
            -float(item.get("distance_to_pass_lane", float("inf"))),
        ),
        reverse=True,
    )


def passer_pressure_attributions(
    attributions: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    max_threat = max((float(item.get("raw_threat_contribution", 0.0)) for item in attributions), default=0.0)
    out = []
    for item in attributions:
        copied = dict(item)
        threat = float(copied.get("raw_threat_contribution", 0.0))
        threat_norm = threat / max_threat if max_threat > 1e-12 else 0.0
        pressure_score = float(copied.get("passer_pressure_score", 0.0))
        copied["passer_pressure_defender_score"] = 0.80 * pressure_score + 0.20 * threat_norm
        out.append(copied)
    return sorted(
        out,
        key=lambda item: (
            float(item["passer_pressure_defender_score"]),
            -float(item.get("distance_to_passer", float("inf"))),
        ),
        reverse=True,
    )



def defender_target_metric_key(args) -> str:
    if args.defender_target_metric == "adjusted":
        return f"adjusted_{lambda_key(args.defender_target_lambda)}"
    return "obso"


def event_ball_xy(
    event_row,
    frame: BundesligaFrame,
    source: str = "passer",
) -> tuple[float, float] | None:
    if source == "passer":
        passer_id = event_row.player_id
        if passer_id in frame.players:
            passer = frame.players[passer_id]
            return passer.x, passer.y
    if source == "tracking-ball" and frame.ball is not None:
        return frame.ball.x, frame.ball.y
    if source == "event" and pd.notna(event_row.x_start) and pd.notna(event_row.y_start):
        return float(event_row.x_start), float(event_row.y_start)
    if frame.ball is not None:
        return frame.ball.x, frame.ball.y
    if pd.notna(event_row.x_start) and pd.notna(event_row.y_start):
        return float(event_row.x_start), float(event_row.y_start)
    return None


def filter_pass_events(events: pd.DataFrame, successful_only: bool, open_play_only: bool) -> pd.DataFrame:
    passes = events[
        events["event_type"].isin(["pass", "cross"])
        & events["frame_id"].notna()
        & events["team_id"].notna()
        & events["player_id"].notna()
    ].copy()
    if successful_only:
        passes = passes[passes["evaluation"] == "successfullyCompleted"]
    if open_play_only:
        passes = passes[passes["from_open_play"] == True]  # noqa: E712
    return passes.sort_values(["period", "frame_id", "event_id"]).reset_index(drop=True)


def run_match(args) -> pd.DataFrame:
    files = find_bundesliga_files(args.data_dir, args.match_id)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])

    print(f"Match: {metadata.home_team_name} vs {metadata.away_team_name} ({metadata.match_id})")
    print("Reading frame clock...")
    clock = load_bundesliga_frame_clock(files["positions"])

    print("Reading events...")
    events = load_bundesliga_events(files["events"], clock=clock)
    pass_events = filter_pass_events(
        events,
        successful_only=args.successful_only,
        open_play_only=args.open_play_only,
    )
    if args.max_events is not None:
        pass_events = pass_events.head(args.max_events)
    print(f"Selected {len(pass_events)} pass/cross events")

    target_frames = set()
    for row in pass_events.itertuples():
        event_frame = int(row.frame_id)
        target_frames.add(event_frame)
        target_frames.add(event_frame - args.horizon_frames)

    print(f"Reading {len(target_frames)} target frames from positions XML...")
    frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)
    print(f"Loaded {len(frames)} frames with player/ball states")

    results = []
    for event in pass_events.itertuples():
        event_frame_id = int(event.frame_id)
        start_frame_id = event_frame_id - args.horizon_frames
        start_frame = frames.get(start_frame_id)
        end_frame = frames.get(event_frame_id)
        if start_frame is None or end_frame is None:
            continue

        ball_xy = event_ball_xy(event, end_frame, source=args.ball_location_source)
        if ball_xy is None:
            continue

        team_id = event.team_id
        attacking_direction = infer_attacking_direction(end_frame, team_id, metadata)
        passer_id = event.player_id
        recipient_id = event.recipient_id if isinstance(event.recipient_id, str) else None

        for runner_id, runner_end in end_frame.players.items():
            if runner_end.team_id != team_id:
                continue
            if runner_id in {passer_id, recipient_id} or is_goalkeeper(metadata, runner_id):
                continue
            runner_start = start_frame.players.get(runner_id)
            if runner_start is None:
                continue

            movement = euclidean(runner_start.x, runner_start.y, runner_end.x, runner_end.y)
            if movement < args.min_run_distance:
                continue

            excluded = {passer_id, runner_id}
            candidate_ids = candidate_receiver_ids(
                end_frame,
                metadata,
                team_id,
                excluded,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                min_receiver_ahead=args.min_receiver_ahead,
                min_receiver_score=args.min_receiver_score,
                max_receiver_goal_distance=args.max_receiver_goal_distance,
                filter_offside=not args.include_offside_candidates,
            )
            if not candidate_ids:
                continue

            actual_metrics = receiver_option_metrics(
                end_frame,
                team_id,
                attacking_direction,
                candidate_ids,
                ball_xy,
                args.local_radius,
                args.local_samples,
                args.space_radius,
                args.space_samples,
                args.space_rings,
                args.adjusted_lambdas,
            )
            actual_value, actual_top = topk_metric(actual_metrics, "obso", args.top_k_options)
            actual_space_value, actual_space_top = topk_metric(actual_metrics, "post_space_m2", args.top_k_options)
            actual_dangerous_space_value, actual_dangerous_space_top = topk_metric(
                actual_metrics,
                "dangerous_space",
                args.top_k_options,
            )
            defender_target_key = defender_target_metric_key(args)
            defender_target_options = topk_metric(actual_metrics, defender_target_key, 1)[1]
            defender_target_receiver_id = defender_target_options[0][0] if defender_target_options else None
            defender_target_receiver_value = defender_target_options[0][1] if defender_target_options else None
            defender_responsibilities = (
                defender_responsibilities_for_receiver(
                    end_frame,
                    team_id,
                    defender_target_receiver_id,
                    ball_xy,
                    attacking_direction,
                )
                if defender_target_receiver_id is not None
                else []
            )
            affected_defender = affected_defender_for_runner(
                start_frame,
                end_frame,
                runner_id,
                target_receiver_id=defender_target_receiver_id,
                attacking_team_id=team_id,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                defender_selection=args.defender_selection,
            )
            runner_suppression = receiver_defensive_suppression_attribution(
                start_frame,
                end_frame,
                runner_id,
                team_id,
                attacking_direction,
                ball_xy,
                local_radius=args.local_radius,
                local_samples=args.local_samples,
                space_radius=args.space_radius,
                space_samples=args.space_samples,
                space_rings=args.space_rings,
                threat_lambda=args.runner_threat_lambda,
                filter_offside=not args.include_offside_candidates,
            )
            all_suppression_attributions = list(runner_suppression["attributions"])
            suppression_attributions = [
                item
                for item in all_suppression_attributions
                if float(item["allocated_contribution"]) > 1e-12
                or float(item["raw_contribution"]) > 1e-12
                or float(item.get("allocated_threat_contribution", 0.0)) > 1e-12
                or float(item.get("raw_threat_contribution", 0.0)) > 1e-12
            ]
            runner_affected_suppression = runner_affected_suppression_attributions(
                all_suppression_attributions,
                nearest_n=args.runner_affected_nearest_n,
                start_frame=start_frame,
                end_frame=end_frame,
                attacking_team_id=team_id,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                target_receiver_id=defender_target_receiver_id,
                runner_id=runner_id,
            )
            pass_lane_suppression = pass_lane_suppression_attributions(all_suppression_attributions)
            passer_pressure = passer_pressure_attributions(all_suppression_attributions)
            top_suppression_defender = suppression_attributions[0] if suppression_attributions else None
            top_suppression_defender_id = (
                str(top_suppression_defender["defender_id"])
                if top_suppression_defender is not None
                else None
            )
            top_runner_affected_defender = (
                runner_affected_suppression[0] if runner_affected_suppression else None
            )
            top_runner_affected_defender_id = (
                str(top_runner_affected_defender["defender_id"])
                if top_runner_affected_defender is not None
                else None
            )
            pass_lane_suppressor = pass_lane_suppression[0] if pass_lane_suppression else None
            pass_lane_suppressor_id = (
                str(pass_lane_suppressor["defender_id"])
                if pass_lane_suppressor is not None
                else None
            )
            passer_pressure_defender = passer_pressure[0] if passer_pressure else None
            passer_pressure_defender_id = (
                str(passer_pressure_defender["defender_id"])
                if passer_pressure_defender is not None
                else None
            )
            cf_affected_defender_id = top_runner_affected_defender_id

            cf_frame = counterfactual_frame_for_runner(
                start_frame,
                end_frame,
                runner_id,
                runner_alpha=args.runner_alpha,
                defender_alpha=args.defender_alpha,
                affected_defender_id=cf_affected_defender_id,
                target_receiver_id=defender_target_receiver_id,
                attacking_team_id=team_id,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                defender_selection=args.defender_selection,
                fallback_to_selected_defender=False,
            )
            cf_candidate_ids = candidate_receiver_ids(
                cf_frame,
                metadata,
                team_id,
                excluded,
                ball_xy=ball_xy,
                attacking_direction=attacking_direction,
                min_receiver_ahead=args.min_receiver_ahead,
                min_receiver_score=args.min_receiver_score,
                max_receiver_goal_distance=args.max_receiver_goal_distance,
                filter_offside=not args.include_offside_candidates,
            )
            if cf_candidate_ids:
                cf_metrics = receiver_option_metrics(
                    cf_frame,
                    team_id,
                    attacking_direction,
                    cf_candidate_ids,
                    ball_xy,
                    args.local_radius,
                    args.local_samples,
                    args.space_radius,
                    args.space_samples,
                    args.space_rings,
                    args.adjusted_lambdas,
                )
                cf_value, cf_top = topk_metric(cf_metrics, "obso", args.top_k_options)
                cf_space_value, cf_space_top = topk_metric(cf_metrics, "post_space_m2", args.top_k_options)
                cf_dangerous_space_value, cf_dangerous_space_top = topk_metric(
                    cf_metrics,
                    "dangerous_space",
                    args.top_k_options,
                )
            else:
                cf_metrics = []
                cf_value, cf_top = 0.0, []
                cf_space_value, cf_space_top = 0.0, []
                cf_dangerous_space_value, cf_dangerous_space_top = 0.0, []

            runner_threat_point = projected_runner_threat_point(start_frame, end_frame, runner_id)

            row = {
                    "match_id": metadata.match_id,
                    "event_id": event.event_id,
                    "period": event.period,
                    "event_frame": event_frame_id,
                    "start_frame": start_frame_id,
                    "event_type": event.event_type,
                    "team_id": team_id,
                    "team_role": metadata.team_role(team_id),
                    "passer_id": passer_id,
                    "passer_name": player_label(metadata, passer_id),
                    "recipient_id": recipient_id,
                    "recipient_name": player_label(metadata, recipient_id),
                    "runner_id": runner_id,
                    "runner_name": player_label(metadata, runner_id),
                    "runner_threat_point_x": (
                        float(runner_threat_point[0]) if runner_threat_point is not None else None
                    ),
                    "runner_threat_point_y": (
                        float(runner_threat_point[1]) if runner_threat_point is not None else None
                    ),
                    "affected_defender_id": affected_defender.object_id if affected_defender is not None else None,
                    "affected_defender_name": (
                        player_label(metadata, affected_defender.object_id)
                        if affected_defender is not None
                        else None
                    ),
                    "affected_defender_distance_to_runner_m": (
                        euclidean(affected_defender.x, affected_defender.y, runner_end.x, runner_end.y)
                        if affected_defender is not None
                        else None
                    ),
                    "defender_selection": args.defender_selection,
                    "defender_target_metric": args.defender_target_metric,
                    "defender_target_lambda": (
                        args.defender_target_lambda if args.defender_target_metric == "adjusted" else None
                    ),
                    "defender_target_receiver_id": defender_target_receiver_id,
                    "defender_target_receiver_name": player_label(metadata, defender_target_receiver_id),
                    "defender_target_receiver_value": defender_target_receiver_value,
                    "defender_responsibility_top": format_defender_responsibilities(
                        metadata,
                        defender_responsibilities,
                        limit=args.defender_responsibility_top_n,
                    ),
                    "runner_actual_obso": runner_suppression["actual_value"],
                    "runner_no_response_obso": runner_suppression["no_response_value"],
                    "runner_defensive_suppression_obso": runner_suppression["suppression"],
                    "runner_defensive_suppression_positive_obso": runner_suppression["positive_suppression"],
                    "runner_actual_threat": runner_suppression["actual_threat_value"],
                    "runner_no_response_threat": runner_suppression["no_response_threat_value"],
                    "runner_threat_suppression": runner_suppression["threat_suppression"],
                    "runner_threat_suppression_positive": runner_suppression["positive_threat_suppression"],
                    "runner_threat_lambda": runner_suppression["threat_lambda"],
                    "runner_actual_offside": runner_suppression["actual_offside"],
                    "runner_no_response_offside": runner_suppression["no_response_offside"],
                    "top_suppression_defender_id": top_suppression_defender_id,
                    "top_suppression_defender_name": player_label(metadata, top_suppression_defender_id),
                    "top_suppression_defender_attribution_obso": (
                        float(top_suppression_defender["allocated_contribution"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_raw_obso": (
                        float(top_suppression_defender["raw_contribution"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_threat_attribution": (
                        float(top_suppression_defender["allocated_threat_contribution"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_raw_threat": (
                        float(top_suppression_defender["raw_threat_contribution"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_released_runner_threat": (
                        float(top_suppression_defender["released_threat_value"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_released_runner_obso": (
                        float(top_suppression_defender["released_value"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_distance_to_runner_m": (
                        float(top_suppression_defender["distance_to_receiver"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_suppression_defender_distance_to_runner_path_m": (
                        float(top_suppression_defender["distance_to_runner_path"])
                        if top_suppression_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_id": top_runner_affected_defender_id,
                    "top_runner_affected_defender_name": (
                        player_label(metadata, top_runner_affected_defender_id)
                    ),
                    "top_runner_affected_defender_raw_obso": (
                        float(top_runner_affected_defender["raw_contribution"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_score": (
                        float(top_runner_affected_defender["runner_direct_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_raw_threat": (
                        float(top_runner_affected_defender["raw_threat_contribution"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_released_runner_threat": (
                        float(top_runner_affected_defender["released_threat_value"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_goal_side_score": (
                        float(top_runner_affected_defender["goal_side_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_goal_side_distance_m": (
                        float(top_runner_affected_defender["goal_side_distance"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_proximity_rank": (
                        int(top_runner_affected_defender["runner_proximity_rank"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_released_runner_obso": (
                        float(top_runner_affected_defender["released_value"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_distance_to_runner_m": (
                        float(top_runner_affected_defender["distance_to_receiver"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_distance_to_runner_end_m": (
                        float(top_runner_affected_defender["distance_to_runner_end_m"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_distance_to_runner_threat_point_m": (
                        float(top_runner_affected_defender["distance_to_runner_threat_point_m"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_distance_to_runner_path_m": (
                        float(top_runner_affected_defender["distance_to_runner_path"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_runner_closing_score": (
                        float(top_runner_affected_defender["runner_closing_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_runner_responsibility_score": (
                        float(top_runner_affected_defender["runner_responsibility_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_follow_score": (
                        float(top_runner_affected_defender["follow_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_distance_to_target_option_m": (
                        float(top_runner_affected_defender["distance_to_target_option"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_target_option_bias_m": (
                        float(top_runner_affected_defender["target_option_bias_m"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_target_assignment_score": (
                        float(top_runner_affected_defender["target_assignment_score"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_runner_assignment_gate": (
                        float(top_runner_affected_defender["runner_assignment_gate"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_runner_assignment_plausible": (
                        bool(top_runner_affected_defender["runner_assignment_plausible"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_target_biased_without_threat": (
                        bool(top_runner_affected_defender["target_biased_without_threat"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_lane_or_passer_biased": (
                        bool(top_runner_affected_defender["lane_or_passer_biased"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "top_runner_affected_defender_far_without_meaningful_threat": (
                        bool(top_runner_affected_defender["far_without_meaningful_threat"])
                        if top_runner_affected_defender is not None
                        else None
                    ),
                    "pass_lane_suppressor_id": pass_lane_suppressor_id,
                    "pass_lane_suppressor_name": player_label(metadata, pass_lane_suppressor_id),
                    "pass_lane_suppressor_score": (
                        float(pass_lane_suppressor["pass_lane_suppressor_score"])
                        if pass_lane_suppressor is not None
                        else None
                    ),
                    "pass_lane_suppressor_distance_to_pass_lane_m": (
                        float(pass_lane_suppressor["distance_to_pass_lane"])
                        if pass_lane_suppressor is not None
                        else None
                    ),
                    "pass_lane_suppressor_distance_to_runner_m": (
                        float(pass_lane_suppressor["distance_to_receiver"])
                        if pass_lane_suppressor is not None
                        else None
                    ),
                    "pass_lane_suppressor_raw_threat": (
                        float(pass_lane_suppressor["raw_threat_contribution"])
                        if pass_lane_suppressor is not None
                        else None
                    ),
                    "passer_pressure_defender_id": passer_pressure_defender_id,
                    "passer_pressure_defender_name": player_label(metadata, passer_pressure_defender_id),
                    "passer_pressure_defender_score": (
                        float(passer_pressure_defender["passer_pressure_defender_score"])
                        if passer_pressure_defender is not None
                        else None
                    ),
                    "passer_pressure_defender_distance_to_passer_m": (
                        float(passer_pressure_defender["distance_to_passer"])
                        if passer_pressure_defender is not None
                        else None
                    ),
                    "passer_pressure_defender_distance_to_runner_m": (
                        float(passer_pressure_defender["distance_to_receiver"])
                        if passer_pressure_defender is not None
                        else None
                    ),
                    "passer_pressure_defender_raw_threat": (
                        float(passer_pressure_defender["raw_threat_contribution"])
                        if passer_pressure_defender is not None
                        else None
                    ),
                    "runner_affected_suppression_top": format_defender_role_attributions(
                        metadata,
                        runner_affected_suppression,
                        score_key="runner_direct_score",
                        limit=args.defensive_suppression_top_n,
                    ),
                    "pass_lane_suppression_top": format_defender_role_attributions(
                        metadata,
                        pass_lane_suppression,
                        score_key="pass_lane_suppressor_score",
                        limit=args.defensive_suppression_top_n,
                    ),
                    "passer_pressure_top": format_defender_role_attributions(
                        metadata,
                        passer_pressure,
                        score_key="passer_pressure_defender_score",
                        limit=args.defensive_suppression_top_n,
                    ),
                    "defensive_suppression_top": format_defensive_suppression_attributions(
                        metadata,
                        suppression_attributions,
                        limit=args.defensive_suppression_top_n,
                    ),
                    "runner_start_x": runner_start.x,
                    "runner_start_y": runner_start.y,
                    "runner_end_x": runner_end.x,
                    "runner_end_y": runner_end.y,
                    "runner_progress_m": attacking_direction * (runner_end.x - runner_start.x),
                    "runner_movement_m": movement,
                    "ball_x": ball_xy[0],
                    "ball_y": ball_xy[1],
                    "ball_location_source": args.ball_location_source,
                    "pass_end_x": float(event.x_end) if pd.notna(event.x_end) else None,
                    "pass_end_y": float(event.y_end) if pd.notna(event.y_end) else None,
                    "attacking_direction": attacking_direction,
                    "actual_team_obso_topk": actual_value,
                    "counterfactual_team_obso_topk": cf_value,
                    "draft_obso_offball_value": actual_value - cf_value,
                    "actual_team_post_reception_space_topk_m2": actual_space_value,
                    "counterfactual_team_post_reception_space_topk_m2": cf_space_value,
                    "draft_post_reception_space_offball_value_m2": actual_space_value - cf_space_value,
                    "actual_team_post_reception_dangerous_space_topk": actual_dangerous_space_value,
                    "counterfactual_team_post_reception_dangerous_space_topk": cf_dangerous_space_value,
                    "draft_post_reception_dangerous_space_offball_value": (
                        actual_dangerous_space_value - cf_dangerous_space_value
                    ),
                    "actual_top_options": format_top_options(metadata, actual_top),
                    "counterfactual_top_options": format_top_options(metadata, cf_top),
                    "actual_top_post_reception_space_options": format_top_options(metadata, actual_space_top),
                    "counterfactual_top_post_reception_space_options": format_top_options(metadata, cf_space_top),
                    "actual_top_post_reception_dangerous_space_options": format_top_options(
                        metadata,
                        actual_dangerous_space_top,
                    ),
                    "counterfactual_top_post_reception_dangerous_space_options": format_top_options(
                        metadata,
                        cf_dangerous_space_top,
                    ),
                    "offside_filter": not args.include_offside_candidates,
                }
            for lam in args.adjusted_lambdas:
                key = lambda_key(lam)
                metric_key = f"adjusted_{key}"
                actual_adjusted_value, actual_adjusted_top = topk_metric(
                    actual_metrics,
                    metric_key,
                    args.top_k_options,
                )
                if cf_metrics:
                    cf_adjusted_value, cf_adjusted_top = topk_metric(
                        cf_metrics,
                        metric_key,
                        args.top_k_options,
                    )
                else:
                    cf_adjusted_value, cf_adjusted_top = 0.0, []
                row[f"actual_team_adjusted_option_topk_{key}"] = actual_adjusted_value
                row[f"counterfactual_team_adjusted_option_topk_{key}"] = cf_adjusted_value
                row[f"draft_adjusted_option_offball_value_{key}"] = (
                    actual_adjusted_value - cf_adjusted_value
                )
                row[f"actual_top_adjusted_options_{key}"] = format_top_options(
                    metadata,
                    actual_adjusted_top,
                )
                row[f"counterfactual_top_adjusted_options_{key}"] = format_top_options(
                    metadata,
                    cf_adjusted_top,
                )

            results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    return out.sort_values("draft_obso_offball_value", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a first Bundesliga OBSO-style off-ball counterfactual demo."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--match-id", type=str, default="J03WMX")
    parser.add_argument("--max-events", type=int, default=80, help="Pass/cross event cap per match. Use 0 for all.")
    parser.add_argument("--horizon-frames", type=int, default=25)
    parser.add_argument("--min-run-distance", type=float, default=4.0)
    parser.add_argument("--top-k-options", type=int, default=3)
    parser.add_argument("--local-radius", type=float, default=3.0)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--space-radius", type=float, default=5.0)
    parser.add_argument("--space-samples", type=int, default=16)
    parser.add_argument("--space-rings", type=int, default=3)
    parser.add_argument(
        "--adjusted-lambdas",
        type=parse_lambdas,
        default=parse_lambdas("0.2,0.5,1.0"),
        help="Comma-separated lambdas for OBSO + lambda * dangerous_space_norm.",
    )
    parser.add_argument("--min-receiver-ahead", type=float, default=None)
    parser.add_argument("--min-receiver-score", type=float, default=None)
    parser.add_argument("--max-receiver-goal-distance", type=float, default=None)
    parser.add_argument("--ball-location-source", choices=["passer", "event", "tracking-ball"], default="passer")
    parser.add_argument("--runner-alpha", type=float, default=0.0)
    parser.add_argument("--defender-alpha", type=float, default=0.3)
    parser.add_argument(
        "--defender-selection",
        choices=["closest-runner", "responsibility"],
        default="responsibility",
        help="Select affected defender by runner proximity or DEFCON-inspired responsibility proxy.",
    )
    parser.add_argument(
        "--defender-target-metric",
        choices=["obso", "adjusted"],
        default="adjusted",
        help="Metric used to choose the receiver option whose defenders get responsibility scores.",
    )
    parser.add_argument(
        "--defender-target-lambda",
        type=float,
        default=0.5,
        help="Adjusted-option lambda used when --defender-target-metric=adjusted.",
    )
    parser.add_argument("--defender-responsibility-top-n", type=int, default=5)
    parser.add_argument("--defensive-suppression-top-n", type=int, default=5)
    parser.add_argument(
        "--runner-threat-lambda",
        type=float,
        default=0.5,
        help="Lambda for runner threat = OBSO + lambda * normalized dangerous post-reception space.",
    )
    parser.add_argument(
        "--runner-affected-nearest-n",
        type=int,
        default=3,
        help="Choose direct runner defender from the N nearest defenders to the projected runner threat point/path.",
    )
    parser.add_argument(
        "--suppression-runner-max-distance",
        type=float,
        default=12.0,
        help="Deprecated; direct runner defender now uses --runner-affected-nearest-n.",
    )
    parser.add_argument(
        "--suppression-runner-path-max-distance",
        type=float,
        default=6.0,
        help="Deprecated; direct runner defender now uses --runner-affected-nearest-n.",
    )
    parser.add_argument("--top-k-output", type=int, default=30)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--all-passes", action="store_true")
    parser.add_argument("--include-set-pieces", action="store_true")
    parser.add_argument("--include-offside-candidates", action="store_true")
    args = parser.parse_args()

    if args.max_events is not None and args.max_events <= 0:
        args.max_events = None

    if args.defender_target_metric == "adjusted" and not any(
        math.isclose(args.defender_target_lambda, lam) for lam in args.adjusted_lambdas
    ):
        args.adjusted_lambdas = [*args.adjusted_lambdas, args.defender_target_lambda]

    args.successful_only = not args.all_passes
    args.open_play_only = not args.include_set_pieces

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if args.match_id.lower() == "all":
        outputs = []
        for match_id in list_bundesliga_match_ids(args.data_dir):
            match_args = copy(args)
            match_args.match_id = normalize_bundesliga_match_id(match_id)
            match_out = run_match(match_args)
            if not match_out.empty:
                outputs.append(match_out)
        out = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        out_name = f"bundesliga_obso_counterfactual_all{suffix}.csv"
    else:
        args.match_id = normalize_bundesliga_match_id(args.match_id)
        out = run_match(args)
        out_name = f"bundesliga_obso_counterfactual_{short_bundesliga_match_id(args.match_id)}{suffix}.csv"

    if out.empty:
        print("No OBSO off-ball candidates found.")
        return

    out = out.sort_values("draft_obso_offball_value", ascending=False).reset_index(drop=True)
    out_file = OUT_DIR / out_name
    out.to_csv(out_file, index=False)

    print("\nTop OBSO-based off-ball candidates:")
    cols = [
        "event_frame",
        "team_role",
        "passer_name",
        "recipient_name",
        "runner_name",
        "affected_defender_name",
        "top_runner_affected_defender_name",
        "pass_lane_suppressor_name",
        "passer_pressure_defender_name",
        "top_suppression_defender_name",
        "defender_target_receiver_name",
        "runner_movement_m",
        "runner_actual_obso",
        "runner_no_response_obso",
        "runner_defensive_suppression_positive_obso",
        "runner_actual_threat",
        "runner_no_response_threat",
        "runner_threat_suppression_positive",
        "actual_team_obso_topk",
        "counterfactual_team_obso_topk",
        "draft_obso_offball_value",
        "actual_team_post_reception_space_topk_m2",
        "counterfactual_team_post_reception_space_topk_m2",
        "draft_post_reception_space_offball_value_m2",
    ]
    for lam in args.adjusted_lambdas:
        cols.append(f"draft_adjusted_option_offball_value_{lambda_key(lam)}")
    print(out[cols].head(args.top_k_output).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {out_file}")


if __name__ == "__main__":
    main()
