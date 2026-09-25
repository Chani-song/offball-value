from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaMatchMeta,
    BundesligaObjectState,
    find_bundesliga_files,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
)
from offball_value.obso import (
    counterfactual_frame_for_runner,
    is_offside_position,
    receiver_option_value,
    receiver_post_reception_dangerous_space_value,
    receiver_post_reception_space_area,
    runner_receiving_threat_map,
    score_at_points,
    transition_at_points,
)


DEFAULT_RESULTS = ROOT / "data" / "processed" / "offball_results.csv"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"
DEFAULT_VISUALIZATIONS_DIR = ROOT / "data" / "processed" / "visualizations"


def player_name(metadata: BundesligaMatchMeta, player_id: str | None) -> str:
    if not player_id:
        return ""
    player = metadata.players.get(player_id)
    return player.short_name if player else player_id


def player_number(metadata: BundesligaMatchMeta, player_id: str | None) -> str:
    if not player_id:
        return ""
    player = metadata.players.get(player_id)
    if player is None:
        return player_id[-4:]
    return player.shirt_number or player.short_name[:5]


def compact_semicolon_names(value: object, limit: int = 2) -> str:
    if value is None or pd.isna(value):
        return "none"
    names = [name for name in str(value).split(";") if name]
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    return f"{shown} +{len(names) - limit}" if len(names) > limit else shown


def parse_top_option(value: str | float | None) -> tuple[str | None, str | None, float | None]:
    if value is None or pd.isna(value):
        return None, None, None
    first = str(value).split(";")[0]
    parts = first.split(":")
    if len(parts) != 3:
        return None, None, None
    try:
        score = float(parts[2])
    except ValueError:
        score = None
    return parts[0], parts[1], score


def parse_option_values(value: str | float | None) -> dict[str, float]:
    if value is None or pd.isna(value):
        return {}
    out = {}
    for item in str(value).split(";"):
        parts = item.split(":")
        if len(parts) != 3:
            continue
        try:
            out[parts[0]] = float(parts[2])
        except ValueError:
            continue
    return out


def fast_top_beneficiary_gain(
    row: pd.Series,
    adjusted_lambda: float,
) -> float:
    key = lambda_key(adjusted_lambda)
    actual = parse_option_values(row.get(f"actual_top_adjusted_options_{key}"))
    counterfactual = parse_option_values(row.get(f"counterfactual_top_adjusted_options_{key}"))
    if not actual:
        return 0.0
    return max(value - counterfactual.get(player_id, 0.0) for player_id, value in actual.items())


def lambda_key(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{text}"


def dangerous_space_norm(dangerous_space: float, space_radius: float) -> float:
    max_space_area = math.pi * space_radius**2 if space_radius > 0 else 1.0
    return max(0.0, min(1.0, dangerous_space / max_space_area))


def is_goalkeeper(metadata: BundesligaMatchMeta, player_id: str) -> bool:
    player = metadata.players.get(player_id)
    return player is not None and player.position == "TW"


def receiver_option_metrics(
    frame: BundesligaFrame,
    player_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
    adjusted_lambda: float,
) -> tuple[float, float, float, float] | None:
    value = receiver_option_value(
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
    if not all(math.isfinite(metric) for metric in (value, space, dangerous_space)):
        return None
    adjusted = value + adjusted_lambda * dangerous_space_norm(dangerous_space, space_radius)
    return value, space, dangerous_space, adjusted


def receiver_obso_values(
    frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    excluded_ids: set[str],
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
    adjusted_lambda: float,
    candidate_sort_by: str,
) -> list[tuple[str, float, float, float, float]]:
    values = []
    for player_id, player in frame.players.items():
        if player.team_id != attacking_team_id:
            continue
        if player_id in excluded_ids or is_goalkeeper(metadata, player_id):
            continue
        if is_offside_position(
            frame,
            player_id,
            attacking_team_id,
            ball_xy,
            attacking_direction,
        ):
            continue
        metrics = receiver_option_metrics(
            frame,
            player_id,
            attacking_team_id,
            attacking_direction,
            ball_xy,
            local_radius,
            local_samples,
            space_radius,
            space_samples,
            space_rings,
            adjusted_lambda,
        )
        if metrics is not None:
            values.append((player_id, *metrics))

    sort_index = {
        "obso": 1,
        "post-space": 2,
        "dangerous-space": 3,
        "adjusted": 4,
    }.get(candidate_sort_by, 1)
    return sorted(values, key=lambda item: item[sort_index], reverse=True)


def top_beneficiary_for_run(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    runner_id: str,
    passer_id: str,
    affected_defender_id: str | None,
    target_receiver_id: str | None,
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
    adjusted_lambda: float,
) -> dict[str, float | str | bool] | None:
    cf_frame = counterfactual_frame_for_runner(
        start_frame,
        end_frame,
        runner_id,
        runner_alpha=0.0,
        defender_alpha=0.3,
        affected_defender_id=affected_defender_id,
        target_receiver_id=target_receiver_id,
        attacking_team_id=attacking_team_id,
        ball_xy=ball_xy,
        attacking_direction=attacking_direction,
        defender_selection="responsibility",
        fallback_to_selected_defender=False,
    )
    excluded_ids = {passer_id, runner_id}
    actual_values = receiver_obso_values(
        end_frame,
        metadata,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        excluded_ids=excluded_ids,
        local_radius=local_radius,
        local_samples=local_samples,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
        adjusted_lambda=adjusted_lambda,
        candidate_sort_by="adjusted",
    )
    cf_values = receiver_obso_values(
        cf_frame,
        metadata,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        excluded_ids=excluded_ids,
        local_radius=local_radius,
        local_samples=local_samples,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
        adjusted_lambda=adjusted_lambda,
        candidate_sort_by="adjusted",
    )
    cf_by_id = {
        player_id: (obso, space, dangerous_space, adjusted)
        for player_id, obso, space, dangerous_space, adjusted in cf_values
    }
    best = None
    for player_id, obso, space, dangerous_space, adjusted in actual_values:
        actual_offside = is_offside_position(
            end_frame,
            player_id,
            attacking_team_id,
            ball_xy,
            attacking_direction,
        )
        counterfactual_offside = is_offside_position(
            cf_frame,
            player_id,
            attacking_team_id,
            ball_xy,
            attacking_direction,
        )
        cf_metrics = cf_by_id.get(player_id)
        if cf_metrics is None:
            cf_metrics = receiver_option_metrics(
                cf_frame,
                player_id,
                attacking_team_id,
                attacking_direction,
                ball_xy,
                local_radius,
                local_samples,
                space_radius,
                space_samples,
                space_rings,
                adjusted_lambda,
            )
        if cf_metrics is None:
            continue
        cf_raw_obso, cf_space, cf_dangerous_space, cf_raw_adjusted = cf_metrics
        cf_legal_obso = 0.0 if counterfactual_offside else cf_raw_obso
        cf_legal_adjusted = 0.0 if counterfactual_offside else cf_raw_adjusted
        item = {
            "player_id": player_id,
            "name": player_name(metadata, player_id),
            "actual_offside": actual_offside,
            "counterfactual_offside": counterfactual_offside,
            "actual_adjusted": float(adjusted),
            "counterfactual_adjusted": float(cf_legal_adjusted),
            "counterfactual_raw_adjusted": float(cf_raw_adjusted),
            "adjusted_gain": float(adjusted - cf_legal_adjusted),
            "actual_obso": float(obso),
            "counterfactual_obso": float(cf_legal_obso),
            "counterfactual_raw_obso": float(cf_raw_obso),
            "obso_gain": float(obso - cf_legal_obso),
            "actual_space_m2": float(space),
            "counterfactual_space_m2": float(cf_space),
            "space_gain_m2": float(space - cf_space),
            "actual_dangerous_space": float(dangerous_space),
            "counterfactual_dangerous_space": float(cf_dangerous_space),
            "dangerous_space_gain": float(dangerous_space - cf_dangerous_space),
        }
        if best is None or float(item["adjusted_gain"]) > float(best["adjusted_gain"]):
            best = item
    if best is None or float(best["adjusted_gain"]) <= 1e-9:
        return None
    target_id = str(best["player_id"])
    actual_context = defensive_context_for_target(
        end_frame,
        target_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        affected_defender_id,
    )
    cf_context = defensive_context_for_target(
        cf_frame,
        target_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        affected_defender_id,
    )
    best.update(
        {
            "nearest_defender_clearance_gain_m": gain_metric(
                actual_context,
                cf_context,
                "nearest_defender_distance_m",
            ),
            "close_defenders_5m_removed": reduction_metric(
                actual_context,
                cf_context,
                "close_defenders_5m",
            ),
            "receiver_pressure_reduction": reduction_metric(
                actual_context,
                cf_context,
                "receiver_pressure_sum",
            ),
            "pass_lane_pressure_reduction": reduction_metric(
                actual_context,
                cf_context,
                "pass_lane_pressure_sum",
            ),
            "goal_side_blocker_reduction": reduction_metric(
                actual_context,
                cf_context,
                "goal_side_blocker_sum",
            ),
            "affected_removed_from_beneficiary_m": gain_metric(
                actual_context,
                cf_context,
                "affected_distance_to_target_m",
            ),
            "affected_pressure_reduction_on_beneficiary": reduction_metric(
                actual_context,
                cf_context,
                "affected_pressure_on_target",
            ),
            "affected_pass_lane_reduction_to_beneficiary": reduction_metric(
                actual_context,
                cf_context,
                "affected_pass_lane_score_to_target",
            ),
            "affected_goal_side_reduction_to_beneficiary": reduction_metric(
                actual_context,
                cf_context,
                "affected_goal_side_score_to_target",
            ),
        }
    )
    best["primary_benefit_type"] = primary_benefit_type(best)
    return best


def gain_metric(actual: dict[str, float | int], counterfactual: dict[str, float | int], key: str) -> float:
    return float(actual.get(key, 0.0)) - float(counterfactual.get(key, 0.0))


def reduction_metric(actual: dict[str, float | int], counterfactual: dict[str, float | int], key: str) -> float:
    return float(counterfactual.get(key, 0.0)) - float(actual.get(key, 0.0))


def defensive_context_for_target(
    frame: BundesligaFrame,
    target_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    affected_defender_id: str | None,
) -> dict[str, float | int]:
    target = frame.players.get(target_id)
    if target is None:
        return {}

    target_xy = (target.x, target.y)
    goal_xy = (attacking_direction * FIELD_LENGTH / 2.0, 0.0)
    nearest = float("inf")
    pressure_sum = 0.0
    pass_lane_sum = 0.0
    goal_side_sum = 0.0
    close_5m = 0
    affected_distance = float("nan")
    affected_pressure = 0.0
    affected_lane = 0.0
    affected_goal_side = 0.0

    for defender in frame.players.values():
        if defender.team_id == attacking_team_id:
            continue
        defender_xy = (defender.x, defender.y)
        d_target = math.hypot(defender.x - target.x, defender.y - target.y)
        nearest = min(nearest, d_target)
        pressure = math.exp(-d_target / 5.0)
        pressure_sum += pressure
        close_5m += int(d_target <= 5.0)

        lane_distance, lane_projection = point_to_segment_distance(defender_xy, ball_xy, target_xy)
        lane_score = 0.0
        if 0.0 <= lane_projection <= 1.0:
            lane_score = math.exp(-0.5 * (lane_distance / 4.0) ** 2)
            pass_lane_sum += lane_score

        goal_distance, goal_projection = point_to_segment_distance(defender_xy, target_xy, goal_xy)
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
            goal_side_sum += goal_score

        if affected_defender_id and defender.object_id == affected_defender_id:
            affected_distance = d_target
            affected_pressure = pressure
            affected_lane = lane_score
            affected_goal_side = goal_score

    return {
        "nearest_defender_distance_m": float(nearest),
        "receiver_pressure_sum": float(pressure_sum),
        "pass_lane_pressure_sum": float(pass_lane_sum),
        "goal_side_blocker_sum": float(goal_side_sum),
        "close_defenders_5m": int(close_5m),
        "affected_distance_to_target_m": float(affected_distance),
        "affected_pressure_on_target": float(affected_pressure),
        "affected_pass_lane_score_to_target": float(affected_lane),
        "affected_goal_side_score_to_target": float(affected_goal_side),
    }


def primary_benefit_type(item: dict[str, float | str | bool]) -> str:
    if bool(item.get("counterfactual_offside")) and not bool(item.get("actual_offside")):
        return "offside_line_gain"

    scores = {
        "receiver_opened": max(
            0.0,
            float(item.get("nearest_defender_clearance_gain_m", 0.0)) / 3.0,
            float(item.get("receiver_pressure_reduction", 0.0)),
        ),
        "pass_lane_opened": max(0.0, float(item.get("pass_lane_pressure_reduction", 0.0))),
        "goal_side_blocker_removed": max(0.0, float(item.get("goal_side_blocker_reduction", 0.0))),
        "direct_defender_pulled": max(0.0, float(item.get("affected_removed_from_beneficiary_m", 0.0)) / 4.0),
        "space_created": max(0.0, float(item.get("space_gain_m2", 0.0)) / 20.0),
        "obso_quality_gain": max(0.0, float(item.get("obso_gain", 0.0)) / 0.02),
    }
    label, score = max(scores.items(), key=lambda value: value[1])
    return label if score > 0.05 else "mixed_or_small"


def readable_benefit_type(value: str | None) -> str:
    labels = {
        "offside_line_gain": "offside line gain",
        "receiver_opened": "receiver opened",
        "pass_lane_opened": "pass lane opened",
        "goal_side_blocker_removed": "goal-side blocker removed",
        "direct_defender_pulled": "direct defender pulled",
        "space_created": "post-reception space created",
        "obso_quality_gain": "OBSO quality gain",
        "mixed_or_small": "mixed/small effects",
    }
    return labels.get(value or "", value or "")


def draw_pitch(ax) -> None:
    half_l = FIELD_LENGTH / 2.0
    half_w = FIELD_WIDTH / 2.0
    ax.set_xlim(-half_l, half_l)
    ax.set_ylim(-half_w, half_w)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#eef4ec")
    ax.axis("off")

    line = "#526157"
    lw = 1.0
    ax.plot([-half_l, half_l, half_l, -half_l, -half_l], [-half_w, -half_w, half_w, half_w, -half_w], color=line, lw=lw)
    ax.plot([0, 0], [-half_w, half_w], color=line, lw=lw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color=line, lw=lw))

    box_depth = 16.5
    box_width = 40.32
    six_depth = 5.5
    six_width = 18.32
    for side in [-1, 1]:
        goal_x = side * half_l
        rect_x = goal_x - side * box_depth if side > 0 else goal_x
        six_x = goal_x - side * six_depth if side > 0 else goal_x
        ax.add_patch(plt.Rectangle((rect_x, -box_width / 2), box_depth, box_width, fill=False, color=line, lw=lw))
        ax.add_patch(plt.Rectangle((six_x, -six_width / 2), six_depth, six_width, fill=False, color=line, lw=lw))


def state_xy(state: BundesligaObjectState | None) -> tuple[float, float] | None:
    if state is None:
        return None
    return state.x, state.y


def expand_interval(lo: float, hi: float, min_span: float, lower_bound: float, upper_bound: float) -> tuple[float, float]:
    span = hi - lo
    if span < min_span:
        center = (lo + hi) / 2.0
        lo = center - min_span / 2.0
        hi = center + min_span / 2.0
    if lo < lower_bound:
        hi = min(upper_bound, hi + (lower_bound - lo))
        lo = lower_bound
    if hi > upper_bound:
        lo = max(lower_bound, lo - (hi - upper_bound))
        hi = upper_bound
    return lo, hi


def apply_action_view(ax, points: list[tuple[float, float] | None]) -> None:
    finite_points = [
        (x, y)
        for point in points
        if point is not None
        for x, y in [point]
        if math.isfinite(x) and math.isfinite(y)
    ]
    if not finite_points:
        return

    half_l = FIELD_LENGTH / 2.0
    half_w = FIELD_WIDTH / 2.0
    xs = [point[0] for point in finite_points]
    ys = [point[1] for point in finite_points]
    x0, x1 = max(-half_l, min(xs) - 12.0), min(half_l, max(xs) + 12.0)
    y0, y1 = max(-half_w, min(ys) - 9.0), min(half_w, max(ys) + 9.0)
    x0, x1 = expand_interval(x0, x1, min_span=42.0, lower_bound=-half_l, upper_bound=half_l)
    y0, y1 = expand_interval(y0, y1, min_span=28.0, lower_bound=-half_w, upper_bound=half_w)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def apply_half_view(
    ax,
    points: list[tuple[float, float] | None],
    attacking_direction: int,
) -> None:
    half_l = FIELD_LENGTH / 2.0
    half_w = FIELD_WIDTH / 2.0
    if attacking_direction >= 0:
        x0, x1 = 0.0, half_l
    else:
        x0, x1 = -half_l, 0.0

    finite_points = [
        (x, y)
        for point in points
        if point is not None
        for x, y in [point]
        if math.isfinite(x) and math.isfinite(y)
    ]
    if finite_points:
        xs = [point[0] for point in finite_points]
        margin = 5.0
        x0 = min(x0, max(-half_l, min(xs) - margin))
        x1 = max(x1, min(half_l, max(xs) + margin))
        x0, x1 = expand_interval(x0, x1, min_span=FIELD_LENGTH / 2.0, lower_bound=-half_l, upper_bound=half_l)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-half_w, half_w)


def closest_point_on_segment(
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
        return start
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    return sx + t * dx, sy + t * dy


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
        return math.hypot(px - sx, py - sy), 0.0
    t = ((px - sx) * dx + (py - sy) * dy) / denom
    clamped = max(0.0, min(1.0, t))
    cx = sx + clamped * dx
    cy = sy + clamped * dy
    return math.hypot(px - cx, py - cy), float(t)


def points_to_segment_distance_and_t(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = end - start
    denom = float(delta @ delta)
    if denom <= 1e-9:
        distance = np.linalg.norm(points - start, axis=1)
        return distance, np.zeros(len(points), dtype=float)
    t = ((points - start) @ delta) / denom
    clamped = np.clip(t, 0.0, 1.0)
    closest = start + clamped[:, None] * delta
    distance = np.linalg.norm(points - closest, axis=1)
    return distance, t


def _unit_or_none(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return vector / norm


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def runner_intent_prior_at_points(
    runner_start: BundesligaObjectState,
    runner_end: BundesligaObjectState,
    points: np.ndarray,
    attacking_direction: int,
    max_ahead: float,
    lateral_sigma: float,
    goal_mix: float,
    runner_threat_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Future receiving prior from the pass frame, using the observed run direction."""
    runner_end_xy = np.asarray([runner_end.x, runner_end.y], dtype=float)
    runner_vec = np.asarray([runner_end.x - runner_start.x, runner_end.y - runner_start.y], dtype=float)
    inertia_unit = _unit_or_none(runner_vec)
    if inertia_unit is None:
        inertia_unit = np.asarray([float(attacking_direction), 0.0], dtype=float)

    goal_xy = np.asarray([attacking_direction * FIELD_LENGTH / 2.0, 0.0], dtype=float)
    goal_unit = _unit_or_none(goal_xy - runner_end_xy)
    if goal_unit is None:
        goal_unit = inertia_unit

    if runner_threat_point is not None:
        threat_xy = np.asarray(runner_threat_point, dtype=float)
        threat_unit = _unit_or_none(threat_xy - runner_end_xy)
        if threat_unit is None:
            threat_unit = inertia_unit
    else:
        projection = min(6.0, max(0.0, max_ahead))
        threat_xy = runner_end_xy + projection * inertia_unit
        threat_xy[0] = float(np.clip(threat_xy[0], -FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0))
        threat_xy[1] = float(np.clip(threat_xy[1], -FIELD_WIDTH / 2.0, FIELD_WIDTH / 2.0))
        threat_unit = inertia_unit

    rel = points - runner_end_xy

    def future_lane_prior(unit: np.ndarray) -> np.ndarray:
        ahead = rel @ unit
        lateral_vec = rel - ahead[:, None] * unit
        lateral = np.linalg.norm(lateral_vec, axis=1)
        lateral_width = lateral_sigma + 0.18 * np.maximum(ahead, 0.0)
        lateral_score = np.exp(-0.5 * (lateral / np.maximum(lateral_width, 1e-6)) ** 2)
        ahead_score = (1.0 - np.exp(-np.maximum(ahead, 0.0) / 2.0)) * np.exp(-np.maximum(ahead, 0.0) / max_ahead)
        gate = (ahead > 0.0) & (ahead <= max_ahead)
        return lateral_score * ahead_score * gate

    movement_lane = future_lane_prior(threat_unit)
    goal_lane = future_lane_prior(goal_unit)

    threat_distance = np.linalg.norm(points - threat_xy, axis=1)
    threat_ahead = rel @ threat_unit
    threat_projection = max(1.0, min(max_ahead, float(np.linalg.norm(threat_xy - runner_end_xy))))
    threat_anchor = (
        np.exp(-0.5 * (threat_distance / np.maximum(lateral_sigma * 1.25, 1e-6)) ** 2)
        * np.exp(-0.5 * ((threat_ahead - threat_projection) / 4.5) ** 2)
        * (threat_ahead > 0.0)
        * (threat_ahead <= max_ahead)
    )

    goal_mix = max(0.0, min(1.0, goal_mix))
    future_prior = np.maximum(movement_lane, 0.65 * threat_anchor)
    prior = (1.0 - goal_mix) * future_prior + goal_mix * goal_lane
    max_prior = float(np.nanmax(prior)) if len(prior) else 0.0
    if max_prior > 1e-12:
        prior = prior / max_prior
    return prior


def runner_control_at_points(
    frame: BundesligaFrame,
    runner_id: str,
    points: np.ndarray,
    runner_speed: float = 5.8,
    defender_speed: float = 5.5,
    time_sigma: float = 0.45,
) -> np.ndarray:
    runner = frame.players.get(runner_id)
    if runner is None:
        return np.zeros(len(points), dtype=float)

    defenders = np.asarray(
        [(player.x, player.y) for player in frame.players.values() if player.team_id != runner.team_id],
        dtype=float,
    )
    if len(defenders) == 0:
        return np.ones(len(points), dtype=float)

    runner_xy = np.asarray([runner.x, runner.y], dtype=float)
    runner_time = np.linalg.norm(points - runner_xy, axis=1) / runner_speed
    defender_time = np.linalg.norm(points[:, None, :] - defenders[None, :, :], axis=2).min(axis=1) / defender_speed
    return _sigmoid((defender_time - runner_time) / time_sigma)


def compute_runner_threat_map(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    runner_threat_point: tuple[float, float] | None,
    resolution: float,
    max_ahead: float,
    lateral_sigma: float,
    goal_mix: float,
) -> dict[str, object] | None:
    return runner_receiving_threat_map(
        start_frame,
        end_frame,
        runner_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        runner_threat_point=runner_threat_point,
        resolution=resolution,
        max_ahead=max_ahead,
        lateral_sigma=lateral_sigma,
        goal_mix=goal_mix,
    )


def draw_runner_threat_map(
    ax,
    threat_map: dict[str, object] | None,
    show_label: bool = True,
) -> None:
    if threat_map is None:
        return
    threat = np.asarray(threat_map["threat"], dtype=float)
    if threat.size == 0 or float(np.nanmax(threat)) <= 1e-12:
        return
    xs = np.asarray(threat_map["xs"], dtype=float)
    ys = np.asarray(threat_map["ys"], dtype=float)
    threshold = float(np.nanmax(threat)) * 0.08
    masked = np.ma.masked_less_equal(threat, threshold)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0.0)
    ax.imshow(
        masked,
        extent=[float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        origin="lower",
        cmap=cmap,
        alpha=0.48,
        interpolation="bilinear",
        zorder=1,
    )
    peak_x = threat_map.get("peak_x")
    peak_y = threat_map.get("peak_y")
    if peak_x is not None and peak_y is not None:
        ax.scatter(
            [float(peak_x)],
            [float(peak_y)],
            s=120,
            c="#fde047",
            marker="*",
            edgecolors="#7c2d12",
            linewidths=0.9,
            zorder=13,
        )
        if show_label:
            draw_label_box(
                ax,
                (float(peak_x), float(peak_y)),
                "map peak",
                (float(peak_x) + 2.2, float(peak_y) + 2.2),
                text_color="#7c2d12",
                edge_color="#f59e0b",
                fontsize=7.5,
                zorder=14,
            )


def draw_arrow(
    ax,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: str,
    label: str,
    linewidth: float = 3.2,
    linestyle: str = "-",
    text_offset: tuple[float, float] = (0.0, 1.5),
    zorder: int = 9,
) -> None:
    if math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]) < 0.25:
        return
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 16,
        },
        zorder=zorder,
    )
    if label:
        mid_x = (start_xy[0] + end_xy[0]) / 2.0 + text_offset[0]
        mid_y = (start_xy[1] + end_xy[1]) / 2.0 + text_offset[1]
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.72,
            },
            zorder=zorder + 1,
        )


def draw_label_box(
    ax,
    anchor_xy: tuple[float, float],
    label: str,
    label_xy: tuple[float, float],
    text_color: str,
    edge_color: str,
    fontsize: float = 8.0,
    weight: str = "bold",
    zorder: int = 15,
    connector: bool = True,
) -> None:
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight=weight,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": edge_color,
            "linewidth": 0.55,
            "alpha": 0.78,
        },
        zorder=zorder,
    )
    if connector and math.hypot(label_xy[0] - anchor_xy[0], label_xy[1] - anchor_xy[1]) > 0.6:
        ax.plot(
            [anchor_xy[0], label_xy[0]],
            [anchor_xy[1], label_xy[1]],
            color=edge_color,
            lw=0.85,
            linestyle=":",
            alpha=0.72,
            zorder=zorder - 1,
        )


def closest_defender_for_runner(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
) -> BundesligaObjectState | None:
    runner = end_frame.players.get(runner_id)
    if runner is None:
        return None
    defenders = [
        player
        for player in end_frame.players.values()
        if player.team_id != runner.team_id and player.object_id in start_frame.players
    ]
    if not defenders:
        return None
    return min(defenders, key=lambda player: (player.x - runner.x) ** 2 + (player.y - runner.y) ** 2)


def affected_defender_from_row(
    row: pd.Series,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
) -> BundesligaObjectState | None:
    if "top_runner_affected_defender_id" in row:
        if pd.notna(row["top_runner_affected_defender_id"]):
            defender_id = str(row["top_runner_affected_defender_id"])
            defender = end_frame.players.get(defender_id)
            if defender is not None and defender.object_id in start_frame.players:
                return defender
        return None
    if "affected_defender_id" in row and pd.notna(row["affected_defender_id"]):
        defender_id = str(row["affected_defender_id"])
        defender = end_frame.players.get(defender_id)
        if defender is not None and defender.object_id in start_frame.players:
            return defender
    return closest_defender_for_runner(start_frame, end_frame, runner_id)


def defender_from_row(
    row: pd.Series,
    column: str,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
) -> BundesligaObjectState | None:
    if column not in row or pd.isna(row[column]):
        return None
    defender_id = str(row[column])
    defender = end_frame.players.get(defender_id)
    if defender is not None and defender.object_id in start_frame.players:
        return defender
    return None


def defender_ids_from_row(row: pd.Series, column: str) -> list[str]:
    if column not in row or pd.isna(row[column]):
        return []
    return [
        defender_id
        for defender_id in str(row[column]).split(";")
        if defender_id
    ]


def defenders_from_row(
    row: pd.Series,
    column: str,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
) -> list[BundesligaObjectState]:
    defenders = []
    for defender_id in defender_ids_from_row(row, column):
        defender = end_frame.players.get(defender_id)
        if defender is not None and defender_id in start_frame.players:
            defenders.append(defender)
    return defenders


def draw_defender_role_rings(
    ax,
    defenders_by_id: dict[str, BundesligaObjectState],
    reactive_ids: set[str],
    holding_ids: set[str],
    blocker_ids: set[str],
    show_labels: bool = True,
) -> None:
    offsets = [(-2.7, 2.8), (2.7, 2.8), (-2.7, -2.8), (2.7, -2.8)]
    for idx, (defender_id, defender) in enumerate(defenders_by_id.items()):
        roles = []
        if defender_id in blocker_ids:
            ax.scatter(
                [defender.x],
                [defender.y],
                s=350,
                facecolors="none",
                edgecolors="#dc2626",
                linewidths=2.0,
                zorder=10,
            )
            roles.append("B")
        if defender_id in holding_ids:
            ax.scatter(
                [defender.x],
                [defender.y],
                s=275,
                facecolors="none",
                edgecolors="#7c3aed",
                linewidths=2.0,
                zorder=11,
            )
            roles.append("H")
        if defender_id in reactive_ids:
            ax.scatter(
                [defender.x],
                [defender.y],
                s=215,
                facecolors="none",
                edgecolors="#c026d3",
                linewidths=2.2,
                zorder=12,
            )
            roles.insert(0, "R")
        if not roles or not show_labels:
            continue
        offset = offsets[idx % len(offsets)]
        draw_label_box(
            ax,
            (defender.x, defender.y),
            "+".join(roles),
            (defender.x + offset[0], defender.y + offset[1]),
            text_color="#111827",
            edge_color="#6b21a8" if defender_id in reactive_ids | holding_ids else "#b91c1c",
            fontsize=7.0,
            zorder=16,
        )


def scatter_players(
    ax,
    frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    key_ids: set[str],
    high_contrast: bool = False,
    show_numbers: bool = True,
) -> None:
    for player in frame.players.values():
        is_attacker = player.team_id == attacking_team_id
        is_key = player.object_id in key_ids
        if high_contrast:
            color = "#2563eb" if is_attacker else "#111827"
            edge = "#ffffff"
            size = 82 if not is_key else 190
            alpha = 0.96
            zorder = 7 if is_key else 6
            ax.scatter(
                [player.x],
                [player.y],
                s=size + 92,
                c="#ffffff",
                edgecolors="#111827",
                linewidths=0.35,
                alpha=0.94,
                zorder=zorder - 1,
            )
        else:
            color = "#2f80ed" if is_attacker else "#9aa3ad"
            edge = "#114f9d" if is_attacker else "#4b5563"
            size = 58 if not is_key else 170
            alpha = 0.58 if not is_key else 1.0
            zorder = 4 if is_key else 3
        ax.scatter(
            [player.x],
            [player.y],
            s=size,
            c=color,
            edgecolors=edge,
            linewidths=1.25 if high_contrast else 0.9,
            alpha=alpha,
            zorder=zorder,
        )
        if is_key and show_numbers:
            ax.text(
                player.x,
                player.y - 1.6,
                player_number(metadata, player.object_id),
                ha="center",
                va="top",
                fontsize=8,
                color="#111827",
                weight="bold",
                zorder=20,
            )


def draw_marker(
    ax,
    state: BundesligaObjectState | None,
    label: str,
    color: str,
    marker: str,
    text_offset: tuple[float, float] = (0.0, 1.9),
) -> None:
    if state is None:
        return
    kwargs = {"s": 185, "c": color, "marker": marker, "linewidths": 1.2, "zorder": 10}
    if marker != "x":
        kwargs["edgecolors"] = "#111827"
    ax.scatter([state.x], [state.y], **kwargs)
    draw_label_box(
        ax,
        (state.x, state.y),
        label,
        (state.x + text_offset[0], state.y + text_offset[1]),
        text_color="#111827",
        edge_color=color,
        fontsize=8,
        zorder=11,
    )


def draw_role_label(
    ax,
    state: BundesligaObjectState | None,
    label: str,
    color: str,
    text_offset: tuple[float, float],
    fontsize: float = 7.6,
    zorder: int = 15,
) -> None:
    if state is None:
        return
    draw_label_box(
        ax,
        (state.x, state.y),
        label,
        (state.x + text_offset[0], state.y + text_offset[1]),
        text_color=color,
        edge_color=color,
        fontsize=fontsize,
        zorder=zorder,
    )


def draw_outline_marker(
    ax,
    state: BundesligaObjectState | None,
    color: str,
    marker: str,
    size: float,
    zorder: int = 14,
) -> None:
    if state is None:
        return
    ax.scatter(
        [state.x],
        [state.y],
        s=size,
        facecolors="none",
        edgecolors=color,
        marker=marker,
        linewidths=2.2,
        zorder=zorder,
    )


def draw_role_link(
    ax,
    state: BundesligaObjectState | None,
    target_xy: tuple[float, float] | None,
    label: str,
    color: str,
    marker: str,
    text_offset: tuple[float, float],
    show_marker: bool = True,
) -> None:
    if state is None:
        return
    if show_marker:
        kwargs = {"s": 185, "c": color, "marker": marker, "linewidths": 1.2, "zorder": 14}
        if marker != "x":
            kwargs["edgecolors"] = "#111827"
        ax.scatter([state.x], [state.y], **kwargs)
    if label:
        draw_label_box(
            ax,
            (state.x, state.y),
            label,
            (state.x + text_offset[0], state.y + text_offset[1]),
            text_color=color,
            edge_color=color,
            fontsize=7.6,
            zorder=15,
        )
    if target_xy is None:
        return
    ax.plot(
        [state.x, target_xy[0]],
        [state.y, target_xy[1]],
        color=color,
        lw=1.8,
        linestyle=":",
        alpha=0.95,
        zorder=9,
    )


def draw_top_option_ring(ax, state: BundesligaObjectState | None) -> None:
    if state is None:
        return
    ax.scatter(
        [state.x],
        [state.y],
        s=310,
        facecolors="none",
        edgecolors="#10b981",
        marker="D",
        linewidths=2.1,
        zorder=12,
    )
    draw_label_box(
        ax,
        (state.x, state.y),
        "top option",
        (state.x, state.y + 3.8),
        text_color="#047857",
        edge_color="#10b981",
        fontsize=8,
        zorder=13,
    )


def draw_beneficiary_marker(
    ax,
    state: BundesligaObjectState | None,
    gain: float | None,
    is_top_option: bool,
    show_marker: bool = True,
    show_label: bool = True,
) -> None:
    if state is None:
        return
    if show_marker:
        ax.scatter(
            [state.x],
            [state.y],
            s=430 if is_top_option else 330,
            c="#67e8f9",
            edgecolors="#0e7490",
            marker="*",
            linewidths=1.5,
            zorder=18,
        )
    if show_label:
        label = "top gain"
        if gain is not None:
            label += f" {gain:+.3f}"
        draw_label_box(
            ax,
            (state.x, state.y),
            label,
            (
                state.x,
                state.y + (5.2 if is_top_option else 3.8),
            ),
            text_color="#0e7490",
            edge_color="#0e7490",
            fontsize=8,
            zorder=19,
        )


def draw_benefit_box(
    ax,
    beneficiary: dict[str, float | str] | None,
    compact: bool = False,
) -> None:
    if beneficiary is None:
        return
    if compact:
        text = (
            f"beneficiary: {beneficiary['name']}  +{float(beneficiary['adjusted_gain']):.3f}\n"
            f"type: {readable_benefit_type(str(beneficiary.get('primary_benefit_type')))}\n"
            f"lane -{float(beneficiary.get('pass_lane_pressure_reduction', 0.0)):.3f} | "
            f"pressure -{float(beneficiary.get('receiver_pressure_reduction', 0.0)):.3f}\n"
            f"nearest +{float(beneficiary.get('nearest_defender_clearance_gain_m', 0.0)):.1f}m | "
            f"affected +{float(beneficiary.get('affected_removed_from_beneficiary_m', 0.0)):.1f}m"
        )
    else:
        text = (
            f"beneficiary: {beneficiary['name']}  +{float(beneficiary['adjusted_gain']):.3f}\n"
            f"type: {readable_benefit_type(str(beneficiary.get('primary_benefit_type')))}\n"
            f"O +{float(beneficiary.get('obso_gain', 0.0)):.3f} | "
            f"S +{float(beneficiary.get('space_gain_m2', 0.0)):.1f}m2 | "
            f"lane -{float(beneficiary.get('pass_lane_pressure_reduction', 0.0)):.3f}\n"
            f"pressure -{float(beneficiary.get('receiver_pressure_reduction', 0.0)):.3f} | "
            f"nearest +{float(beneficiary.get('nearest_defender_clearance_gain_m', 0.0)):.1f}m | "
            f"affected +{float(beneficiary.get('affected_removed_from_beneficiary_m', 0.0)):.1f}m"
        )
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color="#164e63",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ecfeff",
            "edgecolor": "#0891b2",
            "linewidth": 0.9,
            "alpha": 0.92,
        },
        zorder=30,
    )


def draw_obso_labels(
    ax,
    frame: BundesligaFrame,
    obso_values: list[tuple[str, float, float, float, float]],
    top_id: str | None,
    max_labels: int,
    show_adjusted: bool,
) -> None:
    for rank, (player_id, value, space, _dangerous_space, adjusted) in enumerate(obso_values[:max_labels], start=1):
        player = frame.players.get(player_id)
        if player is None:
            continue

        is_top = player_id == top_id
        text_color = "#047857" if is_top else "#14532d"
        bg = "#ecfdf5" if is_top else "#f7fee7"
        label = f"{rank}. O {value:.3f}\nS {space:.0f}m2"
        if show_adjusted:
            label = f"{rank}. A {adjusted:.3f}\nO {value:.3f} S {space:.0f}m2"
        ax.text(
            player.x + 0.9,
            player.y + 1.9,
            label,
            ha="left",
            va="center",
            fontsize=7.4 if not is_top else 8.0,
            color=text_color,
            weight="bold" if is_top else "normal",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": bg,
                "edgecolor": "#16a34a" if is_top else "#bbf7d0",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
            zorder=16,
        )


def row_pass_end(row: pd.Series, recipient: BundesligaObjectState | None) -> tuple[float, float] | None:
    if "pass_end_x" in row and "pass_end_y" in row and pd.notna(row["pass_end_x"]) and pd.notna(row["pass_end_y"]):
        pass_end = float(row["pass_end_x"]), float(row["pass_end_y"])
        pass_start = float(row["ball_x"]), float(row["ball_y"])
        if math.hypot(pass_end[0] - pass_start[0], pass_end[1] - pass_start[1]) >= 1.0:
            return pass_end
    if recipient is not None:
        return recipient.x, recipient.y
    return None


def plot_flow_scene(
    row: pd.Series,
    rank: int,
    metadata: BundesligaMatchMeta,
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    out_file: Path,
    view: str,
    pass_vector_source: str,
    max_obso_labels: int,
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
    sort_by: str,
    adjusted_lambda: float,
    show_runner_threat_map: bool,
    threat_map_resolution: float,
    threat_map_max_ahead: float,
    threat_map_lateral_sigma: float,
    threat_map_goal_mix: float,
    hide_text: bool,
) -> dict[str, object]:
    attacking_team_id = str(row["team_id"])
    runner_id = str(row["runner_id"])
    passer_id = str(row["passer_id"])
    recipient_id = str(row["recipient_id"]) if pd.notna(row["recipient_id"]) else None
    option_sort_by = "adjusted" if sort_by == "top-beneficiary-gain" else sort_by
    top_option_column = "actual_top_options"
    if option_sort_by == "adjusted":
        top_option_column = f"actual_top_adjusted_options_{lambda_key(adjusted_lambda)}"
    top_id, top_name, top_score = parse_top_option(row.get(top_option_column))

    runner_start = start_frame.players.get(runner_id)
    runner_end = end_frame.players.get(runner_id)
    runner_threat_point = None
    if (
        "runner_threat_point_x" in row
        and "runner_threat_point_y" in row
        and pd.notna(row["runner_threat_point_x"])
        and pd.notna(row["runner_threat_point_y"])
    ):
        runner_threat_point = (
            float(row["runner_threat_point_x"]),
            float(row["runner_threat_point_y"]),
        )
    runner_defender_end = defender_from_row(
        row,
        "top_runner_affected_defender_id",
        start_frame,
        end_frame,
    )
    if runner_defender_end is None:
        runner_defender_end = affected_defender_from_row(row, start_frame, end_frame, runner_id)
    runner_defender_start = (
        start_frame.players.get(runner_defender_end.object_id)
        if runner_defender_end is not None
        else None
    )
    reactive_defenders = defenders_from_row(
        row,
        "reactive_affected_defender_ids",
        start_frame,
        end_frame,
    )
    holding_defenders = defenders_from_row(
        row,
        "holding_affected_defender_ids",
        start_frame,
        end_frame,
    )
    threat_blockers = defenders_from_row(
        row,
        "runner_threat_blocker_ids",
        start_frame,
        end_frame,
    )
    if (
        "reactive_affected_defender_ids" not in row
        and runner_defender_end is not None
    ):
        reactive_defenders = [runner_defender_end]
    reactive_ids = {defender.object_id for defender in reactive_defenders}
    holding_ids = {defender.object_id for defender in holding_defenders}
    blocker_ids = {defender.object_id for defender in threat_blockers}
    role_defenders_by_id = {
        defender.object_id: defender
        for defender in [*reactive_defenders, *holding_defenders, *threat_blockers]
    }
    pass_lane_defender = defender_from_row(row, "pass_lane_suppressor_id", start_frame, end_frame)
    passer_pressure_defender = defender_from_row(row, "passer_pressure_defender_id", start_frame, end_frame)
    passer = end_frame.players.get(passer_id)
    recipient = end_frame.players.get(recipient_id) if recipient_id else None
    top_option = end_frame.players.get(top_id) if top_id else None
    ball_xy = (float(row["ball_x"]), float(row["ball_y"]))
    attacking_direction = int(row["attacking_direction"])
    obso_values = receiver_obso_values(
        end_frame,
        metadata,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        excluded_ids={passer_id, runner_id},
        local_radius=local_radius,
        local_samples=local_samples,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
        adjusted_lambda=adjusted_lambda,
        candidate_sort_by=option_sort_by,
    )
    top_space = None
    top_dangerous_space = None
    top_adjusted = None
    if obso_values:
        top_id = obso_values[0][0]
        top_name = player_name(metadata, top_id)
        top_score = obso_values[0][1]
        top_space = obso_values[0][2]
        top_dangerous_space = obso_values[0][3]
        top_adjusted = obso_values[0][4]
        top_option = end_frame.players.get(top_id)

    affected_defender_id = (
        runner_defender_end.object_id
        if runner_defender_end is not None
        else None
    )
    target_receiver_id = (
        str(row["defender_target_receiver_id"])
        if "defender_target_receiver_id" in row and pd.notna(row["defender_target_receiver_id"])
        else top_id
    )
    top_beneficiary = top_beneficiary_for_run(
        start_frame,
        end_frame,
        metadata,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        runner_id,
        passer_id,
        affected_defender_id,
        target_receiver_id,
        local_radius,
        local_samples,
        space_radius,
        space_samples,
        space_rings,
        adjusted_lambda,
    )
    top_beneficiary_id = str(top_beneficiary["player_id"]) if top_beneficiary is not None else None
    top_beneficiary_state = end_frame.players.get(top_beneficiary_id) if top_beneficiary_id else None
    runner_threat_map = (
        compute_runner_threat_map(
            start_frame,
            end_frame,
            runner_id,
            attacking_team_id,
            attacking_direction,
            ball_xy,
            runner_threat_point,
            resolution=threat_map_resolution,
            max_ahead=threat_map_max_ahead,
            lateral_sigma=threat_map_lateral_sigma,
            goal_mix=threat_map_goal_mix,
        )
        if show_runner_threat_map
        else None
    )

    if pass_vector_source == "players" and passer is not None and recipient is not None:
        pass_start = (passer.x, passer.y)
        pass_end = (recipient.x, recipient.y)
        pass_endpoint_is_player = True
    else:
        pass_start = (float(row["ball_x"]), float(row["ball_y"]))
        pass_end = row_pass_end(row, recipient)
        pass_endpoint_is_player = False

    key_ids = {
        pid
        for pid in [
            runner_id,
            passer_id,
            recipient_id,
            top_id,
            top_beneficiary_id,
            runner_defender_end.object_id if runner_defender_end else None,
            *role_defenders_by_id.keys(),
            pass_lane_defender.object_id if pass_lane_defender else None,
            passer_pressure_defender.object_id if passer_pressure_defender else None,
        ]
        if pid
    }

    fig, ax = plt.subplots(figsize=(12.8, 8.6), dpi=170)
    draw_pitch(ax)
    draw_runner_threat_map(ax, runner_threat_map, show_label=not hide_text)
    clean_threat_map_view = runner_threat_map is not None
    scatter_players(
        ax,
        end_frame,
        metadata,
        attacking_team_id,
        key_ids,
        high_contrast=clean_threat_map_view,
        show_numbers=not hide_text,
    )

    if pass_end is not None:
        draw_arrow(
            ax,
            pass_start,
            pass_end,
            color="#dca500",
            label="" if hide_text else "pass vector",
            linewidth=3.6,
            text_offset=(4.0, 4.0),
            zorder=12,
        )
        if clean_threat_map_view:
            if not pass_endpoint_is_player:
                ax.scatter([pass_start[0]], [pass_start[1]], s=42, c="#facc15", edgecolors="#713f12", linewidths=0.9, zorder=5)
                ax.scatter([pass_end[0]], [pass_end[1]], s=28, c="#fff7cc", edgecolors="#dca500", linewidths=0.8, zorder=5)
        else:
            ax.scatter([pass_start[0]], [pass_start[1]], s=90, c="#facc15", edgecolors="#713f12", linewidths=1.0, zorder=13)
            ax.scatter([pass_end[0]], [pass_end[1]], s=55, c="#fff7cc", edgecolors="#dca500", linewidths=1.0, zorder=12)

    if sort_by in {"runner-suppression", "runner-threat-suppression"} and runner_end is not None:
        draw_arrow(
            ax,
            pass_start,
            (runner_end.x, runner_end.y),
            color="#0f766e",
            label="" if hide_text else "runner option lane",
            linewidth=2.6,
            linestyle=":",
            text_offset=(0.0, -2.4),
            zorder=9,
        )

    if runner_start is not None and runner_end is not None:
        draw_arrow(ax, (runner_start.x, runner_start.y), (runner_end.x, runner_end.y), color="#f97316", label="", linewidth=3.2, text_offset=(-3.0, -3.2), zorder=11)
        if runner_threat_point is not None:
            draw_arrow(
                ax,
                (runner_end.x, runner_end.y),
                runner_threat_point,
                color="#fb923c",
                label="" if hide_text else "threat point",
                linewidth=2.0,
                linestyle=":",
                text_offset=(1.6, 2.0),
                zorder=10,
            )
            ax.scatter(
                [runner_threat_point[0]],
                [runner_threat_point[1]],
                s=58 if clean_threat_map_view else 95,
                c="#fb923c",
                marker="x",
                linewidths=1.9 if clean_threat_map_view else 2.2,
                zorder=8 if clean_threat_map_view else 13,
            )

    movement_role_defenders = {
        defender.object_id: (defender, "#c026d3")
        for defender in reactive_defenders
    }
    for defender in holding_defenders:
        movement_role_defenders.setdefault(defender.object_id, (defender, "#7c3aed"))
    if not movement_role_defenders and runner_defender_end is not None:
        movement_role_defenders[runner_defender_end.object_id] = (
            runner_defender_end,
            "#c026d3",
        )
    for defender_id, (defender_end, color) in movement_role_defenders.items():
        defender_start = start_frame.players.get(defender_id)
        if defender_start is None:
            continue
        draw_arrow(
            ax,
            (defender_start.x, defender_start.y),
            (defender_end.x, defender_end.y),
            color=color,
            label="",
            linewidth=3.0,
            linestyle="--",
            text_offset=(0.0, 2.4),
            zorder=10,
        )

    if hide_text:
        draw_outline_marker(ax, passer, "#f59e0b", "s", 255, zorder=15)
        draw_outline_marker(ax, recipient, "#0f766e", "^", 270, zorder=15)
        draw_outline_marker(ax, runner_end, "#f97316", "o", 285, zorder=15)
        draw_outline_marker(ax, top_option, "#10b981", "D", 315, zorder=16)
        draw_outline_marker(ax, top_beneficiary_state, "#0891b2", "*", 390, zorder=17)
        if pass_lane_defender is not None:
            ax.scatter(
                [pass_lane_defender.x],
                [pass_lane_defender.y],
                s=165,
                c="#dc2626",
                marker="x",
                linewidths=2.4,
                zorder=18,
            )
        if passer_pressure_defender is not None:
            ax.scatter(
                [passer_pressure_defender.x],
                [passer_pressure_defender.y],
                s=185,
                c="#2563eb",
                marker="+",
                linewidths=2.4,
                zorder=18,
            )
    elif clean_threat_map_view:
        draw_role_label(ax, passer, "passer", "#b45309", text_offset=(-3.6, -4.2))
        draw_role_label(ax, recipient, "recipient", "#0f766e", text_offset=(-3.0, 4.0))
        if top_option is not None:
            top_option_label = (
                f"top option + gain {float(top_beneficiary['adjusted_gain']):+.3f}"
                if top_beneficiary is not None and top_beneficiary_id == top_id
                else "top option"
            )
            draw_role_label(ax, top_option, top_option_label, "#047857", text_offset=(0.0, 3.8))
    else:
        draw_marker(ax, passer, "passer", "#f59e0b", "s", text_offset=(-3.6, -4.2))
        draw_marker(ax, recipient, "recipient", "#14b8a6", "^", text_offset=(-3.0, 4.0))
        if top_option is not None and top_id == recipient_id:
            draw_top_option_ring(ax, top_option)
        elif top_option is not None:
            draw_marker(ax, top_option, "top option", "#10b981", "D", text_offset=(0.0, 3.8))
    if not hide_text and not (clean_threat_map_view and top_beneficiary_id == top_id):
        draw_beneficiary_marker(
            ax,
            top_beneficiary_state,
            float(top_beneficiary["adjusted_gain"]) if top_beneficiary is not None else None,
            top_beneficiary_id == top_id,
            show_marker=not clean_threat_map_view,
            show_label=True,
        )
    if hide_text:
        pass
    elif clean_threat_map_view:
        draw_role_label(ax, runner_end, "runner", "#c2410c", text_offset=(4.2, -4.0))
    else:
        draw_marker(ax, runner_end, "runner", "#f97316", "o", text_offset=(4.2, -4.0))
    runner_target = runner_threat_point or ((runner_end.x, runner_end.y) if runner_end is not None else None)
    if clean_threat_map_view:
        draw_defender_role_rings(
            ax,
            role_defenders_by_id,
            reactive_ids,
            holding_ids,
            blocker_ids,
            show_labels=not hide_text,
        )
    else:
        draw_role_link(
            ax,
            runner_defender_end,
            runner_target,
            "runner-affected",
            "#c026d3",
            "D",
            text_offset=(-5.4, 4.2),
            show_marker=True,
        )
    lane_end = runner_target if runner_target is not None else pass_end
    pass_lane_target = (
        closest_point_on_segment((pass_lane_defender.x, pass_lane_defender.y), pass_start, lane_end)
        if pass_lane_defender is not None and lane_end is not None
        else None
    )
    draw_role_link(
        ax,
        pass_lane_defender,
        pass_lane_target,
        "" if hide_text else ("L" if clean_threat_map_view else "pass-lane"),
        "#dc2626",
        "X",
        text_offset=(4.6, -3.8),
        show_marker=not clean_threat_map_view,
    )
    passer_target = (passer.x, passer.y) if passer is not None else pass_start
    draw_role_link(
        ax,
        passer_pressure_defender,
        passer_target,
        "" if hide_text else ("P" if clean_threat_map_view else "passer pressure"),
        "#2563eb",
        "P",
        text_offset=(4.8, 3.8),
        show_marker=not clean_threat_map_view,
    )
    if not clean_threat_map_view and max_obso_labels > 0:
        draw_obso_labels(
            ax,
            end_frame,
            obso_values,
            top_id,
            max_obso_labels,
            show_adjusted=option_sort_by == "adjusted",
        )
    if not hide_text:
        draw_benefit_box(ax, top_beneficiary, compact=clean_threat_map_view)

    if view == "action":
        apply_action_view(
            ax,
            [
                pass_start,
                pass_end,
                state_xy(runner_start),
                state_xy(runner_end),
                runner_threat_point,
                (
                    (float(runner_threat_map["peak_x"]), float(runner_threat_map["peak_y"]))
                    if runner_threat_map is not None
                    and runner_threat_map.get("peak_x") is not None
                    and runner_threat_map.get("peak_y") is not None
                    else None
                ),
                state_xy(runner_defender_start),
                state_xy(runner_defender_end),
                *[
                    state_xy(start_frame.players.get(defender_id))
                    for defender_id in role_defenders_by_id
                ],
                *[state_xy(defender) for defender in role_defenders_by_id.values()],
                state_xy(pass_lane_defender),
                state_xy(passer_pressure_defender),
                state_xy(passer),
                state_xy(recipient),
                state_xy(top_option),
                state_xy(top_beneficiary_state),
            ],
        )
    elif view == "half":
        apply_half_view(
            ax,
            [
                pass_start,
                pass_end,
                state_xy(runner_start),
                state_xy(runner_end),
                runner_threat_point,
                (
                    (float(runner_threat_map["peak_x"]), float(runner_threat_map["peak_y"]))
                    if runner_threat_map is not None
                    and runner_threat_map.get("peak_x") is not None
                    and runner_threat_map.get("peak_y") is not None
                    else None
                ),
                state_xy(runner_defender_start),
                state_xy(runner_defender_end),
                *[
                    state_xy(start_frame.players.get(defender_id))
                    for defender_id in role_defenders_by_id
                ],
                *[state_xy(defender) for defender in role_defenders_by_id.values()],
                state_xy(pass_lane_defender),
                state_xy(passer_pressure_defender),
                state_xy(passer),
                state_xy(recipient),
                state_xy(top_option),
                state_xy(top_beneficiary_state),
            ],
            attacking_direction=attacking_direction,
        )

    match_title = f"{metadata.home_team_name} vs {metadata.away_team_name}"
    title_metric_label = "OBSO"
    title_metric_value = float(row["draft_obso_offball_value"])
    if sort_by == "post-space":
        title_metric_label = "post-space"
        title_metric_value = float(row["draft_post_reception_space_offball_value_m2"])
    elif sort_by == "dangerous-space":
        title_metric_label = "dangerous-space"
        title_metric_value = float(row["draft_post_reception_dangerous_space_offball_value"])
    elif sort_by == "adjusted":
        title_metric_label = f"adjusted λ={adjusted_lambda:g}"
        title_metric_value = float(row[f"draft_adjusted_option_offball_value_{lambda_key(adjusted_lambda)}"])
    elif sort_by == "top-beneficiary-gain":
        title_metric_label = f"top gain adjusted λ={adjusted_lambda:g}"
        title_metric_value = (
            float(top_beneficiary["adjusted_gain"]) if top_beneficiary is not None else 0.0
        )
    elif sort_by == "runner-suppression":
        title_metric_label = "runner defensive suppression"
        title_metric_value = float(row["runner_defensive_suppression_positive_obso"])
    elif sort_by == "runner-threat-suppression":
        title_metric_label = "runner threat suppression"
        title_metric_value = float(row["runner_threat_suppression_positive"])
    suffix = "m2" if sort_by == "post-space" else ""
    title_value_text = f"{title_metric_label} off-ball value {title_metric_value:+.4f}{suffix}"
    if sort_by in {"runner-suppression", "runner-threat-suppression"}:
        title_value_text = f"{title_metric_label} {title_metric_value:+.4f}"
    title = f"Rank {rank} | {match_title} | frame {int(row['event_frame'])} | {title_value_text}"
    runner_defender_name = player_name(metadata, runner_defender_end.object_id if runner_defender_end else None)
    runner_defender_role = row.get("top_runner_affected_defender_role")
    runner_defender_role = (
        str(runner_defender_role)
        if runner_defender_role is not None and pd.notna(runner_defender_role)
        else None
    )
    has_multi_roles = "reactive_affected_defender_names" in row
    if has_multi_roles:
        reactive_names = row.get("reactive_affected_defender_names")
        holding_names = row.get("holding_affected_defender_names")
        blocker_names = row.get("runner_threat_blocker_names")
        role_parts = [
            f"R {compact_semicolon_names(reactive_names)}",
            f"H {compact_semicolon_names(holding_names)}",
            f"B {compact_semicolon_names(blocker_names)}",
        ]
        runner_affected_label = " | ".join(role_parts)
    else:
        runner_affected_label = "runner-affected none"
        if runner_defender_name:
            runner_affected_label = f"runner-affected {runner_defender_name}"
            if runner_defender_role:
                runner_affected_label += f" ({runner_defender_role})"
        else:
            candidate_name = row.get("top_runner_affected_candidate_name")
            candidate_role = row.get("top_runner_affected_candidate_role")
            if candidate_name is not None and pd.notna(candidate_name):
                runner_affected_label += f" | best candidate {candidate_name}"
                if candidate_role is not None and pd.notna(candidate_role):
                    runner_affected_label += f" ({candidate_role})"
    subtitle = (
        f"passer {player_name(metadata, passer_id)} -> recipient {player_name(metadata, recipient_id)} | "
        f"non-receiving runner {player_name(metadata, runner_id)}"
    )
    subtitle += (
        f"\n{runner_affected_label}"
        if clean_threat_map_view
        else f" | {runner_affected_label}"
    )
    pass_lane_name = row.get("pass_lane_suppressor_name")
    if not clean_threat_map_view and pass_lane_name is not None and pd.notna(pass_lane_name):
        subtitle += f" | pass-lane {pass_lane_name}"
    passer_pressure_name = row.get("passer_pressure_defender_name")
    if not clean_threat_map_view and passer_pressure_name is not None and pd.notna(passer_pressure_name):
        subtitle += f" | passer pressure {passer_pressure_name}"
    top_suppression_name = row.get("top_suppression_defender_name")
    if not clean_threat_map_view and top_suppression_name is not None and pd.notna(top_suppression_name):
        subtitle += f" | option suppressor {top_suppression_name}"
    target_receiver_name = row.get("defender_target_receiver_name")
    defender_selection = row.get("defender_selection")
    if not clean_threat_map_view and target_receiver_name is not None and pd.notna(target_receiver_name):
        subtitle += f" | target option {target_receiver_name}"
    if not clean_threat_map_view and defender_selection is not None and pd.notna(defender_selection):
        subtitle += f" ({defender_selection})"
    actual_option = float(row["actual_team_obso_topk"])
    counterfactual_option = float(row["counterfactual_team_obso_topk"])
    option_label = "actual option"
    if option_sort_by == "adjusted":
        key = lambda_key(adjusted_lambda)
        actual_option = float(row[f"actual_team_adjusted_option_topk_{key}"])
        counterfactual_option = float(row[f"counterfactual_team_adjusted_option_topk_{key}"])
        option_label = f"actual adjusted λ={adjusted_lambda:g}"
    option_text = f"{option_label} {actual_option:.4f} vs counterfactual {counterfactual_option:.4f}"
    if sort_by in {"runner-suppression", "runner-threat-suppression"}:
        option_text = (
            f"runner OBSO actual {float(row['runner_actual_obso']):.3f}"
            f" / no-response {float(row['runner_no_response_obso']):.3f}"
            f" / suppressed {float(row['runner_defensive_suppression_positive_obso']):.3f}"
        )
        if sort_by == "runner-threat-suppression" and "runner_actual_threat" in row:
            option_text = (
                f"runner threat actual {float(row['runner_actual_threat']):.3f}"
                f" / no-response {float(row['runner_no_response_threat']):.3f}"
                f" / suppressed {float(row['runner_threat_suppression_positive']):.3f}"
            )
        if "top_suppression_defender_attribution_obso" in row and pd.notna(row["top_suppression_defender_attribution_obso"]):
            option_text += (
                f" | top suppressor attribution {float(row['top_suppression_defender_attribution_obso']):.3f}"
            )
        if "top_runner_affected_defender_raw_threat" in row and pd.notna(row["top_runner_affected_defender_raw_threat"]):
            option_text += (
                f" | runner-affected raw threat {float(row['top_runner_affected_defender_raw_threat']):.3f}"
            )
        elif "top_runner_affected_defender_raw_obso" in row and pd.notna(row["top_runner_affected_defender_raw_obso"]):
            option_text += (
                f" | nearby affected raw {float(row['top_runner_affected_defender_raw_obso']):.3f}"
            )
    if sort_by not in {"runner-suppression", "runner-threat-suppression"} and "draft_post_reception_space_offball_value_m2" in row:
        option_text += f" | post-space delta {float(row['draft_post_reception_space_offball_value_m2']):+.1f}m2"
    if sort_by not in {"runner-suppression", "runner-threat-suppression"} and "runner_defensive_suppression_positive_obso" in row:
        option_text += (
            f" | runner OBSO actual {float(row['runner_actual_obso']):.3f}"
            f" / no-response {float(row['runner_no_response_obso']):.3f}"
            f" / suppressed {float(row['runner_defensive_suppression_positive_obso']):.3f}"
        )
    if (
        not clean_threat_map_view
        and sort_by not in {"runner-suppression", "runner-threat-suppression"}
        and top_name
        and top_score is not None
    ):
        option_text += f" | top option: {top_name} "
        if top_adjusted is not None and option_sort_by == "adjusted":
            option_text += f"A {top_adjusted:.3f}, "
        option_text += f"O {top_score:.3f}"
        if top_space is not None:
            option_text += f", S {top_space:.0f}m2"
    if top_beneficiary is not None:
        option_text += (
            f" | top gain: {top_beneficiary['name']} "
            f"+{float(top_beneficiary['adjusted_gain']):.3f}"
            f" ({readable_benefit_type(str(top_beneficiary.get('primary_benefit_type')))})"
        )
    if runner_threat_map is not None:
        option_text += (
            f" | runner map peak {float(runner_threat_map['peak_value']):.3f}"
            f" / total {float(runner_threat_map['integrated_threat']):.2f}"
        )
    if clean_threat_map_view:
        clean_parts = []
        if top_beneficiary is not None:
            clean_parts.append(
                f"top gain: {top_beneficiary['name']} "
                f"+{float(top_beneficiary['adjusted_gain']):.3f}"
                f" ({readable_benefit_type(str(top_beneficiary.get('primary_benefit_type')))})"
            )
        if runner_threat_map is not None:
            clean_parts.append(
                f"runner map peak {float(runner_threat_map['peak_value']):.3f}"
                f" / total {float(runner_threat_map['integrated_threat']):.2f}"
            )
        option_text = " | ".join(clean_parts)

    if hide_text:
        fig.tight_layout(pad=0.15)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.suptitle(title, fontsize=13, y=0.98, color="#111827")
        fig.text(0.5, 0.942, subtitle, ha="center", va="center", fontsize=9.2, color="#111827")
        fig.text(0.5, 0.900, option_text, ha="center", va="center", fontsize=9.0, color="#0f766e")
        pass_footer = "yellow arrow: passer -> recipient" if pass_vector_source == "players" else "yellow arrow: event pass start -> event pass end"
        footer = (
            f"blue: attack | dark: defense | {pass_footer} | orange: runner | "
            "R: reactive | H: holding | B: blocker | L: pass-lane | P: passer pressure"
            if clean_threat_map_view
            else "yellow: actual pass | cyan star: top beneficiary by value gain | orange: runner, x=runner threat point | magenta: runner-affected | red: pass-lane | blue: passer pressure | green labels: A=adjusted, O=OBSO, S=post-reception space"
        )
        fig.text(0.5, 0.025, footer, ha="center", fontsize=9, color="#374151")
        fig.tight_layout(rect=[0, 0.05, 1, 0.875])
        fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return {
        "rank": rank,
        "match_id": row["match_id"],
        "event_frame": int(row["event_frame"]),
        "passer_name": player_name(metadata, passer_id),
        "recipient_name": player_name(metadata, recipient_id),
        "runner_name": player_name(metadata, runner_id),
        "runner_threat_point_x": runner_threat_point[0] if runner_threat_point is not None else None,
        "runner_threat_point_y": runner_threat_point[1] if runner_threat_point is not None else None,
        "affected_defender_id": runner_defender_end.object_id if runner_defender_end else None,
        "affected_defender": player_name(metadata, runner_defender_end.object_id if runner_defender_end else None),
        "reactive_affected_defender_count": len(reactive_defenders),
        "reactive_affected_defender_names": ";".join(
            player_name(metadata, defender.object_id) for defender in reactive_defenders
        ),
        "holding_affected_defender_count": len(holding_defenders),
        "holding_affected_defender_names": ";".join(
            player_name(metadata, defender.object_id) for defender in holding_defenders
        ),
        "runner_threat_blocker_count": len(threat_blockers),
        "runner_threat_blocker_names": ";".join(
            player_name(metadata, defender.object_id) for defender in threat_blockers
        ),
        "top_reactive_affected_defender_name": (
            row.get("top_reactive_affected_defender_name")
            if "top_reactive_affected_defender_name" in row
            else None
        ),
        "top_reactive_affected_defender_score": (
            float(row["top_reactive_affected_defender_score"])
            if "top_reactive_affected_defender_score" in row
            and pd.notna(row["top_reactive_affected_defender_score"])
            else None
        ),
        "top_holding_affected_defender_name": (
            row.get("top_holding_affected_defender_name")
            if "top_holding_affected_defender_name" in row
            else None
        ),
        "top_holding_affected_defender_score": (
            float(row["top_holding_affected_defender_score"])
            if "top_holding_affected_defender_score" in row
            and pd.notna(row["top_holding_affected_defender_score"])
            else None
        ),
        "top_runner_threat_blocker_name": (
            row.get("top_runner_threat_blocker_name")
            if "top_runner_threat_blocker_name" in row
            else None
        ),
        "top_runner_threat_blocker_score": (
            float(row["top_runner_threat_blocker_score"])
            if "top_runner_threat_blocker_score" in row
            and pd.notna(row["top_runner_threat_blocker_score"])
            else None
        ),
        "top_runner_threat_blocker_contribution": (
            float(row["top_runner_threat_blocker_contribution"])
            if "top_runner_threat_blocker_contribution" in row
            and pd.notna(row["top_runner_threat_blocker_contribution"])
            else None
        ),
        "affected_defender_distance_to_runner_m": (
            float(row["top_runner_affected_defender_distance_to_runner_m"])
            if "top_runner_affected_defender_distance_to_runner_m" in row
            and pd.notna(row["top_runner_affected_defender_distance_to_runner_m"])
            else (
                math.hypot(runner_defender_end.x - runner_end.x, runner_defender_end.y - runner_end.y)
                if runner_defender_end is not None and runner_end is not None
                else None
            )
        ),
        "defender_selection": row.get("defender_selection") if "defender_selection" in row else None,
        "defender_target_receiver_name": (
            row.get("defender_target_receiver_name") if "defender_target_receiver_name" in row else None
        ),
        "defender_responsibility_top": (
            row.get("defender_responsibility_top") if "defender_responsibility_top" in row else None
        ),
        "runner_actual_obso": float(row["runner_actual_obso"]) if "runner_actual_obso" in row else None,
        "runner_no_response_obso": (
            float(row["runner_no_response_obso"]) if "runner_no_response_obso" in row else None
        ),
        "runner_defensive_suppression_positive_obso": (
            float(row["runner_defensive_suppression_positive_obso"])
            if "runner_defensive_suppression_positive_obso" in row
            else None
        ),
        "runner_actual_threat": (
            float(row["runner_actual_threat"])
            if "runner_actual_threat" in row and pd.notna(row["runner_actual_threat"])
            else None
        ),
        "runner_no_response_threat": (
            float(row["runner_no_response_threat"])
            if "runner_no_response_threat" in row and pd.notna(row["runner_no_response_threat"])
            else None
        ),
        "runner_threat_suppression_positive": (
            float(row["runner_threat_suppression_positive"])
            if "runner_threat_suppression_positive" in row and pd.notna(row["runner_threat_suppression_positive"])
            else None
        ),
        "top_suppression_defender_name": (
            row.get("top_suppression_defender_name") if "top_suppression_defender_name" in row else None
        ),
        "top_suppression_defender_attribution_obso": (
            float(row["top_suppression_defender_attribution_obso"])
            if "top_suppression_defender_attribution_obso" in row
            and pd.notna(row["top_suppression_defender_attribution_obso"])
            else None
        ),
        "top_runner_affected_defender_name": (
            row.get("top_runner_affected_defender_name")
            if "top_runner_affected_defender_name" in row
            else None
        ),
        "top_runner_affected_defender_role": (
            row.get("top_runner_affected_defender_role")
            if "top_runner_affected_defender_role" in row
            else None
        ),
        "top_runner_affected_candidate_name": (
            row.get("top_runner_affected_candidate_name")
            if "top_runner_affected_candidate_name" in row
            else None
        ),
        "top_runner_affected_candidate_role": (
            row.get("top_runner_affected_candidate_role")
            if "top_runner_affected_candidate_role" in row
            else None
        ),
        "top_runner_affected_candidate_score": (
            float(row["top_runner_affected_candidate_score"])
            if "top_runner_affected_candidate_score" in row
            and pd.notna(row["top_runner_affected_candidate_score"])
            else None
        ),
        "top_runner_affected_candidate_raw_threat": (
            float(row["top_runner_affected_candidate_raw_threat"])
            if "top_runner_affected_candidate_raw_threat" in row
            and pd.notna(row["top_runner_affected_candidate_raw_threat"])
            else None
        ),
        "top_runner_affected_candidate_runner_relevance_score": (
            float(row["top_runner_affected_candidate_runner_relevance_score"])
            if "top_runner_affected_candidate_runner_relevance_score" in row
            and pd.notna(row["top_runner_affected_candidate_runner_relevance_score"])
            else None
        ),
        "top_runner_affected_defender_raw_obso": (
            float(row["top_runner_affected_defender_raw_obso"])
            if "top_runner_affected_defender_raw_obso" in row
            and pd.notna(row["top_runner_affected_defender_raw_obso"])
            else None
        ),
        "top_runner_affected_defender_score": (
            float(row["top_runner_affected_defender_score"])
            if "top_runner_affected_defender_score" in row
            and pd.notna(row["top_runner_affected_defender_score"])
            else None
        ),
        "top_runner_affected_defender_score_norm": (
            float(row["top_runner_affected_defender_score_norm"])
            if "top_runner_affected_defender_score_norm" in row
            and pd.notna(row["top_runner_affected_defender_score_norm"])
            else None
        ),
        "top_runner_affected_defender_runner_relevance_score": (
            float(row["top_runner_affected_defender_runner_relevance_score"])
            if "top_runner_affected_defender_runner_relevance_score" in row
            and pd.notna(row["top_runner_affected_defender_runner_relevance_score"])
            else None
        ),
        "top_runner_affected_defender_relevance_threat_region_score": (
            float(row["top_runner_affected_defender_relevance_threat_region_score"])
            if "top_runner_affected_defender_relevance_threat_region_score" in row
            and pd.notna(row["top_runner_affected_defender_relevance_threat_region_score"])
            else None
        ),
        "top_runner_affected_defender_relevance_observed_path_score": (
            float(row["top_runner_affected_defender_relevance_observed_path_score"])
            if "top_runner_affected_defender_relevance_observed_path_score" in row
            and pd.notna(row["top_runner_affected_defender_relevance_observed_path_score"])
            else None
        ),
        "top_runner_affected_defender_relevance_goal_side_score": (
            float(row["top_runner_affected_defender_relevance_goal_side_score"])
            if "top_runner_affected_defender_relevance_goal_side_score" in row
            and pd.notna(row["top_runner_affected_defender_relevance_goal_side_score"])
            else None
        ),
        "top_runner_affected_defender_relevance_movement_response_score": (
            float(row["top_runner_affected_defender_relevance_movement_response_score"])
            if "top_runner_affected_defender_relevance_movement_response_score" in row
            and pd.notna(row["top_runner_affected_defender_relevance_movement_response_score"])
            else None
        ),
        "top_runner_affected_defender_relevance_pass_lane_score": (
            float(row["top_runner_affected_defender_relevance_pass_lane_score"])
            if "top_runner_affected_defender_relevance_pass_lane_score" in row
            and pd.notna(row["top_runner_affected_defender_relevance_pass_lane_score"])
            else None
        ),
        "top_runner_affected_defender_raw_threat": (
            float(row["top_runner_affected_defender_raw_threat"])
            if "top_runner_affected_defender_raw_threat" in row
            and pd.notna(row["top_runner_affected_defender_raw_threat"])
            else None
        ),
        "top_runner_affected_defender_goal_side_score": (
            float(row["top_runner_affected_defender_goal_side_score"])
            if "top_runner_affected_defender_goal_side_score" in row
            and pd.notna(row["top_runner_affected_defender_goal_side_score"])
            else None
        ),
        "top_runner_affected_defender_proximity_rank": (
            int(row["top_runner_affected_defender_proximity_rank"])
            if "top_runner_affected_defender_proximity_rank" in row
            and pd.notna(row["top_runner_affected_defender_proximity_rank"])
            else None
        ),
        "top_runner_affected_defender_distance_to_target_option_m": (
            float(row["top_runner_affected_defender_distance_to_target_option_m"])
            if "top_runner_affected_defender_distance_to_target_option_m" in row
            and pd.notna(row["top_runner_affected_defender_distance_to_target_option_m"])
            else None
        ),
        "top_runner_affected_defender_target_option_bias_m": (
            float(row["top_runner_affected_defender_target_option_bias_m"])
            if "top_runner_affected_defender_target_option_bias_m" in row
            and pd.notna(row["top_runner_affected_defender_target_option_bias_m"])
            else None
        ),
        "top_runner_affected_defender_target_assignment_score": (
            float(row["top_runner_affected_defender_target_assignment_score"])
            if "top_runner_affected_defender_target_assignment_score" in row
            and pd.notna(row["top_runner_affected_defender_target_assignment_score"])
            else None
        ),
        "pass_lane_suppressor_name": (
            row.get("pass_lane_suppressor_name") if "pass_lane_suppressor_name" in row else None
        ),
        "pass_lane_suppressor_score": (
            float(row["pass_lane_suppressor_score"])
            if "pass_lane_suppressor_score" in row and pd.notna(row["pass_lane_suppressor_score"])
            else None
        ),
        "pass_lane_suppressor_contribution": (
            float(row["pass_lane_suppressor_contribution"])
            if "pass_lane_suppressor_contribution" in row
            and pd.notna(row["pass_lane_suppressor_contribution"])
            else None
        ),
        "pass_lane_suppressor_distance_to_pass_lane_m": (
            float(row["pass_lane_suppressor_distance_to_pass_lane_m"])
            if "pass_lane_suppressor_distance_to_pass_lane_m" in row
            and pd.notna(row["pass_lane_suppressor_distance_to_pass_lane_m"])
            else None
        ),
        "passer_pressure_defender_name": (
            row.get("passer_pressure_defender_name") if "passer_pressure_defender_name" in row else None
        ),
        "passer_pressure_defender_score": (
            float(row["passer_pressure_defender_score"])
            if "passer_pressure_defender_score" in row and pd.notna(row["passer_pressure_defender_score"])
            else None
        ),
        "passer_pressure_defender_distance_to_passer_m": (
            float(row["passer_pressure_defender_distance_to_passer_m"])
            if "passer_pressure_defender_distance_to_passer_m" in row
            and pd.notna(row["passer_pressure_defender_distance_to_passer_m"])
            else None
        ),
        "runner_affected_suppression_top": (
            row.get("runner_affected_suppression_top") if "runner_affected_suppression_top" in row else None
        ),
        "pass_lane_suppression_top": (
            row.get("pass_lane_suppression_top") if "pass_lane_suppression_top" in row else None
        ),
        "passer_pressure_top": (
            row.get("passer_pressure_top") if "passer_pressure_top" in row else None
        ),
        "defensive_suppression_top": (
            row.get("defensive_suppression_top") if "defensive_suppression_top" in row else None
        ),
        "actual_team_obso_topk": float(row["actual_team_obso_topk"]),
        "counterfactual_team_obso_topk": float(row["counterfactual_team_obso_topk"]),
        "draft_obso_offball_value": float(row["draft_obso_offball_value"]),
        "draft_post_reception_space_offball_value_m2": (
            float(row["draft_post_reception_space_offball_value_m2"])
            if "draft_post_reception_space_offball_value_m2" in row
            else None
        ),
        "draft_post_reception_dangerous_space_offball_value": (
            float(row["draft_post_reception_dangerous_space_offball_value"])
            if "draft_post_reception_dangerous_space_offball_value" in row
            else None
        ),
        "adjusted_lambda": adjusted_lambda if option_sort_by == "adjusted" else None,
        "draft_adjusted_option_offball_value": (
            float(row[f"draft_adjusted_option_offball_value_{lambda_key(adjusted_lambda)}"])
            if option_sort_by == "adjusted"
            else None
        ),
        "top_option_name": player_name(metadata, top_id),
        "top_option_obso": float(top_score) if top_score is not None else None,
        "top_option_post_reception_space_m2": float(top_space) if top_space is not None else None,
        "top_option_post_reception_dangerous_space": (
            float(top_dangerous_space) if top_dangerous_space is not None else None
        ),
        "top_option_adjusted": float(top_adjusted) if top_adjusted is not None else None,
        "top_beneficiary_name": (
            str(top_beneficiary["name"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_actual_offside": (
            bool(top_beneficiary["actual_offside"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_counterfactual_offside": (
            bool(top_beneficiary["counterfactual_offside"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_adjusted_actual": (
            float(top_beneficiary["actual_adjusted"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_adjusted_counterfactual": (
            float(top_beneficiary["counterfactual_adjusted"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_adjusted_counterfactual_raw": (
            float(top_beneficiary["counterfactual_raw_adjusted"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_adjusted_gain": (
            float(top_beneficiary["adjusted_gain"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_obso_gain": (
            float(top_beneficiary["obso_gain"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_space_gain_m2": (
            float(top_beneficiary["space_gain_m2"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_space_actual_m2": (
            float(top_beneficiary["actual_space_m2"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_space_counterfactual_m2": (
            float(top_beneficiary["counterfactual_space_m2"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_dangerous_space_gain": (
            float(top_beneficiary["dangerous_space_gain"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_dangerous_space_actual": (
            float(top_beneficiary["actual_dangerous_space"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_dangerous_space_counterfactual": (
            float(top_beneficiary["counterfactual_dangerous_space"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_primary_benefit_type": (
            str(top_beneficiary["primary_benefit_type"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_nearest_defender_clearance_gain_m": (
            float(top_beneficiary["nearest_defender_clearance_gain_m"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_close_defenders_5m_removed": (
            float(top_beneficiary["close_defenders_5m_removed"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_receiver_pressure_reduction": (
            float(top_beneficiary["receiver_pressure_reduction"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_pass_lane_pressure_reduction": (
            float(top_beneficiary["pass_lane_pressure_reduction"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_goal_side_blocker_reduction": (
            float(top_beneficiary["goal_side_blocker_reduction"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_affected_removed_m": (
            float(top_beneficiary["affected_removed_from_beneficiary_m"])
            if top_beneficiary is not None
            else None
        ),
        "top_beneficiary_is_top_option": top_beneficiary_id == top_id if top_beneficiary_id else None,
        "runner_threat_map_peak_x": (
            float(runner_threat_map["peak_x"])
            if runner_threat_map is not None and runner_threat_map.get("peak_x") is not None
            else None
        ),
        "runner_threat_map_peak_y": (
            float(runner_threat_map["peak_y"])
            if runner_threat_map is not None and runner_threat_map.get("peak_y") is not None
            else None
        ),
        "runner_threat_map_peak_value": (
            float(runner_threat_map["peak_value"]) if runner_threat_map is not None else None
        ),
        "runner_threat_map_peak_receive_probability": (
            float(runner_threat_map["peak_receive_probability"]) if runner_threat_map is not None else None
        ),
        "runner_threat_map_integrated_threat": (
            float(runner_threat_map["integrated_threat"]) if runner_threat_map is not None else None
        ),
        "obso_candidates": ";".join(
            f"{player_name(metadata, player_id)}:{value:.6f}"
            for player_id, value, _space, _dangerous_space, _adjusted in obso_values[:max_obso_labels]
        ),
        "post_reception_space_candidates": ";".join(
            f"{player_name(metadata, player_id)}:{space:.6f}"
            for player_id, _value, space, _dangerous_space, _adjusted in obso_values[:max_obso_labels]
        ),
        "post_reception_dangerous_space_candidates": ";".join(
            f"{player_name(metadata, player_id)}:{dangerous_space:.6f}"
            for player_id, _value, _space, dangerous_space, _adjusted in obso_values[:max_obso_labels]
        ),
        "adjusted_candidates": ";".join(
            f"{player_name(metadata, player_id)}:{adjusted:.6f}"
            for player_id, _value, _space, _dangerous_space, adjusted in obso_values[:max_obso_labels]
        ),
        "image_file": out_file.name,
    }


def load_frames_for_rows(data_dir: Path, rows: pd.DataFrame) -> tuple[dict[str, BundesligaMatchMeta], dict[tuple[str, int], BundesligaFrame]]:
    metadata_by_match: dict[str, BundesligaMatchMeta] = {}
    frames_by_key: dict[tuple[str, int], BundesligaFrame] = {}

    for match_id, group in rows.groupby("match_id"):
        files = find_bundesliga_files(data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        metadata_by_match[match_id] = metadata
        target_frames = set(group["start_frame"].astype(int)) | set(group["event_frame"].astype(int))
        frames = load_bundesliga_frames(files["positions"], target_frames=target_frames)
        for frame_id, frame in frames.items():
            frames_by_key[(match_id, frame_id)] = frame

    return metadata_by_match, frames_by_key


def clear_generated_visualizations(out_dir: Path) -> None:
    for image_file in out_dir.glob("rank_*.png"):
        image_file.unlink()
    summary_file = out_dir / "summary.csv"
    if summary_file.exists():
        summary_file.unlink()


def enrich_exact_top_beneficiary_gain(
    rows: pd.DataFrame,
    data_dir: Path,
    adjusted_lambda: float,
    local_radius: float,
    local_samples: int,
    space_radius: float,
    space_samples: int,
    space_rings: int,
) -> pd.DataFrame:
    metadata_by_match, frames_by_key = load_frames_for_rows(data_dir, rows)
    gains = []
    names = []
    benefit_types = []
    for _, row in rows.iterrows():
        match_id = str(row["match_id"])
        start_frame = frames_by_key[(match_id, int(row["start_frame"]))]
        end_frame = frames_by_key[(match_id, int(row["event_frame"]))]
        affected_defender_id = (
            str(row["top_runner_affected_defender_id"])
            if "top_runner_affected_defender_id" in row
            and pd.notna(row["top_runner_affected_defender_id"])
            else None
        )
        target_receiver_id = (
            str(row["defender_target_receiver_id"])
            if "defender_target_receiver_id" in row
            and pd.notna(row["defender_target_receiver_id"])
            else None
        )
        beneficiary = top_beneficiary_for_run(
            start_frame,
            end_frame,
            metadata_by_match[match_id],
            str(row["team_id"]),
            int(row["attacking_direction"]),
            (float(row["ball_x"]), float(row["ball_y"])),
            str(row["runner_id"]),
            str(row["passer_id"]),
            affected_defender_id,
            target_receiver_id,
            local_radius,
            local_samples,
            space_radius,
            space_samples,
            space_rings,
            adjusted_lambda,
        )
        gains.append(
            float(beneficiary["adjusted_gain"])
            if beneficiary is not None
            else 0.0
        )
        names.append(
            str(beneficiary["name"])
            if beneficiary is not None
            else None
        )
        benefit_types.append(
            str(beneficiary["primary_benefit_type"])
            if beneficiary is not None
            else None
        )

    out = rows.copy()
    key = lambda_key(adjusted_lambda)
    out[f"exact_top_beneficiary_adjusted_gain_{key}"] = gains
    out["exact_top_beneficiary_name"] = names
    out["exact_top_beneficiary_primary_benefit_type"] = benefit_types
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ball/runner/defender arrow views for Bundesliga OBSO candidates.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/processed/visualizations/offball_<view>.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--view", choices=["action", "half", "full"], default="half")
    parser.add_argument("--pass-vector-source", choices=["players", "event"], default="players")
    parser.add_argument("--max-obso-labels", type=int, default=8)
    parser.add_argument("--local-radius", type=float, default=3.0)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--space-radius", type=float, default=5.0)
    parser.add_argument("--space-samples", type=int, default=16)
    parser.add_argument("--space-rings", type=int, default=3)
    parser.add_argument("--show-runner-threat-map", action="store_true")
    parser.add_argument(
        "--hide-text",
        action="store_true",
        help="Hide all titles, labels, shirt numbers, and annotation boxes.",
    )
    parser.add_argument("--threat-map-resolution", type=float, default=1.0)
    parser.add_argument("--threat-map-max-ahead", type=float, default=18.0)
    parser.add_argument("--threat-map-lateral-sigma", type=float, default=3.2)
    parser.add_argument(
        "--threat-map-goal-mix",
        type=float,
        default=0.25,
        help="Blend in a goal-directed lane with the observed runner-movement lane.",
    )
    parser.add_argument(
        "--sort-by",
        choices=[
            "obso",
            "post-space",
            "dangerous-space",
            "adjusted",
            "top-beneficiary-gain",
            "runner-suppression",
            "runner-threat-suppression",
        ],
        default="obso",
    )
    parser.add_argument("--adjusted-lambda", type=float, default=0.5)
    parser.add_argument(
        "--exact-rerank-pool",
        type=int,
        default=0,
        help="Exactly recompute top-beneficiary gain for this many fast-ranked rows.",
    )
    parser.add_argument(
        "--exact-results-out",
        type=Path,
        default=None,
        help="Optional CSV path for the exact-reranked candidate pool.",
    )
    parser.add_argument("--include-nonpositive", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = DEFAULT_VISUALIZATIONS_DIR / f"offball_{args.view}"
    if args.exact_rerank_pool > 0 and args.exact_results_out is None:
        args.exact_results_out = (
            ROOT / "data" / "processed" / f"offball_top{args.exact_rerank_pool}.csv"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sort_columns = {
        "obso": "draft_obso_offball_value",
        "post-space": "draft_post_reception_space_offball_value_m2",
        "dangerous-space": "draft_post_reception_dangerous_space_offball_value",
        "adjusted": f"draft_adjusted_option_offball_value_{lambda_key(args.adjusted_lambda)}",
        "runner-suppression": "runner_defensive_suppression_positive_obso",
        "runner-threat-suppression": "runner_threat_suppression_positive",
    }
    rows = pd.read_csv(args.results)
    if args.sort_by == "top-beneficiary-gain":
        key = lambda_key(args.adjusted_lambda)
        actual_col = f"actual_top_adjusted_options_{key}"
        counterfactual_col = f"counterfactual_top_adjusted_options_{key}"
        exact_col = f"exact_top_beneficiary_adjusted_gain_{key}"
        fast_col = "_fast_top_beneficiary_adjusted_gain"
        if args.exact_rerank_pool > 0:
            for col in [actual_col, counterfactual_col]:
                if col not in rows.columns:
                    raise ValueError(f"{col} is not present in {args.results}")
            rows[fast_col] = rows.apply(
                lambda row: fast_top_beneficiary_gain(row, args.adjusted_lambda),
                axis=1,
            )
            rows = (
                rows.sort_values(fast_col, ascending=False)
                .head(args.exact_rerank_pool)
                .reset_index(drop=True)
            )
            rows = enrich_exact_top_beneficiary_gain(
                rows,
                args.data_dir,
                args.adjusted_lambda,
                args.local_radius,
                args.local_samples,
                args.space_radius,
                args.space_samples,
                args.space_rings,
            )
            sort_column = exact_col
            if args.exact_results_out is not None:
                exact_rows = rows.sort_values(sort_column, ascending=False).reset_index(drop=True)
                args.exact_results_out.parent.mkdir(parents=True, exist_ok=True)
                exact_rows.to_csv(args.exact_results_out, index=False)
                print(f"Saved exact-reranked pool to {args.exact_results_out}")
        elif exact_col in rows.columns:
            sort_column = exact_col
        else:
            for col in [actual_col, counterfactual_col]:
                if col not in rows.columns:
                    raise ValueError(f"{col} is not present in {args.results}")
            sort_column = fast_col
            rows[sort_column] = rows.apply(
                lambda row: fast_top_beneficiary_gain(row, args.adjusted_lambda),
                axis=1,
            )
    else:
        sort_column = sort_columns[args.sort_by]
        if sort_column not in rows.columns:
            raise ValueError(f"{sort_column} is not present in {args.results}")
    rows = rows.sort_values(sort_column, ascending=False)
    if not args.include_nonpositive:
        rows = rows[rows[sort_column] > 0]
    rows = rows.head(args.top_k).reset_index(drop=True)
    if rows.empty:
        print("No rows to visualize.")
        return

    clear_generated_visualizations(args.out_dir)
    metadata_by_match, frames_by_key = load_frames_for_rows(args.data_dir, rows)
    summaries = []
    for idx, row in rows.iterrows():
        rank = idx + 1
        match_id = row["match_id"]
        start_frame = frames_by_key[(match_id, int(row["start_frame"]))]
        end_frame = frames_by_key[(match_id, int(row["event_frame"]))]
        out_file = args.out_dir / f"rank_{rank:02d}.png"
        summaries.append(
            plot_flow_scene(
                row,
                rank,
                metadata_by_match[match_id],
                start_frame,
                end_frame,
                out_file,
                args.view,
                args.pass_vector_source,
                args.max_obso_labels,
                args.local_radius,
                args.local_samples,
                args.space_radius,
                args.space_samples,
                args.space_rings,
                args.sort_by,
                args.adjusted_lambda,
                args.show_runner_threat_map,
                args.threat_map_resolution,
                args.threat_map_max_ahead,
                args.threat_map_lateral_sigma,
                args.threat_map_goal_mix,
                args.hide_text,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_file = args.out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved {len(summary)} flow-arrow visualizations to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
