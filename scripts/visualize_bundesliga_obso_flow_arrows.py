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
)


DEFAULT_RESULTS = ROOT / "data" / "processed" / "bundesliga_obso_counterfactual_all_goal_threat.csv"
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "bundesliga-integrated"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "visualizations" / "bundesliga_obso_flow_arrows"


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
        if math.isfinite(value) and math.isfinite(space) and math.isfinite(dangerous_space):
            adjusted = value + adjusted_lambda * dangerous_space_norm(dangerous_space, space_radius)
            values.append((player_id, value, space, dangerous_space, adjusted))

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
) -> dict[str, float | str] | None:
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
        cf_obso, cf_space, cf_dangerous_space, cf_adjusted = cf_by_id.get(
            player_id,
            (0.0, 0.0, 0.0, 0.0),
        )
        item = {
            "player_id": player_id,
            "name": player_name(metadata, player_id),
            "actual_adjusted": float(adjusted),
            "counterfactual_adjusted": float(cf_adjusted),
            "adjusted_gain": float(adjusted - cf_adjusted),
            "actual_obso": float(obso),
            "counterfactual_obso": float(cf_obso),
            "obso_gain": float(obso - cf_obso),
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


def primary_benefit_type(item: dict[str, float | str]) -> str:
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


def scatter_players(
    ax,
    frame: BundesligaFrame,
    metadata: BundesligaMatchMeta,
    attacking_team_id: str,
    key_ids: set[str],
) -> None:
    for player in frame.players.values():
        is_attacker = player.team_id == attacking_team_id
        is_key = player.object_id in key_ids
        color = "#2f80ed" if is_attacker else "#9aa3ad"
        edge = "#114f9d" if is_attacker else "#4b5563"
        size = 58 if not is_key else 170
        alpha = 0.58 if not is_key else 1.0
        ax.scatter([player.x], [player.y], s=size, c=color, edgecolors=edge, linewidths=0.9, alpha=alpha, zorder=4 if is_key else 3)
        if is_key:
            ax.text(
                player.x,
                player.y - 1.6,
                player_number(metadata, player.object_id),
                ha="center",
                va="top",
                fontsize=8,
                color="#111827",
                weight="bold",
                zorder=8,
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


def draw_role_link(
    ax,
    state: BundesligaObjectState | None,
    target_xy: tuple[float, float] | None,
    label: str,
    color: str,
    marker: str,
    text_offset: tuple[float, float],
) -> None:
    if state is None:
        return
    kwargs = {"s": 185, "c": color, "marker": marker, "linewidths": 1.2, "zorder": 14}
    if marker != "x":
        kwargs["edgecolors"] = "#111827"
    ax.scatter([state.x], [state.y], **kwargs)
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
) -> None:
    if state is None:
        return
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


def draw_benefit_box(ax, beneficiary: dict[str, float | str] | None) -> None:
    if beneficiary is None:
        return
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

    if pass_vector_source == "players" and passer is not None and recipient is not None:
        pass_start = (passer.x, passer.y)
        pass_end = (recipient.x, recipient.y)
    else:
        pass_start = (float(row["ball_x"]), float(row["ball_y"]))
        pass_end = row_pass_end(row, recipient)

    key_ids = {
        pid
        for pid in [
            runner_id,
            passer_id,
            recipient_id,
            top_id,
            top_beneficiary_id,
            runner_defender_end.object_id if runner_defender_end else None,
            pass_lane_defender.object_id if pass_lane_defender else None,
            passer_pressure_defender.object_id if passer_pressure_defender else None,
        ]
        if pid
    }

    fig, ax = plt.subplots(figsize=(12.8, 8.6), dpi=170)
    draw_pitch(ax)
    scatter_players(ax, end_frame, metadata, attacking_team_id, key_ids)

    if pass_end is not None:
        draw_arrow(ax, pass_start, pass_end, color="#dca500", label="pass vector", linewidth=3.6, text_offset=(4.0, 4.0), zorder=12)
        ax.scatter([pass_start[0]], [pass_start[1]], s=90, c="#facc15", edgecolors="#713f12", linewidths=1.0, zorder=13)
        ax.scatter([pass_end[0]], [pass_end[1]], s=55, c="#fff7cc", edgecolors="#dca500", linewidths=1.0, zorder=12)

    if sort_by in {"runner-suppression", "runner-threat-suppression"} and runner_end is not None:
        draw_arrow(
            ax,
            pass_start,
            (runner_end.x, runner_end.y),
            color="#0f766e",
            label="runner option lane",
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
                label="threat point",
                linewidth=2.0,
                linestyle=":",
                text_offset=(1.6, 2.0),
                zorder=10,
            )
            ax.scatter(
                [runner_threat_point[0]],
                [runner_threat_point[1]],
                s=95,
                c="#fb923c",
                marker="x",
                linewidths=2.2,
                zorder=13,
            )

    if runner_defender_start is not None and runner_defender_end is not None:
        draw_arrow(
            ax,
            (runner_defender_start.x, runner_defender_start.y),
            (runner_defender_end.x, runner_defender_end.y),
            color="#c026d3",
            label="",
            linewidth=3.0,
            linestyle="--",
            text_offset=(0.0, 2.4),
            zorder=10,
        )

    draw_marker(ax, passer, "passer", "#f59e0b", "s", text_offset=(-3.6, -4.2))
    draw_marker(ax, recipient, "recipient", "#14b8a6", "^", text_offset=(-3.0, 4.0))
    if top_option is not None and top_id == recipient_id:
        draw_top_option_ring(ax, top_option)
    elif top_option is not None:
        draw_marker(ax, top_option, "top option", "#10b981", "D", text_offset=(0.0, 3.8))
    draw_beneficiary_marker(
        ax,
        top_beneficiary_state,
        float(top_beneficiary["adjusted_gain"]) if top_beneficiary is not None else None,
        top_beneficiary_id == top_id,
    )
    draw_marker(ax, runner_end, "runner", "#f97316", "o", text_offset=(4.2, -4.0))
    runner_target = runner_threat_point or ((runner_end.x, runner_end.y) if runner_end is not None else None)
    draw_role_link(
        ax,
        runner_defender_end,
        runner_target,
        "runner-affected",
        "#c026d3",
        "D",
        text_offset=(-5.4, 4.2),
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
        "pass-lane",
        "#dc2626",
        "X",
        text_offset=(4.6, -3.8),
    )
    passer_target = (passer.x, passer.y) if passer is not None else pass_start
    draw_role_link(
        ax,
        passer_pressure_defender,
        passer_target,
        "passer pressure",
        "#2563eb",
        "P",
        text_offset=(4.8, 3.8),
    )
    draw_obso_labels(ax, end_frame, obso_values, top_id, max_obso_labels, show_adjusted=option_sort_by == "adjusted")
    draw_benefit_box(ax, top_beneficiary)

    if view == "action":
        apply_action_view(
            ax,
            [
                pass_start,
                pass_end,
                state_xy(runner_start),
                state_xy(runner_end),
                runner_threat_point,
                state_xy(runner_defender_start),
                state_xy(runner_defender_end),
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
                state_xy(runner_defender_start),
                state_xy(runner_defender_end),
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
    subtitle = (
        f"passer {player_name(metadata, passer_id)} -> recipient {player_name(metadata, recipient_id)} | "
        f"non-receiving runner {player_name(metadata, runner_id)} | "
        f"runner-affected {runner_defender_name}"
    )
    pass_lane_name = row.get("pass_lane_suppressor_name")
    if pass_lane_name is not None and pd.notna(pass_lane_name):
        subtitle += f" | pass-lane {pass_lane_name}"
    passer_pressure_name = row.get("passer_pressure_defender_name")
    if passer_pressure_name is not None and pd.notna(passer_pressure_name):
        subtitle += f" | passer pressure {passer_pressure_name}"
    top_suppression_name = row.get("top_suppression_defender_name")
    if top_suppression_name is not None and pd.notna(top_suppression_name):
        subtitle += f" | option suppressor {top_suppression_name}"
    target_receiver_name = row.get("defender_target_receiver_name")
    defender_selection = row.get("defender_selection")
    if target_receiver_name is not None and pd.notna(target_receiver_name):
        subtitle += f" | target option {target_receiver_name}"
    if defender_selection is not None and pd.notna(defender_selection):
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
    if sort_by not in {"runner-suppression", "runner-threat-suppression"} and top_name and top_score is not None:
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

    fig.suptitle(title, fontsize=13, y=0.98, color="#111827")
    fig.text(0.5, 0.945, subtitle, ha="center", va="center", fontsize=9.5, color="#111827")
    fig.text(0.5, 0.918, option_text, ha="center", va="center", fontsize=9.0, color="#0f766e")
    fig.text(
        0.5,
        0.025,
        "yellow: actual pass | cyan star: top beneficiary by value gain | orange: runner, x=runner threat point | magenta: runner-affected | red: pass-lane | blue: passer pressure | green labels: A=adjusted, O=OBSO, S=post-reception space",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
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
        "top_beneficiary_adjusted_actual": (
            float(top_beneficiary["actual_adjusted"]) if top_beneficiary is not None else None
        ),
        "top_beneficiary_adjusted_counterfactual": (
            float(top_beneficiary["counterfactual_adjusted"]) if top_beneficiary is not None else None
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
        "top_beneficiary_dangerous_space_gain": (
            float(top_beneficiary["dangerous_space_gain"]) if top_beneficiary is not None else None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ball/runner/defender arrow views for Bundesliga OBSO candidates.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--view", choices=["action", "half", "full"], default="half")
    parser.add_argument("--pass-vector-source", choices=["players", "event"], default="players")
    parser.add_argument("--max-obso-labels", type=int, default=8)
    parser.add_argument("--local-radius", type=float, default=3.0)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--space-radius", type=float, default=5.0)
    parser.add_argument("--space-samples", type=int, default=16)
    parser.add_argument("--space-rings", type=int, default=3)
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
    parser.add_argument("--include-nonpositive", action="store_true")
    args = parser.parse_args()

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
        if exact_col in rows.columns:
            sort_column = exact_col
        else:
            for col in [actual_col, counterfactual_col]:
                if col not in rows.columns:
                    raise ValueError(f"{col} is not present in {args.results}")
            sort_column = "_fast_top_beneficiary_adjusted_gain"
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

    metadata_by_match, frames_by_key = load_frames_for_rows(args.data_dir, rows)
    summaries = []
    for idx, row in rows.iterrows():
        rank = idx + 1
        match_id = row["match_id"]
        start_frame = frames_by_key[(match_id, int(row["start_frame"]))]
        end_frame = frames_by_key[(match_id, int(row["event_frame"]))]
        out_file = args.out_dir / f"rank_{rank:02d}_{match_id}_frame_{int(row['event_frame'])}.png"
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
            )
        )

    summary = pd.DataFrame(summaries)
    summary_file = args.out_dir / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved {len(summary)} flow-arrow visualizations to {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
