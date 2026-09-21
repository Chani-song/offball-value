from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaObjectState,
)


GOAL_WIDTH = 7.32
DEFAULT_EPV_GRID_PATH = Path(__file__).resolve().parents[2] / "data" / "static" / "EPV_grid.csv"


def _as_points(points: Iterable[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


def _clip_points(points: np.ndarray) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], -FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0)
    clipped[:, 1] = np.clip(clipped[:, 1], -FIELD_WIDTH / 2.0, FIELD_WIDTH / 2.0)
    return clipped


def _team_xy(frame: BundesligaFrame, team_id: str) -> np.ndarray:
    return np.asarray(
        [(p.x, p.y) for p in frame.players.values() if p.team_id == team_id],
        dtype=float,
    )


def _opponent_xy(frame: BundesligaFrame, team_id: str) -> np.ndarray:
    return np.asarray(
        [(p.x, p.y) for p in frame.players.values() if p.team_id != team_id],
        dtype=float,
    )


def offside_line_projection(
    frame: BundesligaFrame,
    attacking_team_id: str,
    ball_xy: tuple[float, float],
    attacking_direction: int,
    tolerance: float = 0.2,
) -> float | None:
    defenders = [
        attacking_direction * player.x
        for player in frame.players.values()
        if player.team_id != attacking_team_id
    ]
    if len(defenders) < 2:
        return None

    second_last_defender = sorted(defenders, reverse=True)[1]
    ball_projection = attacking_direction * ball_xy[0]
    halfway_projection = 0.0
    return max(second_last_defender, ball_projection, halfway_projection) + tolerance


def is_offside_position(
    frame: BundesligaFrame,
    player_id: str,
    attacking_team_id: str,
    ball_xy: tuple[float, float],
    attacking_direction: int,
    tolerance: float = 0.2,
) -> bool:
    player = frame.players.get(player_id)
    if player is None or player.team_id != attacking_team_id:
        return False

    line = offside_line_projection(
        frame,
        attacking_team_id,
        ball_xy,
        attacking_direction,
        tolerance=tolerance,
    )
    if line is None:
        return False
    return attacking_direction * player.x > line


def _point_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
    projection = start + t[:, None] * segment
    return np.linalg.norm(points - projection, axis=1)


def _segment_projection_fraction(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return np.zeros(len(points), dtype=float)
    return ((points - start) @ segment) / denom


def _softmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    centered = values - float(np.max(values))
    exp_values = np.exp(centered)
    total = float(np.sum(exp_values))
    if total <= 1e-12:
        return np.full(len(values), 1.0 / len(values))
    return exp_values / total


def pitch_control_at_points(
    frame: BundesligaFrame,
    attacking_team_id: str,
    points: Iterable[tuple[float, float]],
    player_speed: float = 5.5,
    time_sigma: float = 0.45,
) -> np.ndarray:
    """Lightweight OBSO-style pitch-control proxy.

    This is intentionally a transparent first pass, not a calibrated Spearman
    model. It estimates which side can arrive first at each candidate point.
    """
    pts = _clip_points(_as_points(points))
    attackers = _team_xy(frame, attacking_team_id)
    defenders = _opponent_xy(frame, attacking_team_id)

    if len(attackers) == 0:
        return np.zeros(len(pts))
    if len(defenders) == 0:
        return np.ones(len(pts))

    att_dist = np.linalg.norm(pts[:, None, :] - attackers[None, :, :], axis=2).min(axis=1)
    def_dist = np.linalg.norm(pts[:, None, :] - defenders[None, :, :], axis=2).min(axis=1)
    att_time = att_dist / player_speed
    def_time = def_dist / player_speed
    logits = (def_time - att_time) / time_sigma
    return 1.0 / (1.0 + np.exp(-logits))


def receiver_space_control_at_points(
    frame: BundesligaFrame,
    receiver_id: str,
    points: Iterable[tuple[float, float]],
    player_speed: float = 5.5,
    defender_speed: float = 5.5,
    time_sigma: float = 0.45,
) -> np.ndarray:
    """Estimate whether the receiver can use each point after receiving.

    Unlike pitch_control_at_points, this is centered on the receiver with the
    ball. It asks whether the receiver can reach/turn into nearby space before
    the nearest defender can close it.
    """
    pts = _clip_points(_as_points(points))
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        return np.full(len(pts), np.nan)

    defenders = np.asarray(
        [(p.x, p.y) for p in frame.players.values() if p.team_id != receiver.team_id],
        dtype=float,
    )
    if len(defenders) == 0:
        return np.ones(len(pts))

    receiver_xy = np.asarray([receiver.x, receiver.y], dtype=float)
    rec_dist = np.linalg.norm(pts - receiver_xy, axis=1)
    def_dist = np.linalg.norm(pts[:, None, :] - defenders[None, :, :], axis=2).min(axis=1)
    rec_time = rec_dist / player_speed
    def_time = def_dist / defender_speed
    logits = (def_time - rec_time) / time_sigma
    return 1.0 / (1.0 + np.exp(-logits))


def transition_at_points(
    ball_xy: tuple[float, float],
    defender_xy: np.ndarray,
    points: Iterable[tuple[float, float]],
    pass_scale: float = 38.0,
    lane_center: float = 2.0,
    lane_sigma: float = 1.2,
) -> np.ndarray:
    """Pass reachability proxy from the current ball location to each point."""
    pts = _clip_points(_as_points(points))
    start = np.asarray(ball_xy, dtype=float)
    distances = np.linalg.norm(pts - start, axis=1)
    distance_score = np.exp(-distances / pass_scale)

    if len(defender_xy) == 0:
        return distance_score

    lane_scores = []
    for point in pts:
        lane_dist = _point_to_segment_distance(defender_xy, start, point)
        min_lane = float(np.min(lane_dist)) if len(lane_dist) else 10.0
        lane_scores.append(1.0 / (1.0 + math.exp(-(min_lane - lane_center) / lane_sigma)))
    return distance_score * np.asarray(lane_scores)


@lru_cache(maxsize=4)
def _load_epv_grid(path: str) -> np.ndarray | None:
    grid_path = Path(path)
    if not grid_path.exists():
        return None

    grid = np.loadtxt(grid_path, delimiter=",")
    if grid.ndim != 2:
        raise ValueError(f"EPV grid must be 2D, got shape {grid.shape}")

    max_value = float(np.nanmax(grid))
    if max_value <= 0:
        raise ValueError("EPV grid maximum must be positive")
    return np.asarray(grid / max_value, dtype=float)


def _epv_score_at_points(
    points: np.ndarray,
    attacking_direction: int,
    epv_grid: np.ndarray,
) -> np.ndarray:
    grid = np.fliplr(epv_grid) if attacking_direction < 0 else epv_grid
    rows, cols = grid.shape
    cell_w = FIELD_LENGTH / cols
    cell_h = FIELD_WIDTH / rows

    x_idx = np.floor((points[:, 0] + FIELD_LENGTH / 2.0) / cell_w).astype(int)
    y_idx = np.floor((points[:, 1] + FIELD_WIDTH / 2.0) / cell_h).astype(int)
    x_idx = np.clip(x_idx, 0, cols - 1)
    y_idx = np.clip(y_idx, 0, rows - 1)
    return grid[y_idx, x_idx]


def _geometric_score_at_points(
    points: Iterable[tuple[float, float]],
    attacking_direction: int,
) -> np.ndarray:
    """Fallback goal-danger proxy used when no EPV grid is available."""
    pts = _clip_points(_as_points(points))
    goal_x = FIELD_LENGTH / 2.0 if attacking_direction >= 0 else -FIELD_LENGTH / 2.0
    goal_center = np.asarray([goal_x, 0.0])
    goal_top = np.asarray([goal_x, GOAL_WIDTH / 2.0])
    goal_bottom = np.asarray([goal_x, -GOAL_WIDTH / 2.0])

    distances = np.linalg.norm(pts - goal_center, axis=1)
    dist_score = np.exp(-distances / 34.0)

    v1 = goal_top - pts
    v2 = goal_bottom - pts
    dot = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    angle = np.arccos(np.clip(dot / np.maximum(norms, 1e-9), -1.0, 1.0))
    angle_score = angle / math.pi

    progress = np.clip((attacking_direction * pts[:, 0] + FIELD_LENGTH / 2.0) / FIELD_LENGTH, 0.0, 1.0)
    return np.clip(0.45 * progress + 0.35 * dist_score + 0.20 * angle_score, 0.0, 1.0)


def score_at_points(
    points: Iterable[tuple[float, float]],
    attacking_direction: int,
    epv_grid_path: str | Path | None = DEFAULT_EPV_GRID_PATH,
) -> np.ndarray:
    """EPV-grid scoring component, with geometric fallback.

    The default grid follows the PAUSA-style static EPV surface: values are
    normalized to [0, 1] and flipped for attacks toward the left goal.
    """
    pts = _clip_points(_as_points(points))
    epv_grid = _load_epv_grid(str(epv_grid_path)) if epv_grid_path is not None else None
    if epv_grid is not None:
        return _epv_score_at_points(pts, attacking_direction, epv_grid)
    return _geometric_score_at_points(pts, attacking_direction)


def obso_at_points(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    points: Iterable[tuple[float, float]],
    ball_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    pts = _clip_points(_as_points(points))
    if ball_xy is None:
        if frame.ball is None:
            raise ValueError("ball_xy is required when the frame has no ball state")
        ball_xy = (frame.ball.x, frame.ball.y)

    defenders = _opponent_xy(frame, attacking_team_id)
    ppcf = pitch_control_at_points(frame, attacking_team_id, pts)
    transition = transition_at_points(ball_xy, defenders, pts)
    score = score_at_points(pts, attacking_direction)
    return ppcf * transition * score


def local_points_around(
    x: float,
    y: float,
    radius: float = 3.0,
    samples: int = 8,
) -> list[tuple[float, float]]:
    points = [(x, y)]
    if radius <= 0 or samples <= 0:
        return points
    for idx in range(samples):
        theta = 2.0 * math.pi * idx / samples
        points.append((x + radius * math.cos(theta), y + radius * math.sin(theta)))
    return [(float(np.clip(px, -FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0)),
             float(np.clip(py, -FIELD_WIDTH / 2.0, FIELD_WIDTH / 2.0))) for px, py in points]


def _disk_points_around(
    x: float,
    y: float,
    radius: float,
    samples: int,
    rings: int,
) -> np.ndarray:
    if radius <= 0 or samples <= 0 or rings <= 0:
        return np.asarray([(x, y)], dtype=float)

    points = []
    for ring_idx in range(rings):
        ring_radius = radius * math.sqrt((ring_idx + 0.5) / rings)
        angle_offset = (ring_idx % 2) * math.pi / samples
        for sample_idx in range(samples):
            theta = 2.0 * math.pi * sample_idx / samples + angle_offset
            points.append((x + ring_radius * math.cos(theta), y + ring_radius * math.sin(theta)))
    return np.asarray(points, dtype=float)


def _inside_pitch(points: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] >= -FIELD_LENGTH / 2.0)
        & (points[:, 0] <= FIELD_LENGTH / 2.0)
        & (points[:, 1] >= -FIELD_WIDTH / 2.0)
        & (points[:, 1] <= FIELD_WIDTH / 2.0)
    )


def receiver_post_reception_space_score(
    frame: BundesligaFrame,
    receiver_id: str,
    space_radius: float = 5.0,
    space_samples: int = 16,
    space_rings: int = 3,
) -> float:
    """Normalized usable space around the receiver after the pass is received.

    The value is the average receiver-vs-defender control over a disk around the
    receiver. Off-pitch sample points count as unusable space.
    """
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        return float("nan")

    points = _disk_points_around(receiver.x, receiver.y, space_radius, space_samples, space_rings)
    if len(points) == 0:
        return float("nan")

    inside = _inside_pitch(points)
    if not np.any(inside):
        return 0.0

    controlled = np.zeros(len(points), dtype=float)
    controlled[inside] = receiver_space_control_at_points(frame, receiver_id, points[inside])
    return float(np.nanmean(controlled))


def receiver_post_reception_space_area(
    frame: BundesligaFrame,
    receiver_id: str,
    space_radius: float = 5.0,
    space_samples: int = 16,
    space_rings: int = 3,
) -> float:
    """Usable post-reception space in square meters."""
    if space_radius <= 0:
        return 0.0
    score = receiver_post_reception_space_score(
        frame,
        receiver_id,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
    )
    if math.isnan(score):
        return score
    return float(score * math.pi * space_radius**2)


def receiver_post_reception_dangerous_space_value(
    frame: BundesligaFrame,
    receiver_id: str,
    attacking_direction: int,
    space_radius: float = 5.0,
    space_samples: int = 16,
    space_rings: int = 3,
) -> float:
    """EPV-weighted post-reception space around the receiver."""
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        return float("nan")
    if space_radius <= 0:
        return 0.0

    points = _disk_points_around(receiver.x, receiver.y, space_radius, space_samples, space_rings)
    inside = _inside_pitch(points)
    if not np.any(inside):
        return 0.0

    weighted = np.zeros(len(points), dtype=float)
    inside_points = points[inside]
    controlled = receiver_space_control_at_points(frame, receiver_id, inside_points)
    score = score_at_points(inside_points, attacking_direction)
    weighted[inside] = controlled * score
    return float(np.nanmean(weighted) * math.pi * space_radius**2)


def defender_responsibilities_for_receiver(
    frame: BundesligaFrame,
    attacking_team_id: str,
    receiver_id: str,
    ball_xy: tuple[float, float],
    attacking_direction: int,
    lane_scale: float = 4.0,
    receiver_pressure_scale: float = 9.0,
    goal_side_scale: float = 8.0,
) -> list[dict[str, float | str]]:
    """DEFCON-inspired responsibility proxy for blocking a receiver option.

    This is not a learned GNN model. It assigns responsibility to defenders
    according to three interpretable cues: pass-lane coverage, pressure on the
    receiver, and goal-side coverage between the receiver and the goal.
    """
    receiver = frame.players.get(receiver_id)
    if receiver is None or receiver.team_id != attacking_team_id:
        return []

    defenders = [
        player
        for player in frame.players.values()
        if player.team_id != attacking_team_id
    ]
    if not defenders:
        return []

    defender_xy = np.asarray([(player.x, player.y) for player in defenders], dtype=float)
    ball = np.asarray(ball_xy, dtype=float)
    receiver_xy = np.asarray([receiver.x, receiver.y], dtype=float)
    goal_x = FIELD_LENGTH / 2.0 if attacking_direction >= 0 else -FIELD_LENGTH / 2.0
    goal_xy = np.asarray([goal_x, 0.0], dtype=float)

    pass_lane_dist = _point_to_segment_distance(defender_xy, ball, receiver_xy)
    receiver_dist = np.linalg.norm(defender_xy - receiver_xy, axis=1)
    goal_side_dist = _point_to_segment_distance(defender_xy, receiver_xy, goal_xy)
    goal_projection = _segment_projection_fraction(defender_xy, receiver_xy, goal_xy)
    between_receiver_goal = (goal_projection >= 0.0) & (goal_projection <= 1.0)

    lane_score = np.exp(-0.5 * (pass_lane_dist / lane_scale) ** 2)
    pressure_score = np.exp(-receiver_dist / receiver_pressure_scale)
    goal_side_score = np.exp(-0.5 * (goal_side_dist / goal_side_scale) ** 2) * between_receiver_goal
    raw_score = 0.45 * lane_score + 0.35 * pressure_score + 0.20 * goal_side_score

    responsibility = raw_score / float(np.sum(raw_score)) if float(np.sum(raw_score)) > 1e-12 else _softmax(raw_score)
    out = []
    for idx, defender in enumerate(defenders):
        out.append(
            {
                "defender_id": defender.object_id,
                "responsibility": float(responsibility[idx]),
                "raw_score": float(raw_score[idx]),
                "lane_score": float(lane_score[idx]),
                "pressure_score": float(pressure_score[idx]),
                "goal_side_score": float(goal_side_score[idx]),
                "distance_to_receiver": float(receiver_dist[idx]),
                "distance_to_pass_lane": float(pass_lane_dist[idx]),
            }
        )
    return sorted(out, key=lambda item: float(item["responsibility"]), reverse=True)


def defender_responsibilities_for_point(
    frame: BundesligaFrame,
    attacking_team_id: str,
    target_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    attacking_direction: int,
    lane_scale: float = 4.0,
    receiver_pressure_scale: float = 9.0,
    goal_side_scale: float = 8.0,
) -> list[dict[str, float | str]]:
    """Responsibility proxy for defending an arbitrary receive/threat point."""
    defenders = [
        player
        for player in frame.players.values()
        if player.team_id != attacking_team_id
    ]
    if not defenders:
        return []

    defender_xy = np.asarray([(player.x, player.y) for player in defenders], dtype=float)
    ball = np.asarray(ball_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    goal_x = FIELD_LENGTH / 2.0 if attacking_direction >= 0 else -FIELD_LENGTH / 2.0
    goal_xy = np.asarray([goal_x, 0.0], dtype=float)

    pass_lane_dist = _point_to_segment_distance(defender_xy, ball, target)
    target_dist = np.linalg.norm(defender_xy - target, axis=1)
    goal_side_dist = _point_to_segment_distance(defender_xy, target, goal_xy)
    goal_projection = _segment_projection_fraction(defender_xy, target, goal_xy)
    between_target_goal = (goal_projection >= 0.0) & (goal_projection <= 1.0)

    lane_score = np.exp(-0.5 * (pass_lane_dist / lane_scale) ** 2)
    pressure_score = np.exp(-target_dist / receiver_pressure_scale)
    goal_side_score = np.exp(-0.5 * (goal_side_dist / goal_side_scale) ** 2) * between_target_goal
    raw_score = 0.45 * lane_score + 0.35 * pressure_score + 0.20 * goal_side_score

    responsibility = raw_score / float(np.sum(raw_score)) if float(np.sum(raw_score)) > 1e-12 else _softmax(raw_score)
    out = []
    for idx, defender in enumerate(defenders):
        out.append(
            {
                "defender_id": defender.object_id,
                "responsibility": float(responsibility[idx]),
                "raw_score": float(raw_score[idx]),
                "lane_score": float(lane_score[idx]),
                "pressure_score": float(pressure_score[idx]),
                "goal_side_score": float(goal_side_score[idx]),
                "distance_to_receiver": float(target_dist[idx]),
                "distance_to_pass_lane": float(pass_lane_dist[idx]),
            }
        )
    return sorted(out, key=lambda item: float(item["responsibility"]), reverse=True)


def projected_runner_threat_point(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
    projection_m: float = 6.0,
) -> tuple[float, float] | None:
    """Project the runner's end point into the space they are attacking."""
    runner_start = start_frame.players.get(runner_id)
    runner_end = end_frame.players.get(runner_id)
    if runner_start is None or runner_end is None:
        return None

    runner_vec = np.asarray([runner_end.x - runner_start.x, runner_end.y - runner_start.y], dtype=float)
    runner_norm = float(np.linalg.norm(runner_vec))
    end_xy = np.asarray([[runner_end.x, runner_end.y]], dtype=float)
    if runner_norm <= 1e-6 or projection_m <= 0:
        threat_xy = end_xy
    else:
        unit = runner_vec / runner_norm
        threat_xy = end_xy + projection_m * unit.reshape(1, 2)
    clipped = _clip_points(threat_xy)[0]
    return (float(clipped[0]), float(clipped[1]))


def affected_defender_for_runner(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
    target_receiver_id: str | None = None,
    attacking_team_id: str | None = None,
    ball_xy: tuple[float, float] | None = None,
    attacking_direction: int = 1,
    defender_selection: str = "responsibility",
) -> BundesligaObjectState | None:
    """Select the defender most likely affected by a non-receiving runner.

    The responsibility mode first asks which defenders are plausibly influenced
    by the runner, then uses target-option responsibility only as a tie-breaker.
    """
    runner_start = start_frame.players.get(runner_id)
    runner_end = end_frame.players.get(runner_id)
    if runner_start is None or runner_end is None:
        return None

    defenders = [
        player
        for player in end_frame.players.values()
        if player.team_id != runner_end.team_id and player.object_id in start_frame.players
    ]
    if not defenders:
        return None

    if defender_selection == "closest-runner":
        return min(defenders, key=lambda player: (player.x - runner_end.x) ** 2 + (player.y - runner_end.y) ** 2)

    runner_threat_xy_tuple = projected_runner_threat_point(start_frame, end_frame, runner_id)
    if runner_threat_xy_tuple is None:
        return None
    runner_threat_xy = np.asarray(runner_threat_xy_tuple, dtype=float)

    responsibility_by_id: dict[str, float] = {}
    if attacking_team_id and ball_xy is not None:
        responsibility_by_id = {
            str(item["defender_id"]): float(item["responsibility"])
            for item in defender_responsibilities_for_point(
                end_frame,
                attacking_team_id,
                runner_threat_xy_tuple,
                ball_xy,
                attacking_direction,
            )
        }

    runner_vec = np.asarray([runner_end.x - runner_start.x, runner_end.y - runner_start.y], dtype=float)
    runner_norm = float(np.linalg.norm(runner_vec))
    runner_start_xy = np.asarray([runner_start.x, runner_start.y], dtype=float)
    runner_end_xy = np.asarray([runner_end.x, runner_end.y], dtype=float)
    max_responsibility = max(responsibility_by_id.values(), default=0.0)

    best_defender = None
    best_score = -1.0
    for defender in defenders:
        defender_start = start_frame.players[defender.object_id]
        defender_vec = np.asarray([defender.x - defender_start.x, defender.y - defender_start.y], dtype=float)
        defender_norm = float(np.linalg.norm(defender_vec))
        runner_distance = math.hypot(defender.x - runner_threat_xy[0], defender.y - runner_threat_xy[1])
        defender_xy = np.asarray([[defender.x, defender.y]], dtype=float)
        path_distance = float(_point_to_segment_distance(defender_xy, runner_start_xy, runner_threat_xy)[0])
        if runner_distance > 10.0 and path_distance > 6.0:
            continue

        runner_proximity = math.exp(-runner_distance / 6.0)
        path_proximity = math.exp(-0.5 * (path_distance / 5.0) ** 2)

        follow_score = 0.0
        if runner_norm > 1e-6 and defender_norm > 1e-6:
            cosine = float(np.dot(runner_vec, defender_vec) / (runner_norm * defender_norm))
            follow_score = max(0.0, cosine) * min(1.0, defender_norm / max(runner_norm, 1e-6))

        defender_start_xy = np.asarray([defender_start.x, defender_start.y], dtype=float)
        start_threat_distance = float(np.linalg.norm(defender_start_xy - runner_threat_xy))
        closing_score = 0.0
        if runner_norm > 1e-6:
            closing_score = max(0.0, min(1.0, (start_threat_distance - runner_distance) / runner_norm))

        responsibility = responsibility_by_id.get(defender.object_id, 0.0)
        influence_score = (
            0.35 * runner_proximity
            + 0.25 * path_proximity
            + 0.20 * follow_score
            + 0.20 * closing_score
        )
        if responsibility_by_id:
            responsibility_score = responsibility / max_responsibility if max_responsibility > 1e-12 else 0.0
            score = influence_score * (0.65 + 0.35 * responsibility_score)
        else:
            score = influence_score

        if score > best_score:
            best_score = score
            best_defender = defender

    return best_defender


def receiver_option_value(
    frame: BundesligaFrame,
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float] | None = None,
    local_radius: float = 3.0,
    local_samples: int = 8,
) -> float:
    receiver = frame.players.get(receiver_id)
    if receiver is None:
        return float("nan")
    points = local_points_around(receiver.x, receiver.y, radius=local_radius, samples=local_samples)
    return float(np.nanmax(obso_at_points(frame, attacking_team_id, attacking_direction, points, ball_xy=ball_xy)))


def team_option_values(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    excluded_player_ids: set[str] | None = None,
    ball_xy: tuple[float, float] | None = None,
    local_radius: float = 3.0,
    local_samples: int = 8,
) -> list[tuple[str, float]]:
    excluded = excluded_player_ids or set()
    values = []
    for player in frame.players.values():
        if player.team_id != attacking_team_id or player.object_id in excluded:
            continue
        value = receiver_option_value(
            frame,
            player.object_id,
            attacking_team_id,
            attacking_direction,
            ball_xy=ball_xy,
            local_radius=local_radius,
            local_samples=local_samples,
        )
        if not math.isnan(value):
            values.append((player.object_id, value))
    return sorted(values, key=lambda item: item[1], reverse=True)


def team_topk_option_value(
    frame: BundesligaFrame,
    attacking_team_id: str,
    attacking_direction: int,
    k: int = 3,
    excluded_player_ids: set[str] | None = None,
    ball_xy: tuple[float, float] | None = None,
    local_radius: float = 3.0,
    local_samples: int = 8,
) -> tuple[float, list[tuple[str, float]]]:
    values = team_option_values(
        frame,
        attacking_team_id,
        attacking_direction,
        excluded_player_ids=excluded_player_ids,
        ball_xy=ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
    )
    top = values[: max(1, k)]
    return float(sum(value for _, value in top)), top


def move_player(frame: BundesligaFrame, player_id: str, x: float, y: float) -> BundesligaFrame:
    player = frame.players.get(player_id)
    if player is None:
        return frame
    return frame.with_player(player_id, replace(player, x=float(x), y=float(y)))


def no_response_defense_frame(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    attacking_team_id: str,
) -> BundesligaFrame:
    """Keep attacking movement actual, but freeze defenders at start positions."""
    frame = end_frame
    for player in end_frame.players.values():
        if player.team_id == attacking_team_id or player.object_id not in start_frame.players:
            continue
        start = start_frame.players[player.object_id]
        frame = move_player(frame, player.object_id, start.x, start.y)
    return frame


def defender_response_removed_frame(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    defender_id: str,
) -> BundesligaFrame:
    """Freeze one defender at the start-frame position while leaving others actual."""
    start = start_frame.players.get(defender_id)
    end = end_frame.players.get(defender_id)
    if start is None or end is None:
        return end_frame
    return move_player(end_frame, defender_id, start.x, start.y)


def receiver_option_value_legal(
    frame: BundesligaFrame,
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    local_radius: float = 3.0,
    local_samples: int = 8,
    filter_offside: bool = True,
) -> float:
    if filter_offside and is_offside_position(
        frame,
        receiver_id,
        attacking_team_id,
        ball_xy,
        attacking_direction,
    ):
        return 0.0
    return receiver_option_value(
        frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
    )


def receiver_threat_value_legal(
    frame: BundesligaFrame,
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    local_radius: float = 3.0,
    local_samples: int = 8,
    space_radius: float = 5.0,
    space_samples: int = 16,
    space_rings: int = 3,
    threat_lambda: float = 0.5,
    filter_offside: bool = True,
) -> float:
    if filter_offside and is_offside_position(
        frame,
        receiver_id,
        attacking_team_id,
        ball_xy,
        attacking_direction,
    ):
        return 0.0

    option_value = receiver_option_value(
        frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy=ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
    )
    dangerous_space = receiver_post_reception_dangerous_space_value(
        frame,
        receiver_id,
        attacking_direction,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
    )
    if not (math.isfinite(option_value) and math.isfinite(dangerous_space)):
        return float("nan")

    max_space_area = math.pi * space_radius**2 if space_radius > 0 else 1.0
    dangerous_space_norm = max(0.0, min(1.0, dangerous_space / max_space_area))
    return float(option_value + threat_lambda * dangerous_space_norm)


def receiver_defensive_suppression_attribution(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    receiver_id: str,
    attacking_team_id: str,
    attacking_direction: int,
    ball_xy: tuple[float, float],
    local_radius: float = 3.0,
    local_samples: int = 8,
    space_radius: float = 5.0,
    space_samples: int = 16,
    space_rings: int = 3,
    threat_lambda: float = 0.5,
    filter_offside: bool = True,
) -> dict[str, object]:
    """Attribute how much defensive response suppresses a receiver option.

    The no-response baseline keeps the receiver and other attackers at their
    actual end-frame positions, while freezing all defenders at start-frame
    positions. Defender-level attribution uses a one-defender-at-a-time version
    of that rollback and allocates the total positive suppression across the
    positive marginal contributions.
    """
    actual_value = receiver_option_value_legal(
        end_frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
        filter_offside=filter_offside,
    )
    no_response_frame = no_response_defense_frame(start_frame, end_frame, attacking_team_id)
    no_response_value = receiver_option_value_legal(
        no_response_frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
        filter_offside=filter_offside,
    )
    suppression = no_response_value - actual_value
    positive_suppression = max(0.0, suppression)
    actual_threat_value = receiver_threat_value_legal(
        end_frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
        threat_lambda=threat_lambda,
        filter_offside=filter_offside,
    )
    no_response_threat_value = receiver_threat_value_legal(
        no_response_frame,
        receiver_id,
        attacking_team_id,
        attacking_direction,
        ball_xy,
        local_radius=local_radius,
        local_samples=local_samples,
        space_radius=space_radius,
        space_samples=space_samples,
        space_rings=space_rings,
        threat_lambda=threat_lambda,
        filter_offside=filter_offside,
    )
    threat_suppression = no_response_threat_value - actual_threat_value
    positive_threat_suppression = max(0.0, threat_suppression)

    receiver = end_frame.players.get(receiver_id)
    receiver_start = start_frame.players.get(receiver_id)
    ball = np.asarray(ball_xy, dtype=float)
    receiver_xy = np.asarray([receiver.x, receiver.y], dtype=float) if receiver is not None else None
    receiver_start_xy = (
        np.asarray([receiver_start.x, receiver_start.y], dtype=float)
        if receiver_start is not None
        else None
    )
    receiver_vec = (
        receiver_xy - receiver_start_xy
        if receiver_xy is not None and receiver_start_xy is not None
        else None
    )
    receiver_norm = float(np.linalg.norm(receiver_vec)) if receiver_vec is not None else 0.0
    goal_x = FIELD_LENGTH / 2.0 if attacking_direction >= 0 else -FIELD_LENGTH / 2.0
    goal_xy = np.asarray([goal_x, 0.0], dtype=float)
    defenders = [
        player
        for player in end_frame.players.values()
        if player.team_id != attacking_team_id and player.object_id in start_frame.players
    ]

    attributions = []
    for defender in defenders:
        released_frame = defender_response_removed_frame(start_frame, end_frame, defender.object_id)
        released_value = receiver_option_value_legal(
            released_frame,
            receiver_id,
            attacking_team_id,
            attacking_direction,
            ball_xy,
            local_radius=local_radius,
            local_samples=local_samples,
            filter_offside=filter_offside,
        )
        raw_contribution = max(0.0, released_value - actual_value)
        released_threat_value = receiver_threat_value_legal(
            released_frame,
            receiver_id,
            attacking_team_id,
            attacking_direction,
            ball_xy,
            local_radius=local_radius,
            local_samples=local_samples,
            space_radius=space_radius,
            space_samples=space_samples,
            space_rings=space_rings,
            threat_lambda=threat_lambda,
            filter_offside=filter_offside,
        )
        raw_threat_contribution = max(0.0, released_threat_value - actual_threat_value)
        defender_xy = np.asarray([[defender.x, defender.y]], dtype=float)
        defender_start = start_frame.players[defender.object_id]
        defender_vec = np.asarray([defender.x - defender_start.x, defender.y - defender_start.y], dtype=float)
        defender_norm = float(np.linalg.norm(defender_vec))
        distance_to_receiver = (
            float(np.linalg.norm(defender_xy[0] - receiver_xy))
            if receiver_xy is not None
            else float("nan")
        )
        distance_to_runner_path = (
            float(_point_to_segment_distance(defender_xy, receiver_start_xy, receiver_xy)[0])
            if receiver_start_xy is not None and receiver_xy is not None
            else float("nan")
        )
        follow_score = 0.0
        if receiver_norm > 1e-6 and defender_norm > 1e-6 and receiver_vec is not None:
            cosine = float(np.dot(receiver_vec, defender_vec) / (receiver_norm * defender_norm))
            follow_score = max(0.0, cosine) * min(1.0, defender_norm / max(receiver_norm, 1e-6))
        distance_to_pass_lane = (
            float(_point_to_segment_distance(defender_xy, ball, receiver_xy)[0])
            if receiver_xy is not None
            else float("nan")
        )
        pass_lane_projection = (
            float(_segment_projection_fraction(defender_xy, ball, receiver_xy)[0])
            if receiver_xy is not None
            else float("nan")
        )
        distance_to_passer = float(np.linalg.norm(defender_xy[0] - ball))
        goal_side_distance = (
            float(_point_to_segment_distance(defender_xy, receiver_xy, goal_xy)[0])
            if receiver_xy is not None
            else float("nan")
        )
        goal_side_projection = (
            float(_segment_projection_fraction(defender_xy, receiver_xy, goal_xy)[0])
            if receiver_xy is not None
            else float("nan")
        )
        if math.isfinite(goal_side_distance) and math.isfinite(goal_side_projection):
            line_score = math.exp(-0.5 * (goal_side_distance / 8.0) ** 2)
            if 0.0 <= goal_side_projection <= 1.0:
                front_weight = 1.0
            elif goal_side_projection < 0.0:
                front_weight = 0.25 * math.exp(goal_side_projection / 0.35)
            else:
                front_weight = 0.50 * math.exp(-(goal_side_projection - 1.0) / 0.35)
            goal_side_score = line_score * front_weight
        else:
            goal_side_score = 0.0
        if math.isfinite(distance_to_pass_lane) and math.isfinite(pass_lane_projection):
            lane_between = 1.0 if 0.0 <= pass_lane_projection <= 1.0 else 0.0
            pass_lane_score = math.exp(-0.5 * (distance_to_pass_lane / 4.0) ** 2) * lane_between
        else:
            pass_lane_score = 0.0
        passer_pressure_score = math.exp(-distance_to_passer / 7.0) if math.isfinite(distance_to_passer) else 0.0
        attributions.append(
            {
                "defender_id": defender.object_id,
                "raw_contribution": float(raw_contribution),
                "allocated_contribution": 0.0,
                "released_value": float(released_value),
                "raw_threat_contribution": float(raw_threat_contribution),
                "allocated_threat_contribution": 0.0,
                "released_threat_value": float(released_threat_value),
                "distance_to_receiver": distance_to_receiver,
                "distance_to_runner_path": distance_to_runner_path,
                "distance_to_pass_lane": distance_to_pass_lane,
                "pass_lane_projection": pass_lane_projection,
                "pass_lane_score": float(pass_lane_score),
                "distance_to_passer": distance_to_passer,
                "passer_pressure_score": float(passer_pressure_score),
                "goal_side_distance": goal_side_distance,
                "goal_side_projection": goal_side_projection,
                "goal_side_score": float(goal_side_score),
                "defender_movement_m": defender_norm,
                "follow_score": float(follow_score),
            }
        )

    raw_total = sum(float(item["raw_contribution"]) for item in attributions)
    if raw_total > 1e-12 and positive_suppression > 0:
        for item in attributions:
            item["allocated_contribution"] = (
                float(item["raw_contribution"]) / raw_total * positive_suppression
            )
    raw_threat_total = sum(float(item["raw_threat_contribution"]) for item in attributions)
    if raw_threat_total > 1e-12 and positive_threat_suppression > 0:
        for item in attributions:
            item["allocated_threat_contribution"] = (
                float(item["raw_threat_contribution"])
                / raw_threat_total
                * positive_threat_suppression
            )

    attributions.sort(
        key=lambda item: (
            float(item["allocated_threat_contribution"]),
            float(item["raw_threat_contribution"]),
            float(item["allocated_contribution"]),
        ),
        reverse=True,
    )

    return {
        "actual_value": float(actual_value),
        "no_response_value": float(no_response_value),
        "suppression": float(suppression),
        "positive_suppression": float(positive_suppression),
        "actual_threat_value": float(actual_threat_value),
        "no_response_threat_value": float(no_response_threat_value),
        "threat_suppression": float(threat_suppression),
        "positive_threat_suppression": float(positive_threat_suppression),
        "threat_lambda": float(threat_lambda),
        "actual_offside": bool(
            is_offside_position(end_frame, receiver_id, attacking_team_id, ball_xy, attacking_direction)
        ),
        "no_response_offside": bool(
            is_offside_position(no_response_frame, receiver_id, attacking_team_id, ball_xy, attacking_direction)
        ),
        "attributions": attributions,
    }


def counterfactual_frame_for_runner(
    start_frame: BundesligaFrame,
    end_frame: BundesligaFrame,
    runner_id: str,
    runner_alpha: float = 0.0,
    defender_alpha: float = 0.3,
    affected_defender_id: str | None = None,
    target_receiver_id: str | None = None,
    attacking_team_id: str | None = None,
    ball_xy: tuple[float, float] | None = None,
    attacking_direction: int = 1,
    defender_selection: str = "closest-runner",
    fallback_to_selected_defender: bool = True,
) -> BundesligaFrame:
    """Rollback runner movement and one defender response.

    runner_alpha=0 freezes the runner at the start-frame position. The affected
    defender is partially pulled back toward its own start-frame position.
    """
    runner_start = start_frame.players.get(runner_id)
    runner_end = end_frame.players.get(runner_id)
    if runner_start is None or runner_end is None:
        return end_frame

    cf = move_player(
        end_frame,
        runner_id,
        (1.0 - runner_alpha) * runner_start.x + runner_alpha * runner_end.x,
        (1.0 - runner_alpha) * runner_start.y + runner_alpha * runner_end.y,
    )

    defenders = [
        p for p in end_frame.players.values()
        if p.team_id != runner_end.team_id and p.object_id in start_frame.players
    ]
    if not defenders:
        return cf

    affected = None
    if affected_defender_id:
        affected = end_frame.players.get(affected_defender_id)
    if (
        fallback_to_selected_defender
        and (affected is None or affected.object_id not in start_frame.players or affected.team_id == runner_end.team_id)
    ):
        affected = affected_defender_for_runner(
            start_frame,
            end_frame,
            runner_id,
            target_receiver_id=target_receiver_id,
            attacking_team_id=attacking_team_id,
            ball_xy=ball_xy,
            attacking_direction=attacking_direction,
            defender_selection=defender_selection,
        )
    if affected is None or affected.object_id not in start_frame.players or affected.team_id == runner_end.team_id:
        return cf

    defender_start = start_frame.players[affected.object_id]
    return move_player(
        cf,
        affected.object_id,
        (1.0 - defender_alpha) * defender_start.x + defender_alpha * affected.x,
        (1.0 - defender_alpha) * defender_start.y + defender_alpha * affected.y,
    )
