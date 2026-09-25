"""Audit tables and lightweight SVG reports for attacker endpoint actions."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .action_space import (
    EndpointActionConfig,
    PlayerEndpointActionSet,
)
from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaMatchMeta,
)


def endpoint_actions_dataframe(
    match_id: str,
    frame_id: int,
    action_sets: Sequence[PlayerEndpointActionSet],
) -> pd.DataFrame:
    rows = []
    for action_set in action_sets:
        for action in action_set.actions:
            row = action.as_record(match_id, frame_id)
            row.update(
                {
                    "start_x": action_set.start_x,
                    "start_y": action_set.start_y,
                    "initial_vx_mps": action_set.initial_vx_mps,
                    "initial_vy_mps": action_set.initial_vy_mps,
                    "initial_speed_mps": action_set.initial_speed_mps,
                    "velocity_sample_count": action_set.velocity_sample_count,
                    "velocity_window_seconds": action_set.velocity_window_seconds,
                    "velocity_estimator": action_set.velocity_estimator,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def endpoint_summary_dataframe(
    match_id: str,
    frame_id: int,
    action_sets: Sequence[PlayerEndpointActionSet],
) -> pd.DataFrame:
    rows = []
    for action_set in action_sets:
        optimization = action_set.optimization_actions
        observed = next(
            (action for action in action_set.actions if "observed" in action.labels),
            None,
        )
        nearest_observed_distance = math.nan
        if observed is not None and optimization:
            nearest_observed_distance = min(
                math.hypot(
                    action.endpoint_x - observed.endpoint_x,
                    action.endpoint_y - observed.endpoint_y,
                )
                for action in optimization
            )
        rows.append(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "player_id": action_set.player_id,
                "start_x": action_set.start_x,
                "start_y": action_set.start_y,
                "initial_vx_mps": action_set.initial_vx_mps,
                "initial_vy_mps": action_set.initial_vy_mps,
                "initial_speed_mps": action_set.initial_speed_mps,
                "velocity_estimator": action_set.velocity_estimator,
                "optimization_action_count": len(optimization),
                "single_acceleration_action_count": sum(
                    action.motion.motion_model == "single_acceleration"
                    for action in optimization
                ),
                "brake_turn_action_count": sum(
                    action.motion.motion_model == "brake_turn_accelerate"
                    for action in optimization
                ),
                "grid_action_count": sum(
                    "grid" in action.labels for action in action_set.actions
                ),
                "hold_feasible": any(
                    "hold" in action.labels and action.motion.feasible
                    for action in action_set.actions
                ),
                "observed_available": observed is not None,
                "observed_feasible": observed.motion.feasible if observed else None,
                "observed_failure_reason": (
                    observed.motion.failure_reason if observed else None
                ),
                "observed_distance_m": (
                    observed.distance_from_start_m if observed else math.nan
                ),
                "nearest_action_to_observed_m": nearest_observed_distance,
            }
        )
    return pd.DataFrame(rows)


def select_speed_stratified_action_sets(
    action_sets_by_frame: Mapping[int, Sequence[PlayerEndpointActionSet]],
    scene_count: int = 6,
    players_per_scene: int = 3,
) -> list[tuple[int, PlayerEndpointActionSet]]:
    """Choose slow, median, and fast runners for compact visual QC."""

    selected = []
    for frame_id in sorted(action_sets_by_frame)[:scene_count]:
        ordered = sorted(
            action_sets_by_frame[frame_id],
            key=lambda item: item.initial_speed_mps,
        )
        if not ordered:
            continue
        if players_per_scene <= 1:
            indices = [len(ordered) - 1]
        else:
            indices = [
                round(index * (len(ordered) - 1) / (players_per_scene - 1))
                for index in range(players_per_scene)
            ]
        for index in dict.fromkeys(indices):
            selected.append((frame_id, ordered[index]))
    return selected


def _pitch_xy(x: float, y: float) -> tuple[float, float]:
    return x + FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0 - y


def _reference_action(
    action_set: PlayerEndpointActionSet,
    label: str,
):
    return next(
        (action for action in action_set.actions if label in action.labels),
        None,
    )


def _endpoint_svg(
    frame: BundesligaFrame,
    action_set: PlayerEndpointActionSet,
) -> str:
    svg = [
        '<svg class="pitch" viewBox="0 0 105 68" role="img" '
        f'aria-label="Reachable endpoints for {escape(action_set.player_id)}">',
        '<rect x="0" y="0" width="105" height="68" fill="#18743a" '
        'stroke="white" stroke-width="0.35"/>',
        '<line x1="52.5" y1="0" x2="52.5" y2="68" stroke="white" '
        'stroke-width="0.25"/>',
        '<circle cx="52.5" cy="34" r="9.15" fill="none" stroke="white" '
        'stroke-width="0.25"/>',
        '<rect x="0" y="13.84" width="16.5" height="40.32" fill="none" '
        'stroke="white" stroke-width="0.25"/>',
        '<rect x="88.5" y="13.84" width="16.5" height="40.32" fill="none" '
        'stroke="white" stroke-width="0.25"/>',
    ]

    for action in action_set.optimization_actions:
        if "grid" not in action.labels:
            continue
        x, y = _pitch_xy(action.endpoint_x, action.endpoint_y)
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.25" '
            'fill="#78ff9a" opacity="0.72"/>'
        )

    for player_id, player in frame.players.items():
        x, y = _pitch_xy(player.x, player.y)
        attacking = player.team_id == frame.players[action_set.player_id].team_id
        fill = "#ff8c42" if attacking else "#3f8efc"
        radius = 1.35 if player_id == action_set.player_id else 0.85
        stroke = "#ffe15d" if player_id == action_set.player_id else "white"
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.3"/>'
        )

    start_x, start_y = _pitch_xy(action_set.start_x, action_set.start_y)
    velocity_x, velocity_y = _pitch_xy(
        action_set.start_x + action_set.initial_vx_mps,
        action_set.start_y + action_set.initial_vy_mps,
    )
    svg.append(
        f'<line x1="{start_x:.2f}" y1="{start_y:.2f}" '
        f'x2="{velocity_x:.2f}" y2="{velocity_y:.2f}" '
        'stroke="#ffe15d" stroke-width="0.55"/>'
    )

    constant_velocity = _reference_action(action_set, "constant_velocity")
    if constant_velocity is not None:
        x, y = _pitch_xy(
            constant_velocity.endpoint_x,
            constant_velocity.endpoint_y,
        )
        svg.append(
            f'<path d="M {x:.2f},{y-0.9:.2f} L {x+0.9:.2f},{y:.2f} '
            f'L {x:.2f},{y+0.9:.2f} L {x-0.9:.2f},{y:.2f} Z" '
            'fill="#55d8ff" stroke="white" stroke-width="0.2"/>'
        )

    observed = _reference_action(action_set, "observed")
    if observed is not None:
        x, y = _pitch_xy(observed.endpoint_x, observed.endpoint_y)
        svg.append(
            f'<path d="M {x-1.0:.2f},{y-1.0:.2f} L {x+1.0:.2f},{y+1.0:.2f} '
            f'M {x+1.0:.2f},{y-1.0:.2f} L {x-1.0:.2f},{y+1.0:.2f}" '
            'stroke="#ff4d67" stroke-width="0.55"/>'
        )
    svg.append("</svg>")
    return "".join(svg)


def render_endpoint_audit_html(
    frames: Mapping[int, BundesligaFrame],
    selected: Sequence[tuple[int, PlayerEndpointActionSet]],
    metadata: BundesligaMatchMeta,
    config: EndpointActionConfig,
) -> str:
    cards = []
    for frame_id, action_set in selected:
        frame = frames[frame_id]
        player_meta = metadata.players.get(action_set.player_id)
        player_name = player_meta.short_name if player_meta else action_set.player_id
        observed = _reference_action(action_set, "observed")
        observed_text = "missing"
        if observed is not None:
            observed_text = (
                "feasible"
                if observed.motion.feasible
                else f"infeasible ({observed.motion.failure_reason})"
            )
        cards.append(
            f"""
            <article class="card">
              <h2>{escape(player_name)} · frame {frame_id}</h2>
              {_endpoint_svg(frame, action_set)}
              <dl>
                <dt>Initial speed</dt><dd>{action_set.initial_speed_mps:.2f} m/s</dd>
                <dt>Initial velocity</dt><dd>({action_set.initial_vx_mps:.2f}, {action_set.initial_vy_mps:.2f}) m/s</dd>
                <dt>Optimization endpoints</dt><dd>{len(action_set.optimization_actions)}</dd>
                <dt>Observed endpoint</dt><dd>{escape(observed_text)}</dd>
              </dl>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attacker endpoint action audit</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f3f5f7; color: #18212b; }}
header, main {{ max-width: 1500px; margin: auto; padding: 24px; }}
.note {{ background: #fff6d8; border-left: 4px solid #e5b100; padding: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); gap: 16px; }}
.card {{ background: white; border: 1px solid #dfe4e8; border-radius: 10px; padding: 14px; }}
.card h2 {{ margin: 0; font-size: 1rem; }}
.pitch {{ width: 100%; margin: 12px 0; background: #18743a; border-radius: 4px; }}
dl {{ display: grid; grid-template-columns: 155px 1fr; margin: 0; font-size: .88rem; }}
dt {{ color: #5a6773; }} dd {{ margin: 0 0 5px; }}
</style>
</head>
<body>
<header>
  <h1>Attacker endpoint action audit</h1>
  <p class="note">Green dots are feasible 1 m grid endpoints. The yellow line is one second of current velocity, the cyan diamond is the constant-velocity endpoint, and the red X is the observed two-second endpoint. The observed future is diagnostic only and never creates an optimizer-only action.</p>
  <p>H={config.horizon_seconds:.1f} s · grid={config.grid_resolution_m:.1f} m · shared max speed={config.max_speed_mps:.1f} m/s · shared max acceleration={config.max_acceleration_mps2:.1f} m/s²</p>
</header>
<main><div class="grid">{''.join(cards)}</div></main>
</body>
</html>"""


def write_endpoint_outputs(
    actions: pd.DataFrame,
    summary: pd.DataFrame,
    html: str,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actions_path = output / "attacker_endpoint_actions.csv"
    summary_path = output / "attacker_endpoint_summary.csv"
    audit_path = output / "attacker_endpoint_audit.html"
    actions.to_csv(actions_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit_path.write_text(html, encoding="utf-8")
    return actions_path, summary_path, audit_path
