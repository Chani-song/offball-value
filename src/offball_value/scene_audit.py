"""Human-auditable HTML and CSV outputs for extracted decision scenes."""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
import random
from typing import Mapping, Sequence

import pandas as pd

from .bundesliga import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    BundesligaFrame,
    BundesligaMatchMeta,
)
from .scene_extractor import SceneCandidate, SceneExtractionResult, SceneExtractorConfig


def write_scene_candidates_csv(
    result: SceneExtractionResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_dataframe().to_csv(path, index=False)
    return path


def select_audit_candidates(
    candidates: Sequence[SceneCandidate],
    config: SceneExtractorConfig,
    accepted_count: int = 20,
    borderline_count: int = 20,
    rejected_count: int = 20,
    seed: int = 0,
) -> list[SceneCandidate]:
    """Select accepted, threshold-borderline, and reason-stratified scenes."""

    rng = random.Random(seed)
    accepted = [candidate for candidate in candidates if candidate.accepted]
    rejected = [candidate for candidate in candidates if not candidate.accepted]
    rng.shuffle(accepted)

    selected: list[SceneCandidate] = accepted[:accepted_count]
    selected_ids = {(candidate.match_id, candidate.frame_id) for candidate in selected}

    def boundary_distance(candidate: SceneCandidate) -> float:
        distance_margin = (
            abs(candidate.ball_carrier_distance_m - config.control_distance_m)
            / config.control_distance_m
            if candidate.ball_carrier_distance_m is not None
            else 10.0
        )
        stability_margin = abs(
            candidate.stable_control_fraction - config.minimum_stable_fraction
        )
        return min(distance_margin, stability_margin)

    borderline_pool = sorted(candidates, key=boundary_distance)
    added_borderline = 0
    for candidate in borderline_pool:
        key = (candidate.match_id, candidate.frame_id)
        if key in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(key)
        added_borderline += 1
        if added_borderline >= borderline_count:
            break

    by_reason: dict[str, list[SceneCandidate]] = defaultdict(list)
    for candidate in rejected:
        for reason in candidate.rejection_reasons:
            by_reason[reason].append(candidate)
    reason_order = sorted(by_reason)
    reason_cursor = 0
    added_rejected = 0
    while reason_order and added_rejected < rejected_count:
        reason = reason_order[reason_cursor % len(reason_order)]
        pool = by_reason[reason]
        rng.shuffle(pool)
        candidate = next(
            (
                item
                for item in pool
                if (item.match_id, item.frame_id) not in selected_ids
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add((candidate.match_id, candidate.frame_id))
            added_rejected += 1
        else:
            reason_order.remove(reason)
            reason_cursor -= 1
        reason_cursor += 1
    return selected


def _pitch_xy(x: float, y: float) -> tuple[float, float]:
    return x + FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0 - y


def _shirt_label(metadata: BundesligaMatchMeta, player_id: str) -> str:
    player = metadata.players.get(player_id)
    if player is None:
        return ""
    return player.shirt_number or ""


def _scene_svg(
    candidate: SceneCandidate,
    frames: Mapping[int, BundesligaFrame],
    history_ids: Sequence[int],
    metadata: BundesligaMatchMeta,
) -> str:
    frame = frames.get(candidate.frame_id)
    if frame is None:
        return '<div class="missing">Frame unavailable</div>'

    svg: list[str] = [
        '<svg class="pitch" viewBox="0 0 105 68" role="img" '
        f'aria-label="Scene {candidate.frame_id}">',
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

    history_frames = [frames[frame_id] for frame_id in history_ids if frame_id in frames]
    for player_id in frame.players:
        trail = []
        for history in history_frames:
            state = history.players.get(player_id)
            if state is not None:
                trail.append(_pitch_xy(state.x, state.y))
        if len(trail) >= 2:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in trail)
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="#f4f4f4" '
                'stroke-width="0.25" opacity="0.35"/>'
            )

    eligible = set(candidate.eligible_attacker_ids)
    offside = set(candidate.offside_attacker_ids)
    for player_id, state in frame.players.items():
        x, y = _pitch_xy(state.x, state.y)
        attacking = state.team_id == candidate.possession_team_id
        fill = "#ff8c42" if attacking else "#3f8efc"
        stroke = "#ffe15d" if player_id == candidate.ball_carrier_id else "#ffffff"
        stroke_width = 0.75 if player_id == candidate.ball_carrier_id else 0.25
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.25" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        if player_id in eligible:
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.85" fill="none" '
                'stroke="#64ff8f" stroke-width="0.35"/>'
            )
        if player_id in offside:
            svg.append(
                f'<path d="M {x-1.2:.2f},{y-1.2:.2f} L {x+1.2:.2f},{y+1.2:.2f} '
                f'M {x+1.2:.2f},{y-1.2:.2f} L {x-1.2:.2f},{y+1.2:.2f}" '
                'stroke="#ff4d67" stroke-width="0.4"/>'
            )
        shirt = escape(_shirt_label(metadata, player_id))
        if shirt:
            svg.append(
                f'<text x="{x:.2f}" y="{y+0.45:.2f}" text-anchor="middle" '
                f'font-size="1.15" fill="#111">{shirt}</text>'
            )

    if frame.ball is not None:
        ball_x, ball_y = _pitch_xy(frame.ball.x, frame.ball.y)
        ball_trail = [
            _pitch_xy(history.ball.x, history.ball.y)
            for history in history_frames
            if history.ball is not None
        ]
        if len(ball_trail) >= 2:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in ball_trail)
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="#111" '
                'stroke-width="0.4" opacity="0.7"/>'
            )
        svg.append(
            f'<circle cx="{ball_x:.2f}" cy="{ball_y:.2f}" r="0.65" '
            'fill="#111" stroke="white" stroke-width="0.2"/>'
        )
    svg.append("</svg>")
    return "".join(svg)


def _reason_counts(candidates: Sequence[SceneCandidate]) -> Counter[str]:
    return Counter(
        reason
        for candidate in candidates
        for reason in candidate.rejection_reasons
    )


def render_scene_audit_html(
    result: SceneExtractionResult,
    metadata: BundesligaMatchMeta,
    config: SceneExtractorConfig,
    selected: Sequence[SceneCandidate],
) -> str:
    accepted_count = sum(candidate.accepted for candidate in result.candidates)
    acceptance_rate = accepted_count / max(1, len(result.candidates))
    reasons = _reason_counts(result.candidates)
    reason_rows = "".join(
        f"<tr><td>{escape(reason)}</td><td>{count}</td></tr>"
        for reason, count in reasons.most_common()
    )
    cards = []
    for candidate in selected:
        status_class = "accepted" if candidate.accepted else "rejected"
        status_text = "ACCEPTED" if candidate.accepted else "REJECTED"
        reason_text = ", ".join(candidate.rejection_reasons) or "none"
        carrier_distance = (
            f"{candidate.ball_carrier_distance_m:.2f} m"
            if candidate.ball_carrier_distance_m is not None
            else "missing"
        )
        svg = _scene_svg(
            candidate,
            result.frames,
            result.history_frame_ids.get(candidate.frame_id, (candidate.frame_id,)),
            metadata,
        )
        cards.append(
            f"""
            <article class="scene-card">
              <div class="card-head">
                <span class="badge {status_class}">{status_text}</span>
                <strong>{escape(candidate.match_id)} · frame {candidate.frame_id}</strong>
              </div>
              {svg}
              <dl>
                <dt>Carrier</dt><dd>{escape(candidate.ball_carrier_id or 'missing')}</dd>
                <dt>Ball distance</dt><dd>{carrier_distance}</dd>
                <dt>Stable control</dt><dd>{candidate.stable_control_fraction:.1%}</dd>
                <dt>Players</dt><dd>{candidate.player_count}</dd>
                <dt>Eligible attackers</dt><dd>{len(candidate.eligible_attacker_ids)}</dd>
                <dt>Nearest event</dt><dd>{escape(candidate.nearest_event_type or 'none')} ({candidate.nearest_event_frame_distance} frames)</dd>
                <dt>Reasons</dt><dd>{escape(reason_text)}</dd>
              </dl>
              <label>Human review
                <select><option>unreviewed</option><option>valid</option><option>invalid</option><option>ambiguous</option></select>
              </label>
              <label>Review note <input type="text" placeholder="e.g. pass already released"></label>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Controlled-possession scene audit</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f3f5f7; color: #18212b; }}
header, main {{ max-width: 1500px; margin: auto; padding: 24px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.metric, .scene-card {{ background: white; border: 1px solid #dfe4e8; border-radius: 10px; padding: 14px; }}
.metric strong {{ display: block; font-size: 1.6rem; }}
.audit-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); gap: 16px; }}
.pitch {{ width: 100%; margin: 12px 0; background: #18743a; border-radius: 4px; }}
.card-head {{ display: flex; gap: 10px; align-items: center; }}
.badge {{ padding: 4px 8px; border-radius: 999px; font-size: .72rem; font-weight: 700; }}
.badge.accepted {{ background: #d9f8e1; color: #176529; }}
.badge.rejected {{ background: #ffe0e4; color: #9b1c31; }}
dl {{ display: grid; grid-template-columns: 145px 1fr; margin: 0; font-size: .88rem; }}
dt {{ color: #5a6773; }} dd {{ margin: 0 0 5px; overflow-wrap: anywhere; }}
label {{ display: block; margin-top: 8px; font-size: .82rem; color: #4b5965; }}
input, select {{ width: 100%; box-sizing: border-box; margin-top: 3px; padding: 7px; }}
table {{ border-collapse: collapse; background: white; }} td, th {{ border: 1px solid #dfe4e8; padding: 6px 10px; }}
.note {{ background: #fff6d8; border-left: 4px solid #e5b100; padding: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Controlled-possession scene audit</h1>
  <p class="note">Selection uses only the current and previous 0.5 seconds. The observed future is not used to accept a scene. Form inputs in this static report are for visual review only and are not saved automatically.</p>
  <div class="summary">
    <div class="metric"><strong>{len(result.candidates)}</strong>Total candidates</div>
    <div class="metric"><strong>{accepted_count}</strong>Accepted</div>
    <div class="metric"><strong>{acceptance_rate:.1%}</strong>Acceptance rate</div>
    <div class="metric"><strong>{len(result.candidates)-accepted_count}</strong>Rejected</div>
    <div class="metric"><strong>{len(selected)}</strong>Audit cards</div>
  </div>
  <h2>v0.1 thresholds</h2>
  <p>Control distance ≤ {config.control_distance_m:.2f} m · prior window {config.stable_control_seconds:.2f} s · stable fraction ≥ {config.minimum_stable_fraction:.0%} · cadence {config.sample_interval_seconds:.1f} s</p>
  <h2>Rejection reasons</h2>
  <table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{reason_rows}</tbody></table>
  <p>Orange: possession team · blue: defense · yellow ring: ball carrier · green ring: eligible attacker · red X: offside attacker.</p>
</header>
<main><div class="audit-grid">{''.join(cards)}</div></main>
</body>
</html>"""


def write_scene_audit_html(
    result: SceneExtractionResult,
    metadata: BundesligaMatchMeta,
    config: SceneExtractorConfig,
    output_path: str | Path,
    accepted_count: int = 20,
    borderline_count: int = 20,
    rejected_count: int = 20,
    seed: int = 0,
) -> Path:
    selected = select_audit_candidates(
        result.candidates,
        config,
        accepted_count=accepted_count,
        borderline_count=borderline_count,
        rejected_count=rejected_count,
        seed=seed,
    )
    html = render_scene_audit_html(result, metadata, config, selected)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_scene_review_template(
    selected: Sequence[SceneCandidate],
    output_path: str | Path,
) -> Path:
    """Write the persistent companion to the read-only HTML form controls."""

    rows = []
    for candidate in selected:
        rows.append(
            {
                "match_id": candidate.match_id,
                "frame_id": candidate.frame_id,
                "extractor_decision": candidate.status,
                "extractor_reasons": "|".join(candidate.rejection_reasons),
                "human_review": "unreviewed",
                "review_note": "",
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def candidate_summary(result: SceneExtractionResult) -> pd.DataFrame:
    """Return one compact table for command-line and notebook inspection."""

    records = []
    reason_counts = _reason_counts(result.candidates)
    accepted = sum(candidate.accepted for candidate in result.candidates)
    records.append({"metric": "total_candidates", "value": len(result.candidates)})
    records.append({"metric": "accepted", "value": accepted})
    records.append(
        {"metric": "acceptance_rate", "value": accepted / max(1, len(result.candidates))}
    )
    records.extend(
        {"metric": f"rejected:{reason}", "value": count}
        for reason, count in reason_counts.most_common()
    )
    return pd.DataFrame(records)
