#!/usr/bin/env python3
"""Extract and render shot-anchored attacking contexts from Bundesliga data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.shot_context import (
    ShotAttackingPhase,
    ShotContextConfig,
    extract_shot_anchors,
    infer_shot_attacking_phase,
)
from offball_value.shot_context_audit import render_shot_context_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/shot_context_v0_1"),
    )
    parser.add_argument("--max-lookback-seconds", type=float, default=60.0)
    parser.add_argument("--audit-per-match", type=int, default=3)
    parser.add_argument("--animation-fps", type=float, default=10.0)
    return parser.parse_args()


def _match_ids(args: argparse.Namespace) -> list[str]:
    if str(args.match_id).lower() == "all":
        return list_bundesliga_match_ids(args.data_dir)
    return [normalize_bundesliga_match_id(str(args.match_id))]


def _target_frame_ids(phases, config: ShotContextConfig) -> set[int]:
    result: set[int] = set()
    for shot in phases:
        result.update(range(shot.frame_id - config.max_lookback_frames, shot.frame_id + 1))
    return result


def _duration_stratified(
    phases: list[ShotAttackingPhase],
    count: int,
) -> list[ShotAttackingPhase]:
    if count <= 0 or not phases:
        return []
    ordered = sorted(phases, key=lambda phase: (phase.duration_seconds, phase.shot_frame_id))
    if len(ordered) <= count:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [len(ordered) // 2]
    return [ordered[index] for index in sorted(set(indices))]


def _phase_payload(
    phase: ShotAttackingPhase,
    frames,
    metadata,
    animation_fps: float,
) -> dict[str, object]:
    step = max(1, int(round(FPS / animation_fps)))
    sampled_ids = list(
        range(phase.available_start_frame_id, phase.shot_frame_id + 1, step)
    )
    if phase.shot_frame_id not in sampled_ids:
        sampled_ids.append(phase.shot_frame_id)
    samples = []
    for frame_id in sampled_ids:
        frame = frames.get(frame_id)
        if frame is None:
            continue
        players = []
        for player_id, player in frame.players.items():
            player_meta = metadata.players.get(player_id)
            players.append(
                [
                    player_id,
                    player.team_id,
                    round(float(player.x), 3),
                    round(float(player.y), 3),
                    player_meta.short_name if player_meta else player_id,
                ]
            )
        samples.append(
            {
                "frame_id": frame_id,
                "time_to_shot_s": round((frame_id - phase.shot_frame_id) / FPS, 3),
                "players": players,
                "ball": (
                    [round(float(frame.ball.x), 3), round(float(frame.ball.y), 3)]
                    if frame.ball is not None
                    else None
                ),
            }
        )
    starts = [
        {
            "key": "attacking_half",
            "label": "마지막 상대 진영 진입",
            "frame_id": phase.attacking_half_entry_frame_id,
            "seconds_before_shot": (phase.shot_frame_id - phase.attacking_half_entry_frame_id) / FPS,
        },
        {
            "key": "possession",
            "label": "추정 possession 시작",
            "frame_id": phase.possession_start_frame_id,
            "seconds_before_shot": phase.duration_seconds,
        },
    ]
    for seconds, frame_id in phase.window_start_frame_ids:
        starts.append(
            {
                "key": f"window_{int(seconds)}",
                "label": f"고정 {seconds:g}초 window",
                "frame_id": frame_id,
                "seconds_before_shot": (phase.shot_frame_id - frame_id) / FPS,
            }
        )
    shooter_meta = metadata.players.get(phase.shot.player_id or "")
    return {
        "match_id": phase.shot.match_id,
        "match_label": f"{metadata.home_team_name} vs {metadata.away_team_name}",
        "event_id": phase.shot.event_id,
        "shot_frame_id": phase.shot_frame_id,
        "shot_team_id": phase.shot.team_id,
        "shooter_id": phase.shot.player_id,
        "shooter_name": shooter_meta.short_name if shooter_meta else phase.shot.player_id or "unknown",
        "xg": phase.shot.xg,
        "build_up": phase.shot.build_up,
        "setup": phase.shot.setup,
        "attacking_direction": phase.attacking_direction,
        "boundary_reason": phase.boundary_reason,
        "available_duration_seconds": (phase.shot_frame_id - phase.available_start_frame_id) / FPS,
        "possession_duration_seconds": phase.duration_seconds,
        "attacking_half_duration_seconds": phase.attacking_half_duration_seconds,
        "attacking_half_fraction": phase.attacking_half_fraction,
        "attacking_team_control_fraction": phase.attacking_team_control_fraction,
        "known_control_fraction": phase.known_control_fraction,
        "start_options": starts,
        "frames": samples,
    }


def main() -> None:
    args = parse_args()
    if args.animation_fps <= 0.0:
        raise ValueError("animation-fps must be positive")
    config = ShotContextConfig(max_lookback_seconds=args.max_lookback_seconds)
    config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_shot_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    audit_payloads: list[dict[str, object]] = []
    match_summaries: list[dict[str, object]] = []
    for match_id in _match_ids(args):
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        events = load_bundesliga_events(files["events"], clock)
        all_shots = extract_shot_anchors(events, open_play_only=False)
        open_play_shots = extract_shot_anchors(events, open_play_only=True)
        all_shot_rows.extend(shot.as_record() for shot in all_shots)
        if not open_play_shots:
            continue

        frames = load_bundesliga_frames(
            files["positions"], _target_frame_ids(open_play_shots, config)
        )
        phases = [
            infer_shot_attacking_phase(shot, frames, metadata, config)
            for shot in open_play_shots
            if shot.frame_id in frames
        ]
        phase_rows.extend(phase.as_record() for phase in phases)
        selected = _duration_stratified(phases, args.audit_per_match)
        audit_payloads.extend(
            _phase_payload(phase, frames, metadata, args.animation_fps)
            for phase in selected
        )
        match_summaries.append(
            {
                "match_id": match_id,
                "all_shots": len(all_shots),
                "open_play_shots": len(open_play_shots),
                "phases": len(phases),
                "median_possession_seconds": (
                    float(pd.Series([phase.duration_seconds for phase in phases]).median())
                    if phases
                    else None
                ),
                "lookback_capped": sum(
                    phase.boundary_reason == "lookback_cap_or_period_start" for phase in phases
                ),
            }
        )
        print(
            f"{match_id}: {len(open_play_shots)}/{len(all_shots)} open-play shots, "
            f"{len(phases)} phases, {len(selected)} audits",
            flush=True,
        )

    shots_frame = pd.DataFrame(all_shot_rows)
    phases_frame = pd.DataFrame(phase_rows)
    summary_frame = pd.DataFrame(match_summaries)
    shots_path = args.output_dir / "shot_anchors.csv"
    phases_path = args.output_dir / "attacking_phases.csv"
    summary_path = args.output_dir / "match_summary.csv"
    audit_json_path = args.output_dir / "shot_context_audit.json"
    audit_html_path = args.output_dir / "shot_context_audit.html"
    manifest_path = args.output_dir / "manifest.json"
    shots_frame.to_csv(shots_path, index=False)
    phases_frame.to_csv(phases_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    audit_json_path.write_text(
        json.dumps(audit_payloads, ensure_ascii=False), encoding="utf-8"
    )
    audit_html_path.write_text(
        render_shot_context_audit(audit_payloads), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "version": "shot_context_v0_1",
                "matches": _match_ids(args),
                "config": {
                    "max_lookback_seconds": config.max_lookback_seconds,
                    "comparison_windows_seconds": config.comparison_windows_seconds,
                    "control_distance_m": config.control_distance_m,
                    "opponent_control_confirmation_seconds": config.opponent_control_confirmation_seconds,
                    "animation_fps": args.animation_fps,
                },
                "notes": [
                    "Shots are sampling anchors, not causal labels for earlier off-ball actions.",
                    "Only provider-classified open-play shots enter attacking-phase extraction.",
                    "The audit sample is duration-stratified within each match.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"shots:  {shots_path}")
    print(f"phases: {phases_path}")
    print(f"audit:  {audit_html_path}")


if __name__ == "__main__":
    main()
