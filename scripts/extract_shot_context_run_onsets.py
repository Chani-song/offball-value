#!/usr/bin/env python3
"""Mine stable off-ball run onsets inside shot-anchored attacking phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offball_value.bundesliga import (
    FPS,
    find_bundesliga_files,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.run_onset import RunOnsetConfig, detect_run_onsets
from offball_value.shot_context_onset import (
    SettledPossessionConfig,
    assess_settled_shot_context_onset,
)
from offball_value.shot_context_onset_audit import render_shot_context_onset_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument(
        "--phase-csv",
        type=Path,
        default=Path("data/processed/shot_context_v0_1/attacking_phases.csv"),
    )
    parser.add_argument("--phase-review-csv", type=Path)
    parser.add_argument("--match-id", default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/shot_context_run_onset_v0_1"),
    )
    parser.add_argument("--audit-per-match", type=int, default=4)
    parser.add_argument("--animation-pre-seconds", type=float, default=2.0)
    parser.add_argument("--animation-post-seconds", type=float, default=3.0)
    parser.add_argument("--animation-fps", type=float, default=12.5)
    return parser.parse_args()


def _reviewed_phases(phases: pd.DataFrame, review_path: Path | None) -> pd.DataFrame:
    result = phases.copy()
    result["phase_human_review"] = "not_audited"
    result["phase_best_start"] = "unreviewed"
    if review_path is None:
        return result
    reviews = pd.read_csv(review_path, dtype={"match_id": str, "event_id": str})
    reviews = reviews[
        ["match_id", "shot_frame_id", "event_id", "phase_review", "best_start"]
    ].drop_duplicates(["match_id", "shot_frame_id", "event_id"])
    result = result.merge(
        reviews,
        on=["match_id", "shot_frame_id", "event_id"],
        how="left",
    )
    result["phase_human_review"] = result["phase_review"].fillna("not_audited")
    result["phase_best_start"] = result["best_start"].fillna("unreviewed")
    return result.drop(columns=["phase_review", "best_start"])


def _search_start(row: pd.Series) -> tuple[int, str]:
    mapping = {
        "attacking_half": "attacking_half_entry_frame_id",
        "possession": "possession_start_frame_id",
        "window_10": "window_10s_start_frame_id",
        "window_15": "window_15s_start_frame_id",
        "window_30": "window_30s_start_frame_id",
        "window_60": "window_60s_start_frame_id",
    }
    choice = str(row.get("phase_best_start", "unreviewed"))
    column = mapping.get(choice, "possession_start_frame_id")
    return int(row[column]), choice if choice in mapping else "possession_fallback"


def _prepare_phase_ranges(phases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in phases.iterrows():
        if str(row["phase_human_review"]) == "not_attack_context":
            continue
        start, source = _search_start(row)
        end = int(row["shot_frame_id"] - round(2.0 * FPS))
        if end < start:
            continue
        record = row.to_dict()
        record["search_start_frame_id"] = start
        record["search_end_frame_id"] = end
        record["search_start_source"] = source
        rows.append(record)
    return pd.DataFrame(rows)


def _target_and_search_ids(phases: pd.DataFrame) -> tuple[set[int], set[int]]:
    target: set[int] = set()
    search: set[int] = set()
    margin = int(round(1.6 * FPS))
    for _, row in phases.iterrows():
        start = int(row["search_start_frame_id"])
        end = int(row["search_end_frame_id"])
        shot = int(row["shot_frame_id"])
        search.update(range(start, end + 1))
        target.update(range(max(0, start - margin), shot + 1))
    return target, search


def _nearest_phase(candidate, phases: pd.DataFrame) -> pd.Series | None:
    eligible = phases[
        (phases["team_id"].astype(str) == candidate.team_id)
        & (phases["search_start_frame_id"].astype(int) <= candidate.frame_id)
        & (phases["search_end_frame_id"].astype(int) >= candidate.frame_id)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(["shot_frame_id", "event_id"]).iloc[0]


def _audit_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0)
    pool = frame.sort_values(
        ["seconds_before_shot", "confidence_score"], ascending=[True, False]
    )
    distinct_shots = pool.drop_duplicates("shot_frame_id", keep="first")
    source = distinct_shots if len(distinct_shots) >= count else pool
    if len(source) <= count:
        return source
    indices = np.linspace(0, len(source) - 1, count).round().astype(int)
    return source.iloc[np.unique(indices)]


def _payload(row: pd.Series, frames, metadata, args: argparse.Namespace) -> dict[str, object]:
    onset = int(row["frame_id"])
    pre = int(round(args.animation_pre_seconds * FPS))
    post = int(round(args.animation_post_seconds * FPS))
    step = max(1, int(round(FPS / args.animation_fps)))
    frame_ids = list(range(onset - pre, onset + post + 1, step))
    if onset not in frame_ids:
        frame_ids.append(onset)
        frame_ids.sort()
    samples = []
    for frame_id in frame_ids:
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
                "relative_time_s": round((frame_id - onset) / FPS, 3),
                "players": players,
                "ball": (
                    [round(float(frame.ball.x), 3), round(float(frame.ball.y), 3)]
                    if frame.ball is not None
                    else None
                ),
            }
        )
    runner_meta = metadata.players.get(str(row["player_id"]))
    carrier_meta = metadata.players.get(str(row["ball_carrier_id"]))
    return {
        "match_id": str(row["match_id"]),
        "match_label": f"{metadata.home_team_name} vs {metadata.away_team_name}",
        "onset_frame_id": onset,
        "runner_id": str(row["player_id"]),
        "runner_name": runner_meta.short_name if runner_meta else str(row["player_id"]),
        "carrier_id": str(row["ball_carrier_id"]),
        "carrier_name": carrier_meta.short_name if carrier_meta else str(row["ball_carrier_id"]),
        "team_id": str(row["team_id"]),
        "attacking_direction": int(row["attacking_direction"]),
        "onset_labels": str(row["onset_labels"]).split("|"),
        "onset_speed_mps": float(row["onset_speed_mps"]),
        "pre_speed_mps": float(row["pre_mean_speed_mps"]),
        "post_speed_mps": float(row["post_mean_speed_mps"]),
        "settled_team_control_fraction": float(row["settled_team_control_fraction"]),
        "settled_same_team_known_fraction": float(row["settled_same_team_known_fraction"]),
        "ball_progress_m": float(row["ball_progress_m"]),
        "seconds_before_shot": float(row["seconds_before_shot"]),
        "shot_frame_id": int(row["shot_frame_id"]),
        "phase_human_review": str(row["phase_human_review"]),
        "frames": samples,
    }


def main() -> None:
    args = parse_args()
    phases = pd.read_csv(
        args.phase_csv,
        dtype={"match_id": str, "event_id": str, "team_id": str, "player_id": str},
    )
    phases = _reviewed_phases(phases, args.phase_review_csv)
    if str(args.match_id).lower() != "all":
        match_id = normalize_bundesliga_match_id(str(args.match_id))
        phases = phases[phases["match_id"] == match_id]
    phase_ranges = _prepare_phase_ranges(phases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    onset_config = RunOnsetConfig()
    settled_config = SettledPossessionConfig()
    all_rows: list[dict[str, object]] = []
    audit_payloads: list[dict[str, object]] = []
    audit_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for match_id, match_phases in phase_ranges.groupby("match_id", sort=True):
        files = find_bundesliga_files(args.data_dir, str(match_id))
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        events = load_bundesliga_events(files["events"], clock)
        target_ids, search_ids = _target_and_search_ids(match_phases)
        frames = load_bundesliga_frames(files["positions"], target_ids)
        detected = detect_run_onsets(
            frames,
            metadata,
            search_frame_ids=search_ids,
            config=onset_config,
        )
        match_records = []
        for candidate in detected:
            phase = _nearest_phase(candidate, match_phases)
            if phase is None:
                continue
            assessment = assess_settled_shot_context_onset(
                candidate,
                frames=frames,
                events=events,
                phase_start_frame_id=int(phase["search_start_frame_id"]),
                shot_frame_id=int(phase["shot_frame_id"]),
                attacking_direction=int(phase["attacking_direction"]),
                config=settled_config,
            )
            phase_review = str(phase["phase_human_review"])
            primary = assessment.accepted and phase_review != "ambiguous"
            record = candidate.as_record()
            record.update(
                {
                    "shot_frame_id": int(phase["shot_frame_id"]),
                    "shot_event_id": str(phase["event_id"]),
                    "shot_player_id": str(phase["player_id"]),
                    "shot_xg": float(phase["xg"]) if pd.notna(phase["xg"]) else None,
                    "attacking_direction": int(phase["attacking_direction"]),
                    "phase_start_frame_id": int(phase["search_start_frame_id"]),
                    "phase_start_source": str(phase["search_start_source"]),
                    "phase_human_review": phase_review,
                    "primary_candidate": primary,
                }
            )
            record.update(assessment.as_record())
            match_records.append(record)
            all_rows.append(record)

        match_frame = pd.DataFrame(match_records)
        primary_frame = (
            match_frame[match_frame["primary_candidate"] == True].copy()  # noqa: E712
            if not match_frame.empty
            else match_frame
        )
        selected = _audit_rows(primary_frame, args.audit_per_match)
        if not selected.empty:
            audit_rows.append(selected)
            audit_payloads.extend(
                _payload(row, frames, metadata, args)
                for _, row in selected.iterrows()
            )
        summaries.append(
            {
                "match_id": match_id,
                "shot_phases": len(match_phases),
                "contextual_run_onsets": len(match_frame),
                "settled_onsets": int(match_frame["settled_possession"].sum()) if not match_frame.empty else 0,
                "primary_onsets": len(primary_frame),
                "audit_scenes": len(selected),
            }
        )
        print(
            f"{match_id}: {len(match_phases)} phases → {len(match_frame)} contextual onsets "
            f"→ {len(primary_frame)} primary settled",
            flush=True,
        )

    candidates = pd.DataFrame(all_rows)
    selection = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    candidates_path = args.output_dir / "shot_context_run_onsets.csv"
    selection_path = args.output_dir / "audit_selection.csv"
    summary_path = args.output_dir / "match_summary.csv"
    audit_json_path = args.output_dir / "shot_context_onset_audit.json"
    audit_html_path = args.output_dir / "shot_context_onset_audit.html"
    candidates.to_csv(candidates_path, index=False)
    selection.to_csv(selection_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit_json_path.write_text(json.dumps(audit_payloads, ensure_ascii=False), encoding="utf-8")
    audit_html_path.write_text(render_shot_context_onset_audit(audit_payloads), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "shot_context_run_onset_v0_1",
                "phase_csv": str(args.phase_csv),
                "phase_review_csv": str(args.phase_review_csv) if args.phase_review_csv else None,
                "run_onset_config": onset_config.__dict__,
                "settled_possession_config": settled_config.__dict__,
                "notes": [
                    "Shot contexts restrict the search universe but do not label causal contribution.",
                    "Primary candidates pass both the original run-onset context gate and the additional settled-possession gate.",
                    "Human-rejected shot phases are excluded; ambiguous reviewed phases remain non-primary.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not candidates.empty:
        rejected = candidates[~candidates["settled_possession"]]
        reason_counts = (
            rejected["settled_rejection_reasons"]
            .str.split("|")
            .explode()
            .value_counts()
        )
        print("settled rejection reasons:")
        print(reason_counts.to_string())
    print(f"candidates: {candidates_path}")
    print(f"audit:      {audit_html_path}")


if __name__ == "__main__":
    main()
