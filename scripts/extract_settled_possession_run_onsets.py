#!/usr/bin/env python3
"""Mine stable off-ball run onsets inside ALL settled attacking possessions.

This widens the labelling pool beyond shot-ending phases: possession phases
are segmented forwards over the whole match with the same strict foot-control
ownership model that the shot-anchored pipeline uses backwards, then the SAME
onset detection + contextual + settled gates run over those phases and the
SAME output bundle (candidates CSV, audit selection, match summary, manifest,
QC HTML/JSON) is emitted.

Anchor semantics: the ``shot_frame_id`` column of every phase and candidate
holds the possession-end frame (last confirmed own-team control sample) and
``shot_xg``/``xg`` is missing.  ``seconds_before_shot`` therefore reads as
"seconds before possession end".

Exclusions enforced here:
- excluded matches (default DFL-MAT-J03WN1: red card at ~7 minutes leaves
  11v10 for ~93% of the match, so it is outside the studied population);
- onsets within a +/- dedupe window of already human-reviewed onsets are kept
  in the candidates CSV flagged ``excluded_reviewed`` but never enter
  ``audit_selection.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from offball_value.bundesliga import (
    FPS,
    expected_player_count_at,
    find_bundesliga_files,
    list_bundesliga_match_ids,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_frames,
    load_bundesliga_match_metadata,
    load_bundesliga_sendings_off,
    normalize_bundesliga_match_id,
)
from offball_value.run_onset import RunOnsetConfig, detect_run_onsets
from offball_value.settled_possession_phase import (
    SettledPossessionPhaseConfig,
    segment_settled_possession_phases,
)
from offball_value.shot_context_onset import (
    SettledPossessionConfig,
    assess_settled_shot_context_onset,
)
from offball_value.shot_context_onset_audit import render_shot_context_onset_audit


DEFAULT_REVIEW_CSVS = (
    Path(
        "examples/research_audit/human_reviews/shot_context/"
        "shot_context_onset_v0_1_reviews.csv"
    ),
    Path(
        "examples/research_audit/human_reviews/shot_context/"
        "shot_context_onset_v0_2_reviews.csv"
    ),
)
MAX_SEGMENTATION_FRAME_ID = 400_000
PRE_ANCHOR_GAP_SECONDS = 2.0
SEARCH_MARGIN_SECONDS = 1.6
TARGET_POST_SECONDS = 4.6


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
        default=Path("data/processed/settled_possession_run_onset_v0_1"),
    )
    parser.add_argument(
        "--exclude-match",
        action="append",
        default=None,
        help="Match ids excluded from the population (default: DFL-MAT-J03WN1).",
    )
    parser.add_argument(
        "--reviewed-onsets-csv",
        type=Path,
        nargs="*",
        default=list(DEFAULT_REVIEW_CSVS),
        help="Review CSVs whose (match_id, onset_frame_id) pairs are deduped.",
    )
    parser.add_argument("--dedupe-window-seconds", type=float, default=3.0)
    parser.add_argument("--sample-stride-frames", type=int, default=5)
    parser.add_argument("--min-phase-duration-seconds", type=float, default=6.0)
    parser.add_argument(
        "--require-ball-in-opponent-half",
        dest="require_ball_in_opponent_half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the ball to be in the opponent half at the onset frame "
            "(default: on, matching v0_1). Turning it OFF admits counter-attacks "
            "and deep build-up, which the population definition -- our team holds "
            "the ball and the moment is not contested -- does not exclude."
        ),
    )
    parser.add_argument(
        "--require-attacking-half-presence",
        dest="require_attacking_half_presence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the possession phase to reach the attacking half (default: on).",
    )
    parser.add_argument("--min-speed-gain-mps", type=float, default=1.50)
    parser.add_argument("--min-direction-change-degrees", type=float, default=30.0)
    parser.add_argument("--min-post-displacement-m", type=float, default=3.0)
    parser.add_argument("--onset-control-distance-m", type=float, default=1.50)
    # Phase splitting, separate from the onset gate above. 0.32 s cuts a
    # phase on a grazed touch: 343 of 720 phases resume with the same team
    # within 5 s, and 129 shots fall into the gaps that creates.
    parser.add_argument("--opponent-confirmation-seconds", type=float, default=0.32)
    parser.add_argument("--phase-control-distance-m", type=float, default=1.50)
    parser.add_argument(
        "--adjust-for-dismissals",
        dest="adjust_for_dismissals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expect 21 players after a red card instead of a constant 22 "
            "(default: on). Off reproduces the pre-fix behaviour, which "
            "discarded 92%% of DFL-MAT-J03WN1."
        ),
    )
    parser.add_argument("--audit-per-match", type=int, default=15)
    parser.add_argument("--audit-total", type=int, default=90)
    parser.add_argument("--animation-pre-seconds", type=float, default=2.0)
    parser.add_argument("--animation-post-seconds", type=float, default=3.0)
    parser.add_argument("--animation-fps", type=float, default=12.5)
    return parser.parse_args()


def _excluded_matches(args: argparse.Namespace) -> set[str]:
    raw = args.exclude_match if args.exclude_match else ["DFL-MAT-J03WN1"]
    if len(raw) == 1 and str(raw[0]).strip().lower() == "none":
        return set()
    return {normalize_bundesliga_match_id(str(match_id)) for match_id in raw}


def _match_ids(args: argparse.Namespace) -> list[str]:
    if str(args.match_id).lower() == "all":
        return list_bundesliga_match_ids(args.data_dir)
    return [normalize_bundesliga_match_id(str(args.match_id))]


def _reviewed_onset_frames(paths: list[Path]) -> dict[str, list[int]]:
    frames_by_match: dict[str, set[int]] = {}
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"reviewed-onsets csv not found: {path}")
        reviews = pd.read_csv(path, dtype={"match_id": str})
        for _, row in reviews.iterrows():
            frames_by_match.setdefault(str(row["match_id"]), set()).add(
                int(row["onset_frame_id"])
            )
    return {
        match_id: sorted(values) for match_id, values in frames_by_match.items()
    }


def _prepare_phase_ranges(phases: pd.DataFrame) -> pd.DataFrame:
    gap = int(round(PRE_ANCHOR_GAP_SECONDS * FPS))
    rows = []
    for _, row in phases.iterrows():
        start = int(row["possession_start_frame_id"])
        end = int(row["shot_frame_id"]) - gap
        if end < start:
            continue
        record = row.to_dict()
        record["search_start_frame_id"] = start
        record["search_end_frame_id"] = end
        record["search_start_source"] = "possession"
        record["phase_human_review"] = "not_audited"
        rows.append(record)
    return pd.DataFrame(rows)


def _target_and_search_ids(phases: pd.DataFrame) -> tuple[set[int], set[int]]:
    target: set[int] = set()
    search: set[int] = set()
    margin = int(round(SEARCH_MARGIN_SECONDS * FPS))
    post = int(round(TARGET_POST_SECONDS * FPS))
    for _, row in phases.iterrows():
        start = int(row["search_start_frame_id"])
        end = int(row["search_end_frame_id"])
        anchor = int(row["shot_frame_id"])
        search.update(range(start, end + 1))
        target.update(range(max(0, start - margin), anchor + post + 1))
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


def _is_reviewed_duplicate(
    frame_id: int,
    reviewed_frames: list[int],
    window_frames: int,
) -> bool:
    return any(abs(frame_id - reviewed) <= window_frames for reviewed in reviewed_frames)


def _audit_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Pick the strongest fresh primary candidates, spread over distinct phases."""

    if count <= 0 or frame.empty:
        return frame.head(0)
    pool = frame.sort_values(
        ["confidence_score", "settled_team_control_fraction", "frame_id"],
        ascending=[False, False, True],
    )
    distinct = pool.drop_duplicates("shot_frame_id", keep="first")
    if len(distinct) >= count:
        return distinct.head(count)
    rest = pool.loc[~pool.index.isin(distinct.index)]
    return pd.concat([distinct, rest.head(count - len(distinct))])


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
        "match_label": (
            f"{metadata.home_team_name} vs {metadata.away_team_name}"
            " · settled-possession pool (기준 시점 = 점유 종료, 슈팅 아님)"
        ),
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
    excluded_matches = _excluded_matches(args)
    reviewed_frames = _reviewed_onset_frames(list(args.reviewed_onsets_csv))
    dedupe_window_frames = int(round(args.dedupe_window_seconds * FPS))
    phase_config = SettledPossessionPhaseConfig(
        sample_stride_frames=args.sample_stride_frames,
        minimum_duration_seconds=args.min_phase_duration_seconds,
        require_attacking_half_presence=args.require_attacking_half_presence,
        opponent_control_confirmation_seconds=args.opponent_confirmation_seconds,
        control_distance_m=args.phase_control_distance_m,
    )
    onset_config = RunOnsetConfig(
        minimum_speed_gain_mps=args.min_speed_gain_mps,
        minimum_direction_change_degrees=args.min_direction_change_degrees,
        minimum_post_displacement_m=args.min_post_displacement_m,
        control_distance_m=args.onset_control_distance_m,
    )
    settled_config = SettledPossessionConfig(
        require_ball_in_opponent_half=args.require_ball_in_opponent_half,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    audit_payloads: list[dict[str, object]] = []
    audit_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for match_id in _match_ids(args):
        if match_id in excluded_matches:
            summaries.append(
                {
                    "match_id": match_id,
                    "excluded_match": True,
                    "phases": 0,
                    "contextual_run_onsets": 0,
                    "settled_onsets": 0,
                    "primary_onsets": 0,
                    "excluded_reviewed_primary": 0,
                    "fresh_primary_onsets": 0,
                    "audit_scenes": 0,
                }
            )
            print(f"{match_id}: excluded from population", flush=True)
            continue
        files = find_bundesliga_files(args.data_dir, match_id)
        metadata = load_bundesliga_match_metadata(files["matchinfo"])
        clock = load_bundesliga_frame_clock(files["positions"])
        events = load_bundesliga_events(files["events"], clock)

        sample_ids = range(0, MAX_SEGMENTATION_FRAME_ID, args.sample_stride_frames)
        sampled_frames = load_bundesliga_frames(files["positions"], sample_ids)
        phases = segment_settled_possession_phases(
            sampled_frames, events, metadata, phase_config
        )
        del sampled_frames
        match_phase_frame = pd.DataFrame([phase.as_record() for phase in phases])
        phase_rows.extend(phase.as_record() for phase in phases)
        if match_phase_frame.empty:
            summaries.append(
                {
                    "match_id": match_id,
                    "excluded_match": False,
                    "phases": 0,
                    "contextual_run_onsets": 0,
                    "settled_onsets": 0,
                    "primary_onsets": 0,
                    "excluded_reviewed_primary": 0,
                    "fresh_primary_onsets": 0,
                    "audit_scenes": 0,
                }
            )
            continue

        phase_ranges = _prepare_phase_ranges(match_phase_frame)
        target_ids, search_ids = _target_and_search_ids(phase_ranges)
        frames = load_bundesliga_frames(files["positions"], target_ids)
        # After a dismissal the pitch holds 21 players, and the onset gate's
        # constant 22 then rejects the rest of the match. Pass the expected
        # count so a dismissal stays in the population and only genuinely
        # missing tracking rejects a frame.
        dismissals = load_bundesliga_sendings_off(files["events"], clock)
        if dismissals:
            print(
                f"  {match_id}: 퇴장 {len(dismissals)}건 "
                f"{[frame_id for frame_id, _t, _p in dismissals]} "
                f"→ 이후 기대 인원 {22 - len(dismissals)}명",
                flush=True,
            )
        detected = detect_run_onsets(
            frames,
            metadata,
            search_frame_ids=search_ids,
            config=onset_config,
            expected_player_count=(
                expected_player_count_at(dismissals)
                if args.adjust_for_dismissals
                else None
            ),
        )
        match_reviewed = reviewed_frames.get(match_id, [])
        match_records = []
        for candidate in detected:
            phase = _nearest_phase(candidate, phase_ranges)
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
            record = candidate.as_record()
            record.update(
                {
                    "shot_frame_id": int(phase["shot_frame_id"]),
                    "shot_event_id": str(phase["event_id"]),
                    "shot_player_id": (
                        str(phase["player_id"]) if pd.notna(phase["player_id"]) else ""
                    ),
                    "shot_xg": None,
                    "attacking_direction": int(phase["attacking_direction"]),
                    "phase_start_frame_id": int(phase["search_start_frame_id"]),
                    "phase_start_source": str(phase["search_start_source"]),
                    "phase_human_review": str(phase["phase_human_review"]),
                    "possession_end_frame_id": int(phase["shot_frame_id"]),
                    "phase_boundary_reason": str(phase["boundary_reason"]),
                    "primary_candidate": assessment.accepted,
                    "excluded_reviewed": _is_reviewed_duplicate(
                        candidate.frame_id, match_reviewed, dedupe_window_frames
                    ),
                }
            )
            record.update(assessment.as_record())
            match_records.append(record)
            all_rows.append(record)

        match_frame = pd.DataFrame(match_records)
        if match_frame.empty:
            primary_frame = match_frame
            fresh_frame = match_frame
            settled_count = 0
            excluded_reviewed_primary = 0
        else:
            primary_frame = match_frame[
                match_frame["primary_candidate"] == True  # noqa: E712
            ].copy()
            fresh_frame = primary_frame[
                primary_frame["excluded_reviewed"] == False  # noqa: E712
            ].copy()
            settled_count = int(match_frame["settled_possession"].sum())
            excluded_reviewed_primary = int(primary_frame["excluded_reviewed"].sum())
        selected = _audit_rows(fresh_frame, args.audit_per_match)
        if not selected.empty:
            audit_rows.append(selected)
            audit_payloads.extend(
                _payload(row, frames, metadata, args)
                for _, row in selected.iterrows()
            )
        summaries.append(
            {
                "match_id": match_id,
                "excluded_match": False,
                "phases": len(match_phase_frame),
                "contextual_run_onsets": len(match_frame),
                "settled_onsets": settled_count,
                "primary_onsets": len(primary_frame),
                "excluded_reviewed_primary": excluded_reviewed_primary,
                "fresh_primary_onsets": len(fresh_frame),
                "audit_scenes": len(selected),
            }
        )
        print(
            f"{match_id}: {len(match_phase_frame)} phases → {len(match_frame)} contextual onsets "
            f"→ {len(primary_frame)} primary settled → {len(fresh_frame)} fresh "
            f"({excluded_reviewed_primary} near reviewed) → {len(selected)} audit",
            flush=True,
        )
        del frames

    candidates = pd.DataFrame(all_rows)
    selection = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    if not selection.empty and len(selection) > args.audit_total:
        selection = (
            selection.sort_values("confidence_score", ascending=False)
            .head(args.audit_total)
            .sort_values(["match_id", "frame_id"])
            .reset_index(drop=True)
        )
        kept = set(
            zip(selection["match_id"], selection["frame_id"], selection["player_id"])
        )
        audit_payloads = [
            payload
            for payload in audit_payloads
            if (payload["match_id"], payload["onset_frame_id"], payload["runner_id"])
            in kept
        ]
    phases_frame = pd.DataFrame(phase_rows)
    summary = pd.DataFrame(summaries)

    phases_path = args.output_dir / "attacking_phases.csv"
    candidates_path = args.output_dir / "shot_context_run_onsets.csv"
    selection_path = args.output_dir / "audit_selection.csv"
    summary_path = args.output_dir / "match_summary.csv"
    audit_json_path = args.output_dir / "shot_context_onset_audit.json"
    audit_html_path = args.output_dir / "shot_context_onset_audit.html"
    phases_frame.to_csv(phases_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    selection.to_csv(selection_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit_json_path.write_text(json.dumps(audit_payloads, ensure_ascii=False), encoding="utf-8")
    audit_html_path.write_text(render_shot_context_onset_audit(audit_payloads), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "settled_possession_run_onset_v0_1",
                "phase_source": "segment_settled_possession_phases (all settled attacking possessions)",
                "excluded_matches": sorted(excluded_matches),
                "excluded_match_reason": (
                    (
                        "DFL-MAT-J03WN1 has a ~7-minute red card, so ~93% of the match "
                        "is 11v10; man-advantage football is outside this labelling "
                        "population."
                    )
                    if excluded_matches
                    else (
                        "No match excluded. DFL-MAT-J03WN1 (11v10 for ~93% after a "
                        "~7-minute red card) is deliberately INCLUDED here: 13 of the "
                        "88 human-labelled shot annotations live in it, and dropping "
                        "them cannot be undone later. It is man-advantage football and "
                        "IS outside the 11v11 population, so any numeric comparison "
                        "must filter it out on match_id."
                    )
                ),
                "reviewed_onsets_csvs": [str(path) for path in args.reviewed_onsets_csv],
                "dedupe_window_seconds": args.dedupe_window_seconds,
                "audit_per_match": args.audit_per_match,
                "audit_total": args.audit_total,
                "adjust_for_dismissals": args.adjust_for_dismissals,
                "require_ball_in_opponent_half": args.require_ball_in_opponent_half,
                "require_attacking_half_presence": args.require_attacking_half_presence,
                "min_phase_duration_seconds": args.min_phase_duration_seconds,
                "settled_possession_phase_config": {
                    "sample_stride_frames": phase_config.sample_stride_frames,
                    "minimum_duration_seconds": phase_config.minimum_duration_seconds,
                    "require_attacking_half_presence": phase_config.require_attacking_half_presence,
                    "control_distance_m": phase_config.control_distance_m,
                    "max_control_ball_height_m": phase_config.max_control_ball_height_m,
                    "opponent_control_confirmation_seconds": phase_config.opponent_control_confirmation_seconds,
                },
                "run_onset_config": onset_config.__dict__,
                "settled_possession_config": settled_config.__dict__,
                "notes": [
                    "Phases are ALL settled attacking possessions in open play, not only shot-ending ones.",
                    "Anchor semantics: shot_frame_id / frame_id hold the possession-end frame "
                    "(last confirmed own-team control sample) and xg / shot_xg are missing; "
                    "seconds_before_shot reads as seconds before possession end.",
                    "Possession spells reuse the strict foot-control ownership model of "
                    "shot_context._possession_boundary, applied forwards: a spell starts at the first "
                    "confirmed own-team control sample, survives unknown-control samples, and ends on "
                    "confirmed opponent control, stoppage/restart events, or period end, trimmed back "
                    "to the last confirmed own-team control sample.",
                    "Candidates within the dedupe window of already-reviewed onsets are flagged "
                    "excluded_reviewed=True in the candidates CSV and never enter audit_selection.",
                    "audit_selection ranks fresh primary settled candidates by confidence, prefers "
                    "distinct possessions, and is capped per match and in total.",
                    "Output file names mirror the shot-anchored bundle so Pass-1 review tooling and "
                    "scene payload builders keep working unchanged.",
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
    print(f"phases:     {phases_path}")
    print(f"candidates: {candidates_path}")
    print(f"audit:      {audit_html_path}")


if __name__ == "__main__":
    main()
