#!/usr/bin/env python3
"""Extract and render controlled-possession audit scenes for one IDSSE match."""

from __future__ import annotations

import argparse
from pathlib import Path

from offball_value.bundesliga import (
    find_bundesliga_files,
    load_bundesliga_events,
    load_bundesliga_frame_clock,
    load_bundesliga_match_metadata,
    normalize_bundesliga_match_id,
)
from offball_value.scene_audit import (
    candidate_summary,
    render_scene_audit_html,
    select_audit_candidates,
    write_scene_candidates_csv,
    write_scene_review_template,
)
from offball_value.scene_extractor import (
    SceneExtractorConfig,
    extract_bundesliga_scenes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bundesliga-integrated"),
    )
    parser.add_argument("--match-id", default="J03WMX")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/scene_extractor_v0_1"),
    )
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Development-only limit; omit for the full match.",
    )
    parser.add_argument("--accepted-audit", type=int, default=20)
    parser.add_argument("--borderline-audit", type=int, default=20)
    parser.add_argument("--rejected-audit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SceneExtractorConfig()
    files = find_bundesliga_files(args.data_dir, args.match_id)
    clock = load_bundesliga_frame_clock(files["positions"])
    events = load_bundesliga_events(files["events"], clock)
    metadata = load_bundesliga_match_metadata(files["matchinfo"])

    result = extract_bundesliga_scenes(
        positions_xml=files["positions"],
        events=events,
        metadata=metadata,
        config=config,
        limit_candidates=args.limit_candidates,
    )

    match_id = normalize_bundesliga_match_id(args.match_id)
    output_dir = args.output_dir / match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = write_scene_candidates_csv(
        result,
        output_dir / "scene_candidates.csv",
    )
    summary = candidate_summary(result)
    summary_path = output_dir / "scene_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected = select_audit_candidates(
        result.candidates,
        config,
        accepted_count=args.accepted_audit,
        borderline_count=args.borderline_audit,
        rejected_count=args.rejected_audit,
        seed=args.seed,
    )
    audit_path = output_dir / "scene_audit.html"
    audit_path.write_text(
        render_scene_audit_html(result, metadata, config, selected),
        encoding="utf-8",
    )
    review_path = write_scene_review_template(
        selected,
        output_dir / "scene_review.csv",
    )

    print(summary.to_string(index=False))
    print(f"candidates: {candidates_path}")
    print(f"summary:    {summary_path}")
    print(f"audit:      {audit_path}")
    print(f"review:     {review_path}")


if __name__ == "__main__":
    main()

