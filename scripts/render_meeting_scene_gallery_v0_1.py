#!/usr/bin/env python3
"""Render the curated eight-scene meeting gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offball_value.meeting_scene_gallery import (
    build_meeting_gallery_payload,
    render_meeting_scene_gallery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload-json",
        type=Path,
        default=Path(
            "examples/research_audit/current_demos/meeting_scene_gallery_payload.json"
        ),
        help=(
            "Curated payload used by default. Pass both --prototype-json and "
            "--observed-json to rebuild a payload from source artifacts instead."
        ),
    )
    parser.add_argument(
        "--prototype-json",
        type=Path,
    )
    parser.add_argument(
        "--observed-json",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/meeting_scene_gallery_v0_1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild = args.prototype_json is not None or args.observed_json is not None
    if rebuild:
        if args.prototype_json is None or args.observed_json is None:
            raise SystemExit(
                "Pass --prototype-json and --observed-json together when "
                "rebuilding the gallery payload."
            )
        prototype = json.loads(args.prototype_json.read_text(encoding="utf-8"))
        observed = json.loads(args.observed_json.read_text(encoding="utf-8"))
        payload = build_meeting_gallery_payload(prototype, observed)
    else:
        payload = json.loads(args.payload_json.read_text(encoding="utf-8"))
        if int(payload.get("scene_count", -1)) != 8:
            raise ValueError("Curated meeting payload must contain eight scenes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.output_dir / "meeting_scene_gallery_payload.json"
    html_path = args.output_dir / "meeting_scene_gallery.html"
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_meeting_scene_gallery(payload), encoding="utf-8")
    print(f"payload: {payload_path.resolve()}")
    print(f"html: {html_path.resolve()}")
    print("gallery scenes: 8 unique observed; Klaus includes an alternate prototype view")


if __name__ == "__main__":
    main()
