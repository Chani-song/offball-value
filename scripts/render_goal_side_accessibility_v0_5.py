#!/usr/bin/env python3
"""Render the scene-1 goal-side accessibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_causal_attribution_v0_4 import TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "data/processed/goal_side_accessibility_v0_5/"
            "goal_side_accessibility_v0_5.json"
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path(
            "data/processed/goal_side_accessibility_v0_5/"
            "goal_side_accessibility_v0_5_audit.html"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(
        TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"audit: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
