"""Build the blind derived-beneficiary labelling screen from audit payloads.

Usage:
    python scripts/render_blind_derived_review.py \
        --audit-json data/processed/candidate_scenes_geometric_g/local_game_payoff_audits.json \
        --out data/processed/blind_derived_review/index.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.blind_derived_review import (
    FORBIDDEN_TOKENS,
    render_blind_derived_review,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    games = json.loads(args.audit_json.read_text(encoding="utf-8"))
    html = render_blind_derived_review(games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    rows = sum(len(game["candidate_defenders"]) for game in games)
    print(f"wrote {args.out} · {len(games)} scene(s) · {rows} label row(s)")
    # Belt and braces: the module asserts internally, re-check the file.
    text = args.out.read_text(encoding="utf-8")
    leaked = [token for token in FORBIDDEN_TOKENS if token in text]
    print(f"leak check: {'FAILED ' + str(leaked) if leaked else 'clean'}")
    if leaked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
