"""Redraw an existing local-game audit's HTML without recomputing anything.

The payoff computation takes about an hour for the seven-scene cohort, so a
display-only change (a new overlay, a clearer caption) should not require a
rerun.  This reads a finished ``local_game_payoff_audits.json`` and renders
the current HTML template over it in place.

Usage:
    python scripts/rerender_local_game_audit_html.py DIR [DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offball_value.local_game_payoff_audit import render_local_game_payoff_audit


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for argument in sys.argv[1:]:
        directory = Path(argument)
        payload = directory / "local_game_payoff_audits.json"
        games = json.loads(payload.read_text(encoding="utf-8"))
        html_path = directory / "local_game_payoff_audit.html"
        html_path.write_text(
            render_local_game_payoff_audit(games), encoding="utf-8"
        )
        print(f"redrew {html_path} ({len(games)} scene(s))")


if __name__ == "__main__":
    main()
