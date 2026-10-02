#!/usr/bin/env python3
"""Interactive off-ball explorer.

    python -m demo_viz.app.interactive_app          # then open http://127.0.0.1:8060

Click an attacker to set the runner, a defender to set who reacts, and a
teammate to set the beneficiary. Everything updates as you click.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.app"

from dash import Dash

from .callbacks import register
from .components import DEFAULTS, INDEX_CSS, layout


def create_app() -> Dash:
    app = Dash(__name__, title="Off-ball explorer", update_title=None)
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}"
        "{%css%}" + INDEX_CSS + "</head><body>{%app_entry%}<footer>{%config%}"
        "{%scripts%}{%renderer%}</footer></body></html>"
    )
    app.layout = layout
    register(app)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scene", default=None, help="scene to open on start")
    parser.add_argument("--mode", default="annotation",
                        choices=("manual", "annotation", "pipeline"))
    args = parser.parse_args(argv)
    DEFAULTS["scene"] = args.scene
    DEFAULTS["mode"] = args.mode
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
