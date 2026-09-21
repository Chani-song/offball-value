#!/usr/bin/env python3
"""Build and open the interactive viewer for one scene.

    python -m demo_viz.view_scene --scene J03WOH:shot_006_P1_1054
    python -m demo_viz.view_scene --scene strong:2 --no-open --offline

``--offline`` inlines plotly.js so the page works without a network connection
(about 3 MB larger).  ``--serve`` starts a local HTTP server instead of opening
the file directly, which some browsers prefer.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from .config import demo_paths
from .loader import load_scene
from .story import beat_table, build_storyboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="strong:0")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=3,
                        help="source frames per animation frame (higher = smaller file)")
    parser.add_argument("--offline", action="store_true", help="inline plotly.js")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--serve", type=int, nargs="?", const=8765, default=None,
                        metavar="PORT", help="serve the exports directory over HTTP")
    parser.add_argument("--grid", type=float, default=1.5)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args(argv)

    from .interactive import export_html

    scene = load_scene(args.scene, surfaces=True, every=args.every, grid_resolution_m=args.grid)
    storyboard = build_storyboard(scene, wake_is_illustrative=scene.surfaces is None)
    print(scene.describe())
    print(beat_table(storyboard, scene))

    out = args.out or (demo_paths().export_dir /
                       f"{_slug(scene.scene_id)}.html")
    export_html(scene, storyboard, out, stride=args.stride,
                include_plotlyjs=("inline" if args.offline else "cdn"))
    print(f"wrote {out}")

    if args.serve is not None:
        _serve(out.parent, args.serve, out.name, open_browser=not args.no_open)
    elif not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _serve(directory: Path, port: int, filename: str, open_browser: bool) -> None:
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{filename}"
        print(f"serving {directory} at {url}  (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


if __name__ == "__main__":
    raise SystemExit(main())
