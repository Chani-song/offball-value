#!/usr/bin/env python3
"""One experimental clip of the OBSO threat view with the candidate-pass fan.

    python -m demo_viz.export_experimental
    python -m demo_viz.export_experimental --scene J03WOY:shot_002_P1_0580

Writes ``demo_viz/exports/experimental/``:

    obso_passes.png        one frame
    obso_passes.mp4        the clip

This is a look at whether the visual language reads, not a demo video. It is
deliberately captured from the *browser* rather than the matplotlib renderer:
the point is to judge what visitors will actually see, and rendering it a
second way would only tell us whether two renderers agree.

Nothing here is part of the narrated strong-five reel, and nothing in the reel
changes.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import subprocess
import sys
import tempfile
import threading
import contextlib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from .config import demo_paths
from .core.candidates import candidate_passes
from .loader import load_scene
from .web.build import build
from .web.validate import find_chrome

#: Frames between captures. The browser timeline is 25 fps; every third frame
#: at 8 fps plays back at roughly a third speed, which is what a look at a
#: field wants.
STRIDE = 3
FPS = 8
WIDTH, HEIGHT = 1600, 1000


@contextlib.contextmanager
def _serve(directory: Path):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="J03WOH:shot_006_P1_1054")
    parser.add_argument("--frames", type=int, default=276)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    chrome = find_chrome()
    if chrome is None:
        print("!! no Chrome/Chromium found", file=sys.stderr)
        return 1
    out_dir = args.out or (demo_paths().export_dir / "experimental")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp) / "site")
        shots = Path(tmp) / "shots"
        shots.mkdir()
        with _serve(root) as port:
            base = (f"http://127.0.0.1:{port}/index.html?scene={args.scene}"
                    "&mode=obso&passes=1&fit=1")
            frames = list(range(0, args.frames, STRIDE))
            for position, frame in enumerate(frames):
                target = shots / f"{position:04d}.png"
                subprocess.run(
                    [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                     "--hide-scrollbars", f"--window-size={WIDTH},{HEIGHT}",
                     "--virtual-time-budget=12000",
                     f"--screenshot={target}", f"{base}&t={frame}"],
                    capture_output=True, timeout=120,
                )
                if position % 20 == 0:
                    print(f"    frame {position + 1}/{len(frames)}", flush=True)

        # the representative still has to show both things at once, so pick the
        # captured frame nearest the middle that actually has a carrier --
        # core.candidates answers that without a second browser pass
        scene = load_scene(args.scene, quantities=False, surfaces=False)
        middle = len(frames) // 2
        order = sorted(range(len(frames)), key=lambda i: abs(i - middle))
        chosen = next((i for i in order if candidate_passes(scene, frames[i])), middle)
        still = out_dir / "obso_passes.png"
        still.write_bytes((shots / f"{chosen:04d}.png").read_bytes())
        print(f"still from frame {frames[chosen]} "
              f"({'carrier' if chosen != middle else 'midpoint'})")

        movie = out_dir / "obso_passes.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(shots / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-preset", "slow", "-crf", "20",
             str(movie)],
            check=True,
        )
    print(f"wrote {still}")
    print(f"wrote {movie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
