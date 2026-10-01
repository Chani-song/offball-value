#!/usr/bin/env python3
"""Export the solved moments of the local bundle for the Game solution view.

    python -m demo_viz.web.export_panels

Writes ``demo_viz/web_data/solver/<scene>.json`` for every published showcase
scene the bundle covers: the equilibrium policy at each real moment
(0.0 / 0.6 / 1.2 s) with **the solver's own 0.6 s path** per command, the pass
policy, the value, and upstream's dilemma flags and static comparison.

LICENCE
-------
Nothing from ``tracking/`` is read or written. Every coordinate emitted is
solver output or a body's start position at one frame -- material the
published scene payload already contains. ``tests/test_bundle_integration.py``
walks the built site and fails on a tracking file or a raw trajectory.

Without the bundle this exits 0 and writes nothing, so a fresh checkout still
builds a site; the Game solution simply stays unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "demo_viz.web"

from ..paper_story.bundle import available, panel_payload, source  # noqa: E402
from .export_data import WEB_DATA, slug  # noqa: E402

SOLVER_DIR = WEB_DATA / "solver"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SOLVER_DIR)
    args = parser.parse_args(argv)

    if not available():
        print("no local bundle; skipping panel export", file=sys.stderr)
        print("  the Game solution stays unavailable and the site still builds",
              file=sys.stderr)
        return 0

    reader = source()
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene_id in reader.scenes():
        payload = panel_payload(reader, scene_id)
        if payload is None:
            continue
        path = args.out / f"{slug(scene_id)}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        total += path.stat().st_size
        print(f"{payload['code']:5s} {scene_id:34s} "
              f"{len(payload['panels'])} moments  {path.stat().st_size / 1024:.0f} KB")
    print(f"\n{len(reader.scenes())} scenes, {total / 1024:.0f} KB total -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
