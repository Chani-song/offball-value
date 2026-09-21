#!/usr/bin/env python3
"""Render one off-ball scene to a still, an MP4 and/or a GIF.

Examples
--------
    python -m demo_viz.render_scene --scene J03WOH:shot_006_P1_1054 --mp4 --gif --png
    python -m demo_viz.render_scene --scene strong:2 --png
    python -m demo_viz.render_scene --scene synthetic --gif --half
    python -m demo_viz.render_scene --list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

if __package__ in (None, ""):                          # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from .animate import (ExportConfig, gif_from_movie, has_ffmpeg, render_movie,
                      render_still, storyboard_contact_sheet)
from .config import demo_paths
from .loader import figure_config_for, list_scenes, load_scene
from .story import beat_table, build_storyboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="strong:0",
                        help="clip id, 'strong[:n]', 'medium[:n]', 'synthetic', or a pipeline .json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--name", default=None, help="basename for the exported files")
    parser.add_argument("--png", action="store_true", help="write a single preview frame")
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--beats", action="store_true", help="write one still per story beat")
    parser.add_argument("--html", action="store_true", help="write the interactive viewer")
    parser.add_argument("--at", type=float, default=None,
                        help="scene time in seconds for --png (default: the payoff beat)")
    parser.add_argument("--wake", choices=("residual", "delta", "geometric", "off"), default=None)
    parser.add_argument("--no-minimap", action="store_true")
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--no-camera", action="store_true", help="render the full pitch instead")
    parser.add_argument("--no-surfaces", action="store_true",
                        help="skip the space computation (much faster, geometric wake only)")
    parser.add_argument("--half", action="store_true", help="960x540 output, faster")
    parser.add_argument("--stride", type=int, default=1, help="source frames per rendered frame")
    parser.add_argument("--grid", type=float, default=1.5, help="space-grid resolution in metres")
    parser.add_argument("--every", type=int, default=5, help="frames between space samples")
    parser.add_argument("--freeze", choices=("onset", "reaction", "start"), default="onset",
                        help="when the counterfactual defender stops responding")
    parser.add_argument("--baseline", choices=("hold", "drift"), default="hold",
                        help="no-response baseline: keep pre-run velocity, or freeze in place")
    parser.add_argument("--list", action="store_true", help="list annotated scenes and exit")
    parser.add_argument("--paths", action="store_true", help="show resolved data paths and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.paths:
        print(demo_paths().describe())
        return 0
    if args.list:
        for effect in ("strong", "medium"):
            print(f"--- {effect} ---")
            for scene_id in list_scenes(effect):
                print(" ", scene_id)
        return 0

    if not any((args.png, args.mp4, args.gif, args.beats, args.html)):
        args.png = True

    started = time.time()
    print(f"loading scene {args.scene!r} ...", flush=True)
    scene = load_scene(
        args.scene,
        stride=args.stride,
        surfaces=not args.no_surfaces,
        every=args.every,
        grid_resolution_m=args.grid,
        freeze_at=args.freeze,
        baseline_mode=args.baseline,
    )
    print(scene.describe())
    storyboard = build_storyboard(
        scene, wake_is_illustrative=(args.wake == "geometric" or scene.surfaces is None)
    )
    print(beat_table(storyboard, scene))

    config = figure_config_for(
        scene,
        wake_mode=args.wake,
        show_minimap=False if args.no_minimap else None,
        show_chart=False if args.no_chart else None,
        use_camera=False if args.no_camera else None,
    )
    if args.half:
        # keep the same figure size in inches so the layout is identical;
        # only the pixel density changes
        config.width_px, config.height_px, config.dpi = 960, 540, 60

    out_dir = args.out_dir or demo_paths().export_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or _slug(scene.scene_id)

    if args.png:
        path = render_still(scene, storyboard, out_dir / f"{name}_frame.png", at_time=args.at,
                            figure_config=config)
        print(f"wrote {path}")
    if args.beats:
        for path in storyboard_contact_sheet(scene, storyboard, out_dir / f"{name}_beat.png",
                                             figure_config=config):
            print(f"wrote {path}")
    if args.mp4:
        if not has_ffmpeg():
            print("!! ffmpeg not found; skipping MP4", file=sys.stderr)
        else:
            path = render_movie(scene, storyboard, out_dir / f"{name}.mp4",
                                ExportConfig(stride=args.stride), figure_config=config)
            print(f"wrote {path}")
    if args.gif:
        gif = out_dir / f"{name}.gif"
        mp4 = out_dir / f"{name}.mp4"
        if has_ffmpeg() and mp4.exists():
            path = gif_from_movie(mp4, gif)
        else:
            path = render_movie(scene, storyboard, gif,
                                ExportConfig(stride=max(args.stride, 2)),
                                figure_config=config)
        print(f"wrote {path}")
    if args.html:
        from .interactive import export_html

        path = export_html(scene, storyboard, out_dir / f"{name}.html")
        print(f"wrote {path}")

    print(f"done in {time.time() - started:.1f}s")
    return 0


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


if __name__ == "__main__":
    raise SystemExit(main())
