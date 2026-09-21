#!/usr/bin/env python3
"""Batch-export the demo artifacts for one or more scenes.

Writes into ``demo_viz/exports/``:

    preview.mp4            the hero scene, full story
    preview.gif            the same, halved frame rate and resolution
    preview_frame.png      a single still at the payoff beat
    preview.html           the interactive viewer
    preview_beat_*.png     one still per story beat
    <scene>.mp4 / .png     the same set for every other requested scene
    manifest.json          what was written, and from which inputs

Example::

    python -m demo_viz.export_preview                     # hero scene only
    python -m demo_viz.export_preview --all-strong        # every 'strong' scene
    python -m demo_viz.export_preview --scenes J03WOY:shot_002_P1_0580 --no-gif
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from .animate import (ExportConfig, gif_from_movie, has_ffmpeg, render_movie,
                      render_still, storyboard_contact_sheet)
from .config import demo_paths
from .loader import figure_config_for, list_scenes, load_scene
from .story import beat_table, build_storyboard

HERO_SCENE = "J03WOH:shot_006_P1_1054"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="*", default=None,
                        help=f"scene ids (default: {HERO_SCENE})")
    parser.add_argument("--all-strong", action="store_true",
                        help="export every annotated scene with effect=strong")
    parser.add_argument("--hero", default=HERO_SCENE,
                        help="which scene becomes preview.mp4 / preview.gif / preview.html")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-beats", action="store_true")
    parser.add_argument("--grid", type=float, default=1.0)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--wake", choices=("residual", "delta", "geometric", "off"), default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = demo_paths()
    out_dir = args.out_dir or paths.export_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(args.scenes or [])
    if args.all_strong:
        scenes = list_scenes("strong")
    if not scenes:
        scenes = [args.hero]
    if args.hero not in scenes:
        scenes.insert(0, args.hero)

    manifest: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "idsse_dir": str(paths.idsse_dir),
            "annotations_xlsx": str(paths.annotations_xlsx),
            "clip_dir": str(paths.clip_dir),
        },
        "scenes": [],
    }

    for scene_id in scenes:
        is_hero = scene_id == args.hero
        stem = "preview" if is_hero else _slug(scene_id)
        print(f"\n=== {scene_id} -> {stem} ===", flush=True)
        try:
            scene = load_scene(scene_id, surfaces=True, every=args.every,
                               grid_resolution_m=args.grid)
        except Exception as error:                       # keep going on bad scenes
            print(f"!! {scene_id}: {error}", file=sys.stderr)
            traceback.print_exc()
            manifest["scenes"].append({"scene_id": scene_id, "error": str(error)})  # type: ignore[union-attr]
            continue

        storyboard = build_storyboard(scene, wake_is_illustrative=scene.surfaces is None)
        print(scene.describe())
        print(beat_table(storyboard, scene))
        config = figure_config_for(scene, wake_mode=args.wake)

        written: list[str] = []
        written.append(str(render_still(scene, storyboard, out_dir / f"{stem}_frame.png",
                                        figure_config=config)))
        if not args.no_beats:
            written += [str(p) for p in storyboard_contact_sheet(
                scene, storyboard, out_dir / f"{stem}_beat.png", figure_config=config)]
        movie = None
        if not args.no_mp4 and has_ffmpeg():
            movie = render_movie(scene, storyboard, out_dir / f"{stem}.mp4",
                                 ExportConfig(), figure_config=config)
            written.append(str(movie))
        elif not args.no_mp4:
            print("!! ffmpeg not found; skipping MP4", file=sys.stderr)
        if not args.no_gif:
            gif = out_dir / f"{stem}.gif"
            if movie is not None:
                # far smaller and truer to the soft space field than a
                # frame-by-frame GIF writer
                written.append(str(gif_from_movie(movie, gif)))
            else:
                gif_config = config.__class__(**{**vars(config)})
                gif_config.width_px, gif_config.height_px, gif_config.dpi = 1120, 630, 70
                written.append(str(render_movie(scene, storyboard, gif,
                                                ExportConfig(stride=3),
                                                figure_config=gif_config)))
        if not args.no_html:
            from .interactive import export_html

            written.append(str(export_html(scene, storyboard, out_dir / f"{stem}.html")))

        manifest["scenes"].append({          # type: ignore[union-attr]
            "scene_id": scene.scene_id,
            "stem": stem,
            "source": scene.source,
            "effect": scene.provenance.get("annotation_effect"),
            "title": scene.title,
            "roles": {
                "runner": [scene.players[i].label for i in scene.runner_ids],
                "defender": [scene.players[i].label for i in scene.defender_ids],
                "beneficiary": [scene.players[i].label for i in scene.beneficiary_ids],
            },
            "moments": {k: round(float(v), 3) for k, v in scene.moments.items()},
            "moment_methods": scene.provenance.get("moment_methods", {}),
            "beat_adjustments": scene.provenance.get("beat_adjustments", {}),
            "scores": {k: round(float(v), 3) for k, v in scene.scores.items()},
            "window_frames": scene.provenance.get("window_frames"),
            "video": str(scene.video_path) if scene.video_path else None,
            "files": written,
        })
        for path in written:
            print(f"  wrote {path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {manifest_path}")
    return 0


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


if __name__ == "__main__":
    raise SystemExit(main())
