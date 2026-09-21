#!/usr/bin/env python3
"""Side-by-side: the existing tracking plot next to the demo_viz render.

The annotation app's clips in ``ALL_SHOT_CLIPS/`` are plain 5 Hz tracking plots
of exactly the same window this renderer uses, which makes them a fair "before"
panel: same data, same scene, same 11 seconds. This script renders the demo
without the opening/closing holds so the two panels stay in sync, then stacks
them with ffmpeg.

One thing to say out loud when showing it: the demo mirrors the pitch so the
attacking team always plays left-to-right, while the source clip keeps the raw
orientation. The two panels can therefore face opposite goals even though they
show the same seconds of the same match.

    python -m demo_viz.compare_clip --scene J03WOH:shot_006_P1_1054
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from matplotlib.patches import FancyBboxPatch

from . import palette
from .animate import ExportConfig, render_movie
from .config import demo_paths
from .loader import figure_config_for, load_scene
from .render.pitch import tracked
from .story import build_storyboard

def _label_png(text: str, path: Path, width: int = 520, height: int = 62) -> Path:
    """Render a caption strip with the demo's own typography.

    ffmpeg's ``drawtext`` filter is not present in every build (it needs
    libfreetype), so the captions are drawn with matplotlib and composited as
    images instead. That also keeps the two panels in the same visual language.
    """

    from matplotlib.figure import Figure

    dpi = 100
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.patch.set_alpha(0.0)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.16), 0.976, 0.68,
            boxstyle="round,pad=0,rounding_size=0.14",
            facecolor=palette.INK, edgecolor=palette.GRID, lw=1.2, alpha=0.92,
        )
    )
    ax.text(0.045, 0.50, tracked(text), fontsize=11, weight="bold",
            color=palette.TEXT_PRIMARY, va="center", ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, transparent=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="J03WOH:shot_006_P1_1054")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--grid", type=float, default=1.0)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required for the side-by-side export", file=sys.stderr)
        return 1

    scene = load_scene(args.scene, surfaces=True, every=args.every, grid_resolution_m=args.grid)
    if scene.video_path is None or not scene.video_path.exists():
        print(f"no source clip found for {scene.scene_id}; "
              f"set OFFBALL_CLIP_DIR to the folder holding the shot mp4s", file=sys.stderr)
        return 1

    out_dir = demo_paths().export_dir
    storyboard = build_storyboard(scene)
    config = figure_config_for(scene)
    # match the source clip's framing: same window, no holds, 16:9 at the
    # requested height
    config.width_px = int(args.height * 16 / 9)
    config.height_px = args.height
    config.dpi = int(round(config.width_px / 16))

    rendered = out_dir / f"{_slug(scene.scene_id)}_nohold.mp4"
    render_movie(scene, storyboard, rendered,
                 ExportConfig(hold_start_s=0.0, hold_end_s=0.0), figure_config=config)

    out = args.out or (out_dir / f"{_slug(scene.scene_id)}_before_after.mp4")
    label_dir = demo_paths().cache_dir / "labels"
    left_label = _label_png("BEFORE  \u00b7  EXISTING TRACKING PLOT",
                            label_dir / "before.png")
    right_label = _label_png("AFTER  \u00b7  DEMO_VIZ", label_dir / "after.png")

    # A title band above the stack keeps the captions clear of both panels'
    # own chrome, which they would otherwise sit on top of.
    band = 62
    filters = (
        f"[0:v]scale=-2:{args.height},fps=25,setsar=1[left];"
        f"[1:v]setsar=1[right];"
        f"[left][right]hstack=inputs=2[stacked];"
        f"[stacked]pad=iw:ih+{band}:0:{band}:color={palette.INK.replace('#', '0x')}[padded];"
        f"[padded][2:v]overlay=18:6[one];"
        f"[one][3:v]overlay=W/2+18:6[v]"
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(scene.video_path),
        "-i", str(rendered),
        "-i", str(left_label),
        "-i", str(right_label),
        "-filter_complex", filters,
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "20", str(out),
    ]
    subprocess.run(command, check=True)
    print(f"wrote {out}")
    print(f"  left : {scene.video_path}")
    print(f"  right: {rendered}")
    print("  note : the demo mirrors the pitch so the attacking team always plays "
          "left-to-right; the source clip does not, so the two panels can face "
          "opposite goals.")
    return 0


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


if __name__ == "__main__":
    raise SystemExit(main())
