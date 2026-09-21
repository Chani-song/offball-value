#!/usr/bin/env python3
"""Render the five human-confirmed `strong` scenes, and a montage of them.

    python -m demo_viz.export_strong                 # all five + montage
    python -m demo_viz.export_strong --no-montage
    python -m demo_viz.export_strong --only 1 2

Writes into ``demo_viz/exports/strong/``:

    strong_01.mp4 … strong_05.mp4      one scene each, 1920x1080
    strong_01.png … strong_05.png      a representative frame from each
    strong_five_montage.mp4            title card + the five clips
    manifest.json                      which scene each number is, and its roles

The roles are the human annotations from ``shot_annotations.xlsx`` — every
annotated runner, defender and beneficiary, with no heuristic substitution. The
shaded field is the beneficiary's **available space** (the factual residual),
not the difference against the held-defender baseline, and the passing lane and
suggested-player layers are off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from . import palette
from .animate import ExportConfig, has_ffmpeg, render_movie, render_still
from .annotations import select_clips
from .config import demo_paths
from .loader import figure_config_for, load_scene
from .story import build_storyboard

#: Layers the brief asks to be off in the rendered video.
DISABLED_LAYERS = ("lane",)
TITLE = "OFF-BALL SPACE"


@dataclass(frozen=True)
class StrongScene:
    position: int
    clip_id: str
    shape: str
    complexity: int

    @property
    def stem(self) -> str:
        return f"strong_{self.position:02d}"


def strong_scenes() -> list[StrongScene]:
    """The five `strong` clips, simplest interaction first."""

    clips = [c for c in select_clips("strong") if c.is_renderable]

    def complexity(clip):
        runners = len(clip.runner_shirts)
        defenders = len(clip.defender_shirts)
        beneficiaries = len(clip.beneficiary_shirts)
        # one-to-one first, then extra defenders, then extra attackers
        return (runners + defenders + beneficiaries, defenders, runners, clip.clip_id)

    clips.sort(key=complexity)
    return [
        StrongScene(position=index, clip_id=clip.clip_id, shape=clip.shape,
                    complexity=complexity(clip)[0])
        for index, clip in enumerate(clips, start=1)
    ]


def _config_for(scene):
    config = figure_config_for(scene)
    config.wake_mode = "residual" if scene.surfaces is not None else "geometric"
    config.delta_outline = False          # the field already is the available space
    config.disabled_layers = DISABLED_LAYERS
    return config


def render_one(entry: StrongScene, out_dir: Path, still_only: bool = False) -> dict:
    scene = load_scene(entry.clip_id, surfaces=True, every=5, grid_resolution_m=1.0)
    storyboard = build_storyboard(
        scene, wake_is_illustrative=scene.surfaces is None, space_view="available",
    )
    config = _config_for(scene)

    png = render_still(scene, storyboard, out_dir / f"{entry.stem}.png",
                       figure_config=config)
    files = [png]
    if not still_only:
        files.append(render_movie(scene, storyboard, out_dir / f"{entry.stem}.mp4",
                                  ExportConfig(), figure_config=config, progress=False))

    def labels(ids):
        return [scene.players[p].label for p in ids if p in scene.players]

    return {
        "position": entry.position,
        "scene_id": entry.clip_id,
        "shape": entry.shape,
        "title": scene.title,
        "roles": {
            "runner": labels(scene.runner_ids),
            "defender": labels(scene.defender_ids),
            "beneficiary": labels(scene.beneficiary_ids),
        },
        "space_view": "available",
        "layers_off": list(DISABLED_LAYERS),
        "files": [f.name for f in files],
    }


# ---------------------------------------------------------------------------
# montage
# ---------------------------------------------------------------------------
def _card(text: str, subtitle: str, path: Path, width: int = 1920, height: int = 1080):
    """A minimal card: the words, nothing else."""

    from matplotlib.figure import Figure

    dpi = 120
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=palette.INK)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_facecolor(palette.INK)
    ax.text(0.5, 0.52, " ".join(text), ha="center", va="center",
            fontsize=34, weight="bold", color=palette.TEXT_PRIMARY)
    if subtitle:
        ax.text(0.5, 0.43, subtitle, ha="center", va="center",
                fontsize=13, color=palette.TEXT_MUTED)
    figure.savefig(path, dpi=dpi, facecolor=palette.INK)
    return path


def _card_clip(png: Path, mp4: Path, seconds: float, fps: int = 25) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
         "-t", f"{seconds}", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "slow", "-crf", "18", "-vf", "scale=1920:1080", str(mp4)],
        check=True,
    )
    return mp4


def build_montage(entries: list[dict], out_dir: Path, chapters: bool = True) -> Path:
    """Title card, then each clip, concatenated without re-encoding the scenes."""

    out = out_dir / "strong_five_montage.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parts: list[Path] = []
        parts.append(_card_clip(_card(TITLE, "five human-confirmed scenes",
                                      tmp_path / "title.png"),
                                tmp_path / "title.mp4", 2.4))
        total = len(entries)
        for entry in entries:
            if chapters:
                label = f"STRONG {entry['position']} / {total}"
                parts.append(_card_clip(
                    _card(label, entry["shape"], tmp_path / f"c{entry['position']}.png"),
                    tmp_path / f"c{entry['position']}.mp4", 1.1,
                ))
            parts.append(out_dir / f"strong_{entry['position']:02d}.mp4")

        listing = tmp_path / "parts.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # different stream parameters: fall back to a single re-encode
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "slow", "-crf", "18", "-r", "25", str(out)],
                check=True,
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--only", type=int, nargs="*", default=None,
                        help="render only these positions (1-5)")
    parser.add_argument("--no-montage", action="store_true")
    parser.add_argument("--stills-only", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (demo_paths().export_dir / "strong")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = strong_scenes()
    if len(entries) != 5:
        print(f"!! expected 5 strong scenes, found {len(entries)}", file=sys.stderr)
    wanted = [e for e in entries if not args.only or e.position in args.only]

    if not args.stills_only and not has_ffmpeg():
        print("!! ffmpeg not found; writing stills only", file=sys.stderr)
        args.stills_only = True

    started = time.time()
    records = []
    for entry in wanted:
        print(f"[{entry.position}/{len(entries)}] {entry.clip_id}  {entry.shape}",
              flush=True)
        record = render_one(entry, out_dir, still_only=args.stills_only)
        print(f"        runner {record['roles']['runner']}  "
              f"defender {record['roles']['defender']}  "
              f"beneficiary {record['roles']['beneficiary']}")
        records.append(record)

    manifest = out_dir / "manifest.json"
    if args.only and manifest.exists():                # keep the untouched entries
        existing = {r["position"]: r for r in json.loads(manifest.read_text())["scenes"]}
        existing.update({r["position"]: r for r in records})
        records = [existing[k] for k in sorted(existing)]
    manifest.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "space_view": "available",
         "roles": "human annotation (shot_annotations.xlsx, effect=strong)",
         "scenes": records},
        indent=2, ensure_ascii=False,
    ))

    if not args.no_montage and not args.stills_only and len(records) == len(entries):
        print("montage …", flush=True)
        print(f"wrote {build_montage(records, out_dir)}")
    print(f"done in {time.time() - started:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
