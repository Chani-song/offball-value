#!/usr/bin/env python3
"""Off-the-ball investigation examples: the five `strong` scenes, and a montage.

    python -m demo_viz.export_strong                 # all five + montage
    python -m demo_viz.export_strong --no-montage
    python -m demo_viz.export_strong --only 1 2

Writes into ``demo_viz/exports/strong/``:

    strong_01.mp4 … strong_05.mp4      one scene each, 1920x1080
    strong_01.png … strong_05.png      a representative frame from each
    strong_five_montage.mp4            title card + the five clips
    manifest.json                      which scene each number is, and its roles

Each video runs in three phases:

    1. the annotated roles are named one at a time over a frozen opening frame,
       about a second each, with the camera pushed in on them;
    2. the camera settles back to the framing the play opens with;
    3. the scene plays, with every role and the space field held at full
       strength from the first frame to the last.

Phase 3 deliberately does not wait for a detector. The beat detectors place
their moments where the measured signals move, which on several of these scenes
is halfway through the clip — fine for an analysis render, useless for a talk.
The captions walk the causal chain on a fixed clock instead, which the frame
says under the timeline.

The roles are the human annotations from ``shot_annotations.xlsx`` — every
annotated runner, defender and beneficiary, with no heuristic substitution. The
shaded field is the beneficiary's **available space** (the factual residual),
not the difference against the held-defender baseline.
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
import numpy as np

matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "demo_viz"

from . import palette
from .animate import ExportConfig, Shot, has_ffmpeg, render_shots, render_still
from .annotations import select_clips
from .config import demo_paths
from .loader import figure_config_for, load_scene
from .render.camera import blend_boxes, focus_box
from .render.figure import SceneFigure
from .story import build_presentation_storyboard, build_role_intro

#: Layers kept off in the rendered video: the passing lane and the forward look
#: at the beneficiary's own path both add ink without adding an answer.
DISABLED_LAYERS = ("lane", "future")
TITLE = "OFF-THE-BALL SPACE"
SUBTITLE = "Off-the-ball investigation examples"

#: How long each annotated role is held on screen before the scene runs.
INTRO_SECONDS_PER_ROLE = 1.0
#: …and how long each player gets when a role has to be visited one at a time.
INTRO_SECONDS_PER_PLAYER = 0.75
#: Two players in the same role further apart than this are visited one at a
#: time; closer than this, one box holds both. In these five scenes the pairs
#: fall cleanly either side of it (8 m apart, or 16-24 m).
SPLIT_SEPARATION_M = 18.0
#: Floor on the close-up width. Tighter than this and the pitch loses every
#: landmark, so the viewer can see the player but not where they are.
INTRO_WIDTH_M = 36.0
INTRO_RESET_SECONDS = 0.9
#: How long the camera takes to travel between two role close-ups.
INTRO_MOVE_SECONDS = 0.38
#: Where the representative still is taken, as a fraction of the clip.
PRESENTATION_STILL = 0.80


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


def _role_ids(scene, role: str) -> list[str]:
    return {
        "runner": list(scene.runner_ids),
        "defender": list(scene.defender_ids),
        "beneficiary": list(scene.beneficiary_ids),
    }[role]


def _split_roles(scene, figure: SceneFigure, context_index: int) -> tuple[str, ...]:
    """Roles whose players are too far apart to introduce in one shot.

    Two runners twenty metres apart would widen a single box until it is no
    longer a close-up; the camera visits them in turn instead. Players standing
    together stay in one shot, which reads better than two near-identical ones.
    """

    out = []
    for role in ("runner", "defender", "beneficiary"):
        ids = [i for i in _role_ids(scene, role) if i in scene.players]
        if len(ids) < 2:
            continue
        points = [scene.view_xy(scene.players[i].xy[context_index]) for i in ids]
        spread = max(float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
                     for a in points for b in points)
        if spread > SPLIT_SEPARATION_M:
            out.append(role)
    return tuple(out)


def _intro_shots(scene, figure: SceneFigure, intro, marks, seconds: float,
                 context_index: int, fps: float) -> list[Shot]:
    """Phase 1 and 2: name each role over a frozen frame, then settle.

    The camera sits on one role (or one player) at a time and eases to the
    next, ending on the exact framing the play opens with so the cut into
    phase 3 is invisible.
    """

    boxes: list[tuple[float, tuple[float, float, float, float]]] = []
    for start, role, ids in marks:
        if role == "cast" or not ids:
            box = figure.camera.viewport(context_index, 1.0)
        else:
            box = focus_box(scene, context_index, ids, figure.aspect,
                            minimum_width_m=INTRO_WIDTH_M)
        boxes.append((start, box))

    def viewport_at(t: float):
        box = boxes[0][1]
        for position, (start, target) in enumerate(boxes):
            if t < start - INTRO_MOVE_SECONDS:
                break
            if position == 0:
                box = target
                continue
            previous = boxes[position - 1][1]
            weight = (t - (start - INTRO_MOVE_SECONDS)) / INTRO_MOVE_SECONDS
            box = blend_boxes(previous, target, min(1.0, max(0.0, weight)))
        return box

    count = max(1, int(round(seconds * fps)))
    return [
        Shot(index=context_index, story_time=step / fps,
             viewport=viewport_at(step / fps), storyboard=intro)
        for step in range(count)
    ]


def render_one(entry: StrongScene, out_dir: Path, still_only: bool = False) -> dict:
    """One scene: role introduction, reset, then the play with the roles held on.

    The play never waits for a detector — every role and the space field are at
    full strength from its first frame — so the sequence is readable from the
    start whatever the scene's own timing turns out to be.
    """

    scene = load_scene(entry.clip_id, surfaces=True, every=5, grid_resolution_m=1.0)
    play = build_presentation_storyboard(scene, space_view="available")
    config = _config_for(scene)

    still_at = float(scene.times[0]) + PRESENTATION_STILL * (
        float(scene.times[-1]) - float(scene.times[0]))
    png = render_still(scene, play, out_dir / f"{entry.stem}.png",
                       at_time=still_at, figure_config=config)
    files = [png]

    intro_seconds = 0.0
    if not still_only:
        export = ExportConfig(hold_start_s=0.0, hold_end_s=1.6)
        figure = SceneFigure(scene, play, config)
        context_index = 0
        intro, marks, intro_seconds = build_role_intro(
            scene, INTRO_SECONDS_PER_ROLE, INTRO_RESET_SECONDS,
            split_roles=_split_roles(scene, figure, context_index),
            seconds_per_player=INTRO_SECONDS_PER_PLAYER,
        )
        shots = _intro_shots(scene, figure, intro, marks, intro_seconds,
                             context_index, export.fps)
        body = list(range(context_index, scene.n_frames))
        tail = [body[-1]] * int(round(export.hold_end_s * export.fps))
        shots += [Shot(index=index) for index in body + tail]
        files.append(render_shots(scene, play, shots,
                                  out_dir / f"{entry.stem}.mp4",
                                  config=export, figure_config=config,
                                  progress=False))

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
        "intro_seconds": round(intro_seconds, 2),
        "pacing": "presentation",
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
        parts.append(_card_clip(_card(TITLE, SUBTITLE,
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
