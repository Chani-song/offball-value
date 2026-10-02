"""Export layer: still frames, MP4, and GIF.

Frames are produced by redrawing :class:`demo_viz.render.figure.SceneFigure`
and handed straight to a matplotlib writer, so the same code path produces the
preview frame, the video and the animated GIF.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

from .render.figure import FigureConfig, SceneFigure
from .scene import Scene
from .story import Storyboard


@dataclass
class ExportConfig:
    """Frame rate, subsampling and the holds at each end of a clip."""

    fps: float = 25.0
    stride: int = 1                 # source frames per rendered frame
    hold_start_s: float = 0.6       # freeze on the opening frame
    hold_end_s: float = 1.6         # freeze on the payoff frame
    bitrate: int = 7200
    dpi: int | None = None


def has_ffmpeg() -> bool:
    """Whether MP4 export is possible on this machine."""

    return shutil.which("ffmpeg") is not None


def frame_plan(scene: Scene, config: ExportConfig) -> list[int]:
    """Indices to render, including the opening and closing holds."""

    stride = max(1, int(config.stride))
    body = list(range(0, scene.n_frames, stride))
    if body[-1] != scene.n_frames - 1:
        body.append(scene.n_frames - 1)
    out_fps = scene.fps / stride
    lead = [body[0]] * int(round(config.hold_start_s * out_fps))
    tail = [body[-1]] * int(round(config.hold_end_s * out_fps))
    return lead + body + tail


def render_still(
    scene: Scene,
    storyboard: Storyboard,
    path: Path,
    at_time: float | None = None,
    figure_config: FigureConfig | None = None,
    dpi: int | None = None,
) -> Path:
    """One frame, defaulting to the moment the story pays off."""

    figure = SceneFigure(scene, storyboard, figure_config)
    if at_time is None:
        at_time = scene.moments.get("beneficiary")
    if at_time is None:
        at_time = float(scene.times[int(scene.n_frames * 0.75)])
    figure.draw(scene.index_at(float(at_time)))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.fig.savefig(path, dpi=dpi or figure.config.dpi, facecolor=figure.fig.get_facecolor())
    return path


@dataclass
class Shot:
    """One frame of a hand-composed sequence.

    ``index`` is the scene frame to draw. ``story_time`` runs the storyboard on
    its own clock, so a sequence can hold on one frame while a narration plays
    over it; ``viewport`` overrides the camera; ``storyboard`` swaps the
    narration itself, which is how a demo video puts a role introduction in
    front of the play without rebuilding the figure.
    """

    index: int
    story_time: float | None = None
    viewport: tuple[float, float, float, float] | None = None
    storyboard: Storyboard | None = None


def render_shots(
    scene: Scene,
    storyboard: Storyboard,
    shots: Sequence[Shot],
    path: Path,
    config: ExportConfig | None = None,
    figure_config: FigureConfig | None = None,
    progress: bool = True,
) -> Path:
    """Write an MP4 from an explicit frame list rather than a frame plan."""

    config = config or ExportConfig()
    figure = SceneFigure(scene, storyboard, figure_config)
    out_fps = config.fps

    path.parent.mkdir(parents=True, exist_ok=True)
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg is required for MP4 export")
    writer = FFMpegWriter(
        fps=out_fps,
        bitrate=config.bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18"],
    )
    dpi = config.dpi or figure.config.dpi
    with writer.saving(figure.fig, str(path), dpi):
        for step, shot in enumerate(shots):
            figure.storyboard = shot.storyboard or storyboard
            figure.draw(int(shot.index), story_time=shot.story_time,
                        viewport=shot.viewport)
            writer.grab_frame(facecolor=figure.fig.get_facecolor())
            if progress and step % 25 == 0:
                print(f"    frame {step + 1}/{len(shots)}", flush=True)
    figure.storyboard = storyboard
    return path


def render_movie(
    scene: Scene,
    storyboard: Storyboard,
    path: Path,
    config: ExportConfig | None = None,
    figure_config: FigureConfig | None = None,
    progress: bool = True,
) -> Path:
    """Write an MP4 (``.mp4``) or animated GIF (``.gif``)."""

    config = config or ExportConfig()
    figure = SceneFigure(scene, storyboard, figure_config)
    plan = frame_plan(scene, config)
    out_fps = scene.fps / max(1, int(config.stride))

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=out_fps)
    else:
        if not has_ffmpeg():
            raise RuntimeError("ffmpeg is required for MP4 export; use a .gif path instead")
        writer = FFMpegWriter(
            fps=out_fps,
            bitrate=config.bitrate,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18"],
        )

    dpi = config.dpi or figure.config.dpi
    with writer.saving(figure.fig, str(path), dpi):
        for step, index in enumerate(plan):
            figure.draw(int(index))
            writer.grab_frame(facecolor=figure.fig.get_facecolor())
            if progress and step % 25 == 0:
                print(f"    frame {step + 1}/{len(plan)}", flush=True)
    return path


def gif_from_movie(
    movie: Path,
    path: Path,
    width: int = 820,
    fps: float = 10.0,
) -> Path:
    """Transcode an MP4 to a GIF with a per-clip palette.

    Matplotlib's ``PillowWriter`` writes an unoptimised GIF with the default
    web palette, which both bloats the file and posterises the soft space
    field. ffmpeg's ``palettegen``/``paletteuse`` pair fixes both at once.
    """

    import subprocess

    if not has_ffmpeg():
        raise RuntimeError("ffmpeg is required for gif_from_movie")
    path.parent.mkdir(parents=True, exist_ok=True)
    chain = f"fps={fps},scale={width}:-2:flags=lanczos"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(movie),
            "-filter_complex",
            f"[0:v]{chain},split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop", "0", str(path),
        ],
        check=True,
    )
    return path


def storyboard_contact_sheet(
    scene: Scene,
    storyboard: Storyboard,
    path: Path,
    figure_config: FigureConfig | None = None,
    dpi: int | None = None,
) -> list[Path]:
    """One still per story beat, useful for slides and for reviewing a render."""

    figure = SceneFigure(scene, storyboard, figure_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for position, beat in enumerate(storyboard.beats):
        nxt = storyboard.beats[position + 1] if position + 1 < len(storyboard.beats) else None
        end = nxt.start if nxt else float(scene.times[-1])
        at = min(beat.start + 0.35 * (end - beat.start), float(scene.times[-1]))
        figure.draw(scene.index_at(at))
        out = path.with_name(f"{path.stem}_{position}_{beat.key}{path.suffix}")
        figure.fig.savefig(out, dpi=dpi or figure.config.dpi,
                           facecolor=figure.fig.get_facecolor())
        written.append(out)
    return written
