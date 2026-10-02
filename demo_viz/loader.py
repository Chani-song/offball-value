"""One entry point for every scene source.

``load_scene`` resolves an id the way a user would expect:

* ``"synthetic"``                    -> generated smoke-test scene
* ``"J03WOH:shot_006_P1_1054"``      -> manual annotation + IDSSE tracking
* ``"strong"`` / ``"strong:2"``      -> the nth renderable scene of that effect
* ``path/to/records.json``           -> pipeline-predicted triplets

Presentation-only overrides (window trim, camera, which space layer to shade)
come from ``scenes.json`` and never touch a measured quantity.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import find_clip, select_clips
from .config import DEMO_ROOT
from .scene import Scene

SCENE_BOOK_PATH = DEMO_ROOT / "scenes.json"

#: ``"strong"``/``"strong:2"`` select the nth annotated scene of that effect.
#: Real clip ids also contain a colon, so the prefix is checked against this set
#: before falling back to a spreadsheet lookup.
EFFECT_SELECTORS = {"strong", "medium", "low"}


def scene_book() -> dict[str, Any]:
    """The presentation-only overrides in ``scenes.json``."""

    if not SCENE_BOOK_PATH.exists():
        return {"defaults": {}, "scenes": {}}
    return json.loads(SCENE_BOOK_PATH.read_text())


def presentation_for(scene_id: str) -> dict[str, Any]:
    """Defaults merged with one scene's overrides."""

    book = scene_book()
    merged = dict(book.get("defaults", {}))
    merged.update(book.get("scenes", {}).get(scene_id, {}))
    return merged


def load_scene(
    scene_id: str,
    stride: int = 1,
    quantities: bool = True,
    surfaces: bool = True,
    every: int = 5,
    grid_resolution_m: float = 1.5,
    before_s: float | None = None,
    after_s: float | None = None,
    freeze_at: str = "onset",
    baseline_mode: str = "hold",
) -> Scene:
    """Resolve ``scene_id`` to a fully populated :class:`~demo_viz.scene.Scene`."""

    scene_id = scene_id.strip()

    if scene_id.lower() in {"synthetic", "smoke", "demo"}:
        from .sources.synthetic import synthetic_scene

        scene = synthetic_scene()
        if quantities:
            _attach(scene, surfaces, every, grid_resolution_m, freeze_at, baseline_mode)
        return scene

    path = Path(scene_id).expanduser()
    if path.suffix.lower() == ".json" and path.exists():
        from .sources.pipeline import load_pipeline_scenes

        scenes = load_pipeline_scenes(path, stride=stride)
        if not scenes:
            raise ValueError(f"{path} contained no pipeline scenes")
        scene = scenes[0]
        if quantities:
            _attach(scene, surfaces, every, grid_resolution_m, freeze_at, baseline_mode)
        return scene

    from .sources.annotation_scene import build_scene_from_clip

    effect, _, position = scene_id.partition(":")
    if effect.lower() not in EFFECT_SELECTORS:
        clip = find_clip(scene_id)
    else:
        clips = select_clips(effect.lower())
        if not clips:
            raise KeyError(f"no renderable annotated clips with effect {effect!r}")
        index = int(position) if position else 0
        clip = clips[index % len(clips)]

    settings = presentation_for(clip.clip_id)
    scene = build_scene_from_clip(
        clip,
        before_s=float(before_s if before_s is not None else settings.get("before_s", 8.0)),
        after_s=float(after_s if after_s is not None else settings.get("after_s", 3.0)),
        stride=stride,
    )
    scene.provenance["presentation"] = settings
    if settings.get("headline"):
        scene.subtitle = f"{scene.subtitle}   ·   {settings['headline']}"
    if quantities:
        _attach(scene, surfaces, every, grid_resolution_m, freeze_at, baseline_mode)
    return scene


def _attach(scene: Scene, surfaces: bool, every: int, grid_resolution_m: float,
            freeze_at: str = "onset", baseline_mode: str = "hold") -> None:
    from .quantities import attach_quantities

    attach_quantities(
        scene,
        surfaces=surfaces,
        every=every,
        grid_resolution_m=grid_resolution_m,
        series_resolution_m=max(grid_resolution_m, 1.5),
        freeze_at=freeze_at,
        baseline_mode=baseline_mode,
    )


def list_scenes(effect: str | tuple[str, ...] = ("strong", "medium")) -> list[str]:
    return [clip.clip_id for clip in select_clips(effect)]


def figure_config_for(scene: Scene, **overrides):
    """Apply a scene's presentation block to a renderer configuration."""

    from .render.camera import CameraConfig
    from .render.figure import FigureConfig

    settings = scene.provenance.get("presentation", {}) or {}
    camera = CameraConfig(**{**settings.get("camera", {})})
    config = FigureConfig(camera=camera)
    if settings.get("wake_mode"):
        config.wake_mode = settings["wake_mode"]
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    if scene.surfaces is None and config.wake_mode in ("residual", "delta"):
        config.wake_mode = "geometric"
    return config
