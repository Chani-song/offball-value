"""Scene and quantity services shared by every callback.

Dash callbacks are stateless, so anything expensive lives here behind a small
cache: loading a scene reads a ~400 MB positions file the first time and a
compressed window cache afterwards, and the influence cache costs about 40 ms.
Both are keyed so switching scenes back and forth stays instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..annotations import AnnotatedClip, load_annotations
from ..config import demo_paths
from ..core.influence import InfluenceCache
from ..core.role_logic import runner_onset_index
from ..loader import load_scene
from ..scene import Scene

#: Coarser than the video renderer, because the app recomputes on every click.
GRID_RESOLUTION_M = 2.0
SAMPLE_EVERY = 5


@dataclass
class SceneBundle:
    scene: Scene
    cache: InfluenceCache
    clip: AnnotatedClip | None

    @property
    def notes(self) -> str:
        return (self.clip.notes if self.clip else "") or self.scene.notes or ""

    @property
    def effect(self) -> str:
        return (self.clip.effect if self.clip else "") or ""


_BUNDLES: dict[str, SceneBundle] = {}
_ONSETS: dict[tuple[str, str], tuple[int, str]] = {}


def scene_options() -> list[dict[str, str]]:
    """Dropdown entries: annotated scenes first, strongest first, then synthetic."""

    options: list[dict[str, str]] = []
    try:
        clips = load_annotations()
    except Exception:
        clips = []
    order = {"strong": 0, "medium": 1, "low": 2}
    renderable = [c for c in clips if c.is_renderable and c.effect in order]
    renderable.sort(key=lambda c: (order.get(c.effect, 9), c.clip_id))
    for clip in renderable:
        options.append({
            "label": f"{clip.effect}  ·  {clip.clip_id}  ·  {clip.shape}",
            "value": clip.clip_id,
        })
    options.append({"label": "synthetic  ·  smoke test", "value": "synthetic"})
    return options


def get_bundle(scene_id: str) -> SceneBundle:
    """Load a scene once and keep it, with its influence cache."""

    bundle = _BUNDLES.get(scene_id)
    if bundle is not None:
        return bundle
    scene = load_scene(
        scene_id, quantities=False, surfaces=False,
        every=SAMPLE_EVERY, grid_resolution_m=GRID_RESOLUTION_M,
    )
    cache = InfluenceCache(scene, every=SAMPLE_EVERY, grid_resolution_m=GRID_RESOLUTION_M)
    clip = _clip_for(scene_id)
    bundle = SceneBundle(scene=scene, cache=cache, clip=clip)
    _BUNDLES[scene_id] = bundle
    return bundle


def onset_for(scene_id: str, runner_id: str | None) -> tuple[int | None, str]:
    """Where the chosen runner's run starts, cached per (scene, runner)."""

    if not runner_id:
        return None, ""
    key = (scene_id, runner_id)
    cached = _ONSETS.get(key)
    if cached is None:
        bundle = get_bundle(scene_id)
        cached = runner_onset_index(bundle.scene, runner_id)
        _ONSETS[key] = cached
    return cached


@lru_cache(maxsize=1)
def _annotation_index() -> dict[str, AnnotatedClip]:
    try:
        return {clip.clip_id: clip for clip in load_annotations()}
    except Exception:
        return {}


def _clip_for(scene_id: str) -> AnnotatedClip | None:
    return _annotation_index().get(scene_id)


#: Where a real pipeline would drop its triplets, in priority order. The last
#: entry is an example file shipped with the repo so the mode is demonstrable
#: on a machine that has never run the pipeline.
PIPELINE_CANDIDATES = (
    ("pipeline", Path("data/processed/pipeline_triplets.json")),
    ("pipeline", Path("demo_viz/data/pipeline_triplets.json")),
    ("example", Path("demo_viz/data/suggested_triplets.example.json")),
)


def pipeline_source() -> tuple[Path | None, str]:
    """The triplet file in use, and whether it is real pipeline output."""

    root = demo_paths().cache_dir.parent.parent
    for kind, relative in PIPELINE_CANDIDATES:
        candidate = root / relative
        if candidate.exists():
            return candidate, kind
    return None, ""


def pipeline_path() -> Path | None:
    return pipeline_source()[0]


def pipeline_triplets(scene_id: str) -> list[dict[str, Any]]:
    """Predicted triplets for this scene, if a triplet file exists."""

    path, _kind = pipeline_source()
    if path is None:
        return []
    import json

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    records = payload.get("scenes", payload) if isinstance(payload, dict) else payload
    return [r for r in records
            if str(r.get("clip_id", r.get("scene_id", ""))) == scene_id]


def time_marks(scene: Scene, count: int = 7) -> dict[int, str]:
    """Slider ticks in seconds relative to the annotated event."""

    if scene.n_frames < 2:
        return {0: "0"}
    steps = np.linspace(0, scene.n_frames - 1, count).astype(int)
    return {int(i): f"{scene.times[int(i)]:+.0f}" for i in steps}
