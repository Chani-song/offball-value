"""demo_viz - a reusable renderer for off-ball value scenes.

Typical use::

    from demo_viz import load_scene, build_storyboard, render_movie

    scene = load_scene("J03WOH:shot_006_P1_1054")
    render_movie(scene, build_storyboard(scene), Path("out.mp4"))
"""

from .scene import Scene, ScenePlayer  # noqa: F401
from .story import Storyboard, build_storyboard  # noqa: F401

__all__ = [
    "Scene",
    "ScenePlayer",
    "Storyboard",
    "build_storyboard",
    "load_scene",
    "render_movie",
    "render_still",
]


def load_scene(scene_id: str, **kwargs):
    """Load a scene by annotation clip id, pipeline id, or ``synthetic``."""

    from .loader import load_scene as _load

    return _load(scene_id, **kwargs)


def render_movie(*args, **kwargs):
    from .animate import render_movie as _render

    return _render(*args, **kwargs)


def render_still(*args, **kwargs):
    from .animate import render_still as _render

    return _render(*args, **kwargs)
