"""Path discovery for the off-ball demo renderer.

Nothing here is hard-wired to one machine: every location is resolved from an
environment variable first, then from a short list of conventional locations.
Missing optional inputs degrade to ``None`` rather than raising, so the renderer
can still run on a synthetic smoke-test scene.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent
CACHE_DIR = Path(os.environ.get("OFFBALL_DEMO_CACHE", DEMO_ROOT / "cache"))
EXPORT_DIR = Path(os.environ.get("OFFBALL_DEMO_EXPORTS", DEMO_ROOT / "exports"))

_IDSSE_CANDIDATES = (
    "$OFFBALL_IDSSE_DIR",
    str(REPO_ROOT / "data" / "raw" / "bundesliga-integrated"),
    str(REPO_ROOT / "data" / "raw" / "idsse-data"),
    "~/idsse_shots/idsse-data",
    str(REPO_ROOT.parent / "idsse_shots" / "idsse-data"),
)

_ANNOTATION_CANDIDATES = (
    "$OFFBALL_ANNOTATIONS",
    str(REPO_ROOT.parent / "shot_annotations.xlsx"),
    str(REPO_ROOT / "shot_annotations.xlsx"),
    "~/idsse_shots/output/shot_annotations.xlsx",
)

_CLIP_CANDIDATES = (
    "$OFFBALL_CLIP_DIR",
    "~/idsse_shots/output/ALL_SHOT_CLIPS",
    str(REPO_ROOT.parent / "idsse_shots" / "output" / "ALL_SHOT_CLIPS"),
)

_OVERLAY_CACHE_CANDIDATES = (
    "$OFFBALL_OVERLAY_CACHE",
    "~/idsse_shots/output/tracking_overlay_cache",
    str(REPO_ROOT.parent / "idsse_shots" / "output" / "tracking_overlay_cache"),
)


def _resolve(candidates: tuple[str, ...], *, must_be_dir: bool) -> Path | None:
    for raw in candidates:
        if raw.startswith("$"):
            value = os.environ.get(raw[1:])
            if not value:
                continue
            raw = value
        path = Path(raw).expanduser()
        if must_be_dir and path.is_dir():
            return path.resolve()
        if not must_be_dir and path.is_file():
            return path.resolve()
    return None


@dataclass(frozen=True)
class DemoPaths:
    """Every input and output location the demo resolved."""

    idsse_dir: Path | None
    annotations_xlsx: Path | None
    clip_dir: Path | None
    overlay_cache_dir: Path | None
    cache_dir: Path
    export_dir: Path

    def describe(self) -> str:
        rows = [
            ("IDSSE raw XML", self.idsse_dir),
            ("shot_annotations.xlsx", self.annotations_xlsx),
            ("shot clip videos", self.clip_dir),
            ("5 Hz overlay cache", self.overlay_cache_dir),
            ("scene cache", self.cache_dir),
            ("exports", self.export_dir),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(
            f"  {name.ljust(width)} : {path if path else '-- not found --'}"
            for name, path in rows
        )


def demo_paths() -> DemoPaths:
    """Resolve every path, creating the cache and export directories."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return DemoPaths(
        idsse_dir=_resolve(_IDSSE_CANDIDATES, must_be_dir=True),
        annotations_xlsx=_resolve(_ANNOTATION_CANDIDATES, must_be_dir=False),
        clip_dir=_resolve(_CLIP_CANDIDATES, must_be_dir=True),
        overlay_cache_dir=_resolve(_OVERLAY_CACHE_CANDIDATES, must_be_dir=True),
        cache_dir=CACHE_DIR,
        export_dir=EXPORT_DIR,
    )


def ensure_repo_on_path() -> None:
    """Make ``offball_value`` importable without installing the package."""

    import sys

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
