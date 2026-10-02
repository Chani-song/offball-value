"""The submission manifest: which scenes the final demo shows, in what order.

The final ten are not chosen yet, and nothing here chooses them. This module
defines the shape that list will have, validates one, and loads it. When no
manifest is present the demo falls back to the existing Strong/Medium
explorer, which is the behaviour today.

Two kinds of entry are allowed, and they are kept apart on purpose:

``scene``
    a tracked Bundesliga scene from ``web_data/``, identified by ``scene_id``.

``solver_reference``
    one of the solver's own declared 2v1 study states. It is **not** a match
    scene and must never be presented as one; it exists so the solver view can
    be shown working against a real artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Where the demo looks, unless told otherwise.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "submission_scenes.json"

KINDS = ("scene", "solver_reference")
ROLE_KEYS = ("runners", "defenders", "beneficiaries")


class ManifestError(ValueError):
    """The manifest exists but does not describe something we can show."""


@dataclass(frozen=True)
class SubmissionEntry:
    scene_id: str
    title: str
    order: int
    kind: str = "scene"
    annotation: dict[str, tuple[str, ...]] = field(default_factory=dict)
    focus_frame: int | None = None
    solver_artifact: str | None = None
    notes: str = ""

    def to_payload(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "order": self.order,
            "kind": self.kind,
            "annotation": {k: list(v) for k, v in self.annotation.items()},
            "focus_frame": self.focus_frame,
            "solver_artifact": self.solver_artifact,
            "notes": self.notes,
        }


def _roles(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: annotation must be an object")
    out: dict[str, tuple[str, ...]] = {}
    for key in ROLE_KEYS:
        value = raw.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ManifestError(f"{where}: annotation.{key} must be a list of ids")
        out[key] = tuple(value)
    unknown = set(raw) - set(ROLE_KEYS)
    if unknown:
        raise ManifestError(f"{where}: unknown annotation keys {sorted(unknown)}")
    return out


def validate_entry(raw: Any, position: int) -> SubmissionEntry:
    where = f"entry {position}"
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: must be an object")
    for required in ("scene_id", "order"):
        if required not in raw:
            raise ManifestError(f"{where}: missing '{required}'")
    scene_id = raw["scene_id"]
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ManifestError(f"{where}: scene_id must be a non-empty string")
    order = raw["order"]
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ManifestError(f"{where}: order must be a positive integer")
    kind = raw.get("kind", "scene")
    if kind not in KINDS:
        raise ManifestError(f"{where}: kind must be one of {KINDS}")
    focus = raw.get("focus_frame")
    if focus is not None and (not isinstance(focus, int) or isinstance(focus, bool) or focus < 0):
        raise ManifestError(f"{where}: focus_frame must be a non-negative integer")
    artifact = raw.get("solver_artifact")
    if artifact is not None and not isinstance(artifact, str):
        raise ManifestError(f"{where}: solver_artifact must be a string or null")
    if kind == "solver_reference" and not artifact:
        raise ManifestError(f"{where}: a solver_reference entry needs a solver_artifact")
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise ManifestError(f"{where}: notes must be a string")
    return SubmissionEntry(
        scene_id=scene_id,
        title=str(raw.get("title") or scene_id),
        order=order,
        kind=kind,
        annotation=_roles(raw.get("annotation"), where),
        focus_frame=focus,
        solver_artifact=artifact,
        notes=notes,
    )


def validate(raw: Any) -> tuple[SubmissionEntry, ...]:
    """Every entry, in ``order``. Raises rather than dropping a bad one."""

    if isinstance(raw, dict):
        raw = raw.get("scenes", raw.get("entries"))
    if not isinstance(raw, list):
        raise ManifestError("manifest must be a list of entries, or {'scenes': [...]}")
    entries = tuple(validate_entry(item, i) for i, item in enumerate(raw))
    orders = [e.order for e in entries]
    if len(set(orders)) != len(orders):
        raise ManifestError(f"duplicate order values: {sorted(orders)}")
    ids = [e.scene_id for e in entries]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate scene_id values")
    return tuple(sorted(entries, key=lambda e: e.order))


def load(path: Path | None = None) -> tuple[SubmissionEntry, ...] | None:
    """The manifest, or ``None`` when there is not one.

    ``None`` is the graceful fallback the browser relies on: no manifest means
    the existing Strong/Medium explorer, unchanged.
    """

    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path}: not valid JSON ({error})") from error
    return validate(raw)


def to_payload(entries: Iterable[SubmissionEntry]) -> dict:
    return {"scenes": [e.to_payload() for e in entries]}
