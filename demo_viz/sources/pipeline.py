"""Adapter for pipeline-predicted role triplets.

The research pipeline does not yet emit a stable scene file, so this adapter is
deliberately format-tolerant: it accepts any JSON object that carries a clip or
match/frame reference plus predicted role ids, and reuses the same IDSSE
loading path as the manual-annotation adapter.  Two shapes are understood:

1. ``{"clip_id": "...", "runner": [...], "defender": [...], "beneficiary": [...]}``
   Role entries may be shirt numbers (``"7"``) or DFL person ids
   (``"DFL-OBJ-002GM1"``); both are resolved against the match roster.

2. ``{"match_id": "J03WOH", "period": 1, "center_seconds": 1054.9, ...}``
   for a scene that has no annotated clip.

Anything else raises with a message naming the keys it needed, rather than
silently rendering the wrong players.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..annotations import AnnotatedClip
from ..scene import Scene
from .annotation_scene import build_scene_from_clip

ROLE_KEYS = {
    "runner": ("runner", "runners", "runner_ids", "offball_attackers"),
    "defender": ("defender", "defenders", "defender_ids", "drawn_defenders",
                 "reacting_defenders"),
    "beneficiary": ("beneficiary", "beneficiaries", "beneficiary_ids",
                    "space_beneficiaries"),
}


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _as_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return tuple(part.strip() for part in str(value).replace(";", ",").split(",") if part.strip())


def scene_from_record(record: Mapping[str, Any], **kwargs) -> Scene:
    """Build a canonical scene from one pipeline record."""

    clip_id = str(_first(record, ("clip_id", "clip", "scene_id")) or "")
    match_id = str(_first(record, ("match_id", "match")) or "")
    period = _first(record, ("period", "half"))
    seconds = _first(record, ("period_seconds", "center_seconds", "seconds", "shot_seconds"))

    if clip_id and ":" in clip_id and not (match_id and period and seconds):
        from ..annotations import find_clip

        try:
            base = find_clip(clip_id)
        except KeyError:
            base = None
        if base is not None:
            match_id, period, seconds = base.match_id, base.period, base.period_seconds

    if not (match_id and period is not None and seconds is not None):
        raise ValueError(
            "pipeline record needs either a known clip_id, or match_id + period + "
            f"period_seconds; got keys {sorted(record)}"
        )

    roles = {name: _as_list(_first(record, keys)) for name, keys in ROLE_KEYS.items()}
    if not any(roles.values()):
        raise ValueError(
            "pipeline record carries no role ids; expected one of "
            f"{sorted(k for keys in ROLE_KEYS.values() for k in keys)}"
        )

    clip = AnnotatedClip(
        clip_id=clip_id or f"{match_id}:P{period}_{float(seconds):.0f}",
        match_id=match_id,
        shot_number=None,
        period=int(float(period)),
        period_seconds=float(seconds),
        match_clock=str(record.get("match_clock", "")),
        team=str(record.get("team", "")),
        shooter=str(record.get("shooter", "")),
        shot_result=str(record.get("shot_result", "")),
        video_filename=str(record.get("video_filename", "")),
        runner_shirts=roles["runner"],
        defender_shirts=roles["defender"],
        beneficiary_shirts=roles["beneficiary"],
        effect=str(record.get("effect", "pipeline")),
        notes=str(record.get("notes", "")),
    )
    scene = build_scene_from_clip(clip, **kwargs)
    scene.source = "pipeline"
    scene.provenance["adapter"] = "pipeline"
    scene.provenance["pipeline_record"] = dict(record)
    _resolve_direct_ids(scene, roles)
    return scene


def _resolve_direct_ids(scene: Scene, roles: Mapping[str, Sequence[str]]) -> None:
    """Accept DFL person ids directly, not only shirt numbers."""

    for name, attr in (("runner", "runner_ids"), ("defender", "defender_ids"),
                       ("beneficiary", "beneficiary_ids")):
        current = list(getattr(scene, attr))
        for token in roles[name]:
            if token in scene.players and token not in current:
                current.append(token)
        setattr(scene, attr, tuple(current))


def load_pipeline_scenes(path: Path, **kwargs) -> list[Scene]:
    """Read a JSON file holding one record or a list of records."""

    payload = json.loads(Path(path).read_text())
    records: Iterable[Mapping[str, Any]]
    if isinstance(payload, dict):
        records = payload.get("scenes", [payload])
    else:
        records = payload
    return [scene_from_record(record, **kwargs) for record in records]
