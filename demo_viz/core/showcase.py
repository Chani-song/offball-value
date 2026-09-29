"""Schema, validator and loader for the curated submission showcase.

``demo_viz/data/submission_showcase.json`` is curation metadata: which scenes
the submission puts forward, who reviewed them and how they rated them, the
hand-assigned roles, and which repository scene each one is played from. It
holds no trajectories, and the validator enforces that -- a record that carries
one is a leak from the local source, not a valid record.

``featured`` and ``order`` are the only fields that change when the final ten
are chosen, and nothing in the code depends on which they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "submission_showcase.json"

PROVENANCE = ("human", "solver")
MAPPING_STATUS = ("verified", "unresolved", "ambiguous")
SOLVER_STATUS = ("available", "unavailable", "unresolved")
TIMING = ("슛", "run_onset")
ROLE_KEYS = ("runners", "defenders", "beneficiaries")

#: A record carrying any of these is a trajectory leak, not curation metadata.
TRACK_KEYS = ("t", "ball", "xy", "tracks", "players")


class ShowcaseError(ValueError):
    """The showcase file exists but does not describe something we can show."""


@dataclass(frozen=True)
class Reviewer:
    name: str
    rating: int | None
    offball: str = ""
    note: str = ""


@dataclass(frozen=True)
class ShowcaseScene:
    showcase_id: str
    scene_id: str | None
    provenance: str
    mapping_status: str
    mapping_evidence: str
    reviewers: tuple[Reviewer, ...]
    agreement_label: str
    roles: dict[str, tuple[str, ...]]
    match: str
    half: str
    event: str
    goal: bool
    timing: str
    annotation: str
    solver_status: str
    scenario_type: str | None
    featured: bool = False
    order: int | None = None
    #: A curated one-line title for the case study, and a sentence under it.
    #: Optional and empty for every scene until someone writes one: an
    #: uncurated scene falls back to its match/event label rather than to a
    #: generated phrase. See PAPER_STORY_TRACE.md section 4.
    story_title: str = ""
    story_summary: str = ""

    @property
    def playable(self) -> bool:
        """Whether the demo can actually show this scene.

        A showcase entry without a verified repository scene has no published
        trajectory -- the curated source's own tracking is deliberately not
        republished -- so it is listed and explained, not played.
        """

        return self.mapping_status == "verified" and bool(self.scene_id)

    @property
    def ratings(self) -> tuple[int, ...]:
        return tuple(r.rating for r in self.reviewers if r.rating is not None)

    def to_payload(self) -> dict:
        return {
            "showcase_id": self.showcase_id,
            "scene_id": self.scene_id,
            "provenance": self.provenance,
            "featured": self.featured,
            "order": self.order,
            "playable": self.playable,
            "mapping": {"status": self.mapping_status, "evidence": self.mapping_evidence},
            "review": {
                "agreement_label": self.agreement_label,
                "reviewers": [
                    {"name": r.name, "rating": r.rating, "offball": r.offball, "note": r.note}
                    for r in self.reviewers
                ],
            },
            "roles": {k: list(v) for k, v in self.roles.items()},
            "match": self.match,
            "half": self.half,
            "event": self.event,
            "goal": self.goal,
            "timing": self.timing,
            "annotation": self.annotation,
            "solver": {"status": self.solver_status, "scenario_type": self.scenario_type},
            "story_title": self.story_title,
            "story_summary": self.story_summary,
        }


def _reviewers(raw: Any, where: str) -> tuple[Reviewer, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ShowcaseError(f"{where}: review.reviewers must be a list")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ShowcaseError(f"{where}: each reviewer must be an object")
        rating = item.get("rating")
        if rating is not None and (not isinstance(rating, int) or isinstance(rating, bool)):
            raise ShowcaseError(f"{where}: reviewer rating must be an integer or null")
        out.append(Reviewer(name=str(item.get("name", "")), rating=rating,
                            offball=str(item.get("offball", "")),
                            note=str(item.get("note", ""))))
    return tuple(out)


def validate_scene(raw: Any, position: int) -> ShowcaseScene:
    where = f"showcase entry {position}"
    if not isinstance(raw, dict):
        raise ShowcaseError(f"{where}: must be an object")
    leaked = [k for k in TRACK_KEYS if k in raw]
    if leaked:
        raise ShowcaseError(
            f"{where}: carries trajectory data {leaked}; the showcase file holds "
            "curation metadata only")

    showcase_id = raw.get("showcase_id")
    if not isinstance(showcase_id, str) or not showcase_id.strip():
        raise ShowcaseError(f"{where}: showcase_id must be a non-empty string")
    provenance = raw.get("provenance")
    if provenance not in PROVENANCE:
        raise ShowcaseError(f"{where}: provenance must be one of {PROVENANCE}")

    mapping = raw.get("mapping") or {}
    status = mapping.get("status")
    if status not in MAPPING_STATUS:
        raise ShowcaseError(f"{where}: mapping.status must be one of {MAPPING_STATUS}")
    scene_id = raw.get("scene_id")
    if scene_id is not None and not isinstance(scene_id, str):
        raise ShowcaseError(f"{where}: scene_id must be a string or null")
    if status == "verified" and not scene_id:
        raise ShowcaseError(f"{where}: a verified mapping needs a scene_id")
    if status != "verified" and scene_id:
        raise ShowcaseError(
            f"{where}: only a verified mapping may name a scene_id; "
            "an unresolved entry must not point at one")

    solver = raw.get("solver") or {}
    solver_status = solver.get("status")
    if solver_status not in SOLVER_STATUS:
        raise ShowcaseError(f"{where}: solver.status must be one of {SOLVER_STATUS}")
    if solver_status == "available" and not solver.get("artifact"):
        raise ShowcaseError(
            f"{where}: solver.status 'available' requires a real artifact reference")

    roles_raw = raw.get("roles") or {}
    roles = {}
    for key in ROLE_KEYS:
        value = roles_raw.get(key, [])
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ShowcaseError(f"{where}: roles.{key} must be a list of shirt numbers")
        roles[key] = tuple(value)

    order = raw.get("order")
    if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 1):
        raise ShowcaseError(f"{where}: order must be a positive integer or null")
    featured = raw.get("featured", False)
    if not isinstance(featured, bool):
        raise ShowcaseError(f"{where}: featured must be a boolean")

    for key in ("story_title", "story_summary"):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            raise ShowcaseError(f"{where}: {key} must be a string")

    review = raw.get("review") or {}
    return ShowcaseScene(
        showcase_id=showcase_id,
        scene_id=scene_id,
        provenance=provenance,
        mapping_status=status,
        mapping_evidence=str(mapping.get("evidence", "")),
        reviewers=_reviewers(review.get("reviewers"), where),
        agreement_label=str(review.get("agreement_label", "")),
        roles=roles,
        match=str(raw.get("match", "")),
        half=str(raw.get("half", "")),
        event=str(raw.get("event", "")),
        goal=bool(raw.get("goal", False)),
        timing=str(raw.get("timing", "")),
        annotation=str(raw.get("annotation", "")),
        solver_status=solver_status,
        scenario_type=solver.get("scenario_type"),
        featured=featured,
        order=order,
        story_title=str(raw.get("story_title", "")),
        story_summary=str(raw.get("story_summary", "")),
    )


def validate(raw: Any) -> tuple[ShowcaseScene, ...]:
    if isinstance(raw, dict):
        raw = raw.get("scenes")
    if not isinstance(raw, list):
        raise ShowcaseError("showcase must be {'scenes': [...]}")
    scenes = tuple(validate_scene(item, i) for i, item in enumerate(raw))
    ids = [s.showcase_id for s in scenes]
    if len(set(ids)) != len(ids):
        raise ShowcaseError("duplicate showcase_id values")
    orders = [s.order for s in scenes if s.order is not None]
    if len(set(orders)) != len(orders):
        raise ShowcaseError(f"duplicate order values: {sorted(orders)}")
    return scenes


def load(path: Path | None = None) -> tuple[ShowcaseScene, ...] | None:
    """The showcase, or ``None`` when there is not one.

    ``None`` keeps the existing Strong/Medium explorer as the whole demo, which
    is what shipped before the showcase existed.
    """

    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ShowcaseError(f"{path}: not valid JSON ({error})") from error
    return validate(raw)


def featured(scenes: Iterable[ShowcaseScene]) -> tuple[ShowcaseScene, ...]:
    chosen = [s for s in scenes if s.featured]
    return tuple(sorted(chosen, key=lambda s: (s.order is None, s.order or 0, s.showcase_id)))


def default_filter(scenes: Sequence[ShowcaseScene]) -> str:
    """``featured`` once anything is marked, otherwise ``all``."""

    return "featured" if any(s.featured for s in scenes) else "all"
