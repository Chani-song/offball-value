"""Read the curated dilemma showcase and map it onto repository scenes.

The source (``local_inputs/dilemma_showcase.html``) is local-only and carries
embedded tracking. Nothing here copies a trajectory: it reads curation metadata
-- reviewer ratings, human role labels, match and event descriptions, timing
convention, solver scenario type -- and then tries to identify which scene
already in ``web_data/`` the entry refers to. Trajectories come from that
existing scene, or the entry stays unresolved.

Mapping is evidence-based and refuses to guess. A candidate has to agree on:

  * the pair of teams, unordered -- the repository titles a scene
    "attacking vs defending", which flips between scenes of the same fixture;
  * the half;
  * the human role labels, where the showcase names one player per role and
    the repository annotation may list several, so containment is the test;

and then either the shooter's name or exact role equality has to confirm it.
Anything weaker is reported as ambiguous or unresolved, never mapped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_SOURCE = (Path(__file__).resolve().parent.parent.parent
                  / "local_inputs" / "dilemma_showcase.html")
WEB_DATA = Path(__file__).resolve().parent.parent / "web_data"

#: The source is Korean. These are the only words this module interprets.
HALF_FIRST, HALF_SECOND = "전반", "후반"
ZERO_SHOT, ZERO_ONSET = "슛", "러너 출발"
PROVENANCE = {"찬의": "human", "솔버": "solver"}
SCENARIO = {"2대1": "2v1", "3대1": "3v1"}

#: Shot outcomes, for the English event line shown publicly.
OUTCOME = {
    "골": "goal", "선방": "saved", "빗나감": "off target",
    "막힘": "blocked", "골대": "woodwork", "기록 없음": "not recorded",
}


@dataclass
class ShowcaseEntry:
    showcase_id: str
    kind: str                       # strong | medium | solver, the source's own
    provenance: str                 # human | solver
    scenario_type: str | None       # 2v1 | 3v1, solver entries only
    zero: str                       # shot | run_onset
    match: str
    half: str                       # first | second
    teams: tuple[str, str]
    event: str
    goal: bool
    annotation: str
    raters: tuple[dict[str, Any], ...]
    roles: dict[str, tuple[str, ...]]      # shirt numbers, as labelled by hand
    role_names: dict[str, str]
    split: str
    scene_id: str | None = None
    mapping_status: str = "unresolved"
    mapping_evidence: str = ""
    candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ratings(self) -> tuple[int, ...]:
        out = []
        for rater in self.raters:
            try:
                out.append(int(rater.get("rating")))
            except (TypeError, ValueError):
                continue
        return tuple(out)

    @property
    def agreement_label(self) -> str:
        """e.g. "5/5". Human reviewer ratings, not a model score."""

        return "/".join(str(r) for r in self.ratings) if self.ratings else ""


def _shirts(text: str | None) -> tuple[str, ...]:
    return tuple(re.findall(r"#(\d+)", text or ""))


def _teams(match: str) -> tuple[str, str]:
    head = match.split("·")[0].strip()
    parts = [p.strip() for p in head.split(" vs ")]
    return (parts[0], parts[1]) if len(parts) == 2 else (head, "")


def read_showcase(path: Path | None = None) -> list[ShowcaseEntry]:
    """Parse the local showcase file into curation records, without tracks."""

    path = Path(path) if path is not None else DEFAULT_SOURCE
    raw = path.read_text()
    head = "const SCENES = "
    start = raw.index(head) + len(head)
    end = raw.index("\nconst GROUPS", start)
    scenes = json.loads(raw[start:end].rstrip().rstrip(";"))

    entries = []
    for scene in scenes:
        info = scene.get("info", {})
        match = info.get("match", "")
        label = scene.get("label", "")
        source = label.split("·")[0].strip()
        scenario = None
        if len(label.split("·")) > 1:
            scenario = SCENARIO.get(label.split("·")[1].strip())
        outcome = (info.get("shot") or "").split("·")[0].strip()
        shooter = (info.get("shot") or "").split("·")[-1].strip() if "·" in (info.get("shot") or "") else ""
        entries.append(ShowcaseEntry(
            showcase_id=scene["code"],
            kind=scene.get("kind", ""),
            provenance=PROVENANCE.get(source, "human"),
            scenario_type=scenario,
            zero=ZERO_SHOT if scene.get("zero") == ZERO_SHOT else "run_onset",
            match=match,
            half="second" if HALF_SECOND in match else "first",
            teams=_teams(match),
            event=" · ".join(p for p in (OUTCOME.get(outcome, outcome), shooter) if p),
            goal=bool(info.get("goal")),
            annotation=info.get("annotation", "") or "",
            raters=tuple(info.get("raters", ())),
            roles={
                "runners": _shirts(info.get("runner")),
                "defenders": _shirts(info.get("defender")),
                "beneficiaries": _shirts(info.get("beneficiary")),
            },
            role_names={
                "runners": info.get("runner", ""),
                "defenders": info.get("defender", ""),
                "beneficiaries": info.get("beneficiary", ""),
            },
            split=info.get("split", "") or "",
        ))
    return entries


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------
def repository_scenes(web_data: Path | None = None) -> list[dict]:
    """The scenes the demo already ships, with what mapping needs."""

    web_data = Path(web_data) if web_data is not None else WEB_DATA
    index_file = web_data / "index.json"
    if not index_file.exists():
        return []
    out = []
    for row in json.loads(index_file.read_text())["scenes"]:
        payload = json.loads((web_data / row["file"]).read_text())
        by_id = {p["id"]: p for p in payload["players"]}
        out.append({
            "scene_id": payload["scene_id"],
            "title": payload["title"],
            "teams": _teams(payload["title"]),
            "subtitle": payload.get("subtitle", ""),
            "effect": payload.get("effect", ""),
            "half": "second" if "_P2_" in payload["scene_id"] else "first",
            "roles": {
                key: tuple(sorted(by_id[i]["shirt"] for i in payload["roles"][role]
                                  if i in by_id))
                for key, role in (("runners", "runner"), ("defenders", "defender"),
                                  ("beneficiaries", "beneficiary"))
            },
        })
    return out


def _shooter(subtitle: str) -> str:
    parts = [p.strip() for p in subtitle.split("·")]
    return parts[2] if len(parts) > 2 else ""


def map_entry(entry: ShowcaseEntry, scenes: Sequence[dict]) -> ShowcaseEntry:
    """Attach a repository scene to one showcase entry, or leave it unresolved."""

    pair = frozenset(t for t in entry.teams if t)
    shooter = entry.event.split("·")[-1].strip()

    scored = []
    for scene in scenes:
        if frozenset(t for t in scene["teams"] if t) != pair:
            continue
        if scene["half"] != entry.half:
            continue
        contained = all(
            set(entry.roles[key]) <= set(scene["roles"][key]) and entry.roles[key]
            for key in ("runners", "defenders", "beneficiaries")
        )
        if not contained:
            continue
        exact = all(tuple(sorted(entry.roles[key])) == scene["roles"][key]
                    for key in ("runners", "defenders", "beneficiaries"))
        shooter_ok = bool(shooter) and shooter.lower() in _shooter(scene["subtitle"]).lower()
        scored.append((exact, shooter_ok, scene["scene_id"]))

    if not scored:
        entry.mapping_status = "unresolved"
        entry.mapping_evidence = (
            "no repository scene shares the team pair, half and hand-labelled roles"
        )
        return entry

    confirmed = [s for s in scored if s[0] or s[1]]
    if len(confirmed) == 1:
        exact, shooter_ok, scene_id = confirmed[0]
        entry.scene_id = scene_id
        entry.mapping_status = "verified"
        why = []
        why.append("teams and half agree")
        why.append("roles identical" if exact else "hand-labelled roles are a subset of the annotation")
        if shooter_ok:
            why.append(f"shooter '{shooter}' matches")
        entry.mapping_evidence = "; ".join(why)
        entry.candidates = (scene_id,)
        return entry

    entry.candidates = tuple(s[2] for s in scored)
    entry.mapping_status = "ambiguous"
    entry.mapping_evidence = (
        f"{len(scored)} repository scenes fit teams, half and roles"
        + (f"; {len(confirmed)} also confirmed by shooter or exact roles" if confirmed else "")
    )
    return entry


def map_all(entries: Iterable[ShowcaseEntry],
            scenes: Sequence[dict] | None = None) -> list[ShowcaseEntry]:
    scenes = repository_scenes() if scenes is None else scenes
    return [map_entry(entry, scenes) for entry in entries]
