"""The stable paper-story contract: what a scene offers the reviewer.

This module is the boundary between the research code and the interface. The
browser reads *this* shape, never a solver run directory, a ``rows.jsonl``
layout or an upstream variable name. When the research implementation changes,
:mod:`demo_viz.paper_story.adapter` changes and this shape does not.

THE ONE INVARIANT
-----------------
A number is on screen only if the research code produced it.

:class:`Metric` enforces that at construction:

    availability == "available"   =>  value is not None  and  source is set
    availability != "available"   =>  value is None

So there is no way to build a record that renders a plausible-looking number
without a computation behind it. A field whose method does not exist yet is a
first-class value -- ``method_not_implemented`` -- not a zero, a dash or a
placeholder digit.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No formula. This module reserves the *names* the abstract uses -- relative
rank, similarity to optimal, regret, static and responsive counterfactual
values, frame and clip aggregates -- and says nothing about how they are
computed, because as of this writing nothing computes them (see
``demo_viz/PAPER_STORY_TRACE.md``). Their definitions arrive with the upstream
output, carried per metric in ``definition_version`` so a later change to a
definition is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: Bumped when the shape changes in a way a consumer must notice.
SCHEMA_VERSION = "paper-story/1"

#: Why a paper-facing value is or is not on screen. Kept here, in one place,
#: rather than as strings scattered through the JavaScript.
AVAILABILITY = (
    "available",                # the research code produced it
    "method_not_implemented",   # nothing computes this quantity yet
    "artifact_missing",         # the method exists; no run covers this scene
    "scene_not_supported",      # this scene cannot carry this quantity
    "mapping_unresolved",       # the scene is not matched to a tracked clip
)

#: What the interface says for each state. The browser looks these up; it does
#: not spell them. Every one of them is a sentence, not a symbol -- "--" and
#: "N/A" collapse five different reasons into one shrug.
AVAILABILITY_LABEL = {
    "available": "",
    "method_not_implemented": "Awaiting updated evaluation pipeline",
    "artifact_missing": "Not computed for this scene",
    "scene_not_supported": "Not defined for this scene",
    "mapping_unresolved": "Scene mapping unresolved",
}

#: Where a value came from. Distinct from availability: a value can be present
#: and still be a reviewer's opinion rather than a model output.
PROVENANCE = (
    "human_reviewed",        # a person's label or rating
    "observed_tracking",     # read straight off the tracking
    "derived_current_demo",  # this repository computed it
    "solver",                # the exact game solver's own output
    "evaluation_pipeline",   # the player-evaluation pipeline
    "future_pipeline",       # reserved: nothing produces this yet
)

#: The three bodies the paper story talks about. ``passer`` is the ball
#: carrier. It is **not** the demo's ``beneficiary``, which is the attacker
#: whose space the run opens -- that stays an off-ball-context role.
STORY_ROLES = ("runner", "passer", "defender")

#: The four story modes, in the order the interface presents them.
MODES = ("observed", "counterfactual", "evaluation", "game_solution")


class ContractError(ValueError):
    """A record that does not satisfy the contract."""


def _one_of(value: Any, allowed: Sequence[str], where: str) -> str:
    if value not in allowed:
        raise ContractError(f"{where}: must be one of {tuple(allowed)}, got {value!r}")
    return str(value)


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Metric:
    """One paper-facing number, or an explicit reason there isn't one.

    ``definition_version`` and ``source`` exist so that a metric whose
    definition is still being settled upstream can change without the demo
    quietly showing the old meaning under the old label.
    """

    name: str
    label: str
    availability: str = "method_not_implemented"
    value: float | int | str | bool | None = None
    unit: str = ""
    provenance: str = "future_pipeline"
    definition_version: str | None = None
    source: str | None = None
    aggregation: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        where = f"metric {self.name!r}"
        if not self.name or not str(self.name).strip():
            raise ContractError("metric: name must be a non-empty string")
        if not self.label or not str(self.label).strip():
            raise ContractError(f"{where}: label must be a non-empty string")
        _one_of(self.availability, AVAILABILITY, where)
        _one_of(self.provenance, PROVENANCE, where)
        if self.availability == "available":
            if self.value is None:
                raise ContractError(f"{where}: available but carries no value")
            if not self.source:
                raise ContractError(f"{where}: available but names no source")
        elif self.value is not None:
            raise ContractError(
                f"{where}: {self.availability} but carries a value {self.value!r}; "
                "an unavailable metric must not show a number")

    @property
    def message(self) -> str:
        """What to show when there is no value."""

        return self.detail or AVAILABILITY_LABEL[self.availability]

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "availability": self.availability,
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "definition_version": self.definition_version,
            "source": self.source,
            "aggregation": self.aggregation,
            "detail": self.detail,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "Metric":
        if not isinstance(raw, dict):
            raise ContractError("metric: must be an object")
        unknown = set(raw) - {
            "name", "label", "availability", "value", "unit", "provenance",
            "definition_version", "source", "aggregation", "detail"}
        if unknown:
            raise ContractError(f"metric: unknown keys {sorted(unknown)}")
        return cls(
            name=str(raw.get("name", "")),
            label=str(raw.get("label", "")),
            availability=raw.get("availability", "method_not_implemented"),
            value=raw.get("value"),
            unit=str(raw.get("unit", "")),
            provenance=raw.get("provenance", "future_pipeline"),
            definition_version=raw.get("definition_version"),
            source=raw.get("source"),
            aggregation=raw.get("aggregation"),
            detail=str(raw.get("detail", "")),
        )


def pending(name: str, label: str, *, unit: str = "",
            availability: str = "method_not_implemented",
            provenance: str = "future_pipeline", detail: str = "") -> Metric:
    """A reserved slot. The interface renders its reason, never a number."""

    return Metric(name=name, label=label, availability=availability, unit=unit,
                  provenance=provenance, detail=detail)


def measured(name: str, label: str, value: float | int | str | bool, *,
             source: str, unit: str = "", provenance: str = "derived_current_demo",
             definition_version: str | None = None, aggregation: str | None = None,
             detail: str = "") -> Metric:
    """A real value. ``source`` is required, and says what produced it."""

    return Metric(name=name, label=label, availability="available", value=value,
                  unit=unit, provenance=provenance, source=source,
                  definition_version=definition_version, aggregation=aggregation,
                  detail=detail)


# ---------------------------------------------------------------------------
# series: one metric over frames
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Series:
    """A metric evaluated at some frames of a clip.

    ``frames`` are scene frame indices, not times, so the timeline lines up
    with playback without a unit conversion in the browser. Evaluation may be
    sparse -- three solved frames in a 250-frame clip is a normal output -- and
    ``interpolate`` says whether joining them is meaningful. It defaults to
    False: a line drawn between two computed points implies values that were
    never computed, and only the producer knows if that is fair.
    """

    name: str
    label: str
    availability: str = "method_not_implemented"
    unit: str = ""
    provenance: str = "future_pipeline"
    definition_version: str | None = None
    source: str | None = None
    detail: str = ""
    frames: tuple[int, ...] = ()
    values: tuple[float | None, ...] = ()
    interpolate: bool = False
    domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        where = f"series {self.name!r}"
        if not self.name or not self.label:
            raise ContractError(f"{where}: name and label are required")
        _one_of(self.availability, AVAILABILITY, where)
        _one_of(self.provenance, PROVENANCE, where)
        if len(self.frames) != len(self.values):
            raise ContractError(f"{where}: {len(self.frames)} frames, "
                                f"{len(self.values)} values")
        if self.availability == "available":
            if not self.frames:
                raise ContractError(f"{where}: available but carries no samples")
            if not self.source:
                raise ContractError(f"{where}: available but names no source")
            if list(self.frames) != sorted(set(self.frames)):
                raise ContractError(f"{where}: frames must be sorted and unique")
            if any(not isinstance(f, int) or isinstance(f, bool) or f < 0
                   for f in self.frames):
                raise ContractError(f"{where}: frames must be non-negative integers")
        elif self.frames:
            raise ContractError(
                f"{where}: {self.availability} but carries {len(self.frames)} samples")

    @property
    def message(self) -> str:
        return self.detail or AVAILABILITY_LABEL[self.availability]

    def to_payload(self) -> dict:
        return {
            "name": self.name, "label": self.label,
            "availability": self.availability, "unit": self.unit,
            "provenance": self.provenance,
            "definition_version": self.definition_version,
            "source": self.source, "detail": self.detail,
            "frames": list(self.frames), "values": list(self.values),
            "interpolate": self.interpolate,
            "domain": list(self.domain) if self.domain else None,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "Series":
        if not isinstance(raw, dict):
            raise ContractError("series: must be an object")
        domain = raw.get("domain")
        return cls(
            name=str(raw.get("name", "")), label=str(raw.get("label", "")),
            availability=raw.get("availability", "method_not_implemented"),
            unit=str(raw.get("unit", "")),
            provenance=raw.get("provenance", "future_pipeline"),
            definition_version=raw.get("definition_version"),
            source=raw.get("source"), detail=str(raw.get("detail", "")),
            frames=tuple(int(f) for f in raw.get("frames", ())),
            values=tuple(raw.get("values", ())),
            interpolate=bool(raw.get("interpolate", False)),
            domain=tuple(float(v) for v in domain) if domain else None,
        )


# ---------------------------------------------------------------------------
# action
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionRef:
    """One action, however the producer chooses to represent it.

    Future evaluation may project continuous tracking onto a discrete action
    library. When it does, ``projection_distance_m`` and ``confidence`` carry
    how far that projection reached, so the interface can show an observed
    action *and* how well it was matched -- rather than presenting a snapped
    label as if it were what the player did.

    ``kind`` is the producer's own word. This module does not enumerate the
    action library, because the library belongs upstream.
    """

    name: str
    label: str
    availability: str = "method_not_implemented"
    action_id: str | None = None
    kind: str = ""
    description: str = ""
    #: centre-origin metres, the demo's frame
    target: tuple[float, float] | None = None
    vector: tuple[float, float] | None = None
    projection_distance_m: float | None = None
    confidence: float | None = None
    provenance: str = "future_pipeline"
    source: str | None = None
    definition_version: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        where = f"action {self.name!r}"
        if not self.name or not self.label:
            raise ContractError(f"{where}: name and label are required")
        _one_of(self.availability, AVAILABILITY, where)
        _one_of(self.provenance, PROVENANCE, where)
        if self.availability == "available":
            if not self.source:
                raise ContractError(f"{where}: available but names no source")
            if not (self.action_id or self.description or self.target or self.vector):
                raise ContractError(f"{where}: available but describes no action")
        else:
            for blocked in ("action_id", "description", "target", "vector",
                            "projection_distance_m", "confidence"):
                if getattr(self, blocked):
                    raise ContractError(
                        f"{where}: {self.availability} but carries {blocked}")

    @property
    def message(self) -> str:
        return self.detail or AVAILABILITY_LABEL[self.availability]

    def to_payload(self) -> dict:
        return {
            "name": self.name, "label": self.label,
            "availability": self.availability, "action_id": self.action_id,
            "kind": self.kind, "description": self.description,
            "target": list(self.target) if self.target else None,
            "vector": list(self.vector) if self.vector else None,
            "projection_distance_m": self.projection_distance_m,
            "confidence": self.confidence, "provenance": self.provenance,
            "source": self.source, "definition_version": self.definition_version,
            "detail": self.detail,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "ActionRef":
        if not isinstance(raw, dict):
            raise ContractError("action: must be an object")
        pair = lambda key: (tuple(float(v) for v in raw[key])   # noqa: E731
                            if raw.get(key) else None)
        return cls(
            name=str(raw.get("name", "")), label=str(raw.get("label", "")),
            availability=raw.get("availability", "method_not_implemented"),
            action_id=raw.get("action_id"), kind=str(raw.get("kind", "")),
            description=str(raw.get("description", "")),
            target=pair("target"), vector=pair("vector"),
            projection_distance_m=raw.get("projection_distance_m"),
            confidence=raw.get("confidence"),
            provenance=raw.get("provenance", "future_pipeline"),
            source=raw.get("source"),
            definition_version=raw.get("definition_version"),
            detail=str(raw.get("detail", "")),
        )


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CounterfactualSide:
    """One half of the static/responsive comparison.

    ``semantics`` is the producer's own description of what is held fixed. It
    is empty until the producer supplies one: this repository has three
    candidate readings of "static" on three different scales
    (``PAPER_STORY_TRACE.md`` section 2) and choosing between them is a
    research decision, not an interface default.
    """

    key: str
    label: str
    semantics: str = ""
    best_action: ActionRef | None = None
    value: Metric | None = None

    def to_payload(self) -> dict:
        return {
            "key": self.key, "label": self.label, "semantics": self.semantics,
            "best_action": self.best_action.to_payload() if self.best_action else None,
            "value": self.value.to_payload() if self.value else None,
        }


@dataclass(frozen=True)
class CounterfactualBlock:
    observed_action: ActionRef
    feasible_actions: Metric
    #: the actions themselves, when a producer supplies them
    feasible: tuple[ActionRef, ...] = ()
    #: the solver's own release action library, when a solved state exists
    release_library: Metric | None = None
    static: CounterfactualSide | None = None
    responsive: CounterfactualSide | None = None
    value_change: Metric | None = None

    def to_payload(self) -> dict:
        return {
            "observed_action": self.observed_action.to_payload(),
            "feasible_actions": self.feasible_actions.to_payload(),
            "feasible": [a.to_payload() for a in self.feasible],
            "release_library": (self.release_library.to_payload()
                                if self.release_library else None),
            "static": self.static.to_payload() if self.static else None,
            "responsive": self.responsive.to_payload() if self.responsive else None,
            "value_change": self.value_change.to_payload() if self.value_change else None,
        }


@dataclass(frozen=True)
class EvaluationBlock:
    """How good the observed action was, once something can say.

    Every field is reserved, none is defined here. ``metrics`` is a list rather
    than named attributes so that the producer can add one without a change to
    this module or to the interface.
    """

    metrics: tuple[Metric, ...] = ()
    optimal_action: ActionRef | None = None
    frame_series: tuple[Series, ...] = ()

    def to_payload(self) -> dict:
        return {
            "metrics": [m.to_payload() for m in self.metrics],
            "optimal_action": self.optimal_action.to_payload() if self.optimal_action else None,
            "frame_series": [s.to_payload() for s in self.frame_series],
        }


@dataclass(frozen=True)
class ClipSummary:
    """Clip-level aggregates, per story role.

    The aggregation is the producer's: each metric carries its own
    ``aggregation`` label ("mean over evaluated frames", "count", whatever it
    turns out to be). Nothing here averages anything, so a later change to the
    aggregation is a change upstream and a re-export, not an interface rewrite.
    """

    roles: dict[str, tuple[Metric, ...]] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {role: [m.to_payload() for m in metrics]
                for role, metrics in self.roles.items()}


@dataclass(frozen=True)
class EquilibriumRef:
    """Where the game solution for this scene lives, if anywhere.

    The payload itself is not inlined: solver files are lazy-loaded one at a
    time and are much larger than the rest of the story. ``key`` names the
    exported file; the browser fetches it only in Game solution mode.
    """

    availability: str = "artifact_missing"
    key: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        _one_of(self.availability, AVAILABILITY, "equilibrium")
        if self.availability == "available" and not self.key:
            raise ContractError("equilibrium: available but names no artifact key")

    @property
    def message(self) -> str:
        return self.detail or AVAILABILITY_LABEL[self.availability]

    def to_payload(self) -> dict:
        return {"availability": self.availability, "key": self.key,
                "detail": self.detail}


@dataclass(frozen=True)
class PaperStoryScene:
    """Everything one scene offers the paper story."""

    scene_id: str
    schema: str = SCHEMA_VERSION
    story_title: str | None = None
    story_summary: str | None = None
    roles: tuple[str, ...] = STORY_ROLES
    #: role -> block, because the counterfactual question is asked of a player
    counterfactual: dict[str, CounterfactualBlock] = field(default_factory=dict)
    evaluation: dict[str, EvaluationBlock] = field(default_factory=dict)
    clip_summary: ClipSummary = field(default_factory=ClipSummary)
    equilibrium: EquilibriumRef = field(default_factory=EquilibriumRef)
    action_value: Metric | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "story_title": self.story_title,
            "story_summary": self.story_summary,
            "roles": list(self.roles),
            "counterfactual": {r: b.to_payload() for r, b in self.counterfactual.items()},
            "evaluation": {r: b.to_payload() for r, b in self.evaluation.items()},
            "clip_summary": self.clip_summary.to_payload(),
            "equilibrium": self.equilibrium.to_payload(),
            "action_value": self.action_value.to_payload() if self.action_value else None,
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# validation of an external payload
# ---------------------------------------------------------------------------
def validate_payload(raw: Any) -> dict:
    """Check a story payload, whoever wrote it, and return it unchanged.

    Every metric, series and action is round-tripped through its dataclass, so
    the invariant -- no value without availability and a source -- is checked
    on data arriving from outside this process, not only on data this process
    built.
    """

    if not isinstance(raw, dict):
        raise ContractError("story payload must be an object")
    schema = raw.get("schema")
    if schema != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema {schema!r}, expected {SCHEMA_VERSION!r}")
    if not raw.get("scene_id"):
        raise ContractError("story payload has no scene_id")

    for role, block in (raw.get("counterfactual") or {}).items():
        where = f"counterfactual[{role}]"
        if not isinstance(block, dict):
            raise ContractError(f"{where}: must be an object")
        ActionRef.from_payload(block.get("observed_action") or {"name": "observed_action",
                                                               "label": "Observed action"})
        Metric.from_payload(block.get("feasible_actions") or {"name": "feasible_actions",
                                                              "label": "Feasible alternatives"})
        for action in block.get("feasible") or ():
            ActionRef.from_payload(action)
        if block.get("release_library"):
            Metric.from_payload(block["release_library"])
        for side in ("static", "responsive"):
            part = block.get(side)
            if part is None:
                continue
            if not isinstance(part, dict):
                raise ContractError(f"{where}.{side}: must be an object")
            if part.get("best_action"):
                ActionRef.from_payload(part["best_action"])
            if part.get("value"):
                Metric.from_payload(part["value"])
        if block.get("value_change"):
            Metric.from_payload(block["value_change"])

    for role, block in (raw.get("evaluation") or {}).items():
        if not isinstance(block, dict):
            raise ContractError(f"evaluation[{role}]: must be an object")
        for metric in block.get("metrics") or ():
            Metric.from_payload(metric)
        if block.get("optimal_action"):
            ActionRef.from_payload(block["optimal_action"])
        for series in block.get("frame_series") or ():
            Series.from_payload(series)

    for role, metrics in (raw.get("clip_summary") or {}).items():
        if not isinstance(metrics, list):
            raise ContractError(f"clip_summary[{role}]: must be a list")
        for metric in metrics:
            Metric.from_payload(metric)

    equilibrium = raw.get("equilibrium") or {}
    EquilibriumRef(availability=equilibrium.get("availability", "artifact_missing"),
                   key=equilibrium.get("key"), detail=str(equilibrium.get("detail", "")))
    if raw.get("action_value"):
        Metric.from_payload(raw["action_value"])
    return raw


def contract_payload() -> dict:
    """The vocabulary, shipped once so the browser never spells it itself."""

    return {
        "schema": SCHEMA_VERSION,
        "availability": list(AVAILABILITY),
        "availability_label": dict(AVAILABILITY_LABEL),
        "provenance": list(PROVENANCE),
        "roles": list(STORY_ROLES),
        "modes": list(MODES),
    }
