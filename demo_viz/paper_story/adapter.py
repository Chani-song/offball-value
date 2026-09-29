"""The one integration boundary between the research code and the demo.

Everything the paper story shows passes through :func:`build_scene`. When the
research implementation gains the player-evaluation quantities the abstract
describes, this module is the file that changes -- not the HTML, not the CSS,
not the drawing code, not the panel layout.

WHAT EXISTS TODAY, AND WHAT DOES NOT
------------------------------------
``demo_viz/PAPER_STORY_TRACE.md`` traced every abstract-facing quantity to
code. The equilibrium half is implemented (the exact 2v1 game: value,
certificate, mixed policies, modal line). The evaluation half -- relative rank,
similarity to the optimal action, regret against the observed action,
frame-by-frame scoring, clip aggregation -- is implemented nowhere, in either
repository.

So this module reserves those fields and fills them with :func:`~.schema.pending`.
It does not compute them, approximate them, or borrow a lookalike from another
lineage. ``dynamic_response_game.normalized_option_regret`` is not the
abstract's regret; ``markov.policy_transfer_bounds`` prices a whole fixed
policy against fresh best responses and requires an identical scenario, so it
is not a comparison of an observed action to an optimum either.

HOW THE FUTURE OUTPUT ARRIVES
-----------------------------
Through :class:`EvaluationSource`. Implement it against the updated research
code, point ``OFFBALL_EVALUATION_SOURCE`` at it, re-export, rebuild. The
reserved names in :data:`EVALUATION_METRICS` and :data:`FRAME_SERIES` are the
keys it returns; the adapter attaches provenance, the definition version and
the source string, and the interface renders whatever comes back without
knowing any of it. ``demo_viz/KYUHYEOK_UPDATE_INTEGRATION.md`` is the checklist.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .schema import (
    ActionRef,
    ClipSummary,
    ContractError,
    CounterfactualBlock,
    CounterfactualSide,
    EquilibriumRef,
    EvaluationBlock,
    Metric,
    PaperStoryScene,
    STORY_ROLES,
    Series,
    measured,
    pending,
)


# ---------------------------------------------------------------------------
# the reserved vocabulary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """A reserved name, its public label, and its unit -- but not its formula.

    ``note`` is what the interface shows while nothing produces the field. It
    says which of the two blockers applies: no method at all, or a method with
    no run covering this scene.
    """

    name: str
    label: str
    unit: str = ""
    note: str = ""


#: The observed-action evaluation fields the abstract names, in panel order.
#: Adding one here is the whole interface change: the panel renders the list.
EVALUATION_METRICS = (
    FieldSpec("observed_action_rank", "Observed action rank",
              note="Requires the observed action to be matched to an action set."),
    FieldSpec("relative_rank", "Relative rank",
              note="No relative-rank definition exists in the research code yet."),
    FieldSpec("similarity_to_optimal", "Similarity to optimal",
              note="No similarity metric exists in the research code yet."),
    FieldSpec("regret", "Regret",
              note="No observed-action regret exists in the research code yet. "
                   "The `regret` in dynamic_response_game compares defender "
                   "options in a different lineage and is not this quantity."),
)

#: The same quantities over time. Only series the payload carries are drawn.
FRAME_SERIES = (
    FieldSpec("relative_rank", "Relative rank"),
    FieldSpec("similarity_to_optimal", "Similarity to optimal"),
    FieldSpec("regret", "Regret"),
)

#: The two halves of the static/responsive comparison. ``semantics`` is left to
#: the producer: this repository has three candidate readings of "static" on
#: three different scales, and picking one is a research decision.
COUNTERFACTUAL_SIDES = (
    ("static", "Static counterfactual"),
    ("responsive", "Responsive counterfactual"),
)

#: What the story roles are called on screen. ``passer`` is the ball carrier,
#: which is not the demo's ``beneficiary`` -- that stays off-ball context.
ROLE_LABEL = {"runner": "Runner", "passer": "Passer", "defender": "Defender"}

_PENDING_SIDE = "Definition pending updated research implementation."
_PENDING_CLIP = ("Clip aggregation is not defined in the research code yet; "
                 "the aggregation method arrives with the metric.")


# ---------------------------------------------------------------------------
# the future source
# ---------------------------------------------------------------------------
@runtime_checkable
class EvaluationSource(Protocol):
    """What the updated research code will implement.

    Every method may return ``None`` for "not available here", and the adapter
    turns that into an explicit unavailable state rather than a gap.

    The returned mappings use the reserved names above as keys. A value entry
    is ``{"value": ..., "definition_version": ..., "aggregation": ..., "unit":
    ...}``; only ``value`` is required. The adapter never inspects the number.
    """

    #: What to show as the source of every value this object produces.
    name: str
    #: Bumped upstream whenever a definition changes, so a changed meaning is
    #: visible on screen instead of silently replacing the old one.
    definition_version: str

    def counterfactual(self, scene_id: str, role: str,
                       frame: int | None) -> Mapping[str, Any] | None: ...

    def evaluation(self, scene_id: str, role: str,
                   frame: int | None) -> Mapping[str, Any] | None: ...

    def frame_series(self, scene_id: str, role: str) -> Mapping[str, Any] | None: ...

    def clip_metrics(self, scene_id: str,
                     role: str) -> Sequence[Mapping[str, Any]] | None: ...


class NoEvaluationSource:
    """Today's source: there isn't one.

    Kept as a real object rather than a ``None`` check so the pending path and
    the future path run through exactly the same code, and the pending path is
    exercised by the same tests.
    """

    name = "none"
    definition_version = "unimplemented"

    def counterfactual(self, scene_id, role, frame):   # noqa: D102
        return None

    def evaluation(self, scene_id, role, frame):       # noqa: D102
        return None

    def frame_series(self, scene_id, role):            # noqa: D102
        return None

    def clip_metrics(self, scene_id, role):            # noqa: D102
        return None


def discover_source() -> EvaluationSource:
    """The evaluation source named by ``OFFBALL_EVALUATION_SOURCE``, or none.

    The value is ``"package.module:attribute"``. The attribute may be the
    source object or a zero-argument factory. An unimportable or malformed
    value raises: silently falling back to pending would make a broken
    integration look like an unimplemented one.
    """

    spec = os.environ.get("OFFBALL_EVALUATION_SOURCE", "").strip()
    if not spec:
        return NoEvaluationSource()
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ContractError(
            f"OFFBALL_EVALUATION_SOURCE must be 'module:attribute', got {spec!r}")
    module = importlib.import_module(module_name)
    found = getattr(module, attribute)
    # a class or a factory is called; an already-built instance is used as is
    source = found() if callable(found) else found
    for required in ("counterfactual", "evaluation", "frame_series", "clip_metrics"):
        if not callable(getattr(source, required, None)):
            raise ContractError(f"{spec}: evaluation source has no {required}()")
    if not getattr(source, "name", ""):
        raise ContractError(f"{spec}: evaluation source must name itself")
    return source


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------
def _entry(raw: Mapping[str, Any] | None, key: str) -> Mapping[str, Any] | None:
    if not raw:
        return None
    found = raw.get(key)
    if found is None:
        return None
    if not isinstance(found, Mapping):
        return {"value": found}
    return found


def _metric(spec: FieldSpec, raw: Mapping[str, Any] | None,
            source: EvaluationSource, *, provenance: str = "evaluation_pipeline",
            unavailable: str = "method_not_implemented") -> Metric:
    """A real metric when the source produced one, an explicit gap otherwise."""

    entry = _entry(raw, spec.name)
    if entry is None or entry.get("value") is None:
        return pending(spec.name, spec.label, unit=spec.unit,
                       availability=unavailable, detail=spec.note)
    return measured(
        spec.name, spec.label, entry["value"],
        source=str(entry.get("source") or source.name),
        unit=str(entry.get("unit", spec.unit)),
        provenance=str(entry.get("provenance", provenance)),
        definition_version=str(entry.get("definition_version")
                               or source.definition_version),
        aggregation=entry.get("aggregation"),
    )


def _action(name: str, label: str, raw: Mapping[str, Any] | None,
            source: EvaluationSource, *, note: str = "",
            unavailable: str = "method_not_implemented") -> ActionRef:
    entry = _entry(raw, name)
    if entry is None:
        return ActionRef(name=name, label=label, availability=unavailable,
                         detail=note)
    pair = lambda key: (tuple(float(v) for v in entry[key])      # noqa: E731
                        if entry.get(key) else None)
    return ActionRef(
        name=name, label=label, availability="available",
        action_id=entry.get("action_id"), kind=str(entry.get("kind", "")),
        description=str(entry.get("description", "")),
        target=pair("target"), vector=pair("vector"),
        projection_distance_m=entry.get("projection_distance_m"),
        confidence=entry.get("confidence"),
        provenance=str(entry.get("provenance", "evaluation_pipeline")),
        source=str(entry.get("source") or source.name),
        definition_version=str(entry.get("definition_version")
                               or source.definition_version),
    )


def _series(spec: FieldSpec, raw: Mapping[str, Any] | None,
            source: EvaluationSource) -> Series:
    entry = _entry(raw, spec.name)
    if entry is None or not entry.get("frames"):
        return Series(name=spec.name, label=spec.label, unit=spec.unit,
                      availability="method_not_implemented", detail=spec.note)
    domain = entry.get("domain")
    return Series(
        name=spec.name, label=spec.label, availability="available",
        unit=str(entry.get("unit", spec.unit)),
        provenance=str(entry.get("provenance", "evaluation_pipeline")),
        definition_version=str(entry.get("definition_version")
                               or source.definition_version),
        source=str(entry.get("source") or source.name),
        frames=tuple(int(f) for f in entry["frames"]),
        values=tuple(entry.get("values", ())),
        interpolate=bool(entry.get("interpolate", False)),
        domain=tuple(float(v) for v in domain) if domain else None,
    )


def _counterfactual(scene_id: str, role: str, frame: int | None,
                    source: EvaluationSource) -> CounterfactualBlock:
    raw = source.counterfactual(scene_id, role, frame)
    sides = []
    for key, label in COUNTERFACTUAL_SIDES:
        entry = _entry(raw, key) or {}
        semantics = str(entry.get("semantics", ""))
        best = _action(f"{key}_best_action", "Best action",
                       {f"{key}_best_action": entry.get("best_action")},
                       source, note=_PENDING_SIDE)
        value = _metric(FieldSpec(f"{key}_value", "Value", note=_PENDING_SIDE),
                        {f"{key}_value": entry.get("value")}, source)
        sides.append(CounterfactualSide(key=key, label=label, semantics=semantics,
                                        best_action=best, value=value))
    # Section 17: the solver's own release library is real code
    # (`solver_native.solver_release_targets`, DEFAULT_PASSES read from the
    # imported module). It is built from a solved state's receiver, and no
    # tracked scene has one -- so the slot exists and says exactly that,
    # rather than being filled with the demo's five exploratory rays.
    release_library = _metric(
        FieldSpec("solver_release_actions", "Solver release actions",
                  note="The solver's 18 releases are built from a solved "
                       "state's receiver; no solver run covers this scene. "
                       "The five exploratory rays are not a substitute."),
        raw, source, unavailable="artifact_missing")

    feasible_raw = _entry(raw, "feasible") or {}
    feasible = tuple(
        _action(f"feasible_{i}", str(item.get("label") or f"Alternative {i + 1}"),
                {f"feasible_{i}": item}, source)
        for i, item in enumerate(feasible_raw.get("actions", ()))
    )
    return CounterfactualBlock(
        observed_action=_action(
            "observed_action", "Observed action", raw, source,
            note="Projecting observed tracking onto an action set is not "
                 "defined in the research code yet."),
        feasible_actions=_metric(
            FieldSpec("feasible_actions", "Feasible alternatives",
                      note="The evaluation action set arrives with the "
                           "evaluation pipeline."),
            raw, source),
        feasible=feasible,
        release_library=release_library,
        static=sides[0], responsive=sides[1],
        value_change=_metric(
            FieldSpec("value_change", "Value change after response",
                      note=_PENDING_SIDE),
            raw, source),
    )


def _evaluation(scene_id: str, role: str, frame: int | None,
                source: EvaluationSource) -> EvaluationBlock:
    raw = source.evaluation(scene_id, role, frame)
    series_raw = source.frame_series(scene_id, role)
    return EvaluationBlock(
        metrics=tuple(_metric(spec, raw, source) for spec in EVALUATION_METRICS),
        optimal_action=_action(
            "optimal_action", "Optimal action", raw, source,
            note="No optimal-action definition exists in the research code yet."),
        frame_series=tuple(_series(spec, series_raw, source) for spec in FRAME_SERIES),
    )


def _clip_summary(scene_id: str, source: EvaluationSource) -> ClipSummary:
    roles: dict[str, tuple[Metric, ...]] = {}
    for role in STORY_ROLES:
        produced = source.clip_metrics(scene_id, role)
        if not produced:
            roles[role] = (pending("clip_summary", "Clip aggregate",
                                   detail=_PENDING_CLIP),)
            continue
        metrics = []
        for item in produced:
            spec = FieldSpec(str(item.get("name", "")), str(item.get("label", "")),
                             unit=str(item.get("unit", "")))
            metrics.append(_metric(spec, {spec.name: item}, source))
        roles[role] = tuple(metrics)
    return ClipSummary(roles=roles)


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------
def build_scene(
    scene_id: str,
    *,
    source: EvaluationSource | None = None,
    equilibrium: EquilibriumRef | None = None,
    action_value: Metric | None = None,
    story_title: str | None = None,
    story_summary: str | None = None,
    frame: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> PaperStoryScene:
    """Everything one scene offers the paper story, in the stable shape.

    ``source`` defaults to whatever ``OFFBALL_EVALUATION_SOURCE`` names, which
    today is nothing -- so every evaluation field comes back as an explicit
    unavailable state carrying the reason.
    """

    source = source or discover_source()
    return PaperStoryScene(
        scene_id=scene_id,
        story_title=story_title,
        story_summary=story_summary,
        counterfactual={role: _counterfactual(scene_id, role, frame, source)
                        for role in STORY_ROLES},
        evaluation={role: _evaluation(scene_id, role, frame, source)
                    for role in STORY_ROLES},
        clip_summary=_clip_summary(scene_id, source),
        equilibrium=equilibrium or EquilibriumRef(),
        action_value=action_value,
        provenance={"evaluation_source": source.name,
                    "definition_version": source.definition_version,
                    **dict(provenance or {})},
    )


def equilibrium_for(scene_id: str, solver_dir: Path) -> EquilibriumRef:
    """Whether a real exported solver state covers this scene.

    Presence of the exported file is the whole test. Nothing is matched by
    name similarity: a solver run is attached to a scene only when something
    upstream said so and wrote the file under that key.
    """

    key = "".join(c if c.isalnum() else "_" for c in scene_id)
    path = Path(solver_dir) / f"{key}.json"
    if path.exists():
        return EquilibriumRef(availability="available", key=key)
    return EquilibriumRef(
        availability="artifact_missing",
        detail="No solver run covers this scene. The exact 2v1 game is a "
               "batch job; its artifacts are not produced by this demo.")
