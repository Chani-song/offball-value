"""The paper-story contract and its one adapter.

``schema`` defines the shape the browser reads. ``adapter`` is the only place
that knows anything about the research code's own output layout.
"""

from .adapter import (
    EVALUATION_METRICS,
    FRAME_SERIES,
    EvaluationSource,
    NoEvaluationSource,
    build_scene,
    discover_source,
    equilibrium_for,
)
from .schema import (
    AVAILABILITY,
    AVAILABILITY_LABEL,
    MODES,
    PROVENANCE,
    SCHEMA_VERSION,
    STORY_ROLES,
    ActionRef,
    ContractError,
    Metric,
    PaperStoryScene,
    Series,
    contract_payload,
    measured,
    pending,
    validate_payload,
)

__all__ = [
    "AVAILABILITY", "AVAILABILITY_LABEL", "MODES", "PROVENANCE",
    "SCHEMA_VERSION", "STORY_ROLES", "ActionRef", "ContractError",
    "EVALUATION_METRICS", "EvaluationSource", "FRAME_SERIES", "Metric",
    "NoEvaluationSource", "PaperStoryScene", "Series", "build_scene",
    "contract_payload", "discover_source", "equilibrium_for", "measured",
    "pending", "validate_payload",
]
