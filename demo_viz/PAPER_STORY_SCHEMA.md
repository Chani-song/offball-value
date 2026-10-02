# The paper-story payload

`paper-story/1`. Defined by [`paper_story/schema.py`](paper_story/schema.py),
produced by [`paper_story/adapter.py`](paper_story/adapter.py), written by
[`web/export_paper_story.py`](web/export_paper_story.py), read by
[`web/site/js/story.js`](web/site/js/story.js).

Two files per build:

| path | when | what |
| --- | --- | --- |
| `data/story/contract.json` | once, with the shell | the vocabulary: availability states and their wording, provenance kinds, roles, modes |
| `data/story/<scene>.json` | lazily, with the scene | that scene's story |

Scene files are ~16 KB each today and load one at a time. Frame-level arrays
land in the scene file, so a clip's evaluation never enters the initial bundle.

---

## The invariant

Everything else is convenience. This is the contract:

```
availability == "available"   =>   value is not None  and  source is set
availability != "available"   =>   value is None
```

Enforced in `Metric.__post_init__`, `Series.__post_init__` and
`ActionRef.__post_init__`, so it holds for records this repository builds; and
re-checked by `validate_payload` for records arriving from anywhere else.

**There is no way to construct a record that shows a number nothing computed.**
A placeholder raises rather than renders.

---

## Availability

Five states, never collapsed into one dash. `contract.json` ships the wording,
so the browser never spells a reason itself.

| state | means | shown as |
| --- | --- | --- |
| `available` | the research code produced it | the value |
| `method_not_implemented` | nothing computes this quantity yet | Awaiting updated evaluation pipeline |
| `artifact_missing` | the method exists; no run covers this scene | Not computed for this scene |
| `scene_not_supported` | this scene cannot carry this quantity | Not defined for this scene |
| `mapping_unresolved` | the scene is not matched to a tracked clip | Scene mapping unresolved |

A record may override the wording with its own `detail`, which is how the
`regret` slot says *which* regret it is not.

## Provenance

Orthogonal to availability: a value can be present and still be a person's
opinion. `human_reviewed`, `observed_tracking`, `derived_current_demo`,
`solver`, `evaluation_pipeline`, `future_pipeline`.

---

## Shape

```jsonc
{
  "schema": "paper-story/1",
  "scene_id": "J03WOY:shot_011_P2_0903",
  "story_title": null,          // curated, optional; falls back to match/event
  "story_summary": null,
  "roles": ["runner", "passer", "defender"],

  "counterfactual": {           // keyed by story role
    "<role>": {
      "observed_action":  ActionRef,
      "feasible_actions": Metric,      // how many alternatives
      "feasible":        [ActionRef],  // the alternatives themselves
      "static":     { "key", "label", "semantics", "best_action": ActionRef,
                      "value": Metric },
      "responsive": { ... same ... },
      "value_change": Metric           // only if the two share a scale
    }
  },

  "evaluation": {               // keyed by story role
    "<role>": {
      "metrics": [Metric],      // observed_action_rank, relative_rank,
                                // similarity_to_optimal, regret
      "optimal_action": ActionRef,
      "frame_series": [Series]  // the same quantities over frames
    }
  },

  "clip_summary": { "<role>": [Metric] },
  "equilibrium": { "availability", "key", "detail" },
  "action_value": Metric,
  "provenance": { "evaluation_source", "definition_version" }
}
```

### Metric

```jsonc
{ "name", "label", "availability", "value", "unit", "provenance",
  "definition_version", "source", "aggregation", "detail" }
```

`aggregation` is the producer's own words ("mean over evaluated frames"). The
interface prints it and assumes nothing: no mean, median or percentile is
implemented anywhere in the UI.

### Series

```jsonc
{ ...Metric fields..., "frames": [int], "values": [float|null],
  "interpolate": false, "domain": [low, high] | null }
```

`frames` are **this repository's scene frame indices** — not times, not
upstream row numbers — so the strip and the scrubber share one clock. They must
be sorted and unique; the validator rejects otherwise, because an unsorted
series would silently draw a sample at the wrong instant.

`interpolate` defaults to **false**. Sparse evaluation is the expected case: a
handful of solved frames in a 250-frame clip. With it false the strip draws a
stem per sample; joining them into a line asserts values nobody computed, and
only the producer knows whether that is fair.

`domain` gives the metric's natural range. Without one the lane auto-scales to
the samples present, which is honest but not comparable across scenes.

### ActionRef

```jsonc
{ "name", "label", "availability", "action_id", "kind", "description",
  "target": [x, y], "vector": [dx, dy], "projection_distance_m", "confidence",
  "provenance", "source", "definition_version", "detail" }
```

Coordinates are centre-origin metres, this repository's frame. The adapter
converts; the browser does not.

`projection_distance_m` and `confidence` exist because future evaluation is
likely to project continuous tracking onto a discrete action library. When it
does, the interface shows the matched action *and* how far the match reached,
so a snapped label never reads as what the player actually did.

This module deliberately does **not** enumerate the action library. That
belongs upstream, and `kind` is whatever word the producer uses.

---

## What the schema does not define

No formula, for any quantity. The abstract's names are reserved here;
their meanings arrive with the output, carried per record in
`definition_version` so a changed definition is visible rather than silently
reusing an old label.

See [`PAPER_STORY_TRACE.md`](PAPER_STORY_TRACE.md) for what is and is not
implemented today, and [`KYUHYEOK_UPDATE_INTEGRATION.md`](KYUHYEOK_UPDATE_INTEGRATION.md)
for how an implementation plugs in.
