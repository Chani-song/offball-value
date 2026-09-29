# Integrating the updated research code

When the player-evaluation quantities land, the demo should need **one file
changed and two commands run**. This is the checklist that keeps it that way.

The design goal, stated once: the browser reads
`demo_viz/paper_story/schema.py`'s shape and nothing else. It does not know
about `rows.jsonl`, run directories, `policies/*.npz`, or any upstream variable
name. If integrating an update requires editing HTML, CSS or drawing code, the
adapter boundary has leaked and that is the bug to fix first.

---

## Audit baseline

| | |
| --- | --- |
| Last audited `origin/kyuhyeok-dev` | **`c6423d4`** (2026-09-28 13:15 -0500) |
| Previous baseline | `3c9965b` (2026-09-25) |
| Between them | one commit: fitted pass-model-A coefficients, no code |
| Evaluation layer at `c6423d4` | **still unimplemented** — all ten quantities |

Step 1 below exists because the local remote-tracking ref was stale once
already. **Always `git fetch origin --prune` before comparing**, and record the
SHA you actually audited here.

## Where things are

| | |
| --- | --- |
| The contract | [`demo_viz/paper_story/schema.py`](paper_story/schema.py) |
| The one adapter | [`demo_viz/paper_story/adapter.py`](paper_story/adapter.py) |
| The exporter | [`demo_viz/web/export_paper_story.py`](web/export_paper_story.py) |
| What the browser reads | `demo_viz/web_data/story/<scene>.json` |
| The vocabulary, shipped once | `demo_viz/web_data/story/contract.json` |
| What is and is not implemented | [`PAPER_STORY_TRACE.md`](PAPER_STORY_TRACE.md) |
| What the abstract needs | [`ABSTRACT_DEMO_MAPPING.md`](ABSTRACT_DEMO_MAPPING.md) |

---

## The eight steps

1. **Fetch first, then diff from the recorded baseline.**

   ```bash
   git fetch origin --prune                      # the ref goes stale silently
   git rev-parse origin/kyuhyeok-dev             # record this above
   git log --oneline <baseline>..origin/kyuhyeok-dev
   git diff --name-status <baseline>..origin/kyuhyeok-dev
   ```

   Diff from the recorded baseline rather than re-reading the whole tree, and
   check `git for-each-ref refs/remotes/` in case the work landed on another
   branch. Do not merge: this repository reads the research code, it does not
   vendor it (one exception, `_passer2on1_payoff.py`, which a test holds
   byte-identical). A commit that touches no `.py` cannot have added a metric —
   confirm that before spending time on the contents.

2. **Identify the output definitions.** For each quantity below, find the
   function that computes it and read the implementation, not the name. The
   last pass found a `regret` that was not the abstract's regret and a
   `policy_transfer_bounds` that was not an observed-action comparison; both
   would have looked right from a grep.

3. **Confirm semantics with the author.** Specifically: what the reference
   optimum is, what the observed action is projected onto, and which scale the
   static and responsive sides share. Write the answers into
   `PAPER_STORY_TRACE.md` before writing code.

4. **Implement `EvaluationSource`.** Four methods, all allowed to return
   `None`:

   ```python
   class KyuhyeokEvaluation:
       name = "kyuhyeok-dev@<commit>"
       definition_version = "<upstream's own version string>"

       def counterfactual(self, scene_id, role, frame): ...
       def evaluation(self, scene_id, role, frame): ...
       def frame_series(self, scene_id, role): ...
       def clip_metrics(self, scene_id, role): ...
   ```

   Keys are the reserved names in `adapter.EVALUATION_METRICS` and
   `adapter.FRAME_SERIES`. A value entry is
   `{"value": ..., "unit": ..., "definition_version": ..., "aggregation": ...}`;
   only `value` is required. The adapter attaches provenance and the source
   string and never inspects the number.

5. **Point the exporter at it** and regenerate:

   ```bash
   OFFBALL_EVALUATION_SOURCE=my_pkg.evaluation:KyuhyeokEvaluation \
       .venv/bin/python -m demo_viz.web.export_paper_story
   .venv/bin/python -m demo_viz.web.build
   ```

   The `Evaluation source` row in the demo's Source section will stop saying
   *none · not implemented in the research code yet* and name the module and
   definition version instead. That row is the integration's own smoke test.

6. **Run the parity tests.** `.venv/bin/python -m unittest discover -s tests`.
   `tests/test_paper_story.py` is the one that matters here: it enforces that a
   value cannot appear without a source, that unavailable fields carry no
   number, and that the public payload cannot contain fixture data.

7. **Fields enable themselves.** There is no allow-list to update. Anything the
   source returns is rendered, with its own label and unit, by `contractRows`
   in `app.js`. Anything it does not return renders its reason.

8. **Do not hand-edit the interface per metric.** If a new metric needs UI work,
   say so in a review rather than special-casing it — the next one will need it
   too.

---

## Per-quantity checklist

Tick each only when the implementation has been read, not when a name matched.

### Relative rank
- [ ] What is ranked: actions in a feasible set, or something else?
- [ ] Rank among how many, and is that count stable across frames?
- [ ] Is it a rank (1 = best) or a normalised position? Which direction is good?
- [ ] If a percentile: over what population, and computed how?
- [ ] Key: `relative_rank`. Also supply `unit` if it is not dimensionless.

### Similarity to optimal
- [ ] Similar in what space — action vector, endpoint, value?
- [ ] Bounded? If so, to what, and does 1 mean identical?
- [ ] What is "optimal" here: the equilibrium action, the best release, the
      argmax of a value function?
- [ ] Key: `similarity_to_optimal`, and `optimal_action` for the action itself.

### Regret
- [ ] Against what reference, and in which value units?
- [ ] Sign convention: is high regret bad?
- [ ] Is it comparable across scenes, or only within one?
- [ ] Key: `regret`.

### Static counterfactual
- [ ] What exactly is held fixed, in one sentence — this becomes `semantics`
      and is shown verbatim to the reviewer.
- [ ] Which of the three readings in `PAPER_STORY_TRACE.md` §2 is it?
- [ ] Key: `static` → `{"semantics": ..., "best_action": {...}, "value": {...}}`.

### Responsive counterfactual
- [ ] Which agents respond, and optimally with respect to what?
- [ ] **Same scale as the static side?** If not, the pair must not be shown as
      a comparison; say so and the adapter will keep them apart.
- [ ] Key: `responsive`, same shape. `value_change` is the paired difference,
      and only exists if the two sides share a scale.

### Observed action
- [ ] Is observed tracking projected onto a discrete action library?
- [ ] If so, supply `projection_distance_m` and `confidence` — the interface
      shows both, so a poorly matched action does not read as a certain one.
- [ ] Key: `observed_action`.

### Frame-level metrics
- [ ] Which frames are evaluated, and why those?
- [ ] Frame indices must be this repository's scene frame indices, not times
      and not upstream row numbers.
- [ ] **Is joining consecutive samples meaningful?** Set `interpolate` only if
      it is; the strip draws stems rather than a line by default, because a
      line asserts values nobody computed.
- [ ] Supply `domain` when the metric has a natural range.
- [ ] Keys: `relative_rank`, `similarity_to_optimal`, `regret`.

### Clip aggregation
- [ ] What aggregation, over which frames, with what weighting?
- [ ] Supply it as the `aggregation` string per metric. The interface prints it
      on hover and never assumes a mean.
- [ ] Any name is accepted; `clip_metrics` returns whatever the pipeline has.

---

## If a definition is still unsettled

Ship it with a `definition_version` that says so and leave the rest pending.
A partially populated Evaluation panel is fine — each field states its own
availability. What is not fine is a placeholder number: `Metric` raises rather
than construct one, so this cannot happen by accident.
