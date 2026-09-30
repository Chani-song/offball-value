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
| Last audited `origin/kyuhyeok-dev` | **`e84553a`** (2026-09-30 01:15 -0500) |
| Previous baseline | `c6423d4` (2026-09-28) |
| Between them | four commits: multi-pass solver, run-pass action set, opening payoff table, evaluation-set builders, A-sym pass model |
| Evaluation layer at `e84553a` | **still unimplemented** — all ten quantities. The inputs are built; nothing computes the comparison |
| Solver state schema | **changed** — see "Reading a multi-pass artifact" below |

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


---

## Reading a multi-pass artifact (`d1bbbb4` onward)

A run solved with `--multi-pass` is not shaped like the meeting-era runs, and
the difference is silent rather than loud. Both layouts are now read, but know
what changed:

| | meeting-era | `--multi-pass` |
| --- | --- | --- |
| `root_attack` length | `actions**2 + 1` | `actions**2 + len(pass_candidates)` |
| last entry | *the* release | **one candidate among many** |
| pass probability | `root_attack[-1]` | **sum over the tail** |
| `rollouts` | four sampled | `[]` — `modal_line` replaces them |
| new fields | — | `pass_candidates`, `multi_pass`, `modal_line`, `root_game`, `commands`, `passes`, `threat`, `terminal`, `pass_reaction_s`, `background_tackles` |

`demo_viz/solver/adapter.py` detects `multi_pass`, sums the tail for the pass
probability, and names the likeliest pass with the solver's own candidate
string. `web/site/js/policy.js` reads `move_columns` and `pass_candidates` off
the payload rather than assuming `n * n`.
`tests/test_paper_story.py::MultiPassArtifactTests` pins both layouts.

**When the first real multi-pass artifact arrives**, check three things before
believing the screen: the pass probability against the artifact's own
`modal_line[0].release_probability`; that `pass_candidates` and
`root_game.columns` agree on the tail; and that the horizon and `step_seconds`
match what `PASS_MODEL_TRACE.md` describes.

## What would unblock the evaluation layer

Not an export. `eval_v1.sbatch` already produces the inputs and says what they
are for — *"the opening payoff table (solve.root_game) for the
static-vs-responsive numbers"* and *"read into a panel … for the
observed-action matching"*. What is missing is the code that consumes them:

1. **observed-action matching** — project the real tracking at a moment onto
   the panel's command endpoints / pass targets. Supply
   `projection_distance_m` and `confidence`; the schema already carries both.
2. **a rank / similarity / regret definition** over `root_game.matrix` columns.
3. **a static/responsive pair on one scale.** `defender_pure_loss` is *not*
   it — it compares a committed defender with a mixing one, not a held opponent
   with a responding one.

Once those exist, the route is unchanged: implement `EvaluationSource`, set
`OFFBALL_EVALUATION_SOURCE`, re-export, rebuild.
