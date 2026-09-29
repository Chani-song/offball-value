# Abstract → demo traceability

Each claim the abstract makes, the demo feature that carries it, the code
behind it, and whether it is real today. Read `PAPER_STORY_TRACE.md` for the
definitions.

Status: **live** (in the demo now) · **blocked** (built, needs a Delta
artifact) · **slot live** (the interface and the typed contract slot exist and
render an explicit reason; the quantity itself has no implementation).

"Slot live" is what changed on `chani-ssac-demo-final-story`. It is not a
weaker kind of "live": the value is still absent, and the demo still says so.
It means the field is declared in `paper_story/schema.py`, validated, exported,
and rendered — so landing the quantity is an adapter change rather than an
interface change. `KYUHYEOK_UPDATE_INTEGRATION.md` is the checklist.

| # | Abstract concept | Demo feature | Code | Status |
| --- | --- | --- | --- | --- |
| 1 | Responsive counterfactuals rather than holding opponents fixed | Game solution mode (equilibrium policy + modal-policy illustration) | `markov.solve_markov_game`, `stage3_read.modal_path` | **blocked** |
| 2 | Frame-by-frame evaluation of every relevant player | Evaluation mode → frame strip, synced to playback | contract `Series`; no producer | **slot live** |
| 3 | Feasible counterfactual actions — runner | Kinematic reachable area | `action_space.solve_endpoint_motion` | **live** (labelled as the endpoint-motion model, not the pipeline steering model) |
| 3b | Feasible counterfactual actions — passer | Counterfactual mode → `solver_release_actions` | `models.DEFAULT_PASSES`, `solver_release_targets` | **blocked** (needs a state with stage-3 semantics) |
| 3c | Feasible counterfactual actions — defender | 5 compass moves | `stage3_read.SOLVER_DIRS`, `world_direction` | **blocked** |
| 4 | Game-theoretic / Nash solution | Game solution view | `markov.solve_markov_game` + `certificate` | **blocked** |
| 5 | Comparing observed actions against feasible counterfactuals | Solver policy / Actual movement / Overlay | `modal_path` vs scene tracks | **blocked** (both halves exist; only the solved half is missing) |
| 6a | Relative rank among feasible counterfactuals | Evaluation mode → `relative_rank` | contract `Metric`; no producer | **slot live** |
| 6b | Similarity to the optimal action | Evaluation mode → `similarity_to_optimal`, `optimal_action` | contract `Metric` / `ActionRef`; no producer | **slot live** |
| 7 | Aggregation of frame-level evaluations across a clip | Clip summary cards, per story role | contract `ClipSummary`; aggregation supplied per metric | **slot live** |
| 8 | Static vs responsive counterfactual comparison | Counterfactual mode → paired cards, each with its own `semantics` | contract `CounterfactualSide`; no single-scale comparison exists | **slot live** (see trace §2) |
| 9 | Pure and mixed equilibrium behaviour | Game solution → weighted policy bars, support counts, pure/mixed | `policy.js` decode + `split_by_step(below=0.8)`, `fsplit0` | **blocked** for a tracked scene; **live** against a real reference state (`story_harness.html`) |
| 10 | Pass/xT-like models are interchangeable components | Action value details, demoted below the story | `payoff.release_payoffs_with_background` | **live** |

## What the demo can honestly claim today

Rows 3 and 10 are live. Row 10 is the one the previous pass made exact, and
this pass demotes it so it stops reading as the contribution.

Rows 1, 4, 5, 9 are **one artifact away**: the adapter, the policy
representation and the renderer are written and tested; no scene has a solved
policy. See `DELTA_EXPORT_REQUIREMENTS.md`.

Rows 2, 6a, 6b, 7, 8 have **no implementation in either repository**. They are
the abstract's evaluation layer, and they are still the gap — what changed is
that the demo now has a validated, exported, rendered slot for each, so filling
them is an adapter change. Nothing about the science moved.

## Screenshot candidates

| Concept | Scene | Exists? |
| --- | --- | --- |
| Observed play + roles + off-ball context | S20, S05 | yes |
| Action valuation drill-down | S20 frame 115 | yes |
| Legality as a gate | S02 frame 45 | yes |
| Equilibrium policy / solver answer, on a tracked scene | — | needs an artifact |
| Equilibrium policy component, on a real solver study state | `story_harness.html` | yes — `final_story_game_solution_reference.png`, labelled as a study state |
| Observed-action evaluation, populated | — | needs the metric to exist |
| Observed-action evaluation, honest pending state | S20 | yes — `final_story_evaluation_pending.png` |
