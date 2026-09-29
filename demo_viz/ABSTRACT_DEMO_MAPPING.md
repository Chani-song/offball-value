# Abstract → demo traceability

Each claim the abstract makes, the demo feature that carries it, the code
behind it, and whether it is real today. Read `PAPER_STORY_TRACE.md` for the
definitions.

Status: **live** (in the demo now) · **blocked** (built, needs a Delta
artifact) · **not implemented** (prose only).

| # | Abstract concept | Demo feature | Code | Status |
| --- | --- | --- | --- | --- |
| 1 | Responsive counterfactuals rather than holding opponents fixed | Game solution view (equilibrium policy + modal line) | `markov.solve_markov_game`, `stage3_read.modal_path` | **blocked** |
| 2 | Frame-by-frame evaluation of every relevant player | — | none | **not implemented** |
| 3 | Feasible counterfactual actions — runner | Kinematic reachable area | `action_space.solve_endpoint_motion` | **live** (labelled as the endpoint-motion model, not the pipeline steering model) |
| 3b | Feasible counterfactual actions — passer | Solver release actions (18) | `models.DEFAULT_PASSES`, `solver_release_targets` | **blocked** (needs a state with stage-3 semantics) |
| 3c | Feasible counterfactual actions — defender | 5 compass moves | `stage3_read.SOLVER_DIRS`, `world_direction` | **blocked** |
| 4 | Game-theoretic / Nash solution | Game solution view | `markov.solve_markov_game` + `certificate` | **blocked** |
| 5 | Comparing observed actions against feasible counterfactuals | Solver answer vs actual movement | `modal_path` vs scene tracks | **blocked** (both halves exist; only the solved half is missing) |
| 6a | Relative rank among feasible counterfactuals | — | none | **not implemented** |
| 6b | Similarity to the optimal action | — | none | **not implemented** |
| 7 | Aggregation of frame-level evaluations across a clip | — | none | **not implemented** |
| 8 | Static vs responsive counterfactual comparison | — | no single-scale comparison exists | **not implemented** (see trace §2) |
| 9 | Pure and mixed equilibrium behaviour | "torn" indicator, per-decision policy shares | `split_by_step(below=0.8)`, `fsplit0` | **blocked** |
| 10 | Pass/xT-like models are interchangeable components | Action value details, demoted below the story | `payoff.release_payoffs_with_background` | **live** |

## What the demo can honestly claim today

Rows 3 and 10 are live. Row 10 is the one the previous pass made exact, and
this pass demotes it so it stops reading as the contribution.

Rows 1, 4, 5, 9 are **one artifact away**: the adapter, the policy
representation and the renderer are written and tested; no scene has a solved
policy. See `DELTA_EXPORT_REQUIREMENTS.md`.

Rows 2, 6a, 6b, 7, 8 have **no implementation in either repository**. They are
the abstract's evaluation layer, and they are the gap.

## Screenshot candidates

| Concept | Scene | Exists? |
| --- | --- | --- |
| Observed play + roles + off-ball context | S20, S05 | yes |
| Action valuation drill-down | S20 frame 115 | yes |
| Legality as a gate | S02 frame 45 | yes |
| Equilibrium policy / solver answer | — | needs an artifact |
| Observed-action evaluation | — | needs the metric to exist |
