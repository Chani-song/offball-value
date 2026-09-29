# Paper story: what the code actually computes

Phase A audit. For every abstract-facing quantity: the exact definition, where
it lives, and whether the demo could honestly show it today.

Traced 2026-09-29 against the current tree, `origin/kyuhyeok-dev` @ `3c9965b`
and `~/Research/offball_demo/mit_ssac2027_ref` @ `e8b0a95` (read-only).

> **Re-audited 2026-09-29 at `origin/kyuhyeok-dev` @ `c6423d4`.** The local
> remote-tracking ref had been stale: the branch had moved `3c9965b..c6423d4`.
> The single new commit adds eight fitted pass-model coefficient files and a
> `.gitignore` rule to share them. **No `.py`, no sbatch, no test changed** —
> `git diff --name-only 3c9965b..c6423d4` touches nothing under `src/`,
> `scripts/`, `tests/`, `andrew-*/` or `deploy/`. Every conclusion below stands
> unaltered, and was re-verified by reading semantics at the new SHA rather
> than by trusting the file list. See §5.

**Headline: the equilibrium half of the abstract is implemented and real. The
player-evaluation half — relative rank, similarity to optimal, regret against
the observed action, frame-by-frame scoring, clip aggregation — is not
implemented anywhere, in either repository.** It is prose in the abstract, not
code. Nothing in this pass invented it.

---

## 0. Two separate lineages, easily confused

| | **Stage-3 exact solver** | **v0.1 localized counterfactual** |
| --- | --- | --- |
| Where | `andrew-passer2on1` / `andrew-fixedpasser` + `mit_ssac2027` | `defender_best_response.py`, `attacker_maximin.py`, `dynamic_response_game.py` |
| Game | full Nash equilibrium, mixed policies, certified | attacker max / defender min over feasible paths |
| Scored by | completion proxy × positional threat | **`reference_obso`** — the OBSO stack |
| Horizon | 3 decisions × 0.6 s from one onset state | 2 s, evaluated at 0.4 … 2.0 s |
| Produced the meeting page | **yes** | no |

The abstract's "game-theoretic / Nash", "responsive", "pure and mixed
equilibrium" language maps to the **stage-3 solver**. The v0.1 lineage is
scored by the very OBSO stack the public demo just demoted, so promoting it
back into the paper story would contradict the previous pass.

---

## 1. The old solver page: exact data path

```
deploy/delta/stage3_passer2on1_agile.sbatch
  └─ andrew-passer2on1/scripts/run.py  --model .../experimental_pass.json
                                       --allow-proxy-labels --steps 3 --step-seconds 0.6
  └─ OUT/  manifest.json · starting_states.json · rows.jsonl · states/ · policies/
  └─ scripts/summarise_stage3.py    → passer2on1_results_agile.csv
  └─ scripts/render_solver_story.py → out/passer2on1_agile/index.html
```

`render_solver_story.py` shows two views behind one button:
`솔버의 답` (the solver's answer) and `실제 움직임` (the real tracking clip).

---

## 2. Quantity-by-quantity

### Implemented and real (stage-3 solver)

| Quantity | Definition | Source | Status |
| --- | --- | --- | --- |
| **Equilibrium value** | root value of the solved Markov game | `markov.solve_markov_game` → `rows.jsonl:value` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Certificate / best-response gap** | root BR bounds recomputed through the whole policy tree, never from the solved value arrays | `markov.certificate` → `{lower, upper, gap}` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Mixed policy (defender)** | 5-vector over compass moves per decision, per reachable state | `policies/state_NNN.npz:defender_k` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Mixed policy (attack)** | 26-vector: 25 movement pairs + release | `…:attack_k` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Action in football terms** | compass move named for the line it points along most by cosine; "sideways" when no cosine ≥ 0.3; `(0,0)` is "brake", not "stand" | `stage3_read.name_move_targets` | `IMPLEMENTED_AND_AVAILABLE` (pure function) |
| **Pure vs mixed ("torn")** | SPLIT when the likeliest move is below **0.8** | `stage3_read.split_by_step(below=0.8)`, `render_solver_story.SPLIT = 0.2` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Root split fraction** | `fsplit0 = 1 − heaviest *merged* share` — merged by football name, because 55/45 east/south may both mean "toward the ball" | `summarise_stage3` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Solver answer trajectory** | the **modal line**: at every decision both sides take the action their policy weights most. Explicitly *"an illustration of the policy, not a sample from it"* | `stage3_read.modal_path` | `IMPLEMENTED_BUT_ARTIFACT_MISSING` |
| **Actual movement** | the observed tracking clip, untouched | scene data | `IMPLEMENTED_AND_AVAILABLE` |
| **Release payoff chain** | `legal × completion_proxy × positional_threat` | `payoff.release_payoffs_with_background` | `IMPLEMENTED_AND_AVAILABLE` (already shipped) |

### Not implemented anywhere

| Abstract concept | Searched | Status |
| --- | --- | --- |
| **Relative rank among feasible counterfactuals** | no match for `relative rank`, `percentile` (as a metric) in `src/` of either repo | `ABSTRACT_ONLY_NOT_IMPLEMENTED` |
| **Similarity to optimal action** | no match for `similarity`, no cosine/distance-to-optimal anywhere | `ABSTRACT_ONLY_NOT_IMPLEMENTED` |
| **Regret against the observed action** | `regret` exists **only** as `dynamic_response_game.normalized_option_regret`, a normalised min over a defender-option **cost matrix** in the v0.1 OBSO lineage. It is not a comparison of an observed action to an optimum. | `IMPLEMENTED_ONLY_FOR_SPECIFIC_PIPELINE` — and the wrong pipeline |
| **Frame-by-frame player evaluation** | the solver solves **one starting state per candidate triple**, with 3 decisions at 0.0 / 0.6 / 1.2 s. There is no per-frame scoring of a player across a clip. | `ABSTRACT_ONLY_NOT_IMPLEMENTED` |
| **Clip-level aggregation of player evaluation** | `summarise_stage3` aggregates **across scenes** for a results table (`split_t{k}`, `top_t{k}` means). It does not aggregate frames into a player score. | `ABSTRACT_ONLY_NOT_IMPLEMENTED` |
| **Exploitability** | no match in either repository | `ABSTRACT_ONLY_NOT_IMPLEMENTED` |
| **Static vs responsive counterfactual, as one comparison** | see below | `AMBIGUOUS_NEEDS_TEAM_CONFIRMATION` |

### The closest thing to observed-action evaluation

`markov.policy_transfer_bounds` reprices a **fixed full-feedback policy**
against fresh best responses on both sides. It is the only machinery that
scores a non-equilibrium strategy. Three reasons it is not the abstract's
metric:

1. it takes a whole **policy**, not one observed action;
2. it requires *"identical scenario, physics and action library"* — it exists
   to compare **payoff models**, not behaviour against an optimum;
3. it is never applied to observed tracking anywhere in either repository.

Building rank or similarity on top of it would be new science, which §5 and
§34 forbid in this pass.

### Static vs responsive: what exists, and why it is ambiguous

Three candidate readings, all implemented, none labelled this way in code:

| Reading | Static side | Responsive side | Scored by |
| --- | --- | --- | --- |
| A. what the demo already ships | **held-defender baseline** ("Space created") | observed defence | `goal_weighted_influence` |
| B. v0.1 localized counterfactual | non-intervened players keep their observed future | one focal defender takes a feasible delayed response | `reference_obso` (demoted stack) |
| C. stage-3 solver | — | every agent plays an equilibrium response | completion × threat |

The abstract's "opponents can optimally respond" is clearly **C**. But C has no
static twin computed on the same scale: nothing runs the stage-3 game with the
defender frozen. Reading A is on a different scale (residual space, not game
value); reading B uses the demoted OBSO stack.

**A direct static-vs-responsive comparison at one scale does not exist.**
Producing one means either running the stage-3 solver with a frozen defender,
or declaring A the intended comparison — both team decisions, not mine.

---

## 3. Data availability

No stage-3 artifact exists locally for any scene. Confirmed again this pass by
the bounded search from the previous one: no `stage3`, `passer2on1` or
`fixedpasser` output on this machine; the runs live on Delta at
`/work/hdd/bbmr/kseo1/offball-out`. The only real solver artifacts here are
`mit_ssac2027_ref/results/exact_100`, which are the solver's own declared 2v1
study states — not Bundesliga scenes, and never attached to one.

For the 21 showcase scenes: **0 have a solved policy**. The five
solver-derived entries also have no published tracking, so they are not
playable at all.

---

## 4. What this means for the demo

| Abstract layer | Can the demo show it today? |
| --- | --- |
| Observed play, roles, off-ball context | **Yes** — already does |
| Feasible action sets | Partly: `action_space` (kinematic reach) ships; the solver's own 5 compass moves and 18 releases are defined but need a state |
| Release-action valuation | **Yes** — already does |
| Equilibrium policy, pure vs mixed, solver answer | **Architecture yes, data no** |
| Static vs responsive at one scale | **No** — not defined in code |
| Observed-action rank / similarity / regret | **No** — not implemented |
| Frame-by-frame and clip-level player evaluation | **No** — not implemented |

Roughly **half** the abstract's narrative is code; the evaluation half is
prose. Per §35 this pass stops at the honest boundary: hierarchy changes and
architecture, no invented metrics.


---

## 5. Re-audit at `c6423d4` (2026-09-29)

`3c9965b..c6423d4` is one commit, linear, no rewrite (`git merge-base
--is-ancestor` confirms the old base is an ancestor of the new head):

    c6423d4  2026-09-28 13:15 -0500
    pass model A: share the fitted coefficient files
    (A_all, A_without_<match>, A_crossfit router)

Nine files: `data/processed/pass_models/A_all.json`, six
`A_without_DFL-MAT-*.json`, `A_crossfit.json`, and `.gitignore` (edited only to
un-ignore `data/processed/pass_models/A_*.json`).

### What pass model A is

`src/offball_value/physics_pass.py` — unchanged since before the audit, and
already covered by it. Candidate A prices a pass from the **arrival race**:
ball speed `v = intercept + slope · distance` fitted on measured launch speeds,
players moved by the solver's own `agile_motion.AGILE`, two time margins
(receiver, lane) clipped to ±3 s, then `P = sigmoid(b0 + b1·receiver +
b2·lane)`. Its own docstring states the point of it: *"The solver's pass model
reads positions only, so a runner sprinting into the space behind a defender
who is running the other way prices the same as two men standing still. This
model reads the race instead."*

`A_crossfit.json` is a router, not a model: `kind: per_match`, mapping
`provenance.match_id` to the model fitted **without** that match, with
`A_all.json` as the default. `run.py:model_router` resolves it and writes a
`pass_model_router` block, with a sha256 per model file, into the run manifest.

### Why this changes nothing in the table above

Model A occupies the **completion-proxy slot** — its docstring says so: *"The
interface is ExpectedPass.predict's, so the game calls it unchanged."* That is
row 10 of `ABSTRACT_DEMO_MAPPING.md`, the row the abstract itself calls a
replaceable component, and the row this demo deliberately demoted. It is not
relative rank, similarity, regret, an observed-action projection, a frame-level
score, a clip aggregate, or a static/responsive pair.

Re-checked at `c6423d4` by reading the implementations, not the names:

| Name that sounds like the abstract | What it actually is |
| --- | --- |
| `scripts/rank_triples.py` | ranks **(runner, defender, beneficiary) triples for human review** so ~1,900 candidates become a shortlist. Its `regret` is `minimax_worst_q` minus the defender's best reply knowing the target — the **defender's** cost of uncertainty in the v0.1 local game. Its own docstring: *"Ranking scenes for review is not the same as defining the dilemma."* |
| `scripts/search_ranking_criterion.py` | AUC of existing ranking columns against Chani's expert labels — asks whether any column separates her scenes. A scene-selection diagnostic. |
| `scripts/criterion_on_matched_runners.py` | the same AUC restricted to triples whose runner she named. Same purpose. |
| `similarity`, `observed_action`, `exploitability` | **zero matches** anywhere in `src/`, `scripts/` or `andrew-*/`. |

### The canonical pipeline still uses the old pass model

`deploy/delta/stage3_passer2on1_agile.sbatch` — the job that produced the
meeting page (§1) — still passes
`--model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels`,
byte-identical to `3c9965b`. The `stage3_*_passA/passB1/passB1SB` sbatch files
are **parallel model-comparison jobs** and predate this commit.

Every file this demo's adapter depends on is byte-identical between the two
SHAs: `summarise_stage3.py`, `render_solver_story.py`, `stage3_read.py`,
`andrew-passer2on1/passer2on1/payoff.py`, `andrew-passer2on1/scripts/run.py`,
`stage3_passer2on1_agile.sbatch`.

### Still not implemented, at `c6423d4`

All ten, unchanged: `observed_action`, feasible actions for evaluation, the
static/responsive pair on one scale, `observed_action_rank`, `relative_rank`,
`similarity_to_optimal`, observed-action regret, frame-level player evaluation,
clip-level aggregation, exploitability.