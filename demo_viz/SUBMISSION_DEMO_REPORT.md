# Submission demo — research inspector

What the public demo exposes, where every number comes from, and what is still
unavailable. Written for a reviewer who has not read the code.

---

## 0. Audit: what exists, and where

Taken from the current tree, `origin/kyuhyeok-dev` @ `3c9965b` (read-only) and
`~/Research/offball_demo/mit_ssac2027_ref` @ `e8b0a95` (read-only, unmodified).

### Already in this tree

| Quantity | Where | Status |
| --- | --- | --- |
| Player/ball tracks, 25 Hz | `web_data/<scene>.json` | measured |
| Velocity (central difference, 0.4 s window) | `js/influence.js` `velocityAt` | derived |
| Run onset | `core/role_logic.runner_onset_index`, exported per attacker | derived |
| Residual space (Available space) | `offball_value.goal_weighted_influence` | derived |
| Held-defender counterfactual (Space created) | `core/influence.InfluenceCache` | derived |
| OBSO = control × transition × EPV | `offball_value.reference_obso` | derived |
| EPV grid | `data/static/EPV_grid.csv` (PAUSA, Apache-2.0) | reference data |
| Candidate pass fan | `core/candidates.py`, `js/obso.js` | fixed geometry |

### On `origin/kyuhyeok-dev`, inspected but **not** brought across

| Module | Why not |
| --- | --- |
| `physics_pass.py` (pass candidate A) | needs a fitted JSON that lives only on Delta |
| `xpass.py`, `kinematic_xpass.py` | training-side; no local fitted artefact |
| `vacated_space.py` | author flags it as post-hoc on the round-2 test |
| `defender_response.py`, `attacker_maximin.py` | need Delta pipeline CSVs |
| `steering_reachable.py` | see below — measured at 9.5 s per player-frame |

### Reachability: three implementations, one choice

| Model | Cost | Used by |
| --- | --- | --- |
| `action_space.solve_endpoint_motion` | microseconds, closed form | endpoint action grids |
| `steering_reachable` | **9.5 s** per player-frame (measured, 1115 endpoints) | defender-response pipeline |
| `dynamic_reachable` | not measured | calibration scripts |

The defender-response pipeline uses the steering model. At 9.5 s per
player-frame it cannot run in a browser and cannot be precomputed for 22
players × 276 frames × 45 scenes. **The Reachable area layer therefore shows
`action_space.solve_endpoint_motion`** — accelerate-then-cruise, 2.0 s horizon,
9 m/s, 3.5 m/s² — ported to JavaScript with the same constants and checked
against Python. The Source panel names both models and says which is drawn.
This is a different model shown exactly, not the steering model approximated.

### Solver artifacts (`mit_ssac2027_ref`)

Real, and present locally: `results/exact_100/` holds 100 solved 2v1 states.

| File | Contents |
| --- | --- |
| `states/state_NNN.json` | scenario (carrier/receiver/defender position, velocity, speed and acceleration caps), `value`, `certificate` {lower, upper, gap}, `root_attack` (26), `root_defender` (5), timings |
| `rows.jsonl` | the same per state, plus `rollouts` |
| `policies/state_NNN.npz` | `attack_k` (n,n,n,26), `defender_k` (n,n,n,5), `pass_index_k`, `terminal_release`, `metadata` (schema, fingerprint, root value) |
| `manifest.json` | `steps`, `step_seconds`, `physics_step`, 5 `directions`, 18 `passes` |

Action encoding, from `markov.py`: 5 directions `[(0,0), (1,0), (0,1), (-1,0),
(0,-1)]`; an attack action `a < 25` splits as `carry, run = divmod(a, 5)`, and
`a == 25` is release. Rollout events are `move`, `release`, `retain`,
`tackled_during_previous_interval`. Coordinates are corner-origin metres on
105 × 68; the demo's are centre-origin, so the adapter shifts them.

**These are the solver's own declared 2v1 study states, not Bundesliga
scenes.** No solver artifact exists locally for any demo scene, and per
Kyuhyeok's 2026-09-25 note exactly one annotated scene reaches the solver at
all, on Delta. So every demo scene reports *Not computed for this scene*, and
the adapter is validated against the real artifacts above.

---

## 1. What was added

| | |
| --- | --- |
| **Analysis panel** | Player / Space & threat / Passing / Solver, each shown only when its selection or layer is live |
| **Player inspector** | click any player; speed, acceleration, distances, run onset, available space, reachable area |
| **Reachable area** | layer, off by default |
| **Solver solution** | layer, off by default, with an honest unavailable state |
| **Threat components** | pitch control / ball transition / EPV / OBSO at a player or a pass endpoint |
| **Candidate endpoint inspector** | click an endpoint: direction, distance, ball travel |
| **Submission manifest** | schema, validator, loader, example; absent ⇒ current explorer, unchanged |
| **Solver adapter** | reads real `mit_ssac2027` artifacts, with provenance |

Clicking a player inspects it *and* still assigns a role exactly as before, so
no existing interaction changed. Multi-runner, multi-beneficiary and
multi-defender assignment, drag and drop, the three space views, Focus, and the
Pages build all behave as they did.

## 2. Quantities exposed, and their provenance

Three categories, never blurred:

| Quantity | Category | Source |
| --- | --- | --- |
| Position, speed, acceleration, distances | **Measured** | tracking at 25 Hz; velocity by central difference over 0.4 s |
| Role | **Human** | manual annotation |
| Run start | **Derived** | `run_onset.detect_kinematic_run_onsets`, else a labelled acceleration cue |
| Available space | **Derived** | `goal_weighted_influence` residual |
| Space created | **Derived** | the same, minus a held-defender baseline |
| Pitch control, Ball transition, EPV, OBSO | **Derived** | `reference_obso`; EPV is the static PAUSA grid |
| Reachable area | **Derived** | `action_space.solve_endpoint_motion`, 2.0 s / 9 m/s / 3.5 m/s² |
| Candidate passes | **Derived** | fixed geometry: 90° sector, five 25 m rays |
| Ball travel | **Derived** | endpoint distance ÷ 15 m/s, the `ReferenceOBSOConfig` ball speed |
| Equilibrium value, certificate gap, root policies, trajectories | **Solver** | read from a real artifact only |

Two things are deliberately absent, because no validated local artifact defines
them: **pass completion probability** and any **expected value** combining
completion with threat. The candidate fan carries neither, and the Passing
section reports geometry and ball travel only.

`Ball travel` is worth one line of honesty: 15 m/s is a single constant from
the OBSO configuration, which the pitch-control term already assumes. It is not
a fitted flight model, and no arrival margin is claimed from it — receiver and
defender arrival would need `physics_pass`, whose coefficients are not local.

## 3. Reachability: which model, and why

`action_space.solve_endpoint_motion` — one constant acceleration, then cruise —
ported to JavaScript and checked against Python on 64 cases per run, feasibility
and solved motion, to 1e-9. It runs live: the 1 m action grid is 7,140
feasibility tests, a few milliseconds.

The defender-response pipeline uses the richer `steering_reachable` instead
(tangential and normal control with plant-and-cut). Measured here at **9.5 s for
one player at one frame**, producing 1,115 endpoints. That cannot run in a
browser and cannot be precomputed for 22 players × 276 frames × 45 scenes, so it
is not drawn. The Source panel names both models and says which one the layer
shows. Nothing is approximated silently.

## 4. Solver adapter format

Input is a run directory from `defensive_positioning.exact_study`, referenced as
`<run directory>#<state index>`:

    manifest.json           steps, step_seconds, directions, passes
    rows.jsonl              scenario, value, certificate, root policies, rollouts
    states/state_NNN.json   the same without rollouts (fallback)
    policies/state_NNN.npz  full conditional policy tables; metadata fingerprint

The adapter normalises corner-origin solver metres to the demo's centre-origin,
decodes the action encoding from the run's own manifest (`carry, run =
divmod(a, 5)`, `a == 25` is release) rather than assuming it, and attaches
repository, commit, artifact, state index, policy fingerprint and run timestamp
to every state.

`load_for_scene` returns `Unavailable(reason)` for a missing, malformed or
unreadable reference. There is no branch that substitutes anything else.

## 5. Which scenes have genuine solver output

**None.** All 45 report *Not computed for this scene*.

That is the true state of the pipeline, not a gap in this work: per the
2026-09-25 pipeline note, exactly one annotated scene reaches the solver at all,
and that run lives on Delta. The adapter is instead validated against real
artifacts — `mit_ssac2027@e8b0a95 results/exact_100`, 100 solved 2v1 states —
and three of them are exported to `web_data/solver/` under a
`solver_reference` kind that the tests forbid from shadowing any tracked scene.

### Integration run (§10)

The solver was run once, locally, on its own declared scenario set. No licensed
data was involved and the reference repository was not modified (verified clean
afterwards); output went to a temporary directory.

```
repo     ~/Research/offball_demo/mit_ssac2027_ref @ e8b0a95
command  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
         python -m defensive_positioning.exact_study \
           --states data/scenarios/conditioning_100.json \
           --model models/experimental_pass.json --allow-proxy-labels \
           --workers 2 --limit 1 --output <tmp>
runtime  12 s wall (one state; the published run reports a 20.3 s median)
input    data/scenarios/conditioning_100.json, state 0
output   manifest.json, rows.jsonl, states/, policies/state_000.npz
```

The freshly produced artifact parses with the adapter, and reproduces the
committed run exactly: value `0.7045360969038921` against `0.7045360969038921`,
and the policy tables are **bit-identical**. The `metadata.fingerprint` string
differs between the two runs even though the tables and the manifest config do
not, so it is not a content hash of the policy alone — worth knowing before
anyone uses it as one.

One dependency (`imageio_ffmpeg`) had to be installed into this project's venv
for the solver's clip-rendering step; `pyproject.toml` was not changed.

## 6. What remains unavailable

* **Pass completion probability.** Candidate A (`physics_pass`) needs a fitted
  JSON that exists only on Delta. The solver's current estimator is described by
  its own author as "an experimental proxy, not a validated replacement".
* **Expected value** of a pass, which needs the above.
* **Receiver / defender arrival margins**, same reason.
* **Solver output for any demo scene** — see §5.
* **Steering reachability** in the browser — see §3.
* **Optimal defender best response** as a separate overlay: the artifacts hold a
  joint equilibrium policy, not a labelled "best response" trajectory, so
  presenting one would be an interpretation rather than a reading.

## 6b. Timeline marks

Two marks on the scrubber, both from data the scene already carries: the run
onset per selected runner (`scene.onsets`, carrying the detector's own method
label as a tooltip) and the annotated shot, which is where the clip's clock
reads zero. Nothing else is marked — the exported scene defines no "defender
reacts" frame, so none is drawn, and a test asserts that.

Scrubbing updates everything together: tracks, the player inspector, available
space, space created, OBSO, the candidate fan and the reachable area are all
recomputed from the same frame index in one render pass.

## 7. Performance

| | |
| --- | --- |
| Initial load | 140 KB (shell + scene index) |
| Scene | 78 KB, lazy |
| OBSO surface | ~122 KB per scene, lazy, only when the threat view is picked |
| Solver artifact | ~4 KB per state, lazy, only when the solver layer is picked |
| Reachable area | computed live, 7,140 feasibility tests, memoised per player and frame |
| Role switch | unchanged at 0.8 ms |

No solver runs in the browser. Nothing added to the initial payload but the
shell, which grew from 103 KB to 132 KB.

## 8. Tests

230 pass. New: reachability JS↔Python parity (feasibility and motion, 1e-9),
solver artifact parsing against the real run (coordinates, action decoding,
probability normalisation, mixed vs pure, events, provenance, pass targets),
the unavailable state in four failure modes, a guard that no exported solver
file shadows a tracked scene, and manifest schema/validator/loader including
graceful absence. Browser↔Python parity for Available space and Space created
is unchanged at 8.5e-14.

## 9. Screenshots

`demo_viz/exports/submission/`

| File | What |
| --- | --- |
| `a_strong_inspector.png` | Strong scene, player inspector (1440p) |
| `b_medium_scene.png` | Medium scene, OBSO threat |
| `c_multi_runner.png` | two runners, two defenders, inspector on #10 |
| `d_obso_passes.png` | OBSO threat with the candidate fan (1440p) |
| `e_reachable.png` | Reachable area for a selected player |
| `g_solver_unavailable.png` | the honest solver state on a demo scene |

There is no solver-populated screenshot of a demo scene, because there is no
such artifact. §18's scenario F is the one thing here that genuinely does not
exist yet.

---

# Submission showcase (2026-09-28)

## 1. Ingestion

`local_inputs/dilemma_showcase.html` is the curated source. It is **local-only,
gitignored and not committed**, because it embeds raw tracking.
`python -m demo_viz.ingest_showcase` reads its curation metadata and writes
`demo_viz/data/submission_showcase.json` (28 KB), which holds **no
trajectories**: a showcase entry is played from the repository scene it maps
to, or it is not played. The generator refuses to write a file containing a
track key, and a test independently scans the committed file for long numeric
arrays.

## 2. What the curated set contains

| | |
| --- | --- |
| Scenes | **21** |
| Human-reviewed | **16** (source kinds: 4 strong, 12 medium) |
| Solver-derived | **5** (3 × 2v1, 2 × 3v1) |
| 5/5 ratings | **7** |
| 5/4 or 4/5 | **14** |
| Timing: shot-centred | 16 |
| Timing: run-onset-centred | 5 (all solver-derived) |

Every entry carries two reviewers, both ratings, their off-ball judgement and
their notes. Timing is preserved per scene rather than normalised: the Analysis
panel reports *Timeline zero — Shot* or *Run starts* accordingly.

## 3. Mapping

| Status | Count |
| --- | --- |
| verified | **16** |
| unresolved | **5** |
| ambiguous | 0 |

A candidate has to agree on the unordered team pair (repository titles are
"attacking vs defending" and flip between scenes of one fixture), the half, and
containment of the hand-labelled roles in the annotation — then be confirmed by
the shooter's name or exact role equality. Anything weaker stays unresolved.
Every record stores its evidence string.

All five unresolved entries are the solver-derived ones: they are run-onset
states from the stage-3 pipeline, not shot annotations, so none is among the 45
published scenes. The schema forbids a non-verified entry from naming a
`scene_id`, so a guess cannot become a mapping.

## 4. Human-reviewed roles

For a verified entry the reviewers' own runner/defender/beneficiary become the
scene's opening state, replacing the stored annotation where they differ — S05
opens on one runner and one defender rather than the annotation's two of each.
Roles are only applied when **all three** resolve on the expected side;
otherwise the annotation stands and the panel says the reviewer's roles could
not be applied. After loading, every existing interaction still works: add,
remove, drag between roles, inspect anyone.

## 5–7. Solver status

**No showcase scene can show solver output, and none pretends to.** See
`SUBMISSION_SOLVER_MAPPING.md` for the per-entry table. In short: the five
solver-derived entries have no published scene data and no local artifact —
their solved policies are on Delta. A bounded search of the locations the docs
name found no stage-3, `passer2on1` or `fixedpasser` output locally.

The only real artifacts present are `mit_ssac2027_ref/results/exact_100`, the
solver's own declared 2v1 study states. They validate the adapter and are
exported under a `solver_reference` kind that tests forbid from shadowing a
tracked scene. They are never attached to a Bundesliga scene.

Three states are distinct in the UI: *Solver solution available*, *Not computed
for this scene*, and — for an entry with no published scene — *scene data not
published* on the selector itself.

## 8. Collections and filters

`Submission showcase` (default) and `Full explorer`. Filters are offered only
when the metadata supports them: All, 5/5, Human-reviewed, Solver-derived, 2v1,
3v1. The selector is one line — `S05 · 5/5 · Human-reviewed` — with the fixture
on the pitch header and in the Analysis panel; reviewer notes and annotations
stay out of it.

## 9. Top-10 readiness

`featured` and `order` are the only fields to change. With nothing featured the
default filter is `all`; the moment any entry is featured it becomes
`featured`, in both Python and the browser, with the same rule and a test on
each side. No code change is needed.

## 10. Performance

| | |
| --- | --- |
| Initial load | **190 KB** (shell 148 KB + scene index 8.6 KB + showcase 29 KB) |
| Scene | 78 KB, lazy |
| OBSO surface | ~122 KB per scene, lazy |
| Solver artifact | ~4 KB, lazy |
| Kinematic reachable area | computed live, memoised per player and frame |

## 11. Tests

**265 pass**, 35 of them new for the showcase: scene count, provenance and
rating-group counts, role and note preservation, per-scene timing, mapping
counts and evidence, the rule that an unresolved entry names no scene, that the
hand-labelled shirts really exist on the right side of the mapped scene,
trajectory-leak guards on both the file and the validator, schema rejections,
featured/order behaviour on both sides, and that the local source is untracked.
Browser/Python parity for Available space and Space created is unchanged at
8.5e-14.

## 12. Visual QA

Screenshots in `demo_viz/exports/submission/` at 1440p and 1080p. Two defects
found and fixed during the pass: a `.chip` class collision that made the role
chips unreadable, and `[hidden]` losing to `.control { display: flex }` so both
collections' controls showed at once. Toggling a layer whose readout lives in
the Analysis panel now scrolls that panel into view.

## 13. Remaining limitations

* Five curated scenes cannot be shown at all, for the data reasons above.
* No solver comparison is visible anywhere in the public demo.
* EPV is still position-only and does not encode the defensive line.
* The Kinematic reachable area is the endpoint-motion model, not the
  pipeline's steering model; both are named in Source.
* No pass completion probability or expected value — no validated local artifact.

## 14. When the final Top-10 arrives

Edit `demo_viz/data/submission_showcase.json`: set `"featured": true` and
`"order": 1..10` on those entries, then `python -m demo_viz.web.build`. The
default filter flips to Featured on its own. If a chosen scene is one of the
five unresolved ones, it needs its scene data published first — otherwise it
will appear in the list and stay disabled.

---

# Solver-native conversion (2026-09-28)

## Why OBSO was demoted

`PASS_MODEL_TRACE.md` established from source that the current solver's release
payoff is

    legal × completion_proxy × positional_threat

and that OBSO, EPV and pitch control appear nowhere in it. They were a
reference diagnostic, not part of the model the meeting results came from.
Leaving them prominent next to solver-native numbers would have implied they
were the same stack.

**The implementation, its export and its tests all stay.** Only the public
controls went. The demo built around it is preserved on
`backup/ssac-obso-demo` @ `26c752f`, untouched.

## What the public UI now shows

| | |
| --- | --- |
| Space views | `Available space`, `Space created` — the OBSO option is gone |
| Pass layer | `Explore pass model` (was `Candidate passes`) |
| Analysis order | Scene → Off-ball effect → **Solver pass model** → Player → Solver |
| Solver pass model | Receiver, Direction, Legal, Completion proxy, Positional threat, **Release payoff** |
| Model details | seven of the 33 features, collapsed by default |
| Source | a `Current solver` entry; OBSO appears once, under `Legacy / reference diagnostic` |

## The chain, and where the numbers come from

Every value is produced by the research implementation at export time and read
by the browser. Nothing is recomputed there and nothing is ported:

* **Completion proxy** — Andrew's `ExpectedPass` on `experimental_pass.json`,
  the 33-feature positions-only logistic, reached exactly as the meeting
  pipeline reaches it.
* **Positional threat** — `positional_threat_all`, vendored byte-identical from
  `origin/kyuhyeok-dev:andrew-passer2on1/passer2on1/payoff.py` so it stays
  diffable. Room to the **nearest of the whole defending side**, bounded [0.2, 1].
* **Legal** — `inside_pitch` and `offside_flags`; with two or more defenders the
  multi-defender rule (second-last), never Andrew's single-defender line.
* **Release payoff** — `legal ? proxy × threat : 0`.

The model is handed the **whole defending side**, as the 2v1 wrapper does, not
a pre-selected defender.

### The gate is visible

An illegal target keeps its completion proxy and takes a payoff of zero. On
S02 frame 45: `Legal: No · offside`, proxy `0.999`, threat `0.571`, payoff
`0.000`. That is `solver_native_legality.png`.

## Exploratory targets are ours, the values are theirs

The five rays are still the demo's own geometry — 90° sector, five 25 m rays
from the carrier. They are **not** the solver's action space and are never
called best, top or recommended. The solver's own 18 releases are built from
the receiver (`receiver + direction × offset`) and belong to stage-3 states this
demo does not publish; `solver_release_targets()` reads that definition from the
imported module so the set cannot drift, and a test asserts the two
constructions differ.

## Receiver selection

The receiver is the **selected beneficiary**, explicitly. With none selected the
panel says *"Select a receiving attacker to inspect the solver pass model."* and
computes nothing. No nearest-attacker fallback exists; a test forbids one.

## Solver trajectories

Unchanged and still honestly unavailable. No scene has a compatible artifact;
the five solver-derived showcase entries remain unresolved pending their Delta
export.

## Payload and performance

| | |
| --- | --- |
| Initial load | 201 KB (shell 158 + index 8.6 + showcase 29) |
| Scene | 78 KB, lazy |
| Release quantities | **83 KB per scene**, lazy, only when the explorer is used |
| Solver artifact | ~4 KB, lazy |
| OBSO (legacy) | **not in the public build**; `--include-legacy-obso` ships it for internal debugging |

Export of all 45 scenes takes 7 s. No research code runs in the browser.

## Optional by construction

The export needs Andrew's `defensive_positioning` and the fitted
`experimental_pass.json`, neither committed. `availability()` reports what is
missing, `export_release.py` **exits 0** with an explanation, and the site
builds without a pass-model payload — the explorer then says so rather than
showing blanks. Neither the model nor the reference repo is tracked.

## Tests

**302 pass.** New: the trace's numbers pinned (logit `6.498830792478147`, proxy
`0.998497064162670`, independent agreement ~1e-16); the background-defender
sensitivity `0.992984 → 0.907020` with only the receiver block and
`same_defender` moving; the legality gate; `payoff == legal × proxy × threat`
across cases and both attacking directions; the vendored payoff byte-identical
to `origin/kyuhyeok-dev`; the export payload carrying no tracking and its stored
payoff matching the product; and the public-UI contract.

Six OBSO tests asserted the *removed controls*. They were rewritten, not
deleted, to assert the new invariants — that the public control no longer
offers OBSO, and that OBSO appears in the page only inside the legacy
disclaimer. The OBSO export, parity and role-blindness tests are untouched and
still pass.

## Remaining limitations

* The solver's 18 real release actions have export support but no scene with
  valid stage-3 semantics, so the mode stays unavailable.
* No genuine solver trajectory for any public scene.
* ~~The legacy OBSO payload is still copied into the build.~~ **Resolved**: the
  public build no longer ships it (see below).

## Legacy payload removed from the public build (2026-09-28)

`build()` gained `include_legacy_obso=False`, and the CLI a matching
`--include-legacy-obso`. `python -m demo_viz.web.build` now produces the
solver-native site with no OBSO surfaces.

| | before | after |
| --- | --- | --- |
| Public build total | 12.82 MB | **7.34 MB** |
| Initial load | 202 KB | **202 KB** (unchanged) |
| Scenes / release / solver | shipped | shipped, untouched |
| OBSO surfaces | 5.48 MB | **0** |

**No numerical coverage was traded for the size.** The two OBSO browser tests
that need a served site pass `include_legacy_obso=True`, so they exercise the
same parity they always did. Every other OBSO test — export integrity, the
role-invariance pair, the score-grid caveat — never needed a built site and is
unchanged. `reference_obso.py`, `obso.js` and `export_obso.py` are all still
here, and `web_data/obso/` is still generated and committed.

Five tests were added to hold the line: the default build ships no OBSO, the
flag puts it back, the difference is the size claimed, the solver-native
payloads survive the change, and no reachable code path can request the missing
payload — the URL now accepts only `space` and `gain`, and both `obsoSurface`
call sites stay behind a mode the public control cannot produce.

The Pages sanity check now **asserts** the public artifact has no
`data/obso/`, so a future change that reintroduces it fails the build rather
than quietly adding 5.5 MB.

307 tests pass.

---

# Paper-story alignment — Phase A, and where it stopped (2026-09-29)

`PAPER_STORY_TRACE.md` traced every abstract-facing quantity to code.
`ABSTRACT_DEMO_MAPPING.md` maps each abstract claim to a demo feature and its
status. `DELTA_EXPORT_REQUIREMENTS.md` states exactly what would unblock the
equilibrium half.

## The finding

**About half the abstract is implemented; the other half is prose.**

Implemented and real, in the stage-3 solver: equilibrium value, the certified
best-response gap, per-decision mixed policies for both sides, pure-vs-mixed
("torn", likeliest move below 0.8), the football naming of compass moves, and
the "solver answer" — the *modal line*, where at every decision both sides take
the action their policy weights most, explicitly *"an illustration of the
policy, not a sample from it"*.

**Not implemented anywhere, in either repository:** relative rank among
feasible counterfactuals, similarity to the optimal action, regret against an
observed action, frame-by-frame player evaluation, and clip-level aggregation
of player scores. The only `regret` in the tree is
`dynamic_response_game.normalized_option_regret`, a normalised min over a
defender-option cost matrix in the v0.1 lineage — scored by the OBSO stack this
demo deliberately demoted, and not a comparison of an observed action to an
optimum.

Static-vs-responsive has no single-scale implementation either: three candidate
readings exist, on three different scales, and none is labelled that way in
code. Choosing one is a team decision.

## What this pass therefore did, and did not, do

Per the instruction to stop before inventing the missing layer, this pass made
only the sanctioned hierarchy change:

| | before | after |
| --- | --- | --- |
| Analysis order | Scene → Off-ball effect → **Solver pass model** → Player | Scene → Player → **Off-ball context** → *Action value details* |
| Release chain | a top-level section, always open | a **collapsed drill-down**, with a one-line `payoff 0.789` summary on its header |
| Off-ball pair | "Off-ball effect" | "Off-ball context" — it explains why the scene matters, not how the player is scored |

The pass-model computation is unchanged and still exact. It simply no longer
reads as the contribution, which is what the abstract asks for.

**No story modes were built.** Two of the three proposed modes (Counterfactual,
Evaluation) have no data and no metric behind them; shipping them would be
controls that do nothing.

## The one blocker

The demo cannot tell the paper's evaluation story because **the evaluation
story is not implemented**. No export unblocks it — it needs a defined method
first. Everything else is one Delta artifact away.

309 tests pass.

---

# The final paper-story architecture (2026-09-29)

Branch `chani-ssac-demo-final-story`, from `8841b31`.

The previous pass established that half the abstract has no implementation.
This pass builds the interface the abstract describes **anyway**, with typed,
validated slots where the missing quantities will land — so that integrating
them is an adapter change, not another redesign.

## The interface

Four story modes over one scene, one pitch and one timeline:

| mode | question | state today |
| --- | --- | --- |
| **Observed** | What happened? | fully populated |
| **Counterfactual** | What else could the player have done? | structure live; kinematic reachable area is real, the evaluation action set is pending |
| **Evaluation** | How good was the observed action? | four reserved fields, each showing its own reason |
| **Game solution** | What does the equilibrium recommend? | component complete, no artifact covers a tracked scene |

The right panel follows the paper's argument:
scene → player → current decision → counterfactual → player evaluation → clip
summary → game solution → off-ball context → *action value details* → source.
Off-ball context and the release chain come last, because both are replaceable
components in the abstract and neither may read as the contribution.

## The contract

[`PAPER_STORY_SCHEMA.md`](PAPER_STORY_SCHEMA.md) documents it. The invariant:

```
availability == "available"   =>   value is not None  and  source is set
availability != "available"   =>   value is None
```

Enforced at construction, and re-checked by `validate_payload` on data from
outside the process. **A placeholder number cannot be constructed.** Five
distinct availability states, each with its own sentence — never a dash.

## The boundary

`demo_viz/paper_story/adapter.py` is the only file that will know anything
about the updated research code's output. The browser reads
`data/story/<scene>.json` and the shipped vocabulary in `contract.json`; a test
asserts the site never names `rows.jsonl`, `policies/`, a run directory or a
state index in code.

`tests/test_paper_story.py::FutureIntegrationTests` builds a scene from a stub
evaluation source and asserts the values arrive with source and definition
version attached, through exactly the code path the real pipeline will use —
including a metric named `mean_regret`, a name no interface file mentions.
If that test ever needs the UI changed to pass, the boundary has leaked.

## What was not built

No invented rank, similarity, regret, static baseline, aggregation or optimal
trajectory. No example numbers, no fixture data in any public payload (a test
enforces both). The development harness for the Game solution component lives
outside the copied site folder and a test asserts the built site contains one
HTML file.

375 tests pass. Initial load 204 KB, total 8.03 MB.
