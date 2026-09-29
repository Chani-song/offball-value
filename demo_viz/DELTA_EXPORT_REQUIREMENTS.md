# What the demo needs exported from Delta

For Kyuhyeok. Everything below already exists on Delta as a by-product of the
stage-3 runs; nothing here asks for new computation or a methodology change.

With these files the demo can show the equilibrium policy, the "solver answer
vs actual movement" comparison, and pure-vs-mixed behaviour — rows 1, 4, 5 and
9 of `ABSTRACT_DEMO_MAPPING.md`. The adapter, the policy reader and the
renderer are already written and tested against real artifacts.

## 1. One solved run directory

Source on Delta:

```
/work/hdd/bbmr/kseo1/offball-out/stage3_passer2on1_agile      (2v1)
/work/hdd/bbmr/kseo1/offball-out/stage3_fixedpasser_agile     (3v1)
```

Per run, these exact files:

| File | Why |
| --- | --- |
| `manifest.json` | `config.steps`, `step_seconds`, `physics_step`, `directions`, `passes`, `physics`. The reader refuses to assume the action encoding and reads it from here. |
| `rows.jsonl` | `value`, `certificate`, `root_defender`, `root_attack`, `mixed_states`, `rollouts`, `scenario`, `stratum`, `index` |
| `states/state_NNN.json` | same per state, without rollouts |
| `policies/state_NNN.npz` | `defender_k`, `attack_k`, `pass_index_k`, `terminal_release`, `metadata` — **required**: the per-decision policies are only here |
| `starting_states.json` | `provenance` per state: `match_id`, `match_label`, `onset_frame_id`, `scene_dir`, `runner_id/name`, `defender_id/name`, `carrier_id/name`, `beneficiary_id/name` — **this is what lets a state be joined to a demo scene** |

A **single state** is enough for a first end-to-end demonstration. Ten would
let the Game solution view be shown across the showcase.

## 2. The join key — the part that matters most

The demo identifies scenes as `J03WOH:shot_006_P1_1054`. The solver states are
keyed by run onset, not by shot. `starting_states.json[].provenance` is the
only place carrying both sides. Please confirm, per exported state:

* `match_id` and `onset_frame_id`
* `runner_id`, `defender_id`, `beneficiary_id` as **player ids or shirt
  numbers**, matching `shot_annotations.xlsx`

Without these the mapping stays unresolved and the artifact cannot be attached
to a scene. **Do not** send states we cannot join — an unjoinable artifact
would have to sit unused rather than be guessed onto a scene.

## 3. Scene data for the five solver-derived showcase entries

`S46, S48, S53, S58, S66` are curated and reviewed but have no published
tracking, so they cannot be played at all. They are run-onset states, not shot
annotations, so they are not among the 45 exported scenes.

Needed per entry: the same per-frame tracking the existing scenes carry
(player and ball xy at 25 Hz over the clip window, with shirt numbers and
sides), at the run-onset-centred timing those entries use. The export path
`demo_viz/web/export_data.py` shows the exact shape.

## 4. Optional, if it already exists

* `passer2on1_results_agile.csv` / `fixedpasser_results_agile.csv` from
  `summarise_stage3.py` — `fsplit0`, `root_top`, `root_second`,
  `path_points`, `split_t{k}`, `top_t{k}`. Saves recomputing the summary.
* The stage-3 **input** states file (`passer2on1_states_agile06_final.json`)
  for the background tracks.

## 5. What is NOT being asked for

* No raw licensed tracking beyond the 21 curated scenes.
* No re-run, no threshold change, no relaxation of the pipeline gates.
* No new metric. In particular the demo does **not** need — and must not be
  given — a hand-made "rank" or "similarity" number; see §6.

## 6. The separate, larger blocker

`PAPER_STORY_TRACE.md` finds no implementation, in either repository, of the
abstract's evaluation layer: relative rank among feasible counterfactuals,
similarity to the optimal action, regret against the observed action,
frame-by-frame player scoring, or clip-level aggregation.

Exporting the artifacts above will **not** unblock those. They need a defined
method first, and that is a team decision, not an export. The nearest existing
machinery is `markov.policy_transfer_bounds`, which reprices a fixed *policy*
against fresh best responses — not a single observed action, and only for
comparing payoff models on an identical scenario.


---

## 7. What happens to an artifact once it lands

The demo now has a place to put one. Dropping a run directory into a local
input folder and regenerating is three commands:

```bash
# 1. point the submission manifest at it: "<run directory>#<state index>"
#    in demo_viz/data/submission_scenes.json, for the scene it belongs to
# 2. read the artifact and normalise it into the shape the browser draws
.venv/bin/python -m demo_viz.web.export_solver --from-manifest
# 3. re-derive the paper-story payloads, which pick the solver file up
.venv/bin/python -m demo_viz.web.export_paper_story
.venv/bin/python -m demo_viz.web.build
```

The Game solution mode then populates for that scene: equilibrium value,
certificate gap, per-side support counts, weighted policy bars, and the
modal-policy illustration. The component is finished and is exercised against
a real solver study state by `demo_viz/web/story_harness.html`; only the
attachment to a tracked scene is missing.

**Nothing is attached by fuzzy match.** `paper_story.adapter.equilibrium_for`
looks for an exported file under the scene's own key and reports
`artifact_missing` otherwise — there is no name-similarity fallback, and
`tests/test_paper_story.py` asserts that no tracked scene currently claims an
equilibrium. When one does, that test fails on purpose, so the trace document
and the claim get updated together.

The export refuses an unusable reference rather than guessing:
`export_solver.py` skips a manifest entry whose `solver_artifact` is malformed
or whose directory is absent, and says which entry it skipped.


---

## 8. If the run used pass model A (`c6423d4`, 2026-09-28)

`origin/kyuhyeok-dev` @ `c6423d4` shares the fitted coefficients for pass model
A (`A_all.json`, six `A_without_<match>.json`, and the `A_crossfit.json`
router). Nothing about this request changes, but two details are worth knowing
before an artifact arrives.

**A `passA` run's manifest carries an extra block.** `run.py:model_router`
writes `pass_model_router` into `manifest.json`: the router's own JSON plus a
sha256 per model file. Our reader
(`demo_viz/solver/adapter.py:read_manifest`) requires only `config` and ignores
unknown keys, so such a manifest loads unchanged — and the sha256 map is exactly
the provenance we would want to display. **Still send the manifest**; do not
strip the block.

**Say which pass model the run used.** The canonical
`stage3_passer2on1_agile.sbatch` uses `andrew/models/experimental_pass.json`;
the `stage3_*_passA` variants use `A_crossfit.json`. These are different
completion models — A prices the arrival race and reads velocities, the other
is a positions-only 33-feature logistic — so a value from one is not comparable
with a value from the other. The demo names the model it is showing, and
`PASS_MODEL_TRACE.md` documents the positions-only one; if a `passA` run is
sent, that document needs a sibling section before the numbers go on screen.

**Not shared, and not being asked for:** `B1_crossfit.json`, `B1SB_all.json`,
`report.json` and `passes.jsonl` remain ignored. Only the `A_*.json` files were
un-ignored.

**This does not touch §6.** Pass model A is a completion proxy. The evaluation
layer is still unimplemented, and no export of coefficients can change that.
