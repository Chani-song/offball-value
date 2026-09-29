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
