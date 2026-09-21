# Interactive explorer — implementation report

Built on top of the existing `demo_viz` renderer, on branch `chani-fancy-viz`.
Nothing was pushed, no repository history was rewritten, and no scientific
method in `src/offball_value/` was modified.

---

## 1. What already existed, and what I reused

The rendered-video system was already in place, so the app is an extension
rather than a second implementation. Reused unchanged:

| Existing piece | How the app uses it |
| --- | --- |
| `demo_viz/scene.py` — canonical `Scene` / `ScenePlayer` | the app's only data model; the whole UI talks to this and nothing else |
| `demo_viz/loader.py` — `load_scene`, `list_scenes`, `figure_config_for` | every scene the app opens, including the `scenes.json` presentation overrides |
| `demo_viz/annotations.py` | the scene dropdown and Annotation mode |
| `demo_viz/sources/*` | IDSSE window reading, the `.npz` window cache, the pipeline record adapter |
| `demo_viz/palette.py` | role colours, surfaces, the CVD-checked palette — the app's visual language is the videos' |
| `demo_viz/quantities.py` | `_frame_at`, `_velocities`, `baseline_tracks`, `marking_series`, `_kinematic_onsets`, `peak_acceleration_time`, `compute_surfaces` |
| `demo_viz/animate.py` — `render_still` | the app's **PNG** button re-renders the current interactive pick through the video renderer |
| `demo_viz/story.py` | the storyboard that PNG export needs |
| `src/offball_value/*` | all of the science: `goal_weighted_influence`, `fernandez_influence`, `dynamic_marking`, `run_onset` |

Two things did **not** exist and had to be built: a way to change roles without
recomputing everything, and a ranking that answers "who reacts / who gains".

### The one real engineering problem

`goal_weighted_influence.target_residual_influence` recomputes every defender's
influence surface for every target it is asked about. That is fine for one
annotated beneficiary in a batch render, and far too slow for an app where the
user clicks a different beneficiary every second.

`demo_viz/core/influence.py` computes each player's Fernández surface **once**
per sampled frame and keeps the defensive sum, so a residual field for any
target is one multiply, and swapping a defender for its no-reaction baseline is
one subtract-and-add. The arithmetic is the repository's, unchanged:

```
intrinsic = target_influence * goal_weighted_space_value(target)
uncovered = exp(-k * sum_of_defender_influences)
residual  = intrinsic * uncovered
```

`tests/test_demo_viz_app.py::InfluenceCacheTests::test_matches_the_repository_residual_exactly`
asserts it reproduces `target_residual_influence` to `atol=1e-12`. Measured on
the hero scene: cache build **40 ms**, then **0.06 ms** per residual lookup, and
**5 ms** to rebuild the whole pitch figure.

---

## 2. Files added and changed

### Added

```
demo_viz/core/influence.py        factored per-frame influence cache
demo_viz/core/role_logic.py       defender / beneficiary rankings, run onset, auto triplet
demo_viz/core/selection.py        role selection state + the click rules
demo_viz/core/figures.py          Plotly pitch and side chart
demo_viz/app/interactive_app.py   Dash entry point
demo_viz/app/components.py        layout and dark theme
demo_viz/app/callbacks.py         clicking, scrubbing, hints, exports
demo_viz/app/state.py             scene/cache service, scene list, pipeline discovery
demo_viz/data/suggested_triplets.example.json   example triplets (see §5)
tests/test_demo_viz_app.py        26 tests
demo_viz/exports/app_*.png        screenshots of the three modes
demo_viz/APP_REPORT.md            this file
```

### Changed

* `demo_viz/quantities.py` — `baseline_tracks` gained an optional `defender_ids`
  argument so the app can build a baseline for a defender the user just picked
  rather than only the annotated one. Backward compatible; the video path is
  untouched.
* `demo_viz/README.md`, `README.md`, `pyproject.toml` — docs and an `app` extra.

Nothing in `src/offball_value/`, `scripts/` or `examples/` was touched. Full
suite: **119 tests, OK** (93 existing + 26 new).

---

## 3. How role selection works

State lives in one JSON-serialisable `Selection` (`core/selection.py`) held in a
`dcc.Store`, so every callback reads and writes the same object.

`Selection.apply_click(scene, player_id)` is the whole rule set, and it is a pure
function so it is unit-tested directly rather than through the browser:

1. a player who already holds a role **loses it** — a second click is always undo;
2. the clicked **side wins** over the armed Pick target, so clicking a defender
   while *Runner* is armed sets the defender rather than mis-assigning;
3. an attacker fills the **runner** slot when empty, otherwise **beneficiary**.

The Pick target then advances (runner -> defender -> beneficiary), so the normal
path is three clicks with no mode fiddling. Clicks resolve exactly: every player
marker carries its player id in Plotly `customdata`, so there is no
nearest-neighbour guessing.

Rows in the hint list carry the same behaviour through a pattern-matching
callback, which is what makes "what changes if I pick that defender instead"
a single click.

### Where the hints come from

* **Reacts most** — for each opponent, the mean alignment of their motion with
  the moving goal-side point in front of the runner
  (`dynamic_marking.moving_goal_side_target`), blended with mean proximity.
  A transparent geometric heuristic, **not** a model of intent. On the hero
  scene it ranks the annotated defender 4th of 11, which is an honest
  disagreement rather than a bug.
* **Gains most** — teammates ranked by the Opened-space difference itself. On the
  hero scene, given the annotated runner and defender, it puts the annotated
  beneficiary (#34 N. Gavory) first. That is asserted as a test.

---

## 4. Scene sources supported

| Source | Status |
| --- | --- |
| `shot_annotations.xlsx` | the **88 rows that carry a complete runner/defender/beneficiary triplet** (5 strong, 40 medium, 43 low), sorted strongest first; Annotation mode preloads the triplet and shows the notes |
| IDSSE tracking | loaded through the existing `sources/idsse.py` window reader and `.npz` cache |
| Synthetic | `synthetic · smoke test` in the dropdown; the app works end to end with no data at all |
| Pipeline triplets | adapter implemented, see below |

---

## 5. Pipeline mode — implemented, but there is no pipeline output

The mode is fully wired: it reads the first file it finds of

```
data/processed/pipeline_triplets.json      <- a real pipeline would write here
demo_viz/data/pipeline_triplets.json
demo_viz/data/suggested_triplets.example.json
```

resolves shirt numbers *or* DFL person ids against the roster, preloads the
triplet, and lets you edit it by clicking.

**This repository still produces no pipeline triplets.** I did not invent any.
The example file that ships is generated by
`demo_viz.core.role_logic.auto_triplet` — the app's own geometry — so the mode is
demonstrable on a machine that has never run the pipeline. Its `_comment` says
so, and the UI labels the selection *from example file* rather than
*from pipeline file*. Drop a real file at the first path and it wins.

---

## 6. Grounded vs illustrative

**Measured** — from data, or from code already in this repository:

* player and ball positions, speeds, trails, displacements: IDSSE 25 Hz tracking
* the shaded space and both curves: `goal_weighted_influence.target_residual_influence`
  via the factored cache, bit-for-bit identical to calling the repo function
* **Opened space**: that quantity minus itself with the selected defenders
  replaced by a no-reaction baseline
* **Gains most**: teammates ranked by exactly that difference
* **run starts**: `run_onset.detect_kinematic_run_onsets` where it fires

**Explanatory — labelled on screen:**

* the **Ghost** marker and the *no reaction* curve are a what-if device: the same
  formula re-evaluated with the defender held at their pre-run position. Not a
  learned or optimised defensive best response; this repository has none.
* **Reacts most** is a geometric heuristic, not a model of defensive intent.
* **run starts** falls back to a *peak acceleration* cue when no onset is
  detected, and the panel names which fired.
* the annotation **Notes** are a qualitative human comment used for scene choice
  and initialisation — never a numeric ground truth.

**Not shown at all:** calibrated xT, pass probability, dribble probability, any
learned threat surface. None exists here.

One consequence worth stating: because before the run starts the baseline
defender *is* the observed defender, Opened space is exactly `0.0` at those
frames. That is correct, not a broken readout, and the app opens on the
peak-gain frame so the first thing you see is not a zero.

---

## 7. Two bugs the tests caught

1. **The counterfactual leaked backwards in time.** The baseline defender's
   influence was computed with the pre-run velocity at *every* frame, including
   before the freeze, so the two curves differed slightly before the run had
   begun. Now frames at or before the freeze reuse the observed surface, so the
   curves are identical until the run starts — matching what `baseline_tracks`
   already promised in its docstring.
2. **PNG export mutated the shared scene.** Writing a still swapped the roles and
   surfaces onto the cached `Scene` and left them there, so switching back to
   Annotation mode afterwards would restore the *exported* roles instead of the
   annotation. Now snapshotted and restored in a `finally`.

---

## 8. Known limitations

* **Full-pitch view has gutters.** A 105x68 pitch in a 16:9 panel leaves side
  margins, and players are ~20 px. `Fit` crops to the picked players; it is off
  by default because cropping hides players you might want to click.
* **Playback is a browser timer.** `dcc.Interval` at 80 ms advancing two frames,
  so it runs at roughly 25 Hz on a good connection and slower on a loaded
  machine. It is a scrubber, not a video player — the MP4 exports are for that.
* **No GIF/clip export from the app.** `PNG` and `JSON` are instant; a clip would
  take 30–60 s inside a callback and block it. `demo_viz.render_scene --gif`
  already does this offline.
* **The app writes PNGs to `demo_viz/exports/`** as well as streaming them to the
  browser, so repeated exports accumulate files there.
* **One process, in-memory cache.** Scene bundles are cached in module state, so
  running the app with multiple workers would load each scene per worker. Fine
  for a local demo.
* **No side-by-side with the source clip.** The clips in `ALL_SHOT_CLIPS/` are
  themselves 5 Hz tracking plots rather than broadcast video, so the comparison
  is much less useful than it sounds; `demo_viz.compare_clip` already does it
  offline for the video path.

---

## 9. Next best improvements

1. **Compare two triplets side by side** — hold one selection, pick a second, and
   show both Opened-space curves on one chart. The cache makes this nearly free;
   it is UI work only.
2. **Brush the time axis to pick the window** the gain is averaged over, instead
   of fixing it at "everything after the run starts".
3. **A defender-by-beneficiary matrix** for the selected runner — every defender
   against every teammate, one cell per gain. About 11x11 cache lookups, so it
   would render instantly and would answer the "what if" questions in one view.
4. **Persist selections** to `scenes.json` so an interesting pick found by
   clicking becomes a curated demo scene.
5. **Settle the counterfactual question** from `OVERNIGHT_REPORT.md` §8 — the app
   makes it much easier to probe, because you can now watch the sign flip as you
   change the defender.
