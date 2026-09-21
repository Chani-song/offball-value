# Overnight build report — `demo_viz`

Built 2026-09-21 on branch `chani-fancy-viz` (branched from `kyuhyeok-dev`).
Nothing was pushed; no repository history was rewritten; no scientific method
in `src/offball_value/` was modified.

---

## 1. What I found in the repository

`offball-value` is a research prototype around a *local off-ball game*: one
runner, one candidate defender response, and the attacking options whose
control changes with that defender's allocation. The README is explicit that
it is not yet a validated off-ball value metric.

Relevant existing code, all of which the renderer now calls rather than
reimplements:

| module | what it gives the demo |
| --- | --- |
| `bundesliga.py` | IDSSE XML parsing, the metre/centre-origin coordinate convention, `FIELD_LENGTH=105`, `FIELD_WIDTH=68`, `FPS=25`, roster and shirt numbers |
| `run_onset.py` | `detect_kinematic_run_onsets(frame_ids, xs, ys)` — a pure function returning acceleration / direction-change / check-run onsets |
| `fernandez_influence.py` | the Fernández & Bornn (2018) player-influence Gaussian |
| `goal_weighted_influence.py` | `target_residual_influence(...)` — goal-weighted space an attacker retains after defensive coverage; the repo labels this a transparent v0.1 operationalization, not a calibrated threat model |
| `dynamic_marking.py` | `moving_goal_side_target`, `marking_sample` — goal-side marking geometry |
| `pass_dynamics.py` | `VelocityEstimate`, the velocity-window convention |
| `local_game_structure.py`, `steering_reachable.py`, `empirical_action_space.py` | the structural audit; read but not called by the demo |

What the repository does **not** currently produce, and which the renderer
therefore does not draw: calibrated xT, pass probability, dribble probability,
a learned threat surface, or a learned/optimised defensive best response.
`Scene` carries typed hooks for the first four; they stay `None`, and a test
asserts it.

### Existing artifacts I did not disturb

`examples/research_audit/` holds three self-contained HTML demos and
`manifests/confirmed_core_scenes.json` (8 human-confirmed scenes, 63 frames
each at 12.5 Hz, players as `[id, team, x, y, name]`, **no shirt numbers and no
ball**). That payload cannot be joined to the spreadsheet's shirt numbers, so
the demo reads the raw IDSSE XML instead. The three existing render scripts
still run unchanged.

---

## 2. Data formats and how the join works

### Inputs found on this machine

| what | where | size |
| --- | --- | --- |
| IDSSE positions XML, 7 matches | `~/idsse_shots/idsse-data/DFL_04_03_positions_raw_observed_*.xml` | ~350–420 MB each |
| IDSSE matchinformation XML | same directory | ~12 KB each |
| IDSSE event XML | same directory | ~700 KB each |
| `shot_annotations.xlsx` | `~/Research/offball_demo/shot_annotations.xlsx` | 170 rows |
| shot clip videos | `~/idsse_shots/output/ALL_SHOT_CLIPS/*.mp4` | 172 clips |
| 5 Hz overlay cache from the annotation app | `~/idsse_shots/output/tracking_overlay_cache/<match>/<clip>.json` | used only to validate |

`data/raw/` inside the repository is empty (and git-ignored), so
`demo_viz/config.py` searches `OFFBALL_IDSSE_DIR`, then
`data/raw/bundesliga-integrated/`, then `~/idsse_shots/idsse-data`.

### Positions XML

```xml
<FrameSet GameSection="firstHalf" MatchId="DFL-MAT-J03WOH"
          TeamId="DFL-CLU-00000P" PersonId="DFL-OBJ-00003X">
<Frame N="10000" T="2022-08-26T16:32:09.720+00:00"
       X="-0.59" Y="8.72" D="0.00" S="0.00" A="0.00" M="1"/>
```

X and Y are already metres with the origin at the centre spot. `S` is speed in
km/h (the loader converts to m/s). One `FrameSet` per object per half; the ball
is `TeamId="BALL"`.

### The join: spreadsheet row -> tracking window

The spreadsheet stores `period` and `period_seconds` of the shot. Frame number
is `section_start[period] + period_seconds * 25`, and **the first frame of each
half is 10000 and 100000 respectively** in every one of the seven matches. See
the bug section below — getting this wrong is silent and destructive.

Shirt numbers resolve to DFL person ids through `matchinformation.xml`,
restricted to the shooting team for runner and beneficiary and to the opponent
for the drawn defenders. The window is the same one the human annotator saw in
the clip app: 8 s before the shot to 3 s after.

### `shot_annotations.xlsx`

170 rows: 82 `ignore`, 43 `low`, 40 `medium`, **5 `strong`**. All five strong
rows carry a complete runner / defender / beneficiary triplet and all five now
render. 40 of the `medium` rows are also complete and renderable.

---

## 3. Scenes located and verified

All five `strong` scenes load, with every annotated shirt number resolved to a
tracked player:

| clip | shape | runner | pulled defender(s) | beneficiary |
| --- | --- | --- | --- | --- |
| `J03WOH:shot_006_P1_1054` | 1R-1D-1B | #7 Kristoffer Peterson | #11 K. Faber | #34 N. Gavory |
| `J03WOH:shot_010_P1_1759` | 2R-2D-1B | #10 D. Ginczek, #7 Peterson | #23 S. Breitkreuz, #11 K. Faber | #9 Dawid Kownacki |
| `J03WOY:shot_002_P1_0580` | 1R-2D-1B | #23 Shinta Appelkamp | #6 D. Dressel, #34 Lukas Fröde | #9 Dawid Kownacki |
| `J03WOY:shot_011_P2_0903` | 1R-2D-1B | #18 J. Verhoek | #25 M. Zimmermann, #15 T. Oberdorf | #34 Lukas Fröde |
| `J03WPY:shot_018_P2_1375` | 2R-2D-1B | #9 Kownacki, #25 Zimmermann | #4 J. Lawrence, #13 E. Wekesser | #11 F. Klaus |

### Independent validation of the tracking join

The annotation app cached the exact 5 Hz, normalized tracks the human reviewer
saw. I reconstructed those tracks from the raw XML through the new loader and
compared them:

* **rosters**: identical attack/defend shirt sets on all 14 scenes tested
  (5 strong + 9 medium).
* **positions**: median error **0.000 m**, p99 **0.001 m** over ~176 sampled
  player-frames per scene, once the ±1-frame phase offset between the 25 Hz and
  5 Hz sampling is allowed for. The only scene that disagreed was one where I
  had deliberately trimmed the window by 1 s; restoring the default window
  brought it to 0.000 m as well.

So the demo is reading the same data the annotator reviewed, and the y-axis
flip that `kloppy` applies in the annotation app is accounted for.

Reproduce:

```bash
.venv/bin/python - <<'EOF'
import json, pathlib, numpy as np
from demo_viz.loader import load_scene
CACHE = pathlib.Path("~/idsse_shots/output/tracking_overlay_cache").expanduser()
s = load_scene("J03WOH:shot_006_P1_1054", quantities=False)
d = json.load(open(CACHE / "J03WOH" / "J03WOH_shot_006_P1_1054.json"))
errs = []
for k in range(0, d["frame_count"], 7):
    for side, key in (("attack", "a"), ("defend", "d")):
        for shirt, xy in d["frames"][k][key].items():
            p = s.by_shirt(shirt, side)
            ref = np.array([xy[0] * 105 - 52.5, (1 - xy[1]) * 68 - 34])
            errs.append(min(np.linalg.norm(p.xy[i] - ref)
                            for i in range(max(0, k * 5 - 2), min(s.n_frames, k * 5 + 3))))
print("median %.4f m   p99 %.4f m" % (np.median(errs), np.percentile(errs, 99)))
EOF
```

---

## 4. Files added and changed

### Added — `demo_viz/` (new, self-contained)

```
demo_viz/__init__.py              package entry points
demo_viz/config.py                path discovery (env var -> conventional locations)
demo_viz/palette.py               role colours + a runnable CVD separation check
demo_viz/scene.py                 the canonical Scene / ScenePlayer objects
demo_viz/annotations.py           shot_annotations.xlsx adapter
demo_viz/quantities.py            optional quantities, each from a named repo function
demo_viz/story.py                 beats, captions, layer strengths, the causal chain
demo_viz/animate.py               still / MP4 / GIF / contact-sheet export
demo_viz/interactive.py           self-contained Plotly HTML viewer
demo_viz/loader.py                one entry point for every scene source
demo_viz/scenes.json              presentation-only per-scene overrides
demo_viz/sources/idsse.py         fast XML window reader + on-disk cache
demo_viz/sources/annotation_scene.py  spreadsheet row -> Scene
demo_viz/sources/pipeline.py      pipeline-predicted triplets -> Scene
demo_viz/sources/synthetic.py     generated smoke-test scene
demo_viz/render/pitch.py          pitch geometry
demo_viz/render/layers.py         trail, tether, wake, ghost, lane, run vector, badge
demo_viz/render/camera.py         smooth following viewport
demo_viz/render/labels.py         collision-aware label placement
demo_viz/render/figure.py         full-frame composition
demo_viz/render_scene.py          CLI: one scene -> artifacts
demo_viz/view_scene.py            CLI: interactive viewer
demo_viz/export_preview.py        CLI: batch export + manifest
demo_viz/compare_clip.py          CLI: side-by-side with the source clip
demo_viz/README.md                architecture, grounding, how to run
demo_viz/OVERNIGHT_REPORT.md      this file
demo_viz/PLAN.md                  the plan written before implementation
```

### Added — tests

```
tests/test_demo_viz.py            26 tests
```

### Changed — three files, none of them scientific

* `.gitignore` — one line, `demo_viz/cache/`.
* `README.md` — a short section pointing at `demo_viz/`.
* `pyproject.toml` — an optional `demo` extra declaring `plotly`, which only the
  interactive viewer imports.

Nothing in `src/offball_value/`, `scripts/` or `examples/` was touched, and the
pre-existing suite still passes: `93 tests, OK` (67 existing + 26 new).

### Written at runtime (git-ignored territory, safe to delete)

```
demo_viz/cache/clocks/*.json      first frame of each half, per match
demo_viz/cache/windows/*.npz      one extracted tracking window each, with a .json sidecar
demo_viz/exports/*                the artifacts below
```

---

## 5. Exact commands to reproduce the outputs

```bash
cd offball-value
./bootstrap.sh                                     # python 3.11 venv + editable install
# (this build used: uv venv --python 3.11 .venv && uv pip install -e . plotly)

.venv/bin/python -m demo_viz.render_scene --paths  # confirm the data was found
.venv/bin/python -m demo_viz.render_scene --list   # list annotated scenes

# every artifact for all five 'strong' scenes (~10 min on a laptop; writes manifest.json)
.venv/bin/python -m demo_viz.export_preview --all-strong

# one scene, everything
.venv/bin/python -m demo_viz.render_scene \
    --scene J03WOH:shot_006_P1_1054 --png --mp4 --gif --html --beats

# interactive viewer in a browser
.venv/bin/python -m demo_viz.view_scene --scene strong:0

# no data at all on this machine
.venv/bin/python -m demo_viz.render_scene --scene synthetic --png --gif

# checks
.venv/bin/python -m demo_viz.palette               # colour separation report
.venv/bin/python -m unittest tests.test_demo_viz -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Useful flags: `--half` (960×540, same layout), `--wake {residual,delta,geometric,off}`,
`--baseline {hold,drift}`, `--freeze {onset,reaction,start}`, `--no-camera`,
`--no-minimap`, `--no-chart`, `--no-surfaces`, `--grid`, `--every`, `--stride`.

---

## 6. What works today

* **Data discovery** that finds the IDSSE XML, the spreadsheet, the clip videos
  and the overlay cache with no configuration on this machine, and via
  environment variables anywhere else.
* **A ~1 s window reader** over a 350 MB positions file, cached to `.npz`
  afterwards (3 ms on the second call).
* **All five annotated `strong` scenes**, including two second-half scenes,
  two multi-runner scenes and three multi-defender scenes.
* **Story mode**: five beats with captions, a persistent four-chip causal rail,
  eased layer transitions, a timeline with beat markers.
* **A following broadcast camera** that opens on a wide establishing shot and
  pushes in to the cast, with a full-pitch minimap inset keeping the 11-v-11
  context the close shot crops away.
* **Collision-aware labels** — role chips, ghost captions, field captions and
  distance readouts are placed against a live occupancy map, so they do not
  stack on each other or on players.
* **Layers**: runner trail and run vector, runner→defender tether with live
  separation, shaded open-space field, counterfactual outline, ghost defender
  with displacement, beneficiary reveal ring, ball trail, pass lane (drawn only
  within 34 m, so a 50 m "lane" is never implied).
* **A live chart** of the beneficiary's space, observed against the
  counterfactual, with the difference shaded and a hero number.
* **An honest provenance strip** in every frame tagging each layer
  MEASURED / HUMAN / EXPLANATORY.
* **Interactive HTML** (~1 MB) with a scrubber, play/pause, hover identities and
  the same provenance footer.
* **Exports**: `preview.mp4` (1920×1080, 25 fps, ~13 s), `preview.gif`,
  `preview_frame.png`, `preview.html`, five beat stills per scene, and a
  `manifest.json` recording the roles, beats, detector names and scores.
* **A before/after video.** The annotation app's clips in `ALL_SHOT_CLIPS/` are
  plain 5 Hz tracking plots of *exactly the same 11 s window*, which makes them
  a fair "before" panel — same data, same scene, same seconds.
  `demo_viz.compare_clip` renders the demo without its holds and stacks the two
  with ffmpeg into `*_before_after.mp4`.
* **Graceful degradation**: `--no-surfaces` or a scene with no quantities falls
  back to a geometric wake that is labelled illustrative in frame.
* **A synthetic scene** that exercises every layer with no data present.

---

## 7. Blocked, or limited by missing quantities

1. **No calibrated threat.** There is no xT, pass probability, dribble
   probability or learned value surface in this repository, so none is drawn.
   The hooks exist (`Scene.pass_probability`, `.dribble_probability`,
   `.threat_surface`) and a test asserts they stay empty.
2. **No learned defensive best response.** The counterfactual is a
   no-response baseline, not an optimised defence. It is labelled EXPLANATORY in
   every frame.
3. **The `reacting defender` beat often has no detector behind it.** The repo's
   kinematic onset detector fires for the annotated defender in only 2 of the 5
   strong scenes. The other three fall back to the geometric pursuit rule,
   which the beat table names explicitly.
4. **Two scenes have no runner onset in the window either**, and use the
   labelled peak-acceleration cue.
5. **`examples/research_audit/manifests/confirmed_core_scenes.json` cannot be
   joined to the spreadsheet** — it has no shirt numbers and no ball track. A
   small change to whatever writes it (emitting `shirt_number` and the ball)
   would let the demo render those eight confirmed scenes without touching the
   raw XML at all.
6. **No `data/raw/` inside the repository.** The demo reaches outside the repo
   to `~/idsse_shots/`. That is fine for a local demo but means the exports are
   not reproducible from a bare clone.

---

## 8. The measurement that matters most — and it is not the comfortable one

The demo's hero number is

```
Δ = residual goal-weighted space for the beneficiary (observed)
  − the same quantity with the reacting defenders not responding
```

Both terms come from `goal_weighted_influence.target_residual_influence`. If the
annotated causal story held under this v0.1 metric, Δ should be positive after
the defender reacts. Measured over the whole window, sampled every 10 frames on
a 2 m grid, for both no-response baselines:

| clip | baseline | max Δ | min Δ | mean Δ after reaction | fraction Δ>0 |
| --- | --- | --- | --- | --- | --- |
| `J03WOH:shot_006_P1_1054` | hold | **+4.2** | −2.5 | **+3.32** | 72 % |
| | drift | +2.2 | −9.1 | −6.05 | 55 % |
| `J03WOH:shot_010_P1_1759` | hold | +2.3 | −8.7 | −4.03 | 34 % |
| | drift | +2.5 | −5.4 | −1.41 | 31 % |
| `J03WOY:shot_002_P1_0580` | hold | +0.7 | −3.3 | −1.24 | 41 % |
| | drift | +0.2 | −3.3 | −1.24 | 28 % |
| `J03WOY:shot_011_P2_0903` | hold | +0.1 | −11.6 | −3.22 | 10 % |
| | drift | +3.8 | −4.6 | −0.63 | 21 % |
| `J03WPY:shot_018_P2_1375` | hold | +1.1 | −8.2 | −3.45 | 59 % |
| | drift | +0.5 | −10.1 | −4.77 | 59 % |

**Only one of five human-labelled `strong` scenes shows the expected sign.** In
the rest, a defender who had not responded to the runner would have left the
beneficiary with *more* goal-weighted residual space, not less.

I did not tune this away, and the renderer draws negative Δ in magenta exactly
as prominently as positive Δ in cyan. Two readings are available, and telling
them apart is research work, not rendering work:

* **The counterfactual is too blunt over a long window.** Freezing (or
  drifting) a defender for six to eight seconds of a moving attack does not
  isolate "did not react to the runner" — it removes that defender from the
  phase. A defender left behind the play has no influence near the beneficiary,
  which mechanically *raises* the counterfactual residual. Both baselines
  suffer from this; `drift` reduces it in two scenes and worsens it in two
  others. A short-horizon counterfactual — hold only from the reaction moment
  and read Δ over the next 1–2 s — would be the obvious next experiment, and
  `--freeze reaction` already lets you try it.
* **Or the target quantity is not the one the annotator saw.** The residual is
  goal-weighted around the *beneficiary's own position*. A beneficiary arriving
  at the near post with three defenders around them scores low on this measure
  at exactly the moment the annotator called the scene strong. What the
  annotator described — a lane, a cut-back corridor, a vacated channel — is a
  property of the *region the defender left*, not of the area around the
  receiver.

Either way this is a finding about the v0.1 operationalization, surfaced by
putting the number on screen. It is the single most useful thing the night
produced, and it is why the demo shows the counterfactual curve at all rather
than just a pretty heat blob.

---

## 9. Bugs and inconsistencies noticed (no scientific logic altered)

1. **Half-start frame numbers — the one that actually mattered.** My first
   window reader took the first `<FrameSet>` of each half as the half's start
   frame. That is wrong: a player substituted on at minute 60 has a second-half
   FrameSet starting tens of thousands of frames late. It silently produced
   windows offset by minutes. One scene came back with **zero** tracked
   players; another loaded 22 plausible-looking players from the wrong moment,
   which is far worse because it looks fine.
   `offball_value.bundesliga.load_bundesliga_frame_clock` already avoids this by
   reading the clock from the BALL track — I now do the same (minimum frame
   number per section, ball authoritative) and cache it. All seven matches give
   `firstHalf = 10000`, `secondHalf = 100000`, which is consistent with the
   `frame_id >= 100000 -> period 2` rule already hard-coded in
   `load_bundesliga_events`. **Worth hoisting that convention into one named
   constant**, since it is currently re-derived in two places with different
   methods.
2. **`load_bundesliga_frames` builds one frozen dataclass per object per frame**
   via `ET.iterparse`, and `with_player` copies the whole player dict on every
   insertion — O(n²) in players per frame. Fine for a handful of frames,
   expensive for an 11 s window. The demo uses its own line-streaming reader
   (~1 s for a 350 MB file) and does not touch the repo function.
3. **IDSSE `S` is km/h, not m/s.** Not a repo bug — the repo stores it raw and
   never uses it — but anything that starts treating `BundesligaObjectState.speed`
   as m/s will be 3.6× out. The demo converts on read and otherwise computes
   velocities by central difference.
4. **`goal_weighted_influence` recomputes every defender's influence surface for
   every target.** With one beneficiary and 22 players that is 22 Gaussian
   evaluations per frame that could be cached across targets. Not a
   correctness issue; it is the main cost when sampling many frames.
5. **The spreadsheet's `shot_result` is blank for some rows**, including the
   hero scene. The renderer treats it as optional. Worth checking whether those
   are genuinely missing or a write bug in the annotation app.
6. **`scripts/render_*.py` are not importable as modules** (no `__main__` guard
   consistency across them) — untouched, just noted.

---

## 10. Top next improvements

**Research side (someone else's call, not mine to make):**

1. Try the short-horizon counterfactual: `--freeze reaction` plus a 1–2 s
   evaluation horizon, and see whether the sign of Δ flips on the four scenes
   where it currently disagrees with the human label.
2. Measure the space in the *region the defender vacated* rather than around
   the beneficiary. The renderer already computes and draws that region (the
   `Δ > 0` contour); integrating it is a small addition to
   `goal_weighted_influence`, not a new model.
3. Emit `shirt_number` and the ball track into `confirmed_core_scenes.json` so
   the confirmed scenes and the spreadsheet share a key.

**Demo side:**

4. A scene gallery page — one HTML with all five scenes and a scene switcher.
   `export_preview.py` already writes a `manifest.json` with everything it needs.
6. Layer toggles in the Plotly viewer (the traces are already in a fixed order,
   so the buttons are mechanical).
7. The optional shallow-3D "value terrain" hero shot. Not attempted; it should
   not be built until the sign question in §8 is settled, because a dramatic 3D
   surface of a quantity that disagrees with the human labels would be the
   wrong thing to put in front of people.
7. Render the 40 renderable `medium` scenes in bulk and eyeball them as a
   contact sheet — cheap, and the fastest way to find more hero candidates.

---

## 11. References used

Design and technique, in rough order of usefulness:

* Karun Singh, *Introducing Expected Threat (xT)* — <https://karun.in/blog/expected-threat.html>.
  The structural idea I borrowed is pedagogical, not visual: build the
  explanation incrementally and let the reader step through it. That became the
  four-chip causal rail and the beat timeline rather than any copied graphic.
* Fernández & Bornn, *Wide Open Spaces* (MIT SSAC 2018) — already implemented in
  `src/offball_value/fernandez_influence.py`; the demo renders that
  implementation, and the citation is in the module docstring.
* `mplsoccer` documentation — <https://mplsoccer.readthedocs.io/> — consulted for
  pitch-marking dimensions and dark-pitch conventions. **Not added as a
  dependency**; the pitch here is drawn directly in `render/pitch.py` to keep the
  dependency list unchanged.
* Matplotlib `FFMpegWriter` / `PillowWriter` docs, and Plotly's
  `frames` + `sliders` animation API, for the export and viewer layers.
* The colour work is computed, not borrowed: `demo_viz/palette.py` implements
  OKLab ΔE and the Viénot (1999) dichromat simulation, and prints the
  separation of every mark pair in normal, protan, deutan and tritan vision.

---

## 12. Assumptions I made

* The spreadsheet's `team` column names the *attacking* team for that clip;
  runner and beneficiary are resolved against it and defenders against the
  opponent. Verified against the annotation app's own team assignment on all
  14 scenes tested.
* The annotated window (8 s before the shot, 3 s after) is the right default,
  because it is the window the human reviewer actually watched. Per-scene trims
  live in `scenes.json` and are marked presentation-only.
* Beat times should come from detectors where detectors fire, and from clearly
  named fallbacks otherwise — never from hand-authored timings. Any pacing
  adjustment is reported in the beat table as `[moved +x.xxs for pacing]`.
* "Attacking left-to-right" is a *view* transform only; every quantity is
  computed in the repository's raw coordinates.
