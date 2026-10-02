# Interactive explorer — scene coverage, role dock, and the static build

Third pass on `demo_viz`, on branch `chani-fancy-viz`. Nothing pushed, no
history rewritten, and no scientific method in `src/offball_value/` modified.

Three things changed: the demo now covers every human-confirmed **strong and
medium** scene, role selection moved into a **spatial dock** beside the pitch,
and there is a **dependency-free browser build** that GitHub Pages can serve,
with the residual-space arithmetic ported to JavaScript so a visitor can pick
any triplet and get the same numbers Python produces.

---

## 1. Architecture

```
demo_viz/
  core/            shared logic: influence cache, role rules, rankings, figures
  app/             local Dash explorer          -> python -m demo_viz.app.interactive_app
  web/
    site/          the static browser explorer  (html + css + 7 JS modules, no build step)
    export_data.py scenes -> compact browser JSON
    build.py       site + data -> one deployable folder
    validate.py    Python vs browser parity, and a latency benchmark
    harness.html   parity harness  (not published)
    bench.html     latency harness (not published)
  web_data/        exported scene JSON, committed
  web_build/       assembled site (generated, git-ignored)
.github/workflows/pages.yml
```

The two front ends share everything that matters. `core/selection.py` and
`web/site/js/selection.js` implement the same four click rules;
`core/influence.py` and `web/site/js/influence.js` implement the same
arithmetic; `core/role_logic.py` and `web/site/js/ranking.js` implement the same
two rankings. The duplication is deliberate and bounded: the browser must run
without Python, and the parity test keeps the pair honest.

### What was reused, not rewritten

| Existing | Used by |
| --- | --- |
| `demo_viz.scene.Scene`, `loader`, `annotations`, `sources/*` | both explorers and the exporter |
| `demo_viz.palette` | Dash, and mirrored into `web/site/js/palette.js` |
| `demo_viz.core.influence.InfluenceCache` | Dash, the exporter's onsets, and the parity reference |
| `demo_viz.core.role_logic` | Dash rankings, `Auto triplet`, and the example triplet file |
| `demo_viz.animate.render_still` | the Dash app's **PNG** button |
| `offball_value.{goal_weighted_influence,fernandez_influence,dynamic_marking,run_onset}` | all of the science, in both front ends |

---

## 2. Scene coverage

`scene_options()` now takes the effect labels to show and defaults to
`("strong", "medium")`. The dropdown lists **45 scenes** — every annotated row
with a complete runner / defender / beneficiary triplet at those two labels
(5 strong, 40 medium) — labelled `STRONG · <clip id> · <shape>`. Low and ignore
rows never appear by default. A `Strong` / `Medium` pair of checkboxes sits next
to the selector in both front ends; unchecking both is refused rather than
leaving an empty list.

Shapes covered: 26 × `1R-1D-1B`, plus multi-defender (`1R-2D-1B`, `2R-2D-1B`,
`1R-3D-1B`, `2R-3D-1B`, …) and multi-beneficiary (`1R-1D-2B`, `2R-2D-2B`) up to
one `4R-2D-4B`. `tests/test_demo_viz_web.py` asserts the clean case, a
multi-defender case and a multi-runner case are all present.

---

## 3. Role selection

The `Pick` radio group is gone. Beside the pitch there is now a dock:

```
ATTACK                 DEFENCE
  Runner  [ #7 … × ]     Defender [ #11 … × ]
      ⇅ Swap
  Beneficiary [ #34 … × ]
```

Runner and Beneficiary sit inside one **Attack** box with the swap control
between them, because the same attacker can hold either. Defender sits in its
own **Defence** box.

### How a player gets a role

`Selection.apply_click` is four rules in order, and it is a pure function so it
is unit-tested directly rather than through a browser:

1. an **armed slot** takes a compatible player, replacing whoever was there;
2. otherwise a player who already holds a role **loses it** — a second click is
   always undo;
3. otherwise the **clicked side** decides: a defender fills Defender;
4. an attacker fills **Runner** when empty, else **Beneficiary**.

So the quick path is three clicks with no mode fiddling, and the explicit path
(click the slot, then the player) always wins when you use it. Clicking a slot
arms it; clicking the × on a chip clears just that player; `⇅ Swap` exchanges
the two attacking roles in one gesture.

### Drag and drop

The **browser** build has real pointer-based drag and drop: press a player or a
chip, drag, and compatible slots light up; drop assigns. Dragging the Runner
chip onto the Beneficiary slot swaps them. A press without movement falls back
to a click, so nothing depends on the drag being recognised.

It is implemented with pointer events rather than the HTML5 drag API, because
HTML5 drag does not work on SVG elements across browsers. The **Dash** build
keeps click-to-arm only: Dash has no drag primitive, and the brief asked not to
trade reliability for the gesture.

### Valid-target feedback

While a slot is armed, the side that can fill it stays at full opacity and gets
a thin ring; the other side drops to ~20 %. No instructions, just the pitch
showing what is clickable.

---

## 4. Live update

Every role change re-renders trail, tether, ghost, opened-space field,
beneficiary lane, the scalar, the chart and the candidate rankings, with no
Apply step. Both front ends do it through the cached influence stacks, so a
change is a subtract-and-add rather than a recomputation.

Candidate lists are unchanged in substance: **Reacts most** (an exploratory
geometric heuristic on `dynamic_marking`'s goal-side target, **not** a pipeline
output and labelled as such in the docs) and **Gains most** (the opened-space
difference itself). Rows are clickable and do exactly what clicking that player
on the pitch does.

---

## 5. Static build and how the maths survives it

The browser is **not** fed pre-rendered fields. `web/site/js/influence.js`
implements the same three steps as Python:

```
intrinsic = target_influence * goal_weighted_space_value(target)
uncovered = exp(-k * sum_of_defender_influences)
residual  = intrinsic * uncovered
```

so any runner / defender / beneficiary the visitor picks is computed on demand.
This was possible because both surfaces are **parametric**: the Fernández
influence needs only a position, a velocity and the ball; the goal-weighted
value needs only the target's position and the attacking direction. Nothing had
to be exported except the tracks, which the browser needs anyway to draw
players. No field grids are shipped.

The one thing that would have been unreasonable to port is the run-onset
detector (~300 lines), so `export_data.py` precomputes a run-start frame for
**every outfield attacker** — 10 or 11 small entries per scene — and the browser
looks up whichever runner is picked. Velocities, the goal-side marking geometry
and both rankings are recomputed in the browser.

### Exported per scene

scene id, effect, shape, title, subtitle, notes, match clock, fps, frame count,
`t0`, pitch size, team names, attacking direction, per-player
`{id, shirt, name, side, gk, x[], y[]}`, the ball track, the annotated roles,
and the per-attacker run starts. Nothing else. A test asserts the payload
contains exactly the declared field list and no `/Users/`, `/home/`, home
directory, cache or source-data path.

---

## 6. Python vs browser validation

`python -m demo_viz.web.validate` picks random scenes, frames, beneficiary sets
(1–2), defender sets (1–2), freeze frames and baselines; computes the values in
Python; then runs the same cases through `js/influence.js` in headless Chrome
and compares. The browser is the thing under test, so the check executes the
JavaScript rather than re-reading it.

**160 cases across 16 scenes, worst disagreement:**

| quantity | absolute | relative |
| --- | --- | --- |
| factual residual space | 1.4e-13 | 4.1e-15 |
| counterfactual | 1.4e-13 | 4.1e-15 |
| gain (difference) | 7.1e-14 | 1.5e-09 |
| field peak | 1.7e-16 | 1.5e-15 |

That is float64 machine precision. There is no approximation and no
"illustrative" fallback: the browser shows the same numbers as the renderer.

The `gain` relative column is larger only because it is a difference of two
similar numbers and several sampled cases have a true gain near zero; the
absolute column is the meaningful one, and the comparison uses
`atol + rtol*|expected|` the way `numpy.isclose` does.

**Getting to that precision exposed a real bug.** The agreement started at
~1e-8, which is float32, not float64: `sources/idsse.py` was storing the cached
tracking window as float32. The IDSSE feed gives two decimal places, so float32
was quantising exact inputs by ~1e-6 m. The window cache now stores float64 and
its key carries a version so old caches are ignored. Physically the difference
was irrelevant; it mattered because it made exact agreement impossible to
demonstrate.

---

## 7. Size and latency

```
python -m demo_viz.web.build
```

| | raw | gzipped |
| --- | --- | --- |
| shell (html + css + 7 JS modules) | 73.9 KB | 22.2 KB |
| scene index (45 entries) | 8.6 KB | ~3 KB |
| **initial load** | **82.6 KB** | **~25 KB** |
| one scene | 79 KB | 26 KB |
| all 45 scenes | 3.41 MB | 1.11 MB |
| total build | 3.49 MB | — |

Scenes are fetched one at a time on selection; nothing but the shell and the
index loads at startup.

**Browser latency** (`python -m demo_viz.web.validate --bench`, headless Chrome,
median / worst over 6 scenes):

| | median | worst |
| --- | --- | --- |
| parse scene JSON | 0.2 ms | 0.3 ms |
| build influence cache | 0.0 ms | 0.1 ms |
| first residual (pays for all 22 influence surfaces) | 9.8 ms | 13.2 ms |
| **role switch** | **0.8 ms** | **0.9 ms** |
| rank defenders | 0.2 ms | 0.5 ms |
| rank beneficiaries (10 targets × 2 series) | 11.3 ms | 12.0 ms |

Scene switch is dominated by the network fetch of ~26 KB. Role switching is
sub-millisecond, which is the target.

The benchmark measures with `performance.now()` over batched iterations and
inlines its payloads, because Chrome's `--virtual-time-budget` freezes the clock
during synchronous work and reports zero — the first version of the harness hung
on exactly that.

---

## 8. Commands

```bash
# local Dash explorer
uv pip install -e ".[app]"
python -m demo_viz.app.interactive_app                  # http://127.0.0.1:8060
python -m demo_viz.app.interactive_app --scene J03WOH:shot_006_P1_1054 --mode manual

# browser build
python -m demo_viz.web.export_data                      # 45 scenes -> demo_viz/web_data/
python -m demo_viz.web.export_data --effects strong     # narrower
python -m demo_viz.web.build                            # -> demo_viz/web_build/ + size report
python -m http.server -d demo_viz/web_build 8090        # http://127.0.0.1:8090

# checks
python -m demo_viz.web.validate --cases 160 --scenes 16 # Python vs browser
python -m demo_viz.web.validate --bench --scenes 6      # latency
python -m unittest discover -s tests -p 'test_*.py'     # 143 tests
```

The browser explorer takes URL state, which makes a particular pick linkable and
the screenshots reproducible:

```
?scene=J03WOH:shot_006_P1_1054&r=7&d=11&b=34&t=240&fit=1&swap=1&about=1
```

`r`/`d`/`b` accept shirt numbers or player ids; `d` and `b` take comma lists.

---

## 9. GitHub Pages deployment

`.github/workflows/pages.yml` runs on push to `main`/`master` (and on demand),
assembles the site with `python -m demo_viz.web.build --out _site`, checks the
build contains `index.html`, `js/influence.js` and every scene file the index
references, then uploads and deploys with the official
`actions/configure-pages`, `actions/upload-pages-artifact` and
`actions/deploy-pages`.

There is no npm step: the site is plain files, so the workflow only copies and
uploads.

**One manual setting is required, and I cannot set it for you:**

> **Settings → Pages → Build and deployment → Source: `GitHub Actions`**

Then push the branch to `main` (or run the workflow manually from the Actions
tab). The site will appear at

```
https://chani-song.github.io/offball-value/
```

derived from the `origin` remote, `https://github.com/Chani-song/offball-value.git`.
The README hero image and badge already point there.

Note: the repository's `docs/` folder holds the research documentation, so the
Pages build deliberately does **not** use the "deploy from `docs/`" option — it
uploads an artifact instead, which leaves `docs/` untouched.

---

## 10. Screenshots

In `demo_viz/exports/`:

| file | what |
| --- | --- |
| `web_annotation.png` | browser, annotated triplet, Focus view (README hero) |
| `web_manual.png` | browser, manual pick (#10 / #23 / #34) |
| `web_multirole.png` | browser, two runners and two defenders at once |
| `web_multidefender.png` | browser, two pulled defenders, two ghosts |
| `web_medium.png` | browser, a `medium` scene |
| `web_source.png` | browser, the Source & method panel |
| `app_dock.png` | local Dash explorer with the role dock |

---

## 10b. Presentation pass

A final pass moved implementation detail out of the default view:

* the per-frame provenance strip is gone; **Source** opens a panel carrying the
  measured / human / explanatory / not-shown breakdown, the method, and the
  links to the reports. Nothing was weakened — the same statements are one
  click away instead of set in 11 px type under the pitch.
* `Run starts -2.9 s` no longer names the detector inline; the detector is the
  element's tooltip and the panel explains both cases.
* in **Annotation** mode the **Reacts most** list is collapsed and dimmed, so a
  geometric heuristic cannot be mistaken for the annotated roles beside it. It
  expands on click, and opens automatically as soon as you edit a role.
* the `Fit` checkbox became a `Full pitch` / `Focus` segmented control next to
  the pitch, where a view control belongs.
* drag feedback: compatible slots light up in their own role colour, the
  incompatible side visibly dims, the hovered slot gets an inset ring, and
  dragging a chip onto a compatible slot previews `#7 → Beneficiary` on the
  cursor before you drop.
* layout: the app is capped at 1840 px wide and 1090 px tall and centred, so a
  1440p screen no longer stretches an aspect-locked pitch into a letterbox.

## 10c. Terminology, layer names, multi-role and the strong-five videos

A later pass changed what things are called and how roles are edited. No
scientific quantity changed: the same `goal_weighted_influence` residual is
computed by the same code, and the parity harness still agrees with Python to
8.5e-14 over 60 random cases.

### The two numbers now say what they are

| was | is | what it actually is |
| --- | --- | --- |
| Opened space | **Space created** | factual minus the held-defender baseline — a signed difference against a what-if |
| Total space | **Available space** | the factual goal-weighted residual space the beneficiary holds |

**Available space is now the default** in the Dash app, in the browser and in
the rendered videos. It is the measured quantity; the difference against a
counterfactual is the interpretation, so the measurement leads. "Total space"
is gone from the interface — it suggested a whole-pitch total, which it never
was. The video panel reads `AVAILABLE SPACE`, and the beneficiary badge reads
`29.2 available space` rather than a signed gain.

### Layers are named for what you see

| internal | label |
| --- | --- |
| `trail` | Runner movement |
| `tether` | Defender response |
| `wake` | Space map |
| `lane` | Passing lane |
| `ghost` | Defender if stayed |
| `labels` | Player numbers |
| `candidates` | Suggested players |
| `paths` | Player movements |

On by default: Runner movement, Defender response, Space map, Defender if
stayed, Player numbers, Suggested players. Off by default: **Passing lane** and
Player movements, in both front ends. The variable names did not change; only
the labels did, and `DEFAULT_LAYER_VALUES` in `app/components.py` and the
initial `state.layers` in `js/app.js` are now the single statements of the
default set. The browser had been spelling one layer `hints` where Dash spelled
it `candidates`; they are both `candidates` now.

### Every role holds several players

`Selection.runners` / `.defenders` / `.beneficiaries` were already lists, but
the interaction treated the first two as single-valued. Now:

* each chip is dragged on its own, and dropping it on another role **moves that
  player and nobody else** — the rest of the role stays;
* a click on the pitch with a slot armed **toggles that one player** in or out;
* `×` removes one player;
* nobody can hold two attack roles at once, so dragging a runner onto
  Beneficiary takes it out of Runner;
* **only an explicit clear or Reset empties a whole role.** Clicking another
  compatible player never replaces the list.

The global **Runner ↔ Beneficiary swap** button is gone from both front ends —
it was a whole-list operation in an interface that is now per-player.
`swap_attack_roles()` survives as a library method (and the `?swap=1` deep link)
because it is still the honest way to express "exchange the two roles".

### Five strong scenes, rendered from the human annotation

`python -m demo_viz.export_strong` renders only the five `effect == strong`
clips, with **all** of their annotated runners, defenders and beneficiaries —
no heuristic auto-triplet. Ordered simplest interaction first:

| # | scene | shape |
| --- | --- | --- |
| 1 | `J03WOH:shot_006_P1_1054` | 1R-1D-1B |
| 2 | `J03WOY:shot_002_P1_0580` | 1R-2D-1B |
| 3 | `J03WOY:shot_011_P2_0903` | 1R-2D-1B |
| 4 | `J03WOH:shot_010_P1_1759` | 2R-2D-1B |
| 5 | `J03WPY:shot_018_P2_1375` | 2R-2D-1B |

Into `demo_viz/exports/strong/`: `strong_01..05.mp4`, `strong_01..05.png`,
`strong_five_montage.mp4` (74.2 s: a minimal `OFF-BALL SPACE` card, then
`STRONG n / 5` before each clip), and `manifest.json` recording which scene each
number is and which roles it used. 1920x1080, 25 fps, the same role colours.

`FigureConfig.disabled_layers` is how the video turns the passing lane off, and
`build_storyboard(space_view="available")` is how the beat captions stop
claiming a gain the panel is not showing.

### Tests

`tests/test_demo_viz_web.py` gained three classes and 166 tests pass:

* `DefaultViewTests` — Available space is the default in both front ends, the
  layer labels are the human-readable ones, Passing lane and Player movements
  start unchecked, the other six start checked, no `btn-swap` survives, and the
  strings `Total space` / `Opened space` appear nowhere in the markup.
* `StrongVideoSelectionTests` — exactly the five strong scenes are rendered, no
  medium or low scene reaches the montage, they are numbered simplest first.
* `BrowserRoleStateTests` — `js/selection.js` is a hand-written mirror of
  `core/selection.py`, so `web/selection_harness.html` runs the dock's own
  cases through the browser copy in headless Chrome. Breaking `move()` in the
  JS copy fails this test, which is the point of it.

## 10d. Helvetica, wording, and the presentation pass on the videos

**Helvetica.** `render/figure.py` sets `font.sans-serif` to Helvetica, then
Arial, then Liberation/DejaVu, and the browser and Dash name the same stack.
One catch worth recording: macOS ships Helvetica as a `.ttc` whose glyphs
FreeType cannot load below about six pixels, so a small-dpi render raised
`RuntimeError: failed to load glyph` rather than falling back. `sans_stack(dpi)`
probes the smallest type in the frame at the figure's own dpi and drops any
leading font that cannot draw it — Helvetica at the 120 dpi the demo renders at,
Arial for a thumbnail or a test. The chain-rail separator moved from U+276F to
U+203A for the same reason: matplotlib does not fall back per character.

**Wording.** "five human-confirmed scenes" is gone; the montage card reads
**Off-the-ball investigation examples**, the site and the README landing read
**Off-the-ball value explorer**, and the hero line is "click a runner, a
defender and a beneficiary, and explore how the space changes". The research
sections of the root README still say *human-confirmed*, because that is the
audit pipeline's own term for what those scripts produce and renaming it there
would misreport them.

**The videos lead with the roles.** The beat detectors place their moments
where the measured signals move, which is right for an analysis render and
wrong for a talk: the first role did not light up until 47 % of the clip on
scene 1 and 62 % on scene 5, and scenes 3 and 4 had dead air between "space
opens" and "beneficiary gains". Each demo video now runs in three phases:

1. **Role introduction.** The clip freezes on its opening frame and each
   annotated role is named in turn — runner, defender, beneficiary — about a
   second each, with the camera pushed in on them and the rest of the pitch
   dropped back. A role whose players stand more than 18 m apart is visited one
   player at a time; closer than that, one box holds both. In these five scenes
   the pairs fall cleanly either side of that line (8 m apart, or 16–24 m), so
   only the two-runner scenes split.
2. **Reset.** A "cast" beat settles on the framing the play opens with, so the
   cut into phase 3 is invisible.
3. **Play.** The scene runs with runner, defender, beneficiary and the space
   field all at full strength from the first frame to the last. They never drop
   out, and nothing waits for a detector.

The captions still walk the causal chain, but on fixed fractions of the clip
(0, 26, 50, 72 %) rather than on detected times. That is a presentation choice,
not a measurement, so the frame says so under the timeline: *caption marks are
presentation-paced, not detector times*. Everything measured — the space field,
the curve, the stat — is still the per-frame value at the frame being drawn,
which is why `SceneFigure.draw` takes the storyboard's clock separately from
the scene's.

The held-defender caption no longer reads "pulled 0.0 m" during the
introduction, and the passing-lane and player-movement layers stay off.

Videos are 1920×1080, 25 fps, about 16–17 s each; the montage is
`strong_five_montage.mp4`.

## 10e. OBSO threat and the candidate-pass fan

Two additions, both built only from code and data already in the repository or
on `origin/kyuhyeok-dev`. No fitted artefact was needed and no solver output is
used.

### OBSO threat, a third space view

`Available space | Space created | OBSO threat`. The surface is

    pitch control  x  ball transition  x  EPV score

from `offball_value.reference_obso.evaluate_reference_obso`, copied from
`origin/kyuhyeok-dev @ 3c9965b` **byte-identical** so it can still be diffed
against that branch. It was a one-file dependency: it needs only
`bundesliga.py`, which this tree already had, and `data/static/EPV_grid.csv`,
which was already committed.

The bare EPV grid is never exposed. It is position-only — a function of pitch
coordinates with no representation of the defensive line, so a cell three
metres behind the last defender scores the same as one three metres in front.
Measured down the central lane it is essentially distance to goal: 0.021 at
36 m, 0.041 at 24 m, 0.119 at 13 m, 0.571 at the goalmouth. The **product** is
still worth showing, because pitch control does move when a defender is beaten,
and that is the whole distinction the caveat in the Source panel draws.

**It is not role-sensitive.** `evaluate_reference_obso` takes no runner,
defender or beneficiary, so the field does not move when the visitor re-picks.
Verified end to end: the same frame under three different role selections
reports the same peak, 0.047.

### Why it is precomputed, and what is not

The surface costs ~260 ms a frame in numpy against 0.8 ms for a role switch, so
it cannot be recomputed in the browser. Being role-blind is what makes
precomputing it possible at all.

Only **pitch control** is stored. The browser computes the other two terms
itself at every frame:

| term | where | why |
| --- | --- | --- |
| pitch control | precomputed, 0.2 s samples, interpolated | expensive, and slow-moving |
| ball transition | in the browser, exact | the ball moves fast enough that interpolating it smears the brightest moment of a clip |
| EPV score | shipped once for all 45 scenes | one static grid, not 45 copies of it |

Sampling error against an every-frame computation, as a percentage of the
scene's peak OBSO:

| stride | max | mean |
| --- | --- | --- |
| 5 frames (0.20 s) | 5.8 % | 0.01 % |
| 10 frames (0.40 s) | 8.4 % | 0.02 % |

with one exception no stride fixes. `apply_offside=True` makes an attacker's
contribution appear or vanish the instant he crosses the line — on the sample
clip at frames 179 and 253 — and interpolating across that step reaches **73 %
of peak**. It is a real discontinuity, not undersampling, so the exporter puts
a sample on **both frames either side of every offside change** and the ramp is
one frame wide. Quantising control to one byte costs a further 0.2 %.

End to end, the browser's reconstruction matches `evaluate_reference_obso` to
**0.76 % of frame peak**, which is the quantisation and nothing else.

Payload: ~118 KB per scene in `web_data/obso/`, fetched only when a visitor
first picks the threat view, plus a 14 KB score grid shared by every scene. The
initial page load is unchanged at 110 KB.

### Candidate passes

Five fixed directions — a 90° sector centred on the attacking direction, cut
into five 25 m rays, clipped to the pitch — drawn from whoever is within 2.5 m
of the ball and at least 0.5 m closer to it than anyone else. That is looser
than the pipeline's own 1.5 m carrier gate on purpose: tracking records the
torso while a dribbler pushes the ball ahead of it. Over ten scenes the
looser threshold moves coverage from 38 % to 43 % of frames — the ball really
is in flight or loose the rest of the time — but it stops the fan vanishing on
frames where the ball is plainly at someone's feet. Off by default.

They are **not** ranked, scored for completion, or optimised. No pass model is
applied, because this repository has no validated one: the solver's current
estimator is described by its own author as "an experimental proxy, not a
validated replacement", and it is position-only, which is exactly the failure
mode a candidate fan would advertise. The endpoint numbers are OBSO values
read off the field, not scores for the passes.

The margin rule is the honest part. During a pass nobody is within 1.5 m of the
ball, and two players converging on a loose one are contesting it rather than
carrying it. Both cases yield no carrier, and the fan is hidden with a short
"No clear carrier this frame" note instead of a guessed passer. The selected
runner is never made the passer; the carrier comes from the ball.

The rule is stated once and shared: `core/candidates.py` and
`web/site/js/obso.js`, with a test asserting the constants agree.

### Tests

16 new tests in `tests/test_demo_viz_obso.py`: the browser-versus-Python
surface parity, fan geometry (five rays, correct bearings either attacking
direction, always inside the pitch), role invariance end to end, the export's
declared range and sampling, and a guard that no completion-probability,
best-pass, ranking or expected-value language reaches the browser. The last one
caught its own false positive — the Source panel's existing *"No calibrated xT
… is drawn"* disclaimer is a denial, not a claim, so the test allows the word
only in that sentence.

## 11. Known limitations

* **Pitch labels can overlap** when players are bunched — the tether and ghost
  captions have a dark outline but no collision avoidance. The rendered-video
  path has a proper label placer (`render/labels.py`); the browser does not.
* **`Full pitch` is the default view** in both front ends, with a `Focus`
  toggle beside the pitch. Focus makes the picked players much larger and the
  opened space far more legible, but it crops players you might want to click,
  so it is not the default.
* **No drag and drop in the Dash build**, by choice — see §3.
* **Playback is a timer**, not a video player: `setInterval` at 80 ms advancing
  two frames. The MP4 exports are the thing to show when smoothness matters.
* **The browser recomputes rankings on the main thread.** `Gains most` costs
  ~11 ms, which is fine, but a scene with many more attackers would want a
  worker.
* **Pipeline mode is Dash-only.** The browser build always starts from the
  annotation and lets you edit; there is still no research-pipeline triplet
  file to read (see `APP_REPORT.md` §5).
* **OBSO threat and Candidate passes are browser-only.** The Dash build keeps
  the two role-sensitive views. Nothing in the Dash path changed.
* **The EPV term does not know where the defensive line is.** See §10e. It is
  the reason the threat view is never called xT, and it is the open modelling
  question this demo makes visible rather than solves.
* **The candidate fan is a geometric rule, not a shortlist.** Five directions
  at fixed angles: no pass is claimed to be available, let alone best. A real
  ranking needs a validated completion model, which this repository does not
  have.
* **The exported data is committed** (3.4 MB) so Pages can deploy without the
  raw Bundesliga files. It is derived data for the 45 annotated scenes only.

---

## 12. Next

1. **A defender-by-beneficiary matrix** for the selected runner: 11 × 11 cached
   lookups, so it would render instantly and would answer "which pairing opens
   the most" in one view instead of by repeated clicking.
2. **Label collision avoidance in the browser**, porting the idea from
   `render/labels.py`.
3. **Brush the chart** to choose the window the gain is averaged over, instead
   of fixing it at "everything after the run starts".
4. **Settle the counterfactual question** in `OVERNIGHT_REPORT.md` §8 — only one
   of five strong scenes shows the sign the human label implies. The explorer
   now makes that easy to probe, because you can watch the sign flip as you
   change the defender.
