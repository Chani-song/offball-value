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
?scene=J03WOH:shot_006_P1_1054&r=7&d=11&b=34&t=240&fit=1&swap=1
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
| `web_annotation.png` | browser, annotated triplet, Fit on (README hero) |
| `web_manual.png` | browser, manual pick (#10 / #23 / #34) |
| `web_swapped.png` | browser, Runner ↔ Beneficiary swapped |
| `web_multidefender.png` | browser, two pulled defenders, two ghosts |
| `web_medium.png` | browser, a `medium` scene |
| `app_dock.png` | local Dash explorer with the role dock |

---

## 11. Known limitations

* **Pitch labels can overlap** when players are bunched — the tether and ghost
  captions have a dark outline but no collision avoidance. The rendered-video
  path has a proper label placer (`render/labels.py`); the browser does not.
* **`Fit` is off by default** in both front ends. It makes the picked players
  much larger and the opened space far more legible, but it crops players you
  might want to click, so the default stays on the full pitch.
* **No drag and drop in the Dash build**, by choice — see §3.
* **Playback is a timer**, not a video player: `setInterval` at 80 ms advancing
  two frames. The MP4 exports are the thing to show when smoothness matters.
* **The browser recomputes rankings on the main thread.** `Gains most` costs
  ~11 ms, which is fine, but a scene with many more attackers would want a
  worker.
* **Pipeline mode is Dash-only.** The browser build always starts from the
  annotation and lets you edit; there is still no research-pipeline triplet
  file to read (see `APP_REPORT.md` §5).
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
