# Paper-figure alignment

What the demo shares with the paper figures, what it could not share, and why.

> **REBASED 2026-10-01 onto `origin/kyuhyeok-dev` @ `8a5c69d`.** The figure
> renderer, which §1 below reports as missing, **was pushed**. Everything this
> document called "reconstructed" has been replaced by the real thing, and the
> corrections are in §7 at the end. §§1-6 are kept as the record of what was
> known at `e84553a`; read §7 for what is true now.

Originally audited against `origin/kyuhyeok-dev` @ `e84553a` (2026-09-30).

---

## 1. The figure renderer is not here

Searched for the figures' own captions — `"Pass + receive"`, `"Follow the
runner"`, `"Stay with the ball carrier"`, `"Equilibrium choices"`, `"Current
state"`, `"Future 1"`, `"Slow down"` — across:

* every ref in this repository (`git grep` over `rev-list --all`);
* `origin/kyuhyeok-dev` at `e84553a`;
* `~/Research/offball_demo/mit_ssac2027_ref`;
* `~/Desktop`, `~/Documents`, `~/Downloads`.

**Zero matches.** The code that draws the figures is not on this machine, so
its colours, marker shapes and English action captions **could not be recovered
from source**. The brief's "Dribble / Slow down / Pass + receive" do not match
any decoder output in the research code either, so the renderer applies its own
English relabelling that I cannot see.

Per the brief's own rule — *"Do NOT imitate the screenshots approximately if
exact code exists"* — exact code does not exist for the visual layer, and
guessing at it would be inventing. So:

* **semantics** (roles, action names, probabilities, geometry) are ported from
  the figures' own data producer, exactly;
* **house style** (colour, shape, weight) follows the brief's description and
  is labelled as such in `figure.js`, not presented as recovered.

## 2. What *is* authoritative, and is ported

| Recovered from | Gives |
| --- | --- |
| `scripts/extract_panel_policy.py` — *"One game's opening decision, as a figure panel needs it"* | the panel contract, `compass_name`, `pass_where`, `ROLES` |
| `src/offball_value/stage3_read.py` | `world_direction`, `name_move_targets`, `targets_for` |
| `deploy/delta/eval_v1.sbatch`, `multipass_full_2v1.sbatch` | both of the abstract's runs use `--commands compass` |

Ported in [`web/site/js/figure.js`](web/site/js/figure.js) and held to the
research functions numerically by
`tests/test_figure_alignment.py::FigureDecoderParityTests`, which runs both
implementations over 40 cases in a headless browser and requires identical
strings.

### Action names

**Attackers** (`compass_name`, attack turned to +x): `stop`, `forward`, `back`,
`left`, `right`.

**Defender** (`name_move_targets`): the target whose cosine is largest and at
least **0.3**; ties to the *later* target; `sideways` when nothing reaches 0.3;
and `(0,0)` is **brake**, not "stand" — *"a running defender keeps sliding the
same way while he slows"*.

The research code is Korean. The gloss is this repository's, and it is the only
invented text in the chain:

| `stage3_read` | shown |
| --- | --- |
| `볼 쪽` | toward the ball |
| `러너 쪽` | toward the runner |
| `수혜자 쪽` | toward the beneficiary |
| `골문 쪽` | toward goal |
| `옆으로` | sideways |
| `멈추기(감속)` | brake |

`볼 쪽` and `러너 쪽` are the dilemma: *stay with the ball carrier* versus
*follow the runner*, in the research code's own words.

Two commands can share a name — 55/45 east/south may both point at the ball.
That is a property of the rule, not a bug, and `stage3_read.merged` sums
same-named shares for exactly this reason. The pitch keeps both arrows with
their own shares, so no mass is hidden.

### Roles

`extract_panel_policy.ROLES` — 2v1: **ball carrier**, **runner**, **defender**;
3v1: **runner**, **beneficiary**, **defender**.

These are *solver* roles and are not the demo's annotation roles. The
annotation names a runner, a defender and a beneficiary — a person's reading.
The solver names slots in a game, and its ball carrier has no annotation role at
all. `app.js:solverRolesAt` maps only what is warranted and returns nothing for
a player whose solver role is undetermined; the ball carrier is read from the
tracking, which wins, because who has the ball is not an opinion.

## 3. House style (from the brief, not recovered)

| | |
| --- | --- |
| Ball carrier | circle |
| Runner | diamond |
| Beneficiary / teammate | triangle |
| Defender | square |
| Attacker colour | the demo's existing role colours (runner pink, beneficiary cyan) |
| Defender colour | orange |
| Equilibrium action | arrow, width **and** opacity by probability |
| Pass | dashed line to the target, ringed endpoint |
| Attack direction | `attack →` in the top-left margin |
| Background players | 22% opacity in the solver modes |

Shape carries the role and colour only reinforces it, so identity survives
greyscale, print and colour-vision deficiency. Probability never rides on
colour: a 3% arrow and an 83% arrow differ in width and opacity, and both are
drawn — only the *label* drops below 10%, and the side panel keeps every
option's share regardless.

**Departure from the brief:** it asks for attackers in blue. The demo's runner
(pink) and beneficiary (cyan) colours are load-bearing across every other view
and in the role dock, so recolouring them would have broken more consistency
than it bought. Defender orange matches. This is the one place the demo keeps
its own language deliberately.

**Superseded for the showcase, 2026-10-01.** That departure now holds only in
the Full explorer. The Submission showcase was moved onto the paper's own
values; section 9 has the mapping and says what is shared.

**Muting is visual only.** Background defenders stay in every calculation.

## 4. Attack direction

`scene.js` sets `flip = attacking_direction < 0` and `view()` reflects the
pitch, so **every scene already renders with the attack running right**. The
indicator is therefore a constant, not a per-scene fact — drawn anyway, because
a reader should not have to infer which way the play goes before reading an
arrow.

A direction is transformed with the *linear* part of that reflection
(`viewVector`). Using the point transform mirrors every arrow in y, which is
exactly what happened on the first attempt and what
`test_a_direction_is_transformed_with_the_linear_part` now prevents.

## 5. What each mode shows

**Observed** — unchanged in substance. Role shapes and the attack indicator are
new.

**Counterfactual / the dilemma** — the defender's five compass commands as an
**unweighted** fan from his real position, each named by
`name_move_targets`. Unweighted is the honest picture: naming the moves needs
only the scene, but saying how they should be mixed needs the solved game, and
no artifact covers any scene here. Entering the mode moves the playhead to a
frame that *has* a ball carrier (usually the run onset), because `toward the
ball` has no meaning once the ball is loose and the timeline opens on the shot.

**Game solution** — the same arrows, weighted by the solved policy, with the
decision following the playhead (`floor(seconds / step_seconds)`). No
Bundesliga scene has an artifact, so publicly this is the honest empty state;
it is exercised against a real study state by
[`web/story_harness.html`](web/story_harness.html), which draws the actual
`Pitch` with the actual `arrows.js` and `figure.js`, so what it validates is
what the demo would show.

**Evaluation** — untouched; still the pending state, still no invented metric.

## 6. Honest limits

* The per-step attack policy is not in the artifact — only the root joint
  policy and `modal_line`'s per-step *defender* policy. `bodyPolicies` returns
  `attackKnown: false` past step 0 rather than reusing the root, and the panel
  says "root only in this artifact".
* The legacy `exact_100` states predate `modal_line`, so the harness shows one
  time step. Time-stepping is implemented and will populate from any artifact
  that carries `modal_line`.
* Probability mass is reported, never normalised: `massOf` returns the residual
  and the panel prints it.


---

# 7. Rebase onto `8a5c69d` (2026-10-01)

Three commits since `e84553a`:

    648eed5  figures: the abstract's Figure 1 and Figure 2 (S05, shared print style)
    e89e78f  cleanup: keep what reproduces the abstract, in English
    8a5c69d  evaluation: static counterfactuals held for the whole window

`e89e78f` cut `scripts/` from 199 files to 32 and translated the Korean to
English; everything removed stays at tag `pre-cleanup-2026-09-30`.

## 7.1 The figure code, found

| | |
| --- | --- |
| `scripts/figure_style.py` | the print style both figures share |
| `scripts/render_figure1_dilemma.py` | Figure 1 |
| `scripts/render_figure2_abstract.py` | Figure 2 |
| `scripts/render_panel_figure.py` | the panel readers Figure 2 imports |

Both figures are S05 and both re-render **pixel-identical** to the shared PNGs
from that commit.

## 7.2 What this demo had wrong

| | reconstructed at `d881899` | **upstream at `8a5c69d`** |
| --- | --- | --- |
| Attack colour | runner pink, beneficiary cyan | **both blue `#0072B2`** — "the runner and the ball carrier are both attackers, so both blue: told apart by marker and by direct labels, so no role rests on colour alone" |
| Defence colour | orange `#FFA62B` | **vermillion `#D55E00`** (the Okabe-Ito pair) |
| Beneficiary marker | triangle | **disc** — the triangle is the 3v1 **teammate**, the second attacker when a scripted passer has the ball |
| Zero command | "brake" | **"slow down"** |
| Target names | "toward the ball/runner" | **"toward ball"**, **"toward runner"**, **"toward beneficiary"** |
| Option arrow | fixed length, width **and** opacity by probability | **the solver's own 0.6 s path, unchanged**; probability sets **width only**, `lw_of(p) = 0.6 + 2.4p`, one scale for every panel |
| Stop | an arrow | **a bar across the path's end, no head** |
| Defender label | "toward the ball 88%" | **"88%"** — the direction is the arrow's job |
| Dilemma | a five-command fan | **two options, "Follow?" and "Stay?"**, 3 m long, pointing at where the runner and the ball carrier really were 0.6 s later. Figure 1 **carries no number by design** |

All of these are now ported and held by `tests/test_figure_alignment.py`
(`FigureLabelParityTests`, `PrintStyleParityTests`, `FigureDecoderParityTests`).

## 7.3 Figure 1's semantics, exactly

Three panels, every position read from the S05 tracking:

* **Current state** — Figure 2's **0.6 s** moment. At it the solver's defender
  is *indifferent* between dropping with the runner and stepping to the ball
  (equal values, **57/43**), and the real defender slows to his slowest of the
  play. Each of the three's last 0.6 s as a line; the runner's run to come; the
  ball carrier's next 0.6 s; the defender's two option arrows.
* **Future 1, follow the runner** — **what really happened**, drawn at the shot.
  Real tracking.
* **Future 2, stay with the ball carrier** — a **counterfactual**, drawn when
  the through ball arrives (1.06 s at the pass model's fitted ball speed). The
  through ball is the one the solver plays at the dilemma (**74%**, along the
  run, 8 m ahead of the runner); the defender is moved by **the solver's own
  motion function** (1.8 m — he was moving toward the runner and has to turn);
  **everyone else is where he really was**.

So Future 2 is a hybrid — solver pass target, solver motion for the defender,
real positions for everyone else — and upstream labels it a counterfactual.
The pairing is the model's, not football sense: the through ball is worth 0.674
against "toward ball" and 0.661 against "toward goal", and by the defender's
indifference the carrier's drive is then worth more against "toward goal"
(0.683 vs 0.645).

**The demo ports the current-state panel only.** Future 1 and Future 2 need the
solved panel file and the pass model's ball speed, and no artifact covers a
published scene.

## 7.4 Figure 2's semantics, exactly

* **Each panel is the game solved from that REAL moment** (0.0 / 0.6 / 1.2 s) —
  not the t=0 world propagated forward. This is what the demo's timeline maps to.
* An option = the solver's own 0.6 s path under that command, drawn unchanged.
* A pass = dashed charcoal from the ball to its target; the receiver's run onto
  it = a **straight** connector in his colour. One joint column, so one label:
  **"Pass + receive · NN%"**.
* Labels: `"Slow down NN%"` (braking, still moving), `"Stop NN%"` (at rest),
  `"Dribble NN%"` (ball carrier), otherwise **the percentage alone**.
* The real next 0.6 s of the three: a thin grey **dotted** line to a **hollow**
  marker.
* `MIN_P` = 2%; small options are drawn, not dropped.
* Probabilities are equilibrium probabilities of *choosing* an option — for the
  two attackers the joint policy summed over the other — **not** pass-completion
  chances.

## 7.5 Vector field and heatmap

**Not in the pushed code.** `render_panel_figure.py` was added "without the
`--heat` debug background, which is not part of the abstract", and `quiver`,
`streamplot`, `imshow`, `pcolormesh` and `--heat` have zero matches anywhere in
`scripts/` or `src/`. Nothing to port, and nothing invented.

## 7.6 What the demo still cannot show

`data/processed/*` and `out/` are gitignored upstream, so no panel file, no
solved state and no evaluation output is available here. Figure 1's futures,
Figure 2's panels and every evaluation number are therefore still the honest
unavailable state. What changed is *why*: the methods exist now, so the slots
say **"Awaiting an evaluation run"** rather than "not defined".


---

# 8. Visual sync with the near-final figure (`139498a`, 2026-10-01)

One commit since `8a5c69d`: *"figures: final abstract Figures 1 and 2 with
captions, Figure 2's defender-start grids"*.

## 8.1 What changed upstream, and is now ported

| | before | **`139498a`** |
| --- | --- | --- |
| Palette | Okabe-Ito `#0072B2` / `#D55E00` | **pure blue `#0000FF` / pure red `#FF0000`** — the earlier pair "read as a stock palette" |
| Key markers | 0.8 pt charcoal outline | **no outline** (`KEY_EDGE` 0.8 → 0) |
| Other players | 45% tint | **full team colour**, thin white edge (the tint "read as blurred") |
| Ball | charcoal | **white with a charcoal edge** |
| Moves | started at the marker | start `MOVE_GAP` = **2.5 pt clear of the marker's edge** (`marker_edge`, `leave`) |
| Defender labels | the percentage alone | **named**: `To goal 57%`, `To ball 43%` (`--defender-names short`) |
| Label crowding | may touch | **`--label-gap 0.35` m** of clear ground from labels already placed |

The near-final figure's own settings, from the README's command:

```
--frames --value-fade 0 --value-grids $GRIDS
--flow-grids $GRIDS --flow-color "#FF8000" --flow-alpha 0.4
--defender-names short --label-gap 0.35
```

## 8.2 Why the defender's moves are named

`--defender-names`' own help text: *"a short move, e.g. 0.6 s toward ball 43% =
**0.57 m**, hides under his marker and its % floats unexplained"*. That is
exactly the team's 0.6 s readability concern. The demo names them by the same
rule and, where a move is shorter than the marker, draws no shaft at all — the
named label is the option.

Upstream's table lists three targets because its final command renders S05, a
2v1 game. `stage3_read.targets_for` gives a 3v1 defender a **fourth**, and four
of the demo's seven scenes are 3v1 (S15 plays it 26% at 0.6 s), so
`toward beneficiary` is title-cased by the same rule rather than left as a bare
percentage.

## 8.3 Label collision

Labels are placed heaviest first, and a stop that has come to rest last —
upstream's `prob - 1.0` ordering, because its label sits on the player and can
go anywhere round him. Each label tries a fixed, ordered list of offsets (tip,
then outward along the arrow, then to each side) and takes the first that
clears every label already placed, with `labelGap` between them. **Nothing is
hidden and nothing is random**; only the 2% probability floor ever removes a
label.

`rests` is upstream's exact test, `hypot(end_velocity) == 0.0`, so S05's
defender stop at 1.2 s reads **Stop 22%**, not "Slow down".

## 8.4 The defender-start grids are **not** ported

`139498a` adds the scalar background (the game's value with the defender
starting at each spot, darker = better for him) and the orange flow field of
his expected 0.6 s move, on a 1 m lattice per moment.

**There is no grid data.** `build_defender_grid.py`,
`extract_defender_grid.py`, `check_defender_grid.py` and
`jobs/solve_defender_grid.sbatch` are committed; the grids themselves
(`data/processed/showcase_v1/defender_grid_S05/grid_S05[@dt]_1m_full.json`) are
not, the 2026-09-30 bundle predates them, and nothing on this machine has one.
Reconstructing a scalar field from a screenshot is exactly what section 7 of
the brief forbids, so nothing was drawn.

What would unblock it: those three JSON files per scene — they are numbers on a
1 m lattice, no tracking. The renderer reads them as
`--value-grids 0.0=... 0.6=... 1.2=...`.

**No attacker vector field**, as upstream cancelled it and the brief confirms.


## 9. Two themes, one grammar (2026-10-01)

The demo serves two audiences from one build, so it carries two themes and the
collection picks between them.

| | Submission showcase | Full explorer |
| --- | --- | --- |
| Page, cards | white | the dark research shell |
| Pitch | `#F7F7F7`, lines `#D0D0D0` | dark green, pale lines |
| Attacker | `#0000FF` (both of them) | runner pink, beneficiary cyan |
| Defender | `#FF0000` | orange |
| Text | `#222222` / `#666666` / `#B3B3B3` | near-white / grey |
| Role marker | ringed, no fill behind it | a filled wash behind it |
| Default overlays | runner movement, defender response, numbers | the same, plus the mode's diagnostics |

The showcase values are `figure_style.py`'s, read from
`origin/kyuhyeok-dev@139498a` rather than from a screenshot:
`PAGE/PITCH/LINE/INK/MUTED/FAINT/ATTACK/DEFENCE`. They live in
`palette.js:PAPER_THEME` with the source named per line, and the three roles in
`ROLE_PAPER`.

**Shape, not colour, is the role** once both attackers are blue. The showcase
leans harder on `ROLE_SHAPE` than the figures do, because a demo is clicked
rather than read: carrier circle, runner diamond, teammate triangle, defender
square, in both themes.

**How a theme reaches a mark.** `Pitch.setTheme()` stores it and redraws the
markings; everything drawn afterwards reads `this.theme` and `this.roles`,
never the palette module directly. The chart takes `theme`, the arrow
language reads `pitch.theme` for the pass, and the stat's inline colour goes
through `tint()`. CSS does the same through a `body[data-paper]` block that
redefines the custom properties and then adjusts the components that had
hard-coded values. Nothing is duplicated per theme except the values.

**Where the paper theme has fewer colours than the explorer.** The research
palette has four accents; the paper has two. So in the showcase a gain is
attack blue and a loss defence red, the run-start tick on the scrubber is
attack blue, and "now" on the evaluation strip is charcoal rather than red —
red stays the defender's, everywhere.

**What the showcase turns off by default.** The space map, the held-defender
ghost, and the reachable set in Counterfactual. All three remain under
Advanced; the showcase's Counterfactual is Figure 1 — three bodies and two
options — and the explorer's keeps its diagnostics.

**What was *not* restyled.** The Full explorer. Its dark theme, its annotation
colours and its layer defaults are unchanged, and a diff of the dark rules
shows it: the theme work is additive (`PAPER_THEME`, `setTheme`, the
`body[data-paper]` block) plus the substitution of `this.theme` for the
module-level palette inside `Pitch`, which resolves to the same values there.

## 10. The commands, and the football words for them (2026-10-01)

The public showcase names the solver's commands in football. Those names are
only defensible if they describe the action, so this section is the trace,
read from `origin/kyuhyeok-dev@99bddd8`. `js/public.js` points here.

**The solved studies run the compass set.** `jobs/solve_evaluation.sbatch:28`,
`jobs/solve_figure2_panels.sbatch:27` and `jobs/solve_defender_grid.sbatch:42`
all pass `--commands compass` (`andrew-*/scripts/run.py:47`). The *relative*
set — `"toward ball"`, `"cut off runner"`, `"hold between"` — exists and is
tested, but is **not** what `eval_v1_2v1`, `eval_v1_3v1` or the figure panels
ran, so it is not what this demo reads.

**One five-element set, shared by every body**, stated against the attack
direction (`agile_motion.advance:182-193`; the repo's own transcription is
`src/offball_value/stage3_read.py:32`):

| index | direction | |
| --- | --- | --- |
| 0 | `(0, 0)` | desired velocity zero |
| 1 | `(1, 0)` | forward, along the attack |
| 2 | `(0, 1)` | lateral |
| 3 | `(-1, 0)` | backward |
| 4 | `(0, -1)` | the other lateral |

Carrier, runner, teammate and defender all get this same set. No diagonals.

**The defender's names are descriptions, not action names.**
`stage3_read.name_move_targets` names each compass move after whichever line —
to the ball, to the runner, to the beneficiary, to the defended goal — it
points along most by cosine, `"sideways"` when none reaches 0.3. The same index
can take different names in different states, which is why `public.js` maps
names and not indices, and why two indices can share a name (S05 at 0.0 s has
`toward ball` on both commands 3 and 4).

| solver name | public | why it is accurate |
| --- | --- | --- |
| `toward ball` | Close down | moves along the line to the man on the ball |
| `toward runner` | Track runner | moves along the line to the runner |
| `toward goal` | Drop | moves toward the goal he is defending |
| `toward beneficiary` | Cover teammate | 3v1 only; see below |
| `sideways` | Slide across | a lateral move with no line to name it after |

**Command 0 is braking, not holding.** It sets desired velocity to zero, and
`agile_motion.steer` then applies maximum deceleration *along the current
heading* (`cap = physics.braking`), so a defender already moving keeps sliding
the same way while he slows. The research code calls it `"slow down"` for
exactly that reason (`stage3_read.py:56-59`). The figure splits it by whether
speed reached zero inside the 0.6 s: `Slow down NN%` while still moving,
`Stop NN%` at rest (`render_figure2_abstract.py:411-415`), and the demo ports
the test (`hypot(end_velocity) == 0`).

It is not a convention either. The imported solver **validates** it, in
`mit_ssac2027_ref/src/defensive_positioning/models.py:125-126`:

```python
if not directions or directions[0] != (0.0, 0.0):
    raise ValueError("the first command must brake to rest: (0, 0)")
```

and names it on the field itself, `models.py:98`: *"Zero desired velocity
brakes; the other commands point at compass axes."* There is no configuration
of this solver in which command 0 is anything but the brake.

**So the showcase does not call it "Hold".** "Hold" would claim he held his
position; the model says he braked. Both themes keep "Slow down" and "Stop",
which are already football English.

**One note on where the numbers come from.** The fork's agile path never reads
`maximum_acceleration`; `agile_motion.steer` uses `physics.speed_up`,
`braking` and `turning` only. The production runs took them from
`data/processed/physics_limits/agile_p999_nodelay.json` (braking 7.39 m/s²),
and that file sets `defender_delay_s = 0.0` — so **the demo must never narrate
a defender reaction delay**, and today it does not.

**The imported solver is not vendored.** `defensive_positioning` lives in the
sibling checkout `mit_ssac2027_ref/src/`, not in this repository, so anything
needing `GameConfig`, `FiniteGame` or `motion.advance` is read from there.

**The 3v1 third attacker is a player, not a label.** `fixedpasser/game.py:9-13`
maps the receiver slot to the beneficiary and gives him his own five commands
(`rollout.py:50-52` decodes the joint attack action as `divmod`), and the
scripted passer may release to him — `game.py:58` `RECEIVERS = ("runner",
"beneficiary")`. `DESIGN.md:11-13`: *"The defender's dilemma: follow the
runner, or guard the beneficiary."* `render_figure2_abstract.py:148-150` draws
him as the **teammate**, so the showcase uses that word too — in the legend and
in "Cover teammate".

In a **2v1** this body does not exist: the ball carrier *is* the beneficiary
(`stage3_read.py:97-104`), which is why S05's legend reads "Kownacki — On ball".

**The attackers' names** come from `extract_panel_policy.compass_name:66-73`:
`stop`, `forward`, `left`, `back`, `right`, against the attack direction. The
showcase gives these football words only where a word is already printed — an
off-ball attacker's arrow carries its percentage alone in both themes, because
the arrow is already showing the direction.

**Not ported, deliberately:** the relative command set's names. No study this
demo reads was solved with it.

## 11. Two audiences, two vocabularies (2026-10-01)

The showcase and the explorer now differ in what they *say*, not only in how
they look. One predicate decides it — `isPublic()` in `app.js`, true while the
Submission showcase is open — and `js/public.js` holds every public word.

| | Showcase | Explorer |
| --- | --- | --- |
| Modes | Play · Dilemma · Player evaluation · Nash equilibrium | Observed · Counterfactual · Evaluation · Game solution |
| Caption | `29:20 · <the reviewer's sentence>` | `29:20 · shot 10 · Kownacki · SAVED · …` |
| Selector | `Fortuna Düsseldorf vs SSV Jahn Regensburg · 29:20` | `S05 · … · 5/5 · Human-reviewed` |
| Cast | a three-line legend | the role dock, drag and drop |
| Commands | Close down · Track runner · Drop · Cover teammate | toward ball · toward runner · toward goal · toward beneficiary |
| Sidebar | the mode's own card | scene, player, decision, mode, clip, source |
| Layers | one collapsed control, three marks | the row plus Advanced |
| Panel | Dilemma · Defender · Attack · If the defender froze | nine rows plus the provenance string |

**No identifier reaches the showcase.** `S05`, shot numbers, `Human-reviewed`,
`Solver-derived`, `Auto triplet`, the `5/5` badge, frame numbers and the
provenance string are all explorer or Details copy. Measured on S05 across the
four modes, the showcase went from 1290 visible words to 380 — **71% less**.

**One defect this closed.** The clicked-player block followed `state.inspect`
while the decision and evaluation rows followed the curated story role, so
clicking any background player relabelled half the card — a probe reproduced
it: clicking #3 A. Hoffmann made the panel read "#3 A. Hoffmann · Role:
Runner" beside "Runner · #10 D. Ginczek". In the showcase a click now changes
nothing (the cast is curated and there is no picker), and the clicked-player
block is not part of the public panel at all. The explorer keeps click-to-pick,
where reassigning the cast is the point.

## 12. The defender-start grids: what to drop in (2026-10-01)

`js/grid.js` reads them and nothing else does. It is dormant today — the
`Defender field` control is hidden unless every moment resolves — and lights up
when the files exist. The field is **never** reconstructed from the figure PNG.

**Upstream produces them** with `jobs/solve_defender_grid.sbatch`, which runs
`build_defender_grid.py` → the 2v1 solver → `extract_defender_grid.py` and
writes, for `CODE` in `S05`, `S05@0.6`, `S05@1.2`:

```
data/processed/showcase_v1/defender_grid_S05/grid_${CODE}_1m_full.json
```

Those paths are confirmed from the sbatch at `origin/kyuhyeok-dev@b5f26cb`.
They are not committed (`.gitignore:8` excludes `data/processed/*`), are not in
the 2026-09-30 bundle, and are not on this machine.

**Where the demo wants them:** `demo_viz/web_data/grid/S05_0.0.json`,
`S05_0.6.json`, `S05_1.2.json` — the same files, renamed by moment.

**The schema `grid.js` reads**, which is what `extract_defender_grid.py` writes:

```json
{ "points": [ { "status": "solved",
                "observed": false,
                "defender_start": [x, y],
                "value": 0.6209,
                "defender": [ { "prob": 0.39, "end": [x, y] }, ... ] } ] }
```

Only `status`, `defender_start`, `value` and `defender[].{prob,end}` are read.
Points that are not `"solved"` are skipped, as upstream skips them.

**The two derived fields, exactly as upstream derives them:**

| | from | upstream |
| --- | --- | --- |
| background shade | `point.value` at `point.defender_start` | `render_figure2_abstract.value_grid` |
| move field | `Σ prob × (end − start)` over his commands | `render_defender_flow_moments.moves` |

Shading runs `#F7F7F7` → `#B8AE9C` with **darker = lower value = a better start
for the defender**, and the figure's own key is `darker: better start for the
defender`. The flow is `--flow-color` `#FF0000` at `--flow-alpha` 0.2.

## 13. Why there is no "Selection checks" block (2026-10-01)

The abstract's extraction has two scene-level filters
(`scripts/filter_stage3_states.py`). Evaluated for S05 with upstream's own
`distance_to_triangle` and `inside`, on bundle tracking whose positions match
the panels to 0.0000 m:

| filter | threshold | S05 | verdict |
| --- | --- | --- | --- |
| run quality (mean run speed over 0.8 s after onset) | ≥ 3.8 m/s | **not applicable** | — |
| isolation (nearest outsider to the carrier/runner/defender triangle, at 0.0 / 0.6 / 1.2 / 1.8 s) | ≥ 2.0 m | **1.81 m** (Benedikt Gimber, at 0.0 s) | **FAIL** |

Run quality does not apply because, as `build_showcase_states.py` says, these
starts are "release − 1.8 s", not run onsets.

Across the seven showcase scenes with solved panels: S13, S15, S20, S36 and S44
pass the isolation filter; **S05 (1.81 m) and S34 (1.19 m) fail it**.

This is not a defect. Upstream applies neither filter to these scenes —
`build_showcase_states.py` calls the isolation filter *"Reported, not applied"*
— **because the showcase scenes were picked by hand**, and a hand-picked scene
the filter would have dropped is still in the set by design.

So the showcase shows **no selection-checks block**, and nothing in it says or
implies that a play was extracted automatically. Dilemma asks "Why is this a
dilemma?" and answers with tracking evidence about the play; `compare.js` and
`export_compare.py` both say so in their headers.

## 14. Dilemma's "Compare players" (2026-10-01)

Three quantities, all repository computations, none of them a selection rule.

| shown as | computed by | formula | unit | better |
| --- | --- | --- | --- | --- |
| Marking distance | `dynamic_marking.marking_sample` → `weighted_error_m` | `\|defender − goal-side target\| + 2.0 × wrong-side displacement`, meaned from the run's onset | m | lower |
| Reaction | `role_logic.defender_reaction_index` | the repository's kinematic onset detector, else speed ≥ 1.5 m/s and pursuit alignment ≥ 0.30 held for 0.40 s | s after the run starts | lower |
| Space created | `goal_weighted_influence.target_residual_influence` | residual space now, minus the same with the defender held to his onset position | m² | higher |

The first two are exported by `demo_viz/web/export_compare.py` into
`data/compare/<scene>.json`, because the browser does not implement those two
modules; the third is the influence cache's own number, which
`tests/test_demo_viz_app.py` pins against the repository to 1e-12.

**Not shown:** `role_logic.rank_defenders`'s `score`. It is
`0.55·max(pursuit,0) + 0.45·clip(1 − d/25)`, and the 0.55/0.45 is the app's own
weighting — `ranking.js` calls the whole thing "a transparent heuristic, not a
learned model of defensive intent". Marking distance and reaction say the same
thing in metres and seconds, from a repository module.

**Reaction reads "already pursuing", not "0.0 s"**, when the pursuit rule's
condition was already true as the run began. Zero there would claim a reaction
the tracking never shows starting.

**Selection is not editing.** A comparison lives in `state.compare` and is
never written to `state.selection`, so no equilibrium, rank, regret or
evaluation number can move with it — those exist only for the curated cast.
Leaving Dilemma clears it, and `Reset` clears it in place. Three tests hold
that line.

For S05 the curated defender is also the tightest marker of the runner
(5.8 m against 8.1 m for the next-nearest, #8 Thalhammer), which is the
evidence the mode exists to show.
