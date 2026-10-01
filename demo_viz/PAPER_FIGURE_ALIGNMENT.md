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
