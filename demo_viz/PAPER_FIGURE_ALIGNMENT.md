# Paper-figure alignment

What the demo shares with the paper figures, what it could not share, and why.

Audited against `origin/kyuhyeok-dev` @ `e84553a` (2026-09-30).

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
