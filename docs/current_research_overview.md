# Current research overview

## One-sentence objective

Identify off-ball runs that force a feasible defender allocation trade-off and
retain attacking threat even after the defender's best response.

## Motivation

An off-ball run can matter in at least two ways. The runner may become a direct
receiving or finishing threat, and the defender's response to that run may
release the ball carrier, another attacker, or a useful progression space.
These effects are often missed when value is assigned only to the player who
receives the ball or when the defense is held fixed.

The project therefore asks a conditional question: if the runner moves, how
should a physically constrained defender reallocate, which attacking options
does that response protect or release, and how much attacking threat remains?

## Main research question

> Can we identify off-ball movements that maximize residual attacking threat
> after a physically feasible defensive best response?

Supporting questions are:

1. Which defenders can materially change both the runner's direct opportunity
   and at least one derived attacking option?
2. Does protecting the runner release another local option, and does protecting
   that option release the runner?
3. Does a compromise response control both options, or does meaningful residual
   threat remain?
4. How does the observed defender movement compare with feasible purpose-built
   responses?
5. Once the value model is validated, which feasible runner action performs
   best under the defensive response set?

## Unit of analysis

The unit is a **local off-ball game embedded in the observed 11-v-11 state**.
It is not defined by a fixed player count or a proximity-only 2-v-1 rule.

- `B`: the ball carrier at run onset;
- `R`: the focal off-ball runner;
- `D`: one candidate defender whose feasible response can change the balance;
- `K_D`: the local attacking option set affected by defender `D`.

`K_D` includes the ball carrier and may include one or more off-ball teammates.
A beneficiary is not fixed before the defender is considered: changing the
candidate defender can change which attacker benefits.

The remaining players are not deleted. They remain in the state as observed
context. The current counterfactual scope changes one runner action and one
defender response at a time; joint multi-defender optimization is future work.

## Intended game sequence

```text
runner action
  -> candidate defender response
  -> attacking option re-selection
  -> residual direct and derived threat
```

For a feasible runner action `a` and defender response `d`, let

\[
T_R(a,d)
\]

denote the runner's direct threat, and let

\[
T_k(a,d), \qquad k\in K_D\setminus\{R\}
\]

denote the threat of each affected non-runner option. The derived threat is an
aggregation over this **local affected set**, not an unconditional maximum over
all attackers on the pitch:

\[
T_{\mathrm{derived}}(a,d)
=
\operatorname{Agg}_{k\in K_D\setminus\{R\}} T_k(a,d).
\]

The aggregation rule is not yet frozen. Candidate definitions include a local
maximum, a top-two aggregation, and an expected value under an attacking choice
model.

Once the threat function and aggregation are validated, the intended outer
problem is:

\[
a^*
=
\arg\max_{a\in\mathcal A_R}
\min_{d\in\mathcal D(a)} T(a,d).
\]

This max-min expression is the target design, not a claim about the current
structural audit.

## Defensive allocation dilemma

For a fixed runner action, define `d_R` as the response that best controls the
runner and `d_K` as the response that best controls the selected derived
option. A two-sided allocation conflict requires both cross-costs to be
material:

\[
\Delta_R = T_R(a,d_K)-T_R(a,d_R),
\]

\[
\Delta_K = T_{\mathrm{derived}}(a,d_R)
-T_{\mathrm{derived}}(a,d_K).
\]

If one feasible response keeps both quantities low, the scene is not a strong
dilemma even if it visually resembles a 2-v-1. The threshold for calling both
cross-costs material remains an empirical research decision.

## Current evidence and implementation boundary

The current development set contains eight human-confirmed observed scenes
from four IDSSE matches. These scenes were selected because a reviewer could
identify a plausible reacting defender, a derived opportunity, and a visible
trade-off. They are development examples, not positive labels produced by the
model and not a population estimate of dilemma frequency.

The structural v0.1 audit currently does the following:

- uses the observed runner and teammate futures retrospectively;
- generates bounded, target-updating responses for each outfield defender;
- ranks three candidate defenders by their ability to control the observed
  runner future;
- compares five displayed local attacking options, always retaining the ball
  carrier;
- measures two-sided changes in dynamic goal-side marking geometry.

This audit is shared as an **in-progress working prototype** so collaborators
can inspect the current representation and guide the next implementation. It
is not presented as a completed local-game model or a research result.

It does **not** yet:

- search counterfactual runner actions in the outer loop;
- predict attacker futures online;
- estimate a calibrated probability of a pass, shot, or goal;
- solve a validated defender best-response objective;
- establish causal attribution from the run to the later shot;
- support player ranking or coaching recommendations.

## Expected research contributions

If the remaining components validate, the intended contributions are:

1. an interpretable local-game formulation centered on an off-ball runner;
2. defender-dependent option attribution rather than a preselected beneficiary;
3. physically bounded, dynamically updated defensive responses;
4. separation of direct runner threat from derived teammate threat;
5. identification of residual threat and defensive allocation dilemmas;
6. comparison between observed defense and feasible purpose-built responses.

## Open decisions

The following choices must be resolved before a final experiment:

1. the calibrated threat definition for pass, carry, reception, and finishing;
2. the rule for including an attacker in a defender's affected local option set;
3. the aggregation of multiple derived options;
4. the final feasible defender response set and objective;
5. the counterfactual runner action set used in the outer maximization;
6. the empirical threshold for a meaningful two-sided dilemma;
7. the evaluation protocol beyond the eight-scene development set.
