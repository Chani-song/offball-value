# Methodology v0.1

## Purpose and scope

This document freezes the definitions used to create the current human-reviewed
scenes and the structural local-game audit. It deliberately separates what is
already implemented from the final threat and optimization model.

## Data convention

The primary data are synchronized IDSSE tracking, event, and metadata files.
Tracking is sampled at 25 Hz on a 105 m by 68 m pitch. Internal player and ball
coordinates are centered at `(0, 0)`; attack direction is normalized only when
a model or visualization explicitly requests it.

## 1. Shot-context sampling

Open-play shot events retrieve attack-like portions of the match. A shot is an
anchor for sampling, not an outcome attributed to an earlier off-ball action.

The current `ShotContextConfig` uses:

| Parameter | Value |
| --- | ---: |
| Maximum lookback | 60.0 s |
| Comparison windows | 10, 15, 30, 60 s |
| Foot-control distance | 1.5 m |
| Maximum controlled ball height | 1.5 m |
| Confirmed opponent-control duration | 0.32 s |

The inferred phase is bounded by the available lookback and confirmed changes
of control. The extraction also records possession-start and attacking-half
entry alternatives so the sampling boundary can be audited.

## 2. Retrospective run-onset detection

An onset is the earliest persistent kinematic departure from the runner's
preceding movement state. It is not simply the first frame above an absolute
speed threshold.

The detector considers three movement types:

- acceleration;
- direction change;
- deceleration followed by a turn and reacceleration.

Important default settings are:

| Parameter | Value |
| --- | ---: |
| Smoothing window | 0.20 s |
| Pre-onset window | 0.50 s |
| Post-onset confirmation | 0.80 s |
| Displacement horizon | 1.00 s |
| Minimum speed gain | 1.50 m/s |
| Minimum direction change | 30 degrees |
| Minimum speed for turn detection | 2.0 m/s |
| Minimum post-onset displacement | 3.0 m |
| Merge window | 0.50 s |
| Controlled-possession history | 0.50 s |
| Control distance | 1.50 m |
| Required tracked players | 22 |

Future samples are used only to verify that a detected change persists. They
create retrospective episode labels and must not be passed to an online action
optimizer.

The v0.3 detector also requires the ball to lie within the tracked pitch plus a
0.20 m tolerance. This deterministically removes throw-in waiting states found
during human review.

## 3. Settled-possession gate inside the shot context

The additional shot-context gate evaluates the one second preceding an onset.
It requires sufficient known control, sufficient attacking-team control, and a
high same-team fraction among frames with known control. It excludes unstable
event windows, recent restarts, onsets with less than two seconds before the
shot, and—by default—states outside the opponent half.

These rules are development filters. Transition and contested states are
retained as a secondary diagnostic cohort rather than being used to tune more
kinematic thresholds from a small number of examples.

## 4. Human review and the development set

Human review is used at several gates:

1. shot-context suitability;
2. run-onset timing and possession context;
3. visible local interaction;
4. second-pass confirmation of reacting defender, derived option, and trade-off.

The current second pass reviewed ten clear candidates:

- eight were included;
- two were held;
- none were rejected.

The eight included scenes come from four matches. They form a high-precision
development set for debugging the local-game representation, not a sampled
estimate of prevalence or model accuracy.

## 5. Feasible movement prototypes

The action-space work separates endpoint coverage from the existence of a
physically plausible witness path. The latest prototype combines curved
steering with plant-and-cut behavior to improve coverage behind a fast-moving
player. Parameters were calibrated from leave-one-match-out movement
primitives, while evaluation scenes were excluded from their calibration
library.

The structural audit does not yet perform the outer search over these attacker
actions. It uses observed attacker futures so that defender-option allocation
can be checked before the threat and optimization layers are added.

## 6. Dynamic goal-side defender response

A response targets a moving point on the attacker's goal side rather than the
attacker's old position. At time `t`, the defender anticipates the attacker at
`t + 0.4 s`, targets a point 1.5 m toward the defended goal, and follows bounded
steering dynamics after a 0.2 s response delay.

The current structural response configuration is:

| Parameter | Value |
| --- | ---: |
| Horizon | scene-dependent, 1.2 to 3.0 s |
| Integration step | 0.1 s |
| Pre-onset velocity history | 0.4 s |
| Response delay | 0.2 s |
| Lookahead | 0.4 s |
| Goal-side offset | 1.5 m |
| Maximum speed | 9.0 m/s |
| Maximum tangential acceleration | 4.5 m/s² |
| Maximum deceleration | 6.0 m/s² |
| Maximum normal acceleration | 6.0 m/s² |

These are unified development parameters, not player-specific physical limits.
They require sensitivity analysis before final inference.

## 7. Structural defender-by-option audit

For each confirmed scene:

1. use the observed runner path from onset to the audit horizon;
2. retain observed futures for the ball carrier and outfield teammates;
3. simulate each outfield defender responding to the runner;
4. simulate the same defender responding to each non-runner option;
5. measure dynamic goal-side marking error at matched times;
6. select three candidate defenders and display five local options.

The ball carrier is always retained among the displayed options. Goalkeepers
and the focal runner are excluded from the non-runner option list.

For defender `D` and option `K`, the audit reports:

- `O`: how much worse control of `K` becomes when `D` responds to the runner
  rather than purposefully covering `K`;
- `R`: how much worse control of the runner becomes when `D` covers `K` rather
  than purposefully responding to the runner;
- structural score: `min(O, R)` after within-cell fractional normalization.

These quantities are geometry allocation effects. They are useful for asking
whether a defender-option pair can express a two-sided conflict, but they are
not probabilities, goal values, or final dilemma labels.

## 8. Provisional threat decomposition

The current direction for one attacking option `k` is:

\[
Q_k(a,d)
=
P_{\mathrm{delivery},k}(a,d)
\times G(z_k)
\times A_{\mathrm{goal-side},k}(a,d).
\]

- `P_delivery`: successful delivery by pass or ball carry;
- `G`: goal danger of the resulting location;
- `A_goal-side`: accessible space between the attacker and goal after delivery.

Prototype modules exist for parts of this decomposition, but their scales are
not calibrated and the formulation is not frozen. The structural audit must
not be interpreted as already using this equation.

## 9. Background convention

Observed futures are used as the development audit background because simple
hold, constant-velocity, damped-velocity, and empirical-reference rollouts
distorted some real short-horizon paths. This is a retrospective diagnostic
choice, not a deployable forecasting solution. In a final counterfactual, the
focal runner and selected defender cannot simultaneously use their observed
future as an optimization input.

## 10. Current claim boundary

The v0.1 outputs support qualitative and structural debugging only. They do not
support causal claims, a population frequency of defensive dilemmas, calibrated
threat estimates, player comparisons, or claims of optimal movement.
