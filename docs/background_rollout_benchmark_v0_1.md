# Background rollout benchmark v0.1

## Decision

This benchmark does not justify using constant velocity as the only causal
background for every non-intervened player over a two-second horizon.

For outfield players, the main background rollout is a deterministic empirical
reference trajectory obtained from leave-one-match-out movement primitives.
Goalkeepers use hold as a conservative fallback. Constant velocity and damped
constant velocity remain required baselines.

The current structural v0.1 audit instead uses observed attacker futures as a
retrospective diagnostic background. That choice is explicitly non-deployable
and does not turn the observed future into an optimizer input. When the outer
attacker search is introduced, empirical reference, constant velocity, and
other causal backgrounds must be compared through sensitivity analysis.

## Cohort and leakage control

- Target match: `DFL-MAT-J03WMX`.
- Source library: the other six public Bundesliga matches.
- Target-match primitives in the source library: zero.
- Decision scenes: 94 extractor-accepted scenes plus human-valid overrides at
  frames 13413 and 19013, for 96 scenes total.
- Players per scene: 22.
- Evaluation horizons: 0.5, 1.0, 1.5, and 2.0 seconds.
- Observed future tracking is used only as the held-out benchmark target.
- The causal state estimate uses the 0.4 seconds ending at the decision frame.

## Compared models

### Hold

The player remains at the decision-frame position and has zero terminal
velocity.

### Constant velocity

The player retains the causally estimated velocity, capped at 9 m/s, for the
full horizon.

### Damped constant velocity

Velocity decays as `v(t) = v(0) exp(-lambda t)`. The source-library fit selected
`lambda = 0.19 /s`. No target-match future is used to fit this value.

### Empirical reference

For each outfield player, 512 source-library primitives are selected using
causal initial speed and longitudinal acceleration. A Gaussian-kernel weighted
conditional mean provides path position and terminal velocity. Paths are
rotated from the primitive's velocity-aligned coordinates into the target
player's current heading.

The current empirical reference does not condition on player identity, role,
team tactics, ball-relative geometry, or the proposed attacker action.

## Held-out player prediction results

Mean endpoint error for all outfield players:

| Horizon | Hold | Constant velocity | Damped CV | Empirical reference |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 1.245 m | 0.191 m | 0.194 m | **0.176 m** |
| 1.0 s | 2.439 m | 0.661 m | 0.662 m | **0.608 m** |
| 1.5 s | 3.562 m | 1.373 m | 1.341 m | **1.254 m** |
| 2.0 s | 4.628 m | 2.285 m | 2.187 m | **2.071 m** |

At two seconds, the empirical reference also had the smallest 90th-percentile
endpoint error:

| Model | Mean error | Median error | 90th percentile |
| --- | ---: | ---: | ---: |
| Hold | 4.628 m | 3.940 m | 8.699 m |
| Constant velocity | 2.285 m | 1.780 m | 4.846 m |
| Damped CV | 2.187 m | 1.752 m | 4.453 m |
| Empirical reference | **2.071 m** | **1.626 m** | **4.348 m** |

Scene-clustered paired differences at two seconds were:

| Comparator minus empirical | Mean | Paired 95% CI | Scenes where empirical was better |
| --- | ---: | ---: | ---: |
| Constant velocity | 0.214 m | [0.138, 0.290] m | 77/96 |
| Damped CV | 0.116 m | [0.079, 0.153] m | 69/96 |
| Hold | 2.557 m | [2.166, 2.948] m | 92/96 |

Positive differences favor the empirical reference. The scene is the
independence unit for the interval calculation.

## Formation geometry at two seconds

The empirical reference also improved most team-shape diagnostics relative to
constant velocity.

| Team phase | Model | Centroid error | Pairwise-distance MAE | Offside-line error |
| --- | --- | ---: | ---: | ---: |
| In possession | Constant velocity | 1.080 m | 1.654 m | n/a |
| In possession | Damped CV | 1.111 m | 1.508 m | n/a |
| In possession | Empirical reference | **1.074 m** | **1.488 m** | n/a |
| Out of possession | Constant velocity | 1.541 m | 1.949 m | 1.413 m |
| Out of possession | Damped CV | 1.567 m | 1.761 m | 1.325 m |
| Out of possession | Empirical reference | **1.522 m** | **1.760 m** | **1.222 m** |

One exception is defending-team depth: empirical reference had 2.120 m error,
compared with 1.826 m for constant velocity and 1.680 m for damped CV. This
metric must remain in the later OBSO sensitivity analysis.

## Interpretation

Constant velocity is a useful causal baseline, especially at 0.5 seconds, but
it is not the best two-second background state for these scenes. The empirical
reference has a modest but consistent held-out advantage and causes less
within-team geometric distortion on most metrics.

The remaining two-second error of 2.071 m is material. Therefore the selected
model must be described as a reference rollout rather than an accurate
prediction of the counterfactual team response. It deliberately does not claim
that non-intervened players react to the proposed run.

## Visual audit

The audit contains 12 scenes chosen without cherry-picking:

- four scenes where empirical reference is worse than constant velocity;
- four scenes near the median difference;
- four scenes where empirical reference is better.

Frame 14913 is intentionally included in the empirical-worse stratum. Reviewers
should inspect whether the gold empirical trajectories are plausible as a
neutral background, not whether they reproduce every observed player exactly.

## Implementation outputs

- The compact summary, model-selection, uncertainty, formation, provenance,
  manifest, and benchmark figure are preserved in
  `examples/research_audit/evidence/background_rollout/`.
- Full player-level predictions and the generated interactive audit remain
  reproducible local outputs under ignored `data/processed/`.

## Next integration rule

For a future decision state `s0`, compute one causal reference background
`b(s0)`. For every attacker action `a`, responding defender `j`, and defender
response `d`, construct

```text
state_H(a, j, d; b)
```

by overriding only the focal attacker and the one responding defender. The
same causal background must be reused across compared actions. Empirical
reference is the current data-driven baseline, not a claim that all
non-intervened players truly follow that trajectory or react to the proposed
run.
