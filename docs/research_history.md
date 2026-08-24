# Research history

This document records superseded approaches only when they explain a current
methodological decision. It is not the specification of the current model.

## Early OBSO-style exploration

The project initially used an OBSO-style proxy to estimate whether a non-
receiving runner improved the ball carrier's best passing option. The proxy
combined a heuristic transition term, simplified pitch control, an EPV-based
location score, and post-reception space. Actual states were compared with a
counterfactual in which the runner and one affected defender were rolled back.

This exploration exposed several limitations that shaped the current project:

- Event coordinates and tracking coordinates can be materially misaligned. A
  passer-to-receiver vector should therefore use a consistent tracking frame;
  one early ranked scene was an artifact of a roughly 25.6 m mismatch.
- Sampling only completed pass events is too narrow for an off-ball decision
  problem. Shot-context windows and controlled-possession states are now used
  to find candidate attacking situations before detecting run onset.
- Preselecting the nearest defender does not capture who can respond to a run
  or which attacking option that response releases. The current framework
  compares multiple candidate defenders and defender-dependent local options.
- A team-wide maximum option value can ignore a local allocation conflict when
  an unrelated attacker has the largest value. The analysis is therefore
  centered on a local off-ball game rather than a global maximum alone.
- The proxy output was a relative score, not a calibrated scoring probability.
  It is not treated as the final threat model.

## Transition to the local off-ball game

The current direction treats the off-ball runner as the focal intervention.
Other players remain in the observed match background, while feasible runner
actions and the response of one candidate defender are varied. The analysis
then measures the runner's direct threat and the threat of other attacking
options affected by that defender's allocation.

The intended threat decomposition is currently expressed as delivery
probability, goal danger at the target location, and goal-side accessibility.
These components and the defender best-response objective remain under
development and should not yet be interpreted as a validated final value
function.
