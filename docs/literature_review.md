# Related-work map

## Scope

The project sits between off-ball opportunity valuation, pass availability,
space generation, trajectory counterfactuals, and multi-agent tactical
response. The specific target is narrower: value an off-ball runner while
allowing a local defender to respond feasibly and exposing the trade-off
between the runner and the attacking options released by that response.

## OBSO and possession-value models

OBSO-style work decomposes a dangerous opportunity into transition,
pitch-control, and scoring-location components. EPV and action-value models
similarly provide a value for a state or a possible on-ball action.

These models are useful evaluators, but they do not by themselves identify
which off-ball run created an option or how a defender would relocate after a
counterfactual run. The project therefore treats a state-value model as a
component inside a response game rather than as the primary contribution.

## Pass availability and reception probability

Pass-probability and availability models estimate whether the ball can reach a
receiver before interception and whether the receiver can secure possession.
This is a necessary component of direct and derived option threat.

The remaining attribution problem is causal and strategic: which runner action
changed the delivery probability, which defender response produced that
change, and what alternative option was released?

## Fernandez-style influence and space generation

Continuous influence models distinguish occupying useful space from generating
space for a teammate. They provide a natural representation for decoy runs and
for goal-side accessibility after reception.

Influence alone is not a pass-success or goal-value model. In this project it
is used as an interpretable geometric ingredient and audit layer, not as a
standalone final threat score.

## Counterfactual trajectory approaches

C-OBSO and related trajectory-prediction work compare actual player movement
with a predicted reference trajectory to attribute opportunity creation. This
is an important precedent for non-receiving runner value.

The current project differs by making the defensive response itself explicit:
candidate defenders and their feasible response paths are compared, and the
benefiting attacking option is allowed to depend on which defender reallocates.
The current v0.1 audit still uses observed attacker futures retrospectively;
online trajectory prediction remains a later extension.

## Reachable-region and self-propelled-particle models

Physics-based reachable regions and self-propelled-particle models provide a
way to restrict counterfactual actions by current velocity, acceleration,
deceleration, and direction-change limits. They motivate the project's bounded
steering and plant-and-cut action spaces.

Reachability defines what a player can do, but it does not decide which action
is tactically valuable. That decision requires the local threat and response
game.

## Defensive reaction optimization

DRSO-style defensive search compares a small number of alternative defender
locations around an identified threat point. It provides an interpretable
baseline for asking where a defender could reduce an off-ball opportunity.

The current direction replaces a static endpoint-only response with a bounded,
target-updating path and compares multiple candidate defenders. A faithful
baseline reproduction and a controlled search-method comparison remain future
experiments.

## TacticAI and graph-based tactical generation

TacticAI represents player relations as a graph and adjusts team positions in
set-piece contexts. It demonstrates that generated tactical states can be
compared with empirical movement distributions.

Its set-piece and team-adjustment scope differs from the current open-play,
runner-centered local allocation game.

## Multi-agent reinforcement learning

Multi-agent RL can model coordination among attacking agents and assign value
to on- and off-ball actions. It is attractive for long-term joint optimization
but requires more data, validation, and interpretability work.

The present framework intentionally starts with one focal runner and one
counterfactual defender so that response paths, released options, and failure
modes remain inspectable.

## Research gap

Existing work covers spatial value, pass availability, trajectory prediction,
reachable actions, and learned tactical coordination. The combination that
remains underdeveloped is:

1. an off-ball runner as the focal attacking intervention;
2. several plausible responding defenders rather than a preselected nearest
   defender;
3. physically feasible, dynamically updated defender trajectories;
4. direct runner threat separated from defender-dependent derived threat;
5. a best-response and max-min interpretation with an explicit two-sided
   defensive allocation cost.

This combination defines the intended contribution of the project. The current
repository implements the scene, motion, and structural-audit layers; it does
not yet validate the final threat or optimization layers.

## Core references to formalize before submission

- Spearman, OBSO / *Beyond Expected Goals* line of work.
- Fernandez and Bornn, *Wide Open Spaces*.
- Fernandez, Bornn, and Cervone, instantaneous EPV framework.
- Dick, Link, and Brefeld, pass availability.
- Teranishi et al., C-OBSO / trajectory-based opportunity creation.
- Reachable-region and self-propelled-particle validation work.
- TacticAI.
- Multi-agent on/off-ball action-value research.

Exact bibliographic records should be maintained in a `.bib` file before a
paper submission. Copyrighted article PDFs are intentionally excluded from the
repository.
