# Literature Review

## Scope
This document summarizes prior work most relevant to an early project on **off-ball attacking value** in soccer. The main question is not simply whether space exists, but whether an attacking player's off-ball movement changes the quality of teammates' options and increases the downstream value of the possession.

A useful decomposition for the project is:

1. off-ball movement,
2. defensive reaction or defensive displacement,
3. change in available attacking options,
4. change in option success probability or option value,
5. change in possession value.

The papers below cover different parts of this chain.

---

## 1. Spearman, Beyond Expected Goals / OBSO

### Main idea
This line of work introduced an off-ball scoring-opportunity view of soccer, modeling how likely it is that a dangerous attacking event will occur at a given location. It is foundational because it moves beyond on-ball outcomes and tries to quantify whether a player is threatening even without the ball.

### Modeling idea
A representative formulation decomposes scoring opportunity into terms such as:
- probability that play reaches location $r$,
- probability that the attacking team controls the ball there,
- probability that the state leads to a goal.

This makes OBSO a product-like decomposition of "reachability", "control", and "danger".

### Why it matters
- Important conceptual starting point for off-ball evaluation.
- Gives a state-value view that later work can extend.
- Influenced later studies such as C-OBSO and OBPV.

### Limitation for our project
- Strongly goal-oriented.
- Better at evaluating dangerous receiving locations than attributing credit to the **non-receiving runner** who created the opportunity.
- Does not directly quantify how one player's run improved a teammate's pass or shot option.

---

## 2. Fernandez and Bornn, Wide Open Spaces

### Main idea
This paper separates:
- **space occupation**, meaning a player finds valuable space for themselves, and
- **space generation**, meaning a player creates space for teammates.

This is one of the closest conceptual precursors to dummy runs, decoy runs, and gravity-like attacking effects.

### Modeling idea
The paper builds a continuous-space model using player influence and pitch control. Space quality is expressed as a combination of:
- who controls a location, and
- how valuable that location is.

In practice, this becomes a framework for measuring whether a player's movement opens useful space for others.

### Why it matters
- Probably the clearest early framing of off-ball movement as a team-serving action.
- Gives language and structure for concepts like decoy movement and defender attraction.
- Very relevant if the project wants to talk about space creation without requiring an immediate pass or shot.

### Limitation for our project
- Strong on **space geometry**, weaker on **option value**.
- Does not fully connect space creation to pass success, receiver value, or possession-value change.
- Good as an upstream ingredient, but not sufficient as the final metric.

---

## 3. Fernandez, Bornn, Cervone, Instantaneous EPV Framework

### Main idea
This work formalizes **expected possession value (EPV)** as the value of a possession state at a given moment. It decomposes possession value into submodels so that both observed and hypothetical actions can be evaluated.

### Modeling idea
EPV is treated as a state-value problem and decomposed into components such as:
- action selection,
- action success,
- action outcome value.

This is especially useful because it turns a possession into a structured set of probabilities and values instead of a single end result like goal or no goal.

### Why it matters
- Strong downstream target for our project.
- Suggests that the right response variable is often **possession value change**, not only xG.
- Makes it possible to say that an off-ball movement was useful even if no shot immediately followed.

### Limitation for our project
- EPV tells us **how valuable a state or action is**.
- It does not directly solve **who should get credit** for increasing that value.
- A separate attribution layer is still needed for off-ball runners.

---

## 4. Dick, Link, Brefeld, Availability

### Main idea
This paper defines **availability** as the probability that a target player can receive a pass without interception. It is a direct option-quality model rather than a pure spatial model.

### Modeling idea
Availability aggregates:
- ball dynamics,
- player movement constraints,
- opponent interception risk,
- execution uncertainty.

The result is a pass-receiving probability for a candidate target.

### Why it matters
- Very relevant for a project about how off-ball runs change teammates' options.
- Can serve as an intermediate model for "which receiver became more open because of a run".
- More actionable than generic space value when the research question is pass-option creation.

### Limitation for our project
- Measures whether a player can receive a pass.
- Does not tell us **which off-ball player created that availability improvement**.
- Excellent module, but not a complete off-ball attribution framework.

---

## 5. Teranishi et al., C-OBSO

### Main idea
This is one of the closest prior studies to our idea. It evaluates players who create scoring opportunities for teammates by comparing actual movement with a predicted reference trajectory.

### Modeling idea
The method combines:
- a modified OBSO-like value,
- trajectory prediction using a graph-based recurrent model,
- a counterfactual comparison between actual and reference movement.

The key idea is that if actual movement creates more teammate scoring opportunity than the predicted baseline movement, the difference can be credited to the player.

### Why it matters
- Strong direct precedent for evaluating a player who does **not** receive the ball.
- Introduces a counterfactual logic that is highly relevant for decoy-run valuation.
- Shows that comparing actual movement against a reference movement is a plausible route.

### Limitation for our project
- Very focused on **scoring opportunity**.
- Less explicit about how the run changes the whole menu of pass options.
- Still leaves room for a more interpretable option-level decomposition.

---

## 6. Multi-agent RL-based on/off-ball valuation

### Main idea
This line of work treats soccer as a multi-agent sequential decision problem and tries to value both on-ball and off-ball actions through learned action-value estimates.

### Modeling idea
The framework uses reinforcement learning and state-action value estimation. In principle, it can evaluate many simultaneous off-ball behaviors at once.

### Why it matters
- Shows that off-ball movement can be embedded in a unified action-value framework.
- Suggests one possible long-term direction if we want a more ambitious multi-agent model.

### Limitation for our project
- Harder to interpret.
- May require more data and engineering than an early-stage open-data project can realistically support.
- Less attractive as a first prototype if our goal is a clear, explainable research scaffold.

---

## 7. Ogawa et al., OBPV

### Main idea
OBPV extends the off-ball valuation idea so that areas far from goal, especially transition-starting areas, are evaluated more meaningfully.

### Modeling idea
The method modifies the standard OBSO style by introducing broader field value and transition-aware components rather than relying too heavily on immediate goal proximity.

### Why it matters
- Important warning against making the metric too shot-centric.
- Relevant if we want to value off-ball movement earlier in the attack, not only near the box.
- Helps motivate a project that captures option creation before the final action.

### Limitation for our project
- Still primarily a **space valuation** method.
- Does not fully solve player-level attribution for off-ball option creation.

---

## Summary of the gap
The literature already covers several important building blocks:
- spatial value,
- pitch control,
- pass availability,
- possession value,
- counterfactual off-ball evaluation.

What remains relatively underdeveloped is a framework that explicitly measures:

> how a non-receiving attacker's off-ball movement changes the quality of teammates' options, and how much of the subsequent possession-value increase should be attributed to that runner.

That is the main gap this project should target.

---

## Working research direction
A practical first research framing is:

> Quantify off-ball attacking value by measuring how an attacker's movement changes teammates' option quality relative to a counterfactual reference state.

A stronger version is:

> For each attacking state, estimate candidate teammate options, their success likelihood, and their downstream possession value; then attribute improvements in those quantities to specific off-ball movements using a counterfactual comparison.

---

## Notes for this repository
The current repository does **not** implement the full research metric yet. At this stage, it provides:
- open-data loaders,
- a Metrica-based toy baseline,
- a problem framing for future work.

So the repo should be presented as an **early research scaffold**, not a finished evaluation system.

---

## Selected references
- Spearman, Beyond Expected Goals / off-ball scoring opportunity line.
- Fernandez and Bornn, *Wide Open Spaces*.
- Fernandez, Bornn, Cervone, *A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions*.
- Dick, Link, Brefeld, *Who can receive the pass?*.
- Teranishi et al., *Evaluation of creating scoring opportunities for teammates in soccer via trajectory prediction*.
- Ogawa et al., *Space evaluation at the starting point of soccer transitions*.
