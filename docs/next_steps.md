# Next Steps

## Project status snapshot
This repository is currently an **early research prototype**.

What is already in place:
- repository scaffold,
- open-data notes,
- loaders for public datasets,
- a runnable Metrica-based toy baseline,
- literature and idea notes.

What is **not** in place yet:
- a validated off-ball value metric,
- robust attribution logic,
- cross-dataset evaluation,
- polished figures or paper-ready experiments.

The immediate goal is therefore **not** to claim a finished method, but to preserve the project structure and make the next research steps easy.

---

## Immediate priorities

### 1. Preserve the current state cleanly
Before adding complexity, the repository should serve as a reliable project snapshot.

Recommended actions:
- keep README honest about current scope,
- commit literature and project notes,
- avoid pushing raw data,
- keep processed outputs lightweight.

### 2. Inspect the toy baseline outputs
The current Metrica baseline should be treated as a sanity-check tool.

Questions to ask:
- Do the top-ranked sequences look like plausible off-ball runs?
- Are high scores mostly driven by real movement or by artifacts of the heuristic?
- Is the counterfactual logic doing something interpretable?

If the top examples look obviously wrong, improve interpretability before adding sophistication.

### 3. Replace team-level heuristics with ball-carrier-relative option logic
The current toy baseline is useful only as a draft. The next real modeling step should focus on **the ball-carrier's candidate options** rather than a generic best teammate score.

Concretely:
- identify the ball carrier at each frame,
- define candidate receivers,
- estimate option quality for each receiver,
- compare option quality before and after the off-ball movement,
- attribute a portion of the change to the runner.

This is the most important methodological step.

---

## Recommended development roadmap

### Phase 1. Stabilize data access
Goal: make dataset handling easy and reliable.

Tasks:
- verify Metrica loader outputs with a few manual frame inspections,
- add a minimal SkillCorner loader,
- add one helper for reading StatsBomb event and 360 data into a consistent format,
- document expected directory structure in README.

Deliverable:
- one clean loader path per dataset.

### Phase 2. Improve the baseline definition
Goal: move from a toy heuristic toward a meaningful research proxy.

Tasks:
- infer or approximate ball-carrier identity frame by frame,
- define candidate pass targets,
- estimate a simple open-option score or availability proxy,
- compute a counterfactual score where defender reaction to the runner is weakened,
- store per-sequence comparisons in a structured CSV.

Deliverable:
- an interpretable draft metric at the option level.

### Phase 3. Add qualitative validation
Goal: make sure the metric behaves plausibly.

Tasks:
- visualize top-ranked sequences,
- compare actual vs counterfactual defender geometry,
- manually inspect whether high-value cases correspond to decoy runs, gravity effects, or lane-opening movements,
- write down a short taxonomy of movement types observed in the output.

Deliverable:
- a small set of convincing examples.

### Phase 4. Expand beyond Metrica
Goal: move from a toy environment to a more realistic open-data setting.

Tasks:
- port the draft metric to SkillCorner broadcast tracking,
- use StatsBomb events or 360 freeze-frames for event-aligned validation,
- check whether conclusions hold across more than one match.

Deliverable:
- a stronger prototype on public data.

---

## Possible metric directions

### Direction A. Option-Creation Value
Measure how an off-ball player's movement improves a teammate's receiving option.

Potential ingredients:
- pass availability,
- local pitch control,
- simple downstream value proxy.

Why this is good:
- relatively interpretable,
- feasible with open data,
- tightly connected to the research question.

### Direction B. Counterfactual Off-Ball Value
Compare actual movement with a reference state where the runner moved less aggressively or where the defense did not shift as much.

Potential ingredients:
- actual state,
- reference trajectory or partial rollback,
- value difference attributed to the runner.

Why this is good:
- conceptually close to C-OBSO,
- directly aligned with dummy-run or decoy-run logic.

### Direction C. Possession-Value Uplift
Treat off-ball value as the amount by which the run increases downstream possession value, not just immediate openness.

Potential ingredients:
- option selection probability,
- option success probability,
- success-conditioned downstream value.

Why this is good:
- strongest long-term framing,
- closer to a publishable final formulation.

---

## Public-data strategy

### Metrica
Use for:
- rapid iteration,
- debugging loaders,
- quick counterfactual prototypes,
- initial visual sanity checks.

Not ideal for:
- broad claims,
- player ranking conclusions,
- final evaluation.

### SkillCorner open data
Use for:
- more realistic off-ball movement analysis,
- broadcast-tracking experiments,
- richer qualitative examples.

Not ideal for:
- huge sample inference,
- strong generalization claims.

### StatsBomb open data and 360
Use for:
- event-aligned context,
- freeze-frame option inspection,
- linking off-ball context to passes or shots.

Not ideal for:
- continuous movement analysis on its own.

---

## What should not be done yet
At the current stage, avoid:
- claiming a final player rating metric,
- over-optimizing the heuristic before validating examples,
- building a heavy deep-learning pipeline too early,
- pushing raw datasets to GitHub,
- presenting early outputs as paper-ready evidence.

---

## Suggested repository positioning
A good one-sentence description is:

> Draft research scaffold for quantifying off-ball attacking value in soccer using open tracking and event data.

A good current-scope note for README is:

> The current version includes open-data loaders, a Metrica-based toy counterfactual baseline, and literature notes toward a more complete option-value framework.

---

## Practical next commit ideas
If continuing development after the first GitHub push, a sensible next sequence is:

1. add `docs/literature_review.md`,
2. add `docs/next_steps.md`,
3. update README status section,
4. add one small visualization script for top Metrica examples,
5. start a minimal SkillCorner loader.

This keeps progress visible without overcommitting to a premature methodology.
