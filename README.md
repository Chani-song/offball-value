# offball-value-soccer

Starter repository for exploring **attacking off-ball value** in soccer using open datasets.

This repo is designed as a practical sandbox for the question:

> How much does a non-receiving attacker's off-ball movement improve the quality of teammates' attacking options?

The working goal is **not** to claim a final metric yet. The goal is to get to a clean, testable prototype quickly.

---

## Core research idea

We want to evaluate sequences like:

1. An attacker makes a run without receiving the ball.
2. That run changes defender positions, marking assignments, or line spacing.
3. Because of that change, a teammate's available options improve.
4. The improvement can be measured as a rise in one or more of:
   - pass availability / receiver openness
   - expected possession value proxy
   - zone value / field value
   - likelihood of line-breaking progression
   - eventual shot quality as a downstream validation target

A practical first draft is:

\[
\text{OffBallValue}(i, t) \approx \text{OptionScore}_{\text{actual}}(t+\Delta) - \text{OptionScore}_{\text{counterfactual no-run}}(t+\Delta)
\]

where player `i` is the off-ball attacker and the counterfactual is approximated with a simple baseline such as:

- freezing the runner near the start of the action,
- replacing the runner with a predicted "reference" trajectory,
- or removing the runner's displacement while keeping the rest of the frame fixed.

This repository currently implements a **very rough baseline** around this idea so you can start testing fast.

---

## What is in the repo

- `scripts/` download and setup helpers
- `src/offball_value/` draft Python package
- `scripts/run_metrica_baseline.py` first toy pipeline on Metrica sample data
- `docs/idea_sketch.md` more explicit formulation and next steps
- `docs/datasets.md` currently usable open datasets

---

## Open datasets to start with

### 1. Metrica Sports sample data
Good for the very first prototype.

- public GitHub repo
- synchronized event + tracking data
- lightweight and easy to inspect
- very small, so it is good for method prototyping, not strong generalization claims

Source:
- https://github.com/metrica-sports/sample-data

### 2. SkillCorner open data
Best current open option here for a more serious tracking-based prototype.

- public GitHub repo
- 10 matches of broadcast tracking data
- includes tracking, dynamic events, phases of play, and season-level physical data

Source:
- https://github.com/SkillCorner/opendata

### 3. StatsBomb open data + 360
Useful for event-level context even though it is not full continuous tracking.

- public GitHub repo
- event data for many competitions
- selected matches include 360 freeze-frame context
- useful for option quality around individual events

Source:
- https://github.com/statsbomb/open-data

### 4. Bundesliga integrated event + position dataset
Potentially very valuable, but not as frictionless as the GitHub repos above.

- official integrated event and position dataset
- seven matches from German Bundesliga 1 and 2
- excellent benchmark-style public resource

Sources:
- https://www.nature.com/articles/s41597-025-04505-y
- https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177

---

## Quick start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download one or more datasets

```bash
python scripts/download_metrica.py
python scripts/download_skillcorner.py
python scripts/download_statsbomb.py
```

For the Bundesliga dataset, see `docs/datasets.md`.

### 3. Run the first toy pipeline on Metrica

```bash
python scripts/run_metrica_baseline.py
```

Expected output:

- a small CSV of candidate off-ball runs
- a rough score based on changes in receiver openness and field progress

---

## Current baseline metric in plain language

The first implementation is intentionally simple.

For each attacking event in a short window:

1. detect nearby attacking teammates who are moving meaningfully,
2. estimate whether their motion increases a teammate's notional option quality,
3. compare the actual frame with a naive counterfactual where the runner is held near the initial position,
4. assign the difference as a **draft off-ball value**.

This is **not** the final research metric. It is just the scaffold.

---

## Immediate next steps

1. Build a cleaner event-window extractor for each dataset.
2. Replace the current toy option score with a better model:
   - pass availability
   - pitch control proxy
   - xT / EPV-style target
3. Add a better counterfactual trajectory baseline.
4. Separate move types:
   - decoy run
   - support run
   - overlap / underlap
   - pinning run
   - box occupation run
5. Add simple visualizations of runner and defenders.

---

## Suggested repo workflow

```bash
git init
git add .
git commit -m "Initial off-ball value prototype scaffold"
```

Then create a remote repo and push as usual.

---

## Important caveat

This repo is a **research bootstrap**, not a polished package. The code is intentionally draft-level so you can iterate quickly.
