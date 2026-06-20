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

## Possession and state convention

For the first modeling pass, an attacking state means the team has **controlled
possession at a player's feet**. In code, this is approximated by a ball carrier
whose tracked position is close enough to the ball.

Pass-flight frames are not treated as separate possession states. Instead, a
completed pass is represented as a transition:

```text
A controlled-possession state -> pass attempt -> B controlled-possession state
```

This means "B clearly receives the pass" is decided after the next controlled
touch is observed. That receiver/outcome label can be used to build transitions
or validation labels, but it should not be included as a feature in the state at
pass time.

The first state-value target should stay simple: start with an action/state
value proxy from prior possession-value work, then replace the toy score only
after the possession and option definitions are stable.

---

## What is in the repo

- `scripts/` download and setup helpers
- `src/offball_value/` draft Python package
- `scripts/run_metrica_baseline.py` first toy pipeline on Metrica sample data
- `docs/idea_sketch.md` more explicit formulation and next steps
- `docs/datasets.md` currently usable open datasets

---

## Current implementation status

The repository has now moved beyond the initial scaffold stage and includes a working internal data pipeline for multiple open soccer datasets.

### Currently implemented

- editable Python package setup via `pyproject.toml`
- reusable internal schema for frame-level and event-level data
- dataset format inspection utilities
- source adapters for:
  - Metrica sample data
  - SkillCorner open data
  - StatsBomb open data
- possession helpers for foot-control ball-carrier inference and completed pass transitions
- working Metrica frame-loading pipeline
- first-draft baseline and counterfactual demo scaffolding

### Internal representations

The current internal schema is centered around the following objects:

- `PlayerSnapshot`
  - one player at one time step
  - fields include player id, team, x/y position, optional z, optional role, and raw metadata

- `BallSnapshot`
  - ball state at one time step
  - fields include x/y position, optional z, optional detection flag, and raw metadata

- `FrameSnapshot`
  - unified frame-level representation across tracking providers
  - fields include source, match id, period, frame id, time in seconds, ball snapshot, player snapshots, optional possession team, and optional possession player

- `EventRecord`
  - unified event-level representation across event providers
  - fields include source, match id, event id, period, time in seconds, event type, player id, team, start/end location, optional frame start/end, optional receiver, optional outcome, and raw metadata

### Current source adapters

The following adapter layer is now available in `src/offball_value/adapters.py`.

#### Metrica
- `load_metrica_frames(...)`
- `iter_metrica_frames(...)`

These functions convert raw Metrica tracking CSV files into `FrameSnapshot` objects.

#### SkillCorner
- `load_skillcorner_matches(...)`
- `load_skillcorner_frames(...)`
- `iter_skillcorner_frames(...)`
- `load_skillcorner_events(...)`
- `iter_skillcorner_events(...)`
- `load_skillcorner_phases_of_play(...)`
- `load_skillcorner_match_metadata(...)`

These functions expose both frame-level and event-level access.

#### StatsBomb
- `load_statsbomb_competitions(...)`
- `load_statsbomb_matches(...)`
- `load_statsbomb_events(...)`
- `iter_statsbomb_events(...)`
- `load_statsbomb_lineups(...)`
- `load_statsbomb_freeze_frames(...)`

These functions support event-level analysis and 360 freeze-frame access.

---

## Data currently usable in practice

### Metrica sample data

Currently usable for:
- first-pass off-ball counterfactual demos
- frame-level motion sanity checks
- quick method prototyping

Main fields currently exposed:
- `period`
- `frame_id`
- `time_s`
- player positions `(x, y)`
- ball position `(x, y)`

### SkillCorner open data

Currently usable for:
- frame-level tracking access
- dynamic event parsing
- phase-of-play access
- future off-ball run and passing-option experiments

Main tracking fields confirmed:
- `frame`
- `timestamp`
- `period`
- `ball_data`
- `possession`
- `player_data`

Main event fields confirmed:
- `event_id`
- `event_type`
- `player_id`
- `team_id`
- `frame_start`
- `frame_end`
- `time_start`
- `x_start`, `y_start`
- `x_end`, `y_end`

### StatsBomb open data

Currently usable for:
- event-sequence analysis
- contextual event inspection
- 360 freeze-frame support for local option structure checks

Main event fields currently used:
- `id`
- `period`
- `timestamp`
- `type`
- `player`
- `team`
- `location`
- pass/carry/shot end locations when available

Main 360 fields:
- `event_uuid`
- `visible_area`
- `freeze_frame`

---

## Smoke-run results

The following smoke checks have already been completed.

### Metrica
- raw tracking files were successfully parsed into the internal frame schema
- home and away data were merged into a unified frame representation
- `load_metrica_frames(game=1)` returned `145006` frame snapshots
- the first frame contained `22` player snapshots

This confirms that the Metrica pipeline is already usable for the first demo experiments.

### SkillCorner
- open-data repository was downloaded successfully
- match metadata, tracking JSONL, dynamic events CSV, and phases-of-play CSV were inspected
- frame-level and event-level adapters were connected to the common schema
- timestamp parsing and schema conversion were debugged and fixed

This confirms that the SkillCorner pipeline is available for follow-up experiments, although it still needs more qualitative checking than the Metrica path.

### StatsBomb
- competitions, events, lineups, and 360 data were inspected successfully
- event adapters are available
- 360 freeze-frame loading is available

This confirms that StatsBomb is ready for event-level contextual validation, although it is not the main source for continuous off-ball movement modeling.

---

## Current interpretation of the codebase

At this stage, the repository should be understood as:

- a working research prototype
- a unified open-data ingestion layer
- a draft experimental sandbox for off-ball value ideas

It should **not** yet be interpreted as:
- a final off-ball metric
- a validated benchmark result
- a player-ranking system

---

## Immediate next engineering steps

1. add a dedicated `demo_offball_counterfactual.py` script
2. inspect top-ranked candidate runs qualitatively
3. refine the counterfactual definition
4. move from team-level best-option proxy to ball-carrier-conditioned option value
5. extend the first reproducible demo from Metrica to SkillCorner
6. use StatsBomb 360 mainly for contextual validation rather than primary modeling


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
