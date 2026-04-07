# Off-Ball Value Project Status Update

_Last updated: 2026-04-07_

## 1. Project snapshot

The project has moved past the pure idea stage and into a usable prototype stage.

What is already in place:
- a repository scaffold for the project
- open-data download scripts for Metrica, SkillCorner, and StatsBomb
- a common package layout under `src/offball_value`
- a common schema layer for frame-level and event-level data
- dataset adapters that convert raw source formats into project-level objects
- a working Metrica smoke test
- literature review and next-step planning documents

What this means in practice:
- we can now load at least one real tracking dataset into a common representation
- we have enough structure to build a first end-to-end demo for off-ball counterfactual evaluation
- the project is still in an early research-prototype state, not yet at a validated metric or paper-ready experiment stage

## 2. Main question of the project

The current research direction is:

> Can we quantify the value of an attacking player's off-ball movement by comparing the actual scene to a counterfactual scene in which that runner did not create the same defensive distortion?

The current working idea is to start with a simple counterfactual proxy:
- detect candidate attacking runs over a short frame window
- compute a simple option-quality score for teammates in the actual frame
- perturb the defender most likely affected by the run
- recompute the option-quality score in the counterfactual frame
- define a draft off-ball value as the change between actual and counterfactual option value

This is currently a **toy proxy**, not the final research metric.

## 3. Current repository structure

### Root-level files
- `README.md`: project overview and initial positioning
- `requirements.txt`: environment requirements
- `bootstrap.sh`: environment setup and data download helper
- `pyproject.toml`: package metadata for editable install
- `.gitignore`: local and generated files to ignore

### Documentation
- `docs/idea_sketch.md`: high-level project idea
- `docs/datasets.md`: dataset notes
- `docs/literature_review.md`: related work summary
- `docs/next_steps.md`: next-step planning

### Source package
- `src/offball_value/__init__.py`
- `src/offball_value/loaders.py`
- `src/offball_value/schema.py`
- `src/offball_value/adapters.py`
- `src/offball_value/baseline.py`

### Scripts
- `scripts/download_metrica.py`
- `scripts/download_skillcorner.py`
- `scripts/download_statsbomb.py`
- `scripts/run_metrica_baseline.py`
- `scripts/inspect_dataset_formats.py`
- `scripts/demo_offball_counterfactual.py` (planned / next to run)

## 4. What has been implemented so far

### 4.1 Environment and packaging
Completed:
- repository initialized and committed
- Python environment created with the required dependencies
- editable install enabled through `pyproject.toml`
- `import offball_value` now works without needing temporary `PYTHONPATH=src` workarounds

Why this matters:
- the repo is now usable as a real Python package
- scripts and tests can import the code in a stable way

### 4.2 Data acquisition
Completed:
- Metrica sample data downloaded successfully
- SkillCorner open data downloaded successfully after installing Git LFS
- StatsBomb open data downloaded successfully

Important note:
- raw datasets are kept locally under `data/raw/`
- raw data is **not** intended to be committed to GitHub

### 4.3 Data normalization
Completed:
- a common schema was introduced so different sources can be handled through similar interfaces
- adapters were added to convert raw source files into project-level objects

This is a major milestone because the project is no longer blocked by raw source formats.

## 5. Common schema now available

The project now uses two main normalized object types.

### 5.1 `FrameSnapshot`
Represents one tracking frame.

Fields:
- `source`: dataset source name
- `match_id`: source-specific match id
- `period`: match period if available
- `frame_id`: frame number if available
- `time_s`: frame timestamp in seconds if available
- `ball`: normalized ball object or `None`
- `players`: list of normalized player objects
- `possession_team`: team in possession if available
- `raw`: original source payload for traceability

### 5.2 `EventRecord`
Represents one event-level record.

Fields:
- `source`
- `match_id`
- `event_id`
- `period`
- `time_s`
- `event_type`
- `player_id`
- `team`
- `x_start`, `y_start`
- `x_end`, `y_end`
- `frame_start`, `frame_end`
- `raw`

### 5.3 `PlayerSnapshot` and `BallSnapshot`
These are normalized helper objects used inside `FrameSnapshot`.

## 6. Function-level status summary

## `src/offball_value/loaders.py`
Purpose:
- low-level loading of raw source files

Implemented responsibilities:
- locate Metrica sample game directories
- load Metrica events
- parse raw Metrica tracking CSV files into structured pandas tables
- handle the multi-row Metrica tracking header format
- expose raw StatsBomb JSON loading helpers

Key result:
- Metrica raw tracking CSVs are now parsed into usable tables instead of remaining as messy multi-header raw files

## `src/offball_value/schema.py`
Purpose:
- define project-level normalized data structures

Implemented responsibilities:
- define `PlayerSnapshot`, `BallSnapshot`, `FrameSnapshot`, `EventRecord`
- define conversion helpers from raw source-specific rows into normalized objects
- robustly parse numeric values and time strings
- fix SkillCorner event time parsing such as `00:00.2`

Key result:
- the project now has a common language for handling heterogeneous datasets

## `src/offball_value/adapters.py`
Purpose:
- bridge raw loaders and normalized schema

Implemented responsibilities:
- load merged Metrica frame snapshots
- load SkillCorner match list and match metadata
- load SkillCorner tracking frames as `FrameSnapshot`
- load SkillCorner dynamic events as `EventRecord`
- load SkillCorner phases of play
- load StatsBomb competitions, matches, lineups, events, and 360 freeze-frame data

Key result:
- source-specific data access is now separated from downstream analysis logic

## `src/offball_value/baseline.py`
Purpose:
- hold the current toy-value logic for early experimentation

Implemented responsibilities:
- define a simple player-state abstraction used in toy demos
- define `detect_candidate_run(...)`
- define `compute_toy_option_score(...)`

Interpretation:
- this file is intentionally simple and is currently used for sanity-check experiments, not final conclusions

## `scripts/inspect_dataset_formats.py`
Purpose:
- inspect raw datasets and print the effective local format

Why it exists:
- before building real experiments, we needed to verify what the downloaded files actually contain
- this script documents the observed local structure in executable form

## `scripts/run_metrica_baseline.py`
Purpose:
- run a first baseline using Metrica data only

What it currently does:
- loads Metrica Home and Away tracking
- merges them into a single frame stream
- detects candidate movement windows
- computes a simple option score before/after counterfactual defender repositioning
- writes a CSV of draft off-ball values

Interpretation:
- this is the earliest working end-to-end baseline
- it is useful mainly for sanity checks and debugging, not for claims

## `scripts/demo_offball_counterfactual.py`
Purpose:
- next experiment script to formalize the first demo more cleanly

Expected role:
- consume `FrameSnapshot`s directly
- produce top-k candidate off-ball scenes
- save a more interpretable counterfactual demo output

## 7. Data currently available and usable

## 7.1 Metrica sample data
Currently usable for:
- frame-level tracking experiments
- simple event alignment
- first toy counterfactual demo

Observed files for Game 1:
- `Sample_Game_1_RawEventsData.csv`
- `Sample_Game_1_RawTrackingData_Away_Team.csv`
- `Sample_Game_1_RawTrackingData_Home_Team.csv`

Observed tracking structure:
- row 1: team header blocks
- row 2: jersey numbers
- row 3: main header row
- data rows contain:
  - `Period`
  - `Frame`
  - `Time [s]`
  - player x/y columns
  - ball x/y columns

Observed example header pattern:
- `Period, Frame, Time [s], Player11, ..., Ball`

How it is used in this project:
- parsed into a unified frame table
- converted into `FrameSnapshot`
- currently the main development dataset for the first demo

Strengths:
- clean enough for rapid prototyping
- fully continuous tracking
- easy to debug

Current limitation:
- very small dataset compared with real club-scale work

## 7.2 SkillCorner open data
Currently usable for:
- richer frame-level tracking
- possession-aware and event-aware analysis
- later validation with off-ball-run-related metadata

Observed top-level structure:
- `matches.json`
- per-match directory under `data/matches/<match_id>/`

Observed per-match files:
- `{match_id}_match.json`
- `{match_id}_tracking_extrapolated.jsonl`
- `{match_id}_dynamic_events.csv`
- `{match_id}_phases_of_play.csv`

Observed tracking frame keys:
- `frame`
- `timestamp`
- `period`
- `ball_data`
- `possession`
- `image_corners_projection`
- `player_data`

Observed `ball_data` keys:
- `x`
- `y`
- `z`
- `is_detected`

Observed event-side usefulness from `dynamic_events.csv`:
- timing fields
- player and team ids
- start/end coordinates
- many precomputed context fields
- off-ball-run associations
- passing-option-related fields
- xThreat-like and pass-quality-like annotations
- line-break information
- shot/goal downstream indicators

Important interpretation:
- SkillCorner is powerful and likely the best public dataset here for later-stage demos
- however, many fields are already highly engineered, so they must be used carefully to avoid leakage

## 7.3 StatsBomb open data
Currently usable for:
- event-level analysis
- freeze-frame context checks
- later event-conditioned validation

Observed top-level structure under `data/`:
- `competitions.json`
- `events/`
- `lineups/`
- `matches/`
- `three-sixty/`

Observed event structure from a sample event file:
- `id`
- `index`
- `period`
- `timestamp`
- `minute`
- `second`
- `type`
- `possession`
- `possession_team`
- `play_pattern`
- `team`
- `duration`
- `tactics`

Observed 360 structure:
- `event_uuid`
- `visible_area`
- `freeze_frame`

Interpretation:
- StatsBomb is not the main dataset for continuous movement-based off-ball valuation
- it is still useful for event-centered sanity checks and later support analysis

## 8. Smoke runs completed so far

## 8.1 Repository / package smoke run
Completed:
- editable package install succeeded
- `import offball_value` works

Impact:
- the codebase is now importable as a proper package

## 8.2 Metrica frame smoke run
Completed result:
- `load_metrica_frames(game=1)` returned `145006` frames
- first frame id was `1`
- first frame contained `22` players

Interpretation:
- Metrica tracking is now being parsed successfully into normalized frame objects
- this is the clearest end-to-end confirmation that the data pipeline is alive

## 8.3 Dataset format inspection smoke run
Completed result:
- Metrica Game 1 file structure was printed and inspected
- SkillCorner match structure, tracking keys, dynamic event header, and phases-of-play header were printed and inspected
- StatsBomb top-level structure, competition count, sample event keys, and sample 360 keys were printed and inspected

Interpretation:
- local dataset formats are now confirmed through code, not just assumptions

## 8.4 SkillCorner parser fixes
Issues encountered and resolved:
- Git LFS was required for checkout
- SkillCorner event times were stored as strings like `00:00.2`, so direct `float(...)` parsing failed
- schema helpers were updated to robustly parse these time formats

Interpretation:
- SkillCorner is now much closer to being usable in downstream demos

## 9. What is already trustworthy vs what is still draft

## Already trustworthy
- repository structure
- local environment setup
- raw data download process
- Metrica tracking parsing
- common schema layer
- adapter layer design
- basic dataset-format inspection

## Still draft / provisional
- candidate-run definition
- toy option score
- counterfactual defender perturbation logic
- attacking direction handling
- any claim about true player value
- any cross-dataset comparison result

This distinction is important. The codebase is operational, but the research metric is still early.

## 10. Recommended immediate next steps

### Priority 1: run the first clean demo script
Implement and run:
- `scripts/demo_offball_counterfactual.py`

Goal:
- produce a ranked CSV of candidate off-ball scenes using Metrica

### Priority 2: qualitative inspection of top scenes
After the demo CSV is generated:
- inspect the top 10 to 20 scenes manually
- check whether they look like meaningful off-ball runs
- verify whether the counterfactual delta matches visual intuition

### Priority 3: make the toy demo more realistic
Current toy simplifications to improve next:
- use actual attacking direction by period
- condition the score on a ball-carrier rather than the global best teammate option
- improve the counterfactual defender-assignment logic

### Priority 4: extend the same logic to SkillCorner
After Metrica sanity checks:
- run the same type of demo on a small SkillCorner subset
- compare whether scenes align with SkillCorner's richer contextual metadata

### Priority 5: avoid leakage when using SkillCorner engineered fields
If SkillCorner annotations are used later:
- separate fields used for modeling from fields used only for validation
- do not accidentally use labels or label-like annotations as inputs to the main metric

## 11. Current working interpretation of the roadmap

A practical roadmap is now visible:

1. **Normalize data**  
   Done at a first-pass level.

2. **Verify format assumptions**  
   Done through inspection scripts and smoke runs.

3. **Run first counterfactual toy demo on Metrica**  
   Immediate next milestone.

4. **Inspect whether top-ranked scenes make football sense**  
   First qualitative validation.

5. **Refine the off-ball value definition**  
   Shift from toy option proxy to a more ball-carrier-conditioned and attack-aware formulation.

6. **Move to richer open data such as SkillCorner**  
   Later-stage public-data validation.

## 12. Bottom line

The project is no longer blocked by setup, packaging, or source-format ambiguity.

The most important achievement so far is this:

> We can now load real public soccer data into a common internal representation and are ready to run the first genuine off-ball counterfactual demo.

The next meaningful step is **not** more repository plumbing. It is to produce and inspect the first demo results.
