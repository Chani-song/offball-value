# Data and reproduction

## Environment

The project requires Python 3.11.

```bash
./bootstrap.sh
```

This creates `.venv/` and installs the package in editable mode. The bootstrap
script does not download soccer data.

## Primary dataset

The current pipeline uses IDSSE: seven full matches from the 2022/23 Bundesliga
and 2. Bundesliga with synchronized tracking, events, and metadata. Tracking is
available at 25 Hz for all players and the ball.

Place the extracted dataset at:

```text
data/raw/bundesliga-integrated/
```

The loader discovers match files below this directory. Raw data and regenerated
outputs are ignored by Git.

### License and citation

IDSSE is distributed under CC BY 4.0.

- Dataset DOI: <https://doi.org/10.6084/m9.figshare.28196177>
- Paper DOI: <https://doi.org/10.1038/s41597-025-04505-y>
- Data owner: Deutsche Fußball Liga (DFL)

```bibtex
@article{bassek2025idsse,
  title   = {An integrated dataset of synchronized spatiotemporal and event data in elite soccer},
  author  = {Bassek, M. and others},
  journal = {Scientific Data},
  volume  = {12},
  number  = {1},
  pages   = {195},
  year    = {2025},
  doi     = {10.1038/s41597-025-04505-y}
}
```

The curated scene payloads and HTML demos contain small derived tracking
excerpts and must retain this attribution when redistributed.

## Fast reproduction without raw match files

The confirmed scene payload in `examples/` is sufficient to rebuild the
current structural audit:

```bash
.venv/bin/python scripts/render_structural_local_game_v0_1.py \
  --scenes-json examples/research_audit/manifests/confirmed_core_scenes.json \
  --output-dir data/processed/structural_local_game_v0_1
```

The meeting gallery can be rendered from its preserved payload:

```bash
.venv/bin/python scripts/render_meeting_scene_gallery_v0_1.py \
  --payload-json examples/research_audit/current_demos/meeting_scene_gallery_payload.json \
  --output-dir data/processed/meeting_scene_gallery_v0_1
```

All HTML outputs are self-contained. Open them locally in a browser; GitHub's
file view shows the HTML source.

## Full scene-extraction pipeline

### 1. Extract shot contexts

```bash
.venv/bin/python scripts/extract_shot_contexts.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id all \
  --output-dir data/processed/shot_context_v0_1
```

Main outputs are `shot_anchors.csv`, `attacking_phases.csv`,
`match_summary.csv`, and a self-contained audit HTML.

### 2. Detect run onsets inside reviewed shot contexts

```bash
.venv/bin/python scripts/extract_shot_context_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --phase-csv data/processed/shot_context_v0_1/attacking_phases.csv \
  --phase-review-csv examples/research_audit/human_reviews/shot_context/shot_context_v0_1_reviews.csv \
  --match-id all \
  --output-dir data/processed/shot_context_run_onset_v0_1
```

The script produces the candidate table and the review audit used by the next
gate.

### 3. Build the ten clear-core candidates

```bash
.venv/bin/python scripts/render_clear_core_scene_audit_v0_1.py \
  --audit-json data/processed/shot_context_run_onset_v0_1/shot_context_onset_audit.json \
  --review-csv examples/research_audit/human_reviews/shot_context/shot_context_onset_v0_1_reviews.csv \
  --output-dir data/processed/clear_core_scene_audit_v0_1
```

### 4. Attach the second-pass human decisions

```bash
.venv/bin/python scripts/compile_clear_core_scene_reviews.py \
  --scenes-json data/processed/clear_core_scene_audit_v0_1/clear_core_scenes.json \
  --reviews-csv examples/research_audit/human_reviews/clear_core/clear_core_scene_reviews_human.csv \
  --output-dir data/processed/clear_core_scene_audit_v0_1
```

This creates `confirmed_core_scenes.json`. The curated copy under `examples/`
contains eight confirmed scenes and is the fixed v0.1 development input.

### 5. Build the structural audit

```bash
.venv/bin/python scripts/render_structural_local_game_v0_1.py \
  --scenes-json data/processed/clear_core_scene_audit_v0_1/confirmed_core_scenes.json
```

## Movement calibration

Movement-library and steering calibration are separate, longer-running steps.
The target match must be excluded when constructing a calibration library for
that match.

```bash
.venv/bin/python scripts/build_empirical_movement_library.py \
  --data-dir data/raw/bundesliga-integrated \
  --target-match-id DFL-MAT-J03WMX

.venv/bin/python scripts/calibrate_steering_reachability.py
.venv/bin/python scripts/calibrate_plant_cut_reachability.py
```

The current structural response uses documented unified development limits;
the empirical library is retained for action-space calibration and sensitivity
work.

## Tests

Run every test with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

For the current handoff core only:

```bash
.venv/bin/python -m unittest -v \
  tests.test_shot_context \
  tests.test_shot_context_onset \
  tests.test_run_onset \
  tests.test_clear_core_scene_audit \
  tests.test_empirical_action_space \
  tests.test_fernandez_influence \
  tests.test_steering_reachable \
  tests.test_dynamic_marking \
  tests.test_dynamic_response_game \
  tests.test_local_game_structure \
  tests.test_meeting_scene_gallery
```

## Reproducibility limits

- Human reviews are development annotations from one reviewer, not ground
  truth labels.
- The eight scenes are intentionally high precision and are not a random sample.
- Some early calibration outputs are cached under ignored `data/processed/`.
- The current structural audit is deterministic given the curated scene JSON
  and code, but it is not the final value experiment.
