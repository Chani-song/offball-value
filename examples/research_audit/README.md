# Research audit artifacts

This directory preserves the non-regenerable human reviews, the confirmed
core-scene manifest, and the three current review demos before legacy generated
outputs are removed from `data/processed/`.

## Contents

- `human_reviews/endpoint_action_space/`: human reviews that informed the
  successive feasible endpoint action-space revisions.
- `human_reviews/scene_extraction/`: early scene-extraction reviews.
- `human_reviews/run_onset/`: run-onset review rounds and the compiled v0.3 QC
  subsets.
- `human_reviews/shot_context/`: shot-window and shot-context-onset reviews.
- `human_reviews/clear_core/`: final human review used to select the current
  core scenes.
- `manifests/`: confirmed core-scene metadata and the self-contained confirmed
  scene payload.
- `evidence/background_rollout/`: compact held-out evidence used to reject
  constant velocity as the only two-second background baseline.
- `current_demos/`: the eight-scene confirmed clear-core audit, meeting
  gallery, in-progress structural local-game audit, the curated meeting
  payload, and the compact structural summary. The two held clear-core scenes
  are not included in these shared demos.

The HTML files are self-contained. Download the file or clone the repository
and open it in a web browser; GitHub's source view does not execute the page.

These are research audit artifacts, not final model outputs. In particular,
the structural local-game audit is shared as an in-progress working prototype.
It is a geometric diagnostic and does not yet represent a completed local-game
model or a validated final threat function.

The derived tracking excerpts originate from IDSSE (Bassek et al., 2025), are
shared under CC BY 4.0, and retain the dataset attribution documented in
`docs/data_and_reproduction.md`.
