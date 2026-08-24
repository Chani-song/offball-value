# Local Off-Ball Game in Soccer

Research prototype for studying how an off-ball run reallocates a defender
between the runner's direct threat and the attacking opportunities left to
teammates.

## Research question

> Can we identify off-ball movements that retain attacking threat even after a
> physically feasible defensive response?

The focal intervention is the **off-ball runner**. The full 11-v-11 state is
kept as context, but counterfactual control is restricted to a small local
game: one runner action, one candidate defender response at a time, and the
attacking options whose control changes with that defender's allocation. A
fixed geometric 2-v-1 is neither required nor assumed.

## Current workflow

```text
Bundesliga tracking + events
  -> shot-context sampling
  -> controlled-possession gate
  -> retrospective run-onset detection
  -> human-confirmed development scenes
  -> bounded movement and dynamic goal-side response
  -> defender-by-attacking-option structural audit
  -> future: calibrated threat, defender best response, attacker max-min
```

Shots are used only to retrieve attack-like match windows. The project does
not assume that an earlier run caused the later shot, and shot outcomes are not
optimizer inputs.

## What is implemented

| Component | Status |
| --- | --- |
| IDSSE Bundesliga loader for seven full matches | Implemented |
| Open-play shot contexts and controlled-possession filtering | Implemented |
| Acceleration, direction-change, and check-run onset detector | Implemented and human-audited |
| Eight human-confirmed local-game development scenes | Available |
| Bounded steering and plant-and-cut action-space prototypes | Implemented and human-audited |
| Dynamic goal-side defender response | Implemented as a geometry prototype |
| Three candidate defenders by five local attacking options | Implemented as a structural audit |
| Delivery x goal danger x goal-side accessibility | Provisional components only |
| Validated defender best response and attacker max-min value | Not yet implemented |
| Seven-match statistical evaluation | Not yet run |

The current structural matrix uses **observed attacker futures** for
retrospective development-set auditing. Its cells are goal-side marking
allocation effects, not a calibrated threat value and not an online trajectory
prediction result.

## Review artifacts

The curated audit package is in
[`examples/research_audit/`](examples/research_audit/README.md). It contains
non-regenerable human reviews, the confirmed scene manifest, and three
self-contained demos:

- [`clear_core_scene_audit.html`](examples/research_audit/current_demos/clear_core_scene_audit.html): eight human-confirmed development scenes shown with observed motion only; the two held scenes are not displayed.
- [`meeting_scene_gallery.html`](examples/research_audit/current_demos/meeting_scene_gallery.html): eight unique confirmed scenes; the Klaus scene also retains one clearly labeled legacy dynamic view.
- [`structural_local_game_audit.html`](examples/research_audit/current_demos/structural_local_game_audit.html): defender-by-option geometry screen over the eight confirmed scenes.

GitHub displays HTML source rather than executing these pages. Clone or
download the repository and open the files in a local web browser.

## Quick start

Python 3.11 is required.

```bash
./bootstrap.sh
```

Download the IDSSE data separately and place it at:

```text
data/raw/bundesliga-integrated/
```

The curated structural audit can be regenerated without loading raw match
files:

```bash
.venv/bin/python scripts/render_structural_local_game_v0_1.py \
  --scenes-json examples/research_audit/manifests/confirmed_core_scenes.json
```

The confirmed observed-motion audit can be regenerated from the same payload:

```bash
.venv/bin/python scripts/render_clear_core_scene_audit_v0_1.py \
  --confirmed-scenes-json examples/research_audit/manifests/confirmed_core_scenes.json
```

The preserved meeting gallery can also be rendered directly from its curated
payload:

```bash
.venv/bin/python scripts/render_meeting_scene_gallery_v0_1.py
```

Run the test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

See [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md) for the
full extraction and review pipeline.

## Documentation

- [`docs/current_research_overview.md`](docs/current_research_overview.md): current research question, game formulation, contributions, and open decisions.
- [`docs/methodology_v0_1.md`](docs/methodology_v0_1.md): frozen definitions for the current scene and structural audits.
- [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md): data setup, commands, outputs, and citation.
- [`docs/review_guide.md`](docs/review_guide.md): how to inspect the three HTML demos.
- [`docs/research_history.md`](docs/research_history.md): why the project moved away from the early OBSO-style proxy.
- [`docs/literature_review.md`](docs/literature_review.md): related-work map.
- [`docs/background_rollout_benchmark_v0_1.md`](docs/background_rollout_benchmark_v0_1.md): held-out comparison of simple causal background rollouts.

## Repository map

```text
src/offball_value/    reusable loaders, detection, motion, and local-game code
scripts/              extraction, calibration, evaluation, and rendering CLIs
tests/                unit tests
examples/             curated human reviews, manifests, and self-contained demos
docs/                 current research and reproduction documentation
data/static/          small static model inputs tracked by Git
data/raw/             local datasets; ignored by Git
data/processed/       regenerated outputs; ignored by Git
```

## Data and claim boundary

The primary dataset is IDSSE: seven synchronized Bundesliga and 2. Bundesliga
matches sampled at 25 Hz. IDSSE is distributed under CC BY 4.0; attribution
details are in [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md).
Raw data are never committed to this repository.

This repository is a research prototype. It must not yet be interpreted as a
validated off-ball value metric, player ranking, causal estimate, or coaching
recommendation system.
