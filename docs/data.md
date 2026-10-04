# Data

## Tracking and event data

All scenes come from IDSSE (Bassek et al., 2025): seven full 2022/23 Bundesliga and 2. Bundesliga
matches with synchronized position data (25 Hz, all players and the ball), event data and match
information. The data owner is Deutsche Fußball Liga (DFL). The figshare record is licensed
CC BY 4.0 (checked 2026-10-01).

- Dataset: <https://doi.org/10.6084/m9.figshare.28196177>
- Paper: <https://doi.org/10.1038/s41597-025-04505-y>

Download the files and place them, unchanged, in `data/raw/bundesliga-integrated/`. The loaders
(`src/offball_value/bundesliga.py`) find them by match id.

CC BY 4.0 permits redistribution with attribution. We still keep raw and tracking-derived positions
out of this branch: the authors have not decided to publish them, and the team's data bundles are
marked not for Git. The public demo is a separate case (below).

## What is in the repository

| Path | Content | Source | Terms |
| --- | --- | --- | --- |
| `data/processed/pass_models_sym/Asym_*.json` | A-sym pass model: three logistic coefficients, ball-speed fit, player constants; one fit on all six pass matches and one per held-out match, plus the router the games use | `scripts/fit_pass_sym.py` on IDSSE passes | coefficients only, no positions |
| `data/processed/pass_models/A_*.json` | pass model A (the candidate A-sym replaced), same layout | `scripts/fit_pass_candidates.py` | as above |
| `data/processed/physics_limits/agile_p999_nodelay.json` | 99.9th-percentile speed-up, braking and turning limits | `scripts/measure_accelerations.py` on IDSSE | numbers only |
| `data/static/EPV_grid.csv` | 32 x 50 static EPV grid | copied from [PAUSA](https://github.com/leemingo/mitssac-pausa) | Apache-2.0 ([data/static/README.md](../data/static/README.md)) |
| `docs/assets/figure1_ssac.pdf`, `figure2_ssac.pdf` (and `.png` page renders) | the paper's Figures 1 and 2 as submitted, with captions | the submitted figure PDFs | images of scene S05; no coordinates |
| `docs/assets/demo_screenshot.png` | the demo's Nash equilibrium view for scene S05 (2026-10-01) | the demo at commit `ae3f38b` | image only |

`tests/test_public_data.py` checks these files, including that every held-out model was fitted
without its match.

## What is not in the repository

| Material | Where it comes from | Why it is absent | Needed for |
| --- | --- | --- | --- |
| IDSSE raw files | figshare (above) | large; obtainable from the source | every pipeline step that reads tracking |
| Scene tracking excerpts (`data/processed/showcase_v1/tracking/*.csv`), game states, solver outputs (`out/runs/`), panels, defender grids | the pipeline and the solver jobs | tracking-derived positions; not published by the authors | solver checks, Figures 1 and 2, the evaluation |
| Evaluation output (`players.csv`, `moments.json`, `summary.json` from `scripts/analyze_eval.py`) | the 2026-09-30 evaluation run | kept with the team's data bundle; these files hold no positions and could be added if the authors decide to | checking the numbers in the README |
| Static comparison output (`static_full/`) | `jobs/static_counterfactual.sbatch` | not available outside the cluster | the static versus equilibrium comparison |
| Shot-clip annotations, raters' scores, the showcase start sheet (`solver_starts_*.csv`) | team members | team data | scene selection steps 3 and 5 in [reproduction.md](reproduction.md), Figure 1 |
| Solver base `defensive_positioning` | `andrewkang12345/mit_ssac2027` @ `e8b0a95` | private repository, no license | every solver run and check |
| Positions-only pass model `andrew/models/experimental_pass.json` | the solver base | part of the solver base; fitted on StatsBomb 360 data | several solver check scripts, `scripts/fit_pass_candidates.py` |

## Coordinates

Solver states, panels and grids use pitch-corner coordinates (0 to 105 m by 0 to 68 m) with the
match's own attack direction. The scene tracking CSVs and the figures use centre-spot coordinates with
the attack to the right. With `attack_direction` = +1 or -1 from the panel file:

```text
x_csv = attack_direction * (x_panel - 52.5)
y_csv = attack_direction * (y_panel - 34.0)
```

## Public demo data

The interactive explorer (<https://chani-song.github.io/offball-value/>) is built from `demo_viz/` on
other branches. The `main` branch carries its exported scene files (`demo_viz/web_data/`, 50 scenes,
about 3.8 MB), which hold every player's and the ball's position at 25 Hz over an 11 s clip per scene,
and the site serves them publicly. None of these files are on this branch.

## Citing the data

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
