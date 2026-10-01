# Off-Ball Runs as a Local Game

Code for our abstract on off-ball runs in soccer. At a real moment of an
off-ball run, the runner, the ball carrier and the defender responsible for the
runner play a simultaneous-move game over the next 1.8 s: three decisions of
0.6 s each, every player choosing among physically limited movement commands,
and the attack also choosing among passes along the run. Every other player
follows his real tracked path; the other defenders can still tackle the ball
carrier. Each game is solved for its Nash equilibrium (a linear program at every
decision state), and every real moment is solved as its own game, so the
equilibrium can be compared with what the players actually did next.

Probabilities reported by the code and drawn in the figures are equilibrium
probabilities of choosing an action, not pass-success probabilities.

## Repository

```text
andrew-passer2on1/    2v1 game: ball carrier and runner against one defender (solver: scripts/run.py)
andrew-fixedpasser/   3v1 game: a scripted passer, the runner and a teammate against one defender
src/offball_value/    tracking loaders, run-onset detection, scene building and local-game code
scripts/              the pipeline steps, the figures and the evaluation numbers (below)
jobs/                 SLURM job scripts for the solver runs
data/processed/       fitted pass-model coefficients and measured movement limits (numbers only)
data/static/          a small static model input (EPV grid)
tests/                unit tests (each solver package has its own tests/ as well)
```

## Setup

Python 3.11:

```bash
./bootstrap.sh
```

Download the IDSSE dataset (below) and place it at `data/raw/bundesliga-integrated/`.
The scene annotations and the raters' scores used to pick the showcase scenes
(`chani/chani_shot_annotation.xlsx`, the rating CSVs) are team data and are not
included. Everything the pipeline writes goes under `data/processed/` and `out/`,
both ignored by Git.

## Reproducing the abstract

The steps below run in order; each script documents its inputs, outputs and
options in its docstring and `--help`.

1. **Run onsets.** `scripts/extract_settled_possession_run_onsets.py`, once per
   match, finds the moments an off-ball run starts inside settled possessions
   (the abstract uses a 25 degree direction change and a 1.5 m/s speed gain):

   ```bash
   PYTHONPATH=src python scripts/extract_settled_possession_run_onsets.py \
     --data-dir data/raw/bundesliga-integrated --match-id <match> \
     --output-dir data/processed/run_onset_v0_5/dir25/<match> --exclude-match none \
     --no-require-ball-in-opponent-half --adjust-for-dismissals \
     --min-phase-duration-seconds 6.0 --min-direction-change-degrees 25.0 \
     --min-speed-gain-mps 1.50 --min-post-displacement-m 3.0 --onset-control-distance-m 1.50 \
     --audit-per-match 100000 --audit-total 100000 --dedupe-window-seconds 3.0 \
     --sample-stride-frames 5 --animation-pre-seconds 2.0 --animation-post-seconds 3.0 \
     --animation-fps 12.5
   ```

2. **Scenes and triples.** `scripts/render_local_game_payoff_audit_v0_1.py`
   builds each candidate scene's local game (the runner, the candidate defenders
   and the attacking options they leave), `scripts/apply_pair_gate.py` applies the
   pair plausibility gate and lists the runner / defender / beneficiary triples,
   and `scripts/build_stage3_states.py` and `scripts/filter_stage3_states.py` turn
   the triples into game start states.
3. **Annotated scenes.** `scripts/map_chani_annotations.py` joins the shot
   annotations to the tracking, `scripts/trace_chani_scenes.py` follows the
   annotated scenes through the pipeline, `scripts/build_rating_package.py` and
   `scripts/merge_ratings.py` prepare and collect the three-rater dilemma rating,
   and `scripts/export_showcase.py` exports the showcase scenes.
4. **Pass model and movement limits.** `scripts/build_pass_dataset.py` collects
   the real passes, `scripts/fit_pass_candidates.py` and `scripts/fit_pass_sym.py`
   fit the pass model the games use (A-sym, `data/processed/pass_models_sym/`),
   and `scripts/measure_accelerations.py` measures the players' speed-up, braking
   and turning limits (`data/processed/physics_limits/`).
5. **Game states.** `scripts/build_showcase_states.py` (the figure scenes) and
   `scripts/build_eval_states.py` (the evaluation set) write the start states of
   every real 0.0 / 0.6 / 1.2 s moment.
6. **Solve.**

   ```bash
   sbatch --account=<account> jobs/solve_figure2_panels.sbatch
   # then Figure 2's defender-start grids, one per moment (lattice covering each panel)
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05     -11 6 -9 7
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@0.6  -9 10 -9 7
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@1.2  -8 7 -8 7
   sbatch --account=<account> --cpus-per-task=37 --time=01:30:00 jobs/solve_evaluation.sbatch 2v1
   sbatch --account=<account> --cpus-per-task=32 --time=02:00:00 jobs/solve_evaluation.sbatch 3v1
   ```

7. **Figures and numbers.**

   ```bash
   PYTHONPATH=andrew-passer2on1 python scripts/render_figure1_dilemma.py \
     --output out/showcase_v1/figure1_S05/S05_figure1.png
   python scripts/add_caption.py --input out/showcase_v1/figure1_S05/S05_figure1.png \
     --output out/showcase_v1/abstract_figures/S05_figure1_caption.png --number "Figure 1." \
     --text "An off-ball run creates a defensive dilemma: follow the runner or stay with the ball carrier."
   G=data/processed/showcase_v1/defender_grid_S05
   GRIDS="0.0=$G/grid_S05_1m_full.json 0.6=$G/grid_S05@0.6_1m_full.json 1.2=$G/grid_S05@1.2_1m_full.json"
   python scripts/render_figure2_abstract.py \
     --panels data/processed/showcase_v1/figure_multi/panels --label S05-full \
     --tracking data/processed/showcase_v1/tracking/S05.csv \
     --frames --value-fade 0 --value-grids $GRIDS \
     --flow-grids $GRIDS --flow-color "#FF8000" --flow-alpha 0.4 \
     --defender-names short --label-gap 0.35 --title "" \
     --output out/showcase_v1/figure_multi/S05_figure2.png
   python scripts/add_caption.py --input out/showcase_v1/figure_multi/S05_figure2.png \
     --output out/showcase_v1/abstract_figures/S05_figure2_caption.png --number "Figure 2." \
     --text "Nash equilibrium choices during an off-ball play at three successive moments.\nPercentages are action probabilities; orange lines show the defender's expected movement from each starting position."
   PYTHONPATH=src:scripts python scripts/analyze_eval.py \
     --solved out/runs/eval_v1_2v1 out/runs/eval_v1_3v1 \
     --panels data/processed/eval_v1/panels --output data/processed/eval_v1/analysis
   ```

The remaining scripts fit or check constants and rules that `src/offball_value/`
uses and cites (delivery and xPass models, carrying speed, beneficiary
assignment, run-onset review cohorts).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The solver packages' tests are scripts, one check list per file; each file's
docstring gives its command, for example:

```bash
PYTHONPATH=andrew-passer2on1:andrew-fixedpasser:src python andrew-passer2on1/tests/test_multi_pass.py
```

Several of them read the processed data or earlier solver runs: they look under
the repository root, or under `OFFBALL_DATA_ROOT` (a checkout holding
`data/processed/`) and `OFFBALL_OUT_ROOT` (solver outputs; default `out/runs`).

## Data

The tracking and event data are IDSSE: seven full 2022/23 Bundesliga and
2. Bundesliga matches with synchronized tracking (25 Hz, all players and the
ball), events and metadata. IDSSE is distributed under CC BY 4.0; the data owner
is Deutsche Fußball Liga (DFL). Raw data are never committed to this repository.

- Dataset: <https://doi.org/10.6084/m9.figshare.28196177>
- Paper: <https://doi.org/10.1038/s41597-025-04505-y>

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
