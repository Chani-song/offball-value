# Reproduction

This file lists every step behind the paper's figures and evaluation numbers, the inputs each step
reads, and what we could and could not re-run from this repository on 2026-10-01. Data sources and
redistribution terms are in [data.md](data.md).

## Status at a glance

| Step | Command | Needs | Checked 2026-10-01 |
| --- | --- | --- | --- |
| Unit tests | `PYTHONPATH=src python -m unittest discover -s tests` | this repository only | 71 tests pass (also in CI) |
| Solver check scripts | `andrew-*/tests/*.py` | solver base, processed scenes | 2 of 18 run without processed data and pass; 16 need `data/processed/` files that are not in the repository |
| Run onsets, scenes, triples | `scripts/extract_settled_possession_run_onsets.py` and steps 2 to 3 below | IDSSE tracking; steps 3 and 5 also the team's annotation and rating sheets | not re-run |
| Pass model, movement limits | `scripts/fit_pass_sym.py`, `scripts/measure_accelerations.py` | IDSSE tracking | not re-run; their outputs are committed |
| Game states | `scripts/build_showcase_states.py`, `scripts/build_eval_states.py` | IDSSE tracking, the hand-picked start sheet (showcase only) | not re-run |
| Solve | `jobs/*.sbatch` | solver base, game states, a SLURM cluster | not re-run (about 30 CPU-minutes per game) |
| Figure 1 | `scripts/render_figure1_dilemma.py` | panel, states file, start sheet, scene tracking | not re-run: the states file and start sheet are not available locally |
| Figure 2 | `scripts/render_figure2_abstract.py` | three panels, scene tracking, three defender grids, `scenes.csv`, IDSSE match information | rendered from the team's data bundles (see below); matches the submitted PDF by eye, not pixel by pixel |
| Evaluation numbers | `scripts/analyze_eval.py` | solver output, panels, IDSSE tracking | not re-run; its 2026-09-30 output was read and the numbers in the README recomputed from it |
| Static comparison | `scripts/static_counterfactual.py`, `scripts/summarize_static.py` | solver base, evaluation run, IDSSE tracking | not re-run; no output available locally |

## Environment

```bash
./bootstrap.sh                 # .venv with requirements.txt (pinned), this package, andrew/ if present
```

The versions in `requirements.txt` are the ones the tests and the Figure 2 render above ran with. The
solver runs behind the paper used the cluster's environment, whose versions were not recorded.

### Solver base

The two solver packages (`andrew-passer2on1/passer2on1`, `andrew-fixedpasser/fixedpasser`) import
`defensive_positioning`, the finite-game solver written by Andrew Kang. It lives in a separate
repository, `andrewkang12345/mit_ssac2027`, which is **private and carries no license**. Both packages
were written against commit `e8b0a95`:

```bash
git clone https://github.com/andrewkang12345/mit_ssac2027.git andrew   # needs access
git -C andrew checkout e8b0a95
.venv/bin/python -m pip install -e andrew --no-deps
```

Without it the solver packages, the solver check scripts, `scripts/static_counterfactual.py`,
`scripts/extract_defender_grid.py`, `scripts/extract_panel_policy.py`, `scripts/build_stage3_states.py`,
`scripts/fit_pass_candidates.py` and the `src/offball_value` modules `agile_motion`,
`physics_pass`, `run_passes`, `ssac_stack` and `stage3_read` do not import. The unit tests in `tests/` do
not need it. Several check scripts also read `andrew/models/experimental_pass.json`, the solver base's
positions-only pass model; the paper's games use the A-sym model in `data/processed/pass_models_sym/`.

### Font

Both figures are set in Nimbus Sans (URW base35, AGPL-3.0 with a font exception), loaded from
`/usr/share/fonts/urw-base35` or from the directory in `OFFBALL_FONT_DIR`. The two files
(`NimbusSans-Regular.otf`, `NimbusSans-Bold.otf`) are in
<https://github.com/ArtifexSoftware/urw-base35-fonts>. `figure_style.py` stops if a text item would
fall back to another face.

## Pipeline

Each script documents its inputs, outputs and options in its docstring and `--help`. Paths below are
the defaults the scripts and jobs use; everything under `data/processed/` (except the committed model
files) and `out/` is written by the pipeline and ignored by Git.

1. **Run onsets.** Once per match, `scripts/extract_settled_possession_run_onsets.py` segments settled
   attacking possessions and detects the moments an off-ball run starts. The paper uses a 25 degree
   direction change and a 1.5 m/s speed gain:

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

2. **Local games and triples.** `scripts/render_local_game_payoff_audit_v0_1.py` builds each candidate
   scene's local game (the runner, the candidate defenders, the attacking options each leaves).
   `scripts/apply_pair_gate.py` applies the pair plausibility gate and lists runner / defender /
   beneficiary triples. `scripts/build_stage3_states.py` and `scripts/filter_stage3_states.py` turn
   the triples into game start states (2v1 when the beneficiary is the ball carrier, 3v1 otherwise).

3. **Annotated scenes and rating.** `scripts/map_chani_annotations.py` joins a team member's shot-clip
   annotations to the tracking and `scripts/trace_chani_scenes.py` follows them through the pipeline.
   `scripts/build_rating_package.py` assembles 75 scenes for a dilemma rating: the 45 annotated clips
   and 30 pipeline scenes that an earlier solver configuration flagged as dilemmas (12 2v1, 18 3v1); `scripts/merge_ratings.py` collects the raters' scores, and
   `scripts/export_showcase.py` keeps the scenes the raters scored 4 or 5 with at least one 5.
   The annotation and rating sheets are team data and are not in the repository.

4. **Pass model and movement limits.** `scripts/build_pass_dataset.py` collects open-play passes from
   six of the seven matches (DFL-MAT-J03WN1 is excluded, see its docstring).
   `scripts/fit_pass_sym.py` fits the A-sym pass model, one fit on all six matches and one per
   held-out match (`data/processed/pass_models_sym/`; the router `Asym_crossfit.json` prices a scene
   with the model fitted without its match). `scripts/measure_accelerations.py` measures players'
   speed-up, braking and turning limits; the games use the 99.9th percentiles with no reaction delay
   (`data/processed/physics_limits/agile_p999_nodelay.json`).

5. **Game states.** `scripts/build_showcase_states.py` builds the start states of the showcase scenes
   from the raw tracking at the start frames a reviewer picked (`solver_starts_<timestamp>.csv`, not in
   the repository). `scripts/build_eval_states.py` builds the evaluation scenes' states at the
   pipeline's own run onsets. Both write one state per real moment: 0.0, 0.6 and 1.2 s after the start.

6. **Solve.** Each moment is solved as its own game with the same settings: 3 decisions of 0.6 s,
   compass commands, the solver base's threat function, pass candidates along the run, every pass its
   own attack column priced after the defender has carried out his command for 0.2 s, the A-sym pass
   model, the measured limits. In 2v1 games the other defenders, on their real tracks, can also tackle
   the ball carrier.

   ```bash
   sbatch --account=<account> jobs/solve_figure2_panels.sbatch          # 4 showcase scenes x 3 moments
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05     -11 6 -9 7
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@0.6  -9 10 -9 7
   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@1.2  -8 7 -8 7
   sbatch --account=<account> --cpus-per-task=37 --time=01:30:00 jobs/solve_evaluation.sbatch 2v1
   sbatch --account=<account> --cpus-per-task=32 --time=02:00:00 jobs/solve_evaluation.sbatch 3v1
   sbatch --account=<account> --cpus-per-task=37 --time=02:00:00 jobs/static_counterfactual.sbatch 2v1
   sbatch --account=<account> --cpus-per-task=32 --time=02:30:00 jobs/static_counterfactual.sbatch 3v1
   ```

   Each run writes the solver base's study layout (`manifest.json`, `starting_states.json`,
   `states/state_NNN.json` with the value, the certificate and the opening payoff table,
   `policies/state_NNN.npz` with the behaviour strategy at every decision state).
   `scripts/extract_panel_policy.py` reads one moment's opening decision into a panel file;
   `scripts/extract_defender_grid.py` and `scripts/check_defender_grid.py` do the same for a defender
   grid and check it (probabilities sum to 1, the observed start reproduces the reference game, the
   certificate gap at every start).

## Figure 1

`scripts/render_figure1_dilemma.py`, scene S05, attack to the right.

| | |
| --- | --- |
| Shows | the moment 0.6 s into S05 and two futures: the defender follows the runner (what happened, drawn at the shot) or stays with the ball carrier (counterfactual, drawn when the solver's through ball would arrive) |
| Reads | `data/processed/showcase_v1/tracking/S05.csv`; `solver_starts_202609281928.csv`; the 0.6 s panel `figure_multi/panels/S05-full_0.6.json` (through-ball target); `figure_2v1/states_2v1_all.json` (the defender's top speed); the A-sym router (ball speed); the movement limits; `scenes.csv` and the IDSSE match information (match line) |
| Model output used | the through ball's target and the defender's counterfactual run, moved with the solver's own motion function; no number is drawn |
| Drawing choices | the two option arrows' length (3 m) and direction (toward where the runner and ball carrier were 0.6 s later); a 5-frame mean on tracking paths |
| Missing here | the start sheet and the states file |

The command used for the submitted version:

```bash
PYTHONPATH=andrew-passer2on1 python scripts/render_figure1_dilemma.py --title-size 12 \
  --output out/showcase_v1/figure1_S05/S05_figure1.png
python scripts/add_caption.py --size 11.5 --margin-in 0.08 --input out/showcase_v1/figure1_S05/S05_figure1.png \
  --output out/showcase_v1/abstract_figures/S05_figure1_caption.png --number "Figure 1." \
  --text "An off-ball run creates a defensive dilemma: follow the runner or stay with the ball carrier."
```

## Figure 2

`scripts/render_figure2_abstract.py`, scene S05 at 0.0, 0.6 and 1.2 s.

| | |
| --- | --- |
| Shows | at each real moment, the equilibrium probability of each player's commands and of the passes, the commands' own 0.6 s paths, the players' real next 0.6 s, the defender's expected first move from each start on a 1 m grid (orange lines), and the game value if the defender had started at each grid point (shading) |
| Reads | `figure_multi/panels/S05-full_{0.0,0.6,1.2}.json` (from `jobs/solve_figure2_panels.sbatch`); `tracking/S05.csv`; `defender_grid_S05/grid_S05{,@0.6,@1.2}_1m_full.json` (from `jobs/solve_defender_grid.sbatch`); `scenes.csv` and the IDSSE match information (match line) |
| Grid | at each grid point only the defender's start position moves; the attackers, the ball, every velocity and the other players' tracks stay as observed. Starts solved: 290 of 298 at 0.0 s, 333 of 340 at 0.6 s, 246 of 253 at 1.2 s. At each moment 7 starts lie within the 1 m tackle radius of the ball carrier or the runner and are not solved; one 0.0 s start is left unsolved by the certificate check |
| Orange lines | the probability-weighted 0.6 s displacement of the defender's equilibrium first move, interpolated to 0.25 m, Gaussian-smoothed (sigma 0.6 m) and drawn with `streamplot` (`render_defender_flow_moments.py`: `moves`, `window_field`) |
| Label positions | set by hand in the command below (`--label-at`, `--two-line`, `--no-leader`, `--exit-angle`, `--stretch-to`) |

The command used for the submitted version:

```bash
G=data/processed/showcase_v1/defender_grid_S05
GRIDS="0.0=$G/grid_S05_1m_full.json 0.6=$G/grid_S05@0.6_1m_full.json 1.2=$G/grid_S05@1.2_1m_full.json"
python scripts/render_figure2_abstract.py \
  --panels data/processed/showcase_v1/figure_multi/panels --label S05-full \
  --tracking data/processed/showcase_v1/tracking/S05.csv \
  --frames --value-fade 0 --value-grids $GRIDS \
  --flow-grids $GRIDS --flow-color "#FF8000" --flow-alpha 0.4 \
  --defender-names short --label-gap 0.35 --title "" \
  --value-key "preferred defender position" --value-key-size 10 --head-size 12 --attack-size 10 \
  --min-arrow 10 --stretch-to 0.0:runner:3=20 --exit-angle 1.2:defender:3=-90 0.6:defender:4=-143 \
  --straight 0.6:defender:4 --pass-under-moves \
  --label-at "0.0|To runner 30%=30.0,-5.35" "0.0|Defender=30.3,-8.0" "0.0|Stop 39%=30.22,-6.75" \
    "0.0|3%=32.9,-1.92" "0.0|Dribble 100%=24.31,-10.02" "1.2|Dribble 70%=30.5,-5.15" \
    "1.2|To ball 47%=35.0,-9.8" "1.2|Dribble 30%=31.9,-9.7" "1.2|Stop 22%=36.4,-7.95" \
  --two-line "0.0|To goal 31%" "1.2|Dribble 30%" "1.2|Dribble 70%" "1.2|To ball 47%" "1.2|Stop 22%" \
  --no-leader "0.0|Dribble 100%" "0.0|To goal 31%" "0.0|To runner 30%" "1.2|Dribble 30%" \
    "1.2|Stop 22%" "1.2|11%" \
  --output out/showcase_v1/figure_multi/S05_figure2.png
python scripts/add_caption.py --size 11.5 --input out/showcase_v1/figure_multi/S05_figure2.png \
  --output out/showcase_v1/abstract_figures/S05_figure2_caption.png --number "Figure 2." \
  --text "Nash equilibrium choices during an off-ball play at 0.0 s, 0.6 s and 1.2 s. Percentages are action probabilities. Orange lines show the defender's expected movement from each starting position. Gray dotted lines and hollow markers show the players' actual movement over the next 0.6 s. Shading is the attack's expected threat in equilibrium if the defender started at that spot (darker = lower threat)."
```

**What was checked.** On 2026-10-01 the command above ran with the S05 panels of the evaluation run
(`eval-S05*.json`, same solver settings) and the three grids from the team's data bundles, with
`--no-match-line` because the match information file was not in place. Two things had to be bridged:
the bundle's panels were written before the solver's command names were translated (commit
`e89e78f`), so their defender command names were mapped back with that commit's own table; and the
font check needed the `str()` fix in `figure_style.py`. Every probability label pinned in the command
appears in the output at the same value, and the panels match the submitted figure PDF by eye
(`docs/assets/figure2.png` is converted from it). It was not compared pixel by pixel.

## Evaluation

`scripts/analyze_eval.py` reads, for every evaluation moment, the opening payoff table `M` that the
solver saved (rows: the defender's commands; columns: the attack's joint options, with the rest of the
game already solved), the equilibrium strategies `p` (defender) and `q` (attack), and the tracking
0.6 s later.

* **Observed option.** Each player's real move is the command whose 0.6 s end point (the solver's own
  movement) is nearest where he was 0.6 s later. The ball carrier's observed option is the pass when
  the ball moved faster than any player can run with it (9 m/s). This is an approximation: the median
  distance to the nearest command end is 0.78 m.
* **Value of an option.** For an attack column `a`, `p · M[:, a]`: its value against the defender's
  equilibrium. An attacker's command is worth the best column containing it, with the teammate
  choosing his best command; the ball carrier's pass is the best pass column. A defender's command `d`
  is worth `M[d, :] · q`, lower being better for him. The panels' `slot_values` come from the same
  definitions in the solver's modal-line output.
* **Rank.** 1 plus the number of legal options strictly better than the observed one; the fractional
  rank splits ties evenly. Ties are frequent: 42.5% of observed options tie with another option, and
  8.7% of decisions have all options equal.
* **Equilibrium probability of the observed option.** The player's marginal in `p` or `q`.
* **Mixing.** A moment counts as mixed when the defender's equilibrium puts more than 1e-9 on more
  than one command.

`scripts/static_counterfactual.py` rebuilds each evaluation game and solves it four times: free (V,
checked against the saved value), the defender held to his observed commands at every decision (S),
the attack's best plan against that held defender then answered by a free defender (R), and the
observed attack held (A). `scripts/summarize_static.py` reports `(S - V) / V`, `(S - R) / S`,
`(S - V) / S` and `(V - A) / V`. The `static_*` fields in `analyze_eval.py`'s output are an earlier
version that holds the opponent for the first 0.6 s only.

## Solver check scripts

Each file under `andrew-passer2on1/tests/` and `andrew-fixedpasser/tests/` is a script with a list of
checks; its docstring gives the command, for example

```bash
PYTHONPATH=andrew-passer2on1:andrew-fixedpasser:src python andrew-passer2on1/tests/test_multi_pass.py
PYTHONPATH=andrew-passer2on1:src:scripts python andrew-passer2on1/tests/test_static_counterfactual.py
```

They read the processed scenes and earlier solver runs from the repository root, or from
`OFFBALL_DATA_ROOT` (a tree holding `data/processed/` and `andrew/models/`) and `OFFBALL_OUT_ROOT`
(solver outputs, default `out/runs`). With the solver base installed and no processed data,
`test_pass_commitment.py` and `test_pass_sym.py` pass; the other 16 stop at a missing input file.

## Claims and evidence

| README statement | Where it can be checked |
| --- | --- |
| Each real moment is solved as its own game, 3 decisions of 0.6 s | `jobs/*.sbatch` (`--steps 3 --step-seconds 0.6`), `scripts/build_eval_states.py` |
| 2v1 when the beneficiary is the ball carrier, 3v1 with a scripted passer otherwise | `andrew-passer2on1/DESIGN.md`, `andrew-fixedpasser/DESIGN.md`, `scripts/build_stage3_states.py` |
| Five commands for each player; passes along the run | `--commands compass`, `--passes run`; `run_passes.py` |
| Measured limits, 99.9th percentile | `data/processed/physics_limits/agile_p999_nodelay.json`, `scripts/measure_accelerations.py` |
| Pass model fitted on our passes with each match held out | `data/processed/pass_models_sym/`, `tests/test_public_data.py` |
| Linear program at every decision state, certificate check | `passer2on1/multi_markov.py` (`solve_multi`, `certificate_multi`), `passer2on1/solve.py` (the gap test); the LP itself is in the solver base |
| Probabilities are equilibrium choice probabilities | `scripts/extract_panel_policy.py` (the defender's row of the opening policy; attackers' marginals of the joint attack policy) |
| Figure 2's grid: only the defender's start moves | `scripts/build_defender_grid.py` |
| Evaluation numbers (26 scenes, 69 moments, 207 decisions; 33 mixed; rank 2.0; probability 0.31) | `scripts/analyze_eval.py`; its 2026-09-30 output, not in the repository |
| Uniform-choice baselines (rank 3.0, probability 0.19) | the median of `(n + 1) / 2` and the mean of `1 / n` over the 207 decisions (170 with five options, 37 with six), computed from the same output |
