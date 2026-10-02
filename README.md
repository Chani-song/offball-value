# The Defender's Dilemma

Game-Theoretic Evaluation of Off-Ball Movement in Soccer

Kyuhyeok Seo, Chan-Eui Song, Andrew Kang, Priya Narasimhan, James Z. Wang

Code for a submission to the MIT Sloan Sports Analytics Conference.

An off-ball run can leave the defender responsible for the runner with two threats to cover: the
runner and the ball carrier (or another teammate). We take real moments of such runs from Bundesliga
tracking data, reduce each to a small game between the two attackers and that defender, and solve the
game for its Nash equilibrium. Every real moment is solved as its own game, so the players' actual
movements can be compared with the options the game makes available to them.

## Method

**Scenes.** Run onsets are detected inside settled attacking possessions of the seven IDSSE matches
(a 25 degree change of direction with a 1.5 m/s gain in speed). Each scene has a runner, the
defender responsible for him and a beneficiary, the teammate who gains if the defender follows the
runner. A rule assigns these roles in pipeline scenes; in annotated scenes they come from the
annotator's labels. When the beneficiary is the ball carrier the game is a 2v1; otherwise it is a
3v1 in which the ball carrier is a scripted passer who follows his real path and only passes.

**Game.** Three simultaneous decisions of 0.6 s each (a 1.8 s horizon). The defender chooses among
five movement commands (stop, or one of four directions relative to the attack); the attack chooses a
joint command for its two players or a pass. Pass candidates are aimed along the receiver's run. All
movement respects speed-up, braking and turning limits measured from the same tracking (99.9th
percentile). Everyone else follows his real tracked path; in 2v1 games the other defenders can still
tackle the ball carrier. A pass is worth its completion probability, from a pass model fitted on our
own passes with each match held out, times a hand-designed positional threat at its target.

**Solution.** The game is zero-sum and finite. It is solved backwards, with a linear program at every
decision state, and every solution is checked against unrestricted best responses for both sides (the
certificate gap must be within the solver's tolerance). The solution is a behaviour strategy: a mixed
choice at every decision state.

Probabilities reported by the code and drawn in the figures are the probability mass that the
equilibrium strategy assigns to an action. They are not pass-completion probabilities, and they are
not the probability that an action is correct.

The game code is in `andrew-passer2on1/` (2v1) and `andrew-fixedpasser/` (3v1). Both build on a
finite-game solver by Andrew Kang (`defensive_positioning`) that is not part of this repository; see
[docs/reproduction.md](docs/reproduction.md#solver-base). The design of each game is in its
`DESIGN.md`.

## Figures

**Figure 1** (`scripts/render_figure1_dilemma.py`) shows the dilemma in scene S05, a scene the authors
chose from those the raters scored highest. From the moment 0.6 s into the play it draws two futures:
the defender follows the runner (what happened, from the tracking) or stays with the ball carrier (a
counterfactual in which the through ball the solver plays at that moment reaches the runner). Positions
come from the tracking; the solver supplies only the through ball's target and the defender's motion in
the counterfactual. The figure carries no numbers.

**Figure 2** (`scripts/render_figure2_abstract.py`) shows the solved games at 0.0, 0.6 and 1.2 s of
the same play: each player's commands with their equilibrium probabilities, the players' real next
0.6 s, and two quantities from re-solving the game with only the defender's start moved over a 1 m
grid: his expected first move from each start (orange lines) and the game value there (shading).

The rendered figures are not committed. [docs/reproduction.md](docs/reproduction.md) lists each
figure's inputs and the exact commands.

## Results

The evaluation set has 26 scenes from the rated pool (7 from a team member's annotated shot clips, 19
from the pipeline), giving 69 real moments and 207 player decisions. `scripts/analyze_eval.py` reads
each moment's opening payoff table from the solver and ranks each player's observed option among his
five or six options by its value against the opponent's equilibrium strategy. Its output from the
2026-09-30 run gives:

| | Observed | Uniform choice |
| --- | --- | --- |
| Moments where the defender's equilibrium is mixed | 33 of 69 | |
| Median rank of the observed option (ties share places) | 2.0 | 3.0 |
| Mean equilibrium probability of the observed option | 0.31 | 0.19 |

These numbers come with three caveats. The observed option is the command whose 0.6 s end point is
nearest the player's real position (median distance 0.78 m), so it is an approximation. Ties are
common: 42.5% of observed options tie with another option. The evaluation set is small, includes the
figure scene, and was drawn from scenes selected for a dilemma rating.

The evaluation output files are not in the repository ([docs/data.md](docs/data.md)). The comparison
between static counterfactuals and the equilibrium (`scripts/static_counterfactual.py`) is implemented,
but its output is not available here, so no result from it is reported.

## Interactive demo

<https://chani-song.github.io/offball-value/> is an explorer for the annotated scenes. Its code
(`demo_viz/`) is developed on separate branches and is not part of this branch.

## Reproducing

| What | Status |
| --- | --- |
| Unit tests | run from this repository alone (CI) |
| Pass model and movement limits | fitted outputs committed; refitting needs the IDSSE files |
| Scenes, game states, solver runs | need the IDSSE files, the private solver base and a SLURM cluster; the showcase scenes also need team annotation sheets |
| Figure 1 | needs the solver output for S05 and a start sheet that are not in the repository |
| Figure 2 | needs the S05 panels, defender grids and tracking excerpt, which are not in the repository; rendered from the team's copies on 2026-10-01 |
| Evaluation numbers | need the evaluation run's solver output and the IDSSE files |

Commands, inputs and what was checked are in [docs/reproduction.md](docs/reproduction.md).

## Repository

```text
andrew-passer2on1/    2v1 game package, solver run script, design notes, check scripts
andrew-fixedpasser/   3v1 game package (scripted passer), the same layout
src/offball_value/    tracking loaders, run-onset detection, scene construction
scripts/              pipeline steps, figures, evaluation
jobs/                 SLURM scripts for the solver runs
data/processed/       fitted pass-model coefficients and measured movement limits
data/static/          static EPV grid (from PAUSA, Apache-2.0)
tests/                unit tests
docs/                 reproduction, data, code status
```

[docs/code_status.md](docs/code_status.md) separates the code on the paper's path from supporting
scripts and from modules left over from earlier formulations, which are kept for now.

## Data

The tracking and event data are IDSSE (Bassek et al., 2025; CC BY 4.0; data owner DFL), which must be
downloaded separately into `data/raw/bundesliga-integrated/`. The repository includes only fitted
coefficients and measured limits derived from it. [docs/data.md](docs/data.md) lists what is included,
what is not and why, and which steps need which inputs.

## Installation

Python 3.11.

```bash
./bootstrap.sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

`requirements.txt` pins the versions the tests were run with. The solver packages also need the solver
base cloned into `andrew/` ([docs/reproduction.md](docs/reproduction.md#solver-base)).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the IDSSE dataset ([docs/data.md](docs/data.md)).

## License

No license has been chosen for this code yet, so it is not licensed for reuse. The solver base it
imports is in a private repository without a license. `data/static/EPV_grid.csv` is Apache-2.0
(PAUSA).
