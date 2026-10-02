#!/bin/bash
#SBATCH --job-name=defender-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/defender_grid_%j.out
# Figure 2's background and its orange lines: one of S05's three moments re-solved with the controlled defender's
# START moved over a 1 m lattice laid through his real position (scripts/build_defender_grid.py), everything else
# as in that moment's own game -- the same record of the Figure 2 study and the same solver flags as
# jobs/solve_figure2_panels.sbatch. Each start is its own game on its own core (~30 CPU-minutes); then every start
# is read into one grid file (scripts/extract_defender_grid.py) and checked (scripts/check_defender_grid.py: sums,
# the real start against the study's own game).
#
# Submit from the repository root once per moment, after jobs/solve_figure2_panels.sbatch, with the lattice that
# covers that moment's Figure 2 panel (goalward steps, then steps to the attack's left):
#   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05     -11 6 -9 7    # 306 starts
#   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@0.6  -9 10 -9 7   # 340 starts
#   sbatch --account=<account> jobs/solve_defender_grid.sbatch S05@1.2  -8 7 -8 7    # 256 starts
# (the starts within 1 m of an attacker are dropped, not solved). With fewer cores the games queue on the workers:
# wall time ~ starts / cores x 40 min.
# Environment: PYTHON (default: python), REF (the Figure 2 study, default out/runs/multipass_full_2v1),
# OUT (default out/runs/defender_grid_<moment>), GRIDS (default data/processed/showcase_v1/defender_grid_S05).
set -uo pipefail
CODE=${1:?moment code, e.g. S05 or S05@0.6}
F0=${2:?} F1=${3:?} L0=${4:?} L1=${5:?}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${PYTHON:-python}"
REF="${REF:-$ROOT/out/runs/multipass_full_2v1}"
OUT="${OUT:-$ROOT/out/runs/defender_grid_$CODE}"
G="${GRIDS:-$ROOT/data/processed/showcase_v1/defender_grid_S05}"
cd "$ROOT"; mkdir -p logs "$G"
echo "== lattice $(date)"
"$PY" scripts/build_defender_grid.py --states "$REF/starting_states.json" --code "$CODE" --spacing 1 \
  --forward "$F0" "$F1" --left "$L0" "$L1" --out "$G" || exit 1
STATES="$G/states_defender_grid_$CODE.json"
echo "== solve $(date)"
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py --states "$STATES" \
  --model "$ROOT/data/processed/pass_models_sym/Asym_crossfit.json" --allow-proxy-labels --steps 3 --step-seconds 0.6 \
  --commands compass --threat andrew --passes run --physics-file "$ROOT/data/processed/physics_limits/agile_p999_nodelay.json" \
  --multi-pass --pass-reaction-s 0.2 --background-tackles --output "$OUT" --workers "${SLURM_CPUS_PER_TASK:-128}"
echo "== read $(date)"
PYTHONPATH="$ROOT/src" "$PY" scripts/extract_defender_grid.py --solved "$OUT" --grid-states "$STATES" \
  --output "$G/grid_${CODE}_1m_full.json" --workers 8 || exit 1
echo "== check $(date)"
"$PY" scripts/check_defender_grid.py --grid "$G/grid_${CODE}_1m_full.json" --reference "$REF" --code "$CODE" \
  --out "$G/checks" || echo "FAILED check"
echo "== DONE $(date)"
