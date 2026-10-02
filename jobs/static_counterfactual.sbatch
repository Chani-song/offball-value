#!/bin/bash
#SBATCH --job-name=static-cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=logs/static_cf_%x_%j.out
# Static counterfactuals held for the whole window, for the abstract's comparison with the equilibrium: every
# evaluation moment's game (jobs/solve_evaluation.sbatch) is rebuilt exactly and solved four times -- free (V,
# checked against the saved value), the defender held to his real commands at every turn (S), the attack's
# static-best plan held (R), the real attack held (A); see scripts/static_counterfactual.py. A 3v1 game takes
# 45-55 min to build, so every moment runs in parallel.
#
# Submit from the repository root:
#   sbatch --account=<your account> --cpus-per-task=37 --time=02:00:00 -J static-2v1 jobs/static_counterfactual.sbatch 2v1
#   sbatch --account=<your account> --cpus-per-task=32 --time=02:30:00 -J static-3v1 jobs/static_counterfactual.sbatch 3v1
# then: python scripts/summarize_static.py data/processed/eval_v1/static_full
# Environment: PYTHON (default: python), SOLVED (default: out/runs/eval_v1_<kind>).
set -uo pipefail
KIND=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${PYTHON:-python}"
SOLVED="${SOLVED:-$ROOT/out/runs/eval_v1_$KIND}"
cd "$ROOT"; mkdir -p logs
if [ "$KIND" = 2v1 ]; then PKG=andrew-passer2on1; else PKG=andrew-fixedpasser; fi
PYTHONPATH="$ROOT/$PKG:$ROOT/src:$ROOT/scripts" "$PY" scripts/static_counterfactual.py "$KIND" --solved "$SOLVED"   --output "$ROOT/data/processed/eval_v1/static_full" --workers "${SLURM_CPUS_PER_TASK:-8}"
echo "== DONE $(date)"
