#!/bin/bash
#SBATCH --job-name=pass-prep
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pass_prep_%j.out
# Pass-model candidates A (physics race) and B1 (준현's velocity logistic):
# real passes from our 6 matches -> fit with leave-one-match-out -> tests.
# The four solver jobs depend on this one (afterok): any failure here stops them.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
PY="$ROOT/.venv-delta/bin/python"
M="$ROOT/data/processed/pass_models"
cd "$ROOT"; mkdir -p logs "$M"
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PY" scripts/build_pass_dataset.py --output "$M/passes.jsonl" --workers 6
PYTHONPATH="$ROOT/src:$ROOT/andrew-passer2on1" "$PY" scripts/fit_pass_candidates.py --passes "$M/passes.jsonl" --output "$M"
PYTHONPATH="$ROOT/andrew-passer2on1:$ROOT/andrew-fixedpasser:$ROOT/src" "$PY" andrew-passer2on1/tests/test_agile_motion.py
PYTHONPATH="$ROOT/andrew-passer2on1:$ROOT/andrew-fixedpasser:$ROOT/src" "$PY" andrew-passer2on1/tests/test_pass_candidates.py "$M"
echo DONE
