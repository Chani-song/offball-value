#!/usr/bin/env bash
#SBATCH --job-name=offball-cond-shift
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/condshift_%j.out
#SBATCH --error=logs/condshift_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/test_conditional_shift.py --audits out/scene_* \
  --output out/delivery_analysis/conditional_shift.json
echo "=== done ==="
