#!/usr/bin/env bash
#SBATCH --job-name=offball-vel-recovery
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/velrec_%j.out
#SBATCH --error=logs/velrec_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/test_statsbomb_velocity_recovery.py \
  --output out/delivery_analysis/velocity_recovery.json
echo "=== done ==="
