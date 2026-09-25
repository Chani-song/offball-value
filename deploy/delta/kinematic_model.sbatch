#!/usr/bin/env bash
#SBATCH --job-name=offball-kinematic
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/kinematic_%j.out
#SBATCH --error=logs/kinematic_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/train_kinematic_delivery.py \
  --output out/delivery_analysis/kinematic_model.json
echo "=== done ==="
