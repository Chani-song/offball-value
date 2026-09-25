#!/usr/bin/env bash
#SBATCH --job-name=offball-xpass-kin
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/xpasskin_%j.out
#SBATCH --error=logs/xpasskin_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/train_xpass_360_kinematic.py \
  --data-root data/raw/statsbomb-open-data/data \
  --output data/processed/xpass_360_kinematic
echo "=== done ==="
