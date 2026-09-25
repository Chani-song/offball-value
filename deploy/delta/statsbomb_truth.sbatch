#!/usr/bin/env bash
#SBATCH --job-name=offball-sb-truth
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/sbtruth_%j.out
#SBATCH --error=logs/sbtruth_%j.out
#
# Ground-truth completion curves from StatsBomb 360 matches, next to what
# xpass360 predicts for the same passes. Reads the open data through the
# symlink into /work/hdd; writes a few KB of JSON.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/statsbomb_truth.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

python scripts/measure_statsbomb_lane_truth.py \
  --data-root data/raw/statsbomb-open-data/data \
  --output out/delivery_analysis/statsbomb_lane_truth.json
echo "=== done ==="
