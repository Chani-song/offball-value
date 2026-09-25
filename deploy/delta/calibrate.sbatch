#!/usr/bin/env bash
#SBATCH --job-name=offball-calibrate
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.out
#
# Fit a monotone calibration for the hybrid delivery model on real Bundesliga
# passes, held out by match. Parses ~350 MB of positions XML per match.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/calibrate.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/calibrate_hybrid_delivery.py \
  --raw-dir data/raw/bundesliga-integrated \
  --output out/delivery_analysis/hybrid_calibration.json
echo "=== done ==="
