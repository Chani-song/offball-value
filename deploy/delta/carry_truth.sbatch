#!/usr/bin/env bash
#SBATCH --job-name=offball-carry-truth
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/carrytruth_%j.out
#SBATCH --error=logs/carrytruth_%j.out
#
# Empirical carry retention from Bundesliga tracking, against the model's own
# logistic((arrival - 0.35)/0.25) product. Parses ~350 MB of positions XML per
# match, so it belongs in a batch job, not on a login node.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/carry_truth.sbatch [--max-matches 1]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/measure_carry_retention_truth.py \
  --raw-dir data/raw/bundesliga-integrated \
  --stride 5 --horizons 1.0 2.0 \
  --output "${CARRY_OUT:-out/delivery_analysis/carry_retention_truth.json}" \
  ${EXTRA_ARGS:-}
echo "=== done ==="
