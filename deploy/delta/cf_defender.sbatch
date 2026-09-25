#!/usr/bin/env bash
#SBATCH --job-name=offball-cf-defender
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/cfdef_%j.out
#SBATCH --error=logs/cfdef_%j.out
#
# Controlled intervention: insert one stationary defender on the lane of a real
# Bundesliga pass and recompute all three delivery models. Six evaluations per
# pass, ~350 MB positions XML per match.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/cf_defender.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/probe_counterfactual_defender.py \
  --raw-dir data/raw/bundesliga-integrated \
  --output out/delivery_analysis/counterfactual_defender.json
echo "=== done ==="
