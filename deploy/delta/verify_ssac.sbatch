#!/usr/bin/env bash
#SBATCH --job-name=offball-ssac-verify
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ssacverify_%j.out
#SBATCH --error=logs/ssacverify_%j.out
#
# The gate in front of the 588-scene run. build_ssac.sbatch is submitted with
# --dependency=afterok on this job, so if the adapter is wrong the array never
# starts and no allocation is spent on 588 confidently wrong audits.
#
# What it can catch that an exception cannot: a mirrored pitch, defender roles
# picked against the wrong anchor, a family index read as a string, a threat
# that returns the same number everywhere.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

# A scene costs about 13 minutes per stack, so the default three is a ~80
# minute job against a 90 minute wall. VERIFY_INDICES trims it for a backup
# gate with a longer limit; PYTHONUNBUFFERED so a timeout still leaves a log.
export PYTHONUNBUFFERED=1
python scripts/verify_value_stack.py \
  --scenes-json data/processed/run_onset_v0_5/dir25/chunk0.json \
  --scene-indices "${VERIFY_INDICES:-1,2,3}"

echo "gate passed"
