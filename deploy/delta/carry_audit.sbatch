#!/usr/bin/env bash
#SBATCH --job-name=offball-carry-audit
#SBATCH --array=1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/carry_%A_%a.out
#SBATCH --error=logs/carry_%A_%a.out
#
# The carry (dribble) delivery model has never been audited. Task 1 runs it
# on the hybrid build, task 2 on the xpass360 build -- the carry model is
# identical in both, so any difference is a cross-check on the reader, not
# on the model.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/carry_audit.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

case "${SLURM_ARRAY_TASK_ID}" in
  1)
    python scripts/audit_carry_delivery_model.py \
      --audits data/processed/goalside_v1_chunk0 \
               data/processed/goalside_v1_chunk1 \
               data/processed/goalside_v1_chunk2 \
      --label "(하이브리드 빌드, 90장면)" \
      --output out/delivery_analysis/carry_model_hybrid.json
    ;;
  2)
    python scripts/audit_carry_delivery_model.py \
      --audits out/scene_* \
      --label "(xpass360 빌드, 90장면)" \
      --output out/delivery_analysis/carry_model_x360.json
    ;;
esac
echo "=== done ==="
