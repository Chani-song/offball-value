#!/usr/bin/env bash
#SBATCH --job-name=offball-delivery-analysis
#SBATCH --array=1-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/analysis_%A_%a.out
#SBATCH --error=logs/analysis_%A_%a.out
#
# Two independent post-hoc analyses of the delivery-model swap. Neither
# rebuilds anything: both read audit JSON that is already on disk.
#
#   task 1  compare_delivery_models_split.py
#           The hybrid-vs-360 defender-response comparison restricted to
#           genuine pass cells, with the terminal_structure fallback (which
#           has no defender in its formula) removed and the lane distance
#           recorded, so "hybrid 0.000" can be read for what it is.
#
#   task 2  measure_action_class_commensurability.py
#           Whether pass options and carry options are priced on the same
#           scale, and the k-sweep that tests whether the lost beneficiary
#           answers are an exchange-rate artefact.
#
# SUBMIT:  sbatch --account=bbmr-delta-cpu deploy/delta/analysis.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

case "${SLURM_ARRAY_TASK_ID}" in
  1)
    echo "=== task 1: 실제 패스 셀만으로 배달 모델 맞대결 ==="
    python scripts/compare_delivery_models_split.py \
      --audits data/processed/goalside_v1_chunk0 \
               data/processed/goalside_v1_chunk1 \
               data/processed/goalside_v1_chunk2 \
      --output out/delivery_analysis/pass_only_comparison.json
    ;;
  2)
    echo "=== task 2: 패스/캐리 통약성과 교환비 스윕 ==="
    python scripts/measure_action_class_commensurability.py \
      --hybrid data/processed/goalside_v1_chunk0 \
               data/processed/goalside_v1_chunk1 \
               data/processed/goalside_v1_chunk2 \
      --x360 out/scene_* \
      --output out/delivery_analysis/commensurability.json
    ;;
  *)
    echo "unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1
    ;;
esac
echo "=== done ==="
