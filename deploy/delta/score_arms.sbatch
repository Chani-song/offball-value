#!/usr/bin/env bash
#SBATCH --job-name=offball-score-arms
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
B=/work/hdd/bbmr/kseo1/offball-out
case "${SLURM_ARRAY_TASK_ID}" in
  1) NAME="goalside_v1 (기존 하이브리드)"; DIRS=(data/processed/goalside_v1_chunk0 data/processed/goalside_v1_chunk1 data/processed/goalside_v1_chunk2) ;;
  2) NAME="control (하이브리드 재빌드)";    DIRS=($B/control/scene_*) ;;
  3) NAME="calhybrid (보정 하이브리드)";    DIRS=($B/calhybrid/scene_*) ;;
  4) NAME="calmech (보정 역학단독)";        DIRS=($B/calmech/scene_*) ;;
esac
echo "### $NAME ###"
python scripts/score_coupled_beneficiary.py "${DIRS[@]}" --exclude-frame 57121
echo "=== done ==="
