#!/usr/bin/env bash
#SBATCH --job-name=offball-qvar
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/qvar_%A_%a.out
#SBATCH --error=logs/qvar_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; export PYTHONPATH=src
B=/work/hdd/bbmr/kseo1/offball-out
case "${SLURM_ARRAY_TASK_ID}" in
  1) L="하이브리드 raw (19/21)"; D=($B/control/scene_*) ;;
  2) L="보정 하이브리드 (17/21)"; D=($B/calhybrid/scene_*) ;;
  3) L="보정 역학단독 (17/21)";   D=($B/calmech/scene_*) ;;
  4) L="xpass360 (17/21)";      D=(out/scene_*) ;;
esac
python scripts/decompose_q_variance.py --label "$L" --audits "${D[@]}"
