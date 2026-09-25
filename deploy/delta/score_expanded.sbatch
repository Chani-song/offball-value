#!/usr/bin/env bash
#SBATCH --job-name=offball-score-exp
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/scoreexp_%A_%a.out
#SBATCH --error=logs/scoreexp_%A_%a.out
#
# The same comparison on the recovered evaluation set. Of 34 labelled scenes
# only 18 were ever built, because the extractor drops onsets that already
# carry a review; 14 have now been rebuilt from the candidates CSV and the
# confirmed-core manifest. Every number before this rested on 21 pairs, where
# the whole argument was a two-answer gap.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; export PYTHONPATH=src
B=/work/hdd/bbmr/kseo1/offball-out
case "${SLURM_ARRAY_TASK_ID}" in
  1) N="control (하이브리드 raw)";   D=($B/control/scene_* $B/extra_control/scene_* $B/recovered_control/scene_*) ;;
  2) N="calhybrid (보정)";          D=($B/calhybrid/scene_* $B/extra_calhybrid/scene_* $B/recovered_calhybrid/scene_*) ;;
  3) N="xpass360";                  D=(out/scene_*) ;;
  4) N="x360kin (운동학, 새 장면만)"; D=($B/extra_x360kin/scene_* $B/recovered_x360kin/scene_*) ;;
esac
echo "### $N ###"
python scripts/score_coupled_beneficiary.py "${D[@]}" --exclude-frame 57121
echo "=== done ==="
