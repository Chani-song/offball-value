#!/usr/bin/env bash
#SBATCH --job-name=offball-extract-v5
#SBATCH --array=1-21
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extract_v5_%A_%a.out
#SBATCH --error=logs/extract_v5_%A_%a.out
#
# v0_4 + minimum_direction_change_degrees 30 -> 25.
#
# 스윕 실측(설정당 7경기 전량 재추출):
#   dir25   장면 369->588 (+219),  strong 3/5 -> 5/5,  s+m 16/45 -> 20/45
#   dir20   장면 369->733 (+364),  strong 5/5,         s+m 21/45   (1건 더에 +145장면)
#   disp20  장면 369->995 (+626),  s+m 18/45           (1건 회수에 313장면)
#   ctrl25  장면 369->410,         회수 0
# 25도에서 strong이 전부 회수되고, 더 내려도 얻는 게 없다.
#
# arm dir25       : 채택안.
# arm dir25_gain10 / dir25_gain08 : 측정용. speed_gain을 얼마나 내릴지는
#   "0.8 m/s 가속을 런의 시작이라 부를 수 있는가"라는 축구적 판단이라
#   숫자만 만들어 두고 채택은 하지 않는다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(dir25 dir25_gain10 dir25_gain08)
GAIN=( 1.50  1.00         0.80 )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/run_onset_v0_5/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} gain=${GAIN[$CFG]} dir=25.0 match=${MATCH}"

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --min-direction-change-degrees 25.0 \
  --min-speed-gain-mps "${GAIN[$CFG]}" \
  --min-post-displacement-m 3.0 \
  --onset-control-distance-m 1.50 \
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
