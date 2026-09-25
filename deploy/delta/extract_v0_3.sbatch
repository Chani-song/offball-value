#!/usr/bin/env bash
#SBATCH --job-name=offball-extract-v3
#SBATCH --array=1-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extract_v3_%A_%a.out
#SBATCH --error=logs/extract_v3_%A_%a.out
#
# 모집단 정의를 코드에 맞추는 재추출. 경기 7 × arm 2 = 14 태스크.
#
# 동기: 모집단은 "우리가 공을 잡고 있고 경합 상황이 아니다"이다. 그런데
# SettledPossessionConfig.require_ball_in_opponent_half=True 가 온셋 순간 공이
# 우리 진영이면 탈락시킨다. 이 조건은 그 정의에 들어있지 않고, 역습을 정의상
# 전부 배제한다. 실측: 탈락 232건 중 186건(80%)이 이 사유 단독이고, 온셋 시점
# 공 위치의 중앙값은 -0.18m -- 분포가 가장 빽빽한 하프라인을 정확히 자른다.
#
# arm halfoff       : 그 조건만 해제. 나머지 게이트(경합·재개 관련)는 그대로 둔다.
# arm halfoff_short : 거기에 phase 최소 지속시간 6.0s -> 3.0s. 빠른 역습은 공
#                     탈취부터 슛까지 6초 안에 끝날 수 있어 phase 단계에서 먼저
#                     떨어질 수 있다. 몇 개가 살아나는지 재기 위한 것이고,
#                     짧은 phase를 실제로 채택할지는 숫자를 보고 정한다.
#
# v0_1 / v0_2_full 출력은 건드리지 않는다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
TASK="${SLURM_ARRAY_TASK_ID}"
ARM_IDX=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"

if [ "$ARM_IDX" -eq 0 ]; then
  ARM="halfoff";       MINDUR=6.0
else
  ARM="halfoff_short"; MINDUR=3.0
fi

OUTDIR="data/processed/run_onset_v0_3_${ARM}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} -> arm=${ARM} min_dur=${MINDUR} match=${MATCH}"
date

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --min-phase-duration-seconds "$MINDUR" \
  --audit-per-match 100000 \
  --audit-total 100000 \
  --dedupe-window-seconds 3.0 \
  --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 \
  --animation-post-seconds 3.0 \
  --animation-fps 12.5

date
echo "done ${ARM} ${MATCH}"
