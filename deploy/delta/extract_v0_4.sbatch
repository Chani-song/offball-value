#!/usr/bin/env bash
#SBATCH --job-name=offball-extract-v4
#SBATCH --array=1-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extract_v4_%A_%a.out
#SBATCH --error=logs/extract_v4_%A_%a.out
#
# 최종 추출. v0_1 대비 세 가지가 다르다.
#
# 1. --no-require-ball-in-opponent-half
#    모집단은 "우리가 공을 잡고 있고 경합이 아니다"이고 거기에 "상대 진영"은
#    없다. 이 조건은 역습을 정의상 전부 배제했다. 실측: 탈락 232건 중 186건이
#    이 사유 단독, 온셋 시점 공 위치 중앙값 -0.18m(하프라인 위)로 분포가 가장
#    빽빽한 곳을 자르고 있었다. 해제 시 primary 181 -> 367 (실측).
#
# 2. --adjust-for-dismissals (기본 on)
#    run_onset 게이트가 "정확히 22명"을 요구해 퇴장 이후를 전부 버렸다.
#    실측: J03WN1은 경기의 92.5%, J03WQQ는 37.2%가 21명이다. 퇴장 없는 5경기는
#    전 구간 정확히 22명이므로 이 수정은 그 두 경기에만 영향을 준다.
#    추적 누락으로 인원이 모자란 프레임은 여전히 탈락한다.
#
# 3. --exclude-match none / 캡 해제
#    J03WN1 포함(1번+2번이 있어야 비로소 쓸모가 있다). 11v10 구간이 있으므로
#    수치 비교 시 match_id로 걸러야 한다. manifest에 기록된다.
#
# min-phase-duration은 6.0s 유지. 3.0s로 낮추면 phase가 107->142로 늘지만
# 장면은 326->335로 +9개뿐이라, 짧은 possession을 받아들이는 위험만 커진다.

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
MATCH="${MATCHES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"
OUTDIR="data/processed/run_onset_v0_4/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${SLURM_ARRAY_TASK_ID} -> ${MATCH}"; date

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --audit-per-match 100000 \
  --audit-total 100000 \
  --dedupe-window-seconds 3.0 \
  --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 \
  --animation-post-seconds 3.0 \
  --animation-fps 12.5

date; echo "done ${MATCH}"
