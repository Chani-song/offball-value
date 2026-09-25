#!/usr/bin/env bash
#SBATCH --job-name=offball-extract-full
#SBATCH --array=1-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extract_full_%A_%a.out
#SBATCH --error=logs/extract_full_%A_%a.out
#
# 분데스리가 전 경기 run-onset 추출, 경기당 1 태스크.
#
# v0_1 대비 바뀐 것 두 가지뿐:
#   1. --exclude-match none  -> J03WN1 포함 (찬의님 레이블 13개가 거기 있음).
#      11v10 경기이므로 수치 비교 시 match_id로 걸러내야 함. manifest에 기록됨.
#   2. --audit-per-match / --audit-total 을 사실상 무제한으로  -> 경기당 15 / 전체 90
#      캡 해제. 90장면은 모델이 고른 게 아니라 레이블링 샘플용 캡이었음.
#
# 그 외 검출 파라미터(dedupe 3.0s, stride 5, min-phase 6.0s, animation 2/3s @12.5Hz)는
# v0_1 그대로다. 바꾸면 기존 42쌍 평가셋과 장면 정의가 어긋난다.
#
# v0_1 출력은 건드리지 않는다. 그게 42쌍 평가셋의 출처다.

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

OUTDIR="data/processed/settled_possession_run_onset_v0_2_full/${MATCH}"
mkdir -p "$OUTDIR"

echo "task ${SLURM_ARRAY_TASK_ID} -> ${MATCH} -> ${OUTDIR}"
date

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --audit-per-match 100000 \
  --audit-total 100000 \
  --dedupe-window-seconds 3.0 \
  --sample-stride-frames 5 \
  --min-phase-duration-seconds 6.0 \
  --animation-pre-seconds 2.0 \
  --animation-post-seconds 3.0 \
  --animation-fps 12.5

date
echo "done ${MATCH}"
