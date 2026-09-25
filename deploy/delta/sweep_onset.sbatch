#!/usr/bin/env bash
#SBATCH --job-name=offball-sweep
#SBATCH --array=1-56
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#
# run-onset 임계값 스윕. 설정 8종 x 경기 7 = 56 태스크.
#
# 목적은 "찬의님 strong/medium 장면이 몇 개 회수되는가" 대 "전체 풀이 몇 개
# 늘어나는가"의 교환비를 재는 것이다. 회수만 보면 임계값을 0으로 밀게 된다.
#
# 각 설정은 v0_4에서 한 값만 바꾼다. 여러 개를 동시에 바꾸면 어느 것이
# 기여했는지 알 수 없다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(base gain12 gain10 gain08 dir25 dir20 disp20 ctrl25)
GAIN=( 1.50  1.20   1.00   0.80   1.50  1.50  1.50   1.50 )
DIR=(  30.0  30.0   30.0   30.0   25.0  20.0  30.0   30.0 )
DISP=(  3.0   3.0    3.0    3.0    3.0   3.0   2.0    3.0 )
CTRL=(  1.50  1.50   1.50   1.50   1.50  1.50  1.50   2.50 )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/sweep_onset/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} gain=${GAIN[$CFG]} dir=${DIR[$CFG]} disp=${DISP[$CFG]} ctrl=${CTRL[$CFG]} match=${MATCH}"

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --min-speed-gain-mps "${GAIN[$CFG]}" \
  --min-direction-change-degrees "${DIR[$CFG]}" \
  --min-post-displacement-m "${DISP[$CFG]}" \
  --onset-control-distance-m "${CTRL[$CFG]}" \
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
