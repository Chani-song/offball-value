#!/usr/bin/env bash
#SBATCH --job-name=offball-ctrl-sweep
#SBATCH --array=1-28
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ctrlsweep_%A_%a.out
#SBATCH --error=logs/ctrlsweep_%A_%a.out
#
# onset 게이트의 control_distance_m 스윕. 설정 4종 x 경기 7 = 28 태스크.
#
# 근거: 소유 국면 프레임의 42%에서 공 최근접 선수가 1.5m 밖이고, 그때 공은
# 중앙값 10 m/s로 이동 중이다(대부분 땅 패스, 공중은 22.8%). 즉 이 게이트는
# "패스가 이동 중인 순간"을 통째로 버린다. 찬의님 strong 장면 J03WOH shot_006
# 에서도 Peterson의 온셋 하나가 carrier_far(3.22m)로 떨어졌다.
#
# 다만 값을 키우면 5.79m 떨어진 선수를 "볼 소유자"라고 부르게 되고, 그
# ball_carrier_id는 1·2단계로 그대로 흘러간다. 그래서 이건 채택이 아니라
# 교환비 측정이다: 장면이 몇 개 늘고, 찬의님 strong/medium 회수가 몇 개 느는가.
#
# v0_5(dir 25도)를 기준으로 control_distance_m만 바꾼다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(ctrl15 ctrl30 ctrl50 ctrl99)
CTRL=( 1.50   3.00   5.00   99.0 )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/sweep_ctrl/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} control_distance=${CTRL[$CFG]} match=${MATCH}"

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --min-direction-change-degrees 25.0 \
  --min-speed-gain-mps 1.50 \
  --min-post-displacement-m 3.0 \
  --onset-control-distance-m "${CTRL[$CFG]}" \
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
