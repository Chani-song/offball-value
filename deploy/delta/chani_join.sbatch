#!/usr/bin/env bash
#SBATCH --job-name=offball-chani-join
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/chani_join_%j.out
#SBATCH --error=logs/chani_join_%j.out
#
# 찬의님 슛 주석(period, period_seconds)을 트래킹 프레임으로 옮기고
# 두 가지를 잰다:
#   1. 시계 정합성 — 주석 슛 시각이 DFL ShotAtGoal 이벤트와 맞는가
#   2. 커버리지   — 레이블된 슛이 후보 풀 possession 안에 들어오는가
# 2번이 0에 가까우면 strong/medium 대조 자체가 성립하지 않는다.
#
# 제출:
#   sbatch --account=bbmr-delta-cpu --export=ALL,CANDIDATES=<csv>,OUT=<csv> deploy/delta/chani_join.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src

CANDIDATES="${CANDIDATES:-data/processed/settled_possession_run_onset_v0_2_full/shot_context_run_onsets.csv}"
OUT="${OUT:-data/processed/chani_annotation_join.csv}"

echo "candidates=${CANDIDATES}"
echo "output=${OUT}"
date

python scripts/map_chani_annotations.py \
  --annotations chani/chani_shot_annotation.xlsx \
  --raw-dir data/raw/bundesliga-integrated \
  --candidates "$CANDIDATES" \
  --output "$OUT"

date
