#!/usr/bin/env bash
#SBATCH --job-name=offball-full
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.out
#
# 2단계(수혜자) 산출을 전체 장면 풀에 적용. 한 태스크 = 한 장면.
#
# 배달 모델은 ARM 환경변수로 고른다. 두 팔을 다 돌리는 이유:
# 42쌍에서 hybrid 34 / xpass360 33 은 검정력이 없어 구분 불가다. 지금 하나를
# 고르면 찬의님 88개 레이블이 붙었을 때 비교할 기회를 버리게 된다.
#
# 제출:
#   N=$(python -c "import json;print(json.load(open('data/processed/settled_possession_run_onset_v0_2_full/manifest.json'))['scene_count'])")
#   sbatch --account=bbmr-delta-cpu --array=1-$N --export=ALL,ARM=hybrid   deploy/delta/build_full.sbatch
#   sbatch --account=bbmr-delta-cpu --array=1-$N --export=ALL,ARM=xpass360 deploy/delta/build_full.sbatch
#
# 플래그는 --delivery-model 을 뺀 전부가 goalside_v1 매니페스트 그대로다.
# 하나라도 다르면 두 팔의 비교가 배달 모델 차이인지 알 수 없게 된다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

ARM="${ARM:?ARM must be set: hybrid | xpass360}"
ROOT="data/processed/settled_possession_run_onset_v0_2_full"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")

TASK="${SLURM_ARRAY_TASK_ID}"
CHUNK=$(( (TASK - 1) / CHUNK_SIZE ))
INDEX=$(( (TASK - 1) % CHUNK_SIZE + 1 ))

SCENES="${ROOT}/chunk${CHUNK}.json"
OUTDIR="/work/hdd/bbmr/kseo1/offball-out/full_${ARM}/scene_$(printf '%04d' "$TASK")"
MODEL="data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"

mkdir -p "$OUTDIR"
echo "arm=${ARM} task=${TASK} -> chunk${CHUNK} index ${INDEX} -> ${OUTDIR}"

EXTRA=()
if [ "$ARM" = "xpass360" ]; then
  EXTRA+=(--xpass-360-model "$MODEL")
fi

python scripts/render_local_game_payoff_audit_v0_1.py \
  --scenes-json "$SCENES" \
  --scene-index "$INDEX" \
  --output-dir "$OUTDIR" \
  --release-step-seconds 0.4 \
  --influence-grid-resolution-m 3.0 \
  --response-search-mode target_agnostic \
  --local-game-selection-mode pairwise_cross_cost \
  --raw-response-limit 160 \
  --screen-keep-per-option 8 \
  --screen-keep-minimax 16 \
  --screen-keep-low-effort 8 \
  --candidate-defender-count 3 \
  --local-non-runner-limit 3 \
  --through-ball-leads 2,4,6 \
  --record-candidate-grid \
  --vacated-union 2 \
  --delivery-model "$ARM" \
  "${EXTRA[@]}"
