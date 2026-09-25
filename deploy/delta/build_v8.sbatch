#!/usr/bin/env bash
#SBATCH --job-name=offball-v8
#SBATCH --array=1-1634
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/v8_%A_%a.out
#SBATCH --error=logs/v8_%A_%a.out
#
# Everything today's measurements pointed at, in one pool.
#
#   stage 0  opponent confirmation 0.32 -> 2.0 s
#   stage 1  min post displacement 3.0 -> 1.5 m
#   stage 2  adopted threat only, plus R9 for the beneficiary
#
# Chani's strong five go 2/5 -> 5/5 on the runner, and s+m 4/45 -> 12/45,
# with the two stage gates together; neither alone gets past 3/5, because a
# run dropped by both is recovered by neither. Stage 2 takes the adopted
# threat and nothing else: on the labelled scenes the full adopted stack
# costs two, while threat alone costs none.
#
# 1,634 scenes against v0_5's 588, so about 1,450 CPU-hours at the measured
# 13.3 min per scene. That is roughly half the remaining allocation.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

ROOT="data/processed/stack01_sweep/s0s1"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")

TASK="${SLURM_ARRAY_TASK_ID}"
CHUNK=$(( (TASK - 1) / CHUNK_SIZE ))
INDEX=$(( (TASK - 1) % CHUNK_SIZE + 1 ))

SCENES="${ROOT}/chunk${CHUNK}.json"
OUTDIR="/work/hdd/bbmr/kseo1/offball-out/v8_s0s1_threat/scene_$(printf '%04d' "$TASK")"
mkdir -p "$OUTDIR"
echo "task=${TASK} -> chunk${CHUNK} index ${INDEX} -> ${OUTDIR}"

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
  --delivery-model hybrid \
  --value-stack ssac_threat \
  --derived-option-source coupled_r9

echo "done ssac task ${TASK}"
