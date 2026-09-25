#!/usr/bin/env bash
#SBATCH --job-name=offball-keepacc
#SBATCH --array=1-18
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/keepacc_%A_%a.out
#SBATCH --error=logs/keepacc_%A_%a.out
#
# Separating the two changes "ssac" makes at once.
#
# Adopting the full stack costs two labels (20/23 -> 18/23) while each single
# substitution costs none. But "ssac" is not three changes, it is four: the
# three models, plus dropping goal_side_accessibility.
#
# That drop was justified as redundancy -- the adopted threat has a defender
# term of its own. Junhyun confirmed on 2026-09-21 that his xT only ever used
# defenders near the ball, and the code agrees: the term is
# 1 - exp(-d/9) for ONE defender, while goal_side_accessibility reads the
# whole defensive shape off an influence grid including the keeper. One
# player against eleven is not redundancy, so the justification was wrong.
#
# This arm is the full adopted stack with accessibility KEPT. Against the 18
# labelled scenes:
#   20/23  -> dropping accessibility was the cost, not the models
#   18/23  -> the models themselves, and the drop is incidental
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES=(15 23 125 167 168 245 253 293 348 388 425 437 449 460 521 546 580 581)
ARMS=(ssac_keepaccess)
N=18

TASK="${SLURM_ARRAY_TASK_ID}"
ARM="${ARMS[$(( (TASK - 1) / N ))]}"
SCENE="${SCENES[$(( (TASK - 1) % N ))]}"

ROOT="data/processed/run_onset_v0_5/dir25"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/label_${ARM}/scene_$(printf '%04d' "$SCENE")"
mkdir -p "$OUTDIR"
echo "task=${TASK} arm=${ARM} scene=${SCENE} -> chunk${CHUNK} index ${INDEX}"

python scripts/render_local_game_payoff_audit_v0_1.py \
  --scenes-json "${ROOT}/chunk${CHUNK}.json" \
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
  --value-stack "$ARM" \
  --derived-option-source coupled_r9

echo "done ${ARM} scene ${SCENE}"
