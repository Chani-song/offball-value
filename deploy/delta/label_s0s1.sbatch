#!/usr/bin/env bash
#SBATCH --job-name=offball-lbl01
#SBATCH --array=1-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/lbl01_%A_%a.out
#SBATCH --error=logs/lbl01_%A_%a.out
#
# Step B: before spending ~500 CPU-hours rebuilding the whole s0s1 pool
# (1,634 scenes against v0_5's 588), check on the labelled scenes that the
# new stage 0/1 has not cost anything at stage 2.
#
# It could: conf200 merges phases, so a scene's horizon and its background
# are different objects than they were, and the beneficiary rule reads both.
# Recall going up at stage 1 does not guarantee the payoff still names the
# right player.
#
# Two arms x 18 labelled scenes:
#   ssac_threat   the recommended stack (adopted threat, ours elsewhere)
#   native        the reference, since native scored 20/23 on the old pool
#
# Both with R9. If ssac_threat holds near 20/23 here, A is worth running.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES=(67 89 398 519 520 685 709 825 975 1071 1153 1182 1216 1217 1250 1432 1491 1617)
ARMS=(ssac_threat native)
N=18

TASK="${SLURM_ARRAY_TASK_ID}"
ARM="${ARMS[$(( (TASK - 1) / N ))]}"
SCENE="${SCENES[$(( (TASK - 1) % N ))]}"

ROOT="data/processed/stack01_sweep/s0s1"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/s0s1_${ARM}/scene_$(printf '%04d' "$SCENE")"
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
