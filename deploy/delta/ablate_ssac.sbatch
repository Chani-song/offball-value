#!/usr/bin/env bash
#SBATCH --job-name=offball-ablate
#SBATCH --array=1-450
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ablate_%A_%a.out
#SBATCH --error=logs/ablate_%A_%a.out
#
# Adopting the stack reordered the shortlist completely: 0 of the top 20
# survived on three of the four criteria. Before reading that as "the adopted
# threat sees something ours does not", it has to be attributed, because the
# switch bundles three substitutions AND one judgement that is not in the
# adopted code at all -- dropping the accessibility factor.
#
#   ssac_threat       adopted threat, our delivery, accessibility kept
#   ssac_pass         adopted completion + tackle, our threat, accessibility kept
#   native_no_access  our everything, accessibility dropped
#
# Each differs from native in exactly one factor, so the three rank
# correlations against native decompose the full switch.
#
# 150 scenes per arm, not 588: a rank correlation over ~500 triples is already
# tight, and three full arms would cost more than the question is worth.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES_PER_ARM=150
ARMS=(ssac_threat ssac_pass native_no_access)

TASK="${SLURM_ARRAY_TASK_ID}"
ARM_INDEX=$(( (TASK - 1) / SCENES_PER_ARM ))
SCENE=$(( (TASK - 1) % SCENES_PER_ARM + 1 ))
ARM="${ARMS[$ARM_INDEX]}"

ROOT="data/processed/run_onset_v0_5/dir25"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/ablate_${ARM}/scene_$(printf '%04d' "$SCENE")"
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
  --value-stack "$ARM"

echo "done ${ARM} scene ${SCENE}"
