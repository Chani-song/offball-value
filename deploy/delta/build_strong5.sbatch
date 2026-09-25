#!/usr/bin/env bash
#SBATCH --job-name=offball-strong
#SBATCH --array=1-13
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/strong_%A_%a.out
#SBATCH --error=logs/strong_%A_%a.out
#
# The five scenes Chani marked strong, under the settings that finally
# detect all five.
#
# This is not an extension of the v7 shortlist and cannot be merged into it:
# v7 is R1 + the full adopted stack on the v0_5 pool, these are R9 + adopted
# threat on the s0s1 pool, and Q is on a different scale in each. They answer
# a different question anyway --
#
#   v7 shortlist   what our ranking picks out of everything
#   these five     whether what the expert picked is in there at all
#
# Stage 0/1 already showed the runner is recovered 5/5. This builds stage 2
# so the five can be looked at rather than counted.
#
# 13 scenes because her five clips span 13 onsets. Submitted against
# bidp-delta-cpu; bbmr is down to 2,806 hours with other people on it.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

# `read` returns non-zero at EOF when the file has no trailing newline,
# and set -e then kills the task before it prints anything -- which is
# exactly what 83 empty logs looked like.
read -r -a SCENES < data/processed/stack01_sweep/s0s1_strong_scenes.txt || true
SCENE="${SCENES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"

ROOT="data/processed/stack01_sweep/s0s1"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/v8_strong/scene_$(printf '%04d' "$SCENE")"
mkdir -p "$OUTDIR"
echo "task=${SLURM_ARRAY_TASK_ID} scene=${SCENE} -> chunk${CHUNK} index ${INDEX}"

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
  --value-stack ssac_threat \
  --derived-option-source coupled_r9

echo "done scene ${SCENE}"
