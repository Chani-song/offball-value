#!/usr/bin/env bash
#SBATCH --job-name=offball-sm
#SBATCH --array=1-83
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/sm_%A_%a.out
#SBATCH --error=logs/sm_%A_%a.out
#
# The scenes that carry the claim, and no others.
#
# Building the whole s0s1 pool costs ~1,450 CPU-hours out of 2,800 left on a
# SHARED allocation. What it would buy is a wider candidate pool, and v7
# (588 scenes) already serves that. The claim being tested -- that the new
# stage 0/1 recovers Chani's runners without costing stage-2 accuracy -- is
# about her 45 strong/medium clips, and the scenes inside those phases are
# 83 of the 1,634. That is 74 CPU-hours for the same answer.
#
#   stage 0  opponent confirmation 2.0 s   (0.32 cut phases on grazed touches)
#   stage 1  min post displacement 1.5 m   (3.0 dropped short runs)
#   stage 2  adopted threat only + R9      (the full adopted stack costs two
#                                           labels; threat alone costs none)
#
# Submitted against bidp-delta-cpu: bbmr has been carrying two days of
# sweeps and is down to 2,806 hours with other people on it.

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
read -r -a SCENES < data/processed/stack01_sweep/s0s1_sm_scenes.txt || true
SCENE="${SCENES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"

ROOT="data/processed/stack01_sweep/s0s1"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/v8_sm/scene_$(printf '%04d' "$SCENE")"
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
