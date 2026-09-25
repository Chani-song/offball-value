#!/usr/bin/env bash
#SBATCH --job-name=offball-ourcarry
#SBATCH --array=1-18
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ourcarry_%A_%a.out
#SBATCH --error=logs/ourcarry_%A_%a.out
#
# Our own carry model, on the labelled scenes, for the first time.
#
# carry_dynamics.py has existed since 2026-09-17 and has never run in the
# build path -- only in a prototype script. It prices a carry the way the
# pass side is priced:
#
#     pass   margin = defender_time - max(ball_time, receiver_time)
#     carry  margin = defender_time - carrier_time        (he is the ball)
#     P_carry = carrier_first x secure x path_retention
#
# with the carry speed measured rather than assumed: 0.944 of a player's own
# top speed at +/-0.4 s, over 5,059 pairs. The adopted alternative is a
# tackle hazard with three hand-set constants.
#
# `ssac_ourcarry` = adopted threat + adopted completion + our carry, so it
# differs from the full adopted stack in the carry term alone. Scored on the
# same 18 labelled scenes as ablate_on_labels.sbatch, so the five arms are
# directly comparable.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES=(15 23 125 167 168 245 253 293 348 388 425 437 449 460 521 546 580 581)
ARMS=(ssac_ourcarry)
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
