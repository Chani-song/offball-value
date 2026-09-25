#!/usr/bin/env bash
#SBATCH --job-name=offball-ablabel
#SBATCH --array=1-72
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ablabel_%A_%a.out
#SBATCH --error=logs/ablabel_%A_%a.out
#
# The first accuracy evidence about adopting the stack came out AGAINST it:
#
#   v5_hybrid (native)  R9 20/23
#   v6_ssac             R9 18/23
#   v7_r9_ssac          R9 18/23
#
# Two scenes on n=23, so not decisive, but it is the only accuracy number
# there is and every other comparison so far has been a rank correlation.
# Before reading it either way it has to be attributed, and the existing
# ablation arms cannot do that: they cover the first 150 scenes of the pool,
# which contain only 3 of the 34 labelled ones.
#
# So build the labelled scenes themselves, under each ablation, 18 of them
# that live in this pool. Four arms x 18:
#
#   ssac_threat       adopted threat only
#   ssac_pass         adopted completion + tackle only
#   native_no_access  accessibility dropped, nothing else
#   ssac              the full stack, as the reference point
#
# Each differs from native in one factor, so whichever one costs the two
# scenes is the one to argue about -- and if it is native_no_access, the
# cost is mine, not the adopted stack's.
#
# R9 throughout, because that is the rule being scored.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES=(15 23 125 167 168 245 253 293 348 388 425 437 449 460 521 546 580 581)
ARMS=(ssac_threat ssac_pass native_no_access ssac)
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
