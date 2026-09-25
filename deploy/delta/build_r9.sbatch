#!/usr/bin/env bash
#SBATCH --job-name=offball-r9
#SBATCH --array=1-588
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/r9_%A_%a.out
#SBATCH --error=logs/r9_%A_%a.out
#
# The v0_5 scene pool on the adopted value stack AND the R9 beneficiary rule.
#
# R9 is what the labels favour -- round 1 24/24, round 2 19/21, against R1's
# 20 and 14.5/22 -- and the pipeline had been shipping R1 anyway. Not from a
# disagreement: R1 needs only positions so it can be picked before anything
# is priced, while R9 needs Q_i(t), which does not exist until the responses
# are priced. The build prices every option under every response before it
# forms the local pair, so the curves are there by then; R9 just had to be
# called at that point instead of at the top.
#
# Changing the beneficiary changes the local option set, so Q, the minimax
# and the ranking all move with it. This is not comparable to v6_ssac
# scene-for-scene -- it is a different pair being solved.
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

ROOT="data/processed/run_onset_v0_5/dir25"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")

TASK="${SLURM_ARRAY_TASK_ID}"
CHUNK=$(( (TASK - 1) / CHUNK_SIZE ))
INDEX=$(( (TASK - 1) % CHUNK_SIZE + 1 ))

SCENES="${ROOT}/chunk${CHUNK}.json"
OUTDIR="/work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac/scene_$(printf '%04d' "$TASK")"
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
  --value-stack ssac \
  --derived-option-source coupled_r9

echo "done ssac task ${TASK}"
