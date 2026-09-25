#!/usr/bin/env bash
#SBATCH --job-name=offball-ssac
#SBATCH --array=1-588
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ssac_%A_%a.out
#SBATCH --error=logs/ssac_%A_%a.out
#
# The v0_5 scene pool rebuilt on stage 3's own value stack: its completion
# model, its positional threat, and its tackle hazard for carries.
#
# Why all three and not the pass model alone: swapping the delivery model by
# itself was measured here and moved pass options 1.9x while leaving carry
# options at 1.0x, so the two action classes stopped being comparable. These
# three are coherent with each other. `--value-stack ssac` switches them
# together and folds the accessibility factor away, because the adopted
# threat already carries a defender-room term.
#
# Everything else is byte-identical to build_v0_5.sbatch, so a difference
# between v5_hybrid and v6_ssac is the value stack and nothing else.
#
# Submitted with --dependency=afterok on verify_ssac.sbatch.

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
OUTDIR="/work/hdd/bbmr/kseo1/offball-out/v6_ssac/scene_$(printf '%04d' "$TASK")"
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
  --value-stack ssac

echo "done ssac task ${TASK}"
