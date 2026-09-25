#!/usr/bin/env bash
#SBATCH --job-name=offball-payoff
#SBATCH --array=1-90
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/payoff_%A_%a.out
#SBATCH --error=logs/payoff_%A_%a.out
#
# One array task per scene. The 90 scenes live in three 30-scene manifests,
# so the task id maps onto (chunk, 1-based index within chunk).
#
# SUBMIT WITH YOUR ALLOCATION:
#   mkdir -p logs && sbatch --account=<allocation> deploy/delta/build.sbatch
#
# VERIFY BEFORE FIRST USE (these are the three things I could not check from
# the laptop):
#   sinfo -s                      -> is the CPU partition called "cpu"?
#   accounts                      -> the --account string
#   module avail python           -> the module setup.sh loads
#
# Tasks are independent and write to separate directories, so a partial run
# is still usable: score whatever out/scene_* exist. That is a real gain over
# the laptop, where the renderer writes nothing until all its scenes finish
# and a crash loses the lot.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate

# One task, one core-bound process: let Slurm do the parallelism rather than
# BLAS threads fighting each other across 90 concurrent tasks.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

TASK="${SLURM_ARRAY_TASK_ID}"
CHUNK=$(( (TASK - 1) / 30 ))
INDEX=$(( (TASK - 1) % 30 + 1 ))

SCENES="data/processed/settled_possession_run_onset_v0_1/chunk${CHUNK}.json"
OUTDIR="out/scene_$(printf '%02d' "$TASK")"
MODEL="data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib"

echo "task ${TASK} -> chunk${CHUNK} index ${INDEX} -> ${OUTDIR}"

# Every flag below is goalside_v1's manifest, with delivery_model as the only
# deliberate change. Anything else differing would confound the comparison.
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
  --delivery-model xpass360 \
  --xpass-360-model "$MODEL"
