#!/usr/bin/env bash
#SBATCH --job-name=offball-extra
#SBATCH --array=1-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/extra_%A_%a.out
#SBATCH --error=logs/extra_%A_%a.out
#
# Five labelled scenes the 90-scene build never covered, in the same three
# delivery arms, so they extend the evaluation set rather than sitting beside
# it. Of 34 labelled scenes only 18 were built; every comparison today rested
# on 21 pairs, where a one-answer difference is noise.
#
# SUBMIT: sbatch --account=bbmr-delta-cpu deploy/delta/build_extra.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

TASK="${SLURM_ARRAY_TASK_ID}"
ARM=$(( (TASK - 1) / 5 ))
INDEX=$(( (TASK - 1) % 5 + 1 ))
CAL="out/delivery_analysis/hybrid_calibration.json"
case "$ARM" in
  0) NAME=control;   MODEL=hybrid;      EXTRA=() ;;
  1) NAME=calhybrid; MODEL=hybrid;      EXTRA=(--delivery-calibration "$CAL") ;;
  2) NAME=x360kin;   MODEL=xpass360_kinematic; EXTRA=() ;;
esac
BASE=/work/hdd/bbmr/kseo1/offball-out
OUTDIR="$BASE/extra_$NAME/scene_$(printf '%02d' "$INDEX")"
mkdir -p "$OUTDIR"
echo "task $TASK -> arm $ARM ($NAME, $MODEL) scene $INDEX"

python scripts/render_local_game_payoff_audit_v0_1.py \
  --scenes-json data/processed/settled_possession_run_onset_v0_1/labelled_extra.json \
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
  --delivery-model "$MODEL" \
  "${EXTRA[@]}"
echo "=== done ==="
