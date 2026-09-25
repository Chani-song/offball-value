#!/usr/bin/env bash
#SBATCH --job-name=offball-rebuild-variants
#SBATCH --array=1-270
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/rebuild_%A_%a.out
#SBATCH --error=logs/rebuild_%A_%a.out
#
# Three delivery variants over the same 90 scenes, so the only thing that
# differs between arms is the priced P.
#
#   arm 0  calhybrid  hybrid + isotonic calibration   (level fixed, response flattened)
#   arm 1  calmech    mechanistic + calibration       (also drops the xPass factor,
#                                                      which measurement showed
#                                                      contributes 2.9 % of log-variance)
#   arm 2  control    hybrid, no calibration          (must reproduce goalside_v1)
#
# Arm 2 is the control: if it does not reproduce the existing hybrid build, the
# calibration wiring changed something it should not have, and arms 0 and 1 mean
# nothing. Check it first.
#
# Output goes to /work/hdd (2.4 TB free); /projects/bbmr is at 95 %.
#
# SUBMIT: sbatch --account=bbmr-delta-cpu deploy/delta/rebuild_variants.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

TASK="${SLURM_ARRAY_TASK_ID}"
ARM=$(( (TASK - 1) / 90 ))
SCENE=$(( (TASK - 1) % 90 + 1 ))
CHUNK=$(( (SCENE - 1) / 30 ))
INDEX=$(( (SCENE - 1) % 30 + 1 ))

CAL="out/delivery_analysis/hybrid_calibration.json"
case "$ARM" in
  0) NAME=calhybrid; MODEL=hybrid;      EXTRA=(--delivery-calibration "$CAL") ;;
  1) NAME=calmech;   MODEL=mechanistic; EXTRA=(--delivery-calibration "$CAL") ;;
  2) NAME=control;   MODEL=hybrid;      EXTRA=() ;;
  *) echo "bad arm $ARM" >&2; exit 1 ;;
esac

BASE=/work/hdd/bbmr/kseo1/offball-out
OUTDIR="$BASE/$NAME/scene_$(printf '%02d' "$SCENE")"
mkdir -p "$OUTDIR"
echo "task $TASK -> arm $ARM ($NAME, $MODEL) scene $SCENE (chunk$CHUNK idx $INDEX)"

# Every flag below is goalside_v1's manifest. Only the delivery model and the
# calibration differ between arms; anything else would confound the comparison.
python scripts/render_local_game_payoff_audit_v0_1.py \
  --scenes-json "data/processed/settled_possession_run_onset_v0_1/chunk${CHUNK}.json" \
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
