#!/usr/bin/env bash
#SBATCH --job-name=offball-fill-xpass
#SBATCH --array=1-104
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/fill_%A_%a.out
#SBATCH --error=logs/fill_%A_%a.out
#
# Put the two xPass arms on the same 42-pair footing as control/calhybrid.
#   tasks   1- 14  xpass360            on the 14 recovered labelled scenes
#   tasks  15-104  xpass360_kinematic  on the original 90
# Until now xpass360 was scored on 21 pairs and the kinematic model on 14
# scenes only, so neither was comparable with the arms that got the rebuild.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
B=/work/hdd/bbmr/kseo1/offball-out
D=data/processed/settled_possession_run_onset_v0_1
TASK="${SLURM_ARRAY_TASK_ID}"

if [ "$TASK" -le 14 ]; then
  MODEL=xpass360; NAME=x360_extra
  MODELFLAG=(--xpass-360-model data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib)
  if [ "$TASK" -le 5 ]; then
    SCENES="$D/labelled_extra.json"; INDEX="$TASK"; TAG=e
  else
    SCENES="$D/labelled_recovered.json"; INDEX=$(( TASK - 5 )); TAG=r
  fi
  OUTDIR="$B/$NAME/scene_${TAG}$(printf '%02d' "$INDEX")"
else
  MODEL=xpass360_kinematic; NAME=x360kin_main
  MODELFLAG=()
  SCENE=$(( TASK - 14 ))
  CHUNK=$(( (SCENE - 1) / 30 )); INDEX=$(( (SCENE - 1) % 30 + 1 ))
  SCENES="$D/chunk${CHUNK}.json"
  OUTDIR="$B/$NAME/scene_$(printf '%02d' "$SCENE")"
fi
mkdir -p "$OUTDIR"
echo "task $TASK -> $MODEL $SCENES index $INDEX -> $OUTDIR"

python scripts/render_local_game_payoff_audit_v0_1.py \
  --scenes-json "$SCENES" --scene-index "$INDEX" --output-dir "$OUTDIR" \
  --release-step-seconds 0.4 --influence-grid-resolution-m 3.0 \
  --response-search-mode target_agnostic \
  --local-game-selection-mode pairwise_cross_cost \
  --raw-response-limit 160 --screen-keep-per-option 8 \
  --screen-keep-minimax 16 --screen-keep-low-effort 8 \
  --candidate-defender-count 3 --local-non-runner-limit 3 \
  --through-ball-leads 2,4,6 --record-candidate-grid --vacated-union 2 \
  --delivery-model "$MODEL" "${MODELFLAG[@]}"
echo "=== done ==="
