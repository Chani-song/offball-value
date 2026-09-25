#!/usr/bin/env bash
#SBATCH --job-name=offball-tackle
#SBATCH --array=1-300
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/tackle_%A_%a.out
#SBATCH --error=logs/tackle_%A_%a.out
#
# Adopting the stack means adopting three hand-set constants with it:
#
#   tackle_rate 2.2   tackle_radius 1.0 m   tackle_softness 0.45 m
#
# They are defaults in the other package's GameConfig, not a fit. At those
# values a defender sitting 0.5 m away for two seconds takes the ball 96 % of
# the time, which is severe enough to be worth testing rather than assuming.
#
# Two perturbations that bracket the default rather than tune it:
#   gentle  half the hazard rate
#   wide    twice the radius, so the danger zone reaches further
#
# If the ranking barely moves across these, the constants are not load-bearing
# and adopting them costs nothing. If it moves, adopting the stack means
# inheriting somebody's tuning, and that has to be said out loud.
#
# Same 150 scenes as ablate_ssac.sbatch, so the arms are comparable.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

SCENES_PER_ARM=150
NAMES=(gentle wide)
RATES=(1.1 2.2)
RADII=(1.0 2.0)

TASK="${SLURM_ARRAY_TASK_ID}"
ARM_INDEX=$(( (TASK - 1) / SCENES_PER_ARM ))
SCENE=$(( (TASK - 1) % SCENES_PER_ARM + 1 ))
NAME="${NAMES[$ARM_INDEX]}"
RATE="${RATES[$ARM_INDEX]}"
RADIUS="${RADII[$ARM_INDEX]}"

ROOT="data/processed/run_onset_v0_5/dir25"
CHUNK_SIZE=$(python -c "import json;print(json.load(open('$ROOT/manifest.json'))['chunk_size'])")
CHUNK=$(( (SCENE - 1) / CHUNK_SIZE ))
INDEX=$(( (SCENE - 1) % CHUNK_SIZE + 1 ))

OUTDIR="/work/hdd/bbmr/kseo1/offball-out/tackle_${NAME}/scene_$(printf '%04d' "$SCENE")"
mkdir -p "$OUTDIR"
echo "task=${TASK} arm=${NAME} rate=${RATE} radius=${RADIUS} scene=${SCENE}"

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
  --value-stack ssac \
  --ssac-tackle-rate "$RATE" \
  --ssac-tackle-radius "$RADIUS"

echo "done ${NAME} scene ${SCENE}"
