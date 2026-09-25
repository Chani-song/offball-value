#!/usr/bin/env bash
#SBATCH --job-name=offball-stack01
#SBATCH --array=1-21
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/stack01_%A_%a.out
#SBATCH --error=logs/stack01_%A_%a.out
#
# Stage 0 and stage 1 together, because each was swept alone and each helped
# alone:
#
#   conf200  opponent confirmation 0.32 -> 2.0 s   strong (1)(2) 4/5 -> 5/5
#   disp15   min post displacement 3.0 -> 1.5 m    strong (3)   2/5 -> 3/5
#
# Whether that composes is not something either sweep can answer. conf200
# merges phases, which changes which runs are even inside a phase, so the
# displacement gate is being applied to a different population than it was
# when it was measured alone.
#
# Three arms x 7 matches, and no stage-2 build yet: recall is decided here,
# and a 588-scene payoff run costs ~500 CPU-hours that should not be spent
# before the recall is known.
#
#   s0only   conf200 alone                    the stage-0 reference
#   s0s1     conf200 + disp15                 the proposed combination
#   s0combo  conf200 + all four gates loose   the ceiling, precision be damned
#
# s0combo is expected to produce a very large onset pool -- gate combo alone
# gave 4,768 against base's 726. It is here to bound what recall is reachable
# at all, not as a candidate.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(s0only s0s1  s0combo)
DISP=( 3.0    1.5   1.5    )
SPEED=(1.50   1.50  0.80   )
DIRDEG=(25.0  25.0  15.0   )
CTRL=( 1.50   1.50  3.00   )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/stack01_sweep/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} match=${MATCH} disp=${DISP[$CFG]} speed=${SPEED[$CFG]} dir=${DIRDEG[$CFG]} ctrl=${CTRL[$CFG]}"

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --min-direction-change-degrees "${DIRDEG[$CFG]}" \
  --min-speed-gain-mps "${SPEED[$CFG]}" \
  --min-post-displacement-m "${DISP[$CFG]}" \
  --onset-control-distance-m "${CTRL[$CFG]}" \
  --opponent-confirmation-seconds 2.00 \
  --phase-control-distance-m 1.50 \
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
