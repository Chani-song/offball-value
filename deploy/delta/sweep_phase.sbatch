#!/usr/bin/env bash
#SBATCH --job-name=offball-phase
#SBATCH --array=1-35
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/phase_%A_%a.out
#SBATCH --error=logs/phase_%A_%a.out
#
# Stage 0. The gate sweep cannot reach two of Chani's strong five: shot_010
# has its shot 23.7 s outside any phase, and shot_002 sits in a 47 s phase
# whose five onsets are all the wrong player. Neither is a run-shape problem.
#
# The suspect is `opponent_control_confirmation_seconds = 0.32`. A phase ends
# as soon as an opponent is judged in control for a third of a second, which
# a grazed touch satisfies, and the measurement already says this fires far
# too eagerly: 343 of 720 phases resume with the SAME team within 5 s
# (median 1.08 s), and the gaps so created swallow 129 shots.
#
# Five settings x 7 matches:
#   base    0.32 s            the control
#   conf60  0.60 s            past a touch, short of a pass
#   conf100 1.00 s            a full second of opponent possession
#   conf200 2.00 s            deliberately too long, to find where it breaks
#   pctrl30 0.32 s, 3.0 m     loosen who counts as controlling instead
#
# Longer confirmation merges fragments, so phases get longer and fewer. That
# is not automatically good: a phase that runs 47 s is already the failure
# mode in shot_002. Read the recall, and watch the phase-length distribution
# beside it.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(base  conf60 conf100 conf200 pctrl30)
CONF=( 0.32  0.60   1.00    2.00    0.32   )
PCTRL=(1.50  1.50   1.50    1.50    3.00   )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/phase_sweep/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} conf=${CONF[$CFG]} pctrl=${PCTRL[$CFG]} match=${MATCH}"

python scripts/extract_settled_possession_run_onsets.py \
  --data-dir data/raw/bundesliga-integrated \
  --match-id "$MATCH" \
  --output-dir "$OUTDIR" \
  --exclude-match none \
  --no-require-ball-in-opponent-half \
  --adjust-for-dismissals \
  --min-phase-duration-seconds 6.0 \
  --min-direction-change-degrees 25.0 \
  --min-speed-gain-mps 1.50 \
  --min-post-displacement-m 3.0 \
  --onset-control-distance-m 1.50 \
  --opponent-confirmation-seconds "${CONF[$CFG]}" \
  --phase-control-distance-m "${PCTRL[$CFG]}" \
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
