#!/usr/bin/env bash
#SBATCH --job-name=offball-gates
#SBATCH --array=1-42
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/gates_%A_%a.out
#SBATCH --error=logs/gates_%A_%a.out
#
# Stage 1, not stage 2. The build has all five of Chani's strong phases and
# names her runner in two of them, so the bottleneck is which runs survive
# the onset gates, and nothing done to the beneficiary rule or the value
# stack can move that.
#
# Six settings x 7 matches. Each changes ONE gate away from v0_5, so a
# recovered runner is attributable:
#
#   base      v0_5 exactly, as the control
#   ctrl30    carrier distance 1.5 -> 3.0 m   (Peterson was dropped at 3.22)
#   ctrl50    carrier distance 1.5 -> 5.0 m
#   speed08   min speed gain 1.50 -> 0.80 m/s
#   dir15     min direction change 25 -> 15 degrees
#   disp15    min post displacement 3.0 -> 1.5 m
#
# The count of scenes is NOT the result. score_chani_runner_recall.py reads
# these as recall against her named runners: a gate that doubles the pool and
# recovers nobody is a worse gate, not a looser one.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src

MATCHES=(DFL-MAT-J03WMX DFL-MAT-J03WN1 DFL-MAT-J03WOH DFL-MAT-J03WOY \
         DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9)
NAMES=(base   ctrl30 ctrl50 speed08 dir15  disp15)
CTRL=( 1.50   3.00   5.00   1.50    1.50   1.50  )
SPEED=(1.50   1.50   1.50   0.80    1.50   1.50  )
DIRDEG=(25.0  25.0   25.0   25.0    15.0   25.0  )
DISP=( 3.0    3.0    3.0    3.0     3.0    1.5   )

TASK="${SLURM_ARRAY_TASK_ID}"
CFG=$(( (TASK - 1) / 7 ))
MATCH="${MATCHES[$(( (TASK - 1) % 7 ))]}"
NAME="${NAMES[$CFG]}"

OUTDIR="data/processed/gate_sweep/${NAME}/${MATCH}"
mkdir -p "$OUTDIR"
echo "task ${TASK} cfg=${NAME} match=${MATCH} ctrl=${CTRL[$CFG]} speed=${SPEED[$CFG]} dir=${DIRDEG[$CFG]} disp=${DISP[$CFG]}"

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
  --audit-per-match 100000 --audit-total 100000 \
  --dedupe-window-seconds 3.0 --sample-stride-frames 5 \
  --animation-pre-seconds 2.0 --animation-post-seconds 3.0 --animation-fps 12.5

echo "done ${NAME} ${MATCH}"
