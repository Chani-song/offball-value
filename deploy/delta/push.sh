#!/usr/bin/env bash
# Run this ON THE LAPTOP. Copies only what a payoff build reads (~49 MB).
#
# data/raw (16 GB) is deliberately absent: the scene frames are embedded in
# data/processed/settled_possession_run_onset_v0_1/chunk*.json, and a running
# build holds no raw file open.  Verified 2026-09-17.
set -euo pipefail

DELTA_USER="${DELTA_USER:-kseo1}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
# /scratch, not /projects: the bbmr allocation sits at 475 of 500 GB.
# Moved there 2026-09-18. Durable output still goes to /work/hdd --
# see OUTDIR in the build*.sbatch files.
DELTA_PATH="${DELTA_PATH:-/scratch/bbmr/kseo1/offball-value}"

cd "$(dirname "$0")/../.."

PATHS=()
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  PATHS+=("$line")
done < deploy/delta/files.txt

echo "== 전송 목록 =="
du -sk "${PATHS[@]}" \
  | awk '{s+=$1; printf "%8.1f MB  %s\n", $1/1024, $2} END {printf "\n합계 %.0f MB\n\n", s/1024}'

# -R (relative) rather than --files-from: macOS ships openrsync ("rsync 2.6.9
# compatible"), whose --files-from does NOT recurse into directories named in
# the list even with -a. It silently created the tree and copied one 602-byte
# file. Passing the paths as arguments with -R preserves the same layout and
# recurses properly on both openrsync and real rsync.
rsync -avzR --progress \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  "${PATHS[@]}" "${DELTA_USER}@${DELTA_HOST}:${DELTA_PATH}/"

echo
echo "전송 완료.  다음:"
echo "  ssh ${DELTA_USER}@${DELTA_HOST}"
echo "  cd ${DELTA_PATH} && bash deploy/delta/setup.sh"
echo "  sbatch --account=\$(accounts | awk '/cpu/{print \$1; exit}') deploy/delta/build.sbatch"
