#!/usr/bin/env bash
#SBATCH --job-name=offball-cuts
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/cuts_%j.out
#SBATCH --error=logs/cuts_%j.out
#
# The phase sweep reported how long phases got, which is a side effect, not
# the target. The target is the 129 shots that fall into the gaps between
# phases -- one of Chani's strong five (shot_010, its shot 23.7 s outside any
# phase) dies exactly there.
#
# audit_possession_cuts.py measures that directly: how often a phase ends and
# the same team touches the ball again within 5 s, and how many shots land in
# the gaps. Run per arm, it says whether raising the confirmation time
# actually closes the holes or merely lengthens what was already contiguous.
#
# Read it against the length distribution already measured:
#   base    720 phases  median 15.0s  131 over 30s
#   conf200 692         median 19.4s  189 over 30s   <- merging, but bloating
#   pctrl30 854         median 13.2s  106 over 30s   <- more and shorter
# pctrl30 moving the opposite way is the interesting one: it is the arm that
# does NOT trade gap-closing for 47-second phases.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

mkdir -p data/processed/cut_audits

for CFG in base conf60 conf100 conf200 pctrl30; do
  D="data/processed/phase_sweep/$CFG"
  [ -d "$D" ] || { echo "($CFG 없음)"; continue; }
  [ -f "$D/attacking_phases.csv" ] || python scripts/merge_sweep_arm.py "$D" >/dev/null 2>&1
  echo
  echo "############################################################"
  echo "# $CFG"
  echo "############################################################"
  python scripts/audit_possession_cuts.py \
    --root "$D" \
    --output "data/processed/cut_audits/cuts_${CFG}.csv" 2>&1 | tail -22 \
    || echo "  ($CFG 실패)"
done
