#!/usr/bin/env bash
#SBATCH --job-name=offball-missed
#SBATCH --cpus-per-task=4
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/missed_onsets_%j.out
#SBATCH --error=logs/missed_onsets_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src
python scripts/diagnose_missed_onsets2.py --effects strong medium
