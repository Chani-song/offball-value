#!/usr/bin/env bash
#SBATCH --job-name=offball-rankxt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/rankxt_%j.out
#SBATCH --error=logs/rankxt_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src
python scripts/rank_by_vacated_xt.py
