#!/usr/bin/env bash
#SBATCH --job-name=offball-pcount
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/pcount_%j.out
#SBATCH --error=logs/pcount_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src
python scripts/measure_player_count_gate.py
