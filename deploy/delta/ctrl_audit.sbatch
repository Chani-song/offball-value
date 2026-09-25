#!/usr/bin/env bash
#SBATCH --job-name=offball-ctrl
#SBATCH --cpus-per-task=4
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ctrl_audit_%j.out
#SBATCH --error=logs/ctrl_audit_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src
python scripts/audit_control_distance_gate.py
