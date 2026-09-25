#!/usr/bin/env bash
#SBATCH --job-name=offball-vacated-reach
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/vacreach_%j.out
#SBATCH --error=logs/vacreach_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH=src
python scripts/measure_vacated_space_reachability.py \
  --audits data/processed/goalside_v1_chunk0 \
           data/processed/goalside_v1_chunk1 \
           data/processed/goalside_v1_chunk2 \
  --output out/delivery_analysis/vacated_reachability.json
echo "=== done ==="
