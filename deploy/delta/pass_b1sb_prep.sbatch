#!/bin/bash
#SBATCH --job-name=pass-b1sb
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pass_b1sb_%j.out
# B1-SB: 준현's velocity logistic fitted on StatsBomb 360 men's club passes
# (velocities from the previous freeze frame, mirrored when it is the other team's;
# intended-at-kick targets from the pass direction only), then scored on our
# Bundesliga passes, which it never sees. Needs passes.jsonl from the prep job.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
PY="$ROOT/.venv-delta/bin/python"
M="$ROOT/data/processed/pass_models"
cd "$ROOT"; mkdir -p logs
"$PY" scripts/build_statsbomb_pass_dataset.py --output "$M/statsbomb_passes.jsonl" --workers 16
PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/andrew-passer2on1" "$PY" scripts/fit_pass_b1sb.py \
  --statsbomb "$M/statsbomb_passes.jsonl" --ours "$M/passes.jsonl" --output "$M"
PYTHONPATH="$ROOT/andrew-passer2on1:$ROOT/src" "$PY" andrew-passer2on1/tests/test_pass_b1sb.py "$M"
echo DONE
