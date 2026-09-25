#!/bin/bash
#SBATCH --job-name=s3-fp-18
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s3_fixedpasser_18s_%j.out

# 3v1 fixed passer (passer on his real track) with background defenders, 3 x 0.6 s = 1.8 s.
# Inputs: run speed >= 3.8 m/s AND nobody else inside the carrier-runner-defender
# triangle (beneficiary-defender-runner) at 0 / 0.6 / 1.2 s -> 335 states.
# Game checked for shared pricing with 2v1, receiver symmetry, monotonicity
# and exactness (andrew-fixedpasser/tests/test_game.py).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_fixedpasser_18s
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-fixedpasser" "$PY" andrew-fixedpasser/scripts/run.py \
  --states "$ROOT/data/processed/stage3/fixedpasser_states_06_final.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --workers 32
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/fixedpasser_results_18s.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/fixedpasser_results_18s.csv" --solved "$OUT" \
  --title "3대1 · 솔버의 답 (0.6초 × 3턴)" --output "$ROOT/out/fixedpasser_18s/index.html"
echo DONE
