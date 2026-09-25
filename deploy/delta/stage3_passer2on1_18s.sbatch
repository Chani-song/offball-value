#!/bin/bash
#SBATCH --job-name=s3-p2on1-18
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s3_passer2on1_18s_%j.out

# 2v1 (beneficiary = carrier) with background defenders, 3 turns x 0.6 s = 1.8 s.
# Inputs: run speed >= 3.8 m/s AND nobody else inside the carrier-runner-defender
# triangle at 0 / 0.6 / 1.2 s (scripts/filter_stage3_states.py) -> 80 states.
# Game code verified bit-for-bit against the original with no background
# (andrew-passer2on1/tests/test_equivalence.py).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_passer2on1_18s
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py \
  --states "$ROOT/data/processed/stage3/passer2on1_states_06_final.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --workers 16
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/passer2on1_results_18s.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/passer2on1_results_18s.csv" --solved "$OUT" \
  --title "2대1 · 솔버의 답 (0.6초 × 3턴)" --output "$ROOT/out/passer2on1_18s/index.html"
echo DONE
