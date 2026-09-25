#!/bin/bash
#SBATCH --job-name=s3-fixedpasser-ag
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s3_fixedpasser_agile_%j.out
# 3대1, 3 x 0.6 s, AGILE physics (agile_motion.AGILE: 4.5 up / 6.0 brake / 6.0 turn,
# plant-and-cut 9.0, defender 0.2 s delay at onset; top speed 9.0, 2v1 carrier 7.2).
# Inputs: run speed >= 3.8 m/s and nobody inside the triangle at 0 / 0.6 / 1.2 / 1.8 s -> 332 states.
# Tests run first; any failure stops the job before solving.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_fixedpasser_agile
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1:$ROOT/andrew-fixedpasser:$ROOT/src" "$PY" andrew-passer2on1/tests/test_agile_motion.py
:
PYTHONPATH="$ROOT/andrew-fixedpasser" "$PY" andrew-fixedpasser/scripts/run.py \
  --states "$ROOT/data/processed/stage3/fixedpasser_states_agile06_final.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --physics agile --workers 32
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/fixedpasser_results_agile.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/fixedpasser_results_agile.csv" --solved "$OUT" \
  --title "3대1 · 솔버의 답 (0.6초 × 3턴 · 예전 가속)" --output "$ROOT/out/fixedpasser_agile/index.html"
echo DONE
