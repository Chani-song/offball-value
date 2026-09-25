#!/bin/bash
#SBATCH --job-name=s3-passer2on1-ag
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s3_passer2on1_agile_%j.out
# 2대1, 3 x 0.6 s, AGILE physics (agile_motion.AGILE: 4.5 up / 6.0 brake / 6.0 turn,
# plant-and-cut 9.0, defender 0.2 s delay at onset; top speed 9.0, 2v1 carrier 7.2).
# Inputs: run speed >= 3.8 m/s and nobody inside the triangle at 0 / 0.6 / 1.2 / 1.8 s -> 76 states.
# Tests run first; any failure stops the job before solving.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_passer2on1_agile
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1:$ROOT/andrew-fixedpasser:$ROOT/src" "$PY" andrew-passer2on1/tests/test_agile_motion.py
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/tests/test_equivalence.py
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py \
  --states "$ROOT/data/processed/stage3/passer2on1_states_agile06_final.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --physics agile --workers 16
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/passer2on1_results_agile.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/passer2on1_results_agile.csv" --solved "$OUT" \
  --title "2대1 · 솔버의 답 (0.6초 × 3턴 · 예전 가속)" --output "$ROOT/out/passer2on1_agile/index.html"
echo DONE
