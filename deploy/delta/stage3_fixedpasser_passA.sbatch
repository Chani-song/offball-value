#!/bin/bash
#SBATCH --job-name=s3-fixed-pA
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/s3_fixedpasser_passA_%j.out
# 3대1, agile physics, 3 x 0.6 s, triangle-2 m scenes, pass model candidate A (A 물리 경주).
# Cross-fitted: a scene from match M is priced by the model fitted without M's passes
# (data/processed/pass_models/A_crossfit.json, written by the prep job).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_fixedpasser_agile_passA
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-fixedpasser" "$PY" andrew-fixedpasser/scripts/run.py \
  --states "$ROOT/data/processed/stage3/fixedpasser_states_agile06_final2m.json" \
  --model "$ROOT/data/processed/pass_models/A_crossfit.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --physics agile --workers 32
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/fixedpasser_results_passA.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/fixedpasser_results_passA.csv" --solved "$OUT" \
  --title "3대1 · 솔버의 답 (예전 가속 · 삼각형 2m · 패스 모델 A: A 물리 경주)" --output "$ROOT/out/fixedpasser_passA/index.html"
echo DONE
