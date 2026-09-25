#!/bin/bash
#SBATCH --job-name=s3-passe-pB1
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s3_passer2on1_passB1_%j.out
# 2대1, agile physics, 3 x 0.6 s, triangle-2 m scenes, pass model candidate B1 (B1 속도 로지스틱).
# Cross-fitted: a scene from match M is priced by the model fitted without M's passes
# (data/processed/pass_models/B1_crossfit.json, written by the prep job).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_passer2on1_agile_passB1
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py \
  --states "$ROOT/data/processed/stage3/passer2on1_states_agile06_final2m.json" \
  --model "$ROOT/data/processed/pass_models/B1_crossfit.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --physics agile --workers 16
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/passer2on1_results_passB1.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/passer2on1_results_passB1.csv" --solved "$OUT" \
  --title "2대1 · 솔버의 답 (예전 가속 · 삼각형 2m · 패스 모델 B1: B1 속도 로지스틱)" --output "$ROOT/out/passer2on1_passB1/index.html"
echo DONE
