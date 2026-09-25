#!/bin/bash
#SBATCH --job-name=s3-fixed-pSB
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s3_fixedpasser_passB1SB_%j.out
# 3대1, agile physics, 3 x 0.6 s, triangle-2 m scenes, pass model B1-SB (준현님 속도 로지스틱, StatsBomb 360 학습).
# Fitted on StatsBomb 360 club passes only, so no scene is priced by a model that saw its match.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_fixedpasser_agile_passB1SB
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-fixedpasser" "$PY" andrew-fixedpasser/scripts/run.py \
  --states "$ROOT/data/processed/stage3/fixedpasser_states_agile06_final2m.json" \
  --model "$ROOT/data/processed/pass_models/B1SB_all.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 0.6 --physics agile --workers 32
PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py --solved "$OUT" \
  --output "$ROOT/data/processed/stage3/fixedpasser_results_passB1SB.csv"
cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_solver_story.py \
  --results "$ROOT/data/processed/stage3/fixedpasser_results_passB1SB.csv" --solved "$OUT" \
  --title "3대1 · 솔버의 답 (예전 가속 · 삼각형 2m · 패스 모델 B1-SB: 속도 로지스틱·StatsBomb)" --output "$ROOT/out/fixedpasser_passB1SB/index.html"
echo DONE
