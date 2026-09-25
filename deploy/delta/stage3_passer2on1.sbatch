#!/bin/bash
#SBATCH --job-name=s3-p2on1
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/s3_passer2on1_%j.out

# Stage 3, 2v1 (beneficiary = carrier), 167 triples, WITH background defenders.
#
# Game: andrew-passer2on1 (Background2on1Game). The collaborator's andrew/ is
# imported unmodified. Differences from stage3_carrier_3s (job 22331772) are
# only in the terminal payoffs: every other defender is present on his real
# track (no decisions), the pass model sees them all, offside uses the full
# second-last-defender rule, threat's room is to the nearest defender.
# Verified bit-for-bit equal to the original when the background is empty
# (andrew-passer2on1/tests/test_equivalence.py).
#
# Trial on 9 scenes: ~97 CPU s each -> 167 x 97 / 8 workers ~ 35 min, ~4.5 SU.
# Then summarise and render the review page, so results exist without a session.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_passer2on1_3s
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"
mkdir -p logs

PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py \
  --states "$ROOT/data/processed/stage3/passer2on1_states.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 1.0 --workers 8

PYTHONPATH="$ROOT/src" "$PY" scripts/summarise_stage3.py \
  --solved "$OUT" --output "$ROOT/data/processed/stage3/passer2on1_results_3s.csv"

cd "$ROOT/scripts"
PYTHONPATH="$ROOT/src" "$PY" render_dilemma_review.py \
  --results "$ROOT/data/processed/stage3/passer2on1_results_3s.csv" \
  --solved "$OUT" \
  --compare "$ROOT/data/processed/stage3/carrier_results_3s.csv" \
  --output "$ROOT/out/passer2on1/index.html"
echo "DONE: page at out/passer2on1/index.html"
