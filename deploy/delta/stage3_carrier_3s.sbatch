#!/bin/bash
#SBATCH --job-name=s3-carrier-3s
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/s3_carrier_3s_%j.out

# Stage 3 on the 167 carrier-beneficiary triples, over the WHOLE scene.
#
# The first run (job 22316190) used exact_study as shipped, which fixes the
# game at 3 turns of 0.4 s = 1.2 s. Our scenes are the 2.96 s after onset, so
# it solved the first 40% of each run. This run uses 3 turns of 1.0 s = 3.0 s
# through scripts/run_stage3.py, which calls the same solve_one with its own
# GameConfig and leaves the imported code untouched.
#
# Same number of states (5^(3k) depends on turn count only); measured 33 CPU s
# per scene on two scenes, against 31 at 0.4 s. What changes is resolution:
# the defender re-decides at 0, 1 and 2 s instead of every 0.4 s.
#
# Inputs: gate kept AND runner_is_protagonist AND beneficiary == carrier,
# carrier speed limit 7.2 m/s (see build_stage3_states.py).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/stage3_carrier_3s
cd "$ROOT"
mkdir -p logs

PYTHONPATH="$ROOT/src" "$ROOT/.venv-delta/bin/python" scripts/run_stage3.py \
  --states "$ROOT/data/processed/stage3/carrier_beneficiary_states_v2.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" --allow-proxy-labels \
  --output "$OUT" --steps 3 --step-seconds 1.0 --workers 8
