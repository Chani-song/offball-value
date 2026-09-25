#!/bin/bash
#SBATCH --job-name=s3-carrier
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/s3_carrier_%j.out

# Stage 3 on the 226 triples whose beneficiary is the ball carrier.
#
# Those are the ones the imported solver already expresses: the defender is
# caught between the ball and the run, which is exactly its carrier-receiver
# game. Nothing about the solver is modified here -- this is his model, his
# defaults, our scenes. The 786 triples whose beneficiary is a third attacker
# wait for the two-receiver question.
#
# The carrier is given the solver's own 7.2 m/s default rather than
# scenarios.py's uncommented 5.6, which on real tracking would have dropped
# the counter-attacks as a class. The reasoning rides in the states file
# under "limits"; see scripts/build_stage3_states.py.
#
# Eight workers is the solver's own ceiling (exact_study rejects more), so
# the core request matches it. Single-thread the BLAS on top: the solver
# parallelises across starting states and nested threading oversubscribes.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=/scratch/bbmr/kseo1/offball-value
cd "$ROOT"
mkdir -p logs /work/hdd/bbmr/kseo1/offball-out/stage3_carrier

"$ROOT/.venv-delta/bin/python" -m defensive_positioning.exact_study \
  --states "$ROOT/data/processed/stage3/carrier_beneficiary_states.json" \
  --model "$ROOT/andrew/models/experimental_pass.json" \
  --allow-proxy-labels \
  --output /work/hdd/bbmr/kseo1/offball-out/stage3_carrier \
  --workers 8
