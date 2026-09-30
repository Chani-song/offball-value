#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --output=logs/evaluation_%x_%j.out
# The abstract's evaluation set: every real 0.0 / 0.6 / 1.2 s moment of the evaluation scenes (the eight showcase
# scenes and the annotated pipeline scenes; scripts/build_eval_states.py) solved as its own game with the same
# solver as Figure 2 (background tackles in 2v1 only: in 3v1 the passer is scripted and has no dribble). The
# solver also stores each game's opening payoff table for scripts/analyze_eval.py. Every moment is then read
# into a panel (options' ends and paths) for matching the observed moves.
#
# Submit from the repository root:
#   sbatch --account=<your account> --cpus-per-task=37 --time=01:30:00 -J eval-2v1 jobs/solve_evaluation.sbatch 2v1
#   sbatch --account=<your account> --cpus-per-task=32 --time=02:00:00 -J eval-3v1 jobs/solve_evaluation.sbatch 3v1
# Environment: PYTHON (default: python), OUT (default: out/runs/eval_v1_<kind>).
set -uo pipefail
KIND=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${PYTHON:-python}"
OUT="${OUT:-$ROOT/out/runs/eval_v1_$KIND}"
E="$ROOT/data/processed/eval_v1"
cd "$ROOT"; mkdir -p logs "$E/panels"
if [ "$KIND" = 2v1 ]; then PKG=andrew-passer2on1; EXTRA=--background-tackles; else PKG=andrew-fixedpasser; EXTRA=; fi
PYTHONPATH="$ROOT/$PKG" "$PY" $PKG/scripts/run.py --states "$E/states_${KIND}_eval.json" \
  --model "$ROOT/data/processed/pass_models_sym/Asym_crossfit.json" --allow-proxy-labels --steps 3 --step-seconds 0.6 \
  --commands compass --threat andrew --passes run --physics-file "$ROOT/data/processed/physics_limits/agile_p999_nodelay.json" \
  --multi-pass --pass-reaction-s 0.2 $EXTRA --workers "${SLURM_CPUS_PER_TASK:-8}" --output "$OUT"
codes=$("$PY" -c "import json; print(' '.join(r['provenance']['code'] for r in json.load(open('$E/states_${KIND}_eval.json'))['states']))")
for code in $codes; do
  PYTHONPATH="$ROOT/src" "$PY" scripts/extract_panel_policy.py --solved "$OUT" --code "$code" --label "eval" \
    --output "$E/panels/eval-$code.json" > /dev/null || echo "FAILED extract $code" &
done
wait
echo "panels: $(ls "$E/panels" | wc -l)"
echo "== DONE $(date)"
