#!/bin/bash
#SBATCH --job-name=eval-v1
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --output=logs/eval_v1_%x_%j.out
# 2026-09-29: the abstract's evaluation set, solved with the full solver of the figures (every pass along the
# run its own column, A-sym, defender committed 0.2 s, background tackles in 2v1, P99.9 no delay, 1.8 s horizon
# with the imported best-pass terminal). Each state is one real moment (0 / 0.6 / 1.2 s) of a scene: the eight
# showcase scenes + the pipeline scenes S46-S75 not marked "not an off-ball run" (data/processed/eval_v1). The
# states now also carry the opening payoff table (solve.root_game) for the static-vs-responsive numbers; every
# moment is then read into a panel (options' ends and paths) for the observed-action matching.
# Usage: sbatch --cpus-per-task=37 --time=01:30:00 -J eval-2v1 deploy/delta/eval_v1.sbatch 2v1
#        sbatch --cpus-per-task=32 --time=02:00:00 -J eval-3v1 deploy/delta/eval_v1.sbatch 3v1
set -uo pipefail
KIND=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out/eval_v1_$KIND
E=$ROOT/data/processed/eval_v1
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs "$E/panels"
if [ "$KIND" = 2v1 ]; then PKG=andrew-passer2on1; EXTRA=--background-tackles; else PKG=andrew-fixedpasser; EXTRA=; fi
PYTHONPATH="$ROOT/$PKG" "$PY" $PKG/scripts/run.py --states "$E/states_${KIND}_eval.json" \
  --model "$ROOT/data/processed/pass_models_sym/Asym_crossfit.json" --allow-proxy-labels --steps 3 --step-seconds 0.6 \
  --commands compass --threat andrew --passes run --physics-file "$ROOT/data/processed/physics_limits/agile_p999_nodelay.json" \
  --multi-pass --pass-reaction-s 0.2 $EXTRA --workers "$SLURM_CPUS_PER_TASK" --output "$OUT" > logs/eval_v1_$KIND.log 2>&1
grep -h "state \|FAILED" logs/eval_v1_$KIND.log
codes=$("$PY" -c "import json; print(' '.join(r['provenance']['code'] for r in json.load(open('$E/states_${KIND}_eval.json'))['states']))")
for code in $codes; do
  PYTHONPATH="$ROOT/src" "$PY" scripts/extract_panel_policy.py --solved "$OUT" --code "$code" --label "eval" \
    --output "$E/panels/eval-$code.json" > /dev/null || echo "FAILED extract $code" &
done
wait
echo "panels: $(ls $E/panels | wc -l)"
echo "== DONE $(date)"
