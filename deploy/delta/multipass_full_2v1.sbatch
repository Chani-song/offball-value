#!/bin/bash
#SBATCH --job-name=multipass-full
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/multipass_full_2v1_%j.out
# 2026-09-29: the four 2v1 showcase scenes at the real 0/0.6/1.2 s with everything of the day: every pass
# along the run its own column (18 run passes), A-sym (arrival race + lane), the controlled defender
# committed 0.2 s to his command when a pass is played, and the background defenders tackling the carrier
# like the controlled one (NPCs: no decisions, their real tracks). Andrew's xT, compass moves, 99.9th
# percentile accelerations, no delay.
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/scratch/bbmr/kseo1/offball-value
OUT=/work/hdd/bbmr/kseo1/offball-out
P=$ROOT/data/processed/showcase_v1
PY="$ROOT/.venv-delta/bin/python"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py --states "$P/figure_2v1/states_2v1_all.json" \
  --model "$ROOT/data/processed/pass_models_sym/Asym_crossfit.json" --allow-proxy-labels --steps 3 --step-seconds 0.6 \
  --commands compass --threat andrew --passes run --physics-file "$ROOT/data/processed/physics_limits/agile_p999_nodelay.json" \
  --multi-pass --pass-reaction-s 0.2 --background-tackles --output "$OUT/multipass_full_2v1" --workers 12 > logs/multipass_full_2v1.log 2>&1
grep -h "state \|FAILED" logs/multipass_full_2v1.log
PAN=$P/figure_multi/panels; FIG=$ROOT/out/showcase_v1/figure_multi; mkdir -p "$PAN" "$FIG"
for scene in S05 S13 S34 S53; do
  for dt in 0.0 0.6 1.2; do
    code=$scene; [ $dt != 0.0 ] && code=$scene@$dt
    PYTHONPATH="$ROOT/src" "$PY" scripts/extract_panel_policy.py --solved "$OUT/multipass_full_2v1" --code "$code" \
      --label "$scene-full" --output "$PAN/$scene-full_$dt.json"
  done
  "$PY" scripts/render_panel_figure.py --panels "$PAN" --label "$scene-full" --tracking "$P/tracking/$scene.csv" \
    --title "$scene — every pass along the run, A-sym race, defender committed 0.2 s, other defenders can tackle" \
    --output "$FIG/${scene}_full.png" || echo "FAILED $scene"
done
ls "$FIG"
echo "== DONE $(date)"
