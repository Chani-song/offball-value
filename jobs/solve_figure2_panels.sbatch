#!/bin/bash
#SBATCH --job-name=figure2-panels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/figure2_panels_%j.out
# Figure 2's games: the four 2v1 showcase scenes at their real 0.0 / 0.6 / 1.2 s moments, each moment solved as
# its own game with the abstract's solver -- every pass along the run its own column, the A-sym pass model, the
# controlled defender committed 0.2 s to his command when a pass is played, the other defenders able to tackle
# the ball carrier (their real tracks), measured 99.9th-percentile movement limits, no reaction delay, compass
# commands, xT threat, 3 decisions of 0.6 s -- then every moment read into a figure panel.
#
# Submit from the repository root (about 30 CPU-minutes per game, 12 games):
#   sbatch --account=<your account> jobs/solve_figure2_panels.sbatch
# Environment: PYTHON (default: python), OUT (default: out/runs/multipass_full_2v1).
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="${PYTHON:-python}"
P="$ROOT/data/processed/showcase_v1"
OUT="${OUT:-$ROOT/out/runs/multipass_full_2v1}"
cd "$ROOT"; mkdir -p logs
PYTHONPATH="$ROOT/andrew-passer2on1" "$PY" andrew-passer2on1/scripts/run.py --states "$P/figure_2v1/states_2v1_all.json" \
  --model "$ROOT/data/processed/pass_models_sym/Asym_crossfit.json" --allow-proxy-labels --steps 3 --step-seconds 0.6 \
  --commands compass --threat andrew --passes run --physics-file "$ROOT/data/processed/physics_limits/agile_p999_nodelay.json" \
  --multi-pass --pass-reaction-s 0.2 --background-tackles --output "$OUT" --workers "${SLURM_CPUS_PER_TASK:-12}"
PAN="$P/figure_multi/panels"; mkdir -p "$PAN"
for scene in S05 S13 S34 S53; do
  for dt in 0.0 0.6 1.2; do
    code=$scene; [ $dt != 0.0 ] && code=$scene@$dt
    PYTHONPATH="$ROOT/src" "$PY" scripts/extract_panel_policy.py --solved "$OUT" --code "$code" \
      --label "$scene-full" --output "$PAN/$scene-full_$dt.json" || echo "FAILED extract $code"
  done
done
echo "== DONE $(date)"
