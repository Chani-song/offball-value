#!/bin/bash
#SBATCH --job-name=pass-compare
#SBATCH --account=bidp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pass_compare_%j.out
# Same scenes, three pass models (shipped positions-only, A, B1): torn scenes,
# defender retreat on real goalward runs, where the line ends. Runs after the
# four solver jobs whatever their outcome; a missing study prints "결과 없음".
set -uo pipefail
ROOT=/scratch/bbmr/kseo1/offball-value
cd "$ROOT"
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$ROOT/.venv-delta/bin/python" scripts/compare_pass_models.py
cat "$ROOT/data/processed/pass_models/report.json" | head -c 4000
echo DONE
