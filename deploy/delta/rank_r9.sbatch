#!/usr/bin/env bash
#SBATCH --job-name=offball-r9-rank
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/r9rank_%j.out
#SBATCH --error=logs/r9rank_%j.out
#
# Runs once the v7_r9_ssac array finishes. Ranks its triples on the same four
# criteria as the native build, then answers the only question that matters
# for adopting the stack: does the ORDER change? Levels are incomparable by
# construction -- the two stacks put Q on different scales on purpose.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src

echo "=========================================================="
echo "빌드된 장면 수"
echo "=========================================================="
ls /work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac 2>/dev/null | wc -l

echo
echo "=========================================================="
echo "ssac 삼중항 순위표"
echo "=========================================================="
python scripts/rank_triples.py \
  --build /work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac \
  --output data/processed/triple_ranking_r9.csv

echo
echo "=========================================================="
echo "vacated_xt 열 추가 (개요가 요구함)"
echo "=========================================================="
python scripts/rank_by_vacated_xt.py \
  --build /work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac \
  --ranking data/processed/triple_ranking_r9.csv \
  --output data/processed/triple_ranking_r9_xt.csv

echo
echo "=========================================================="
echo "ssac 스택으로 본 검토 대상 40장면"
echo "=========================================================="
python scripts/render_triple_overview.py \
  --ranking data/processed/triple_ranking_r9_xt.csv \
  --build /work/hdd/bbmr/kseo1/offball-out/v7_r9_ssac \
  --output out/triple_overview_r9/index.html \
  --top-n 20 || echo "개요 생성 실패 (vacated_xt 열이 없으면 예상된 결과)"
