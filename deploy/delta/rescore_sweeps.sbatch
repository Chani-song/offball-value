#!/usr/bin/env bash
#SBATCH --job-name=offball-rescore
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/rescore_%j.out
#SBATCH --error=logs/rescore_%j.out
#
# Last night's 84 extractions all succeeded; only the scoring was wrong. The
# throwaway scorer looked for a `runner_name` column that does not exist and
# treated `offball_attackers` as names when they are shirt numbers, so every
# arm printed the same "열을 못 찾음".
#
# score_against_chani2.py is the scorer that produced chani_runner_score.csv
# and already handles both: shirt -> player_id through the match metadata,
# and a shot matched only to a phase owned by the SHOOTING team, ending at or
# just before the shot. Its own docstring records why the naive window was
# wrong. Using it instead of a second implementation means the sweep numbers
# are comparable to the ones already on record.

set -uo pipefail   # -e 제외: arm 하나가 실패해도 나머지는 채점한다
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

score_arm () {
  local DIR="$1" LABEL="$2"
  [ -d "$DIR" ] || { echo "  ($LABEL 없음)"; return 0; }
  python scripts/merge_sweep_arm.py "$DIR" >/dev/null 2>&1 || {
    echo "  ($LABEL 병합 실패)"; return 0; }
  echo
  echo "----- $LABEL -----"
  local OUT="logs/_rescore_${LABEL}.txt"
  python scripts/score_against_chani2.py \
    --root "$DIR" \
    --output "data/processed/sweep_scores/chani_runner_${LABEL}.csv" \
    >"$OUT" 2>&1 || echo "  ($LABEL 채점 실패)"
  sed -n '/① 슛이 속한/,/^$/p;/②③/,/^$/p' "$OUT" | head -30
  grep -E '창|strong|s\+m' "$OUT" | head -14
}

mkdir -p data/processed/sweep_scores

echo "##############################################################"
echo "# 대조군 — v0_5 현행"
echo "##############################################################"
python scripts/score_against_chani2.py \
  --root data/processed/run_onset_v0_5/dir25 \
  --output data/processed/sweep_scores/chani_runner_v05.csv \
  >logs/_rescore_v05.txt 2>&1 || echo "(대조군 실패)"
grep -E '창|strong|s\+m' logs/_rescore_v05.txt | head -14

echo
echo "##############################################################"
echo "# 1단계 온셋 게이트 (하나씩) + 조합"
echo "##############################################################"
for CFG in base ctrl30 ctrl50 speed08 dir15 disp15 combo; do
  score_arm "data/processed/gate_sweep/$CFG" "gate_$CFG"
done

echo
echo "##############################################################"
echo "# 0단계 possession 분할"
echo "##############################################################"
for CFG in base conf60 conf100 conf200 pctrl30; do
  score_arm "data/processed/phase_sweep/$CFG" "phase_$CFG"
done
