#!/usr/bin/env bash
#SBATCH --job-name=offball-analyse
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/analyse_%j.out
#SBATCH --error=logs/analyse_%j.out
#
# The cheap half of tonight: everything answerable from builds that already
# exist, so none of it waits on the gate sweep.
#
#  1 beneficiary accuracy on the labels, for all three value stacks, so
#    "adopt the stack" finally has an accuracy number beside the rank
#    correlations rather than only "it changed things".
#  2 the same for the control-distance sweep already on disk: does loosening
#    the carrier gate recover Chani's named runners, or only add scenes?
#  3 where the strong five actually fail.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

echo "##################################################################"
echo "# 1. 수혜자 정확도 — 가치 스택 3종 (라벨 47개 / 장면 34개)"
echo "##################################################################"
for BUILD in v5_hybrid v6_ssac v7_r9_ssac; do
  echo
  echo "===== $BUILD ====="
  python scripts/score_coupled_beneficiary.py \
    /work/hdd/bbmr/kseo1/offball-out/$BUILD/scene_* 2>&1 | head -8 \
    || echo "  (채점 실패)"
done

echo
echo "##################################################################"
echo "# 2. carrier 게이트를 풀면 찬의님 러너가 살아나는가"
echo "##################################################################"
for CFG in ctrl15 ctrl30 ctrl50; do
  python scripts/score_chani_runner_recall.py \
    data/processed/sweep_ctrl/$CFG/* --label "$CFG" 2>&1 | tail -8 \
    || echo "  ($CFG 실패)"
done

echo
echo "##################################################################"
echo "# 3. strong 5개가 어디서 깨지나"
echo "##################################################################"
python - <<'PY'
import pandas as pd
t = pd.read_csv("data/processed/chani_triple_score.csv")
r = pd.read_csv("data/processed/chani_runner_score.csv")
m = t.merge(r[["clip_id", "gap_s", "phase_seconds", "our_runners"]],
            on="clip_id", how="left")
for effect in ("strong", "medium"):
    rows = m[m["effect"] == effect]
    if rows.empty:
        continue
    print(f"\n[{effect}]  {len(rows)}개 · runner_hit "
          f"{int(rows['runner_match'].sum())}")
    cols = ["clip_id", "our_scenes_in_phase", "our_runners", "gap_s",
            "phase_seconds", "stage"]
    print(rows[cols].to_string(index=False, max_colwidth=26))
PY
