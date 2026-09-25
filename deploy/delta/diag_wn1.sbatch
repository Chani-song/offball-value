#!/usr/bin/env bash
#SBATCH --job-name=offball-diag-wn1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/diag_wn1_%j.out
#SBATCH --error=logs/diag_wn1_%j.out
#
# J03WN1은 101 phases에서 contextual onset이 2개만 나왔다. 다른 경기는 같은
# phase 수에서 52~85개가 나온다. phase 통계(지속시간/공격진영 비율/점유)와
# 트래킹(21명, 퇴장 반영)은 모두 정상이므로 검출 단계에서 무언가 떨어진다.
# detect_run_onsets가 몇 개를 내놓는지, _nearest_phase가 몇 개를 버리는지 가른다.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export PYTHONPATH=src
python scripts/diagnose_wn1_onsets.py
