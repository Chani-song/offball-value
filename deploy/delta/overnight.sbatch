#!/usr/bin/env bash
#SBATCH --job-name=offball-night
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/night_%A_%a.out
#SBATCH --error=logs/night_%A_%a.out
#
# 자리를 비운 사이 돌릴 네 가지. 전부 읽기 전용 측정이고, 서로 독립이다.
#
# 1 vacated_xt   수비수가 비워둔 공간의 PAUSA 합으로 삼중항 정렬.
#                (첫 시도는 runner_response_path_txy 키가 없어 전부 NaN이었다.
#                 빌드 출력에서 러너 추격 경로는 responses의 focus_runner다.)
# 2 possession   phase가 끊긴 직후 같은 팀이 계속 공을 만지는 빈도, 그리고
#                phase 사이 공백에 들어간 슛 수. 찬의님 strong 한 장면(J03WOH
#                shot_010)이 21초 공백에 통째로 들어가 사라졌다.
# 3 arms         hybrid vs xpass360, 587장면 전체 일치도.
# 4 triples      삼중항 순위표 재생성(정렬 기준 비교의 입력).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src

case "${SLURM_ARRAY_TASK_ID}" in
  1) python scripts/rank_by_vacated_xt.py ;;
  2) python scripts/audit_possession_cuts.py ;;
  3) python scripts/compare_full_arms.py \
       --arm hybrid   /work/hdd/bbmr/kseo1/offball-out/v5_hybrid \
       --arm xpass360 /work/hdd/bbmr/kseo1/offball-out/v5_xpass360 ;;
  4) python scripts/rank_triples.py ;;
esac
