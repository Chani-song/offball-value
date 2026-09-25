#!/usr/bin/env bash
# Run this ON THE LAPTOP, once, if the project is moving to Delta for good.
#
# push.sh sends only what a payoff build reads (49 MB). This sends the things
# you need to do NEW work: extract scenes, retrain a delivery model, or
# compare against the hybrid baseline.
#
# Split by provenance, because it decides how the bytes should travel:
#
#   PROPRIETARY -> must be uploaded from here.
#     data/raw/bundesliga-integrated   2.4 GB   DFL/IDSSE tracking, licensed
#     data/processed/goalside_v1_*     685 MB   the hybrid baseline audits;
#                                               regenerating them costs ~6 h
#
#   PUBLIC -> do NOT upload. Fetch on Delta, where the network is far faster
#   than a home uplink:
#     python scripts/download_statsbomb.py     12 GB  (git clone, xPass training)
#     python scripts/download_skillcorner.py  925 MB
#     python scripts/download_metrica.py      175 MB
set -euo pipefail

DELTA_USER="${DELTA_USER:-kseo1}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DELTA_PATH="${DELTA_PATH:-/scratch/bbmr/kseo1/offball-value}"

cd "$(dirname "$0")/../.."

PATHS=(
  data/raw/bundesliga-integrated
  data/processed/goalside_v1_chunk0
  data/processed/goalside_v1_chunk1
  data/processed/goalside_v1_chunk2
)

echo "== 업로드 목록 (독점 데이터만) =="
du -sk "${PATHS[@]}" \
  | awk '{s+=$1; printf "%8.2f GB  %s\n", $1/1048576, $2} END {printf "\n합계 %.2f GB\n\n", s/1048576}'

echo "공개 데이터(StatsBomb 12 GB 등)는 업로드하지 않습니다."
echo "Delta에서 'python scripts/download_statsbomb.py' 로 직접 받는 쪽이 훨씬 빠릅니다."
echo
read -r -p "계속할까요? [y/N] " reply
[ "$reply" = "y" ] || { echo "취소"; exit 0; }

# -R for the same reason as push.sh: macOS openrsync's --files-from does not
# recurse. --partial so an interrupted multi-GB transfer resumes.
rsync -avzR --partial --progress \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  "${PATHS[@]}" "${DELTA_USER}@${DELTA_HOST}:${DELTA_PATH}/"

echo
echo "완료. Delta에서 공개 데이터 받기:"
echo "  cd ${DELTA_PATH}"
echo "  . .venv-delta/bin/activate"
echo "  python scripts/download_statsbomb.py    # 12 GB, git clone"
