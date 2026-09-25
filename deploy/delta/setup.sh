#!/usr/bin/env bash
# Run this ON A DELTA LOGIN NODE. Building a venv is light enough for a login
# node; the BUILD ITSELF IS NOT — that goes through sbatch. See build.sbatch.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Delta's login shell already starts in conda `base`, which is python enough
# to build a venv from. If that ever stops being true, load a module instead:
#   module avail python
if ! command -v python3 >/dev/null; then
  module load python || module load anaconda3_cpu
fi

echo "== 인터프리터 =="
python3 -V
case "$(python3 -V 2>&1)" in
  *" 3.1"[1-9]*) ;;
  *) echo "경고: pyproject는 >=3.11을 요구합니다. numpy 2.4.4가 안 깔릴 수 있습니다." ;;
esac

# sbatch fails immediately if the --output directory does not exist.
mkdir -p logs out

python3 -m venv .venv-delta
. .venv-delta/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r deploy/delta/requirements.txt

echo
echo "== 검증: 모델이 같은 버전으로 로드되는가 =="
PYTHONPATH=src python - <<'PY'
import sklearn, numpy
from offball_value.xpass import load_xpass_model, predict_pass_success_360
print("sklearn", sklearn.__version__, "· numpy", numpy.__version__)
m = load_xpass_model("data/processed/xpass_360_nochoice/xpass_360_hist_gbdt.joblib")
print("feature columns:", getattr(m, "_offball_feature_columns", None))
# These two numbers must match the laptop exactly: 0.9408 / 0.7003.
# A mismatch means the pickle was loaded under a different scikit-learn and
# every downstream comparison would be against a different model.
print("open ", round(predict_pass_success_360(m, (0, 0), (20, 0), 1, [(30.0, 30.0)]), 4))
print("block", round(predict_pass_success_360(m, (0, 0), (20, 0), 1, [(10.0, 0.5)]), 4))
PY

echo
echo "위 두 값이 0.9408 / 0.7003 이어야 합니다. 다르면 멈추고 버전을 확인하세요."
echo
echo "== 청구 계정 후보 (sbatch --account 에 넣을 값) =="
accounts 2>/dev/null || echo "  'accounts' 명령이 없습니다. NCSA 문서/할당 메일을 확인하세요."
echo
echo "다음:  sbatch --account=<위 목록의 cpu 계정> deploy/delta/build.sbatch"
