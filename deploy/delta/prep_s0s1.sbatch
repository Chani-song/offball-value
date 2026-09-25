#!/usr/bin/env bash
#SBATCH --job-name=offball-prep01
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/prep01_%j.out
#SBATCH --error=logs/prep01_%j.out
#
# Turn the s0s1 extraction (conf200 + disp15) into the chunked pool stage 2
# reads, and report which of the labelled scenes are in it.
#
# The labelled scene indices from run_onset_v0_5 do NOT carry over: conf200
# merges phases and disp15 admits shorter runs, so both the phases and the
# onsets inside them are different objects. The indices have to be recovered
# from this pool by (match_id, onset_frame_id).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python || module load anaconda3_cpu || true
. .venv-delta/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

ROOT=data/processed/stack01_sweep/s0s1

echo "=== chunk 생성 ==="
python scripts/merge_full_extraction.py --root "$ROOT" --chunk-size 30

echo
echo "=== 라벨 장면이 이 풀의 몇 번인가 ==="
python - <<'PY'
import glob
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "sv", "scripts/score_vacated_space_rule.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
want = {(k[0], str(k[1])) for k in m.load_labels()}

root = Path("data/processed/stack01_sweep/s0s1")
manifest = json.loads((root / "manifest.json").read_text())
size = int(manifest["chunk_size"])
print(f"scene_count {manifest.get('scene_count')} · chunk_size {size}")

index = 0
hits = []
for chunk_path in sorted(root.glob("chunk*.json"),
                         key=lambda p: int(p.stem.replace("chunk", ""))):
    for scene in json.loads(chunk_path.read_text()):
        index += 1
        key = (str(scene.get("match_id")), str(scene.get("onset_frame_id")))
        if key in want:
            hits.append(index)
print(f"라벨 장면 {len(hits)}개 (전체 라벨 {len(want)})")
print(",".join(str(h) for h in hits))
Path("data/processed/stack01_sweep/s0s1_label_tasks.txt").write_text(
    ",".join(str(h) for h in hits))
PY
