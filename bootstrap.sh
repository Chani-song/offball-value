#!/usr/bin/env bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

echo "\n[1/3] Downloading Metrica sample data"
python scripts/download_metrica.py

echo "\n[2/3] Downloading SkillCorner open data"
python scripts/download_skillcorner.py

echo "\n[3/3] Running toy baseline on Metrica"
python scripts/run_metrica_baseline.py
