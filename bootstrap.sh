#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "\n[1/3] Downloading Metrica sample data"
python scripts/download_metrica.py

echo "\n[2/3] Downloading SkillCorner open data"
python scripts/download_skillcorner.py

echo "\n[3/3] Running toy baseline on Metrica"
python scripts/run_metrica_baseline.py
