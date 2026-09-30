#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

printf '%s\n' \
  "Environment ready." \
  "Download IDSSE separately and place it in data/raw/bundesliga-integrated/." \
  "See README.md for the pipeline and the commands."
