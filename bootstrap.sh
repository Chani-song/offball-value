#!/usr/bin/env bash
# Create .venv with the pinned versions in requirements.txt and this package in editable mode.
# The solvers also need the collaborator's solver base (defensive_positioning); see docs/reproduction.md.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps

if [ -d andrew ]; then
  .venv/bin/python -m pip install -e andrew --no-deps
else
  echo "andrew/ not found: the solver packages will not import (docs/reproduction.md, 'Solver base')."
fi
echo "Environment ready. Data: docs/data.md."
