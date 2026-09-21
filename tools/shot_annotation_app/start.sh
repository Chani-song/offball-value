#!/bin/sh
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$VENV/bin/python" -m ensurepip --upgrade
fi
"$VENV/bin/python" -m pip install -q -r requirements.txt
exec "$VENV/bin/python" -u app.py
