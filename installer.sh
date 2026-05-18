#!/usr/bin/env bash
set -e
if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "[error] Python not found. Install Python 3.10+ and re-run."
  exit 1
fi
PY=$(command -v python3 || command -v python)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PY" "$DIR/installer.py" "$@"
