#!/usr/bin/env bash
# One-line Adam installer for Mac/Linux.
set -e
if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found. Install Python 3.10+ and re-run."
  exit 1
fi
PYV=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found python3 = $PYV"
python3 install.py "$@"
