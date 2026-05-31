#!/usr/bin/env bash
# One-line Adam upgrade for Mac/Linux: git pull + re-run installer (deps/kernels/model). KaTeX rides with the pull.
set -e
cd "$(dirname "$0")"
if [ ! -d .git ]; then
  echo "[upgrade] not a git checkout - get the latest with a fresh clone (data in ~/.amni-ai survives):"
  echo "  git clone https://github.com/Amnibro/Amni-Ai && cd Amni-Ai && python3 install.py"
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then echo "[error] python3 not found. Install Python 3.10+ and re-run."; exit 1; fi
python3 upgrade.py "$@"
