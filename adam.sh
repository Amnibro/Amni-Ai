#!/usr/bin/env bash
# Launch Adam in a standalone desktop window (Mac/Linux). Activates the local venv if present,
# serves Adam, and opens a chromeless Chrome/Edge --app window once the model is ready.
cd "$(dirname "$0")" || exit 1
PY=python3
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
echo "Starting Adam -- a standalone window will open when the model is ready (first warmup takes a minute)..."
exec "$PY" -m amni.cli serve --app-window "$@"
