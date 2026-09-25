#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYO3_PYTHON:-$(cd .. && pwd)/.venv-linux/bin/python}"
DEST="${DEST:-$(cd .. && pwd)}"
EXT="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYO3_PYTHON="$PY" cargo build --release -j "${JOBS:-8}" -p adam-gf17 --features py
mkdir -p "$DEST"
cp target/release/libadam_gf17.so "$DEST/adam_gf17$EXT"
echo "installed $DEST/adam_gf17$EXT"
