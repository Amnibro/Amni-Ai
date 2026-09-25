#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYO3_PYTHON:-$(cd .. && pwd)/.venv-linux/bin/python}"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=x86-64-v3}" CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/ks}"
cargo build --release -j "${JOBS:-4}" -p adam-ks --bin adam-ks
PYO3_PYTHON="$PY" cargo build --release -j "${JOBS:-4}" -p adam-ks --lib --features py
EXT="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
cp "$CARGO_TARGET_DIR/release/libadam_ks.so" "../adam_ks$EXT"
echo "installed $(cd .. && pwd)/adam_ks$EXT and $CARGO_TARGET_DIR/release/adam-ks"
