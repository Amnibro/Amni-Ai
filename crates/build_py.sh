#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYO3_PYTHON:-$(cd .. && pwd)/.venv-linux/bin/python}"
PYO3_PYTHON="$PY" cargo build --release -j "${JOBS:-8}" -p adam-ray-py
cp target/release/libadam_ray.so "../adam_ray$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
echo "installed $(cd .. && pwd)/adam_ray$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
