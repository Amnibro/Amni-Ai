#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export PATH="$ROCM_HOME/bin:$PATH"
echo "[gf17_hip] compiling for gfx1100+gfx1101 (RX 7800 XT w/ HSA override)..."
hipcc -shared -fPIC -O3 --offload-arch=gfx1100 --offload-arch=gfx1101 \
    -I"$ROCM_HOME/include" \
    -o libgf17_hip.so gf17_hip.cpp \
    -L"$ROCM_HOME/lib" -lamdhip64 -Wl,-rpath,"$ROCM_HOME/lib"
echo "[gf17_hip] built: $DIR/libgf17_hip.so ($(stat -c%s libgf17_hip.so) bytes)"
echo "[gf17_hip] verifying symbols..."
nm -D libgf17_hip.so | grep "T gf17_" | head -20
echo "[gf17_hip] done"
