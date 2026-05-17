#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export ROCM_HOME="${ROCM_HOME:-/opt/rocm-7.2.0}"
export PATH="$ROCM_HOME/bin:$PATH"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
echo "[build_f32] ROCM_HOME=$ROCM_HOME"
echo "[build_f32] compiling gf17_hip.cpp with f32 dequant kernels..."
hipcc -shared -fPIC -O3 --offload-arch=gfx1100 --offload-arch=gfx1101 \
    -I"$ROCM_HOME/include" \
    -o libgf17_hip.so gf17_hip.cpp \
    -L"$ROCM_HOME/lib" -lamdhip64 -Wl,-rpath,"$ROCM_HOME/lib"
SIZE=$(stat -c%s libgf17_hip.so)
echo "[build_f32] built: $DIR/libgf17_hip.so ($SIZE bytes)"
echo "[build_f32] verifying f32 symbols..."
nm -D libgf17_hip.so | grep -E "T gf17_(dq_gemv|rms_norm_f32|elem_add_f32|silu_inp_f32|alloc_f32)"
echo "[build_f32] done"
