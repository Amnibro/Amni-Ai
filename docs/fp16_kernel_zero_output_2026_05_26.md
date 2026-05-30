# fp16 GEMV/matmul kernels returning zero on RGBA4 bake — finding 2026-05-26

## Symptom
`engine.gemv_exact_fp16(x_dev, tb, N, K)` and `engine.tex_matmul_fp16(x_dev, tb, M, K, N)` both return all-zero output buffers when tested against:
- A fresh GF(17) RGBA4 bake (v5.0.4 bf16→fp16 cast, weight-decode cos=1.0)
- A trivially-encoded diagonal weight (4×4 and 64×64 tests, also output zero)
- Both fp16 and bf16 activation bit conventions

## Confirmed working
- `_load()` returns True; `libari_hip.dll` loads
- `bind_texture(page)` returns valid `tex_idx`
- `decode_rgba4_to_fp16` roundtrip is bit-exact (cos=1.0 verified)

## What's NOT confirmed
- Whether `k_gemv_rgba_fp16` (line 393 in `ari_hip.cpp`) actually executes
- Whether `hipDeviceSynchronize` is being called after kernel launch
- Whether texture binding alignment matters for arbitrary page widths (tested 4 and 64 — neither worked)
- Whether HIP texture objects need specific dimension constraints on RDNA3

## Hypotheses (untested)
1. **Adam doesn't use these kernels.** `amni/model/adam.py` exclusively uses `tex_matmul_t` (the INT8 GF(17) path with `g_mul`/`g_add` LUTs at line 61 of `ari_hip.cpp`). The fp16 GEMV/matmul kernels may be untested in production and have a latent bug.
2. **Page width constraint.** `hipTextureObject_t` on RDNA3 may require page width ≥ some threshold or power-of-2. My test pages were w=4 and w=64.
3. **Kernel launch silently failing.** No HIP error reporting in the bindings; if the kernel fails to launch, output stays zero.

## Recommended next steps
- **Substrate work**: add `hipGetLastError()` polling after kernel launch in `ari_*_fp16` C functions. Surface kernel-launch errors to Python instead of silently zeroing.
- **Or**: skip TMU kernel debugging and use a torch-native `PtexLinear`:
  1. `PtexBank` mmap's `.ptex` files on disk (already works)
  2. On forward, decode via `decode_rgba4_to_fp16` in numpy/CPU → upload as torch fp16 tensor → standard `F.linear(x, W)`
  3. Loses TMU advantage but proves the bake is usable end-to-end
  4. Pair with LRU eviction for the VRAM win (only one block's weights resident at a time)

## Status
- Bake: WORKS (v5.0.4 + chunked-verify patch, FLUX2 9B done in 265s, all 233 tensors lossless cos=1.0 vs source)
- `PtexBank` loader: WORKS (mmap, manifest, decode)
- `PtexLinear.forward` via HIP kernel: BROKEN (returns zeros)
- Substrate GEMV/matmul fp16 kernels: NEED INVESTIGATION
