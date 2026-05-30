# bf16 Substrate Gap — v5.0.4 finding

**Date:** 2026-05-26

## Discovery
`scripts/v5_0_3_bake.py`'s `to_uint16_np()` for `torch.bfloat16` used `t.view(torch.float16)` (bit reinterpret) instead of `t.to(torch.float16)` (numeric cast). Bit reinterpret preserves the 16 bits unchanged but the numerical *value* changes because bf16 (1+8+7) and fp16 (1+5+10) split exponent/mantissa differently.

Concrete: bf16 `1.0` → bits `0x3F80` → viewed as fp16 → `1.875`. Then GF(17) stores those bits losslessly, kernel decodes via fp16 interpretation, math is wrong by ~10-90% per weight.

## Why Gemma 4 worked anyway
TBD — possibly Gemma's weights were fp16 in the HF release, not bf16, so the view-vs-cast distinction was a no-op. Or the eval tolerance hid it. Verify before declaring any bake "lossless" for bf16-source models.

## Fix (v5.0.4)
`v5_0_3_bake.py::to_uint16_np` now does `t.to(torch.float16).numpy().view(np.uint16)` for bf16, float32, float64 sources. fp16 source is unchanged (no cast needed).

## Substrate gap (open)
GPU kernels assume fp16 weight bits. bf16-native weight decode would need a new kernel in `amni/compute/hip/ari_hip.cpp` that uses `bf16b_to_f32()` for weight reads (those helper functions already exist on the activation side but aren't wired to weight texture sampling).

Without that kernel, bf16-source models lose ~3 bits of mantissa precision at bake time. For FLUX.2's range (mostly within fp16's [-65504, 65504]), this is acceptable for inference. For ranges that exceed fp16, clipping would occur — needs per-tensor validation.

## Action items
- [x] Patch bake to numeric-cast bf16 → fp16
- [ ] Re-verify gemma-e2b-prism manifest is sound (was it baked with the bug? if so, recovery TBD)
- [ ] Long-term: add bf16-native weight GEMV kernel for true CLAUDE.md compliance on bf16-source models
