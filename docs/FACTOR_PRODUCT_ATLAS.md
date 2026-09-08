# Factor+Product Atlas (`gf17_fpa` / `factor_product_atlas_v1`)

Lane A compressed-domain train/serve for Amni-AI.

**Status: bootstrap. Not Done. Not a near-1 / near-lossless claim.**

Lane B (freeze a dense checkpoint, then quantize) was **REFUTED**: measured
`bpw ~ 0.93` at `rel_err ~ 0.72`. That path is closed. The process-node story
is: multi-B models are **born as FPA parameters** and trained in the compressed
domain so dense `W` is never materialised.

This is **not** Gf17Atex `.codes` / `.scale` int4-group packing
(`AtexLin` / `GraniteAtexChatService` / `int4grp_gemv`). Do not load an FPA
folder through those classes, and do not load an ATEX bake through `FpaBake`.

## Forward (required math)

For each FPA Linear, with `x` shaped `(..., inn)`:

```
y = U_deq @ (V_deq.T @ x) + pq_gemv(x) + sparse_gemv(x)
```

Implemented as two thin GEMVs plus a product-VQ residual GEMV plus an optional
COO scatter. `W = U @ V.T` (shape `(out, inn)`) is never allocated.

## Bake contract

Manifest (`bake_manifest.json` or `manifest.json`):

| field | value |
|---|---|
| `format` | `gf17_fpa` |
| `packing` | `factor_product_atlas_v1` |
| `rank` | even integer (e.g. 64) |
| `pq` | `{gs:128, M:8, K:256, scale_dtype:f16}` |
| `kf` | opaque sidecar (typically sparse keep-fraction); stored, not interpreted |

Per-tensor keys in `*.safetensors` — example Qwen L15 `up_proj` `[9216, 2560]`,
`r=64`:

| key | dtype | shape | notes |
|---|---|---|---|
| `.U` | U8 | `(out, rank/2)` e.g. `(9216, 32)` | packed int4 nibbles |
| `.V` | U8 | `(inn, rank/2)` e.g. `(2560, 32)` | packed int4 nibbles |
| `.U_scale` | F32 | `(rank,)` | per-column |
| `.V_scale` | F32 | `(rank,)` | per-column |
| `.pq_idx` | U8 | `(out, inn/gs, M)` e.g. `(9216, 20, 8)` | product-VQ indices |
| `.pq_scale` | F16 | `(out, inn/gs)` | per residual group |

Int4 layout matches `amni/inference/int4_linear.py`: even index → low nibble,
odd → high nibble, stored `0..15 = signed -8..7` (zero-point 8).

Sidecars in the bake folder:

| file | role |
|---|---|
| `_fpa_codebooks.npy` | shared product-VQ tables. Canonical shape `(M, K, subvec)` with `subvec = gs/M` (16 when `gs=128`, `M=8`). Stacked / permuted layouts are normalised on load. |
| `_fpa_sparse.npz` | optional COO residual. Typical keys: `idx` `(2, nnz)` or `(nnz, 2)` + `vals`, or `rows`/`cols`/`vals`. Per-tensor prefixes accepted. |
| `_fpa_report.json` | optional bake notes (printed by the smoke CLI). |

Reference folders (Antman; may be absent in CI):

- `bakes/qwen35_4b_hc_fpa_onetensor_probe`
- `bakes/granite41_3b_hc_fpa_onetensor_probe`

## Product-VQ residual

Hypothesis implemented here (kept behind `pq_gemv` so the interface stays
stable if a later `_fpa_report.json` disagrees):

```
R[o, g*gs:(g+1)*gs] = pq_scale[o, g] * concat_m( codebooks[m, pq_idx[o, g, m]] )
y_pq[o]             = sum_g  R[o, g] · x[g]
```

Equivalent GEMV that never builds `R` as `(out, inn)`:

```
y_pq[o] = sum_g  pq_scale[o, g] * sum_m  ( codebooks[m, idx[o,g,m]] · x_m )
```

## Process-node / distill

`FpaLinear.born(in, out, rank=…, trainable=True)` constructs a layer that
exists only as FPA params (random U/V, zeroed pq_scale).

`FpaBake(folder, trainable=True).linear()` dequants packed U/V into
`nn.Parameter`s. `parameters()` / `distill_parameters()` yield `U`, `V`,
`pq_scale`, `codebooks`. Packed codes and `pq_idx` stay discrete buffers.

```python
from amni.inference.fpa_linear import FpaBake, FpaLinear, load_fpa_linear

lin = load_fpa_linear("bakes/qwen35_4b_hc_fpa_onetensor_probe", trainable=True)
y = lin(x)   # no dense W
opt = torch.optim.AdamW(lin.parameters(), lr=1e-4)
```

## Smoke

```bash
python -m amni.inference.gf17_fpa
python -m amni.inference.gf17_fpa --bake bakes/qwen35_4b_hc_fpa_onetensor_probe
python -m unittest tests.test_gf17_fpa -v
```

CPU is required and is the default. CUDA/HIP work via ordinary `lin.to(device)`
(device-agnostic torch; no custom kernel in this bootstrap).
