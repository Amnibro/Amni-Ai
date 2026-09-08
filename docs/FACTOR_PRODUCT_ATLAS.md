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
indexed sparse contrib. `W = U @ V.T` (shape `(out, inn)`) is never allocated.
v1 CPU may materialise the **sparse residual only**; UV+PQ stay factorised.

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

Sidecars in the bake folder (Antman ground truth):

| file | role |
|---|---|
| `_fpa_codebooks.npy` | float32 **`(M=8, K=256, subvec_dim=16)`**. Other stacked / permuted layouts are still normalised on load. |
| `_fpa_sparse.npz` | **`indices`** int64 `(nnz,)` flat into `(out, inn)`; **`values`** int8 `(nnz,)`; **`vmax`** float32 `(1,)`; **`shape`** int64 `(2,)` = `(out, inn)`. Reconstruct: scatter `values/127 * vmax` at flat indices. Indexed GEMV is the default (no full sparse matrix). |
| `_fpa_report.json` | optional bake notes (printed by the smoke CLI). |

Reference folders (Antman; may be absent in CI):

- `bakes/qwen35_4b_hc_fpa_onetensor_probe`
- `bakes/granite41_3b_hc_fpa_onetensor_probe`

## Product-VQ residual

Per output row, over groups `g = inn/gs` (Antman sketch):

```
y_pq[row] ≈ sum_g  pq_scale[row, g] * sum_m
    codebook[m, pq_idx[row, g, m]] · x[g*gs + m*subvec : g*gs + (m+1)*subvec]
```

`codebook` entries are length `subvec_dim=16`. Equivalent concat form
(`scale * concat_m(lookup_m) · x_group`) is used only in unit tests on tiny
rows. The serve/train path never builds `(out, inn)`.

## Sparse residual

Antman `_fpa_sparse.npz`:

```
deq = values.float() / 127 * vmax          # values: int8
row, col = divmod(indices, inn)            # indices: flat int64
y[row] += deq * x[col]
```

`FpaLinear.materialize_sparse()` exists for v1 CPU / tests (scatter into
`(out, inn)`). Forward uses indexed contrib.

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
