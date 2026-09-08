"""gf17_fpa / factor_product_atlas_v1 — Lane A compressed-domain forward.

Bootstrap only. This is NOT Done and NOT a near-1 / near-lossless claim.

Lane B freeze-quantize of dense W was REFUTED (measured bpw~0.93 @ rel_err~0.72).
The native FPA path exists so train/distill/serve can run without ever materialising
dense W:

    y = U_deq @ (V_deq.T @ x) + pq_gemv(x) + sparse_gemv(x)

This module is the packing + math kernel. ``FpaLinear`` (trainable factors for a
later Kimahri distill loop) lives in ``amni.inference.fpa_linear``.

Bake contract (one-tensor probe folders on Antman; shapes mirrored here):

    format:  gf17_fpa
    packing: factor_product_atlas_v1
    rank, pq: {gs:128, M:8, K:256, scale_dtype:f16}, kf

Per-tensor safetensors keys (Qwen L15 up_proj [9216,2560], r=64)::

    .U        U8 packed int4 nibbles (out, rank/2)     e.g. (9216, 32)
    .V        U8 packed int4          (inn, rank/2)     e.g. (2560, 32)
    .U_scale  F32 (rank,) per-column
    .V_scale  F32 (rank,)
    .pq_idx   U8  (out, inn/gs, M)                      e.g. (9216, 20, 8)
    .pq_scale F16 (out, inn/gs)

Sidecars::

    _fpa_codebooks.npy  shared product-VQ tables; canonical (M, K, subvec)
                        with subvec = gs/M (16 when gs=128, M=8). Other
                        stacked layouts are accepted and normalised.
    _fpa_sparse.npz     optional COO residual (idx + vals, or rows/cols/vals)
    _fpa_report.json    optional bake notes (printed by the smoke CLI)

Int4 nibble layout matches ``int4_linear`` / packed ATEX gemv: even index →
low nibble, odd → high nibble, stored as uint8 0..15 = signed -8..7 (zp=8).

PQ residual (product-VQ, not Gf17Atex ``.codes/.scale``)::

    R[o, g*gs:(g+1)*gs] = pq_scale[o,g] * concat_m(codebooks[m, pq_idx[o,g,m]])
    y_pq[o] = sum_g R[o, g] · x[g]

i.e. a GEMV of the reconstructed residual groups — implemented as
``scale * sum_m (lookup_m · x_m)`` so the (out, inn) matrix is never built.

Reference bake dirs (docs/tests; may be absent on this disk)::

    bakes/qwen35_4b_hc_fpa_onetensor_probe
    bakes/granite41_3b_hc_fpa_onetensor_probe

Smoke::

    python -m amni.inference.gf17_fpa
    python -m amni.inference.gf17_fpa --bake bakes/qwen35_4b_hc_fpa_onetensor_probe
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

FORMAT = "gf17_fpa"
PACKING = "factor_product_atlas_v1"
DEFAULT_GS = 128
DEFAULT_M = 8
DEFAULT_K = 256
INT4_ZP = 8
INT4_ABSMAX = 7

Pathish = Union[str, os.PathLike]


def is_fpa_manifest(man: Mapping[str, Any]) -> bool:
    fmt = str(man.get("format") or "").strip().lower()
    packing = str(man.get("packing") or "").strip().lower()
    if fmt in (FORMAT, "fpa") or packing == PACKING:
        return True
    return False


def is_atex_codes_bake(man: Mapping[str, Any]) -> bool:
    """True for Gf17Atex ``.codes/.scale`` int4-group bakes — not FPA."""
    if is_fpa_manifest(man):
        return False
    tensors = man.get("tensors") or {}
    return bool(man.get("gs") is not None and any((t or {}).get("q") == 1 for t in tensors.values()))


def read_bake_manifest(folder: Pathish) -> Dict[str, Any]:
    folder = Path(folder)
    for name in ("bake_manifest.json", "manifest.json"):
        p = folder / name
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"no bake_manifest.json / manifest.json under {folder}")


def is_fpa_bake(folder: Pathish) -> bool:
    try:
        return is_fpa_manifest(read_bake_manifest(folder))
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return False


def pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 ``[..., n]`` (n even, values in [-8, 7]) to uint8 ``[..., n/2]``.

    Even index → low nibble, odd index → high nibble. Matches ``Int4GroupLinear``.
    """
    if q.shape[-1] % 2:
        raise ValueError(f"int4 pack last-dim must be even, got {tuple(q.shape)}")
    qf = (q.to(torch.int16) + INT4_ZP).clamp(0, 15).to(torch.uint8)
    return (qf[..., 0::2] | (qf[..., 1::2] << 4)).contiguous()


def unpack_int4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 ``[..., n/2]`` to int16 ``[..., n]`` in [-8, 7]."""
    p = packed.to(torch.int16)
    lo = (p & 0xF) - INT4_ZP
    hi = ((p >> 4) & 0xF) - INT4_ZP
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int16, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def quantize_int4_cols(W: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-column symmetric int4 of a factor ``(rows, rank)`` → packed uint8 + f32 scale."""
    Wf = W.detach().float()
    if scale is None:
        scale = Wf.abs().amax(dim=0).clamp_min(1e-8) / float(INT4_ABSMAX)
    else:
        scale = scale.detach().float().reshape(-1)
        if scale.numel() != Wf.shape[-1]:
            raise ValueError(f"scale length {scale.numel()} != rank {Wf.shape[-1]}")
    q = torch.clamp(torch.round(Wf / scale.unsqueeze(0)), -INT4_ABSMAX, INT4_ABSMAX)
    return pack_int4_nibbles(q), scale.contiguous()


def dequant_int4_cols(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(rows, rank/2)`` packed + ``(rank,)`` per-col scale → ``(rows, rank)``."""
    q = unpack_int4_nibbles(packed).to(dtype)
    return q * scale.to(device=packed.device, dtype=dtype).reshape(-1)


def uv_gemv(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """``y = U @ (V.T @ x)`` for ``x (..., inn)``, ``U (out, r)``, ``V (inn, r)``.

    Never forms ``W = U @ V.T``.
    """
    xf = x.to(dtype=U.dtype)
    z = xf @ V
    return z @ U.T


def normalize_codebooks(arr: Any, M: int, K: int, subvec: int) -> torch.Tensor:
    """Accept common bake layouts; return float32 ``(M, K, subvec)``."""
    if isinstance(arr, torch.Tensor):
        t = arr.detach().cpu()
        a = t.numpy() if t.dtype != torch.bfloat16 else t.float().numpy()
    else:
        a = np.asarray(arr)
    a = np.ascontiguousarray(a)
    want = (M, K, subvec)
    if a.shape == want:
        return torch.from_numpy(a.copy()).float()
    if a.shape == (K, M, subvec):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (1, 0, 2)))).float()
    if a.shape == (M, subvec, K):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (0, 2, 1)))).float()
    if a.shape == (K, subvec, M):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (2, 0, 1)))).float()
    if a.size == M * K * subvec:
        return torch.from_numpy(np.ascontiguousarray(a.reshape(M, K, subvec))).float()
    raise ValueError(
        f"codebooks shape {a.shape} does not match (M,K,subvec)={want} "
        f"(or a stacked permutation of {M * K * subvec} elements)"
    )


def load_fpa_codebooks(path: Pathish, M: int, K: int, subvec: int) -> torch.Tensor:
    return normalize_codebooks(np.load(os.fspath(path), allow_pickle=False), M, K, subvec)


def pq_reconstruct_groups(
    pq_idx: torch.Tensor,
    pq_scale: torch.Tensor,
    codebooks: torch.Tensor,
    gs: int,
) -> torch.Tensor:
    """Reconstruct residual as ``(out, G, gs)`` — for tests / small rows only.

    Still not a dense ``(out, inn)`` allocation when the caller keeps G*gs scoped.
    Product-VQ: each group is ``scale * concat_m(codebooks[m, idx[m]])``.
    """
    out, G, M = pq_idx.shape
    D = codebooks.shape[-1]
    if M * D != gs:
        raise ValueError(f"product-VQ expects gs=M*subvec, got gs={gs} M={M} subvec={D}")
    idx = pq_idx.to(dtype=torch.long)
    chunks = []
    for m in range(M):
        chunks.append(codebooks[m][idx[:, :, m]])
    groups = torch.cat(chunks, dim=-1)
    return groups * pq_scale.to(groups.dtype).unsqueeze(-1)


def pq_gemv(
    x: torch.Tensor,
    pq_idx: torch.Tensor,
    pq_scale: torch.Tensor,
    codebooks: torch.Tensor,
    gs: int,
) -> torch.Tensor:
    """Product-VQ residual GEMV. Peak extra is ``(out, G, subvec)``, never ``(out, inn)``."""
    *batch, inn = x.shape
    out, G, M = pq_idx.shape
    D = int(codebooks.shape[-1])
    if M * D != gs:
        raise ValueError(f"product-VQ expects gs=M*subvec, got gs={gs} M={M} subvec={D}")
    if G * gs != inn:
        raise ValueError(f"x inn={inn} != G*gs={G * gs} (G={G} gs={gs})")
    xf = x.reshape(-1, inn).to(dtype=codebooks.dtype)
    B = xf.shape[0]
    xg = xf.reshape(B, G, M, D)
    idx = pq_idx.to(dtype=torch.long)
    dots = xf.new_zeros(B, out, G)
    for m in range(M):
        vecs = codebooks[m][idx[:, :, m]]
        dots = dots + torch.einsum("bgd,ogd->bog", xg[:, :, m, :], vecs)
    y = (dots * pq_scale.to(dtype=dots.dtype).unsqueeze(0)).sum(-1)
    return y.reshape(*batch, out)


def parse_sparse_npz(npz: Mapping[str, Any], tensor_name: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
    """Parse ``_fpa_sparse.npz`` into COO ``rows, cols, vals``.

    Accepts idx(2,nnz)|idx(nnz,2)+vals, rows/cols/vals, row/col/val, i/j/v,
    and optional per-tensor prefixes (``{name}.idx`` / ``{name}/idx``).
    """
    keys = list(npz.keys())

    def _as_t(v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.detach()
        return torch.from_numpy(np.ascontiguousarray(np.asarray(v)))

    prefixes: List[str] = [""]
    if tensor_name:
        stem = tensor_name
        prefixes = [
            f"{stem}.",
            f"{stem}/",
            f"{stem}.weight.",
            f"{stem}.weight/",
            "",
        ]

    def _first(cands: Sequence[str]) -> Optional[str]:
        for pfx in prefixes:
            for c in cands:
                k = f"{pfx}{c}"
                if k in npz:
                    return k
        return None

    idx_k = _first(("idx", "indices", "index", "ij"))
    val_k = _first(("vals", "val", "values", "data", "v"))
    row_k = _first(("rows", "row", "i"))
    col_k = _first(("cols", "col", "j"))

    rows = cols = vals = None
    if idx_k is not None and val_k is not None:
        idx = _as_t(npz[idx_k])
        vals = _as_t(npz[val_k]).reshape(-1).float()
        if idx.ndim == 2 and idx.shape[0] == 2:
            rows, cols = idx[0], idx[1]
        elif idx.ndim == 2 and idx.shape[1] == 2:
            rows, cols = idx[:, 0], idx[:, 1]
        else:
            raise ValueError(f"sparse idx shape {tuple(idx.shape)} — expected (2,nnz) or (nnz,2)")
    elif row_k is not None and col_k is not None and val_k is not None:
        rows, cols = _as_t(npz[row_k]), _as_t(npz[col_k])
        vals = _as_t(npz[val_k]).reshape(-1).float()
    elif not keys:
        return None
    else:
        if tensor_name is not None:
            return None
        raise ValueError(f"unrecognised sparse npz keys: {keys}")

    if rows is None:
        return None
    return {
        "rows": rows.reshape(-1).to(torch.int64).contiguous(),
        "cols": cols.reshape(-1).to(torch.int64).contiguous(),
        "vals": vals.contiguous(),
    }


def load_fpa_sparse(path: Pathish, tensor_name: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
    if not Path(path).is_file():
        return None
    with np.load(os.fspath(path), allow_pickle=False) as z:
        return parse_sparse_npz({k: z[k] for k in z.files}, tensor_name=tensor_name)


def sparse_gemv(
    x: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """COO residual GEMV: ``y[row] += val * x[col]``."""
    *batch, inn = x.shape
    xf = x.reshape(-1, inn).to(dtype=vals.dtype)
    B = xf.shape[0]
    y = xf.new_zeros(B, out_features)
    if rows.numel() == 0:
        return y.reshape(*batch, out_features)
    contrib = vals.to(dtype=xf.dtype).unsqueeze(0) * xf[:, cols.long()]
    y.index_add_(1, rows.long(), contrib)
    return y.reshape(*batch, out_features)


def _load_safetensors_map(path: Pathish) -> Dict[str, torch.Tensor]:
    """NumPy-first load so U8 packed tensors avoid the ROCm safetensors pt-ctor crash."""
    try:
        from safetensors.numpy import load_file as load_np

        raw = load_np(os.fspath(path))
        out = {}
        for k, v in raw.items():
            a = np.ascontiguousarray(v)
            out[k] = torch.from_numpy(a)
        return out
    except Exception:
        from safetensors.torch import load_file as load_pt

        return load_pt(os.fspath(path))


def load_bake_tensors(folder: Pathish) -> Dict[str, torch.Tensor]:
    folder = Path(folder)
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {folder}")
    tens: Dict[str, torch.Tensor] = {}
    for sh in shards:
        tens.update(_load_safetensors_map(sh))
    return tens


def pq_cfg_from_manifest(man: Mapping[str, Any]) -> Dict[str, Any]:
    pq = dict(man.get("pq") or {})
    gs = int(pq.get("gs", man.get("gs", DEFAULT_GS)))
    M = int(pq.get("M", man.get("M", DEFAULT_M)))
    K = int(pq.get("K", man.get("K", DEFAULT_K)))
    scale_dtype = str(pq.get("scale_dtype", "f16"))
    if gs <= 0 or M <= 0 or K <= 0 or gs % M:
        raise ValueError(f"invalid pq cfg gs={gs} M={M} K={K} (gs must divide by M)")
    return {"gs": gs, "M": M, "K": K, "scale_dtype": scale_dtype, "subvec": gs // M}


def _stem_from_u_key(key: str) -> str:
    if key.endswith(".weight.U"):
        return key[: -len(".U")]
    if key.endswith(".U"):
        return key[: -len(".U")]
    raise ValueError(f"not an FPA .U key: {key}")


def discover_fpa_stems(keys: Iterable[str]) -> List[str]:
    stems = []
    for k in keys:
        if k.endswith(".U"):
            stems.append(_stem_from_u_key(k))
    if not stems and "U" in set(keys):
        stems.append("")
    return sorted(set(stems))


def _key(stem: str, suffix: str) -> str:
    return suffix if stem == "" else f"{stem}.{suffix}"


def extract_fpa_tensor_pack(tens: Mapping[str, torch.Tensor], stem: str) -> Dict[str, torch.Tensor]:
    def g(*names: str) -> torch.Tensor:
        for n in names:
            if n in tens:
                return tens[n]
        raise KeyError(f"missing FPA key(s) {names} for stem={stem!r}")

    pack = {
        "U": g(_key(stem, "U"), "U"),
        "V": g(_key(stem, "V"), "V"),
        "U_scale": g(_key(stem, "U_scale"), "U_scale"),
        "V_scale": g(_key(stem, "V_scale"), "V_scale"),
        "pq_idx": g(_key(stem, "pq_idx"), "pq_idx"),
        "pq_scale": g(_key(stem, "pq_scale"), "pq_scale"),
    }
    bias_k = _key(stem, "bias") if stem else "bias"
    if bias_k in tens:
        pack["bias"] = tens[bias_k]
    elif stem.endswith(".weight"):
        b = stem[: -len(".weight")] + ".bias"
        if b in tens:
            pack["bias"] = tens[b]
    return pack


def validate_fpa_shapes(pack: Mapping[str, torch.Tensor], pq: Mapping[str, Any], rank: Optional[int] = None) -> Tuple[int, int, int]:
    U, V = pack["U"], pack["V"]
    us, vs = pack["U_scale"], pack["V_scale"]
    pqi, pqs = pack["pq_idx"], pack["pq_scale"]
    out, half_u = U.shape
    inn, half_v = V.shape
    r = int(us.numel())
    if rank is not None and int(rank) != r:
        raise ValueError(f"manifest rank={rank} != U_scale.numel()={r}")
    if half_u != r // 2 or half_v != r // 2 or r % 2:
        raise ValueError(f"packed U/V last-dim must be rank/2 (rank={r} U={tuple(U.shape)} V={tuple(V.shape)})")
    if vs.numel() != r:
        raise ValueError(f"V_scale.numel()={vs.numel()} != rank={r}")
    gs, M = int(pq["gs"]), int(pq["M"])
    if inn % gs:
        raise ValueError(f"inn={inn} not divisible by pq.gs={gs}")
    G = inn // gs
    if tuple(pqi.shape) != (out, G, M):
        raise ValueError(f"pq_idx shape {tuple(pqi.shape)} != {(out, G, M)}")
    if tuple(pqs.shape) != (out, G):
        raise ValueError(f"pq_scale shape {tuple(pqs.shape)} != {(out, G)}")
    return out, inn, r


def write_synthetic_bake(
    folder: Pathish,
    *,
    out: int = 64,
    inn: int = 128,
    rank: int = 8,
    gs: int = 128,
    M: int = 8,
    K: int = 256,
    name: str = "model.layers.15.mlp.up_proj.weight",
    sparse_nnz: int = 4,
    seed: int = 0,
    kf: float = 0.01,
) -> Path:
    """Write a contract-shaped one-tensor FPA bake (for tests / smoke). Never used as a quality claim."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if inn % gs:
        raise ValueError(f"inn={inn} must be divisible by gs={gs}")
    if rank % 2:
        raise ValueError(f"rank={rank} must be even")
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(out, rank, generator=g) / (rank ** 0.5)
    V = torch.randn(inn, rank, generator=g) / (rank ** 0.5)
    U_p, U_s = quantize_int4_cols(U)
    V_p, V_s = quantize_int4_cols(V)
    G = inn // gs
    subvec = gs // M
    codebooks = torch.randn(M, K, subvec, generator=g) * 0.05
    pq_idx = torch.randint(0, K, (out, G, M), generator=g, dtype=torch.int64).to(torch.uint8)
    pq_scale = (torch.rand(out, G, generator=g) * 0.1).to(torch.float16)
    tens = {
        f"{name}.U": U_p.contiguous(),
        f"{name}.V": V_p.contiguous(),
        f"{name}.U_scale": U_s.contiguous(),
        f"{name}.V_scale": V_s.contiguous(),
        f"{name}.pq_idx": pq_idx.contiguous(),
        f"{name}.pq_scale": pq_scale.contiguous(),
    }
    try:
        from safetensors.numpy import save_file

        save_file({k: v.detach().cpu().numpy() for k, v in tens.items()}, str(folder / "model.safetensors"))
    except Exception:
        from safetensors.torch import save_file as save_pt

        save_pt(tens, str(folder / "model.safetensors"))
    np.save(folder / "_fpa_codebooks.npy", codebooks.numpy())
    if sparse_nnz > 0:
        rows = torch.randint(0, out, (sparse_nnz,), generator=g)
        cols = torch.randint(0, inn, (sparse_nnz,), generator=g)
        vals = torch.randn(sparse_nnz, generator=g) * 0.2
        np.savez(
            folder / "_fpa_sparse.npz",
            idx=torch.stack([rows, cols], 0).numpy().astype(np.int64),
            vals=vals.numpy().astype(np.float32),
        )
    man = {
        "format": FORMAT,
        "packing": PACKING,
        "rank": rank,
        "kf": kf,
        "pq": {"gs": gs, "M": M, "K": K, "scale_dtype": "f16"},
        "tensors": {
            name: {
                "shape": [out, inn],
                "q": "fpa",
                "rank": rank,
            }
        },
        "note": "synthetic one-tensor probe — Lane A bootstrap; not a quality claim; Lane B freeze-quantize REFUTED",
    }
    with open(folder / "bake_manifest.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    with open(folder / "_fpa_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pq": "product-VQ: residual group = pq_scale * concat_m(codebooks[m, idx])",
                "forward": "y = U_deq @ (V_deq.T @ x) + pq_gemv(x) + sparse_gemv(x)",
                "lane_b": "freeze-quantize REFUTED (bpw~0.93 @ rel_err~0.72)",
                "status": "bootstrap — not Done, not near-1",
            },
            f,
            indent=2,
        )
    return folder


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="gf17_fpa one-tensor GEMV smoke (Lane A bootstrap, not Done)")
    p.add_argument("--bake", default="", help="FPA bake folder (synthetic if omitted)")
    p.add_argument("--name", default="", help="tensor stem (default: the only / first tensor)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=int, default=64)
    p.add_argument("--inn", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    args = p.parse_args(list(argv) if argv is not None else None)

    import tempfile

    from amni.inference.fpa_linear import FpaBake

    device = torch.device(args.device)
    tmp = None
    if args.bake:
        bake = FpaBake(args.bake, trainable=False).to(device)
        lin = bake.linear(args.name or None)
        src = args.bake
    else:
        tmp = tempfile.TemporaryDirectory(prefix="gf17_fpa_smoke_")
        folder = write_synthetic_bake(
            tmp.name,
            out=args.out,
            inn=args.inn,
            rank=args.rank,
            seed=args.seed,
        )
        bake = FpaBake(folder, trainable=False).to(device)
        lin = bake.linear(args.name or None)
        src = str(folder) + "  (synthetic)"
    torch.manual_seed(args.seed)
    x = torch.randn(lin.in_features, device=device)
    y = lin(x)
    print("gf17_fpa / factor_product_atlas_v1  [Lane A bootstrap — NOT Done; freeze-quantize REFUTED]")
    print(f"  bake     {src}")
    print(f"  name     {lin.name}")
    print(f"  format   {bake.manifest.get('format')} packing={bake.manifest.get('packing')}")
    print(f"  shapes   x={tuple(x.shape)} y={tuple(y.shape)} U={tuple(lin.U.shape)} V={tuple(lin.V.shape)} rank={lin.rank}")
    print(f"  pq       idx={tuple(lin.pq_idx.shape)} scale={tuple(lin.pq_scale.shape)} cb={tuple(lin.codebooks.shape)} gs={lin.gs}")
    print(f"  sparse   nnz={lin.sparse_nnz}")
    print(f"  y.norm   {float(y.float().norm()):.6f}  y.mean={float(y.float().mean()):.6e}")
    report = bake.report
    if report:
        print(f"  report   {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
