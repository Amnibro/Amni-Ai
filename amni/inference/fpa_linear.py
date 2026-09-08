"""FpaLinear — native Factor+Product Atlas layer for Lane A train / serve.

``parameters()`` exposes trainable U, V, pq_scale, and shared codebooks when
``trainable=True`` so a later Kimahri distill loop can step compressed-domain
factors without ever allocating dense W.

This is not Gf17Atex ``AtexLin`` (``.codes`` / ``.scale`` int4-group). Do not
route an FPA bake through ``GraniteAtexChatService``.

Status: bootstrap. Not Done. Not a near-1 quality claim. Lane B freeze-quantize
was REFUTED (bpw~0.93 @ rel_err~0.72).

Usage::

    from amni.inference.fpa_linear import FpaBake, FpaLinear

    bake = FpaBake("bakes/qwen35_4b_hc_fpa_onetensor_probe", trainable=True)
    lin = bake.linear()
    y = lin(x)                          # U@(Vᵀx) + pq + sparse
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-4)

    # born-as-FPA (process-node: multi-B starts as FPA params)
    lin = FpaLinear.born(2560, 9216, rank=64, trainable=True)

Smoke: ``python -m amni.inference.fpa_linear`` (same CLI as gf17_fpa).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from amni.inference.gf17_fpa import (
    DEFAULT_GS,
    DEFAULT_K,
    DEFAULT_M,
    FORMAT,
    PACKING,
    dequant_int4_cols,
    discover_fpa_stems,
    extract_fpa_tensor_pack,
    is_atex_codes_bake,
    is_fpa_manifest,
    load_bake_tensors,
    load_fpa_codebooks,
    load_fpa_sparse,
    pq_cfg_from_manifest,
    pq_gemv,
    quantize_int4_cols,
    read_bake_manifest,
    sparse_gemv,
    uv_gemv,
    validate_fpa_shapes,
)

Pathish = Union[str, Path]


def _reg(mod: nn.Module, name: str, tensor: torch.Tensor, trainable: bool) -> None:
    if trainable and tensor.is_floating_point():
        mod.register_parameter(name, nn.Parameter(tensor))
    else:
        if isinstance(getattr(mod, name, None), nn.Parameter):
            delattr(mod, name)
        mod.register_buffer(name, tensor)


class FpaLinear(nn.Module):
    """Compressed-domain Linear: ``y = U @ (V.T @ x) + pq_gemv(x) + sparse_gemv(x)``."""

    def __init__(
        self,
        U_packed: torch.Tensor,
        V_packed: torch.Tensor,
        U_scale: torch.Tensor,
        V_scale: torch.Tensor,
        pq_idx: torch.Tensor,
        pq_scale: torch.Tensor,
        codebooks: torch.Tensor,
        *,
        gs: int = DEFAULT_GS,
        sparse: Optional[Mapping[str, torch.Tensor]] = None,
        bias: Optional[torch.Tensor] = None,
        trainable: bool = False,
        name: str = "",
        kf: Any = None,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self.kf = kf
        self.gs = int(gs)
        self.trainable_factors = bool(trainable)

        U_packed = U_packed.contiguous().to(torch.uint8)
        V_packed = V_packed.contiguous().to(torch.uint8)
        U_scale = U_scale.detach().float().reshape(-1).contiguous()
        V_scale = V_scale.detach().float().reshape(-1).contiguous()
        pq_idx = pq_idx.contiguous().to(torch.uint8)
        pq_scale = pq_scale.detach().contiguous()
        if pq_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            pq_scale = pq_scale.float()
        codebooks = codebooks.detach().float().contiguous()

        pack = {
            "U": U_packed,
            "V": V_packed,
            "U_scale": U_scale,
            "V_scale": V_scale,
            "pq_idx": pq_idx,
            "pq_scale": pq_scale,
        }
        pq = {"gs": self.gs, "M": int(codebooks.shape[0]), "K": int(codebooks.shape[1]), "subvec": int(codebooks.shape[2])}
        out, inn, r = validate_fpa_shapes(pack, pq, rank=rank)
        self.out_features = out
        self.in_features = inn
        self.rank = r

        self.register_buffer("U_packed", U_packed)
        self.register_buffer("V_packed", V_packed)
        self.register_buffer("U_scale", U_scale)
        self.register_buffer("V_scale", V_scale)
        self.register_buffer("pq_idx", pq_idx)

        U = dequant_int4_cols(U_packed, U_scale)
        V = dequant_int4_cols(V_packed, V_scale)
        _reg(self, "U", U, trainable)
        _reg(self, "V", V, trainable)
        _reg(self, "pq_scale", pq_scale.float() if trainable else pq_scale, trainable)
        _reg(self, "codebooks", codebooks, trainable)

        if sparse and sparse.get("rows") is not None and int(sparse["rows"].numel()) > 0:
            self.register_buffer("sp_rows", sparse["rows"].to(torch.int64).contiguous())
            self.register_buffer("sp_cols", sparse["cols"].to(torch.int64).contiguous())
            self.register_buffer("sp_vals", sparse["vals"].float().contiguous())
        else:
            self.register_buffer("sp_rows", torch.zeros(0, dtype=torch.int64))
            self.register_buffer("sp_cols", torch.zeros(0, dtype=torch.int64))
            self.register_buffer("sp_vals", torch.zeros(0, dtype=torch.float32))

        if bias is not None:
            _reg(self, "bias", bias.detach().float().reshape(-1).contiguous(), trainable)
        else:
            self.bias = None

    @property
    def sparse_nnz(self) -> int:
        return int(self.sp_rows.numel())

    def distill_parameters(self) -> Iterable[nn.Parameter]:
        """U, V, pq_scale, codebooks — the compressed-domain trainables."""
        for n in ("U", "V", "pq_scale", "codebooks"):
            p = getattr(self, n)
            if isinstance(p, nn.Parameter):
                yield p

    def uv_gemv(self, x: torch.Tensor) -> torch.Tensor:
        return uv_gemv(x, self.U, self.V)

    def pq_gemv(self, x: torch.Tensor) -> torch.Tensor:
        return pq_gemv(x, self.pq_idx, self.pq_scale, self.codebooks, self.gs)

    def sparse_gemv(self, x: torch.Tensor) -> torch.Tensor:
        return sparse_gemv(x, self.sp_rows, self.sp_cols, self.sp_vals, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"FpaLinear[{self.name}] expected inn={self.in_features}, got {tuple(x.shape)}")
        y = self.uv_gemv(x)
        y = y + self.pq_gemv(x)
        if self.sparse_nnz:
            y = y + self.sparse_gemv(x)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"pq=({self.gs},{int(self.codebooks.shape[0])},{int(self.codebooks.shape[1])}), "
            f"sparse_nnz={self.sparse_nnz}, trainable={self.trainable_factors}, name={self.name!r}"
        )

    @classmethod
    def from_pack(
        cls,
        pack: Mapping[str, torch.Tensor],
        codebooks: torch.Tensor,
        *,
        gs: int,
        sparse: Optional[Mapping[str, torch.Tensor]] = None,
        trainable: bool = False,
        name: str = "",
        kf: Any = None,
        rank: Optional[int] = None,
    ) -> "FpaLinear":
        return cls(
            pack["U"],
            pack["V"],
            pack["U_scale"],
            pack["V_scale"],
            pack["pq_idx"],
            pack["pq_scale"],
            codebooks,
            gs=gs,
            sparse=sparse,
            bias=pack.get("bias"),
            trainable=trainable,
            name=name,
            kf=kf,
            rank=rank,
        )

    @classmethod
    def born(
        cls,
        in_features: int,
        out_features: int,
        rank: int = 64,
        *,
        gs: int = DEFAULT_GS,
        M: int = DEFAULT_M,
        K: int = DEFAULT_K,
        trainable: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        name: str = "born",
    ) -> "FpaLinear":
        """Process-node constructor: a Linear that exists only as FPA params (no dense W)."""
        if in_features % gs:
            raise ValueError(f"in_features={in_features} must be divisible by gs={gs}")
        if rank % 2:
            raise ValueError(f"rank={rank} must be even (packed int4 nibbles)")
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        U = torch.randn(out_features, rank, generator=g, dtype=torch.float32) / (rank ** 0.5)
        V = torch.randn(in_features, rank, generator=g, dtype=torch.float32) / (rank ** 0.5)
        U_p, U_s = quantize_int4_cols(U)
        V_p, V_s = quantize_int4_cols(V)
        G = in_features // gs
        subvec = gs // M
        codebooks = torch.randn(M, K, subvec, generator=g) * 0.02
        pq_idx = torch.randint(0, K, (out_features, G, M), generator=g, dtype=torch.int64).to(torch.uint8)
        pq_scale = torch.zeros(out_features, G, dtype=torch.float16)
        lin = cls(
            U_p, V_p, U_s, V_s, pq_idx, pq_scale, codebooks,
            gs=gs, trainable=trainable, name=name, rank=rank,
        )
        # born-as-FPA uses the float factors (not the int4 snapshot) as the live params
        with torch.no_grad():
            lin.U.copy_(U.to(dtype=lin.U.dtype))
            lin.V.copy_(V.to(dtype=lin.V.dtype))
        if device is not None:
            lin = lin.to(device)
        return lin


class FpaBake:
    """Load a ``gf17_fpa`` / ``factor_product_atlas_v1`` folder (one-tensor probe or multi)."""

    def __init__(self, folder: Pathish, trainable: bool = False, device: Optional[torch.device] = None):
        self.folder = Path(folder)
        self.manifest = read_bake_manifest(self.folder)
        if is_atex_codes_bake(self.manifest):
            raise ValueError(
                f"{self.folder} looks like a Gf17Atex .codes/.scale bake — "
                "FPA is a different packing (factor_product_atlas_v1). "
                "Do not load it through FpaBake / AtexLin interchangeably."
            )
        if not is_fpa_manifest(self.manifest):
            raise ValueError(
                f"{self.folder} manifest is not {FORMAT}/{PACKING} "
                f"(format={self.manifest.get('format')!r} packing={self.manifest.get('packing')!r})"
            )
        self.pq = pq_cfg_from_manifest(self.manifest)
        self.rank = int(self.manifest.get("rank") or 0) or None
        self.kf = self.manifest.get("kf")
        tens = load_bake_tensors(self.folder)
        stems = discover_fpa_stems(tens.keys())
        if not stems:
            raise KeyError(f"no *.U FPA tensors in {self.folder} (keys={list(tens)[:12]})")

        cb_path = self.folder / "_fpa_codebooks.npy"
        if not cb_path.is_file():
            raise FileNotFoundError(f"missing sidecar {cb_path}")
        self.codebooks = load_fpa_codebooks(cb_path, self.pq["M"], self.pq["K"], self.pq["subvec"])

        report_p = self.folder / "_fpa_report.json"
        self.report: Optional[Dict[str, Any]] = None
        if report_p.is_file():
            try:
                self.report = json.loads(report_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.report = None

        sparse_path = self.folder / "_fpa_sparse.npz"
        self.linears: Dict[str, FpaLinear] = {}
        for stem in stems:
            pack = extract_fpa_tensor_pack(tens, stem)
            sparse = load_fpa_sparse(sparse_path, tensor_name=stem or None) if sparse_path.is_file() else None
            display = stem or next(iter((self.manifest.get("tensors") or {}), ""), "tensor")
            self.linears[display] = FpaLinear.from_pack(
                pack,
                self.codebooks,
                gs=self.pq["gs"],
                sparse=sparse,
                trainable=trainable,
                name=display,
                kf=self.kf,
                rank=self.rank,
            )
        if device is not None:
            self.to(device)

    def names(self) -> Sequence[str]:
        return list(self.linears.keys())

    def linear(self, name: Optional[str] = None) -> FpaLinear:
        if name:
            if name in self.linears:
                return self.linears[name]
            for k, lin in self.linears.items():
                if k.endswith(name) or k.rstrip(".weight").endswith(name.rstrip(".weight")):
                    return lin
            raise KeyError(f"{name!r} not in {self.names()}")
        if len(self.linears) == 1:
            return next(iter(self.linears.values()))
        raise KeyError(f"bake has {len(self.linears)} tensors {self.names()}; pass name=")

    def to(self, device: Union[str, torch.device]) -> "FpaBake":
        for lin in self.linears.values():
            lin.to(device)
        self.codebooks = self.codebooks.to(device)
        return self

    def parameters(self):
        for lin in self.linears.values():
            yield from lin.parameters()


def load_fpa_linear(folder: Pathish, name: Optional[str] = None, trainable: bool = False, device: Optional[torch.device] = None) -> FpaLinear:
    """Kimahri one-liner: load a one-tensor (or named) FPA Linear from a bake folder."""
    return FpaBake(folder, trainable=trainable, device=device).linear(name)


def load_fpa_bake(folder: Pathish, trainable: bool = False, device: Optional[torch.device] = None) -> FpaBake:
    return FpaBake(folder, trainable=trainable, device=device)


if __name__ == "__main__":
    from amni.inference.gf17_fpa import _cli

    raise SystemExit(_cli())
