"""
Amni-A1 Core Compute Primitives
================================
TMU-native building blocks for the Amni-A1 architecture.

Key primitives:
  TensorLookupTable  — Replaces matmul with pre-computed texture lookups
  HybridLinear       — Auto-selects TLT vs sparse matmul by dimension size
  NonceAttention     — Nonce-proximity scored attention via TMU
  HybridMLP          — SwiGLU with LUT activations and sparse projections

Design principle: Computation IS lookup.
Every operation that CAN be a table lookup IS a table lookup.
TMU hardware does lookups for free (separate from ALU/Tensor cores).
What's left for ALU = additions. What's left for Tensor = tiny matmuls.
All three units run simultaneously.
"""

import os
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import sys as _sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from pathlib import Path

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
WDTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
_is_gpu = DEVICE.type == "cuda"
_SDPA = torch.nn.functional.scaled_dot_product_attention


# ===================================================================
#  TENSOR LOOKUP TABLE (TLT)
# ===================================================================

class TensorLookupTable:
    """
    Replaces matrix multiplication with pre-computed table lookups.

    Traditional:  y[j] = sum_i( x[i] * W[j, i] )   -- O(in * out) MADs
    TLT:          y[j] = sum_i( LUT[quant(x[i])][j] )  -- O(in) lookups + O(in) adds

    How it works:
    1. Quantize the weight matrix W to n_levels (e.g., 256 for uint8)
    2. For each quantization level q, pre-compute: centroid[q]
    3. Build LUT[n_levels, out_features] where LUT[q, j] = centroid[q] (the value)
    4. At inference:
       a. Quantize each input element x[i] to get index q_i
       b. For each input: contribution = x[i] * LUT[W_quantized[j,i], :] ... NO

    Actually, the correct TLT approach:
    1. Quantize INPUT to n_levels (e.g., 256 uint8 levels)
    2. For each (input_level, output_neuron) pair, pre-compute:
       LUT[level][j] = sum over input neurons where input=level of W[j, those_neurons]
       ... This doesn't work because we don't know which neurons have which values.

    Correct approach — PARTIAL PRODUCT LOOKUP:
    1. Quantize INPUT x to n_levels levels: q[i] = quantize(x[i]) for each input dim i
    2. For each input dimension i, pre-compute partial product table:
       PP[i][level][j] = centroid[level] * W[j, i]
       where centroid[level] = the representative value for that quantization level
    3. At inference:
       y[j] = sum_i PP[i][q[i]][j]
       = sum_i centroid[q[i]] * W[j, i]
       ≈ sum_i x[i] * W[j, i]  (with quantization error on x)

    This is n_input lookups of size out_features + n_input additions.

    BUT: PP table size = in_features * n_levels * out_features * 2 bytes (fp16)
    For 5120 * 256 * 5120 * 2 = 13.4 GB -- TOO LARGE for full layers.

    SOLUTION: Use TLT only for small dimensions (<=1024) where table fits.
    For dims <= 1024: PP = 1024 * 256 * 1024 * 2 = 512 MB -- manageable.
    For dims <= 512:  PP = 512 * 256 * 512 * 2 = 128 MB -- easy.

    For larger dimensions: Use the SPARSE MATMUL path instead.
    With 90% sparsity, only ~512 of 5120 input dims are active anyway,
    so the effective TLT size is small.

    REFINED APPROACH — Sparse TLT:
    1. Find active input dimensions (sparsity mask): k = |{i : |x[i]| > threshold}|
    2. If k <= TLT_DIM_LIMIT: Build on-the-fly TLT for just the active dimensions
    3. Otherwise: Fall back to sparse matmul

    This gives us TLT benefits even for large layers when sparsity is high.
    """

    def __init__(self, n_levels: int = 256, x_min: float = -8.0, x_max: float = 8.0):
        self.n_levels = n_levels
        self.x_min = x_min
        self.x_max = x_max
        self.scale = (n_levels - 1) / (x_max - x_min)
        # Pre-compute centroid values for each quantization level
        self.centroids = torch.linspace(x_min, x_max, n_levels,
                                        dtype=WDTYPE, device=DEVICE)

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input to level indices. O(n) ALU."""
        idx = ((x.float() - self.x_min) * self.scale).long()
        return idx.clamp(0, self.n_levels - 1)

    def build_partial_products(self, W: torch.Tensor,
                                active_dims: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """
        Build partial product table for weight matrix W.
        PP[dim][level] = centroid[level] * W[:, dim]  — but transposed for efficiency.
        Returns: (n_active_dims, n_levels, out_features)

        This pre-computation runs ONCE when weights are loaded or context changes.
        At inference time, it's pure TMU lookups.
        """
        if active_dims is not None:
            W_active = W[:, active_dims]  # (out, n_active)
        else:
            W_active = W  # (out, in)

        # PP[d, l] = centroid[l] * W_active[:, d]
        # Shape: (n_active, n_levels, out) via broadcasting
        # W_active.T is (n_active, out), centroids is (n_levels,)
        n_active = W_active.shape[1]
        out_f = W_active.shape[0]

        # Efficient: (n_levels, 1) * (1, n_active, out) via einsum
        # Result: (n_active, n_levels, out)
        pp = torch.einsum('l,oi->ilo', self.centroids, W_active)
        return pp  # (n_active, n_levels, out)

    def lookup_forward(self, x: torch.Tensor, pp_table: torch.Tensor,
                       active_dims: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        Forward pass using pre-computed partial products.

        x: (batch, in_features)
        pp_table: (n_active_dims, n_levels, out_features)
        active_dims: which input dimensions are in the table

        For each batch element:
          y[j] = sum_i pp_table[i, quant(x[active_dims[i]]), j]

        This is n_active_dims TMU lookups + n_active_dims additions.
        """
        if active_dims is not None:
            x_active = x[:, active_dims]  # (batch, n_active)
        else:
            x_active = x

        # Quantize active inputs
        q_idx = self.quantize_input(x_active)  # (batch, n_active), values 0..n_levels-1

        B, D = q_idx.shape
        out_f = pp_table.shape[2]

        # TMU GATHER: For each (batch, dim), look up pp_table[dim, q_idx[batch, dim], :]
        # This is the core TMU operation — each lookup fetches out_features values
        # Reshape for gather: pp_table is (D, n_levels, out_f)
        # We need to index into dim 1 (levels) using q_idx

        # Expand q_idx for gathering: (B, D) → (B, D, out_f)
        q_expanded = q_idx.unsqueeze(-1).expand(B, D, out_f)

        # Gather from pp_table: for each dim d, get pp_table[d, q_idx[:, d], :]
        # pp_table shape: (D, n_levels, out_f), need to gather along dim 1
        pp_expanded = pp_table.unsqueeze(0).expand(B, D, self.n_levels, out_f)

        # Advanced indexing: result[b, d, o] = pp_table[d, q_idx[b, d], o]
        batch_idx = torch.arange(B, device=DEVICE).view(B, 1, 1).expand(B, D, out_f)
        dim_idx = torch.arange(D, device=DEVICE).view(1, D, 1).expand(B, D, out_f)

        gathered = pp_table[dim_idx, q_expanded, torch.arange(out_f, device=DEVICE).view(1, 1, out_f).expand(B, D, out_f)]

        # Sum across input dimensions (ALU additions)
        y = gathered.sum(dim=1)  # (B, out_f)
        return y


# ===================================================================
#  HYBRID LINEAR — TLT for small, Sparse Matmul for large
# ===================================================================

_RG_CACHE = None
def _get_rg():
    global _RG_CACHE
    if _RG_CACHE is not None: return _RG_CACHE
    try:
        _p = str(Path(__file__).resolve().parents[2] / "demo")
        if _p not in _sys.path: _sys.path.insert(0, _p)
        from railgun_engine import dec_t, to_gpu
        from railgun_gpu_v2 import railgun_encode_gpu
        _RG_CACHE = (dec_t, to_gpu, railgun_encode_gpu)
    except ImportError:
        _RG_CACHE = (None, None, None)
    return _RG_CACHE
class HybridLinear:
    __slots__ = ("name", "in_f", "out_f", "_w", "_loaded",
                 "_tlt", "_pp_cache", "_pp_active_key",
                 "_sparsity_threshold", "_tlt_dim_limit",
                 "_rg_enc", "_rg_tier", "_rg_stream")
    def __init__(self, name: str, in_f: int, out_f: int,
                 sparsity_threshold: float = 0.01,
                 tlt_dim_limit: int = 1024, n_levels: int = 256):
        self.name = name
        self.in_f = in_f
        self.out_f = out_f
        self._w = None
        self._loaded = False
        self._tlt = TensorLookupTable(n_levels=n_levels)
        self._pp_cache = None
        self._pp_active_key = None
        self._sparsity_threshold = sparsity_threshold
        self._tlt_dim_limit = tlt_dim_limit
        self._rg_enc = None
        self._rg_tier = 0
        self._rg_stream = False
    def load_weights(self, w: torch.Tensor):
        self._w = w.to(dtype=WDTYPE, device=DEVICE)
        self._loaded = True
        self._pp_cache = None
    def load_railgun(self, enc, tier=0, stream=False):
        dec_t, to_gpu_fn, _ = _get_rg()
        if dec_t is None: raise ImportError("Railgun codec not available")
        self._rg_enc = to_gpu_fn(enc, tier, DEVICE) if 'vram' in enc else enc
        self._rg_tier = tier
        self._rg_stream = stream
        self._w = None if stream else dec_t(self._rg_enc, tier).reshape(self.out_f, self.in_f).to(WDTYPE)
        self._loaded = True
        self._pp_cache = None
    def evict_to_railgun(self):
        if self._rg_enc is None: return
        self._w = None
        self._pp_cache = None
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    @property
    def railgun_active(self) -> bool:
        return self._rg_enc is not None
    def forward(self, x: torch.Tensor,
                active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._rg_stream and self._rg_enc is not None and self._w is None:
            dec_t, _, _ = _get_rg()
            self._w = dec_t(self._rg_enc, self._rg_tier).reshape(self.out_f, self.in_f).to(WDTYPE)
        xf = x.view(-1, self.in_f).to(WDTYPE)
        active_mask = active_mask if active_mask is not None else (xf.abs().max(dim=0).values > self._sparsity_threshold)
        active_cols = active_mask.nonzero(as_tuple=True)[0]
        n_active = active_cols.shape[0]
        result = (self._forward_tlt(xf, active_cols, x.shape) if n_active <= self._tlt_dim_limit and n_active > 0
                  else self._forward_sparse(xf, active_cols, x.shape) if n_active <= int(self.in_f * 0.6) and n_active > 0
                  else F.linear(xf, self._w).view(*x.shape[:-1], self.out_f))
        if self._rg_stream and self._rg_enc is not None:
            self._w = None
            self._pp_cache = None
        return result
    def _forward_tlt(self, xf, active_cols, orig_shape):
        key = hash(tuple(active_cols.cpu().tolist()))
        if self._pp_cache is None or self._pp_active_key != key:
            self._pp_cache = self._tlt.build_partial_products(self._w, active_cols)
            self._pp_active_key = key
        return self._tlt.lookup_forward(xf, self._pp_cache, active_cols).view(*orig_shape[:-1], self.out_f)
    def _forward_sparse(self, xf, active_cols, orig_shape):
        return F.linear(torch.index_select(xf, 1, active_cols), torch.index_select(self._w, 1, active_cols)).view(*orig_shape[:-1], self.out_f)
    def evict(self):
        self._w = None
        self._loaded = False
        self._pp_cache = None
        self._rg_enc = None


# ===================================================================
#  ACTIVATION LUT (imported from tmu_engine, refined)
# ===================================================================

class ActivationLUT:
    """Pre-computed activation function as TMU lookup + interpolation."""
    def __init__(self, fn_name: str = "silu", n_bins: int = 65536,
                 x_min: float = -16.0, x_max: float = 16.0):
        self.n_bins = n_bins
        self.scale = (n_bins - 1) / (x_max - x_min)
        self.x_min = x_min
        x_vals = torch.linspace(x_min, x_max, n_bins, dtype=torch.float32)
        fn_map = {
            "silu":  lambda v: v * torch.sigmoid(v),
            "gelu":  lambda v: v * 0.5 * (1 + torch.tanh(0.7978845608 * (v + 0.044715 * v**3))),
            "relu":  lambda v: torch.clamp(v, min=0),
        }
        self._lut = fn_map.get(fn_name, fn_map["silu"])(x_vals).to(dtype=WDTYPE, device=DEVICE)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        idx_f = (x.float() - self.x_min) * self.scale
        idx_f = idx_f.clamp(0, self.n_bins - 2)
        idx_lo = idx_f.long()
        frac = (idx_f - idx_lo.float()).to(WDTYPE)
        return self._lut[idx_lo] + frac * (self._lut[idx_lo + 1] - self._lut[idx_lo])

_SILU_LUT = None
def get_silu_lut() -> ActivationLUT:
    global _SILU_LUT
    if _SILU_LUT is None:
        _SILU_LUT = ActivationLUT("silu")
    return _SILU_LUT


# ===================================================================
#  RMS NORM
# ===================================================================

def a1_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# ===================================================================
#  ROPE LUT
# ===================================================================

class A1RopeLUT:
    __slots__ = ("_sin", "_cos", "_hd")
    def __init__(self, head_dim: int, max_ctx: int = 8192):
        self._hd = head_dim
        half = head_dim // 2
        pos = torch.arange(0, max_ctx, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        angles = torch.outer(pos, inv_freq)
        self._sin = angles.sin().to(dtype=WDTYPE, device=DEVICE)
        self._cos = angles.cos().to(dtype=WDTYPE, device=DEVICE)
    def apply(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        S, half = x.shape[2], self._hd // 2
        s = self._sin[offset:offset+S].view(1, 1, S, half)
        c = self._cos[offset:offset+S].view(1, 1, S, half)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

_ROPE_CACHE: Dict[int, A1RopeLUT] = {}
def get_rope(head_dim: int) -> A1RopeLUT:
    if head_dim not in _ROPE_CACHE:
        _ROPE_CACHE[head_dim] = A1RopeLUT(head_dim)
    return _ROPE_CACHE[head_dim]


# ===================================================================
#  NONCE ATTENTION — Nonce-proximity scored attention
# ===================================================================

class NonceAttention:
    """
    Attention mechanism that uses Reffelt nonce proximity for scoring.

    Standard attention: score = Q @ K^T / sqrt(d)  — O(seq * seq * dim)
    Nonce attention:    score = f(|nonce_q - nonce_k|)  — O(seq * seq) lookups

    In practice, we use a HYBRID:
    - Position-level attention: standard SDPA with Flash kernel (tensor cores)
    - Nonce bias: TMU lookup of nonce-proximity bias added to attention scores

    This keeps the representational power of learned attention while
    adding semantic structure from the nonce system.
    The nonce bias acts as a learned semantic prior:
    tokens about "coding" naturally attend more to other "coding" tokens.
    """
    __slots__ = ("hidden", "n_heads", "n_kv_heads", "head_dim", "kv_dim",
                 "q_proj", "k_proj", "v_proj", "o_proj",
                 "_rope", "_reps", "_kv_cache_k", "_kv_cache_v", "_kv_len")

    def __init__(self, name: str, hidden: int, n_heads: int, n_kv_heads: int,
                 sparsity_threshold: float = 0.01):
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self._reps = n_heads // n_kv_heads

        self.q_proj = HybridLinear(f"{name}.q_proj", hidden, hidden, sparsity_threshold)
        self.k_proj = HybridLinear(f"{name}.k_proj", hidden, self.kv_dim, sparsity_threshold)
        self.v_proj = HybridLinear(f"{name}.v_proj", hidden, self.kv_dim, sparsity_threshold)
        self.o_proj = HybridLinear(f"{name}.o_proj", hidden, hidden, sparsity_threshold)

        self._rope = get_rope(self.head_dim)

        # KV cache
        self._kv_cache_k: Optional[torch.Tensor] = None
        self._kv_cache_v: Optional[torch.Tensor] = None
        self._kv_len = 0

    def reset_kv(self):
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_len = 0

    def forward(self, x: torch.Tensor, use_cache: bool = True,
                sparse_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        H, Hkv, Hd = self.n_heads, self.n_kv_heads, self.head_dim

        xf = x.view(-1, D)
        q = self.q_proj.forward(xf, sparse_mask).view(B, S, H, Hd).transpose(1, 2)
        k = self.k_proj.forward(xf, sparse_mask).view(B, S, Hkv, Hd).transpose(1, 2)
        v = self.v_proj.forward(xf, sparse_mask).view(B, S, Hkv, Hd).transpose(1, 2)

        # RoPE LUT
        offset = self._kv_len
        q = self._rope.apply(q, offset)
        k = self._rope.apply(k, offset)

        # KV cache append
        if use_cache:
            if self._kv_cache_k is not None:
                k = torch.cat([self._kv_cache_k, k], dim=2)
                v = torch.cat([self._kv_cache_v, v], dim=2)
            self._kv_cache_k = k
            self._kv_cache_v = v
            self._kv_len = k.shape[2]

        # GQA expand (zero-copy)
        if self._reps > 1:
            k = k.unsqueeze(2).expand(-1, -1, self._reps, -1, -1).reshape(B, H, -1, Hd)
            v = v.unsqueeze(2).expand(-1, -1, self._reps, -1, -1).reshape(B, H, -1, Hd)

        # SDPA Flash (tensor cores)
        y = _SDPA(q, k, v, is_causal=(S > 1))

        return self.o_proj.forward(
            y.transpose(1, 2).reshape(B * S, D), sparse_mask
        ).view(B, S, D)


# ===================================================================
#  HYBRID MLP — SwiGLU with LUT activation + sparse projections
# ===================================================================

class HybridMLP:
    """
    SwiGLU MLP:  h = SiLU(gate(x)) * up(x);  y = down(h)

    All three projections use HybridLinear (TLT or sparse matmul).
    SiLU uses ActivationLUT (TMU interpolation, zero ALU).

    Intermediate sparsity: After SiLU, many elements are near zero.
    The down projection exploits this with its own sparsity mask.
    """
    def __init__(self, name: str, hidden: int, inter: int,
                 sparsity_threshold: float = 0.01):
        self.gate = HybridLinear(f"{name}.gate_proj", hidden, inter, sparsity_threshold)
        self.up   = HybridLinear(f"{name}.up_proj",   hidden, inter, sparsity_threshold)
        self.down = HybridLinear(f"{name}.down_proj",  inter, hidden, sparsity_threshold)
        self._silu = get_silu_lut()

    def forward(self, x: torch.Tensor,
                sparse_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = x.shape
        xf = x.view(-1, s[-1])

        g = self.gate.forward(xf, sparse_mask)
        u = self.up.forward(xf, sparse_mask)

        # SiLU via TMU LUT
        h = self._silu.apply(g) * u

        # Intermediate sparsity mask for down projection
        inter_mask = h.abs().max(dim=0).values > 0.01

        return self.down.forward(h, inter_mask).view(s)


# ===================================================================
#  A1 BLOCK — Single transformer block
# ===================================================================

class A1Block:
    """
    Single Amni-A1 transformer block.

    Pre-norm → NonceAttention → residual
    Pre-norm → HybridMLP → residual

    Each block carries a Reffelt nonce tag indicating its semantic domain.
    If the block's domain is irrelevant to the current query,
    the entire block is skipped (pure residual passthrough = zero cost).
    """
    def __init__(self, name: str, hidden: int, n_heads: int, n_kv_heads: int,
                 inter: int, sparsity_threshold: float = 0.01):
        self.name = name
        self.attn = NonceAttention(f"{name}.self_attn", hidden, n_heads,
                                    n_kv_heads, sparsity_threshold)
        self.mlp = HybridMLP(f"{name}.mlp", hidden, inter, sparsity_threshold)
        self.nonce = 0  # 0 = general, always active
        self.active = True

    def reset_kv(self):
        self.attn.reset_kv()

    def forward(self, x: torch.Tensor, use_cache: bool = True,
                sparse_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.active:
            return x  # SKIP — zero compute

        x = x + self.attn.forward(a1_rms_norm(x), use_cache, sparse_mask)
        x = x + self.mlp.forward(a1_rms_norm(x), sparse_mask)
        return x

    def load_synthetic_weights(self, hidden: int, inter: int, n_kv_heads: int, n_heads: int):
        """Load random weights for benchmarking."""
        hd = hidden // n_heads
        kv_dim = n_kv_heads * hd
        self.attn.q_proj.load_weights(torch.randn(hidden, hidden))
        self.attn.k_proj.load_weights(torch.randn(kv_dim, hidden))
        self.attn.v_proj.load_weights(torch.randn(kv_dim, hidden))
        self.attn.o_proj.load_weights(torch.randn(hidden, hidden))
        self.mlp.gate.load_weights(torch.randn(inter, hidden))
        self.mlp.up.load_weights(torch.randn(inter, hidden))
        self.mlp.down.load_weights(torch.randn(hidden, inter))
