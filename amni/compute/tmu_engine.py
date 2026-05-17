import os
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
WDTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
_is_gpu = DEVICE.type == "cuda"
_COMPUTE_STREAM = torch.cuda.Stream() if _is_gpu else None
_FETCH_STREAM   = torch.cuda.Stream() if _is_gpu else None
_SDPA = torch.nn.functional.scaled_dot_product_attention
_SILU = torch.nn.functional.silu
class ActivationLUT:
    def __init__(self, fn_name: str = "silu", n_bins: int = 65536,
                 x_min: float = -16.0, x_max: float = 16.0):
        self.fn_name = fn_name
        self.n_bins  = n_bins
        self.x_min   = x_min
        self.x_max   = x_max
        self.scale   = (n_bins - 1) / (x_max - x_min)
        x_vals = torch.linspace(x_min, x_max, n_bins, dtype=torch.float32)
        fn_map = {
            "silu":  lambda x: x * torch.sigmoid(x),
            "gelu":  lambda x: x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3))),
            "relu":  lambda x: torch.clamp(x, min=0),
            "swish": lambda x: x * torch.sigmoid(x),
        }
        fn = fn_map.get(fn_name, fn_map["silu"])
        self._lut = fn(x_vals).to(dtype=WDTYPE, device=DEVICE)
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        idx_f = (x.float() - self.x_min) * self.scale
        idx_f = idx_f.clamp(0, self.n_bins - 2)
        idx_lo = idx_f.long()
        frac   = (idx_f - idx_lo.float()).to(WDTYPE)
        v_lo = self._lut[idx_lo]
        v_hi = self._lut[idx_lo + 1]
        return v_lo + frac * (v_hi - v_lo)
_ACT_LUTS: Dict[str, ActivationLUT] = {}
def get_activation_lut(name: str) -> ActivationLUT:
    if name not in _ACT_LUTS:
        _ACT_LUTS[name] = ActivationLUT(name)
    return _ACT_LUTS[name]
try:
    import amni_kernels as _ak
    _HAS_AK = True
except ImportError:
    _HAS_AK = False
class TmuLinear:
    __slots__ = ("name", "in_f", "out_f", "_w", "_sparsity_threshold",
                 "_bin_path", "_loaded", "_load_lock", "_gf17_page", "_gf17_scale",
                 "_gf17_bias", "_use_gf17")
    def __init__(self, name: str, in_f: int, out_f: int,
                 sparsity_threshold: float = 0.01):
        self.name = name
        self.in_f  = in_f
        self.out_f = out_f
        self._w: Optional[torch.Tensor] = None
        self._sparsity_threshold = sparsity_threshold
        self._bin_path: Optional[Path] = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._gf17_page: Optional[np.ndarray] = None
        self._gf17_scale = 1.0
        self._gf17_bias = 0.0
        self._use_gf17 = False
    def load_from_tensor(self, w: torch.Tensor):
        self._w = w.to(dtype=WDTYPE, device=DEVICE)
        self._loaded = True
    def load_from_disk(self, path: Path):
        self._bin_path = path
        if path.exists():
            raw = np.fromfile(str(path), dtype=np.float16)
            self._w = torch.from_numpy(raw.reshape(self.out_f, self.in_f).copy()
                                       ).to(dtype=WDTYPE, device=DEVICE)
            self._loaded = True
    def load_from_catalog(self, catalog, tex_mgr):
        from amni.model.gpu_engine import _parallel_decode_pages, _upload_u8_dequant_gpu, _upload_f32_gpu
        meta_entry = catalog.get(self.name)
        if not meta_entry or "weights" not in meta_entry:
            raise RuntimeError(f"TmuLinear: no weights for {self.name}")
        t_info = meta_entry["weights"]
        files  = t_info["files"]
        is_q   = t_info.get("quantized", False)
        scale  = float(t_info.get("scale", 1.0))
        bias_s = float(t_info.get("bias", 0.0))
        total  = self.out_f * self.in_f
        raw_pages = _parallel_decode_pages(files, total, workers=16)
        if is_q:
            self._w = _upload_u8_dequant_gpu(raw_pages, self.out_f, self.in_f, scale, bias_s)
        else:
            raw_f32 = raw_pages.view(np.float32) if raw_pages.dtype == np.uint8 else raw_pages.astype(np.float32)
            self._w = _upload_f32_gpu(raw_f32, self.out_f, self.in_f)
        self._loaded = True
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(WDTYPE), self._w)
    def forward_sparse(self, x: torch.Tensor,
                       active_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        B = x.shape[0]
        xf = x.view(-1, self.in_f).to(WDTYPE)
        if active_mask is None:
            active_mask = xf.abs().max(dim=0).values > self._sparsity_threshold
        active_cols = active_mask.nonzero(as_tuple=True)[0]
        n_active = active_cols.shape[0]
        if n_active > self.in_f * 0.6:
            return self.forward_dense(x)
        W_active = torch.index_select(self._w, 1, active_cols)
        x_active = torch.index_select(xf, 1, active_cols)     
        out = F.linear(x_active, W_active)
        return out.view(B, -1, self.out_f) if x.dim() == 3 else out
    def forward(self, x: torch.Tensor,
                active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError(f"TmuLinear {self.name}: weights not loaded")
        return self.forward_sparse(x, active_mask)
    def evict(self):
        self._w = None
        self._gf17_page = None
        self._loaded = False
        self._use_gf17 = False
    def load_gf17_page(self, page_data: np.ndarray, scale: float, bias: float):
        self._gf17_page = np.ascontiguousarray(page_data)
        self._gf17_scale = scale
        self._gf17_bias = bias
        self._use_gf17 = True
        self._loaded = True
    def forward_gf17(self, x: torch.Tensor) -> torch.Tensor:
        if self._gf17_page is None or not _HAS_AK:
            return self.forward_dense(x)
        x_np = x.detach().cpu().float().numpy().flatten()
        x_q, _, _ = _ak.gf17_quantize_f32_to_state(x_np)
        x_q = np.array(x_q)
        out_q = np.array(_ak.gf17_log_matmul(
            self._gf17_page, np.ascontiguousarray(x_q), self.out_f, self.in_f))
        out_f32 = np.array(_ak.gf17_dequantize_page(out_q, self._gf17_scale, self._gf17_bias))
        return torch.from_numpy(out_f32).to(dtype=WDTYPE, device=DEVICE).view(1, -1)
    def forward(self, x: torch.Tensor,
                active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError(f"TmuLinear {self.name}: weights not loaded")
        return self.forward_gf17(x) if self._use_gf17 else self.forward_sparse(x, active_mask)
class TmuPipeline:
    def __init__(self, layers: List[TmuLinear]):
        self.layers = layers
        self._prefetch_buffers: Dict[str, torch.Tensor] = {}
        self._fetch_events: Dict[str, torch.cuda.Event] = {}
    def prefetch_layer(self, layer: TmuLinear):
        if not _is_gpu or layer.is_loaded:
            return
        if _FETCH_STREAM is not None:
            with torch.cuda.stream(_FETCH_STREAM):
                layer.load_from_disk(layer._bin_path)
                event = torch.cuda.Event()
                event.record(_FETCH_STREAM)
                self._fetch_events[layer.name] = event
    def wait_for_prefetch(self, layer_name: str):
        if layer_name in self._fetch_events and _COMPUTE_STREAM is not None:
            _COMPUTE_STREAM.wait_event(self._fetch_events[layer_name])
            del self._fetch_events[layer_name]
    def forward_pipelined(self, x: torch.Tensor,
                          sparsity_mask: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        current = x
        for i, layer in enumerate(self.layers):
            if i + 1 < len(self.layers):
                self.prefetch_layer(self.layers[i + 1])
            self.wait_for_prefetch(layer.name)
            if _COMPUTE_STREAM is not None:
                with torch.cuda.stream(_COMPUTE_STREAM):
                    current = layer.forward(current, sparsity_mask)
            else:
                current = layer.forward(current, sparsity_mask)
        return current
    def forward_staggered_4s(self, batch: List[torch.Tensor],
                              sparsity_masks: Optional[List[torch.Tensor]] = None,
                              use_compression: bool = True) -> List[torch.Tensor]:
        from amni.compute.staggered_pipeline import StaggeredPipeline, Stage, TokenCompressor
        n = len(batch)
        if n == 0:
            return []
        compressor = TokenCompressor()
        _streams = [torch.cuda.Stream() for _ in range(4)] if _is_gpu else [None] * 4
        results = [None] * n
        layer_list = self.layers
        n_layers = len(layer_list)
        for tok_idx in range(n):
            current = batch[tok_idx]
            mask = sparsity_masks[tok_idx] if sparsity_masks and tok_idx < len(sparsity_masks) else None
            for li, layer in enumerate(layer_list):
                stream_idx = li % 4
                pf_depth = min(3, n_layers - li - 1)
                for pf in range(1, pf_depth + 1):
                    pf_li = li + pf
                    if pf_li < n_layers:
                        pf_stream = (li + pf) % 4
                        if _is_gpu and _streams[pf_stream]:
                            with torch.cuda.stream(_streams[pf_stream]):
                                self.prefetch_layer(layer_list[pf_li])
                self.wait_for_prefetch(layer.name)
                if _is_gpu and _streams[stream_idx]:
                    with torch.cuda.stream(_streams[stream_idx]):
                        current = layer.forward(current, mask)
                else:
                    current = layer.forward(current, mask)
                if use_compression and li < n_layers - 1 and current.is_floating_point():
                    q, scale = compressor.compress_i8(current, f"t{tok_idx}_l{li}")
                    current = compressor.decompress_i8(q, scale, current.dtype)
            results[tok_idx] = current
        if _is_gpu:
            torch.cuda.synchronize()
        return results
class ReffeltRouter:
    def __init__(self, nonce_threshold: int = 2000):
        self.nonce_threshold = nonce_threshold
        self.active_nonces: List[int] = []
        self._layer_nonces: Dict[str, int] = {}
    def set_active_nonces(self, nonces: List[int]):
        self.active_nonces = nonces
    def register_layer_nonce(self, layer_name: str, nonce: int):
        self._layer_nonces[layer_name] = nonce
    def is_layer_active(self, layer_name: str) -> bool:
        if not self.active_nonces:
            return True
        layer_nonce = self._layer_nonces.get(layer_name)
        if layer_nonce is None or layer_nonce == 0:
            return True
        min_dist = min(abs(layer_nonce - an) for an in self.active_nonces)
        return min_dist <= self.nonce_threshold
    def get_active_layers(self, layer_names: List[str]) -> List[str]:
        return [n for n in layer_names if self.is_layer_active(n)]
    def compute_sparsity_mask(self, x: torch.Tensor,
                              threshold: float = 0.01) -> torch.Tensor:
        return x.abs().max(dim=0).values > threshold
class LayerCollapser:
    def __init__(self, max_collapse: int = 4):
        self.max_collapse = max_collapse
        self._collapsed_tables: Dict[str, torch.Tensor] = {}
        self._collapse_keys: Dict[str, str] = {}
    def _make_key(self, layer_names: List[str],
                  nonces: Tuple[int, ...]) -> str:
        return f"{'|'.join(layer_names)}@{hash(nonces)}"
    def try_collapse(self, layers: List[TmuLinear],
                     sparsity_mask: torch.Tensor,
                     active_nonces: Tuple[int, ...]) -> Optional[torch.Tensor]:
        if len(layers) < 2 or len(layers) > self.max_collapse:
            return None
        names = [l.name for l in layers]
        key = self._make_key(names, active_nonces)
        if key in self._collapsed_tables:
            return self._collapsed_tables[key]
        W = layers[0]._w
        for i in range(1, len(layers)):
            W_next = layers[i]._w
            W = W_next @ W
        self._collapsed_tables[key] = W
        return W
    def clear(self):
        self._collapsed_tables.clear()
def tmu_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
class TmuRopeLUT:
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
_TMU_ROPE_CACHE: Dict[int, TmuRopeLUT] = {}
class TmuBenchmark:
    @staticmethod
    def measure_gather_throughput(sizes: List[Tuple[int, int]],
                                  n_iters: int = 100) -> Dict:
        results = {}
        sync = torch.cuda.synchronize if _is_gpu else lambda: None
        for (rows, cols) in sizes:
            W = torch.randn(rows, cols, dtype=WDTYPE, device=DEVICE)
            for sparsity in [0.5, 0.8, 0.9, 0.95]:
                n_active = max(1, int(cols * (1.0 - sparsity)))
                indices = torch.randperm(cols, device=DEVICE)[:n_active]
                for _ in range(5):
                    _ = torch.index_select(W, 1, indices)
                sync()
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    _ = torch.index_select(W, 1, indices)
                sync()
                dt = time.perf_counter() - t0
                samples_per_iter = rows * n_active
                total_samples = samples_per_iter * n_iters
                throughput = total_samples / dt
                key = f"{rows}x{cols}_sp{int(sparsity*100)}"
                results[key] = {
                    "shape": (rows, cols),
                    "sparsity": sparsity,
                    "n_active": n_active,
                    "samples_per_fetch": samples_per_iter,
                    "throughput_samples_sec": throughput,
                    "throughput_Gsamples_sec": throughput / 1e9,
                    "time_per_fetch_us": (dt / n_iters) * 1e6,
                    "bandwidth_GBs": (total_samples * 2) / dt / 1e9,
                }
        return results
    @staticmethod
    def measure_sparse_matmul_speedup(M: int, K: int, N: int,
                                       sparsity: float = 0.9,
                                       n_iters: int = 100) -> Dict:
        sync = torch.cuda.synchronize if _is_gpu else lambda: None
        W = torch.randn(N, K, dtype=WDTYPE, device=DEVICE)
        x = torch.randn(M, K, dtype=WDTYPE, device=DEVICE)
        mask = torch.rand(K, device=DEVICE) > sparsity
        x_masked = x * mask.unsqueeze(0)
        for _ in range(5):
            _ = F.linear(x_masked, W)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = F.linear(x_masked, W)
        sync()
        dense_time = (time.perf_counter() - t0) / n_iters
        active_cols = mask.nonzero(as_tuple=True)[0]
        n_active = active_cols.shape[0]
        W_sparse = torch.index_select(W, 1, active_cols)
        x_sparse = torch.index_select(x_masked, 1, active_cols)
        for _ in range(5):
            _ = F.linear(x_sparse, W_sparse)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = torch.index_select(W, 1, active_cols)
            x_sp = torch.index_select(x_masked, 1, active_cols)
            _ = F.linear(x_sp, torch.index_select(W, 1, active_cols))
        sync()
        sparse_time = (time.perf_counter() - t0) / n_iters
        return {
            "dense_ms": dense_time * 1000,
            "sparse_ms": sparse_time * 1000,
            "speedup": dense_time / sparse_time if sparse_time > 0 else float('inf'),
            "sparsity": sparsity,
            "active_features": n_active,
            "total_features": K,
            "dense_ops": 2 * M * K * N,
            "sparse_ops": 2 * M * n_active * N,
        }
    @staticmethod
    def measure_pipeline_benefit(n_layers: int = 48,
                                  hidden: int = 5120,
                                  inter: int = 13824,
                                  n_iters: int = 20) -> Dict:
        sync = torch.cuda.synchronize if _is_gpu else lambda: None
        layers_w = []
        for i in range(min(n_layers, 8)):
            w = torch.randn(inter, hidden, dtype=WDTYPE, device=DEVICE)
            layers_w.append(w)
        n_test = len(layers_w)
        x = torch.randn(1, hidden, dtype=WDTYPE, device=DEVICE)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            cur = x
            for w in layers_w:
                cur = F.linear(cur, w)
                cur = _SILU(cur)
                if cur.shape[-1] != hidden:
                    cur = cur[..., :hidden]
        sync()
        seq_time = (time.perf_counter() - t0) / n_iters
        if _is_gpu:
            sync()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                cur = x
                for i, w in enumerate(layers_w):
                    if i + 1 < n_test:
                        with torch.cuda.stream(_FETCH_STREAM):
                            _ = layers_w[(i+1) % n_test].data_ptr()
                    with torch.cuda.stream(_COMPUTE_STREAM):
                        cur = F.linear(cur, w)
                        cur = _SILU(cur)
                        if cur.shape[-1] != hidden:
                            cur = cur[..., :hidden]
            sync()
            pipe_time = (time.perf_counter() - t0) / n_iters
        else:
            pipe_time = seq_time
        return {
            "sequential_ms": seq_time * 1000,
            "pipelined_ms": pipe_time * 1000,
            "pipeline_speedup": seq_time / pipe_time if pipe_time > 0 else 1.0,
            "n_layers_tested": n_test,
            "projected_48L_seq_ms": (seq_time / n_test) * 48 * 1000,
            "projected_48L_pipe_ms": (pipe_time / n_test) * 48 * 1000,
        }
    @staticmethod
    def measure_lut_vs_compute(n_elements: int = 5120 * 13824,
                                n_iters: int = 200) -> Dict:
        sync = torch.cuda.synchronize if _is_gpu else lambda: None
        x = torch.randn(n_elements, dtype=WDTYPE, device=DEVICE)
        lut = get_activation_lut("silu")
        for _ in range(5):
            _ = _SILU(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = _SILU(x)
        sync()
        alu_time = (time.perf_counter() - t0) / n_iters
        for _ in range(5):
            _ = lut.apply(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = lut.apply(x)
        sync()
        lut_time = (time.perf_counter() - t0) / n_iters
        ref = _SILU(x.float()).to(WDTYPE)
        lut_out = lut.apply(x)
        max_err = (ref - lut_out).abs().max().item()
        mean_err = (ref - lut_out).abs().mean().item()
        return {
            "alu_us": alu_time * 1e6,
            "lut_us": lut_time * 1e6,
            "speedup": alu_time / lut_time if lut_time > 0 else 1.0,
            "n_elements": n_elements,
            "max_error": max_err,
            "mean_error": mean_err,
            "alu_throughput_Gsamples": n_elements / alu_time / 1e9,
            "lut_throughput_Gsamples": n_elements / lut_time / 1e9,
        }
    @staticmethod
    def measure_layer_scaling(hidden: int = 5120,
                               max_layers: int = 48,
                               sparsity: float = 0.9,
                               n_iters: int = 20) -> Dict:
        sync = torch.cuda.synchronize if _is_gpu else lambda: None
        results = {}
        n_active = max(1, int(hidden * (1.0 - sparsity)))
        for n_layers in [2, 4, 8, 16, 24, 48]:
            actual_layers = min(n_layers, max_layers)
            W = torch.randn(hidden, hidden, dtype=WDTYPE, device=DEVICE)
            indices = torch.randperm(hidden, device=DEVICE)[:n_active]
            W_sparse = torch.index_select(W, 1, indices)
            x = torch.randn(1, hidden, dtype=WDTYPE, device=DEVICE)
            for _ in range(3):
                cur = x
                for _ in range(actual_layers):
                    x_sp = torch.index_select(cur, 1, indices)
                    cur = F.linear(x_sp, W_sparse)
            sync()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                cur = x
                for _ in range(actual_layers):
                    x_sp = torch.index_select(cur, 1, indices)
                    cur = F.linear(x_sp, W_sparse)
            sync()
            dt = (time.perf_counter() - t0) / n_iters
            tps = 1.0 / dt if dt > 0 else float('inf')
            results[f"{n_layers}L"] = {
                "layers": n_layers,
                "ms_per_token": dt * 1000,
                "tokens_per_sec": tps,
                "ms_per_layer": (dt / actual_layers) * 1000,
                "sparsity": sparsity,
                "active_features": n_active,
            }
        return results
