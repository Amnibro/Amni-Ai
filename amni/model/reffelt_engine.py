import os, gc, time
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore", message=".*experimental.*")
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from amni.core.reffelt import ReffeltNonces, Atlas4D, ReffeltConfig, ExpertRegistry, DEVICE, WDTYPE
from amni.core.codec import load_reffelt_nonces
_SILU = F.silu
_SDPA = F.scaled_dot_product_attention
def _is_gpu() -> bool:
    return DEVICE.type == "cuda"
class ReffeltRMSNorm:
    __slots__ = ("_w","eps")
    def __init__(self, w: torch.Tensor, eps: float = 1e-6):
        self._w = w.to(dtype=WDTYPE, device=DEVICE)
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self._w
class ReffeltEmbedding:
    __slots__ = ("_w","vocab","dim")
    def __init__(self, w: torch.Tensor):
        self._w = w.to(dtype=WDTYPE, device=DEVICE)
        self.vocab, self.dim = self._w.shape
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._w[ids]
    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._w)
class ReffeltAtlasCore:
    def __init__(self, atlas_dir: str, vram_budget_mb: int = 0):
        self.dir = Path(atlas_dir)
        self.cfg = ReffeltConfig.load(str(self.dir / "reffelt_config.json"))
        self.nonces = ReffeltNonces(str(self.dir / "reffelt_nonces.bin"), top_k=self.cfg.top_k, temperature=self.cfg.temperature)
        self.atlas = Atlas4D(str(self.dir / "reffelt_atlas.bin"), vram_budget_mb=vram_budget_mb)
        embed_raw = np.fromfile(str(self.dir / "embed.bin"), dtype=np.float16).reshape(self.cfg.vocab_size, self.cfg.hidden)
        self.embed = ReffeltEmbedding(torch.from_numpy(embed_raw.copy()))
        norm_raw = np.fromfile(str(self.dir / "final_norm.bin"), dtype=np.float16).reshape(self.cfg.hidden)
        self.norm = ReffeltRMSNorm(torch.from_numpy(norm_raw.copy()))
        lm_path = self.dir / "lm_head.bin"
        self._lm_head = None
        if lm_path.exists():
            lm_raw = np.fromfile(str(lm_path), dtype=np.float16).reshape(self.cfg.vocab_size, self.cfg.hidden)
            self._lm_head = torch.from_numpy(lm_raw.copy()).to(dtype=WDTYPE, device=DEVICE)
        self._pos_enc = self._build_pos_enc(self.cfg.n_buckets, self.cfg.hidden)
        expert_path = self.dir / "expert_registry.json"
        self.experts: Optional[ExpertRegistry] = ExpertRegistry.load(str(expert_path)) if expert_path.exists() else None
        self._expert_mode = self.experts is not None and len(self.experts.profiles) > 0
        N, B, H, D = self.atlas.shape
        s = self.atlas.summary()
        print(f"  ReffeltAtlasCore ready on {DEVICE}")
        print(f"    nonces={self.cfg.n_nonces} buckets={self.cfg.n_buckets} heads={H} dim={D}")
        print(f"    atlas entries={s['total_entries']} active_nonces={s['active_nonces']} sparsity={s['sparsity']:.3f}")
        print(f"    embed={self.cfg.vocab_size}x{self.cfg.hidden} top_k={self.cfg.top_k} temp={self.cfg.temperature}")
        if self._expert_mode:
            es = self.experts.summary()
            print(f"    EXPERT MODE: {es['n_profiles']} profiles, {es['n_groups']} groups, affinity={'yes' if es['has_affinity'] else 'no'}")
            for d, sz in sorted(es['group_sizes'].items(), key=lambda x: -x[1]):
                print(f"      [{d}]: {sz} nonces")
        else:
            print(f"    expert mode: off (no registry found)")
    def _build_pos_enc(self, n_buckets: int, dim: int) -> torch.Tensor:
        pos = torch.arange(n_buckets, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(pos, inv_freq)
        pe = torch.zeros(n_buckets, dim)
        pe[:, 0::2] = angles.sin()
        pe[:, 1::2] = angles.cos()
        return pe.to(dtype=WDTYPE, device=DEVICE)
    def _token_to_bucket(self, pos: int) -> int:
        return pos % self.cfg.n_buckets
    def preload(self):
        t0 = time.perf_counter()
        self.atlas.preload_all()
        dt = time.perf_counter() - t0
        print(f"    atlas preload: {dt:.1f}s")
    def get_active_domains(self, x: torch.Tensor) -> Dict[str, float]:
        if not self._expert_mode: return {"general": 1.0}
        return self.experts.domain_weights(x, self.nonces)
    def expert_error_signal(self, baseline: torch.Tensor, expert_output: torch.Tensor) -> float:
        delta = expert_output - baseline
        dn = delta.norm().item()
        bn = baseline.norm().item()
        return dn / max(bn, 1e-8)
    def _scatter_chunk(self, chunk, bucket: int) -> torch.Tensor:
        chunk.to_gpu()
        bmask = (chunk._bucket_idx == bucket)
        if not bmask.any(): return torch.zeros(self.cfg.hidden, dtype=WDTYPE, device=DEVICE)
        matched_heads = chunk._head_idx[bmask]
        matched_vals = chunk._vals[bmask]
        hd = chunk.dim
        out = torch.zeros(self.cfg.hidden, dtype=WDTYPE, device=DEVICE)
        for j in range(matched_heads.shape[0]):
            h = int(matched_heads[j])
            out[h * hd:(h + 1) * hd] += matched_vals[j]
        return out
    def _forward_token(self, x: torch.Tensor, pos: int, active_nonces: torch.Tensor, nonce_weights: torch.Tensor) -> torch.Tensor:
        bucket = self._token_to_bucket(pos)
        if not self._expert_mode:
            return self._forward_token_flat(x, bucket, active_nonces, nonce_weights)
        domain_map = self.experts.classify_active(active_nonces, nonce_weights)
        expert_contribs = {}
        expert_weights = {}
        for domain, (nids, wts) in domain_map.items():
            contrib = torch.zeros(self.cfg.hidden, dtype=WDTYPE, device=DEVICE)
            dw_total = 0.0
            for nid, w in zip(nids, wts):
                chunk = self.atlas.get_chunk(nid)
                if chunk.n_entries == 0: continue
                sc = self._scatter_chunk(chunk, bucket)
                collab_scale = 1.0
                if len(nids) > 1:
                    for other_nid in nids:
                        collab_scale = max(collab_scale, self.experts.collab_weight(nid, other_nid)) if other_nid != nid else collab_scale
                contrib += w * collab_scale * sc
                dw_total += w * collab_scale
            if dw_total > 0:
                expert_contribs[domain] = contrib
                expert_weights[domain] = dw_total
        if not expert_contribs: return x
        total_w = sum(expert_weights.values()) or 1.0
        fused = torch.zeros(self.cfg.hidden, dtype=WDTYPE, device=DEVICE)
        for domain, contrib in expert_contribs.items():
            fused += (expert_weights[domain] / total_w) * contrib
        return x + fused
    def _forward_token_flat(self, x: torch.Tensor, bucket: int, active_nonces: torch.Tensor, nonce_weights: torch.Tensor) -> torch.Tensor:
        contrib = torch.zeros(self.cfg.hidden, dtype=WDTYPE, device=DEVICE)
        n_active = active_nonces.shape[0]
        for i in range(n_active):
            nid = int(active_nonces[i])
            w = float(nonce_weights[i])
            if w < 1e-6: continue
            chunk = self.atlas.get_chunk(nid)
            if chunk.n_entries == 0: continue
            contrib += w * self._scatter_chunk(chunk, bucket)
        return x + contrib
    def _forward_pass(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, S, D = x.shape
        for t in range(S):
            bucket = self._token_to_bucket(t)
            x[:, t] = x[:, t] + self._pos_enc[bucket]
        active_n, n_weights = self.nonces.activate(x.view(-1, D))
        self.atlas.preload_nonces(active_n.cpu().tolist())
        for t in range(S):
            x[:, t] = self._forward_token(x[:, t].squeeze(0), t, active_n, n_weights)
        for it in range(self.cfg.max_iters - 1):
            active_n, n_weights = self.nonces.activate(x[:, -1])
            x[:, -1] = self._forward_token(x[:, -1].squeeze(0), S - 1, active_n, n_weights)
        return x
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm.forward(x[:, -1:])
        return F.linear(h.squeeze(1), self._lm_head) if self._lm_head is not None else self.embed.unembed(h.squeeze(1))
    def generate(self, ids, max_new: int = 20, verbose: bool = True) -> tuple:
        id_t = (ids.to(DEVICE) if isinstance(ids, torch.Tensor) else torch.from_numpy(ids).to(DEVICE))
        times = []
        _sync = torch.cuda.synchronize if _is_gpu() else lambda: None
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            x = self.embed.forward(id_t)
            x = self._forward_pass(x)
            logits = self._logits(x)
        _sync()
        prefill_t = time.perf_counter() - t0
        nxt = int(logits[0].argmax())
        id_t = torch.cat([id_t, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        times.append(prefill_t)
        (print(f"  prefill ({ids.shape[1] if hasattr(ids,'shape') else len(ids)} tok): {prefill_t*1000:.0f}ms") if verbose else None)
        gc.disable()
        for step in range(1, max_new):
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                tok_embed = self.embed.forward(id_t[:, -1:])
                pos_idx = self._token_to_bucket(id_t.shape[1] - 1)
                tok_embed = tok_embed + self._pos_enc[pos_idx]
                active_n, n_weights = self.nonces.activate(tok_embed.view(-1, self.cfg.hidden))
                self.atlas.preload_nonces(active_n.cpu().tolist())
                out = self._forward_token(tok_embed[:, 0], id_t.shape[1] - 1, active_n, n_weights)
                for it in range(self.cfg.max_iters - 1):
                    active_n, n_weights = self.nonces.activate(out.unsqueeze(0))
                    out = self._forward_token(out, id_t.shape[1] - 1, active_n, n_weights)
                logits = self._logits(out.unsqueeze(0).unsqueeze(0))
            _sync()
            dt = time.perf_counter() - t0
            times.append(dt)
            nxt = int(logits[0].argmax())
            id_t = torch.cat([id_t, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            (print(f"  tok {step+1:3d}: {dt*1000:.1f}ms  ({1/dt:.1f} t/s)") if verbose else None)
        gc.enable()
        return id_t.cpu().numpy(), times
