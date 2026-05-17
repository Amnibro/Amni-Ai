import numpy as np, torch, json, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import OrderedDict
from amni.core.codec import (load_reffelt_nonces, load_4d_atlas, save_reffelt_nonces,
    save_4d_atlas, load_truth_atlas, save_truth_atlas, load_term_ontology,
    save_term_ontology, load_delta_weights, save_delta_weights,
    save_expert_registry, load_expert_registry)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
WDTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
DEFAULT_DOMAINS = ["general","code","math","science","language","art","history","logic","creative","technical"]
class ReffeltNonces:
    __slots__ = ("n_nonces","hidden","_nonces","_nonces_norm","_active","_scores","top_k","temperature")
    def __init__(self, path: str, top_k: int = 32, temperature: float = 0.1):
        raw = load_reffelt_nonces(path)
        self.n_nonces, self.hidden = raw.shape
        self._nonces = torch.from_numpy(raw.copy()).to(dtype=WDTYPE, device=DEVICE)
        nrm = self._nonces.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._nonces_norm = self._nonces / nrm
        self._active: Optional[torch.Tensor] = None
        self._scores: Optional[torch.Tensor] = None
        self.top_k = min(top_k, self.n_nonces)
        self.temperature = temperature
    def activate(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = query.view(-1, self.hidden)
        qn = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sims = qn @ self._nonces_norm.T
        scores = sims.mean(dim=0) if q.shape[0] > 1 else sims.squeeze(0)
        topv, topi = scores.topk(self.top_k)
        weights = torch.softmax(topv / self.temperature, dim=0)
        self._active = topi
        self._scores = weights
        return topi, weights
    def get_nonce_vec(self, idx: int) -> torch.Tensor:
        return self._nonces[idx]
    @property
    def active_indices(self) -> Optional[torch.Tensor]:
        return self._active
    @property
    def active_weights(self) -> Optional[torch.Tensor]:
        return self._scores
class Atlas4DChunk:
    __slots__ = ("nonce_id","_bucket_idx","_head_idx","_vals","_on_gpu","n_entries","n_heads","dim")
    def __init__(self, nonce_id: int, bucket_idx: np.ndarray, head_idx: np.ndarray, vals: np.ndarray, n_heads: int, dim: int):
        self.nonce_id = nonce_id
        self._bucket_idx = bucket_idx
        self._head_idx = head_idx
        self._vals = vals
        self._on_gpu = False
        self.n_entries = len(bucket_idx)
        self.n_heads = n_heads
        self.dim = dim
    def to_gpu(self):
        if self._on_gpu: return
        self._bucket_idx = torch.from_numpy(self._bucket_idx).to(device=DEVICE)
        self._head_idx = torch.from_numpy(self._head_idx.astype(np.int64)).to(device=DEVICE)
        self._vals = torch.from_numpy(self._vals.copy()).to(dtype=WDTYPE, device=DEVICE)
        self._on_gpu = True
    def evict(self):
        if not self._on_gpu: return
        self._bucket_idx = self._bucket_idx.cpu().numpy()
        self._head_idx = self._head_idx.cpu().numpy().astype(np.int16)
        self._vals = self._vals.cpu().numpy()
        self._on_gpu = False
    def vram_bytes(self) -> int:
        return self.n_entries * (4 + 8 + self.dim * 2) if self._on_gpu else 0
    def fetch(self, bucket_mask: torch.Tensor) -> torch.Tensor:
        self.to_gpu()
        match = bucket_mask[self._bucket_idx]
        return (self._vals[match].sum(dim=0) if match.any() else torch.zeros(self.dim, dtype=WDTYPE, device=DEVICE))
    def fetch_scatter(self, bucket_ids: torch.Tensor, out: torch.Tensor):
        self.to_gpu()
        for i in range(self.n_entries):
            b = int(self._bucket_idx[i])
            h = int(self._head_idx[i])
            out[b, h] += self._vals[i]
class Atlas4D:
    __slots__ = ("shape","val_scale","_chunks","_lru","_budget_bytes","_loaded","path")
    def __init__(self, path: str, vram_budget_mb: int = 0):
        self.path = path
        nonce_ptr, bucket_idx, head_idx, vals, shape, vs = load_4d_atlas(path)
        self.shape = shape
        self.val_scale = vs
        N, B, H, D = shape
        self._chunks: List[Atlas4DChunk] = []
        for n in range(N):
            s, e = int(nonce_ptr[n]), int(nonce_ptr[n + 1])
            ch = Atlas4DChunk(n, bucket_idx[s:e].copy(), head_idx[s:e].copy(), vals[s:e].copy(), H, D) if e > s else Atlas4DChunk(n, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int16), np.zeros((0, D), dtype=np.float16), H, D)
            self._chunks.append(ch)
        self._budget_bytes = vram_budget_mb * 1024 * 1024 if vram_budget_mb > 0 else 0
        self._lru: OrderedDict = OrderedDict()
        self._loaded: Set[int] = set()
    def ensure_nonce(self, nonce_id: int):
        if nonce_id in self._loaded:
            self._lru.move_to_end(nonce_id)
            return
        if self._budget_bytes > 0:
            while self._lru and self._used_vram() + self._chunks[nonce_id].n_entries * self.shape[3] * 2 > self._budget_bytes:
                old, _ = self._lru.popitem(last=False)
                self._chunks[old].evict()
                self._loaded.discard(old)
        self._chunks[nonce_id].to_gpu()
        self._loaded.add(nonce_id)
        self._lru[nonce_id] = True
    def _used_vram(self) -> int:
        return sum(self._chunks[n].vram_bytes() for n in self._loaded)
    def preload_all(self):
        for i, ch in enumerate(self._chunks):
            ch.to_gpu()
            self._lru[i] = True
        self._loaded = set(range(len(self._chunks)))
    def preload_nonces(self, nonce_ids: List[int]):
        for n in nonce_ids: self.ensure_nonce(n)
    def get_chunk(self, nonce_id: int) -> Atlas4DChunk:
        self.ensure_nonce(nonce_id)
        return self._chunks[nonce_id]
    def total_entries(self) -> int:
        return sum(ch.n_entries for ch in self._chunks)
    def summary(self) -> dict:
        total = sum(ch.n_entries for ch in self._chunks)
        nonempty = sum(1 for ch in self._chunks if ch.n_entries > 0)
        return {"shape": self.shape, "total_entries": total, "active_nonces": nonempty, "total_nonces": len(self._chunks), "sparsity": 1.0 - total / max(1, self.shape[0] * self.shape[1] * self.shape[2])}
class ReffeltConfig:
    __slots__ = ("n_nonces","n_buckets","n_heads","hidden","top_k","temperature","vocab_size","max_iters")
    def __init__(self, **kw):
        self.n_nonces = kw.get("n_nonces", 512)
        self.n_buckets = kw.get("n_buckets", 256)
        self.n_heads = kw.get("n_heads", 16)
        self.hidden = kw.get("hidden", 896)
        self.top_k = kw.get("top_k", 32)
        self.temperature = kw.get("temperature", 0.1)
        self.vocab_size = kw.get("vocab_size", 151936)
        self.max_iters = kw.get("max_iters", 3)
    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}
    @classmethod
    def from_dict(cls, d: dict) -> 'ReffeltConfig':
        return cls(**d)
    def save(self, path: str):
        with open(path, "w") as f: json.dump(self.to_dict(), f, indent=2)
    @classmethod
    def load(cls, path: str) -> 'ReffeltConfig':
        with open(path) as f: return cls.from_dict(json.load(f))
class ExpertProfile:
    __slots__ = ("nonce_id","domain","spec_scores","collab_affinities","strength")
    def __init__(self, nonce_id: int, domain: str = "general", spec_scores: Optional[Dict[str,float]] = None, collab_affinities: Optional[Dict[int,float]] = None, strength: float = 1.0):
        self.nonce_id = nonce_id
        self.domain = domain
        self.spec_scores = spec_scores or {d: (1.0 if d == domain else 0.0) for d in DEFAULT_DOMAINS}
        self.collab_affinities = collab_affinities or {}
        self.strength = strength
    def to_dict(self) -> dict:
        return {"nonce_id": self.nonce_id, "domain": self.domain, "spec_scores": self.spec_scores, "collab_affinities": {str(k): v for k, v in self.collab_affinities.items()}, "strength": self.strength}
    @classmethod
    def from_dict(cls, d: dict) -> 'ExpertProfile':
        ca = {int(k): v for k, v in d.get("collab_affinities", {}).items()}
        return cls(d["nonce_id"], d.get("domain", "general"), d.get("spec_scores"), ca, d.get("strength", 1.0))
    def domain_score(self, domain: str) -> float:
        return self.spec_scores.get(domain, 0.0) * self.strength
    def top_domains(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.spec_scores.items(), key=lambda x: -x[1])[:n]
class ExpertGroup:
    __slots__ = ("group_id","domain","member_nonces","fusion_mode","_weights")
    def __init__(self, group_id: int, domain: str, member_nonces: Optional[List[int]] = None, fusion_mode: str = "weighted_sum"):
        self.group_id = group_id
        self.domain = domain
        self.member_nonces = member_nonces or []
        self.fusion_mode = fusion_mode
        self._weights: Optional[torch.Tensor] = None
    def to_dict(self) -> dict:
        return {"group_id": self.group_id, "domain": self.domain, "member_nonces": self.member_nonces, "fusion_mode": self.fusion_mode}
    @classmethod
    def from_dict(cls, d: dict) -> 'ExpertGroup':
        return cls(d["group_id"], d["domain"], d.get("member_nonces", []), d.get("fusion_mode", "weighted_sum"))
    def add_nonce(self, nonce_id: int):
        if nonce_id not in self.member_nonces: self.member_nonces.append(nonce_id)
    @property
    def size(self) -> int:
        return len(self.member_nonces)
class ExpertRegistry:
    __slots__ = ("profiles","groups","_affinity_matrix","_domain_index","n_nonces","domains")
    def __init__(self, n_nonces: int = 512, domains: Optional[List[str]] = None):
        self.n_nonces = n_nonces
        self.domains = domains or list(DEFAULT_DOMAINS)
        self.profiles: Dict[int, ExpertProfile] = {}
        self.groups: Dict[str, ExpertGroup] = {}
        self._affinity_matrix: Optional[np.ndarray] = None
        self._domain_index: Dict[str, List[int]] = {d: [] for d in self.domains}
    def register(self, profile: ExpertProfile):
        self.profiles[profile.nonce_id] = profile
        if profile.domain in self._domain_index:
            nlist = self._domain_index[profile.domain]
            if profile.nonce_id not in nlist: nlist.append(profile.nonce_id)
    def build_groups(self):
        self.groups = {}
        for gid, domain in enumerate(self.domains):
            members = self._domain_index.get(domain, [])
            self.groups[domain] = ExpertGroup(gid, domain, list(members), "weighted_sum") if members else ExpertGroup(gid, domain, [], "weighted_sum")
    def classify_active(self, active_nonces: torch.Tensor, nonce_weights: torch.Tensor) -> Dict[str, Tuple[List[int], List[float]]]:
        result: Dict[str, Tuple[List[int], List[float]]] = {}
        for i in range(active_nonces.shape[0]):
            nid = int(active_nonces[i])
            w = float(nonce_weights[i])
            if w < 1e-6: continue
            prof = self.profiles.get(nid)
            domain = prof.domain if prof else "general"
            if domain not in result: result[domain] = ([], [])
            result[domain][0].append(nid)
            result[domain][1].append(w)
        return result
    def domain_weights(self, query: torch.Tensor, nonces: 'ReffeltNonces') -> Dict[str, float]:
        act_idx, act_w = nonces.activate(query)
        dw: Dict[str, float] = {}
        for i in range(act_idx.shape[0]):
            nid = int(act_idx[i])
            w = float(act_w[i])
            prof = self.profiles.get(nid)
            if not prof: continue
            for d, s in prof.spec_scores.items():
                dw[d] = dw.get(d, 0.0) + w * s
        total = sum(dw.values()) or 1.0
        return {d: v / total for d, v in dw.items()}
    def collab_weight(self, nid_a: int, nid_b: int) -> float:
        if self._affinity_matrix is not None: return float(self._affinity_matrix[nid_a, nid_b])
        pa = self.profiles.get(nid_a)
        return pa.collab_affinities.get(nid_b, 0.5) if pa else 0.5
    def set_affinity_matrix(self, mat: np.ndarray):
        self._affinity_matrix = mat
    def compute_affinity_from_coactivation(self, coact_counts: np.ndarray):
        row_sums = coact_counts.sum(axis=1, keepdims=True).clip(min=1)
        self._affinity_matrix = (coact_counts / row_sums).astype(np.float32)
    def to_dict(self) -> dict:
        return {"n_nonces": self.n_nonces, "domains": self.domains, "profiles": {str(k): v.to_dict() for k, v in self.profiles.items()}, "groups": {k: v.to_dict() for k, v in self.groups.items()}}
    @classmethod
    def from_dict(cls, d: dict) -> 'ExpertRegistry':
        reg = cls(d.get("n_nonces", 512), d.get("domains"))
        for k, v in d.get("profiles", {}).items():
            reg.register(ExpertProfile.from_dict(v))
        for k, v in d.get("groups", {}).items():
            reg.groups[k] = ExpertGroup.from_dict(v)
        return reg
    def save(self, path: str):
        save_expert_registry(path, self.to_dict(), self._affinity_matrix)
    @classmethod
    def load(cls, path: str) -> 'ExpertRegistry':
        d, mat = load_expert_registry(path)
        reg = cls.from_dict(d)
        if mat is not None: reg.set_affinity_matrix(mat)
        return reg
    def summary(self) -> dict:
        grp_sizes = {d: g.size for d, g in self.groups.items() if g.size > 0}
        return {"n_profiles": len(self.profiles), "n_groups": len(grp_sizes), "domains": self.domains, "group_sizes": grp_sizes, "has_affinity": self._affinity_matrix is not None}
