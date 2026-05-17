import numpy as np, torch, json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
class BlockFingerprint:
    __slots__ = ("block_idx","categories","centroid","energy","connections")
    def __init__(self, block_idx: int):
        self.block_idx = block_idx
        self.categories: Dict[str, float] = {}
        self.centroid: Optional[np.ndarray] = None
        self.energy: float = 0.0
        self.connections: List[int] = []
    def to_dict(self) -> dict:
        return {"idx": self.block_idx, "cats": self.categories, "energy": self.energy, "conns": self.connections, "centroid": self.centroid.tolist() if self.centroid is not None else None}
    @classmethod
    def from_dict(cls, d: dict) -> 'BlockFingerprint':
        fp = cls(d["idx"])
        fp.categories = d.get("cats", {})
        fp.energy = d.get("energy", 0.0)
        fp.connections = d.get("conns", [])
        fp.centroid = np.array(d["centroid"], dtype=np.float32) if d.get("centroid") else None
        return fp
class CategoryIndex:
    __slots__ = ("categories","block_fps","n_blocks","_cat_blocks","embed_dim")
    def __init__(self):
        self.categories: Dict[str, np.ndarray] = {}
        self.block_fps: List[BlockFingerprint] = []
        self.n_blocks: int = 0
        self._cat_blocks: Dict[str, List[int]] = {}
        self.embed_dim: int = 0
    def blocks_for_category(self, cat: str) -> List[int]:
        return self._cat_blocks.get(cat, [])
    def all_categories(self) -> List[str]:
        return list(self.categories.keys())
    def save(self, path: str):
        d = {"embed_dim": self.embed_dim, "n_blocks": self.n_blocks, "categories": {k: v.tolist() for k, v in self.categories.items()}, "fingerprints": [fp.to_dict() for fp in self.block_fps], "cat_blocks": self._cat_blocks}
        with open(path, "w") as f:
            json.dump(d, f)
    @classmethod
    def load(cls, path: str) -> 'CategoryIndex':
        with open(path) as f:
            d = json.load(f)
        ci = cls()
        ci.embed_dim = d.get("embed_dim", 0)
        ci.n_blocks = d.get("n_blocks", 0)
        ci.categories = {k: np.array(v, dtype=np.float32) for k, v in d.get("categories", {}).items()}
        ci.block_fps = [BlockFingerprint.from_dict(fp) for fp in d.get("fingerprints", [])]
        ci._cat_blocks = d.get("cat_blocks", {})
        return ci
def build_category_index(embed_w: np.ndarray, block_weights: Dict[int, Dict[str, np.ndarray]], n_cats: int = 32, min_energy: float = 0.01) -> CategoryIndex:
    idx = CategoryIndex()
    idx.embed_dim = embed_w.shape[1]
    E = embed_w.astype(np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-8)
    E_norm = E / norms
    from sklearn.cluster import MiniBatchKMeans
    n_vocab = E.shape[0]
    sample_n = min(n_vocab, 50000)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_vocab, sample_n, replace=False)
    km = MiniBatchKMeans(n_clusters=n_cats, batch_size=4096, random_state=42, n_init=3)
    km.fit(E_norm[sample_idx])
    centroids = km.cluster_centers_
    labels = km.predict(E_norm)
    cat_names = [f"cat_{i:03d}" for i in range(n_cats)]
    for i, name in enumerate(cat_names):
        idx.categories[name] = centroids[i]
    all_profiles = []
    for block_idx, layers in block_weights.items():
        fp = BlockFingerprint(block_idx)
        combined_energy = np.zeros(n_cats, dtype=np.float32)
        n_mats = 0
        for lname, w in layers.items():
            if w.ndim != 2: continue
            if w.shape[1] != idx.embed_dim: continue
            W = w.astype(np.float32)
            projections = np.linalg.norm(W @ centroids.T, axis=0)
            combined_energy += projections
            n_mats += 1
        total_e = combined_energy.sum()
        fp.energy = float(total_e)
        profile = combined_energy / max(total_e, 1e-8)
        all_profiles.append(profile)
        c_norm = np.linalg.norm(combined_energy)
        fp.centroid = combined_energy / max(c_norm, 1e-8)
        idx.block_fps.append(fp)
    if all_profiles:
        profiles = np.stack(all_profiles)
        global_mean = profiles.mean(axis=0)
        global_std = profiles.std(axis=0).clip(min=1e-8)
        for bi, fp in enumerate(idx.block_fps):
            z_scores = (profiles[bi] - global_mean) / global_std
            for i in range(n_cats):
                if z_scores[i] > 0.5:
                    fp.categories[cat_names[i]] = float(profiles[bi, i])
    idx.n_blocks = len(block_weights)
    for fp in idx.block_fps:
        for cat in fp.categories:
            idx._cat_blocks.setdefault(cat, []).append(fp.block_idx)
    build_connections(idx)
    return idx
def build_connections(idx: CategoryIndex, conn_percentile: float = 75.0):
    n = len(idx.block_fps)
    all_sims = []
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ci = idx.block_fps[i].centroid
        if ci is None: continue
        for j in range(i + 1, n):
            cj = idx.block_fps[j].centroid
            if cj is None: continue
            sim = float(np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-8))
            sim_matrix[i, j] = sim_matrix[j, i] = sim
            all_sims.append(sim)
    threshold = float(np.percentile(all_sims, conn_percentile)) if all_sims else 0.9
    for i in range(n):
        conns = []
        for j in range(n):
            if i != j and sim_matrix[i, j] > threshold:
                conns.append(j)
        idx.block_fps[i].connections = conns
class NonceRouter:
    __slots__ = ("cat_idx","embed_w","_E_norm","_centroids","_cat_names","select_ratio","max_blocks","min_blocks","_always_blocks")
    def __init__(self, cat_idx: CategoryIndex, embed_w: np.ndarray, threshold: float = 0.15, max_blocks: int = 0, min_blocks: int = 4, select_ratio: float = 0.4):
        self.cat_idx = cat_idx
        self.embed_w = embed_w.astype(np.float32)
        norms = np.linalg.norm(self.embed_w, axis=1, keepdims=True).clip(min=1e-8)
        self._E_norm = self.embed_w / norms
        self._centroids = np.stack(list(cat_idx.categories.values()))
        self._cat_names = list(cat_idx.categories.keys())
        self.select_ratio = select_ratio
        self.max_blocks = max_blocks if max_blocks > 0 else cat_idx.n_blocks
        self.min_blocks = min_blocks
        self._always_blocks = set()
        for fp in cat_idx.block_fps:
            if fp.block_idx < 2 or fp.block_idx >= cat_idx.n_blocks - 2:
                self._always_blocks.add(fp.block_idx)
    def route(self, token_ids: List[int]) -> List[int]:
        if not token_ids: return list(range(self.cat_idx.n_blocks))
        valid_ids = [t for t in token_ids if 0 <= t < len(self._E_norm)]
        if not valid_ids: return list(range(self.cat_idx.n_blocks))
        tok_vecs = self._E_norm[valid_ids]
        prompt_vec = tok_vecs.mean(axis=0)
        pnorm = np.linalg.norm(prompt_vec)
        if pnorm < 1e-8: return list(range(self.cat_idx.n_blocks))
        prompt_vec /= pnorm
        cat_sims = self._centroids @ prompt_vec
        cat_weights = np.maximum(cat_sims, 0)
        cat_name_to_idx = {n: i for i, n in enumerate(self._cat_names)}
        block_scores = {}
        for fp in self.cat_idx.block_fps:
            score = 0.0
            for cat, weight in fp.categories.items():
                ci = cat_name_to_idx.get(cat, -1)
                if ci >= 0:
                    score += weight * cat_weights[ci]
            block_scores[fp.block_idx] = score
        sorted_blocks = sorted(block_scores.items(), key=lambda x: -x[1])
        best_score = sorted_blocks[0][1] if sorted_blocks else 0
        score_threshold = best_score * self.select_ratio if best_score > 0 else 0
        selected = set(self._always_blocks)
        for bidx, score in sorted_blocks:
            if score >= score_threshold:
                selected.add(bidx)
                for conn in self.cat_idx.block_fps[bidx].connections:
                    if block_scores.get(conn, 0) >= score_threshold * 0.5:
                        selected.add(conn)
        while len(selected) < self.min_blocks and sorted_blocks:
            bidx, _ = sorted_blocks.pop(0)
            selected.add(bidx)
        if len(selected) > self.max_blocks:
            always = selected & self._always_blocks
            rest = [(b, block_scores.get(b, 0)) for b in selected - always]
            rest.sort(key=lambda x: -x[1])
            selected = always | {b for b, _ in rest[:self.max_blocks - len(always)]}
        return sorted(selected)
    def route_torch(self, token_ids: torch.Tensor) -> List[int]:
        ids = token_ids.flatten().cpu().tolist()
        return self.route(ids)
