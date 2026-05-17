import numpy as np, time, json, hashlib
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from amni.compute.gf17_ops import (P, gf17_matmul_t, gf17_add, gf17_rms_norm, gf17_init_weights, gf17_to_float, gf17_fused_mlp, dq_matmul_t, f32_rms_norm, float_to_gf17)
from amni.compute.noncelex import (NonceLexCodec, nonce_to_rgba, rgba_to_nonce, nonce_to_rgba_batch, rgba_to_nonce_batch, save_noncelex_ptex, load_noncelex_ptex, CHAR_MAX, WORD_BASE, MAX_NONCE)
try:
    from amni.compute.ptex_encyclopedia import query_encyclopedia, EncyclopediaIndex, EncyclopediaManifest
    _ENCYC_OK = True
except Exception:
    _ENCYC_OK = False
H = 51
N_HEADS = 3
N_KV_HEADS = 1
HEAD_DIM = 17
INTER = 102
N_BLOCKS = 3
VOCAB = 17
MAX_PARAM = P ** 4
CTX_STEPS = 92
MEM_SLOTS = 278
CONF_CTX = 0.9
CONF_MEM = 0.8
CONF_ENC = 0.6
class MicroLinear:
    __slots__ = ('in_f', 'out_f', 'w')
    def __init__(self, in_f: int, out_f: int):
        self.in_f, self.out_f = in_f, out_f
        self.w = gf17_init_weights((out_f, in_f), "centered")
    def forward_f32(self, x, s=2.0):
        return dq_matmul_t(x.reshape(-1, self.in_f), self.w, s).reshape(*x.shape[:-1], self.out_f)
    def param_count(self):
        return self.in_f * self.out_f
class MicroAttn:
    __slots__ = ('q', 'k', 'v', 'o', 'n_heads', 'n_kv', 'hd', '_reps')
    def __init__(self):
        self.n_heads, self.n_kv, self.hd = N_HEADS, N_KV_HEADS, HEAD_DIM
        self._reps = self.n_heads // self.n_kv
        self.q = MicroLinear(H, N_HEADS * HEAD_DIM)
        self.k = MicroLinear(H, N_KV_HEADS * HEAD_DIM)
        self.v = MicroLinear(H, N_KV_HEADS * HEAD_DIM)
        self.o = MicroLinear(N_HEADS * HEAD_DIM, H)
    def forward_f32(self, x, s=2.0):
        B = 1
        S = x.shape[0] if x.ndim == 2 else 1
        xin = x.reshape(1, S, H) if x.ndim <= 2 else x
        q = self.q.forward_f32(xin, s).reshape(B, S, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k.forward_f32(xin, s).reshape(B, S, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        v = self.v.forward_f32(xin, s).reshape(B, S, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        k, v = (np.repeat(k, self._reps, axis=1), np.repeat(v, self._reps, axis=1)) if self._reps > 1 else (k, v)
        sc = (q @ k.transpose(0, 1, 3, 2) / np.float32(np.sqrt(self.hd))).astype(np.float32)
        sc = sc + np.triu(np.full((S, S), np.float32(-1e9)), k=1) if S > 1 else sc
        aw = np.exp(sc - sc.max(axis=-1, keepdims=True))
        aw = (aw / aw.sum(axis=-1, keepdims=True)).astype(np.float32)
        out = self.o.forward_f32((aw @ v).transpose(0, 2, 1, 3).reshape(B, S, N_HEADS * HEAD_DIM), s)
        return out.reshape(S, H) if x.ndim == 2 else out
    def param_count(self):
        return sum(p.param_count() for p in [self.q, self.k, self.v, self.o])
class MicroMLP:
    __slots__ = ('gate', 'up', 'down')
    def __init__(self):
        self.gate = MicroLinear(H, INTER)
        self.up = MicroLinear(H, INTER)
        self.down = MicroLinear(INTER, H)
    def forward_f32(self, x, s=2.0):
        g = self.gate.forward_f32(x, s)
        u = self.up.forward_f32(x, s)
        return self.down.forward_f32((g / (1.0 + np.exp(-np.clip(g, -20, 20)))) * u, s)
    def param_count(self):
        return sum(p.param_count() for p in [self.gate, self.up, self.down])
class MicroBlock:
    __slots__ = ('attn', 'mlp')
    def __init__(self):
        self.attn = MicroAttn()
        self.mlp = MicroMLP()
    def forward_f32(self, x, s=2.0):
        x = x + self.attn.forward_f32(f32_rms_norm(x), s)
        return x + self.mlp.forward_f32(f32_rms_norm(x), s)
    def param_count(self):
        return self.attn.param_count() + self.mlp.param_count()
class MicroRouter:
    __slots__ = ('embed', 'blocks', 'head')
    def __init__(self):
        self.embed = gf17_init_weights((VOCAB, H), "centered")
        self.blocks = [MicroBlock() for _ in range(N_BLOCKS)]
        self.head = gf17_init_weights((VOCAB, H), "centered")
    def forward_f32(self, nonce_ids: np.ndarray, s=2.0) -> np.ndarray:
        gf_ids = (nonce_ids % VOCAB).astype(np.uint8)
        x = gf17_to_float(self.embed[gf_ids], scale=2.0)
        for blk in self.blocks:
            x = blk.forward_f32(x, s)
        logits = dq_matmul_t(f32_rms_norm(x.reshape(-1, H)), self.head, s)
        return logits
    def hidden_state(self, nonce_ids: np.ndarray, s=2.0) -> np.ndarray:
        gf_ids = (nonce_ids % VOCAB).astype(np.uint8)
        x = gf17_to_float(self.embed[gf_ids], scale=2.0)
        for blk in self.blocks:
            x = blk.forward_f32(x, s)
        return f32_rms_norm(x.reshape(-1, H))
    def param_count(self):
        return VOCAB * H * 2 + sum(b.param_count() for b in self.blocks)
def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel().astype(np.float32), b.ravel().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    na, nb = np.linalg.norm(a_flat), np.linalg.norm(b_flat)
    return float(dot / max(na * nb, 1e-8))
def _cos_sim_batch(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = query.ravel().astype(np.float32)
    b = bank.reshape(bank.shape[0], -1).astype(np.float32)
    qn = q / max(np.linalg.norm(q), 1e-8)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return (bn @ qn).astype(np.float32)
class ContextPTEX:
    __slots__ = ('_buf', '_pos', '_cap', '_dim')
    def __init__(self, capacity: int = CTX_STEPS, dim: int = H):
        self._cap, self._dim = capacity, dim
        self._buf = np.zeros((capacity, dim), dtype=np.float32)
        self._pos = 0
    def push(self, vec: np.ndarray):
        self._buf[self._pos % self._cap] = vec.ravel()[:self._dim]
        self._pos += 1
    def query(self, vec: np.ndarray) -> Tuple[float, int]:
        used = min(self._pos, self._cap)
        if used == 0:
            return 0.0, -1
        sims = _cos_sim_batch(vec, self._buf[:used])
        best = int(np.argmax(sims))
        return float(sims[best]), best
    def get(self, idx: int) -> np.ndarray:
        return self._buf[idx % self._cap].copy()
    @property
    def size(self):
        return min(self._pos, self._cap)
    def to_array(self) -> np.ndarray:
        return self._buf[:min(self._pos, self._cap)].copy()
    def clear(self):
        self._buf[:] = 0
        self._pos = 0
class MemoryPTEX:
    __slots__ = ('_vecs', '_labels', '_nonces', '_cap')
    def __init__(self, capacity: int = MEM_SLOTS):
        self._vecs: List[np.ndarray] = []
        self._labels: List[str] = []
        self._nonces: List[int] = []
        self._cap = capacity
    def store(self, vec: np.ndarray, label: str = "", nonce_id: int = 0):
        if len(self._vecs) >= self._cap:
            sims = _cos_sim_batch(vec, np.array(self._vecs))
            worst = int(np.argmin(sims))
            self._vecs[worst] = vec.ravel().copy()
            self._labels[worst] = label
            self._nonces[worst] = nonce_id
        else:
            self._vecs.append(vec.ravel().copy())
            self._labels.append(label)
            self._nonces.append(nonce_id)
    def query(self, vec: np.ndarray) -> Tuple[float, int, str]:
        if not self._vecs:
            return 0.0, -1, ""
        bank = np.array(self._vecs)
        sims = _cos_sim_batch(vec, bank)
        best = int(np.argmax(sims))
        return float(sims[best]), self._nonces[best], self._labels[best]
    def get_vec(self, idx: int) -> np.ndarray:
        return self._vecs[idx].copy() if 0 <= idx < len(self._vecs) else np.zeros(H, dtype=np.float32)
    @property
    def size(self):
        return len(self._vecs)
    def clear(self):
        self._vecs.clear()
        self._labels.clear()
        self._nonces.clear()
class LearningsPTEX:
    __slots__ = ('_corrections',)
    def __init__(self):
        self._corrections: List[Dict] = []
    def add_correction(self, query_vec: np.ndarray, wrong_nonce: int, correct_nonce: int, label: str = ""):
        self._corrections.append({'q': query_vec.ravel().copy(), 'wrong': wrong_nonce, 'correct': correct_nonce, 'label': label, 'ts': time.time()})
    def check(self, vec: np.ndarray, candidate_nonce: int) -> Optional[int]:
        for c in reversed(self._corrections):
            sim = _cos_sim(vec, c['q'])
            if sim > 0.85 and c['wrong'] == candidate_nonce:
                return c['correct']
        return None
    @property
    def size(self):
        return len(self._corrections)
    def clear(self):
        self._corrections.clear()
class MicroEngine:
    __slots__ = ('router', 'codec', 'ctx', 'mem', 'learn', '_enc_dir', '_enc_idx')
    def __init__(self, enc_dir: Optional[str] = None):
        self.router = MicroRouter()
        self.codec = NonceLexCodec()
        self.ctx = ContextPTEX()
        self.mem = MemoryPTEX()
        self.learn = LearningsPTEX()
        self._enc_dir = enc_dir
        self._enc_idx = None
        if enc_dir and _ENCYC_OK:
            idx_path = str(Path(enc_dir) / "encyclopedia_index.bin")
            if Path(idx_path).exists():
                self._enc_idx = EncyclopediaIndex(idx_path)
                self._enc_idx.load()
    def ingest_text(self, text: str):
        self.codec.build_vocab(text)
    def query(self, text: str) -> Dict:
        pixels = self.codec.encode(text)
        if pixels.size == 0:
            return {'answer': None, 'confidence': 0.0, 'source': 'none'}
        nonce_ids = rgba_to_nonce_batch(pixels)
        hidden = self.router.hidden_state(nonce_ids)
        query_vec = hidden.mean(axis=0) if hidden.ndim == 2 else hidden.ravel()
        self.ctx.push(query_vec)
        ctx_sim, ctx_idx = self.ctx.query(query_vec)
        if ctx_sim >= CONF_CTX and ctx_idx >= 0:
            ctx_vec = self.ctx.get(ctx_idx)
            override = self.learn.check(query_vec, ctx_idx)
            return {'answer': ctx_vec, 'confidence': ctx_sim, 'source': 'context', 'nonce_id': override if override else ctx_idx}
        mem_sim, mem_nonce, mem_label = self.mem.query(query_vec)
        if mem_sim >= CONF_MEM:
            mem_vec = self.mem.get_vec(int(np.argmax(_cos_sim_batch(query_vec, np.array(self.mem._vecs)))))
            override = self.learn.check(query_vec, mem_nonce)
            return {'answer': mem_vec, 'confidence': mem_sim, 'source': 'memory', 'nonce_id': override if override else mem_nonce, 'label': mem_label}
        if self._enc_dir and _ENCYC_OK:
            best_nonce = int(nonce_ids[0]) if len(nonce_ids) > 0 else 0
            centroid = query_vec[:512] if query_vec.size >= 512 else np.pad(query_vec, (0, 512 - query_vec.size))
            enc_result = query_encyclopedia(self._enc_dir, best_nonce, centroid)
            if enc_result is not None:
                enc_sim = _cos_sim(query_vec, enc_result[:H])
                if enc_sim >= CONF_ENC:
                    override = self.learn.check(query_vec, best_nonce)
                    return {'answer': enc_result, 'confidence': enc_sim, 'source': 'encyclopedia', 'nonce_id': override if override else best_nonce}
        return {'answer': None, 'confidence': 0.0, 'source': 'none'}
    def update_context(self, text: str):
        pixels = self.codec.encode(text)
        if pixels.size == 0:
            return
        nonce_ids = rgba_to_nonce_batch(pixels)
        hidden = self.router.hidden_state(nonce_ids)
        self.ctx.push(hidden.mean(axis=0) if hidden.ndim == 2 else hidden.ravel())
    def commit_to_memory(self, text: str, label: str = "", nonce_id: int = 0):
        pixels = self.codec.encode(text)
        if pixels.size == 0:
            return
        nonce_ids = rgba_to_nonce_batch(pixels)
        hidden = self.router.hidden_state(nonce_ids)
        vec = hidden.mean(axis=0) if hidden.ndim == 2 else hidden.ravel()
        self.mem.store(vec, label=label or text[:64], nonce_id=nonce_id)
    def add_correction(self, query_text: str, wrong_nonce: int, correct_nonce: int):
        pixels = self.codec.encode(query_text)
        if pixels.size == 0:
            return
        nonce_ids = rgba_to_nonce_batch(pixels)
        hidden = self.router.hidden_state(nonce_ids)
        vec = hidden.mean(axis=0) if hidden.ndim == 2 else hidden.ravel()
        self.learn.add_correction(vec, wrong_nonce, correct_nonce, label=query_text[:64])
    def save_state(self, out_dir: str):
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        ws = []
        ws.append(self.router.embed)
        for blk in self.router.blocks:
            for lin in [blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o, blk.mlp.gate, blk.mlp.up, blk.mlp.down]:
                ws.append(lin.w)
        ws.append(self.router.head)
        np.savez_compressed(str(od / "micro17_weights.npz"), **{f'w{i}': w for i, w in enumerate(ws)})
        ctx_data = self.ctx.to_array()
        np.savez_compressed(str(od / "context.npz"), buf=ctx_data, pos=np.array([self.ctx._pos]))
        if self.mem.size > 0:
            np.savez_compressed(str(od / "memory.npz"), vecs=np.array(self.mem._vecs), nonces=np.array(self.mem._nonces), labels=np.array(self.mem._labels, dtype=object))
        if self.learn.size > 0:
            learn_data = json.dumps([{'wrong': c['wrong'], 'correct': c['correct'], 'label': c['label'], 'ts': c['ts']} for c in self.learn._corrections])
            (od / "learnings.json").write_text(learn_data)
            if self.learn._corrections:
                np.savez_compressed(str(od / "learnings_vecs.npz"), **{f'q{i}': c['q'] for i, c in enumerate(self.learn._corrections)})
        save_noncelex_ptex(str(od / "vocab.nlx.ptex"), self.codec, np.zeros((1, 4), dtype=np.uint8))
        meta = {'h': H, 'heads': N_HEADS, 'kv': N_KV_HEADS, 'hd': HEAD_DIM, 'inter': INTER, 'blocks': N_BLOCKS, 'vocab': VOCAB, 'params': self.router.param_count(), 'ctx_size': self.ctx.size, 'mem_size': self.mem.size, 'learn_size': self.learn.size, 'codec_vocab': self.codec.vocab_size, 'saved_at': time.time()}
        (od / "meta.json").write_text(json.dumps(meta, indent=2))
    def load_state(self, state_dir: str) -> bool:
        sd = Path(state_dir)
        wp = sd / "micro17_weights.npz"
        if wp.exists():
            d = np.load(str(wp))
            ws = [d[f'w{i}'] for i in range(len(d.files))]
            idx = 0
            self.router.embed[:] = ws[idx]; idx += 1
            for blk in self.router.blocks:
                for lin in [blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o, blk.mlp.gate, blk.mlp.up, blk.mlp.down]:
                    lin.w[:] = ws[idx]; idx += 1
            self.router.head[:] = ws[idx]
        cp = sd / "context.npz"
        if cp.exists():
            d = np.load(str(cp))
            buf = d['buf']
            n = min(buf.shape[0], self.ctx._cap)
            self.ctx._buf[:n] = buf[:n]
            self.ctx._pos = int(d['pos'][0])
        mp = sd / "memory.npz"
        if mp.exists():
            d = np.load(str(mp), allow_pickle=True)
            vecs = d['vecs']
            nonces = d['nonces']
            labels = d['labels']
            for i in range(len(vecs)):
                self.mem._vecs.append(vecs[i])
                self.mem._nonces.append(int(nonces[i]))
                self.mem._labels.append(str(labels[i]))
        vp = sd / "vocab.nlx.ptex"
        if vp.exists():
            loaded_codec, _, _ = load_noncelex_ptex(str(vp))
            self.codec = loaded_codec
        lp = sd / "learnings.json"
        lvp = sd / "learnings_vecs.npz"
        if lp.exists() and lvp.exists():
            entries = json.loads(lp.read_text())
            vd = np.load(str(lvp))
            for i, e in enumerate(entries):
                qk = f'q{i}'
                qv = vd[qk] if qk in vd else np.zeros(H, dtype=np.float32)
                self.learn.add_correction(qv, e['wrong'], e['correct'], e.get('label', ''))
        return True
    def stats(self) -> Dict:
        return {'model_params': self.router.param_count(), 'max_params': MAX_PARAM, 'param_usage': f"{self.router.param_count()/MAX_PARAM*100:.1f}%", 'context_size': self.ctx.size, 'context_cap': CTX_STEPS, 'memory_size': self.mem.size, 'memory_cap': MEM_SLOTS, 'learnings': self.learn.size, 'vocab_words': self.codec.vocab_size, 'vocab_total': self.codec.total_nonces, 'enc_dir': self._enc_dir}
