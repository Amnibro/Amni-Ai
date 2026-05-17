import numpy as np, time, json, re, hashlib
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from amni.inference.formal_logic import extract_and_reason
    _LOGIC_OK = True
except ImportError:
    _LOGIC_OK = False
from amni.compute.noncelex import NonceLexCodec, load_noncelex_ptex, rgba_to_nonce_batch, nonce_to_rgba_batch
from amni.compute.gf17_ops import dq_matmul_t, f32_rms_norm, gf17_to_float
try:
    from amni.model.adam_micro17 import MicroRouter, MicroEngine, H as MICRO_H, VOCAB as MICRO_V, CTX_STEPS, ContextPTEX, MemoryPTEX
    _MICRO_OK = True
except ImportError:
    _MICRO_OK = False
    MICRO_H = 51
try:
    from amni.model.adam import AdamModel, ADAM_CONFIGS
    _ADAM_OK = True
except ImportError:
    _ADAM_OK = False
try:
    from amni.training.foundational_filter import score_foundational, _REJECT_IDEOLOGY_RE, _REJECT_FRAMING_RE
    _FILTER_OK = True
except ImportError:
    _FILTER_OK = False
try:
    from amni.a1.asimov import AsimovLayer
    _ASIMOV_OK = True
except ImportError:
    _ASIMOV_OK = False
try:
    from amni.a1.causal_engine import CausalEngine
    _CAUSAL_OK = True
except ImportError:
    _CAUSAL_OK = False
try:
    from amni.a1.metacognition import MetacognitiveMonitor
    _META_OK = True
except ImportError:
    _META_OK = False
try:
    from amni.a1.grail import DampingLoop
    _GRAIL_OK = True
except ImportError:
    _GRAIL_OK = False
try:
    from amni.a1.triumvirate import Triumvirate
    _TRIUMPH_OK = True
except ImportError:
    _TRIUMPH_OK = False
_STOP = set("the a an is are was were be been being have has had do does did will would could should shall may might can can't won't it its it's this that these those i me my we our you your he she they them his her and or but if in on at to for of with by from as not no".split())
_CODE_RE = re.compile(r'^(import |from .+ import |def |class |#!/|\{\{|<\?php|function\s+\w+\()', re.MULTILINE)
H_DIM = MICRO_H if _MICRO_OK else 51
IDX_MAGIC = b'PTXI'
IDX_VER = 1
HYBRID_VEC_W = 0.4
HYBRID_KW_W = 0.6
MAX_CONTEXT_CHARS = 3000
MAX_AMALG_FILES = 5
PARALLEL_WORKERS = 4
GROWTH_THRESH = 0.15
def _tok(text):
    return [w for w in re.findall(r'[a-zA-Z]{2,}', text.lower()) if w not in _STOP]
def _bigrams(toks):
    return [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)]
def _extract_title(text):
    m = re.match(r'^(?:Title|Chapter|Course Unit|Lesson|Lesson Overview)[:\s]+(.+?)(?:\n|$)', text.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else ''
def _is_code(text):
    lines = text.split('\n')[:30]
    cl = sum(1 for l in lines if _CODE_RE.search(l))
    return cl >= 5 or (len(lines) > 5 and cl / len(lines) > 0.4)
def _cos_batch(query, bank):
    q = query.ravel().astype(np.float32)
    b = bank.reshape(bank.shape[0], -1).astype(np.float32)
    qn = np.linalg.norm(q)
    return (b @ q / max(qn, 1e-8)).astype(np.float32) if qn > 1e-8 else np.zeros(b.shape[0], dtype=np.float32)
def _detect_vram():
    try:
        import ctypes
        hip = ctypes.CDLL('/opt/rocm-7.2.0/lib/libamdhip64.so')
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(0)
    except Exception:
        pass
    return 0, 0
class PointerEntry:
    __slots__ = ('path', 'domain', 'vocab_size', 'n_quads', 'title', 'centroid', 'keywords')
    def __init__(self, path, domain, vocab_size, n_quads, title, centroid, keywords):
        self.path, self.domain, self.vocab_size = path, domain, vocab_size
        self.n_quads, self.title, self.centroid = n_quads, title, centroid
        self.keywords = keywords
class PtexPointerIndex:
    __slots__ = ('entries', '_centroids', '_normed', 'inv_idx', 'title_idx', 'bigram_idx', 'domain_idx', '_built')
    def __init__(self):
        self.entries: List[PointerEntry] = []
        self._centroids: Optional[np.ndarray] = None
        self._normed: Optional[np.ndarray] = None
        self.inv_idx = defaultdict(list)
        self.title_idx = defaultdict(list)
        self.bigram_idx = defaultdict(list)
        self.domain_idx = defaultdict(list)
        self._built = False
    def add(self, entry: PointerEntry):
        idx = len(self.entries)
        self.entries.append(entry)
        self.domain_idx[entry.domain].append(idx)
        for kw in set(entry.keywords):
            self.inv_idx[kw].append(idx)
        if entry.title:
            for kw in set(_tok(entry.title)):
                self.title_idx[kw].append(idx)
        for bg in set(_bigrams(entry.keywords)):
            self.bigram_idx[bg].append(idx)
    def finalize(self):
        if not self.entries:
            return
        self._centroids = np.array([e.centroid for e in self.entries], dtype=np.float32)
        norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
        self._normed = self._centroids / np.maximum(norms, 1e-8)
        self._built = True
    def vector_search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._built or self._normed is None:
            return []
        qn = query_vec.ravel().astype(np.float32)
        qnorm = np.linalg.norm(qn)
        qn = qn / max(qnorm, 1e-8) if qnorm > 1e-8 else qn
        sims = self._normed @ qn
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]
    def keyword_search(self, query: str, top_k: int = 5, domain: str = None) -> List[Tuple[int, float]]:
        qtokens = _tok(query)
        if not qtokens:
            return []
        n = len(self.entries)
        scores = Counter()
        for tok in qtokens:
            matches = self.inv_idx.get(tok, [])
            if not matches:
                continue
            idf = max(0.1, np.log(max(n, 1) / (len(matches) + 1)))
            for idx in matches:
                if domain and self.entries[idx].domain != domain:
                    continue
                scores[idx] += idf
            for idx in self.title_idx.get(tok, []):
                if domain and self.entries[idx].domain != domain:
                    continue
                scores[idx] += idf * 3.0
        for bg in _bigrams(qtokens):
            for idx in self.bigram_idx.get(bg, []):
                if domain and self.entries[idx].domain != domain:
                    continue
                scores[idx] += max(0.5, np.log(max(n, 1) / (len(self.bigram_idx.get(bg, [])) + 1))) * 4.0
        for idx in scores:
            prev_low = self.entries[idx].keywords
            qtoks_set = set(qtokens)
            if any(w in prev_low for w in qtoks_set if len(w) > 3):
                scores[idx] *= 1.3
        if not scores and domain:
            return self.keyword_search(query, top_k=top_k, domain=None)
        ranked = scores.most_common(top_k * 2)
        mx = max((s for _, s in ranked), default=1)
        return [(idx, sc / mx) for idx, sc in ranked[:top_k]]
    def hybrid_search(self, query: str, query_vec: np.ndarray, top_k: int = 5, domain: str = None) -> List[Tuple[PointerEntry, float]]:
        kw_results = self.keyword_search(query, top_k=top_k * 2, domain=domain)
        vec_results = self.vector_search(query_vec, top_k=top_k * 2) if self._built else []
        combined = Counter()
        for idx, sc in kw_results:
            combined[idx] += sc * HYBRID_KW_W
        for idx, sc in vec_results:
            combined[idx] += max(0, sc) * HYBRID_VEC_W
        ranked = combined.most_common(top_k)
        return [(self.entries[idx], sc) for idx, sc in ranked]
    @property
    def size(self):
        return len(self.entries)
    @property
    def domains(self):
        return {d: len(v) for d, v in self.domain_idx.items()}
    def save(self, path: str):
        od = Path(path)
        od.parent.mkdir(parents=True, exist_ok=True)
        paths = [e.path for e in self.entries]
        domains = [e.domain for e in self.entries]
        titles = [e.title for e in self.entries]
        vocab_sizes = np.array([e.vocab_size for e in self.entries], dtype=np.int32)
        n_quads = np.array([e.n_quads for e in self.entries], dtype=np.int32)
        centroids = np.array([e.centroid for e in self.entries], dtype=np.float32)
        kw_data = [' '.join(e.keywords[:100]) for e in self.entries]
        meta = {'magic': IDX_MAGIC.decode(), 'ver': IDX_VER, 'n_entries': len(self.entries), 'h_dim': centroids.shape[1] if len(centroids) > 0 else H_DIM, 'built_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        np.savez_compressed(str(od), centroids=centroids, vocab_sizes=vocab_sizes, n_quads=n_quads, paths=np.array(paths, dtype=object), domains=np.array(domains, dtype=object), titles=np.array(titles, dtype=object), keywords=np.array(kw_data, dtype=object), meta=json.dumps(meta))
    @classmethod
    def load(cls, path: str) -> 'PtexPointerIndex':
        d = np.load(str(path), allow_pickle=True)
        meta = json.loads(str(d['meta']))
        idx = cls()
        centroids = d['centroids']
        paths = d['paths']
        domains = d['domains']
        titles = d['titles']
        vocab_sizes = d['vocab_sizes']
        n_quads_arr = d['n_quads']
        keywords = d['keywords']
        for i in range(len(paths)):
            kws = str(keywords[i]).split() if len(keywords) > i else []
            entry = PointerEntry(str(paths[i]), str(domains[i]), int(vocab_sizes[i]), int(n_quads_arr[i]), str(titles[i]), centroids[i].astype(np.float32), kws)
            idx.add(entry)
        idx.finalize()
        return idx
def _compute_centroid(ptex_path: str, router: 'MicroRouter' = None) -> Optional[Tuple[np.ndarray, int, int, str, str, list]]:
    try:
        codec, px, _ = load_noncelex_ptex(ptex_path)
        if codec.vocab_size < 50:
            return None
        preview = codec.decode(px[:300]).strip()
        if len(preview) < 20 or _is_code(preview):
            return None
        nonce_ids = rgba_to_nonce_batch(px[:min(px.shape[0], 500)])
        if router and _MICRO_OK:
            hidden = router.hidden_state(nonce_ids)
            centroid = hidden.mean(axis=0).ravel().astype(np.float32)
        else:
            centroid = np.zeros(H_DIM, dtype=np.float32)
            for nid in nonce_ids[:200]:
                centroid[int(nid) % H_DIM] += 1.0
            norm = np.linalg.norm(centroid)
            centroid = centroid / max(norm, 1e-8)
        domain = Path(ptex_path).name.split('_')[0]
        title = _extract_title(preview)
        kws = _tok(preview[:1000])
        return (centroid, codec.vocab_size, px.shape[0], domain, title, kws)
    except Exception:
        return None
def build_pointer_index(ptex_dir: str, max_files: int = 0, use_router: bool = True) -> PtexPointerIndex:
    files = sorted(Path(ptex_dir).rglob('*.nlx.ptex'))
    files = files[:max_files] if max_files > 0 else files
    print(f"  Building pointer index from {len(files)} PTEX files...", flush=True)
    router = MicroRouter() if use_router and _MICRO_OK else None
    if router:
        ckpt = Path(ptex_dir).parent / 'checkpoints' / 'rag_engine' / 'micro17_weights.npz'
        if ckpt.exists():
            d = np.load(str(ckpt))
            ws = [d[f'w{i}'] for i in range(len(d.files))]
            wi = 0
            router.embed[:] = ws[wi]; wi += 1
            for blk in router.blocks:
                for lin in [blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o, blk.mlp.gate, blk.mlp.up, blk.mlp.down]:
                    lin.w[:] = ws[wi]; wi += 1
            router.head[:] = ws[wi]
            print(f"  Loaded MicroRouter weights from checkpoint", flush=True)
    idx = PtexPointerIndex()
    t0 = time.time()
    skipped = 0
    for fi, f in enumerate(files):
        result = _compute_centroid(str(f), router)
        if result is None:
            skipped += 1
            continue
        centroid, vs, nq, dom, title, kws = result
        idx.add(PointerEntry(str(f), dom, vs, nq, title, centroid, kws))
        if (fi + 1) % 500 == 0:
            print(f"    [{fi+1}/{len(files)}] indexed={idx.size} skipped={skipped}", flush=True)
    idx.finalize()
    el = time.time() - t0
    print(f"  Done in {el:.1f}s: {idx.size} entries, {skipped} skipped, {len(idx.inv_idx)} keywords", flush=True)
    print(f"  Domains: {idx.domains}", flush=True)
    return idx
def _decode_ptex_file(path: str, query_tokens: set, max_chars: int = 2000, decode_quads: int = 5000) -> Optional[Dict]:
    try:
        if not Path(path).exists():
            return None
        codec, px, _ = load_noncelex_ptex(path)
        text = codec.decode(px[:min(px.shape[0], decode_quads)])
        paragraphs = re.split(r'\n\s*\n', text)
        scored = []
        for pi, para in enumerate(paragraphs):
            ptoks = set(_tok(para))
            overlap = len(query_tokens & ptoks)
            scored.append((overlap, pi, para.strip()))
        scored.sort(key=lambda x: (-x[0], x[1]))
        answer = ""
        for _, _, para in scored:
            if not para or len(para) < 20:
                continue
            if len(answer) + len(para) + 2 > max_chars:
                break
            answer += (para + "\n\n") if answer else para
        return {'text': answer.strip() if answer.strip() else text[:max_chars].strip(), 'path': path, 'domain': Path(path).name.split('_')[0]}
    except Exception:
        return None
class TieredInference:
    __slots__ = ('pointer_idx', '_router', '_codec', '_ctx', '_mem', '_vram_budget', '_reason_enabled', '_causal', '_metacog', '_grail', '_triumvirate')
    def __init__(self, pointer_idx: PtexPointerIndex, vram_budget_gb: float = 14.0, reason: bool = True):
        self.pointer_idx = pointer_idx
        self._router = MicroRouter() if _MICRO_OK else None
        self._codec = NonceLexCodec()
        self._ctx = ContextPTEX() if _MICRO_OK else None
        self._mem = MemoryPTEX() if _MICRO_OK else None
        self._vram_budget = int(vram_budget_gb * 1024 ** 3)
        self._reason_enabled = reason and _LOGIC_OK
        self._causal = CausalEngine() if _CAUSAL_OK else None
        self._metacog = MetacognitiveMonitor() if _META_OK else None
        try:
            self._grail = DampingLoop(None, None) if _GRAIL_OK else None
        except Exception:
            self._grail = None
        self._triumvirate = None
        if self._router:
            ckpt = Path(__file__).resolve().parent.parent.parent / 'checkpoints' / 'rag_engine' / 'micro17_weights.npz'
            if ckpt.exists():
                d = np.load(str(ckpt))
                ws = [d[f'w{i}'] for i in range(len(d.files))]
                wi = 0
                self._router.embed[:] = ws[wi]; wi += 1
                for blk in self._router.blocks:
                    for lin in [blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o, blk.mlp.gate, blk.mlp.up, blk.mlp.down]:
                        lin.w[:] = ws[wi]; wi += 1
                self._router.head[:] = ws[wi]
    def _route(self, query: str, top_k: int = MAX_AMALG_FILES) -> Tuple[List[Tuple[PointerEntry, float]], np.ndarray, float]:
        t0 = time.perf_counter()
        self._codec.build_vocab(query)
        pixels = self._codec.encode(query)
        nonce_ids = rgba_to_nonce_batch(pixels)
        query_vec = np.zeros(H_DIM, dtype=np.float32)
        if self._router and _MICRO_OK and len(nonce_ids) > 0:
            hidden = self._router.hidden_state(nonce_ids)
            query_vec = hidden.mean(axis=0).ravel().astype(np.float32)
        else:
            for nid in nonce_ids[:200]:
                query_vec[int(nid) % H_DIM] += 1.0
            norm = np.linalg.norm(query_vec)
            query_vec = query_vec / max(norm, 1e-8) if norm > 1e-8 else query_vec
        if self._ctx:
            self._ctx.push(query_vec)
        results = self.pointer_idx.hybrid_search(query, query_vec, top_k=top_k)
        route_ms = (time.perf_counter() - t0) * 1000
        return results, query_vec, route_ms
    def _amalgamate(self, query: str, routed: List[Tuple[PointerEntry, float]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict], float]:
        t0 = time.perf_counter()
        qtokens = set(_tok(query))
        per_file = max(max_chars // max(len(routed), 1), 400)
        decode_tasks = [(e.path, qtokens, per_file) for e, _ in routed]
        results = []
        with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, len(decode_tasks))) as pool:
            futures = {pool.submit(_decode_ptex_file, path, qt, mc): (path, sc) for (path, qt, mc), (_, sc) in zip(decode_tasks, routed)}
            for fut in as_completed(futures):
                path, sc = futures[fut]
                res = fut.result()
                if res:
                    res['score'] = sc
                    results.append(res)
        results.sort(key=lambda x: -x['score'])
        context = ""
        sources = []
        for r in results:
            chunk = r['text']
            if len(context) + len(chunk) + 4 > max_chars:
                remaining = max_chars - len(context) - 4
                chunk = chunk[:remaining] if remaining > 100 else ""
            if chunk:
                context += (("\n\n" + chunk) if context else chunk)
                sources.append({'domain': r['domain'], 'file': Path(r['path']).name, 'score': round(r['score'], 3)})
        amalg_ms = (time.perf_counter() - t0) * 1000
        return context.strip(), sources, amalg_ms
    def _select_model(self) -> Optional[str]:
        if not _ADAM_OK:
            return None
        free, total = _detect_vram()
        if free < 256 * 1024 * 1024:
            return None
        tiers = ["adam-small", "adam-medium", "adam-large", "adam-1"]
        chosen = None
        for t in tiers:
            if t not in ADAM_CONFIGS:
                continue
            m = AdamModel(t, max_ctx=512, auto_load=False)
            vb = m.vram_bytes()
            if vb["streaming_vram"] <= min(free, self._vram_budget):
                chosen = t
            else:
                break
        return chosen
    def _generate(self, context: str, query: str, model_cfg: str, max_gen: int = 80) -> Optional[Dict]:
        if not _ADAM_OK:
            return None
        try:
            words = _tok(context)
            freq = Counter(words)
            ranked = [w for w, _ in freq.most_common() if len(w) > 3][:180]
            q_words = _tok(query)
            prompt_text = " ".join(ranked) + " | " + " ".join(q_words)
            codec = NonceLexCodec()
            codec.build_vocab(context + " " + query)
            pixels = codec.encode(prompt_text)
            tokens = pixels.ravel().astype(np.int64)
            if len(tokens) < 4:
                return None
            m = AdamModel(model_cfg, max_ctx=512, auto_load=True)
            window = tokens[-512:] if len(tokens) > 512 else tokens
            t0 = time.perf_counter()
            generated = m.generate_sampled(window, max_len=max_gen, temp=0.8, top_p=0.9, rep_penalty=1.2) if hasattr(m, 'generate_sampled') else m.generate_cached(window, max_len=max_gen)
            gen_ms = round((time.perf_counter() - t0) * 1000, 1)
            gen_arr = generated.ravel().astype(np.uint8)
            n_quads = len(gen_arr) // 4
            if n_quads == 0:
                return None
            quads = gen_arr[:n_quads * 4].reshape(-1, 4)
            nids = rgba_to_nonce_batch(quads)
            valid = sum(1 for nid in nids if 0 < int(nid) < codec.total_nonces)
            nonce_rate = valid / max(n_quads, 1)
            text = codec.decode(quads)
            return {'text': text, 'nonce_rate': round(nonce_rate, 4), 'gen_ms': gen_ms, 'model': model_cfg}
        except Exception:
            return None
    def _safety_gate(self, query: str, context: str, generated: str, domain: str) -> Tuple[bool, Dict]:
        info = {'domain': domain, 'question': query[:100], 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if _FILTER_OK:
            fscore = score_foundational(generated)
            fscore_val = fscore[0] if isinstance(fscore, tuple) else fscore
            ideo = len(_REJECT_IDEOLOGY_RE.findall(generated))
            frame = len(_REJECT_FRAMING_RE.findall(generated))
            info['foundational_score'] = round(fscore_val, 4)
            info['ideology_hits'] = ideo
            info['framing_hits'] = frame
            if ideo > 0 or frame > 0 or fscore_val < GROWTH_THRESH:
                info['rejected'] = True
                return False, info
        if _ASIMOV_OK:
            layer = AsimovLayer()
            ok, _ = layer.check_output(generated)
            info['asimov_ok'] = ok
            if not ok:
                info['rejected'] = True
                return False, info
        info['rejected'] = False
        return True, info
    def _reason(self, context: str, query: str) -> Tuple[str, Optional[Dict]]:
        if not self._reason_enabled:
            return context, None
        try:
            result = extract_and_reason(context, query)
            prefix = result.get('enriched_prefix', '')
            enriched = (prefix + "\n\n" + context) if prefix else context
            if self._causal:
                causal_info = self._causal.detect_causal_intent(query)
                if causal_info.get('has_causal'):
                    for conc in result.get('conclusions', [])[:5]:
                        words = _tok(conc)
                        for w in words[:3]:
                            chains = self._causal.chain_forward(w, max_depth=3)
                            if chains:
                                chain_str = " → ".join(chains[0]) if chains[0] else ""
                                if chain_str:
                                    enriched = f"Causal chain: {chain_str}\n" + enriched
                                    break
                    result['causal'] = causal_info
            return enriched, result
        except Exception:
            return context, None
    def _commit_growth(self, context: str, query: str, query_vec: np.ndarray):
        if self._mem and self._router:
            self._mem.store(query_vec, label=query[:64], nonce_id=0)
    def _assess_confidence(self, answer: str, domain: str, sources: List[Dict]) -> Optional[Dict]:
        if not self._metacog:
            return None
        try:
            sig = self._metacog.assess_confidence(answer, domain=domain, knowledge_hits=len(sources), validation_score=min(len(sources)/3.0, 1.0))
            return {'confidence': round(sig.calibrated, 4), 'uncertainty': round(sig.uncertainty, 4), 'hedge': self._metacog.should_hedge_response(sig), 'seek_validation': self._metacog.should_seek_validation(sig)}
        except Exception:
            return None
    def ask(self, query: str, top_k: int = MAX_AMALG_FILES, max_chars: int = MAX_CONTEXT_CHARS, generate: bool = True, model_cfg: Optional[str] = None) -> Dict:
        t_total = time.perf_counter()
        routed, query_vec, route_ms = self._route(query, top_k=top_k)
        if not routed:
            return {'question': query, 'answer': "I don't have information about that topic.", 'sources': [], 'confidence': 0.0, 'stage': 'route', 'route_ms': round(route_ms, 1), 'total_ms': 0}
        context, sources, amalg_ms = self._amalgamate(query, routed, max_chars=max_chars)
        if not context:
            return {'question': query, 'answer': "Found files but could not extract relevant content.", 'sources': sources, 'confidence': 0.1, 'stage': 'amalgamate', 'route_ms': round(route_ms, 1), 'amalg_ms': round(amalg_ms, 1)}
        context, reason_info = self._reason(context, query)
        model_cfg = model_cfg if (generate and model_cfg in ADAM_CONFIGS) else (self._select_model() if generate else None)
        gen_result = None
        gen_ms = 0.0
        if model_cfg:
            gen_result = self._generate(context, query, model_cfg)
            if gen_result:
                gen_ms = gen_result['gen_ms']
                safe, safety_info = self._safety_gate(query, context, gen_result['text'], sources[0]['domain'] if sources else 'unknown')
                if safe:
                    self._commit_growth(context, query, query_vec)
                    total_ms = round((time.perf_counter() - t_total) * 1000, 1)
                    domain = sources[0]['domain'] if sources else 'unknown'
                    meta_info = self._assess_confidence(gen_result['text'], domain, sources)
                    r = {'question': query, 'answer': gen_result['text'], 'sources': sources, 'confidence': round(gen_result['nonce_rate'], 3), 'stage': f"generate({model_cfg})", 'route_ms': round(route_ms, 1), 'amalg_ms': round(amalg_ms, 1), 'gen_ms': gen_ms, 'total_ms': total_ms, 'nonce_rate': gen_result['nonce_rate'], 'growth': safety_info}
                    if meta_info:
                        r['metacog'] = meta_info
                    return r
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        stage_name = 'amalgamate' if not model_cfg else f"generate_fallback({model_cfg})"
        stage_name = f"reason+{stage_name}" if reason_info and reason_info.get('conclusions') else stage_name
        domain = sources[0]['domain'] if sources else 'unknown'
        meta_info = self._assess_confidence(context[:500], domain, sources)
        result = {'question': query, 'answer': context[:max_chars], 'sources': sources, 'confidence': 0.5, 'stage': stage_name, 'route_ms': round(route_ms, 1), 'amalg_ms': round(amalg_ms, 1), 'gen_ms': round(gen_ms, 1), 'total_ms': total_ms}
        if reason_info:
            result['reasoning'] = reason_info
        if meta_info:
            result['metacog'] = meta_info
        return result
    def quality_check(self, questions: List[str] = None) -> List[Dict]:
        if questions is None:
            questions = [
                "How does technology affect education?",
                "Explain base number conversions in math",
                "How to create a wreath for home decoration?",
                "What is the scientific method?",
                "How does machine learning work?",
                "What is good sportsmanship?",
                "How do courts handle legal disputes?",
                "What are neural networks?",
                "How does music theory work?",
                "What is the Academy Awards?",
            ]
        print(f"\n{'='*60}\n  TIERED INFERENCE QUALITY CHECK — {len(questions)} questions\n{'='*60}", flush=True)
        results = []
        for q in questions:
            r = self.ask(q)
            results.append(r)
            ans = r['answer'][:300].replace('\n', ' ')
            stage = r.get('stage', '?')
            print(f"\n  Q: {q}", flush=True)
            print(f"  A: {ans}{'...' if len(r['answer']) > 300 else ''}", flush=True)
            timing = f"route={r.get('route_ms',0):.0f}ms"
            timing += f" amalg={r.get('amalg_ms',0):.0f}ms" if 'amalg_ms' in r else ""
            timing += f" gen={r.get('gen_ms',0):.0f}ms" if r.get('gen_ms') else ""
            timing += f" total={r.get('total_ms',0):.0f}ms"
            src = r['sources'][0] if r['sources'] else None
            src_str = f"[{src['domain']}] {src['file'][:40]}" if src else "[no source]"
            reason_str = ""
            if 'reasoning' in r:
                ri = r['reasoning']
                reason_str = f" | premises={ri.get('premises_found',0)} derived={ri.get('derived_total',0)} conclusions={len(ri.get('conclusions',[]))} reason_ms={ri.get('total_ms',0):.0f}ms"
                if ri.get('conclusions'):
                    print(f"  DERIVED: {'; '.join(ri['conclusions'][:3])}", flush=True)
            print(f"  {src_str} | stage={stage} | {timing}{reason_str}", flush=True)
        return results
    def interactive(self):
        free, total = _detect_vram()
        vram_str = f"{free/(1024**3):.1f}/{total/(1024**3):.1f}GB" if total else "N/A"
        print(f"\n  Adam Tiered Inference — {self.pointer_idx.size} documents indexed")
        print(f"  Domains: {', '.join(sorted(self.pointer_idx.domains.keys()))}")
        print(f"  VRAM: {vram_str} | Router: {'MicroRouter' if self._router else 'keyword-only'}")
        print(f"  Logic: {'formal reasoning ON' if self._reason_enabled else 'OFF'}")
        print(f"  Type a question (or 'quit' to exit)\n")
        while True:
            try:
                q = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ('quit', 'exit', 'q'):
                break
            r = self.ask(q)
            stage = r.get('stage', '?')
            timing = f"route={r.get('route_ms',0):.0f}ms"
            timing += f" amalg={r.get('amalg_ms',0):.0f}ms" if 'amalg_ms' in r else ""
            timing += f" gen={r.get('gen_ms',0):.0f}ms" if r.get('gen_ms') else ""
            timing += f" total={r.get('total_ms',0):.0f}ms"
            print(f"\n  Adam: {r['answer'][:800]}")
            src = r['sources'][0] if r['sources'] else None
            src_str = f"[{src['domain']}]" if src else ""
            reason_str = ""
            if 'reasoning' in r:
                ri = r['reasoning']
                reason_str = f" | premises={ri.get('premises_found',0)} derived={ri.get('derived_total',0)}"
                if ri.get('conclusions'):
                    print(f"  DERIVED: {'; '.join(ri['conclusions'][:3])}")
            print(f"  {src_str} stage={stage} | {timing}{reason_str}\n")
