import numpy as np, re, hashlib, time
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
ASSERT, NEG, COND, DISJ, CONJ, UNIV, EXIST, DNEG = range(8)
CONN_NAMES = {ASSERT:'assert', NEG:'neg', COND:'if→then', DISJ:'or', CONJ:'and', UNIV:'∀', EXIST:'∃', DNEG:'¬¬'}
_P = np.uint32(17)
MAX_DERIVE_ROUNDS = 3
MAX_DERIVED = 50
MIN_CONTEST_CHAIN = 1
_COND_RE = re.compile(
    r'(?:if\s+(.+?)\s*[,;]?\s*then\s+(.+?)(?:\.|$))|'
    r'(?:when\s+(.+?)\s*[,;]\s*(.+?)(?:\.|$))|'
    r'(?:(.+?)\s+(?:leads?\s+to|causes?|results?\s+in|implies?|means?\s+that)\s+(.+?)(?:\.|$))',
    re.IGNORECASE
)
_DISJ_RE = re.compile(r'(?:either\s+)?(.+?)\s+or\s+(.+?)(?:\.|$)', re.IGNORECASE)
_CONJ_RE = re.compile(r'(?:both\s+)?(.+?)\s+and\s+(.+?)(?:\.|$)', re.IGNORECASE)
_NEG_RE = re.compile(r'(?:(?:it\s+is\s+)?not\s+(?:true\s+that\s+|the\s+case\s+that\s+)?(.+?)(?:\.|$))|(?:(?:there\s+is\s+)?no\s+(.+?)(?:\.|$))|(?:(.+?)\s+(?:is\s+not|are\s+not|does\s+not|do\s+not|cannot|can\'t|won\'t|isn\'t|aren\'t|doesn\'t|don\'t)\s+(.+?)(?:\.|$))', re.IGNORECASE)
_UNIV_RE = re.compile(r'(?:all|every|each|any)\s+(.+?)\s+(?:is|are|has|have|must|will|can)\s+(.+?)(?:\.|$)', re.IGNORECASE)
_EXIST_RE = re.compile(r'(?:some|there\s+exist(?:s)?|at\s+least\s+one|certain)\s+(.+?)\s+(?:is|are|has|have)\s+(.+?)(?:\.|$)', re.IGNORECASE)
_ASSERT_RE = re.compile(r'^(.+?)\s+(?:is|are|was|were|has|have|requires?|involves?|consists?\s+of|includes?|provides?|contains?|reaches?|conducts?|produces?|receives?|flows?|turns?|makes?|comes?|goes?|runs?|works?|gets?|gives?|takes?|shows?|becomes?|remains?)\s+(.+?)(?:\.|$)', re.IGNORECASE)
def _gf17_hash(text: str) -> np.uint32:
    h = np.uint32(0)
    for b in text.encode('utf-8', errors='replace')[:128]:
        h = np.uint32((int(h) * 17 + b) % (2**31))
    return h
def _norm_phrase(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())[:120]
class Proposition:
    __slots__ = ('subj', 'pred', 'obj', 'conn', 'neg', 'subj_h', 'pred_h', 'obj_h', 'source_idx', 'chain_len', 'rule_name', 'text')
    def __init__(self, subj: str, pred: str, obj: str = '', conn: int = ASSERT, neg: bool = False, source_idx: int = -1, chain_len: int = 0, rule_name: str = ''):
        self.subj, self.pred, self.obj = _norm_phrase(subj), _norm_phrase(pred), _norm_phrase(obj)
        self.conn, self.neg = conn, neg
        self.subj_h, self.pred_h, self.obj_h = _gf17_hash(self.subj), _gf17_hash(self.pred), _gf17_hash(self.obj)
        self.source_idx, self.chain_len, self.rule_name = source_idx, chain_len, rule_name
        self.text = self._reconstruct()
    def _reconstruct(self) -> str:
        neg_pre = "not " if self.neg else ""
        if self.conn == COND:
            return f"if {self.subj} then {neg_pre}{self.pred}"
        elif self.conn == DISJ:
            return f"{self.subj} or {neg_pre}{self.pred}"
        elif self.conn == CONJ:
            return f"{self.subj} and {neg_pre}{self.pred}"
        elif self.conn == UNIV:
            return f"all {self.subj} {neg_pre}{self.pred}"
        elif self.conn == EXIST:
            return f"some {self.subj} {neg_pre}{self.pred}"
        elif self.conn == NEG:
            return f"not {self.subj} {self.pred}" if self.pred else f"not {self.subj}"
        elif self.conn == DNEG:
            return f"{self.subj} {self.pred}" if self.pred else self.subj
        return f"{neg_pre}{self.subj} {self.pred}" + (f" {self.obj}" if self.obj else "")
    def hash_key(self) -> int:
        return hash((self.subj_h, self.pred_h, self.obj_h, self.conn, self.neg))
    def to_row(self) -> np.ndarray:
        return np.array([self.subj_h, self.pred_h, self.obj_h, self.conn, int(self.neg)], dtype=np.int64)
class PropositionExtractor:
    def extract(self, text: str, source_idx: int = 0) -> List[Proposition]:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        props = []
        for si, sent in enumerate(sents):
            sent = sent.strip()
            if len(sent) < 10:
                continue
            found = False
            for m in _COND_RE.finditer(sent):
                g = [x for x in m.groups() if x]
                if len(g) >= 2:
                    props.append(Proposition(g[0], g[1], conn=COND, source_idx=source_idx))
                    found = True
            for m in _UNIV_RE.finditer(sent):
                props.append(Proposition(m.group(1), m.group(2), conn=UNIV, source_idx=source_idx))
                found = True
            for m in _EXIST_RE.finditer(sent):
                props.append(Proposition(m.group(1), m.group(2), conn=EXIST, source_idx=source_idx))
                found = True
            for m in _NEG_RE.finditer(sent):
                g = [x for x in m.groups() if x]
                if g:
                    subj = g[0] if len(g) == 1 else g[0]
                    pred = g[1] if len(g) > 1 else ''
                    props.append(Proposition(subj, pred, conn=NEG, neg=True, source_idx=source_idx))
                    found = True
            if not found:
                for m in _DISJ_RE.finditer(sent):
                    a, b = m.group(1).strip(), m.group(2).strip()
                    if len(a) > 3 and len(b) > 3:
                        props.append(Proposition(a, b, conn=DISJ, source_idx=source_idx))
                        found = True
            if not found:
                for m in _ASSERT_RE.finditer(sent):
                    subj, pred = m.group(1).strip(), m.group(2).strip()
                    if len(subj) > 2 and len(pred) > 2 and len(subj) < 80:
                        props.append(Proposition(subj, pred, conn=ASSERT, source_idx=source_idx))
                        break
        return props
_STRIP_RE = re.compile(r'^(?:a |an |the |it |they |we |he |she |this |that |these |those )', re.IGNORECASE)
def _match_phrase(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) > 4 and len(b) > 4 and (a in b or b in a):
        return True
    sa, sb = _STRIP_RE.sub('', a), _STRIP_RE.sub('', b)
    return sa == sb or (len(sa) > 4 and len(sb) > 4 and (sa in sb or sb in sa))
class RuleEngine:
    def __init__(self):
        self._rules = [
            ('modus_ponens', self._modus_ponens),
            ('modus_tollens', self._modus_tollens),
            ('hypothetical_syllogism', self._hypo_syl),
            ('disjunctive_syllogism', self._disj_syl),
            ('constructive_dilemma', self._constr_dilemma),
            ('conjunction_intro', self._conj_intro),
            ('conjunction_elim', self._conj_elim),
            ('double_neg_elim', self._double_neg),
            ('universal_instantiation', self._univ_inst),
            ('existential_generalization', self._exist_gen),
        ]
    def _modus_ponens(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        conds = [p for p in props if p.conn == COND and not p.neg]
        asserts = [p for p in props if p.conn == ASSERT and not p.neg]
        for c in conds:
            for a in asserts:
                if _match_phrase(c.subj, a.subj + ' ' + a.pred if a.pred else a.subj) or _match_phrase(c.subj, a.subj):
                    if a.source_idx != c.source_idx or a.source_idx == -1:
                        derived.append(Proposition(c.pred, '', conn=ASSERT, chain_len=max(c.chain_len, a.chain_len) + 1, rule_name='modus_ponens'))
        return derived
    def _modus_tollens(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        conds = [p for p in props if p.conn == COND and not p.neg]
        negs = [p for p in props if p.neg or p.conn == NEG]
        for c in conds:
            for n in negs:
                target = n.subj + (' ' + n.pred if n.pred else '')
                if _match_phrase(c.pred, target) or _match_phrase(c.pred, n.subj):
                    derived.append(Proposition(c.subj, '', conn=NEG, neg=True, chain_len=max(c.chain_len, n.chain_len) + 1, rule_name='modus_tollens'))
        return derived
    def _hypo_syl(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        conds = [p for p in props if p.conn == COND and not p.neg]
        for c1 in conds:
            for c2 in conds:
                if c1 is c2:
                    continue
                if _match_phrase(c1.pred, c2.subj):
                    derived.append(Proposition(c1.subj, c2.pred, conn=COND, chain_len=max(c1.chain_len, c2.chain_len) + 1, rule_name='hypothetical_syllogism'))
        return derived
    def _disj_syl(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        disjs = [p for p in props if p.conn == DISJ]
        negs = [p for p in props if p.neg or p.conn == NEG]
        for d in disjs:
            for n in negs:
                target = n.subj + (' ' + n.pred if n.pred else '')
                if _match_phrase(d.subj, target) or _match_phrase(d.subj, n.subj):
                    derived.append(Proposition(d.pred, '', conn=ASSERT, chain_len=max(d.chain_len, n.chain_len) + 1, rule_name='disjunctive_syllogism'))
                elif _match_phrase(d.pred, target) or _match_phrase(d.pred, n.subj):
                    derived.append(Proposition(d.subj, '', conn=ASSERT, chain_len=max(d.chain_len, n.chain_len) + 1, rule_name='disjunctive_syllogism'))
        return derived
    def _constr_dilemma(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        conds = [p for p in props if p.conn == COND and not p.neg]
        disjs = [p for p in props if p.conn == DISJ]
        for i, c1 in enumerate(conds):
            for c2 in conds[i+1:]:
                for d in disjs:
                    if (_match_phrase(d.subj, c1.subj) and _match_phrase(d.pred, c2.subj)) or \
                       (_match_phrase(d.subj, c2.subj) and _match_phrase(d.pred, c1.subj)):
                        derived.append(Proposition(c1.pred, c2.pred, conn=DISJ, chain_len=max(c1.chain_len, c2.chain_len, d.chain_len) + 1, rule_name='constructive_dilemma'))
        return derived
    def _conj_intro(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        asserts = [p for p in props if p.conn == ASSERT and not p.neg]
        for i, a1 in enumerate(asserts):
            for a2 in asserts[i+1:]:
                if a1.source_idx != a2.source_idx:
                    derived.append(Proposition(a1.subj + ' ' + a1.pred, a2.subj + ' ' + a2.pred, conn=CONJ, chain_len=max(a1.chain_len, a2.chain_len) + 1, rule_name='conjunction_intro'))
                    if len(derived) > MAX_DERIVED:
                        return derived
        return derived
    def _conj_elim(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        conjs = [p for p in props if p.conn == CONJ]
        for c in conjs:
            derived.append(Proposition(c.subj, '', conn=ASSERT, chain_len=c.chain_len + 1, rule_name='conjunction_elim'))
            derived.append(Proposition(c.pred, '', conn=ASSERT, chain_len=c.chain_len + 1, rule_name='conjunction_elim'))
        return derived
    def _double_neg(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        dnegs = [p for p in props if p.conn == DNEG]
        for d in dnegs:
            derived.append(Proposition(d.subj, d.pred, conn=ASSERT, chain_len=d.chain_len + 1, rule_name='double_neg_elim'))
        negs = [p for p in props if p.conn == NEG and p.neg]
        neg_map = defaultdict(list)
        for n in negs:
            neg_map[n.subj_h].append(n)
        for nh, ns in neg_map.items():
            if len(ns) >= 2:
                derived.append(Proposition(ns[0].subj, ns[0].pred, conn=ASSERT, neg=False, chain_len=max(n.chain_len for n in ns) + 1, rule_name='double_neg_elim'))
        return derived
    def _univ_inst(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        univs = [p for p in props if p.conn == UNIV]
        entities = set()
        for p in props:
            if p.conn == ASSERT and p.subj and len(p.subj.split()) <= 3:
                entities.add(p.subj)
        for u in univs:
            for ent in list(entities)[:10]:
                derived.append(Proposition(ent, u.pred, conn=ASSERT, chain_len=u.chain_len + 1, rule_name='universal_instantiation'))
        return derived
    def _exist_gen(self, props: List[Proposition]) -> List[Proposition]:
        derived = []
        seen_preds = set()
        for p in props:
            if p.conn == ASSERT and not p.neg and p.pred and p.pred_h not in seen_preds:
                seen_preds.add(p.pred_h)
                derived.append(Proposition(p.subj, p.pred, conn=EXIST, chain_len=p.chain_len + 1, rule_name='existential_generalization'))
        return derived
    def apply_all(self, propositions: List[Proposition], max_rounds: int = MAX_DERIVE_ROUNDS) -> List[Proposition]:
        all_props = list(propositions)
        seen = {p.hash_key() for p in all_props}
        for rnd in range(max_rounds):
            new_derived = []
            with ThreadPoolExecutor(max_workers=min(len(self._rules), 4)) as pool:
                futures = {pool.submit(fn, all_props): name for name, fn in self._rules}
                for fut in futures:
                    try:
                        result = fut.result()
                        for d in result:
                            hk = d.hash_key()
                            if hk not in seen:
                                seen.add(hk)
                                new_derived.append(d)
                    except Exception:
                        pass
            if not new_derived:
                break
            all_props.extend(new_derived[:MAX_DERIVED - len(all_props)])
            if len(all_props) >= MAX_DERIVED:
                break
        return [p for p in all_props if p.chain_len > 0]
class ContestEngine:
    def resolve(self, derived: List[Proposition], query_tokens: Set[str]) -> List[Proposition]:
        if not derived:
            return []
        scored = []
        for d in derived:
            d_toks = set(d.text.lower().split())
            relevance = len(query_tokens & d_toks) / max(len(query_tokens), 1)
            chain_penalty = 1.0 / (1.0 + d.chain_len * 0.3)
            score = relevance * chain_penalty
            scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        resolved = []
        seen_subj = set()
        for sc, d in scored:
            if sc < 0.01:
                continue
            key = (d.subj_h, d.conn)
            if key in seen_subj:
                continue
            seen_subj.add(key)
            resolved.append(d)
            if len(resolved) >= 8:
                break
        contradictions = self._find_contradictions(resolved)
        if contradictions:
            resolved = self._resolve_contradictions(resolved, contradictions)
        return resolved
    def _find_contradictions(self, props: List[Proposition]) -> List[Tuple[int, int]]:
        contras = []
        for i, a in enumerate(props):
            for j, b in enumerate(props[i+1:], i+1):
                if a.subj_h == b.subj_h and a.pred_h == b.pred_h and a.neg != b.neg:
                    contras.append((i, j))
        return contras
    def _resolve_contradictions(self, props: List[Proposition], contras: List[Tuple[int, int]]) -> List[Proposition]:
        drop = set()
        for i, j in contras:
            drop.add(j if props[i].chain_len <= props[j].chain_len else i)
        return [p for idx, p in enumerate(props) if idx not in drop]
def extract_and_reason(context: str, query: str) -> Dict:
    t0 = time.perf_counter()
    extractor = PropositionExtractor()
    engine = RuleEngine()
    contest = ContestEngine()
    props = extractor.extract(context)
    extract_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    derived = engine.apply_all(props)
    derive_ms = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    qtoks = set(re.findall(r'[a-zA-Z]{2,}', query.lower())) - {'the','a','an','is','are','how','what','does','do','can','will'}
    resolved = contest.resolve(derived, qtoks)
    contest_ms = (time.perf_counter() - t2) * 1000
    conclusions = [d.text for d in resolved]
    rule_counts = Counter(d.rule_name for d in derived)
    enriched = ""
    if conclusions:
        enriched = "Based on logical inference: " + ". ".join(c.capitalize() for c in conclusions[:5]) + ". "
    return {
        'premises_found': len(props),
        'derived_total': len(derived),
        'conclusions': conclusions,
        'enriched_prefix': enriched,
        'rule_counts': dict(rule_counts),
        'extract_ms': round(extract_ms, 2),
        'derive_ms': round(derive_ms, 2),
        'contest_ms': round(contest_ms, 2),
        'total_ms': round(extract_ms + derive_ms + contest_ms, 2),
        'premise_types': dict(Counter(CONN_NAMES.get(p.conn, '?') for p in props)),
        'derived_types': dict(Counter(CONN_NAMES.get(d.conn, '?') for d in derived)),
    }
