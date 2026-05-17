import hashlib

def _hept_hash(data: bytes, length: int = 12) -> str:
    n = int.from_bytes(hashlib.sha256(data).digest(), 'big')
    if n == 0: return '0'
    digits = []
    while n:
        digits.append('0123456789abcdefg'[n % 17])
        n //= 17
    return ''.join(digits[::-1])[:length]
import json, time, os, hashlib, re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np
_JUNK_RE=re.compile(r'\[OPTIMIZE\]|Rao!|urlopen error|HTTP Error|Generation error|Traceback|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError|ConnectionError|TimeoutError|FileNotFoundError|PermissionError|OSError|RuntimeError|KeyError|IndexError|AttributeError|Exception:|Error:|Failed:|error occurred|stack trace',re.IGNORECASE)
_COT_RE=re.compile(r'^\s*(\*\*Analyze|\*\*Step|\*\*First|My reasoning:|Let me think|Chain of thought|Step \d+:|First,\s|Second,\s|Third,\s|Finally,\s|In conclusion,\s|To summarize,\s|Therefore,\s|Thus,\s|Hence,\s)',re.IGNORECASE)
def _validate_fact(text):
    if not text or len(text.strip())<10:return None
    if _JUNK_RE.search(text):return None
    cleaned=_COT_RE.sub('',text).strip()
    return cleaned if len(cleaned)>=10 else None
from amni.a1.asimov import scrub_pii, has_pii, has_spam
_MAX_DELTA_SIZE_KB = 512
_MAX_TOTAL_MB = 256
class DeltaPage:
    __slots__ = ('subject', 'facts', 'sources', 'created', 'version',
                 'coverage', 'checksum')
    def __init__(self, subject: str, facts: List[Dict],
                 sources: List[str], coverage: float):
        self.subject = subject
        self.facts = facts
        self.sources = sources
        self.created = time.time()
        self.version = 1
        self.coverage = coverage
        raw = json.dumps({"s": subject, "f": facts}, sort_keys=True)
        self.checksum = _hept_hash(raw.encode(), 64)[:16]
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "facts": self.facts,
            "sources": self.sources,
            "created": self.created,
            "version": self.version,
            "coverage": self.coverage,
            "checksum": self.checksum,
        }
    @staticmethod
    def from_dict(d: Dict) -> 'DeltaPage':
        pg = DeltaPage(d["subject"], d["facts"],
                       d.get("sources", []), d.get("coverage", 0.0))
        pg.created = d.get("created", time.time())
        pg.version = d.get("version", 1)
        pg.checksum = d.get("checksum", "")
        return pg
class CreativityEntry:
    __slots__ = ('pattern', 'context', 'quality', 'created', 'uses', 'fingerprint', 'lineage')
    def __init__(self, pattern: str, context: str, quality: float, fingerprint: Optional[Dict] = None, lineage: Optional[Dict] = None):
        self.pattern = pattern
        self.context = context
        self.quality = quality
        self.created = time.time()
        self.uses = 0
        self.fingerprint = fingerprint
        self.lineage = lineage
    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern, "context": self.context,
            "quality": self.quality, "created": self.created,
            "uses": self.uses, "fingerprint": self.fingerprint,
            "lineage": self.lineage
        }
    @staticmethod
    def from_dict(d: Dict) -> 'CreativityEntry':
        e = CreativityEntry(d["pattern"], d["context"], d["quality"], d.get("fingerprint"), d.get("lineage"))
        e.created = d.get("created", time.time())
        e.uses = d.get("uses", 0)
        return e
class DeltaWriter:
    def __init__(self, learnings_dir: str):
        self.root = Path(learnings_dir)
        self.deltas_dir = self.root / "deltas"
        self.creativity_dir = self.root / "creativity_memory"
        self.reflections_dir = self.root / "reflections"
        self.queue_dir = self.root / "knowledge_queue"
        self.web_dir = self.root / "web_cache"
        for d in (self.deltas_dir, self.creativity_dir,
                  self.reflections_dir, self.queue_dir, self.web_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._delta_index: Dict[str, DeltaPage] = {}
        self._creativity_bank: List[CreativityEntry] = []
        self._lockfile = self.root / "lockfile"
        self._knowledge_packer = None
        self._load_index()
    def set_knowledge_packer(self, packer):
        self._knowledge_packer = packer
    @property
    def creativity_bank(self)->List[CreativityEntry]:
        return self._creativity_bank
    def _load_index(self):
        self.rebuild_index()
        cbank = self.creativity_dir / "creativity_bank.json"
        if cbank.exists():
            with open(str(cbank)) as f:
                entries = json.load(f)
            self._creativity_bank = [CreativityEntry.from_dict(e) for e in entries]
    def rebuild_index(self):
        self._delta_index.clear()
        for fp in self.deltas_dir.glob("*.json"):
            if fp.name == "delta_index.json":
                continue
            try:
                with open(str(fp)) as f:
                    data = json.load(f)
                subj = data.get("subject", fp.stem.replace("_", " "))
                self._delta_index[subj] = DeltaPage.from_dict(data)
            except Exception:
                continue
        self._save_index()
    def _save_index(self):
        idx_path = self.deltas_dir / "delta_index.json"
        data = {subj: pg.to_dict() for subj, pg in self._delta_index.items()}
        with open(str(idx_path), 'w') as f:
            json.dump(data, f, indent=1)
    def _save_creativity(self):
        cbank = self.creativity_dir / "creativity_bank.json"
        entries = [e.to_dict() for e in self._creativity_bank[-500:]]
        with open(str(cbank), 'w') as f:
            json.dump(entries, f, indent=1)
    def _acquire_lock(self) -> bool:
        if self._lockfile.exists():
            try:
                lock_time = float(self._lockfile.read_text().strip())
                if time.time() - lock_time > 300:
                    self._lockfile.unlink()
                else:
                    return False
            except (ValueError, OSError):
                self._lockfile.unlink()
        try:
            self._lockfile.write_text(str(time.time()))
            return True
        except OSError:
            return False
    def _release_lock(self):
        try:
            self._lockfile.unlink(missing_ok=True)
        except OSError:
            pass
    def _check_size_limits(self) -> bool:
        total = sum(f.stat().st_size for f in self.root.rglob("*")
                    if f.is_file() and f.name != "lockfile")
        return total < _MAX_TOTAL_MB * 1024 * 1024
    def write_delta(self, subject: str, facts: List[Dict],
                    sources: List[str], coverage: float) -> Dict:
        clean=[]
        for f in facts:
            txt=f.get("text","")
            v=_validate_fact(txt)
            if v is not None:
                f["text"]=v
                clean.append(f)
        if not clean:
            return {"ok":False,"reason":"all facts filtered by validation"}
        facts=clean
        if not self._acquire_lock():
            return {"ok": False, "reason": "locked"}
        try:
            if not self._check_size_limits():
                return {"ok": False, "reason": "size limit exceeded"}
            existing = self._delta_index.get(subject)
            if existing:
                merged_facts = list(existing.facts)
                existing_checksums = {f.get("checksum", "") for f in merged_facts}
                for f in facts:
                    fck = _hept_hash(json.dumps(f, sort_keys=True).encode(), 64)[:16]
                    f["checksum"] = fck
                    if fck not in existing_checksums:
                        merged_facts.append(f)
                page = DeltaPage(subject, merged_facts, 
                                list(set(existing.sources + sources)), coverage)
                page.version = existing.version + 1
            else:
                for f in facts:
                    fck = _hept_hash(json.dumps(f, sort_keys=True).encode(), 64)[:16]
                    f["checksum"] = fck
                page = DeltaPage(subject, facts, sources, coverage)
            self._delta_index[subject] = page
            page_file = self.deltas_dir / f"{self._safe_filename(subject)}.json"
            with open(str(page_file), 'w') as f:
                json.dump(page.to_dict(), f, indent=1)
            self._save_index()
            if self._knowledge_packer:
                try:
                    self._knowledge_packer.pack_incremental(subject, facts)
                except Exception:
                    pass
            return {"ok": True, "subject": subject, "version": page.version,
                    "n_facts": len(page.facts)}
        finally:
            self._release_lock()
    def read_delta(self, subject: str) -> Optional[DeltaPage]:
        return self._delta_index.get(subject)
    def has_delta(self, subject: str) -> bool:
        return subject in self._delta_index
    def all_subjects(self) -> List[str]:
        return list(self._delta_index.keys())
    def add_creativity(self, pattern: str, context: str,
                       quality: float) -> Dict:
        entry = CreativityEntry(pattern, context, quality)
        self._creativity_bank.append(entry)
        self._save_creativity()
        return {"ok": True, "bank_size": len(self._creativity_bank)}
    def get_creativity(self, context: str = "", top_k: int = 5) -> List[Dict]:
        if not self._creativity_bank:
            return []
        scored = []
        ctx_words = set(context.lower().split()) if context else set()
        for e in self._creativity_bank:
            score = e.quality * 0.5 + (0.5 if any(w in e.context.lower() for w in ctx_words) else 0.0)
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        return [e.to_dict() for _, e in scored[:top_k]]
    def ingest_corpus(self, corpus_path: str, verbose: bool = True) -> Dict:
        src = Path(corpus_path)
        if not src.exists():
            return {"ok": False, "reason": f"path not found: {corpus_path}"}
        files = list(src.rglob("*.txt")) if src.is_dir() else [src] if src.suffix == ".txt" else []
        files += list(src.rglob("*.md")) if src.is_dir() else ([src] if src.suffix == ".md" else [])
        if not files:
            return {"ok": False, "reason": "no .txt or .md files found"}
        total_entries = 0
        for fp in files:
            n = self._ingest_file(fp, verbose)
            total_entries += n
        self._save_creativity()
        if verbose:
            print(f"  [INGEST] {total_entries} entries from {len(files)} files")
        return {"ok": True, "files": len(files), "entries": total_entries,
                "bank_size": len(self._creativity_bank)}
    def _ingest_file(self, filepath: Path, verbose: bool) -> int:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 0
        source = filepath.stem
        paragraphs = re.split(r'\n\s*\n', text)
        added = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20 or len(para) > 2000:
                continue
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent) > 500:
                    continue
                existing = any(e.pattern == sent for e in self._creativity_bank[-200:])
                if existing:
                    continue
                quality = min(1.0, len(sent) / 200.0) * 0.6 + 0.4
                entry = CreativityEntry(sent, f"corpus:{source}", quality)
                self._creativity_bank.append(entry)
                added += 1
        if verbose:
            print(f"    [{source}] {added} entries ingested")
        return added
    def queue_fact(self, subject: str, claim: str, source: str = "self") -> Dict:
        if not subject or not claim:
            return {"ok":False,"reason":"empty subject or claim"}
        if len(claim)<10:
            return {"ok":False,"reason":"claim too short (<10 chars)"}
        if len(claim)>2000:
            return {"ok":False,"reason":"claim too long (>2000 chars)"}
        if has_spam(claim):
            return {"ok":False,"reason":"spam detected"}
        claim=scrub_pii(claim)
        existing=[qf for qf in sorted(self.queue_dir.glob("q_*.json"))[-50:]]
        for qf in existing:
            try:
                with open(str(qf)) as ef:
                    ex=json.load(ef)
                if ex.get("subject")==subject and ex.get("claim","").lower().strip()==claim.lower().strip():
                    return {"ok":False,"reason":"duplicate in queue"}
            except (json.JSONDecodeError,OSError):
                continue
        entry = {
            "subject": subject, "claim": claim,
            "source": source, "queued": time.time(),
            "status": "pending",
        }
        q_file = self.queue_dir / f"q_{int(time.time()*1000)}.json"
        with open(str(q_file), 'w') as f:
            json.dump(entry, f)
        return {"ok": True, "file": q_file.name}
    def get_pending_queue(self, limit: int = 10) -> List[Dict]:
        items = []
        for qf in sorted(self.queue_dir.glob("q_*.json"))[:limit]:
            try:
                with open(str(qf)) as f:
                    entry = json.load(f)
                entry["_file"] = qf.name
                items.append(entry)
            except (json.JSONDecodeError, OSError):
                continue
        return items
    def resolve_queue_item(self, filename: str, status: str) -> bool:
        qf = self.queue_dir / filename
        if not qf.exists():
            return False
        try:
            with open(str(qf)) as f:
                entry = json.load(f)
            entry["status"] = status
            entry["resolved"] = time.time()
            with open(str(qf), 'w') as f:
                json.dump(entry, f)
            return True
        except (json.JSONDecodeError, OSError):
            return False
    def log_reflection(self, cycle: int, gaps: List, actions: List[Dict],
                       results: List[Dict]) -> Dict:
        entry = {
            "cycle": cycle, "timestamp": time.time(),
            "gaps": gaps, "actions": actions, "results": results,
        }
        log_file = self.reflections_dir / f"cycle_{cycle:06d}.json"
        with open(str(log_file), 'w') as f:
            json.dump(entry, f, indent=1)
        return {"ok": True, "file": log_file.name}
    def stats(self) -> Dict:
        total_bytes = sum(f.stat().st_size for f in self.root.rglob("*")
                          if f.is_file() and f.name != "lockfile")
        return {
            "n_deltas": len(self._delta_index),
            "n_creativity": len(self._creativity_bank),
            "n_queued": len(list(self.queue_dir.glob("q_*.json"))),
            "n_reflections": len(list(self.reflections_dir.glob("cycle_*.json"))),
            "total_size_kb": round(total_bytes / 1024, 1),
            "subjects": list(self._delta_index.keys())[:20],
        }
    @staticmethod
    def _safe_filename(s: str) -> str:
        return re.sub(r'[^\w\-.]', '_', s.lower())[:64]
