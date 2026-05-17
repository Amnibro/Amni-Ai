"""SessionATEXWriter: capture successful CoT traces as PTEX KB entries for compounding self-cache.
Per the maintainer's v5.5.135 design: each high-confidence ask_with_loop run writes (query, answer, trace) to a session-scoped KnowledgeBase. Future similar queries retrieve the cached trace as a primary source, compounding reasoning across the session.
Storage: lossless PTEX bytes, exactly the same substrate as wiki/canonical KBs, just session-scoped.
"""
import hashlib,json,time
from pathlib import Path
from amni.learning.knowledge_base import KnowledgeBase
class SessionATEXWriter:
    def __init__(self,session_root,confidence_threshold=0.6):
        self.kb=KnowledgeBase(session_root)
        self.threshold=confidence_threshold
        self.n_written=0;self.n_skipped=0
    def _nonce_key(self,query):
        return f'session::{hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]}'
    def write(self,query,answer,confidence,trace=None,meta=None):
        if confidence<self.threshold:
            self.n_skipped+=1;return None
        key=self._nonce_key(query)
        body=[f'Q: {query.strip()}',f'A: {(answer or "").strip()}']
        if trace:
            body.append('Trace:')
            for step in trace[-3:]:
                step_str=json.dumps(step,default=str) if isinstance(step,dict) else str(step)
                body.append(f'  {step_str[:300]}')
        text='\n'.join(body)
        m={'kind':'session_cot','confidence':float(confidence),'ts':time.time()}
        if meta:m.update(meta)
        try:
            self.kb.add(key,text,meta=m,allow_overwrite=True)
            self.n_written+=1
            return key
        except Exception as e:
            self.n_skipped+=1;return None
    def flush(self):
        try:self.kb.flush()
        except Exception:pass
    def stats(self):
        s=self.kb.stats()
        s['session_writes']=self.n_written;s['session_skipped']=self.n_skipped
        s['threshold']=self.threshold
        return s
    def close(self):
        self.flush();self.kb.close()
