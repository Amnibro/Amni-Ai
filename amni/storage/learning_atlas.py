"""LearningAtlas — per-cell metadata overlay for Adam's sem_lut. Tracks confidence, provenance sources, last_reinforced_at, consensus_count for every taught fact.
This is the substrate that turns Adam's lesson bank from a flat "stored answer" cache into a verifiable, decay-aware, multi-source-aggregated knowledge graph. Mainstream models can't do this because their weights are frozen.
Persistence: meta.jsonl (one row per fact-cell). Cell key = blake2b(question+answer) so meta survives sem_lut refits."""
import json,time,hashlib,threading
from pathlib import Path
from typing import Dict,Any,List,Optional,Tuple
_CONF_NEW=0.5
_CONF_VERIFIED_THRESHOLD=0.85
_STALE_AFTER_S=30*24*3600
def _cell_key(q:str,a:str)->str:
    h=hashlib.blake2b(digest_size=8);h.update((q or '').encode('utf-8','ignore'));h.update(b'\x1f');h.update((a or '').encode('utf-8','ignore'))
    return h.hexdigest()
class LearningAtlas:
    def __init__(self,root:str='experiences/learning_atlas'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self._path=self.root/'meta.jsonl'
        self._lock=threading.Lock()
        self._meta:Dict[str,Dict[str,Any]]={}
        self._load()
    def _load(self):
        if not self._path.exists():return
        try:
            for ln in self._path.read_text(encoding='utf-8').strip().splitlines():
                if ln.strip():
                    try:r=json.loads(ln)
                    except Exception:continue
                    k=r.get('key')
                    if k:self._meta[k]=r
        except Exception as e:print(f'[LearningAtlas] load failed: {e}',flush=True)
    def _save_all(self):
        try:
            tmp=self._path.with_suffix('.jsonl.tmp')
            with tmp.open('w',encoding='utf-8') as f:
                for r in self._meta.values():f.write(json.dumps(r,default=str)+'\n')
            tmp.replace(self._path)
        except Exception as e:print(f'[LearningAtlas] save failed: {e}',flush=True)
    def record(self,question:str,answer:str,source:str='',confidence:Optional[float]=None,kind:str='ingest')->Dict[str,Any]:
        k=_cell_key(question,answer);now=time.time()
        with self._lock:
            cur=self._meta.get(k)
            if cur is None:
                cur={'key':k,'q':question[:400],'a':answer[:1200],'sources':[],'consensus_count':1,'confidence':float(confidence) if confidence is not None else _CONF_NEW,'created_ts':now,'last_reinforced_at':now,'kind':kind,'verified':False,'debated':False,'retest_count':0}
            else:
                cur['last_reinforced_at']=now
                if source and source not in cur['sources']:
                    cur['sources']=(cur['sources']+[source])[:20]
                    cur['consensus_count']=int(cur.get('consensus_count',1))+1
                    cur['confidence']=min(1.0,float(cur.get('confidence',_CONF_NEW))+0.15)
                if cur['confidence']>=_CONF_VERIFIED_THRESHOLD:cur['verified']=True
            if source and source not in cur['sources']:cur['sources']=(cur['sources']+[source])[:20]
            self._meta[k]=cur
            self._save_all()
        return cur
    def mark_debated(self,question:str,answer:str,competing_answer:str,source:str='')->Optional[Dict[str,Any]]:
        k=_cell_key(question,answer);now=time.time()
        with self._lock:
            cur=self._meta.get(k)
            if cur is None:return None
            cur['debated']=True;cur['competing_answer']=competing_answer[:400];cur['debated_source']=source;cur['debated_at']=now
            cur['confidence']=max(0.0,float(cur.get('confidence',_CONF_NEW))-0.2)
            self._save_all();return cur
    def reinforce(self,question:str,answer:str,bump:float=0.05)->Optional[Dict[str,Any]]:
        k=_cell_key(question,answer);now=time.time()
        with self._lock:
            cur=self._meta.get(k)
            if cur is None:return None
            cur['last_reinforced_at']=now
            cur['retest_count']=int(cur.get('retest_count',0))+1
            cur['confidence']=min(1.0,float(cur.get('confidence',_CONF_NEW))+bump)
            if cur['confidence']>=_CONF_VERIFIED_THRESHOLD:cur['verified']=True
            self._save_all();return cur
    def get(self,question:str,answer:str)->Optional[Dict[str,Any]]:
        return self._meta.get(_cell_key(question,answer))
    def stale_cells(self,older_than_s:int=_STALE_AFTER_S,limit:int=50)->List[Dict[str,Any]]:
        now=time.time()
        with self._lock:
            out=[r for r in self._meta.values() if (now-r.get('last_reinforced_at',0))>older_than_s]
        out.sort(key=lambda r:r.get('last_reinforced_at',0))
        return out[:limit]
    def verified_facts(self,limit:int=100)->List[Dict[str,Any]]:
        with self._lock:out=[r for r in self._meta.values() if r.get('verified')]
        out.sort(key=lambda r:-(r.get('consensus_count',1)));return out[:limit]
    def debated_facts(self,limit:int=50)->List[Dict[str,Any]]:
        with self._lock:return [r for r in self._meta.values() if r.get('debated')][:limit]
    def stats(self)->Dict[str,Any]:
        with self._lock:m=list(self._meta.values())
        if not m:return {'total':0,'verified':0,'debated':0,'stale':0,'avg_confidence':0.0,'avg_sources':0.0}
        now=time.time()
        return {'total':len(m),'verified':sum(1 for r in m if r.get('verified')),'debated':sum(1 for r in m if r.get('debated')),'stale':sum(1 for r in m if (now-r.get('last_reinforced_at',0))>_STALE_AFTER_S),'avg_confidence':round(sum(float(r.get('confidence',0)) for r in m)/len(m),3),'avg_sources':round(sum(int(r.get('consensus_count',1)) for r in m)/len(m),2),'max_sources':max(int(r.get('consensus_count',1)) for r in m)}
