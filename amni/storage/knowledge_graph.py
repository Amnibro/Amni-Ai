"""KnowledgeGraph — SPO triple store on top of the LearningAtlas + sem_lut substrate.
Each triple: (subject, predicate, object, confidence, sources, ts, kind). Subject + object are normalized to lowercase token strings; predicate is a short relation slug. Indexed both directions for O(1) neighbor lookup.
Persistence: triples.jsonl (full-rewrite on save; the file stays small per triple). In-memory adjacency dicts give path queries.
Mainstream models can't reason relationally because their facts live in opaque weights. Adam can: given two subjects, BFS the graph and emit the path that connects them."""
import json,time,hashlib,threading,re
from pathlib import Path
from typing import List,Dict,Any,Optional,Tuple,Set
_TOKEN_RE=re.compile(r'\w[\w\-]*')
def _norm(s:str,max_len:int=80)->str:
    if not s:return ''
    toks=_TOKEN_RE.findall(s.lower())
    return ' '.join(toks)[:max_len].strip()
def _slug_predicate(p:str)->str:
    p=(p or '').strip().lower().replace(' ','_')
    return re.sub(r'[^a-z0-9_]','',p)[:40] or 'related_to'
def _triple_key(s:str,p:str,o:str)->str:
    h=hashlib.blake2b(digest_size=8)
    for x in (s,p,o):h.update(x.encode('utf-8','ignore'));h.update(b'\x1f')
    return h.hexdigest()
class KnowledgeGraph:
    def __init__(self,root:str='experiences/knowledge_graph'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self._path=self.root/'triples.jsonl'
        self._lock=threading.Lock()
        self._triples:Dict[str,Dict[str,Any]]={}
        self._by_subject:Dict[str,Set[str]]={}
        self._by_object:Dict[str,Set[str]]={}
        self._by_predicate:Dict[str,Set[str]]={}
        self._load()
    def _index_one(self,k:str,t:Dict[str,Any]):
        self._by_subject.setdefault(t['s'],set()).add(k)
        self._by_object.setdefault(t['o'],set()).add(k)
        self._by_predicate.setdefault(t['p'],set()).add(k)
    def _unindex_one(self,k:str,t:Dict[str,Any]):
        for d,key in ((self._by_subject,t['s']),(self._by_object,t['o']),(self._by_predicate,t['p'])):
            if key in d:d[key].discard(k)
            if key in d and not d[key]:d.pop(key,None)
    def _load(self):
        if not self._path.exists():return
        try:
            for ln in self._path.read_text(encoding='utf-8').strip().splitlines():
                if ln.strip():
                    try:t=json.loads(ln)
                    except Exception:continue
                    k=t.get('key')
                    if k:self._triples[k]=t;self._index_one(k,t)
        except Exception as e:print(f'[KnowledgeGraph] load failed: {e}',flush=True)
    def _save_all(self):
        try:
            tmp=self._path.with_suffix('.jsonl.tmp')
            with tmp.open('w',encoding='utf-8') as f:
                for t in self._triples.values():f.write(json.dumps(t,default=str)+'\n')
            tmp.replace(self._path)
        except Exception as e:print(f'[KnowledgeGraph] save failed: {e}',flush=True)
    def add(self,subject:str,predicate:str,object_:str,source:str='',confidence:Optional[float]=None,kind:str='ingest')->Optional[Dict[str,Any]]:
        s=_norm(subject);p=_slug_predicate(predicate);o=_norm(object_)
        if not s or not p or not o:return None
        if len(s)<2 or len(o)<2:return None
        k=_triple_key(s,p,o);now=time.time()
        with self._lock:
            cur=self._triples.get(k)
            if cur is None:
                cur={'key':k,'s':s,'p':p,'o':o,'sources':[source] if source else [],'consensus_count':1,'confidence':float(confidence) if confidence is not None else 0.5,'created_ts':now,'last_seen_ts':now,'kind':kind}
                self._triples[k]=cur;self._index_one(k,cur)
            else:
                cur['last_seen_ts']=now
                if source and source not in cur['sources']:
                    cur['sources']=(cur['sources']+[source])[:20]
                    cur['consensus_count']=int(cur.get('consensus_count',1))+1
                    cur['confidence']=min(1.0,float(cur.get('confidence',0.5))+0.15)
            self._save_all()
            return dict(cur)
    def add_many(self,triples:List[Tuple[str,str,str]],source:str='',confidence:Optional[float]=None)->int:
        n=0
        for s,p,o in triples:
            if self.add(s,p,o,source=source,confidence=confidence) is not None:n+=1
        return n
    def neighbors_of(self,subject:str,direction:str='both',limit:int=50)->List[Dict[str,Any]]:
        s=_norm(subject);out=[]
        with self._lock:
            if direction in ('out','both'):
                for k in self._by_subject.get(s,set()):
                    t=self._triples.get(k)
                    if t:out.append({'s':t['s'],'p':t['p'],'o':t['o'],'direction':'out','conf':t.get('confidence',0.5),'consensus':t.get('consensus_count',1)})
            if direction in ('in','both'):
                for k in self._by_object.get(s,set()):
                    t=self._triples.get(k)
                    if t:out.append({'s':t['s'],'p':t['p'],'o':t['o'],'direction':'in','conf':t.get('confidence',0.5),'consensus':t.get('consensus_count',1)})
        out.sort(key=lambda x:(-x['consensus'],-x['conf']))
        return out[:limit]
    def by_predicate(self,predicate:str,limit:int=50)->List[Dict[str,Any]]:
        p=_slug_predicate(predicate);out=[]
        with self._lock:
            for k in self._by_predicate.get(p,set()):
                t=self._triples.get(k)
                if t:out.append({'s':t['s'],'p':t['p'],'o':t['o'],'conf':t.get('confidence',0.5),'consensus':t.get('consensus_count',1)})
        out.sort(key=lambda x:(-x['consensus'],-x['conf']))
        return out[:limit]
    def path_between(self,a:str,b:str,max_hops:int=3)->Optional[List[Dict[str,Any]]]:
        sa=_norm(a);sb=_norm(b)
        if not sa or not sb:return None
        if sa==sb:return []
        visited={sa};frontier=[(sa,[])];hop=0
        with self._lock:
            while frontier and hop<max_hops:
                next_frontier=[]
                for node,path in frontier:
                    out_keys=self._by_subject.get(node,set());in_keys=self._by_object.get(node,set())
                    for k in out_keys|in_keys:
                        t=self._triples.get(k)
                        if t is None:continue
                        nxt=t['o'] if t['s']==node else t['s']
                        edge={'s':t['s'],'p':t['p'],'o':t['o'],'via':node,'next':nxt}
                        if nxt==sb:return path+[edge]
                        if nxt in visited:continue
                        visited.add(nxt);next_frontier.append((nxt,path+[edge]))
                frontier=next_frontier;hop+=1
        return None
    def search_subject(self,query:str,limit:int=20)->List[str]:
        q=_norm(query)
        with self._lock:
            keys=list(self._by_subject.keys())
        keys=[s for s in keys if q and (q in s or s in q)]
        return sorted(keys,key=lambda s:abs(len(s)-len(q)))[:limit]
    def stats(self)->Dict[str,Any]:
        with self._lock:
            tn=len(self._triples)
            if tn==0:return {'triples':0,'subjects':0,'objects':0,'predicates':0}
            avg_conf=sum(float(t.get('confidence',0)) for t in self._triples.values())/tn
            avg_consensus=sum(int(t.get('consensus_count',1)) for t in self._triples.values())/tn
            return {'triples':tn,'subjects':len(self._by_subject),'objects':len(self._by_object),'predicates':len(self._by_predicate),'avg_confidence':round(avg_conf,3),'avg_consensus':round(avg_consensus,2),'top_predicates':[(p,len(ks)) for p,ks in sorted(self._by_predicate.items(),key=lambda kv:-len(kv[1]))[:10]]}
    def forget(self,subject:Optional[str]=None,predicate:Optional[str]=None,object_:Optional[str]=None)->int:
        removed=0
        with self._lock:
            kill=[]
            for k,t in self._triples.items():
                if subject is not None and t.get('s')!=_norm(subject):continue
                if predicate is not None and t.get('p')!=_slug_predicate(predicate):continue
                if object_ is not None and t.get('o')!=_norm(object_):continue
                kill.append(k)
            for k in kill:
                t=self._triples.pop(k);self._unindex_one(k,t);removed+=1
            if removed:self._save_all()
        return removed
