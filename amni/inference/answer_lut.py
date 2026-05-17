"""AnswerLUT: adaptive-match Q→A cache for AdamLoop tier-1.
Per Anthony's vision: "Adam gets asked the same question: becomes LUT" — return cached answer instantly with zero inference. ADAPTIVE: normalize away trivial restatements (case, whitespace, contractions, trailing punctuation) so "What's the capital?" hits the same entry as "What is the Capital".
Storage: single JSON index keyed by sha256(normalized_q)[:24]; values include original query + answer + meta.
Atomic write via temp+rename. Rollback support: track recent additions, drop on regression.
"""
import json,hashlib,re,time
from pathlib import Path
from typing import Optional,Dict,Any,List
_CONTRACTIONS={"what's":"what is","it's":"it is","that's":"that is","there's":"there is","here's":"here is","who's":"who is","how's":"how is","let's":"let us","i'm":"i am","you're":"you are","they're":"they are","we're":"we are","i've":"i have","you've":"you have","we've":"we have","they've":"they have","i'd":"i would","you'd":"you would","he'd":"he would","she'd":"she would","we'd":"we would","they'd":"they would","i'll":"i will","you'll":"you will","he'll":"he will","she'll":"she will","we'll":"we will","they'll":"they will","isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not","hasn't":"has not","haven't":"have not","hadn't":"had not","doesn't":"does not","don't":"do not","didn't":"did not","won't":"will not","wouldn't":"would not","shouldn't":"should not","couldn't":"could not","can't":"cannot","cannot":"cannot","mustn't":"must not"}
_QUOTE_RE=re.compile(r"[‘’‚‛′‵`]")
_DQUOTE_RE=re.compile(r"[“”„‟″‶]")
_WS_RE=re.compile(r"\s+")
_END_PUNCT_RE=re.compile(r"[?!.,;:]+\s*$")
_BOILER_RE=re.compile(r"^(please\s+|kindly\s+|hey,?\s+|hi,?\s+|hello,?\s+|so,?\s+|um,?\s+|uh,?\s+|i\s+(?:was\s+)?wonder(?:ing)?\s+)+",re.I)
def _normalize_query(q:str)->str:
    s=q.strip().lower()
    s=_QUOTE_RE.sub("'",s);s=_DQUOTE_RE.sub('"',s)
    for c,exp in _CONTRACTIONS.items():s=re.sub(rf"\b{re.escape(c)}\b",exp,s)
    s=_BOILER_RE.sub("",s)
    s=_END_PUNCT_RE.sub("",s)
    s=_WS_RE.sub(" ",s).strip()
    return s
class AnswerLUT:
    def __init__(self,root:str):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.index_path=self.root/'lut_index.json'
        self._index:Dict[str,Dict[str,Any]]={}
        self._recent_keys:List[str]=[]
        self._load()
    def _load(self):
        if self.index_path.exists():
            try:self._index=json.loads(self.index_path.read_text(encoding='utf-8'))
            except Exception:self._index={}
    def _save(self):
        tmp=self.index_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self._index,ensure_ascii=False),encoding='utf-8')
        import os
        os.replace(str(tmp),str(self.index_path))
    def _key(self,q:str)->str:
        return hashlib.sha256(_normalize_query(q).encode('utf-8')).hexdigest()[:24]
    def normalized(self,q:str)->str:return _normalize_query(q)
    def lookup(self,q:str)->Optional[Dict[str,Any]]:
        e=self._index.get(self._key(q))
        if e is not None:
            e['hit_count']=e.get('hit_count',0)+1;e['last_hit_ts']=time.time()
        return e
    def store(self,q:str,answer:str,subject:Optional[str]=None,source:str='cold_solve',meta:Optional[Dict[str,Any]]=None,track_recent:bool=True):
        k=self._key(q)
        self._index[k]={'q':q[:300],'a':answer,'subject':subject,'source':source,'ts':time.time(),'hit_count':0,'meta':meta or {}}
        if track_recent:self._recent_keys.append(k)
        self._save();return k
    def commit_recent(self):
        n=len(self._recent_keys);self._recent_keys=[];return n
    def rollback_recent(self):
        n=0
        for k in self._recent_keys:
            if k in self._index:del self._index[k];n+=1
        self._recent_keys=[];self._save();return n
    def stats(self):
        n=len(self._index)
        hits=sum(int(e.get('hit_count',0)) for e in self._index.values())
        return {'n_entries':n,'total_hits':hits,'recent_uncommitted':len(self._recent_keys)}
    def keys(self):return list(self._index.keys())
