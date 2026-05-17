"""NonceKBRetriever: Reffelt-nonce-addressed retrieval over a PTEX KB.
Per the v5.5.120 'MoE 1.5 lossless LUT first' framing: nonce encodes scale + biome + forest + content_hash. Query gets parsed into a (scale_mask, biome_mask) prefilter, then TF-keyword scoring runs ONLY on the matching subset. Net: smaller candidate pool → higher signal-to-noise → less TF inversion.
Nonce layout (48 bits in uint64):
    [4 bits scale][12 bits biome][12 bits forest][20 bits content]
"""
import hashlib,re,struct
from pathlib import Path
from typing import List,Tuple,Optional
import numpy as np
from amni.learning.knowledge_base import KnowledgeBase
_TOK_RE=re.compile(r"[a-zA-Z][a-zA-Z0-9_\-\.]*")
_STOP=frozenset({'the','a','an','and','or','of','to','in','for','is','are','what','how','do','i','my','this','that','with','on','at','by','as','it','be','can','use','using','from','some','any','have','has','will','would','should','could','make','get','set','put','show','give','tell','please','help'})
_BIOME_KEYWORDS={
    'python':['python','py','pip','def','self','__init__','pathlib','numpy','pandas','torch','flask','django','asyncio'],
    'rust':['rust','cargo','impl','trait','async','tokio','serde','vec','option','result'],
    'javascript':['javascript','js','node','npm','async','await','promise','dom','event'],
    'typescript':['typescript','ts','interface','generic','tsx'],
    'java':['java','jvm','spring','maven','gradle'],
    'cpp':['cpp','c++','stl','template','header','iostream'],
    'c':['c','header','stdlib','printf'],
    'go':['golang','goroutine','chan','interface'],
    'haskell':['haskell','monad','functor'],
    'ruby':['ruby','rails','gem','attr_accessor'],
    'php':['php','composer','laravel','symfony','wordpress'],
    'css':['css','selector','flexbox','grid','margin','padding'],
    'html':['html','tag','element','attribute','dom'],
    'sql':['sql','query','select','join','postgres','mysql'],
    'docker':['docker','container','dockerfile','compose'],
    'kubernetes':['kubernetes','k8s','pod','deployment','helm'],
    'ansible':['ansible','playbook','inventory'],
    'dart':['dart','flutter','widget'],
    'swift':['swift','ios','xcode'],
    'kotlin':['kotlin','android'],
}
def _tokenize(text):return [t.lower() for t in _TOK_RE.findall(text or '') if t.lower() not in _STOP and len(t)>=2]
def _h12(s):return int(hashlib.sha256(s.encode()).hexdigest()[:3],16)
def _h20(s):return int(hashlib.sha256(s.encode()).hexdigest()[:5],16)
def _classify_query_biome(query):
    q_low=query.lower()
    scores={}
    for biome,kws in _BIOME_KEYWORDS.items():
        scores[biome]=sum(1 for k in kws if k in q_low)
    if not any(v>0 for v in scores.values()):return None,0
    top=max(scores,key=scores.get)
    return top,scores[top]
def _classify_query_scale(query):
    q_tokens=_tokenize(query)
    n=len(q_tokens)
    if n<=3:return 5
    if n<=6:return 4
    if n<=10:return 3
    return 2
class NonceKBRetriever:
    def __init__(self,kb_root,nonce_index_path=None):
        self.kb=KnowledgeBase(kb_root)
        if nonce_index_path is None:nonce_index_path=Path(kb_root)/'nonce_index.ptex'
        self._load_nonce_index(nonce_index_path)
    def _load_nonce_index(self,path):
        path=Path(path)
        if not path.exists():
            self.nonces=None;self.entries=[];return
        data=path.read_bytes()
        if data[:8]!=b'NONCEPTX':raise ValueError(f'bad magic in {path}')
        n=struct.unpack('<I',data[8:12])[0]
        self.nonces=np.zeros(n,dtype=np.uint64)
        self.entries=[]
        off=12
        for i in range(n):
            nonce,page,offset,length=struct.unpack('<QHIH',data[off:off+16])
            self.nonces[i]=nonce
            self.entries.append((page,offset,length))
            off+=16
        self._scales=(self.nonces>>44)&0xF
        self._biomes=(self.nonces>>32)&0xFFF
    def retrieve(self,query,k=3,max_chars_per=600,min_score=1,use_nonce_filter=True,scale_relax=2):
        if self.nonces is None or not use_nonce_filter:return self._retrieve_keyword(query,k,max_chars_per,min_score,None)
        biome,biome_conf=_classify_query_biome(query)
        scale=_classify_query_scale(query)
        if biome is None:return self._retrieve_keyword(query,k,max_chars_per,min_score,None)
        biome_id=_h12(biome)
        scale_min=max(1,scale-scale_relax);scale_max=min(6,scale+scale_relax)
        sel=(self._biomes==biome_id)&(self._scales>=scale_min)&(self._scales<=scale_max)
        cand_idxs=np.where(sel)[0]
        if len(cand_idxs)==0:return self._retrieve_keyword(query,k,max_chars_per,min_score,None)
        keys=self.kb.keys()
        cand_keys=[keys[i] for i in cand_idxs if i<len(keys)]
        return self._retrieve_keyword(query,k,max_chars_per,min_score,cand_keys)
    def _retrieve_keyword(self,query,k,max_chars_per,min_score,cand_keys):
        q_tokens=_tokenize(query)
        if not q_tokens:return []
        keys=cand_keys if cand_keys is not None else self.kb.keys()
        scored=[]
        for key in keys:
            key_l=key.lower()
            score=sum(1 for t in q_tokens if t in key_l)
            if score==0:
                txt=self.kb.lookup(key) or ''
                if not txt:continue
                txt_l=txt.lower()
                score=sum(1 for t in q_tokens if t in txt_l)
            if score<min_score:continue
            scored.append((score,key))
        scored.sort(reverse=True)
        out=[]
        for score,key in scored[:k]:
            txt=self.kb.lookup(key) or ''
            if len(txt)>max_chars_per:txt=txt[:max_chars_per]+'...'
            out.append((key,txt,score))
        return out
    def format_as_context(self,results):
        if not results:return ''
        lines=['Reference docs (retrieved from knowledge base):']
        for key,txt,score in results:
            lines.append(f'--- {key} (score={score})')
            lines.append(txt)
        return '\n'.join(lines)
    def stats(self):return self.kb.stats()
    def close(self):self.kb.close()
