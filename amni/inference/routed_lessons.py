"""RoutedSemanticLUT — federated map-PTEX lesson store. Drop-in for SemanticPTEXLUT but partitioned by
(language, topic): each partition is its own small SemanticPTEXLUT "pack"; a tiny map routes a query to the
1-2 relevant packs so only those load + refit. Distant packs stay cold (zero-load) — the Reffelt nonce idea.
Wins: teach refits ONE small pack (not all N) → no 130s/24GB; recall searches within-type → sharp; one
shared CPU encoder across packs → no MiniLM duplication. Enable with AMNI_ROUTED_LESSONS=1."""
import os,json,re
from pathlib import Path
from typing import Optional,List,Tuple
from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
_LANGS=[('python',('python',' py ','def ','import ','lambda','numpy','pandas','pytest','async def')),('rust',('rust','cargo','fn ','impl ','trait ','borrow','&mut','tokio')),('javascript',('javascript',' js ','node','npm','const ','=>','typescript',' ts ')),('go',('golang',' go ','goroutine','func ')),('cpp',('c++','cpp','std::','template<','#include')),('java',('java','public class','void ','jvm')),('sql',('sql','select ','insert into','join ','where ')),('bash',('bash','shell script','#!/bin'))]
_TOPICS=[('sort',('sort','order','rank','ascending','descending')),('string',('string','substring','palindrome','anagram','char','reverse a ')),('search',('search','find','lookup','binary search','index of')),('math',('math','prime','factorial','fibonacci','gcd','arithmetic','sum of')),('dedup',('duplicate','dedup','unique','distinct')),('io',('file','read','write','path','directory','stream')),('concurrency',('thread','async','await','lock','queue','parallel','mutex')),('web',('http',' request ',' url ',' api ','fetch',' server ','endpoint')),('data',('list','dict','set','tuple','array','map','hash')),('recursion',('recursion','recursive','backtrack')),('test',('test','assert','unittest','pytest'))]
def classify_key(text:str)->str:
    t=(' '+(text or '').lower()+' ')
    lang=next((name for name,kws in _LANGS if any(k in t for k in kws)),'general')
    topic=next((name for name,kws in _TOPICS if any(k in t for k in kws)),'general')
    return f'{lang}:{topic}'
def _safe(name:str)->str:return re.sub(r'[^a-z0-9_]+','_',name.lower())
class RoutedSemanticLUT:
    def __init__(self,grid:int=64,pca_dim:int=8,encoder=None,root:Optional[str]=None):
        self.grid=grid;self.pca_dim=pca_dim;self._encoder=encoder
        self.root=Path(root) if root else None
        self._raw:List[Tuple[str,str]]=[]
        self._keys:List[str]=[]
        self._packs={}
        self._dirty=set()
    def _ensure_encoder(self):
        if self._encoder is None:
            self._encoder=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim)._ensure_encoder()
        return self._encoder
    def _pack(self,key:str)->SemanticPTEXLUT:
        if key not in self._packs:
            p=None
            if self.root is not None:
                sp=self.root/('pack_'+_safe(key))
                if (sp.with_suffix('.npz')).exists():
                    try:p=SemanticPTEXLUT.load(str(sp),encoder=self._ensure_encoder())
                    except Exception:p=None
            self._packs[key]=p or SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self._ensure_encoder())
        return self._packs[key]
    def add(self,question:str,answer:str):
        key=classify_key(question)
        self._raw.append((question,answer));self._keys.append(key)
        self._pack(key).add(question,answer);self._dirty.add(key)
    def purge_indices(self,indices):
        idx=set(int(i) for i in indices)
        affected={self._keys[i] for i in idx if i<len(self._keys)}
        keep=[(self._raw[i],self._keys[i]) for i in range(len(self._raw)) if i not in idx]
        n=len(self._raw);self._raw=[p for p,k in keep];self._keys=[k for p,k in keep]
        for k in affected:
            self._packs[k]=SemanticPTEXLUT(grid=self.grid,pca_dim=self.pca_dim,encoder=self._ensure_encoder())
            for (q,a),kk in zip(self._raw,self._keys):
                if kk==k:self._packs[k].add(q,a)
            self._dirty.add(k)
        self.fit()
        return n-len(self._raw)
    def fit(self):
        for key in list(self._dirty):
            try:self._pack(key).fit()
            except Exception as e:print(f'[routed] pack {key} fit failed: {e}',flush=True)
        self._dirty.clear()
    def _route(self,q:str)->List[SemanticPTEXLUT]:
        key=classify_key(q);out=[self._pack(key)]
        if key!='general:general':out.append(self._pack('general:general'))
        return out
    def lookup(self,q:str):
        for p in self._route(q):
            if p._cells:
                hit=p.lookup(q)
                if hit is not None:return hit
        return None
    def lookup_soft(self,q:str,k:int=3,tol:int=None,cos_gate:float=None,margin:float=None,return_diag:bool=False):
        best=None
        for p in self._route(q):
            if not p._cells:continue
            r=p.lookup_soft(q,k=k,tol=tol,cos_gate=cos_gate,margin=margin,return_diag=return_diag)
            hit=r[0] if (return_diag and isinstance(r,tuple)) else r
            if hit is not None:return r
            if best is None:best=r
        return best if best is not None else ((None,None,None,None) if return_diag else None)
    def auto_margin(self)->float:
        ms=[p.auto_margin() for p in self._packs.values() if p._cells]
        return sum(ms)/len(ms) if ms else 0.08
    def save(self,path:str):
        base=Path(path);base.parent.mkdir(parents=True,exist_ok=True)
        self.root=base.parent/(base.name+'_packs')
        self.root.mkdir(parents=True,exist_ok=True)
        packs={}
        for key,p in self._packs.items():
            if not p._raw:continue
            fn=self.root/('pack_'+_safe(key))
            try:p.save(str(fn));packs[key]={'pairs':len(p._raw),'cells':len(p._cells),'file':fn.name}
            except Exception as e:print(f'[routed] save pack {key} failed: {e}',flush=True)
        mp={'version':1,'grid':self.grid,'pca_dim':self.pca_dim,'packs':packs,'raw':[[q,a] for q,a in self._raw],'keys':self._keys}
        (base.with_suffix('.map.json')).write_text(json.dumps(mp,ensure_ascii=False),encoding='utf-8')
    @classmethod
    def load(cls,path:str,encoder=None):
        base=Path(path);mapf=base.with_suffix('.map.json')
        obj=cls(encoder=encoder,root=str(base.parent/(base.name+'_packs')))
        if mapf.exists():
            try:
                mp=json.loads(mapf.read_text(encoding='utf-8'))
                obj.grid=mp.get('grid',64);obj.pca_dim=mp.get('pca_dim',8)
                obj._raw=[(q,a) for q,a in mp.get('raw',[])]
                obj._keys=mp.get('keys') or [classify_key(q) for q,_ in obj._raw]
            except Exception as e:print(f'[routed] load map failed: {e}',flush=True)
        return obj
    @classmethod
    def from_flat(cls,flat,encoder=None,root:Optional[str]=None):
        obj=cls(grid=getattr(flat,'grid',64),pca_dim=getattr(flat,'pca_dim',8),encoder=encoder or getattr(flat,'encoder',None),root=root)
        for q,a in getattr(flat,'_raw',[]):obj.add(q,a)
        obj.fit()
        return obj
    def stats(self):
        return {'pairs':len(self._raw),'packs':len(self._packs),'unique_cells':sum(len(p._cells) for p in self._packs.values()),'pack_sizes':{k:len(p._raw) for k,p in self._packs.items()}}
