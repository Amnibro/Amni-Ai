"""EmbeddingCosineRetriever: dense-vector retrieval as TF-score sidecar.
Per memory feedback_amni_ai_tf_score_inverts.md: KBRetriever TF score INVERTS on math word problems; cosine on a small embedding model is the real fix. Uses sentence-transformers all-MiniLM-L6-v2 (~80MB) for query+key vectors, cached to disk under <kb_root>/_emb_cache.npy.
Drop-in interface match with KBRetriever: retrieve(query,k,max_chars_per,min_score,slug) returns (key,text,score) triples; format_as_context(results) returns ready-to-prepend context block; stats() returns kb.stats().
"""
import json,hashlib,numpy as np
from pathlib import Path
from typing import List,Tuple,Optional
from amni.learning.knowledge_base import KnowledgeBase
_MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2'
class EmbeddingCosineRetriever:
    def __init__(self,kb_root:str,model_name:str=_MODEL_NAME,device:str='cpu',rebuild_cache:bool=False):
        self.kb=KnowledgeBase(kb_root)
        self.kb_root=Path(kb_root)
        self.model_name=model_name
        self.device=device
        self._model=None
        self._cache_path=self.kb_root/'_emb_cache.npz'
        self._meta_path=self.kb_root/'_emb_cache_meta.json'
        self._key_to_idx={}
        self._embs=None
        self._keys=[]
        self._load_or_build_cache(rebuild=rebuild_cache)
    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer,AutoModel
            tok=AutoTokenizer.from_pretrained(self.model_name)
            mdl=AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            class _MiniEncoder:
                def __init__(s,t,m,dev):s.t=t;s.m=m;s.dev=dev
                def encode(s,texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False):
                    out=[]
                    for i in range(0,len(texts),batch_size):
                        b=texts[i:i+batch_size]
                        enc=s.t(b,padding=True,truncation=True,max_length=256,return_tensors='pt').to(s.dev)
                        with torch.no_grad():o=s.m(**enc)
                        h=o.last_hidden_state
                        mask=enc['attention_mask'].unsqueeze(-1).float()
                        pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-6)
                        if normalize_embeddings:pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
                        out.append(pooled.cpu().numpy())
                    import numpy as np
                    return np.concatenate(out,axis=0) if convert_to_numpy else out
            self._model=_MiniEncoder(tok,mdl,self.device)
        return self._model
    def _kb_signature(self):
        keys=self.kb.keys();h=hashlib.sha256()
        for k in keys[:1000]:h.update(k.encode('utf-8'))
        h.update(str(len(keys)).encode())
        return h.hexdigest()[:16]
    def _load_or_build_cache(self,rebuild:bool=False):
        sig=self._kb_signature()
        if not rebuild and self._cache_path.exists() and self._meta_path.exists():
            try:
                meta=json.loads(self._meta_path.read_text())
                if meta.get('sig')==sig and meta.get('model')==self.model_name:
                    d=np.load(self._cache_path,allow_pickle=True)
                    self._embs=d['embs'];self._keys=list(d['keys'])
                    self._key_to_idx={k:i for i,k in enumerate(self._keys)}
                    return
            except Exception as ex:print(f'  [warn] cache load failed: {ex}; rebuilding',flush=True)
        self._build_cache(sig)
    def _build_cache(self,sig:str):
        m=self._ensure_model()
        keys=self.kb.keys()
        texts=[]
        for k in keys:
            t=self.kb.lookup(k) or ''
            texts.append(f'{k}\n{t[:512]}')
        embs=m.encode(texts,batch_size=64,show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True)
        self._embs=embs.astype(np.float32);self._keys=list(keys)
        self._key_to_idx={k:i for i,k in enumerate(self._keys)}
        from pathlib import Path as _P
        tmp_npz=_P(str(self._cache_path)+'.tmp.npz')
        tmp_meta=_P(str(self._meta_path)+'.tmp')
        np.savez(str(tmp_npz)[:-4],embs=self._embs,keys=np.array(self._keys,dtype=object))
        tmp_meta.write_text(json.dumps({'sig':sig,'model':self.model_name,'n':len(keys)}))
        import os
        os.replace(str(tmp_npz),str(self._cache_path))
        os.replace(str(tmp_meta),str(self._meta_path))
    def retrieve(self,query:str,k:int=3,max_chars_per:int=600,min_score:float=0.2,slug:Optional[str]=None)->List[Tuple[str,str,float]]:
        if self._embs is None or len(self._keys)==0:return []
        m=self._ensure_model()
        q=m.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0].astype(np.float32)
        scores=self._embs@q
        if slug:
            slug_prefix=f'{slug}::'
            mask=np.array([kk.startswith(slug_prefix) for kk in self._keys])
            scores=np.where(mask,scores,-1.0)
        order=np.argsort(-scores)[:k*4]
        out=[]
        for i in order:
            s=float(scores[i])
            if s<min_score:break
            key=self._keys[i]
            txt=self.kb.lookup(key) or ''
            if len(txt)>max_chars_per:txt=txt[:max_chars_per]+'...'
            out.append((key,txt,s))
            if len(out)>=k:break
        return out
    def format_as_context(self,results)->str:
        if not results:return ''
        lines=['Reference docs (retrieved from knowledge base):']
        for key,txt,score in results:
            lines.append(f'--- {key} (cos={score:.3f})')
            lines.append(txt)
        return '\n'.join(lines)
    def stats(self):
        s=self.kb.stats()
        s['emb_indexed']=len(self._keys);s['emb_dim']=int(self._embs.shape[1]) if self._embs is not None else 0
        return s
