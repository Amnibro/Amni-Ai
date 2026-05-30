"""SemanticPTEXLUT — proposed tier-1.5 between hash-LUT and embed-template.
Built from probe v0.5 finding (2026-05-14): MiniLM + PCA-8D + grid achieves 100% paraphrase recovery on 24-paraphrase test, vs 0% for hash-LUT.
Architecture: each Q is encoded via MiniLM-L6-v2 -> 384-dim, projected via fitted PCA-8D, discretized to N-dim grid coordinate (one PTEX "address"). The answer is stored at that address. Lookup hashes a query to its address; cell-hit returns the stored answer in zero LLM inference.
This is the practical first slice of the maintainer's PTEX-arrangement vision: store coordinates, not numbers; retrieve by spatial layout, not by autoregressive decoding.
Tier-1.5 fires only on cells with a stored answer (no soft NN by default to keep precision high). Soft-NN mode available via `lookup_soft(k=3)`.
"""
import numpy as np
from pathlib import Path
import json
def _kmeans(X,K,iters=30,seed=42):
    rng=np.random.RandomState(seed)
    N=X.shape[0]
    if K>=N:return X.copy(),np.arange(N)
    idx=rng.choice(N,K,replace=False)
    centroids=X[idx].copy()
    for _ in range(iters):
        d=-X@centroids.T
        assignments=np.argmin(d,axis=1)
        new_centroids=np.zeros_like(centroids)
        for k in range(K):
            mask=assignments==k
            if mask.sum()>0:new_centroids[k]=X[mask].mean(axis=0)
            else:new_centroids[k]=centroids[k]
        new_centroids/=np.linalg.norm(new_centroids,axis=1,keepdims=True)+1e-12
        if np.allclose(centroids,new_centroids):break
        centroids=new_centroids
    return centroids,assignments
class SemanticPTEXLUT:
    def __init__(self,grid:int=64,pca_dim:int=8,encoder=None,routing:str='flat',k_clusters:int=32):
        self.grid=grid;self.pca_dim=pca_dim;self.encoder=encoder
        self.routing=routing;self.k_clusters=k_clusters
        self._raw=[];self._cells={};self._pca_mean=None;self._pca_Vt=None
        self._cmin=None;self._cmax=None;self._stored_embs=None
        self._centroids=None;self._cluster_embs=[];self._cluster_ans=[];self._cluster_indices=[]
    def _ensure_encoder(self):
        if self.encoder is not None:return self.encoder
        import torch
        from transformers import AutoTokenizer,AutoModel
        name='sentence-transformers/all-MiniLM-L6-v2'
        dev='cuda' if torch.cuda.is_available() else 'cpu'
        tok=AutoTokenizer.from_pretrained(name)
        mdl=AutoModel.from_pretrained(name).eval().to(dev)
        from amni.inference.gpu_queue import run_on_gpu
        def enc(texts):
            e=tok(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
            def _job():
                ee={k:v.to(dev) for k,v in e.items()}
                with torch.no_grad():o=mdl(**ee)
                h=o.last_hidden_state;mask=ee['attention_mask'].unsqueeze(-1).float()
                p=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-6)
                p=torch.nn.functional.normalize(p,p=2,dim=1)
                return p.cpu().numpy().astype('float32')
            return run_on_gpu(_job)
        try:_=enc(['warmup'])
        except Exception:pass
        self.encoder=enc
        return self.encoder
    def add(self,question:str,answer:str):
        self._raw.append((question,answer))
    def fit(self):
        enc=self._ensure_encoder()
        qs=[q for q,_ in self._raw]
        embs=enc(qs)
        self._stored_embs=embs
        centered=embs-embs.mean(axis=0)
        self._pca_mean=embs.mean(axis=0)
        _,_,Vt=np.linalg.svd(centered,full_matrices=False)
        self._pca_Vt=Vt[:self.pca_dim]
        proj=embs@self._pca_Vt.T
        self._cmin=proj.min(axis=0);self._cmax=proj.max(axis=0)
        span=self._cmax-self._cmin+1e-6
        for i,(q,a) in enumerate(self._raw):
            cell=tuple(int(c) for c in np.clip(((proj[i]-self._cmin)/span*self.grid).astype(int),0,self.grid-1))
            if cell not in self._cells:self._cells[cell]={'q':q,'a':a,'idx':i}
        if self.routing=='kmeans' and len(self._raw)>=self.k_clusters:
            self._centroids,assignments=_kmeans(self._stored_embs,self.k_clusters)
            self._cluster_embs=[None]*self.k_clusters
            self._cluster_ans=[None]*self.k_clusters
            self._cluster_indices=[None]*self.k_clusters
            for k in range(self.k_clusters):
                mask=assignments==k
                self._cluster_embs[k]=self._stored_embs[mask]
                self._cluster_ans[k]=[self._raw[i][1] for i,m in enumerate(mask) if m]
                self._cluster_indices[k]=np.where(mask)[0]
    def _project(self,q:str):
        enc=self._ensure_encoder()
        e=enc([q])[0]
        proj=e@self._pca_Vt.T
        span=self._cmax-self._cmin+1e-6
        cell=tuple(int(c) for c in np.clip(((proj-self._cmin)/span*self.grid).astype(int),0,self.grid-1))
        return cell,proj,e
    def lookup(self,q:str):
        if not self._cells:return None
        cell,_,_=self._project(q)
        hit=self._cells.get(cell)
        return hit['a'] if hit else None
    def auto_margin(self)->float:
        n=len(self._raw)
        if n>=500:return 0.05
        if n>=100:return 0.08
        if n>=50:return 0.10
        if n>=20:return 0.15
        if n>=10:return 0.20
        if n>=5:return 0.30
        return 0.40
    def lookup_soft(self,q:str,k:int=3,tol:int=None,cos_gate:float=None,margin:float=None,return_diag:bool=False):
        if not self._cells:return (None,None,None,None) if return_diag else None
        if margin=='auto':margin=self.auto_margin()
        cell,proj,e=self._project(q)
        if self.routing=='kmeans' and self._centroids is not None:
            cd=self._centroids@e
            kbest=int(np.argmax(cd))
            if len(self._cluster_ans[kbest])==0:return (None,None,None,None) if return_diag else None
            cos=self._cluster_embs[kbest]@e
            order=np.argsort(-cos)
            top=float(cos[order[0]]);second=float(cos[order[1]]) if len(order)>1 else 0.0
            if margin is not None and top-second<margin:return (None,None,top,second) if return_diag else None
            if cos_gate is not None and top<cos_gate:return (None,None,top,second) if return_diag else None
            return (self._cluster_ans[kbest][order[0]],None,top,second) if return_diag else self._cluster_ans[kbest][order[0]]
        best=None;best_d=None
        for c2,v in self._cells.items():
            d=sum((cell[i]-c2[i])**2 for i in range(len(cell)))
            if best is None or d<best_d:best=c2;best_d=d
        if tol is not None and best_d>tol**2*len(cell):return (None,best_d,None,None) if return_diag else None
        hit=self._cells[best]
        cos_top=float(self._stored_embs[hit['idx']]@e) if self._stored_embs is not None else None
        cos_2nd=None
        if margin is not None and self._stored_embs is not None:
            all_cos=self._stored_embs@e
            sorted_cos=sorted(all_cos.tolist(),reverse=True)
            cos_2nd=sorted_cos[1] if len(sorted_cos)>1 else 0.0
            if cos_top-cos_2nd<margin:return (None,best_d,cos_top,cos_2nd) if return_diag else None
        if cos_gate is not None and cos_top is not None and cos_top<cos_gate:return (None,best_d,cos_top,cos_2nd) if return_diag else None
        return (hit['a'],best_d,cos_top,cos_2nd) if return_diag else hit['a']
    def stats(self):
        return {'pairs':len(self._raw),'unique_cells':len(self._cells),'collisions':len(self._raw)-len(self._cells),'grid':self.grid,'pca_dim':self.pca_dim}
    def save(self,path:str):
        import json
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(str(p),embs=self._stored_embs,Vt=self._pca_Vt,cmin=self._cmin,cmax=self._cmax,pca_mean=self._pca_mean)
        meta={'grid':self.grid,'pca_dim':self.pca_dim,'pairs':[(q,a) for q,a in self._raw]}
        (p.with_suffix('.json')).write_text(json.dumps(meta))
    @classmethod
    def load(cls,path:str,encoder=None):
        import json
        p=Path(path)
        d=np.load(str(p) if str(p).endswith('.npz') else str(p)+'.npz')
        meta=json.loads((p.with_suffix('.json')).read_text())
        lut=cls(grid=meta['grid'],pca_dim=meta['pca_dim'],encoder=encoder)
        lut._raw=[(q,a) for q,a in meta['pairs']]
        lut._stored_embs=d['embs']
        lut._pca_Vt=d['Vt']
        lut._cmin=d['cmin']
        lut._cmax=d['cmax']
        lut._pca_mean=d['pca_mean']
        proj=lut._stored_embs@lut._pca_Vt.T
        span=lut._cmax-lut._cmin+1e-6
        for i,(q,a) in enumerate(lut._raw):
            cell=tuple(int(c) for c in np.clip(((proj[i]-lut._cmin)/span*lut.grid).astype(int),0,lut.grid-1))
            if cell not in lut._cells:lut._cells[cell]={'q':q,'a':a,'idx':i}
        return lut
