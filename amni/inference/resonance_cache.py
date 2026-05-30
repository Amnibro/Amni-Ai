import numpy as np
from collections import defaultdict
def _proj_codes(x,mu,Vk,levels,scale=None):
    xp=(x-mu)@Vk.T;h=(levels-1)/2
    if scale is None:s=np.abs(xp).max(0,keepdims=True);scale=np.where(s<1e-8,1.0,s)/h
    return np.clip(np.round(xp/scale),-h,h).astype(np.int16),scale
class ResonanceCache:
    def __init__(self,Vk,Vr,mu,scale,base,meanx,kidx,M,levels):
        self.Vk=Vk;self.Vr=Vr;self.mu=mu;self.scale=scale;self.base=base;self.meanx=meanx;self.kidx=kidx;self.M=M;self.levels=levels
    @classmethod
    def build(cls,X,W,pcs=4,resid_pcs=8,levels=17,min_support=4):
        X=np.ascontiguousarray(X,np.float32);W=np.ascontiguousarray(W,np.float32)
        mu=X.mean(0,keepdims=True);_,_,Vt=np.linalg.svd((X-mu)[:4000],full_matrices=False);Vk=Vt[:pcs];Vr=Vt[pcs:pcs+resid_pcs]
        codes,scale=_proj_codes(X,mu,Vk,levels);cells=defaultdict(list)
        for i,c in enumerate(codes):cells[c.tobytes()].append(i)
        ks=[k for k,idx in cells.items() if len(idx)>=min_support]
        base=np.stack([X[cells[k]].mean(0)@W.T for k in ks]).astype(np.float32) if ks else np.zeros((0,W.shape[0]),np.float32)
        meanx=np.stack([X[cells[k]].mean(0) for k in ks]).astype(np.float32) if ks else np.zeros((0,W.shape[1]),np.float32)
        return cls(Vk.astype(np.float32),Vr.astype(np.float32),mu.astype(np.float32),scale.astype(np.float32),base,meanx,{k:i for i,k in enumerate(ks)},(W@Vr.T).astype(np.float32),levels)
    def lookup(self,X):
        X=np.ascontiguousarray(X,np.float32);n=X.shape[0];hit=np.zeros(n,bool)
        if self.base.shape[0]==0:return np.zeros((n,self.M.shape[0]),np.float32),hit
        codes,_=_proj_codes(X,self.mu,self.Vk,self.levels,self.scale);y=np.zeros((n,self.base.shape[1]),np.float32)
        for i in range(n):
            ci=self.kidx.get(codes[i].tobytes())
            if ci is None:continue
            y[i]=self.base[ci]+self.M@(self.Vr@(X[i]-self.meanx[ci]));hit[i]=True
        return y,hit
    def stats(self):return {'cells':len(self.kidx),'pcs':self.Vk.shape[0],'resid_pcs':self.Vr.shape[0],'levels':self.levels,'out':int(self.M.shape[0]),'in':int(self.Vk.shape[1])}
