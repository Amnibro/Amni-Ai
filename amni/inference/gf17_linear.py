import torch,torch.nn as nn
from amni.inference.gf17_gemv_triton import gf17_quantize_rows,gf17_recover_rows,gf17_gemv
def _rows(W,recover):return gf17_recover_rows(W) if recover else gf17_quantize_rows(W)
class GF17Linear(nn.Module):
    def __init__(self,linear,recover=False):
        super().__init__()
        W=linear.weight.detach().float()
        idx,s=_rows(W,recover)
        self.register_buffer('idx',idx)
        self.register_buffer('s',s.half())
        self.out_features,self.in_features=W.shape
        self.register_buffer('bias',linear.bias.detach().half()) if linear.bias is not None else setattr(self,'bias',None)
    def forward(self,x):
        flat=x.reshape(-1,self.in_features)
        if flat.shape[0]==1:
            y=gf17_gemv(self.idx,self.s,flat[0].half().contiguous()).unsqueeze(0)
        else:
            W=(self.idx.to(torch.float16)-8)*self.s.unsqueeze(1)
            y=flat.half()@W.T
        y=y+self.bias if self.bias is not None else y
        return y.view(*x.shape[:-1],self.out_features).to(x.dtype)
class GF17Fused(nn.Module):
    def __init__(self,linears,recover=False):
        super().__init__()
        W=torch.cat([l.weight.detach().float() for l in linears],0)
        idx,s=_rows(W,recover)
        self.register_buffer('idx',idx)
        self.register_buffer('s',s.half())
        self.splits=[l.weight.shape[0] for l in linears]
        self.out_features,self.in_features=W.shape
        self._y=None;self._shape=None
    def compute(self,x,i):
        if i==0:
            flat=x.reshape(-1,self.in_features)
            if flat.shape[0]==1:
                y=gf17_gemv(self.idx,self.s,flat[0].half().contiguous()).unsqueeze(0)
            else:
                W=(self.idx.to(torch.float16)-8)*self.s.unsqueeze(1)
                y=flat.half()@W.T
            self._y=torch.split(y,self.splits,dim=-1);self._shape=x.shape
        return self._y[i].reshape(*self._shape[:-1],self.splits[i]).to(x.dtype)
class GF17Part(nn.Module):
    def __init__(self,hub,i):
        super().__init__()
        self._hub=[hub];self.i=i
        self.out_features=hub.splits[i];self.in_features=hub.in_features
    def forward(self,x):return self._hub[0].compute(x,self.i)
def convert_model_gf17(model,fuse=True,recover=False):
    n=0
    for L in model.model.layers:
        groups=((L.self_attn,('q_proj','k_proj','v_proj'),('o_proj',)),(L.mlp,('gate_proj','up_proj'),('down_proj',)))
        for parent,fused_names,solo_names in groups:
            lins=[getattr(parent,nm,None) for nm in fused_names]
            if fuse and all(isinstance(l,nn.Linear) and l.bias is None for l in lins):
                hub=GF17Fused(lins,recover)
                for i,nm in enumerate(fused_names):setattr(parent,nm,GF17Part(hub,i));n+=1
                parent._gf17_hub=hub
            else:
                for nm in fused_names:
                    old=getattr(parent,nm,None)
                    if isinstance(old,nn.Linear):setattr(parent,nm,GF17Linear(old,recover));n+=1
            for nm in solo_names:
                old=getattr(parent,nm,None)
                if isinstance(old,nn.Linear):setattr(parent,nm,GF17Linear(old,recover));n+=1
        torch.cuda.empty_cache()
    return n
