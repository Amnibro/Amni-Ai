import torch
def _scale(a,levels,dim):return a.abs().amax(dim=dim,keepdim=True).clamp_min(1e-8)/((levels-1)/2)
def _nearest(a,levels,dim):
    s=_scale(a,levels,dim);h=(levels-1)/2
    return torch.clamp(torch.round(a/s),-h,h)*s
def gf17_lattice_quantize(a,levels=17,dim=None):
    dim=tuple(range(a.dim())) if dim is None else dim
    q=_nearest(a,levels,dim);return a+(q-a).detach()
def lattice_penalty(a,levels=17,dim=None):
    dim=tuple(range(a.dim())) if dim is None else dim
    q=_nearest(a,levels,dim).detach();return ((a-q)**2).mean()
def cardinality_estimate(a,levels=17,dim=None):
    dim=tuple(range(a.dim())) if dim is None else dim
    s=_scale(a,levels,dim);h=(levels-1)/2;idx=torch.clamp(torch.round(a/s),-h,h).to(torch.int64)
    return int(torch.unique(idx).numel()) if idx.dim()<=1 else int(torch.unique(idx.reshape(idx.shape[0],-1),dim=0).shape[0])
def lattice_reg_total(acts,levels=17,weight=1.0):
    return weight*sum(lattice_penalty(a,levels=levels) for a in acts)/max(1,len(acts)) if acts else torch.zeros(())
