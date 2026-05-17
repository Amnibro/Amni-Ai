"""LUT softmax: Talos-inspired precomputed exp lookup, Triton-fused on GPU.
Replaces torch.softmax exp with addressed-memory reads + linear interp.
Reference (pure PyTorch) path kept for CPU/fallback; Triton kernel is the GPU production path.
"""
import torch
_X_RANGE=12.0
_LUT_SIZE=4096
_LUT_CACHE={}
def _build_lut(device,dtype):
    xs=torch.linspace(-_X_RANGE,_X_RANGE,_LUT_SIZE,device=device,dtype=torch.float32)
    return torch.exp(xs).to(dtype)
def _get_lut(device,dtype):
    key=(str(device),dtype)
    if key not in _LUT_CACHE:_LUT_CACHE[key]=_build_lut(device,dtype)
    return _LUT_CACHE[key]
try:
    import triton
    import triton.language as tl
    _HAVE_TRITON=True
    @triton.jit
    def _lut_softmax_kernel(x_ptr,lut_ptr,out_ptr,N,X_RANGE:tl.constexpr,LUT_SIZE:tl.constexpr,BLOCK_N:tl.constexpr):
        row=tl.program_id(0)
        cols=tl.arange(0,BLOCK_N)
        mask=cols<N
        x=tl.load(x_ptr+row*N+cols,mask=mask,other=-1e30).to(tl.float32)
        x_max=tl.max(x,axis=0)
        x_shift=x-x_max
        step=(2.0*X_RANGE)/(LUT_SIZE-1)
        pos=(x_shift+X_RANGE)/step
        pos_c=tl.minimum(tl.maximum(pos,0.0),float(LUT_SIZE-2))
        i0=pos_c.to(tl.int32)
        frac=pos_c-i0.to(tl.float32)
        e0=tl.load(lut_ptr+i0).to(tl.float32)
        e1=tl.load(lut_ptr+i0+1).to(tl.float32)
        e=e0+(e1-e0)*frac
        e=tl.where(mask,e,0.0)
        s=tl.sum(e,axis=0)
        y=e/s
        tl.store(out_ptr+row*N+cols,y,mask=mask)
except ImportError:
    _HAVE_TRITON=False
def lut_exp_torch(x):
    lut=_get_lut(x.device,x.dtype)
    n=lut.numel();step=(2*_X_RANGE)/(n-1)
    xc=x.clamp(-_X_RANGE,_X_RANGE);pos=(xc+_X_RANGE)/step
    i0=pos.long().clamp(0,n-2);frac=pos-i0.to(pos.dtype)
    return lut[i0]+(lut[i0+1]-lut[i0])*frac
def lut_softmax_torch(x,dim=-1):
    x_max=x.max(dim=dim,keepdim=True).values
    e=lut_exp_torch(x-x_max)
    return e/e.sum(dim=dim,keepdim=True)
def lut_softmax(x,dim=-1):
    if not _HAVE_TRITON or not x.is_cuda:return lut_softmax_torch(x,dim=dim)
    if dim!=-1 and dim!=x.ndim-1:return lut_softmax_torch(x,dim=dim)
    orig_shape=x.shape
    N=orig_shape[-1]
    if N>4096:return lut_softmax_torch(x,dim=dim)
    M=x.numel()//N
    x_flat=x.contiguous().reshape(M,N).to(torch.float32)
    out=torch.empty_like(x_flat)
    lut=_get_lut(x.device,torch.float32)
    BLOCK_N=triton.next_power_of_2(N)
    _lut_softmax_kernel[(M,)](x_flat,lut,out,N,X_RANGE=_X_RANGE,LUT_SIZE=_LUT_SIZE,BLOCK_N=BLOCK_N,num_warps=4)
    return out.to(x.dtype).reshape(orig_shape)
