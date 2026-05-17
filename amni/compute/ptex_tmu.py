import torch
import numpy as np
_DEVICE=None
_KERNELS=None
_BLOCK=1024
def _dev():
    global _DEVICE
    _DEVICE=_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE
def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:return _KERNELS
    try:
        import triton
        import triton.language as tl
        @triton.jit
        def _decode_kernel(px_ptr,out_ptr,n,BLOCK:tl.constexpr):
            pid=tl.program_id(0);off=pid*BLOCK+tl.arange(0,BLOCK);mask=off<n;base=off*8
            d0=tl.load(px_ptr+base+0,mask=mask,other=0).to(tl.uint32)
            d1=tl.load(px_ptr+base+1,mask=mask,other=0).to(tl.uint32)
            d2=tl.load(px_ptr+base+2,mask=mask,other=0).to(tl.uint32)
            d3=tl.load(px_ptr+base+3,mask=mask,other=0).to(tl.uint32)
            d4=tl.load(px_ptr+base+4,mask=mask,other=0).to(tl.uint32)
            d5=tl.load(px_ptr+base+5,mask=mask,other=0).to(tl.uint32)
            d6=tl.load(px_ptr+base+6,mask=mask,other=0).to(tl.uint32)
            d7=tl.load(px_ptr+base+7,mask=mask,other=0).to(tl.uint32)
            v=d0+d1*17+d2*289+d3*4913+d4*83521+d5*1419857+d6*24137569+d7*410338673
            tl.store(out_ptr+off,v.to(tl.float32),mask=mask)
        @triton.jit
        def _encode_kernel(inp_ptr,px_ptr,n,BLOCK:tl.constexpr):
            pid=tl.program_id(0);off=pid*BLOCK+tl.arange(0,BLOCK);mask=off<n
            v=tl.load(inp_ptr+off,mask=mask,other=0).to(tl.uint32);base=off*8
            d0=v%17;v=v//17;d1=v%17;v=v//17;d2=v%17;v=v//17;d3=v%17;v=v//17
            d4=v%17;v=v//17;d5=v%17;v=v//17;d6=v%17;v=v//17;d7=v%17
            tl.store(px_ptr+base+0,d0.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+1,d1.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+2,d2.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+3,d3.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+4,d4.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+5,d5.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+6,d6.to(tl.uint8),mask=mask)
            tl.store(px_ptr+base+7,d7.to(tl.uint8),mask=mask)
        _KERNELS=(_decode_kernel,_encode_kernel)
    except ImportError:
        _KERNELS=False
    return _KERNELS
def decode_f32b17_gpu(pixels_flat:torch.Tensor)->torch.Tensor:
    k=_get_kernels();n=pixels_flat.numel()//8
    out=torch.empty(n,dtype=torch.float32,device=pixels_flat.device)
    if k:k[0][((n+_BLOCK-1)//_BLOCK,)](pixels_flat.contiguous(),out,n,BLOCK=_BLOCK)
    else:
        v=pixels_flat.view(-1,8).to(torch.int64)
        K=torch.tensor([1,17,289,4913,83521,1419857,24137569,410338673],dtype=torch.int64,device=pixels_flat.device)
        out.copy_((v*K).sum(dim=1).to(torch.int32).view(torch.float32))
    return out
def encode_f32b17_gpu(weights:torch.Tensor)->torch.Tensor:
    k=_get_kernels();flat=weights.contiguous().view(-1);n=flat.numel()
    raw=flat.view(torch.uint32);px=torch.empty(n*8,dtype=torch.uint8,device=flat.device)
    if k:k[1][((n+_BLOCK-1)//_BLOCK,)](raw,px,n,BLOCK=_BLOCK)
    else:
        v=raw.to(torch.int64).unsqueeze(1)
        digits=torch.zeros(n,8,dtype=torch.uint8,device=flat.device)
        for i in range(8):digits[:,i]=(v[:,0]%17).to(torch.uint8);v=v//17
        px.copy_(digits.view(-1))
    return px.view(-1,4)
def decode_f32b17(pixels_np:np.ndarray,n_weights:int)->np.ndarray:
    dev=_dev()
    if dev=='cpu':
        K=np.array([1,17,289,4913],dtype=np.uint64);Kh=np.array([83521,1419857,24137569,410338673],dtype=np.uint64)
        C=1<<22;out=np.empty(n_weights,dtype=np.float32)
        for s in range(0,n_weights,C):
            e=min(s+C,n_weights);lo=pixels_np[2*s:2*e:2].astype(np.uint64);hi=pixels_np[2*s+1:2*e:2].astype(np.uint64)
            out[s:e]=np.frombuffer(((lo*K).sum(axis=1)+(hi*Kh).sum(axis=1)).astype(np.uint32).tobytes(),dtype=np.float32)
        return out
    flat=torch.from_numpy(pixels_np[:n_weights*2].ravel()).to(dev)
    return decode_f32b17_gpu(flat).cpu().numpy()
def encode_f32b17(f32_values:np.ndarray)->'tuple[np.ndarray,int]':
    dev=_dev()
    if dev=='cpu':
        raw=np.frombuffer(f32_values.astype(np.float32).tobytes(),dtype=np.uint32)
        n=raw.size;px=np.empty((n*2,4),dtype=np.uint8);C=1<<22
        for s in range(0,n,C):
            e=min(s+C,n);v=raw[s:e].astype(np.uint64)
            for j in range(4):px[2*s:2*e:2,j]=(v%17).astype(np.uint8);v//=17
            for j in range(4):px[2*s+1:2*e:2,j]=(v%17).astype(np.uint8);v//=17
        return px,n
    t=torch.from_numpy(f32_values.astype(np.float32)).to(dev)
    px_gpu=encode_f32b17_gpu(t)
    return px_gpu.cpu().numpy(),f32_values.size
