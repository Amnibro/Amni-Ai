import torch,torch.nn.functional as F,triton,triton.language as tl
@triton.jit
def _causal_conv1d_kernel(x_ptr,w_ptr,b_ptr,out_ptr,B,D,L,sxb,sxd,sxl,swd,swk,sob,sod,sol,K:tl.constexpr,BLOCK_L:tl.constexpr,HAS_BIAS:tl.constexpr,APPLY_SILU:tl.constexpr):
    pid_bd=tl.program_id(0);pid_l=tl.program_id(1)
    b=pid_bd//D;d=pid_bd%D
    bias=tl.load(b_ptr+d).to(tl.float32) if HAS_BIAS else 0.0
    l_start=pid_l*BLOCK_L
    l_off=l_start+tl.arange(0,BLOCK_L)
    accs=tl.zeros([BLOCK_L],dtype=tl.float32)
    for k in tl.static_range(K):
        xl=l_off-(K-1)+k
        m=(l_off<L)&(xl>=0)&(xl<L)
        xv=tl.load(x_ptr+b*sxb+d*sxd+xl*sxl,mask=m,other=0.0).to(tl.float32)
        wk=tl.load(w_ptr+d*swd+k*swk).to(tl.float32)
        accs+=xv*wk
    accs=accs+bias
    if APPLY_SILU:accs=accs*tl.sigmoid(accs)
    om=l_off<L
    tl.store(out_ptr+b*sob+d*sod+l_off*sol,accs.to(out_ptr.dtype.element_ty),mask=om)
@triton.jit
def _causal_conv1d_update_kernel(x_ptr,s_ptr,w_ptr,b_ptr,out_ptr,B,D,sxb,sxd,ssb,ssd,ssk,swd,swk,sob,sod,K:tl.constexpr,HAS_BIAS:tl.constexpr,APPLY_SILU:tl.constexpr):
    pid=tl.program_id(0)
    b=pid//D;d=pid%D
    k_off=tl.arange(0,K)
    w=tl.load(w_ptr+d*swd+k_off*swk).to(tl.float32)
    state=tl.load(s_ptr+b*ssb+d*ssd+k_off*ssk).to(tl.float32)
    x_new=tl.load(x_ptr+b*sxb+d*sxd).to(tl.float32)
    shifted_idx=k_off+1
    shifted_mask=shifted_idx<K
    next_state=tl.load(s_ptr+b*ssb+d*ssd+shifted_idx*ssk,mask=shifted_mask,other=0.0).to(tl.float32)
    new_state=tl.where(k_off<K-1,next_state,x_new)
    tl.store(s_ptr+b*ssb+d*ssd+k_off*ssk,new_state.to(s_ptr.dtype.element_ty))
    acc=tl.sum(new_state*w)
    if HAS_BIAS:
        bias=tl.load(b_ptr+d).to(tl.float32)
        acc=acc+bias
    if APPLY_SILU:acc=acc*tl.sigmoid(acc)
    tl.store(out_ptr+b*sob+d*sod,acc.to(out_ptr.dtype.element_ty))
_DEBUG_CC1D=False
def causal_conv1d_fn(x,weight,bias=None,activation=None,seq_idx=None,initial_states=None,return_final_states=False,final_states_out=None,**_unused):
    assert activation in (None,'silu','swish'),f'unsupported activation {activation}'
    assert seq_idx is None,'seq_idx variable-length packing not supported'
    assert initial_states is None and not return_final_states,'final-state IO not supported in this Triton port'
    if _DEBUG_CC1D:print(f'[cc1d_fn] x.shape={tuple(x.shape)} dim={x.dim()} contig={x.is_contiguous()} weight.shape={tuple(weight.shape)} bias={None if bias is None else tuple(bias.shape)} act={activation}',flush=True)
    assert x.dim()==3,f'expected 3D x (B,D,L), got shape {tuple(x.shape)}'
    B,D,L=x.shape;K=weight.shape[-1]
    if weight.dim()==3:weight=weight.squeeze(1)
    out=torch.empty_like(x)
    BLOCK_L=64
    grid=(B*D,triton.cdiv(L,BLOCK_L))
    _causal_conv1d_kernel[grid](x,weight,bias if bias is not None else torch.empty(0,device=x.device,dtype=x.dtype),out,B,D,L,x.stride(0),x.stride(1),x.stride(2),weight.stride(0),weight.stride(1),out.stride(0),out.stride(1),out.stride(2),K=K,BLOCK_L=BLOCK_L,HAS_BIAS=bias is not None,APPLY_SILU=activation in ('silu','swish'))
    if _DEBUG_CC1D:print(f'[cc1d_fn] out.shape={tuple(out.shape)} dim={out.dim()}',flush=True)
    return out
def causal_conv1d_update(x,conv_state,weight,bias=None,activation=None,cache_seqlens=None,conv_state_indices=None,**_unused):
    assert activation in (None,'silu','swish'),f'unsupported activation {activation}'
    assert cache_seqlens is None and conv_state_indices is None,'variable cache layout not supported'
    was_3d=x.dim()==3
    if was_3d:
        assert x.shape[-1]==1,'update path expects single timestep'
        x=x.squeeze(-1)
    B,D=x.shape
    if weight.dim()==3:weight=weight.squeeze(1)
    K=weight.shape[-1]
    out=torch.empty_like(x)
    grid=(B*D,)
    _causal_conv1d_update_kernel[grid](x,conv_state,weight,bias if bias is not None else torch.empty(0,device=x.device,dtype=x.dtype),out,B,D,x.stride(0),x.stride(1),conv_state.stride(0),conv_state.stride(1),conv_state.stride(2),weight.stride(0),weight.stride(1),out.stride(0),out.stride(1),K=K,HAS_BIAS=bias is not None,APPLY_SILU=activation in ('silu','swish'))
    return out.unsqueeze(-1) if was_3d else out
def causal_conv1d_torch_ref(x,weight,bias=None,activation=None):
    B,D,L=x.shape
    if weight.dim()==2:weight=weight.unsqueeze(1)
    K=weight.shape[-1]
    x_pad=F.pad(x,(K-1,0))
    out=F.conv1d(x_pad,weight,bias,groups=D)
    return F.silu(out) if activation in ('silu','swish') else out
def causal_conv1d_update_torch_ref(x,conv_state,weight,bias=None,activation=None):
    if x.dim()==3:x=x.squeeze(-1)
    K=weight.shape[-1] if weight.dim()==2 else weight.shape[2]
    if weight.dim()==3:weight=weight.squeeze(1)
    new_state=torch.cat([conv_state[...,1:],x.unsqueeze(-1)],dim=-1)
    conv_state.copy_(new_state)
    out=(new_state*weight.unsqueeze(0)).sum(dim=-1)
    if bias is not None:out=out+bias
    return F.silu(out) if activation in ('silu','swish') else out
