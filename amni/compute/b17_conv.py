import torch,torch.nn as nn,torch.nn.functional as F,triton,triton.language as tl,numpy as np
def encode_conv_weight_to_b17(w):
    assert w.dtype==torch.float16 and w.ndim==4
    Cout,Cin,Kh,Kw=w.shape
    arr=w.detach().cpu().contiguous().view(Cout,-1).numpy().view(np.uint16).astype(np.uint32)
    d0=arr%17;d1=(arr//17)%17;d2=(arr//289)%17;d3=(arr//4913)%17
    packed=(d0|(d1<<8)|(d2<<16)|(d3<<24)).astype(np.uint32)
    return torch.from_numpy(packed).to(w.device).contiguous()
def decode_b17_to_weight(w_b17,shape):
    arr=w_b17.detach().cpu().contiguous().numpy().astype(np.uint32)
    d0=arr&0xFF;d1=(arr>>8)&0xFF;d2=(arr>>16)&0xFF;d3=(arr>>24)&0xFF
    bits=(d0+d1*17+d2*289+d3*4913).astype(np.uint16)
    return torch.from_numpy(bits.view(np.float16).reshape(shape)).to(w_b17.device)
@triton.jit
def _conv2d_b17_kernel(x_ptr,w_b17_ptr,bias_ptr,y_ptr,N,Cin,H,W,Cout,Kh:tl.constexpr,Kw:tl.constexpr,Hout,Wout,stride_h:tl.constexpr,stride_w:tl.constexpr,pad_h:tl.constexpr,pad_w:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pid_m=tl.program_id(0);pid_n=tl.program_id(1)
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    npix=Hout*Wout
    n_idx=offs_m//npix
    pix=offs_m%npix
    hout=pix//Wout
    wout=pix%Wout
    mask_m=offs_m<(N*npix)
    mask_n=offs_n<Cout
    acc=tl.zeros([BLOCK_M,BLOCK_N],dtype=tl.float32)
    K=Cin*Kh*Kw
    for k_start in range(0,K,BLOCK_K):
        k_offs=k_start+tl.arange(0,BLOCK_K)
        mask_k=k_offs<K
        cin=k_offs//(Kh*Kw)
        khw=k_offs%(Kh*Kw)
        kh=khw//Kw
        kw=khw%Kw
        h_in=hout[:,None]*stride_h-pad_h+kh[None,:]
        w_in=wout[:,None]*stride_w-pad_w+kw[None,:]
        valid=(h_in>=0)&(h_in<H)&(w_in>=0)&(w_in<W)&mask_m[:,None]&mask_k[None,:]
        x_addr=((n_idx[:,None]*Cin+cin[None,:])*H+h_in)*W+w_in
        x_tile=tl.load(x_ptr+x_addr,mask=valid,other=0.0)
        w_addr=offs_n[None,:]*K+k_offs[:,None]
        w_mask=mask_n[None,:]&mask_k[:,None]
        packed=tl.load(w_b17_ptr+w_addr,mask=w_mask,other=0).to(tl.uint32)
        d0=packed&0xFF
        d1=(packed>>8)&0xFF
        d2=(packed>>16)&0xFF
        d3=(packed>>24)&0xFF
        bits=(d0+d1*17+d2*289+d3*4913).to(tl.uint16)
        w_tile=bits.to(tl.float16,bitcast=True)
        acc+=tl.dot(x_tile,w_tile,allow_tf32=False)
    if HAS_BIAS:
        b=tl.load(bias_ptr+offs_n,mask=mask_n,other=0.0)
        acc+=b[None,:]
    out_mask=mask_m[:,None]&mask_n[None,:]
    y_addr=((n_idx[:,None]*Cout+offs_n[None,:])*Hout+hout[:,None])*Wout+wout[:,None]
    tl.store(y_ptr+y_addr,acc.to(tl.float16),mask=out_mask)
def b17_conv2d(x,w_b17,bias,shape_meta,stride=1,padding=0):
    Cout,Cin,Kh,Kw=shape_meta
    N,Cin_x,H,W=x.shape
    assert Cin==Cin_x,f"channel mismatch {Cin} vs {Cin_x}"
    sh=stride if isinstance(stride,int) else stride[0]
    sw=stride if isinstance(stride,int) else stride[1]
    ph=padding if isinstance(padding,int) else padding[0]
    pw=padding if isinstance(padding,int) else padding[1]
    Hout=(H+2*ph-Kh)//sh+1
    Wout=(W+2*pw-Kw)//sw+1
    y=torch.empty(N,Cout,Hout,Wout,device=x.device,dtype=torch.float16)
    M=N*Hout*Wout
    BLOCK_M=64;BLOCK_N=64;BLOCK_K=32
    grid=(triton.cdiv(M,BLOCK_M),triton.cdiv(Cout,BLOCK_N))
    x_c=x.contiguous().to(torch.float16)
    bias_ptr=bias if bias is not None else x_c
    _conv2d_b17_kernel[grid](x_c,w_b17,bias_ptr,y,N,Cin,H,W,Cout,Kh,Kw,Hout,Wout,sh,sw,ph,pw,bias is not None,BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,BLOCK_K=BLOCK_K)
    return y
class GF17Conv2d(nn.Module):
    def __init__(self,conv:nn.Conv2d):
        super().__init__()
        assert conv.groups==1,"GF17Conv2d does not support groups>1 yet"
        assert conv.dilation==(1,1),"GF17Conv2d does not support dilation>1 yet"
        self.shape_meta=tuple(conv.weight.shape)
        self.stride=conv.stride
        self.padding=conv.padding
        w_fp16=conv.weight.detach().to(torch.float16).contiguous()
        self.register_buffer("w_b17",encode_conv_weight_to_b17(w_fp16),persistent=False)
        if conv.bias is not None:
            self.register_buffer("bias",conv.bias.detach().to(torch.float16).contiguous(),persistent=False)
        else:
            self.bias=None
    def forward(self,x):
        return b17_conv2d(x,self.w_b17,self.bias,self.shape_meta,stride=self.stride,padding=self.padding)
def patch_unet_to_b17(unet,verbose=False):
    swapped=0;skipped=0
    for name,mod in list(unet.named_modules()):
        for child_name,child in list(mod.named_children()):
            if isinstance(child,nn.Conv2d):
                if child.groups!=1 or child.dilation!=(1,1):
                    skipped+=1
                    if verbose:print(f"  [skip] {name}.{child_name} (groups={child.groups} dil={child.dilation})")
                    continue
                gf17=GF17Conv2d(child).to(child.weight.device)
                setattr(mod,child_name,gf17)
                swapped+=1
                if verbose and swapped<5:print(f"  [swap] {name}.{child_name} -> GF17Conv2d shape={child.weight.shape}")
    return swapped,skipped
