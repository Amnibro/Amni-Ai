import torch,torch.nn as nn,torch.nn.functional as F,triton,triton.language as tl,numpy as np
from amni.compute.b17_conv import encode_conv_weight_to_b17,decode_b17_to_weight
_BT=torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],dtype=torch.float32)
_G=torch.tensor([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]],dtype=torch.float32)
_AT=torch.tensor([[1,1,1,0],[0,1,-1,-1]],dtype=torch.float32)
def winograd_transform_weight(w):
    assert w.dtype==torch.float16 and w.ndim==4 and w.shape[2]==3 and w.shape[3]==3
    Cout,Cin=w.shape[0],w.shape[1]
    g=w.float().reshape(Cout*Cin,3,3)
    G=_G.to(w.device)
    V=(G@g@G.transpose(0,1)).reshape(Cout,Cin,4,4).permute(2,3,0,1).contiguous()
    return V.half()
def winograd_transform_weight_b17(w):
    V=winograd_transform_weight(w)
    Cout,Cin=w.shape[0],w.shape[1]
    Vp=V.reshape(16,Cout,Cin).permute(0,2,1).contiguous()
    V_b17=torch.stack([encode_conv_weight_to_b17(Vp[i].reshape(Cin,Cout,1,1)).reshape(Cin,Cout) for i in range(16)],dim=0).contiguous()
    return V_b17
@triton.jit
def _winograd_input_kernel(x_ptr,u_ptr,N,Cin,H,W,Th,Tw,BLOCK_C:tl.constexpr,BLOCK_T:tl.constexpr):
    pid_t=tl.program_id(0)
    pid_c=tl.program_id(1)
    pid_n=tl.program_id(2)
    offs_t=pid_t*BLOCK_T+tl.arange(0,BLOCK_T)
    offs_c=pid_c*BLOCK_C+tl.arange(0,BLOCK_C)
    mask_t=offs_t<(Th*Tw)
    mask_c=offs_c<Cin
    th=offs_t//Tw
    tw=offs_t%Tw
    n=pid_n
    h0=th*2-1
    w0=tw*2-1
    h00=h0+0;h01=h0+1;h02=h0+2;h03=h0+3
    w00=w0+0;w01=w0+1;w02=w0+2;w03=w0+3
    v00=(h00>=0)&(h00<H);v01=v00;v02=v00;v03=v00
    v10=(h01>=0)&(h01<H);v11=v10;v12=v10;v13=v10
    v20=(h02>=0)&(h02<H);v21=v20;v22=v20;v23=v20
    v30=(h03>=0)&(h03<H);v31=v30;v32=v30;v33=v30
    g0=(w00>=0)&(w00<W);g1=(w01>=0)&(w01<W);g2=(w02>=0)&(w02<W);g3=(w03>=0)&(w03<W)
    base=(n*Cin+offs_c[None,:])*H
    sc=mask_c[None,:]&mask_t[:,None]
    d00=tl.load(x_ptr+(base+h00[:,None])*W+w00[:,None],mask=(v00&g0)[:,None]&sc,other=0.0).to(tl.float32)
    d01=tl.load(x_ptr+(base+h00[:,None])*W+w01[:,None],mask=(v01&g1)[:,None]&sc,other=0.0).to(tl.float32)
    d02=tl.load(x_ptr+(base+h00[:,None])*W+w02[:,None],mask=(v02&g2)[:,None]&sc,other=0.0).to(tl.float32)
    d03=tl.load(x_ptr+(base+h00[:,None])*W+w03[:,None],mask=(v03&g3)[:,None]&sc,other=0.0).to(tl.float32)
    d10=tl.load(x_ptr+(base+h01[:,None])*W+w00[:,None],mask=(v10&g0)[:,None]&sc,other=0.0).to(tl.float32)
    d11=tl.load(x_ptr+(base+h01[:,None])*W+w01[:,None],mask=(v11&g1)[:,None]&sc,other=0.0).to(tl.float32)
    d12=tl.load(x_ptr+(base+h01[:,None])*W+w02[:,None],mask=(v12&g2)[:,None]&sc,other=0.0).to(tl.float32)
    d13=tl.load(x_ptr+(base+h01[:,None])*W+w03[:,None],mask=(v13&g3)[:,None]&sc,other=0.0).to(tl.float32)
    d20=tl.load(x_ptr+(base+h02[:,None])*W+w00[:,None],mask=(v20&g0)[:,None]&sc,other=0.0).to(tl.float32)
    d21=tl.load(x_ptr+(base+h02[:,None])*W+w01[:,None],mask=(v21&g1)[:,None]&sc,other=0.0).to(tl.float32)
    d22=tl.load(x_ptr+(base+h02[:,None])*W+w02[:,None],mask=(v22&g2)[:,None]&sc,other=0.0).to(tl.float32)
    d23=tl.load(x_ptr+(base+h02[:,None])*W+w03[:,None],mask=(v23&g3)[:,None]&sc,other=0.0).to(tl.float32)
    d30=tl.load(x_ptr+(base+h03[:,None])*W+w00[:,None],mask=(v30&g0)[:,None]&sc,other=0.0).to(tl.float32)
    d31=tl.load(x_ptr+(base+h03[:,None])*W+w01[:,None],mask=(v31&g1)[:,None]&sc,other=0.0).to(tl.float32)
    d32=tl.load(x_ptr+(base+h03[:,None])*W+w02[:,None],mask=(v32&g2)[:,None]&sc,other=0.0).to(tl.float32)
    d33=tl.load(x_ptr+(base+h03[:,None])*W+w03[:,None],mask=(v33&g3)[:,None]&sc,other=0.0).to(tl.float32)
    u00=d00-d02-d20+d22
    u01=d01+d02-d21-d22
    u02=-d01+d02+d21-d22
    u03=d01-d03-d21+d23
    u10=d10-d12+d20-d22
    u11=d11+d12+d21+d22
    u12=-d11+d12-d21+d22
    u13=d11-d13+d21-d23
    u20=-d10+d12+d20-d22
    u21=-d11-d12+d21+d22
    u22=d11-d12-d21+d22
    u23=-d11+d13+d21-d23
    u30=d10-d12-d30+d32
    u31=d11+d12-d31-d32
    u32=-d11+d12+d31-d32
    u33=d11-d13-d31+d33
    NT=N*Th*Tw
    obase=((n*(Th*Tw)+offs_t[:,None])*Cin+offs_c[None,:])
    om=mask_t[:,None]&mask_c[None,:]
    tl.store(u_ptr+0*NT*Cin+obase,u00.to(tl.float16),mask=om)
    tl.store(u_ptr+1*NT*Cin+obase,u01.to(tl.float16),mask=om)
    tl.store(u_ptr+2*NT*Cin+obase,u02.to(tl.float16),mask=om)
    tl.store(u_ptr+3*NT*Cin+obase,u03.to(tl.float16),mask=om)
    tl.store(u_ptr+4*NT*Cin+obase,u10.to(tl.float16),mask=om)
    tl.store(u_ptr+5*NT*Cin+obase,u11.to(tl.float16),mask=om)
    tl.store(u_ptr+6*NT*Cin+obase,u12.to(tl.float16),mask=om)
    tl.store(u_ptr+7*NT*Cin+obase,u13.to(tl.float16),mask=om)
    tl.store(u_ptr+8*NT*Cin+obase,u20.to(tl.float16),mask=om)
    tl.store(u_ptr+9*NT*Cin+obase,u21.to(tl.float16),mask=om)
    tl.store(u_ptr+10*NT*Cin+obase,u22.to(tl.float16),mask=om)
    tl.store(u_ptr+11*NT*Cin+obase,u23.to(tl.float16),mask=om)
    tl.store(u_ptr+12*NT*Cin+obase,u30.to(tl.float16),mask=om)
    tl.store(u_ptr+13*NT*Cin+obase,u31.to(tl.float16),mask=om)
    tl.store(u_ptr+14*NT*Cin+obase,u32.to(tl.float16),mask=om)
    tl.store(u_ptr+15*NT*Cin+obase,u33.to(tl.float16),mask=om)
@triton.jit
def _winograd_gemm_kernel(u_ptr,v_b17_ptr,m_ptr,NT,Cin,Cout,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pid_p=tl.program_id(0)
    pid_m=tl.program_id(1)
    pid_n=tl.program_id(2)
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    mask_m=offs_m<NT
    mask_n=offs_n<Cout
    acc=tl.zeros([BLOCK_M,BLOCK_N],dtype=tl.float32)
    u_base=pid_p*NT*Cin
    v_base=pid_p*Cin*Cout
    for k_start in range(0,Cin,BLOCK_K):
        k_offs=k_start+tl.arange(0,BLOCK_K)
        mask_k=k_offs<Cin
        u_addr=u_base+offs_m[:,None]*Cin+k_offs[None,:]
        u_tile=tl.load(u_ptr+u_addr,mask=mask_m[:,None]&mask_k[None,:],other=0.0)
        v_addr=v_base+k_offs[:,None]*Cout+offs_n[None,:]
        v_packed=tl.load(v_b17_ptr+v_addr,mask=mask_k[:,None]&mask_n[None,:],other=0).to(tl.uint32)
        d0=v_packed&0xFF
        d1=(v_packed>>8)&0xFF
        d2=(v_packed>>16)&0xFF
        d3=(v_packed>>24)&0xFF
        v_bits=(d0+d1*17+d2*289+d3*4913).to(tl.uint16)
        v_tile=v_bits.to(tl.float16,bitcast=True)
        acc+=tl.dot(u_tile,v_tile,allow_tf32=False)
    m_base=pid_p*NT*Cout
    out_addr=m_base+offs_m[:,None]*Cout+offs_n[None,:]
    om=mask_m[:,None]&mask_n[None,:]
    tl.store(m_ptr+out_addr,acc.to(tl.float16),mask=om)
@triton.jit
def _winograd_output_kernel(m_ptr,bias_ptr,y_ptr,N,Cout,Th,Tw,HAS_BIAS:tl.constexpr,BLOCK_F:tl.constexpr,BLOCK_T:tl.constexpr):
    pid_t=tl.program_id(0)
    pid_f=tl.program_id(1)
    pid_n=tl.program_id(2)
    offs_t=pid_t*BLOCK_T+tl.arange(0,BLOCK_T)
    offs_f=pid_f*BLOCK_F+tl.arange(0,BLOCK_F)
    mask_t=offs_t<(Th*Tw)
    mask_f=offs_f<Cout
    th=offs_t//Tw
    tw=offs_t%Tw
    n=pid_n
    NT=N*Th*Tw
    base=((n*(Th*Tw)+offs_t[:,None])*Cout+offs_f[None,:])
    lm=mask_t[:,None]&mask_f[None,:]
    m00=tl.load(m_ptr+0*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m01=tl.load(m_ptr+1*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m02=tl.load(m_ptr+2*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m03=tl.load(m_ptr+3*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m10=tl.load(m_ptr+4*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m11=tl.load(m_ptr+5*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m12=tl.load(m_ptr+6*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m13=tl.load(m_ptr+7*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m20=tl.load(m_ptr+8*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m21=tl.load(m_ptr+9*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m22=tl.load(m_ptr+10*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m23=tl.load(m_ptr+11*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m30=tl.load(m_ptr+12*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m31=tl.load(m_ptr+13*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m32=tl.load(m_ptr+14*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    m33=tl.load(m_ptr+15*NT*Cout+base,mask=lm,other=0.0).to(tl.float32)
    y00=m00+m01+m02+m10+m11+m12+m20+m21+m22
    y01=m01-m02-m03+m11-m12-m13+m21-m22-m23
    y10=m10+m11+m12-m20-m21-m22-m30-m31-m32
    y11=m11-m12-m13-m21+m22+m23-m31+m32+m33
    if HAS_BIAS:
        b=tl.load(bias_ptr+offs_f,mask=mask_f,other=0.0).to(tl.float32)
        y00=y00+b[None,:]
        y01=y01+b[None,:]
        y10=y10+b[None,:]
        y11=y11+b[None,:]
    H=Th*2
    W=Tw*2
    h0=th*2
    w0=tw*2
    sm=mask_t[:,None]&mask_f[None,:]
    tl.store(y_ptr+((n*Cout+offs_f[None,:])*H+(h0[:,None]+0))*W+(w0[:,None]+0),y00.to(tl.float16),mask=sm)
    tl.store(y_ptr+((n*Cout+offs_f[None,:])*H+(h0[:,None]+0))*W+(w0[:,None]+1),y01.to(tl.float16),mask=sm)
    tl.store(y_ptr+((n*Cout+offs_f[None,:])*H+(h0[:,None]+1))*W+(w0[:,None]+0),y10.to(tl.float16),mask=sm)
    tl.store(y_ptr+((n*Cout+offs_f[None,:])*H+(h0[:,None]+1))*W+(w0[:,None]+1),y11.to(tl.float16),mask=sm)
def winograd_b17_conv2d(x,v_b17,bias,Cout,Cin):
    assert x.dtype==torch.float16 and x.ndim==4
    N,_,H,W=x.shape
    assert H%2==0 and W%2==0,f"H,W must be even (got {H}x{W})"
    Th,Tw=H//2,W//2
    NT=N*Th*Tw
    x=x.contiguous()
    u=torch.empty(16,NT,Cin,device=x.device,dtype=torch.float16)
    BLOCK_T_IN=32;BLOCK_C_IN=64
    grid_in=(triton.cdiv(Th*Tw,BLOCK_T_IN),triton.cdiv(Cin,BLOCK_C_IN),N)
    _winograd_input_kernel[grid_in](x,u,N,Cin,H,W,Th,Tw,BLOCK_C_IN,BLOCK_T_IN)
    m=torch.empty(16,NT,Cout,device=x.device,dtype=torch.float16)
    BLOCK_M=128;BLOCK_N=128;BLOCK_K=32
    grid_g=(16,triton.cdiv(NT,BLOCK_M),triton.cdiv(Cout,BLOCK_N))
    _winograd_gemm_kernel[grid_g](u,v_b17,m,NT,Cin,Cout,BLOCK_M,BLOCK_N,BLOCK_K)
    y=torch.empty(N,Cout,H,W,device=x.device,dtype=torch.float16)
    BLOCK_T_OUT=32;BLOCK_F_OUT=64
    grid_out=(triton.cdiv(Th*Tw,BLOCK_T_OUT),triton.cdiv(Cout,BLOCK_F_OUT),N)
    _winograd_output_kernel[grid_out](m,bias if bias is not None else x,y,N,Cout,Th,Tw,bias is not None,BLOCK_F_OUT,BLOCK_T_OUT)
    return y
class GF17WinogradConv2d(nn.Module):
    def __init__(self,conv:nn.Conv2d):
        super().__init__()
        assert conv.groups==1 and conv.dilation==(1,1) and conv.kernel_size==(3,3) and conv.stride==(1,1) and conv.padding==(1,1)
        self.Cout,self.Cin=conv.weight.shape[0],conv.weight.shape[1]
        w_fp16=conv.weight.detach().to(torch.float16).contiguous()
        self.register_buffer("v_b17",winograd_transform_weight_b17(w_fp16),persistent=False)
        if conv.bias is not None:
            self.register_buffer("bias",conv.bias.detach().to(torch.float16).contiguous(),persistent=False)
        else:
            self.bias=None
    def forward(self,x):
        return winograd_b17_conv2d(x,self.v_b17,self.bias,self.Cout,self.Cin)
