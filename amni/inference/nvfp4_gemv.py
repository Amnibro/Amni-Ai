import torch,triton,triton.language as tl
@triton.jit
def _e2m1(c):
    s=tl.where(c>=8,-1.0,1.0);m=c&7
    v=tl.where(m==0,0.0,tl.where(m==1,0.5,tl.where(m==2,1.0,tl.where(m==3,1.5,tl.where(m==4,2.0,tl.where(m==5,3.0,tl.where(m==6,4.0,6.0)))))))
    return s*v
@triton.jit
def _nvfp4_gemv(c_ptr,s_ptr,x_ptr,y_ptr,ws2,OUT:tl.constexpr,INN:tl.constexpr,NG:tl.constexpr,HALF:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0);rows=pid*ROWS+tl.arange(0,ROWS);rm=rows<OUT
    k=tl.arange(0,8);acc=tl.zeros((ROWS,8),dtype=tl.float32)
    for g in range(0,NG):
        bcol=8*g+k;bm=bcol<HALF
        b=tl.load(c_ptr+rows[:,None]*HALF+bcol[None,:],mask=rm[:,None]&bm[None,:],other=0).to(tl.int32)
        vlo=_e2m1(b&0xF);vhi=_e2m1((b>>4)&0xF)
        ie=16*g+2*k;io=ie+1
        xe=tl.load(x_ptr+ie,mask=ie<INN,other=0.0).to(tl.float32)
        xo=tl.load(x_ptr+io,mask=io<INN,other=0.0).to(tl.float32)
        sc=tl.load(s_ptr+rows*NG+g,mask=rm,other=0.0).to(tl.float32)*ws2
        acc+=sc[:,None]*(vlo*xe[None,:]+vhi*xo[None,:])
    y=tl.sum(acc,axis=1);tl.store(y_ptr+rows,y.to(tl.float16),mask=rm)
def nvfp4_gemv(codes,scale,ws2,x,y=None,ROWS=None):
    out,half=codes.shape;ng=scale.shape[1];inn=half*2
    if y is None:y=torch.empty(out,device=x.device,dtype=torch.float16)
    if ROWS is not None:_nvfp4_gemv[((out+ROWS-1)//ROWS,)](codes,scale,x,y,float(ws2),OUT=out,INN=inn,NG=ng,HALF=half,ROWS=ROWS)
    elif out>inn:_nvfp4_gemv[((out+7)//8,)](codes,scale,x,y,float(ws2),OUT=out,INN=inn,NG=ng,HALF=half,ROWS=8,num_warps=4)
    else:_nvfp4_gemv[((out+31)//32,)](codes,scale,x,y,float(ws2),OUT=out,INN=inn,NG=ng,HALF=half,ROWS=32)
    return y
@triton.jit
def _nvfp4_gemm(c_ptr,s_ptr,x_ptr,y_ptr,ws2,M,OUT:tl.constexpr,INN:tl.constexpr,NG:tl.constexpr,HALF:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    k=tl.arange(0,8);acc=tl.zeros((BM,BN),dtype=tl.float32)
    for g in range(0,NG):
        bcol=8*g+k;bmask=bcol<HALF
        b=tl.load(c_ptr+rn[None,:]*HALF+bcol[:,None],mask=nm[None,:]&bmask[:,None],other=0).to(tl.int32)
        vlo=_e2m1(b&0xF);vhi=_e2m1((b>>4)&0xF)
        ie=16*g+2*k;io=ie+1
        xe=tl.load(x_ptr+rm[:,None]*INN+ie[None,:],mask=mm[:,None]&(ie[None,:]<INN),other=0.0)
        xo=tl.load(x_ptr+rm[:,None]*INN+io[None,:],mask=mm[:,None]&(io[None,:]<INN),other=0.0)
        part=tl.dot(xe.to(tl.float32),vlo)+tl.dot(xo.to(tl.float32),vhi)
        sc=tl.load(s_ptr+rn*NG+g,mask=nm,other=0.0).to(tl.float32)*ws2
        acc+=part*sc[None,:]
    tl.store(y_ptr+rm[:,None]*OUT+rn[None,:],acc.to(tl.float16),mask=mm[:,None]&nm[None,:])
def nvfp4_gemm(codes,scale,ws2,x,BM=16,BN=64):
    out,half=codes.shape;ng=scale.shape[1];inn=half*2;M=x.shape[0]
    y=torch.empty(M,out,device=x.device,dtype=torch.float16)
    _nvfp4_gemm[((M+BM-1)//BM,(out+BN-1)//BN)](codes,scale,x,y,float(ws2),M,OUT=out,INN=inn,NG=ng,HALF=half,BM=BM,BN=BN)
    return y
@triton.jit
def _nvfp4_dq(c_ptr,s_ptr,w_ptr,ws2,OUT,HALF:tl.constexpr,NG:tl.constexpr,INN:tl.constexpr,BR:tl.constexpr,BH:tl.constexpr):
    pr=tl.program_id(0);ph=tl.program_id(1)
    rows=pr*BR+tl.arange(0,BR);cols=ph*BH+tl.arange(0,BH);rm=rows<OUT;cm=cols<HALF
    b=tl.load(c_ptr+rows[:,None]*HALF+cols[None,:],mask=rm[:,None]&cm[None,:],other=0).to(tl.int32)
    vlo=_e2m1(b&0xF);vhi=_e2m1((b>>4)&0xF);g=cols//8
    sc=tl.load(s_ptr+rows[:,None]*NG+g[None,:],mask=rm[:,None]&cm[None,:],other=0.0).to(tl.float32)*ws2
    tl.store(w_ptr+rows[:,None]*INN+2*cols[None,:],(vlo*sc).to(tl.bfloat16),mask=rm[:,None]&cm[None,:])
    tl.store(w_ptr+rows[:,None]*INN+(2*cols[None,:]+1),(vhi*sc).to(tl.bfloat16),mask=rm[:,None]&cm[None,:])
_PFS={}
def nvfp4_prefill(codes,scale,ws2,x,BR=16,BH=128):
    out,half=codes.shape;ng=scale.shape[1];inn=half*2;need=out*inn;dev=codes.device
    buf=_PFS.get(dev)
    if buf is None or buf.numel()<need:buf=torch.empty(need,device=dev,dtype=torch.bfloat16);_PFS[dev]=buf
    W=buf[:need].view(out,inn)
    _nvfp4_dq[((out+BR-1)//BR,(half+BH-1)//BH)](codes,scale,W,float(ws2),out,HALF=half,NG=ng,INN=inn,BR=BR,BH=BH)
    return torch.matmul(x,W.t())
@triton.jit
def _nvfp4_fused(c,s,x,y,ws2,M,OUT:tl.constexpr,K:tl.constexpr,NG:tl.constexpr,HALF:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    acc=tl.zeros((BM,BN),tl.float32)
    for g0 in range(0,NG,BG):
        kb=8*g0+tl.arange(0,8*BG);kbm=kb<HALF
        b=tl.load(c+rn[None,:]*HALF+kb[:,None],mask=nm[None,:]&kbm[:,None],other=0).to(tl.int32)
        gg=kb//8;sc=tl.load(s+rn[None,:]*NG+gg[:,None],mask=nm[None,:]&kbm[:,None],other=0.0)*ws2
        vlo=(_e2m1(b&0xF)*sc).to(tl.bfloat16);vhi=(_e2m1((b>>4)&0xF)*sc).to(tl.bfloat16)
        ke=2*kb;ko=2*kb+1
        xe=tl.load(x+rm[:,None]*K+ke[None,:],mask=mm[:,None]&(ke[None,:]<K),other=0.0)
        xo=tl.load(x+rm[:,None]*K+ko[None,:],mask=mm[:,None]&(ko[None,:]<K),other=0.0)
        acc+=tl.dot(xe,vlo)+tl.dot(xo,vhi)
    tl.store(y+rm[:,None]*OUT+rn[None,:],acc.to(tl.bfloat16),mask=mm[:,None]&nm[None,:])
def nvfp4_fused(codes,scale,ws2,x,BM=64,BN=64,BG=4):
    OUT,HALF=codes.shape;K=HALF*2;NG=scale.shape[1];M=x.shape[0]
    y=torch.empty(M,OUT,device=x.device,dtype=torch.bfloat16)
    _nvfp4_fused[((M+BM-1)//BM,(OUT+BN-1)//BN)](codes,scale,x,y,float(ws2),M,OUT=OUT,K=K,NG=NG,HALF=HALF,BM=BM,BN=BN,BG=BG)
    return y
_DQE={}
def nvfp4_dequant(codes,scale,ws2):
    dev=codes.device
    if dev not in _DQE:_DQE[dev]=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],device=dev,dtype=torch.float16)
    E=_DQE[dev];lo=codes&0xF;hi=codes>>4
    one=torch.tensor(1.,device=dev,dtype=torch.float16);neg=torch.tensor(-1.,device=dev,dtype=torch.float16)
    vlo=E[(lo&7).long()]*torch.where(lo>=8,neg,one);vhi=E[(hi&7).long()]*torch.where(hi>=8,neg,one)
    nib=torch.stack([vlo,vhi],-1).reshape(codes.shape[0],-1)
    return nib*(scale.to(torch.float16)*float(ws2)).repeat_interleave(16,1)
def _ref_decode(codes,scale,ws2):
    out,half=codes.shape;lo=(codes&0xF).int();hi=((codes>>4)&0xF).int()
    E=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],device=codes.device)
    def d(n):return E[n&7]*torch.where(n>=8,-1.,1.)
    nib=torch.stack([d(lo),d(hi)],-1).reshape(out,half*2)
    bs=(scale.float()*float(ws2)).repeat_interleave(16,1);return nib*bs
if __name__=='__main__':
    import torch.nn.functional as F
    torch.manual_seed(0);out,inn,NG=4096,2560,2560//16
    codes=torch.randint(0,256,(out,inn//2),dtype=torch.uint8).cuda()
    scale=(torch.rand(out,NG)*0.5+0.1).to(torch.float16).cuda();ws2=0.0007
    W=_ref_decode(codes,scale,ws2).float()
    x=torch.randn(inn).cuda()
    ref=F.linear(x.unsqueeze(0).to(torch.float16),W.to(torch.float16)).squeeze(0).float()
    y=nvfp4_gemv(codes,scale,ws2,x.half()).float()
    cos=F.cosine_similarity(ref,y,0).item();rel=(ref-y).abs().mean().item()/ref.abs().mean().clamp_min(1e-9).item()
    print(f'GEMV vs ref: cos={cos:.5f} rel_err={rel*100:.3f}%','OK' if cos>0.999 else 'FAIL',flush=True)
    M=16;xm=torch.randn(M,inn).cuda().half()
    refm=F.linear(xm,W.to(torch.float16)).float();ym=nvfp4_gemm(codes,scale,ws2,xm).float()
    cosm=F.cosine_similarity(refm.flatten(),ym.flatten(),0).item()
    print(f'GEMM(M={M}) vs ref: cos={cosm:.5f}','OK' if cosm>0.999 else 'FAIL',flush=True)
