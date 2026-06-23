import torch,triton,triton.language as tl
@triton.jit
def _e2m1(c):
    s=tl.where(c>=8,-1.0,1.0);m=c&7
    v=tl.where(m==0,0.0,tl.where(m==1,0.5,tl.where(m==2,1.0,tl.where(m==3,1.5,tl.where(m==4,2.0,tl.where(m==5,3.0,tl.where(m==6,4.0,6.0)))))))
    return s*v
@triton.jit
def _nvfp4_gemv(c_ptr,s_ptr,x_ptr,y_ptr,ws2,OUT:tl.constexpr,INN:tl.constexpr,NG:tl.constexpr,HALF:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0);rows=pid*ROWS+tl.arange(0,ROWS);rm=rows<OUT
    k=tl.arange(0,8);acc=tl.zeros((ROWS,),dtype=tl.float32)
    for g in range(0,NG):
        bcol=8*g+k;bm=bcol<HALF
        b=tl.load(c_ptr+rows[:,None]*HALF+bcol[None,:],mask=rm[:,None]&bm[None,:],other=0).to(tl.int32)
        vlo=_e2m1(b&0xF);vhi=_e2m1((b>>4)&0xF)
        ie=16*g+2*k;io=ie+1
        xe=tl.load(x_ptr+ie,mask=ie<INN,other=0.0).to(tl.float32)
        xo=tl.load(x_ptr+io,mask=io<INN,other=0.0).to(tl.float32)
        gsum=tl.sum(vlo*xe[None,:]+vhi*xo[None,:],axis=1)
        sc=tl.load(s_ptr+rows*NG+g,mask=rm,other=0.0).to(tl.float32)*ws2
        acc+=gsum*sc
    tl.store(y_ptr+rows,acc.to(tl.float16),mask=rm)
def nvfp4_gemv(codes,scale,ws2,x,y=None,ROWS=8):
    out,half=codes.shape;ng=scale.shape[1];inn=half*2
    if y is None:y=torch.empty(out,device=x.device,dtype=torch.float16)
    _nvfp4_gemv[((out+ROWS-1)//ROWS,)](codes,scale,x,y,float(ws2),OUT=out,INN=inn,NG=ng,HALF=half,ROWS=ROWS)
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
