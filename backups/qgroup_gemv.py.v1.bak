import torch,triton,triton.language as tl
PL={2:(2,0),3:(2,1),4:(4,0),8:(8,0)}
@triton.jit
def _qg(p0,p1,a_ptr,c_ptr,x_ptr,y_ptr,M,OUT,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    acc=tl.zeros((BM,BN),dtype=tl.float32);kk=tl.arange(0,GS)
    for g in range(0,NG):
        k=g*GS+kk
        b0=tl.load(p0+rn[None,:]*(INN*B0//8)+(k[:,None]*B0//8),mask=nm[None,:],other=0).to(tl.int32)
        c=(b0>>((k[:,None]*B0)%8))&((1<<B0)-1)
        if B1>0:
            b1=tl.load(p1+rn[None,:]*(INN*B1//8)+(k[:,None]*B1//8),mask=nm[None,:],other=0).to(tl.int32)
            c=c|(((b1>>((k[:,None]*B1)%8))&((1<<B1)-1))<<B0)
        a=tl.load(a_ptr+rn*NG+g,mask=nm,other=0.0).to(tl.float32)
        o=tl.load(c_ptr+rn*NG+g,mask=nm,other=0.0).to(tl.float32)
        w=(c.to(tl.float32)*a[None,:]+o[None,:]).to(tl.float16)
        x=tl.load(x_ptr+rm[:,None]*INN+k[None,:],mask=mm[:,None],other=0.0).to(tl.float16)
        acc+=tl.dot(x,w)
    tl.store(y_ptr+rm[:,None]*OUT+rn[None,:],acc.to(tl.float16),mask=mm[:,None]&nm[None,:])
def qgemm(planes,a,c,x,bits,gs,y=None,BN=None,nw=None):
    p0=planes[0];p1=planes[1] if len(planes)>1 else planes[0];B0,B1=PL[bits];out,ng=a.shape;inn=ng*gs;M=x.shape[0]
    y=torch.empty(M,out,device=x.device,dtype=torch.float16) if y is None else y
    BM=16 if M<=16 else 32 if M<=64 else 64;BN=BN or (16 if M<=16 else 64);nw=nw or (1 if M<=16 else 4)
    _qg[((M+BM-1)//BM,(out+BN-1)//BN)](p0,p1,a,c,x,y,M,out,INN=inn,NG=ng,GS=gs,B0=B0,B1=B1,BM=BM,BN=BN,num_warps=nw)
    return y
def pack(codes,B):
    n=8//B;o,i=codes.shape;s=(torch.arange(n,device=codes.device,dtype=torch.int32)*B)
    return (codes.to(torch.int32).reshape(o,i//n,n)<<s).sum(-1).to(torch.uint8).contiguous()
def split(codes,bits):
    B0,B1=PL[bits];lo=codes&((1<<B0)-1);return [pack(lo,B0)]+([pack(codes>>B0,B1)] if B1 else [])
def unpack(planes,bits,inn):
    B0,B1=PL[bits];f=lambda p,B:((p.to(torch.int32).unsqueeze(-1)>>(torch.arange(8//B,device=p.device,dtype=torch.int32)*B))&((1<<B)-1)).reshape(p.shape[0],-1)[:,:inn]
    return f(planes[0],B0)|((f(planes[1],B1)<<B0) if B1 else 0)
def quantize(W,bits,gs,iters=3,grid=(1.0,0.95,0.9,0.85,0.8,0.75,0.7)):
    o,i=W.shape;Wg=W.float().reshape(o,i//gs,gs);qm=(1<<bits)-1;mn=Wg.amin(-1,keepdim=True);mx=Wg.amax(-1,keepdim=True);best=None
    for r in grid:
        lo,hi=mn*r,mx*r;s=((hi-lo)/qm).clamp_min(1e-8);q=torch.clamp(torch.round((Wg-lo)/s),0,qm);e=((q*s+lo-Wg)**2).sum(-1,keepdim=True)
        best=(e,q,s,lo) if best is None else tuple(torch.where(e<best[0],n,b) for n,b in zip((e,q,s,lo),best))
    _,q,s,lo=best
    for _ in range(iters):
        qc=q-q.mean(-1,keepdim=True);wc=Wg-Wg.mean(-1,keepdim=True);den=(qc*qc).sum(-1,keepdim=True)
        a=torch.where(den>0,(qc*wc).sum(-1,keepdim=True)/den.clamp_min(1e-12),s);c=Wg.mean(-1,keepdim=True)-a*q.mean(-1,keepdim=True)
        a16=a.half().float();c16=c.half().float();q=torch.clamp(torch.round((Wg-c16)/a16.clamp_min(1e-8)),0,qm)
    a16=a.half();c16=c.half()
    return q.reshape(o,i).to(torch.int32),a16.squeeze(-1).contiguous(),c16.squeeze(-1).contiguous()
def dequant(planes,a,c,bits,gs,inn):
    q=unpack(planes,bits,inn).float().reshape(a.shape[0],-1,gs);return (q*a.float().unsqueeze(-1)+c.float().unsqueeze(-1)).reshape(a.shape[0],inn)
