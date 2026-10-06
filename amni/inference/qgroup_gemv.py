import torch,triton,triton.language as tl
PL={2:(2,0),3:(2,1),4:(4,0),8:(8,0)}
@triton.jit
def _qg(p0,p1,a_ptr,c_ptr,x_ptr,y_ptr,M,OUT,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    NU:tl.constexpr=8 if B1>0 else 8//B0
    R0:tl.constexpr=INN*B0//8
    R1:tl.constexpr=INN*B1//8
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for t in range(0,INN//NU,BK):
        q=t+tl.arange(0,BK);qm=q<INN//NU;g=(q*NU)//GS;mk=qm[:,None]&nm[None,:]
        a=tl.load(a_ptr+rn[None,:]*NG+g[:,None],mask=mk,other=0.0).to(tl.float32)
        o=tl.load(c_ptr+rn[None,:]*NG+g[:,None],mask=mk,other=0.0).to(tl.float32)
        if B1>0:
            ba=tl.load(p0+rn[None,:]*R0+2*q[:,None],mask=mk,other=0).to(tl.int32)
            bb=tl.load(p0+rn[None,:]*R0+2*q[:,None]+1,mask=mk,other=0).to(tl.int32)
            bh=tl.load(p1+rn[None,:]*R1+q[:,None],mask=mk,other=0).to(tl.int32)
        else:
            b0=tl.load(p0+rn[None,:]*R0+q[:,None],mask=mk,other=0).to(tl.int32)
        for j in tl.static_range(NU):
            if B1>0:
                c=(((ba if j<4 else bb)>>(2*(j%4)))&3)|(((bh>>j)&1)<<2)
            else:
                c=(b0>>(j*B0))&((1<<B0)-1)
            w=(c.to(tl.float32)*a+o).to(tl.float16)
            xj=tl.load(x_ptr+rm[:,None]*INN+(q[None,:]*NU+j),mask=mm[:,None]&qm[None,:],other=0.0).to(tl.float16)
            acc+=tl.dot(xj,w)
    tl.store(y_ptr+rm[:,None]*OUT+rn[None,:],acc.to(tl.float16),mask=mm[:,None]&nm[None,:])
def qgemm(planes,a,c,x,bits,gs,y=None,BN=None,nw=None,BK=32):
    p0=planes[0];p1=planes[1] if len(planes)>1 else planes[0];B0,B1=PL[bits];out,ng=a.shape;inn=ng*gs;M=x.shape[0]
    y=torch.empty(M,out,device=x.device,dtype=torch.float16) if y is None else y
    BM=16 if M<=16 else 32 if M<=64 else 64;BN=BN or 64;nw=nw or (2 if M<=16 else 4)
    _qg[((M+BM-1)//BM,(out+BN-1)//BN)](p0,p1,a,c,x,y,M,out,INN=inn,NG=ng,GS=gs,B0=B0,B1=B1,BM=BM,BN=BN,BK=BK,num_warps=nw)
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
def gptq(W,H,bits,gs,damp=0.01,blk=128):
    W=W.float().clone();o,n=W.shape;qm=(1<<bits)-1;H=H.clone();dead=H.diagonal()==0;H[dead,dead]=1;W[:,dead]=0;H.diagonal().add_(damp*H.diagonal().mean());Hi=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
    Qc=torch.zeros(o,n,dtype=torch.int32,device=W.device);A=torch.zeros(o,n//gs,device=W.device);C=torch.zeros_like(A)
    for i1 in range(0,n,blk):
        i2=min(i1+blk,n);W1=W[:,i1:i2].clone();Er=torch.zeros_like(W1);H1=Hi[i1:i2,i1:i2]
        for j in range(i2-i1):
            c=i1+j;g=c//gs
            if c%gs==0:_,a,b=quantize(W1[:,j:j+gs],bits,gs,iters=2);A[:,g]=a.float().reshape(-1);C[:,g]=b.float().reshape(-1)
            w=W1[:,j];q=torch.clamp(torch.round((w-C[:,g])/A[:,g].clamp_min(1e-8)),0,qm);e=(w-(q*A[:,g]+C[:,g]))/H1[j,j];W1[:,j:]-=e[:,None]*H1[j,j:][None,:];Er[:,j]=e;Qc[:,c]=q.to(torch.int32)
        W[:,i2:]-=Er@Hi[i1:i2,i2:]
    return Qc,A.half().contiguous(),C.half().contiguous()
