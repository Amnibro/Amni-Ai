import torch,triton,triton.language as tl
@triton.jit
def _i8_gemv(c_ptr,s_ptr,x_ptr,y_ptr,OUT,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0);rows=pid*ROWS+tl.arange(0,ROWS);rm=rows<OUT
    k=tl.arange(0,GS);acc=tl.zeros((ROWS,GS),dtype=tl.float32)
    for g in range(0,NG):
        col=g*GS+k;cm=col<INN
        c=tl.load(c_ptr+rows[:,None]*INN+col[None,:],mask=rm[:,None]&cm[None,:],other=0).to(tl.float32)
        x=tl.load(x_ptr+col,mask=cm,other=0.0).to(tl.float32)
        sc=tl.load(s_ptr+rows*NG+g,mask=rm,other=0.0).to(tl.float32)
        acc+=sc[:,None]*(c*x[None,:])
    y=tl.sum(acc,axis=1);tl.store(y_ptr+rows,y.to(tl.float16),mask=rm)
def int8grp_gemv(codes,scale,x,y=None,GS=128):
    out,inn=codes.shape;ng=scale.shape[1]
    if y is None:y=torch.empty(out,device=x.device,dtype=torch.float16)
    if out>inn:_i8_gemv[((out+7)//8,)](codes,scale,x,y,out,INN=inn,NG=ng,GS=GS,ROWS=8,num_warps=4,num_stages=2)
    else:_i8_gemv[((out+31)//32,)](codes,scale,x,y,out,INN=inn,NG=ng,GS=GS,ROWS=32,num_warps=4,num_stages=2)
    return y
@triton.jit
def _i8_gemm(c_ptr,s_ptr,x_ptr,y_ptr,M,OUT,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for g in range(0,NG):
        ck=g*GS+tl.arange(0,GS);km=ck<INN
        x=tl.load(x_ptr+rm[:,None]*INN+ck[None,:],mask=mm[:,None]&km[None,:],other=0.0).to(tl.float32)
        c=tl.load(c_ptr+ck[:,None]+rn[None,:]*INN,mask=km[:,None]&nm[None,:],other=0).to(tl.float32)
        sc=tl.load(s_ptr+rn*NG+g,mask=nm,other=0.0).to(tl.float32)
        acc+=tl.dot(x,c*sc[None,:])
    tl.store(y_ptr+rm[:,None]*OUT+rn[None,:],acc.to(tl.float16),mask=mm[:,None]&nm[None,:])
def int8grp_gemm(codes,scale,x,GS=128,BM=32,BN=64):
    out,inn=codes.shape;ng=scale.shape[1];M=x.shape[0]
    y=torch.empty(M,out,device=x.device,dtype=torch.float16)
    _i8_gemm[((M+BM-1)//BM,(out+BN-1)//BN)](codes,scale,x,y,M,out,INN=inn,NG=ng,GS=GS,BM=BM,BN=BN);return y
def quant_int8(W,GS=128):
    out,inn=W.shape;Wg=W.float().reshape(out,-1,GS);sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
    codes=torch.clamp(torch.round(Wg/sc),-128,127).to(torch.int8).reshape(out,inn).contiguous()
    return codes,sc.squeeze(-1).float().contiguous()
if __name__=='__main__':
    import torch.nn.functional as F
    torch.manual_seed(0);out,inn,GS=4096,3840,128
    W=torch.randn(out,inn).cuda();codes,scale=quant_int8(W,GS)
    Wdq=(codes.reshape(out,-1,GS).float()*scale.unsqueeze(-1)).reshape(out,inn)
    x=torch.randn(inn).cuda()
    refdq=F.linear(x.unsqueeze(0).to(torch.float16),Wdq.to(torch.float16)).squeeze(0).float()
    refbf=F.linear(x.unsqueeze(0).to(torch.bfloat16),W.to(torch.bfloat16)).squeeze(0).float()
    y=int8grp_gemv(codes,scale,x.half()).float()
    print('GEMV vs int8-dequant: cos=%.6f (kernel correct)'%F.cosine_similarity(refdq,y,0).item())
    print('int8 vs bf16 (quant fidelity): cos=%.6f rel=%.3f%%'%(F.cosine_similarity(refbf,y,0).item(),(refbf-y).abs().mean().item()/refbf.abs().mean().item()*100))
    M=8;xm=torch.randn(M,inn).cuda().half();refm=F.linear(xm,Wdq.to(torch.float16)).float();ym=int8grp_gemm(codes,scale,xm).float()
    print('GEMM vs int8-dequant: cos=%.6f'%F.cosine_similarity(refm.flatten(),ym.flatten(),0).item())
    print('INT8_KERNEL_OK')
