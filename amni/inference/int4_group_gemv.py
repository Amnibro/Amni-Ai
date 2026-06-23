import torch,triton,triton.language as tl
@triton.jit
def _int4grp_gemv(c_ptr,s_ptr,x_ptr,y_ptr,OUT:tl.constexpr,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0);rows=pid*ROWS+tl.arange(0,ROWS);rm=rows<OUT
    goffs=tl.arange(0,GS);acc=tl.zeros((ROWS,),dtype=tl.float32)
    for g in range(0,NG):
        col=g*GS+goffs;cm=col<INN
        x=tl.load(x_ptr+col,mask=cm,other=0.0).to(tl.float32)
        c=tl.load(c_ptr+rows[:,None]*INN+col[None,:],mask=rm[:,None]&cm[None,:],other=8).to(tl.float32)
        gsum=tl.sum(x[None,:]*(c-8.0),axis=1)
        sc=tl.load(s_ptr+rows*NG+g,mask=rm,other=0.0).to(tl.float32)
        acc+=gsum*sc
    tl.store(y_ptr+rows,acc.to(tl.float16),mask=rm)
def int4grp_gemv(codes,scale,x,y=None,GS=128,ROWS=8):
    out,inn=codes.shape;ng=scale.shape[1]
    if y is None:y=torch.empty(out,device=x.device,dtype=torch.float16)
    _int4grp_gemv[((out+ROWS-1)//ROWS,)](codes,scale,x,y,OUT=out,INN=inn,NG=ng,GS=GS,ROWS=ROWS)
    return y
@triton.jit
def _int4grp_gemm(c_ptr,s_ptr,x_ptr,y_ptr,M,OUT:tl.constexpr,INN:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);mm=rm<M;nm=rn<OUT
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for g in range(0,NG):
        ck=g*GS+tl.arange(0,GS);km=ck<INN
        x=tl.load(x_ptr+rm[:,None]*INN+ck[None,:],mask=mm[:,None]&km[None,:],other=0.0).to(tl.float32)
        c=tl.load(c_ptr+ck[:,None]+rn[None,:]*INN,mask=km[:,None]&nm[None,:],other=8).to(tl.float32)
        sc=tl.load(s_ptr+rn*NG+g,mask=nm,other=0.0).to(tl.float32)
        acc+=tl.dot(x,(c-8.0)*sc[None,:])
    tl.store(y_ptr+rm[:,None]*OUT+rn[None,:],acc.to(tl.float16),mask=mm[:,None]&nm[None,:])
def int4grp_gemm(codes,scale,x,GS=128,BM=32,BN=64):
    out,inn=codes.shape;ng=scale.shape[1];M=x.shape[0]
    y=torch.empty(M,out,device=x.device,dtype=torch.float16)
    _int4grp_gemm[((M+BM-1)//BM,(out+BN-1)//BN)](codes,scale,x,y,M,OUT=out,INN=inn,NG=ng,GS=GS,BM=BM,BN=BN)
    return y
if __name__=='__main__':
    import torch.nn.functional as F
    torch.manual_seed(0);out,inn,GS=4096,2560,128
    W=torch.randn(out,inn).cuda();x=torch.randn(inn).cuda()
    Wg=W.reshape(out,-1,GS);sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7.0
    codes=(torch.clamp(torch.round(Wg/sc),-8,7).to(torch.uint8)+8).reshape(out,inn).contiguous()
    scale=sc.squeeze(-1).float().contiguous()
    Wdq=((codes.reshape(out,-1,GS).float()-8.0)*sc).reshape(out,inn)
    ref=F.linear(x.unsqueeze(0).to(torch.bfloat16),Wdq.to(torch.bfloat16)).squeeze(0).float()
    y=int4grp_gemv(codes,scale,x.half()).float()
    cos=F.cosine_similarity(ref,y,dim=0).item();rel=(ref-y).abs().mean().item()/ref.abs().mean().item()
    print(f'int4grp_gemv vs dequant-F.linear: cos={cos:.5f} rel_err={rel*100:.2f}%')
    print('VERDICT:','KERNEL CORRECT' if cos>0.999 else 'CHECK kernel')
    import time
    xh=x.half();yb=torch.empty(out,device='cuda',dtype=torch.float16);Wb=Wdq.to(torch.float16)
    for _ in range(10):int4grp_gemv(codes,scale,xh,y=yb)
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(200):int4grp_gemv(codes,scale,xh,y=yb)
    torch.cuda.synchronize();tk=(time.perf_counter()-t0)/200*1e6
    for _ in range(10):F.linear(xh.unsqueeze(0),Wb)
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(200):F.linear(xh.unsqueeze(0),Wb)
    torch.cuda.synchronize();tf=(time.perf_counter()-t0)/200*1e6
    floor=codes.numel()*0.5/1e12/1e-6
    print(f'SPEED us/call: int4grp_gemv={tk:.1f}  bf16 F.linear={tf:.1f}  4bit-mem-floor~{floor:.1f}  (kernel/floor={tk/floor:.1f}x)')
    M=32;xm=torch.randn(M,inn).cuda().half()
    ygm=int4grp_gemm(codes,scale,xm).float()
    refm=F.linear(xm,Wb).float()
    cosm=F.cosine_similarity(refm.flatten(),ygm.flatten(),dim=0).item()
    for _ in range(10):int4grp_gemm(codes,scale,xm)
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(200):int4grp_gemm(codes,scale,xm)
    torch.cuda.synchronize();tg=(time.perf_counter()-t0)/200*1e6
    print(f'GEMM (M={M}): cos={cosm:.4f}  gemm={tg:.1f}us  vs {M}x-gemv={tk*M:.1f}us  (gemm wins {tk*M/tg:.1f}x for prefill)')
    xh2=x.half().contiguous()
    for _ in range(5):int4grp_gemv(codes,scale,xh2,y=yb)
    torch.cuda.synchronize()
    try:
        st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):int4grp_gemv(codes,scale,xh2,y=yb)
        torch.cuda.current_stream().wait_stream(st)
        gg=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):int4grp_gemv(codes,scale,xh2,y=yb)
        gg.replay();torch.cuda.synchronize();print('KERNEL GRAPH-CAPTURE: OK (my kernel is graph-safe -> blocker is Gemma4 forward)')
    except Exception as e:print('KERNEL GRAPH-CAPTURE FAIL:',type(e).__name__,str(e)[:90])
