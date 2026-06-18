import torch,triton,triton.language as tl
@triton.jit
def _gf17_gemv(idx_ptr,s_ptr,x_ptr,y_ptr,OUT:tl.constexpr,INN:tl.constexpr,BLOCK:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0)
    rows=pid*ROWS+tl.arange(0,ROWS)
    rm=rows<OUT
    offs=tl.arange(0,BLOCK)
    acc=tl.zeros((ROWS,BLOCK),dtype=tl.float32)
    for off in range(0,INN,BLOCK):
        m=off+offs<INN
        x=tl.load(x_ptr+off+offs,mask=m,other=0.0).to(tl.float32)
        idx=tl.load(idx_ptr+rows[:,None]*INN+off+offs[None,:],mask=rm[:,None]&m[None,:],other=8).to(tl.float32)
        acc+=x[None,:]*(idx-8.0)
    s=tl.load(s_ptr+rows,mask=rm,other=0.0).to(tl.float32)
    tl.store(y_ptr+rows,(tl.sum(acc,axis=1)*s).to(tl.float16),mask=rm)
@triton.jit
def _gf17_gemv_p3(pk_ptr,s_ptr,x_ptr,y_ptr,NP:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0)
    offs=tl.arange(0,BLOCK)
    acc=tl.zeros((BLOCK,),dtype=tl.float32)
    for off in range(0,NP,BLOCK):
        m=off+offs<NP
        w=tl.load(pk_ptr+r*NP+off+offs,mask=m,other=2602).to(tl.int32)
        d0=w%17;d1=(w//17)%17;d2=(w//289)%17
        x0=tl.load(x_ptr+(off+offs)*3,mask=m,other=0.0).to(tl.float32)
        x1=tl.load(x_ptr+(off+offs)*3+1,mask=m,other=0.0).to(tl.float32)
        x2=tl.load(x_ptr+(off+offs)*3+2,mask=m,other=0.0).to(tl.float32)
        acc+=x0*(d0.to(tl.float32)-8.0)+x1*(d1.to(tl.float32)-8.0)+x2*(d2.to(tl.float32)-8.0)
    s=tl.load(s_ptr+r)
    tl.store(y_ptr+r,(tl.sum(acc,axis=0)*s).to(tl.float16))
def gf17_quantize_rows(W,h=8.0,clip=0.62):
    s=W.abs().amax(1,keepdim=True)*clip/h
    idx=(torch.clamp(torch.round(W/s),-h,h)+h).to(torch.uint8)
    return idx,s.squeeze(1).contiguous()
def gf17_recover_rows(W,h=8.0):
    A=W.abs();nz=torch.where(A>0,A,torch.full_like(A,float('inf')));s=nz.amin(1,keepdim=True)
    s=torch.where(torch.isinf(s),torch.ones_like(s),s)
    idx=(torch.clamp(torch.round(W/s),-h,h)+h).to(torch.uint8)
    return idx,s.squeeze(1).contiguous()
def gf17_pack3(idx):
    out,inn=idx.shape;pad=(-inn)%3
    a=torch.nn.functional.pad(idx,(0,pad)).view(out,-1,3).to(torch.int32)
    return (a[:,:,0]+17*a[:,:,1]+289*a[:,:,2]).to(torch.int16).contiguous(),inn+pad
def gf17_gemv(idx,s,x,BLOCK=None,ROWS=None):
    out,inn=idx.shape
    ROWS=(1 if inn>4096 else 8 if out>4096 else 2) if ROWS is None else ROWS
    BLOCK=(512 if inn>4096 or out>4096 else 1024) if BLOCK is None else BLOCK
    y=torch.empty(out,device=x.device,dtype=torch.float16)
    _gf17_gemv[((out+ROWS-1)//ROWS,)](idx,s,x,y,OUT=out,INN=inn,BLOCK=BLOCK,ROWS=ROWS)
    return y
def gf17_gemv_packed(pk,s,x,inn3,BLOCK=512):
    out,np_=pk.shape;y=torch.empty(out,device=x.device,dtype=torch.float16)
    xp=torch.nn.functional.pad(x,(0,inn3-x.shape[0])) if x.shape[0]<inn3 else x
    _gf17_gemv_p3[(out,)](pk,s,xp,y,NP=np_,BLOCK=BLOCK)
    return y
@triton.jit
def _gf17_gemv_sel(idx_ptr,s_ptr,x_ptr,y_ptr,sel_ptr,NSEL,INN:tl.constexpr,BLOCK:tl.constexpr,ROWS:tl.constexpr):
    pid=tl.program_id(0);rows=pid*ROWS+tl.arange(0,ROWS);rm=rows<NSEL
    rr=tl.load(sel_ptr+rows,mask=rm,other=0)
    offs=tl.arange(0,BLOCK);acc=tl.zeros((ROWS,BLOCK),dtype=tl.float32)
    for off in range(0,INN,BLOCK):
        m=off+offs<INN
        x=tl.load(x_ptr+off+offs,mask=m,other=0.0).to(tl.float32)
        idx=tl.load(idx_ptr+rr[:,None]*INN+off+offs[None,:],mask=rm[:,None]&m[None,:],other=8).to(tl.float32)
        acc+=x[None,:]*(idx-8.0)
    s=tl.load(s_ptr+rr,mask=rm,other=0.0).to(tl.float32)
    tl.store(y_ptr+rr,(tl.sum(acc,axis=1)*s).to(tl.float16),mask=rm)
def gf17_gemv_sel(idx,s,x,sel,y=None,BLOCK=None,ROWS=8):
    out,inn=idx.shape
    y=torch.zeros(out,device=x.device,dtype=torch.float16) if y is None else y
    B=BLOCK or (512 if inn>4096 else 1024)
    n=sel.numel()
    if n:_gf17_gemv_sel[(triton.cdiv(n,ROWS),)](idx,s,x,y,sel,n,INN=inn,BLOCK=B,ROWS=ROWS)
    return y
