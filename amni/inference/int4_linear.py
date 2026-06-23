import torch,torch.nn as nn,torch.nn.functional as F
class Int4GroupLinear(nn.Module):
    def __init__(s,W,bias=None,gs=128):
        super().__init__()
        out,inf=W.shape;s.out,s.inf,s.gs=out,inf,gs
        pad=(gs-inf%gs)%gs;s.pad=pad;s.infp=inf+pad
        Wp=F.pad(W.float(),(0,pad)) if pad else W.float()
        Wg=Wp.reshape(out,-1,gs)
        sc=Wg.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7.0
        q=torch.clamp(torch.round(Wg/sc),-8,7).to(torch.int16)
        qf=(q.reshape(out,-1)+8).to(torch.uint8)
        s.register_buffer('packed',(qf[:,0::2]|(qf[:,1::2]<<4)).contiguous())
        s.register_buffer('scale',sc.squeeze(-1).to(torch.bfloat16).contiguous())
        s.register_buffer('bias',bias.to(torch.bfloat16).contiguous() if bias is not None else None)
    def dequant(s):
        p=s.packed
        q=torch.empty(s.out,s.infp,dtype=torch.int16,device=p.device)
        q[:,0::2]=(p&0xF).to(torch.int16)-8;q[:,1::2]=((p>>4)&0xF).to(torch.int16)-8
        W=(q.reshape(s.out,-1,s.gs).to(torch.bfloat16)*s.scale.unsqueeze(-1)).reshape(s.out,s.infp)
        return W[:,:s.inf] if s.pad else W
    def forward(s,x):
        return F.linear(x.to(torch.bfloat16),s.dequant(),s.bias).to(x.dtype)
def quantize_linears(model,gs=128,skip=('lm_head','embed')):
    import torch.nn as nn
    reps=[]
    for name,mod in model.named_modules():
        if isinstance(mod,nn.Linear) and mod.weight.dim()==2 and not any(k in name for k in skip):
            reps.append((name,mod))
    n=0
    for name,mod in reps:
        parent=model;parts=name.split('.')
        for p in parts[:-1]:parent=getattr(parent,p)
        dev=mod.weight.device
        ql=Int4GroupLinear(mod.weight.data,mod.bias.data if mod.bias is not None else None,gs).to(dev)
        setattr(parent,parts[-1],ql);del mod;n+=1
    return n
if __name__=='__main__':
    torch.manual_seed(0)
    lin=torch.nn.Linear(2560,4096,bias=True).cuda().bfloat16()
    x=torch.randn(2,2560).cuda().bfloat16()
    ref=lin(x)
    ql=Int4GroupLinear(lin.weight.data,lin.bias.data,128).cuda()
    out=ql(x)
    cos=F.cosine_similarity(ref.float().flatten(),out.float().flatten(),dim=0).item()
    bf16_bytes=lin.weight.numel()*2;int4_bytes=ql.packed.numel()+ql.scale.numel()*2
    print(f'Int4GroupLinear correctness: cos={cos:.4f} | storage {int4_bytes/1e6:.1f}MB vs bf16 {bf16_bytes/1e6:.1f}MB ({int4_bytes/bf16_bytes:.2f}x)')
    print('VERDICT:','OK module works + ~4x smaller' if cos>0.99 and int4_bytes/bf16_bytes<0.3 else 'check')
