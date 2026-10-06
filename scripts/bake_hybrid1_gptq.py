import os,sys,json,glob,time,shutil,argparse,gc,torch,safetensors.torch as st
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from amni.inference import qgroup_gemv as Q
ap=argparse.ArgumentParser();ap.add_argument("--src",required=True);ap.add_argument("--out",required=True);ap.add_argument("--bits",type=int,default=3);ap.add_argument("--gs",type=int,default=64);ap.add_argument("--text",nargs="+",required=True);ap.add_argument("--nseq",type=int,default=96);ap.add_argument("--seqlen",type=int,default=512);ap.add_argument("--damp",type=float,default=0.01);a=ap.parse_args()
dev="cuda";t0=time.time();tok=AutoTokenizer.from_pretrained(a.src);m=AutoModelForCausalLM.from_pretrained(a.src,dtype=torch.bfloat16).to(dev).eval();m.requires_grad_(False)
T=" ".join(Path(f).read_text(errors="ignore") for f in a.text);ids=tok(T[:6000000],return_tensors="pt").input_ids[0];g=torch.Generator().manual_seed(11);st0=torch.randint(0,ids.numel()-a.seqlen-1,(a.nseq,),generator=g);X=torch.stack([ids[i:i+a.seqlen] for i in st0.tolist()]).to(dev)
L=m.model.layers;qm=(1<<a.bits)-1
def gptq(W,H):
    W=W.float().clone();o,n=W.shape;d=torch.arange(n,device=dev);dead=H.diag()==0;H[dead,dead]=1;W[:,dead]=0;H[d,d]+=a.damp*H.diag().mean();Hi=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True);Qc=torch.zeros_like(W,dtype=torch.int32);A=torch.zeros(o,n//a.gs,device=dev);C=torch.zeros_like(A)
    for i1 in range(0,n,128):
        i2=min(i1+128,n);W1=W[:,i1:i2].clone();Err=torch.zeros_like(W1);Hi1=Hi[i1:i2,i1:i2]
        for j in range(i2-i1):
            col=i1+j
            if col%a.gs==0:
                q0,sa,sc=Q.quantize(W1[:,j:j+a.gs],a.bits,a.gs,iters=2);A[:,col//a.gs]=sa.float().squeeze(-1);C[:,col//a.gs]=sc.float().squeeze(-1)
            gi=col//a.gs;s=A[:,gi];c=C[:,gi];w=W1[:,j];q=torch.clamp(torch.round((w-c)/s.clamp_min(1e-8)),0,qm);dq=q*s+c;Qc[:,col]=q.to(torch.int32);err=(w-dq)/Hi1[j,j];W1[:,j:]-=err.unsqueeze(1)*Hi1[j,j:].unsqueeze(0);Err[:,j]=err
        W[:,i2:]-=Err@Hi[i1:i2,i2:]
    return Qc,A.half(),C.half()
inps=[];cache={}
class Stop(Exception):pass
def grab(mod,args,kw):inps.append((args,kw));raise Stop
h=L[0].register_forward_pre_hook(grab,with_kwargs=True)
with torch.no_grad():
    for i in range(0,a.nseq,8):
        try:m.model(input_ids=X[i:i+8],use_cache=False)
        except Stop:pass
h.remove();E=[(x[0][0] if x[0] else x[1]["hidden_states"],x[1]) for x in inps]
out={};man={"gs":a.gs,"bits":a.bits,"emb_bits":0,"emb_gs":128,"fmt":"qgroup-affine-v1-gptq","tensors":{}};rep={}
sd0={}
for sh in sorted(glob.glob(a.src+"/*.safetensors")):
    from safetensors import safe_open
    with safe_open(sh,framework="pt") as f:
        for k in f.keys():sd0[k]=sh
for li,lay in enumerate(L):
    lins={n:mod for n,mod in lay.named_modules() if isinstance(mod,torch.nn.Linear)};Hs={n:torch.zeros(mod.in_features,mod.in_features,device=dev) for n,mod in lins.items()};cnt={n:0 for n in lins}
    def mk(n):
        def f(mod,args):
            x=args[0].reshape(-1,args[0].shape[-1]).float();Hs[n]+=x.T@x;cnt[n]+=x.shape[0]
        return f
    hs=[mod.register_forward_pre_hook(mk(n)) for n,mod in lins.items()]
    with torch.no_grad():
        for hid,kw in E:lay(hid,**{k:v for k,v in kw.items()})
    [x.remove() for x in hs]
    for n,mod in lins.items():
        Hn=Hs[n]*2/max(1,cnt[n]);Wq=mod.weight.data;Qc,sa,sc=gptq(Wq,Hn);pl=Q.split(Qc,a.bits);D=Q.dequant(pl,sa,sc,a.bits,a.gs,Wq.shape[1]);Wf=Wq.float();key=f"model.layers.{li}.{n}.weight"
        rep[key]={"cos":torch.nn.functional.cosine_similarity(D.flatten(),Wf.flatten(),dim=0).item(),"rel_err":((D-Wf).norm()/Wf.norm()).item(),"codes_roundtrip_exact":bool((Q.unpack(pl,a.bits,Wq.shape[1])==Qc).all()),"bits":a.bits,"gs":a.gs,"numel":Wq.numel()}
        out.update({key+".p0":pl[0].cpu(),**({key+".p1":pl[1].cpu()} if len(pl)>1 else {}),key+".a":sa.cpu(),key+".c":sc.cpu()});man["tensors"][key]={"q":1,"shape":list(Wq.shape),"bits":a.bits,"gs":a.gs};mod.weight.data=D.to(mod.weight.dtype)
    del Hs
    with torch.no_grad():E=[((lambda r:r[0] if isinstance(r,tuple) else r)(lay(hid,**kw)),kw) for hid,kw in E]
    print(f"layer {li} {time.time()-t0:.0f}s",flush=True);torch.cuda.empty_cache()
nb=sum(v.numel()*v.element_size() for v in out.values())
for k,sh in sd0.items():
    if k in man["tensors"] or any(k==x for x in out):continue
    from safetensors import safe_open
    with safe_open(sh,framework="pt") as f:w=f.get_tensor(k)
    out[k]=w.to(torch.bfloat16) if w.is_floating_point() else w;man["tensors"][k]={"q":0,"shape":list(w.shape)};nb+=w.numel()*2
os.makedirs(a.out,exist_ok=True);st.save_file(out,os.path.join(a.out,"model-gptq.safetensors"),metadata={"format":"pt"})
Lr=list(rep.values());agg={"n":len(Lr),"cos_min":min(v["cos"] for v in Lr),"cos_mean":sum(v["cos"] for v in Lr)/len(Lr),"rel_err_mean":sum(v["rel_err"] for v in Lr)/len(Lr),"rel_err_max":max(v["rel_err"] for v in Lr),"rel_err_weighted":sum(v["rel_err"]*v["numel"] for v in Lr)/sum(v["numel"] for v in Lr),"codes_exact_all":all(v["codes_roundtrip_exact"] for v in Lr)}
man["bake_bytes"]=nb;man["fidelity"]={"layers":agg,"embed_head":{}};json.dump(man,open(os.path.join(a.out,"bake_manifest.json"),"w"));json.dump(rep,open(os.path.join(a.out,"fidelity_per_tensor.json"),"w"))
[shutil.copy(f,a.out) for f in glob.glob(a.src+"/*") if not f.endswith((".safetensors",".md")) and os.path.basename(f)!="model.safetensors.index.json"]
print(json.dumps({"bake_GB":round(nb/1e9,3),"s":round(time.time()-t0),**agg}))
