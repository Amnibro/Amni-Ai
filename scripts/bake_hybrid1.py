import os,sys,json,glob,time,shutil,argparse,re,torch,safetensors.torch as st
from safetensors import safe_open
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from amni.inference import qgroup_gemv as Q
ap=argparse.ArgumentParser();ap.add_argument("--src",required=True);ap.add_argument("--out",required=True);ap.add_argument("--bits",type=int,default=4);ap.add_argument("--gs",type=int,default=128);ap.add_argument("--emb-bits",type=int,default=0);ap.add_argument("--emb-gs",type=int,default=128);ap.add_argument("--mix",default="");a=ap.parse_args()
MIX=[(re.compile(p),int(b)) for p,b in (x.split("=") for x in a.mix.split(",") if x)]
os.makedirs(a.out,exist_ok=True);dev="cuda"
LIN=lambda k,w:w.dim()==2 and ".layers." in k and k.endswith(".weight") and "norm" not in k
EMB=lambda k,w:w.dim()==2 and (k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"))
man={"gs":a.gs,"bits":a.bits,"emb_bits":a.emb_bits,"emb_gs":a.emb_gs,"fmt":"qgroup-affine-v1","tensors":{}};rep={};t0=time.time();nb=0
def bake(k,w,bits,gs):
    W=w.to(dev);q,sa,sc=Q.quantize(W,bits,gs);pl=Q.split(q,bits);D=Q.dequant(pl,sa,sc,bits,gs,W.shape[1]);Wf=W.float()
    cos=torch.nn.functional.cosine_similarity(D.flatten(),Wf.flatten(),dim=0).item();rel=((D-Wf).norm()/Wf.norm()).item();ok=bool((Q.unpack(pl,bits,W.shape[1])==q).all())
    return {k+".p0":pl[0].cpu(),**({k+".p1":pl[1].cpu()} if len(pl)>1 else {}),k+".a":sa.cpu(),k+".c":sc.cpu()},{"cos":cos,"rel_err":rel,"codes_roundtrip_exact":ok,"bits":bits,"gs":gs,"numel":W.numel()}
for sh in sorted(glob.glob(a.src+"/*.safetensors")):
    sd={}
    with safe_open(sh,framework="pt") as f:
        for k in f.keys():
            w=f.get_tensor(k)
            if LIN(k,w) or (a.emb_bits and EMB(k,w)):
                bits,gs=(a.bits,a.gs) if LIN(k,w) else (a.emb_bits,a.emb_gs)
                bits=next((b for p,b in MIX if p.search(k)),bits)
                o,r=bake(k,w,bits,gs);sd.update(o);rep[k]=r;man["tensors"][k]={"q":1,"shape":list(w.shape),"bits":bits,"gs":gs};nb+=sum(v.numel()*v.element_size() for v in o.values())
            else:
                sd[k]=w.to(torch.bfloat16) if w.is_floating_point() else w;man["tensors"][k]={"q":0,"shape":list(w.shape)};nb+=w.numel()*2
    st.save_file(sd,os.path.join(a.out,os.path.basename(sh)),metadata={"format":"pt"});del sd;torch.cuda.empty_cache();print(os.path.basename(sh),f"{time.time()-t0:.0f}s",flush=True)
L=[v for k,v in rep.items() if ".layers." in k];E=[v for k,v in rep.items() if ".layers." not in k];tot=sum(v["numel"] for v in L)
agg=lambda X:{"n":len(X),"cos_min":min(v["cos"] for v in X),"cos_mean":sum(v["cos"] for v in X)/len(X),"rel_err_mean":sum(v["rel_err"] for v in X)/len(X),"rel_err_max":max(v["rel_err"] for v in X),"rel_err_weighted":sum(v["rel_err"]*v["numel"] for v in X)/sum(v["numel"] for v in X),"codes_exact_all":all(v["codes_roundtrip_exact"] for v in X)} if X else {}
man["bake_bytes"]=nb;man["fidelity"]={"layers":agg(L),"embed_head":agg(E)};json.dump(man,open(os.path.join(a.out,"bake_manifest.json"),"w"));json.dump(rep,open(os.path.join(a.out,"fidelity_per_tensor.json"),"w"))
[shutil.copy(f,a.out) for f in glob.glob(a.src+"/*") if not f.endswith((".safetensors",".md")) and os.path.basename(f)!="model.safetensors.index.json"]
print(json.dumps({"bake_GB":round(nb/1e9,3),"s":round(time.time()-t0),**man["fidelity"]},indent=1))
