import os,sys,json,glob,time,shutil,argparse,re,hashlib,torch,torch.nn as nn,safetensors.torch as st
from safetensors import safe_open
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from amni.inference import qgroup_gemv as Q
ap=argparse.ArgumentParser();ap.add_argument("--src",required=True);ap.add_argument("--out",required=True);ap.add_argument("--bits",type=int,default=4);ap.add_argument("--gs",type=int,default=128);ap.add_argument("--emb-bits",type=int,default=0);ap.add_argument("--emb-gs",type=int,default=128);ap.add_argument("--mix",default="");ap.add_argument("--calib",default="");ap.add_argument("--nseq",type=int,default=128);ap.add_argument("--seqlen",type=int,default=512);ap.add_argument("--damp",type=float,default=0.01);ap.add_argument("--seed",type=int,default=11);ap.add_argument("--dev",default="cuda");a=ap.parse_args()
MIX=[(re.compile(p),int(b)) for p,b in (x.split("=") for x in a.mix.split(",") if x)]
os.makedirs(a.out,exist_ok=True);dev=a.dev
LIN=lambda k,w:w.dim()==2 and ".layers." in k and k.endswith(".weight") and "norm" not in k
EMB=lambda k,w:w.dim()==2 and (k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"))
CF=json.load(open(a.src+"/config.json"));TX="text_config" in CF and "vision_config" in CF
RK=lambda k:None if re.match(r"(model\.visual\.|model\.mtp|mtp\.)",k) else k.replace("model.language_model.","model.")
TX and json.dump({**CF["text_config"],"architectures":[CF["architectures"][0].replace("ForConditionalGeneration","ForCausalLM")],"tie_word_embeddings":CF.get("tie_word_embeddings",CF["text_config"].get("tie_word_embeddings",False))},open(os.path.join(a.out,"config.json"),"w"),indent=1)
man={"gs":a.gs,"bits":a.bits,"emb_bits":a.emb_bits,"emb_gs":a.emb_gs,"fmt":"qgroup-affine-v1"+("-gptq" if a.calib else ""),"text_only_from_multimodal":TX,"tensors":{}};rep={};t0=time.time();nb=[0]
def bake(k,w,bits,gs,H=None):
    W=w.to(dev);q,sa,sc,dp=Q.gptq(W,H,bits,gs,a.damp) if H is not None else Q.quantize(W,bits,gs)+(None,);pl=Q.split(q,bits);D=Q.dequant(pl,sa,sc,bits,gs,W.shape[1]);Wf=W.float();E=D-Wf
    cos=torch.nn.functional.cosine_similarity(D.flatten().double(),Wf.flatten().double(),dim=0).item();rel=(E.norm()/Wf.norm()).item();ok=bool((Q.unpack(pl,bits,W.shape[1])==q).all());hr=((E@H*E).sum()/(Wf@H*Wf).sum()).clamp_min(0).sqrt().item() if H is not None else None
    return {k+".p0":pl[0].cpu(),**({k+".p1":pl[1].cpu()} if len(pl)>1 else {}),k+".a":sa.cpu(),k+".c":sc.cpu()},{"cos":cos,"rel_err":rel,"out_rel_err":hr,"damp":dp,"codes_roundtrip_exact":ok,"bits":bits,"gs":gs,"numel":W.numel()},D
def put(sd,k,w,H=None):
    q=LIN(k,w) or (a.emb_bits and EMB(k,w));bits,gs=(a.bits,a.gs) if LIN(k,w) else (a.emb_bits,a.emb_gs);bits=next((b for p,b in MIX if p.search(k)),bits)
    o,r,D=bake(k,w,bits,gs,H) if q else ({k:w.to(torch.bfloat16) if w.is_floating_point() and w.dim()>1 else w},None,None);sd.update(o);r and rep.__setitem__(k,r);man["tensors"][k]={"q":1,"shape":list(w.shape),"bits":bits,"gs":gs} if q else {"q":0,"shape":list(w.shape)};nb[0]+=sum(v.numel()*v.element_size() for v in o.values()) if q else w.numel()*(2 if w.is_floating_point() and w.dim()>1 else w.element_size());return D
SH={RK(k):sh for sh in sorted(glob.glob(a.src+"/*.safetensors")) for k in safe_open(sh,framework="pt").keys() if RK(k)}
def calib():
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    cfg=AutoConfig.from_pretrained(a.out if TX else a.src);tok=AutoTokenizer.from_pretrained(a.src);F={sh:safe_open(sh,framework="pt") for sh in set(SH.values())};RV={RK(k):k for f in F.values() for k in f.keys() if RK(k)}
    with init_empty_weights():m=AutoModelForCausalLM.from_config(cfg).eval()
    m.load_state_dict({k:F[SH[k]].get_tensor(RV[k]).to(torch.bfloat16) for k in SH},assign=True,strict=False);m.requires_grad_(False);[c.to(dev) for n,c in m.model.named_children() if n!="layers"]
    T=open(a.calib).read();ids=tok(T,return_tensors="pt",add_special_tokens=False).input_ids[0];g=torch.Generator().manual_seed(a.seed);X=torch.stack([ids[i:i+a.seqlen] for i in torch.randint(0,ids.numel()-a.seqlen-1,(a.nseq,),generator=g).tolist()])
    man["calib"]={"file":os.path.basename(a.calib),"sha256":hashlib.sha256(T.encode()).hexdigest(),"nseq":a.nseq,"seqlen":a.seqlen,"seed":a.seed,"damp":a.damp,"tokens":int(X.numel())};E=[]
    class Stop(Exception):pass
    def grab(mod,args,kw):E.append((args[0] if args else kw.pop("hidden_states"),kw));raise Stop
    h=m.model.layers[0].register_forward_pre_hook(grab,with_kwargs=True)
    with torch.no_grad():
        for i in range(0,a.nseq,8):
            try:m.model(input_ids=X[i:i+8].to(dev),use_cache=False)
            except Stop:pass
    h.remove();lt=list(getattr(cfg,"layer_types",None) or []);assert len(set(lt))<=1 or E[0][1].get("attention_mask") is None,"mixed layer types need per-type masks";sd={};k0=0
    for li,lay in enumerate(m.model.layers):
        lay.to(dev);L={n:x for n,x in lay.named_modules() if isinstance(x,nn.Linear)};Hs={n:torch.zeros(x.in_features,x.in_features,device=dev) for n,x in L.items()};G={};N={n:0 for n in L}
        def mk(n):
            def f(mod,args):x=args[0].reshape(-1,args[0].shape[-1]).float();Hs[n].addmm_(x.T,x);N[n]+=x.shape[0];G.setdefault(n,args[0].data_ptr())
            return f
        hs=[x.register_forward_pre_hook(mk(n)) for n,x in L.items()]
        with torch.no_grad():[lay(hd,**kw) for hd,kw in E]
        [x.remove() for x in hs]
        for gp in dict.fromkeys(G[n] for n in L):
            ns=[n for n in L if G[n]==gp];ks=[f"model.layers.{li}.{n}.weight" for n in ns];H=Hs[ns[0]]*2/max(1,N[ns[0]])
            for n,k in zip(ns,ks):L[n].weight.data=put(sd,k,L[n].weight.data,H).to(L[n].weight.dtype)
        Hs.clear()
        with torch.no_grad():E=[((lambda r:r[0] if isinstance(r,tuple) else r)(lay(hd,**kw)),kw) for hd,kw in E]
        [put(sd,k,F[SH[k]].get_tensor(RV[k])) for k in SH if k.startswith(f"model.layers.{li}.") and k not in man["tensors"]];lay.to("meta")
        (li%8==7 or li==len(m.model.layers)-1) and (st.save_file(sd,os.path.join(a.out,f"model-gptq-{k0:05d}.safetensors"),metadata={"format":"pt"}),sd.clear());k0+=li%8==7;print(f"layer {li} {time.time()-t0:.0f}s",flush=True);torch.cuda.empty_cache()
    [put(sd,k,F[SH[k]].get_tensor(RV[k])) for k in SH if k not in man["tensors"]];st.save_file(sd,os.path.join(a.out,"model-rest.safetensors"),metadata={"format":"pt"})
if a.calib:calib()
else:
    for sh in sorted(set(SH.values())):
        sd={}
        with safe_open(sh,framework="pt") as f:[put(sd,RK(k),f.get_tensor(k)) for k in f.keys() if RK(k)]
        st.save_file(sd,os.path.join(a.out,os.path.basename(sh)),metadata={"format":"pt"});del sd;torch.cuda.empty_cache();print(os.path.basename(sh),f"{time.time()-t0:.0f}s",flush=True)
L=[v for k,v in rep.items() if ".layers." in k];E=[v for k,v in rep.items() if ".layers." not in k]
agg=lambda X:{"n":len(X),"cos_min":min(v["cos"] for v in X),"cos_mean":sum(v["cos"] for v in X)/len(X),"rel_err_mean":sum(v["rel_err"] for v in X)/len(X),"rel_err_max":max(v["rel_err"] for v in X),"rel_err_weighted":sum(v["rel_err"]*v["numel"] for v in X)/sum(v["numel"] for v in X),"out_rel_err_mean":(lambda Y:sum(Y)/len(Y) if Y else None)([v["out_rel_err"] for v in X if v["out_rel_err"] is not None]),"codes_exact_all":all(v["codes_roundtrip_exact"] for v in X)} if X else {}
man["bake_bytes"]=nb[0];man["fidelity"]={"layers":agg(L),"embed_head":agg(E)};json.dump(man,open(os.path.join(a.out,"bake_manifest.json"),"w"));json.dump(rep,open(os.path.join(a.out,"fidelity_per_tensor.json"),"w"))
[shutil.copy(f,a.out) for f in glob.glob(a.src+"/*") if not f.endswith((".safetensors",".md")) and os.path.basename(f) not in ("model.safetensors.index.json","model.safetensors.index.json")+(("config.json","preprocessor_config.json","video_preprocessor_config.json") if TX else ())]
print(json.dumps({"bake_GB":round(nb[0]/1e9,3),"s":round(time.time()-t0),**man["fidelity"]},indent=1))
