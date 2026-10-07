import os,sys,json,gc,time,argparse,hashlib,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)];os.environ["AMNI_BLENDER_NOTHR"]="1"
ap=argparse.ArgumentParser();ap.add_argument("--bake",required=True);ap.add_argument("--cfgs",required=True);ap.add_argument("--out",required=True);ap.add_argument("--set",default="orig",choices=("orig","dev"));ap.add_argument("--n-new",type=int,default=48);ap.add_argument("--long",type=int,default=0);ap.add_argument("--ref",default="");ap.add_argument("--max-dev",type=int,default=0);a=ap.parse_args()
from amni.serve import blender_atex as B
P=["User: Marlowe Bridge: Marlowe Bridge is a stone arch bridge over the River Tane in the county of Eastmere. It carries a single lane of traffic. What is Marlowe Bridge?\nAdam:","User: Calculator: The answer is 84. Working: 12 x 7 = 84. What is 12 times 7?\nAdam:","User: Hi Adam, long day at work and I am wiped out.\nAdam:"]
D=ROOT/"data"/"adam_t"/"eval"/"hybrid1_dev";J=lambda f:[json.loads(l) for l in open(D/f) if l.strip()];BL=J("blend_probe.jsonl");GV=J("greet_vent_probe.jsonl")
P+=([f"User: {' '.join(r['material']+[r['q']])}\nAdam:" for r in BL]+[f"User: {r['q']}\nAdam:" for r in GV])[:a.max_dev or None] if a.set=="dev" else []
P+=["".join(f"User: {' '.join(r['material']+[r['q']])}\nAdam: {r.get('ref') or 'I am not sure about that.'}\n\n" for r in BL[k*6%45:k*6%45+6])+f"User: {GV[k%len(GV)]['q']} Also, {BL[(k*6+7)%len(BL)]['q']}\nAdam:" for k in range(a.long)]
R={"prompts":len(P),"n_new":a.n_new,"alloc_conf":os.environ.get("PYTORCH_HIP_ALLOC_CONF",""),"cfgs":{}};base=None;REF=None;rn=a.ref or next(iter(json.loads(a.cfgs)))
for name,o in json.loads(a.cfgs).items():
    t0=time.time();b=B.Blender("atex:"+a.bake,{**o,"sys":"split4"});out=[];LG=[];NP=[]
    for p in P:
        lg=[];ids=b.prompt_ids(p);NP.append(int(ids.shape[-1]));b.e.generate(ids,a.n_new,(),0.0,lg_out=lg);out.append(list(b.e.last_ids));LG.append(torch.stack(lg) if lg else torch.zeros(0))
    REF=REF if REF is not None or name!=rn else (out,LG);base=base or out
    dv=lambda u,v:next((i for i,(x,y) in enumerate(zip(u,v)) if x!=y),None if len(u)==len(v) else min(len(u),len(v)))
    PI=[{"div":dv(u,r),"n":len(u),"n_ref":len(r),"np":n,"maxabs":round(float(max(((l[:k+1]-q[:k+1]).abs().max() for k in [min(len(l),len(q))-1 if dv(u,r) is None else min(dv(u,r),len(l)-1,len(q)-1)] if k>=0),default=0.0)),5)} for u,r,l,q,n in zip(out,REF[0],LG,REF[1],NP)] if REF is not None else []
    m=[sum(x==y for x,y in zip(u,v))/max(1,max(len(u),len(v))) for u,v in zip(out,base)];bad=[i for i,u in enumerate(out) if len(u)>=8 and len(set(u[-8:]))==1 and (len(base[i])<8 or len(set(base[i][-8:]))>1)]
    R["cfgs"][name]={"opts":o,"ref":rn,"identical_to_first":all(u==v for u,v in zip(out,base)),"token_match_frac":[round(x,3) for x in m[:3]],"sample":b.e.tok.decode(out[0],skip_special_tokens=True)[:300],"n_identical_to_ref":sum(x["div"] is None for x in PI),"n_cmp":len(PI),"diverged":[{"i":i,**x} for i,x in enumerate(PI) if x["div"] is not None],"maxabs_identical_max":max((x["maxabs"] for x in PI if x["div"] is None),default=None),"maxabs_all_max":max((x["maxabs"] for x in PI),default=None),"degenerate_repeat_prompts":bad,"toks_sha1":[hashlib.sha1(json.dumps(u).encode()).hexdigest()[:12] for u in out],"max_prompt_tokens":max(NP),"s":round(time.time()-t0,1)}
    print(name,R["cfgs"][name]["identical_to_first"],R["cfgs"][name]["token_match_frac"],"identical_to_ref",R["cfgs"][name]["n_identical_to_ref"],"/",len(PI),"maxabs",R["cfgs"][name]["maxabs_all_max"],"degenerate",len(bad),flush=True)
    del b,LG;gc.collect();torch.cuda.empty_cache()
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(R,open(a.out,"w"),indent=1)
