import os,sys,json,numpy as np
H=os.environ.get("AMNI_HOME","/home/anmire/ai/Amni-Ai");src=sys.argv[1]
z=np.load(src) if src.endswith(".npz") else {k:np.load(os.path.join(src,k+".npy")) for k in ("cand","val","vlen")}
rows=[json.loads(l) for l in open(os.path.join(H,"data","chat3","hold_eval.jsonl"))];KINDS=("simple","multi","correction","pronoun","twohop","longgap")
by={k:[0,0] for k in KINDS};ranks={k:[] for k in KINDS};unk=uz=0
for i,r in enumerate(rows):
 b,bi=divmod(i,8);c=z["cand"][b,bi];cv=["" if int(c[k])<0 else bytes(z["val"][b,bi,int(c[k]),:int(z["vlen"][b,bi,int(c[k])])].astype(np.uint8)).decode("ascii","replace") for k in range(12)]
 direct=[v for v in cv[:8] if v];vals=direct+[v for v in cv[8:] if v];k=r["kind"];ans=r["answer"]
 if k=="unknown":unk+=1;uz+=1 if not direct else 0;continue
 by[k][1]+=1;by[k][0]+=int(ans in vals and (not ans or any(v[:1]==ans[:1] for v in vals)));rank=next((j+1 for j,v in enumerate(cv) if v==ans),None);ranks[k].append(rank) if rank else None
print("CANDIDATE_RECALL",src)
for k in KINDS:a,n=by[k];rs=ranks[k];print(f"{k:12} recall8={a/max(1,n):.4f} {a}/{n} mean_rank={sum(rs)/len(rs):.2f}" if rs else f"{k:12} recall8={a/max(1,n):.4f} {a}/{n} mean_rank=na")
print(f"unknown_zero_direct={uz/max(1,unk):.4f} {uz}/{unk}")
