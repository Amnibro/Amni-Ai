import os,json,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");TA=os.path.join(ROOT,"data","torus","a2b");TC=os.path.join(ROOT,"data","torus","c");SEED=6029;NSC=10
def main():
 d=np.load(os.path.join(TA,"a2b_data.npz"),allow_pickle=True);H,T,R,SPo=d["h"],d["t"],d["r"].astype(np.int64),d["split"];N=len(d["names"]);rels=[str(x) for x in d["rels"]];NL=len(rels);rng=np.random.default_rng(SEED)
 pk=np.minimum(H,T).astype(np.int64)*N+np.maximum(H,T);u,inv,cn=np.unique(pk,return_inverse=True,return_counts=True);uniq=cn[inv]==1;elig=(SPo==0)&(R<NSC)&(H!=T)&uniq
 deg=np.bincount(np.concatenate([H,T]),minlength=N).astype(np.int64);tc=np.bincount(R*N+T,minlength=NL*N).astype(np.int64);cnt=np.bincount(R,minlength=NL);cap=np.ceil(cnt*0.1).astype(np.int64);nh=np.zeros((NL,3),np.int64);SP=np.zeros(len(H),np.int8)
 for i in rng.permutation(np.nonzero(elig)[0]).tolist():
  r,h,t=int(R[i]),int(H[i]),int(T[i])
  if deg[h]<3 or deg[t]<3 or tc[r*N+t]<2:continue
  s=2 if nh[r,2]<=nh[r,1] and nh[r,2]<cap[r] else (1 if nh[r,1]<cap[r] else 0)
  if s:SP[i]=s;nh[r,s]+=1;deg[h]-=1;deg[t]-=1;tc[r*N+t]-=1
 tr=SP==0;dg=np.bincount(np.concatenate([H[tr],T[tr]]),minlength=N);ho=~tr;assert dg[H[ho]].min()>=2 and dg[T[ho]].min()>=2 and (SPo[ho]==0).all() and uniq[ho].all()
 for r in range(NL):m=R==r;print(f"{rels[r]} facts={int(m.sum())} test={int((SP[m]==2).sum())} dev={int((SP[m]==1).sum())} old_dev_test_now_train={int(((SPo[m]>0)&(SP[m]==0)).sum())}",flush=True)
 out={k:d[k] for k in d.files};out["split"]=SP;np.savez_compressed(os.path.join(TC,"c_data.npz"),**out)
 print(json.dumps({"entities":N,"facts":int(len(H)),"train":int(tr.sum()),"dev":int((SP==1).sum()),"test":int((SP==2).sum())}),flush=True)
if __name__=="__main__":main()
