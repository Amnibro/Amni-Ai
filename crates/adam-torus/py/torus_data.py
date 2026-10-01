import os,sys,time,json,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");A2B="--a2b" in sys.argv;OUT=os.path.join(ROOT,"data","torus","a2b","a2b_data.npz") if A2B else os.path.join(ROOT,"data","torus","a2_data.npz");SEED=2930 if A2B else 17;AUX=[27,20,106] if A2B else []
RELS=["wn:hypernym","wn:part_holonym","wn:member_holonym","cn:UsedFor","cn:AtLocation","cn:CapableOf","wd:P19","wd:P17","wd:P131","wd:P171"]+[f"wd:P{p}" for p in AUX];NSC=10;RTYPE=np.array([0,0,0,1,1,1,2,2,2,2]+[2]*len(AUX),np.int8)
WNT={"wn:hypernym":1,"wn:part_holonym":4,"wn:member_holonym":6};CNR={"cn:UsedFor":1,"cn:AtLocation":6,"cn:CapableOf":2}
def whash(w):
 h=0
 for c in w.encode():h=(h*83492791+c+1)&0xFFFFFFFF
 return h
def dsplit(H,T,R,SP,N,rng):
 deg=np.bincount(np.concatenate([H,T]),minlength=N).astype(np.int64);tc=np.bincount(R.astype(np.int64)*N+T,minlength=len(RELS)*N).astype(np.int64);cnt=np.bincount(R,minlength=len(RELS));cap=np.ceil(cnt*0.1).astype(np.int64);nh=np.zeros((len(RELS),3),np.int64)
 for i in rng.permutation(len(H)).tolist():
  r,h,t=int(R[i]),int(H[i]),int(T[i])
  if r>=NSC or h==t or deg[h]<3 or deg[t]<3 or tc[r*N+t]<2:continue
  s=2 if nh[r,2]<=nh[r,1] and nh[r,2]<cap[r] else (1 if nh[r,1]<cap[r] else 0)
  if s:SP[i]=s;nh[r,s]+=1;deg[h]-=1;deg[t]-=1;tc[r*N+t]-=1
 tr=SP==0;dg=np.bincount(np.concatenate([H[tr],T[tr]]),minlength=N);ho=~tr;assert dg[H[ho]].min()>=2 and dg[T[ho]].min()>=2
 for r in range(len(RELS)):m=R==r;print(f"{RELS[r]} facts={int(m.sum())} test={int((SP[m]==2).sum())} dev={int((SP[m]==1).sum())} heads={len(np.unique(H[m]))} tails={len(np.unique(T[m]))}",flush=True)
def main():
 t0=time.time();rng=np.random.default_rng(SEED);sp=np.load(os.path.join(ROOT,"data","concepts","sphere.npz"),allow_pickle=True);spx=None if A2B else np.load(os.environ.get("SPHERE_NPZ",os.path.join(ROOT,"data","torus","sphere_noleak.npz")))["xyz"];wn=np.load(os.path.join(ROOT,"data","concepts","wordnet_graph.npz"),allow_pickle=True)
 Nsp=len(sp["node_names"]);w2n=dict(zip(sp["word_hash"].tolist(),sp["word_node"].tolist()));names=[str(x) for x in sp["node_names"]];src=[0]*Nsp;lift=list(range(Nsp))
 facts={}
 es,ed,et=wn["edge_src"].astype(np.int64),wn["edge_dst"].astype(np.int64),wn["edge_type"]
 for r,k in WNT.items():m=et==k;facts[r]=(es[m],ed[m])
 cd=os.path.join(ROOT,"data","concepts","conceptnet");ep=np.load(cd+"/e_p.npy");eD=np.load(cd+"/e_d.npy").astype(np.int64);eR=np.load(cd+"/e_r.npy");no=np.load(cd+"/n_o.npy");txt=open(cd+"/n_txt.bin","rb").read()
 eS=np.repeat(np.arange(len(ep)-1),np.diff(ep));cnid={}
 def cn(i):
  if i in cnid:return cnid[i]
  t=txt[no[i]:no[i+1]].decode("utf-8","replace");j=w2n.get(whash(t.replace(" ","_")),-1)
  if j<0:j=len(names);names.append("cn:"+t);src.append(1);lift.append(-1)
  cnid[i]=j;return j
 for r,k in CNR.items():m=eR==k;facts[r]=(np.array([cn(int(i)) for i in eS[m]],np.int64),np.array([cn(int(i)) for i in eD[m]],np.int64))
 st=os.path.join(ROOT,"data","wikifact","v3","store");fs=np.load(st+"/f_s.npy",mmap_mode="r");fp=np.load(st+"/f_p.npy",mmap_mode="r");fk=np.load(st+"/f_k.npy",mmap_mode="r");fv=np.load(st+"/f_v.npy",mmap_mode="r")
 sel=np.nonzero((fk==1)&np.isin(fp,[19,17,131,171]+AUX))[0];S=fs[sel].astype(np.int64);P=fp[sel];V=fv[sel].astype(np.int64)
 def sub(p,subj):m=(P==p)&np.isin(S,subj);return S[m],V[m]
 per=np.unique(S[P==19]);per=np.intersect1d(per,np.unique(S[np.isin(P,AUX)])) if A2B else per;per=rng.choice(per,150000,replace=False);a=sub(19,per);ax={f"wd:P{p}":sub(p,per) for p in AUX};r1=sub(131,np.unique(np.concatenate([a[1]]+([ax["wd:P20"][1]] if A2B else []))));r2=sub(131,np.unique(r1[1]))
 pl=np.unique(np.concatenate([a[1],r1[1],r2[1]]+([ax["wd:P20"][1]] if A2B else [])));b=sub(17,pl);tx=np.unique(S[P==171]);tx=rng.choice(tx,100000,replace=False);c1=sub(171,tx);c2=sub(171,np.unique(c1[1]))
 wdf={"wd:P19":a,"wd:P17":b,"wd:P131":(np.concatenate([r1[0],r2[0]]),np.concatenate([r1[1],r2[1]])),"wd:P171":(np.concatenate([c1[0],c2[0]]),np.concatenate([c1[1],c2[1]])),**ax}
 wde=np.unique(np.concatenate([np.concatenate(v) for v in wdf.values()]));base=len(names);wmap=np.full(int(wde.max())+1,-1,np.int64);wmap[wde]=base+np.arange(len(wde))
 lo=np.load(st+"/e_lo.npy",mmap_mode="r");lab=np.memmap(st+"/e_lab.bin",dtype=np.uint8,mode="r");nlift=0
 for e in wde.tolist():
  t=bytes(lab[lo[e]:lo[e+1]]).decode("utf-8","replace");names.append("wd:"+t);src.append(2);j=w2n.get(whash(t.lower().replace(" ","_")),-1);lift.append(j);nlift+=j>=0
 for r,(h,t) in wdf.items():facts[r]=(wmap[h],wmap[t])
 H,T,R,SP=[],[],[],[]
 for ri,r in enumerate(RELS):
  h,t=facts[r];k=np.unique(np.stack([h,t],1)[h!=t],axis=0);k=k[rng.permutation(len(k))];n=len(k);nt=int(np.ceil(n*0.1));nd=int(np.ceil((n-nt)*0.05));s=np.zeros(n,np.int8);s[:nt]=0 if A2B else 2;s[nt:nt+nd]=0 if A2B else 1
  H.append(k[:,0]);T.append(k[:,1]);R.append(np.full(n,ri,np.int8));SP.append(s);A2B or print(f"{r} facts={n} test={nt} dev={nd} heads={len(np.unique(k[:,0]))} tails={len(np.unique(k[:,1]))}",flush=True)
 H,T,R,SP=map(np.concatenate,(H,T,R,SP));N=len(names);A2B and dsplit(H,T,R,SP,N,rng)
 xyz=np.zeros((N,3),np.float32);li=np.array(lift);m=(li>=0)&(spx is not None);xyz[m]=spx[li[m]] if spx is not None else 0
 np.savez_compressed(OUT,h=H,t=T,r=R,split=SP,rels=np.array(RELS),rtype=RTYPE,names=np.array(names,dtype=object),src=np.array(src,np.int8),lift=li,xyz=xyz,n_sphere=np.int64(Nsp),wd_qidx=wde,nscored=np.int64(NSC),has_xyz=np.int64(spx is not None))
 print(json.dumps({"entities":N,"sphere":Nsp,"cn_new":int(sum(1 for x in src if x==1)),"wd":len(wde),"wd_lifted":int(nlift),"facts":int(len(H)),"test":int((SP==2).sum()),"dev":int((SP==1).sum()),"sec":round(time.time()-t0,1)}),flush=True)
if __name__=="__main__":main()
