import os,time,json,numpy as np
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");OUT=os.path.join(ROOT,"data","torus","a2_data.npz");SEED=17
RELS=["wn:hypernym","wn:part_holonym","wn:member_holonym","cn:UsedFor","cn:AtLocation","cn:CapableOf","wd:P19","wd:P17","wd:P131","wd:P171"];RTYPE=np.array([0,0,0,1,1,1,2,2,2,2],np.int8)
WNT={"wn:hypernym":1,"wn:part_holonym":4,"wn:member_holonym":6};CNR={"cn:UsedFor":1,"cn:AtLocation":6,"cn:CapableOf":2}
def whash(w):
 h=0
 for c in w.encode():h=(h*83492791+c+1)&0xFFFFFFFF
 return h
def main():
 t0=time.time();rng=np.random.default_rng(SEED);sp=np.load(os.path.join(ROOT,"data","concepts","sphere.npz"),allow_pickle=True);spx=np.load(os.environ.get("SPHERE_NPZ",os.path.join(ROOT,"data","torus","sphere_noleak.npz")))["xyz"];wn=np.load(os.path.join(ROOT,"data","concepts","wordnet_graph.npz"),allow_pickle=True)
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
 sel=np.nonzero((fk==1)&np.isin(fp,[19,17,131,171]))[0];S=fs[sel].astype(np.int64);P=fp[sel];V=fv[sel].astype(np.int64)
 def sub(p,subj):m=(P==p)&np.isin(S,subj);return S[m],V[m]
 per=np.unique(S[P==19]);per=rng.choice(per,150000,replace=False);a=sub(19,per);r1=sub(131,np.unique(a[1]));r2=sub(131,np.unique(r1[1]))
 pl=np.unique(np.concatenate([a[1],r1[1],r2[1]]));b=sub(17,pl);tx=np.unique(S[P==171]);tx=rng.choice(tx,100000,replace=False);c1=sub(171,tx);c2=sub(171,np.unique(c1[1]))
 wdf={"wd:P19":a,"wd:P17":b,"wd:P131":(np.concatenate([r1[0],r2[0]]),np.concatenate([r1[1],r2[1]])),"wd:P171":(np.concatenate([c1[0],c2[0]]),np.concatenate([c1[1],c2[1]]))}
 wde=np.unique(np.concatenate([np.concatenate(v) for v in wdf.values()]));base=len(names);wmap=np.full(int(wde.max())+1,-1,np.int64);wmap[wde]=base+np.arange(len(wde))
 lo=np.load(st+"/e_lo.npy",mmap_mode="r");lab=np.memmap(st+"/e_lab.bin",dtype=np.uint8,mode="r");nlift=0
 for e in wde.tolist():
  t=bytes(lab[lo[e]:lo[e+1]]).decode("utf-8","replace");names.append("wd:"+t);src.append(2);j=w2n.get(whash(t.lower().replace(" ","_")),-1);lift.append(j);nlift+=j>=0
 for r,(h,t) in wdf.items():facts[r]=(wmap[h],wmap[t])
 H,T,R,SP=[],[],[],[]
 for ri,r in enumerate(RELS):
  h,t=facts[r];k=np.unique(np.stack([h,t],1)[h!=t],axis=0);k=k[rng.permutation(len(k))];n=len(k);nt=int(np.ceil(n*0.1));nd=int(np.ceil((n-nt)*0.05));s=np.zeros(n,np.int8);s[:nt]=2;s[nt:nt+nd]=1
  H.append(k[:,0]);T.append(k[:,1]);R.append(np.full(n,ri,np.int8));SP.append(s);print(f"{r} facts={n} test={nt} dev={nd} heads={len(np.unique(k[:,0]))} tails={len(np.unique(k[:,1]))}",flush=True)
 H,T,R,SP=map(np.concatenate,(H,T,R,SP));N=len(names)
 xyz=np.zeros((N,3),np.float32);li=np.array(lift);m=li>=0;xyz[m]=spx[li[m]]
 np.savez_compressed(OUT,h=H,t=T,r=R,split=SP,rels=np.array(RELS),rtype=RTYPE,names=np.array(names,dtype=object),src=np.array(src,np.int8),lift=li,xyz=xyz,n_sphere=np.int64(Nsp),wd_qidx=wde)
 print(json.dumps({"entities":N,"sphere":Nsp,"cn_new":int(sum(1 for x in src if x==1)),"wd":len(wde),"wd_lifted":int(nlift),"facts":int(len(H)),"test":int((SP==2).sum()),"dev":int((SP==1).sum()),"sec":round(time.time()-t0,1)}),flush=True)
if __name__=="__main__":main()
