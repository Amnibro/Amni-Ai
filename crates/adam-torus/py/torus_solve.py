import os,time,json,argparse,numpy as np,taichi as ti
from collections import defaultdict
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");TD0=os.path.join(ROOT,"data","torus")
ap=argparse.ArgumentParser()
for k,v in dict(mode="dev",D=64,Ds=-1,sweeps=30,S="lin",lam=16,rho=500,wrel=1,seed=17,tag="run",nq=2000,tau="64",recall=1,a3=0,save=0,threads=3,fit=0,lift=1,tm="lut",sphere=0,cav=1,warm=0,prior=0,bidir=0,data="a2_data.npz",sph="",mix=1,bounce="",bk=4,bw=0,lvl=0,calib=0,load="",ebins=10,export=0,pw="").items():ap.add_argument("--"+k,type=type(v),default=v)
A=ap.parse_args();D=A.D;DS=D if A.Ds<0 else A.Ds;HY=DS<D;TD=os.path.dirname(os.path.join(TD0,A.data));SPHP=A.sph or os.path.join(TD,"sphere_noleak.npz");KM=A.mix;MK=17**KM;P17=[1,17,289][:KM];TAU=float(A.tau);PW=[int(x) for x in A.pw.split(",")] if A.pw else []
assert KM==1 or (A.bidir==0 and A.tm=="lut" and DS%4==0 and D%4==0)
SP={"lin":[8-min(v,17-v) for v in range(17)],"sharp":[[16,9,4,1,0,0,0,0,0][min(v,17-v)] for v in range(17)],"mid":[[12,10,7,4,2,1,0,0,0][min(v,17-v)] for v in range(17)]}[A.S];Sa=np.array(SP,np.int64)
COMP=[("c1 P19>P17",6,7),("c2 P19>P131",6,8),("c3 hyp>hyp",0,0),("c4 P171>P171",9,9)]
X17=np.arange(17)
def log(*a):print(*a,flush=True)
def lift(xyz,nd,seed):
 r=np.random.default_rng(seed);out=np.zeros((len(xyz),nd),np.int32)
 for j in range(nd):q,_=np.linalg.qr(r.normal(size=(3,3)));p=xyz@q;out[:,j]=np.floor((np.arctan2(p[:,1],p[:,0])+np.pi)/(2*np.pi)*17).astype(np.int32)%17
 return out
def csr(n,src,dst,rel,role):
 o=np.argsort(src,kind="stable");p=np.zeros(n+1,np.int64);np.add.at(p,src+1,1);return np.cumsum(p).astype(np.int32),dst[o].astype(np.int32),rel[o].astype(np.int32),role[o].astype(np.int32)
def auc(p,n):
 a=np.concatenate([p,n]).astype(np.float64);o=np.argsort(a,kind="stable");sa=a[o];rk=np.empty(len(a));i=0
 while i<len(a):
  j=i
  while j+1<len(a) and sa[j+1]==sa[i]:j+=1
  rk[o[i:j+1]]=(i+j)/2+1;i=j+1
 return float((rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
def hits(sc,true_idx,filt,nc):
 g=np.zeros(len(true_idx));e=np.zeros(len(true_idx))
 for i in range(len(true_idx)):row=sc if sc.ndim==1 else sc[i];st=row[true_idx[i]];m=np.zeros(len(row),bool);m[:nc]=True;m[filt[i]]=False;m[true_idx[i]]=False;g[i]=(row[m]>st).sum();e[i]=(row[m]==st).sum()
 return np.clip((10-g)/(e+1),0,1)
def ece(p,y,nb):b=np.minimum((p*nb).astype(np.int64),nb-1);n=np.bincount(b,minlength=nb);return float(sum(abs(p[b==i].mean()-y[b==i].mean())*n[i] for i in range(nb) if n[i])/max(len(p),1))
def pav(y,w):
 v,ww,c=[],[],[]
 for a,b in zip(y.tolist(),w.tolist()):
  v.append(a);ww.append(b);c.append(1)
  while len(v)>1 and v[-2]>v[-1]:v[-2]=(v[-2]*ww[-2]+v[-1]*ww[-1])/(ww[-2]+ww[-1]);ww[-2]+=ww[-1];c[-2]+=c[-1];v.pop();ww.pop();c.pop()
 return np.repeat(v,c)
def calfit(sp,sn,nb):
 a=np.concatenate([sp,sn]);y=np.concatenate([np.ones(len(sp)),np.zeros(len(sn))]);ed=np.unique(np.quantile(a,np.arange(1,nb)/nb,method="lower").astype(np.int64));b=np.searchsorted(ed,a,side="right");k=len(ed)+1;ps=np.bincount(b,weights=y,minlength=k);n=np.bincount(b,minlength=k)
 return ed,np.round(288*pav((ps+1)/(n+2),n+1)).astype(np.int64)
def calap(s,c):return c[1][np.searchsorted(c[0],s,side="right")]/288
def nbc(a,b,L):return (a>=0)&(b>=0)&(a//(L*L)==b//(L*L))&(np.abs((a//L)%L-(b//L)%L)<=1)&(np.abs(a%L-b%L)<=1)
def recall_eval(top,NS,gc,gC):
 res={}
 for gname,g in (("cur",gc),("C",gC)):
  r2=np.random.default_rng(8);nb=defaultdict(list)
  for a,b in zip(g["edge_src"].tolist(),g["edge_dst"].tolist()):nb[a].append(b);nb[b].append(a)
  cand=np.array([i for i in range(NS) if nb[i]]);samp=r2.choice(cand,2000,replace=False)
  def rec(ids,nbs):
   h=t=0.0
   for b0 in range(0,len(ids),200):
    sc=top(np.array(ids[b0:b0+200]))
    for i0,row in enumerate(sc):
     i=b0+i0;s20=np.partition(row,-20)[-20];gg=(row>s20).sum();ee=(row==s20).sum();ns=[j for j in dict.fromkeys(nbs[i]) if j!=ids[i]]
     h+=sum(1.0 if row[j]>s20 else ((20-gg)/ee if row[j]==s20 else 0.0) for j in ns);t+=len(ns)
   return h/max(t,1),int(t)
  ra=rec(samp.tolist(),[nb[int(i)] for i in samp])
  for i in samp:r2.integers(0,NS,size=max(len(set(nb[int(i)])-{int(i)})*3,8))
  out={"recall20":round(ra[0],4),"n":ra[1]}
  for nm,(s_,d_) in (("wn",(g["wn_src"],g["wn_dst"])),("hyp",(g["hyp_src"],g["hyp_dst"]))):
   nb2=defaultdict(set)
   for a,b in zip(s_.tolist(),d_.tolist()):nb2[a].add(b);nb2[b].add(a)
   c=[i for i in range(NS) if nb2[i]];s2=r2.choice(c,min(1000,len(c)),replace=False);x=rec(s2.tolist(),[list(nb2[int(i)]) for i in s2]);out[nm]=round(x[0],4);out[nm+"_n"]=x[1]
  res[gname]=out;log("recall",gname,out)
 return res
def main():
 t0=time.time();gc=np.load(os.path.join(TD0,"sphere_graph_cur.npz"));gC=np.load(os.path.join(TD0,"sphere_graph_C.npz"))
 if A.sphere:
  sp=np.load(A.sph or os.path.join(ROOT,"data","concepts","sphere.npz"));P=sp["xyz"].astype(np.float32);NS=len(P)
  def top(ids):s=np.einsum("qk,nk->qn",P[ids],P);s[np.arange(len(ids)),ids]=-9;return s
  r=recall_eval(top,NS,gc,gC);json.dump({"sphere_recall":r,"sphere":A.sph or "live"},open(os.path.join(TD,f"res_{A.tag}.json"),"w"),indent=1);return
 d=np.load(os.path.join(TD0,A.data),allow_pickle=True);H,T,R,SPL=d["h"],d["t"],d["r"],d["split"];RT=d["rtype"];N=len(d["names"]);NS=int(d["n_sphere"]);NF=len(d["rels"]);NSC=int(d["nscored"]) if "nscored" in d.files else NF;REL=NF;NREL=NF+1+NF*A.bidir;rtype=np.concatenate([RT,[0],RT[:NF*A.bidir]]);RR=[r for r in range(NREL) if r!=REL];RV=lambda r:r+NF+1
 tr=SPL==0 if A.mode in ("dev","test") else SPL<2;ev=SPL==1 if A.mode=="dev" else SPL==2
 ge=np.concatenate([np.stack([gc["edge_src"],gc["edge_dst"]],1),np.stack([gC["edge_src"],gC["edge_dst"]],1)]).astype(np.int64);ge=np.sort(ge,1);ge=np.unique(ge[ge[:,0]!=ge[:,1]],axis=0)
 hold=np.sort(np.stack([H[~tr],T[~tr]],1),1);hk=np.unique(hold[:,0]*N+hold[:,1]);ge=ge[~np.isin(ge[:,0]*N+ge[:,1],hk)]
 gh,gt,trr=np.concatenate([H[tr],ge[:,0]]+([T[tr]] if A.bidir else [])),np.concatenate([T[tr],ge[:,1]]+([H[tr]] if A.bidir else [])),np.concatenate([R[tr].astype(np.int64),np.full(len(ge),REL)]+([R[tr].astype(np.int64)+NF+1] if A.bidir else []))
 log(f"mode={A.mode} data={A.data} D={D} Ds={DS} S={A.S} tm={A.tm} mix={KM} bounce={A.bounce}/{A.bk}/{A.bw} lvl={A.lvl} calib={A.calib} load={A.load} bidir={A.bidir} prior={A.prior} cav={A.cav} warm={A.warm} sweeps={A.sweeps} lam={A.lam} rho={A.rho} wrel={A.wrel} lift={A.lift} train_facts={int(tr.sum())} related_edges={len(ge)} eval={int(ev.sum())}")
 ty=rtype[trr];NR=N;trow={}
 if HY:
  for k in range(3):m=ty==k;ents=np.unique(np.concatenate([gh[m],gt[m]]));mp=np.full(N,-1,np.int64);mp[ents]=NR+np.arange(len(ents));trow[k]=mp;NR+=len(ents)
 li=d["lift"];SPH=np.load(SPHP) if os.path.exists(SPHP) else None;xyz=d["xyz"].copy();has=(li>=0)&(A.lift>0);rng=np.random.default_rng(A.seed)
 if "has_xyz" in d.files and not int(d["has_xyz"]):m=li>=0;xyz[m]=SPH["xyz"][li[m]]
 X0=rng.integers(0,17,size=(NR,D)).astype(np.int32);L=lift(xyz,D,A.seed);X0[:N][has]=L[has]
 if HY:
  for k in range(3):m=trow[k]>=0;X0[trow[k][m]]=X0[np.nonzero(m)[0]]
 ti.init(arch=ti.cpu,cpu_max_num_threads=A.threads,random_seed=A.seed,default_ip=ti.i32,offline_cache=False,log_level=ti.ERROR)
 X=ti.field(ti.i32,shape=(NR,D));Xn=ti.field(ti.i32,shape=(NR,D));TF=ti.field(ti.i32,shape=(NREL,D,MK));Sf=ti.field(ti.i32,shape=17);Wf=ti.field(ti.i32,shape=NREL)
 PCC=ti.field(ti.i64,shape=(NREL,D,17));NGT=ti.field(ti.i64,shape=(NREL,D,17));ID=np.tile(np.arange(MK)%17,(NREL,D,1)).astype(np.int32)
 X.from_numpy(X0);Sf.from_numpy(Sa.astype(np.int32));Wf.from_numpy(np.array([A.wrel if r==REL else 1 for r in range(NREL)],np.int32));TFa=ID.copy();TF.from_numpy(TFa)
 groups=[]
 def mk(e_h,e_t,rel,lo,hi):
  src=np.concatenate([e_h,e_t]);dst=np.concatenate([e_t,e_h]);rr=np.concatenate([rel,rel]);ro=np.concatenate([np.zeros(len(e_h),np.int64),np.ones(len(e_t),np.int64)]);ro[np.concatenate([rel,rel])==REL]=0
  arrs=csr(NR,src,dst,rr,ro);dg=np.diff(arrs[0]).astype(np.int64);cvw=np.where(dg>0,256-(256*A.cav)//np.maximum(dg,1),256).astype(np.int32);arrs=arrs+(cvw,);f=[ti.field(ti.i32,shape=len(x)) for x in arrs];[a.from_numpy(b) for a,b in zip(f,arrs)]
  cnt=np.zeros((NR,NREL),np.int64);np.add.at(cnt,(e_h,rel),1);np.add.at(cnt,(e_t[rel==REL],rel[rel==REL]),1);cw=np.zeros((NR,NREL),np.int64)
  for r in range(NREL):
   m=rel==r
   if m.any():c=np.unique(e_t[m]) if r!=REL else np.unique(np.concatenate([e_h[m],e_t[m]]));cw[c,r]=max(1,int(round(16*m.sum()*(2 if r==REL else 1)/len(c))))
  nz=(cnt>0)|(cw>0);rp=np.zeros(NR+1,np.int64);rp[1:]=np.cumsum(nz.sum(1));ri,rc=np.nonzero(nz);ga=[rp,rc,cnt[ri,rc],cw[ri,rc]]
  g=[ti.field(ti.i32,shape=max(1,len(x))) for x in ga];[a.from_numpy(b.astype(np.int32)) for a,b in zip(g,ga) if len(b)];groups.append((f,g,lo,hi))
 mk(gh,gt,trr,0,DS)
 if HY:mk(np.concatenate([trow[k][gh[ty==k]] for k in range(3)]),np.concatenate([trow[k][gt[ty==k]] for k in range(3)]),np.concatenate([trr[ty==k] for k in range(3)]),DS,D)
 @ti.kernel
 def icm(p:ti.template(),o:ti.template(),rr:ti.template(),rl:ti.template(),cv:ti.template(),rp:ti.template(),rc:ti.template(),rh:ti.template(),rw:ti.template(),d0:ti.i32,d1:ti.i32,lam:ti.i64,rho:ti.i32,seed:ti.i32):
  for e,dd in ti.ndrange(NR,d1-d0):
   d=d0+dd;hsh=(ti.cast(e,ti.u32)*ti.u32(2654435761)+ti.cast(d,ti.u32)*ti.u32(40503)+ti.cast(seed,ti.u32)*ti.u32(2246822519))>>ti.u32(7)
   Xn[e,d]=X[e,d]
   if ti.cast(hsh%ti.u32(1000),ti.i32)<rho and p[e+1]>p[e]:
    g=ti.Vector.zero(ti.i64,17);xe=X[e,d]
    for k in range(p[e],p[e+1]):
     q=o[k];r=rr[k];w=ti.cast(Wf[r]*cv[q],ti.i64)
     if rl[k]==1:
      ix=0
      for s in ti.static(range(KM)):ix+=P17[s]*X[q,d^s]
      u=TF[r,d,ix]
      for v in ti.static(range(17)):g[v]+=w*Sf[(u-v+17)%17]
     else:
      for s in ti.static(range(KM)):
       oo=d^s;ix=-P17[s]*xe;y=X[q,oo]
       for s2 in ti.static(range(KM)):ix+=P17[s2]*X[e,oo^s2]
       for v in ti.static(range(17)):g[v]+=w*Sf[(TF[r,oo,ix+P17[s]*v]-y+17)%17]
    for k in range(rp[e],rp[e+1]):
     r=rc[k];a=ti.cast(rh[k],ti.i64)*16;b=ti.cast(rw[k],ti.i64);hv=ti.Vector.zero(ti.i64,17)
     for s in ti.static(range(KM)):
      oo=d^s;ix=-P17[s]*xe
      for s2 in ti.static(range(KM)):ix+=P17[s2]*X[e,oo^s2]
      for v in ti.static(range(17)):hv[v]+=a*PCC[r,oo,TF[r,oo,ix+P17[s]*v]]
     for v in ti.static(range(17)):g[v]-=lam*(hv[v]+b*NGT[r,d,v])//256
    cur=X[e,d];bg=g[0]
    for v in ti.static(range(17)):
     if v==cur:bg=g[v]
    bv=cur
    for v in ti.static(range(17)):
     if g[v]>bg:bg=g[v];bv=v
    Xn[e,d]=bv
 @ti.kernel
 def cp():
  for e,dd in X:X[e,dd]=Xn[e,dd]
 def ends(r):
  m=trr==r;h,t=gh[m],gt[m];k=rtype[r]
  return [(h,t,0,DS)]+([(trow[k][h],trow[k][t],DS,D)] if HY else [])
 def conv(p):return np.stack([(p*Sa[(y-X17)%17]).sum(-1) for y in range(17)],-1)
 ixf=lambda Xa,e,lo,hi:sum(P17[s]*Xa[e][:,np.arange(lo,hi)^s] for s in range(KM))
 def fitT(Xa):
  T=ID.copy()
  for r in RR:
   for h,t,lo,hi in ends(r):
    nd=hi-lo;J=np.bincount((np.arange(nd)*MK*17+ixf(Xa,h,lo,hi)*17+Xa[t,lo:hi]).ravel(),minlength=nd*MK*17).reshape(nd,MK,17).astype(np.float64)
    c=np.unique(t);pc=np.stack([np.bincount(Xa[c,lo+j],minlength=17) for j in range(nd)])/max(len(c),1)
    G=16*conv(J)-A.lam*J.sum(2)[:,:,None]*conv(pc)[:,None,:];G=G if KM==1 else (lambda g:(g+g.sum(1,keepdims=True)/(MK//17)).reshape(nd,MK,17))(G.reshape(nd,MK//17,17,17))
    if A.tm=="lut":T[r,lo:hi]=G.argmax(2)
    elif A.tm=="trans":b=np.stack([G[:,X17,(X17+bb)%17].sum(1) for bb in range(17)],1).argmax(1);T[r,lo:hi]=(X17[None,:]+b[:,None])%17
    else:sc=np.stack([np.stack([G[:,X17,(aa*X17+bb)%17].sum(1) for bb in range(17)],1) for aa in range(17)],1).reshape(nd,289).argmax(1);T[r,lo:hi]=((sc//17)[:,None]*X17[None,:]+(sc%17)[:,None])%17
  return T
 def updNG(Xa,T):
  pcc=np.zeros((NREL,D,17),np.int64);ngt=np.zeros((NREL,D,17),np.int64)
  for r in range(NREL):
   for h,t,lo,hi in ends(r) if r!=REL else [(gh[trr==r],gt[trr==r],0,DS)]+([(trow[0][gh[trr==r]],trow[0][gt[trr==r]],DS,D)] if HY else []):
    nd=hi-lo;c=np.unique(t) if r!=REL else np.unique(np.concatenate([h,t]));pc=np.stack([np.bincount(Xa[c,lo+j],minlength=17) for j in range(nd)])/max(len(c),1)
    lnd=T[r][np.arange(lo,hi)[None,:],ixf(Xa,h,lo,hi)];la=np.stack([np.bincount(lnd[:,j],minlength=17) for j in range(nd)])/max(len(h),1)
    pcc[r,lo:hi]=np.round(256*conv(pc)).astype(np.int64);ngt[r,lo:hi]=np.round(256*conv(la)).astype(np.int64)
  PCC.from_numpy(pcc);NGT.from_numpy(ngt)
 if A.load:
  ml=np.load(os.path.join(TD,f"torus_{A.load}.npz"));assert ml["TF"].shape==TFa.shape and ml["X"].shape==(NR,D);X.from_numpy(ml["X"].astype(np.int32));TFa=ml["TF"].astype(np.int32)
 else:
  for s in range(A.sweeps):
   Xa=X.to_numpy();TFa=fitT(Xa) if s>=A.warm else ID.copy();TF.from_numpy(TFa);updNG(Xa,TFa)
   for f,g,lo,hi in groups:icm(*f,*g,lo,hi,A.lam,A.rho,A.seed*1000+s)
   cp();ch=int((X.to_numpy()!=Xa).sum());(s%5==4 or s==0) and log(f"sweep {s+1}/{A.sweeps} changed={ch} ({ch/(NR*D):.4f}) t={time.time()-t0:.0f}s")
  TFa=fitT(X.to_numpy().astype(np.int32))
 Xa=X.to_numpy().astype(np.int32);TF.from_numpy(TFa)
 res={"cfg":vars(A),"rows":int(NR),"entities":int(N),"stored_digits":int(N*DS+(NR-N)*(D-DS)) if HY else int(N*(D+1)),"transform_digits":int(TFa[RR].size),"digits_per_query":D,"solve_s":round(time.time()-t0,1)}
 rq=np.random.default_rng(1);ia=rq.integers(0,len(ge),20000);r1,r2=rq.integers(0,N,20000),rq.integers(0,N,20000);sx=lambda p,q:Sa[(Xa[p,:DS]-Xa[q,:DS])%17].sum(1);res["collapse"]={"dim_max_share":round(float(np.mean([np.bincount(Xa[:N,j],minlength=17).max()/N for j in range(DS)])),4),"related_pair_score":round(float(sx(ge[ia,0],ge[ia,1]).mean()),1),"random_pair_score":round(float(sx(r1,r2).mean()),1),"distinct_rows":int(len(np.unique(Xa[:N,:DS],axis=0)))};log("collapse",res["collapse"])
 if A.save:np.savez_compressed(os.path.join(TD,f"torus_{A.tag}.npz"),X=Xa.astype(np.uint8),TF=TFa.astype(np.uint8),S=Sa.astype(np.uint8),Ds=np.int64(DS),trow=np.stack([trow[k] for k in range(3)]) if HY else np.zeros(0),mix=np.int64(KM))
 crow=lambda c,k:(c,np.where(trow[k][c]>=0,trow[k][c],c) if HY else c)
 @ti.kernel
 def score(land:ti.types.ndarray(),hx:ti.types.ndarray(),ri:ti.types.ndarray(),c0:ti.types.ndarray(),c1:ti.types.ndarray(),bl:ti.types.ndarray(),nb:ti.types.ndarray(),out:ti.types.ndarray(),bo:ti.types.ndarray()):
  for q,c in ti.ndrange(out.shape[0],out.shape[1]):
   s=0;u=0
   for dd in range(D):
    row=ti.cast(c0[c],ti.i32)
    if dd>=DS:row=ti.cast(c1[c],ti.i32)
    xc=X[row,dd];s+=Sf[(land[q,dd]-xc+17)%17]
    if ri.shape[0]>0:
     y=xc
     for j in range(ri.shape[0]):y=TF[ri[j],dd,y]
     s+=Sf[(y-hx[q,dd]+17)%17]
    for b in range(nb[q]):u+=Sf[(bl[q,b,dd]-xc+17)%17]
   out[q,c]=s;bo[q,c]=u
 def lands(h,rs,k):
  l=np.zeros((len(h),D),np.int32);l[:,:DS]=Xa[h,:DS]
  if HY:l[:,DS:]=Xa[np.where(trow[k][h]>=0,trow[k][h],h),DS:]
  for r in rs:l=TFa[r][np.arange(D)[None,:],sum(P17[s]*l[:,np.arange(D)^s] for s in range(KM))]
  return l.astype(np.int32)
 rinv=lambda rs:np.array([RV(r) for r in rs[::-1]] if A.bidir else [],np.int32)
 CL={L:np.where(li>=0,SPH[f"cell_id_{L}"][np.maximum(li,0)],-1).astype(np.int64) for L in (64,256,1024)} if SPH is not None else None
 if SPH is not None and "tan" in A.bounce:
  ts=SPH["tangent_src"].astype(np.int64);tc=(SPH["tangent_dst_gf"].astype(np.int64)*17**np.arange(6)).sum(1);c1n=SPH["cell_id_1024"].astype(np.int64);oc=np.argsort(c1n,kind="stable");scn=c1n[oc];a_=np.searchsorted(scn,tc);cn_=np.searchsorted(scn,tc,side="right")-a_
  rows=np.repeat(np.arange(len(ts)),cn_);tj=oc[np.repeat(a_,cn_)+np.arange(cn_.sum())-np.repeat(np.cumsum(cn_)-cn_,cn_)];tsr=ts[rows];bad=np.zeros(len(ts),bool);np.logical_or.at(bad,rows,np.isin(np.minimum(tsr,tj)*N+np.maximum(tsr,tj),hk))
  kp=~bad[rows]&(tj!=tsr);ks=tsr[kp];kj=tj[kp];TP=np.zeros(len(SPH["xyz"])+1,np.int64);np.add.at(TP,ks+1,1);TP=np.cumsum(TP);TJ=kj[np.argsort(ks,kind="stable")];res["tangents"]={"links":int(len(ts)),"dropped_heldout":int(bad.sum()),"bounce_nodes":int(len(TJ))};log("tangents",res["tangents"])
 if "nbr" in A.bounce:
  ao=np.argsort(np.concatenate([gh,gt]),kind="stable");AJ=np.concatenate([gt,gh])[ao];AP=np.zeros(N+1,np.int64);np.add.at(AP,np.concatenate([gh,gt])+1,1);AP=np.cumsum(AP)
 def bnc(hs):
  out=[]
  for h in hs.tolist():
   L=[]
   if "tan" in A.bounce and li[h]>=0:L+=TJ[TP[li[h]]:TP[li[h]+1]][:A.bk].tolist()
   if "nbr" in A.bounce:
    nb=np.unique(AJ[AP[h]:AP[h+1]]);nb=nb[nb!=h]
    if len(nb):L+=nb[np.argsort(-Sa[(Xa[h,:DS][None,:]-Xa[nb,:DS])%17].sum(1),kind="stable")[:A.bk]].tolist()
   out.append([x for x in dict.fromkeys(L) if x!=h])
  bid=np.full((len(hs),max([len(x) for x in out]+[1])),-1,np.int64)
  for i,x in enumerate(out):bid[i,:len(x)]=x
  return bid,np.array([len(x) for x in out],np.int32)
 LCt={}
 for r in range(NSC if CL is not None else 0):
  m=trr==r;ch,ct=CL[64][gh[m]],CL[64][gt[m]];v=(ch>=0)&(ct>=0);u,c=np.unique(ch[v]*24576+ct[v],return_counts=True);kh=u//24576;o=np.lexsort((-c,kh));kh,kt=kh[o],(u%24576)[o];f=np.r_[True,kh[1:]!=kh[:-1]] if len(kh) else np.zeros(0,bool);LCt[r]=(kh[f],kt[f])
 def lvb(h,c,r,pair=False):
  ga=lambda L:CL[L][h] if pair else CL[L][h][:,None];gb=lambda L:CL[L][c] if pair else CL[L][c][None,:];a64,b64=ga(64),gb(64);kh,kt=LCt[r];ah=CL[64][h];land=np.full(len(h),-1,np.int64)
  if len(kh):j=np.clip(np.searchsorted(kh,ah),0,len(kh)-1);land=np.where((ah>=0)&(kh[j]==ah),kt[j],-1)
  land=land if pair else land[:,None];lv=np.where((a64<0)|(b64<0),4,np.where(nbc(ga(1024),gb(1024),1024),3,np.where(nbc(ga(256),gb(256),256),2,np.where(nbc(a64,b64,64),1,0))))
  return (lv*3+np.where((land<0)|(b64<0),2,nbc(land,b64,64).astype(np.int64))).astype(np.int8)
 def wfit(lp,ln):cp_,cn_=np.bincount(lp,minlength=15),np.bincount(ln,minlength=15);return np.round(A.lvl*np.log2(((cp_+1)/(cp_.sum()+15))/((cn_+1)/(cn_.sum()+15)))).astype(np.int64)
 allf=defaultdict(set)
 for h,t,r in zip(H.tolist(),T.tolist(),R.tolist()):allf[(h,r)].add(t)
 def evalq(qh,qt,rs,cand,freq,filt_sets,k,rng,pop=None):
  qh,qt=np.asarray(qh),np.asarray(qt);ci={c:i for i,c in enumerate(cand.tolist())};keep=np.array([t in ci for t in qt.tolist()]);n=len(qh);nc=len(cand);pop=np.zeros(nc,np.int32) if pop is None else pop
  ca=np.concatenate([cand,np.unique(qt[~keep])]).astype(np.int64);cia={c:i for i,c in enumerate(ca.tolist())};pa=np.zeros(len(ca),np.int64);pa[:nc]=pop;c0,c1=crow(ca,k);c0,c1=np.asarray(c0,np.int64),np.asarray(c1,np.int64)
  bid,nb=bnc(qh) if A.bounce else (np.full((n,1),-1,np.int64),np.zeros(n,np.int32));lv=A.lvl>0 and len(rs)==1 and CL is not None;B=np.zeros((n,len(ca)),np.int32);Lb=np.zeros((n,len(ca)),np.int8) if lv else None
  for s0 in range(0,n,256):
   s1=min(n,s0+256);o=np.zeros((s1-s0,len(ca)),np.int32);bo=np.zeros_like(o);bl=np.zeros((s1-s0,bid.shape[1],D),np.int32);v=bid[s0:s1]>=0
   if v.any():bl[v]=lands(bid[s0:s1][v],rs,k)
   score(lands(qh[s0:s1],rs,k),lands(qh[s0:s1],[],k),rinv(rs),c0,c1,bl,np.ascontiguousarray(nb[s0:s1]),o,bo);nbs=nb[s0:s1,None].astype(np.int64)
   B[s0:s1]=o+np.where(nbs>0,(A.bw*bo.astype(np.int64))//(4*np.maximum(nbs,1)),(A.bw*o.astype(np.int64))//4)+(PW[rs[0]] if len(rs)==1 and rs[0]<len(PW) else A.prior)*pa[None,:]
   if lv:Lb[s0:s1]=lvb(qh[s0:s1],ca,rs[0])
  ti_=np.array([cia[t] for t in qt.tolist()]);filt=[[ci[x] for x in fs if x in ci] for fs in filt_sets]
  neg=np.array([rng.choice(np.setdiff1d(np.arange(nc),np.array(f+[ti_[i]],np.int64))) if nc>len(f)+1 else 0 for i,f in enumerate(filt)])
  hb=hits(np.concatenate([freq,np.zeros(len(ca)-nc)]),ti_,filt,nc);hb[~keep]=0
  return dict(n=n,keep=keep,B=B,Lb=Lb,ti=ti_,filt=filt,neg=neg,nc=nc,hb=hb,nbm=float(nb.mean()),lp=lvb(qh,qt,rs[0],True) if lv else None,ln=lvb(qh,ca[neg],rs[0],True) if lv else None)
 def fin(q,ix,W=None,cal=None):
  tot=q["B"][ix].astype(np.int64)+(W[q["Lb"][ix]] if W is not None else 0);n=len(ix);ar=np.arange(n);ti_=q["ti"][ix];neg=q["neg"][ix];keep=q["keep"][ix];nc=q["nc"]
  ht=hits(tot,ti_,[q["filt"][i] for i in ix],nc);ht[~keep]=0;sp=tot[ar,ti_];sn=tot[ar,neg]
  if cal is not None:pp,pn=calap(sp,cal),calap(sn,cal)
  else:
   K=min(32,nc);cs=tot[:,:nc];top=np.argpartition(-cs,K-1,axis=1)[:,:K];smax=cs.max(1);mp=np.maximum(smax,sp);lut=np.round(65535*np.exp(-np.arange(int(mp.max()-min(cs.min(),sp.min()))+2)/TAU)).astype(np.int64);ts=cs[ar[:,None],top];et=lut[smax[:,None]-ts].sum(1);ep=lut[mp[:,None]-ts].sum(1)
   ins_p=keep&(top==ti_[:,None]).any(1);ins_n=(top==neg[:,None]).any(1);zp=lut[mp-sp];zn=lut[smax-sn];pp=zp/np.maximum(ep+np.where(ins_p,0,zp),1);pn=zn/np.maximum(et+np.where(ins_n,0,zn),1)
  return dict(ht=ht,hb=q["hb"][ix],sp=sp,sn=sn,pp=pp,pn=pn)
 popd=lambda f:np.minimum(16,np.floor(2*np.log2(1+f))).astype(np.int32)
 seen=np.zeros(N,bool);seen[gh]=True;seen[gt]=True;rng=np.random.default_rng(A.seed+5);rf=np.random.default_rng(A.seed+6);per={};pool=defaultdict(list);luts={};popx={}
 for r in range(NSC):
  m=(tr if A.fit else ev)&(R==r);idx=np.nonzero(m)[0];idx=idx[rng.permutation(len(idx))[:A.nq]];cand=np.unique(T[tr&(R==r)]);ci={c:i for i,c in enumerate(cand.tolist())};fr=np.zeros(len(cand))
  for t in T[tr&(R==r)].tolist():fr[ci[t]]+=1
  popx[r]=(cand,popd(fr));q=evalq(H[idx],T[idx],[r],cand,fr,[allf[(h,r)]-{t} for h,t in zip(H[idx].tolist(),T[idx].tolist())],int(rtype[r]),rng,popd(fr));n=q["n"]
  if A.mode=="test":ii=np.nonzero((SPL==1)&(R==r))[0];ii=ii[rf.permutation(len(ii))[:A.nq]];qf=evalq(H[ii],T[ii],[r],cand,fr,[allf[(h,r)]-{t} for h,t in zip(H[ii].tolist(),T[ii].tolist())],int(rtype[r]),rf,popd(fr));parts=[(np.arange(n),qf,np.arange(qf["n"]))]
  else:hv=np.arange(n)%2;parts=[(np.nonzero(hv==h_)[0],q,np.nonzero(hv!=h_)[0]) for h_ in (0,1)] if (A.lvl or A.calib) else [(np.arange(n),None,None)]
  o={}
  for ix,qf_,fx in parts:
   W=wfit(qf_["lp"][fx],qf_["ln"][fx]) if qf_ is not None and qf_["lp"] is not None else None;cal=(lambda f:calfit(f["sp"],f["sn"],A.calib))(fin(qf_,fx,W)) if A.calib and qf_ is not None else None;f=fin(q,ix,W,cal);luts[r]=(W,cal)
   for k_,v_ in f.items():o.setdefault(k_,np.zeros(n,v_.dtype))[ix]=v_
  o.update(n=n,keep=q["keep"],seen=seen[H[idx]]&seen[T[idx]],nbm=q["nbm"]);per[str(d["rels"][r])]=o;[pool[k_].append(o[k_]) for k_ in ("ht","hb","sp","sn","pp","pn","seen")];del q
 P={k_:np.concatenate(v) for k_,v in pool.items()};sv=P["seen"];yy=lambda o_,m_=slice(None):(np.concatenate([o_["pp"][m_],o_["pn"][m_]]),np.concatenate([np.ones(len(o_["pp"][m_])),np.zeros(len(o_["pn"][m_]))]))
 st=lambda o_,m_=slice(None):{"n":int(len(o_["ht"][m_])),"hits10":round(float(o_["ht"][m_].mean()),4),"base_hits10":round(float(o_["hb"][m_].mean()),4),"ratio":round(float(o_["ht"][m_].mean()/max(o_["hb"][m_].mean(),1e-9)),3),"auc_raw":round(auc(o_["sp"][m_],o_["sn"][m_]),4),"auc_P":round(auc(o_["pp"][m_],o_["pn"][m_]),4),"ece":round(ece(*yy(o_,m_),A.ebins),4)}
 res["pooled"]=st(P);res["pooled_transductive"]=st(P,sv);res["per_relation"]={k_:dict(st(v),transductive_frac=round(float(v["seen"].mean()),3),in_cand=round(float(v["keep"].mean()),3),mean_bounces=round(v["nbm"],2)) for k_,v in per.items()}
 log(json.dumps(res["pooled"]));log("transductive",json.dumps(res["pooled_transductive"]));[log(k_,v) for k_,v in res["per_relation"].items()]
 if A.mode=="test":np.savez(os.path.join(TD,f"luts_{A.tag}.npz"),**{f"W{r}":(w if w is not None else np.zeros(15,np.int64)) for r,(w,c) in luts.items()},**{f"ce{r}":(c[0] if c is not None else np.zeros(0,np.int64)) for r,(w,c) in luts.items()},**{f"cp{r}":(c[1] if c is not None else np.zeros(0,np.int64)) for r,(w,c) in luts.items()},**{f"lk{r}":LCt[r][0] for r in LCt},**{f"lv{r}":LCt[r][1] for r in LCt})
 if A.export:
  ed=os.path.join(TD,f"l2_{A.tag}");os.makedirs(ed,exist_ok=True);dg=lambda v,nd:np.stack([(np.asarray(v,np.int64)//17**k)%17 for k in range(nd)],-1).astype(np.uint8);wl=(np.arange(N)>=NS)&(li>=0);tn=np.bincount(SPH["tangent_src"],minlength=len(SPH["xyz"])) if SPH is not None else np.zeros(0,np.int64)
  secs=[("X",Xa.astype(np.uint8)),("TF",TFa[RR].astype(np.uint8)),("S",Sa.astype(np.uint8)),("pw",dg([PW[r] if r<len(PW) else A.prior for r in range(NSC)],2)),("lift_wd",np.concatenate([dg(np.nonzero(wl)[0],5),dg(li[wl],5)],1))]+([("cell1024",SPH["cell_gf_1024"].astype(np.uint8)),("tan_cnt",dg(tn,1)),("tan_dst",SPH["tangent_dst_gf"].astype(np.uint8))] if SPH is not None else [])
  secs+=[(f"pop{r}",np.concatenate([dg(c,5),dg(p_,1)],1)) for r,(c,p_) in popx.items()]+[(f"lvlW{r}",dg(w+144,2)) for r,(w,c) in luts.items() if w is not None]+[(f"land{r}",np.concatenate([dg(LCt[r][0],4),dg(LCt[r][1],4)],1)) for r in LCt if A.lvl]+[(f"cal{r}",np.concatenate([dg(c[0]+41760,4).ravel(),dg(c[1],2).ravel()])) for r,(w,c) in luts.items() if c is not None]
  [np.save(os.path.join(ed,f"{n}.npy"),np.ascontiguousarray(v)) for n,v in secs];open(os.path.join(ed,"manifest.tsv"),"w").write("".join(f"{n}\t{n}.npy\n" for n,v in secs))
  tot=sum(v.size for n,v in secs);res["layer2_digits"]={"total":int(tot),"per_entity":round(tot/N,3),"positions_per_entity":D,"by_section":{n:int(v.size) for n,v in secs},"train_facts":int(tr.sum())};log("layer2_digits",res["layer2_digits"])
 if A.recall:
  allc=np.arange(NS);c0,c1=crow(allc,0);c0,c1=np.asarray(c0,np.int64),np.asarray(c1,np.int64)
  def top(ids):
   out=np.zeros((len(ids),NS),np.int32);l=np.zeros((len(ids),D),np.int32);l[:,:DS]=Xa[ids,:DS]
   if HY:l[:,DS:]=Xa[np.where(trow[0][ids]>=0,trow[0][ids],ids),DS:]
   score(l,l,np.zeros(0,np.int32),c0,c1,np.zeros((len(ids),1,D),np.int32),np.zeros(len(ids),np.int32),out,np.zeros_like(out));out[np.arange(len(ids)),ids]=-1;return out
  res["recall"]=recall_eval(top,NS,gc,gC)
 if A.a3:
  res["a3"]={};trs=defaultdict(set);stored=set();pa=defaultdict(list)
  for h,t,r in zip(H[tr].tolist(),T[tr].tolist(),R[tr].tolist()):trs[(r,h)].add(t)
  for h,t in zip(H.tolist(),T.tolist()):stored.add((h,t));stored.add((t,h))
  for name,r1,r2_ in COMP:
   allt=defaultdict(int)
   for i in np.nonzero(tr&(R==r1))[0].tolist():
    for t2 in trs.get((r2_,int(T[i])),()):allt[t2]+=1
   cand=np.unique(T[tr&(R==r2_)]);fr=np.array([allt.get(c,0) for c in cand.tolist()],float)
   for var in ("stored_hops","hop1_held_out"):
    two=defaultdict(set)
    for i in np.nonzero((tr if var=="stored_hops" else ev)&(R==r1))[0].tolist():
     for t2 in trs.get((r2_,int(T[i])),()):
      if t2!=H[i] and (int(H[i]),t2) not in stored:two[int(H[i])].add(t2)
    heads=sorted(two);rr_=np.random.default_rng(A.seed+31);heads=[heads[j] for j in rr_.permutation(len(heads))[:A.nq]]
    if not heads:continue
    qt=[sorted(two[h])[rr_.integers(len(two[h]))] for h in heads]
    q=evalq(np.array(heads),np.array(qt),[r1,r2_],cand,fr,[two[h]-{t} for h,t in zip(heads,qt)],int(rtype[r1]),rr_,popd(np.unique(T[tr&(R==r2_)],return_counts=True)[1].astype(float)));f=fin(q,np.arange(q["n"]))
    res["a3"][f"{name} {var}"]={"n":q["n"],"in_cand":round(float(q["keep"].mean()),4),"hits10":round(float(f["ht"].mean()),4),"base_hits10":round(float(f["hb"].mean()),4),"auc_raw":round(auc(f["sp"],f["sn"]),4)}
    if var=="stored_hops":pa["ht"].append(f["ht"]);pa["hb"].append(f["hb"])
    log("a3",name,var,res["a3"][f"{name} {var}"])
  if pa:a,b=np.concatenate(pa["ht"]),np.concatenate(pa["hb"]);res["a3_pooled_stored_hops"]={"n":int(len(a)),"hits10":round(float(a.mean()),4),"base_hits10":round(float(b.mean()),4),"ratio":round(float(a.mean()/max(b.mean(),1e-9)),3)};log("a3 pooled",res["a3_pooled_stored_hops"])
 res["total_s"]=round(time.time()-t0,1);json.dump(res,open(os.path.join(TD,f"res_{A.tag}.json"),"w"),indent=1);log("saved",f"res_{A.tag}.json",res["total_s"])
if __name__=="__main__":main()
