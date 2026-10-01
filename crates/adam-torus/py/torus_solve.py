import os,time,json,argparse,numpy as np,taichi as ti
from collections import defaultdict
ROOT=os.environ.get("AMNI_ROOT","/home/anmire/ai/Amni-Ai");TD=os.path.join(ROOT,"data","torus")
ap=argparse.ArgumentParser()
for k,v in dict(mode="dev",D=64,Ds=-1,sweeps=30,S="lin",lam=16,rho=500,wrel=1,seed=17,tag="run",nq=2000,tau="",recall=1,a3=0,save=0,threads=3,fit=0,lift=1,tm="lut",sphere=0,cav=1,warm=0,prior=0,bidir=0).items():ap.add_argument("--"+k,type=type(v),default=v)
A=ap.parse_args();D=A.D;DS=D if A.Ds<0 else A.Ds;HY=DS<D
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
def hits(sc,true_idx,filt):
 st=sc[np.arange(len(sc)),true_idx];g=np.zeros(len(sc));e=np.zeros(len(sc))
 for i in range(len(sc)):m=np.ones(sc.shape[1],bool);m[filt[i]]=False;m[true_idx[i]]=False;g[i]=(sc[i][m]>st[i]).sum();e[i]=(sc[i][m]==st[i]).sum()
 return np.clip((10-g)/(e+1),0,1)
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
 t0=time.time();gc=np.load(os.path.join(TD,"sphere_graph_cur.npz"));gC=np.load(os.path.join(TD,"sphere_graph_C.npz"))
 if A.sphere:
  sp=np.load(os.path.join(ROOT,"data","concepts","sphere.npz"));P=sp["xyz"].astype(np.float32);NS=len(P)
  def top(ids):s=np.einsum("qk,nk->qn",P[ids],P);s[np.arange(len(ids)),ids]=-9;return s
  r=recall_eval(top,NS,gc,gC);json.dump({"sphere_recall":r},open(os.path.join(TD,f"res_{A.tag}.json"),"w"),indent=1);return
 d=np.load(os.path.join(TD,"a2_data.npz"),allow_pickle=True);H,T,R,SPL=d["h"],d["t"],d["r"],d["split"];RT=d["rtype"];N=len(d["names"]);NS=int(d["n_sphere"]);NF=len(d["rels"]);REL=NF;NREL=NF+1+NF*A.bidir;rtype=np.concatenate([RT,[0],RT[:NF*A.bidir]]);RR=[r for r in range(NREL) if r!=REL];RV=lambda r:r+NF+1
 tr=SPL==0 if A.mode=="dev" else SPL<2;ev=SPL==1 if A.mode=="dev" else SPL==2
 ge=np.concatenate([np.stack([gc["edge_src"],gc["edge_dst"]],1),np.stack([gC["edge_src"],gC["edge_dst"]],1)]).astype(np.int64);ge=np.sort(ge,1);ge=np.unique(ge[ge[:,0]!=ge[:,1]],axis=0)
 hold=np.sort(np.stack([H[~tr],T[~tr]],1),1);hk=np.unique(hold[:,0]*N+hold[:,1]);ge=ge[~np.isin(ge[:,0]*N+ge[:,1],hk)]
 gh,gt,trr=np.concatenate([H[tr],ge[:,0]]+([T[tr]] if A.bidir else [])),np.concatenate([T[tr],ge[:,1]]+([H[tr]] if A.bidir else [])),np.concatenate([R[tr].astype(np.int64),np.full(len(ge),REL)]+([R[tr].astype(np.int64)+NF+1] if A.bidir else []))
 log(f"mode={A.mode} D={D} Ds={DS} S={A.S} tm={A.tm} bidir={A.bidir} prior={A.prior} cav={A.cav} warm={A.warm} sweeps={A.sweeps} lam={A.lam} rho={A.rho} wrel={A.wrel} lift={A.lift} train_facts={int(tr.sum())} related_edges={len(ge)} eval={int(ev.sum())}")
 ty=rtype[trr];NR=N;trow={}
 if HY:
  for k in range(3):m=ty==k;ents=np.unique(np.concatenate([gh[m],gt[m]]));mp=np.full(N,-1,np.int64);mp[ents]=NR+np.arange(len(ents));trow[k]=mp;NR+=len(ents)
 li=d["lift"];has=(li>=0)&(A.lift>0);rng=np.random.default_rng(A.seed)
 X0=rng.integers(0,17,size=(NR,D)).astype(np.int32);L=lift(d["xyz"],D,A.seed);X0[:N][has]=L[has]
 if HY:
  for k in range(3):m=trow[k]>=0;X0[trow[k][m]]=X0[np.nonzero(m)[0]]
 ti.init(arch=ti.cpu,cpu_max_num_threads=A.threads,random_seed=A.seed,default_ip=ti.i32,offline_cache=False,log_level=ti.ERROR)
 X=ti.field(ti.i32,shape=(NR,D));Xn=ti.field(ti.i32,shape=(NR,D));TF=ti.field(ti.i32,shape=(NREL,D,17));Sf=ti.field(ti.i32,shape=17);Wf=ti.field(ti.i32,shape=NREL)
 NGH=ti.field(ti.i64,shape=(NREL,D,17));NGT=ti.field(ti.i64,shape=(NREL,D,17))
 X.from_numpy(X0);Sf.from_numpy(Sa.astype(np.int32));Wf.from_numpy(np.array([A.wrel if r==REL else 1 for r in range(NREL)],np.int32));TFa=np.tile(X17,(NREL,D,1)).astype(np.int32);TF.from_numpy(TFa)
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
    g=ti.Vector.zero(ti.i64,17)
    for k in range(p[e],p[e+1]):
     q=o[k];r=rr[k];w=ti.cast(Wf[r]*cv[q],ti.i64)
     if rl[k]==1:
      u=TF[r,d,X[q,d]]
      for v in ti.static(range(17)):g[v]+=w*Sf[(u-v+17)%17]
     else:
      y=X[q,d]
      for v in ti.static(range(17)):g[v]+=w*Sf[(TF[r,d,v]-y+17)%17]
    for k in range(rp[e],rp[e+1]):
     r=rc[k];a=ti.cast(rh[k],ti.i64)*16;b=ti.cast(rw[k],ti.i64)
     for v in ti.static(range(17)):g[v]-=lam*(a*NGH[r,d,v]+b*NGT[r,d,v])//256
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
 def fitT(Xa):
  T=np.tile(X17,(NREL,D,1)).astype(np.int32)
  for r in RR:
   for h,t,lo,hi in ends(r):
    nd=hi-lo;J=np.bincount((np.arange(nd)*289+Xa[h,lo:hi]*17+Xa[t,lo:hi]).ravel(),minlength=nd*289).reshape(nd,17,17).astype(np.float64)
    c=np.unique(t);pc=np.stack([np.bincount(Xa[c,lo+j],minlength=17) for j in range(nd)])/max(len(c),1)
    G=16*np.stack([conv(J[:,x,:]) for x in range(17)],1)-A.lam*J.sum(2)[:,:,None]*conv(pc)[:,None,:]
    if A.tm=="lut":T[r,lo:hi]=G.argmax(2)
    elif A.tm=="trans":b=np.stack([G[:,X17,(X17+bb)%17].sum(1) for bb in range(17)],1).argmax(1);T[r,lo:hi]=(X17[None,:]+b[:,None])%17
    else:sc=np.stack([np.stack([G[:,X17,(aa*X17+bb)%17].sum(1) for bb in range(17)],1) for aa in range(17)],1).reshape(nd,289).argmax(1);T[r,lo:hi]=((sc//17)[:,None]*X17[None,:]+(sc%17)[:,None])%17
  return T
 def updNG(Xa,T):
  gh_=np.zeros((NREL,D,17),np.int64);gt_=np.zeros((NREL,D,17),np.int64)
  for r in range(NREL):
   for h,t,lo,hi in ends(r) if r!=REL else [(gh[trr==r],gt[trr==r],0,DS)]+([(trow[0][gh[trr==r]],trow[0][gt[trr==r]],DS,D)] if HY else []):
    nd=hi-lo;c=np.unique(t) if r!=REL else np.unique(np.concatenate([h,t]));pc=np.stack([np.bincount(Xa[c,lo+j],minlength=17) for j in range(nd)])/max(len(c),1)
    la=np.stack([np.bincount(T[r,lo+j][Xa[h,lo+j]],minlength=17) for j in range(nd)])/max(len(h),1);cp_=conv(pc);cl=conv(la)
    gh_[r,lo:hi]=np.round(256*np.take_along_axis(cp_,T[r,lo:hi],1)).astype(np.int64);gt_[r,lo:hi]=np.round(256*cl).astype(np.int64)
  NGH.from_numpy(gh_);NGT.from_numpy(gt_)
 for s in range(A.sweeps):
  Xa=X.to_numpy();TFa=fitT(Xa) if s>=A.warm else np.tile(X17,(NREL,D,1)).astype(np.int32);TF.from_numpy(TFa);updNG(Xa,TFa)
  for f,g,lo,hi in groups:icm(*f,*g,lo,hi,A.lam,A.rho,A.seed*1000+s)
  cp();ch=int((X.to_numpy()!=Xa).sum());(s%5==4 or s==0) and log(f"sweep {s+1}/{A.sweeps} changed={ch} ({ch/(NR*D):.4f}) t={time.time()-t0:.0f}s")
 Xa=X.to_numpy().astype(np.int32);TFa=fitT(Xa);TF.from_numpy(TFa)
 res={"cfg":vars(A),"rows":int(NR),"entities":int(N),"stored_digits":int(N*DS+(NR-N)*(D-DS)) if HY else int(N*(D+1)),"transform_digits":int(TFa[RR].size),"digits_per_query":D,"solve_s":round(time.time()-t0,1)}
 rq=np.random.default_rng(1);ia=rq.integers(0,len(ge),20000);r1,r2=rq.integers(0,N,20000),rq.integers(0,N,20000);sx=lambda p,q:Sa[(Xa[p,:DS]-Xa[q,:DS])%17].sum(1);res["collapse"]={"dim_max_share":round(float(np.mean([np.bincount(Xa[:N,j],minlength=17).max()/N for j in range(DS)])),4),"related_pair_score":round(float(sx(ge[ia,0],ge[ia,1]).mean()),1),"random_pair_score":round(float(sx(r1,r2).mean()),1),"distinct_rows":int(len(np.unique(Xa[:N,:DS],axis=0)))};log("collapse",res["collapse"])
 if A.save:np.savez_compressed(os.path.join(TD,f"torus_{A.tag}.npz"),X=Xa.astype(np.uint8),TF=TFa.astype(np.uint8),S=Sa.astype(np.uint8),Ds=np.int64(DS),trow=np.stack([trow[k] for k in range(3)]) if HY else np.zeros(0))
 crow=lambda c,k:(c,np.where(trow[k][c]>=0,trow[k][c],c) if HY else c)
 @ti.kernel
 def score(land:ti.types.ndarray(),hx:ti.types.ndarray(),ri:ti.types.ndarray(),c0:ti.types.ndarray(),c1:ti.types.ndarray(),out:ti.types.ndarray()):
  for q,c in ti.ndrange(out.shape[0],out.shape[1]):
   s=0
   for dd in range(D):
    row=ti.cast(c0[c],ti.i32)
    if dd>=DS:row=ti.cast(c1[c],ti.i32)
    xc=X[row,dd];s+=Sf[(land[q,dd]-xc+17)%17]
    if ri.shape[0]>0:
     y=xc
     for j in range(ri.shape[0]):y=TF[ri[j],dd,y]
     s+=Sf[(y-hx[q,dd]+17)%17]
   out[q,c]=s
 @ti.kernel
 def pscore(land:ti.types.ndarray(),hx:ti.types.ndarray(),ri:ti.types.ndarray(),c0:ti.types.ndarray(),c1:ti.types.ndarray(),out:ti.types.ndarray()):
  for q in range(out.shape[0]):
   s=0
   for dd in range(D):
    row=ti.cast(c0[q],ti.i32)
    if dd>=DS:row=ti.cast(c1[q],ti.i32)
    xc=X[row,dd];s+=Sf[(land[q,dd]-xc+17)%17]
    if ri.shape[0]>0:
     y=xc
     for j in range(ri.shape[0]):y=TF[ri[j],dd,y]
     s+=Sf[(y-hx[q,dd]+17)%17]
   out[q]=s
 def lands(h,rs,k):
  l=np.zeros((len(h),D),np.int32);l[:,:DS]=Xa[h,:DS]
  if HY:l[:,DS:]=Xa[np.where(trow[k][h]>=0,trow[k][h],h),DS:]
  for r in rs:l=TFa[r][np.arange(D)[None,:],l]
  return l.astype(np.int32)
 def pair(h,rs,k,t):o=np.zeros(len(h),np.int32);c0,c1=crow(t,k);pscore(lands(h,rs,k),lands(h,[],k),rinv(rs),c0.astype(np.int64),c1.astype(np.int64),o);return o
 rinv=lambda rs:np.array([RV(r) for r in rs[::-1]] if A.bidir else [],np.int32)
 allf=defaultdict(set)
 for h,t,r in zip(H.tolist(),T.tolist(),R.tolist()):allf[(h,r)].add(t)
 taus=[float(x) for x in A.tau.split(",")] if A.tau else [1,2,4,8,16,32,64]
 def evalq(qh,qt,rs,cand,freq,filt_sets,k,rng,pop=None):
  ci={c:i for i,c in enumerate(cand.tolist())};keep=np.array([t in ci for t in qt.tolist()]);n=len(qh);pop=np.zeros(len(cand),np.int32) if pop is None else pop
  c0,c1=crow(cand,k);out=np.zeros((n,len(cand)),np.int32);score(lands(qh,rs,k),lands(qh,[],k),rinv(rs),c0.astype(np.int64),c1.astype(np.int64),out);out+=A.prior*pop[None,:]
  ti_=np.array([ci.get(t,0) for t in qt.tolist()]);filt=[[ci[x] for x in fs if x in ci] for fs in filt_sets]
  h_t=hits(out,ti_,filt);h_t[~keep]=0;h_b=hits(np.tile(freq,(n,1)),ti_,filt);h_b[~keep]=0
  neg=np.array([rng.choice(np.setdiff1d(np.arange(len(cand)),np.array(f+[ti_[i]],np.int64))) if len(cand)>len(f)+1 else 0 for i,f in enumerate(filt)])
  sp_=pair(qh,rs,k,np.asarray(qt))+A.prior*np.array([pop[ci[t]] if t in ci else 0 for t in qt.tolist()],np.int32);sn=out[np.arange(n),neg];K=min(32,len(cand));top=np.argpartition(-out,K-1,axis=1)[:,:K];smax=out.max(1);pr={}
  for tau in taus:
   lut=np.round(65535*np.exp(-np.arange(32*D+17*A.prior+1)/tau)).astype(np.int64);mp=np.maximum(smax,sp_);ts=out[np.arange(n)[:,None],top];et=lut[smax[:,None]-ts].sum(1);ep=lut[mp[:,None]-ts].sum(1)
   ins_p=keep&(top==ti_[:,None]).any(1)&(sp_==out[np.arange(n),ti_]);ins_n=(top==neg[:,None]).any(1);zp=lut[mp-sp_];zn=lut[smax-sn]
   pr[tau]=(zp/np.maximum(ep+np.where(ins_p,0,zp),1),zn/np.maximum(et+np.where(ins_n,0,zn),1))
  return dict(n=int(n),keep=keep,ht=h_t,hb=h_b,sp=sp_,sn=sn,pr=pr)
 popd=lambda f:np.minimum(16,np.floor(2*np.log2(1+f))).astype(np.int32)
 seen=np.zeros(N,bool);seen[gh]=True;seen[gt]=True;rng=np.random.default_rng(A.seed+5);per={};pool=defaultdict(list)
 for r in range(NF):
  m=(tr if A.fit else ev)&(R==r);idx=np.nonzero(m)[0];idx=idx[rng.permutation(len(idx))[:A.nq]];cand=np.unique(T[tr&(R==r)]);ci={c:i for i,c in enumerate(cand.tolist())};fr=np.zeros(len(cand))
  for t in T[tr&(R==r)].tolist():fr[ci[t]]+=1
  q=evalq(H[idx],T[idx],[r],cand,fr,[allf[(h,r)]-{t} for h,t in zip(H[idx].tolist(),T[idx].tolist())],int(rtype[r]),rng,popd(fr))
  q["tseen"]=seen[T[idx]]
  q["seen"]=seen[H[idx]]&q["tseen"];per[str(d["rels"][r])]=q;pool["seen"].append(q["seen"]);[pool[k_].append(q[k_]) for k_ in ("ht","hb","sp","sn")];[pool[f"p{tau}"].append(q["pr"][tau][0]) for tau in taus];[pool[f"n{tau}"].append(q["pr"][tau][1]) for tau in taus]
 P={k_:np.concatenate(v) for k_,v in pool.items()};sv=P["seen"]
 res["pooled"]={"n":int(len(P["ht"])),"hits10":round(float(P["ht"].mean()),4),"base_hits10":round(float(P["hb"].mean()),4),"ratio":round(float(P["ht"].mean()/max(P["hb"].mean(),1e-9)),3),"auc_raw":round(auc(P["sp"],P["sn"]),4),"auc_softmax_by_tau":{str(tau):round(auc(P[f"p{tau}"],P[f"n{tau}"]),4) for tau in taus}}
 res["pooled_transductive"]={"n":int(sv.sum()),"hits10":round(float(P["ht"][sv].mean()),4),"base_hits10":round(float(P["hb"][sv].mean()),4),"auc_raw":round(auc(P["sp"][sv],P["sn"][sv]),4),"auc_softmax_by_tau":{str(tau):round(auc(P[f"p{tau}"][sv],P[f"n{tau}"][sv]),4) for tau in taus}}
 res["per_relation"]={k_:{"n":v["n"],"transductive_frac":round(float(v["seen"].mean()),3),"in_cand":round(float(v["keep"].mean()),3),"hits10":round(float(v["ht"].mean()),4),"base_hits10":round(float(v["hb"].mean()),4),"hits10_trans":round(float(v["ht"][v["seen"]].mean()),4) if v["seen"].any() else None,"base_hits10_trans":round(float(v["hb"][v["seen"]].mean()),4) if v["seen"].any() else None,"auc_raw":round(auc(v["sp"],v["sn"]),4)} for k_,v in per.items()}
 log(json.dumps(res["pooled"]));log("transductive",json.dumps(res["pooled_transductive"]));[log(k_,v) for k_,v in res["per_relation"].items()]
 if A.recall:
  allc=np.arange(NS);c0,c1=crow(allc,0)
  def top(ids):
   out=np.zeros((len(ids),NS),np.int32);l=np.zeros((len(ids),D),np.int32);l[:,:DS]=Xa[ids,:DS]
   if HY:l[:,DS:]=Xa[np.where(trow[0][ids]>=0,trow[0][ids],ids),DS:]
   score(l,l,np.zeros(0,np.int32),c0.astype(np.int64),c1.astype(np.int64),out);out[np.arange(len(ids)),ids]=-1;return out
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
    q=evalq(np.array(heads),np.array(qt),[r1,r2_],cand,fr,[two[h]-{t} for h,t in zip(heads,qt)],int(rtype[r1]),rr_,popd(np.unique(T[tr&(R==r2_)],return_counts=True)[1].astype(float)))
    res["a3"][f"{name} {var}"]={"n":q["n"],"in_cand":round(float(q["keep"].mean()),4),"hits10":round(float(q["ht"].mean()),4),"base_hits10":round(float(q["hb"].mean()),4),"auc_raw":round(auc(q["sp"],q["sn"]),4)}
    if var=="stored_hops":pa["ht"].append(q["ht"]);pa["hb"].append(q["hb"])
    log("a3",name,var,res["a3"][f"{name} {var}"])
  if pa:a,b=np.concatenate(pa["ht"]),np.concatenate(pa["hb"]);res["a3_pooled_stored_hops"]={"n":int(len(a)),"hits10":round(float(a.mean()),4),"base_hits10":round(float(b.mean()),4),"ratio":round(float(a.mean()/max(b.mean(),1e-9)),3)};log("a3 pooled",res["a3_pooled_stored_hops"])
 res["total_s"]=round(time.time()-t0,1);json.dump(res,open(os.path.join(TD,f"res_{A.tag}.json"),"w"),indent=1);log("saved",f"res_{A.tag}.json",res["total_s"])
if __name__=="__main__":main()
