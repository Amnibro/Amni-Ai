import os,sys,json,time,argparse,numpy as np,taichi as ti
sys.argv.extend(["--S","sharp"]);sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import torus_solve as TS
import torus_c_seed as SD
Sa=TS.Sa;log=TS.log;X17=TS.X17
ap=argparse.ArgumentParser()
for k,v in dict(tag="k32",model="",split="dev",clu="m1",K=0,tori=8,seed=17,ctl=0,final=0,nq=20000,nneg=99,npos=300000,B=8,comp=0,a2b=0,export=0,out="",ksub=400000,kit=10,novel=0,dir="b",dset="a2b_data.npz",Bp=4,seedk=32,seedf="",lutk=1).items():ap.add_argument("--"+k,type=type(v),default=v)
A=ap.parse_known_args()[0];TD=os.path.join(TS.ROOT,"data","torus",A.dir)
ti.init(arch=ti.cpu,cpu_max_num_threads=3,default_ip=ti.i32,offline_cache=False,log_level=ti.ERROR)
Sf=ti.field(ti.i32,shape=17);Sf.from_numpy(Sa.astype(np.int32));CH=400000
SECT,SLa=TS.SECT,TS.SLa;SLc=Sa[(X17[:,None]-X17[None,:])%17].astype(np.int32)
@ti.kernel
def ksc(X:ti.types.ndarray(),eh:ti.types.ndarray(),et:ti.types.ndarray(),T:ti.types.ndarray(),du:ti.i32,bk:ti.types.ndarray(),bs:ti.types.ndarray(),d0:ti.i32,SL:ti.types.ndarray()):
 for e in range(bk.shape[0]):
  b=-1;bi=0
  for k in range(T.shape[0]):
   s=0
   for d in range(du):s+=SL[T[k,d,X[eh[e],d0+d]],X[et[e],d0+d]]
   if s>b:b=s;bi=k
  bk[e]=bi;bs[e]=b
@ti.kernel
def csc(X:ti.types.ndarray(),land:ti.types.ndarray(),cand:ti.types.ndarray(),out:ti.types.ndarray()):
 for q,c in ti.ndrange(out.shape[0],out.shape[1]):
  s=0
  for d in range(land.shape[1]):s+=Sf[(land[q,d]-X[cand[c],d]+17)%17]
  out[q,c]=s
d=np.load(os.path.join(TD,A.dset),allow_pickle=True);H,T,R,SPL,src,names=d["h"],d["t"],d["r"].astype(np.int64),d["split"],d["src"].astype(np.int64),d["names"];N=len(src);rels=[str(x) for x in d["rels"]];NL=len(rels);NS=10;AS=[0,1,2,6,7,8,9]
Mf=np.load(os.path.join(TD,f"torus_{A.model or A.tag}.npz"));X=Mf["X"].astype(np.int32);D=X.shape[1];P0=int(Mf["psub"]) if "psub" in Mf else 0;Mf={k:Mf[k] for k in Mf.files};Mf.setdefault("phi",np.zeros(0));NK=Mf["TF"].shape[0]-1;K=A.K or NK;DU=min(D-P0,8*A.tori);T1=np.ascontiguousarray(Mf["TF"][:NK,P0:P0+DU]).astype(np.int32);tt=lambda P,ang:np.repeat(P[:,:,None],17,2).astype(np.int32) if ang else ((X17[None,None,:]+P[:,:,None])%17).astype(np.int32)
tr=SPL==0;eh0,et0,Rt=H[tr].copy(),T[tr].copy(),R[tr];fl=Mf["orient_flip"] if "orient_flip" in Mf else np.zeros(0,np.int64);eh0[fl],et0[fl]=et0[fl].copy(),eh0[fl].copy()
deg=np.bincount(np.concatenate([eh0,et0]),minlength=N);bn=np.floor(np.log2(1+deg)).astype(np.int64)
mkk=lambda h,t:np.unique(np.concatenate([h*N+t,t*N+h]))
KN_ALL,KN_TR=mkk(H,T),mkk(eh0,et0)
isk=lambda kn,a,b:(lambda k:(kn[np.minimum(np.searchsorted(kn,k),len(kn)-1)]==k))(a*N+b)
SDm=dict(np.load(os.path.join(TD,A.seedf or f"seed_K{A.seedk}.npz"))) if A.clu=="seed" else None
W=50
ORD={}
def order(mode):
 if mode in ORD:return ORD[mode]
 ok=np.nonzero(deg>0)[0];g=src[ok] if mode=="src" else np.zeros(len(ok),np.int64);o=ok[np.lexsort((np.random.default_rng(7).random(len(ok)),deg[ok],g))];ps=np.full(N,-1,np.int64);ps[o]=np.arange(len(o));go=src[o] if mode=="src" else np.zeros(len(o),np.int64);gs=np.searchsorted(go,np.arange(4));ge=np.searchsorted(go,np.arange(4),side="right")
 ORD[mode]=(o,ps,gs,ge);return ORD[mode]
def draw(h,t,mode,kn,n,rng):
 o,ps,gs,ge=order(mode if mode!="uni" else "deg");g=src[t] if mode=="src" else np.zeros(len(t),np.int64)
 if mode=="uni":
  c=o[rng.integers(0,len(o),(len(t),n))]
  for _ in range(40):
   bad=(c==t[:,None])|(c==h[:,None])|isk(kn,h[:,None],c)
   if not bad.any():break
   c=np.where(bad,o[rng.integers(0,len(o),c.shape)],c)
  return c
 if mode=="deg":gs,ge=np.zeros(4,np.int64),np.full(4,len(o))
 lo=np.clip(ps[t]-W,gs[g],np.maximum(ge[g]-2*W-1,gs[g]));hi=np.minimum(lo+2*W+1,ge[g]);c=o[lo[:,None]+(rng.random((len(t),n))*(hi-lo)[:,None]).astype(np.int64)]
 for _ in range(40):
  bad=(c==t[:,None])|(c==h[:,None])|isk(kn,h[:,None],c)
  if not bad.any():break
  c=np.where(bad,o[lo[:,None]+(rng.random(c.shape)*(hi-lo)[:,None]).astype(np.int64)],c)
 return c
rays=lambda Xm,h,t,nd:((Xm[t][:,P0:P0+nd]-Xm[h][:,P0:P0+nd])%17).astype(np.int64)
codes=lambda ry:SECT[ry[:,0::2],ry[:,1::2]].astype(np.int32)
def prox(Xm,h,t):return np.concatenate([Sa[(Xm[h[s0:s0+CH],:P0]-Xm[t[s0:s0+CH],:P0])%17].sum(1) for s0 in range(0,len(h),CH)]).astype(np.int32) if P0 and len(h) else np.zeros(len(h),np.int32)
def cellmap():
 if "cell" in ORD:return ORD["cell"]
 sp=np.load(os.path.join(TD,"sphere_noleak.npz"));li=d["lift"];c=np.where(li>=0,sp["cell_id_64"].astype(np.int64)[np.maximum(li,0)]//64+1,0);key=c[eh0]*400+c[et0];u,n=np.unique(key,return_counts=True);top=u[np.argsort(-n,kind="stable")[:K]];ORD["cell"]=(c,top);return ORD["cell"]
def cassign(C,Xm,eh,et):
 kd=C["kind"];n=len(eh);p=prox(Xm,eh,et)
 if kd=="seed":return SD.apply(SDm,eh,et),np.zeros(n,np.int32),np.zeros(n,np.int32)
 if kd=="cell":
  c,top=cellmap();key=c[eh]*400+c[et];o=np.argsort(top,kind="stable");st=top[o];jj=np.minimum(np.searchsorted(st,key),len(st)-1);hit=st[jj]==key;return np.where(hit,o[jj],len(top)).astype(np.int64),np.zeros(n,np.int32),p
 if kd=="m3":
  out=[]
  for s0 in range(0,n,200000):
   r=(Xm[et[s0:s0+200000],P0:P0+8]-Xm[eh[s0:s0+200000],P0:P0+8])%17;c=np.where(np.minimum(r,17-r)<=1,0,np.where(r<=8,1,2));out.append((c[:,None,:]!=C["cent"][None,:,:]).sum(2))
  hm=np.concatenate(out);k=hm.argmin(1);return k.astype(np.int64),(8-hm[np.arange(len(k)),k]).astype(np.int32),p
 bk=np.zeros(n,np.int32);bs=np.zeros(n,np.int32)
 for s0 in range(0,n,CH):
  h_,t_=eh[s0:s0+CH],et[s0:s0+CH];m=len(h_);b1=np.zeros(m,np.int32);b2=np.zeros(m,np.int32)
  C.get("ang") and ksc(np.concatenate([np.zeros((1,C["du"]),np.int32),codes(rays(Xm,h_,t_,C["nd"]))]),np.zeros(m,np.int32),np.arange(1,m+1,dtype=np.int32),C["T"],C["du"],b1,b2,0,SLa)
  C.get("ang") or ksc(Xm,np.ascontiguousarray(h_,dtype=np.int32),np.ascontiguousarray(t_,dtype=np.int32),C["T"],C["du"],b1,b2,P0,SLc)
  bk[s0:s0+CH],bs[s0:s0+CH]=b1,b2
 return bk.astype(np.int64),bs,p
def kmodes(Xm,rng,Kc,du,it,ang,P=None):
 sub=rng.choice(len(eh0),min(len(eh0),A.ksub),replace=False);ry=rays(Xm,eh0[sub],et0[sub],du);ry=(codes(ry) if ang else ry).astype(np.int32);dd=ry.shape[1];SLm=SLa if ang else SLc;Z=np.concatenate([np.zeros((1,dd),np.int32),ry]);zi=np.zeros(len(sub),np.int32);ti_=np.arange(1,len(sub)+1,dtype=np.int32);P=(TS.kpp(ry[:30000],Kc,rng,Sa).astype(np.int32) if P is None else P);Tt=tt(P,ang)
 for i in range(it):
  bk=np.zeros(len(sub),np.int32);bs=np.zeros(len(sub),np.int32);ksc(Z,zi,ti_,Tt,dd,bk,bs,0,SLm);J=np.bincount(((bk[:,None].astype(np.int64)*dd+np.arange(dd)[None,:])*17+ry).ravel(),minlength=Kc*dd*17).reshape(Kc,dd,17).astype(np.float64);G=np.stack([(J*SLm[y][None,None,:]).sum(-1) for y in range(17)],-1);Pn=G.argmax(2).astype(np.int32);sz=np.bincount(bk,minlength=Kc);dead=np.nonzero(sz<50)[0];Pn[sz==0]=P[sz==0]
  len(dead) and Pn.__setitem__(dead,ry[np.argpartition(bs,len(dead))[:len(dead)]]);ch=int((Pn!=P).any(1).sum());P=Pn;Tt=tt(P,ang)
  if ch==0:break
 return P
def fit(kind,Xm,rng,Kc,du):
 if kind in ("seed","cell"):return dict(kind=kind,K=int(SDm["table"].max()+1) if kind=="seed" else len(cellmap()[1])+1,du=0)
 if kind=="m1":return dict(kind="m1",T=np.ascontiguousarray(T1[:,:du]),du=du,K=NK)
 if kind=="m2":return dict(kind="m2",T=tt(kmodes(Xm,rng,Kc,du,A.kit,0),0),du=du,K=Kc)
 if kind=="a2":j=du//2;ix_=rng.choice(len(eh0),Kc,replace=False);P0c=codes(rays(Xm,eh0[ix_],et0[ix_],du)).astype(np.int32);return dict(kind="a1",T=tt(kmodes(Xm,rng,Kc,du,A.kit,1,P0c),1),du=j,nd=du,K=Kc,ang=1)
 if kind in ("a1","a0"):j=du//2;P0c=Mf["phi"][:,:j].astype(np.int32) if len(Mf["phi"]) else codes(T1[:,:du,0].astype(np.int64)).astype(np.int32);return dict(kind="a1",T=tt(kmodes(Xm,rng,NK,du,A.kit,1,P0c) if kind=="a1" else P0c,1),du=j,nd=du,K=NK,ang=1)
 sub=rng.choice(len(eh0),min(len(eh0),A.ksub),replace=False);r=rays(Xm,eh0[sub],et0[sub],8);c=np.where(np.minimum(r,17-r)<=1,0,np.where(r<=8,1,2));cd=(c*3**np.arange(8)).sum(1);u,n=np.unique(cd,return_counts=True);top=u[np.argsort(-n,kind="stable")[:Kc]];return dict(kind="m3",cent=np.stack([(top//3**j)%3 for j in range(8)],1),du=8,K=len(top))
def dfit(k,s,p,kn,sn,pn,Kc,B,Bp):
 k,kn,Kc=k*A.lutk,kn*A.lutk,Kc if A.lutk else 1;th=np.zeros((Kc,B-1),np.int64)
 for c in range(Kc):
  m=s[k==c];th[c]=np.quantile(m,np.arange(1,B)/B,method="lower") if len(m)>=B else 0
 tp_=np.quantile(p,np.arange(1,Bp)/Bp,method="lower").astype(np.int64) if Bp>1 and P0 else np.zeros(0,np.int64);Bq=Bp if (Bp>1 and P0) else 1
 bk=lambda k_,s_,p_:(k_*B+(s_[:,None]>=th[k_]).sum(1))*Bq+(p_[:,None]>=tp_[None,:]).sum(1)
 cp,cn=np.bincount(bk(k,s,p),minlength=Kc*B*Bq),np.bincount(bk(kn,sn,pn),minlength=Kc*B*Bq);return (th,tp_),np.round(288*(cp+1)/(cp+cn+2)).astype(np.int64),(B,Bq)
dap=lambda L,k,s,p:L[1][(k*A.lutk*L[2][0]+(s[:,None]>=L[0][0][k*A.lutk]).sum(1))*L[2][1]+(p[:,None]>=L[0][1][None,:]).sum(1)]/288
def hitsq(pt,pn):return np.clip((10-(pn>pt[:,None]).sum(1))/((pn==pt[:,None]).sum(1)+1),0,1)
def rnmi(c,l,m,minn=200):
 um,mi_=np.unique(m,return_inverse=True);cu=np.unique(c,return_inverse=True)[1];lu=np.unique(l,return_inverse=True)[1];nm_,nc_,nl_=len(um),cu.max()+1,lu.max()+1;J=np.bincount((mi_.astype(np.int64)*nc_+cu)*nl_+lu,minlength=nm_*nc_*nl_).reshape(nm_,nc_,nl_).astype(np.float64);n=J.sum();Hs=lambda P:-float((P[P>0]*np.log(P[P>0])).sum());cmi=hc=hl=sn=sw=0.0
 for i in range(nm_):
  ni=J[i].sum();p=J[i]/ni;pc,pl=p.sum(1),p.sum(0);z=p>0;mi=float((p[z]*np.log(p[z]/(pc[:,None]*pl[None,:])[z])).sum());w=ni/n;cmi+=w*mi;hc+=w*Hs(pc);hl+=w*Hs(pl);ok=ni>=minn;sn+=ok*ni*mi/max(0.5*(Hs(pc)+Hs(pl)),1e-12);sw+=ok*ni
 return cmi/max(0.5*(hc+hl),1e-12),sn/max(sw,1)
def pipe(C,Xm,hold,seed,full=1):
 rng=np.random.default_rng(seed);Kc=C["K"];hh,ht,hr=hold;res={}
 sub=rng.choice(len(eh0),min(len(eh0),A.npos),replace=False);ph,pt=eh0[sub],et0[sub];nt=draw(ph,pt,"src",KN_TR,1,rng)[:,0];kp,sp,pq=cassign(C,Xm,ph,pt);kn,sn,pn_=cassign(C,Xm,ph,nt);L=dfit(kp,sp,pq,kn,sn,pn_,Kc,A.B,A.Bp)
 ka,sa,pa=cassign(C,Xm,eh0,et0);aux={};cnt=np.bincount(ka*NL+Rt,minlength=Kc*NL).reshape(Kc,NL);tab=cnt.argmax(1)
 hk,hs,hp=cassign(C,Xm,hh,ht);sp_=lambda a,b:src[a]*3+src[b];Mh=sp_(hh,ht)*9+np.minimum(2,bn[hh]//3)*3+np.minimum(2,bn[ht]//3)
 res["nmi_heldout"],res["purity_heldout"]=TS.nmi(hk,hr);res["nmi_train"],res["purity_train"]=TS.nmi(ka,Rt);res["rnmi_heldout"],res["strata_nmi_heldout"]=rnmi(hk,hr,Mh);mxs=16*C["du"] if C["kind"] not in ("m3","seed","cell") else (8 if C["kind"]=="m3" else 1);res["train_fit"]=float(sa.mean()/mxs);res["clusters_used"]=int((np.bincount(ka,minlength=Kc)>0).sum());res["max_share"]=float(np.bincount(ka,minlength=Kc).max()/len(ka))
 ng=draw(hh,ht,"src",KN_ALL,1,np.random.default_rng(seed+1))[:,0];aux.update(ng=ng,hk=hk);nk,ns,np_=cassign(C,Xm,hh,ng);pp,pn=dap(L,hk,hs,hp),dap(L,nk,ns,np_);res["auc_matched"]=TS.auc(pp,pn);res["auc_train_matched"]=TS.auc(dap(L,kp,sp,pq),dap(L,kn,sn,pn_));res["auc_raw_matched"]=TS.auc(hs,ns);res["auc_prox_matched"]=TS.auc(hp,np_);res["auc_pop_matched"]=TS.auc(np.log1p(deg[ht]),np.log1p(deg[ng]))
 gd=draw(hh,ht,"deg",KN_ALL,1,np.random.default_rng(seed+2))[:,0];k2,s2,p2=cassign(C,Xm,hh,gd);res["auc_degmatched"]=TS.auc(pp,dap(L,k2,s2,p2));res["auc_pop_degmatched"]=TS.auc(np.log1p(deg[ht]),np.log1p(deg[gd]))
 un=draw(hh,ht,"uni",KN_ALL,1,np.random.default_rng(seed+3))[:,0];aux.update(un=un);k3,s3,p3=cassign(C,Xm,hh,un);res["auc_uniform"]=TS.auc(pp,dap(L,k3,s3,p3));res["auc_pop_uniform"]=TS.auc(np.log1p(deg[ht]),np.log1p(deg[un]))
 res["per_relation_auc_matched"]={rels[r]:round(TS.auc(pp[hr==r],pn[hr==r]),4) for r in range(NS)}
 if full:
  qi=np.random.default_rng(seed+4).choice(len(hh),min(A.nq,len(hh)),replace=False);n99=draw(hh[qi],ht[qi],"src",KN_ALL,A.nneg,np.random.default_rng(seed+5));aux.update(qi=qi,n99=n99);k4,s4,p4=cassign(C,Xm,np.repeat(hh[qi],A.nneg),n99.ravel());pr4=dap(L,k4,s4,p4).reshape(len(qi),A.nneg);res["hits10_matched"]=float(hitsq(pp[qi],pr4).mean());res["hits10_raw_matched"]=float(hitsq(hs[qi].astype(np.float64),s4.reshape(len(qi),A.nneg)).mean());res["hits10_pop_matched"]=float(hitsq(np.log1p(deg[ht[qi]]),np.log1p(deg[n99])).mean())
  rk,rs,rp=cassign(C,Xm,ht,hh);pr=dap(L,rk,rs,rp);ori=(pp>pr)+0.5*(pp==pr);res["orientation_acc"]={rels[r]:round(float(ori[hr==r].mean()),4) for r in range(NS)};res["orientation_acc_pooled"]=float(ori.mean());pol=(hs>rs)+0.5*(hs==rs);res["polarity_acc"]={rels[r]:round(float(pol[hr==r].mean()),4) for r in range(NS)};res["polarity_acc_pooled"]=float(pol.mean());ma=np.isin(hr,AS);res["polarity_acc_antisym"]=float(pol[ma].mean());res["orientation_acc_antisym"]=float(ori[ma].mean())
 pred=tab[hk];pi=np.bincount(Rt,minlength=NL)/len(Rt);rc=np.array([(pred[hr==r]==r).mean() for r in range(NS)]);st=np.zeros(9,np.int64);st=np.bincount(sp_(eh0,et0)*NL+Rt,minlength=9*NL).reshape(9,NL).argmax(1);sr=np.array([(st[sp_(hh,ht)][hr==r]==r).mean() for r in range(NS)])
 res["recall"]={rels[r]:round(float(rc[r]),4) for r in range(NS)};res["prior"]={rels[r]:round(float(pi[r]),4) for r in range(NS)};res["srcpair_recall"]={rels[r]:round(float(sr[r]),4) for r in range(NS)};res["recall_ge_2x_prior"]=int((rc>=2*pi[:NS]).sum());res["srcpair_ge_2x_prior"]=int((sr>=2*pi[:NS]).sum());res["accuracy"]=float((pred==hr).mean());res["srcpair_accuracy"]=float((st[sp_(hh,ht)]==hr).mean())
 if full and "T" in C and not C.get("ang"):
  du=C["du"];sb=rng.choice(len(eh0),min(len(eh0),400000),replace=False);ry=rays(Xm,eh0[sb],et0[sb],du);J=np.bincount(((Rt[sb][:,None]*du+np.arange(du)[None,:])*17+ry).ravel(),minlength=NL*du*17).reshape(NL,du,17);LP=np.round(64*np.log((J+1)/(J.sum(2,keepdims=True)+17))).astype(np.int64);lp=np.round(64*np.log(np.maximum(pi,1e-9))).astype(np.int64);rh=rays(Xm,hh,ht,du);nb=np.stack([LP[l][np.arange(du)[None,:],rh].sum(1)+lp[l] for l in range(NL)],1).argmax(1);rn=np.array([(nb[hr==r]==r).mean() for r in range(NS)]);res["ray_nb_accuracy"]=float((nb==hr).mean());res["ray_nb_recall"]={rels[r]:round(float(rn[r]),4) for r in range(NS)};res["ray_nb_ge_2x_prior"]=int((rn>=2*pi[:NS]).sum());res["ray_nb_nmi_heldout"]=TS.nmi(nb,hr)[0]
 tb=(cnt/np.maximum(cnt.sum(0,keepdims=True),1)).argmax(1);pb=tb[hk];rb=np.array([(pb[hr==r]==r).mean() for r in range(NS)]);pcn=np.array([((pb==r)&(hr==r)).sum()/max((pb==r).sum(),1) for r in range(NS)]);res["recall_balanced"]={rels[r]:round(float(rb[r]),4) for r in range(NS)};res["precision_balanced"]={rels[r]:round(float(pcn[r]),4) for r in range(NS)};res["balanced_ge_2x_prior"]=int((rb>=2*pi[:NS]).sum());res["strict_ge_2x_prior"]=int(((rb>=2*pi[:NS])&(pcn>=2*pi[:NS])).sum());res["accuracy_balanced"]=float((pb==hr).mean())
 res["meta_nmi_heldout"]=TS.nmi(Mh,hr)[0];return res,L,tab,ka,sa,aux
def shufflex(Xm,seed):
 rng=np.random.default_rng(seed);Y=Xm.copy()
 for s in range(3):ix=np.nonzero(src==s)[0];Y[ix]=Xm[ix[rng.permutation(len(ix))]]
 return Y
def lift0(seed):
 SPH=np.load(os.path.join(TD,"sphere_noleak.npz"));li=d["lift"];xyz=d["xyz"].copy();xyz[li>=0]=SPH["xyz"][li[li>=0]];rng=np.random.default_rng(seed);X0=rng.integers(0,17,size=(N,D)).astype(np.int32);L=TS.lift(xyz,D,seed);m=li>=0;X0[m]=L[m];return X0
def novel(C,ka,sa,rng,largest=0):
 Kc=C["K"];cnt=np.bincount(ka*NL+Rt,minlength=Kc*NL).reshape(Kc,NL);n=cnt.sum(1);p=cnt/np.maximum(n[:,None],1);ent=-(np.where(p>0,p*np.log(np.where(p>0,p,1)),0)).sum(1)/np.log(NL);maj=p.max(1);mx=16*C["du"] if C["kind"] not in ("m3","seed","cell") else (8 if C["kind"]=="m3" else 1);coh=np.array([sa[ka==c].mean()/mx if n[c] else 0 for c in range(Kc)]);q=(n>=500)&((ent>=0.5)|(maj<0.5))&(coh>=0.5);o=np.argsort(-n,kind="stable");sel=[c for c in o if q[c]][:10] if not largest else [];fill=[c for c in o if not q[c] and n[c]>=(0 if largest else 500)][:10-len(sel)];out=[]
 for c in sel+fill:
  mi=np.nonzero(ka==c)[0];sm=rng.choice(mi,min(10,len(mi)),replace=False);out.append(dict(cluster=int(c),n_train=int(n[c]),qualifies=bool(q[c]),cohesion=round(float(coh[c]),3),entropy=round(float(ent[c]),3),majority_share=round(float(maj[c]),3),labels={rels[l]:round(float(p[c,l]),3) for l in np.argsort(-p[c])[:4] if p[c,l]>0},members=[[str(names[eh0[i]]),str(names[et0[i]]),rels[Rt[i]]] for i in sm]))
 return out
def inverse(C,Xm,ka):
 if "T" not in C or C.get("ang") or not (C["T"]==tt(C["T"][:,:,0],0)).all():return {"note":"LUT or angle-only codes: ray negation undefined for the stored table"}
 Kc=len(C["T"]);du=C["du"];P=C["T"][:,:,0];ng=(-P)%17;sc=np.stack([Sa[(ng[k][None,:]-P)%17].sum(1) for k in range(Kc)]);rd=np.random.default_rng(3).integers(0,17,size=(Kc,du));rs=np.stack([Sa[(rd[k][None,:]-P)%17].sum(1) for k in range(Kc)]);cnt=np.bincount(ka*NL+Rt,minlength=Kc*NL).reshape(Kc,NL);out={"neg_proto_best_sim":round(float((sc.max(1)/(16*du)).mean()),4),"random_vec_best_sim":round(float((rs.max(1)/(16*du)).mean()),4),"frac_clusters_with_inverse_partner_ge_0.5":round(float((sc.max(1)>=8*du).mean()),4),"relations":{}}
 for r in range(NL):
  m=Rt==r;rk,_,_=cassign(C,Xm,et0[m],eh0[m]);fk=int(cnt[:,r].argmax());rm=int(np.bincount(rk,minlength=Kc).argmax());out["relations"][rels[r]]=dict(fwd_cluster=fk,rev_cluster=rm,rev_share=round(float((rk==rm).mean()),3),neg_partner=int(sc[fk].argmax()),rev_is_partner=bool(rm==sc[fk].argmax()),rev_is_same=bool(rm==fk),shared_fwd_rev=round(float((rk==ka[m]).mean()),3))
 return out
CH_=[("P19>P17",6,7),("P19>P131",6,8),("hyp>hyp",0,0),("P171>P171",9,9)]
def compose(C,Xm,ka,rng,hold,aux,var,nmax=2000):
 if "T" not in C or C.get("ang"):return {"note":"angle-only codes carry no magnitude: the sum of two sector codes is undefined; composition needs the translation prototypes (m1)"}
 out={};T_=C["T"];ph=pb=0.0;pn=0;hh,ht,hr=hold;Xr=np.ascontiguousarray(Xm[:,P0:P0+C["du"]])
 for nm,r1,r2 in CH_:
  e2=np.nonzero(Rt==r2)[0];o=e2[np.argsort(eh0[e2],kind="stable")];hs=eh0[o];tr1=np.nonzero(Rt==r1)[0];ho1=np.nonzero(hr==r1)[0]
  a_all,b_all,k_all=(eh0[tr1],et0[tr1],ka[tr1]) if var=="stored" else (hh[ho1],ht[ho1],aux["hk"][ho1]);pm=rng.permutation(len(a_all))[:30*nmax];a_,b_,k1_=a_all[pm],b_all[pm],k_all[pm];lo=np.searchsorted(hs,b_);hi=np.searchsorted(hs,b_,side="right");m=hi>lo;a_,b_,k1_,lo,hi=a_[m],b_[m],k1_[m],lo[m],hi[m];pk=o[lo+(rng.random(len(a_))*(hi-lo)).astype(np.int64)];c_=et0[pk];ok=(a_!=c_)&~isk(KN_ALL,a_,c_);a,c,k1,pk=a_[ok][:nmax],c_[ok][:nmax],k1_[ok][:nmax],pk[ok][:nmax];k2=ka[pk]
  a1={};a2={}
  for x,y in list(zip(eh0[tr1].tolist(),et0[tr1].tolist()))+list(zip(hh[ho1].tolist(),ht[ho1].tolist())):a1.setdefault(x,[]).append(y)
  for x,y in zip(eh0[e2].tolist(),et0[e2].tolist()):a2.setdefault(x,[]).append(y)
  dd=np.arange(C["du"]);land=T_[k2[:,None],dd[None,:],T_[k1[:,None],dd[None,:],Xr[a]]].astype(np.int32);ht_=np.zeros(len(a));hb=np.zeros(len(a));inc=np.zeros(len(a))
  for k in np.unique(k2):
   cand=np.unique(et0[ka==k]);fq=np.bincount(np.searchsorted(cand,et0[ka==k]),minlength=len(cand));qs=np.nonzero(k2==k)[0]
   for q0 in range(0,len(qs),100):
    qq=qs[q0:q0+100];out_=np.zeros((len(qq),len(cand)),np.int32);csc(Xr,np.ascontiguousarray(land[qq]),np.ascontiguousarray(cand,dtype=np.int32),out_)
    for j,qi_ in enumerate(qq):
     ti_=np.searchsorted(cand,c[qi_]);ina=ti_<len(cand) and cand[min(ti_,len(cand)-1)]==c[qi_];inc[qi_]=ina
     if not ina:continue
     tl=np.array(sorted({x for b_x in a1.get(int(a[qi_]),[]) for x in a2.get(b_x,[])}-{int(c[qi_]),int(a[qi_])}),np.int64);tl=tl[~isk(KN_ALL,np.full(len(tl),a[qi_]),tl)] if len(tl) else tl;fi=np.searchsorted(cand,tl);fi=fi[fi<len(cand)];mk_=np.ones(len(cand),bool);mk_[fi]=False;mk_[ti_]=False
     for arr,dst in ((out_[j],ht_),(fq,hb)):g=(arr[mk_]>arr[ti_]).sum();e=(arr[mk_]==arr[ti_]).sum();dst[qi_]=np.clip((10-g)/(e+1),0,1)
  out[nm]=dict(n=int(len(a)),in_cand=round(float(inc.mean()),4),hits10=round(float(ht_.mean()),4),base_hits10=round(float(hb.mean()),4),ratio=round(float(ht_.mean()/max(hb.mean(),1e-9)),3));ph+=ht_.sum();pb+=hb.sum();pn+=len(a);log("compose",var,nm,out[nm])
 out["pooled"]=dict(n=pn,hits10=round(ph/max(pn,1),4),base_hits10=round(pb/max(pn,1),4),ratio=round(ph/max(pb,1e-9),3));return out
def a2b(hold,aux):
 M=np.load(os.path.join(TD,"..","a2b","torus_dev_mix3.npz"));Xa=M["X"].astype(np.int32);TF=M["TF"].astype(np.int32);hh,ht,hr=hold;dd=np.arange(64)
 def sc(h,r,t):
  o=np.zeros(len(h),np.int64)
  for s0 in range(0,len(h),20000):
   xh=Xa[h[s0:s0+20000]];ix=xh+17*xh[:,dd^1]+289*xh[:,dd^2];o[s0:s0+20000]=Sa[(TF[r[s0:s0+20000][:,None],dd[None,:],ix]-Xa[t[s0:s0+20000]])%17].sum(1)
  return o
 sp_=sc(hh,hr,ht);ng=sc(hh,hr,aux["ng"]);un=sc(hh,hr,aux["un"]);qi=aux["qi"];n99=aux["n99"];s99=sc(np.repeat(hh[qi],n99.shape[1]),np.repeat(hr[qi],n99.shape[1]),n99.ravel()).reshape(n99.shape)
 return dict(auc_matched=TS.auc(sp_,ng),auc_uniform=TS.auc(sp_,un),hits10_matched=float(hitsq(sp_[qi].astype(np.float64),s99).mean()),per_relation_auc_matched={rels[r]:round(TS.auc(sp_[hr==r],ng[hr==r]),4) for r in range(NS)})
def export(C,L,tab):
 dg=lambda v,nd:np.stack([(np.asarray(v,np.int64)//17**k)%17 for k in range(nd)],-1).astype(np.uint8);ed=os.path.join(TD,f"l2_{A.tag}_{A.clu}");os.makedirs(ed,exist_ok=True);secs=[("X",X.astype(np.uint8)),("cal",dg(L[1],2)),("th",dg(L[0][0],3)),("thp",dg(L[0][1],3)),("lab",dg(tab,1))]+([("proto",(C["T"][:,:,0]).astype(np.uint8))] if C["kind"] in ("m1","m2","a1") else [("cent",C["cent"].astype(np.uint8))] if C["kind"]=="m3" else [])+([("sect",SECT.astype(np.uint8)),("angd",SLa.astype(np.uint8))] if C.get("ang") else [])
 [np.save(os.path.join(ed,f"{n}.npy"),np.ascontiguousarray(v)) for n,v in secs];open(os.path.join(ed,"manifest.tsv"),"w").write("".join(f"{n}\t{n}.npy\n" for n,v in secs));tot=sum(v.size for n,v in secs);return dict(total_digits=int(tot),digits_per_entity=round(tot/N,3),positions=int(D),cluster_table=int(sum(v.size for n,v in secs if n!="X")),sections={n:int(v.size) for n,v in secs},dir=ed)
def main():
 t0=time.time();sp={"dev":1,"test":2}[A.split];m=SPL==sp;hold=(H[m],T[m],R[m]);lock=os.path.join(TD,f"test_scored_{A.tag}_{A.clu}.lock");assert not(A.split=="test" and (os.path.exists(lock) or not A.final)),"test needs --final 1 and one scoring per tag"
 rng=np.random.default_rng(A.seed);C=fit(A.clu,X,rng,K,DU);log("fit",A.clu,"K",C["K"],"du",C["du"],"P0",P0,f"{time.time()-t0:.0f}s")
 res,L,tab,ka,sa,aux=pipe(C,X,hold,A.seed);res["cfg"]=dict(tag=A.tag,clu=A.clu,K=C["K"],tori=A.tori,split=A.split,seed=A.seed,D=D,P0=P0,du=C["du"]);log(json.dumps({k:v for k,v in res.items() if not isinstance(v,dict)}))
 if A.ctl:
  res["controls"]={}
  for nm,Xc,Cc in (("shuffled",shufflex(X,A.seed+11),C),("random_rays",np.random.default_rng(A.seed+12).integers(0,17,size=X.shape).astype(np.int32),C),("lift_only_kmodes",lift0(A.seed),fit("m2",lift0(A.seed),np.random.default_rng(A.seed+13),K,DU) if C["kind"] in ("m1","m2","a1") else C)):
   r_=pipe(Cc,Xc,hold,A.seed,0)[0];res["controls"][nm]={k:v for k,v in r_.items() if k in ("nmi_heldout","purity_heldout","nmi_train","rnmi_heldout","strata_nmi_heldout","auc_matched","auc_raw_matched","auc_prox_matched","auc_uniform","auc_pop_matched","recall_ge_2x_prior","strict_ge_2x_prior","balanced_ge_2x_prior")};log("control",nm,res["controls"][nm])
 sub=np.random.default_rng(A.seed+21)
 A.novel and res.update(novel_clusters=novel(C,ka,sa,sub),largest_clusters=novel(C,ka,sa,np.random.default_rng(A.seed+23),1),inverse=inverse(C,X,ka))
 A.comp and res.update(composition={v:compose(C,X,ka,np.random.default_rng(A.seed+22),hold,aux,v) for v in ("held","stored")})
 A.a2b and A.dir=="b" and res.update(a2b_labeled_rayonly=a2b(hold,aux))
 A.export and res.update(storage=export(C,L,tab))
 A.split=="test" and (open(lock,"w").write(json.dumps(res["cfg"])))
 json.dump(res,open(os.path.join(TD,A.out or f"resb_{A.tag}_{A.clu}_{A.split}.json"),"w"),indent=1);log("saved",f"{time.time()-t0:.0f}s")
if __name__=="__main__":main()
