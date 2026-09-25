import os,re,sys,json,time,glob,pickle,argparse
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as AN
import amni.compute.rknow as RK
EV=ROOT/"data"/"concepts"/"eval"/"rknow"
def has(x:str,gs:list)->bool:
 n=AN.NORM(x);return any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",n) for g in gs)
def dump(split:str,out:str,shard:str,noq:bool=False)->int:
 bl=AN.netguard();i,n=map(int,shard.split("/"));D=json.loads(Path(split).read_text())[i::n];T=RK.RL._kx()[1];Z=[]
 for x in D:
  q=x["q"].rstrip("?").strip() if noq else x["q"];skip=not RK.askq(q);qt,O,_=RK.occ(q) if not skip else ("skip",[],[]);R=RK.feats(qt,O,T,q=q) if O else [];gs=[g for g in map(AN.NORM,x.get("answers",[])) if g]
  Z.append({"q":q,"gold":x.get("answers",[]),"qt":qt,"R":[{"v":z["v"],"sp":RK.span(qt,z["v"],T),"t":z["t"],"x":z["x"],"y":has(RK.span(qt,z["v"],T),gs),"y0":has(z["v"],gs),"ya":has(RK.span(qt,z["v"],T),gs) or (qt in ("person","place","other") and any(has(a,gs) for a in RK.alias(z["v"],T)))} for z in R]})
 pickle.dump(Z,open(out,"wb"));return len(Z)
def load(pat:str)->list:return [z for f in sorted(glob.glob(pat)) for z in pickle.load(open(f,"rb"))]
def mat(Z:list)->tuple:
 X=np.array([[c["x"][k] for k in RK.FEAT] for z in Z for c in z["R"]],np.float64);y=np.array([c["y"] for z in Z for c in z["R"]],np.float64);q=np.array([i for i,z in enumerate(Z) for _ in z["R"]]);return X,y,q
def fit(X:np.ndarray,y:np.ndarray,l2:float=1e-2,it:int=3000,lr:float=0.5)->np.ndarray:
 w=np.zeros(X.shape[1]);n=len(y)
 for _ in range(it):p=1/(1+np.exp(-np.einsum("ij,j->i",X,w)));g=np.einsum("ij,i->j",X,p-y)/n+l2*np.r_[0,w[1:]];w-=lr*g
 return w
def ranked(z:dict,w:np.ndarray)->list:return sorted([{**c,"s":float(1/(1+np.exp(-sum(wi*c["x"][k] for wi,k in zip(w,RK.FEAT)))))} for c in z["R"]],key=lambda c:-c["s"])
def qrows(Z:list,w:np.ndarray)->list:return [(np.array([RK.qfeat(z["qt"],R,z["q"])[k] for k in RK.CF]),bool(R[0]["y"]),z) for z in Z for R in [ranked(z,w)] if R]
def curve(S:np.ndarray,H:np.ndarray,n:int)->list:
 o=np.argsort(-S);return [{"answered":k,"precision":round(float(H[o[:k]].mean()),4),"acc":round(float(H[o[:k]].sum())/n,4),"th":round(float(S[o[k-1]]),4)} for k in (5,10,15,20,30,40,60,80,120,160,240) if k<=len(S)]
def oof(Z:list,l2:float,k:int=5)->list:return [r for f in range(k) for r in qrows([z for i,z in enumerate(Z) if i%k==f],fit(*mat([z for i,z in enumerate(Z) if i%k!=f])[:2],l2))]
def evalw(Z:list,w:np.ndarray,v:np.ndarray)->dict:
 Q=qrows(Z,w);S=np.array([1/(1+np.exp(-(a@v))) for a,_,_ in Q]);H=np.array([h for _,h,_ in Q]);return {"n":len(Z),"top1":int(H.sum()),"oracle":sum(any(c["y"] for c in z["R"]) for z in Z),"curve":curve(S,H,len(Z)),"at":{str(t):{"answered":int((S>=t).sum()),"correct":int(H[S>=t].sum()),"precision":round(float(H[S>=t].mean()),4) if (S>=t).any() else None} for t in (0.4,0.5,0.6,0.7,0.8)}}
FX=list(RK.FEAT)+["hnv","hng","hna","gq"]+(["yr0","ymin","ymax","qf","ql"] if os.environ.get("RKNOW_YF","1")=="1" else []);REL=["s0","cov","fx","share","pw","lc","sup","n","fw","gq"]
def gmat(Z:list,lab:str="y")->tuple:
 m={"feat":FX,"rel":REL};Q=[(i,z) for i,z in enumerate(Z) if z["R"]];return np.vstack([RK.gbx(z["R"],m) for _,z in Q]),np.array([c[lab] for _,z in Q for c in z["R"]],float),np.array([i for i,z in Q for _ in z["R"]])
def pdump(split:str,out:str,shard:str,n:int=0,k:int=200)->int:
 bl=AN.netguard();i,m=map(int,shard.split("/"));D=json.loads(Path(split).read_text())[:n or None][i::m];I=RK.ix();X,Y,Q,G=[],[],[],[]
 for j,x in enumerate(D):
  q=x["q"].rstrip("?").strip();gs=[g for g in map(AN.NORM,x.get("answers",[])) if g];h=I.search(q,k)
  if not h or not RK.askq(q):continue
  F=I.pfeat(q,h);X.append(F);Y+=[has(I.text(p,I.rc)[1],gs) for _,p in h];Q+=[j]*len(h);G+=[p for _,p in h]
 np.savez_compressed(out,X=np.vstack(X) if X else np.zeros((0,len(RK.PRF))),y=np.array(Y,bool),q=np.array(Q,np.int32),p=np.array(G,np.int64),n=len(D));return len(set(Q))
def pload(pat:str)->tuple:
 Z=[np.load(f) for f in sorted(glob.glob(pat))];off=np.cumsum([0]+[int(z["n"]) for z in Z]);return np.vstack([z["X"] for z in Z]),np.concatenate([z["y"] for z in Z]),np.concatenate([z["q"]+o for z,o in zip(Z,off)]),int(off[-1])
def precall(S:np.ndarray,y:np.ndarray,q:np.ndarray,nq:int,ks=(1,5,10,30,50,100,200))->dict:
 o=np.lexsort((-S,q));qq,yy=q[o],y[o];st=np.r_[0,np.flatnonzero(np.diff(qq))+1];rk=np.arange(len(qq))-np.repeat(st,np.diff(np.r_[st,len(qq)]));fr={}
 for a,b,c in zip(qq,yy,rk):b and a not in fr and fr.__setitem__(a,c)
 return {f"R@{k}":round(sum(v<k for v in fr.values())/nq,4) for k in ks}
def pfit(tr:str,te:str,out:str,bag:int=3,it:int=300,leaf:int=31)->dict:
 from sklearn.ensemble import HistGradientBoostingClassifier as HG
 X,y,q,_=pload(tr);hs=[HG(max_iter=it,max_leaf_nodes=leaf,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=100,random_state=i).fit(X,y) for i in range(bag)];m=export(hs,feat=list(RK.PRF),rel=[]);m.update({kk:np.asarray(m[kk]) for kk in ("f","t","l","r","v","roots")});r={}
 for t in te.split(","):Xt,yt,qt,nq=pload(t);r[t]={"base":precall(-np.log1p(np.arange(len(qt)))+0*qt,yt,qt,nq),"prr":precall(RK.gbp(Xt,m),yt,qt,nq)}
 out and json.dump({kk:(v.tolist() if isinstance(v,np.ndarray) else v) for kk,v in m.items()},open(out,"w"),separators=(",",":"));return r
def wnx(out:str)->int:
 import nltk;nltk.data.path.insert(0,str(ROOT/"data"/"concepts"/"nltk_data"));from nltk.corpus import wordnet as wn;I=RK.ix();K={};ok=lambda w:I.lookup(w)>=0 and I.idf(w)>=2.5
 L={l.name().lower() for s in wn.all_synsets() for l in s.lemmas() if l.name().isalpha()}
 for w in L|{x for m in wn._exception_map.values() for x in m}|{f for l in L for f in (l+"ed",l+"d",l+"ing",l[:-1]+"ing",l[:-1]+"ied",l+l[-1:]+"ed",l+l[-1:]+"ing",l+"er",l+"ers") if I.lookup(f)>=0}:
  t=RK.toks(w)
  if len(t)!=1:continue
  b={m for p in "nv" for m in [wn.morphy(w,p)] if m and m.isalpha()};V=b|{d.name().lower() for m in b for s in wn.synsets(m)[:2] for l in s.lemmas() if l.name().lower()==m for d in l.derivationally_related_forms() if d.name().isalpha() and d.synset().pos()!=s.pos()}
  V=[z for z in dict.fromkeys(y for v in sorted(V,key=lambda v:(v not in b,len(v))) for y in RK.toks(v)) if z!=t[0] and ok(z)][:4];V and K.setdefault(t[0],V)
 json.dump(K,open(out,"w"),separators=(",",":"),sort_keys=True);return len(K)
def export(hs:list,base:dict={},feat:list=None,rel:list=None)->dict:
 F,T,Lf,Rt,V,ro,off,dep=[],[],[],[],[],[],0,0
 for h in hs:
  for p in h._predictors:nd=p[0].nodes;ro.append(off);F+=[-1 if lf else int(x) for x,lf in zip(nd["feature_idx"],nd["is_leaf"])];T+=nd["num_threshold"].tolist();Lf+=(nd["left"].astype(np.int64)+off).tolist();Rt+=(nd["right"].astype(np.int64)+off).tolist();V+=(nd["value"]/len(hs)).tolist();dep=max(dep,int(nd["depth"].max()));off+=len(nd)
 return {**base,"feat":feat or FX,"rel":REL if rel is None else rel,"base":float(np.mean([np.ravel(h._baseline_prediction)[0] for h in hs])),"f":F,"t":T,"l":Lf,"r":Rt,"v":V,"roots":ro,"depth":dep}
def gfit(X:np.ndarray,y:np.ndarray,bag:int=3)->dict:
 from sklearn.ensemble import HistGradientBoostingClassifier as HG
 m=export([HG(max_iter=400,max_leaf_nodes=31,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=40,random_state=i).fit(X,y) for i in range(bag)]);m.update({k:np.asarray(m[k]) for k in ("f","t","l","r","v","roots")});return m
def granked(Z:list,m:dict)->list:return [(z,sorted([{**c,"s":float(s)} for c,s in zip(z["R"],RK.gbp(RK.gbx(z["R"],m),m))],key=lambda c:-c["s"])) for z in Z if z["R"]]
def tiers(Z:list,m:dict,cw:np.ndarray,c:float,c2:float,lab:str="y")->dict:
 T={"high":[0,0],"medium":[0,0],"low":[0,0]};n=len(Z)
 for z,R in granked(Z,m):f=RK.qfeat(z["qt"],R,z["q"]);p=1/(1+np.exp(-np.dot(cw,[f[k] for k in RK.CF])));vt=RK.veto(R[0],f,z["q"]);t="high" if p>=c and not vt else "medium" if p>=c2 and not vt else "low";T[t][0]+=1;T[t][1]+=bool(R[0][lab])
 return {"n":n,"top1":round(sum(v[1] for v in T.values())/n,4),**{k:{"n":v[0],"acc":round(v[1]/max(1,v[0]),4)} for k,v in T.items()}}
def gbfit(tr:list,te:dict,out:str,lab:str="ya",l2c:float=1e-2,k:int=5,pr1:float=0.85,pr2:float=0.5)->dict:
 X,y,q=gmat(tr,lab);m=gfit(X,y);Q=[i for i,z in enumerate(tr) if z["R"]];oof=[r for f in range(k) for r in granked([tr[i] for i in Q if i%k==f],gfit(*[a[np.isin(q,[i for i in Q if i%k!=f])] for a in (X,y)]))]
 A=np.array([[RK.qfeat(z["qt"],R,z["q"])[kk] for kk in RK.CF] for z,R in oof]);H=np.array([bool(R[0]["y"]) for z,R in oof],float);V=np.array([RK.veto(R[0],RK.qfeat(z["qt"],R,z["q"]),z["q"]) for z,R in oof]);cw=fit(A,H,l2c,6000);P=1/(1+np.exp(-np.einsum("ij,j->i",A,cw)))
 pick=lambda pr:min([t for t in np.round(np.arange(0.05,0.96,0.01),2) if (~V&(P>=t)).sum()>=20 and H[~V&(P>=t)].mean()>=pr] or [0.95]);c,c2=pick(pr1),pick(pr2)
 r={"c":float(c),"c2":float(c2),"curve":[(float(t),int((~V&(P>=t)).sum()),round(float(H[~V&(P>=t)].mean()),4)) for t in np.round(np.arange(0.1,0.95,0.05),2) if (~V&(P>=t)).any()],"cw":[round(float(x),4) for x in cw],"lab":lab,"oof":tiers([z for z,_ in oof],m,cw,c,c2),"test":{s:tiers(Z,m,cw,c,c2) for s,Z in te.items()}}
 out and json.dump({**{kk:(v.tolist() if isinstance(v,np.ndarray) else v) for kk,v in m.items()},**{kk:r[kk] for kk in ("c","c2","cw","lab")}},open(out,"w"),separators=(",",":"));return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["dump","fit","gbfit","pdump","pfit","wnx"]);ap.add_argument("--n",type=int,default=0);ap.add_argument("--split");ap.add_argument("--dump");ap.add_argument("--shard",default="0/1");ap.add_argument("--train");ap.add_argument("--test");ap.add_argument("--l2",type=float,default=3e-4);ap.add_argument("--l2c",type=float,default=1e-2);ap.add_argument("--oof",type=int,default=1);ap.add_argument("--out");ap.add_argument("--noq",action="store_true");a=ap.parse_args()
 if a.cmd=="dump":print(json.dumps(dump(a.split,a.dump,a.shard,a.noq)));return
 if a.cmd=="pdump":print(json.dumps(pdump(a.split,a.dump,a.shard,a.n)));return
 if a.cmd=="wnx":print(wnx(a.out));return
 if a.cmd=="pfit":print(json.dumps(pfit(a.train,a.test,a.out),indent=1));return
 if a.cmd=="gbfit":r=gbfit(load(a.train),{t:load(t) for t in a.test.split(",")},a.out);print(json.dumps({k:v for k,v in r.items() if k!="cw"},indent=1));return
 Ztr,Zte=load(a.train),load(a.test);X,y,_=mat(Ztr);w=fit(X,y,a.l2);Q=oof(Ztr,a.l2) if a.oof else qrows(Ztr,w);v=fit(np.array([f for f,_,_ in Q]),np.array([h for _,h,_ in Q],float),a.l2c,6000);r={"feat":RK.FEAT,"w":[round(float(x),4) for x in w],"cf":RK.CF,"v":[round(float(x),4) for x in v],"l2":a.l2,"l2c":a.l2c,"train":evalw(Ztr,w,v),"test":evalw(Zte,w,v)};a.out and json.dump(r,open(a.out,"w"),indent=1)
 print(json.dumps({k:{"n":r[k]["n"],"top1":r[k]["top1"],"oracle":r[k]["oracle"],"curve":[(c["answered"],c["precision"],c["th"]) for c in r[k]["curve"]],"at":r[k]["at"]} for k in ("train","test")}))
if __name__=="__main__":main()
