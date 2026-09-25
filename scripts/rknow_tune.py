import os,re,sys,json,time,glob,pickle,argparse
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as AN
import amni.compute.rknow as RK
EV=ROOT/"data"/"concepts"/"eval"/"rknow"
def has(x:str,gs:list)->bool:
 n=AN.NORM(x);return any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",n) for g in gs)
def dump(split:str,out:str,shard:str)->int:
 bl=AN.netguard();i,n=map(int,shard.split("/"));D=json.loads(Path(split).read_text())[i::n];T=RK.RL._kx()[1];Z=[]
 for x in D:
  skip=not RK.factq(x["q"]);qt,O,_=RK.occ(x["q"]) if not skip else ("skip",[],[]);R=RK.feats(qt,O,T,q=x["q"]) if O else [];gs=[g for g in map(AN.NORM,x.get("answers",[])) if g]
  Z.append({"q":x["q"],"gold":x.get("answers",[]),"qt":qt,"R":[{"v":z["v"],"sp":RK.span(qt,z["v"],T),"t":z["t"],"x":z["x"],"y":has(RK.span(qt,z["v"],T),gs),"y0":has(z["v"],gs)} for z in R]})
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
def evalw(Z:list,w:np.ndarray,v:np.ndarray)->dict:
 Q=qrows(Z,w);S=np.array([1/(1+np.exp(-(a@v))) for a,_,_ in Q]);H=np.array([h for _,h,_ in Q]);return {"n":len(Z),"top1":int(H.sum()),"oracle":sum(any(c["y"] for c in z["R"]) for z in Z),"curve":curve(S,H,len(Z)),"at":{str(t):{"answered":int((S>=t).sum()),"correct":int(H[S>=t].sum()),"precision":round(float(H[S>=t].mean()),4) if (S>=t).any() else None} for t in (0.4,0.5,0.6,0.7,0.8)}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["dump","fit"]);ap.add_argument("--split");ap.add_argument("--dump");ap.add_argument("--shard",default="0/1");ap.add_argument("--train");ap.add_argument("--test");ap.add_argument("--l2",type=float,default=3e-4);ap.add_argument("--l2c",type=float,default=1e-2);ap.add_argument("--out");a=ap.parse_args()
 if a.cmd=="dump":print(json.dumps(dump(a.split,a.dump,a.shard)));return
 Ztr,Zte=load(a.train),load(a.test);X,y,_=mat(Ztr);w=fit(X,y,a.l2);Q=qrows(Ztr,w);v=fit(np.array([f for f,_,_ in Q]),np.array([h for _,h,_ in Q],float),a.l2c,6000);r={"feat":RK.FEAT,"w":[round(float(x),4) for x in w],"cf":RK.CF,"v":[round(float(x),4) for x in v],"l2":a.l2,"l2c":a.l2c,"train":evalw(Ztr,w,v),"test":evalw(Zte,w,v)};a.out and json.dump(r,open(a.out,"w"),indent=1)
 print(json.dumps({k:{"n":r[k]["n"],"top1":r[k]["top1"],"oracle":r[k]["oracle"],"curve":[(c["answered"],c["precision"],c["th"]) for c in r[k]["curve"]],"at":r[k]["at"]} for k in ("train","test")}))
if __name__=="__main__":main()
