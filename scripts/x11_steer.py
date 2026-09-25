import argparse,json,os,sys,time,warnings
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from x0_code_eval import DATA,load_jsonl,selfchecks,problem_prompt,load_ds
from amni.compute import code_steer as CS,tile_synth as TS,eval_lock
TRAIN=("mbpp_holdout","apps_fn_dev","apps_fn_lock")
def data(ex=""):
 warnings.simplefilter("ignore");os.environ["X0_SAMPLE"]=ex;xs={r["id"] for r in load_ds("x11_dev")} if ex else set();rows=[r for f in TRAIN for r in load_jsonl(f) if r["id"] not in xs];pr=lambda r:problem_prompt(r,"code");return CS.fit_rows(rows,pr,lambda r:selfchecks(pr(r)))
def cover(S,Y,ks=(8,12,16,24)):
 o=np.argsort(-S,1);out={}
 for k in ks:top=np.zeros_like(Y);np.put_along_axis(top,o[:,:k],True,1);out[f"recall@{k}"]=round(float((top&Y).sum()/Y.sum()),4);out[f"full@{k}"]=round(float(((top|~Y).all(1)).mean()),4)
 return out
def cv(a):
 X,Y,ids=data();n=len(X);g=np.random.default_rng(17).permutation(n)%5;S=np.zeros(Y.shape);P=np.zeros(Y.shape);T=CS.targets()
 for f in range(5):
  tr=np.flatnonzero(g!=f);te=np.flatnonzero(g==f);m=CS.train([X[i] for i in tr],Y[tr],it=a.it,lr=a.lr,l2=a.l2,pw=a.pw)
  for i in te:S[i]=CS.score(m,X[i])
 freq=np.tile(Y.mean(0),(n,1))
 res={"n":n,"targets":len(T),"mean_pos":float(Y.sum(1).mean()),"steer_cv":cover(S,Y),"freq_baseline":cover(freq+1e-9*np.random.default_rng(1).random(freq.shape),Y),"cfg":vars(a)}
 print(json.dumps(res,indent=1));return res
def fit(a):X,Y,ids=data(a.exclude);m=CS.train(X,Y,it=a.it,lr=a.lr,l2=a.l2,pw=a.pw);m.save(a.out);print(json.dumps({"n":len(X),"saved":a.out,"sha":eval_lock.sha256(a.out)}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["cv","fit"]);ap.add_argument("--it",type=int,default=120);ap.add_argument("--lr",type=float,default=0.3);ap.add_argument("--l2",type=float,default=0.02);ap.add_argument("--pw",type=float,default=3.0);ap.add_argument("--out",default=str(CS.MODEL));ap.add_argument("--exclude",default="");a=ap.parse_args()
 cv(a) if a.cmd=="cv" else fit(a)
if __name__=="__main__":main()
