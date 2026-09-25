import os,re,sys,json,time,argparse,resource
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as AN
import amni.compute.rknow as RK
from v19_eval_run import DECLINE
KS=(1,5,10,30,100)
def has(x:str,gs:list)->bool:
 n=AN.NORM(x);return any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",n) for g in gs)
def cover(I,q:str,golds:list)->tuple:
 qw=[w for w in dict.fromkeys(RK.toks(q)) if w not in RK.QSTOP];W={w:I.idf(w) for w in qw};tot=sum(W.values()) or 1.0;gs=[g for g in map(AN.NORM,golds) if g];loose=strict=False
 for g in golds:
  at=[I.lookup(w) for w in dict.fromkeys(RK.toks(g))]
  if not at or min(at)<0:continue
  P=None
  for j in sorted(at,key=lambda j:int(I.df[j])):p,_=I.post(j);P=p if P is None else P[np.isin(P,p,assume_unique=True)]
  if not len(P):continue
  loose=True;c=np.zeros(len(P))
  for w in qw:
   j=I.lookup(w)
   if j>=0:p,_=I.post(j);c+=W[w]*np.isin(P,p,assume_unique=True)
  o=np.argsort(-c)[:60];cache={}
  if any(c[i]/tot>=0.5 and has(I.text(int(P[i]),cache)[1],gs) for i in o):strict=True;break
 return loose,strict
def run(split:str,n:int,out:str,old:bool)->dict:
 os.environ.update({"AMNI_LEARN_WEB":"0"});bl=AN.netguard();I=RK.ix();D=json.loads(Path(split).read_text())[:n or None];R=[];t0=time.time()
 import amni.compute.ray_learn as RL
 for x in D:
  q,golds=x["q"],x["answers"];gs=[g for g in map(AN.NORM,golds) if g];lo,st=cover(I,q,golds);h=I.search(q,max(KS));cache={};rk=next((i+1 for i,(_,p) in enumerate(h) if has(I.text(p,cache)[1],gs)),None)
  od=next((i+1 for i,(t,txt,_,_) in enumerate(RL.local(q)) if has(txt,gs)),None) if old else None
  qt,C,_=RK.candidates(q);ci=next((i+1 for i,z in enumerate(C) if has(z["v"],gs)),None);a=RK.ask(q);pred=a["reply"] if a else "I'm not sure about that one yet."
  R.append({"q":q,"gold":golds,"qtype":qt,"loose":lo,"cover":st,"rank":rk,"old_rank":od,"cand_rank":ci,"top":C[0]["v"] if C else None,"top_title":C[0]["t"] if C else None,"gate":RK.factq(q),"conf":(RK.decide(qt,C,q=q) or {}).get("conf") if C else None,"pred":pred,**AN.score_qa(pred,golds,DECLINE)})
 k=max(1,len(R));f=lambda c:round(sum(1 for r in R if c(r))/k,4);ans=[r for r in R if not r["decline"]]
 m={"split":split,"n":len(R),"coverage_loose":f(lambda r:r["loose"]),"coverage":f(lambda r:r["cover"]),**{f"R@{kk}":f(lambda r,kk=kk:r["rank"] is not None and r["rank"]<=kk) for kk in KS},**({"old_R@24":f(lambda r:r["old_rank"] is not None)} if old else {}),"gate_pass":f(lambda r:r["gate"]),"cand_top1":f(lambda r:r["cand_rank"]==1),"cand_top5":f(lambda r:r["cand_rank"] is not None and r["cand_rank"]<=5),"acc":f(lambda r:r["hit"]),"em":f(lambda r:r["em"]),"decline":f(lambda r:r["decline"]),"answered":len(ans),"precision":round(sum(r["hit"] for r in ans)/max(1,len(ans)),4),"qtypes":dict(__import__("collections").Counter(r["qtype"] for r in R)),"s_per_q":round((time.time()-t0)/k,2),"maxrss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"net_blocked":bl}
 Path(out).write_text(json.dumps({**m,"items":R},indent=0));return m
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--split",default=str(ROOT/"data"/"concepts"/"eval"/"rknow"/"nq_dev_tune.json"));ap.add_argument("--n",type=int,default=0);ap.add_argument("--old",action="store_true");ap.add_argument("--out",required=True);a=ap.parse_args();print(json.dumps(run(a.split,a.n,a.out,a.old)))
