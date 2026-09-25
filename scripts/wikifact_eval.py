import os,re,sys,json,time,argparse,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as AN
from amni.compute import eval_lock as EL
from amni.compute import wikifact as WF
DEV=ROOT/"data"/"concepts"/"eval"/"rknow";OUT=ROOT/"data"/"wikifact"/"v2"/"eval";CODE=[ROOT/"amni"/"compute"/"wikifact.py",Path(__file__).resolve()]
def score(a:str|None,forms:list,golds:list)->dict:
 gs={g for g in map(AN.NORM,golds) if g};n=AN.NORM(a or "")
 return {"strict":bool(a) and n in gs,"alias":bool(a) and any(AN.NORM(f) in gs for f in [a,*forms] if f),"hit":bool(a) and any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",n) for g in gs)}
def summ(R:list,thr:float)->dict:
 n=max(1,len(R));A=[r for r in R if r["answer"] and r["conf"]>=thr and (WF.TEMPORAL or not r["temporal"])];k=max(1,len(A))
 return {"n":len(R),"answered":len(A),"coverage":round(len(A)/n,4),"precision_strict":round(sum(r["strict"] for r in A)/k,4),"precision_alias":round(sum(r["alias"] for r in A)/k,4),"precision_hit":round(sum(r["hit"] for r in A)/k,4),"wrong_rate_alias":round(sum(not r["alias"] for r in A)/n,4),"correct_rate_alias":round(sum(r["alias"] for r in A)/n,4),"wrong_rate_hit":round(sum(not r["hit"] for r in A)/n,4),"correct_rate_hit":round(sum(r["hit"] for r in A)/n,4)}
def run(splits:list,thr:float,out:str,rk:bool=False,limit:int=0)->dict:
 bl=AN.netguard();RK=None
 if rk:
  from v19_eval_run import DECLINE
  import amni.compute.rknow as RK
 M={};t0=time.time()
 for sp in splits:
  D=json.loads(Path(sp).read_text())[:limit or None];R=[]
  for i,x in enumerate(D):
   t=time.time();r=WF.ask(x["q"]) or {};a=r.get("answer");R.append({"i":i,"q":x["q"],"gold":x["answers"],"answer":a,"conf":float(r.get("conf") or 0),"rule":r.get("rule"),"source":r.get("source"),"temporal":bool(r.get("temporal")),**score(a,r.get("forms") or [],x["answers"]),"s":round(time.time()-t,3)})
   if RK:
    y=RK.ask(x["q"]) or {};p=y.get("reply") or "I'm not sure about that one yet.";sc=AN.score_qa(p,x["answers"],DECLINE);w=R[-1]["answer"] and R[-1]["conf"]>=thr and (WF.TEMPORAL or not R[-1]["temporal"])
    R[-1].update({"rk_reply":p[:200],"rk_tier":y.get("tier"),"rk_hit":sc["hit"],"rk_decline":sc["decline"],"comb_hit":R[-1]["hit"] if w else sc["hit"],"comb_decline":False if w else sc["decline"]})
  s=summ(R,thr);s["by_rule"]={k:summ([r for r in R if (r["rule"] or "").split(":")[0]==k],thr) for k in sorted({(r["rule"] or "").split(":")[0] for r in R if r["rule"]})};s["sweep"]={str(t):summ(R,t) for t in (0.3,0.4,0.5,0.6,0.7,0.8,0.9)};s["temporal"]={"n":sum(r["temporal"] and bool(r["answer"]) and r["conf"]>=thr for r in R),"hit":sum(r["temporal"] and bool(r["answer"]) and r["conf"]>=thr and r["hit"] for r in R)};s["s_per_q"]=round(sum(r["s"] for r in R)/max(1,len(R)),4)
  if RK:n=max(1,len(R));s["rknow"]={"acc":round(sum(r["rk_hit"] for r in R)/n,4),"answer_rate":round(sum(not r["rk_decline"] for r in R)/n,4)};s["combined"]={"acc":round(sum(r["comb_hit"] for r in R)/n,4),"answer_rate":round(sum(not r["comb_decline"] for r in R)/n,4)}
  M[Path(sp).stem]={**s,"items":R}
 rep={"thr":thr,"store":WF.store().man if WF.store() else None,"splits":{k:{a:b for a,b in v.items() if a!="items"} for k,v in M.items()},"items":{k:v["items"] for k,v in M.items()},"maxrss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"sec":round(time.time()-t0,1),"net_blocked":bl}
 EL.record(out,rep,code_files=CODE,splits=[Path(s) for s in splits],kind="dry");return rep["splits"]
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--splits",nargs="+",default=[str(DEV/"nq_dev_tune.json"),str(DEV/"nq_dev_hold.json")]);ap.add_argument("--thr",type=float,default=WF.THR);ap.add_argument("--out",default=str(OUT/"wikifact_dev_dry.json"));ap.add_argument("--rknow",action="store_true");ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
 print(json.dumps({k:{x:y for x,y in v.items() if x not in ("sweep","by_rule")} for k,v in run(a.splits,a.thr,a.out,a.rknow,a.limit).items()},indent=1))
