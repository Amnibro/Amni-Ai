import os,re,sys,json,time,shutil,argparse,tempfile,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
DEV=ROOT/"data"/"adam_t"/"retrieve"/"dev.jsonl";SRC=ROOT/"data"/"adam_t"/"eval"/"chat"/"chat_src.jsonl";BUDGET=2928
CHT=lambda s:re.findall(r"[a-z0-9]+(?:'[a-z]+)?",(s or "").lower())
g6=lambda r:{tuple(t[i:i+6]) for s in [x for u in r.get("turns",[]) for x in u]+[r["q"]] for t in [CHT(s)] for i in range(len(t)-5)}
def overlap()->dict:
 S=set().union(*map(g6,map(json.loads,open(SRC))));D=[json.loads(l) for l in open(DEV)];H=[(r["id"],sorted(" ".join(g) for g in g6(r)&S)) for r in D];H=[h for h in H if h[1]]
 return {"dev":len(D),"src_6grams":len(S),"overlap_items":len(H),"hits":H}
def run(out:str,split:str="")->dict:
 d=tempfile.mkdtemp(prefix="retr_umem_");[os.environ.pop(k) for k in [k for k in os.environ if re.match(r"AMNI_(?:ADAM_T(?!_KW$)|RKNOW|MATH|UMEM|USER_MEMORY|WIKIFACT)",k)]];os.environ.update(AMNI_UMEM_DIR=d)
 from amni.serve import adam_t_client as C,user_memory as U;from amni.compute import wikifact as WF;from adam_nightly import score_qa
 NODEC=re.compile(r"(?!x)x");hit=lambda X,g:bool(X) and score_qa(" ".join(X),g,NODEC)["hit"];cap={};oa=WF.ask;WF.ask=lambda q:cap.__setitem__("w",oa(q)) or cap["w"]
 D=[r for r in map(json.loads,open(DEV)) if not split or r["split"]==split];C.warm();O=[]
 for r in D:
  u=f"retr-{r['id']}";h=[tuple(t) for t in r["turns"]];[U.turn(q,h[:i],u,u) for i,(q,_) in enumerate(h)];cap.clear();t=time.perf_counter();c=C.context(r["q"],h,u,u,BUDGET);ms=(time.perf_counter()-t)*1000
  K=[x for x in c.get("rc",[]) if not x.startswith("Calculator: ")];w=cap.get("w") or {};wd=next((k for k in K if w.get("answer") and k.endswith(": "+C.asc(w["answer"])+".")),"")
  O.append({"id":r["id"],"cat":r["cat"],"split":r["split"],"q":r["q"],"gold":r["gold"],"fired":bool(K),"hit":hit(K,r["gold"]),"wd":wd,"wd_hit":hit([wd] if wd else [],r["gold"]),"psg_hit":hit([k for k in K if k!=wd],r["gold"]),"mem":bool(c.get("pre")),"ms":round(ms,1),"n":c.get("n",{}),"rq":c.get("rq",""),"know":K})
 shutil.rmtree(d,ignore_errors=True);R={"summary":summ(O),"items":O};out and Path(out).write_text(json.dumps(R,indent=1));return R["summary"]
def summ(O:list)->dict:
 f=lambda X,k:round(sum(bool(x[k]) for x in X)/len(X),4) if X else None;md=lambda X:{"median_ms":round(float(np.median([x["ms"] for x in X])),1),"p90_ms":round(float(np.percentile([x["ms"] for x in X],90)),1)} if X else {}
 def one(X):
  k=[x for x in X if x["cat"]=="know"];m=[x for x in X if x["cat"]=="multi"];p=[x for x in X if x["cat"]=="personal"];g=[x for x in X if x["cat"]=="gate"];c=[x for x in k if x["id"][0]=="c"];kd=[x for x in k+m if x["fired"]];wd=[x for x in k+m if x["wd"]]
  return {"n":len(X),"know":{"n":len(k),"attached":f(k,"fired"),"gold_in_ctx":f(k,"hit"),"gold_if_attached":f([x for x in k if x["fired"]],"hit"),"wd_emitted":f(k,"wd"),"wd_precision":f(wd,"wd_hit"),"psg_hit":f(k,"psg_hit"),"common":{"n":len(c),"attached":f(c,"fired"),"gold_in_ctx":f(c,"hit")}},"multi":{"n":len(m),"attached":f(m,"fired"),"gold_in_ctx":f(m,"hit"),"junk":round(sum(x["fired"] and not x["hit"] for x in m)/len(m),4) if m else None},"personal":{"n":len(p),"junk":f(p,"fired")},"multi_junk_all":round((sum(x["fired"] and not x["hit"] for x in m)+sum(x["fired"] for x in p))/max(1,len(m)+len(p)),4),"gate_false_fire":f(g,"fired"),"gold_if_attached_all":f(kd,"hit"),"latency":{"all":md(X),"fired":md([x for x in X if x["fired"]])}}
 return {"all":one(O),**{s:one([x for x in O if x["split"]==s]) for s in ("tune","hold")}}
def compare(a:str,b:str)->dict:
 A,B=({x["id"]:x for x in json.load(open(p))["items"]} for p in (a,b));return {"gained":[(i,B[i]["q"]) for i in B if B[i]["hit"] and not A.get(i,{}).get("hit")],"lost":[(i,B[i]["q"],A[i]["know"][:1]) for i in B if A.get(i,{}).get("hit") and not B[i]["hit"]]}
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=("overlap","run","compare"));ap.add_argument("--out",default="");ap.add_argument("--split",default="");ap.add_argument("files",nargs="*");a=ap.parse_args()
 print(json.dumps(overlap() if a.cmd=="overlap" else run(a.out,a.split) if a.cmd=="run" else compare(*a.files),indent=1))
