import os,re,sys,json,time,shutil,argparse,tempfile,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import adam_nightly as AN
def has(x:str,gs:list)->bool:
 n=AN.NORM(x);return any(re.search(r"(?<!\S)"+re.escape(g)+r"(?!\S)",n) for g in gs)
def run(split:str,n:int,out:str,mode:str,noq:bool=False,shard:str="0/1")->dict:
 td=tempfile.mkdtemp(prefix="rknow_eval_");os.environ.update({k:v for k,v in AN.child_env(td).items() if k.startswith(("AMNI_","RAYON","OMP","MKL","OPENBLAS","NUMEXPR","TI_","POLARS"))});web=mode=="web";web and os.environ.update({"AMNI_LEARN_WEB":"1"});bl=[] if web else AN.netguard()
 from v19_eval_run import DECLINE
 from amni.serve import gguf_runtime as g
 import amni.compute.rknow as RK
 f=(lambda q:(g.chat(q,do_sample=False,user="rknow-dev",session="rknow-dev")["answer"],{})) if mode=="chat" else (lambda q:(lambda r:((r or {}).get("reply") or "I'm not sure about that one yet.",r or {}))(RK.ask(q)))
 i,k=map(int,shard.split("/"));D=json.loads(Path(split).read_text())[:n or None][i::k];R=[];t0=time.time()
 for j,x in enumerate(D):
  q=x["q"].rstrip("?").strip() if noq else x["q"];t=time.time();a,r=f(q);gs=[z for z in map(AN.NORM,x.get("answers",[])) if z];sc=AN.score_qa(a,x.get("answers",[]),DECLINE);al=bool(sc["hit"] or (r.get("span") and r.get("tier")!="none" and any(has(z,gs) for z in RK.alias(r["span"]))))
  R.append({"i":j,"q":q,"gold":x.get("answers",[]),"pred":a,"tier":r.get("tier","decline" if sc["decline"] else "chat"),"conf":r.get("conf"),"web":r.get("web"),**({"pre_tier":r["pre"]["tier"],"pre_hit":AN.score_qa(r["pre"]["reply"],x.get("answers",[]),DECLINE)["hit"]} if r.get("pre") else {}),**sc,"alias":al and not sc["decline"],"s":round(time.time()-t,2)})
 kk=max(1,len(R));ans=[r for r in R if not r["decline"]];T={t:{"n":sum(r["tier"]==t for r in R),"acc":round(sum(r["hit"] for r in R if r["tier"]==t)/max(1,sum(r["tier"]==t for r in R)),4),"acc_alias":round(sum(r["alias"] for r in R if r["tier"]==t)/max(1,sum(r["tier"]==t for r in R)),4)} for t in ("high","medium","low","none","decline","chat")}
 m={"split":split,"mode":mode,"noq":noq,"n":len(R),"acc":sum(r["hit"] for r in R)/kk,"acc_alias":sum(r["alias"] for r in R)/kk,"em":sum(r["em"] for r in R)/kk,"decline":sum(r["decline"] for r in R)/kk,"answered":len(ans),"answer_rate":len(ans)/kk,"committed_rate":sum(r["tier"] in ("high","medium","low") for r in R)/kk,"precision":sum(r["hit"] for r in ans)/max(1,len(ans)),"tiers":T,"s_per_q":round((time.time()-t0)/kk,2),"maxrss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"net_blocked":bl}
 Path(out).write_text(json.dumps({**m,"items":R},indent=0));shutil.rmtree(td,ignore_errors=True);return m
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--split",default=str(ROOT/"data"/"concepts"/"eval"/"rknow"/"nq_dev_tune.json"));ap.add_argument("--n",type=int,default=0);ap.add_argument("--mode",default="chat",choices=["chat","rknow","web"]);ap.add_argument("--noq",action="store_true");ap.add_argument("--shard",default="0/1");ap.add_argument("--out",required=True);a=ap.parse_args();print(json.dumps(run(a.split,a.n,a.out,a.mode,a.noq,a.shard)))
