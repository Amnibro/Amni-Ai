import os,sys,json,time,numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));H=os.environ.get("AMNI_HOME",_H0);sys.path[:0]=[_H0,os.path.join(_H0,"scripts"),os.path.join(H,"scripts")]
import adam_ray
import v19_eval_run as ev
P=sys.argv[1];P=P if P.endswith("_sid.npy") else P+"_sid.npy";P=P if os.path.isabs(P) or os.path.exists(P) else os.path.join(H,P);TAG=sys.argv[2] if len(sys.argv)>2 else "v19_rust";N=int(os.environ.get("EVAL_N","0"));CTX=int(os.environ.get("CTX","1048576"));B=os.path.join(H,"data","big2")
e=adam_ray.RayField(P,os.path.join(B,"synsets.txt"),os.path.join(B,"words.txt"),os.path.join(B,"wordcounts.txt"),os.environ.get("AMNI_RAY_ARCH","cpu"),H).load()
def gen(prompt:str,maxb:int=90)->str:
 s=prompt.encode("ascii","replace")[-(CTX-maxb):];lg=e.prime(s)[0];out=b""
 for _ in range(maxb):
  c=int(np.argmax(lg))
  if c==10 or c==30:break
  out+=bytes([c]);lg=e._advance([0],[c],True)[0]
 return out.decode("ascii","replace").strip()
def chat3(n:int)->dict:
 rows=[json.loads(l) for l in open(os.path.join(H,"data","chat3","hold_eval.jsonl"))];rows=rows[:n] if n else rows;out=[];by={}
 for r in rows:
  p=gen(r["dialogue"]);h=ev.score_contains(p,r["answer"]);out.append({"kind":r.get("kind"),"pred":p,"answer":r["answer"],"hit":h});k=by.setdefault(r.get("kind"),[0,0]);k[0]+=int(h);k[1]+=1
 return {"suite":"chat3_hold","n":len(out),"acc":sum(x["hit"] for x in out)/max(1,len(out)),"by_kind":{k:round(a/b,4) for k,(a,b) in by.items()},"n_by_kind":{k:b for k,(a,b) in by.items()},"items":out}
if __name__=="__main__":
 t=time.time();ev._load=(lambda f:(lambda name:f(name)[:N] if N else f(name)))(ev._load);rep={"recall":ev.run_recall(gen),"twohop":ev.run_twohop(gen),"related":ev.run_related(gen),"chat3":chat3(int(os.environ.get("C3_N","0"))),"pack":P,"engine":f"adam_ray {e.backend} L={e.L} copy={e.copy}","wall_s":0}
 rep["wall_s"]=round(time.time()-t);dst=os.path.join(str(ev.EVAL),f"report_{TAG}.json");json.dump(rep,open(dst,"w"),indent=1)
 it=rep["recall"]["items"];kn=[i for i in it if not i["unknown"]];un=[i for i in it if i["unknown"]]
 print(json.dumps({"recall_known":round(sum(i["hit"] for i in kn)/max(1,len(kn)),4),"unknown":round(sum(i["hit"] for i in un)/max(1,len(un)),4),"twohop":rep["twohop"]["exact_or_contains"],"related":rep["related"]["mean_coverage"],"chat3":rep["chat3"]["acc"],"chat3_by_kind":rep["chat3"]["by_kind"],"engine":rep["engine"],"wall_s":rep["wall_s"],"dst":dst}))
