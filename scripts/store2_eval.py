import os,sys,json,time,random,argparse,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
os.environ["AMNI_STORE2"]="1"
from store1_eval import ok,score
from store1_evalset import rd
ED=ROOT/"data"/"store2"/"evalset";OUT=ROOT/"data"/"store2"/"results"
SY="zor bla quen vik tam rel oss dur mav pex lor fen yb ruk sa tiv gol nem".split()
def warmq()->list:
 g=random.Random(7);nm=lambda:" ".join("".join(g.choice(SY) for _ in range(3)).capitalize() for _ in range(2))
 T=["Who founded the village of {}?","When was {} born?","Where did {} die?","What is the population of {}?","Who directed {}?","What river flows through {}?","Who is married to {}?","How tall is {}?"]
 A=["What is a baby {} called?","What do {} eat?","What is the opposite of {}?","Where does a {} live?","What sound does a {} make?","What is a group of {} called?","What is {} made of?","What is a {} used for?"];w=lambda:"".join(g.choice(SY) for _ in range(3))
 return [T[i%len(T)].format(nm()) for i in range(24)]+[A[i%len(A)].format(w()) for i in range(8)]+[f"What is {g.randint(10,99)} plus {g.randint(10,99)}?" for _ in range(8)]
def lat(L:list)->tuple:L=sorted(L);return (round(statistics.median(L),3),round(L[min(len(L)-1,int(0.99*len(L)))],3)) if L else (None,None)
def run(a):
 if a.abl:os.environ["AMNI_STORE1_ABL"]=a.abl
 from amni.serve import store1 as S;from amni.compute import wikifact as WF
 X=rd(ED/f"{a.split}.jsonl");lk=json.loads((ED/"lock.json").read_text())
 if hashlib.sha256((ED/f"{a.split}.jsonl").read_bytes()).hexdigest()!=lk["sha256"][a.split]:raise SystemExit("eval set hash mismatch")
 OUT.mkdir(parents=True,exist_ok=True);out=OUT/f"{a.name}_{a.split}.json"
 if a.split=="test" and (out.exists() or a.primary and (OUT/"test_scores.json").exists()):raise SystemExit("test already scored")
 S.resolver(a.fd,True);[S.resolve(q,a.fd) for q in warmq()]
 rows=[];L=[]
 for x in X:
  S.clear();r=S.resolve(x["q"],a.fd);L.append(r["ms"]);b=r["best"];g=x["gold"];w=r["wd"]
  cor=bool(b) and ok(b.get("forms") or [b.get("answer","")],x,b.get("val"),b.get("kind",""))
  er=w.get("ents",[]).index(g["e"]) if g.get("store")=="wd" and g["e"] in w.get("ents",[]) else -1
  po=g.get("store")=="wd" and bool(w.get("props")) and w["props"][0] in g["props"]
  rows.append({"id":x["id"],"cat":x["cat"],"q":x["q"],"gstore":g["store"],"instore":x["instore"],"abstain":r["abstain"],"store":b["store"] if b else None,"answer":b.get("answer") if b else None,"correct":cor,"ent_rank":er,"prop_ok":po,"top_props":w.get("props",[])[:3],"wd_score":round(w.get("score",0),3),"fb":bool(w.get("fb")),"ms":r["ms"],"v3_ok":False,"gold":x["answers"][:3]})
 W=[S.resolve(x["q"],a.fd)["ms"] for x in X]
 for x,rw in zip(X,rows):v3=WF.ask(x["q"]) if a.v3 else None;rw["v3_ok"]=bool(v3 and v3.get("answer")) and ok(v3.get("forms") or [v3["answer"]],x)
 Cd=[]
 if a.cold:
  for x in X:S.clear(True);Cd.append(S.resolve(x["q"],a.fd)["ms"])
 m=score(rows);m["lat_median_ms"],m["lat_p99_ms"]=lat(L);m["lat_warm_median_ms"],m["lat_warm_p99_ms"]=lat(W);m["lat_cold_median_ms"],m["lat_cold_p99_ms"]=lat(Cd)
 m["lat_protocol"]="unseen: fresh process, 40 fictional/arithmetic warm-up questions, every in-process cache (Python LRU + drive block caches) cleared before each question, OS page cache kept; warm: second pass, caches kept; cold: plus madvise/fadvise DONTNEED on the drive before each question"
 res={"name":a.name,"split":a.split,"fd":a.fd,"abl":a.abl,"cf":S.CF,"metrics":m,"rows":rows,"time":time.strftime("%Y-%m-%dT%H:%M:%S")};out.write_text(json.dumps(res,indent=1,ensure_ascii=False))
 a.split=="test" and a.primary and (OUT/"test_scores.json").write_text(json.dumps({"source":str(out.relative_to(ROOT)),**m},indent=1))
 print(json.dumps({k:v for k,v in m.items() if k!="by_cat" and k!="lat_protocol"}),"\n",json.dumps(m["by_cat"]))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--split",default="dev");ap.add_argument("--name",default="full");ap.add_argument("--fd",type=int,default=2);ap.add_argument("--abl",default="");ap.add_argument("--v3",type=int,default=1);ap.add_argument("--cold",type=int,default=1);ap.add_argument("--primary",type=int,default=1);run(ap.parse_args())
