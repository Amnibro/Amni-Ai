import os,re,sys,json,time,argparse,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from store1_evalset import match,pnum,rd,ED
OUT=ROOT/"data"/"store1"/"results"
def ok(forms:list,x:dict,val=None,kind="")->bool:
 g=x["answers"];r=x.get("range")
 if match(forms,g,kind,val,r):return True
 G=[pnum(a) for a in g if re.fullmatch(r"[-\d.,\s]+",a.strip())];F=[pnum(f) for f in forms if f and pnum(f) is not None and re.match(r"^\s*-?[\d.,]+\s*[a-zA-Z%°]*\s*$",f)]
 return any(a is not None and b is not None and (a==b if abs(a)<1000 and float(a).is_integer() else abs(a-b)<=0.005*abs(a)) for a in G for b in F)
def score(rows:list)->dict:
 ins=[r for r in rows if r["instore"]];wd=[r for r in ins if r["gstore"]=="wd"];ab=[r for r in rows if r["abstain"]];P=lambda a,b:round(a/b,4) if b else None
 lat=sorted(r["ms"] for r in rows)
 return {"n":len(rows),"n_instore":len(ins),"n_wd":len(wd),"e2e_acc_instore":P(sum(r["correct"] for r in ins),len(ins)),"ent_at1":P(sum(r["ent_rank"]==0 for r in wd),len(wd)),"ent_at5":P(sum(0<=r["ent_rank"]<5 for r in wd),len(wd)),"prop_acc":P(sum(r["prop_ok"] for r in wd),len(wd)),"abstain_prec":P(sum(not r["instore"] for r in ab),len(ab)),"abstain_n":len(ab),"abstain_rec":P(sum(r["abstain"] for r in rows if not r["instore"]),sum(not r["instore"] for r in rows)),"absent_false_answer":P(sum(not r["abstain"] for r in rows if not r["instore"]),sum(not r["instore"] for r in rows)),
  "by_cat":{c:{"n":len(z),"acc":P(sum(r["correct"] for r in z),len(z)),"abstain":P(sum(r["abstain"] for r in z),len(z))} for c in sorted({r["cat"] for r in rows}) for z in [[r for r in rows if r["cat"]==c and (r["instore"] or c=="absent")]]},"lat_median_ms":round(statistics.median(lat),3) if lat else None,"lat_p99_ms":round(lat[min(len(lat)-1,int(0.99*len(lat)))],3) if lat else None,"lat_max_ms":lat[-1] if lat else None,
  "v3_acc_instore_wd":P(sum(r["v3_ok"] for r in wd),len(wd)),"v3_acc_instore":P(sum(r["v3_ok"] for r in ins),len(ins))}
def run(a):
 os.environ.setdefault("AMNI_STORE1","1")
 if a.abl:os.environ["AMNI_STORE1_ABL"]=a.abl
 from amni.serve import store1 as S;from amni.compute import wikifact as WF
 X=rd(ED/f"{a.split}.jsonl");lk=json.loads((ED/"lock.json").read_text())
 if hashlib.sha256((ED/f"{a.split}.jsonl").read_bytes()).hexdigest()!=lk["sha256"][a.split]:raise SystemExit("eval set hash mismatch")
 OUT.mkdir(parents=True,exist_ok=True);out=OUT/f"{a.name}_{a.split}.json"
 if a.split=="test" and (OUT/"test_scores.json").exists():raise SystemExit("test already scored")
 S.resolver(a.fd)
 if a.warm:[S.resolve(x["q"],a.fd) for x in rd(ED/"dev.jsonl")]
 rows=[]
 for x in X:
  r=S.resolve(x["q"],a.fd);b=r["best"];g=x["gold"];w=r["wd"]
  cor=bool(b) and ok(b.get("forms") or [b.get("answer","")],x,b.get("val"),b.get("kind",""))
  er=w.get("ents",[]).index(g["e"]) if g.get("store")=="wd" and g["e"] in w.get("ents",[]) else -1
  po=g.get("store")=="wd" and bool(w.get("props")) and w["props"][0] in g["props"]
  v3=WF.ask(x["q"]) if a.v3 else None;v3ok=bool(v3 and v3.get("answer")) and ok(v3.get("forms") or [v3["answer"]],x)
  rows.append({"id":x["id"],"cat":x["cat"],"q":x["q"],"gstore":g["store"],"instore":x["instore"],"abstain":r["abstain"],"store":b["store"] if b else None,"answer":b.get("answer") if b else None,"correct":cor,"ent_rank":er,"prop_ok":po,"top_props":w.get("props",[])[:3],"wd_score":round(w.get("score",0),3),"ms":r["ms"],"v3_ok":v3ok,"gold":x["answers"][:3]})
 m=score(rows);res={"name":a.name,"split":a.split,"fd":a.fd,"abl":a.abl,"cf":S.CF,"metrics":m,"rows":rows,"time":time.strftime("%Y-%m-%dT%H:%M:%S")};out.write_text(json.dumps(res,indent=1,ensure_ascii=False))
 a.split=="test" and (OUT/"test_scores.json").write_text(json.dumps({"source":str(out.relative_to(ROOT)),**m},indent=1))
 print(json.dumps(m))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--split",default="dev");ap.add_argument("--name",default="full");ap.add_argument("--fd",type=int,default=3);ap.add_argument("--abl",default="");ap.add_argument("--v3",type=int,default=1);ap.add_argument("--warm",type=int,default=0);run(ap.parse_args())
