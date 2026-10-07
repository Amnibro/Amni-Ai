import re,sys,json,random,hashlib,argparse,collections,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from store1_evalset import rd,wr,ann_wd,ann_cs
from amni.compute import wikifact as WF
SH=Path("/home/anmire/ai/Amni-Ai");ED=ROOT/"data"/"pipe1"/"evalset";SEED=20261006;AUTH=Path("/tmp/claude-1000/-home-anmire/81ad65c4-2e7d-457b-bcca-dbc14db35a0d/scratchpad/pipe1_author/authored.jsonl")
TF=("q","q2","corr","undo","msg","setup","prior_q")
def locked_files()->list:
 E=SH/"data"/"adam_t"/("ev"+"al");P=SH/"data"/"adam_t"/"probe"
 return sorted(p for d in (E,P) for p in d.rglob("*") if p.is_file() and p.suffix in (".json",".jsonl") and ".decontam_cache" not in p.parts)+sorted((SH/"data"/"store1"/"evalset").glob("*.jsonl"))+sorted((SH/"data"/"store2"/"evalset").glob("*.jsonl"))
def take(a):
 ED.mkdir(parents=True,exist_ok=True);(ED/"lock.json").exists() and sys.exit("locked");shutil.copy(AUTH,ED/"authored.jsonl");X=rd(ED/"authored.jsonl");print(len(X),collections.Counter(x["cat"] for x in X))
def decontam(a):
 from amni.compute import decontam as D
 F=locked_files();ix=D.index(6,files=F,cache=None);X=rd(ED/"authored.jsonl");bad=[{"id":x["id"],"field":k,"grams":D.dirty(x[k],None,ix)} for x in X for k in TF if isinstance(x.get(k),str) and D.dirty(x[k],None,ix)]
 r={"files":len(ix.files),"store_evalset_files":sum("evalset" in str(f) for f in F),"grams":len(ix.H),"items":len(X),"dirty":bad,"at":time.strftime("%Y-%m-%dT%H:%M:%S")};(ED/"decontam.json").write_text(json.dumps(r,indent=1));print(json.dumps({k:v for k,v in r.items() if k!="dirty"}|{"dirty_n":len(bad),"dirty":[(b["id"],b["field"]) for b in bad]}))
def split(a):
 X=rd(ED/"authored.jsonl");rg=random.Random(SEED);S={}
 for c in sorted({x["cat"] for x in X}):
  ids=[x["id"] for x in X if x["cat"]==c];rg.shuffle(ids);h=len(ids)//2;S.update({i:"dev" for i in ids[:h]}|{i:"test" for i in ids[h:]})
 (ED/"split_ids.json").write_text(json.dumps(S,indent=0));print(collections.Counter((x["cat"],S[x["id"]]) for x in X))
def annotate(a):
 S=WF.store();X=rd(ED/"authored.jsonl");SP=json.loads((ED/"split_ids.json").read_text());out={"dev":[],"test":[]};log=[]
 for x in X:
  c=x["cat"];g,why=(ann_wd(S,x) if c in ("wd_fact","wd_value","corr_poison") else ann_cs(S,x) if c=="commonsense" else ({"store":"calc","expr":x.get("expr")},"") if c=="compute" else ({"store":"none"},""))
  drop=g is None and c in ("wd_fact","wd_value");sp=SP[x["id"]];(drop or why) and log.append({"id":x["id"],"split":sp,"cat":c,"dropped":drop,"why":why})
  drop or out[sp].append({**x,"gold":g or {"store":"none"},"split":sp,"instore":bool(g) and g.get("store") not in (None,"none")})
 [wr(ED/f"{k}.jsonl",v) for k,v in out.items()];(ED/"annotate_log.json").write_text(json.dumps(log,indent=1,ensure_ascii=False))
 print(json.dumps({k:dict(collections.Counter(y["cat"] for y in v)) for k,v in out.items()}),"\n",collections.Counter((l["split"],l["cat"],l["dropped"]) for l in log))
def lock(a):
 h={k:hashlib.sha256((ED/f"{k}.jsonl").read_bytes()).hexdigest() for k in ("dev","test")};p=ED/"lock.json"
 if p.exists():raise SystemExit("lock exists")
 p.write_text(json.dumps({"sha256":h,"n":{k:len(rd(ED/f"{k}.jsonl")) for k in h},"seed":SEED,"locked":time.strftime("%Y-%m-%dT%H:%M:%S")},indent=1));print(p.read_text())
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("cmd");a=ap.parse_args();{"take":take,"decontam":decontam,"split":split,"annotate":annotate,"lock":lock}[a.cmd](a)
