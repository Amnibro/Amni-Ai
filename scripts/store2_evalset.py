import re,sys,json,random,hashlib,argparse,collections,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from store1_evalset import rd,wr,match,wd_forms,ann_wd,ann_cs
from amni.compute import wikifact as WF
SH=Path("/home/anmire/ai/Amni-Ai");ED=ROOT/"data"/"store2"/"evalset";SEED=20261017;AUTH=Path("/tmp/claude-1000/-home-anmire/81ad65c4-2e7d-457b-bcca-dbc14db35a0d/scratchpad/store2_author/authored.jsonl")
def locked_files()->list:return sorted(p for p in (SH/"data"/"adam_t"/("ev"+"al")).rglob("*") if p.is_file() and p.suffix in (".json",".jsonl") and ".decontam_cache" not in p.parts)+sorted((ROOT/"data"/"store1"/"evalset").glob("*.jsonl"))
def take(a):
 ED.mkdir(parents=True,exist_ok=True);(ED/"lock.json").exists() and sys.exit("locked");shutil.copy(AUTH,ED/"authored.jsonl");X=rd(ED/"authored.jsonl");print(len(X),collections.Counter(x["cat"] for x in X))
def decontam(a):
 from amni.compute import decontam as D
 F=locked_files();ix=D.index(6,files=F,cache=None);X=rd(ED/"authored.jsonl");bad=[{"id":x["id"],"grams":D.dirty(x["q"],None,ix)} for x in X if D.dirty(x["q"],None,ix)]
 r={"files":len(ix.files),"store1_evalset_files":sum("store1" in str(f) for f in F),"grams":len(ix.H),"items":len(X),"dirty":bad};(ED/"decontam.json").write_text(json.dumps(r,indent=1));print(json.dumps({k:v for k,v in r.items() if k!="dirty"}|{"dirty_n":len(bad),"dirty_ids":[b["id"] for b in bad]}))
def split(a):
 X=rd(ED/"authored.jsonl");rg=random.Random(SEED);S={}
 for c in sorted({x["cat"] for x in X}):
  ids=[x["id"] for x in X if x["cat"]==c];rg.shuffle(ids);h=len(ids)//2;S.update({i:"dev" for i in ids[:h]}|{i:"test" for i in ids[h:]})
 (ED/"split_ids.json").write_text(json.dumps(S,indent=0));print(collections.Counter((x["cat"],S[x["id"]]) for x in X))
def pmap()->dict:
 J=json.loads((ROOT/"data"/"store1"/"props_en.json").read_text())["props"];M={}
 for k,v in J.items():
  for a in [v["label"]]+v["aliases"]:M.setdefault(" ".join(WF.lem(w) for w in WF.nrm(a).split()),set()).add(int(k[1:]))
 return M
def ann_abs(S,x,PM):
 sp=[m[0] for m in re.finditer(r"(?:[A-Z][\w'’-]+)(?:\s+(?:of|the|de|du|la|von|van|and)?\s*[A-Z][\w'’-]+)*",x["q"])]+[x.get("subject") or ""]
 C={};[C.setdefault(e,s) for s in dict.fromkeys(filter(None,sp)) for v in dict.fromkeys([s,re.sub(r"^(?:The|the) ","",s)]) for e,_ in S.ents(WF.nrm(v))]
 ph=" ".join(WF.lem(w) for w in WF.nrm(x.get("prop_phrase") or "").split());P=PM.get(ph,set()) if ph else set();hit=[]
 for e in C:
  a,n=int(S.e_fs[e]),int(S.e_fn[e])
  for f in range(a,a+n):
   forms,k,val=wd_forms(S,f);pp=int(S.f_p[f])
   if x["answers"] and match(forms,x["answers"],k,val,x.get("range")):hit.append(("answer",int(e),pp))
   if pp in P:hit.append(("prop",int(e),pp))
 subj=bool(x.get("subject")) and bool(S.ents(WF.nrm(x["subject"])))
 why="absent check: "+",".join(sorted({f"{h[0]}:Q{int(S.e_qid[h[1]])}:P{h[2]}" for h in hit})) if hit else "fictional subject resolves in wikidata" if x.get("kind")=="fictional" and subj else ""
 return {"store":"none","kind":x.get("kind"),"cands":len(C),"pids_checked":sorted(P),"facts_checked":sum(int(S.e_fn[e]) for e in C)},why
def annotate(a):
 S=WF.store();X=rd(ED/"authored.jsonl");SP=json.loads((ED/"split_ids.json").read_text());out={"dev":[],"test":[]};log=[];PM=pmap()
 for x in X:
  c=x["cat"];g,why=(ann_wd(S,x) if c in ("wd_entity","wd_value") else ann_cs(S,x) if c=="commonsense" else ({"store":"calc","expr":x.get("expr"),"kind":x.get("kind")},"") if c=="compute" else ann_abs(S,x,PM))
  sp=SP[x["id"]];drop=g is None or c=="absent" and bool(why)
  log.append({"id":x["id"],"split":sp,"cat":c,"dropped":drop,"why":why}) if drop or why else None
  drop or out[sp].append({**x,"gold":g,"split":sp,"instore":g["store"]!="none"})
 [wr(ED/f"{k}.jsonl",v) for k,v in out.items()];(ED/"annotate_log.json").write_text(json.dumps(log,indent=1,ensure_ascii=False))
 print(json.dumps({k:{f"{a_}/{b}":n for (a_,b),n in collections.Counter((y["cat"],y["gold"]["store"]) for y in v).items()} for k,v in out.items()}),"\n",collections.Counter((l["split"],l["cat"],l["dropped"]) for l in log))
def lock(a):
 h={k:hashlib.sha256((ED/f"{k}.jsonl").read_bytes()).hexdigest() for k in ("dev","test")};p=ED/"lock.json"
 if p.exists():raise SystemExit("lock exists")
 p.write_text(json.dumps({"sha256":h,"n":{k:len(rd(ED/f"{k}.jsonl")) for k in h},"seed":SEED,"locked":__import__("time").strftime("%Y-%m-%dT%H:%M:%S")},indent=1));print(p.read_text())
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("cmd");a=ap.parse_args();{"take":take,"decontam":decontam,"split":split,"annotate":annotate,"lock":lock}[a.cmd](a)
