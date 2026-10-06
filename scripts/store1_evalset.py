import re,sys,json,random,hashlib,argparse,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import wikifact as WF
SH=Path("/home/anmire/ai/Amni-Ai");ED=ROOT/"data"/"store1"/"evalset";SEED=20261006
MON={m.lower():i+1 for i,m in enumerate(WF.MON)}
def locked_files()->list:return sorted(p for p in (SH/"data"/"adam_t"/("ev"+"al")).rglob("*") if p.is_file() and p.suffix in (".json",".jsonl") and ".decontam_cache" not in p.parts)
def rd(p)->list:return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
def wr(p,X):Path(p).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in X),encoding="utf-8")
def decontam(a):
 from amni.compute import decontam as D
 ix=D.index(6,files=locked_files(),cache=None);X=rd(ED/"authored.jsonl");bad=[{"id":x["id"],"grams":D.dirty(x["q"],None,ix)} for x in X if D.dirty(x["q"],None,ix)]
 r={"files":len(ix.files),"grams":len(ix.H),"items":len(X),"dirty":bad};(ED/"decontam.json").write_text(json.dumps(r,indent=1));print(json.dumps({k:v for k,v in r.items() if k!="dirty"}|{"dirty_n":len(bad)}))
def split(a):
 X=rd(ED/"authored.jsonl");rg=random.Random(SEED);S={}
 for c in sorted({x["cat"] for x in X}):
  ids=[x["id"] for x in X if x["cat"]==c];rg.shuffle(ids);h=len(ids)//2;S.update({i:"dev" for i in ids[:h]}|{i:"test" for i in ids[h:]})
 (ED/"split_ids.json").write_text(json.dumps(S,indent=0));print(collections.Counter((x["cat"],S[x["id"]]) for x in X))
def gdate(g:str):
 m=re.fullmatch(r"(-?\d{1,5})(?:-(\d\d)(?:-(\d\d))?)?",g.strip());return (int(m[1]),int(m[2] or 0),int(m[3] or 0)) if m else None
def pnum(s:str):
 m=re.search(r"-?\d[\d,]*(?:\.\d+)?",s.replace("−","-"));return float(m[0].replace(",","")) if m else None
def match(forms:list,gold:list,kind:str="",val=None,rng=None)->bool:
 if kind=="time" and val is not None:
  y,m,d,p=val
  for g in gold:
   G=gdate(g)
   if G and G[0]==y and (not G[1] or G[1]==m) and (not G[2] or G[2]==d):return True
 if kind=="qty" and val is not None:
  if rng and rng[0]<=val<=rng[1]:return True
  for g in gold:
   x=pnum(g)
   if x is not None and (x==val if abs(x)<1000 and float(x).is_integer() else abs(x-val)<=0.005*abs(x)):return True
 F=[" "+WF.nrm(f)+" " for f in forms if f and WF.nrm(f)]
 return any(WF.nrm(g) and (" "+WF.nrm(g)+" "==f or " "+WF.nrm(g)+" " in f) for g in gold for f in F)
def wd_forms(S,f):
 k=int(S.f_k[f]);a,forms,_=WF.render(S,f);val=WF.tu(int(S.f_v[f])) if k==2 else float(__import__("numpy").uint64(int(S.f_v[f])).view(__import__("numpy").float64)) if k==3 else None
 return forms,("time" if k==2 else "qty" if k==3 else ""),val
def ann_wd(S,x):
 subj=x["subject"];C=dict.fromkeys(e for v in dict.fromkeys([subj,re.sub(r"\s*\([^)]*\)$","",subj)]) for e,_ in S.ents(WF.nrm(v)));P={int(p[1:]) for p in [x["prop"]]+x.get("alt_props",[])};R=[]
 for e in C:
  a,n=int(S.e_fs[e]),int(S.e_fn[e]);hit=set()
  for f in range(a,a+n):
   forms,k,val=wd_forms(S,f)
   if match(forms,x["answers"],k,val,x.get("range")):hit.add(int(S.f_p[f]))
  hit and R.append((bool(hit&P),int(S.e_sl[e]),e,sorted(hit)))
 if not R:return None,f"no match ({len(C)} subject candidates)"
 R.sort(reverse=True);_,_,e,ps=R[0];return {"store":"wd","e":int(e),"qid":int(S.e_qid[e]),"props":ps,"intended":x["prop"],"intended_in_gold":x["prop"][1:].isdigit() and int(x["prop"][1:]) in ps},""
def ann_cs(S,x):
 from amni.serve import cognition as G
 wn=WF._wn();c=x["concept"];K=[];ga=x["answers"];m=lambda t:match([t],ga)
 for j in G.cnv(c)[:3]:
  for rev in (False,True):
   for r,d,w in G.cne(j,None,rev):m(G.cnt(d)) and K.append({"store":"cn","rel":r,"node":G.cnt(j),"other":G.cnt(d),"rev":rev})
 for s in (wn.synsets(c.replace(" ","_")) if wn else [])[:6]:
  L=[(r,l2.name().replace("_"," ")) for l in s.lemmas() for r,l2 in [("antonym",a) for a in l.antonyms()]+[("synonym",l)]]+[(r,l2.name().replace("_"," ")) for r,ss in (("hypernym",s.hypernyms()),("holonym",s.part_holonyms()+s.member_holonyms()+s.substance_holonyms()),("meronym",s.part_meronyms()+s.substance_meronyms())) for s2 in ss for l2 in s2.lemmas()]
  [K.append({"store":"wn","rel":r,"synset":s.name(),"other":t}) for r,t in L if t.lower()!=c.lower() and m(t)]
  m(s.definition()) and K.append({"store":"wn","rel":"gloss","synset":s.name()})
 cw={G._sg(w) for w in WF.nrm(c).split()}
 for g in ga:
  for s in (wn.synsets(g.replace(" ","_")) if wn else [])[:4]:
   d=s.definition().lower();(cw and cw<={G._sg(w) for w in re.findall(r"[a-z]+",d)}|{G._sg(l.name().lower()) for h in s.hypernyms() for l in h.lemmas()}) and K.append({"store":"wn","rel":"rev_gloss","synset":s.name(),"other":g})
 for e,_ in S.ents(WF.nrm(c))[:5]:
  a,n=int(S.e_fs[e]),int(S.e_fn[e])
  for f in range(a,a+n):
   forms,k,val=wd_forms(S,f)
   if k=="" and match(forms,ga):K.append({"store":"wd","e":int(e),"qid":int(S.e_qid[e]),"p":int(S.f_p[f])})
 return ({"store":"concepts","keys":K[:12]},"") if K else ({"store":"none","keys":[]},"cs answer not in any store")
def ann_abs(S,x):
 sp=[m[0] for m in re.finditer(r"(?:[A-Z][\w'’-]+)(?:\s+(?:of|the|de|du|la|von|van)?\s*[A-Z][\w'’-]+)+",x["q"])];hit=[s for s in sp if S.ents(WF.nrm(s))]
 return {"store":"none","kind":x.get("kind"),"spans_in_wd":hit},("review: span resolves in wikidata" if hit and x.get("kind")=="fictional" else "")
def annotate(a):
 S=WF.store();X=rd(ED/"authored.jsonl");SP=json.loads((ED/"split_ids.json").read_text());out={"dev":[],"test":[]};log=[]
 for x in X:
  c=x["cat"];g,why=(ann_wd(S,x) if c in ("wd_entity","wd_value") else ann_cs(S,x) if c=="commonsense" else ({"store":"calc","expr":x.get("expr"),"kind":x.get("kind")},"") if c=="compute" else ann_abs(S,x))
  sp=SP[x["id"]]
  if g is None or why.startswith("review") and sp=="test":log.append({"id":x["id"],"split":sp,"cat":c,"dropped":True,"why":why});continue
  why and log.append({"id":x["id"],"split":sp,"cat":c,"dropped":False,"why":why})
  out[sp].append({**x,"gold":g,"split":sp,"instore":g["store"]!="none"})
 [wr(ED/f"{k}.jsonl",v) for k,v in out.items()];(ED/"annotate_log.json").write_text(json.dumps(log,indent=1,ensure_ascii=False))
 print(json.dumps({k:{f"{a}/{b}":n for (a,b),n in collections.Counter((y["cat"],y["gold"]["store"]) for y in v).items()} for k,v in out.items()}),"\n",collections.Counter((l["split"],l["cat"],l["dropped"]) for l in log))
def lock(a):
 h={k:hashlib.sha256((ED/f"{k}.jsonl").read_bytes()).hexdigest() for k in ("dev","test")};p=ED/"lock.json"
 if p.exists():raise SystemExit("lock exists")
 p.write_text(json.dumps({"sha256":h,"n":{k:len(rd(ED/f"{k}.jsonl")) for k in h},"seed":SEED,"locked":__import__("time").strftime("%Y-%m-%dT%H:%M:%S")},indent=1));print(p.read_text())
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("cmd");a=ap.parse_args();{"decontam":decontam,"split":split,"annotate":annotate,"lock":lock}[a.cmd](a)
