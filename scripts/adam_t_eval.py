import os,re,io,sys,json,time,math,base64,shlex,hashlib,argparse,functools,resource,subprocess,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
[os.environ.__setitem__(k,str(min(4,int(os.environ.get(k) or 4)))) for k in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_MAX_THREADS","TI_NUM_THREADS","POLARS_MAX_THREADS")];os.environ["AMNI_RAY_ARCH"]="cpu"
from amni.compute import eval_lock as EL
from adam_nightly import score_qa,NORM
from v19_eval_run import DECLINE,score_contains
EV=ROOT/"data"/"adam_t"/"eval";BN=EV/"bench";RK=ROOT/"data"/"concepts"/"eval"/"rknow";D3=ROOT/"data"/"concepts"/"eval"/"d3";PRE=ROOT/"docs"/"adam_t"/"eval_prereg.md";MAN=EV/"prereg_lock.json";STREAM=ROOT/"data"/"adam_t"/"stream"/"manifest.json";C2=ROOT/"data"/"clean2"/"hold.bin"
SEED=20260925;N_ZS=500;RC_N=400;RC_K=30;RC_MAXB=512;BPB_W=8192;BPB_T=256;MAXB={"qa":64,"rc":64,"mem":80};BATCH=256
ARC_REV="210d026faf9955653af8916fad021475a3f00453";PIQA_REV="16c6be5562fdf2e56a6d5ee4f39a047494eb0617";PQ=f"https://raw.githubusercontent.com/ybisk/ybisk.github.io/{PIQA_REV}/piqa/data/"
BENCH={"arc_easy":{"url":[f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{ARC_REV}/ARC-Easy/test-00000-of-00001.parquet"],"license":"CC BY-SA 4.0 (allenai/ai2_arc dataset card; ARC-Easy test split)"},"piqa":{"url":[PQ+"valid.jsonl",PQ+"valid-labels.lst"],"license":"Academic Free License v3.0 (stated by the authors in ybisk.github.io/piqa/data/README.md; HF mirror card says unknown, upstream README is authoritative)"}}
NQF=EV/"nq_mc.json";RCF=EV/"rc_dev.json";D3F=EV/"d3dev.jsonl";ZSF={k:BN/f"{k}_{N_ZS}.jsonl" for k in BENCH}
SPLITS=[RK/"nq_dev_tune.json",RK/"nq_dev_hold.json",NQF,RCF,D3F,C2,*ZSF.values(),BN/"meta.json",PRE]
CODE=[Path(__file__).resolve(),*[ROOT/p for p in ("scripts/adam_nightly.py","scripts/v19_eval_run.py","amni/compute/ray_field.py","amni/compute/eval_lock.py")],*sorted(ROOT.glob("adam_ray.cpython-*.so"))]
PUB={"SmolLM2-135M":{"arc_avg":43.9,"piqa":68.4,"hellaswag":42.1,"src":"huggingface.co/HuggingFaceTB/SmolLM2-135M model card (lighteval zero-shot; ARC = mean of Easy+Challenge)"},"SmolLM2-360M":{"arc_avg":53.0,"piqa":71.7,"hellaswag":54.5,"src":"huggingface.co/HuggingFaceTB/SmolLM2-360M model card (lighteval zero-shot; ARC = mean of Easy+Challenge)"},"Qwen2.5-0.5B":{"arc_avg":45.4,"piqa":69.9,"hellaswag":51.2,"src":"SmolLM2-360M model card comparison column (HF lighteval re-run; not from Qwen)"}}
TPL={"plain":{"qa":"Question: {q}\nAnswer:","rc":"{t}\n{p}\n\nQuestion: {q}\nAnswer:","zs":"Question: {q}\nAnswer:"},"chat":{"qa":"User: {q}\nAdam:","rc":"User: {t}: {p} {q}\nAdam:","zs":"User: {q}\nAdam:"}}
B64=lambda b:base64.b64encode(b).decode();U64=lambda s:base64.b64decode(s)
WB=lambda a:r"(?<![A-Za-z0-9])"+re.escape(a)+r"(?![A-Za-z0-9])"
shape=lambda a:"date" if re.search(r"\b(1\d{3}|20\d\d)\b",a) and re.search(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",a) else "year" if re.fullmatch(r"\d{3,4}",a.strip()) else "num" if re.search(r"\d",a) else "cap" if a[:1].isupper() else "low"
wh=lambda q:next((w for w in ("how many","how much","who","when","where","what year") if re.search(r"\b"+w+r"\b",q.lower())),"other")
clash=lambda o,G:any(n and g and (n in g or g in n) for n in [NORM(o)] for g in map(NORM,G))
def pick(G:list,prefer:list,pool:list,rng,k:int=4)->tuple:
 ch=[];[ch.append(o) for o in [*[prefer[i] for i in rng.permutation(len(prefer))],*[pool[i] for i in rng.permutation(len(pool))]] if len(ch)<k-1 and o.strip() and not clash(o,G+ch)];op=[G[0]]+ch;p=rng.permutation(len(op));return [op[i] for i in p],int(np.argwhere(p==0)[0][0])
def trim(x:str,a:int,e:int,n:int=RC_MAXB)->str:
 S=[(m.start(),m.end()) for m in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)",x)];i=next(j for j,(s,t) in enumerate(S) if t>a);lo=hi=i
 while True:
  c=[z for z in ([(S[hi+1][1]-S[lo][0],lo,hi+1)] if hi+1<len(S) else [])+([(S[hi][1]-S[lo-1][0],lo-1,hi)] if lo>0 else []) if z[0]<=n]
  if not c:break
  _,lo,hi=min(c)
 y=x[S[lo][0]:S[hi][1]];w=max(S[lo][0],min(a-n//2,e-n//2));return y if len(y)<=n else x[w:w+n].strip()
def fetch(u:str)->bytes:
 with urllib.request.urlopen(u,timeout=60) as r:return r.read()
def build_bench()->dict:
 import pyarrow.parquet as pq
 BN.mkdir(parents=True,exist_ok=True);raw={u:fetch(u) for b in BENCH.values() for u in b["url"]};t=pq.read_table(io.BytesIO(raw[BENCH["arc_easy"]["url"][0]])).to_pylist()
 arc=[{"id":r["id"],"q":r["question"],"options":list(r["choices"]["text"]),"gold":list(r["choices"]["label"]).index(r["answerKey"])} for r in t]
 pv=[json.loads(l) for l in raw[BENCH["piqa"]["url"][0]].decode().splitlines() if l.strip()];pl=[int(l) for l in raw[BENCH["piqa"]["url"][1]].decode().split()];assert len(pv)==len(pl)
 piqa=[{"id":f"piqa_valid#{i}","q":r["goal"],"options":[r["sol1"],r["sol2"]],"gold":y} for i,(r,y) in enumerate(zip(pv,pl))]
 for k,rows in (("arc_easy",arc),("piqa",piqa)):ix=sorted(np.random.default_rng(SEED).choice(len(rows),N_ZS,replace=False).tolist());ZSF[k].write_text("".join(json.dumps({**rows[i],"license":BENCH[k]["license"]})+"\n" for i in ix))
 m={k:{**v,"raw_sha256":{u:hashlib.sha256(raw[u]).hexdigest() for u in v["url"]},"n_full":len(arc if k=="arc_easy" else piqa),"n":N_ZS,"seed":SEED,"file":EL.rel(ZSF[k])} for k,v in BENCH.items()};(BN/"meta.json").write_text(json.dumps(m,indent=1));return m
def build_nq()->list:
 R=[{"split":s,"i":i,**r} for s in ("tune","hold") for i,r in enumerate(json.loads((RK/f"nq_dev_{s}.json").read_text()))];out=[]
 for j,r in enumerate(R):
  k=(shape(r["answers"][0]),wh(r["q"]));same=[x["answers"][0] for x in R if x is not r and (shape(x["answers"][0]),wh(x["q"]))==k];sh=[x["answers"][0] for x in R if x is not r and shape(x["answers"][0])==k[0] and x["answers"][0] not in same]
  op,g=pick(r["answers"],same,sh,np.random.default_rng(SEED+j));out.append({"split":r["split"],"i":r["i"],"q":r["q"],"answers":r["answers"],"options":op,"gold":g,"type":list(k)})
 NQF.write_text(json.dumps(out,indent=0));return out
def build_rc(nq:list)->list:
 from amni.compute import rknow as K
 I=K.ix();out=[];miss=0
 for j in np.random.default_rng(SEED).permutation(len(nq)).tolist():
  r=nq[j];c={};hit=None
  for sc,p in I.search(r["q"],RC_K):
   t,x,_=I.text(p,c);x=" ".join(x.split());m=next((m for a in r["answers"] for m in [re.search(WB(a),x,re.I)] if m),None)
   if m:hit=(t,x,m,p);break
  if not hit:miss+=1;continue
  t,x,m,p=hit;y=trim(x,m.start(),m.end());gm=re.search(WB(m.group(0)),y,re.I)
  if not gm:miss+=1;continue
  ents=[e for e,_,_ in K.ents(y)]+re.findall(r"\b\d[\d,.]*\b",y);sh=shape(r["answers"][0])
  op,g=pick([m.group(0)]+r["answers"],[e for e in dict.fromkeys(ents) if shape(e)==sh and NORM(e) not in NORM(r["q"])],[z["answers"][0] for z in nq if shape(z["answers"][0])==sh],np.random.default_rng(SEED+j))
  out.append({"split":r["split"],"i":r["i"],"q":r["q"],"answers":r["answers"],"title":t,"passage":y,"span":m.group(0),"pid":int(p),"options":op,"gold":g})
  if len(out)==RC_N:break
 RCF.write_text(json.dumps({"n":len(out),"tried":len(out)+miss,"no_gold_in_top":miss,"k":RC_K,"maxb":RC_MAXB,"seed":SEED,"index":EL.rel(K.IXD),"items":out},indent=0));return out
def build_d3()->list:
 src=(D3/"d3_build.py").read_text();a,b="random.Random(8675309)",'OUT = os.path.join(HERE, "d3.jsonl")';assert a in src and b in src;tmp=EV/"d3dev_raw.jsonl"
 exec(compile(src.replace(a,f"random.Random({SEED})").replace(b,f"OUT = {str(tmp)!r}"),str(D3/"d3_build.py"),"exec"),{"__name__":"d3dev","__file__":str(D3/"d3_build.py")})
 lk={json.loads(l)["dialogue"] for l in open(D3/"d3.jsonl")};R=[json.loads(l) for l in open(tmp)];R=[r for r in R if r["dialogue"] not in lk];pool=sorted({v for r in R for v in r["values"]});out=[]
 for j,r in enumerate(R):op,g=pick([r["answer"]],[v for v in r["values"] if v!=r["answer"]],pool,np.random.default_rng(SEED+j)) if r["answer"]!="UNKNOWN" else (None,None);out.append({**r,"options":op,"gold":g})
 D3F.write_text("".join(json.dumps(r)+"\n" for r in out));tmp.unlink();return out
def prereg(force:bool=False)->dict:
 assert PRE.exists(),f"write {EL.rel(PRE)} first";assert force or not MAN.exists(),f"{EL.rel(MAN)} exists; prereg is one-shot (use --force only before any model is scored)"
 EV.mkdir(parents=True,exist_ok=True);t=time.time();bm=build_bench();nq=build_nq();rc=build_rc(nq);d3=build_d3();m=EL.lock(SPLITS,MAN)
 return {"lock":EL.rel(MAN),"splits_hash":m["splits_hash"],"nq":len(nq),"rc":len(rc),"d3dev":len(d3),"bench":{k:v["n"] for k,v in bm.items()},"s":round(time.time()-t,1)}
class Ray:
 def __init__(s,pack:str,chat:bool):
  from amni.compute.ray_field import engine
  s.pack=pack;s.e=engine(pack).load();s.kw=dict(lock=True,stop_sentence=True,rep=1.0) if chat else {};s.info={"pack":EL.rel(pack),"engine":f"adam_ray {s.e.backend} L={s.e.L}","ctx":"unbounded (streaming hash state)","decode":{"greedy":True,"stop":"\\n","veto":False,**s.kw}}
 def gen(s,p:str,n:int)->str:return "".join(s.e.stream(p,max_bytes=n,temp=1.0,topk=1,stop=["\n"],veto=False,**s.kw)).strip()
 def lpb(s,P:list)->list:return [float(x) for i in range(0,len(P),BATCH) for x in s.e.score([a for a,_ in P[i:i+BATCH]],[b for _,b in P[i:i+BATCH]])]
 def logprob(s,c:bytes,x:bytes)->float:return s.lpb([(c,x)])[0]
class Proto:
 def __init__(s,cmd:str):s.cmd=cmd;s.p=subprocess.Popen(shlex.split(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1);s.info={"cmd":cmd,**s.rpc(op="hello")}
 def rpc(s,**k)->dict:
  s.p.stdin.write(json.dumps(k)+"\n");s.p.stdin.flush();l=s.p.stdout.readline();r=json.loads(l) if l else {"error":f"server exited rc={s.p.poll()}"};assert "error" not in r,r;return r
 def gen(s,p:str,n:int)->str:return U64(s.rpc(op="gen",prompt=B64(p.encode()),max=n,stop=[B64(b"\n")])["text"]).decode("utf-8","replace").strip()
 def lpb(s,P:list)->list:return [float(x) for i in range(0,len(P),64) for x in s.rpc(op="lp",items=[[B64(a),B64(b)] for a,b in P[i:i+64]])["lp"]]
 def logprob(s,c:bytes,x:bytes)->float:return s.lpb([(c,x)])[0]
 def close(s):s.p.poll() is None and (s.rpc(op="bye"),s.p.wait(30))
def serve(m)->None:
 for l in sys.stdin:
  q=json.loads(l);op=q.get("op")
  try:r={"model":m.name,"step":m.step,"fmt":m.fmt,"ctx":0,"proto":1} if op=="hello" else {"lp":m.lpb([(U64(a),U64(b)) for a,b in q["items"]])} if op=="lp" else {"text":B64(m.gen(U64(q["prompt"]).decode("utf-8","replace"),int(q["max"])).encode())} if op=="gen" else {"bye":True} if op=="bye" else {"error":f"bad op {op}"}
  except Exception as e:r={"error":repr(e)}
  print(json.dumps(r),flush=True)
  if op=="bye":return
def ci(k:int,n:int,c:float)->dict:
 p=k/max(1,n);return {"acc":round(p,4),"n":n,"se":round(math.sqrt(p*(1-p)/max(1,n)),4),"chance":round(c,4),"z":round((p-c)/math.sqrt(max(1e-12,c*(1-c)/max(1,n))),2)}
def mcf(lpb,items:list,ctx)->list:
 P=[(ctx(r).encode(),(" "+o).encode()) for r in items for o in r["options"]];lp=lpb(P) if P else [];out=[];i=0
 for r in items:k=len(r["options"]);v=np.array(lp[i:i+k]);ln=np.array([len(x) for _,x in P[i:i+k]]);i+=k;out.append((int(v.argmax())==r["gold"],int((v/ln).argmax())==r["gold"],k))
 return out
def msum(F:list)->dict:c=float(np.mean([1/k for *_,k in F])) if F else 0.0;return {"acc_sum":ci(sum(a for a,_,_ in F),len(F),c),"acc_norm":ci(sum(b for _,b,_ in F),len(F),c)}
mc=lambda lpb,items,ctx:msum(mcf(lpb,items,ctx))
def gsum(X:list,keys)->dict:return {"n":len(X),**{k:round(sum(bool(x[k]) for x in X)/max(1,len(X)),4) for k in keys}}
def mem_strict(p:str,a:str,vals:list)->bool:
 f=re.split(r"(?<=[.!?])\s",p.strip().lower(),maxsplit=1)[0];o=[v.lower() for v in vals if not clash(v,[a])]
 return bool(DECLINE.search(f)) and not any(re.search(WB(v),f) for v in o) if a=="UNKNOWN" else bool(re.search(WB(a.lower()),f)) and not any(re.search(WB(v),f) for v in o)
def bpb(lpb,path,nw:int,T:int=BPB_T)->dict:
 d=np.memmap(path,np.uint8,"r");ix=np.random.default_rng(SEED).integers(0,len(d)-T,nw);W=[bytes(d[i:i+T]) for i in ix];t=time.time();lp=lpb([(w[:1],w[1:]) for w in W])
 return {"file":EL.rel(path),"bpb":round(-float(np.sum(lp))/(nw*(T-1))/math.log(2),4),"windows":nw,"T":T,"scored_bytes":nw*(T-1),"seed":SEED,"bytes_per_s":round(nw*(T-1)/max(1e-9,time.time()-t))}
def stream_hold()->list:
 if not STREAM.exists():return []
 m=json.loads(STREAM.read_text());S=[]
 def walk(v):S.extend([v] if isinstance(v,str) and re.search(r"hold|heldout|val",v,re.I) and v.endswith(".bin") else []);[walk(x) for x in (v.values() if isinstance(v,dict) else v if isinstance(v,list) else [])]
 walk(m);return [p for p in (next((q for q in (Path(x),STREAM.parent/x,ROOT/x) if q.exists()),None) for x in dict.fromkeys(S)) if p]
def evaluate(gen,logprob,fmt:str="plain",suites=("qa","rc","mem","bpb","zs"),limit:int=0,lpb=None,log=print)->dict:
 lpb=lpb or (lambda P:[logprob(a,b) for a,b in P]);T=TPL[fmt];cut=lambda x:x[:limit] if limit else x;R={};W={}
 def gens(items,prompt,sc,mb):t=time.time();o=[sc(r,gen(prompt(r),mb)) for r in items];return o,round(time.time()-t,1)
 for s in suites:
  t=time.time();log(f"[{time.strftime('%H:%M:%S')}] suite {s}")
  if s=="qa":
   nq=json.loads(NQF.read_text());P=lambda r:T["qa"].format(q=r["q"]);it=[r for sp in ("tune","hold") for r in cut([r for r in nq if r["split"]==sp])];F=mcf(lpb,it,P);G,gs=gens(it,P,lambda r,p:{"split":r["split"],"i":r["i"],"q":r["q"],"gold":r["answers"],"pred":p,**score_qa(p,r["answers"],DECLINE)},MAXB["qa"])
   R[s]={**{sp:{"gen":gsum([x for x in G if x["split"]==sp],("em","hit","decline")),"mc":msum([f for f,r in zip(F,it) if r["split"]==sp])} for sp in ("tune","hold")},"all":{"gen":gsum(G,("em","hit","decline")),"mc":msum(F)},"gen_s":gs,"items":G}
  if s=="rc":
   it=cut(json.loads(RCF.read_text())["items"]);P=lambda r:T["rc"].format(t=r["title"],p=r["passage"],q=r["q"]);G,gs=gens(it,P,lambda r,p:{"i":r["i"],"split":r["split"],"q":r["q"],"gold":[r["span"]]+r["answers"],"pred":p,**score_qa(p,[r["span"]]+r["answers"],DECLINE)},MAXB["rc"]);R[s]={"gen":gsum(G,("em","hit","decline")),"mc":mc(lpb,it,P),"gen_s":gs,"items":G}
  if s=="mem":
   it=cut([json.loads(l) for l in open(D3F)]);G,gs=gens(it,lambda r:r["dialogue"],lambda r,p:{"kind":r["kind"],"answer":r["answer"],"pred":p,"hit":score_contains(p,r["answer"]),"strict":mem_strict(p,r["answer"],r["values"])},MAXB["mem"]);kn=[r for r in it if r["options"]]
   R[s]={"gen":gsum(G,("strict","hit")),"by_kind":{k:gsum([x for x in G if x["kind"]==k],("strict","hit")) for k in sorted({x["kind"] for x in G})},"mc":msum(F:=mcf(lpb,kn,lambda r:r["dialogue"])),"mc_by_kind":{k:msum([f for f,r in zip(F,kn) if r["kind"]==k])["acc_norm"] for k in sorted({r["kind"] for r in kn})},"gen_s":gs,"items":G}
  if s=="bpb":
   nw=min(BPB_W,limit*16) if limit else BPB_W;sh=stream_hold();R[s]={"clean2_hold":bpb(lpb,C2,nw),**({"stream_hold":bpb(lpb,sh[0],nw),"stream_hold_sha256":EL.sha256(sh[0])} if sh else {"stream_hold":"pending: "+EL.rel(STREAM)+(" has no *hold*.bin path" if STREAM.exists() else " not built yet")})}
  if s=="zs":R[s]={k:mc(lpb,cut([json.loads(l) for l in open(f)]),lambda r:T["zs"].format(q=r["q"])) for k,f in ZSF.items()} if all(f.exists() for f in ZSF.values()) else {"skipped":"bench files missing; run prereg"}
  W[s]=round(time.time()-t,1);log(f"  {s} {W[s]}s {json.dumps(headline({s:R[s]}))}")
 return {"suites":R,"suite_s":W}
def headline(S:dict)->dict:
 g=lambda *k:(lambda v:None if isinstance(v,dict) else v)(functools.reduce(lambda d,x:d.get(x) if isinstance(d,dict) else None,k,S))
 return {k:v for k,v in {"nq_em":g("qa","all","gen","em"),"nq_hit":g("qa","all","gen","hit"),"nq_mc":g("qa","all","mc","acc_norm","acc"),"rc_em":g("rc","gen","em"),"rc_hit":g("rc","gen","hit"),"rc_mc":g("rc","mc","acc_norm","acc"),"d3_strict":g("mem","gen","strict"),"d3_hit":g("mem","gen","hit"),"d3_mc":g("mem","mc","acc_norm","acc"),"bpb_clean2":g("bpb","clean2_hold","bpb"),"bpb_stream":g("bpb","stream_hold","bpb"),"arc_e":g("zs","arc_easy","acc_sum","acc"),"arc_e_norm":g("zs","arc_easy","acc_norm","acc"),"piqa":g("zs","piqa","acc_sum","acc"),"piqa_norm":g("zs","piqa","acc_norm","acc")}.items() if v is not None}
from amni.compute.ray_field import DEFAULT_PACK,BASE_PACK
NAMES={"base":("hashbase",re.search(r"base_(v\d+_s\d+k)",BASE_PACK).group(1),BASE_PACK,"plain"),"v19":("v19j2",re.search(r"_(s\d+k)_sid",DEFAULT_PACK).group(1),DEFAULT_PACK,"chat")}
COLS=(("model",None),("step",None),("NQ gen EM","nq_em"),("NQ gen hit","nq_hit"),("NQ MC","nq_mc"),("RC gen EM","rc_em"),("RC gen hit","rc_hit"),("RC MC","rc_mc"),("D3 strict","d3_strict"),("D3 hit","d3_hit"),("D3 MC","d3_mc"),("bpb clean2","bpb_clean2"),("bpb stream","bpb_stream"),("ARC-E","arc_e"),("ARC-E norm","arc_e_norm"),("PIQA","piqa"),("PIQA norm","piqa_norm"))
def table(out=EV/"table.md")->str:
 f=lambda v,k:"-" if v is None else f"{v:.4f}" if k and k.startswith("bpb") else f"{100*v:.1f}" if isinstance(v,float) else str(v)
 rows=[(json.loads(p.read_text()),p) for p in sorted(EV.glob("report_*.json"),key=os.path.getmtime)];L=["| "+" | ".join(c for c,_ in COLS)+" |","|"+"---|"*len(COLS)]
 L+=["| "+" | ".join(f((r["m"].get(k) if k else r["model"]+("" if r.get("kind")=="prod" else f" ({r.get('kind')})") if c=="model" else r["step"]),k) for c,k in COLS)+" |" for r,_ in rows]
 L+=["","Published zero-shot references (reference only, never run here; ARC is the Easy+Challenge mean, so it is not directly comparable with our ARC-Easy column):","","| model | ARC (avg) | PIQA | HellaSwag | source |","|---|---|---|---|---|"]+[f"| {k} | {v['arc_avg']} | {v['piqa']} | {v['hellaswag']} | {v['src']} |" for k,v in PUB.items()]
 L+=["","Chance: NQ/RC/D3 MC 25.0, ARC-Easy ~25 (3-5 options), PIQA 50.0. Accuracies in %, bpb in bits per byte."];s="\n".join(L)+"\n";Path(out).write_text(s);return s
def main():
 ap=argparse.ArgumentParser(description="ADAM-T eval harness (dev suites only)");ap.add_argument("cmd",choices=("prereg","run","serve","table"));ap.add_argument("--model",default="base",help="base | v19 | proto");ap.add_argument("--proto-cmd",default="",help="command speaking the JSON-lines protocol, e.g. 'crates/target/release/adam-t serve exports/adam_t/X/model.gf17'");ap.add_argument("--name",default="");ap.add_argument("--step",default="");ap.add_argument("--fmt",default="");ap.add_argument("--suites",default="qa,rc,mem,bpb,zs");ap.add_argument("--limit",type=int,default=0);ap.add_argument("--force",action="store_true");a=ap.parse_args()
 os.nice(max(0,10-os.nice(0)))
 if a.cmd=="prereg":print(json.dumps(prereg(a.force),indent=1));return
 if a.cmd=="table":print(table());return
 if a.cmd=="serve":m=Ray(NAMES[a.model][2],NAMES[a.model][3]=="chat");m.name,m.step,m.fmt=NAMES[a.model][0]+"-proto",NAMES[a.model][1],NAMES[a.model][3];serve(m);return
 assert MAN.exists(),"run prereg first";man=json.loads(MAN.read_text());t=time.time()
 M=Proto(a.proto_cmd) if a.model=="proto" else Ray(NAMES[a.model][2],NAMES[a.model][3]=="chat");nm,st,fm=(a.name or M.info.get("model") or "adam-t",a.step or str(M.info.get("step","na")),a.fmt or M.info.get("fmt","plain")) if a.model=="proto" else (a.name or NAMES[a.model][0],a.step or NAMES[a.model][1],a.fmt or NAMES[a.model][3])
 r=evaluate(M.gen,M.logprob,fm,tuple(s for s in a.suites.split(",") if s),a.limit,M.lpb,lambda x:print(x,flush=True));isinstance(M,Proto) and M.close()
 ch=EL.verify(man);kind="dry" if a.limit or ch or a.suites!="qa,rc,mem,bpb,zs" else "prod";p=EV/f"report_{re.sub(r'[^A-Za-z0-9.-]+','-',nm)}_{re.sub(r'[^A-Za-z0-9.-]+','-',st)}{'' if kind=='prod' else '_'+kind}.json"
 rep=EL.record(p,{"model":nm,"step":st,"fmt":fm,"adapter":M.info,"prereg":EL.rel(PRE),"seed":SEED,"limit":a.limit,**r,"m":headline(r["suites"]),"published_refs":PUB,"wall_s":round(time.time()-t,1),"maxrss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"threads":int(os.environ["RAYON_NUM_THREADS"]),"nice":os.nice(0)},shlex.join([sys.executable]+sys.argv),CODE,man,kind,{"packs":EL.file_hashes([NAMES[a.model][2]]) if a.model in NAMES else {}})
 print(json.dumps({"report":EL.rel(p),"kind":kind,"m":rep["m"],"splits_changed":ch,"wall_s":rep["wall_s"]},indent=1));table()
if __name__=="__main__":main()
