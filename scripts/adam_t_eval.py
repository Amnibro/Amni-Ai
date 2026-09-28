import os,re,io,sys,json,time,math,base64,shlex,hashlib,argparse,functools,resource,subprocess,collections,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
[os.environ.__setitem__(k,str(min(4,int(os.environ.get(k) or 4)))) for k in ("RAYON_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_MAX_THREADS","TI_NUM_THREADS","POLARS_MAX_THREADS")];os.environ["AMNI_RAY_ARCH"]="cpu"
from amni.compute import eval_lock as EL
from adam_nightly import score_qa,NORM
from v19_eval_run import DECLINE,score_contains
from amni.serve import emergency as EMG
from amni.serve.adam_t_client import Proto as ProtoClient,B64,U64
EV=ROOT/"data"/"adam_t"/"eval";BN=EV/"bench";RK=ROOT/"data"/"concepts"/"eval"/"rknow";D3=ROOT/"data"/"concepts"/"eval"/"d3";PRE=ROOT/"docs"/"adam_t"/"eval_prereg.md";MAN=EV/"prereg_lock.json";STREAM=ROOT/"data"/"adam_t"/"stream"/"manifest.json";C2=ROOT/"data"/"clean2"/"hold.bin"
SEED=20260925;N_ZS=500;RC_N=400;RC_K=30;RC_MAXB=512;BPB_W=8192;BPB_T=256;MAXB={"qa":64,"rc":64,"mem":80,"chat":400};BATCH=256
CHD=EV/"chat";CHS=CHD/"chat_src.jsonl";CHF={s:CHD/f"chat_{s}.jsonl" for s in ("dev","test")};CHH=CHD/"chat_human40.json";CHM=CHD/"chat_meta.json";CHL=CHD/"report_test_ledger.jsonl"
CHCAT=("chat","know","explain","advice","multi","nearmiss","emergency");CHH40={"chat":7,"know":7,"explain":6,"advice":6,"multi":6,"nearmiss":4,"emergency":4};CHDEV=2/3;CHW=(1,60);CHREP=0.25;CHLOOP=0.5;CHTOP=25;CHASC=0.6;PRODS=("qa,rc,mem,bpb,zs","chat","qa,rc,mem,bpb,zs,chat")
CHMS={"C1":{"chat_ans":.50,"chat_coh":.60,"-chat_dup":.20},"C2":{"chat_ans":.80,"chat_coh":.75,"chat_fact":.20,"chat_ctx":.40,"chat_em":.75,"-chat_nmfp":.10,"-chat_dup":.10},"C3":{"chat_ans":.95,"chat_good":.60,"chat_coh":.90,"chat_fact":.35,"chat_ctx":.60,"chat_em":.90,"-chat_nmfp":.05,"-chat_dup":.05},"C4":{"chat_ans":.99,"chat_ans_lo95":.97,"chat_good":.80,"chat_coh":.95,"chat_fact":.50,"chat_ctx":.75,"chat_em":1.0,"-chat_nmfp":.03,"-chat_dup":.05}}
CHDEC=re.compile(r"\b(?:i (?:do not|don'?t|dont) (?:know|really know|have (?:an? |any )?(?:answer|opinions?|information|feelings|preferences?|favou?rites?|personal))|(?:i'?m|i am) not (?:sure|certain|able|qualified|allowed|permitted)|not sure|no idea|(?:i|we) (?:can'?t|cannot|can not|won'?t|am unable to) (?:help|answer|say|tell|assist|provide|give|discuss|comment|recall|remember|be sure|know)|i'?m unable to|i (?:have|'ve) no (?:idea|information|opinion)|(?:haven'?t|never|didn'?t|did not|have not) (?:told|tell|said|say|mentioned|mention)|as an? (?:ai|(?:large )?language model)|i'?m (?:just |only )?an? (?:ai|bot|language model|computer program|machine)|i'?d rather not|let'?s (?:talk about something else|change the subject)|ask (?:someone|somebody) else)\b",re.I)
CHDEF=re.compile(r"\b(?:as an? (?:ai|(?:large )?language model)|i'?m (?:just |only )?an? (?:ai|bot|language model|computer program)|let'?s (?:talk about something else|change the subject)|ask (?:someone|somebody) else|i (?:can'?t|cannot|won'?t) (?:help|assist) (?:you )?with (?:that|this))\b",re.I)
CH911=re.compile(r"\b(?:911|999|112|000)\b|\bemergency (?:number|services|line)\b",re.I);CHINT=re.compile(r"(?i)(?:oh no|oh god|oh|no|please|ok(?:ay)?|hey|listen|quick(?:ly)?|first|right now|immediately|now|yes)");NODEC=re.compile(r"(?!x)x")
STOP=set("a an the and or but if so of to in on at for with by from as is are was were be been being am do does did have has had i me my mine you your yours he him his she her hers it its we us our they them their this that these those what which who whom whose when where why how not no yes can could would should will shall may might must just very really too also about into out up down over than then there here all any some more most much many get got go going like know think one oh well im ive id dont cant thats let lets make say said good great sure ooh ah aah wow hi hey hello hmm yay aw aww ok okay nice cool noted thanks thank yeah yep uh um huh".split())
ARC_REV="210d026faf9955653af8916fad021475a3f00453";PIQA_REV="16c6be5562fdf2e56a6d5ee4f39a047494eb0617";PQ=f"https://raw.githubusercontent.com/ybisk/ybisk.github.io/{PIQA_REV}/piqa/data/"
BENCH={"arc_easy":{"url":[f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{ARC_REV}/ARC-Easy/test-00000-of-00001.parquet"],"license":"CC BY-SA 4.0 (allenai/ai2_arc dataset card; ARC-Easy test split)"},"piqa":{"url":[PQ+"valid.jsonl",PQ+"valid-labels.lst"],"license":"Academic Free License v3.0 (stated by the authors in ybisk.github.io/piqa/data/README.md; HF mirror card says unknown, upstream README is authoritative)"}}
NQF=EV/"nq_mc.json";RCF=EV/"rc_dev.json";D3F=EV/"d3dev.jsonl";ZSF={k:BN/f"{k}_{N_ZS}.jsonl" for k in BENCH}
SPLITS=[RK/"nq_dev_tune.json",RK/"nq_dev_hold.json",NQF,RCF,D3F,C2,*ZSF.values(),BN/"meta.json",PRE,CHS,*CHF.values(),CHH]
CODE=[Path(__file__).resolve(),*[ROOT/p for p in ("scripts/adam_nightly.py","scripts/v19_eval_run.py","amni/compute/ray_field.py","amni/compute/eval_lock.py","amni/serve/emergency.py","amni/serve/adam_t_client.py")],*sorted(ROOT.glob("adam_ray.cpython-*.so"))]
PUB={"SmolLM2-135M":{"arc_avg":43.9,"piqa":68.4,"hellaswag":42.1,"src":"huggingface.co/HuggingFaceTB/SmolLM2-135M model card (lighteval zero-shot; ARC = mean of Easy+Challenge)"},"SmolLM2-360M":{"arc_avg":53.0,"piqa":71.7,"hellaswag":54.5,"src":"huggingface.co/HuggingFaceTB/SmolLM2-360M model card (lighteval zero-shot; ARC = mean of Easy+Challenge)"},"Qwen2.5-0.5B":{"arc_avg":45.4,"piqa":69.9,"hellaswag":51.2,"src":"SmolLM2-360M model card comparison column (HF lighteval re-run; not from Qwen)"}}
TPL={"plain":{"qa":"Question: {q}\nAnswer:","rc":"{t}\n{p}\n\nQuestion: {q}\nAnswer:","zs":"Question: {q}\nAnswer:"},"chat":{"qa":"User: {q}\nAdam:","rc":"User: {t}: {p} {q}\nAdam:","zs":"User: {q}\nAdam:"}}
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
CHT=lambda s:re.findall(r"[a-z0-9]+(?:'[a-z]+)?",(s or "").lower())
FS=lambda p:re.split(r"(?<=[.!?])\s",(p or "").strip(),maxsplit=1)[0]
def stem(w:str)->str:
 w=w.replace("'","");a,b=next(((a,b) for a,b in (("sses","ss"),("ies","i"),("xes","x"),("ches","ch"),("shes","sh"),("ss","ss"),("ing",""),("ed",""),("ly",""),("s","")) if w.endswith(a) and len(w)-len(a)+len(b)>=3),("",""));w=w[:len(w)-len(a)]+b if a else w;w=w[:-1]+("" if w[-1]=="e" else "i") if len(w)>3 and w[-1] in "ey" else w
 return w[:-1] if len(w)>3 and w[-1]==w[-2] and w[-1] not in "ls" else w
def kwhit(t:list,kws:list)->list:S=[stem(x) for x in t];return [k for k in kws if any(all((t[i+j].startswith(p[:-1]) if p.endswith("*") else S[i+j]==stem(p)) for j,p in enumerate(k.split())) for i in range(len(t)-len(k.split())+1))]
cset=lambda t:{stem(x) for x in t if x.replace("'","") not in STOP}
chat_prompt=lambda r:"".join(f"User: {u}\nAdam: {a}\n\n" for u,a in r["turns"])+TPL["chat"]["qa"].format(q=r["q"])
def chat_score(r:dict,p:str,maxb:int=MAXB["chat"])->dict:
 p=(p or "").strip();t=CHT(p);fs=FS(p);nw=len(t);tq=CHT(r["q"]);cr,cq=cset(t),cset(tq);g=bool(r["gold"]) and score_qa(p,r["gold"],NODEC)["hit"];kq=set(kwhit(tq,r["kw"]));kh=[k for k in kwhit(t[:CHTOP],r["kw"]) if k not in kq];ch=kwhit(t,r["ctx"]);g3=[tuple(t[i:i+3]) for i in range(max(0,nw-2))];rep=round(1-len(set(g3))/len(g3),4) if len(g3)>=3 else 0.0
 ns=[c for c in p if not c.isspace()];asc=all(31<ord(c)<127 for c in p) and sum(c.isalnum() for c in ns)>=CHASC*len(ns);echo=bool(cr) and cr<=cq and not g;leak=bool(re.search(r"\b(?:User|Adam):",p));cap=len(p.encode())>=maxb-2
 dec=not g and bool(CHDEC.search(fs) or CHDEF.search(p));top=bool(kh) or g or (r["cat"]!="know" and len(cset(t[:CHTOP])&cq)>=2) or (r["cat"]=="know" and fs==p and nw<=4 and bool(cr-cq));ans=nw>0 and not dec and top and not echo and rep<=CHLOOP;coh=nw>0 and asc and CHW[0]<=nw<=CHW[1] and rep<=CHREP and not echo and not leak and not cap;cl=[x.strip() for x in re.split(r"[,;:.!?]|\bthen\b|\band\b",p) if x.strip()];em=bool(cl) and bool(CH911.search(cl[1] if len(cl)>1 and CHINT.fullmatch(cl[0]) else cl[0]));fp=bool(EMG.OUT.search(p));c=r["cat"]
 good=ans and coh and (g or not r["gold"]) and (bool(ch) or not r["ctx"]) and (em or c!="emergency") and (not fp or c!="nearmiss")
 return {"id":r["id"],"cat":c,"pred":p,"words":nw,"empty":nw==0,"decline":dec,"on_topic":top,"echo":echo,"answered":ans,"coherent":coh,"good":good,"ascii":asc,"rep3":rep,"role_leak":leak,"capped":cap,"kw":kh,**({"diff":r["diff"]} if r.get("diff") else {}),**({"fact":g,"fact_first":score_qa(fs,r["gold"],NODEC)["hit"]} if r["gold"] else {}),**({"ctx":bool(ch),"ctx_hits":ch} if r["ctx"] else {}),**({"em_lead":em,"em_any":fp} if c=="emergency" else {}),**({"em_fp":fp} if c=="nearmiss" else {})}
def chat_sum(G:list)->dict:
 K=("answered","good","coherent","decline","on_topic","echo","empty","capped","role_leak");A=lambda X:{"n":len(X),**{k:round(sum(bool(x[k]) for x in X)/max(1,len(X)),4) for k in K}};f=lambda X,k:ci(sum(bool(x[k]) for x in X if k in x),sum(k in x for x in X),0.0)
 T=[w for x in G for w in CHT(x["pred"])];B=list(zip(T,T[1:]));P=collections.Counter(x["pred"] for x in G);top=P.most_common(1)[0] if P else ("",0)
 return {**A(G),**{k:f(G,k) for k in ("answered","good","coherent")},"fact":f(G,"fact"),"fact_know":{d:f([x for x in G if x.get("diff")==d],"fact") for d in ("easy","medium","hard")},"fact_first":f(G,"fact_first"),"ctx":f(G,"ctx"),"em_lead":f(G,"em_lead"),"em_any":f(G,"em_any"),"nm_fp":f(G,"em_fp"),"rep3_mean":round(float(np.mean([x["rep3"] for x in G])) if G else 0.0,4),"words_mean":round(float(np.mean([x["words"] for x in G])) if G else 0.0,2),"distinct1":round(len(set(T))/max(1,len(T)),4),"distinct2":round(len(set(B))/max(1,len(B)),4),"dup_rate":round(1-len(P)/max(1,len(G)),4),"top_reply":{"text":top[0],"share":round(top[1]/max(1,len(G)),4)},"by_cat":{c:A([x for x in G if x["cat"]==c]) for c in CHCAT}}
def chat_ms(m:dict)->dict:return {k:all(m.get(q.lstrip("-")) is not None and (m[q.lstrip("-")]<=v if q[0]=="-" else m[q]>=v) for q,v in T.items()) for k,T in CHMS.items()}
def build_chat(force:bool=False)->dict:
 man=json.loads(MAN.read_text());assert force or EL.rel(CHS) not in man["splits"],"chat suite already locked (one-shot)";R=[json.loads(l) for l in open(CHS)];ids=[r["id"] for r in R];assert len(set(ids))==len(ids) and {r["cat"] for r in R}==set(CHCAT)
 bad=[r["id"] for r in R if kwhit(CHT(r["q"]),r["ctx"]) or not r["kw"] or (r["cat"]=="know" and not r["gold"]) or (r["cat"]=="multi")!=bool(r["turns"] and r["ctx"]) or not all(x.isascii() for x in [r["q"],*r["kw"],*r["gold"],*r["ctx"],*sum(r["turns"],[])])];assert not bad,f"invalid items {bad}"
 k=collections.Counter(" ".join(CHT(" ".join(sum(r["turns"],[]))+" "+r["q"])) for r in R);assert not [q for q,v in k.items() if v>1],"duplicate prompts"
 S={}
 for j,c in enumerate(CHCAT):X=sorted(r["id"] for r in R if r["cat"]==c);n=round(len(X)*CHDEV);S.update({X[i]:"dev" if o<n else "test" for o,i in enumerate(np.random.default_rng(SEED+j).permutation(len(X)).tolist())})
 [f.write_text("".join(json.dumps({**r,"split":s})+"\n" for r in R if S[r["id"]]==s)) for s,f in CHF.items()];h={c:sorted(np.random.default_rng(SEED+100+j).choice(sorted(r["id"] for r in R if r["cat"]==c and S[r["id"]]=="dev"),CHH40[c],replace=False).tolist()) for j,c in enumerate(CHCAT)}
 CHH.write_text(json.dumps({"seed":SEED,"n":sum(CHH40.values()),"from":"dev","by_cat":h,"ids":[i for c in CHCAT for i in h[c]]},indent=1))
 meta={"n":len(R),"by_cat":{c:{s:sum(1 for r in R if r["cat"]==c and S[r["id"]]==s) for s in CHF} for c in CHCAT},"n_split":{s:sum(v==s for v in S.values()) for s in CHF},"dev_frac":CHDEV,"seed":SEED,"human40":CHH40,"emergency_detector_fires":{c:sum(bool(EMG.detect(r["q"])) for r in R if r["cat"]==c) for c in ("emergency","nearmiss")},"license":R[0]["license"],"source":EL.rel(CHS),"files":{s:EL.rel(f) for s,f in CHF.items()}}
 CHM.write_text(json.dumps(meta,indent=1));m=EL.amend(MAN,SPLITS,"ADAM-T-CHATEVAL: add the chat / answer-rate suite (data/adam_t/eval/chat) and its pre-registration section; no locked split changed, no ADAM-T model scored before this amendment",[PRE])
 return {**meta,"lock":EL.rel(MAN),"splits_hash":m["splits_hash"],"amendment":m["amendments"][-1]}
def human(rep:dict,out=None)->str:
 H=json.loads(CHH.read_text());by={x["id"]:x for x in rep["suites"]["chat"]["items"]};src={r["id"]:r for s in CHF.values() for r in map(json.loads,open(s))};yn=lambda x,k:("yes" if x[k] else "NO") if k in x else None
 L=["# ADAM-T chat: human read","",f"Model `{rep['model']}`, step `{rep['step']}`, report kind `{rep['kind']}`, {rep['eval_lock']['timestamp']}. These are the 40 frozen dev items in `{EL.rel(CHH)}`, stratified by category. Rate each reply from 1 to 5 for **helpful** (did it actually answer or help?) and **natural** (does it read like a good chat partner?). The auto flags come from the pre-registered scorer; disagree with them freely.",""]
 for i,k in enumerate(H["ids"],1):
  r=src[k];x=by.get(k);fl=", ".join(f"{n} {v}" for n,v in ((n,yn(x,n)) for n in ("answered","coherent","fact","ctx","em_lead","em_fp")) if v) if x else "not generated in this run"
  L+=[f"## {i}. {r['cat']} `{k}`",""]+[f"> **User:** {u}  \n> **Adam:** {a}  " for u,a in r["turns"]]+[f"> **User:** {r['q']}","",f"**Adam (model):** {(x['pred'] or '(empty reply)') if x else '-'}","",f"Auto: {fl}","","Helpful: ___ /5, Natural: ___ /5, Note:",""]
 s="\n".join(L);out and Path(out).write_text(s);return s
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
class Proto(ProtoClient):
 def gen(s,p:str,n:int)->str:return s.text(p,n,["\n"]).strip()
 def lpb(s,P:list)->list:return s.lp(P,64)
 def logprob(s,c:bytes,x:bytes)->float:return s.lpb([(c,x)])[0]
def serve(m)->None:
 for l in sys.stdin:
  q=json.loads(l);op=q.get("op")
  try:r={"model":m.name,"step":m.step,"fmt":m.fmt,"ctx":0,"proto":1} if op=="hello" else {"lp":m.lpb([(U64(a),U64(b)) for a,b in q["items"]])} if op=="lp" else {"text":B64(m.gen(U64(q["prompt"]).decode("utf-8","replace"),int(q["max"])).encode())} if op=="gen" else {"bye":True} if op=="bye" else {"error":f"bad op {op}"}
  except Exception as e:r={"error":repr(e)}
  print(json.dumps(r),flush=True)
  if op=="bye":return
def ci(k:int,n:int,c:float)->dict:
 p=k/max(1,n);m=max(1,n);w=1.96**2/m;return {"acc":round(p,4),"n":n,"se":round(math.sqrt(p*(1-p)/m),4),"chance":round(c,4),"z":round((p-c)/math.sqrt(max(1e-12,c*(1-c)/m)),2) if 0<c<1 else None,"lo95":round((p+w/2-1.96*math.sqrt(p*(1-p)/m+w/(4*m)))/(1+w),4)}
def mcf(lpb,items:list,ctx)->list:
 P=[(ctx(r).encode(),(" "+o).encode()) for r in items for o in r["options"]];lp=lpb(P) if P else [];out=[];i=0
 for r in items:k=len(r["options"]);v=np.array(lp[i:i+k]);ln=np.array([len(x) for _,x in P[i:i+k]]);i+=k;out.append((int(v.argmax())==r["gold"],int((v/ln).argmax())==r["gold"],k))
 return out
def msum(F:list)->dict:c=float(np.mean([1/k for *_,k in F])) if F else 0.0;return {"acc_sum":ci(sum(a for a,_,_ in F),len(F),c),"acc_norm":ci(sum(b for _,b,_ in F),len(F),c)}
mc=lambda lpb,items,ctx:msum(mcf(lpb,items,ctx))
def gsum(X:list,keys)->dict:return {"n":len(X),**{k:round(sum(bool(x[k]) for x in X)/max(1,len(X)),4) for k in keys}}
def mem_strict(p:str,a:str,vals:list)->bool:
 f=FS(p).lower();o=[v.lower() for v in vals if not clash(v,[a])]
 return bool(DECLINE.search(f)) and not any(re.search(WB(v),f) for v in o) if a=="UNKNOWN" else bool(re.search(WB(a.lower()),f)) and not any(re.search(WB(v),f) for v in o)
def bpb(lpb,path,nw:int,T:int=BPB_T)->dict:
 d=np.memmap(path,np.uint8,"r");ix=np.random.default_rng(SEED).integers(0,len(d)-T,nw);W=[bytes(d[i:i+T]) for i in ix];t=time.time();lp=lpb([(w[:1],w[1:]) for w in W])
 return {"file":EL.rel(path),"bpb":round(-float(np.sum(lp))/(nw*(T-1))/math.log(2),4),"windows":nw,"T":T,"scored_bytes":nw*(T-1),"seed":SEED,"bytes_per_s":round(nw*(T-1)/max(1e-9,time.time()-t))}
def stream_hold()->list:
 if not STREAM.exists():return []
 m=json.loads(STREAM.read_text());S=[]
 def walk(v):S.extend([v] if isinstance(v,str) and re.search(r"hold|heldout|val",v,re.I) and v.endswith(".bin") else []);[walk(x) for x in (v.values() if isinstance(v,dict) else v if isinstance(v,list) else [])]
 walk(m);return [p for p in (next((q for q in (Path(x),STREAM.parent/x,ROOT/x) if q.exists()),None) for x in dict.fromkeys(S)) if p]
def evaluate(gen,logprob,fmt:str="plain",suites=("qa","rc","mem","bpb","zs"),limit:int=0,lpb=None,log=print,csplit:str="dev")->dict:
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
  if s=="chat":it=cut([json.loads(l) for l in open(CHF[csplit])]);G,gs=gens(it,chat_prompt,chat_score,MAXB["chat"]);R[s]={"split":csplit,"prompt":"chat template for every model: prior turns as 'User: {u}\\nAdam: {a}\\n\\n', then 'User: {q}\\nAdam:'",**chat_sum(G),"gen_s":gs,"items":G}
  if s=="zs":R[s]={k:mc(lpb,cut([json.loads(l) for l in open(f)]),lambda r:T["zs"].format(q=r["q"])) for k,f in ZSF.items()} if all(f.exists() for f in ZSF.values()) else {"skipped":"bench files missing; run prereg"}
  W[s]=round(time.time()-t,1);log(f"  {s} {W[s]}s {json.dumps(headline({s:R[s]}))}")
 return {"suites":R,"suite_s":W}
def headline(S:dict)->dict:
 g=lambda *k:(lambda v:None if isinstance(v,dict) else v)(functools.reduce(lambda d,x:d.get(x) if isinstance(d,dict) else None,k,S))
 return {k:v for k,v in {"nq_em":g("qa","all","gen","em"),"nq_hit":g("qa","all","gen","hit"),"nq_mc":g("qa","all","mc","acc_norm","acc"),"rc_em":g("rc","gen","em"),"rc_hit":g("rc","gen","hit"),"rc_mc":g("rc","mc","acc_norm","acc"),"d3_strict":g("mem","gen","strict"),"d3_hit":g("mem","gen","hit"),"d3_mc":g("mem","mc","acc_norm","acc"),"bpb_clean2":g("bpb","clean2_hold","bpb"),"bpb_stream":g("bpb","stream_hold","bpb"),"arc_e":g("zs","arc_easy","acc_sum","acc"),"arc_e_norm":g("zs","arc_easy","acc_norm","acc"),"piqa":g("zs","piqa","acc_sum","acc"),"piqa_norm":g("zs","piqa","acc_norm","acc"),"chat_ans":g("chat","answered","acc"),"chat_ans_lo95":g("chat","answered","lo95"),"chat_good":g("chat","good","acc"),"chat_fact":g("chat","fact","acc"),"chat_ctx":g("chat","ctx","acc"),"chat_coh":g("chat","coherent","acc"),"chat_em":g("chat","em_lead","acc"),"chat_nmfp":g("chat","nm_fp","acc"),"chat_dec":g("chat","decline"),"chat_dup":g("chat","dup_rate"),"chat_d2":g("chat","distinct2")}.items() if v is not None}
from amni.compute.ray_field import DEFAULT_PACK,BASE_PACK
NAMES={"base":("hashbase",re.search(r"base_(v\d+_s\d+k)",BASE_PACK).group(1),BASE_PACK,"plain"),"v19":("v19j2",re.search(r"_(s\d+k)_sid",DEFAULT_PACK).group(1),DEFAULT_PACK,"chat")}
COLS=(("model",None),("step",None),("NQ gen EM","nq_em"),("NQ gen hit","nq_hit"),("NQ MC","nq_mc"),("RC gen EM","rc_em"),("RC gen hit","rc_hit"),("RC MC","rc_mc"),("D3 strict","d3_strict"),("D3 hit","d3_hit"),("D3 MC","d3_mc"),("bpb clean2","bpb_clean2"),("bpb stream","bpb_stream"),("ARC-E","arc_e"),("ARC-E norm","arc_e_norm"),("PIQA","piqa"),("PIQA norm","piqa_norm"),("Chat answer","chat_ans"),("Chat good","chat_good"),("Chat fact","chat_fact"),("Chat ctx","chat_ctx"),("Chat coherent","chat_coh"),("911 lead","chat_em"),("Near-miss FP","chat_nmfp"),("Chat dup","chat_dup"))
def table(out=EV/"table.md")->str:
 f=lambda v,k:"-" if v is None else f"{v:.4f}" if k and k.startswith("bpb") else f"{100*v:.1f}" if isinstance(v,float) else str(v)
 rows=[(json.loads(p.read_text()),p) for p in sorted(EV.glob("report_*.json"),key=os.path.getmtime)];L=["| "+" | ".join(c for c,_ in COLS)+" |","|"+"---|"*len(COLS)]
 L+=["| "+" | ".join(f((r["m"].get(k) if k else r["model"]+(f" [chat {r['chat_split']}{' '+r['milestone'] if r.get('milestone') else ''}]" if "chat" in r.get("suites",{}) else "")+("" if r.get("kind")=="prod" else f" ({r.get('kind')})") if c=="model" else r["step"]),k) for c,k in COLS)+" |" for r,_ in rows]
 L+=["","Published zero-shot references (reference only, never run here; ARC is the Easy+Challenge mean, so it is not directly comparable with our ARC-Easy column):","","| model | ARC (avg) | PIQA | HellaSwag | source |","|---|---|---|---|---|"]+[f"| {k} | {v['arc_avg']} | {v['piqa']} | {v['hellaswag']} | {v['src']} |" for k,v in PUB.items()]
 L+=["","Chance: NQ/RC/D3 MC 25.0, ARC-Easy ~25 (3-5 options), PIQA 50.0. Accuracies in %, bpb in bits per byte. Chat columns come from the chat suite (docs/adam_t/eval_prereg.md, 'Chat and answer rate'); Near-miss FP and Chat dup are lower-is-better."];s="\n".join(L)+"\n";Path(out).write_text(s);return s
def main():
 ap=argparse.ArgumentParser(description="ADAM-T eval harness (dev suites only; chat test split once per milestone)");ap.add_argument("cmd",choices=("prereg","chat-prereg","run","serve","table","human"));ap.add_argument("--model",default="base",help="base | v19 | proto");ap.add_argument("--proto-cmd",default="",help="command speaking the JSON-lines protocol, e.g. 'crates/target/release/adam-t serve exports/adam_t/X/model.gf17'");ap.add_argument("--name",default="");ap.add_argument("--step",default="");ap.add_argument("--fmt",default="");ap.add_argument("--suites",default="qa,rc,mem,bpb,zs",help="qa,rc,mem,bpb,zs (default) | chat | qa,rc,mem,bpb,zs,chat");ap.add_argument("--limit",type=int,default=0);ap.add_argument("--force",action="store_true");ap.add_argument("--chat-split",default="dev",choices=("dev","test"));ap.add_argument("--milestone",default="",help="C1..C4; required with --chat-split test");ap.add_argument("--report",default="");ap.add_argument("--out",default="");ap.add_argument("--tune",action="store_true",help="dev tuning run (e.g. decode settings): recorded as dry, never prod, dev split only");a=ap.parse_args()
 os.nice(max(0,10-os.nice(0)));sn=lambda x:re.sub(r"[^A-Za-z0-9.-]+","-",x)
 if a.cmd=="prereg":print(json.dumps(prereg(a.force),indent=1));return
 if a.cmd=="chat-prereg":print(json.dumps(build_chat(a.force),indent=1));return
 if a.cmd=="table":print(table());return
 if a.cmd=="human":print(human(json.loads(Path(a.report).read_text()),a.out or None));return
 if a.cmd=="serve":m=Ray(NAMES[a.model][2],NAMES[a.model][3]=="chat");m.name,m.step,m.fmt=NAMES[a.model][0]+"-proto",NAMES[a.model][1],NAMES[a.model][3];serve(m);return
 assert MAN.exists(),"run prereg first";man=json.loads(MAN.read_text());t=time.time();su=tuple(s for s in a.suites.split(",") if s);tst=a.chat_split=="test";led=[json.loads(l) for l in open(CHL)] if CHL.exists() else []
 assert not (tst and a.tune),"--tune runs never touch the chat test split";assert not tst or (a.suites=="chat" and not a.limit and a.milestone in CHMS and not EL.verify(man) and not any(x["milestone"]==a.milestone for x in led)),f"chat test split needs --suites chat, no --limit, --milestone in {list(CHMS)}, a clean lock, and is touched once per milestone ({EL.rel(CHL)})"
 M=Proto(a.proto_cmd) if a.model=="proto" else Ray(NAMES[a.model][2],NAMES[a.model][3]=="chat");nm,st,fm=(a.name or M.info.get("model") or "adam-t",a.step or str(M.info.get("step","na")),a.fmt or M.info.get("fmt","plain")) if a.model=="proto" else (a.name or NAMES[a.model][0],a.step or NAMES[a.model][1],a.fmt or NAMES[a.model][3])
 assert not tst or any(d.get("kind")=="prod" and d.get("chat_split")=="dev" and chat_ms(d["m"]).get(a.milestone) for d in (json.loads(q.read_text()) for q in EV.glob(f"report_{sn(nm)}_{sn(st)}*.json"))),f"{nm} {st} has no prod chat dev report meeting {a.milestone}; test stays untouched"
 r=evaluate(M.gen,M.logprob,fm,su,a.limit,M.lpb,lambda x:print(x,flush=True),a.chat_split);isinstance(M,Proto) and M.close();hl=headline(r["suites"]);ck="chat" in su
 ch=EL.verify(man);kind="dry" if a.limit or ch or a.tune or a.suites not in PRODS else "prod";p=EV/f"report_{sn(nm)}_{sn(st)}{f'_chattest-{a.milestone}' if tst else '_chat' if a.suites=='chat' else ''}{'_tune' if a.tune else ''}{'' if kind=='prod' else '_'+kind}.json"
 rep=EL.record(p,{"model":nm,"step":st,"fmt":fm,"adapter":M.info,"prereg":EL.rel(PRE),"seed":SEED,"limit":a.limit,**({"chat_split":a.chat_split,"milestone":a.milestone or None,"chat_milestones_met":chat_ms(hl)} if ck else {}),**r,"m":hl,"published_refs":PUB,"wall_s":round(time.time()-t,1),"maxrss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),"threads":int(os.environ["RAYON_NUM_THREADS"]),"nice":os.nice(0)},shlex.join([sys.executable]+sys.argv),CODE,man,kind,{"packs":EL.file_hashes([NAMES[a.model][2]]) if a.model in NAMES else {}})
 tst and open(CHL,"a").write(json.dumps({"milestone":a.milestone,"model":nm,"step":st,"report":EL.rel(p),"kind":kind,"met":rep["chat_milestones_met"].get(a.milestone),"at":rep["eval_lock"]["timestamp"],"head":rep["eval_lock"]["git"]["head"]})+"\n")
 hp=CHD/f"human40_{sn(nm)}_{sn(st)}{'_tune' if a.tune else ''}{'' if kind=='prod' else '_'+kind}.md" if ck and not tst else None;hp and human(rep,hp)
 print(json.dumps({"report":EL.rel(p),"kind":kind,"m":rep["m"],**({"chat_milestones_met":rep["chat_milestones_met"],"human40":hp and EL.rel(hp)} if ck else {}),"splits_changed":ch,"wall_s":rep["wall_s"]},indent=1));table()
if __name__=="__main__":main()
