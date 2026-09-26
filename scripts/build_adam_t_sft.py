import os,re,sys,json,gzip,time,random,hashlib,argparse,importlib.util,collections,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get("AMNI_HOME") or Path(__file__).resolve().parents[1]);sys.path.insert(0,str(ROOT));OUT=ROOT/"data"/"adam_t"/"sft";RAW=OUT/"raw"
[os.environ.setdefault(k,v) for k,v in (("HF_HUB_DISABLE_XET","1"),("HF_HOME",str(RAW/"hf_home")),("GEN_CHAT","chat4c"),("MAXA","900"),("TOKENIZERS_PARALLELISM","false"))]
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download,HfApi
from amni.compute import decontam as DC
from amni.serve import emergency as EM
E=os.environ.get;SEED=int(E("SEED","20260926"));HOLD=int(E("HOLD_PERMIL","20"));CTX=int(E("CTX","512"));MAXP=int(E("MAXP","1200"));N=lambda k,d:int(E(k,str(d)))
def imp(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C5=imp("build_chat5_corpus",ROOT/"scripts"/"build_chat5_corpus.py");G=C5.G;NF=C5.NF
MK="https://raw.githubusercontent.com/amazon-science/mintaka/b6146ffb41562803c45df3c028b4cdad8c09dd4d/data/mintaka_%s.json"
HF={"squad":("rajpurkar/squad","7b6d24c440a36b6815f21b70d25016731768db1f",["plain_text/train-00000-of-00001.parquet"]),"adversarial_qa":("UCLNLP/adversarial_qa","c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9",["adversarialQA/train-00000-of-00001.parquet"]),"hotpot":("hotpotqa/hotpot_qa","1908d6afbbead072334abe2965f91bd2709910ab",["distractor/train-00000-of-00002.parquet","distractor/train-00001-of-00002.parquet"]),"drop":("ucinlp/drop","95cda593fae71b60b5b19f82de3fcf3298c1239c",["data/train-00000-of-00001.parquet"]),"tydiqa":("google-research-datasets/tydiqa","da78f23f9119363459acbaf46bf89426ff26c259",["secondary_task/train-00000-of-00001.parquet"]),"gsm8k":("openai/gsm8k","740312add88f781978c0658806c59bc2815b9866",["main/train-00000-of-00001.parquet"]),"dolly_ctx":("databricks/databricks-dolly-15k","bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",["databricks-dolly-15k.jsonl"]),"csqa":("tau/commonsense_qa","94630fe30dad47192a8546eb75f094926d47e155",["data/train-00000-of-00001.parquet"]),"qasc":("allenai/qasc","a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",["data/train-00000-of-00001.parquet"])}
LIC={"oasst1":("apache-2.0","OpenAssistant Conversations Dataset (OASST1), OpenAssistant (LAION) volunteers, Apache-2.0, https://huggingface.co/datasets/OpenAssistant/oasst1","chat","chat5 selection rules (scripts/build_chat5_corpus.py oasst): human-written English trees, ready_for_export, label/detoxify filters, one deepest thread (<=6 exchanges) + up to 2 alternative root replies"),
"oasst2":("apache-2.0","OpenAssistant Conversations Dataset 2 (OASST2), OpenAssistant (LAION) volunteers, Apache-2.0, https://huggingface.co/datasets/OpenAssistant/oasst2","chat","English trees not already taken from oasst1, same rules"),
"dolly":("cc-by-sa-3.0","databricks-dolly-15k, Copyright Databricks, Inc., CC BY-SA 3.0, https://huggingface.co/datasets/databricks/databricks-dolly-15k","chat","rows with empty context: instruction -> response (chat5 rules)"),
"aya":("apache-2.0","Aya Dataset, Cohere Labs, Apache-2.0, https://huggingface.co/datasets/CohereLabs/aya_dataset","chat","language_code=eng, original-annotations only (chat5 rules)"),
"mem":("cc0-own","Own work: scripts/v19_train_dialogues.py templates and value pools (CHAT=chat4c mode); distractor turns are chat5 first-turn pairs from OASST1/2 (Apache-2.0), Dolly (CC BY-SA 3.0) and Aya (Apache-2.0)","mem","memory dialogues (simple/multi/correction/pronoun/twohop/longgap/unknown), fresh seed; value pools disjoint from D3/D3-dev values"),
"emergency":("cc0-own","Own work: prompts from builder templates, replies from amni/serve/emergency.py (Law 0 table)","emerg","emergency prompts whose production Law-0 reply leads with 'Call 911 (or your local emergency number)'; plus a few informational emergency-number questions"),
"squad":("cc-by-sa-4.0","SQuAD v1.1, Rajpurkar et al. 2016, CC BY-SA 4.0, https://huggingface.co/datasets/rajpurkar/squad (passages from Wikipedia, CC BY-SA)","rc","train split; passage + up to 4 questions per dialogue, extractive answers"),
"adversarial_qa":("cc-by-sa-4.0","AdversarialQA, Bartolo et al. 2020, CC BY-SA 4.0, https://huggingface.co/datasets/UCLNLP/adversarial_qa (passages from Wikipedia, CC BY-SA)","rc","adversarialQA train config; passage + up to 4 questions per dialogue"),
"hotpot_rc":("cc-by-sa-4.0","HotpotQA, Yang et al. 2018, CC BY-SA 4.0, https://huggingface.co/datasets/hotpotqa/hotpot_qa (passages from Wikipedia, CC BY-SA)","rc","distractor train; gold paragraphs (supporting sentences) as passage, multi-hop question"),
"hotpot_cb":("cc-by-sa-4.0","HotpotQA, Yang et al. 2018, CC BY-SA 4.0, https://huggingface.co/datasets/hotpotqa/hotpot_qa","cb","distractor train questions asked closed-book (disjoint from hotpot_rc rows); some answers hedged"),
"drop":("cc-by-sa-4.0","DROP, Dua et al. 2019, CC BY-SA 4.0, https://huggingface.co/datasets/ucinlp/drop (passages from Wikipedia, CC BY-SA)","rc","train; passage + up to 4 questions per dialogue (numbers, dates, spans)"),
"tydiqa":("apache-2.0","TyDi QA, Clark et al. 2020, Apache-2.0, https://huggingface.co/datasets/google-research-datasets/tydiqa (passages from Wikipedia)","rc","secondary_task (GoldP) train, English only"),
"mintaka":("cc-by-4.0","Mintaka, Sen et al. 2022 (Amazon), CC BY 4.0, https://github.com/amazon-science/mintaka","cb","train+dev+test English questions closed-book; some answers hedged by complexity type"),
"csqa":("mit","CommonsenseQA, Talmor et al. 2019, MIT, https://huggingface.co/datasets/tau/commonsense_qa","mc","train; question + options -> answer text"),
"qasc":("cc-by-4.0","QASC, Khot et al. 2020, CC BY 4.0, https://huggingface.co/datasets/allenai/qasc","mc","train; question + options -> answer text (+ combined fact)"),
"gsm8k":("mit","GSM8K, Cobbe et al. 2021 (OpenAI), MIT, https://huggingface.co/datasets/openai/gsm8k","math","main train; worked solution on one line, calculator annotations removed, ends 'The answer is N.'"),
"dolly_ctx":("cc-by-sa-3.0","databricks-dolly-15k, Copyright Databricks, Inc., CC BY-SA 3.0, https://huggingface.co/datasets/databricks/databricks-dolly-15k","rc","rows with a context: closed_qa, information_extraction, summarization, classification (passage + instruction)")}
REJ=[("google-research-datasets/nq_open, natural_questions, and NQ derivatives (BoolQ, AmbigQA/ASQA, PAQ, EfficientQA, KILT)","cc-by-sa-3.0 / mixed","ADAM-T qa/rc dev suites are built from NQ-open train; BoolQ/PAQ/AmbigQA share the NQ query pool","excluded"),
("allenai/ai2_arc (Easy and Challenge, all splits)","cc-by-sa-4.0","ADAM-T zs suite (ARC-Easy test); whole dataset excluded to be safe","excluded"),
("ybisk/piqa (all splits)","afl-3.0","ADAM-T zs suite (PIQA valid); whole dataset excluded","excluded"),
("data/concepts/eval/d3, data/adam_t/eval/d3dev.jsonl","own","ADAM-T mem suite; new memory dialogues regenerated and checked against both","excluded"),
("mandarjoshi/trivia_qa","unknown on HF card","licence not stated on card","flagged, excluded"),
("stanfordnlp/web_questions","unknown on HF card","licence not stated on card","flagged, excluded"),
("akariasai/PopQA","none on HF card (repo code MIT)","card has no licence; would be the best source for popularity-calibrated hedging","flagged, excluded"),
("allenai/quac","mit on card, CC BY-SA 4.0 upstream","card/upstream mismatch, loading script only, passages too long for 512-token ctx","flagged, excluded"),
("allenai/quoref, allenai/ropes","cc-by-4.0","clean; left out for scope (quoref is script-only)","clean, not used"),
("allenai/social_i_qa, Rowan/hellaswag, allenai/winogrande","none on card","licence not stated","flagged, excluded"),
("allenai/openbookqa","unknown","licence unknown (also skipped by the eval prereg)","excluded"),
("allenai/sciq","cc-by-nc-3.0","non-commercial","excluded"),
("stanfordnlp/coqa, microsoft/wiki_qa, ms_marco","other / non-commercial","mixed or restrictive licences; MS MARCO is also a webeval set","excluded"),
("ChilleD/SVAMP, gsm8k test","mit","decontam eval sets (data/math)","excluded"),
("data/qa_extra (openhermes.txt, yahoo.txt, ultrachat.txt)","unclear","OpenHermes (GPT-4 outputs), Yahoo Answers (ToS), UltraChat (gpt-3.5 outputs)","excluded"),
("Grokipedia, big2, big3, chat1-4e","unclear","design.md 'never used' list","excluded"),
("data/chat6c/train.bin, data/chat10j/train.bin","own + big2","packed with big2 slices; memory dialogues regenerated instead","excluded")]+[(d,c,r,x) for d,c,r,x in C5.REJ]
Q2=re.compile(r"[^a-z0-9]+")
def nrm(s:str)->str:return Q2.sub(" ",(s or "").lower().replace("'","").replace("’","")).strip()
SYM=str.maketrans({"\u00a3":"GBP ","\u20ac":"EUR ","\u00a5":"JPY ","\u00b0":" degrees","\u2044":"/","\u2212":"-","\u00d7":"x","\u00bd":" 1/2","\u00bc":" 1/4","\u00be":" 3/4"})
def a1(t)->str:return re.sub(r"[\x00-\x1f\x7f]","",re.sub(r"\s+"," ",C5.asc(str(t or "").translate(SYM)).encode("ascii","ignore").decode())).strip()
def wk(p:str)->str:return re.sub(r"\s+([.,;:])",r"\1",re.sub(r" {2,}"," ",re.sub(r"\(\s*[,;]?\s*\)","",re.sub(r"\(/[^()]*\)","",re.sub(r"\[(?:\d+|[a-z]|citation needed|note \d+)\]|\(listen\)","",p))))).strip()
def cap(a:str)->str:return a[:1].upper()+a[1:] if a[:1].islower() and not re.match(r"^[a-z]+[A-Z]",a) else a
def fin(a:str)->str:return cap(a)+("" if re.search(r"[.!?\"')\]]$",a) else ".")
HG=["I think it's {a}, but I'm not certain.","I believe it's {a}, though I'm not completely sure.","Probably {a}, but I'd double-check that.","If I remember right, it's {a}.","My best guess is {a}, but I'm not sure."]
HY=["I think so, but I'm not certain.","I believe so, though I'm not completely sure."];HN=["I don't think so, but I'm not certain.","Probably not, though I'm not completely sure."]
def hedge(R,a:str)->str:l=a.lower().rstrip(".");return R.choice(HY) if l=="yes" else R.choice(HN) if l=="no" else R.choice(HG).format(a=a.rstrip("."))
RCT=[(60,"{t}: {p} {q}"),(15,"{p} {q}"),(8,"Based on this passage: {p} {q}"),(6,"Read this: {p} Question: {q}"),(6,"{q} Here is some context: {p}"),(5,"{t}. {p} Answer from the text above: {q}")]
def rcu(R,t,p,q):return R.choices([x for _,x in RCT],[w for w,_ in RCT])[0].format(t=t or "Passage",p=p,q=q) if t else R.choice(["{p} {q}","Based on this passage: {p} {q}","{p} Question: {q}"]).format(p=p,q=q)
def ex(s,k,t,qs=(),c=None):return {"s":s,"k":str(k),"t":[(u,a) for u,a in t],"q":list(qs),"c":c or LIC[s][2]}
def sents(p:str)->list:return [x for x in re.split(r"(?<=[.!?])\s+",p) if x]
def trim(p:str,a:str,mx:int=MAXP)->str:
 if len(p)<=mx:return p
 s=sents(p);i=next((j for j,x in enumerate(s) if a and a.lower() in x.lower()),-1)
 if i<0:return ""
 lo,hi=i,i+1
 while True:
  cands=[x for x in ((lo-1,hi),(lo,hi+1)) if x[0]>=0 and x[1]<=len(s) and len(" ".join(s[x[0]:x[1]]))<=mx]
  if not cands:break
  lo,hi=cands[0] if len(cands)==1 or hi-i>i-lo else cands[1]
 o=" ".join(s[lo:hi]);return o if len(o)<=mx else ""
def fetch(names=None)->dict:
 RAW.mkdir(parents=True,exist_ok=True);out={}
 for n,(r,v,fs) in HF.items():
  if names and n not in names:continue
  out[n]=[hf_hub_download(r,f,repo_type="dataset",revision=v,local_dir=RAW/n) for f in fs]
 if not names or "mintaka" in names:
  (RAW/"mintaka").mkdir(exist_ok=True);out["mintaka"]=[]
  for s in ("train","dev","test"):
   p=RAW/"mintaka"/f"mintaka_{s}.json";p.exists() or urllib.request.urlretrieve(MK%s,p);out["mintaka"].append(str(p))
 return out
def sha(p)->str:
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<22),b""):h.update(b)
 return h.hexdigest()
def rows(n,cols=None,bs=4096):
 for f in HF[n][2]:
  pf=pq.ParquetFile(RAW/n/f)
  for b in pf.iter_batches(batch_size=bs,columns=cols):yield from b.to_pylist()
def turns(txt:str)->list:return re.findall(r"User: ([^\n]*)\nAdam: ([^\n]*)\n\n",txt)
DEG=re.compile(r"\b\d{1,2}\.\s+\d{1,2}\.(?:\s|$)")
def chat5(st)->list:
 ds=C5.oasst(st)+C5.dolly(st)+C5.aya(st);cut=lambda t:t[:next((j for j,(_,a) in enumerate(t) if DEG.search(a)),len(t))];out=[ex(s,i,cut(turns(t))) for s,i,t,_,_ in ds];st["chat5_degenerate_list_cut"]=sum(len(e["t"])<len(turns(t)) for e,(_,_,t,_,_) in zip(out,ds));return out
IX=[];DIS=[]
def dis(st)->list:
 if not DIS:D=G.load_dis(ROOT/"data"/"chat5"/"pairs.bin");DIS.extend(d for d in D if not (IX and DC.dirty(d[0]+" "+d[1],ix=IX[0])));st["distractor_pairs"]=len(DIS);st["distractor_pairs_contam13_dropped"]=len(D)-len(DIS)
 return DIS
def mem(st)->list:
 R=random.Random(SEED+3);D=dis(st);ov=G.overlap();assert not any(v for v in ov.values()),ov
 dd=[json.loads(l) for f in (ROOT/"data"/"adam_t"/"eval"/"d3dev.jsonl",ROOT/"data"/"concepts"/"eval"/"d3"/"d3.jsonl") for l in open(f) if l.strip()]
 HX={d["dialogue"] for d in dd}|{json.loads(l)["dialogue"] for f in DC.eval_sources() if f.name=="hold_eval.jsonl" for l in open(f) if l.strip()};DV={str(v).lower() for d in dd for v in d.get("values",[])+[d.get("answer","")]}-{"unknown",""}
 DU={u for d in dd for u in re.findall(r"User: ([^\n]*)",d["dialogue"])};ks=[k for k,c in G.quota(N("MEM_N",16000)).items() for _ in range(c)];R.shuffle(ks);seen=set();out=[]
 for i,k in enumerate(ks):
  for n in range(400):
   r=G.one(R,k,D);why="dup" if r["text"] in seen else "eval_prompt" if r["prompt"] in HX else "d3_value" if r["answer"].lower() in DV or any(v in DV for v in [f[2].lower() for f in r["log"]]) else "d3_user_line" if any(u in DU for u in r["users"]) else "contam13" if IX and DC.dirty(r["text"],ix=IX[0]) else "nf" if NF.clean(r["text"])[0]!=r["text"].strip() else ""
   why and st.update(["mem_retry_"+why])
   if not why:break
  if why:st["mem_skip_"+why]+=1;continue
  seen.add(r["text"]);out.append(ex("mem",i,turns(r["text"]),r["asks"],"mem"));st["mem_kind_"+k]+=1
 sc=G.sent_checks([{"users":[u for u,_ in e["t"]],"asks":e["q"]} for e in out]);assert not sc["user_line_exact"] and not sc["user_sentence_exact"] and not sc["ask_template_norm"],sc;st["mem_distinct_asks"]=sc["distinct_ask_templates"];return out
def rcgroups(R,s,groups,per=4,capd=2)->list:
 out=[]
 for key,(t,p,qa) in groups.items():
  R.shuffle(qa)
  for d in range(min(capd,(len(qa)+per-1)//per)):
   c=qa[d*per:(d+1)*per];q0,a0=c[0];pp=trim(p,a0) if s!="drop" else p
   if not pp or len(pp)>MAXP*1.35:continue
   c=[(q,a) for q,a in (c if pp==p else c[:1]) if s=="drop" or a.lower() in pp.lower()]
   if c:out.append(ex(s,key,[(rcu(R,t,pp,c[0][0]),fin(c[0][1]))]+[(q,fin(a)) for q,a in c[1:]],[q for q,_ in c]))
 return out
def squadlike(R,s,src)->list:
 g={}
 for r in rows(src,["title","context","question","answers"]):
  p=wk(a1(r["context"]));q=a1(r["question"]);a=a1((r["answers"]["text"] or [""])[0]);k=hashlib.md5(p.encode()).hexdigest()[:16]
  if p and q and a:g.setdefault(k,(a1(r["title"]).replace("_"," "),p,[]))[2].append((q,a))
 return rcgroups(R,s,g,4,int(E("RC_CAPD","1")))
def drop(R)->list:
 g={}
 for r in rows("drop",["section_id","passage","question","answers_spans"]):
  p=a1(r["passage"]);q=a1(r["question"]);sp=[a1(x) for x in r["answers_spans"]["spans"] if a1(x)];a=", ".join(dict.fromkeys(sp))
  if p and q and a and len(p)<=MAXP*1.35:g.setdefault(hashlib.md5(p.encode()).hexdigest()[:16],(None,p,[]))[2].append((q,a))
 return rcgroups(R,"drop",g,4,int(E("DROP_CAPD","2")))
def hotpot(R)->list:
 rc,cb=[],[];nr,nc=N("HOTPOT_RC",30000),N("HOTPOT_CB",20000);ids=[]
 for r in rows("hotpot",["id","question","answer","type","level","supporting_facts","context"],2000):
  u=R.random();q=a1(r["question"]);a=a1(r["answer"])
  if not q or not a:continue
  if u<.4 and len(cb)<nc:cb.append(ex("hotpot_cb",r["id"],[(q,hedge(R,a) if R.random()<float(E("HEDGE_HOTPOT",".35")) else fin(a))],[q]));continue
  if len(rc)>=nr:continue
  ctx=dict(zip(r["context"]["title"],r["context"]["sentences"]));sf=collections.defaultdict(set);[sf[t].add(i) for t,i in zip(r["supporting_facts"]["title"],r["supporting_facts"]["sent_id"])]
  ps=[f"{a1(t)}: "+" ".join(a1(x) for j,x in enumerate(ctx.get(t,[])) if j in sf[t] or j==0 or len("".join(ctx.get(t,[])))<=500) for t in sf if t in ctx];p=" ".join(ps)
  if len(p)<=MAXP and (a.lower() in ("yes","no") or a.lower() in p.lower()):rc.append(ex("hotpot_rc",r["id"],[(rcu(R,None,p,q),fin(a))],[q]))
 return rc+cb
def tydi(R)->list:
 out=[]
 for r in rows("tydiqa",["id","title","context","question","answers"]):
  if not r["id"].startswith("english"):continue
  p=wk(a1(r["context"]));q=a1(r["question"]);a=a1((r["answers"]["text"] or [""])[0]);q=q if q.endswith("?") else q+"?";pp=trim(p,a) if a else ""
  if pp and a.lower() in pp.lower():out.append(ex("tydiqa",hashlib.md5(pp.encode()).hexdigest()[:16],[(rcu(R,a1(r["title"]),pp,q),fin(a))],[q]))
 return out
def mans(a)->str:
 v=a.get("answer") or [];t=a.get("answerType")
 xs=[a1((x.get("label") or {}).get("en") or "") if isinstance(x,dict) else ("Yes" if x is True else "No" if x is False else a1(x)) for x in v]
 xs=[x for x in xs if x];return "" if not xs else xs[0] if len(xs)==1 else ", ".join(xs[:-1])+" and "+xs[-1] if len(xs)<=4 else ""
def mintaka(R)->list:
 out=[]
 for f in sorted((RAW/"mintaka").glob("mintaka_*.json")):
  for r in json.load(open(f)):
   q=a1(r["question"]);a=mans(r["answer"]);h=float(E("HEDGE_MINTAKA_GENERIC",".1")) if r.get("complexityType")=="generic" else float(E("HEDGE_MINTAKA",".25"))
   if q and a:out.append(ex("mintaka",r["id"],[(q,hedge(R,a) if R.random()<h else fin(a))],[q]))
 return out
def opts(xs):return ", ".join(xs[:-1])+", or "+xs[-1]
def mc(R)->list:
 out=[];T=["{q} Options: {o}.","{q} ({o})","{q} Is it {o}?","{q} Choose from: {o}."]
 for r in rows("csqa",["id","question","choices","answerKey"]):
  lab=r["choices"]["label"];tx=[a1(x) for x in r["choices"]["text"]];q=a1(r["question"])
  if r["answerKey"] in lab and q and all(tx):out.append(ex("csqa",r["id"],[(R.choice(T).format(q=q,o=opts(tx)),fin(tx[lab.index(r["answerKey"])]))],[q]))
 for r in rows("qasc",["id","question","choices","answerKey","combinedfact"]):
  lab=r["choices"]["label"];tx=[a1(x) for x in r["choices"]["text"]];q=a1(r["question"]);cf=a1(r["combinedfact"])
  if r["answerKey"] in lab and q and all(tx):a=fin(tx[lab.index(r["answerKey"])]);out.append(ex("qasc",r["id"],[(R.choice(T).format(q=q,o=opts(tx)),a+(" "+fin(cf) if cf and R.random()<.5 else ""))],[q]))
 return out
def gsm(R)->list:
 out=[]
 for i,r in enumerate(rows("gsm8k",["question","answer"])):
  b,_,n=r["answer"].partition("####");st=[fin(a1(re.sub(r"<<[^>]*>>","",x))) for x in b.split("\n") if a1(x)];n=a1(n).replace(",","")
  if n and st:out.append(ex("gsm8k",i,[(a1(r["question"]),a1(" ".join(st))+f" The answer is {n}.")],[a1(r["question"])]))
 return out
def dolly_ctx(R,st)->list:
 out=[]
 for i,l in enumerate(open(RAW/"dolly_ctx"/"databricks-dolly-15k.jsonl")):
  r=json.loads(l);c=a1(r["context"]);ins=a1(r["instruction"]);a=a1(C5.pair.__globals__["short"](r["response"])) if "```" not in r["response"] else ""
  if r["category"] not in ("closed_qa","information_extraction","summarization","classification") or not c or not ins or len(a)<2 or len(c)>MAXP*1.35 or C5.IDN.search(a):st["dolly_ctx_skip"]+=1;continue
  out.append(ex("dolly_ctx",f"dolly:{i}",[(f"{c} {ins}" if R.random()<.5 else f"{ins} {c}",a)],[ins]))
 return out
SUBJ=[("my dad","he"),("my mom","she"),("my grandpa","he"),("my grandma","she"),("my husband","he"),("my wife","she"),("my brother","he"),("my sister","she"),("my son","he"),("my daughter","she"),("my roommate","they"),("my friend","they"),("my neighbor","they"),("my coworker","they"),("my uncle","he"),("my aunt","she"),("my boyfriend","he"),("my girlfriend","she"),("a guy at the bus stop","he"),("an old lady in the store","she"),("my partner","they"),("my teenage son","he")]
T3=["{S} collapsed and {p} {be} not breathing","{S} passed out and won't wake up","{S} is lying on the floor and not responding","{S} is choking on food and can't breathe","{S} has crushing chest pain and is sweating a lot","{S}'s face is drooping on one side and {p} can't lift {q} arm","{S} is slurring {q} words all of a sudden and {q} arm is numb","{S} cut {q} hand with a knife and it won't stop bleeding","{S} took a whole bottle of pills and {p} {be} barely awake","{S} swallowed some bleach","{S} is having a seizure and shaking all over","{S} fell off a ladder and hit {q} head hard","{S} got stung by a bee and {q} throat is swelling up","{S} ate peanuts and now {p} can't breathe","{S} got shocked by a wire and isn't moving","{S} was pulled out of the pool and isn't breathing","{S} has been in the sun all day and is confused and not sweating","{S} is diabetic and is shaking and confused","{S} is having an asthma attack and the inhaler isn't helping","{S} fell down the stairs and can't get up"]
T1=["I think I'm having a heart attack","my chest hurts really bad and my left arm is numb","I can't breathe and my throat feels tight","I cut my leg on glass and it's bleeding a lot","I took way too many of my sleeping pills","someone is breaking into my house right now","my house is on fire","I smell gas all through my apartment","our carbon monoxide alarm is going off and we all have headaches","I don't want to live anymore","I'm going to kill myself tonight","my baby isn't breathing","my toddler is locked in the car and it's really hot out","there's been a car crash and someone is trapped inside","my water broke and the baby is coming right now","my boyfriend is hitting me and I'm scared","someone is outside my door with a knife","my kitchen pan caught fire","my toddler swallowed a button battery","I'm bleeding and dizzy after I fell","there's a man with a gun in the building","my kid is missing and I can't find him anywhere"]
PRE=["","help ","please help, ","oh my god ","hurry, ","what do I do, ","I need help, "];SUF=["",", what do I do?"," please help"," right now","!",". What should I do?"]
INF=[("What's the emergency number in the UK?","999. You can also dial 112 in the UK and across the EU."),("What number do you call for an emergency in Australia?","000, often called triple zero."),("What's the emergency number in most of Europe?","112. It works across the EU, and many countries also keep their own numbers."),("What's the emergency phone number in the US?","911. It connects you to police, fire, and ambulance services."),("What is the emergency number in Japan?","110 for police, and 119 for fire and ambulance."),("What's the phone number for poison control in the US?","1-800-222-1222. It's free and open 24/7."),("What's the emergency number in India?","112. It's the single emergency number for police, fire, and ambulance."),("What number do you dial for an ambulance in New Zealand?","111.")]
def emergency(R,st)->list:
 D=dis(st);cand=[];out=[];seen=set()
 for S,p in SUBJ:
  for t in T3:cand.append(t.format(S=S,p=p,be="are" if p=="they" else "is",q={"he":"his","she":"her","they":"their"}[p]))
 cand+=T1;cand=[R.choice(PRE)+c+R.choice(SUF) for c in cand for _ in range(10 if c in T1 else 3)];cand=[c[:1].upper()+c[1:] if R.random()<.6 else c for c in cand]
 for i,c in enumerate(dict.fromkeys(cand)):
  r=EM.respond(c)
  if not r or not r.startswith("Call 911") or nrm(c) in seen:st["emerg_skip"]+=1;continue
  seen.add(nrm(c));pre=[R.choice(D)[:2]] if R.random()<.3 else [];out.append(ex("emergency",f"e{i}",pre+[(c,r)],[c]))
 for j,(q,a) in enumerate(INF*4):
  qq=q if j<len(INF) else R.choice(["","Quick question: ","Just curious, "])+q[:1].lower()+q[1:] if j>=len(INF) else q
  if EM.respond(qq) is None and nrm(qq) not in seen:seen.add(nrm(qq));out.append(ex("emergency",f"i{j}",[(qq,a)],[qq]))
 return out
class Bpe:
 PAT=re.compile(rb" ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+")
 def __init__(s,p):
  j=json.load(open(p));s.rk={(a,b):i for i,(a,b) in enumerate(j["merges"])};s.voc=[bytes([b]) for b in range(256)];[s.voc.append(s.voc[a]+s.voc[b]) for a,b in j["merges"]];s.L=np.array([len(v) for v in s.voc],np.int64);s.c={};s.pat=j["pattern"];assert len(s.voc)==j["vocab_size"]
 def word(s,w:bytes)->list:
  v=list(w)
  while len(v)>1:
   b=min(((s.rk[(v[i],v[i+1])],i) for i in range(len(v)-1) if (v[i],v[i+1]) in s.rk),default=None)
   if b is None:break
   v[b[1]:b[1]+2]=[256+b[0]]
  return v
 def enc(s,x:bytes)->list:
  o=[]
  for m in s.PAT.finditer(x):
   w=m.group();r=s.c.get(w);r=r if r is not None else s.c.setdefault(w,s.word(w)) if len(s.c)<(1<<21) else s.word(w);o.extend(r)
  return o
 def dec(s,ids)->bytes:return b"".join(s.voc[i] for i in ids)
def render(t)->tuple:
 x="";sp=[]
 for u,a in t:h=f"User: {u}\nAdam:";sp.append((len(x)+len(h),len(x)+len(h)+len(a)+3));x+=h+f" {a}\n\n"
 return x,sp
def tok(B,e):
 t=e["t"]
 while t:
  x,sp=render(t);b=x.encode("ascii");ids=np.array(B.enc(b),np.int64)
  if len(ids)<=CTX:break
  t=t[:-1] if e["c"]!="mem" else []
 if not t:return None
 bm=np.zeros(len(b)+1,np.uint8);[bm.__setitem__(slice(a,z),1) for a,z in sp];L=B.L[ids];s0=np.cumsum(L)-L;m=bm[s0];assert (m==bm[s0+L-1]).all() and s0[-1]+L[-1]==len(b),("mask/token misalignment",e["s"],e["k"])
 return ids.astype(np.uint16),m,len(t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fetch-only",action="store_true");ap.add_argument("--no-fetch",action="store_true");ap.add_argument("--only",default="");a=ap.parse_args();t0=time.time();p=lambda *x:print(time.strftime("%H:%M:%S"),*x,flush=True)
 fl={} if a.no_fetch else fetch();p("fetched",{k:len(v) for k,v in fl.items()})
 if a.fetch_only:return
 st=collections.Counter();R=random.Random(SEED);only=set(filter(None,a.only.split(",")))
 L=[("chat5",lambda:chat5(st)),("mem",lambda:mem(st)),("emergency",lambda:emergency(R,st)),("squad",lambda:squadlike(R,"squad","squad")),("adversarial_qa",lambda:squadlike(R,"adversarial_qa","adversarial_qa")),("hotpot",lambda:hotpot(R)),("drop",lambda:drop(R)),("tydiqa",lambda:tydi(R)),("mintaka",lambda:mintaka(R)),("mc",lambda:mc(R)),("gsm8k",lambda:gsm(R)),("dolly_ctx",lambda:dolly_ctx(R,st))]
 ix=DC.index(13,cache=None);IX.append(ix);XF=sorted({f for f in DC.eval_sources() if DC.group(f) in ("adam_t",) or "rknow" in str(f) or "nq" in f.name or f.name=="d3.jsonl"}|set((ROOT/"tests"/"data"/"emergency").glob("*.tsv")));ix8=DC.index(8,files=XF,cache=None);p("decontam",repr(ix),repr(ix8));X=[]
 for n,f in L:
  if only and n not in only:continue
  y=f();X+=y;p(n,len(y))
 EQ={nrm(u) for f in list(DC.eval_sources())+XF for s in DC.leaves(DC.load(f)) for u in [s]+re.findall(r"User: ([^\n]+)",s)+s.split("\n")+[x.split("\t")[-1] for x in s.split("\n")] if len(nrm(u).split())>=3};EQA={nrm(u) for f in XF for s in DC.leaves(DC.load(f)) for u in [s]+re.findall(r"User: ([^\n]+)",s)+s.split("\n")+[x.split("\t")[-1] for x in s.split("\n")] if len(nrm(u).split())>=3}
 seen=set();keep=[];dc=collections.Counter();dg=collections.Counter();ex8=[]
 for e in X:
  e["t"]=[(a1(u),a1(v)) for u,v in e["t"] if a1(u) and a1(v)];txt="".join(f"User: {u}\nAdam: {v}\n\n" for u,v in e["t"])
  if not e["t"]:dc[e["s"]+":empty"]+=1;continue
  k=hashlib.blake2b(nrm(txt).encode(),digest_size=12).digest()
  if k in seen:dc[e["s"]+":dup"]+=1;continue
  seen.add(k);chk=lambda t,x:(lambda w,q:(lambda h13,h8:("contam13" if len(h13[1]) else "contam8" if len(h8[1]) else "eval_line" if any(y in (EQA if e["s"]=="mem" else EQ) for y in q) else "nf" if e["s"] not in ("mem","emergency") and NF.clean(x)[0]!=x.strip() else "identity" if C5.IDN.search(x) else "",h13,h8,w))(DC._hits(w,ix),DC._hits(w,ix8)))(DC.words(x),[nrm(y) for y in e["q"][:len(t)]+[u for u,_ in t]]);why,(_,hp,hm),(_,hp8,hm8),ws=chk(e["t"],txt)
  [dg.__setitem__(f"13:{g}",dg[f"13:{g}"]+1) for b,g in enumerate(ix.groups) if np.any((hm>>np.uint32(b))&np.uint32(1))];[dg.__setitem__(f"8:{g}",dg[f"8:{g}"]+1) for b,g in enumerate(ix8.groups) if np.any((hm8>>np.uint32(b))&np.uint32(1))];ch=why=="contam13" and not len(hp8) and all(g.startswith("chat") for b,g in enumerate(ix.groups) if np.any((hm>>np.uint32(b))&np.uint32(1)));dc[e["s"]+":contam13_chat_hold_only"]+=ch
  if why:len(ex8)<40 and why in ("contam13","contam8") and ex8.append({"src":e["s"],"why":why,"gram":" ".join(ws[int((hp if len(hp) else hp8)[0]):][:13 if why=="contam13" else 8])});pf=next((j for j in range(len(e["t"])-1,0,-1) if e["c"]!="mem" and why in ("contam13","contam8","eval_line") and not chk(e["t"][:j],"".join(f"User: {u}\nAdam: {v}\n\n" for u,v in e["t"][:j]))[0]),0);dc[e["s"]+(":salvaged_prefix_after_"+why if pf else ":"+why)]+=1;e["t"]=e["t"][:pf]
  if not why or e["t"]:keep.append(e)
 p("kept",len(keep),"dropped",sum(dc.values()));B=Bpe(ROOT/"data"/"clean2"/"bpe4096.json");R2=random.Random(SEED+1);R2.shuffle(keep);sp={"train":[],"hold":[]}
 for e in keep:sp["hold" if int(hashlib.md5(f"{e['s']}|{e['k']}".encode()).hexdigest(),16)%1000<HOLD else "train"].append(e)
 SI={s:i for i,s in enumerate(LIC)};cnt=collections.defaultdict(collections.Counter);OUT.mkdir(parents=True,exist_ok=True);att=gzip.open(OUT/"attribution.jsonl.gz","wt");smp=[]
 for s,es in sp.items():
  I=[];M=[];ids=[];o=0
  for e in es:
   r=tok(B,e)
   if r is None:dc[e["s"]+":too_long"]+=1;continue
   t,m,nt=r;cnt[e["s"]][s+"_examples"]+=1;cnt[e["s"]][s+"_turns"]+=nt;cnt[e["s"]][s+"_tokens"]+=len(t);cnt[e["s"]][s+"_loss_tokens"]+=int(m.sum());cnt[e["s"]][s+"_bytes"]+=int(B.L[t.astype(np.int64)].sum())
   ids.append(t);M.append(m);I.append((o,len(t),SI[e["s"]],int(m.sum())));o+=len(t);att.write(json.dumps({"src":e["s"],"lic":LIC[e["s"]][0],"id":e["k"],"split":s})+"\n");s=="train" and len(smp)<50000 and smp.append((e,t,m))
  T=np.concatenate(ids);Mk=np.concatenate(M);T.tofile(OUT/f"{s}.bpe4096.u16");Mk.tofile(OUT/f"{s}.bpe4096.mask.u8");np.save(OUT/f"{s}.bpe4096.idx.npy",np.array(I,dtype=[("off","<u8"),("n","<u4"),("src","u1"),("loss","<u4")]));p(s,len(I),"examples",len(T),"tokens",int(Mk.sum()),"loss tokens")
 att.close();R3=random.Random(SEED+2);ss=R3.sample(smp,min(50,len(smp)))
 with open(OUT/"samples.txt","w") as f:
  for j,(e,t,m) in enumerate(ss):
   seg=[];cur=None
   for i,mm in zip(t.tolist(),m.tolist()):
    (seg.append([mm,[i]]) if cur!=mm else seg[-1][1].append(i));cur=mm
   f.write(f"===== sample {j+1}/50 | src={e['s']} ({LIC[e['s']][0]}) | id={e['k']} | tokens={len(t)} loss={int(m.sum())}\n"+"".join(("[[" if mm else "")+B.dec(ii).decode("ascii")+("]]" if mm else "") for mm,ii in seg)+"\n")
 tot={s:sum(v[s+"_tokens"] for v in cnt.values()) for s in sp};REV={s:HfApi().dataset_info(C5.SRC[s]["hf"]).sha for s in ("oasst1","oasst2","dolly","aya") if cnt[s]};bs=lambda s:{k:v for k,v in cnt[s].items()}
 meta={"name":"ADAM-T SFT set","built":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_s":round(time.time()-t0,1),"cmd":E("SFT_CMD"," ".join([sys.executable]+sys.argv)),"seed":SEED,"env":{k:E(k) for k in ("MAXA","SEED","HOLD_PERMIL","CTX","MAXP","MEM_N","HOTPOT_RC","HOTPOT_CB","RC_CAPD","DROP_CAPD","HEDGE_HOTPOT","HEDGE_MINTAKA","HEDGE_MINTAKA_GENERIC") if E(k)},
 "template":{"format":"User: {user}\\nAdam: {reply}\\n\\n per exchange, examples concatenated with no extra separator; ASCII only; one line per turn","why":"byte-level role markers identical to the ADAM-T eval chat fmt ('User: {q}\\nAdam:'), the D3/D3-dev memory dialogues and chat5/chat6c; BPE-4096 has no spare ids (256 bytes + 3840 merges = 4096), so <|user|>-style specials would need a vocab resize (see proposal)","proposal":"optional later: extend vocab to 4099 with <|user|> <|adam|> <|end|> (embedding/head rows +3, one-off surgery at the SFT start); not done here","pretoken_boundaries":"'Adam:' always ends a pre-token and every reply starts with ' ', so the reply/prompt boundary is a token boundary (asserted for every example)","example_limit_tokens":CTX,"multi_turn_truncation":"trailing exchanges dropped until the example fits in CTX tokens"},
 "files":{"train":"train.bpe4096.u16 (uint16 token ids)","hold":"hold.bpe4096.u16","mask":"<split>.bpe4096.mask.u8: 1 per token, 1 = token belongs to an Adam reply (incl. the closing \\n\\n); put loss only where mask==1","index":"<split>.bpe4096.idx.npy: structured (off u8 token offset, n u4 tokens, src u1 = index into sources order, loss u4 masked tokens); every example starts with 'User:'","attribution":"attribution.jsonl.gz: one row per example (src, lic, id, split) in file order","samples":"samples.txt: 50 random train examples decoded from tokens, reply tokens wrapped in [[ ]]"},
 "tokenizer":{"bpe":"data/clean2/bpe4096.json","sha256":sha(ROOT/"data"/"clean2"/"bpe4096.json"),"encoder":"Python replica of crates/adam-t/src/bpe.rs on ASCII input (pattern restricted to ASCII classes, same merge order)","pattern":B.pat},
 "tokens":tot,"sources":{s:{"license":LIC[s][0],"attribution":LIC[s][1],"kind":LIC[s][2],"use":LIC[s][3],"hf":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (C5.SRC.get(s,{}).get("hf"),))[0],"revision":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (None,REV.get(s)))[1],**bs(s)} for s in LIC},
 "share_train_tokens":{s:round(cnt[s]["train_tokens"]/max(1,tot["train"]),4) for s in LIC},"share_train_loss_tokens":{s:round(cnt[s]["train_loss_tokens"]/max(1,sum(v["train_loss_tokens"] for v in cnt.values())),4) for s in LIC},
 "raw_files":{n:[{"file":os.path.relpath(f,RAW),"sha256":sha(f)} for f in fs] for n,fs in fl.items()},"mintaka":{"url":MK%"{split}","commit":"b6146ffb41562803c45df3c028b4cdad8c09dd4d"},
 "decontam":{"module":"amni.compute.decontam","n13":{"index_key":ix.key,"files":len(ix.files),"grams":int(len(ix.H)),"groups":ix.groups},"n8_adam_t":{"index_key":ix8.key,"files":ix8.files,"grams":int(len(ix8.H)),"groups":ix8.groups},"exact_lines":len(EQ),"exact_lines_adam_t":len(EQA),"mem_exact_line_scope":"memory dialogues come from the same template family as the chat3-chat10j hold sets, so their user lines are checked against the ADAM-T/NQ/D3/emergency files only; they still pass the full 13-gram check (as retries during generation) and the D3/D3-dev prompt, value and user-line checks","rule":"drop an example on any shared 13-gram hit (all decontam.eval_sources()), any 8-gram hit against the ADAM-T/NQ/D3/emergency-test files, or any user line/question equal (normalised) to an eval string of >=3 words","hits_by_group":dict(sorted(dg.items())),"examples":ex8},
 "drops":dict(sorted(dc.items())),"stats":dict(sorted(st.items())),"rejected":[dict(zip(("dataset","card_license","reason","decision"),x)) for x in REJ],"licenses":sorted({v[0] for v in LIC.values()})}
 json.dump(meta,open(OUT/"meta.json","w"),indent=1)
 with open(OUT/"attribution.md","w") as f:
  f.write("# ADAM-T SFT set: attribution\n\nThis set mixes sources under the licences below. Rows from CC BY-SA sources (SQuAD, AdversarialQA, HotpotQA, DROP, Dolly) are share-alike: the set, and any redistribution of it, is under CC BY-SA 4.0 terms for those rows. `attribution.jsonl.gz` maps every example (file order) to its source and id.\n\n| source | licence | examples (train/hold) | tokens (train) | credit |\n|---|---|---|---|---|\n"+"".join(f"| {s} | {LIC[s][0]} | {cnt[s]['train_examples']}/{cnt[s]['hold_examples']} | {cnt[s]['train_tokens']} | {LIC[s][1]} |\n" for s in LIC)+"\nModel outputs of third-party LLMs are not used. Own work (`mem`, `emergency`) is CC0. Excluded and flagged datasets are listed in `meta.json` under `rejected`.\n")
 p("done",json.dumps({"tokens":tot,"drops":sum(dc.values())}))
if __name__=="__main__":main()
