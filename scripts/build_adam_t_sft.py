import os,re,sys,json,gzip,math,time,random,hashlib,argparse,importlib.util,collections,urllib.request,datetime as dt,multiprocessing as mp
from pathlib import Path
from fractions import Fraction as Fr
import numpy as np
ROOT=Path(os.environ.get("AMNI_HOME") or Path(__file__).resolve().parents[1]);sys.path.insert(0,str(ROOT));OUT=Path(os.environ.get("SFT_OUT") or ROOT/"data"/"adam_t"/"sft");RAW=Path(os.environ.get("SFT_RAW") or OUT/"raw")
[os.environ.setdefault(k,v) for k,v in (("HF_HUB_DISABLE_XET","1"),("HF_HOME",str(RAW/"hf_home")),("GEN_CHAT","chat4c"),("MAXA","900"),("TOKENIZERS_PARALLELISM","false"),("AMNI_UMEM_DIR",str(RAW/"umem")),("AMNI_UMEM_LT","1"),("AMNI_UMEM_GN","1"),("AMNI_UMEM_G2","1"))]
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download,HfApi
from amni.compute import decontam as DC,wikifact as WF
from amni.serve import emergency as EM,adam_t_client as AC,user_memory as UM
from amni.math.answer import solve_math as msolve,fmt as mfmt
E=os.environ.get;SEED=int(E("SEED","20260926"));HOLD=int(E("HOLD_PERMIL","20"));CTX=int(E("CTX","1024"));MEMCTX=int(E("MEM_CTX","512"));MAXP=int(E("MAXP","1200"));N=lambda k,d:int(E(k,str(d)))
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
"dolly_ctx":("cc-by-sa-3.0","databricks-dolly-15k, Copyright Databricks, Inc., CC BY-SA 3.0, https://huggingface.co/datasets/databricks/databricks-dolly-15k","rc","rows with a context: closed_qa, information_extraction, summarization, classification (passage + instruction)"),
"calc":("cc0-own","Own work: math questions from builder templates; the 'Calculator:' line is the exact serve text of amni/serve/adam_t_client._math (amni/math/answer.py engine); history turns are chat5 first-turn pairs (OASST1/2 Apache-2.0, Dolly CC BY-SA 3.0, Aya Apache-2.0)","calc","serve calculator context 'User: Calculator: <result>. <question>' (ADAM-T-CTX RC form); every engine result is cross-checked against an independent exact computation (mismatches dropped and logged); replies restate the result"),
"calc_gsm8k":("mit","GSM8K, Cobbe et al. 2021 (OpenAI), MIT, https://huggingface.co/datasets/openai/gsm8k; 'Calculator:' line from amni/math/answer.py (own)","calc","GSM8K train rows where the serve's math engine fires (conf >= 0.6) and its value equals the GSM8K answer: same worked solution as the gsm8k source, prompt carries the calculator line the serve would add; split follows the gsm8k row"),
"wikidata":("cc-by-sa-4.0","Wikidata (CC0 1.0, wikidata-20260921 dump, data/wikifact/v2/store) for subjects, answers and 'Label: answer.' lines; passages from English Wikipedia (CC BY-SA 4.0) retrieved by the serve's own R-KNOW wiki_v2 index (data/ks/index_rknow/wiki_v2); questions/replies own templates; history turns chat5 pairs","kctx","serve knowledge context '<Wikidata line> <title>: <passage> <title>: <passage> <question>' built by adam_t_client._know/_fit; relevant lines, natural and synthetic off-target lines (other subject, other property), passage-only prompts; off-target/unsupported answers hedged"),
"kctx":("cc-by-sa-4.0","HotpotQA, Yang et al. 2018, CC BY-SA 4.0 (distractor-train questions and answers not used by hotpot_rc/hotpot_cb); passages from English Wikipedia (CC BY-SA 4.0) retrieved by the serve's R-KNOW wiki_v2 index via adam_t_client._know; history turns chat5 pairs","kctx","long serve prompts: history turns + 1-2 retrieved passages (often one off-target) + optional Wikidata line + question, assembled by adam_t_client.template within the 2928-byte serve budget; unsupported answers hedged"),
"umem":("cc0-own","Own work: facts from scripts/v19_train_dialogues.py value pools (D3/D3-dev values excluded), stored and rendered by amni/serve/user_memory (production parser, temporary store) and adam_t_client._mem; history turns chat5 pairs","umem","serve memory context: remembered facts as earlier 'User: <fact>\\nAdam: Got it, thanks for telling me.' turns (memory budget 640 bytes) + history + question; known, corrected, told-this-session and never-told facts")}
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
VER=3;VP=OUT.parent/"sft_v2";CS=ROOT/"data"/"adam_t"/"eval"/"chat"/"chat_src.jsonl"
EXEMPT=tuple(f"data/{c}/hold_eval.jsonl" for c in ("chat3","chat4","chat4c","chat4e","chat6c","chat7e","chat8e","chat8e_r1","chat9g","chat9h","chat10j"))
EXWHY="CTO decision 2026-09-25 (ADAM-T-SFT2): the v19 chat KPIs are retired and ADAM-T chat is judged on the locked suite data/adam_t/eval/chat (commit 78dbc26, docs/adam_t/eval_prereg.md amendment 1), so the retired v19 chat hold/eval sets chat3-chat10j are exempt from decontam for this SFT set only (13-gram, exact-line and memory eval-prompt checks); amni/compute/decontam.py is unchanged and every other eval source is still checked"
def isx(f)->bool:return DC.rel(f) in EXEMPT
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
def ex(s,k,t,qs=(),c=None,**kw):return {"s":s,"k":str(k),"t":[(u,a) for u,a in t],"q":list(qs),"c":c or LIC[s][2],**kw}
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
IX=[];DIS=[];XM=[np.uint32(0)]
def hits(ws,k=0)->tuple:
 _,p,m=DC._hits(ws,IX[k]);x=(m&~XM[0])!=0 if k==0 else np.ones(len(p),bool);return p[x],m[x],m[~x]
def dirt(t:str)->str:ws=DC.words(t);return "" if not IX else "contam13" if len(hits(ws)[0]) else "contam8" if len(hits(ws,1)[0]) else "chat6" if len(hits(ws,2)[0]) else ""
def dis(st)->list:
 if not DIS:D=G.load_dis(ROOT/"data"/"chat5"/"pairs.bin");w=[dirt(d[0]+" "+d[1]) for d in D];DIS.extend(d for d,y in zip(D,w) if not y);st["distractor_pairs"]=len(DIS);st.update("distractor_pairs_dropped_"+y for y in w if y);st["distractor_pairs_exempt13_kept"]=sum(not y and len(hits(DC.words(d[0]+" "+d[1]))[2])>0 for d,y in zip(D,w))
 return DIS
def d3()->tuple:dd=[json.loads(l) for f in (ROOT/"data"/"adam_t"/"eval"/"d3dev.jsonl",ROOT/"data"/"concepts"/"eval"/"d3"/"d3.jsonl") for l in open(f) if l.strip()];return dd,{str(v).lower() for d in dd for v in d.get("values",[])+[d.get("answer","")]}-{"unknown",""}
def mem(st,B)->list:
 R=random.Random(SEED+3);D=dis(st);ov=G.overlap();assert not any(v for v in ov.values()),ov;lg=[int(x) for x in E("LONGGAP","4,7,200").split(",")];G.DK["longgap"]=tuple(lg[:2]);G.SHORT=lg[2];G._SH.clear();LG=[]
 dd,DV=d3();hd=lambda x:{json.loads(l)["dialogue"] for f in DC.eval_sources() if f.name=="hold_eval.jsonl" and isx(f)==x for l in open(f) if l.strip()}
 HX={d["dialogue"] for d in dd}|hd(False);HXX=hd(True)-HX
 DU={u for d in dd for u in re.findall(r"User: ([^\n]*)",d["dialogue"])};ks=[k for k,c in G.quota(N("MEM_N",16000)).items() for _ in range(c)];R.shuffle(ks);seen=set();out=[]
 for i,k in enumerate(ks):
  for n in range(400):
   r=G.one(R,k,D);why="dup" if r["text"] in seen else "eval_prompt" if r["prompt"] in HX else "d3_value" if r["answer"].lower() in DV or any(v in DV for v in [f[2].lower() for f in r["log"]]) else "d3_user_line" if any(u in DU for u in r["users"]) else dirt(r["text"]) or ("nf" if NF.clean(r["text"])[0]!=r["text"].strip() else "too_long" if len(B.enc(r["text"].encode("ascii","ignore")))>MEMCTX else "")
   why and st.update(["mem_retry_"+why])
   if not why:break
  if why:st["mem_skip_"+why]+=1;continue
  seen.add(r["text"]);out.append(ex("mem",i,turns(r["text"]),r["asks"],"mem"));st["mem_kind_"+k]+=1;st["mem_exempt_eval_prompt_kept"]+=r["prompt"] in HXX;st["mem_exempt13_kept"]+=len(hits(DC.words(r["text"]))[2])>0;k=="longgap" and LG.append((len(B.enc(r["text"].encode("ascii","ignore"))),r["distance_bytes"],len(r["users"])))
 L=np.array(LG or [(0,0,0)]);st.update({f"mem_longgap_{a}_{b}":int(f(L[:,j])) for j,a in enumerate(("tokens","gap_bytes","user_turns")) for b,f in (("min",np.min),("p50",np.median),("max",np.max))});st["mem_longgap_mode_turns_lo"],st["mem_longgap_mode_turns_hi"],st["mem_longgap_mode_short_bytes"]=lg
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
T3=["{S} collapsed and {p} {be} not breathing","{S} passed out and won't wake up","{S} is lying on the floor and not responding","{S} is choking on food and can't breathe","{S} has crushing chest pain and is sweating a lot","one side of {S}'s face is drooping and {p} can't lift {q} arm","{S} is slurring {q} words all of a sudden and {q} arm is numb","{S} cut {q} hand with a knife and it won't stop bleeding","{S} overdosed on pills and {p} {be} barely awake","{S} swallowed some bleach","{S} is having a seizure and shaking all over","{S} fell off a ladder and hit {q} head hard","{S} got stung by a bee and {q} throat is swelling up","{S} ate peanuts and now {p} can't breathe","{S} got shocked by a wire and isn't moving","{S} was pulled out of the pool and isn't breathing","{S} has been in the sun all day and is confused and not sweating","{S} is diabetic and is shaking and confused","{S} is having an asthma attack and the inhaler isn't helping","{S} fell down the stairs and can't get up"]
T1=["I'm pretty sure I'm having a heart attack","my chest hurts really bad and my left arm went numb","I can't breathe and my throat feels tight","I cut my leg on glass and it's bleeding a lot","I took way too many of my sleeping pills","someone is breaking into my house right now","my house is on fire","I smell gas all through my apartment","our carbon monoxide alarm is going off and we all have headaches","I don't want to live anymore","I'm going to kill myself tonight","my baby isn't breathing","my toddler is locked in the car and it's really hot out","there's been a car crash and someone is trapped inside","my water broke and the baby is coming right now","my boyfriend is hitting me and I'm scared","someone is outside my door with a knife","my kitchen pan caught fire","my toddler swallowed a button battery","I'm bleeding and dizzy after I fell","there's a man with a gun in the building","my kid is missing and I can't find him anywhere"]
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
BUD=int(E("SERVE_BUDGET","2928"));XK=("calc","kctx","umem");WK=N("WORKERS",2);CMM=[];EQS=set()
def srv(msg,hist=(),pre=(),rc=())->tuple:
 rc=[x for x in rc if x];u=" ".join(rc+[AC._nl(msg)]);p=AC.template(msg,[tuple(h)[:2] for h in hist],BUD,list(pre),rc);c=f"User: {u}\nAdam:";assert p.endswith(c),("serve prompt tail",p[-160:]);h=p[:-len(c)];T=turns(h);assert "".join(f"User: {a}\nAdam: {b}\n\n" for a,b in T)==h,("serve prompt head",h[:160]);return T,u,len(p.encode())
def hist(R,D,qa=(),w=(35,22,16,12,9,6))->list:return [R.choice(qa) if qa and R.random()<.3 else tuple(a1(x) for x in R.choice(D)[:2]) for _ in range(R.choices((0,1,2,3,4,6)[:len(w)],w)[0])]
ONES="zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split();TENS="_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def n2w(n:int)->str:return ONES[n] if n<20 else TENS[n//10]+(" "+ONES[n%10] if n%10 else "") if n<100 else ONES[n//100]+" hundred"+(" "+n2w(n%100) if n%100 else "")
CL=["what is ","what's ","whats ","how much is ","calculate ","compute ","work out ","quick question, what is ","hey, what's ","can you calculate ","what do you get for ","","","","evaluate "];CW=["","","","hey, ","quick question: ","ok so ","please "]
OPS={"+":(["{a} + {b}","{a}+{b}","{a} plus {b}"],"plus"),"-":(["{a} - {b}","{a} minus {b}","{a}-{b}"],"minus"),"x":(["{a} * {b}","{a} x {b}","{a} times {b}","{a} multiplied by {b}","{a}*{b}"],"times"),"/":(["{a} / {b}","{a} divided by {b}","{a}/{b}"],"divided by")}
UNITS=[("convert {n} miles to km",Fr("1.609344"),0),("{n} miles in km",Fr("1.609344"),0),("{n} km in miles",1/Fr("1.609344"),0),("{n} kg in pounds",1/Fr("0.45359237"),0),("{n} pounds to kg",Fr("0.45359237"),0),("{n} inches in cm",Fr("2.54"),0),("{n} cm to inches",1/Fr("2.54"),0),("{n} feet in meters",Fr("0.3048"),0),("how many ounces in {n} pounds",Fr(16),0),("how many minutes in {n} hours",Fr(60),0),("how many hours in {n} days",Fr(24),0),("how many seconds in {n} minutes",Fr(60),0),("{n} fahrenheit in celsius",None,1),("convert {n} celsius to fahrenheit",None,2)]
WP=[("If I have {a} {o} and give away {b}, how many are left?",lambda a,b:a-b,"You'd have {v} {o} left.","{a} - {b} = {v}"),("{n} has {a} {o}. {n} buys {b} more. How many {o} does {n} have now?",lambda a,b:a+b,"{n} has {v} {o} now.","{a} + {b} = {v}"),("{n} had {a} {o} and gave away {b}. How many {o} does {n} have left?",lambda a,b:a-b,"{n} has {v} {o} left.","{a} - {b} = {v}"),("There are {a} {o} in a jar. I add {b} more. How many {o} are in the jar now?",lambda a,b:a+b,"There are {v} {o} in the jar now.","{a} + {b} = {v}"),("{n} has {a} {o} and gets {b} more. How many {o} does {n} have?",lambda a,b:a+b,"{n} has {v} {o}.","{a} + {b} = {v}")]
OBJ="apples,stickers,marbles,cookies,pencils,books,cards,coins,shells,balloons,crayons,stamps".split(",")
def cn(R)->str:k=R.random();return f"{R.randint(1,99999)/100:.2f}".rstrip("0").rstrip(".") if k<.15 else str(R.randint(2,12) if k<.35 else R.randint(10,99) if k<.6 else R.randint(100,999) if k<.85 else R.randint(1000,99999))
def cw(R,s:str)->str:return n2w(int(s)) if s.isdigit() and int(s)<100 and R.random()<.2 else s
def cq(R,b:str,L=None)->str:s=(R.choice(CL) if L is None else L)+b+R.choice(["?","","?"]);return s[:1].upper()+s[1:] if R.random()<.4 else s
def dstr(R,d)->str:return R.choice([f"{WF.MON[d.month-1]} {d.day}, {d.year}",f"{WF.MON[d.month-1]} {d.day} {d.year}",f"{d.day} {WF.MON[d.month-1]} {d.year}"])
def cgen(R)->tuple:
 k=R.choices(range(11),(28,6,6,5,4,10,9,6,6,7,6))[0]
 if k==0:
  o=R.choice("+-x/");a,b=cn(R),cn(R);b=str(R.randint(2,99)) if o=="/" else b;a=str(int(b)*R.randint(2,999)) if o=="/" and R.random()<.6 else a;v={"+":Fr(a)+Fr(b),"-":Fr(a)-Fr(b),"x":Fr(a)*Fr(b),"/":Fr(a)/Fr(b)}[o];return "arith",cq(R,R.choice(OPS[o][0]).format(a=cw(R,a),b=cw(R,b))),v,None,(a,OPS[o][1],b)
 if k==1:
  a,b,c=(R.randint(2,60) for _ in range(3));t=R.randrange(3);return "arith",cq(R,["{a} + {b} x {c}","{a} plus {b} times {c}","{a} x {b} - {c}"][t].format(a=a,b=b,c=c)),Fr([a+b*c,a+b*c,a*b-c][t]),None,None
 if k==2:
  a=R.randint(2,40);t=R.randrange(4);b=R.randint(2,5 if a>12 else 9);return "arith",cq(R,["{a} squared","{a} cubed","{a} to the power of {b}","{a}^{b}"][t].format(a=a,b=b)),Fr(a)**[2,3,b,b][t],None,None
 if k==3:
  a=R.randint(2,40);t=R.randrange(5);return "arith",cq(R,["the square root of {x}","square root of {x}","sqrt({x})","sqrt {x}","cube root of {y}"][t].format(x=a*a,y=a**3)),Fr(a),None,None
 if k==4:
  n=R.randint(3,12);return "arith",cq(R,R.choice(["{n} factorial","{n}!"]).format(n=n)),Fr(math.factorial(n)),None,None
 if k==5:
  ps=R.choice(["5","10","12.5","15","18","20","25","30","40","50","60","75","80","90","8","3","45","35"]);n=R.choice([R.randint(10,99),R.randint(100,2000),R.randint(2,40)*10]);t=R.randrange(6);f=Fr(ps)*n/100
  return "sent",cq(R,["{p}% of {n}","{p} percent of {n}","{p}% off {d}{n}","increase {n} by {p}%","decrease {n} by {p}%","a {p}% tip on {n} dollars"][t].format(p=ps,n=n,d=R.choice(["","$"])),R.choice(CW) if t in (3,4) else None),[f,f,n-f,n+f,n-f,f][t],None,None
 if k==6:
  u,f,m=R.choice(UNITS);n=R.choice([R.randint(1,100),R.randint(1,20)*5]);return "sent",cq(R,u.format(n=n),R.choice(CW)),Fr(n)*f if m==0 else (Fr(n)-32)*Fr(5,9) if m==1 else Fr(n)*Fr(9,5)+32,None,None
 if k==7:
  d1=dt.date(R.randint(1950,2035),R.randint(1,12),R.randint(1,28));t=R.randrange(3);n=R.randint(2,900);d2=d1+dt.timedelta(days=n)
  return ("sent",cq(R,f"how many days between {dstr(R,d1)} and {dstr(R,d2)}",R.choice(CW)),Fr(n),None,None) if t==0 else ("sent",cq(R,f"what day of the week {'was' if d1<dt.date(2026,9,26) else 'is'} {dstr(R,d1)}",R.choice(CW)),None,f"is a {d1.strftime('%A')}",None) if t==1 else ("sent",cq(R,f"what date is {n} days after {dstr(R,d1)}",R.choice(CW)),None,f"{WF.MON[d2.month-1]} {d2.day}, {d2.year}",None)
 if k==8:
  a=R.randint(2,12);x=R.randint(-20,30);b=R.randint(1,50);s=R.choice("+-");return "alg",cq(R,R.choice(["solve {a}x {s} {b} = {c}","solve for x: {a}x {s} {b} = {c}","if {a}x {s} {b} = {c}, what is x","find x if {a}x {s} {b} = {c}"]).format(a=a,s=s,b=b,c=a*x+(b if s=="+" else -b)),R.choice(CW)),Fr(x),None,x
 if k==9:
  t=R.randrange(5);a,b=R.randint(2,60),R.randint(2,60);xs=[R.randint(1,100) for _ in range(R.randint(3,6))];n=R.randint(11,997);lo=R.randint(1,50);hi=lo+R.randint(5,500)
  return [("sent",cq(R,R.choice(["lcm of {a} and {b}","least common multiple of {a} and {b}"]).format(a=a,b=b)),Fr(a*b//math.gcd(a,b)),None,None),("sent",cq(R,R.choice(["gcd of {a} and {b}","greatest common divisor of {a} and {b}"]).format(a=a,b=b)),Fr(math.gcd(a,b)),None,None),("sent",cq(R,R.choice(["average of ","mean of ","the average of "])+", ".join(map(str,xs[:-1]))+" and "+str(xs[-1])),Fr(sum(xs),len(xs)),None,None),("sent",cq(R,R.choice(["is {n} prime","is {n} a prime number"]).format(n=n),R.choice(CW)),Fr(int(all(n%i for i in range(2,int(n**.5)+1)))),None,None),("sent",cq(R,R.choice(["sum of the numbers from {a} to {b}","sum of the integers from {a} to {b}","the sum of all numbers from {a} to {b}"]).format(a=lo,b=hi)),Fr((lo+hi)*(hi-lo+1),2),None,None)][t]
 w=R.choice(WP);a=R.randint(5,90);b=R.randint(2,a-1);v=w[1](a,b);nm=R.choice(G.NAMES);o=R.choice(OBJ);return "word",w[0].format(a=a,b=b,o=o,n=nm),Fr(v),None,(w[2].format(v=v,o=o,n=nm),w[3].format(a=a,b=b,v=v))
def exact(d:str,v)->bool:
 try:return v is None or Fr(d.replace(",",""))==v
 except (ValueError,ZeroDivisionError):return False
def cre(R,kind,body,x,v=None)->str:
 d=AC.asc(mfmt(v)) if v is not None else "";s=re.sub(r"\b(\d+)/(\d+) (\d[\d.,]*)",r"about \3",re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)","",body).strip().rstrip("."))+".";s=s if exact(d,v) or " " in d else s.replace(f" is {d}",f" is about {d}",1);m=re.fullmatch(r"(.+) = (\S+)(?: (\S+))?\.",body)
 if kind=="arith" and m:e,r,ap=m.groups();ab="" if exact(r,v) else "about ";return f"It's {r}, which is about {ap}." if ap else R.choice([cap(f"{ab}{r}."),f"It's {ab}{r}.",f"{e} is {ab}{r}." if ab else f"{e} = {r}.",f"The answer is {ab}{r}.",f"That comes to {ab}{r}."]+([f"{x[0]} {x[1]} {x[2]} is {ab}{r}."]*2 if x else []))
 return R.choice([f"x = {x}.",f"The solution is x = {x}.",s]) if kind=="alg" else R.choice([x[0],x[0],f"{x[0]} ({x[1]})"]) if kind=="word" else R.choice([s,s,s,s if re.search(r"\d/\d+ \d",body) else body])
def cline(q:str,body:str)->str:return AC.clip("Calculator: "+body,min(240,BUD-len(f"User: {q}\nAdam:".encode())-1))
def calc(R,st,D)->list:
 out=[];seen=set();n=N("CALC_N",6000);tr=0
 while len(out)<n and tr<n*8:
  tr+=1;kind,q,v,need,x=cgen(R)
  if q in seen:continue
  seen.add(q);body=AC._math(q);r=msolve(q) if body else None;val=r and r.get("value")
  if not body:st["calc_engine_silent_"+kind]+=1;continue
  if not ((need in body) if need else val is not None and abs(float(val)-float(v))<=1e-9*max(1.0,abs(float(v)))):st["calc_engine_mismatch_"+kind]+=1;len(CMM)<25 and CMM.append({"q":q,"line":body,"want":str(need or v)});continue
  T,u,pb=srv(q,hist(R,D) if R.random()<.25 else [],(),[cline(q,body)]);st["calc_kind_"+kind]+=1;out.append(ex("calc",f"c{len(out)}",T+[(u,cre(R,kind,body,x,v))],[q],"calc",lo=1,np=0,sh="calc:"+kind+(":hist" if T else ""),pb=pb))
 st["calc_tries"]=tr;return out
def calc_gsm(st)->list:
 out=[]
 for i,r in enumerate(rows("gsm8k",["question","answer"])):
  q=a1(r["question"]);b,_,n=r["answer"].partition("####");s=[fin(a1(re.sub(r"<<[^>]*>>","",x))) for x in b.split("\n") if a1(x)];n=a1(n).replace(",","");body=AC._math(q) if n and s and re.fullmatch(r"-?\d+(?:\.\d+)?",n) else ""
  if not body:continue
  rr=msolve(q);ok=bool(rr and rr.get("value") is not None and abs(float(rr["value"])-float(n))<1e-9);st["gsm8k_calc_fired"]+=1;st["gsm8k_calc_agree" if ok else "gsm8k_calc_disagree"]+=1;st["gsm8k_calc_"+("agree_" if ok else "disagree_")+str((rr or {}).get("intent"))]+=1
  if ok:T,u,pb=srv(q,(),(),[cline(q,body)]);out.append(ex("calc_gsm8k",i,T+[(u,a1(" ".join(s))+f" The answer is {n}.")],[q],"calc",lo=1,np=0,sh="calc_gsm8k",pb=pb,sk=f"gsm8k|{i}"))
 return out
def kq(q:str)->int:return min(1100,BUD-len(f"User: {q}\nAdam:".encode())-2)
def kx(q:str)->list:
 w=WF.ask(q);ok=bool(w and w.get("answer") and float(w.get("conf") or 0)>=WF.THR and (WF.TEMPORAL or not w.get("temporal")));n=kq(q);cap=min(560,n//2);K=AC._know(q,2,cap) if n>=64 else [];wl=AC.clip(f"{AC.asc(WF.store().label(w['e'])) or 'Wikidata'}: {AC.asc(w['answer'])}.",cap) if ok else None
 return [q,{"e":int(w["e"]),"a":w["answer"],"p":int(w.get("p") or 0),"conf":float(w["conf"])} if ok else None,K,wl if ok and K and K[0]==wl else None]
def kxall(qs:list,st)->dict:
 cf=RAW/"kx_cache.jsonl";C={}
 cf.exists() and [C.__setitem__(r[0],r) for l in open(cf) if l.strip() for r in [json.loads(l)]];todo=[q for q in dict.fromkeys(qs) if q not in C];st["kx_questions"]=len(set(qs));st["kx_cache_hits"]=len(set(qs))-len(todo);t0=time.time()
 with mp.get_context("spawn").Pool(WK) as pool,open(cf,"a") as f:
  for j,r in enumerate(pool.imap(kx,todo,chunksize=8)):C[r[0]]=r;f.write(json.dumps(r)+"\n");j%1000==999 and (f.flush(),print(time.strftime("%H:%M:%S"),"kx",j+1,"/",len(todo),f"{time.time()-t0:.0f}s",flush=True))
 return C
WQ={36:(["what is the capital of {s}?","what's the capital of {s}?","whats the capital of {s}","what is the capital city of {s}?","capital of {s}?","what is {s}'s capital?"],["The capital of {s} is {a}.","{a}.","It's {a}.","{a} is the capital of {s}."],10),
50:(["who wrote {s}?","who is the author of {s}?","who was the author of {s}?","who wrote the book {s}?"],["{s} was written by {a}.","{a}.","{a} wrote it.","It was written by {a}."],25),
57:(["who directed {s}?","who is the director of {s}?","who was the director of {s}?","who directed the movie {s}?"],["{s} was directed by {a}.","{a}.","{a} directed it."],25),
86:(["who composed {s}?","who is the composer of {s}?","who wrote the music for {s}?"],["{s} was composed by {a}.","{a}.","{a} composed it."],20),
175:(["who sang {s}?","who sings {s}?","who performed {s}?","who recorded {s}?"],["{s} was performed by {a}.","{a}.","That was {a}."],20),
112:(["who founded {s}?","who is the founder of {s}?","who established {s}?"],["{s} was founded by {a}.","{a}.","{a} founded it."],20),
61:(["who invented {s}?","who discovered {s}?","who is the inventor of {s}?"],["{s} was invented by {a}.","{a}.","That was {a}."],20),
84:(["who designed {s}?","who was the architect of {s}?","who is the architect of {s}?"],["{s} was designed by {a}.","{a}.","The architect was {a}."],15),
170:(["who painted {s}?","who created {s}?","who is the creator of {s}?"],["{s} was created by {a}.","{a}.","That was {a}."],20),
569:(["when was {s} born?","what is {s}'s birthday?","what is the date of birth of {s}?","when is {s}'s birthday?"],["{s} was born {pp} {a}.","{a}.","{PP} {a}."],30),
570:(["when did {s} die?","when did {s} pass away?"],["{s} died {pp} {a}.","{a}.","{PP} {a}."],30),
19:(["where was {s} born?"],["{s} was born in {a}.","In {a}.","{a}."],30),
20:(["where did {s} die?"],["{s} died in {a}.","In {a}.","{a}."],30),
571:(["when was {s} founded?","when was {s} established?","what year was {s} founded?"],["{s} was founded {pp} {a}.","{a}.","{PP} {a}."],20),
577:(["when was {s} released?","when was {s} published?","what year was {s} released?","when did {s} come out?"],["{s} came out {pp} {a}.","{a}.","It was released {pp} {a}."],25),
38:(["what currency is used in {s}?","what currency does {s} use?"],["{s} uses the {a}.","The {a}.","It's the {a}."],10),
37:(["what language is spoken in {s}?","what languages are spoken in {s}?"],["The official language of {s} is {a}.","{a}."],10),
30:(["what continent is {s} in?","which continent is {s} on?"],["{s} is in {a}.","{a}.","It's in {a}."],10),
17:(["what country is {s} in?","which country is {s} located in?"],["{s} is in {a}.","{a}.","It's in {a}."],15),
138:(["who is {s} named after?","what is {s} named after?"],["{s} is named after {a}.","{a}.","It's named after {a}."],15),
403:(["where does the {s} flow into?","where does the {s} end?"],["The {s} flows into {a}.","{a}.","Into {a}."],15)}
NMOK=re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,'&!?-]{1,46}")
def wdisp(S,f:int,yo=False)->tuple:a,fo,_=WF.render(S,int(f),yo);return (AC.asc(a) if a else ""),sorted({WF.nrm(x) for x in fo if x and len(WF.nrm(x))>=2})
def wtpl(R,p,s,a)->str:pp="on" if re.search(r"\d{1,2}, \d{3,4}$",a) else "in";return re.sub(r"(?<!\.)\.\.$",".",fin(R.choice(WQ[p][1]).format(s=s,a=a,pp=pp,PP=pp.capitalize())))
def wd_q(R,st)->list:
 S=WF.store();assert S is not None,"wikifact store missing";P=np.asarray(S.f_p);SL=np.asarray(S.e_sl);out=[];qp=N("WD_N",7500)//len(WQ)
 for p,(qt,_,msl) in WQ.items():
  F=np.flatnonzero(P==p);U=np.unique(np.asarray(S.f_s[F]));U=U[SL[U]>=msl].tolist();R.shuffle(U);k=0
  for e in U:
   if k>=qp:break
   nm=AC.asc(WF.fold(S.name(e)));F2=S.facts(e,[p])
   if not NMOK.fullmatch(nm) or nm.lower().startswith(("list of","category:")) or len(F2)!=1 or int(S.f_k[F2[0]]) not in (1,2):st["wd_skip_subject"]+=1;continue
   t=R.choice(qt);a,fo=wdisp(S,F2[0],t.startswith("what year"))
   if not a or not NMOK.fullmatch(a) or WF.nrm(a) in WF.nrm(nm) or WF.nrm(nm) in WF.nrm(a):st["wd_skip_answer"]+=1;continue
   if any(e2!=e and any(WF.nrm(wdisp(S,f3)[0]) not in fo for f3 in S.facts(e2,[p])) for e2,_ in S.ents(WF.nrm(nm))[:40]):st["wd_skip_ambiguous"]+=1;continue
   q=t.format(s=nm);q=q[:1].upper()+q[1:] if R.random()<.6 else q.lower();k+=1;out.append({"q":q,"e":int(e),"p":p,"s":nm,"a":a,"fo":fo})
  st[f"wd_q_p{p}"]=k
 return out
def kc_q(R,used:set)->list:
 out=[];n=N("KC_N",9000)
 for r in rows("hotpot",["id","question","answer"],4000):
  q,a=a1(r["question"]),a1(r["answer"])
  if r["id"] not in used and q and a and a.lower() not in ("yes","no") and len(q)<=250 and R.random()<.3:out.append({"q":q,"a":a,"id":r["id"]})
  if len(out)>=n:break
 return out
def insup(ps,fo)->bool:t=" "+" ".join(WF.nrm(x) for x in ps)+" ";return any(len(g)>=3 and f" {g} " in t for g in fo)
def know(st,X)->list:
 R=random.Random(SEED+11);D=dis(st);WC=wd_q(R,st);KC=kc_q(R,{e["k"] for e in X if e["s"] in ("hotpot_rc","hotpot_cb")});KX=kxall([c["q"] for c in WC+KC],st);S=WF.store();QA=[(c["q"],fin(c["a"])) for c in WC[::7]+KC[::7]];out=[];BP=collections.defaultdict(list);HW=(25,20,18,15,12,10)
 for c in WC:
  _,w,K,wl=KX[c["q"]];c["lab"]="rel" if wl and w["e"]==c["e"] and WF.nrm(w["a"]) in c["fo"] else "off_nat" if wl else "none";c["lab"]=="rel" and BP[c["p"]].append(c)
 for i,c in enumerate(WC):
  q=c["q"];_,w,K,wl=KX[q];lab=c["lab"];n=kq(q);cap=min(560,n//2)
  if not K:st["wd_no_context"]+=1;continue
  if lab=="rel" and R.random()<float(E("WD_SYN",".35")):
   o=[x for x in BP[c["p"]] if x["e"]!=c["e"]];F3=[(p2,f) for p2 in WQ if p2!=c["p"] for f in S.facts(c["e"],[p2])[:1]];z=R.choice(F3) if F3 and R.random()<.5 else None;d2=wdisp(S,z[1])[0] if z else ""
   nl=AC.clip(f"{AC.asc(S.label(c['e'])) or 'Wikidata'}: {d2}.",cap) if d2 and NMOK.fullmatch(d2) and WF.nrm(d2) not in c["fo"] else KX[R.choice(o)["q"]][3] if o else None
   K,lab=([nl]+K[1:],"off_syn_prop" if d2 and nl.endswith(f": {d2}.") else "off_syn_ent") if nl else (K,lab);wl=nl or wl
  rc=AC._fit(K[::-1],0,n)[0][::-1];hl=bool(wl) and bool(rc) and rc[0]==wl;ps=rc[1:] if hl else rc;sup=insup(ps,c["fo"]);lab=lab if hl else "none"
  if lab!="rel" and not sup and R.random()>float(E("WD_UNSUP",".5")):st["wd_unsup_dropped"]+=1;continue
  a=wtpl(R,c["p"],c["s"],AC.asc(w["a"])) if lab=="rel" else wtpl(R,c["p"],c["s"],c["a"]) if sup else hedge(R,c["a"]);T,u,pb=srv(q,hist(R,D,QA,HW),(),rc)
  st["wd_"+lab]+=1;st["wd_"+lab+("_sup" if sup else "_unsup")]+=1;out.append(ex("wikidata",f"w{i}",T+[(u,a)],[q],"kctx",lo=1,np=0,sh=f"wd:{lab}:{'sup' if sup else 'unsup'}:p{len(ps)}"+(":dis" if any(not insup([x],c["fo"]) for x in ps) else "")+(":hist" if T else ""),pb=pb))
 for i,c in enumerate(KC):
  q=c["q"];_,w,K,wl=KX[q]
  if not K:st["kctx_no_context"]+=1;continue
  rc=AC._fit(K[::-1],0,kq(q))[0][::-1];hl=bool(wl) and rc[0]==wl;ps=rc[1:] if hl else rc;fo=[WF.nrm(c["a"])];sup=insup(ps,fo);lab="sup" if sup else "line" if hl and WF.nrm(w["a"])==fo[0] else "unsup"
  if lab=="unsup" and R.random()>float(E("KC_UNSUP",".25")):st["kctx_unsup_dropped"]+=1;continue
  T,u,pb=srv(q,hist(R,D,QA,HW),(),rc);st["kctx_"+lab]+=1;out.append(ex("kctx",c["id"],T+[(u,fin(c["a"]) if lab!="unsup" else hedge(R,c["a"]))],[q],"kctx",lo=1,np=0,sh=f"kctx:{lab}:p{len(ps)}"+(":dis" if any(not insup([x],fo) for x in ps) else "")+(":wdline" if hl else "")+(":hist" if T else ""),pb=pb))
 return out
UQ={4:("city",["Where do I live?","Where am I living these days?","Where do I live again?","Remind me where I live.","Do you remember where I live?","Which city do I live in?","Where did I say I live?"],["Where does my {k} live?","Where does my {k} live again?","Remind me where my {k} lives.","Which city does my {k} live in?","Do you remember where my {k} lives?","Where did I say my {k} lives?"],["You live in {v}."],["Your {k} lives in {v}."],"where you live","where your {k} lives"),
5:("job",["What do I do for work?","What's my job?","Remind me what I do for work.","Do you remember my job?","What job do I have?","What did I say I do for work?"],["What does my {k} do?","What does my {k} do for a living?","What does my {k} do for work?","Remind me what my {k} does for work.","What job does my {k} have?","Do you remember what my {k} does for work?"],["You work as {a} {v}.","You're {a} {v}."],["Your {k} works as {a} {v}.","Your {k} is {a} {v}."],"what you do for work","what your {k} does for work"),
6:("names",["What's my name?","Do you remember my name?","Remind me what my name is.","What did I say my name was?"],["What's my {k}'s name?","What is my {k} called?","Do you remember my {k}'s name?","Remind me of my {k}'s name.","What did I say my {k}'s name was?"],["Your name is {v}."],["Your {k}'s name is {v}."],"your name","your {k}'s name"),
7:("like",["What do I love doing?","What did I say I love?","Remind me what I love.","Do you remember what I love?"],["What does my {k} love?","What does my {k} love doing?","Remind me what my {k} loves.","Do you remember what my {k} loves?"],["You love {v}."],["Your {k} loves {v}."],"what you love","what your {k} loves"),
8:("allergy",["What am I allergic to?","Remind me what I'm allergic to.","Do you remember my allergy?","What was I allergic to again?"],["What is my {k} allergic to?","What's my {k} allergic to?","Remind me what my {k} is allergic to.","Do you remember my {k}'s allergy?"],["You're allergic to {v}."],["Your {k} is allergic to {v}."],"what you're allergic to","what your {k} is allergic to"),
9:("drive",["What car do I drive?","What do I drive?","Remind me what car I drive.","Do you remember what I drive?"],["What does my {k} drive?","What car does my {k} drive?","Remind me what my {k} drives.","Do you remember what car my {k} drives?"],["You drive {a} {v}."],["Your {k} drives {a} {v}."],"what you drive","what your {k} drives"),
10:("color",["What's my favorite color?","Remind me of my favorite color.","Do you remember my favorite color?","What color do I like best?"],["What's my {k}'s favorite color?","Remind me of my {k}'s favorite color.","What color does my {k} like best?","Do you remember my {k}'s favorite color?"],["Your favorite color is {v}."],["Your {k}'s favorite color is {v}."],"your favorite color","your {k}'s favorite color"),
11:("food",["What's my favorite food?","Remind me of my favorite food.","Do you remember my favorite food?","What food do I like best?"],["What's my {k}'s favorite food?","Remind me of my {k}'s favorite food.","What food does my {k} like best?","Do you remember my {k}'s favorite food?"],["Your favorite food is {v}."],["Your {k}'s favorite food is {v}."],"your favorite food","your {k}'s favorite food"),
12:("study",["What did I study?","What did I major in?","Remind me what I studied.","Do you remember what I studied?"],["What did my {k} study?","What did my {k} major in?","Remind me what my {k} studied.","Do you remember what my {k} studied?"],["You studied {v}."],["Your {k} studied {v}."],"what you studied","what your {k} studied"),
13:("school",["Where did I go to school?","Where did I go to college?","Remind me where I went to school.","Which school did I go to?"],["Where did my {k} go to school?","Where did my {k} go to college?","Which school did my {k} go to?","Remind me where my {k} went to school."],["You went to {v}."],["Your {k} went to {v}."],"where you went to school","where your {k} went to school"),
14:("age",["How old am I?","Remind me how old I am.","Do you remember my age?","What's my age?"],["How old is my {k}?","How old is my {k} now?","Remind me how old my {k} is.","What's my {k}'s age?"],["You're {v}.","You're {v} years old."],["Your {k} is {v}.","Your {k} is {v} years old."],"how old you are","how old your {k} is"),
15:("birthday",["When is my birthday?","When's my birthday?","Remind me when my birthday is.","What month is my birthday?"],["When is my {k}'s birthday?","When's my {k}'s birthday?","What month is my {k}'s birthday?","Remind me when my {k}'s birthday is."],["Your birthday is in {v}."],["Your {k}'s birthday is in {v}."],"when your birthday is","when your {k}'s birthday is")}
UNK=["You haven't told me {d}.","I don't know {d}. You haven't told me yet.","You haven't mentioned {d} yet.","I don't know {d} yet. Want to tell me?"];ACKS=["Got it!","Good to know.","Thanks for telling me.","Nice, I'll remember that.","Okay, noted."]
def uval(R,rel,k,DV)->str:
 for _ in range(50):
  v=str(R.randint(*G.AGER.get(k or "I",(18,80)))) if rel==14 else R.choice(G.NAMES if rel==6 else G.POOL[UQ[rel][0]])
  if v.lower() not in DV:return v
 return v
def ustore(uid,stmt,rel,v,sid)->bool:AC._mem(stmt,[],uid,sid);return any(f.rel==rel and f.val.decode("latin-1").lower() in v.lower() for f in UM.latest(UM.load(UM.user_key(uid))))
def umem(st,DV)->list:
 R=random.Random(SEED+12);D=dis(st);assert UM.enabled(),"AMNI_USER_MEMORY must be on";out=[];n=N("UMEM_N",3000);kin=[k for k in G.PERSONS if " " not in k and "-" not in k]
 for i in range(n*4):
  if len(out)>=n:break
  uid=f"sft3-{SEED}-{i}";told={};corr=None;ins=None;UM.save(UM.user_key(uid),[])
  for _ in range(R.randint(2,9)):
   rel=R.choice(list(UQ));k=None if R.random()<.35 else R.choice(kin);v=uval(R,rel,k,DV);stmt=R.choice([t for t in UM._TPL[rel] if ("{e}" in t)==(k is not None)]).format(e=k,v=v)
   ok=ustore(uid,stmt,rel,v,"s0");corr=(k,rel) if ok and (k,rel) in told else corr;ok and told.__setitem__((k,rel),v);st["umem_fact_"+("stored" if ok else "unparsed")]+=1
  if not told:continue
  H=hist(R,D,(),(30,25,20,15,10))
  if R.random()<.2:
   rel=R.choice(list(UQ));k=None if R.random()<.35 else R.choice(kin);v=uval(R,rel,k,DV);stmt=R.choice([t for t in UM._TPL[rel] if ("{e}" in t)==(k is not None)]).format(e=k,v=v)
   ok=ustore(uid,stmt,rel,v,"s1");ins=(k,rel) if ok else None;ok and (told.__setitem__((k,rel),v),H.insert(R.randint(0,len(H)),(stmt,R.choice(ACKS))))
  kn=R.random()<.8;key=(ins if ins and R.random()<.5 else corr if corr and R.random()<.5 else R.choice(list(told))) if kn else next(((k,rel) for _ in range(30) for rel in [R.choice(list(UQ))] for k in [None if R.random()<.3 else R.choice(kin)] if (k,rel) not in told),None)
  if key is None:continue
  k,rel=key;qs=[x.format(k=k) for x in UQ[rel][1 if k is None else 2] if nrm(x.format(k=k)) not in EQS];q=R.choice(qs) if qs else None
  if q is None:st["umem_skip_all_phrasings_eval"]+=1;continue
  m=AC._mem(q,H,uid,"s1")
  if m["reply"] or (kn and not m["route"]):st["umem_skip_"+("action" if m["reply"] else "noroute")]+=1;continue
  pre=AC._fit(m["pre"],0,min(640,BUD-len(f"User: {q}\nAdam:".encode())))[0] if m["route"] else [];T,u,pb=srv(q,m["hist"],pre,[])
  if kn and not any(told[key].lower() in x.lower() for t in T for x in t[:1]):st["umem_skip_value_missing"]+=1;continue
  v=told.get(key,"");a=R.choice(UQ[rel][3 if k is None else 4]).format(k=k,v=v,a=G.art(v)) if kn else R.choice(UNK).format(d=UQ[rel][5 if k is None else 6].format(k=k));tx="".join(f"User: {x}\nAdam: {y}\n\n" for x,y in T+[(u,a)])
  y=dirt(tx) or ("eval_line" if any(nrm(x) in EQS for x,_ in T) else "")
  if y:st["umem_skip_"+y]+=1;continue
  sh="umem:"+("known" if kn else "unknown")+(":corr" if kn and key==corr else "")+(":insession" if kn and key==ins else "")+(":hist" if len(T)>len(pre) else "");st[sh]+=1;out.append(ex("umem",f"u{i}",T+[(u,a)],[q],"umem",lo=1,np=len(pre),sh=sh,pb=pb))
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
 t=e["t"];n0=e.get("np",0)
 while t:
  x,sp=render(t);b=x.encode("ascii");ids=np.array(B.enc(b),np.int64)
  if len(ids)<=CTX:break
  t,n0=([],0) if e["c"]=="mem" else (t[:-1],n0) if e["c"] not in XK else (t[:n0]+t[n0+1:],n0) if len(t)-1>n0 else (t[1:],n0-1) if n0 else ([],0)
 if not t:return None
 bm=np.zeros(len(b)+1,np.uint8);[bm.__setitem__(slice(a,z),1) for a,z in (sp[-1:] if e.get("lo") else sp)];L=B.L[ids];s0=np.cumsum(L)-L;m=bm[s0];assert (m==bm[s0+L-1]).all() and s0[-1]+L[-1]==len(b),("mask/token misalignment",e["s"],e["k"])
 return ids.astype(np.uint16),m,len(t)
def gb(m,ix)->list:return [g for j,g in enumerate(ix.groups) if np.any((m>>np.uint32(j))&np.uint32(1))]
def lines(fs)->set:return {nrm(u) for f in fs for s in DC.leaves(DC.load(f)) for u in [s]+re.findall(r"User: ([^\n]+)",s)+s.split("\n")+[x.split("\t")[-1] for x in s.split("\n")] if len(nrm(u).split())>=3}
def checks()->dict:
 ES=DC.eval_sources();ix=DC.index(13,cache=None);XF=sorted({f for f in ES if DC.group(f) in ("adam_t",) or "rknow" in str(f) or "nq" in f.name or f.name=="d3.jsonl"}|set((ROOT/"tests"/"data"/"emergency").glob("*.tsv")));ix8=DC.index(8,files=XF,cache=None);ix6=DC.index(6,files=[CS],cache=None);IX[:]=[ix,ix8,ix6]
 XS=[f for f in ES if isx(f)];XG=sorted({DC.group(f) for f in XS});assert len(XS)==len(EXEMPT) and not any(DC.group(f) in XG and not isx(f) for f in ES) and not any(isx(f) for f in XF) and CS.resolve() in XF,(XS,XG);XM[0]=np.uint32(sum(1<<ix.groups.index(g) for g in XG));EQ=lines([f for f in ES if not isx(f)]+XF)
 EQS.update(EQ);return {"ix":ix,"ix8":ix8,"ix6":ix6,"XF":XF,"XS":XS,"XG":XG,"EQ":EQ,"EQX":lines(XS)-EQ,"CQE":{nrm(json.loads(l)["q"]) for l in open(CS) if l.strip() and json.loads(l)["cat"]=="emergency"}}
def rescan(p)->dict:
 K=checks();ix,ix8,ix6=IX;B=Bpe(ROOT/"data"/"clean2"/"bpe4096.json");CQ=lines([CS]);SRC=list(LIC);o={"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"index_keys":{"n13":ix.key,"n8":ix8.key,"n6_chat_suite":ix6.key},"exempt_groups":K["XG"],"unit":"one decoded example (no grams across example boundaries)","splits":{}}
 for s in ("train","hold"):
  T=np.fromfile(OUT/f"{s}.bpe4096.u16",np.uint16);I=np.load(OUT/f"{s}.bpe4096.idx.npy");c=collections.Counter();x=collections.defaultdict(list)
  for off,n,si,_ in I.tolist():
   t=B.dec(T[off:off+n].tolist()).decode("ascii");ws=DC.words(t);us=[nrm(u) for u,_ in turns(t)];m13=DC._hits(ws,ix)[2];k=(m13&~XM[0])!=0
   f={**{"13:"+g:1 for g in gb(m13[k],ix)},**{"13x:"+g:1 for g in gb(m13[~k],ix)},**{"8:"+g:1 for g in gb(DC._hits(ws,ix8)[2],ix8)},"6:chat_suite":len(DC._hits(ws,ix6)[1])>0,"eval_line":any(u in K["EQ"] for u in us),"chat_suite_line":any(u in CQ for u in us),"emergency_copy":SRC[si]=="emergency" and any(q in u for q in K["CQE"] for u in us)}
   c.update({a for a,v in f.items() if v});c["examples"]+=1;[len(x[a])<3 and x[a].append({"src":SRC[si],"text":t[:300]}) for a,v in f.items() if v and not a.startswith("13x:")]
  o["splits"][s]={"counts":dict(sorted(c.items())),"examples":dict(x)};p(s,dict(sorted(c.items())))
 o["adam_t_and_chat_suite_hits"]=sum(v for q in o["splits"].values() for a,v in q["counts"].items() if a=="13:adam_t" or a.startswith("8:") or a in ("6:chat_suite","chat_suite_line","emergency_copy"));o["non_exempt_hits"]=sum(v for q in o["splits"].values() for a,v in q["counts"].items() if a.startswith(("13:","8:","6:")) or a in ("eval_line","chat_suite_line","emergency_copy"))
 (OUT/"logs").mkdir(exist_ok=True);json.dump(o,open(OUT/"logs"/"posthoc_scan.json","w"),indent=1);p("rescan","adam_t_and_chat_suite_hits",o["adam_t_and_chat_suite_hits"],"non_exempt_hits",o["non_exempt_hits"]);return o
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fetch-only",action="store_true");ap.add_argument("--no-fetch",action="store_true");ap.add_argument("--only",default="");ap.add_argument("--rescan",action="store_true");a=ap.parse_args();t0=time.time();p=lambda *x:print(time.strftime("%H:%M:%S"),*x,flush=True)
 if a.rescan:return rescan(p)
 mv=(OUT/"meta.json").exists() and json.load(open(OUT/"meta.json")).get("version",1)<VER;assert not (mv and VP.exists()),VP;mv and OUT.rename(VP);mv and p("moved v2",OUT,"->",VP)
 fl={} if a.no_fetch else fetch();p("fetched",{k:len(v) for k,v in fl.items()})
 if a.fetch_only:return
 st=collections.Counter();R=random.Random(SEED);only=set(filter(None,a.only.split(",")));B=Bpe(ROOT/"data"/"clean2"/"bpe4096.json")
 L=[("chat5",lambda:chat5(st)),("mem",lambda:mem(st,B)),("emergency",lambda:emergency(R,st)),("squad",lambda:squadlike(R,"squad","squad")),("adversarial_qa",lambda:squadlike(R,"adversarial_qa","adversarial_qa")),("hotpot",lambda:hotpot(R)),("drop",lambda:drop(R)),("tydiqa",lambda:tydi(R)),("mintaka",lambda:mintaka(R)),("mc",lambda:mc(R)),("gsm8k",lambda:gsm(R)),("dolly_ctx",lambda:dolly_ctx(R,st)),("calc",lambda:calc(random.Random(SEED+10),st,dis(st))),("calc_gsm8k",lambda:calc_gsm(st)),("know",lambda:know(st,X)),("umem",lambda:umem(st,d3()[1]))]
 K=checks();ix,ix8,ix6=K["ix"],K["ix8"],K["ix6"];p("decontam",repr(ix),repr(ix8),repr(ix6),"exempt",K["XG"]);X=[]
 for n,f in L:
  if only and n not in only:continue
  y=f();X+=y;p(n,len(y))
 seen=set();keep=[];dc=collections.Counter();dx=collections.Counter();dg=collections.Counter();ex8=[]
 def chk(e,t):
  x="".join(f"User: {u}\nAdam: {v}\n\n" for u,v in t);w=DC.words(x);q=[nrm(y) for y in e["q"][:len(t)]+[u for u,_ in t]];h=[hits(w,k) for k in range(3)]
  return ("contam13" if len(h[0][0]) else "contam8" if len(h[1][0]) else "chat6" if len(h[2][0]) else "eval_line" if any(y in K["EQ"] for y in q) else "nf" if e["s"] not in ("mem","emergency","umem") and NF.clean(x)[0]!=x.strip() else "identity" if C5.IDN.search(x) else ""),h,w,any(y in K["EQX"] for y in q)
 for e in X:
  o=e["t"];e["t"]=[(a1(u),a1(v)) for u,v in e["t"] if a1(u) and a1(v)];e["c"] in XK and e["t"]!=o and st.update([e["s"]+"_serve_text_changed_by_a1"]);txt="".join(f"User: {u}\nAdam: {v}\n\n" for u,v in e["t"])
  if not e["t"]:dc[e["s"]+":empty"]+=1;continue
  k=hashlib.blake2b(nrm(txt).encode(),digest_size=12).digest()
  if k in seen:dc[e["s"]+":dup"]+=1;continue
  seen.add(k);why,h,ws,xl=chk(e,e["t"]);dg.update([f"13:{g}" for g in gb(h[0][1],ix)]+[f"13x:{g}" for g in gb(h[0][2],ix)]+[f"8:{g}" for g in gb(h[1][1],ix8)]+["6:chat_suite"]*(len(h[2][0])>0));dx[e["s"]+":any_chat6"]+=len(h[2][0])>0;xw=not why and (len(h[0][2])>0 or xl);dx[e["s"]+":exempt_won"]+=xw;dx[e["s"]+":exempt_won_13"]+=xw and len(h[0][2])>0;dx[e["s"]+":exempt_won_line"]+=xw and xl
  if why:j={"contam13":(0,13),"contam8":(1,8),"chat6":(2,6)}.get(why);len(ex8)<60 and j and ex8.append({"src":e["s"],"why":why,"gram":" ".join(ws[int(h[j[0]][0][0]):][:j[1]])});pf=next((jj for jj in range(len(e["t"])-1,0,-1) if e["c"] not in ("mem",)+XK and why in ("contam13","contam8","chat6","eval_line") and not chk(e,e["t"][:jj])[0]),0);dc[e["s"]+(":salvaged_prefix_after_"+why if pf else ":"+why)]+=1;e["t"]=e["t"][:pf]
  if not why or e["t"]:keep.append(e)
 assert not [u for e in keep if e["s"]=="emergency" for t in e["t"] for u in t if any(c in nrm(u) for c in K["CQE"])],"emergency example copies a chat-suite emergency prompt"
 p("kept",len(keep),"dropped",sum(dc.values()));R2=random.Random(SEED+1);R2.shuffle(keep);sp={"train":[],"hold":[]}
 for e in keep:sp["hold" if int(hashlib.md5((e.get("sk") or f"{e['s']}|{e['k']}").encode()).hexdigest(),16)%1000<HOLD else "train"].append(e)
 SI={s:i for i,s in enumerate(LIC)};cnt=collections.defaultdict(collections.Counter);OUT.mkdir(parents=True,exist_ok=True);att=gzip.open(OUT/"attribution.jsonl.gz","wt");smp=[];NS=[s for s in LIC if LIC[s][2] in XK];nsm=collections.defaultdict(list);LN=collections.defaultdict(list);SH=collections.defaultdict(collections.Counter);PB=collections.defaultdict(list)
 for s,es in sp.items():
  I=[];M=[];ids=[];o=0
  for e in es:
   r=tok(B,e)
   if r is None:dc[e["s"]+":too_long"]+=1;continue
   t,m,nt=r;cnt[e["s"]][s+"_examples"]+=1;cnt[e["s"]][s+"_turns"]+=nt;cnt[e["s"]][s+"_tokens"]+=len(t);cnt[e["s"]][s+"_loss_tokens"]+=int(m.sum());cnt[e["s"]][s+"_bytes"]+=int(B.L[t.astype(np.int64)].sum())
   ids.append(t);M.append(m);I.append((o,len(t),SI[e["s"]],int(m.sum())));o+=len(t);att.write(json.dumps({"src":e["s"],"lic":LIC[e["s"]][0],"id":e["k"],"split":s})+"\n");s=="train" and len(smp)<50000 and smp.append((e,t,m));LN[e["s"]].append(len(t));"sh" in e and (SH[s].update([e["sh"]]),PB[e["s"]].append(e["pb"]),nt<len(e["t"]) and st.update([e["s"]+"_ctx_history_turns_trimmed_to_fit"]*(len(e["t"])-nt)));s=="train" and e["s"] in NS and len(nsm[e["s"]])<400 and nsm[e["s"]].append((e,t,m))
  T=np.concatenate(ids);Mk=np.concatenate(M);T.tofile(OUT/f"{s}.bpe4096.u16");Mk.tofile(OUT/f"{s}.bpe4096.mask.u8");np.save(OUT/f"{s}.bpe4096.idx.npy",np.array(I,dtype=[("off","<u8"),("n","<u4"),("src","u1"),("loss","<u4")]));p(s,len(I),"examples",len(T),"tokens",int(Mk.sum()),"loss tokens")
 att.close();R3=random.Random(SEED+2);ss=R3.sample(smp,min(50,len(smp)));ns=[x for s in NS for x in R3.sample(nsm[s],min(4,len(nsm[s])))]
 with open(OUT/"samples.txt","w") as f:
  for j,(e,t,m) in enumerate(ss+ns):
   seg=[];cur=None
   for i,mm in zip(t.tolist(),m.tolist()):
    (seg.append([mm,[i]]) if cur!=mm else seg[-1][1].append(i));cur=mm
   f.write((f"===== sample {j+1}/{len(ss)}" if j<len(ss) else f"===== new-shape sample {j-len(ss)+1}/{len(ns)} | shape={e.get('sh')} | prompt_bytes={e.get('pb')}")+f" | src={e['s']} ({LIC[e['s']][0]}) | id={e['k']} | tokens={len(t)} loss={int(m.sum())}\n"+"".join(("[[" if mm else "")+B.dec(ii).decode("ascii")+("]]" if mm else "") for mm,ii in seg)+"\n")
 tot={s:sum(v[s+"_tokens"] for v in cnt.values()) for s in sp};REV={s:HfApi().dataset_info(C5.SRC[s]["hf"]).sha for s in ("oasst1","oasst2","dolly","aya") if cnt[s]};bs=lambda s:{k:v for k,v in cnt[s].items()}
 vp=json.load(open(VP/"meta.json")) if (VP/"meta.json").exists() else {};nz=lambda c:{k:v for k,v in sorted(c.items()) if v};RK=ROOT/"data"/"ks"/"index_rknow"/"wiki_v2"/"meta.json";rk=json.load(open(RK)) if RK.exists() else {};WS=WF.store()
 qd=lambda a,bk=((0,128),(128,256),(256,512),(512,768),(768,1024)):(lambda A:{"n":len(A),"min":int(A.min()),"p50":int(np.percentile(A,50)),"p90":int(np.percentile(A,90)),"p99":int(np.percentile(A,99)),"max":int(A.max()),"hist":{f"{lo+1}-{hi}":int(((A>lo)&(A<=hi)).sum()) for lo,hi in bk}})(np.array(a)) if a else {}
 meta={"name":"ADAM-T SFT set v3","version":VER,"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_s":round(time.time()-t0,1),"cmd":E("SFT_CMD"," ".join([sys.executable]+sys.argv)),"seed":SEED,"env":{k:E(k) for k in ("MAXA","SEED","HOLD_PERMIL","CTX","MEM_CTX","MAXP","MEM_N","LONGGAP","HOTPOT_RC","HOTPOT_CB","RC_CAPD","DROP_CAPD","HEDGE_HOTPOT","HEDGE_MINTAKA","HEDGE_MINTAKA_GENERIC","CALC_N","WD_N","KC_N","KC_UNSUP","WD_UNSUP","WD_SYN","UMEM_N","SERVE_BUDGET","WORKERS","SFT_OUT","SFT_RAW") if E(k)},
 "template":{"format":"User: {user}\\nAdam: {reply}\\n\\n per exchange, examples concatenated with no extra separator; ASCII only; one line per turn","why":"byte-level role markers identical to the ADAM-T eval chat fmt ('User: {q}\\nAdam:'), the D3/D3-dev memory dialogues and chat5/chat6c; BPE-4096 has no spare ids (256 bytes + 3840 merges = 4096), so <|user|>-style specials would need a vocab resize (see proposal)","proposal":"optional later: extend vocab to 4099 with <|user|> <|adam|> <|end|> (embedding/head rows +3, one-off surgery at the SFT start); not done here","pretoken_boundaries":"'Adam:' always ends a pre-token and every reply starts with ' ', so the reply/prompt boundary is a token boundary (asserted for every example)","example_limit_tokens":CTX,"example_limit_why":"v3: raised from 512 to the model ctx 1024 (the trainer packs whole examples into CTX+1 windows, so an example of <=1024 tokens is never cut)","mem_generation_limit_tokens":MEMCTX,"mem_generation_why":"the v2 memory-dialogue generator keeps its 512-token retry limit so the mem source is identical to v2; long memory prompts come from the new umem source","multi_turn_truncation":"trailing exchanges dropped until the example fits in CTX tokens (new-shape examples: oldest history turn first, never the final exchange)"},
 "files":{"train":"train.bpe4096.u16 (uint16 token ids)","hold":"hold.bpe4096.u16","mask":"<split>.bpe4096.mask.u8: 1 per token, 1 = token belongs to an Adam reply (incl. the closing \\n\\n); put loss only where mask==1","index":"<split>.bpe4096.idx.npy: structured (off u8 token offset, n u4 tokens, src u1 = index into sources order, loss u4 masked tokens); every example starts with 'User:'","attribution":"attribution.jsonl.gz: one row per example (src, lic, id, split) in file order","samples":"samples.txt: 50 random train examples decoded from tokens, reply tokens wrapped in [[ ]]"},
 "tokenizer":{"bpe":"data/clean2/bpe4096.json","sha256":sha(ROOT/"data"/"clean2"/"bpe4096.json"),"encoder":"Python replica of crates/adam-t/src/bpe.rs on ASCII input (pattern restricted to ASCII classes, same merge order)","pattern":B.pat},
 "tokens":tot,"sources":{s:{"license":LIC[s][0],"attribution":LIC[s][1],"kind":LIC[s][2],"use":LIC[s][3],"hf":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (C5.SRC.get(s,{}).get("hf"),))[0],"revision":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (None,REV.get(s)))[1],**bs(s)} for s in LIC},
 "share_train_tokens":{s:round(cnt[s]["train_tokens"]/max(1,tot["train"]),4) for s in LIC},"share_train_loss_tokens":{s:round(cnt[s]["train_loss_tokens"]/max(1,sum(v["train_loss_tokens"] for v in cnt.values())),4) for s in LIC},
 "raw_files":{n:[{"file":os.path.relpath(f,RAW),"sha256":sha(f)} for f in fs] for n,fs in fl.items()},"mintaka":{"url":MK%"{split}","commit":"b6146ffb41562803c45df3c028b4cdad8c09dd4d"},
 "decontam":{"module":"amni.compute.decontam","n13":{"index_key":ix.key,"files":len(ix.files),"grams":int(len(ix.H)),"groups":ix.groups},"n8_adam_t":{"index_key":ix8.key,"files":ix8.files,"grams":int(len(ix8.H)),"groups":ix8.groups},"n6_chat_suite":{"index_key":ix6.key,"files":ix6.files,"grams":int(len(ix6.H)),"why":"docs/adam_t/eval_prereg.md chat section: most chat-suite prompts are shorter than the 13-gram window, so SFT builders also run decontam.index(n=6, files=[chat_src.jsonl]) with dirty()","dropped":{s:dc[s+":chat6"] for s in LIC if dc[s+":chat6"]},"salvaged_prefix":{s:dc[s+":salvaged_prefix_after_chat6"] for s in LIC if dc[s+":salvaged_prefix_after_chat6"]},"examples_with_any_hit":{k.split(":")[0]:v for k,v in dx.items() if k.endswith(":any_chat6") and v}},"exempt":{"files":list(EXEMPT),"groups":K["XG"],"reason":EXWHY,"scope":"this builder only: 13-gram hits whose only matching groups are exempt, exact user-line/question matches found only in exempt files, and memory-dialogue exact-prompt matches against the exempt hold_eval dialogues are ignored; distractor pairs for memory dialogues are filtered the same way","exact_lines_exempt_only":len(K["EQX"]),"won_back_examples":{k:v for k,v in nz(dx).items() if ":exempt_won" in k},"hits_13_exempt_only_by_group":{k[4:]:v for k,v in sorted(dg.items()) if k.startswith("13x:")},"distractor_pairs_kept":st["distractor_pairs_exempt13_kept"],"mem_kept_with_exempt_13_hit":st["mem_exempt13_kept"],"mem_kept_matching_exempt_hold_prompt":st["mem_exempt_eval_prompt_kept"]},"exact_lines":len(K["EQ"]),"mem_exact_line_scope":"v2: memory dialogue user lines are checked against the same non-exempt exact-line set as every other source (v1 used only the ADAM-T/NQ/D3/emergency subset because of the chat hold sets, now exempt)","rule":"drop an example on any 13-gram hit against a non-exempt group of decontam.eval_sources(), any 8-gram hit against the ADAM-T/NQ/D3/emergency-test files (incl. the chat suite), any 6-gram hit against the chat suite, or any user line/question equal (normalised) to a non-exempt eval string of >=3 words; multi-turn non-memory examples keep their longest clean prefix","emergency_guard":"4 emergency prompt templates that shared 6-grams with chat-suite emergency prompts were reworded (face drooping, whole bottle of pills, heart attack, left arm numb); the 6-gram check still applies to every emergency example, and the build asserts that no emergency example turn contains a chat-suite emergency prompt (normalised substring)","hits_by_group":dict(sorted(dg.items())),"examples":ex8},
 "drops":nz(dc),"stats":nz(st),"previous":{"version":2,"moved_to":"data/adam_t/sft_v2","moved_this_run":bool(mv),"v2_built":vp.get("built"),"v2_builder_commit":"615d8ea","note":"v2 is kept only until the CTO reviews v3, then deleted by the CTO"},"v2_vs_v3":{"tokens":{"v2":vp.get("tokens"),"v3":tot},"train_by_source":{s:{f"{k}_{x}":(vp.get("sources",{}).get(s,{}).get("train_"+x,0) if k=="v2" else cnt[s]["train_"+x]) for x in ("examples","tokens","loss_tokens") for k in ("v2","v3")} for s in LIC}},
 "serve_context":{"client":"amni/serve/adam_t_client.py at c9583d0 (Merge ADAM-T-CTX)","budget_bytes":BUD,"budget_why":"adam-t serve reports ctx = int(1024 symbols x 3.25) = 3328 bytes for BPE-4096; answer() reserves min(AMNI_ADAM_T_MAX_BYTES 400, 4 x max_new_tokens) bytes for the reply, so template() gets 3328 - 400 = 2928 bytes","order":"memory pre-turns ('User: <fact>\\nAdam: Got it, thanks for telling me.'), then history turns (newest kept first), then 'User: <rc items joined by one space> <question>\\nAdam:'; rc is ['Calculator: <engine reply>'] or ['<Wikidata label>: <answer>.', '<title>: <passage>', '<title>: <passage>']","budgets_bytes":{"memory":640,"math":240,"knowledge":1100,"passages":2,"passage":"min(560, knowledge // 2) = 550"},"exclusivity":"context(): the calculator line is offered only when memory gave no direct reply; knowledge is fetched only when there is no calculator line and the memory route did not fire; memory pre-turns only when it did. Memory pre-turns therefore never share a prompt with passages or a Wikidata line: v3 puts passages with history turns and memory pre-turns with history, exactly as the client assembles them","assembly":"the builder calls adam_t_client._math, clip, _know (2-process pool, cached in RAW/kx_cache.jsonl), _fit, _mem and template itself and asserts every prompt parses back into its turns, so new-shape prompts are byte-identical to the flag-gated serve route","loss":"new-shape examples (calc, calc_gsm8k, wikidata, kctx, umem) put loss only on the final Adam reply; history, memory ACK turns and context are prompt-only","truncation":"a new-shape example over CTX tokens drops its oldest history turn first, then its oldest memory turn; the current turn and its context are never cut","env":{"AMNI_WIKIFACT_THR":WF.THR,"AMNI_WIKIFACT_TEMPORAL":WF.TEMPORAL,**{k:E(k) for k in ("AMNI_MATH","AMNI_MATH_MIN_CONF","AMNI_RKNOW_WF","AMNI_ADAM_T_KNOW_COV","AMNI_UMEM_LT","AMNI_UMEM_GN","AMNI_UMEM_G2","AMNI_RAY_HIST","AMNI_UMEM_CAP") if E(k)}},"a1_changed":{k:v for k,v in st.items() if k.endswith("_serve_text_changed_by_a1")}},
 "new_shapes":{"train":dict(sorted(SH["train"].items())),"hold":dict(sorted(SH["hold"].items())),"summary":{sp_:dict(sorted(sum((collections.Counter({":".join(k.split(":")[:2]):v}) for k,v in SH[sp_].items()),collections.Counter()).items())) for sp_ in ("train","hold")}},
 "length_tokens":{"all":qd([x for v in LN.values() for x in v]),"by_source":{s_:qd(LN[s_]) for s_ in LIC if LN[s_]},"note":"train+hold examples after tokenization"},"prompt_bytes":{s_:qd(PB[s_],((0,512),(512,1024),(1024,2048),(2048,BUD))) for s_ in LIC if PB[s_]},
 "calc_engine":{"mismatch_examples":CMM,"stats":{k:v for k,v in sorted(st.items()) if k.startswith(("calc_","gsm8k_calc"))}},
 "local_sources":{"wikifact":{"dir":"data/wikifact/v2/store","license":WS.man.get("license") if WS else None,"dump":WS.man.get("source") if WS else None,"dump_md5":WS.man.get("dump_md5_stream") if WS else None,"built":WS.man.get("built") if WS else None},"rknow":{"dir":"data/ks/index_rknow/wiki_v2","version":rk.get("version"),"source":rk.get("source"),"n_pass":rk.get("n_pass"),"n_doc":rk.get("n_doc")}},"rejected":[dict(zip(("dataset","card_license","reason","decision"),x)) for x in REJ],"licenses":sorted({v[0] for v in LIC.values()})}
 json.dump(meta,open(OUT/"meta.json","w"),indent=1)
 with open(OUT/"attribution.md","w") as f:
  f.write("# ADAM-T SFT set v3: attribution\n\nThis set mixes sources under the licences below. Rows from CC BY-SA sources (SQuAD, AdversarialQA, HotpotQA, DROP, Dolly, and the v3 `wikidata`/`kctx` rows, whose passages are English Wikipedia text retrieved by the R-KNOW wiki_v2 index) are share-alike: the set, and any redistribution of it, is under CC BY-SA 4.0 terms for those rows. `attribution.jsonl.gz` maps every example (file order) to its source and id.\n\n| source | licence | examples (train/hold) | tokens (train) | credit |\n|---|---|---|---|---|\n"+"".join(f"| {s} | {LIC[s][0]} | {cnt[s]['train_examples']}/{cnt[s]['hold_examples']} | {cnt[s]['train_tokens']} | {LIC[s][1]} |\n" for s in LIC)+"\nModel outputs of third-party LLMs are not used. Own work (`mem`, `emergency`, `calc`, `umem`) is CC0; Wikidata facts and lines (`wikidata`) are CC0 1.0 (wikidata-20260921 dump). Excluded and flagged datasets are listed in `meta.json` under `rejected`.\n")
 p("done",json.dumps({"tokens":tot,"drops":sum(dc.values())}))
if __name__=="__main__":main()
