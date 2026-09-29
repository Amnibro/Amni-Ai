import os,re,sys,json,gzip,math,time,random,hashlib,argparse,subprocess,importlib.util,collections,unicodedata,urllib.request,urllib.parse,datetime as dt,multiprocessing as mp
from pathlib import Path
from fractions import Fraction as Fr
import numpy as np
ROOT=Path(os.environ.get("AMNI_HOME") or Path(__file__).resolve().parents[1]);sys.path.insert(0,str(ROOT));OUT=Path(os.environ.get("SFT_OUT") or ROOT/"data"/"adam_t"/"sft");RAW=Path(os.environ.get("SFT_RAW") or OUT/"raw")
[os.environ.setdefault(k,v) for k,v in (("HF_HUB_DISABLE_XET","1"),("HF_HOME",str(RAW/"hf_home")),("GEN_CHAT","chat4c"),("MAXA","900"),("TOKENIZERS_PARALLELISM","false"),("AMNI_UMEM_DIR",str(RAW/"umem")),("AMNI_UMEM_LT","1"),("AMNI_UMEM_GN","1"),("AMNI_UMEM_G2","1"))]
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download,HfApi
from amni.compute import decontam as DC,wikifact as WF,rknow as RKN
from amni.serve import emergency as EM,adam_t_client as AC,user_memory as UM
from amni.math.answer import solve_math as msolve,fmt as mfmt
E=os.environ.get;SEED=int(E("SEED","20260926"));HOLD=int(E("HOLD_PERMIL","20"));CTX=int(E("CTX","1024"));MEMCTX=int(E("MEM_CTX","512"));MAXP=int(E("MAXP","1200"));N=lambda k,d:int(E(k,str(d)))
def imp(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C5=imp("build_chat5_corpus",ROOT/"scripts"/"build_chat5_corpus.py");G=C5.G;NF=C5.NF;TK=imp("adam_t_talk",Path(__file__).resolve().parent/"adam_t_talk.py");TKI={};TM=imp("adam_t_tmem",Path(__file__).resolve().parent/"adam_t_tmem.py")
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
"wikidata":("cc-by-sa-4.0","Wikidata (CC0 1.0, wikidata-20260921 dump, "+os.path.relpath(WF.DIR,ROOT)+") for subjects, answers and 'Label: answer.' lines; passages from English Wikipedia (CC BY-SA 4.0) retrieved by the serve's own R-KNOW wiki_v2 index (data/ks/index_rknow/wiki_v2); questions/replies own templates; history turns chat5 pairs","kctx","serve knowledge context '<Wikidata line> <title>: <passage> <title>: <passage> <question>' built by adam_t_client._know/_fit; relevant lines, natural and synthetic off-target lines (other subject, other property), passage-only prompts; off-target/unsupported answers hedged"),
"kctx":("cc-by-sa-4.0","HotpotQA, Yang et al. 2018, CC BY-SA 4.0 (distractor-train questions and answers not used by hotpot_rc/hotpot_cb); passages from English Wikipedia (CC BY-SA 4.0) retrieved by the serve's R-KNOW wiki_v2 index via adam_t_client._know; history turns chat5 pairs","kctx","long serve prompts: history turns + 1-2 retrieved passages (often one off-target) + optional Wikidata line + question, assembled by adam_t_client.template within the 2928-byte serve budget; unsupported answers hedged"),
"umem":("cc0-own","Own work: facts from scripts/v19_train_dialogues.py value pools (D3/D3-dev values excluded), stored and rendered by amni/serve/user_memory (production parser, temporary store) and adam_t_client._mem; history turns chat5 pairs","umem","serve memory context: remembered facts as earlier 'User: <fact>\\nAdam: Got it, thanks for telling me.' turns (memory budget 640 bytes) + history + question; known, corrected, told-this-session and never-told facts"),
"emerg4":("cc0-own","Own work (ADAM-T-SFT4): emergency prompts and replies written fresh for this set (not from any eval set, not copied from the Law-0 table); history turns chat5 pairs","emerg","varied first- and third-person emergency prompts incl. self-harm; every reply leads with 'Call 911 (or your local emergency number)', short, with calm supportive text after the 911 line for self-harm; each example is pre-checked (13/8/6-gram, exact line, chat-suite emergency substring) before it is kept"),
"talk":("cc0-own","Own work (ADAM-T-SMALLTALK), CC0 1.0: everyday small-talk scenarios written for this set in scripts/adam_t_talk/*.txt (openers, reactions, follow-up questions, answer branches, sign-offs, slot fillers) and combined by scripts/adam_t_talk.py; no third-party datasets and no third-party LLM outputs","chat","1-3 exchange everyday small talk (events, feelings, pets, food, hobbies, weather, family, work, school, sports, media, plans, small wins, annoyances, idiom near-misses, greetings); Adam replies are 1-2 sentences, react to the specifics and often ask one follow-up; every reply template (reaction, question, response, sign-off and each reaction+question pair) is capped at TALK_CAP uses; its own hold split holds out whole scenarios; each dialogue is pre-checked (13/8/6-gram, exact eval line) before it is kept"),
"reader":("cc-by-sa-4.0","Questions and answers from SQuAD v1.1 (Rajpurkar et al. 2016, CC BY-SA 4.0, train), AdversarialQA (Bartolo et al. 2020, CC BY-SA 4.0, train), HotpotQA (Yang et al. 2018, CC BY-SA 4.0, distractor-train rows not used by hotpot_rc/hotpot_cb/kctx, yes/no answers excluded) and TyDi QA GoldP English (Clark et al. 2020, Apache-2.0, train), plus Wikidata subject/answer pairs (CC0 1.0, wikidata-20260921 dump, "+os.path.relpath(WF.DIR,ROOT)+"; subjects not used by the wikidata source) asked with own templates; every passage is English Wikipedia text (CC BY-SA 4.0) from the live R-KNOW index data/ks/index_rknow/wiki_v2 (FineWiki re-source), retrieved or re-grounded through adam_t_client; hedge and reply templates own work (CC0)","kctx","reading: serve knowledge prompt '<optional Wikidata line> <1-3 live-index passages> <question>' assembled by adam_t_client._know/_fit/template; answer examples hold one gold passage (answer span verified in it) among 0-2 distractor passages (answer absent; same-query or related-query retrievals) in random order and reply with the short answer taken from the passage; ~RD_HEDGE of examples carry only distractors and reply with a brief hedge plus a best guess"),
"tmem":("cc0-own","Own work (ADAM-T-SFT7), CC0 1.0: slot frames, statement/question/answer/abstain/hedge templates and value pools written for this set in scripts/adam_t_tmem.py, plus value pools of scripts/v19_train_dialogues.py (D3/D3-dev values excluded); memory lines rendered by amni/serve/user_toroid (real observe/recall on the statements, line() for the gold slot); history turns are chat5 first-turn pairs (OASST1/2 Apache-2.0, Dolly CC BY-SA 3.0, Aya Apache-2.0)","umem","toroid memory-line use in the exact serve shape 'User: [history] Memory: <label>, <attr>: <value>. <question>\\nAdam:' (adam_t_client.template rc slot, AMNI_UMEM_TOROID=1): value lines answered from the line, corrections (latest value wins, the line beats the history), unknown lines and no memory answered honestly, unsure top-2 lines hedged, two-line entity/slot choice; questions heavily paraphrased; loss on the final reply only"),
"cmem":("cc0-own","Own work (ADAM-T-SFT7), CC0 1.0: same frames and templates as tmem (scripts/adam_t_tmem.py); history turns are chat5 first-turn pairs (OASST1/2 Apache-2.0, Dolly CC BY-SA 3.0, Aya Apache-2.0)","umem","contrastive memory twins: every question is asked once with the fact told earlier in the chat history (answer from it, latest value after a correction) and once with the same history minus that turn (honest 'you haven't told me'), both with the same near-miss facts (other entity, same slot / same entity, other slot) and different paraphrases; loss on the final reply only")}
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
VER=7;VP=OUT.parent/f"sft_v{VER-1}";CS=ROOT/"data"/"adam_t"/"eval"/"chat"/"chat_src.jsonl";TDV=ROOT/"data"/"adam_t"/"eval"/"toroid"/"toroid_dev.jsonl"
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
ABR=re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|St|Jr|Sr|No|approx|Inc|Ltd|Co|U\.S|a\.m|p\.m|Fig|al|ca|cf|Prof|Gen|Mt|Ft|Vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.|\b[A-Z]\.|(?:^|\s)\d{1,2}\.)$")
def sents(p:str,ab:bool=False)->list:
 s=[x for x in re.split(r"(?<=[.!?])\s+",p) if x]
 if not ab:return s
 o=[""]
 for x in s:o[-1]=(o[-1]+" "+x).strip();ABR.search(o[-1]) or o.append("")
 return [x for x in o if x]
LST=re.compile(r"(?:^|[\s:])(?:\d{1,2}[.)]|[a-h]\))\s+\S");DSH=re.compile(r"(?:^|\s)-\s?[A-Za-z]");CAV=re.compile(r"(?i)^(?:however|but|although|though|that said|note|keep in mind|unless|except|also note|be careful|warning|caution|important)\b");FIL=re.compile(r"(?i)^(?:sure|certainly|of course|absolutely|great question|good question|hello|hi there|hi|hey|ok|okay)[.!,]*$")
PROM=re.compile(r"(?i)(?::\s*$|\b(?:the following|as follows|below|here are|here is|here's|these are|these steps|some (?:tips|ways|steps|options|ideas|examples|suggestions|things)|a few (?:tips|ways|steps|options|ideas|things|suggestions)|several (?:tips|ways|steps|options|ideas|things|suggestions|factors|reasons))\b)")
BLIM,BKEEP,BMAX,BMIN=N("BRIEF_LIM",60),N("BRIEF_KEEP",15),N("BRIEF_MAX",160),N("BRIEF_MIN",30)
def brief(a:str)->tuple:
 w=len(a.split());ls=bool(LST.search(a)) or len(DSH.findall(a))>=2
 if w<=BLIM:return a,"short"
 if not ls and ":" not in a and not PROM.search(a) and w<=BMAX and int(hashlib.md5(a.encode()).hexdigest(),16)%100<BKEEP:return a,"long_kept"
 s=sents(a,True);s=s[1:] if len(s)>1 and FIL.match(s[0]) else s;o=[];n=lambda z:len(" ".join(z).split())
 for x in s[:4]:
  if n(o+[x])>BLIM or len(o)>=3 and not CAV.match(x) or o and n(o)>=BMIN and not CAV.match(x):break
  o.append(x)
 if not o:return "","drop_first_sentence_long"
 t=" ".join(o);nx=s[len(o)] if len(o)<len(s) else "";why="drop_unended" if not re.search(r"[.!?\"')]$",t) else "drop_promise" if PROM.search(o[-1]) or PROM.search(t[-2:]) else "drop_intro_only" if PROM.search(nx) else "drop_list_cut" if LST.search(t) or len(DSH.findall(t))>=2 else "drop_caveat" if CAV.match(nx) else "drop_tiny" if len(t.split())<6 else ""
 return ("",why) if why else (t,"trimmed")
def briefs(e,st)->list:
 o=[]
 for u,a in e["t"]:
  b,why=brief(a);st[f"brief_{e['s']}_{why}"]+=1
  if not b:break
  o.append((u,b))
  if b!=a:break
 st[f"brief_{e['s']}_threads_cut"]+=0<len(o)<len(e["t"]);st[f"brief_{e['s']}_examples_dropped"]+=not o;return o
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
 ds=C5.oasst(st)+C5.dolly(st)+C5.aya(st);cut=lambda t:t[:next((j for j,(_,a) in enumerate(t) if DEG.search(a)),len(t))];out=[ex(s,i,cut(turns(t))) for s,i,t,_,_ in ds];st["chat5_degenerate_list_cut"]=sum(len(e["t"])<len(turns(t)) for e,(_,_,t,_,_) in zip(out,ds))
 return [e for e in out if not e["t"] or e.__setitem__("t",briefs(e,st)) or e["t"]]
IX=[];DIS=[];XM=[np.uint32(0)]
def hits(ws,k=0)->tuple:
 _,p,m=DC._hits(ws,IX[k]);x=(m&~XM[0])!=0 if k==0 else np.ones(len(p),bool);return p[x],m[x],m[~x]
def dirt(t:str)->str:ws=DC.words(t);return "" if not IX else "contam13" if len(hits(ws)[0]) else "contam8" if len(hits(ws,1)[0]) else "chat6" if len(hits(ws,2)[0]) else "tor6" if len(IX)>3 and len(hits(ws,3)[0]) else ""
def dis(st)->list:
 if not DIS:D=G.load_dis(ROOT/"data"/"chat5"/"pairs.bin");w=[dirt(d[0]+" "+d[1]) for d in D];DIS.extend(d for d,y in zip(D,w) if not y);st["distractor_pairs"]=len(DIS);st.update("distractor_pairs_dropped_"+y for y in w if y);st["distractor_pairs_exempt13_kept"]=sum(not y and len(hits(DC.words(d[0]+" "+d[1]))[2])>0 for d,y in zip(D,w))
 return DIS
def d3()->tuple:dd=[json.loads(l) for f in (ROOT/"data"/"adam_t"/"eval"/"d3dev.jsonl",ROOT/"data"/"concepts"/"eval"/"d3"/"d3.jsonl") for l in open(f) if l.strip()];return dd,{str(v).lower() for d in dd for v in d.get("values",[])+[d.get("answer","")]}-{"unknown",""}
P4=lambda x:[y.strip() for y in x.split("|") if y.strip()]
AK={"city":("Oh nice|Oh cool|Nice|Got it|Ah, okay|Okay|Oh, neat|Ah, nice","you live in {V}|you're in {V}|so you're based in {V}|{V} is home for you|you live over in {V}","{y} lives in {V}|{y} is in {V}|so {y} is based in {V}|{s}'s in {V}|{y} lives over in {V}","Do you like living there?|Have you been there long?|Is it a nice place to live?|Did you grow up there?|Do you like it there?","Does {s} like it there?|Have you visited {o} there?|Has {s} been there long?|Is that far from you?|Do you get to see {o} much?"),
"job":("Oh nice|Oh cool|Nice|Got it|Ah, okay|Oh, interesting|Okay","you work as {a} {V}|you're {a} {V}|so you're {a} {V}|you work as {a} {V} these days","{y} works as {a} {V}|{y} is {a} {V}|so {y} is {a} {V}|{s} works as {a} {V}","Do you enjoy it?|Have you been doing that long?|Is it keeping you busy?|Is that what you always wanted to do?|Do you like the work?","Does {s} enjoy it?|Has {s} been doing that long?|Is {s} busy with it these days?|Does {s} like the work?"),
"name":("Got it|Nice|Oh, nice|Okay|Lovely|Ah, okay","your name is {V}|you go by {V}|I'll call you {V}|nice to meet you, {V}","{y} is called {V}|{y}'s name is {V}|{y} goes by {V}","Is there a story behind your name?|Do people ever shorten it?|Is that a family name?","Is there a story behind the name?|Does {s} like that name?|Is that a family name?"),
"like":("Oh fun|Oh nice|Nice|Cool|Oh cool|Ah, nice|Got it","you love {V}|you're into {V}|{V} is your thing|you really enjoy {V}","{y} loves {V}|{y} is into {V}|{s} really enjoys {V}|{y} is really into {V}","Do you get to do much of it lately?|Have you been into it long?|Do you do it with friends?|Is that how you like to unwind?","Does {s} do it often?|Have you ever tried it with {o}?|Has {s} been into it long?|Does {s} get much time for it?"),
"allergy":("Oh no|Oof|Ah, okay|Got it|Good to know|Okay, noted|Thanks for the heads up","you're allergic to {V}|you have an allergy to {V}|so no {V} for you|you need to avoid {V}","{y} is allergic to {V}|{y} has an allergy to {V}|so no {V} for {o}|{y} needs to avoid {V}","Is it a bad allergy?|Do you carry medicine for it?|Is it hard to avoid?|Have you had it since you were a kid?","Is it a serious allergy?|Does {s} carry medicine for it?|Is it hard for {o} to avoid?|Has {s} had it long?"),
"drive":("Oh nice|Nice|Oh cool|Got it|Ah, okay|Okay","you drive {a} {V}|you've got {a} {V}|so you're driving {a} {V}|your car is {a} {V}","{y} drives {a} {V}|{y} has {a} {V}|so {y} is driving {a} {V}|{y}'s car is {a} {V}","Do you like it?|Have you had it long?|Is it good on long drives?|Is it reliable?","Does {s} like it?|Has {s} had it long?|Is it reliable?|Does {s} drive it much?"),
"color":("Nice|Oh nice|Good choice|Got it|Oh, lovely|Okay","your favorite color is {V}|{V} is your favorite color|you love {V}","{y}'s favorite color is {V}|{V} is {y}'s favorite color|{y} loves {V}","Do you wear it a lot?|Has it always been your favorite?|Do you have a lot of things in that color?","Does {s} wear it a lot?|Has it always been {p} favorite?|Does {s} have a lot of things in that color?"),
"food":("Mmm|Oh nice|Yum|Nice|Oh, tasty|Got it|Good choice","your favorite food is {V}|you love {V}|{V} is your favorite","{y}'s favorite food is {V}|{y} loves {V}|{V} is {y}'s favorite","Do you make it yourself?|Do you have a favorite place for it?|Do you have it often?|Is that your go-to comfort food?","Does {s} make it at home?|Do you ever cook it for {o}?|Does {s} have it often?"),
"study":("Oh nice|Oh, interesting|Nice|Got it|Cool|Okay","you studied {V}|your background is in {V}|you majored in {V}","{y} studied {V}|{y}'s background is in {V}|{y} majored in {V}","Did you enjoy it?|Do you still use it much?|Was it a tough program?|Do you work in that field now?","Did {s} enjoy it?|Does {s} work in that field now?|Was it a tough program?"),
"school":("Oh nice|Nice|Got it|Oh cool|Okay","you went to {V}|you studied at {V}|you went to school at {V}","{y} went to {V}|{y} studied at {V}|{y} went to school at {V}","Did you like it there?|Do you keep in touch with people from there?|Was it far from home?","Did {s} like it there?|Does {s} keep in touch with people from there?|Was it far from home?"),
"age":("Got it|Okay|Noted|Ah, okay|Alright","you're {V}|you're {V} years old|you're {V} now","{y} is {V}|{y} is {V} years old|{y} is {V} now","Any big plans for the year?|Did you do anything fun for your last birthday?","Did you celebrate with {o} on {p} last birthday?|Any plans for {p} next birthday?"),
"birthday":("Got it|Oh nice|Okay|Noted|Oh, fun","your birthday is in {V}|you celebrate in {V}|you're {a} {V} baby","{y}'s birthday is in {V}|{y} celebrates in {V}|{y} has {a} {V} birthday","Do you usually celebrate?|Do you have plans for it this year?|Do you like having {a} {V} birthday?","Do you usually do something for {o}?|Do you have anything planned for it?|Does {s} like to celebrate?")}
AK={k:tuple(P4(x) for x in v) for k,v in AK.items()}
RM4=P4("I'll remember that.|Noted.|I'll keep that in mind.|Thanks for telling me.|Got it.|Thanks, I've got it.|I'll remember.|Thanks for letting me know.")
FM4=[("{Rx}, {e}.",26),("{Rx}, {e}. {q}",22),("{E}. {q}",16),("{E}. {rm}",14),("{Rx}! {q}",8),("{E}, got it.",5),("{Rx}, {e}. {rm}",5)]
GK4=P4("Got it, thanks for telling me.|Good to know!|Okay, I'll remember that.|Thanks, noted.|Got it.|Okay, noted.|Thanks for letting me know.|I'll keep that in mind.|Good to know, thanks.|Got it, I'll remember.|Okay, got it.|Thanks for sharing that.|Noted, thanks!|Makes sense, I'll remember that.|Got it. Anything else I should know?|Okay! I'll keep that in mind.|Thanks, that's good to know.|Sure, I'll remember that.|Alright, got it.|Understood, thanks.|Okay, good to know.|Thanks for filling me in.|I've got it, thanks.|Good to know. Thanks for sharing.")
MK4=P4("Got it, thanks for filling me in.|Okay, I'll keep those straight.|Good to know, thanks!|Got it, I'll remember both.|Okay, noted both of those.|Thanks, I've got them both.|Okay, I'll keep track of both.|Nice, got both of those.|Alright, I'll remember who's who.|Thanks, that helps me keep them straight.|Got it, noted for both of them.|Okay, both noted.")
RK4=P4("Got it, {C} is in {R}.|Ah, {R}! Thanks for the context.|Good to know, {C}'s in {R}.|Thanks, I'll keep that in mind.|Oh, {R}. I'll remember that.|Okay, so {C} is in {R}.|Noted, {C} is in {R}.|Thanks, that helps me place {C}.|Got it, {C} is in {R}. Thanks!|Ah, okay, {R}. Good to know.")
CK4=P4("Ah, {V} then. Thanks for the update!|Got it, {V} instead of {O}.|Okay, updated to {V}.|No problem, I'll go with {V} from now on.|Thanks for the fix! So it's {V}, not {O}.|Good catch. I'll remember {V}, not {O}.|Ah, got it. So it's {V} now.|No problem, {V} it is.|Thanks for clarifying, so it's {V}.|Okay, I've changed it to {V}.|Got it, it's {V} now, not {O}.|Ah, {V}. I'll remember that instead.|Okay, updated. {V}, not {O}.")
UFP={"I think so, yeah.","Ha, pretty much!","Mostly, yeah.","Yeah, for sure."};AFP4=P4("Nice, glad to hear it.|That's great.|Makes sense.|Love that.|Good to hear!|Nice!|Oh, that's nice.|Glad to hear it.|Oh good.|Ha, nice.|That sounds good.|Oh, fun.|Nice, that's good.|Sounds about right.");AFN4=P4("Fair enough!|That's okay.|No worries.|Ha, fair enough.|Makes sense.|All good.|Totally fair.|Ha, no worries.|That's fine.|Understandable.")
BC=collections.Counter();AT=collections.Counter();ACAP=N("ACK_CAP",600);AKP4=float(E("AKP4",".85"))
def bal(R,pool,tag)->str:m=min(BC[tag+x] for x in pool);x=R.choice([y for y in pool if BC[tag+y]<=m+2 and (BC[tag+y]<ACAP or m>=ACAP)]);BC[tag+x]+=1;return x
def ack4(R,rel,k,c)->str:
 A=AK.get(rel)
 if not A:x=bal(R,GK4,"g:");AT["g:"+x]+=1;return x
 for _ in range(30):
  f=R.choices([a for a,_ in FM4],[w for _,w in FM4])[0];rx,e,q,rm=bal(R,A[0],"x:"),bal(R,A[1 if k=="1" else 2],rel+"e:"),bal(R,A[3 if k=="1" else 4],rel+"q:"),bal(R,RM4,"m:");key=f.format(Rx=rx,e=e,E="^"+e,q=q,rm=rm)
  if AT[key]<ACAP:break
 AT[key]+=1;ef=G.fm(e,c);return f.format(Rx=rx,e=ef,E=cap(ef),q=G.fm(q,c),rm=rm)
def fu(R,a,rest)->list:
 if not a.endswith("?") or not (rest or R.random()<.3):return []
 uf=rest[0][0] if rest else R.choice(G.UF);return [(uf,bal(R,AFP4 if uf in UFP else AFN4,"f:"),[])]
RA=[random.Random(SEED+41),random.Random(SEED+42)]
def g4():
 if getattr(G,"_v4",0):return
 U,RU,O=G.unit,G.runit,G.one;G.GACK=[f"\x01G{i}" for i in range(len(G.GACK))];G.MACK=[f"\x01M{i}" for i in range(len(G.MACK))];R=RA[0]
 def unit(R0,S,h,rel,v=None,pron=False,old=None):
  o=U(R0,S,h,rel,v,pron,old);u,a,f=o[0];c=G.ctx(h,S["g"][h],f[0][2],False,rel,O=old or "")
  a=G.fm(bal(R,CK4,"c:"),c) if old else ack4(R,rel,G.kk(h),c) if R.random()<AKP4 else ack4(R,"",0,c);return [(u,a,f)]+fu(R,a,o[1:])
 def runit(R0,C):o=RU(R0,C);return [(o[0][0],G.fm(bal(R,RK4,"r:"),{"C":C,"R":G.CITY[C][0]}),o[0][2])]
 def one(R0,kind,D):r=O(R0,kind,D);p=re.sub("\x01([GM])\\d+",lambda m:bal(R,GK4 if m.group(1)=="G" else MK4,"g:" if m.group(1)=="G" else "M:"),r["prompt"]);return {**r,"prompt":p,"text":p+" "+r["reply"]+"\n\n"}
 G.unit,G.runit,G.one,G._v4=unit,runit,one,1
def uack(k,rel,v)->str:rn={"names":"name"}.get(UQ[rel][0],UQ[rel][0]);h=k or "I";return ack4(RA[1],rn,"1" if k is None else "3",G.ctx(h,"f" if h in G.FEM else "m" if h in G.MAS else RA[1].choice("fm"),v,False,rn))
def mem(st,B)->list:
 g4();R=random.Random(SEED+3);D=dis(st);ov=G.overlap();assert not any(v for v in ov.values()),ov;lg=[int(x) for x in E("LONGGAP","4,7,200").split(",")];G.DK["longgap"]=tuple(lg[:2]);G.SHORT=lg[2];G._SH.clear();LG=[]
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
C9="Call 911 (or your local emergency number)"
EM4=[("breath",8,"","","{S} just collapsed and {p} {pisnt} breathing|{S} fainted and won't wake up no matter what I do|{S} is on the ground and not responding to me|I found {S} unconscious on the floor|{S} suddenly dropped to the floor and isn't moving|{S}'s lips are blue and {p} {pisnt} breathing|{S} slumped over at the table and doesn't respond when I shake {o}|{S} went limp in the chair and I can't feel {q} breath","{C} now and put the phone on speaker. If they're not breathing normally, start CPR: push hard and fast in the center of the chest, about two pushes a second. The dispatcher can coach you until help arrives.|{C} right now. If they aren't breathing or are only gasping, start chest compressions in the middle of the chest and don't stop until help takes over. If someone can grab an AED, have them bring it.|{C} immediately. Check for normal breathing, and if there isn't any, begin hard, fast chest compressions. Keep going until paramedics arrive or they start breathing.",0),
("uncon",3,"","","{S} passed out but {p} {pis} still breathing|{S} is breathing but I can't wake {o} up|{S} fainted and is breathing but won't open {q} eyes|{S} blacked out on the couch and is snoring strangely","{C} now. Roll them onto their side so their airway stays clear, and keep checking that they're breathing. If they stop breathing, start CPR.|{C} right away. Put them in the recovery position on their side, loosen anything tight around their neck, and stay with them until help arrives.",0),
("choke",5,"","","{S} is choking and can't talk|{S} got food stuck in {q} throat and can't breathe|{S} is grabbing {q} throat and can't make a sound|a piece of steak is stuck in {S}'s throat|{S} is choking on a hard candy and can't get any air","{C} now. If they can't cough, speak, or breathe, stand behind them and give quick upward thrusts just above the belly button until the object comes out. If they pass out, lower them to the floor and start CPR.|{C} right away. If they can still cough hard, let them keep coughing. If they can't breathe or speak, give firm abdominal thrusts (the Heimlich maneuver) until it clears.",0),
("heart",6,"I have a crushing pain in my chest and I'm sweating|my chest feels really tight and the pain is spreading to my jaw|I think I'm having a heart attack|there's heavy pressure in my chest and I feel sick|my chest hurts and I'm short of breath and dizzy|I've had chest pain for twenty minutes and it's getting worse","{C} now, and don't drive yourself. Sit down, stay as calm as you can, and unlock the door so responders can get in. If you're not allergic to aspirin and a doctor hasn't told you to avoid it, chew one regular aspirin while you wait.|{C} right away. Stop what you're doing and rest in a comfortable position. Chest pain like that needs to be checked immediately, even if it starts to ease.","{S} has sudden chest pain and is short of breath|{S} says {q} chest feels like it's being squeezed|{S} is clutching {q} chest and sweating|{S} has pain going down {q} arm and looks pale and clammy","{C} now. Have them stop and sit down, loosen tight clothing, and if they're not allergic, they can chew one regular aspirin. If they collapse and stop breathing, start CPR.|{C} right away, and don't let them drive. Keep them resting and calm, and be ready to start CPR if they become unresponsive.",0),
("stroke",5,"my arm and leg on one side suddenly went weak|I suddenly can't see out of one eye and my words sound wrong|half my face went numb and I can't smile right","{C} now. Sudden weakness, numbness, or vision loss like that can be a stroke. Don't drive yourself, unlock the door if you can, and note when it started.","{S}'s face looks lopsided and {q} speech is garbled|{S} suddenly can't move {q} right arm and {q} words are coming out wrong|{S} is suddenly confused and one side of {q} body is weak|half of {S}'s smile sagged all of a sudden and {p} can't find {q} words|{S} suddenly can't understand what I'm saying and {q} leg gave out","{C} now. These are signs of a stroke, and fast treatment matters. Note the time it started, and don't give them anything to eat, drink, or swallow.|{C} right away. Sudden weakness, a sagging face, or trouble speaking can mean a stroke, so every minute counts. Stay with them and write down when it started for the paramedics.",0),
("bleed",6,"I sliced my hand badly on a knife and it won't stop bleeding|blood is soaking through the towel on my leg|I cut my foot on broken glass and it's pouring out|I gashed my arm on a sheet of metal and it's bleeding fast","{C} now. Press hard on the cut with a clean cloth and don't let up, and add more cloth on top if it soaks through. Sit or lie down so you don't faint.|{C} right away. Keep firm pressure on the wound and raise it above your heart if you can. Unlock the door so help can get to you.","{S} sliced {q} arm open and there's blood everywhere|{S} is bleeding heavily from a deep cut on {q} leg|blood is spurting from {S}'s wrist after {p} put {q} hand through a window|{S} cut {q} hand on a table saw and it's gushing|{S} has a deep gash on {q} thigh that won't stop bleeding","{C} now. Press hard on the wound with a clean cloth or your hand and keep steady pressure. If blood soaks through, add more cloth on top instead of lifting it.|{C} right away. Keep firm, constant pressure on the wound. If it's an arm or leg and pressure isn't stopping it, tie a tight tourniquet a few inches above the wound and note the time.",0),
("anaph",5,"I think I'm having an allergic reaction, my throat feels tight and my face is swelling|I got stung and now I'm breaking out in hives and it's hard to breathe|my tongue is swelling up after I ate something with nuts","{C} now. If you have an epinephrine auto-injector, use it in your outer thigh right away. Sit up if breathing is hard, and unlock your door so help can get in.","{S} ate something with shrimp and now {q} lips and tongue are swelling|{S} got stung by a wasp and is wheezing and covered in hives|{S}'s throat is closing up after eating cashews|{S} took a new antibiotic and now {q} face is puffing up and {p} can't breathe well","{C} now. If they have an epinephrine auto-injector like an EpiPen, use it right away in the outer thigh. Have them lie down with legs raised, or sit up if breathing is hard.|{C} right away; this sounds like a severe allergic reaction. Use their epinephrine auto-injector if they have one, and a second dose can be given after 5 to 15 minutes if they aren't improving.",0),
("seize",4,"","","{S} is having a seizure and {q} whole body is jerking|{S} collapsed and started convulsing|{S} has been seizing for more than five minutes|{S} is twitching all over and not responding","{C} now if it lasts more than five minutes, it's their first seizure, or they're hurt. Move hard objects away, cushion their head, and don't hold them down or put anything in their mouth. When the shaking stops, roll them onto their side.|{C} now. Keep them safe by clearing the area and putting something soft under their head, and time the seizure. Once it stops, turn them on their side and stay with them.",0),
("od",5,"","","{S} took a bunch of pills and is hard to wake up|I think {S} overdosed, {p} {pis} barely breathing|{S} passed out after using drugs and {q} lips are turning blue|{S} mixed alcohol with pills and now {p} won't respond","{C} now. If opioids could be involved and you have naloxone (Narcan), give it. Lay them on their side and start CPR if they stop breathing.|{C} right away. Keep them on their side so they don't choke, stay with them, and keep any pill bottles or packets to show the paramedics.",0),
("poison",3,"","","{S} drank some drain cleaner by accident|{S} swallowed something from under the sink|{S} accidentally drank from a bottle of antifreeze|{S} drank lamp oil thinking it was juice","{C} now if they're drowsy, having trouble breathing, or having a seizure. Otherwise call Poison Control right away (1-800-222-1222 in the US). Don't make them throw up, and keep the container so you can say exactly what it was.",0),
("burn",3,"","","{S} spilled boiling oil on {q} arm and the skin is peeling|{S} got badly burned by a firework|{S} has a huge burn on {q} back from a campfire accident","{C} now for a large or deep burn. Cool it under cool running water for at least 10 minutes, and cover it loosely with clean plastic wrap or a clean cloth. Don't use ice, butter, or creams.",0),
("fire",4,"there's smoke pouring out of my kitchen and I can't put it out|the apartment next door is on fire and smoke is coming into mine|I woke up and my bedroom is full of smoke|flames are coming out of our garage|the fire alarm is going off and I can see flames in the hallway","{C} as soon as you're outside. Get everyone out now, stay low under the smoke, and don't go back in for anything.|{C} once you're out safely. Leave right away, close doors behind you to slow the fire, and meet everyone at a spot outside.","","",0),
("trap",2,"I'm stuck in my room and smoke is coming under the door|there's fire in the stairwell and I can't get out of my apartment","{C} now and tell them exactly where you are. Keep the door closed, block the gap under it with a towel or clothes, and signal from a window. Stay low where the air is clearer.","","",0),
("gas",2,"I smell rotten eggs all over the house|the stove is hissing and the whole place smells like gas|there's a strong gas smell in my basement","{C} once you're out of the building. Leave now, and don't flip switches, light anything, or use your phone inside. You can call the gas company from outside too.","","",0),
("co",2,"the carbon monoxide detector keeps beeping and I feel dizzy|everyone in the house has a headache and the CO alarm went off","{C} from outside. Get everyone, including pets, into fresh air right away, and don't go back in until responders say it's safe.","","",0),
("intr",3,"I just heard glass break downstairs and someone is walking around|a stranger is trying to force my front door open|someone is in my backyard trying to get in the back door","{C} now, and keep your voice low. Get to a room you can lock, stay quiet, and don't go looking for the intruder.|{C} right away. Lock yourself in a room if you can, stay on the line, and let the police handle it.","","",0),
("intr2",1,"I came home and my door is kicked in and I can hear someone inside|my front door is smashed open and I think someone is still in the house","{C} now, and don't go inside. Get somewhere safe, like your car with the doors locked or a neighbor's place, and wait for police.","","",0),
("assault",3,"my partner just hit me and says he'll do it again|my ex showed up and is threatening to hurt me|a man is following me and just grabbed my arm|my roommate is attacking me|my husband is throwing things at me and I'm scared","{C} now if you're in danger. If you can, get to a safe place like a neighbor's or a locked room, and keep your phone with you. You don't deserve to be hurt.|{C} right now. Get away from the person if you safely can and go somewhere with other people around. Your safety comes first.","","",0),
("crash",4,"I just saw a car flip over on the highway|I crashed my car and my passenger is bleeding|there's been a wreck in front of me and someone is stuck inside|a motorcycle hit a truck and the rider isn't moving","{C} now and tell them exactly where you are. Stay clear of traffic, turn on your hazard lights, and don't move anyone who's hurt unless there's fire or other immediate danger.","","",0),
("drown",3,"","","{S} went under at the lake and just got pulled out not breathing|{S} fell into the pool and isn't coming up|{S} is struggling in the river and going under","{C} now. Reach or throw something that floats instead of jumping in if you can. Once they're out of the water, start CPR if they're not breathing.",0),
("heat",3,"","","{S} has been working outside in the heat and now is confused and stumbling|{S} collapsed at the park in this heat and {q} skin is hot and dry|{S} ran a race in the heat and now is talking nonsense","{C} now; that sounds like heat stroke. Move them to shade or air conditioning, remove extra clothing, and cool them fast with cold water or ice packs on the neck, armpits, and groin.",0),
("cold",2,"","","{S} fell through the ice and is shaking and confused|{S} was out in the snow for hours and now is slurring and drowsy","{C} now. Get them somewhere warm, take off wet clothing, and wrap them in dry blankets, covering the head too. Handle them gently, and give warm drinks only if they're fully awake.",0),
("shock",2,"","","{S} touched a live wire and is lying next to it|{S} got electrocuted fixing an outlet and isn't responding","{C} now. Don't touch them until the power is off. Switch it off at the breaker if you can, then check their breathing and start CPR if needed.",0),
("baby",3,"my newborn isn't breathing|my baby is limp and turning blue|my infant stopped breathing in the crib|my baby won't wake up and isn't breathing right","{C} now and put the phone on speaker. If your baby isn't breathing, start infant CPR: 30 quick pushes with two fingers in the center of the chest, then 2 gentle puffs over the mouth and nose. Keep going, and the dispatcher will guide you.","","",0),
("bchoke",2,"my baby is choking on something and can't cry or breathe|my infant swallowed a small toy and can't breathe","{C} now. Hold the baby face down along your forearm with the head lower than the chest and give 5 firm back blows between the shoulder blades, then 5 chest thrusts with two fingers. Repeat until it comes out or help arrives.","","",0),
("labor",1,"my wife's water broke and I can see the baby's head|my girlfriend is giving birth on the bathroom floor","{C} now; the dispatcher can talk you through the delivery. Have her lie down on something clean, support the baby's head as it comes out without pulling, and put the baby skin to skin on her chest, covered with a towel.","","",0),
("labor1",1,"I'm in labor and the baby is coming too fast to get to the hospital|I'm pregnant and I think the baby is coming right now at home","{C} now, and unlock the door. Lie down somewhere clean with towels nearby, and let the dispatcher talk you through it. Once the baby is out, keep them skin to skin on your chest and covered.","","",0),
("missing",3,"my daughter didn't come home from school and nobody knows where she is|I lost my son at the mall|my 4 year old wandered off at the beach|my little boy was playing in the yard and now he's gone","{C} now; don't wait. Tell them what your child looks like, what they were wearing, and where you last saw them. Have someone stay at that spot in case they come back.|{C} right away, and alert staff or lifeguards nearby too. Give a clear description and the last place you saw your child.","","",0),
("hotcar",2,"there's a baby locked in a car in this heat and nobody's around|my kid is stuck in the car with the keys inside and it's really hot out","{C} now. If the child looks limp, sweaty, or unresponsive and help isn't close, break a window on the side farthest from the child to get them out. Then move them to shade and cool them down.","","",0),
("armed",2,"someone is shooting in the parking lot|a man with a knife is threatening people in the store|there's someone with a rifle walking into the building","{C} as soon as it's safe to. Get away if you can. If you can't, hide out of sight, lock or block the door, silence your phone, and stay quiet.","","",0),
("asthma",2,"I can't catch my breath and my inhaler isn't helping","{C} now. Sit upright, keep using your rescue inhaler as your plan directs, and unlock the door so help can get in.","{S} is having a bad asthma attack and the inhaler isn't working|{S} can't catch {q} breath and {q} lips look gray","{C} now. Help them sit upright, keep using their rescue inhaler as their plan directs, and stay calm with them while you wait.",0),
("snake",1,"","","{S} got bitten by a rattlesnake on {q} ankle|a snake bit {S} on the hand while we were hiking","{C} now. Keep them still and calm with the bite at about heart level, and take off rings or tight clothing near it. Don't cut the wound, suck out venom, or use ice or a tourniquet.",0),
("bone",1,"","","{S}'s leg is bent the wrong way after a fall off {q} bike|{S} broke {q} arm and the bone is sticking out","{C} now. Don't try to straighten it; keep the limb still and supported, and cover any wound with a clean cloth. Keep them lying down and warm.",0),
("head",2,"","","{S} got hit in the head with a baseball and now is throwing up|{S} hit {q} head in a car accident and is acting confused|{S} took a hard knock to the head playing football and keeps passing out","{C} now; vomiting, confusion, or passing out after a head injury can be serious. Keep them still and lying down, and watch their breathing until help arrives.",0),
("eye",1,"","","{S} got bleach splashed in {q} eyes|oven cleaner sprayed into {S}'s eye","{C} now, and start rinsing the eye with lukewarm running water right away. Keep rinsing for at least 15 minutes, holding the eyelids open, and don't let them rub it.",0),
("stab",2,"","","{S} got stabbed in the stomach|{S} has been shot in the leg|{S} has a knife stuck in {q} side","{C} now. Press hard on the wound with a clean cloth and keep pressing. If something is stuck in the wound, leave it in and press around it.",0),
("ha",1,"I have the worst headache of my life and it came on all at once|my head suddenly exploded with pain and I'm throwing up","{C} now; a sudden, severe headache can be a sign of bleeding in the brain. Lie down, don't take anything, and unlock your door so help can reach you.","","",0),
("sh",14,"I don't want to be alive anymore|I've been thinking about ending my life|I just want to die|I can't do this anymore, I want it all to end|everyone would be better off without me|I'm thinking about killing myself|I don't see a way out and I keep thinking about suicide|there's no point in going on|I'm done with life|I feel like disappearing forever|I want to hurt myself|I keep thinking about hurting myself|I can't stop thinking about dying|I've decided I don't want to be here anymore|I'm so tired of living","{C} now if you might act on these thoughts or you're in danger. I'm really glad you told me, and I'm here with you. You don't have to carry this alone. Is there someone you trust who could be with you right now?|{C} if you feel you might hurt yourself. Thank you for telling me; that took courage. Your life matters, and I want to keep talking with you. What's been weighing on you most?|{C} right away if you're in danger of acting on this. I'm so sorry you're hurting this much. You deserve support, and reaching out to someone you trust right now could really help.|{C} if you're not safe right now. I hear you, and I'm staying right here with you. Would you tell me a little about what's going on?","","",1),
("shpill",2,"I have a bottle of pills in front of me and I want to take them all|I'm holding all my pills and thinking about swallowing them","{C} now. Please put the pills in another room or give them to someone, even just for now. I'm really glad you reached out, and I'm here with you.","","",1),
("shblade",2,"I'm holding a razor and I want to cut|I have a knife and I want to hurt myself with it","{C} now if you're in danger. Please set the blade down and put some distance between you and it. I'm here with you, and I'm glad you told me.","","",1),
("shhigh",2,"I'm on the roof and thinking about jumping|I'm standing on a bridge and I want to jump|I'm at the edge of a cliff and I want to end it","{C} now. Please step back from the edge and move somewhere safer while we talk. Your life matters, and I'm right here with you.","","",1),
("shgun",1,"I have a gun and I'm thinking about using it on myself|I'm holding my dad's gun and I want to die","{C} now. Please unload the gun and put it away from you, or leave the room it's in. I'm really glad you told me, and I'm here with you.","","",1),
("shcar",1,"I'm sitting in my car in the garage with the engine running|I closed the garage and started the car because I want to die","{C} now. Turn off the engine and get out into fresh air right away. I'm glad you reached out, and I'm staying right here with you.","","",1),
("shod",3,"I just swallowed a bunch of pills|I took a whole pack of my medication on purpose|I took too many pills to hurt myself and now I'm scared","{C} right now, even if you feel okay; some pills cause serious harm hours later. Unlock your door and tell them what you took. I'm here with you, and you deserve help.","","",1),
("shcut",2,"I cut myself and it's deeper than I meant|I've been cutting and this time it won't stop bleeding","{C} now if the bleeding is heavy or won't stop, and press firmly on it with a clean cloth. I'm really glad you told me. You matter, and I'm here with you.","","",1),
("sh3",6,"","","{S} said {p} {pwants} to kill {x}|{S} is talking about ending {q} life|{S} locked {x} in the bathroom and says {p} {pwants} to die|{S} is giving away {q} things and saying goodbye to everyone|{S} told me {p} {phas} been planning to take {q} own life","{C} now if they're in immediate danger or have hurt themselves. Stay with them if it's safe, keep them talking, and move anything they could use to hurt themselves out of reach. You're doing the right thing by acting.|{C} now if they might act on it. Take what they said seriously, stay close, and let them know you care and want to help.",1),
("sh3t",2,"","","{S} texted me a goodbye message and isn't answering|{S} posted a goodbye note online and won't pick up {q} phone","{C} now and ask for a welfare check at their address. Keep trying to reach them, and ask anyone nearby to go check on them.",1),
("sh3c",1,"","","I found {S} cutting {q} arms|{S} hurt {x} on purpose and it's bleeding a lot","{C} now if the cuts are deep or bleeding heavily. Press a clean cloth on any bleeding, stay calm, and stay with them. Let them know you care and aren't angry.",1)]
EM4=[(k,w,P4(a),[x.replace("{C}",C9) for x in P4(b)],P4(c),[x.replace("{C}",C9) for x in P4(d)],z) for k,w,a,b,c,d,z in EM4]
PRN={"he":("he","his","him","himself","is","isn't","wants","has"),"she":("she","her","her","herself","is","isn't","wants","has"),"they":("they","their","them","themselves","are","aren't","want","have")}
SUBJ4=SUBJ+[("my little brother","he"),("my mother-in-law","she"),("my niece","she"),("my nephew","he"),("my stepdad","he"),("my classmate","they"),("my best friend","they"),("my grandfather","he"),("a kid at the park","they"),("a man at the gym","he"),("a woman on the train","she"),("my teammate","they")]
PRE4=["","","","","help, ","help! ","please help, ","pls help ","omg ","oh no, ","Adam, ","quick, ","urgent: ","I don't know what to do, ","hey, ","someone help, "];SUF4=["","","","",". What do I do?",", what should I do?","!","!!",". Please hurry.",", help me",". I'm panicking.",". What now?"," and I'm scared",". Is this serious?","."];SUF4S=["","","","",".","...",". I don't know what to do."," and I'm scared.",". Can you talk to me?",". I just needed to tell someone."]
def talk(st)->list:
 R=random.Random(SEED+44);EF=[];GR=(("contam13",13),("contam8",8),("chat6",6))
 def ok(T):
  ws=DC.words("".join(f"User: {u}\nAdam: {a}\n\n" for u,a in T));b=next(((w," ".join(ws[int(p[0]):int(p[0])+n])) for k,(w,n) in enumerate(GR) for p in [hits(ws,k)[0]] if len(p)),None);e=next((u for u,_ in T if nrm(u) in EQS),None)
  return b or (("eval_line",e) if e else ("",""))
 out,info=TK.build(R,N("TALK_N",8000),ok,st,float(E("TALK_HOLD_FRAC",".06")),N("TALK_CAP",30));[(st.update(["talk_em_fires"]),len(EF)<40 and EF.append(u)) for _,T,_,_ in out for u,_ in T if EM.respond(u)];TKI.update(info,em_fires=EF)
 return [ex("talk",k,T,[u for u,_ in T],"chat",ho=h,sc=sc) for k,T,sc,h in out]
def emerg4(st)->list:
 R=random.Random(SEED+43);D=dis(st);out=[];seen=set();n=N("EMERG_N",2400);CL=[x for x in SUBJ4 if x[0].startswith("my ")];RC=collections.Counter();t=0
 assert all(r.startswith(C9) for x in EM4 for r in x[3]+x[5])
 for t in range(n*40):
  if len(out)>=n:break
  k,w,fp,fr,tp,tr,sh=R.choices(EM4,[x[1] for x in EM4])[0];f=bool(fp) and (not tp or R.random()<.5);S,pn=R.choice(CL if sh else SUBJ4);v=PRN[pn]
  c,rs=(R.choice(fp),fr) if f else (R.choice(tp).format(S=S,p=v[0],q=v[1],o=v[2],x=v[3],pis=v[4],pisnt=v[5],pwants=v[6],phas=v[7]),tr)
  c=R.choice(PRE4)+c+R.choice(SUF4S if sh else SUF4);c=c[:1].upper()+c[1:] if R.random()<.6 else c;r=min(rs,key=lambda y:(RC[y],R.random()));x=f"User: {c}\nAdam: {r}\n\n"
  why="dup" if nrm(c) in seen else "chat_suite_emergency" if any(q in nrm(c) for q in CQE) else "eval_line" if nrm(c) in EQS else dirt(x)
  if why:st["emerg4_skip_"+why]+=1;continue
  seen.add(nrm(c));RC[r]+=1;st["emerg4_kind_"+k]+=1;st["emerg4_selfharm"]+=sh;st["emerg4_first_person"]+=f;out.append(ex("emerg4",f"x{t}",([R.choice(D)[:2]] if R.random()<.3 else [])+[(c,r)],[c],"emerg"))
 st["emerg4_tries"]=t;st["emerg4_distinct_replies"]=len(RC);st["emerg4_max_reply_repeat"]=max(RC.values());return out
BUD=int(E("SERVE_BUDGET","2928"));XK=("calc","kctx","umem");WK=N("WORKERS",2);CMM=[];EQS=set();CQE=set()
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
  m=smem(q,hist(R,D) if R.random()<.25 else [])
  if m["reply"]:st["calc_skip_memory_reply"]+=1;continue
  T,u,pb=srv(q,m["hist"],(),[cline(q,body)]);st["calc_kind_"+kind]+=1;out.append(ex("calc",f"c{len(out)}",T+[(u,cre(R,kind,body,x,v))],[q],"calc",lo=1,np=0,sh="calc:"+kind+(":hist" if T else ""),pb=pb))
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
def kx(q:str,h=())->list:
 n=kq(q);K,rq=AC._know(q,2,min(560,n//2),[tuple(x)[:2] for x in h]) if n>=64 else ([],q);w=AC._wd(RKN,rq) if K and rq else None
 return [q,{"e":int(w["e"]),"a":w["answer"],"p":int(w.get("p") or 0),"conf":float(w["conf"])} if w else None,K,K[0] if w else None]
def smem(q,h)->dict:m=AC._mem(q,[tuple(x)[:2] for x in h],"sft-ctx-"+hashlib.md5(q.encode()).hexdigest()[:12],None);return {**m,"hist":[tuple(x)[:2] for x in m["hist"]]}
def khist(R,D,QA,HW,q,KX,st)->tuple:
 H=hist(R,D,QA,HW)
 for h in ([H,[]] if H else [[]]):
  m=smem(q,h);rq=AC._resolve(RKN,q,h) if h else q;r=KX[q] if rq==q else kx(q,h);ok=bool(r[2]) and not m["reply"] and (not m["route"] or rq not in ("",q));st["kx_hist_"+("none" if not h else "same" if rq==q else "skip" if not rq else "resolved")+("" if ok else "_dropped")]+=1
  if ok:return m["hist"],r
 return [],[q,None,[],None]
def kxall(qs:list,st)->dict:
 cf=RAW/f"kx_cache_{hashlib.sha1(Path(AC.__file__).read_bytes()).hexdigest()[:12]}.jsonl";C={}
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
def wd_q(R,st,n=0,skip=frozenset(),tag="wd")->list:
 S=WF.store();assert S is not None,"wikifact store missing";P=np.asarray(S.f_p);SL=np.asarray(S.e_sl);out=[];qp=(n or N("WD_N",7500))//len(WQ)
 for p,(qt,_,msl) in WQ.items():
  F=np.flatnonzero(P==p);U=np.unique(np.asarray(S.f_s[F]));U=U[SL[U]>=msl].tolist();R.shuffle(U);k=0
  for e in U:
   if k>=qp:break
   if e in skip:continue
   nm=AC.asc(WF.fold(S.name(e)));F2=S.facts(e,[p])
   if not NMOK.fullmatch(nm) or nm.lower().startswith(("list of","category:")) or len(F2)!=1 or int(S.f_k[F2[0]]) not in (1,2):st[tag+"_skip_subject"]+=1;continue
   t=R.choice(qt);a,fo=wdisp(S,F2[0],t.startswith("what year"))
   if not a or not NMOK.fullmatch(a) or WF.nrm(a) in WF.nrm(nm) or WF.nrm(nm) in WF.nrm(a):st[tag+"_skip_answer"]+=1;continue
   if any(e2!=e and any(WF.nrm(wdisp(S,f3)[0]) not in fo for f3 in S.facts(e2,[p])) for e2,_ in S.ents(WF.nrm(nm))[:40]):st[tag+"_skip_ambiguous"]+=1;continue
   q=t.format(s=nm);q=q[:1].upper()+q[1:] if R.random()<.6 else q.lower();k+=1;out.append({"q":q,"e":int(e),"p":p,"s":nm,"a":a,"fo":fo})
  st[f"{tag}_q_p{p}"]=k
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
  q=c["q"];H,(_,w,K,wl)=khist(R,D,QA,HW,q,KX,st);lab="rel" if wl and w["e"]==c["e"] and WF.nrm(w["a"]) in c["fo"] else "off_nat" if wl else "none";n=kq(q);cap=min(560,n//2)
  if not K:st["wd_no_context"]+=1;continue
  if lab=="rel" and R.random()<float(E("WD_SYN",".35")):
   o=[x for x in BP[c["p"]] if x["e"]!=c["e"]];F3=[(p2,f) for p2 in WQ if p2!=c["p"] for f in S.facts(c["e"],[p2])[:1]];z=R.choice(F3) if F3 and R.random()<.5 else None;d2=wdisp(S,z[1])[0] if z else ""
   nl=AC._wl({"label":S.name(c["e"]),"p":z[0],"answer":d2},cap) if d2 and NMOK.fullmatch(d2) and WF.nrm(d2) not in c["fo"] else KX[R.choice(o)["q"]][3] if o else None
   K,lab=([nl]+K[1:],"off_syn_prop" if d2 and nl.endswith(f": {d2}.") else "off_syn_ent") if nl else (K,lab);wl=nl or wl
  rc=AC._fit(K[::-1],0,n)[0][::-1];hl=bool(wl) and bool(rc) and rc[0]==wl;ps=rc[1:] if hl else rc;sup=insup(ps,c["fo"]);lab=lab if hl else "none"
  if lab!="rel" and not sup and R.random()>float(E("WD_UNSUP",".5")):st["wd_unsup_dropped"]+=1;continue
  a=wtpl(R,c["p"],c["s"],AC.asc(w["a"])) if lab=="rel" else wtpl(R,c["p"],c["s"],c["a"]) if sup else hedge(R,c["a"]);T,u,pb=srv(q,H,(),rc)
  st["wd_"+lab]+=1;st["wd_"+lab+("_sup" if sup else "_unsup")]+=1;out.append(ex("wikidata",f"w{i}",T+[(u,a)],[q],"kctx",lo=1,np=0,sh=f"wd:{lab}:{'sup' if sup else 'unsup'}:p{len(ps)}"+(":dis" if any(not insup([x],c["fo"]) for x in ps) else "")+(":hist" if T else ""),pb=pb))
 for i,c in enumerate(KC):
  q=c["q"];H,(_,w,K,wl)=khist(R,D,QA,HW,q,KX,st)
  if not K:st["kctx_no_context"]+=1;continue
  rc=AC._fit(K[::-1],0,kq(q))[0][::-1];hl=bool(wl) and rc[0]==wl;ps=rc[1:] if hl else rc;fo=[WF.nrm(c["a"])];sup=insup(ps,fo);lab="sup" if sup else "line" if hl and WF.nrm(w["a"])==fo[0] else "unsup"
  if lab=="unsup" and R.random()>float(E("KC_UNSUP",".25")):st["kctx_unsup_dropped"]+=1;continue
  T,u,pb=srv(q,H,(),rc);st["kctx_"+lab]+=1;out.append(ex("kctx",c["id"],T+[(u,fin(c["a"]) if lab!="unsup" else hedge(R,c["a"]))],[q],"kctx",lo=1,np=0,sh=f"kctx:{lab}:p{len(ps)}"+(":dis" if any(not insup([x],fo) for x in ps) else "")+(":wdline" if hl else "")+(":hist" if T else ""),pb=pb))
 return out
RDN={"squad":N("RD_SQUAD",9000),"adversarial_qa":N("RD_ADV",3000),"tydiqa":N("RD_TYDI",3500),"hotpot":N("RD_HOTPOT",7000),"wikidata":N("RD_WD",7000)};RDH=float(E("RD_HEDGE",".25"));RDK=[int(x) for x in E("RD_DW","25,45,30").split(",")];RDG=float(E("RD_NAT",".7"));RDW=float(E("RD_WDL",".6"));RDS=float(E("RD_SENT",".3"));BASE=[None]
DEIX=re.compile(r"(?i)\b(?:passage|article|paragraph|the text|this text|the author|the above|the following)\b");PRO=re.compile(r"^(?:He|She|It|They|This|These|Those|His|Her|Its|Their|The latter|The former)\b");SPL=lambda x:[z for z in re.split(r"(?<=[.!?])\s+",x) if z]
RH=["I'm not sure. The {p} {d}n't say, but I think it's {a}.","The {p} {d}n't mention that, so I can't be certain. My best guess is {a}.","I'm not sure; the {p} {d}n't cover it. I think it might be {a}.","That isn't in the {p}, so I'm not certain, but I believe it's {a}.","The {p} {d}n't say. If I had to guess, I'd say {a}.","I can't find that in the {p}. I think it's {a}, but I'm not sure."]
RHD=["I'm not sure. The {p} {d}n't say, but I think it was {D}.","The {p} {d}n't give the date, so I can't be exact. My best guess is sometime {D}.","That isn't in the {p}. I think it was {D}, but I don't know the exact date.","I can't find the date in the {p}. If I had to guess, I'd say {D}.","I'm not certain; the {p} {d}n't mention it. Probably {D}, but I can't be more exact."]
RHN=["I'm not sure. The {p} {d}n't give the number, but I think it's {D}.","The {p} {d}n't say, so I can't give an exact figure. My rough guess is {D}.","That number isn't in the {p}. I'd guess {D}, but I'm not certain.","I can't find the figure in the {p}. It's probably {D}, but I don't know exactly.","I'm not certain; the {p} {d}n't mention it. I think it's {D}, but that's only a rough guess."]
MONR="|".join(WF.MON);DATEP=re.compile(r"(?i)(?:\b(?:in|on|by|around|circa|about|the|early|late|mid|of|and|to|or|between)\b|c\.|\d{1,4}(?:s|st|nd|rd|th)?\b|\b(?:%s)\b|\bB\.?C\.?E?\b\.?|\bA\.?D\b\.?|\bC\.?E\b\.?|[\s,/-])+"%MONR);NWD={**{w:i for i,w in enumerate(ONES)},**{w:10*i for i,w in enumerate(TENS) if w!="_"}}
NUMP=re.compile(r"(?i)(?:(?:about|around|approximately|roughly|over|nearly|almost|more than|less than|up to|some)\s+)?(\$|GBP |EUR |USD |JPY )?(\d[\d,]*(?:\.\d+)?|[a-z]+)(?:\s*(%|percent\b))?(?:\s+(thousand|million|billion|trillion|hundred|dozen)\b)?(?:\s+([A-Za-z][A-Za-z -]{0,38}))?");MAG=[(1e12,"trillions"),(1e11,"hundreds of billions"),(1e10,"tens of billions"),(1e9,"billions"),(1e8,"hundreds of millions"),(1e7,"tens of millions"),(1e6,"millions"),(1e5,"hundreds of thousands"),(1e4,"tens of thousands"),(1e3,"thousands"),(100,"hundreds")]
def ordn(c:int)->str:return f"{c}{'th' if 10<=c%100<=20 else {1:'st',2:'nd',3:'rd'}.get(c%10,'th')}"
def hphr(a:str,q:str)->tuple:
 s=a.strip().rstrip(".");qt=RKN.RL.qtype(q);mo=re.findall(r"\b(%s)\b"%MONR,s);bc=bool(re.search(r"(?i)\bB\.?C",s));er=bc or bool(re.search(r"(?i)\b(?:A\.?D|C\.?E)\b",s));ys=[int(y) for y in re.findall(r"\b(\d{1,4})s?\b",s) if er or 100<=int(y)<=2100];m=NUMP.fullmatch(s)
 if DATEP.fullmatch(s) and (qt=="date" or mo or qt!="num" and ys and all(1000<=y<=2100 for y in ys) and re.search(r"(?i)\b(?:y(?:ea|ae)rs?|when|date|decade|century|period|era|time|founded|established|built|born|died|organi[sz]ed|formed|opened|released|published|began|started|ended|created|completed)\b",q)) and (ys or mo):
  y=ys[0] if ys else 0;c=(y-1)//100+1;return ("entity",a) if not ys and not mo else ("date_month",f"in {mo[0]}") if not ys else ("date_decade",f"in the {y//10*10}s") if not bc and y>=1500 and len({z//10 for z in ys})==1 else ("date_century",f"in the {ordn(c)} century{' BC' if bc else ''}")
 n=(float(m[2].replace(",","")) if m[2][0].isdigit() else NWD.get(m[2].lower())) if m else None;n=n*{"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12,"hundred":100,"dozen":12}.get((m[4] or "").lower(),1) if n is not None else None
 u=((m[5] or "").strip() or {"$":"dollars","GBP ":"pounds","EUR ":"euros","USD ":"dollars","JPY ":"yen"}.get(m[1] or "","")) if m else "";u=u if len(u.split())<=5 and not re.search(r"(?i)\b(?:and|or|to)\b",u) else None
 if n is not None and n>=0 and u is not None and qt!="date" and (m[2][0].isdigit() or qt=="num"):
  lo=int(n//10*10);w=next((x for t,x in MAG if n>=t),None);return ("num_percent",f"somewhere between {lo} and {lo+10} percent" if n>=10 else "under 10 percent") if m[3] else ("num_magnitude",(f"{w} of {u}" if u else f"somewhere in the {w}") if w else (f"somewhere between ten and a hundred {u}" if n>=10 else f"fewer than ten {u}").strip())
 return None if s[:1].isdigit() or qt in ("date","num") and re.search(r"\d",s) else ("entity",a)
def has(x:str,f:str)->bool:f=nrm(f);return bool(f) and f" {f} " in f" {nrm(x)} "
def surf(x:str,f:str)->str:m=re.search(r"(?i)(?<![A-Za-z0-9])"+r"[^A-Za-z0-9]+".join(map(re.escape,nrm(f).split()))+r"(?![A-Za-z0-9])",x);return m[0] if m else f
def dfm(a:str)->list:m=re.fullmatch(r"([A-Z][a-z]+) (\d{1,2}), (\d{3,4})",a);n=re.fullmatch(r"(\d{1,2}) ([A-Z][a-z]+) (\d{3,4})",a);return [a]+([f"{m[2]} {m[1]} {m[3]}"] if m else [f"{n[2]} {n[1]}, {n[3]}"] if n else [])
def battr(ss)->set:return {r["id"] for l in gzip.open(BASE[0]/"attribution.jsonl.gz","rt") for r in [json.loads(l)] if r["src"] in ss} if BASE[0] else set()
def qw(q:str)->dict:I=RKN.ix();return {x:I.widf(x) for x in dict.fromkeys(RKN.toks(q)) if x not in RKN.QSTOP}
def cov(W:dict,z:str)->float:t=set(RKN.toks(z));return sum(v for k,v in W.items() if k in t)/(sum(W.values()) or 1.0)
def rtrim(l:str,cp:int,q:str,f=None)->str:
 if len(l)<=cp:return l
 t,_,x=l.partition(": ");W=qw(q);S=SPL(x);return f"{t}: "+AC._trim(S,[cov(W,z)+(5*has(z,f) if f else 0) for z in S],cp-len(t)-2)
def rd_items(R,st,X)->list:
 U=lambda t:a1(urllib.parse.unquote(t or "")).replace("_"," ");out=[];ok=lambda q,a:bool(q and a and nrm(a)) and len(a.split())<=10 and len(q)<=250 and nrm(a) not in ("yes","no") and not DEIX.search(q) and nrm(q) not in EQS and not has(q,a)
 for s in ("squad","adversarial_qa","tydiqa"):
  Y=[(U(r["title"]),a1(r["question"]),a1((r["answers"]["text"] or [""])[0]),a1(r["context"]),r["id"]) for r in rows(s,["id","title","context","question","answers"]) if s!="tydiqa" or r["id"].startswith("english")];R.shuffle(Y)
  Y=[{"k":f"{s}:{i}","s":s,"q":q if q.endswith("?") else q+"?","a":[a],"t":[t],"c":c,"g":f"{s}|{t}"} for t,q,a,c,i in Y if ok(q,a)][:RDN[s]];st[f"rd_items_{s}"]=len(Y);out+=Y
 HU={e["k"] for e in X if e["s"] in ("hotpot_rc","hotpot_cb","kctx")}|battr(("hotpot_rc","hotpot_cb","kctx"));Y=[];st["rd_hotpot_ids_used_before"]=len(HU)
 for r in rows("hotpot",["id","question","answer","supporting_facts","context"],2000):
  q,a=a1(r["question"]),a1(r["answer"])
  if r["id"] in HU or not ok(q,a):continue
  ts=list(dict.fromkeys(r["supporting_facts"]["title"]));cx=dict(zip(r["context"]["title"],r["context"]["sentences"]));Y.append({"k":"hotpot:"+r["id"],"s":"hotpot","q":q,"a":[a],"t":[a1(t) for t in ts],"c":" ".join(a1(x) for t in ts for x in cx.get(t,[])),"g":"hotpot|"+a1(ts[0]) if ts else "hotpot|"})
 R.shuffle(Y);Y=Y[:RDN["hotpot"]];st["rd_items_hotpot"]=len(Y);out+=Y;S=WF.store();V5={c["e"] for c in wd_q(random.Random(SEED+11),collections.Counter())};st["rd_wd_subjects_of_wikidata_source_excluded"]=len(V5)
 Y=[{"k":f"wikidata:{c['e']}:{c['p']}","s":"wikidata","q":c["q"],"a":dfm(c["a"]),"t":[S.title(c["e"]).replace("_"," ")],"c":"","g":f"wikidata|{c['p']}","p":c["p"],"sn":c["s"]} for c in wd_q(random.Random(SEED+53),st,RDN["wikidata"],V5,"rd_wd") if ok(c["q"],c["a"]) and all(unicodedata.normalize("NFKD",x).encode("ascii","ignore") for x in S.name(c["e"]) if x.isalpha())];st["rd_items_wikidata"]=len(Y);return out+Y
def rdx(it)->dict:
 q,F=it["q"],it["a"];n=kq(q);c2,c3=min(560,n//2),min(560,n//3);K,rq=AC._know(q,3,c2) if n>=64 else ([],q);w=AC._wd(RKN,rq) if K and rq else None;I=RKN.ix();getattr(I,"tz",None) is None and I.link(q);W=qw(q);ct=set(RKN.toks(it["c"]));cache={};b=None
 for t in it["t"]:
  for g in (AC._pages(I,RKN,t)[:2] if t else []):
   if AC._BADT.search(I.text(int(I.first[g]),cache)[0]):continue
   for p in range(int(I.first[g]),min(int(I.first[g+1]),int(I.first[g])+400)):
    tt,x,_=I.text(p,cache)
    if AC._NAV.search(x[:300]):continue
    x=AC.asc(AC._fix(x));f=next((f for f in F if has(x,f)),None)
    if not f:continue
    sc=len(set(RKN.toks(x))&ct)/(len(ct) or 1)+cov(W,x);b=b if b and b[0]>=sc else (sc,p,AC.asc(tt),x,f)
 S=SPL(b[3]) if b else [];G={str(cp):l for cp in (c2,c3) for l in [re.sub(r"\s*\x01\s*"," ",f"{b[2]}: "+AC._trim(S,[cov(W,z)+5*has(z,b[4]) for z in S],cp-len(b[2])-2)).strip()] if has(l,b[4])} if b else {}
 return {"k":it["k"],"rq":rq,"K":K,"wl":bool(w) and bool(K) and K[0]==AC._wl(w,c2),"G":G,"gp":b[1] if b else -1,"gt":b[2] if b else ""}
def rdall(IT,st)->dict:
 cf=RAW/f"rd_cache_{hashlib.sha1(Path(AC.__file__).read_bytes()+Path(RKN.__file__).read_bytes()+(Path(RKN.IXD)/'meta.json').read_bytes()).hexdigest()[:12]}.jsonl";C={}
 cf.exists() and [C.__setitem__(r["k"],r) for l in open(cf) if l.strip() for r in [json.loads(l)]];todo=[it for it in IT if it["k"] not in C];st["rd_retrieved"]=len(todo);st["rd_cache_hits"]=len(IT)-len(todo);t0=time.time()
 with mp.get_context("spawn").Pool(WK) as pool,open(cf,"a") as f:
  for j,r in enumerate(pool.imap(rdx,todo,chunksize=8)):C[r["k"]]=r;f.write(json.dumps(r)+"\n");j%1000==999 and (f.flush(),print(time.strftime("%H:%M:%S"),"rd",j+1,"/",len(todo),f"{time.time()-t0:.0f}s",flush=True))
 return C
def reader(st,X)->list:
 R=random.Random(SEED+51);IT=rd_items(R,st,X);RX=rdall(IT,st);P={};BG=collections.defaultdict(list);out=[];nh=na=0
 for it in IT:r=RX[it["k"]];P[it["k"]]=(r["K"][1:] if r["wl"] else r["K"],r["K"][0] if r["wl"] else None);BG[it["g"]].append(it["k"]);st["rd_gold_found_"+it["s"]+("_yes" if r["G"] or any(has(l,f) for l in P[it["k"]][0] for f in it["a"]) else "_no")]+=1
 R.shuffle(IT)
 for it in IT:
  r=RX[it["k"]];q=it["q"];F=it["a"];n=kq(q);ps,wdl=P[it["k"]];gl=lambda l:next((f for f in F if has(l,f)),None);NG=[l for l in ps if gl(l)];qs=set(qw(q));AT=[x for x in dict.fromkeys(z for f in F for z in RKN.toks(f)) if x not in set(RKN.toks(q))|RKN.STOP];tp=max(AT,key=RKN.ix().widf) if AT else None
  lk=lambda l:bool(gl(l)) or bool(AT) and (lambda z:tp in z or 2*len(set(AT)&z)>=len(AT))(set(RKN.toks(l or "")))
  SB=list(dict.fromkeys(l for o in R.sample(BG[it["g"]],min(8,len(BG[it["g"]]))) if o!=it["k"] for l in P[o][0] if l not in ps and not lk(l)))[:4];st["rd_distractor_partial_leak_skipped"]+=sum(1 for l in ps if lk(l) and not gl(l));DP=[(l,"same") for l in ps if not lk(l)]+[(l,"rel") for l in SB];hz=hphr(F[0],q);h=bool(DP) and hz is not None and nh<RDH*(nh+na+1);st["rd_hedge_uncoarsenable_not_hedged"]+=bool(DP) and hz is None
  if not (h or NG or r["G"]):st["rd_skip_"+("no_context" if not DP else "hedge_quota_full")]+=1;continue
  for tr in range(5):
   k=min(len(DP),R.choices((1,2),(55,45))[0] if h else R.choices((0,1,2),RDK)[0]-(tr>2));D=R.sample(DP,max(0,k));cp=min(560,n//(3 if len(D)+(not h)>=3 else 2));gn=R.choice(NG) if NG else None;gs="nat" if gn and (str(cp) not in r["G"] or R.random()<RDG) else "con";g=None if h else rtrim(gn,cp,q,gl(gn)) if gs=="nat" else r["G"].get(str(cp))
   if not h and not (g and gl(g)):st["rd_retry_gold_trim"]+=1;continue
   L=([] if h else [(g,"gold")])+[(rtrim(l,cp,q),t) for l,t in D];R.shuffle(L);k0=bool(wdl) and R.random()<RDW and not (h and lk(wdl));rc=[l for l,_ in L]
   kw=k0 and len(AC._fit(([wdl]+rc)[::-1],0,n)[0])==len(rc)+1;st["rd_wdline_dropped_to_fit"]+=k0 and not kw;rc=([wdl] if kw else [])+rc
   if not rc or len(AC._fit(rc[::-1],0,n)[0])<len(rc):st["rd_retry_fit"]+=1;continue
   f=F[0] if h else gl(g);sf=a1(F[0].rstrip(".") if h else surf(g,f));SN=[] if h else [z for z in SPL(g.partition(": ")[2]) if has(z,f) and 5<=len(z.split())<=20 and z.endswith(".") and not PRO.match(z) and not ABR.search(z) and "(" not in z and " - " not in z and set(RKN.toks(z))&qs]
   pd=dict(p="passage" if len(L)==1 else "passages",d="does" if len(L)==1 else "do");rk="hedge" if h else "wd_tpl" if it["s"]=="wikidata" else "sentence" if SN and R.random()<RDS else "span";a=(R.choice(RH).format(a=sf,**pd) if hz[0]=="entity" else R.choice(RHD if hz[0].startswith("date") else RHN).format(D=hz[1],**pd)) if h else wtpl(R,it["p"],it["sn"],sf) if rk=="wd_tpl" else R.choice(SN) if rk=="sentence" else fin(sf)
   T,u,pb=srv(q,[],(),rc);ua,aa,ga=a1(u),a1(a),a1(g or "");sv=h or (has(ga,f) and ga in ua and sf in ga and (rk!="sentence" or aa in ga));ab=not h or not any(lk(x) for x in rc)
   if not (sv and ab):st["rd_retry_"+("span_unverified" if not sv else "answer_leak")]+=1;continue
   cc=not h or hz[0]=="entity" or not has(a,F[0]) or nrm(F[0]) in nrm(hz[1])
   if not cc:st["rd_retry_coarse_leak"]+=1;continue
   tx=f"User: {u}\nAdam: {a}\n\n";why=dirt(tx) or ("eval_line" if nrm(q) in EQS else "nf" if NF.clean(tx)[0]!=tx.strip() else "identity" if C5.IDN.search(tx) else "")
   if why:st["rd_retry_"+why]+=1;continue
   break
  else:st["rd_skip_after_retries"]+=1;continue
  nh+=h;na+=not h;nd=sum(t=="same" for _,t in L);nr=sum(t=="rel" for _,t in L);gp=next((i for i,(_,t) in enumerate(L) if t=="gold"),-1)
  tg=[f"sub:{it['s']}",f"kind:{'hedge' if h else 'answer'}",f"npass:{len(L)}",f"ndist:{nd+nr}",f"dist_same:{nd}",f"dist_rel:{nr}",f"gold:{'none' if h else gs}",f"wdline:{'none' if not kw else 'with_answer' if gl(wdl) else 'no_answer'}",f"reply:{rk}",f"serve_retrieves:{bool(r['K'])}",f"cap:{cp}"]+([] if h else [f"goldpos:{gp+1}of{len(L)}","span_verified:1",f"span_case_exact:{int(sf in g)}"])+(["answer_absent_verified:1",f"hedge_type:{hz[0]}"]+([] if hz[0]=="entity" else ["coarse_exact_absent_verified:1"]) if h else [])
  out.append(ex("reader",it["k"],T+[(u,a)],[q],"kctx",lo=1,np=0,sh=f"rd:{'hedge' if h else 'answer'}:{it['s']}:p{len(L)}",pb=pb,rd=tg))
 st["rd_examples"]=len(out);st["rd_hedge"]=nh;st["rd_answer"]=na;return out
def rdmeta(st,cnt,RD)->dict:
 A=RD["train"]+RD["hold"];n=A["kind:answer"]+A["kind:hedge"];g=lambda p:{k[len(p):]:v for k,v in sorted(A.items()) if k.startswith(p)};sm=lambda p:sum(int(k[len(p):])*v for k,v in A.items() if k.startswith(p));rk=json.load(open(Path(RKN.IXD)/"meta.json"))
 return {"what":f"v{VER} = v{VER-1} (token, mask, index and attribution rows byte-identical, same source order) + the new reader source (ADAM-T-READER): reading practice in the exact serve knowledge-prompt format, so the model pulls the answer out of a passage that is present, ignores distractor passages and hedges when the answer is absent","why":"E2E report data/adam_t/eval/report_adam-t-t30m-sft_4638_e2e.json (SFT v3 step 4638): with the gold answer in context, fact was right 36% of the time (12.5% without); the FineWiki live index now puts gold in context for ~85% of knowledge questions, so reading is the biggest quality lever",
 "examples":{"train":cnt["reader"]["train_examples"],"hold":cnt["reader"]["hold_examples"],"total":n},"by_subsource":g("sub:"),"hedge_share":round(A["kind:hedge"]/max(1,n),4),"kind":g("kind:"),"hedge_type":g("hedge_type:"),"coarse_hedges":{"n":sum(v for k,v in A.items() if k.startswith("hedge_type:") and k!="hedge_type:entity"),"exact_answer_absent_verified":A["coarse_exact_absent_verified:1"],"rule":"CTO 2026-09-28 (quality first): a hedge never states an exact date or number it cannot see. Date answers (ray_learn.qtype date or a month name, the whole answer a date expression) are guessed by decade (years >= 1500 within one decade: 'in the 1960s'), otherwise by century ('in the 18th century', BC kept), or by month when there is no year ('in May'); number answers (a leading number, or a number word for a how-many/much question, optional currency, scale word, percent and a <=5-word letters-only unit) are guessed by magnitude ('somewhere in the thousands', 'tens of thousands of people', 'somewhere between ten and a hundred', 'fewer than ten') or a 10-point percent band; named entities keep the specific guess; era-marked years (BC/AD/CE) of any size are years, a date question never gets a number hedge, year-only answers also count as dates when the question has a time word (year incl. the 'yaer' typo, when, date, decade, century, period, era, time, founded, established, built, born, died, organized, formed, opened, released, published, began, started, ended, created, completed); an answer that starts with a digit, or a date/number question's answer with digits, that fits neither is not used as a hedge (named entities with digits such as 'Apollo 11' keep the specific guess). The coarse reply is checked not to contain the exact answer (unless the answer is itself the coarse form, e.g. 'the 1960s'); %d date and %d number templates"%(len(RHD),len(RHN))},
 "span_verification":{"answer_examples":A["kind:answer"],"verified":A["span_verified:1"],"rate":round(A["span_verified:1"]/max(1,A["kind:answer"]),4),"case_exact":A["span_case_exact:1"],"hedge_answer_absent_verified":A["answer_absent_verified:1"],"hedge_rate":round(A["answer_absent_verified:1"]/max(1,A["kind:hedge"]),4),"rule":"answer example: the answer (normalised, whole words) is in the gold passage line, the gold line is verbatim in the final user text, the reply's answer text is verbatim in the gold line (for sentence replies the whole sentence), and no distractor passage leaks the answer; hedge example: no passage and no kept Wikidata line leaks the answer; a line leaks the answer when it contains it (normalised, whole words), its rarest answer token (R-KNOW idf, question words and stopwords excluded) or at least half of its answer tokens, so name variants such as 'Evelyn P. Stevens' for 'Evelyn Stevens' count as present; examples failing a check are recomposed (up to 5 tries) or dropped"},
 "passages_per_example":g("npass:"),"distractors":{"per_example":g("ndist:"),"total":sm("ndist:"),"same_query":sm("dist_same:"),"related_query":sm("dist_rel:"),"same_query_def":"passages the serve's own _know retrieval returned for this question that do not contain the answer","related_query_def":"answer-free passages the serve's _know retrieval returned for a related question (another question on the same article, or the same Wikidata property)","count_weights":{"answer_examples_0_1_2":RDK,"hedge_examples_1_2":[55,45]},"order":"gold and distractor passages shuffled per example (Wikidata line, when kept, stays first as in the serve)"},"gold":g("gold:"),"gold_position":g("goldpos:"),"wikidata_line":g("wdline:"),"reply":g("reply:"),"serve_retrieves":g("serve_retrieves:"),"passage_cap_bytes":g("cap:"),
 "rules":{"prompt":"User: [<Wikidata line>] <passage> [<passage> [<passage>]] <question>\\nAdam: <reply> via adam_t_client.template (history-free), loss on the reply only; passage lines are '<title>: <sentences>' rendered like adam_t_client._know (asc(_fix(text)), _trim around the best sentence) at the serve cap min(560, knowledge_budget // 2), or // 3 when an example carries three passages","gold":"nat = the serve's own _know(question) line that contains the answer (kept with p=%s when a re-grounded line also exists); con = re-grounded: the article(s) named by the dataset (SQuAD/AdversarialQA/TyDi title, HotpotQA supporting titles, Wikidata sitelink) are looked up in the live index (adam_t_client._pages), the passage containing the answer with the best overlap with the original dataset paragraph and the question is chosen, and it is trimmed around the answer sentence"%RDG,"replies":"span = the answer as written in the gold passage; sentence = the gold sentence containing the answer (5-20 words, not starting with a pronoun, no parenthesis or ' - ' heading/list join, not ending in an abbreviation, shares a question word; p=%s when available); wd_tpl = the wikidata source reply templates with the passage's surface form; hedge = one of %d own templates with the dataset answer as best guess for named entities, or a coarse date/number template (see coarse_hedges)"%(RDS,len(RH)),"hedge_target":RDH,"wikidata_line":"the serve's gated Wikidata line is kept with p=%s when _know produced one (never when it states the answer in a hedge example)"%RDW,"filters":"question <=250 chars, answer <=10 words, no yes/no answers, answer not already in the question, no 'passage/article/paragraph' questions, question not an eval/chat-suite line, Wikidata subjects whose name loses letters in ASCII folding (e.g. 'Dinh' -> 'inh') skipped; each composed example is pre-checked (13-gram non-exempt, 8-gram ADAM-T/NQ/D3/emergency, 6-gram chat suite, exact eval line, NF, identity) and recomposed on a hit; the main decontam pass runs again after"},
 "excluded":{"drop":"DROP questions presuppose one specific passage ('the game', 'the first quarter') that the serve never has, most answers are computed (not a span of any passage) and its paragraphs have no article title to re-ground on the live index","hotpot_yes_no":"yes/no answers cannot be span-verified"},
 "datasets":{x:{"hf":HF[x][0],"revision":HF[x][1],"files":HF[x][2]} for x in ("squad","adversarial_qa","hotpot","tydiqa")},"index":{k:rk.get(k) for k in ("version","built","source","n_pass","n_doc","license")},"client":{"file":"amni/serve/adam_t_client.py","sha1":hashlib.sha1(Path(AC.__file__).read_bytes()).hexdigest()},"generation_stats":{k:v for k,v in sorted(st.items()) if k.startswith("rd_")}}
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
   ok=ustore(uid,stmt,rel,v,"s1");ins=(k,rel) if ok else None;ok and (told.__setitem__((k,rel),v),H.insert(R.randint(0,len(H)),(stmt,[R.choice(ACKS),uack(k,rel,v)][1])))
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
 ES=DC.eval_sources();ix=DC.index(13,cache=None);XF=sorted({f for f in ES if DC.group(f) in ("adam_t",) or "rknow" in str(f) or "nq" in f.name or f.name=="d3.jsonl"}|set((ROOT/"tests"/"data"/"emergency").glob("*.tsv")));ix8=DC.index(8,files=XF,cache=None);ix6=DC.index(6,files=[CS],cache=None);ix6t=DC.index(6,files=[TDV],cache=None);IX[:]=[ix,ix8,ix6,ix6t]
 XS=[f for f in ES if isx(f)];XG=sorted({DC.group(f) for f in XS});assert len(XS)==len(EXEMPT) and not any(DC.group(f) in XG and not isx(f) for f in ES) and not any(isx(f) for f in XF) and CS.resolve() in XF,(XS,XG);XM[0]=np.uint32(sum(1<<ix.groups.index(g) for g in XG));EQ=lines([f for f in ES if not isx(f)]+XF)
 EQS.update(EQ);CQE.update(nrm(json.loads(l)["q"]) for l in open(CS) if l.strip() and json.loads(l)["cat"]=="emergency");return {"ix":ix,"ix8":ix8,"ix6":ix6,"ix6t":ix6t,"XF":XF,"XS":XS,"XG":XG,"EQ":EQ,"EQX":lines(XS)-EQ,"CQE":set(CQE)}
def rescan(p)->dict:
 K=checks();ix,ix8,ix6,ix6t=IX;B=Bpe(ROOT/"data"/"clean2"/"bpe4096.json");CQ=lines([CS]);SRC=list(LIC);o={"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"index_keys":{"n13":ix.key,"n8":ix8.key,"n6_chat_suite":ix6.key,"n6_toroid_dev":ix6t.key},"exempt_groups":K["XG"],"unit":"one decoded example (no grams across example boundaries)","splits":{}}
 for s in ("train","hold"):
  T=np.fromfile(OUT/f"{s}.bpe4096.u16",np.uint16);I=np.load(OUT/f"{s}.bpe4096.idx.npy");c=collections.Counter();x=collections.defaultdict(list)
  for off,n,si,_ in I.tolist():
   t=B.dec(T[off:off+n].tolist()).decode("ascii");ws=DC.words(t);us=[nrm(u) for u,_ in turns(t)];m13=DC._hits(ws,ix)[2];k=(m13&~XM[0])!=0
   f={**{"13:"+g:1 for g in gb(m13[k],ix)},**{"13x:"+g:1 for g in gb(m13[~k],ix)},**{"8:"+g:1 for g in gb(DC._hits(ws,ix8)[2],ix8)},"6:chat_suite":len(DC._hits(ws,ix6)[1])>0,"6:toroid_dev":len(DC._hits(ws,ix6t)[1])>0,"eval_line":any(u in K["EQ"] for u in us),"chat_suite_line":any(u in CQ for u in us),"emergency_copy":SRC[si] in ("emergency","emerg4") and any(q in u for q in K["CQE"] for u in us)}
   c.update({a for a,v in f.items() if v});c.update({f"src:{SRC[si]}:{a}" for a,v in f.items() if v and not a.startswith("13x:")});c["examples"]+=1;c[f"src:{SRC[si]}:examples"]+=1;[len(x[a])<3 and x[a].append({"src":SRC[si],"text":t[:300]}) for a,v in f.items() if v and not a.startswith("13x:")]
  o["splits"][s]={"counts":dict(sorted(c.items())),"examples":dict(x)};p(s,dict(sorted(c.items())))
 o["adam_t_and_chat_suite_hits"]=sum(v for q in o["splits"].values() for a,v in q["counts"].items() if a=="13:adam_t" or a.startswith("8:") or a in ("6:chat_suite","6:toroid_dev","chat_suite_line","emergency_copy"));o["non_exempt_hits"]=sum(v for q in o["splits"].values() for a,v in q["counts"].items() if a.startswith(("13:","8:","6:")) or a in ("eval_line","chat_suite_line","emergency_copy"))
 BS=set(json.load(open(OUT/"base_meta.json"))["sources"]) if (OUT/"base_meta.json").exists() else set();hk=lambda a:a.startswith(("13:","8:","6:")) or a in ("eval_line","chat_suite_line","emergency_copy");HP={"base":collections.Counter(),"new":collections.Counter()};[HP["base" if a.split(":")[1] in BS else "new"].update({a.split(":",2)[2]:v}) for q in o["splits"].values() for a,v in q["counts"].items() if a.startswith("src:") and hk(a.split(":",2)[2])];o["hits_by_part"]={k:dict(sorted(v.items())) for k,v in HP.items()};o["new_source_hits"]=sum(o["hits_by_part"]["new"].values());o["hits_by_part_note"]="base = sources copied byte for byte from the --base set (not regenerated, not re-checked by this build); new = sources generated by this build"
 (OUT/"logs").mkdir(exist_ok=True);json.dump(o,open(OUT/"logs"/"posthoc_scan.json","w"),indent=1);p("rescan","adam_t_and_chat_suite_hits",o["adam_t_and_chat_suite_hits"],"non_exempt_hits",o["non_exempt_hits"],"new_source_hits",o["new_source_hits"]);return o
CHS=("oasst1","oasst2","dolly","aya")
def rst(D,B,S)->dict:
 T=np.fromfile(D/"train.bpe4096.u16",np.uint16);M=np.fromfile(D/"train.bpe4096.mask.u8",np.uint8).astype(np.int8);I=np.load(D/"train.bpe4096.idx.npy");P3,P2,F,EL=collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter();W=collections.defaultdict(list);PS=collections.defaultdict(collections.Counter);FS=collections.defaultdict(collections.Counter);TS=("talk",)
 for off,n,si,_ in I.tolist():
  d=np.flatnonzero(np.diff(np.r_[0,M[off:off+n],0]))
  for a,z in zip(d[::2].tolist(),d[1::2].tolist()):r=B.dec(T[off+a:off+z].tolist()).decode("ascii").strip();w=r.split();k=[re.sub(r"[^a-z0-9']","",x.lower()) for x in w[:3]];P3[" ".join(k)]+=1;P2[" ".join(k[:2])]+=1;F[r]+=1;S[si] in TS and (PS[S[si]].update([" ".join(k)]),FS[S[si]].update([r]));W[S[si]].append(len(w));S[si].startswith("emerg") and EL.update([S[si]+(":lead911" if r.startswith(C9) else ":other")])
 A=[x for v in W.values() for x in v];H=[x for s in CHS for x in W.get(s,[])];nr=len(A);bk=((1,5),(6,10),(11,20),(21,40),(41,60),(61,100),(101,200),(201,10**6))
 q=lambda X:(lambda Y:{"n":len(X),"median":float(np.median(Y)),"mean":round(float(Y.mean()),1),"p90":float(np.percentile(Y,90)),"share_gt60":round(float((Y>60).mean()),4),"hist":{f"{a}-{b}" if b<10**6 else f">{a-1}":int(((Y>=a)&(Y<=b)).sum()) for a,b in bk}})(np.array(X)) if X else {}
 return {"dir":os.path.relpath(D,ROOT),"replies":nr,"top30_prefix3":[[k,v,round(100*v/nr,3)] for k,v in P3.most_common(30)],"top30_prefix2":[[k,v,round(100*v/nr,3)] for k,v in P2.most_common(30)],"top20_full_replies":[[k,v] for k,v in F.most_common(20)],"top30_prefix3_by_source":{s:[[k,v,round(100*v/sum(c.values()),3)] for k,v in c.most_common(30)] for s,c in PS.items()},"top20_full_replies_by_source":{s:[[k,v] for k,v in c.most_common(20)] for s,c in FS.items()},"reply_words_all":q(A),"reply_words_chat_sources":q(H),"reply_words_by_source":{s:q(v) for s,v in sorted(W.items())},"emergency_reply_lead":dict(EL),"prefix_def":"first 3 (or 2) whitespace words of each loss-bearing Adam reply, lowercased, non [a-z0-9'] chars removed; train split"}
def v4meta(st,cnt,V3,B)->dict:
 pc=collections.Counter();[pc.update({k.split(":",1)[1]:v}) for k,v in BC.items() if k[:2] in ("g:","G:","M:","r:","c:","f:")];ne={s:cnt[s]["train_examples"]+cnt[s]["hold_examples"] for s in LIC};tn=sum(ne.values())
 return {"what":"v5 = the v4 builder (736dd62) with every v4 source regenerated from the same seed, plus the new talk source (ADAM-T-SMALLTALK); the acks/brief/emergency sections below describe v4 rules that still apply","builder_base":"e227af6 (v3)","why":"first tuned checkpoint (data/adam_t/eval/chat/human40_adam-t-t30m-sft_2144.md): canned mem/umem acknowledgements dominated chat style, human-chat replies ran long, and the 911-first emergency lead was not learned (emergency ~0.7% of examples)",
 "acks":{"what":"memory-dialogue fact acknowledgements (scripts/v19_train_dialogues.py k1/k3/GACK/MACK/RACK/CACK and the UF->AF follow-up) are replaced in the builder, not in the shared generator: G.unit/G.runit/G.one are wrapped so the dialogue RNG stream (facts, values, questions, answers) is identical to v3 and only the Adam acknowledgement text changes, drawn from a separate RNG","contextual_share":AKP4,"forms":FM4,"pools":{"relations":{k:[len(x) for x in v] for k,v in AK.items()},"generic":len(GK4),"multi":len(MK4),"region":len(RK4),"correction":len(CK4),"follow_up":[len(AFP4),len(AFN4)],"remember":len(RM4)},"balanced_choice":"every pool is drawn near-uniformly (least-used within +2) and a whole-sentence template stops being drawn at ACK_CAP uses unless its whole pool is at the cap","cap":ACAP,"cap_rule":"a composed acknowledgement template (form + parts, values masked) is redrawn (up to 30 tries) once it has been used ACK_CAP times","max_composed_template_count":max(AT.values()) if AT else 0,"top_composed_templates":AT.most_common(15),"max_whole_sentence_count":max(pc.values()) if pc else 0,"top_whole_sentences":pc.most_common(15),"removed":["Ooh, {a} {V}! Nice ride.","Ooh, {V}. Nice pick.","Ooh, {V}. Now I'm hungry!","{V}, that's a great name.","Nice, thanks for the geography lesson!","Oh, cool! Thanks for telling me."],"umem":"in-session told-fact history turns use the same contextual acknowledgements (prompt-only); serve memory pre-turns keep the serve's fixed 'Got it, thanks for telling me.' text because they must be byte-identical to adam_t_client._mem"},
 "brief":{"rule":"oasst1/oasst2/dolly/aya replies over %d words: keep %d%% (md5 of the reply) intact when they are plain prose of <=%d words (no list, no ':', no 'here are'-style lead-in); otherwise trim to the first 1-3 sentences (stop once >=%d words, never over %d, a trailing caveat sentence is kept) after dropping a lone 'Sure!'-style opener; drop the reply when the kept text ends in a lead-in, only introduces a list/steps, cuts into a list, is followed by an unkept caveat, is under 6 words or the first sentence alone is too long; multi-turn threads end at a dropped reply (earlier turns kept) or right after a trimmed reply (later user turns may refer to the removed text)"%(BLIM,BKEEP,BMAX,BMIN,BLIM),"stats":{k:v for k,v in sorted(st.items()) if k.startswith("brief_")}},
 "emergency":{"examples":{s:ne[s] for s in ("emergency","emerg4")},"share_of_examples":round((ne["emergency"]+ne["emerg4"])/max(1,tn),4),"total_examples":tn,"emerg4_stats":{k:v for k,v in sorted(st.items()) if k.startswith("emerg4_")},"lead":"every emerg4 reply starts with '%s'; self-harm replies add calm supportive text and no hotline numbers (CEO: 911, not 988); the v3 'emergency' source (Law-0 table replies + 32 informational emergency-number questions) is unchanged"%C9},
 "talk":{"examples":ne["talk"],"train_hold":[cnt["talk"]["train_examples"],cnt["talk"]["hold_examples"]],"generator":"scripts/adam_t_talk.py","sources":"scripts/adam_t_talk/*.txt","license":"CC0 1.0 (own work, ADAM-T-SMALLTALK); attribution rows src=talk lic=cc0-own id=<scenario>:<n>","hold":"own split: whole scenarios held out (TALK_HOLD_FRAC of the holdable scenarios, lowest md5 of the scenario id; idiom and greeting scenarios are never held) instead of the %d permil example hash"%HOLD,"rules":{"reply":"1-2 sentences, specific reaction + optional follow-up question; no 'thanks for telling/sharing', no therapy-speak, no refusals, no emergency/911 wording (generator regex BAN, asserted per reply)","cap":"every Adam template part and every reaction+question pair is drawn at most TALK_CAP times; reply 3-word prefixes are capped at prefix_cap","prune":"before generation every template line (each slot value substituted) is checked alone; a line with a 13/8/6-gram hit or an exact eval user line is removed (listed in pruned_lines)","precheck":"each dialogue is dropped and redrawn on any 13-gram (non-exempt), 8-gram (ADAM-T/NQ/D3/emergency tests) or 6-gram (chat suite) hit or an exact eval user line; the main decontam pass runs again after"},**TKI,"stats":{k:v for k,v in sorted(st.items()) if k.startswith("talk_") and not k.startswith("talk_scen_")},"per_scenario":{k[10:]:v for k,v in sorted(st.items()) if k.startswith("talk_scen_")}},
 "reply_stats":{"prev":rst(V3,B,list(json.load(open(V3/"meta.json"))["sources"])) if (V3/"meta.json").exists() else {},"this":rst(OUT,B,list(LIC))}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fetch-only",action="store_true");ap.add_argument("--no-fetch",action="store_true");ap.add_argument("--only",default="");ap.add_argument("--rescan",action="store_true");ap.add_argument("--base",default=E("SFT_BASE",""));a=ap.parse_args();t0=time.time();p=lambda *x:print(time.strftime("%H:%M:%S"),*x,flush=True)
 if a.rescan:return rescan(p)
 mv=(OUT/"meta.json").exists() and json.load(open(OUT/"meta.json")).get("version",1)<VER;assert not (mv and VP.exists()),VP;assert not mv or subprocess.run(["systemctl","--user","is-active","adam-t-sft"],capture_output=True,text=True).stdout.strip()!="active","adam-t-sft is active and reads v3 from data/adam_t/sft: build into SFT_OUT=data/adam_t/sft_v4 instead";mv and OUT.rename(VP);mv and p("moved v3",OUT,"->",VP);V3=VP if (VP/"meta.json").exists() else OUT.parent/"sft"
 only=set(filter(None,a.only.split(",")));BD=Path(a.base) if a.base else None;BD=BD if not BD or BD.is_absolute() else ROOT/BD;BASE[0]=BD;BM=json.load(open(BD/"meta.json")) if BD else {};BS=list(BM.get("sources",{}));assert not BD or (only and BS==list(LIC)[:len(BS)] and BD.resolve()!=OUT.resolve()),("--base needs --only and a base whose sources are a prefix of LIC",BS)
 fl={} if a.no_fetch else fetch(sorted({x for n in only for x in {"reader":["squad","adversarial_qa","hotpot","tydiqa"],"mc":["csqa","qasc"],"calc_gsm8k":["gsm8k"],"know":["hotpot"]}.get(n,[n])}) if only else None);p("fetched",{k:len(v) for k,v in fl.items()})
 if a.fetch_only:return
 st=collections.Counter();R=random.Random(SEED);B=Bpe(ROOT/"data"/"clean2"/"bpe4096.json")
 L=[("chat5",lambda:chat5(st)),("mem",lambda:mem(st,B)),("emergency",lambda:emergency(R,st)),("squad",lambda:squadlike(R,"squad","squad")),("adversarial_qa",lambda:squadlike(R,"adversarial_qa","adversarial_qa")),("hotpot",lambda:hotpot(R)),("drop",lambda:drop(R)),("tydiqa",lambda:tydi(R)),("mintaka",lambda:mintaka(R)),("mc",lambda:mc(R)),("gsm8k",lambda:gsm(R)),("dolly_ctx",lambda:dolly_ctx(R,st)),("calc",lambda:calc(random.Random(SEED+10),st,dis(st))),("calc_gsm8k",lambda:calc_gsm(st)),("know",lambda:know(st,X)),("umem",lambda:umem(st,d3()[1])),("emerg4",lambda:emerg4(st)),("talk",lambda:talk(st)),("reader",lambda:reader(st,X)),("tmem",lambda:TM.tmem(st)),("cmem",lambda:TM.cmem(st))]
 K=checks();ix,ix8,ix6,ix6t=K["ix"],K["ix8"],K["ix6"],K["ix6t"];p("decontam",repr(ix),repr(ix8),repr(ix6),repr(ix6t),"exempt",K["XG"]);X=[];(not only or only&{"tmem","cmem"}) and TM.setup(sys.modules[__name__])
 for n,f in L:
  if only and n not in only:continue
  y=f();X+=y;p(n,len(y));assert not {e["s"] for e in y}&set(BS),("--base source regenerated",n)
 seen=set();keep=[];dc=collections.Counter();dx=collections.Counter();dg=collections.Counter();ex8=[]
 def chk(e,t):
  x="".join(f"User: {u}\nAdam: {v}\n\n" for u,v in t);w=DC.words(x);q=[nrm(y) for y in e["q"][:len(t)]+[u for u,_ in t]];h=[hits(w,k) for k in range(len(IX))]
  return ("contam13" if len(h[0][0]) else "contam8" if len(h[1][0]) else "chat6" if len(h[2][0]) else "tor6" if len(h[3][0]) else "eval_line" if any(y in K["EQ"] for y in q) else "nf" if e["s"] not in ("mem","emergency","umem","emerg4") and NF.clean(x)[0]!=x.strip() else "identity" if C5.IDN.search(x) else ""),h,w,any(y in K["EQX"] for y in q)
 for e in X:
  o=e["t"];e["t"]=[(a1(u),a1(v)) for u,v in e["t"] if a1(u) and a1(v)];e["c"] in XK and e["t"]!=o and st.update([e["s"]+"_serve_text_changed_by_a1"]);txt="".join(f"User: {u}\nAdam: {v}\n\n" for u,v in e["t"])
  if not e["t"]:dc[e["s"]+":empty"]+=1;continue
  k=hashlib.blake2b(nrm(txt).encode(),digest_size=12).digest()
  if k in seen:dc[e["s"]+":dup"]+=1;continue
  seen.add(k);why,h,ws,xl=chk(e,e["t"]);dg.update([f"13:{g}" for g in gb(h[0][1],ix)]+[f"13x:{g}" for g in gb(h[0][2],ix)]+[f"8:{g}" for g in gb(h[1][1],ix8)]+["6:chat_suite"]*(len(h[2][0])>0)+["6:toroid_dev"]*(len(h[3][0])>0));dx[e["s"]+":any_chat6"]+=len(h[2][0])>0;xw=not why and (len(h[0][2])>0 or xl);dx[e["s"]+":exempt_won"]+=xw;dx[e["s"]+":exempt_won_13"]+=xw and len(h[0][2])>0;dx[e["s"]+":exempt_won_line"]+=xw and xl
  if why:j={"contam13":(0,13),"contam8":(1,8),"chat6":(2,6),"tor6":(3,6)}.get(why);len(ex8)<60 and j and ex8.append({"src":e["s"],"why":why,"gram":" ".join(ws[int(h[j[0]][0][0]):][:j[1]])});pf=next((jj for jj in range(len(e["t"])-1,0,-1) if e["c"] not in ("mem",)+XK and why in ("contam13","contam8","chat6","tor6","eval_line") and not chk(e,e["t"][:jj])[0]),0);dc[e["s"]+(":salvaged_prefix_after_"+why if pf else ":"+why)]+=1;e["t"]=e["t"][:pf]
  if not why or e["t"]:keep.append(e)
 assert not [u for e in keep if e["s"] in ("emergency","emerg4") for t in e["t"] for u in t if any(c in nrm(u) for c in K["CQE"])],"emergency example copies a chat-suite emergency prompt"
 p("kept",len(keep),"dropped",sum(dc.values()));R2=random.Random(SEED+1);R2.shuffle(keep);sp={"train":[],"hold":[]}
 for e in keep:sp["hold" if (e["ho"] if "ho" in e else int(hashlib.md5((e.get("sk") or f"{e['s']}|{e['k']}").encode()).hexdigest(),16)%1000<HOLD) else "train"].append(e)
 SI={s:i for i,s in enumerate(LIC)};cnt=collections.defaultdict(collections.Counter);RD=collections.defaultdict(collections.Counter);BA=list(gzip.open(BD/"attribution.jsonl.gz","rt")) if BD else [];[cnt[b].update({k:v for k,v in BM["sources"][b].items() if k.startswith(("train_","hold_")) and isinstance(v,int)}) for b in BS];OUT.mkdir(parents=True,exist_ok=True);att=gzip.open(OUT/"attribution.jsonl.gz","wt");smp=[];NS=[s for s in LIC if LIC[s][2] in XK]+["talk"];nsm=collections.defaultdict(list);LN=collections.defaultdict(list);SH=collections.defaultdict(collections.Counter);PB=collections.defaultdict(list)
 for s,es in sp.items():
  I=[];M=[];ids=[];o=0
  if BD:ids,M,I=[np.fromfile(BD/f"{s}.bpe4096.u16",np.uint16)],[np.fromfile(BD/f"{s}.bpe4096.mask.u8",np.uint8)],[tuple(x) for x in np.load(BD/f"{s}.bpe4096.idx.npy").tolist()];o=len(ids[0]);[att.write(l) for l in BA if json.loads(l)["split"]==s];assert len(M[0])==o and sum(x[1] for x in I)==o
  for e in es:
   r=tok(B,e)
   if r is None:dc[e["s"]+":too_long"]+=1;continue
   t,m,nt=r;cnt[e["s"]][s+"_examples"]+=1;cnt[e["s"]][s+"_turns"]+=nt;cnt[e["s"]][s+"_tokens"]+=len(t);cnt[e["s"]][s+"_loss_tokens"]+=int(m.sum());cnt[e["s"]][s+"_bytes"]+=int(B.L[t.astype(np.int64)].sum())
   ids.append(t);M.append(m);I.append((o,len(t),SI[e["s"]],int(m.sum())));o+=len(t);att.write(json.dumps({"src":e["s"],"lic":LIC[e["s"]][0],"id":e["k"],"split":s})+"\n");s=="train" and len(smp)<50000 and smp.append((e,t,m));LN[e["s"]].append(len(t));"sh" in e and (SH[s].update([e["sh"]]),PB[e["s"]].append(e["pb"]),nt<len(e["t"]) and st.update([e["s"]+"_ctx_history_turns_trimmed_to_fit"]*(len(e["t"])-nt)));s=="train" and e["s"] in NS and len(nsm[e["s"]])<(4000 if e["s"] in ("talk","reader","tmem","cmem") else 400) and nsm[e["s"]].append((e,t,m));"rd" in e and RD[s].update(e["rd"])
  T=np.concatenate(ids);Mk=np.concatenate(M);T.tofile(OUT/f"{s}.bpe4096.u16");Mk.tofile(OUT/f"{s}.bpe4096.mask.u8");np.save(OUT/f"{s}.bpe4096.idx.npy",np.array(I,dtype=[("off","<u8"),("n","<u4"),("src","u1"),("loss","<u4")]));p(s,len(I),"examples",len(T),"tokens",int(Mk.sum()),"loss tokens")
 att.close();R3=random.Random(SEED+2);ss=R3.sample(smp,min(50,len(smp)));ns=[x for s in NS for x in R3.sample(nsm[s],min(20 if s in ("talk","reader","tmem","cmem") else 4,len(nsm[s])))]
 with open(OUT/"samples.txt","w") as f:
  for j,(e,t,m) in enumerate(ss+ns):
   seg=[];cur=None
   for i,mm in zip(t.tolist(),m.tolist()):
    (seg.append([mm,[i]]) if cur!=mm else seg[-1][1].append(i));cur=mm
   f.write((f"===== sample {j+1}/{len(ss)}" if j<len(ss) else f"===== talk sample {j-len(ss)+1} | scenario={e.get('sc')}" if e["s"]=="talk" else f"===== new-shape sample {j-len(ss)+1}/{len(ns)} | shape={e.get('sh')} | prompt_bytes={e.get('pb')}")+f" | src={e['s']} ({LIC[e['s']][0]}) | id={e['k']} | tokens={len(t)} loss={int(m.sum())}\n"+"".join(("[[" if mm else "")+B.dec(ii).decode("ascii")+("]]" if mm else "") for mm,ii in seg)+"\n")
 tot={s:sum(v[s+"_tokens"] for v in cnt.values()) for s in sp};REV={s:HfApi().dataset_info(C5.SRC[s]["hf"]).sha for s in ("oasst1","oasst2","dolly","aya") if cnt[s] and s not in BS};bs=lambda s:{k:v for k,v in cnt[s].items()}
 vp=json.load(open(V3/"meta.json")) if (V3/"meta.json").exists() else {};nz=lambda c:{k:v for k,v in sorted(c.items()) if v};RK=ROOT/"data"/"ks"/"index_rknow"/"wiki_v2"/"meta.json";rk=json.load(open(RK)) if RK.exists() else {};WS=WF.store()
 qd=lambda a,bk=((0,128),(128,256),(256,512),(512,768),(768,1024)):(lambda A:{"n":len(A),"min":int(A.min()),"p50":int(np.percentile(A,50)),"p90":int(np.percentile(A,90)),"p99":int(np.percentile(A,99)),"max":int(A.max()),"hist":{f"{lo+1}-{hi}":int(((A>lo)&(A<=hi)).sum()) for lo,hi in bk}})(np.array(a)) if a else {}
 meta={"name":f"ADAM-T SFT set v{VER}","version":VER,"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_s":round(time.time()-t0,1),"cmd":E("SFT_CMD"," ".join([sys.executable]+sys.argv)),"seed":SEED,"env":{k:E(k) for k in ("MAXA","SEED","HOLD_PERMIL","CTX","MEM_CTX","MAXP","MEM_N","LONGGAP","HOTPOT_RC","HOTPOT_CB","RC_CAPD","DROP_CAPD","HEDGE_HOTPOT","HEDGE_MINTAKA","HEDGE_MINTAKA_GENERIC","CALC_N","WD_N","KC_N","KC_UNSUP","WD_UNSUP","WD_SYN","UMEM_N","SERVE_BUDGET","WORKERS","SFT_OUT","SFT_RAW","ACK_CAP","AKP4","EMERG_N","BRIEF_LIM","BRIEF_KEEP","BRIEF_MAX","BRIEF_MIN","TALK_N","TALK_HOLD_FRAC","TALK_CAP","SFT_BASE","RD_SQUAD","RD_ADV","RD_TYDI","RD_HOTPOT","RD_WD","RD_HEDGE","RD_DW","RD_NAT","RD_WDL","RD_SENT","TMEM_N","CMEM_N") if E(k)},
 "template":{"format":"User: {user}\\nAdam: {reply}\\n\\n per exchange, examples concatenated with no extra separator; ASCII only; one line per turn","why":"byte-level role markers identical to the ADAM-T eval chat fmt ('User: {q}\\nAdam:'), the D3/D3-dev memory dialogues and chat5/chat6c; BPE-4096 has no spare ids (256 bytes + 3840 merges = 4096), so <|user|>-style specials would need a vocab resize (see proposal)","proposal":"optional later: extend vocab to 4099 with <|user|> <|adam|> <|end|> (embedding/head rows +3, one-off surgery at the SFT start); not done here","pretoken_boundaries":"'Adam:' always ends a pre-token and every reply starts with ' ', so the reply/prompt boundary is a token boundary (asserted for every example)","example_limit_tokens":CTX,"example_limit_why":"v3: raised from 512 to the model ctx 1024 (the trainer packs whole examples into CTX+1 windows, so an example of <=1024 tokens is never cut)","mem_generation_limit_tokens":MEMCTX,"mem_generation_why":"the v2 memory-dialogue generator keeps its 512-token retry limit so the mem source is identical to v2; long memory prompts come from the new umem source","multi_turn_truncation":"trailing exchanges dropped until the example fits in CTX tokens (new-shape examples: oldest history turn first, never the final exchange)"},
 "files":{"train":"train.bpe4096.u16 (uint16 token ids)","hold":"hold.bpe4096.u16","mask":"<split>.bpe4096.mask.u8: 1 per token, 1 = token belongs to an Adam reply (incl. the closing \\n\\n); put loss only where mask==1","index":"<split>.bpe4096.idx.npy: structured (off u8 token offset, n u4 tokens, src u1 = index into sources order, loss u4 masked tokens); every example starts with 'User:'","attribution":"attribution.jsonl.gz: one row per example (src, lic, id, split) in file order","samples":"samples.txt: 50 random train examples decoded from tokens, reply tokens wrapped in [[ ]], then 4 per new-shape source and 20 random talk train examples"},
 "tokenizer":{"bpe":"data/clean2/bpe4096.json","sha256":sha(ROOT/"data"/"clean2"/"bpe4096.json"),"encoder":"Python replica of crates/adam-t/src/bpe.rs on ASCII input (pattern restricted to ASCII classes, same merge order)","pattern":B.pat},
 "tokens":tot,"sources":{s:BM["sources"][s] if s in BS else {"license":LIC[s][0],"attribution":LIC[s][1],"kind":LIC[s][2],"use":LIC[s][3],"hf":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (C5.SRC.get(s,{}).get("hf"),))[0],"revision":(HF.get({"hotpot_rc":"hotpot","hotpot_cb":"hotpot"}.get(s,s)) or (None,REV.get(s)))[1],**bs(s)} for s in LIC},
 "share_train_tokens":{s:round(cnt[s]["train_tokens"]/max(1,tot["train"]),4) for s in LIC},"share_train_loss_tokens":{s:round(cnt[s]["train_loss_tokens"]/max(1,sum(v["train_loss_tokens"] for v in cnt.values())),4) for s in LIC},
 "raw_files":{**BM.get("raw_files",{}),**{n:[{"file":os.path.relpath(f,RAW),"sha256":sha(f)} for f in fs] for n,fs in fl.items()}},"mintaka":{"url":MK%"{split}","commit":"b6146ffb41562803c45df3c028b4cdad8c09dd4d"},
 "decontam":{"module":"amni.compute.decontam","n13":{"index_key":ix.key,"files":len(ix.files),"grams":int(len(ix.H)),"groups":ix.groups},"n8_adam_t":{"index_key":ix8.key,"files":ix8.files,"grams":int(len(ix8.H)),"groups":ix8.groups},"n6_chat_suite":{"index_key":ix6.key,"files":ix6.files,"grams":int(len(ix6.H)),"why":"docs/adam_t/eval_prereg.md chat section: most chat-suite prompts are shorter than the 13-gram window, so SFT builders also run decontam.index(n=6, files=[chat_src.jsonl]) with dirty()","dropped":{s:dc[s+":chat6"] for s in LIC if dc[s+":chat6"]},"salvaged_prefix":{s:dc[s+":salvaged_prefix_after_chat6"] for s in LIC if dc[s+":salvaged_prefix_after_chat6"]},"examples_with_any_hit":{k.split(":")[0]:v for k,v in dx.items() if k.endswith(":any_chat6") and v}},"n6_toroid_dev":{"index_key":ix6t.key,"files":ix6t.files,"grams":int(len(ix6t.H)),"why":"ADAM-T-SFT7: the tmem/cmem memory sources share the toroid dev set's format, so every new example is also checked for 6-gram overlap with data/adam_t/eval/toroid/toroid_dev.jsonl (docs/adam_t/user_toroid.md section 11); its questions are also in the exact-line set (eval_sources() covers data/adam_t/eval/**)","dropped":{s:dc[s+":tor6"] for s in LIC if dc[s+":tor6"]}},"exempt":{"files":list(EXEMPT),"groups":K["XG"],"reason":EXWHY,"scope":"this builder only: 13-gram hits whose only matching groups are exempt, exact user-line/question matches found only in exempt files, and memory-dialogue exact-prompt matches against the exempt hold_eval dialogues are ignored; distractor pairs for memory dialogues are filtered the same way","exact_lines_exempt_only":len(K["EQX"]),"won_back_examples":{k:v for k,v in nz(dx).items() if ":exempt_won" in k},"hits_13_exempt_only_by_group":{k[4:]:v for k,v in sorted(dg.items()) if k.startswith("13x:")},"distractor_pairs_kept":st["distractor_pairs_exempt13_kept"],"mem_kept_with_exempt_13_hit":st["mem_exempt13_kept"],"mem_kept_matching_exempt_hold_prompt":st["mem_exempt_eval_prompt_kept"]},"exact_lines":len(K["EQ"]),"mem_exact_line_scope":"v2: memory dialogue user lines are checked against the same non-exempt exact-line set as every other source (v1 used only the ADAM-T/NQ/D3/emergency subset because of the chat hold sets, now exempt)","rule":"drop an example on any 13-gram hit against a non-exempt group of decontam.eval_sources(), any 8-gram hit against the ADAM-T/NQ/D3/emergency-test files (incl. the chat suite), any 6-gram hit against the chat suite, or any user line/question equal (normalised) to a non-exempt eval string of >=3 words; multi-turn non-memory examples keep their longest clean prefix","emergency_guard":"4 emergency prompt templates that shared 6-grams with chat-suite emergency prompts were reworded (face drooping, whole bottle of pills, heart attack, left arm numb); the 6-gram check still applies to every emergency example, and the build asserts that no emergency example turn contains a chat-suite emergency prompt (normalised substring)","hits_by_group":dict(sorted(dg.items())),"examples":ex8},
 "drops":nz(dc),"stats":nz(st),"previous":{"version":vp.get("version"),"path":os.path.relpath(V3,ROOT),"moved_this_run":bool(mv),"prev_built":vp.get("built"),"prev_builder_commit":{3:"e227af6",4:"736dd62"}.get(vp.get("version")),"note":f"v{VER} is written to {os.path.relpath(OUT,ROOT)}; data/adam_t/sft (v3, the adam-t-sft default CORPUS), sft_v4, sft_v5 and sft_v6 are left untouched; point CORPUS at this dir to train on it"},**({"base":{"dir":os.path.relpath(BD,ROOT),"version":BM.get("version"),"built":BM.get("built"),"rule":f"--base: v{VER} files start with the base's train/hold tokens, masks, index rows and attribution rows byte for byte (same source ids); new sources are appended after them in each split; sources/tokens/shares cover the whole set, while decontam/drops/stats/new_shapes/length_tokens/prompt_bytes/samples describe this run's new examples only; the base's own meta.json is copied to base_meta.json","files_sha256":{f:sha(BD/f) for f in ("train.bpe4096.u16","train.bpe4096.mask.u8","train.bpe4096.idx.npy","hold.bpe4096.u16","hold.bpe4096.mask.u8","hold.bpe4096.idx.npy","attribution.jsonl.gz","meta.json")},"examples":{sp_:sum(BM["sources"][b][sp_+"_examples"] for b in BS) for sp_ in ("train","hold")},"tokens":BM.get("tokens")}} if BD else {}),"prev_vs_this":{"tokens":{"prev":vp.get("tokens"),"this":tot},"train_by_source":{s:{f"{k}_{x}":(vp.get("sources",{}).get(s,{}).get("train_"+x,0) if k=="prev" else cnt[s]["train_"+x]) for x in ("examples","tokens","loss_tokens") for k in ("prev","this")} for s in LIC}},**({k:v for k,v in BM.items() if re.fullmatch(r"v\d+",k)} if BD else {"v5":v4meta(st,cnt,V3,B),"v6":rdmeta(st,cnt,RD)}),f"v{VER}":TM.meta(st,cnt),
 "serve_context":{"client":"amni/serve/adam_t_client.py after 5720780 (Merge ADAM-T-RETRIEVE): _know(message,n,cap,history)->(lines,resolved query), follow-up resolution, gated Wikidata line '<label>, <property>: <answer>.' (_wl)","budget_bytes":BUD,"budget_why":"adam-t serve reports ctx = int(1024 symbols x 3.25) = 3328 bytes for BPE-4096; answer() reserves min(AMNI_ADAM_T_MAX_BYTES 400, 4 x max_new_tokens) bytes for the reply, so template() gets 3328 - 400 = 2928 bytes","order":"memory pre-turns ('User: <fact>\\nAdam: Got it, thanks for telling me.'), then history turns (newest kept first), then 'User: <rc items joined by one space> <question>\\nAdam:'; rc is ['Calculator: <engine reply>'] or ['<Wikidata label>: <answer>.', '<title>: <passage>', '<title>: <passage>']","budgets_bytes":{"memory":640,"math":240,"knowledge":1100,"passages":2,"passage":"min(560, knowledge // 2) = 550"},"exclusivity":"context(): the calculator line is offered only when memory gave no direct reply; knowledge is fetched only when there is no calculator line, with the turn's history (pronoun and 'what about X?' follow-ups are resolved against the last turn, personal follow-ups get none), and kept under a memory route only when the history rewrote the question; memory pre-turns only when the route fired. Memory pre-turns therefore never share a prompt with passages or a Wikidata line: v3 puts passages with history turns and memory pre-turns with history, exactly as the client assembles them","assembly":"the builder calls adam_t_client._math, clip, _know (history-free retrieval in a 2-process pool, cached in RAW/kx_cache_<client sha1>.jsonl; each kctx/wikidata row draws its history first and re-retrieves inline when adam_t_client._resolve rewrites the question with it, falling back to no history when that leaves no context), _fit, _mem and template itself and asserts every prompt parses back into its turns, so new-shape prompts are byte-identical to the flag-gated serve route","loss":"new-shape examples (calc, calc_gsm8k, wikidata, kctx, umem) put loss only on the final Adam reply; history, memory ACK turns and context are prompt-only","truncation":"a new-shape example over CTX tokens drops its oldest history turn first, then its oldest memory turn; the current turn and its context are never cut","env":{"AMNI_WIKIFACT_THR":WF.THR,"AMNI_WIKIFACT_TEMPORAL":WF.TEMPORAL,**{k:E(k) for k in ("AMNI_MATH","AMNI_MATH_MIN_CONF","AMNI_RKNOW_WF","AMNI_ADAM_T_KNOW_COV","AMNI_UMEM_LT","AMNI_UMEM_GN","AMNI_UMEM_G2","AMNI_RAY_HIST","AMNI_UMEM_CAP") if E(k)}},"a1_changed":{k:v for k,v in st.items() if k.endswith("_serve_text_changed_by_a1")}},
 "new_shapes":{"train":dict(sorted(SH["train"].items())),"hold":dict(sorted(SH["hold"].items())),"summary":{sp_:dict(sorted(sum((collections.Counter({":".join(k.split(":")[:2]):v}) for k,v in SH[sp_].items()),collections.Counter()).items())) for sp_ in ("train","hold")}},
 "length_tokens":{"all":qd([x for v in LN.values() for x in v]),"by_source":{s_:qd(LN[s_]) for s_ in LIC if LN[s_]},"note":"train+hold examples after tokenization"},"prompt_bytes":{s_:qd(PB[s_],((0,512),(512,1024),(1024,2048),(2048,BUD))) for s_ in LIC if PB[s_]},
 "calc_engine":{"mismatch_examples":CMM,"stats":{k:v for k,v in sorted(st.items()) if k.startswith(("calc_","gsm8k_calc"))}},
 "local_sources":{"wikifact":{"dir":os.path.relpath(WF.DIR,ROOT),"license":WS.man.get("license") if WS else None,"dump":WS.man.get("source") if WS else None,"dump_md5":WS.man.get("dump_md5_stream") if WS else None,"built":WS.man.get("built") if WS else None},"rknow":{"dir":"data/ks/index_rknow/wiki_v2","version":rk.get("version"),"source":rk.get("source"),"n_pass":rk.get("n_pass"),"n_doc":rk.get("n_doc")}},"rejected":[dict(zip(("dataset","card_license","reason","decision"),x)) for x in REJ],"licenses":sorted({v[0] for v in LIC.values()})}
 json.dump(meta,open(OUT/"meta.json","w"),indent=1);BD and json.dump(BM,open(OUT/"base_meta.json","w"),indent=1)
 with open(OUT/"attribution.md","w") as f:
  f.write(f"# ADAM-T SFT set v{VER}: attribution\n\nThis set mixes sources under the licences below. Rows from CC BY-SA sources (SQuAD, AdversarialQA, HotpotQA, DROP, Dolly, and the v3 `wikidata`/`kctx` rows, whose passages are English Wikipedia text retrieved by the R-KNOW wiki_v2 index) are share-alike: the set, and any redistribution of it, is under CC BY-SA 4.0 terms for those rows. `attribution.jsonl.gz` maps every example (file order) to its source and id.\n\n| source | licence | examples (train/hold) | tokens (train) | credit |\n|---|---|---|---|---|\n"+"".join(f"| {s} | {LIC[s][0]} | {cnt[s]['train_examples']}/{cnt[s]['hold_examples']} | {cnt[s]['train_tokens']} | {LIC[s][1]} |\n" for s in LIC)+"\nModel outputs of third-party LLMs are not used. Own work (`mem`, `emergency`, `calc`, `umem`, `emerg4`, `talk`, `tmem`, `cmem`) is CC0; Wikidata facts and lines (`wikidata`) are CC0 1.0 (wikidata-20260921 dump). Excluded and flagged datasets are listed in `meta.json` under `rejected`.\n")
 p("done",json.dumps({"tokens":tot,"drops":sum(dc.values())}))
if __name__=="__main__":main()
