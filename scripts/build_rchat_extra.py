import os,re,sys,json,gzip,html,time,collections,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault("HF_DATASETS_OFFLINE","1");os.environ.setdefault("HF_HUB_OFFLINE","1")
from amni.compute import rchat as R,decontam as DC
from scripts.neutral_filter import DEFAULT_FILTER as NF
from scripts.build_chat5_corpus import asc,BADL
from scripts.ks_ingest import EMAIL,PHONE
SRC=ROOT/"data"/"rchat"/"src";OUT=ROOT/"data"/"rchat"/"extra.jsonl.gz";MAN=ROOT/"data"/"rchat"/"extra_manifest.json";SEED=ROOT/"amni"/"compute"/"rchat_seed.jsonl";CUT="2022-11-30";DUMP="stackexchange_20230308"
SITES={"interpersonal":"Interpersonal Skills","parenting":"Parenting","workplace":"The Workplace","pets":"Pets","lifehacks":"Lifehacks","fitness":"Physical Fitness","cooking":"Seasoned Advice","gardening":"Gardening & Landscaping"}
LICENSES={"se":{"name":"Stack Exchange network, official data dump %s (archive.org/details/%s), sites: %s"%(DUMP[-8:],DUMP,", ".join(k+".stackexchange.com" for k in SITES)),"license":"CC BY-SA 3.0 / CC BY-SA 4.0 (per post ContentLicense; commercial use allowed with attribution + share-alike)","terms":"dump license.txt (cc-wiki): attribute site, link the question/answer, name and link the author; pre-2024 archive.org dump, no clickwrap; posts created before %s only (pre-ChatGPT)"%CUT,"attribution":"per reply: site, answer URL, author name + profile URL, post license"},
"topical":{"name":"Topical-Chat (Amazon Alexa AI), github.com/alexa/Topical-Chat conversations/*.json","license":"CDLA-Sharing-1.0 (DATALICENSE in the repo; commercial use allowed, share-alike on published Data)","terms":"human-human MTurk conversations collected 2019; reading sets (Reddit/Wikipedia/Washington Post) not used","attribution":"per reply: Topical-Chat, Amazon, CDLA-Sharing-1.0"},
"oasst1":{"name":"OpenAssistant/oasst1 extra branches of chat5 train trees","license":"Apache-2.0","terms":"same filters as chat5 (human, reviewed, not deleted, label/detoxify gates)","attribution":"not required"},
"oasst2":{"name":"OpenAssistant/oasst2 extra branches of chat5 train trees","license":"Apache-2.0","terms":"same filters as chat5","attribution":"not required"},
"own":{"name":"Amni everyday reply seeds (amni/compute/rchat_seed.jsonl)","license":"CC0-1.0","terms":"written in-house for Amni by Claude subagents from a situation brief only (no dataset, eval or web text shown); released CC0","attribution":"not required"}}
REJECTED=[("Wikipedia talk pages","CC BY-SA","argumentative, off-tone for chat","not used"),("Project Gutenberg plays/dialogue","public domain (US)","dated tone; template-only value, covered by own seeds","not used"),("CC-BY forums","varies","no forum found with a clean, site-wide CC BY licence and dumps","not used"),("common-pile/stackexchange (HF, cached)","per-doc CC BY-SA","built from Dec-2024 community dumps + July-2024 dumps (new export tool, clickwrap terms flagged by QA)","not used; 2023-03 archive.org dump used instead"),("Stack Exchange dumps after 2024-06","CC BY-SA + clickwrap","clickwrap download terms flagged for counsel","not used"),("Stack Exchange posts created on/after %s"%CUT,"CC BY-SA","may contain LLM-generated answers","dropped"),("Microsoft Personality Chat (qnamaker chit-chat)","MIT repo licence, data licence not stated","licence of the data files not separately stated; bot-persona replies","not used"),("ParlAI blended_skill_talk / PersonaChat / Wizard of Wikipedia","unclear on cards","licence unclear","not used"),("DailyDialog, EmpatheticDialogues, ESConv, Commonsense-Dialogues, OpenDialKG, SAMSum, DialogSum, LIMA, no_robots","NC / ND","non-commercial","not used"),("SODA, UltraChat, ShareGPT, WildChat, smoltalk, OpenHermes, Synthetic-Persona-Chat, HH-RLHF","various","model-generated dialogue","not used"),("Yahoo Answers, Reddit, Grokipedia, data/big2, big3, chat1-chat4e","unclear / excluded by policy","excluded by policy","not used")]
TAG=re.compile(r"<[^>]+>");BLK=re.compile(r"(?is)<(blockquote|pre|code|table|img|hr)\b.*?(</\1>|/?>)");SEBAD=re.compile(r"(?i)\b(OP|the question|your question|this question|other answers?|this answer|my answer|comments?|downvot\w*|upvot\w*|stack ?exchange|this site|IPS|edit|update|tl;?dr|disclaimer|footnote|see also|as mentioned|as others|source|citation|reference|wikipedia|study|studies|research)\b|@|\[|https?:|www\.|\d{3,}")
AUTO=re.compile(r"(?i)\b(my (wife|husband|partner|son|daughter|kids?|children|sister|brother|mom|mother|dad|father|parents|family|boss|manager|team|company|coworkers?|colleagues?|friends?|girlfriend|boyfriend|dogs?|cats?|experience|job|office|employer|house|garden|plants?|horse|puppy|kitten|toddler|baby)|i'?m a|i am a|as a (parent|manager|mother|father|mom|dad|teacher|nurse|developer|lead|boss|vet|chef|trainer|gardener)|when i was|i have (been|had|worked|raised|owned|seen)|i'?ve (been|had|worked|raised|owned|seen|done)|i (had|used to|once|remember|worked|grew|raised|own|owned)|in my (experience|case|house|garden|family|opinion as))\b");POL=re.compile(r"(?i)\b(politic\w*|president\w*|election\w*|vote\w*|voting|government|congress|senat\w*|democrat\w*|republican\w*|liberal|conservative|party|parties|trump|obama|clinton|biden|tax\w*|debt|deficit|news|fox|cnn|war|military|religio\w*|church|god|abortion|immigra\w*)\b")
def txt(h:str)->str:return re.sub(r"\s+"," ",html.unescape(TAG.sub(" ",BLK.sub(" ",re.sub(r"(?i)<li>","<li>- ",h or ""))))).strip()
def paras(h:str,n:int=600)->str:
 o=""
 for p in re.findall(r"(?is)<p>(.*?)</p>",BLK.sub(" ",h or "")):
  p=asc(txt(p))
  if not p:continue
  if o and len(o)+len(p)>n:break
  o=(o+" "+p).strip()
 return o
def se_site(site:str,st:collections.Counter)->list:
 d=SRC/"se"/site;U={};Q={};A=collections.defaultdict(list)
 for _,e in ET.iterparse(d/"Users.xml"):e.tag=="row" and U.__setitem__(e.get("Id"),e.get("DisplayName") or "");e.clear()
 for _,e in ET.iterparse(d/"Posts.xml"):
  if e.tag=="row":
   g=e.get;t=g("PostTypeId");ok=(g("CreationDate") or "9")<CUT and not g("ClosedDate") and not g("CommunityOwnedDate")
   t=="1" and ok and int(g("Score") or 0)>=2 and Q.__setitem__(g("Id"),(asc(html.unescape(g("Title") or "")),paras(g("Body"),260),g("AcceptedAnswerId")))
   t=="2" and ok and A[g("ParentId")].append((int(g("Score") or 0),g("Id"),g("Body"),g("ContentLicense") or "CC BY-SA 4.0",g("OwnerUserId"),g("OwnerDisplayName")))
  e.clear()
 out=[];host=site+".stackexchange.com"
 for qi,(ti,bd,acc) in Q.items():
  st["se_q"]+=1
  if not 3<=len(R.toks(ti))<=25 or SEBAD.search(ti) or re.search(r"(?<!^)\b(?!I\b|I'm\b|I've\b|I'd\b)[A-Z][A-Za-z]+",ti):st["se_q_drop"]+=1;continue
  aa=sorted(A.get(qi,[]),key=lambda x:(x[1]!=acc,-x[0]))
  for sc,ai,body,lic,ou,od in [x for x in aa if x[0]>=3 or (x[1]==acc and x[0]>=1)][:2]:
   a=paras(body);s=[x for x in R.SENT.split(a) if x.strip()][:3];a2=" ".join(s)
   if not s or len(R.toks(a2))<6 or SEBAD.search(a2) or AUTO.search(a2) or '"' in a2 or re.search(r"\d|\b[A-Z]{3,}\b",a2) or re.search(r"(?<![.!?]\s)(?<!^)\b(?!I\b|I'm\b|I've\b|I'd\b|I'll\b)[A-Z][a-z]+",a2) or s[0].rstrip().endswith("?") or re.match(r"(?i)(i (cannot|can'?t|don'?t know|do not know)|i'?m not sure|it depends|i (think|would|'d) (start|begin) by|first,? (let|i)|to answer|short answer|long answer|you ask|i'?m not sure (what|if) you)",a2):st["se_a_drop"]+=1;continue
   nm=re.sub(r"[()]","",U.get(ou,od or "") or "anonymous").strip() or "anonymous";pu=f"https://{host}/users/{ou}" if ou and ou in U else ""
   out.append({"src":"se","id":f"se:{site}:{ai}","lic":lic,"att":f"Stack Exchange, {lic}: answer by {nm}{' '+pu if pu else ''}, https://{host}/a/{ai}","ex":[[ti,a]],"ctx":bd});st["se_a_"+site]+=1
 return out
def topical(st:collections.Counter)->list:
 out=[];OPQ=re.compile(r"(?i)\b(do you (like|love|enjoy|prefer|watch|listen|play|read|follow|have a favou?rite|ever)|what do you think|have you (ever|seen|heard|read|been)|are you (a fan|into)|what('?s| is) your favou?rite|how about you|what about you)\b");FACT=re.compile(r"(?i)did you know|apparently|i read|i heard|article|fun fact|interesting fact|according|wikipedia|million|billion|percent|\bwas (born|founded|created|invented)")
 for sp in ("train","valid_freq","valid_rare","test_freq","test_rare"):
  for k,c in json.load(open(SRC/"topical"/f"{sp}.json")).items():
   m=[asc(re.sub(r"\s+"," ",t["message"])).strip() for t in c["content"]];rt=set(c.get("conversation_rating",{}).values())
   if not rt<= {"Good","Excellent"}:st["tc_conv_drop"]+=1;continue
   for i in range(len(m)-1):
    qs=[x for x in R.SENT.split(m[i]) if OPQ.search(x) and x.rstrip().endswith("?")];q=qs[-1].strip() if qs else "";a=m[i+1]
    if not q or not 2<=len(R.toks(q))<=20 or not 4<=len(R.toks(a))<=45 or re.search(r"\d",a) or FACT.search(a) or POL.search(q+" "+a) or any(re.search(r"(?<![.!?]\s)(?<!^)\b(?!I\b|I'm\b|I've\b|I'd\b|I'll\b)[A-Z][A-Za-z]+",x) for x in (q,a)) or R.cover(q,a)<0.3:st["tc_drop"]+=1;continue
    out.append({"src":"topical","id":f"topical:{k}:{i}","lic":"CDLA-Sharing-1.0","att":"Topical-Chat, Amazon, CDLA-Sharing-1.0","ex":[[q,a]],"ctx":""});st["tc_kept"]+=1
 return out
def oasst_extra(st:collections.Counter)->list:
 from datasets import load_dataset
 att={};[att.setdefault(r["id"],r["split"]) for r in map(json.loads,gzip.open(ROOT/"data"/"chat5"/"attribution.jsonl.gz","rt")) if r["src"]!="gen"]
 have=set(re.findall(r"User: ([^\n]*)\nAdam: ([^\n]*)\n\n",(ROOT/"data"/"chat5"/"train.bin").read_bytes().decode("ascii")));hq={q for q,_ in have};ms={};sr={}
 for s in ("oasst1","oasst2"):
  for sp in ("train","validation"):
   for r in load_dataset("OpenAssistant/"+s,split=sp).select_columns(["message_id","parent_id","text","role","lang","rank","deleted","review_result","synthetic","model_name","message_tree_id","tree_state","labels","detoxify"]):r["message_id"] in ms or (ms.__setitem__(r["message_id"],r),sr.__setitem__(r["message_id"],s))
 bad=lambda r:r["deleted"] or r["review_result"] is not True or r["synthetic"] or bool(r["model_name"]) or r["tree_state"]!="ready_for_export" or r["lang"]!="en" or any(n in BADL and v>.5 for n,v in zip((r["labels"] or {}).get("name",[]),(r["labels"] or {}).get("value",[]))) or ((r["detoxify"] or {}).get("toxicity") or 0)>.5
 from scripts.build_chat_corpus import clean,short
 T=lambda t,a=False:re.sub(r"(?i)open[- ]?assistant","Adam",(short if a else clean)(asc(t)).replace("\n"," ")).strip();ch=collections.defaultdict(list);[ch[m["parent_id"]].append(m) for m in ms.values() if m["parent_id"]];out=[]
 for m in ms.values():
  if m["role"]!="prompter" or att.get(m["message_tree_id"])!="train" or bad(m):continue
  kids=sorted([k for k in ch[m["message_id"]] if k["role"]=="assistant" and not bad(k)],key=lambda k:(k["rank"] is None,k["rank"] or 0))
  if not kids:continue
  q=T(m["text"]);a=T(kids[0]["text"],True);p=ms.get(m["parent_id"]) if m["parent_id"] else None;pp=ms.get(p["parent_id"]) if p and p["parent_id"] else None
  if (q,a) in have or q in hq or R.act(q) not in R.CHATTY or not 2<=len(R.toks(q))<=60:st["oa_skip"]+=1;continue
  if p and (not pp or bad(p) or bad(pp)):st["oa_ctx_bad"]+=1;continue
  out.append({"src":sr[m["message_id"]],"id":m["message_tree_id"]+":"+m["message_id"],"lic":"Apache-2.0","att":"","ex":([[T(pp["text"]),T(p["text"],True)]] if p else [])+[[q,a]],"ctx":""});st["oa_kept"]+=1
 return out
def seeds(st:collections.Counter)->list:
 out=[]
 for l in SEED.read_text().splitlines():
  r=json.loads(l)
  for j,a in enumerate(r["replies"]):
   for i,q in enumerate(r["prompts"]):out.append({"src":"own","id":f"own:{r['sit']}:{j}:{i}","lic":"CC0-1.0","att":"","ex":[[q,a]],"ctx":""});st["own_units"]+=1
 return out
def main():
 t0=time.time();st=collections.Counter();rows=seeds(st)+topical(st)+[r for s in SITES for r in se_site(s,st)]+oasst_extra(st)
 ev=[ROOT/"data"/"rchat"/"eval"/f for f in ("conv_eval.jsonl","conv_dev.jsonl")];em=sorted((ROOT/"tests"/"data"/"emergency").glob("*.tsv"));ix13=DC.index();ix8=DC.index(8,files=ev+em,cache=ROOT/"data"/"rchat"/".dc")
 NRM=lambda s:re.sub(r"[^a-z0-9]+"," ",s.lower()).strip();EQ={NRM(x) for f in ev for l in open(f) for r in [json.loads(l)] for x in [r["prompt"]]+[u for u,_ in r["history"]]}|{NRM(l.split("\t")[0]) for f in em for l in open(f,errors="replace") if l.strip()}
 keep=[];seen=set()
 for r in rows:
  ex=[[re.sub(r"\s+"," ",q).strip(),re.sub(r"\s+"," ",a).strip()] for q,a in r["ex"]];full=" ".join(q+" "+a for q,a in ex)+" "+r["ctx"];k=(r["src"],NRM(ex[-1][0]),NRM(ex[-1][1]))
  why="dup" if k in seen else "pii" if EMAIL.search(full) or PHONE.search(full) else "neutral" if any(NF.clean(x)[0]!=x for q,a in ex for x in (q,a)) or (r["ctx"] and NF.clean(r["ctx"])[0]!=r["ctx"]) else "unsafe" if not all(R.safe(q,True) and R.safe(a) for q,a in ex) or (r["ctx"] and not R.safe(r["ctx"],True)) else "eval_prompt" if any(NRM(q) in EQ for q,_ in ex) else "decontam13" if DC.dirty(full,ix=ix13) else "decontam8" if DC.dirty(full,ix=ix8) else ""
  st[r["src"]+"_"+(why or "kept")]+=1;seen.add(k);why or keep.append(dict(r,ex=ex))
 with gzip.open(str(OUT)+".part","wt") as f:[f.write(json.dumps(r)+"\n") for r in keep]
 os.replace(str(OUT)+".part",OUT);by=collections.Counter(r["src"] for r in keep)
 m={"built":time.strftime("%Y-%m-%dT%H:%M:%S"),"s":round(time.time()-t0,1),"docs":len(keep),"by_src":dict(by),"stats":dict(sorted(st.items())),"licenses":LICENSES,"rejected":[dict(zip(("source","license","reason","decision"),x)) for x in REJECTED],"filters":"neutral_filter DEFAULT_FILTER.clean must leave every text unchanged; rchat.safe (harm keywords, unsafe words); email/phone drop; exact prompt match vs rchat eval/dev + emergency sets; decontam 13-gram vs every registered eval/hold set (%d files, incl. data/rchat/eval) and 8-gram vs rchat eval/dev + tests/data/emergency/*.tsv"%len(ix13.files),"decontam":{"ix13":ix13.key[:16],"ix8":ix8.key[:16],"ix13_files":len(ix13.files),"ix8_files":[DC.rel(f) for f in ev+em]}}
 MAN.write_text(json.dumps(m,indent=1));print(json.dumps({k:v for k,v in m.items() if k not in ("licenses","rejected")},indent=1))
if __name__=="__main__":main()
