import os,re,sys,json,gzip,time,math,random,zlib,argparse,tempfile,resource,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import rchat as R,decontam as DC,eval_lock as EL
E=ROOT/"data"/"rchat"/"eval";EVAL=E/"conv_eval.jsonl";DEV=E/"conv_dev.jsonl";LOCK=E/"lock.json";HOLD=E/"conv_hold.jsonl";HLOCK=E/"lock_hold.json";TRAIN=ROOT/"data"/"chat5"/"train.bin"
OWN="""hey! how's your weekend going so far?
I just got back from a week in Portugal.
Do you have a favorite season?
What's something that always cheers you up?
I can't decide what to cook for dinner tonight.
It's finally sunny here after a whole week of rain!
I started learning guitar last month.
Any fun ideas for a lazy Sunday?
I'm so bored today.
Coffee or tea, what's your pick?
What do you like to do for fun?
I just watched a really good movie last night.
My cat keeps knocking things off the table, it's driving me nuts.
Good morning!
I think I'm finally getting the hang of sourdough.
How do I stop procrastinating on my assignments?
What's a good way to make friends as an adult?
Should I buy a new laptop or repair my old one?
How can I get better sleep when my mind won't stop racing?
Any tips for saving money on groceries?
How do I tell my boss I'm overwhelmed without sounding like I'm complaining?
What's the best way to start running if I've never done it before?
How can I be more confident when speaking in meetings?
Is it rude to leave a party early without saying goodbye?
How do I keep houseplants alive? I kill every one I buy.
What should I bring to a potluck?
How can I stay motivated to exercise in winter?
What do you think about pineapple on pizza?
Do you like music?
Would you rather live in the mountains or by the beach?
What's your opinion on working from home?
Is it better to rent or buy a house?
Do you think people are too attached to their phones?
What's your favorite book?
Are cats better than dogs?
I'm really stressed about my exams next week.
I feel lonely since I moved to a new city.
I got the promotion I was hoping for!
I'm so tired of feeling like I'm behind everyone else.
I've been feeling kind of down lately.
I just adopted a puppy and I'm so excited!
I'm nervous about my first day at a new job tomorrow.
My best friend is moving away and I'm really sad about it.
I finally finished my first marathon today!
I'm frustrated because nothing I cook turns out right.
My sister Priya just moved to Chicago for a new job. || That's a big move! How is she settling in? || She says she's lonely. How can I help her feel less alone from far away?
My dad turns 60 next week. || Nice milestone! Are you planning anything? || I want to surprise him but he hates parties. Any ideas?
I've been friends with Jake since high school. || That's a long friendship! || Lately he cancels every plan we make. Should I say something to him?
My daughter Emma starts kindergarten on Monday. || Exciting! How is she feeling about it? || She's nervous and honestly so am I. How do we make the first day easier?
My coworker Tom keeps taking credit for my work. || That's frustrating. Has it happened often? || Yes, three times this month. How should I bring it up with him?
My grandma is teaching me to bake bread. || That sounds lovely! || Mine keeps coming out dense though. What am I doing wrong?
I adopted a rescue dog named Biscuit. || Congrats! How is Biscuit doing? || He's scared of everything, even the vacuum. How can I help him feel safe?
My girlfriend and I have been together two years. || Congratulations! || Our anniversary is on Friday. || Any plans yet? || Not yet. What's a thoughtful idea that isn't too expensive?
My best friend Lisa is getting married in June. || How exciting! Are you in the wedding? || I'm her maid of honor and I have to give a speech. Any tips?
My son failed his math test. || Oh no, how is he taking it? || He's really upset and says he's just bad at math. What should I tell him?
My roommate Alex never does the dishes. || Ugh, that gets old fast. || I've asked him twice already. What else can I try?
I'm training for a half marathon. || Nice! When is the race? || In eight weeks. || Plenty of time to build up. || My knees hurt after long runs though. Should I be worried?
My mom has been really forgetful lately. || That must be worrying. || Should I talk to her about seeing a doctor?
I met someone named Sophie at a friend's party. || Oh nice! Did you two hit it off? || I think so. Is it too soon to ask her out for coffee?
My brother Mark is visiting this weekend. || Fun! Do you have plans? || He loves hiking but I'm not very fit. What could we do together?
I just started a new job as a nurse. || Congratulations! How's it going so far? || The night shifts are exhausting. How do people cope with them?
My friend Carlos lost his job last week. || I'm sorry to hear that. How is he holding up? || Not great. What can I say to him that doesn't sound fake?
We're moving to a new house next month. || Exciting! Where to? || Just across town, but the kids will change schools. || That's a big change for them. || How can we make it easier on them?
I've been learning Spanish on my own. || That's great! How long have you been at it? || About six months, but I freeze when I try to speak. Any advice?
My aunt Maria sent me a really nice gift. || That's sweet of her! || What's a good way to thank her besides just texting?
I have a job interview on Thursday. || Good luck! What's the role? || A marketing position. What questions should I ask them at the end?
My husband wants to get a dog but I'm not sure. || What's holding you back? || We both work full time. Is it fair to the dog?
What's a good plant for a beginner? || A pothos is a great choice; it's hard to kill and grows in low light. || What about something that flowers?
Should I learn to cook or just get meal kits? || Learning to cook pays off long term, and meal kits can be a good bridge. || Why is cooking better in the long run?
I want to start journaling. || Great habit! Do you want prompts or free writing? || Prompts, I think. What should I write about first?
What's the most relaxing way to spend an evening?
I just finished a really long week at work.
Tell me something that made you smile recently.
Do you ever get tired of answering questions?
I'm thinking about getting a tattoo.
I tried yoga for the first time today.
What would you do with a free afternoon?
I can't stop listening to the same song on repeat.
Is it weird that I like rainy days more than sunny ones?
My neighbor's dog barks all night and I can't sleep.
How do I say no to people without feeling guilty?
Should I text my ex happy birthday?
How do I get back into reading when my attention span is shot?
What's a good gift for someone who has everything?
How do I deal with a friend who always cancels at the last minute?
Is it okay to take a mental health day from work?
What's the best way to learn to cook without wasting a lot of food?
How can I make my small apartment feel cozier?
How do I stop comparing myself to people on social media?
Should I tell my friend her boyfriend is rude to her?
Do you think money can buy happiness?
What's better for a first date, dinner or a walk in the park?
Is it worth learning a musical instrument as an adult?
Do you prefer mornings or evenings?
What's your take on New Year's resolutions?
Would you rather be able to fly or be invisible?
I'm anxious about a presentation I have to give on Monday.
I had a huge fight with my best friend and I don't know what to do.
I passed my driving test!
I'm homesick and I miss my family a lot.
I feel like nobody listens to me at work.
I'm really proud of myself, I cooked a full dinner from scratch!
Thanks, that was really helpful!
Bye, talk to you tomorrow!
How's your day been?
My friend Nina keeps asking to borrow money. || That can get awkward. How much are we talking? || Small amounts, but it adds up. How do I tell her no without hurting her?
My brother Leo is starting college in the fall. || That's exciting! What is he studying? || Engineering. He's nervous about living away from home. What should I tell him?
My mom just retired after thirty years of teaching. || Wow, what a career! How is she feeling? || She's a bit lost without a routine. How can I help her find something new?
I've been dating Ryan for three months. || That's nice! How is it going? || Really well, but he hasn't met my friends yet. Is it too early to introduce him?
My grandpa is turning 90. || That's amazing! || He loves fishing and old westerns. What's a good gift for him?
My coworker Hannah just had a baby. || How lovely! || What's a nice gesture from the team that isn't over the top?
My little sister is being bullied at school. || I'm sorry, that's really hard. Does she talk to you about it? || Sometimes. How can I support her?
My friend Omar is running his first marathon on Sunday. || That's a big deal! || What's a fun way to cheer him on?
My partner snores really loudly. || Oh no, that must make sleeping tough. || Is there anything we can try besides separate rooms?
I've been playing piano for a year. || Nice! How are you finding it? || I'm stuck on the same songs. How do I get to the next level?
My dog Pepper is getting old. || How old is she? || Thirteen. She's slowing down a lot. How can I keep her comfortable?
My best friend Chloe forgot my birthday. || Oh, that stings. || Should I say something to her or let it go?
My dad keeps sending me news articles at 6am. || Ha, early bird! || It's sweet but it wakes me up. How do I bring it up nicely with him?
I'm planning a trip with my friends. || Fun! Where are you going? || We can't agree. Two want the beach and two want the mountains. How do we decide?
I just moved in with my girlfriend. || Congratulations! How's it going? || Good, but we argue about chores. Any tips?
I got a new plant for my desk. || Nice! What kind? || A succulent. How often should I water it?
I want to get fit this year. || Great goal! What are you thinking of doing? || Maybe the gym, but it intimidates me. How do I start?
I've been feeling burned out. || I'm sorry. Is it mostly work? || Yeah. I don't even enjoy my hobbies anymore. What can I do?
I'm hosting Thanksgiving for the first time. || How exciting! How many people? || Twelve. I'm panicking a little. Any tips?
My son wants a phone but he's only nine. || That's a common debate. || Is nine too young?
Do you like movies? || I do! What kind do you enjoy? || Mostly comedies. Any recommendations for a movie night with friends?
I started a book club with my coworkers. || That's a great idea! || What should we read first?
I'm learning to bake. || Fun! What have you made so far? || Just cookies. What should I try next?
I've been trying to wake up earlier. || How is it going? || Terrible, I keep hitting snooze. Any tricks?
My friend Sam says I'm always late. || Is that true? || Kind of. How do I stop being late all the time?
My sister and I haven't talked in months. || I'm sorry to hear that. Did something happen? || We argued about money. How do I reach out to her?
I want to surprise my wife for our anniversary. || Lovely! How long have you been married? || Ten years. What's something special I could do for her?""".splitlines()
OWN_DEV="""hi there, how's it going?
I just got a kitten!
What's a good hobby I can do at home?
Should I go back to school at 30?
Do you like the ocean?
I'm worried I'm not good enough at my job.
I've been cooking more at home lately.
My friend Elena is moving to Berlin. || Oh wow, big move! || I'm going to miss her so much. How do we stay close?
My dad is learning to use a smartphone. || That's great! How is it going? || He gets frustrated fast. How can I help him without making him feel dumb?
I want to run a 5k. || Awesome goal! || I get winded after two minutes. How should I train?
What do you think about camping?
Any tips for a long flight?
I got accepted into my dream school!
My roommate keeps eating my food. || That's annoying! || How do I bring it up with her?
How do I make my mornings less rushed?
thanks!
I'm feeling really unmotivated today.
Is it okay to not want kids?
My brother Dan borrowed my car and scratched it. || Oh no. Did he tell you? || No, I found it myself. How should I talk to him about it?
What's a good way to relax after work?""".splitlines()
BAD=re.compile(r"(?i)chat-?gpt|gpt-?\d|openai|\bllama\b|\bbard\b|\bbing\b|\bgemini\b|\bclaude\b|anthropic|```|\bdef \w+\(|[{};<>]{2}|\\[a-z]+|\$\$|https?://")
TECH=re.compile(r"(?i)\b(code|coding|python|script|program(ming)?|html|css|sql|server|api|apis|database|linux|windows|install\w*|command|terminal|function|algorithm|javascript|node\.?js|java|rust|c\+\+|docker|git|github|excel|spreadsheet|formula|equation|calculat\w*|math\w*|physics|chemistry|theorem|proof|translate|translation|essay|poem|story|stories|lyrics|song|list|table|json|csv|regex|compiler|library|framework|software|app|website|browser|model|neural|dataset|ai|gpu|cpu|computer|laptop pc|crypto|bitcoin|stock|stocks|invest\w*|tax|taxes|law|legal|medical|diagnos\w*|dosage|prescription|pretend|role ?play|act as|you are now|imagine you|character|game master|dungeon|numpy|pandas|ggplot\w*|plot|array|latex|unreal|engine|blueprint|pars(e|ing)|emails?|calendar|reminder|summar\w*|rewrite|domain|subdomain|derivative|integral|whitepaper|acronym|chatbots?|question:|answer:|news|article|paper|research|election|politic\w*|religio\w*|god|war)\b|\b[a-d][.)] \w+.{0,80}\b[b-e][.)] \w+");CONV=re.compile(r"(?i)\b(i|i'?m|i'?ve|i'?d|my|me|mine|you|your|you'?re|we|our|us)\b");OA=re.compile(r"(?i)open[- ]?assistant");NRM=lambda s:re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
def _oasst()->tuple:
 os.environ.setdefault("HF_DATASETS_OFFLINE","1");os.environ.setdefault("HF_HUB_OFFLINE","1")
 from datasets import load_dataset
 import scripts.build_chat5_corpus as B
 from scripts.build_chat_corpus import clean,short
 ms={}
 for s in ("oasst1","oasst2"):
  for sp in ("train","validation"):
   for r in load_dataset("OpenAssistant/"+s,split=sp).select_columns(["message_id","parent_id","text","role","lang","deleted","review_result","synthetic","model_name","message_tree_id","labels","detoxify"]):ms.setdefault(r["message_id"],r)
 ok=lambda r:r["lang"]=="en" and not r["deleted"] and r["review_result"] is not False and not r["synthetic"] and not r["model_name"] and not any(n in B.BADL and v>.5 for n,v in zip((r["labels"] or {}).get("name",[]),(r["labels"] or {}).get("value",[]))) and ((r["detoxify"] or {}).get("toxicity") or 0)<=.5
 T=lambda t,a=False:OA.sub("Adam",(short if a else clean)(B.asc(t)).replace("\n"," ")).strip();return ms,ok,T
def build(n_eval:int=300,n_dev:int=150,seed:int=17)->dict:
 t0=time.time();ms,ok,T=_oasst();att={}
 [att.setdefault(r["id"],r["split"]) for r in map(json.loads,gzip.open(ROOT/"data"/"chat5"/"attribution.jsonl.gz","rt")) if r["src"]!="gen"]
 root={m["message_tree_id"]:m for m in ms.values() if m["parent_id"] is None};ixT=DC.index(13,files=[TRAIN],cache=ROOT/"data"/"rchat"/".dc");ixE=DC.index()
 TU={NRM(u) for u in re.findall(r"User: ([^\n]*)\n",TRAIN.read_text(encoding="ascii"))};st=collections.Counter();pool=[];seen=set()
 for m in ms.values():
  t=m["message_tree_id"];sp=att.get(t,"none")
  if m["role"]!="prompter" or sp=="train" or root[t]["lang"]!="en" or not ok(m):continue
  path=[];p=m
  while p["parent_id"]:p=ms.get(p["parent_id"]);path.append(p) if p else None;p=p or {"parent_id":None}
  path=path[::-1];ex=[(T(path[i]["text"]),T(path[i+1]["text"],True)) for i in range(0,len(path)-1,2) if path[i]["role"]=="prompter" and path[i+1]["role"]=="assistant"]
  if len(ex)*2!=len(path) or not all(ok(x) for x in path):st["bad_path"]+=1;continue
  q=T(m["text"]);ex=ex[-3:];ac=R.act(q);full=" ".join(u+" "+a for u,a in ex)+" "+q;k=NRM(q)
  why="len" if not 1<=len(R.toks(q))<=70 else "act" if ac in ("fact","task") or any(R.act(u) in ("task",) for u,_ in ex) else "tech" if TECH.search(q+" "+" ".join(u for u,_ in ex)) else "nonconv" if not CONV.search(q) and ac not in R.PHATIC else "other_noctx" if ac=="other" and not ex else "bad" if BAD.search(full) or any(not R.safe(x,True) for x in [q]+[u for u,_ in ex]) or any(not R.safe(a) for _,a in ex[-1:]) else "dup" if k in seen else "contam_train13" if DC.dirty(full,ix=ixT) else "contam_evals13" if sp=="none" and DC.dirty(full,ix=ixE) else "train_line" if k in TU and len(R.toks(q))>6 else ""
  st[why or "kept"]+=1;seen.add(k)
  why or pool.append({"src":"oasst_hold" if sp=="hold" else "oasst_unused","lic":"apache-2.0","tree":t,"id":m["message_id"],"act":ac,"history":[list(x) for x in ex],"prompt":q})
 from datasets import load_dataset
 for i,r in enumerate(load_dataset("CohereLabs/aya_dataset",split="test")):
  if r["language_code"]!="eng" or r["annotation_type"]!="original-annotations":continue
  q=re.sub(r"\s+"," ",r["inputs"]).strip();ac=R.act(q);k=NRM(q)
  ok2=1<=len(R.toks(q))<=70 and ac not in ("fact","task","other") and not BAD.search(q) and not TECH.search(q) and R.safe(q,True) and k not in seen and not DC.dirty(q,ix=ixT);st["aya_"+("kept" if ok2 else "drop")]+=1;seen.add(k)
  ok2 and pool.append({"src":"aya_test","lic":"apache-2.0","tree":"aya-test:%d"%i,"id":"aya-test:%d"%i,"act":ac,"history":[],"prompt":q})
 rnd=random.Random(seed);rnd.shuffle(pool);dv=lambda it:zlib.crc32(it["tree"].encode())%4==0;own=[]
 mk=lambda L,p:[{"src":"own","lic":"cc0-own","tree":p+str(i),"id":p+str(i),"act":R.act(x[-1]),"history":[x[j:j+2] for j in range(0,len(x)-1,2)],"prompt":x[-1]} for i,l in enumerate(L) for x in [[s.strip() for s in l.split("||")]]];own=mk(OWN,"own:");odev=mk(OWN_DEV,"own-dev:")
 Q={"advice":62,"opinion":28,"share":32,"feel":10,"greet":8,"howru":8,"thanks":12,"bye":2,"other":68};need=n_eval-len(own)
 def take(ps,n):
  out=[];c=collections.Counter();tr=set()
  for w in (lambda it:bool(it["history"]),lambda it:True):
   for it in ps:
    if len(out)>=n:break
    q=max(1,round(Q.get(it["act"],0)*n/sum(Q.values())))
    if it["id"] not in {o["id"] for o in out} and it["tree"] not in tr and c[it["act"]]<q and w(it):out.append(it);c[it["act"]]+=1;tr.add(it["tree"])
  for it in ps:
   if len(out)>=n:break
   it["id"] not in {o["id"] for o in out} and it["tree"] not in tr and (out.append(it),tr.add(it["tree"]))
  return out
 ev=own+take([p for p in pool if not dv(p)],need);de=odev+take([p for p in pool if dv(p)],n_dev);E.mkdir(parents=True,exist_ok=True)
 for f,rows in ((EVAL,ev),(DEV,de)):f.write_text("".join(json.dumps(dict(r,n=i))+"\n" for i,r in enumerate(rows)))
 lk=EL.lock([EVAL,DEV],LOCK);meta={"eval":len(ev),"dev":len(de),"stats":dict(st),"pool":len(pool),"eval_by_src":dict(collections.Counter(r["src"] for r in ev)),"eval_by_act":dict(collections.Counter(r["act"] for r in ev)),"eval_ctx_turns":dict(collections.Counter(len(r["history"])+1 for r in ev)),"dev_by_act":dict(collections.Counter(r["act"] for r in de)),"decontam":{"train_13gram_hits_eval":sum(bool(DC.dirty(" ".join(u+" "+a for u,a in r["history"])+" "+r["prompt"],ix=ixT)) for r in ev),"train_exact_prompt_lines_eval":sum(NRM(r["prompt"]) in TU for r in ev),"train_exact_prompt_lines_eval_phatic_le6w":sum(NRM(r["prompt"]) in TU and len(R.toks(r["prompt"]))<=6 for r in ev)},"lock":lk["splits_hash"],"s":round(time.time()-t0,1)}
 (E/"build_meta.json").write_text(json.dumps(meta,indent=1));return meta
_IDF={}
def idf(w:str)->float:
 if not _IDF:
  c=collections.Counter()
  for l in open(ROOT/"data"/"clean1"/"wordcounts.txt","rb").read().decode("utf-8","replace").splitlines():
   a=l.rsplit(None,1)
   if len(a)==2 and a[1].isdigit():c[R.stem(a[0])]+=int(a[1])
  _IDF.update({k:v for k,v in c.items()});_IDF[""]=sum(c.values())
 return min(12.0,math.log((_IDF[""]+1)/(_IDF.get(w,0)+1)))
PH={"greet":r"\b(hi|hello|hey|greetings|morning|welcome|good to)\b|\?","howru":r"\b(well|good|great|fine|doing|thanks|you)\b","thanks":r"\b(welcome|glad|happy|anytime|no problem|pleasure|you bet)\b","bye":r"\b(bye|night|care|later|see you|soon|sleep)\b"}
ATTR=re.compile(r"\s*\(Adapted from human-written repl(y|ies) in [^)]*\)\s*$")
def metrics(it:dict,out:str,ixT)->dict:
 t=ATTR.sub("",out).strip();att=bool(ATTR.search(out));h=it["history"];q=it["prompt"];cu=" ".join(u for u,_ in h);ca=" ".join(a for _,a in h[-1:])
 P=set(R.cwords(q));C=set(R.cwords(cu+" "+ca))-P;Rw=set(R.cwords(t));Rs=set().union(*[R.syns(w) for w in Rw]) if Rw else set()
 hit=lambda W:sum(idf(w)*(1.0 if w in Rw else 0.7 if R.syns(w)&Rs else 0.0) for w in W)/max(1e-9,sum(idf(w) for w in W)) if W else 0.0
 pb=R.cwords(q);bg={zlib.crc32((a+" "+b).encode()) for a,b in zip(pb,pb[1:])};rb=R.cwords(t);rg={zlib.crc32((a+" "+b).encode()) for a,b in zip(rb,rb[1:])};bi=len(bg&rg)/len(bg) if bg else 0.0
 ph=it["act"] in PH and len(P)<=1;rel=float(bool(re.search(PH[it["act"]],t,re.I))) if ph else (0.6*hit(P)+0.15*bi+0.25*hit(C) if C else 0.8*hit(P)+0.2*bi)
 w=R.toks(t);nw=R.nonword(t);gar=[g for g,x in (("triple",re.search(r"(?i)\b(\w+)\b(?:\W+\1\b){2}",t)),("longtok",any(len(x)>25 for x in w)),("camel",re.search(r"\b[a-z]+[A-Z][a-z]+\b",t)),("parens",t.count("(")!=t.count(")")),("quotes",t.count('"')%2==1),("unfinished",not re.search(r"[.!?)\"']$",t)),("empty",not w)) if x]
 ng=[" ".join(x.lower() for x in w[i:i+4]) for i in range(max(0,len(w)-3))];srep=max(collections.Counter(ng).values(),default=0)>=2;vb=DC.dirty(t,ix=ixT)
 us=q+" "+cu;allc=us+" "+" ".join(a for _,a in h);nm=R.names_set();rn={x for x in w if x in nm and x!="Adam"}-set(R.toks(allc));g=R.gender(us);fr=sum(x.lower() in (R.PRO_F if g<0 else R.PRO_M) for x in w) if g else 0;sm=sum(x.lower() in (R.PRO_M if g<0 else R.PRO_F) for x in w) if g else 0
 mine={m.lower() for m in re.findall(r"(?i)\bmy ((?:best )?[a-z]+)",us) if R.stem(m.split()[-1]) in R.KIN or m.lower() in ("dog","cat","puppy","roommate","coworker","boss","friend","best friend","husband","wife","partner","kids","son","daughter")};flip=[m for m in mine if re.search(r"(?i)\bmy "+re.escape(m)+r"\b",t)]
 cv=[v for v,x in (("foreign_name",rn),("gender",fr and not sm),("my_flip",flip)) if x];testable=bool({x for x in R.toks(allc) if x in nm} or g or mine)
 r={"rel":round(rel,3),"rel_pass":rel>=0.2,"nonword":round(nw,3),"garble":gar,"coh_pass":nw<=0.05 and not gar,"verbatim13":bool(vb),"attributed":att,"self_rep":srep,"norep_pass":(not vb or att) and not srep,"cons_viol":cv,"cons_testable":testable,"cons_pass":not cv,"words":len(w),"len_pass":3<=len(w)<=120}
 r["all_pass"]=r["rel_pass"] and r["coh_pass"] and r["norep_pass"] and r["cons_pass"] and r["len_pass"];r.update(human(it,t));return r
DECL=re.compile(r"(?i)^\W*i'?m not sure about that one yet\.?\W*$");UNC=re.compile(r"(?i)\b(i'?m not sure (about|if|what|how)|i don'?t know (that|much|for certain|what|how)|not sure i can)\b");ASKC=re.compile(r"(?i)\b(ask\w*|question\w*|follow[- ]?up\w*|invit\w*|curious)\b");EMPC=re.compile(r"(?i)\b(empath\w*|sympath\w*|acknowledg\w*|validat\w*|congratul\w*|celebrat\w*|comfort\w*|reassur\w*|support\w*|excite\w*|happy for)\b");STANCE=re.compile(r"(?i)\b(i think|i'?d|i love|i like|i really like|i prefer|i'?m (a big fan|a fan|into|team)|honestly|personally|my take|to me|i find|i enjoy|i'?m not (a|sure)|depends|can be|i'?m all for|i'?d say|i agree|i lean)\b")
TPLW=set(R.stem(w) for w in "interesting story thinking thought curious wonder wondering mind going celebrate celebrating part best hardest toughest matters matter option options tried take appeal fun love lovely enjoy exciting news weighing carry exhausting common wondering pulling call both sides certain sure occasion shared sharing telling glad hear heard talk anything else carry feel feeling wow ooh oh nice".split())
RBX=set(R.stem(w) for w in "acknowledge acknowledges mention mentions mentioning reply response specific specifically relevant follow followup ask asks question questions such generic praise offer offers suggest suggests suggestion practical advice without instead rather user their them they person situation express expresses validate validates empathy empathetic show shows sympathy tone warm encourage encourages engage engages refer refers reference only just like eg rather than feeling feelings share shares sharing concrete idea ideas option options step steps either whether take view balanced actual reason reasons point points note notes something someone include includes including address addresses specific validates recognizes recognize recognise supportive support".split())
def human(it:dict,t:str)->dict:
 h=it["history"];q=it["prompt"];cat=it.get("cat") or it["act"];ss=[x for x in R.SENT.split(t) if x.strip()];Rw=set(R.cwords(t));Rs=set().union(*[R.syns(w) for w in Rw]) if Rw else set()
 hitw=lambda w:w in Rw or bool(R.syns(w)&Rs) or any(R.cmatch(w,y) for y in Rw)
 U=[w for w in dict.fromkeys(R.cwords(q+" "+" ".join(u for u,_ in h))) if w not in R.GENERIC and w not in R.NGEN];spec=any(hitw(w) for w in U) if U else True;Aw=set(R.cwords(q+" "+" ".join(u+" "+a for u,a in h)));sub=it["act"] in R.PHATIC or any(w not in Aw and w not in R.GENERIC and w not in TPLW and not R.syns(w)&set().union(*[R.syns(x) for x in Aw]) for w in Rw)
 qs=[x for x in ss if x.rstrip().endswith("?")];qrel=any(set(R.cwords(x))&(set(U)|set(R.cwords(" ".join(it.get("rubric") or []))))for x in qs)
 def crit(c):
  kw=[w for w in R.cwords(c) if w not in RBX];ch=sum(hitw(w) for w in kw)/len(kw) if kw else 0.0;v=min(1.0,2*ch)
  v=max(v,1.0 if qs and (qrel or not kw) else 0.5 if qs else 0.0) if ASKC.search(c) else v
  return max(v,0.5+0.5*min(1.0,2*ch)) if EMPC.search(c) and (R.OPEN.match(t) or R.celeb(t) or set(x.lower() for x in R.toks(ss[0] if ss else ""))&R.WARM) else v
 rb=[crit(c) for c in it.get("rubric") or []];rub=sum(rb)/len(rb) if rb else None
 adv=it["act"]=="advice" or cat in ("advice","planning");opi=it["act"]=="opinion" or cat in ("opinions","what_do_you_think");dec=bool(DECL.match(t))
 subst=any(not x.rstrip().endswith("?") and len(R.toks(x))>=4 and set(R.cwords(x))-R.GENERIC for x in ss)
 eng=(subst and not UNC.search(t)) if adv else (subst and bool(STANCE.search(t) or not qs)) or bool(STANCE.search(t) and qs) if opi else bool(qs) or it["act"] in ("thanks","bye")
 return {"decline":dec,"unc":bool(UNC.search(t)),"spec_pass":spec,"eng_pass":eng,"qrel":qrel,"rubric":None if rub is None else round(rub,3),"substance":sub,"good_proxy":(not dec) and spec and sub and eng and (rub is None or rub>=0.5) and len(R.toks(t))>=3 and R.nonword(t)<=0.05,"ontopic_proxy":(not dec) and spec}
def hold(paths:list,seed:int=17,tag:str="")->dict:
 HOLD,HLOCK=E/f"conv_hold{tag}.jsonl",E/f"lock_hold{tag}.json";ixE=DC.index(13,files=[f for f in DC.eval_sources() if f!=HOLD.resolve()],cache=ROOT/"data"/"rchat"/".dc");PV={NRM(json.loads(l)["prompt"]) for f in E.glob("conv_*.jsonl") if f.resolve()!=HOLD.resolve() for l in f.open() if l.strip()}
 rows=[json.loads(l) for p in paths for l in open(p) if l.strip()];random.Random(seed).shuffle(rows);ixT=DC.index(13,files=[TRAIN],cache=ROOT/"data"/"rchat"/".dc");ix=R.index();IQ={NRM(ix.txt("q",i)) for i in range(ix.n)};TU={NRM(u) for u in re.findall(r"User: ([^\n]*)\n",TRAIN.read_text(encoding="ascii"))}
 out=[{"src":"own-hold","lic":"cc0-own","tree":r["id"],"id":r["id"],"cat":r["cat"],"act":R.act(r["prompt"]),"history":[[str(u),str(a)] for u,a in r.get("history") or []],"prompt":r["prompt"].strip(),"rubric":[str(x) for x in r["rubric"]],"n":i} for i,r in enumerate(rows)]
 assert len({o["id"] for o in out})==len(out) and all(2<=len(o["rubric"])<=3 and len(o["history"])<=3 for o in out);HOLD.write_text("".join(json.dumps(o)+"\n" for o in out));lk=EL.lock([HOLD],HLOCK)
 full=lambda o:" ".join(u+" "+a for u,a in o["history"])+" "+o["prompt"];meta={"n":len(out),"by_cat":dict(collections.Counter(o["cat"] for o in out)),"by_act":dict(collections.Counter(o["act"] for o in out)),"ctx_turns":dict(collections.Counter(len(o["history"]) for o in out)),"decontam":{"train_13gram_hits":sum(bool(DC.dirty(full(o),ix=ixT)) for o in out),"train_exact_prompt":sum(NRM(o["prompt"]) in TU for o in out),"rchat_index_exact_prompt":sum(NRM(o["prompt"]) in IQ for o in out),"rchat_index_exact_prompt_items":[o["prompt"] for o in out if NRM(o["prompt"]) in IQ],"other_evals_13gram_hits":sum(bool(DC.dirty(full(o),ix=ixE)) for o in out),"other_evals_exact_prompt":sum(NRM(o["prompt"]) in PV for o in out)},"registered":DC.rel(HOLD) in {DC.rel(f) for f in DC.eval_sources()},"lock":lk["splits_hash"]}
 (E/f"hold{tag}_meta.json").write_text(json.dumps(meta,indent=1));return meta
def hand(split:Path,res:dict,a:str,b:str,k:int=60,seed:int=11,tag:str="_q")->Path:
 rows=[json.loads(l) for l in split.open()];rnd=random.Random(seed);by=collections.defaultdict(list);[by[r.get("cat") or r["act"]].append(r) for r in rows];pick=[]
 while len(pick)<min(k,len(rows)):
  for c in sorted(by):by[c] and len(pick)<k and pick.append(by[c].pop(rnd.randrange(len(by[c]))))
 A={r["n"]:r["out"] for r in res[a]};B={r["n"]:r["out"] for r in res[b]};it=[(r,s,(A if s==a else B)[r["n"]]) for r in pick for s in (a,b)];rnd.shuffle(it);md=["# Blind hand rating","","Rate each reply on its own. G = good: specific to the user's point, fluent and natural, helpful or warm, not generic. T = on-topic but weak (generic, awkward, only partly helpful, or slightly off). X = off-topic, wrong, broken grammar, or nonsense.",""];key=[]
 for i,(r,s,o) in enumerate(it,1):key.append({"i":i,"n":r["n"],"sys":s});md+=["## %d"%i,""]+["> User: "+u+"\n> Adam: "+v for u,v in r["history"]]+["**User:** "+r["prompt"],"","**Reply:** "+(ATTR.sub("",o).strip() or "(empty)"),"","Rating: ___",""]
 (E/f"hand_blind{tag}.md").write_text("\n".join(md));(E/f"hand_key{tag}.json").write_text(json.dumps({"a":a,"b":b,"key":key},indent=1));return E/f"hand_blind{tag}.md"
def handscore(tag:str="_q")->dict:
 tag=os.environ.get("RCHAT_HAND_TAG",tag)
 K=json.loads((E/f"hand_key{tag}.json").read_text());G=json.loads((E/f"hand_ratings{tag}.json").read_text());o={}
 for s in (K["a"],K["b"]):v=[G[str(k["i"])] for k in K["key"] if k["sys"]==s];o[s]={"n":len(v),"good":round(sum(x=="G" for x in v)/len(v),3),"on_topic":round(sum(x in "GT" for x in v)/len(v),3)}
 (E/f"hand_score{tag}.json").write_text(json.dumps(o,indent=1));return o
SYS={"current":{"AMNI_RCHAT":"0","AMNI_RAY_RKNOW":"0"},"gen":{"AMNI_RCHAT":"0","AMNI_RAY_QA_DECLINE":"0","AMNI_RAY_RKNOW":"0"},"new":{"AMNI_RCHAT":"1","AMNI_RAY_RKNOW":"0"},"new_ray":{"AMNI_RCHAT":"1","AMNI_RCHAT_RAY":"1","AMNI_RAY_RKNOW":"0"},"rknow":{"AMNI_RCHAT":"0","AMNI_RAY_RKNOW":"1"},"new_rknow":{"AMNI_RCHAT":"1","AMNI_RAY_RKNOW":"1"}}
def respond(split:Path,systems:list,out:Path)->dict:
 os.environ.update(AMNI_GEN_BACKEND="ray",AMNI_UMEM_DIR=tempfile.mkdtemp(prefix="rchat_umem_",dir=os.environ.get("TMPDIR")),AMNI_LEARN_WEB="0")
 from amni.serve import gguf_runtime as G
 rows=[json.loads(l) for l in split.open()];orig=G._rchat;log={}
 def wrap(message,history=None,user=None,session=None,e=None,fb=True):
  t=time.perf_counter();r=orig(message,history,user,session,e,fb);log["rc"]=r if r or "rc" not in log else log["rc"];log["rc_ms"]=log.get("rc_ms",0.0)+(time.perf_counter()-t)*1000;return r
 G._rchat=wrap;res={s:[] for s in systems}
 for s in systems:
  for k in set().union(*SYS.values()):os.environ.pop(k,None)
  os.environ.update(SYS[s]);R._RECENT.clear()
  for it in rows:
   log.clear();t=time.perf_counter();cr=G.chat(it["prompt"],history=[tuple(x) for x in it["history"]],user="rceval_%s_%d"%(s,it["n"]),session="rceval_%s"%s);o=(cr.get("answer") or "").strip();ms=(time.perf_counter()-t)*1000;tr=str(cr.get("tier") or "")
   res[s].append({"n":it["n"],"out":o,"ms":round(ms,1),"tier":tr,"route":"concern" if tr=="tier_concern" else "emergency" if tr=="tier_emergency" else "rchat" if log.get("rc") and o==log["rc"].strip() else "decline" if o=="I'm not sure about that one yet." else "other","rc_ms":round(log.get("rc_ms",0.0),2)})
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(res));return res
def score(split:Path,res:dict)->dict:
 rows={json.loads(l)["n"]:json.loads(l) for l in split.open()};ixT=DC.index(13,files=[TRAIN],cache=ROOT/"data"/"rchat"/".dc");M={}
 for s,rs in res.items():
  ms=[dict(metrics(rows[r["n"]],r["out"],ixT),n=r["n"],ms=r["ms"],route=r["route"],rc_ms=r["rc_ms"]) for r in rs];f=lambda k,sub=ms:round(sum(bool(m[k]) for m in sub)/max(1,len(sub)),4);ct=[m for m in ms if m["cons_testable"]];lat=sorted(m["ms"] for m in ms);rl=sorted(m["rc_ms"] for m in ms if m["route"]=="rchat")
  M[s]={"n":len(ms),"all_pass":f("all_pass"),"rel_mean":round(sum(m["rel"] for m in ms)/len(ms),4),"rel_pass":f("rel_pass"),"coh_pass":f("coh_pass"),"nonword_mean":round(sum(m["nonword"] for m in ms)/len(ms),4),"norep_pass":f("norep_pass"),"verbatim13":f("verbatim13"),"verbatim13_unattributed":round(sum(m["verbatim13"] and not m["attributed"] for m in ms)/len(ms),4),"attributed":f("attributed"),"self_rep":f("self_rep"),"cons_pass":f("cons_pass"),"cons_pass_testable":f("cons_pass",ct),"cons_testable_n":len(ct),"len_pass":f("len_pass"),"words_mean":round(sum(m["words"] for m in ms)/len(ms),1),"routes":dict(collections.Counter(m["route"] for m in ms)),"decline":f("decline"),"unc":f("unc"),"spec_pass":f("spec_pass"),"substance":f("substance"),"eng_pass":f("eng_pass"),"good_proxy":f("good_proxy"),"ontopic_proxy":f("ontopic_proxy"),"rubric_mean":(lambda v:round(sum(v)/len(v),4) if v else None)([m["rubric"] for m in ms if m["rubric"] is not None]),"lat_ms_p50":round(lat[len(lat)//2],1),"lat_ms_p95":round(lat[int(len(lat)*.95)],1),"rchat_ms_p50":round(rl[len(rl)//2],2) if rl else None,"rchat_ms_p95":round(rl[int(len(rl)*.95)],2) if rl else None,"by_act":{a:{"n":len(x),"all_pass":f("all_pass",x),"good_proxy":f("good_proxy",x),"rel_mean":round(sum(m["rel"] for m in x)/len(x),3)} for a in sorted({rows[m["n"]]["act"] for m in ms}) for x in [[m for m in ms if rows[m["n"]]["act"]==a]]},"by_cat":{c:{"n":len(x),"good_proxy":f("good_proxy",x),"ontopic_proxy":f("ontopic_proxy",x)} for c in sorted({rows[m["n"]].get("cat","") for m in ms}) if c for x in [[m for m in ms if rows[m["n"]].get("cat")==c]]},"items":ms}
 return M
def pairwise(split:Path,res:dict,a:str,b:str,k:int=30,seed:int=5,tag:str="")->tuple:
 A0={r["n"]:r["out"] for r in res[a]};B0={r["n"]:r["out"] for r in res[b]};rows=[json.loads(l) for l in split.open() if A0[json.loads(l)["n"]].strip()!=B0[json.loads(l)["n"]].strip()];rnd=random.Random(seed);by=collections.defaultdict(list);[by[r["act"]].append(r) for r in rows];pick=[]
 while len(pick)<min(k,len(rows)):
  for ac in sorted(by):
   by[ac] and len(pick)<k and pick.append(by[ac].pop(rnd.randrange(len(by[ac]))))
 A={r["n"]:r["out"] for r in res[a]};B={r["n"]:r["out"] for r in res[b]};md=["# Adam open chat: blind pairwise sample","","For each prompt, pick the better reply (1 or 2), or write = for a tie. Judge helpfulness, warmth, fit to the conversation, and naturalness. Attribution notes were removed from both sides so they do not reveal the system. The key is in pairwise_key%s.json; do not open it until you have judged."%tag,""];key=[]
 for i,r in enumerate(pick,1):
  sw=rnd.random()<0.5;x,y=(B[r["n"]],A[r["n"]]) if sw else (A[r["n"]],B[r["n"]]);key.append({"i":i,"n":r["n"],"1":b if sw else a,"2":a if sw else b})
  md+=["## %d"%i,""]+(["**Earlier in the conversation:**",""]+[l for u,v in r["history"] for l in ("> User: "+u,">","> Adam: "+v,">")]+[""] if r["history"] else [])+["**User:** "+r["prompt"],"","**Reply 1:** "+(ATTR.sub("",x).strip() or "(empty)"),"","**Reply 2:** "+(ATTR.sub("",y).strip() or "(empty)"),"","Better: ___",""]
 (E/f"pairwise_blind{tag}.md").write_text("\n".join(md));(E/f"pairwise_key{tag}.json").write_text(json.dumps({"a":a,"b":b,"key":key},indent=1));return E/f"pairwise_blind{tag}.md",E/f"pairwise_key{tag}.json"
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["build","hold","run","handscore"]);ap.add_argument("--split",default="eval");ap.add_argument("--systems",default="current,gen,new");ap.add_argument("--kind",default="prod");ap.add_argument("--pair",default="");ap.add_argument("--tag",default="");ap.add_argument("--base",default="");ap.add_argument("--hand",default="");ap.add_argument("--hand-k",type=int,default=40);ap.add_argument("files",nargs="*");a=ap.parse_args(argv)
 if a.cmd in ("build","hold","handscore"):print(json.dumps(build() if a.cmd=="build" else hold(a.files,tag=a.tag) if a.cmd=="hold" else handscore(),indent=1));return
 sp,lk={"eval":(EVAL,LOCK),"dev":(DEV,LOCK),"hold":(HOLD,HLOCK),"hold2":(E/"conv_hold2.jsonl",E/"lock_hold2.json"),"hold3":(E/"conv_hold3.jsonl",E/"lock_hold3.json"),"hold4":(E/"conv_hold4.jsonl",E/"lock_hold4.json")}.get(a.split,(Path(a.split),None));bad=EL.verify(lk) if lk else {};assert not bad,bad;a.split=a.split if lk else sp.stem.replace("conv_","");sy=[x for x in a.systems.split(",") if x];t0=time.time();res=respond(sp,sy,E/f"responses_{a.split}{a.tag}.json") if sy else {}
 [res.__setitem__(n,json.loads(Path(f).read_text())[o]) for b in a.base.split(";") if b for n,f,o in [b.split(":")]];M=score(sp,res)
 rep=EL.record(E/f"report_{a.split}{a.tag}.json",{"split":a.split,"systems":{s:{k:v for k,v in m.items() if k!="items"} for s,m in M.items()},"items":{s:m["items"] for s,m in M.items()},"rss_gb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,2),"s":round(time.time()-t0,1)},code_files=[ROOT/"amni"/"compute"/"rchat.py",Path(__file__),ROOT/"amni"/"serve"/"gguf_runtime.py"],splits=json.loads(lk.read_text()) if lk else [sp],kind=a.kind if lk else "dry")
 [pairwise(sp,res,*p.split(":")[0].split(","),tag=(p.split(":")+[""])[1]) for p in a.pair.split(";") if p];a.hand and hand(sp,res,*a.hand.split(","),k=a.hand_k,tag="_"+a.split);print(json.dumps({s:{k:v for k,v in m.items() if k not in ("by_act","by_cat")} for s,m in rep["systems"].items()},indent=1))
if __name__=="__main__":main()
