import re,math,hashlib,collections
from pathlib import Path
DIR=Path(__file__).resolve().parent/"adam_t_talk";P=lambda s,c="|":[x.strip() for x in s.split(c) if x.strip()]
GV={"day":P("Saturday|Sunday|Friday|this weekend"),"rel":P("sister|brother|cousin|best friend|aunt|uncle|roommate|coworker"),"friend":P("Sam|Priya|Marcus|Jen|Diego|Hannah|Theo|Aisha|Luis|Megan|Omar|Kate")}
UP={"pos":P("Guess what,|Good news:|Okay so|So|Hey!|Yay,|Big news:"),"neg":P("Ugh,|Man,|Honestly,|Hey,|So|Sigh.|Okay,"),"neu":P("So|Hey,|Okay so|Random thing, but|Hi!|Hey there,")}
UC={"pos":P("Thanks!|Thank you!|Aw, thanks.|Haha, thank you!|Ha, thanks.|Yeah!|Haha, yeah.|Thanks, you're sweet."),"neg":P("Yeah, true.|Thanks.|Ha, yeah.|True, thanks.|Okay, thanks.|Yeah. Thanks for listening.|Ha, fair.|Yeah, I know."),"neu":P("Yeah.|Ha, true.|Thanks!|Okay, cool.|Fair enough.|Sounds good.|Ha, yeah.|True.")}
LW=re.compile(r"(?i)^(?:my|we|the|our|just|so|today|this|it|there|finally|got|went|guess|all|work|last|tomorrow|yesterday|tonight|a|an|me|everyone|nobody|someone|what|how|why|is|are|do|does|did|can|should|have|has|been|spent|made|tried|found|started|had|one|some|no|not|any|every|who|when|where|any|that|these|those|its|it's|there's|here's|looks|feels|kind|sort|pretty|really|still|almost|nearly|our|his|her|their|two|three|first|new|hot|cold|rain|snow)\b")
BAN=re.compile(r"(?i)thanks?(?: you)? for (?:telling|sharing|letting me know)|\bas an ai\b|\bi can'?t help\b|\bi(?:'m| am) (?:unable|not able)\b|feelings are valid|\bit'?s valid\b|\bi hear you\b|\bthat must be (?:so )?(?:hard|difficult)\b|\b911\b|\b988\b|\bemergenc|\bhotline|\btherap|\bcounsel|\bprofessional help\b|\breach out to\b|\bself-care\b|\bbe gentle with yourself\b|\bhold space\b|\bprocess (?:your|those|these) (?:feelings|emotions)\b|\bsafe space\b|\bi'?m just an? (?:ai|language model|computer)\b|[^\x20-\x7e]")
SN=re.compile(r"[.!?]+(?=\s|$)")
def ns(t:str)->int:return len(SN.findall(t)) or 1
def cap1(t:str)->str:return t[:1].upper()+t[1:]
def parse(txt:str,src:str="")->list:
 S=[];E=[]
 for j,l in enumerate(txt.splitlines()):
  l=l.strip()
  if not l:continue
  k,_,v=l.partition(" ");v=v.strip()
  if k.startswith("@"):f=v.split();S.append({"id":k[1:],"mood":f[0],"cat":f[1],"w":next((float(x[2:]) for x in f[2:] if x.startswith("w=")),1.0),"hold":"nohold" not in f[2:],"src":src,"U":[],"V":{},"A":[],"Q":[],"X":[],"C":[]});continue
  s=S[-1] if S else None
  if s is None:E.append((src,j+1,"no scenario"));continue
  if k=="V":[s["V"].__setitem__(a.strip(),P(b,",")) for a,_,b in (x.partition("=") for x in P(v,";"))]
  elif k in ("U","A","X","C"):s[k]+=P(v)
  elif k=="Q":s["Q"].append({"p":P(v),"b":[]})
  elif k==">":s["Q"][-1]["b"].append({"u":P(v),"r":[]}) if s["Q"] else E.append((src,j+1,"> without Q"))
  elif k=="=":s["Q"][-1]["b"][-1]["r"].extend(P(v)) if s["Q"] and s["Q"][-1]["b"] else E.append((src,j+1,"= without >"))
  else:E.append((src,j+1,"bad key "+k))
 for s in S:
  for q in s["Q"]:q["b"]=[b for b in q["b"] if b["u"] and b["r"]]
  s["Q"]=[q for q in s["Q"] if q["p"]];s["QB"]=[q for q in s["Q"] if q["b"]]
  [E.append((src,s["id"],"missing "+k)) for k in ("U","A") if not s[k]]
  [E.append((src,s["id"],"long adam line: "+x)) for x in s["A"]+s["X"]+[r for q in s["Q"] for b in q["b"] for r in b["r"]] if ns(x)>2]
  [E.append((src,s["id"],"question must be one sentence ending in ?: "+x)) for q in s["Q"] for x in q["p"] if ns(x)>1 or not x.endswith("?")]
  [E.append((src,s["id"],"reaction/sign-off must not be a question: "+x)) for x in s["A"]+s["X"] if x.endswith("?")]
  [E.append((src,s["id"],"banned: "+x)) for x in s["A"]+s["X"]+[p for q in s["Q"] for p in q["p"]]+[r for q in s["Q"] for b in q["b"] for r in b["r"]] if BAN.search(x)]
  [E.append((src,s["id"],"unknown slot "+m)) for x in s["U"]+s["A"]+s["X"]+s["C"]+[y for q in s["Q"] for y in q["p"]+[z for b in q["b"] for z in b["u"]+b["r"]]] for m in re.findall(r"\{(\w+)\}",x) if m not in s["V"] and m not in GV]
 return S,E
def load(d=DIR)->tuple:
 S=[];E=[];F={}
 for f in sorted(Path(d).glob("*.txt")):t=f.read_text();F[f.name]=hashlib.sha256(t.encode()).hexdigest();s,e=parse(t,f.name);S+=s;E+=e
 ids=collections.Counter(s["id"] for s in S);E+=[("",k,"duplicate id") for k,v in ids.items() if v>1];return S,E,F
def pf(t:str)->str:return " ".join(re.sub(r"[^a-z0-9']","",x.lower()) for x in t.split()[:3])
def pick(R,xs,C,cap,pre=""):
 ys=[x for x in xs if C[pre+x]<cap]
 if not ys:return None
 m=min(C[pre+x] for x in ys);return R.choice([x for x in ys if C[pre+x]<=m+1])
GRT=re.compile(r"(?i)^(?:hi|hey|hello|yo|ugh|man|so|okay|ok|oh|wow|yay|sigh|honestly|guess|omg|well|argh|ha|haha|yikes|oof|ooh|big news|good news|random|morning|evening|good (?:morning|night|evening|afternoon)|g'?night|sup|what'?s up|how are|how'?s)\b");KI=re.compile(r"^(?:I|I'm|I've|I'd|I'll|\{\w+\})(?:\W|$)")
def low(t:str)->str:return t if KI.match(t) or t[1:2].isupper() else t[:1].lower()+t[1:]
def sty(R,t:str,mood:str,first:bool,cat:str="")->str:
 if first and cat!="greet" and not GRT.match(t) and R.random()<.22:p=R.choice(UP[mood]);t=p+" "+(t if p[-1] in ".!?" else low(t))
 if R.random()<.12:t=low(t);t=t[:-1] if t.endswith(".") and not t.endswith("..") else t
 return t
def one(R,s,nt,C,cap):
 v={};g=lambda t:re.sub(r"\{(\w+)\}",lambda m:v.setdefault(m.group(1),R.choice(s["V"].get(m.group(1)) or GV[m.group(1)])),t);f=lambda t:cap1(g(t));y=lambda t,fi=False:g(sty(R,t,s["mood"],fi,s["cat"]));ks=[]
 u=pick(R,s["U"],C,10**9,"u:");qb=s["QB"] if nt>1 else s["Q"];wq=nt>1 or (bool(qb) and R.random()<.78)
 A=[a for a in s["A"] if ns(a)==1] if wq and qb else s["A"]
 a=pick(R,A,C,cap,"a:")
 if a is None:return None
 qm={y["p"][0]:y for y in qb};q=qm.get(pick(R,list(qm),C,10**9,"qn:")) if wq and qb else None
 if nt>1 and q is None:return None
 qp=pick(R,q["p"],C,cap,"q:") if q else None
 if q and qp is None:return None
 key="A:"+a+(" "+qp if qp else "")
 if C[key]>=cap:return None
 ks+=["u:"+u,"a:"+a,key]+(["q:"+qp,"qn:"+q["p"][0]] if qp else []);T=[(y(u,True),f(a)+(" "+f(qp) if qp else ""))]
 if nt>1:
  bi=pick(R,[str(i) for i in range(len(q["b"]))],C,10**9,"b:"+q["p"][0]+"#");b=q["b"][int(bi)];ub=R.choice(b["u"]);rs=[r for r in b["r"] if nt==2 or not r.rstrip().endswith("?")];r=pick(R,rs,C,cap,"r:")
  if r is None:return None
  ks+=["b:"+q["p"][0]+"#"+bi,"r:"+r];T.append((y(ub),f(r)))
  if nt>2:
   x=pick(R,s["X"],C,cap,"x:")
   if x is None:return None
   ks.append("x:"+x);T.append((y(R.choice(s["C"] or UC[s["mood"]])),f(x)))
 return T,ks
def xp(s,t:str)->list:
 ks=re.findall(r"\{(\w+)\}",t);vs=[s["V"].get(k) or GV[k] for k in ks];m=max([len(x) for x in vs]+[1])
 return [re.sub(r"\{(\w+)\}",lambda z,i=i:(lambda w:w[i%len(w)])(s["V"].get(z.group(1)) or GV[z.group(1)]),t) for i in range(m)]
def prune(S,ok,st)->list:
 D=[];bad=lambda s,t,u:next((w for x in xp(s,t) for w in [ok([(x,"x")] if u else [("x",x)])[0]] if w),"")
 def kp(s,xs,u,k):
  o=[]
  for x in xs:w=bad(s,x,u);w and (D.append([s["id"],k,w,x]),st.update(["talk_line_pruned_"+w]));w or o.append(x)
  return o
 for s in S:
  s["U"]=kp(s,s["U"],1,"U");s["C"]=kp(s,s["C"],1,"C");s["A"]=kp(s,s["A"],0,"A");s["X"]=kp(s,s["X"],0,"X")
  for q in s["Q"]:
   q["p"]=kp(s,q["p"],0,"Q")
   for b in q["b"]:b["u"]=kp(s,b["u"],1,">");b["r"]=kp(s,b["r"],0,"=")
   q["b"]=[b for b in q["b"] if b["u"] and b["r"]]
  s["Q"]=[q for q in s["Q"] if q["p"]];s["QB"]=[q for q in s["Q"] if q["b"]]
 Z=[s["id"] for s in S if not s["U"] or not s["A"]];st["talk_scenarios_dropped"]+=len(Z);return D,[s for s in S if s["id"] not in Z]
def build(R,n:int,ok,st,hold_frac:float=.06,cap:int=30,pcap:int=0,d=DIR)->tuple:
 S,E,F=load(d);assert not E,E[:40];PD,S=prune(S,ok,st);C=collections.Counter();PC=collections.Counter();seen=set();out=[];tw=sum(s["w"] for s in S);pcap=pcap or max(40,int(n*2*.012));BG=collections.Counter()
 H=sorted([s for s in S if s["hold"]],key=lambda s:hashlib.md5(s["id"].encode()).hexdigest());HS={s["id"] for s in H[:math.ceil(hold_frac*len(H))]}
 def tr(s):
  nt=R.choices((1,2,3),(35,40,25))[0];nt=min(nt,1 if not s["QB"] else 2 if not s["X"] else 3);r=one(R,s,nt,C,cap)
  if r is None:st["talk_try_capped"]+=1;return 0
  T,ks=r;p=[pf(a) for _,a in T]
  if any(PC[x]>=pcap for x in p):st["talk_try_prefix_cap"]+=1;return 0
  k=re.sub(r"[^a-z0-9]+"," ","".join(u+" "+a for u,a in T).lower()).strip()
  if k in seen:st["talk_try_dup"]+=1;return 0
  bad=[a for _,a in T if BAN.search(a)];assert not bad,bad
  why,g=ok(T)
  if why:st["talk_try_"+why]+=1;BG[(why,g)]+=1;return 0
  seen.add(k);C.update(ks);PC.update(p);out.append((f"{s['id']}:{len(out)}",T,s["id"],s["id"] in HS));st["talk_turns_%d"%len(T)]+=1;st["talk_scen_"+s["id"]]+=1;return 1
 for s in S:
  m=round(n*s["w"]/tw);g=0
  for _ in range(m*25):
   if g>=m:break
   g+=tr(s)
 for _ in range(60):
  if len(out)>=n:break
  pr=0
  for s in sorted(S,key=lambda s:st["talk_scen_"+s["id"]]):
   if len(out)>=n:break
   pr+=any(tr(s) for _ in range(6))
  if not pr:break
 ad=collections.Counter({k:v for k,v in C.items() if k[:2] in ("a:","q:","r:","x:","A:")});idm={s["id"]:s for s in S}
 info={"files":F,"scenarios":len(S),"pruned_lines":PD,"held_scenarios":sorted(HS),"by_category":dict(collections.Counter(idm[x]["cat"] for _,_,x,_ in out)),"by_mood":dict(collections.Counter(idm[x]["mood"] for _,_,x,_ in out)),"hold_examples":sum(h for *_,h in out),"template_cap":cap,"prefix_cap":pcap,"max_template_uses":max(ad.values()) if ad else 0,"top_templates":[[k,v] for k,v in ad.most_common(12)],"blocked":[[w,g,c] for (w,g),c in BG.most_common(60)],"shortfall":{s["id"]:round(n*s["w"]/tw)-st["talk_scen_"+s["id"]] for s in S if st["talk_scen_"+s["id"]]<round(n*s["w"]/tw)}}
 return out,info
