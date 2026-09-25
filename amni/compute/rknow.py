import os,re,json,math,unicodedata,collections
import numpy as np,pyarrow as pa
import amni.compute.ray_learn as RL
import amni.compute.ks_retrieve as KR
from amni.serve.user_memory import bh
IXD=os.environ.get("AMNI_RKNOW_INDEX",os.path.join(KR.KS,"index_rknow","wiki_v2"));STOP=set("a an the of and or in on at to for by with from as is was are were be been it its that this which who he she his her they their has had have not but also into than then there".split());QSTOP=set("what when where which whom whose why how many much did doe do name called call".split())
K1,B,DFMAX,M64=0.9,0.4,300000,(1<<64)-1;LB=float(os.environ.get("AMNI_RKNOW_LB","3.0"));LIDF=float(os.environ.get("AMNI_RKNOW_LIDF","9.0"));QCI=6.0;TEMP=re.compile(r"(?i)\b(latest|current|currently|now|newest|next|recent|recently|today|this year|upcoming|so far|to date|as of)\b");BYV=re.compile(r"(?i)\b(?:played|portrayed|voiced|performed|sung|recorded|written|directed|composed|produced|created|founded|invented|discovered|built|designed|painted|developed|won|hosted|narrated|owned|led)\s+by\s*$");ASV=re.compile(r"\s*(?:\(\s*)?(?:(?:stars?|starring|starred|plays?|played|portrays?|portrayed|voices?|voiced|appears?|appeared|cast)\s+)?as\s");POSS=re.compile(r"(?i)\b([\w.-]+(?:\s+[\w.-]+)?)['’]s\s+(?:real\s+|biological\s+|first\s+|second\s+|ex[- ]?)?(?:mother|father|mom|mum|dad|wife|husband|son|daughter|sister|brother|boyfriend|girlfriend|friend|best friend|boss|partner|uncle|aunt|cousin|niece|nephew|grandmother|grandfather|grandma|grandpa|baby|twin|fianc[eé]e?|lover|love interest|parents?|kids?|children|family|dog|cat)\b");ACT=re.compile(r"(?i)\b(?:play|plays|played|playing|portray\w*|voice[sd]?|star\w*|cast)\b");SOFT=re.compile(r"(?i)unknown|not sure|don'?t know|do not know|no idea|can'?t recall|not certain")
GBM=(lambda f:json.load(open(f)) if os.environ.get("AMNI_RKNOW_GB","1")=="1" and os.path.exists(f) else None)(os.environ.get("AMNI_RKNOW_GBF",os.path.join(os.path.dirname(os.path.abspath(__file__)),"rknow_rank.json")));GBM and GBM.update({k:np.asarray(GBM[k]) for k in ("f","t","l","r","v","roots")});GATE={"c":float(os.environ.get("AMNI_RKNOW_C",(GBM or {}).get("c",0.633)))}
def fnv(b:bytes)->int:
 h=0xcbf29ce484222325
 for x in b:h=((h^x)*0x100000001b3)&M64
 return h
_MT={}
def fold(t:str)->str:
 if t.isascii():return t
 _MT or _MT.update({c:None for c in range(0x300,0x110000) if unicodedata.category(chr(c)).startswith("M")})
 return unicodedata.normalize("NFKD",t).translate(_MT)
def toks(t:str)->list:
 out=[]
 for w in re.findall(r"[^\W_]+",fold(t).lower().replace("ς","σ")):
  n=len(w.encode());w=w if len(w)==1 else w[:-3]+"y" if n>4 and w.endswith("ies") else w[:-1] if n>3 and w.endswith("s") and not w.endswith(("ss","us","is")) else w
  (len(w)>1 or w.isnumeric()) and w not in STOP and out.append(w)
 return out
class Index:
 def __init__(s,d:str=IXD):
  s.d=d;s.m=json.load(open(os.path.join(d,"meta.json")));mm=lambda n,t:np.memmap(os.path.join(d,n),t,"r");s.terms,s.ptr,s.df,s.ntok=mm("terms.u64",np.uint64),mm("ptr.u64",np.uint64),mm("df.u32",np.uint32),mm("p_ntok.u16",np.uint16)
  s.fd=os.open(os.path.join(d,"pid.u32"),os.O_RDONLY);s.pf=[os.open(os.path.join(d,x),os.O_RDONLY) for x in ("p_doc.u32","p_off.u32","p_len.u32")];s.ft=os.open(os.path.join(d,"tf.u8"),os.O_RDONLY);s.N=int(s.m["n_pass"]);s.avg=float(s.m["avgdl"]);s.db=np.asarray(s.m["doc_base"],np.int64);s.segs=s.m["segments"];s.fs={};s.first=s.dfirst()
 def dfirst(s)->np.ndarray:
  f=os.path.join(s.d,"d_first.u32")
  if not os.path.exists(f):
   n=int(s.m["n_doc"]);c=np.zeros(n+1,np.int64);[np.add.at(c,np.fromfile(os.path.join(s.d,"p_doc.u32"),np.uint32,1<<24,offset=4*i).astype(np.int64)+1,1) for i in range(0,s.N,1<<24)];np.cumsum(c).astype(np.uint32).tofile(f+".tmp");os.replace(f+".tmp",f)
  return np.fromfile(f,np.uint32)
 def lookup(s,w:str)->int:
  h=np.uint64(fnv(w.encode()));j=int(np.searchsorted(s.terms,h));return j if j<len(s.terms) and s.terms[j]==h else -1
 def idf(s,w:str)->float:
  j=s.lookup(w);df=int(s.df[j]) if j>=0 else 0;return math.log(1+(s.N-df+0.5)/(df+0.5))
 def post(s,j:int)->tuple:
  a,b=int(s.ptr[j]),int(s.ptr[j+1]);return np.frombuffer(os.pread(s.fd,4*(b-a),4*a),np.uint32).astype(np.int64),np.frombuffer(os.pread(s.ft,b-a,a),np.uint8).astype(np.float32)
 def bm25(s,j:int)->tuple:
  p,tf=s.post(j);df=int(s.df[j]);dl=s.ntok[p].astype(np.float32);return p,math.log(1+(s.N-df+0.5)/(df+0.5))*tf*(K1+1)/(tf+K1*(1-B+B*dl/s.avg))
 def search(s,q:str,k:int=30)->list:
  J=[j for j in dict.fromkeys(s.lookup(w) for w in toks(q) if w not in QSTOP) if j>=0]
  if not J:return []
  L=[s.bm25(j) for j in J if int(s.df[j])<=DFMAX] or [s.bm25(min(J,key=lambda j:int(s.df[j])))];P=np.concatenate([p for p,_ in L]);u,inv=np.unique(P,return_inverse=True);S=np.bincount(inv,np.concatenate([x for _,x in L]))
  for j in J:
   if int(s.df[j])>DFMAX and len(L)<len(J):p,x=s.bm25(j);i=np.searchsorted(p,u);i=np.minimum(i,len(p)-1);h=p[i]==u;S[h]+=x[i[h]]
  lk=s.link(q) if LB else [];s.lk=set(lk)
  if lk:rg=[(int(s.first[g]),int(s.first[g+1])) for g in lk];S=S+LB*np.logical_or.reduce([(u>=lo)&(u<hi) for lo,hi in rg])*S.max()/(1.0+S.max())
  t=np.argpartition(-S,min(k,len(S))-1)[:k];t=t[np.argsort(-S[t])];return [(float(S[i]),int(u[i])) for i in t]
 def link(s,q:str)->list:
  s.tz=getattr(s,"tz",None) or RL._kx()[1].z[0][1];w=re.findall(r"[\w'&.-]+",q);out=[];i=0
  while i<len(w):
   hit=None
   for n in range(min(6,len(w)-i),0,-1):
    ph=" ".join(w[i:i+n]);tk=toks(ph)
    if not tk or (n==1 and (s.idf(tk[0])<LIDF or w[i].lower() in QSTOP)) or w[i].lower() in STOP:continue
    h=np.uint64(KR.th64(ph));j=int(np.searchsorted(s.tz["h"],h));J=range(j,int(np.searchsorted(s.tz["h"],h,"right")))
    if len(J):hit=(n,[int(s.db[int(s.tz["seg"][x])])+int(s.tz["doc"][x]) for x in J]);break
   out+=hit[1] if hit else [];i+=hit[0] if hit else 1
  return out
 def doc(s,g:int)->bytes:
  i=int(np.searchsorted(s.db,g,"right")-1);sd=os.path.join(KR.FETCH,"wikipedia",s.segs[i]);f=s.fs.get(i) or s.fs.setdefault(i,{"off":np.load(os.path.join(sd,"off.npy")),"raw":np.load(os.path.join(sd,"raw.npy")),"fd":os.open(os.path.join(sd,"blob.bin"),os.O_RDONLY)});d=g-int(s.db[i]);a=int(f["off"][d]);return pa.decompress(os.pread(f["fd"],int(f["off"][d+1])-a,a),int(f["raw"][d]),codec="zstd",asbytes=True)
 def text(s,pid:int,cache:dict=None)->tuple:
  g,o,n=(int(np.frombuffer(os.pread(f,4,4*pid),np.uint32)[0]) for f in s.pf);b=(cache if cache is not None else {}).setdefault(g,None) or s.doc(g);cache is not None and cache.__setitem__(g,b);return b.split(b"\x00",1)[0].decode("utf-8","replace"),b[o:o+n].decode("utf-8","replace"),g
_IX=[]
def ix()->Index:
 _IX or _IX.append(Index());return _IX[0]
CONN=set("of the de van von da del der la le du di bin al y".split());ENT=re.compile(r"(?<![\w'’.-])(?:[A-Z][\w'’.&-]*|[A-Z]\.)(?:[ \t]+(?:(?:of|the|de|van|von|da|del|der|la|le|du|di|bin|al|y)[ \t]+)*(?:[A-Z][\w'’.&-]*|[A-Z]\.|\d+(?:st|nd|rd|th)?\b))*")
FEAT=("bias","s0","pw","r","cov","lc","d","sup","n","share","rk","relhi","rello","relna","tok","tmiss","tbad","qover","ntok","top","title","q_date","q_num","q_person","q_place","q_other","tchar","gap","actqo","pas","pby","ppar","lnk","lead","lnklead","occ","occna","pasx","fx","fx5","fxl","fxn","fw","fpar","kb","ko","yorig","ynew","yold","dmin","fxu","fxa","sbad")
WTS=dict(zip(FEAT,json.loads(os.environ.get("AMNI_RKNOW_WTS","null") or "null") or (-1.8786,0.3376,-0.9641,-0.9774,-0.4726,-0.3256,-0.5874,0.219,-0.0181,0.2884,1.0231,0.2221,-0.3363,0.1453,0.4407,-0.4275,-0.6348,-0.2651,0.3533,0.3643,0.2292,-0.1632,-0.0085,-0.2075,-0.4141,-0.5528,-0.2444,0.0169,-0.352,0.3307,0.1693,0.1219,0.1938,0.1589,0.3825,0.4554,-0.4159,-0.049,0.1384,0.2919,0.0668,0.1309,-0.1985,-0.5796,-0.0518,0.0743,-0.0023,0.019,-0.0539,0.0843,-1.1724,0.0559,-0.2777)))
_TY={}
def ents(y:str)->list:
 out=[]
 for m in ENT.finditer(y):
  tk=m.group(0).split();a=m.start();ci=next((i for i,t in enumerate(tk[:-1]) if t.endswith(".") and not re.fullmatch(r"(?:[A-Z]|Jr|Sr|St|Mt|Dr|Mr|Mrs|Ms|Inc|Co|Ltd|Prof|Gen|Col|Lt|Capt|Sgt|Rev|U\.S|D\.C|Ft|No|Vol|Jan|Feb|Aug|Sept?|Oct|Nov|Dec)\.",t)),None);tk=tk[:ci+1] if ci is not None else tk;tk and re.search(r"['’]s$",tk[-1]) and tk.__setitem__(-1,tk[-1][:-2])
  while tk and (tk[0] in RL.STOPC or tk[0].lower() in CONN or not tk[0][:1].isupper()):a=y.index(tk[0],a)+len(tk[0]);tk=tk[1:]
  while tk and tk[-1].lower() in CONN:tk=tk[:-1]
  tk and not re.fullmatch(r"[A-Z]\.|(?:Jr|Sr|St|Mt|Dr|Inc|Co|Ltd)\.",tk[-1]) and tk[-1].endswith(".") and tk.__setitem__(-1,tk[-1][:-1])
  v=" ".join(tk);a=y.find(tk[0],a) if tk else -1
  if not tk or not tk[0][0].isupper() or len(tk)>6 or a<0 or len(v)<2 or all(t in RL.STOPC for t in tk):continue
  e=y.find(tk[-1],a)+len(tk[-1]);out.append((y[a:e],a,e))
 return out
def gen(qt:str,y:str,q:str)->list:
 if qt=="date":ql=q.lower();return [(m.group(0),k,m.start(),m.end()) for m in RL.DATE.finditer(y) for k in [fkey(qt,m.group(0),ql)] if k]
 if qt=="num":return RL.cands(qt,y,q) if re.search(r"\bhow (?:many|much|old|long|far|tall|big|high|deep|fast)\s+",q.lower()) else []
 E=ents(y);X=[(y[a:e2],a,e2) for (v,a,e),(v2,a2,e2) in zip(E,E[1:]) if qt=="place" and re.fullmatch(r",\s*",y[e:a2])]
 return [(v,AN_NORM(v),a,e) for v,a,e in E+X]
def AN_NORM(s:str)->str:return " ".join(re.sub(r"\b(a|an|the)\b"," ",re.sub(r"[^\w\s]"," ",fold(s).lower())).split())
RELW=((1,"play plays played playing portray portrays portrayed portraying star stars starred starring cast role roles actor actress"),(2,"voice voices voiced voicing"),(3,"sing sings sang sung singing singer singers recorded perform performs performed performer vocals vocalist song single"),(4,"write writes wrote written writing writer writers author authored penned composed composer lyricist lyrics novel poem"),(5,"direct directs directed director"),(6,"produce produces produced producer"),(7,"invent invented inventor discover discovered discoverer found founded founder create created creator design designed designer develop developed developer coined establish established built formulated proposed"),(8,"win wins won winner winners champion champions"),(9,"host hosts hosted presenter presented narrate narrated narrator"),(10,"own owns owned owner"),(11,"lead leads led leader coach coached captain"),(12,"release released releases premiere premiered premieres aired airs debut debuted publish published come came opened launched began begin start started"),(13,"die died dies death killed"),(14,"born birth"),(15,"located situated held filmed shot set headquartered based lies place"),(16,"married marry marries wife husband spouse"))
KREL={bh(w.encode()):r for r,ws in RELW for w in ws.split()};RFAM={2:1};AG=set(range(1,12))|{16};NOUN=set("actor actress role roles cast singer singers song single vocals vocalist writer writers author composer lyricist lyrics novel poem director producer inventor discoverer founder creator designer developer winner winners champion champions presenter narrator owner leader coach captain death birth place wife husband spouse premiere release debut star".split())
QGEN=set(toks("actor actress character role name person people movie film show series tv television song single album novel book original first voice part who what out"));QKIND=((r"\b(?:movie|film)s?\b","film"),(r"\b(?:tv|shows?|series|sitcom|season|episodes?|television)\b","tv"),(r"\b(?:song|single)\b","song"),(r"\balbum\b","album"),(r"\b(?:novel|book)\b","book"))
AKW=((r"film|movie","film"),(r"television|tv|series|sitcom|soap|miniseries|season|show","tv"),(r"song|single","song"),(r"album|\bep\b","album"),(r"novel|book","book"),(r"game","game"));AKT=re.compile(r"\(((?:\d{4}\s+)?(?:[\w'-]+\s+){0,3}?(?:film|movie|TV series|television series|miniseries|series|season \d+|sitcom|soap opera|song|single|album|EP|novel|book|video game|game|musical|play))\)\s*$",re.I);AKL=re.compile(r"(?i)\b(?:is|was)\s+(?:an?|the)\s+((?:\d{4}\s+)?(?:[\w'-]+\s+){0,6}?(?:film|movie|television series|tv series|miniseries|sitcom|soap opera|series|television show|show|song|single|album|novel|book|video game|game))\b")
CHARQ=re.compile(r"(?i)\b(?:who|what)\s+(?:did|does|do|is|was|will|has)\s+(?!the\b|an?\b)\S.{0,60}?\s+(?:play|playing|played|portray\w*|voice|voicing)\b|\b(?:what|which)\s+(?:character|role|part)\b")
ASR=re.compile(r"\s*(?:\([^()]{0,40}\)\s*)?(?:(?:is|was|co-?stars?|stars?|starred|starring|appears?|appeared|returns?|returned|reprises?|reprised|plays?|played|portrays?|portrayed|voices?|voiced|guest[- ]stars?|cast)\s+)?as\s+(?!(?:well|of|such|a\s+result|part|far|long|soon|much|if|though|opposed|to|follows?|many|early|late|and|a\s+whole|usual|possible|expected|before|mentioned|described|shown|seen|in|with|for|from|by)\b)(?:the\s+voice\s+of\s+)?");CHP=re.compile(r"(?:(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Prof|Lt|Col|Gen|Capt|Sgt|Rev|[A-Z])\.|[^,;.()\n]){1,60}?(?=\s*(?:[,;.()\n–—:]|\s(?:in|on|and|who|which|that|from|with|for|while|during|opposite|alongside|to|was|is|were|are|has|had|but|\w+ing)\s|$))")
def rid(w:str)->int:r=KREL.get(bh(w.lower().encode()),0);return RFAM.get(r,r)
def qparse(q:str)->tuple:
 ws=re.findall(r"[\w'’]+",q);rs=[x for x in map(rid,ws) if x];r=(rs[-1] if re.match(r"(?i)\W*(?:when|where|what\s+year|in\s+what\s+year)\b",q) else rs[0]) if rs else 0;ql=q.lower();drop={z for w in ws if rid(w)==r for z in toks(w)}|QGEN
 return r,"char" if r==1 and CHARQ.search(q) else "agent",[z for z in dict.fromkeys(toks(q)) if z not in QSTOP and z not in drop],next((v for p,v in QKIND if re.search(p,ql)),None),bool(re.search(r"(?i)\b(?:original|first)\b",q)),ql,pslot(q,r)
PQ=re.compile(r"(?i)\b(?:plays?|played|playing|portrays?|portrayed|voices?|voiced|(?:is|was|does|did|do)\s+the\s+voice\s+(?:of|for)|stars?\s+as|starred\s+as)\s+(?:the\s+(?:role|part|voice|character)\s+of\s+)?(.+?)(?:\s+(?:in|on|from|of|for)\s+(.+?))?\s*\??$");CQ=re.compile(r"(?i)\bwho\s+(?:did|does|do|is|was|will|has)\s+(.+?)\s+(?:play|playing|played|portray\w*|voice|voicing)\b")
def pslot(q:str,r:int)->tuple:
 m=(CQ.search(q) if CHARQ.search(q) else PQ.search(q)) if r==1 else None;c=m.group(1) if m else "";return set(toks(c))-QGEN,bool(re.search(r"[’']s\s+[a-z]",c))
def akind(t:str,lead:str)->tuple:
 m=AKT.search(t);k=(m.group(1) if m else "") or next((x.group(1) for x in [AKL.search(lead[:300])] if x),"");y=re.search(r"\b(1[89]\d\d|20\d\d)\b",k)
 return next((v for p,v in AKW if re.search(p,k.lower())),None) if k else None,int(y.group(1)) if y else None
def chain(E:list,p:int,y:str,back:bool=False)->list:
 out=[]
 for v,a,b in (E[::-1] if back else E):
  if (back and b>p) or (not back and a<p):continue
  if not re.fullmatch((r"\s*(?:(?:had|has|have)\s+been|were|was|are|is|all|both|also)?\s*" if back else r"\s*(?:the\s+)?") if not out else r"\s*(?:,|and|&|,\s*and)\s*",y[b:p] if back else y[p:a]):break
  out.append((v,a,b));p=a if back else b
 return out[::-1] if back else out
def lent(E:list,a:int,y:str)->tuple:return next(((v,x,b) for v,x,b in E[::-1] if b<=a and re.fullmatch(r"[\s,(]*(?:(?:is|was|are|were|has|had|have|been|who|which|also|later|originally|first|famously|previously|once|then|currently|and|being)\s+)*",y[b:a])),None)
def facts(y:str)->list:
 E=ents(y);F=[];rs=[(m,rid(m.group(0))) for m in re.finditer(r"[\w'’-]+",y)];pl=any(r==1 for _,r in rs) or bool(re.search(r"(?i)\b(?:regulars?|recurring|guest|appear\w*|characters?|interns?|cameos?|reprise\w*)\b",y));rp=re.search(r"\brespectively\b",y)
 cl=lambda a:re.split(r"[;:]",y[:a])[-1][-150:]
 for m,r in rs:
  if not r:continue
  a,b=m.span();w=m.group(0).lower();pb=re.match(r"(?:\s*,?\s*(?:and|&)\s+[a-z]+(?:ed|en))?\s+by\s+",y[b:b+40]) if r in AG else None;ag=chain(E,b+pb.end(),y) if pb else []
  if ag:le=lent(E,a,y) if r==1 else None;F.append({"r":r,"S":{"agent":ag,**({"char":[le]} if le else {})},"c":cl(a),"k":"pas"})
  st=chain(E,b+1,y) if r==1 and w in ("stars","starred","starring","star") else []
  st and F.append({"r":1,"S":{"agent":st},"c":cl(a),"k":"star"})
  le=lent(E,a,y) if r in AG and w not in NOUN and not pb and not st and w not in ("stars","starred","starring","star") else None
  if le and not re.match(r"\s*[:,\"“]",y[b:b+3]):
   am=re.match(r"\s*(?:the\s+(?:role|part|voice)\s+of\s+|as\s+|in\s+the\s+role\s+of\s+)?",y[b:]);ch=CHP.match(y,b+am.end()) if r==1 else None;F.append({"r":r,"S":{"agent":[le],**({"char":[(ch.group(0).strip(" \"“”"),ch.start(),ch.end())]} if ch else {})},"c":re.split(r"[;.]",y[b:b+100])[0],"k":"act"})
  dm=[d for d in RL.DATE.finditer(y) if (b<=d.start()<b+70 and ";" not in y[b:d.start()]) or (a-60<d.end()<=a and ";" not in y[d.end():a])]
  dm and F.append({"r":r,"S":{"date":[(d.group(0),d.start(),d.end()) for d in sorted(dm,key=lambda d:abs(d.start()-b))[:1]]},"c":cl(a)+" "+y[b:b+70],"k":"dat"})
  pm=re.match(r"\s+(?:[\w-]+\s+){0,2}?(?:in|at|on|near|from|inside|within)\s+",y[b:b+45]) if r in (15,14,13,12,7) else None;pc=chain(E,b+pm.end(),y) if pm else []
  pc and F.append({"r":r,"S":{"place":[pc[0]]+([(y[pc[0][1]:pc[1][2]],pc[0][1],pc[1][2])] if len(pc)>1 and re.fullmatch(r",\s*",y[pc[0][2]:pc[1][1]]) else [])},"c":cl(a),"k":"loc"})
 if rp and pl:
  vm=re.search(r"\b(?:cast|chosen|selected|play|played|portray\w*|voice[sd]?|star\w*)\s+(?:as|in\s+the\s+roles\s+of|the\s+roles\s+of)\s+",y);L=chain(E,vm.start(),y,True) if vm else [];R=chain(E,vm.end(),y) if vm else []
  len(L)==len(R)>1 and F.extend({"r":1,"S":{"agent":[l],"char":[c]},"c":"","k":"resp"} for l,c in zip(L,R))
 for v,a,b in E:
  am=ASR.match(y,b);ch=CHP.match(y,am.end()) if am and not rp and (pl or not y[:a].strip()) and not re.search(r"(?i)\b(?:play\w*|portray\w*|voic\w*|as|star\w*)\s+(?:the\s+(?:role|part)\s+of\s+)?$",y[max(0,a-40):a]) else None
  ch and F.append({"r":1,"S":{"agent":[(v,a,b)],"char":[(ch.group(0).strip(" \"“”"),ch.start(),ch.end())]},"c":re.split(r"\bas\b",y[ch.end():ch.end()+100])[0],"k":"as"})
  pm=re.match(r"\s*\(\s*((?:[A-Z][\w'’.-]*\s+){1,3}[A-Z][\w'’.-]*)\s*\)",y[b:b+45])
  pm and F.append({"r":1,"S":{"agent":[(pm.group(1),b+pm.start(1),b+pm.end(1))],"char":[(v,a,b)]},"c":re.split(r"\.\s|;",y[b+pm.end():b+pm.end()+100])[0],"k":"par"})
 return F
def fkey(qt:str,v:str,ql:str="")->str:
 d=RL.DATE.search(v) if qt=="date" else None;return (d.group(1) if d.group(1) not in ql else AN_NORM(v) if re.search(RL.MON,v) else "") if d else AN_NORM(v)
def fmatch(F:list,qp:tuple,qt:str,W:dict,tw:set)->tuple:
 qr,role,qe,_,_,ql=qp[:6];role={"date":"date","place":"place"}.get(qt,role);tot=sum(W.get(z,0.0) for z in qe) or 1.0;FX={};FW={}
 if not qr or qe==[] or (qt not in ("person","date","place") and role!="char"):return FX,FW
 for f in F:
  if f["r"]!=qr or role not in f["S"]:continue
  sm=(slot(f,role,qp[6][0] or set(qe),qp[6][1],W) if ("char" if role=="agent" else "agent") in f["S"] else 0.7) if f["r"]==1 and f["k"]!="star" else 1.0
  cx=set(toks(f["c"]+" "+" ".join(v for rr,vs in f["S"].items() if rr!=role for v,_,_ in vs)))|tw;m=round(sum(W.get(z,0.0) for z in qe if z in cx)/tot*(0.6 if f["k"]=="star" else 1.0)*sm,4);u=max([W.get(z,0.0) for z in qe if z not in cx] or [0.0])/(max(W.get(z,0.0) for z in qe) or 1.0)
  for v,a,b in f["S"][role] if role=="char" else [z for z in f["S"][role] if z[0][:1].isupper() or z[0][:1].isdigit()]:k=fkey(qt,v,ql);k and (m,-u)>FX.get(k,(-1,0,0,0,0,0))[::5] and FX.__setitem__(k,(m,v,a,b,f["k"],-u))
  for rr,vs in f["S"].items():rr!=role and rr in ("agent","char") and [FW.__setitem__(fkey(qt,v,ql),max(FW.get(fkey(qt,v,ql),0.0),m)) for v,_,_ in vs]
 return FX,FW
def slot(f:dict,role:str,qe:set,kin:bool=False,W:dict={})->float:
 o=f["S"]["char" if role=="agent" else "agent"][0][0];pm=re.match(r"(.+?)[’']s\s+([a-z][\w-]*)",o) if role=="agent" else None;nm=set(toks(pm.group(2) if pm else o))&qe;dm=re.match(r"\W*(?:[\w–-]+\W+){0,5}?([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*)*)[’']s?\s+([a-z][\w-]*)",(pm.group(1)+"'s "+pm.group(2)) if pm else f["c"][:60]) if role=="agent" else None
 dk=bool(dm and set(toks(dm.group(2)))<=qe and set(toks(dm.group(1)))&qe);return 1.0 if dk else 0.5+0.5*sum(W.get(z,1.0) for z in nm)/(sum(W.get(z,1.0) for z in qe) or 1.0) if nm and not kin else 0.5
NW=dict(zip("one two three four five six seven eight nine ten eleven twelve".split(),range(1,13)));OW="first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()
def snum(q:str):
 m=re.search(r"(?i)\b(season|series|episode|part|chapter|volume)\s+(\d{1,2}|%s)\b"%"|".join(NW),q);n=(int(m.group(2)) if m.group(2).isdigit() else NW[m.group(2).lower()]) if m else 0;k=m.group(1).lower() if m else ""
 return re.compile(r"(?i)\b%s\s+(?:%d|%s)\b|\b(?:%s|%d(?:st|nd|rd|th))\s+%s\b"%(k,n,[*NW][n-1] if n<=12 else "x",OW[n-1] if n<=12 else "x",n,k)) if m and n else None
def ss(x:str)->list:
 s=RL.sents(x);s=s+[l.strip() for l in x.split("\n") if re.match(r"\s*[A-Z][^\n]{2,60}?\s+as\s+\S",l) and not any(l.strip() in z for z in s)];return s or [z for l in x.split("\n") for z in re.split(r"(?<=[.!?])\s+(?=[A-Z])",l.strip()) if 15<len(z)<600]
def subseq(a:list,b:list)->bool:
 it=iter(b);return all(any(y.startswith(x) for y in it) for x in a)
HON=set("sir dame dr lord lady jr sr ii iii iv mr mrs ms".split())
def ent(v:str,T)->tuple:
 if v in _TY:return _TY[v]
 d=T.doc(v) if T is not None else None;g=KR.lead(*d)[2] or "?" if d else None;m=re.match(r"\s*([^()\n,;:\"]{3,80}?)\s*\(([^()]{0,120})",d[1][:400]) if d else None;bn=re.search(r"\bborn\s+((?:[A-Z][\w.'’-]*\s+)+[A-Z][\w'’-]*)\s*[;,)]",m.group(2)+")") if m else None;vw=v.split();f=next((z for z in [bn and bn.group(1),m and m.group(1).strip()] if z and z!=v and len(z.split())<=6 and z[:1].isupper() and len(z.split())>len(vw) for c in [[fold(w).lower() for w in z.replace(",","").split() if w.strip(".").lower() not in HON]] if len(c)>len(vw) and c[-1].startswith(fold(vw[-1]).lower()) and subseq([fold(w).lower() for w in vw[:-1]],c[:-1])),None) if len(vw)>1 else None
 return _TY.setdefault(v,(g,f))
OCCS=((r"\b(?:play|plays|played|playing|portray\w*|star\w*|cast)\b",r"actor|actress|comedian|entertainer|performer|model|personality|presenter|host|singer|rapper"),(r"\bvoice[sd]?\b",r"actor|actress|voice|comedian|singer|performer"),(r"\b(?:sing|sings|sang|sung|perform\w*|record\w*)\b",r"singer|band|musician|rapper|group|duo|vocalist|songwriter|artist|trio|quartet|performer"),(r"\b(?:wrote|write|writes|written|author\w*|compos\w*|lyric\w*|pen\w*)\b",r"writer|author|novelist|poet|songwriter|composer|lyricist|playwright|screenwriter|journalist|band|singer|musician|rapper|producer"),(r"\bdirect\w*\b",r"director|filmmaker|producer|screenwriter"))
def occf(q:str,d)->dict:
 f=[o for a,o in OCCS if re.search(a,q,re.I)];return {"occ":float(bool(f and d and d!="?" and any(re.search(r"(?i)\b(?:%s)"%o,d) for o in f))),"occna":float(bool(f) and (not d or d=="?"))}
def _typeok(qt:str,v:str,T)->int:
 if qt not in ("person","place") or T is None:return 0
 d=ent(v,T)[0];return 1 if not d or d=="?" else 0 if (RL.PERSONY if qt=="person" else RL.PLACEY).search(d) and not RL.NEGD.search(d) else 2
AKA=re.compile(r"(?i)\b(?:also\s+known\s+as|better\s+known\s+as|known\s+professionally\s+as|professionally\s+known\s+as|known\s+as|nicknamed|stage\s+name|born|n[ée]e|formerly|real\s+name|birth\s+name|or\s+simply)\s+(?:the\s+)?[\"“']?((?:[A-Z][\w.'’-]*\s*){1,6})")
def alias(v:str,T=None)->set:
 d=(T or RL._kx()[1]).doc(v);x=re.sub(r"\s+"," ",d[1][:700]) if d else "";m=re.match(r"\s*([^()\n,;:\"]{2,80}?)\s*\(",x);return {v,*([d[0],re.sub(r"\s*\([^()]*\)$","",d[0])] if d else []),*([m.group(1).strip()] if m else []),*(z.strip() for z in AKA.findall(re.split(r"(?<=[a-z]{3}\.)\s",x,maxsplit=2)[0] if x else "") if not re.match(RL.MON,z))}-{""}
def span(qt:str,v:str,T=None)->str:
 f=ent(v,T or RL._kx()[1])[1] if qt=="person" else None;return "%s (full name %s)"%(v,f) if f else v
def occ(q:str,k:int=None)->tuple:
 I=ix();qt=RL.qtype(q);h=I.search(q,k or int(os.environ.get("AMNI_RKNOW_K","30")));O=[];P=[]
 if not h:return qt,O,P
 qa=set(toks(q));qw=[w for w in dict.fromkeys(toks(q)) if w not in QSTOP];W={w:I.idf(w) for w in qw};tot=sum(W.values()) or 1.0;F=RL.fam(q);qn=set(RL.words(q.encode()));yrs=set(re.findall(r"\b(?:1[0-9]|20)\d\d\b",q));cache={};qi=[w[0] for w in re.findall(r"[a-z0-9]+",fold(q).lower())];ini={"".join(qi[i:j]) for i in range(len(qi)) for j in range(i+2,min(len(qi),i+6)+1)}
 qc={z for z in qw if W[z]>=QCI};sn=snum(q);pm=POSS.search(q);pt=set(toks(pm.group(1))) if pm else set();lks=getattr(I,"lk",set());qp=qparse(q);AK={}
 for r,(sc,pid) in enumerate(h):
  t,x,g=I.text(pid,cache);tw=set(toks(t));P.append((sc,pid,t,x));fp=int(I.first[g]);ak=AK.get(g) or AK.setdefault(g,akind(t,x if fp==pid else I.text(fp,cache)[1]));kb,ko=(float(bool(qp[3] and ak[0] and ak[0]!=qp[3])),float(bool(qp[3] and ak[0]==qp[3])))
  tm=re.match(r"(.+?) \((.+?)(?:['’]s?)? (?:song|single)\)$",t) if qp[0]==3 else None
  for y in ss(x):
   sw=set(toks(y));cov=(sum(W[z] for z in qw if z in sw)+0.5*sum(W[z] for z in qw if z in tw and z not in sw))/tot;FX,FW=fmatch(facts(y)+([{"r":3,"S":{"agent":[(tm.group(2),0,0)]},"c":tm.group(1),"k":"ttl"}] if tm and fp==pid else []),qp,qt,W,tw) if qp[0] else ({},{})
   if (cov<0.3 and max([z[0] for z in FX.values()] or [0])<0.5) or (yrs and not all(z in y+" "+t for z in yrs)):continue
   yl=fold(y).lower();pos=[(m.start(),m.end(),z) for z in qw if z in sw for m in re.finditer(r"\b"+re.escape(z),yl)]
   C=gen(qt,y,q);ks={c[1] for c in C}
   for v,key,a,b in C+[(z[1],k,z[2],z[3]) for k,z in FX.items() if k not in ks]:
    ct=set(toks(v));qo=1.0 if v.isupper() and 1<len(v)<=6 and v.lower() in ini else len(ct&qa)/max(1,len(ct))
    if not ct or qo>=1.0:continue
    d=min([a-p0 if p0<a else max(0,p0-b) for p0,_,_ in pos] or [200]);lc=sum(W[z] for z in {z for p0,p1,z in pos if p1>=a-80 and p0<=b+80})/tot
    qn2=lambda z:bool(set(toks(z))&(qc-tw));am=ASV.match(y[b:b+40]);ae=" ".join(v for v,_,_ in ents(y[b+am.end():b+am.end()+50])[:1]) if am else "";bm=BYV.search(y[max(0,a-40):a]);be=" ".join(v for v,_,_ in ents(y[max(0,a-110):a])[-1:]) if bm else "";O.append({"key":key,"v":t+", "+v if qt=="date" and re.fullmatch(r"\d{4}",v) and re.fullmatch(r"(?:%s) \d{1,2}"%RL.MON,t) else v,"g":g,"t":t,"y":y,"r":r,"pw":sc/h[0][0],"cov":cov,"lc":lc,"d":d,"qo":qo,"rel":RL.rel(qt,y[max(0,a-100):b+100],a-max(0,a-100),b-max(0,a-100),F,qn) if F is not None else None,"pas":bool(am) and qn2(ae) and not set(toks(ae))&pt,"pasx":bool(pt and ((am and set(toks(ae))&pt) or (bm and set(toks(be))&pt))),"pby":bool(bm) and qn2(y[max(0,a-90):a]) and not (pt and set(toks(be))&pt),"ppar":y[max(0,a-2):a].strip()=="(" and qn2(y[max(0,a-50):a]),"lnk":g in lks,"lead":fp==pid,"fx":FX[key][0] if key in FX else 0.0,"fu":-FX[key][5] if key in FX else 1.0,"fk":FX[key][4] if key in FX else "","fw":FW.get(key,0.0),"kb":kb,"ko":ko,"ay":ak[1],"sbad":float(bool(sn and not sn.search(y+" "+t)))})
 Y=[o["ay"] for o in O if o["fx"]>=0.5 and o["ay"]];lo,hi=(min(Y),max(Y)) if Y else (0,0)
 D=[int(o["key"]) for o in O if qt=="date" and o["fx"]>=0.5 and not o["kb"] and o["key"].isdigit()]
 for o in O:o.update({"dmin":float(bool(D) and o["key"]==str(min(D))),"yorig":float(bool(qp[4] and lo<hi and o["ay"]==lo)),"ynew":float(bool(not qp[4] and lo<hi and o["ay"]==hi)),"yold":float(bool(not qp[4] and lo<hi and o["ay"]==lo))})
 return qt,O,P
HN=re.compile(r"(?i)\b(?:what|which)\s+(?:(?:is|was|are|were)\s+(?:the\s+)?)?(?:kind\s+of\s+|type\s+of\s+|sort\s+of\s+)?([a-z]+)");HNX=set(toks("is was are were did does do has had can will would name kind type one part thing year day date time the a an of"))
def feats(qt:str,O:list,T=None,n:int=int(os.environ.get("AMNI_RKNOW_N","25")),q:str="")->list:
 C=collections.defaultdict(list)
 for o in O:C[o["key"]].append({**o,"s":o["pw"]**2*o["cov"]**2*(0.2+o["lc"])})
 for kk in sorted(C,key=len):
  sup=[j for j in C if j!=kk and set(kk.split())<set(j.split())];len(sup)==1 and C[sup[0]].extend(C.pop(kk))
 R=[];qa=set(toks(q))
 for kk,o in C.items():
  per={};[per.__setitem__(z["g"],max(per.get(z["g"],0.0),z["s"])) for z in o];f=collections.Counter(z["v"] for z in o);f=(not kk.isdigit() and collections.Counter({z:c for z,c in f.items() if len(z.split())>1})) or f;tf=max(f.values());fv=collections.Counter(z["v"] for z in o if z["fx"]>=0.5) or f;v=max(fv,key=lambda z:(len(z),fv[z])) if kk[:1].isdigit() and RL.DATE.fullmatch(max(fv,key=len)) else max([z for z in f if f[z]>=max(2,tf/3)] or f,key=lambda z:(len(z),f[z]) if f[z]>=max(2,tf/3) else (f[z],len(z)));bz=max([z for z in o if z["v"]==v],key=lambda z:(round(z["fx"],1),bool(z.get("lnk")),len(set(toks(z["t"]))&qa),bool(z.get("pas") or z.get("pby")),round(z["cov"],1),z["s"]));rl=[z["rel"] for z in o]
  R.append({"key":kk,"v":v,"t":bz["t"],"y":bz["y"],"raw":sum(per.values()),"sup":len(per),"x":{"pw":max(z["pw"] for z in o),"r":min(z["r"] for z in o)/30,"cov":max(z["cov"] for z in o),"lc":max(z["lc"] for z in o),"d":math.log1p(min(z["d"] for z in o))/5,"sup":math.log1p(len(per)),"n":math.log1p(len(o)),"relhi":float(any(x is not None and x>=1.0 for x in rl)),"rello":float(all(x is not None and x<0.4 for x in rl)),"relna":float(rl[0] is None),"qover":min(z["qo"] for z in o),"ntok":len(kk.split())/4,"top":float(any(z["r"]==0 for z in o)),"title":float(any(AN_NORM(z["t"]).startswith(kk) for z in o)),"pas":float(any(z.get("pas") for z in o)),"pby":float(any(z.get("pby") for z in o)),"ppar":float(any(z.get("ppar") for z in o)),"pasx":float(any(z.get("pasx") for z in o)),"lnk":float(any(z.get("lnk") for z in o)),"lead":float(any(z.get("lead") for z in o)),"lnklead":float(any(z.get("lnk") and z.get("lead") for z in o)),"fx":max(z["fx"] for z in o),"fxu":min(z["fu"] for z in o if z["fx"]==max(y["fx"] for y in o)),"fxa":float(any(z["fx"]>=0.5 and z["fu"]==0 for z in o)),"fx5":float(max(z["fx"] for z in o)>=0.6),"fxl":max(z["fx"]*z["lnk"] for z in o),"fxn":math.log1p(len({z["g"] for z in o if z["fx"]>=0.5})),"fw":max(z["fw"] for z in o),"fpar":float(any(z["fx"]>=0.5 for z in o) and all(z["fk"] in ("par","star","dat") for z in o if z["fx"]>=0.5)),**{k:float(any(z[k] and z["fx"]>=0.5 for z in o)) for k in ("kb","ko","yorig","ynew","yold","dmin")},"sbad":float(all(z["sbad"] for z in o)),**{"q_"+x:float(qt==x) for x in ("date","num","person","place","other")}}})
 R=sorted(R,key=lambda z:-z["raw"]);R=R[:n]+[z for z in R[n:] if z["x"]["fx"]>=0.5][:5];tot=sum(z["raw"] for z in R) or 1.0;T=T if T is not None or not R else RL._kx()[1];hm=HN.search(q);hn=set(toks(hm.group(1)))-HNX if hm else set();qw=[w for w in dict.fromkeys(toks(q)) if w not in QSTOP];I=ix();W={w:I.idf(w) for w in qw};wt=sum(W.values()) or 1.0
 for i,z in enumerate(R):ty=_typeok(qt,z["v"],T);gl=set(toks(ent(z["v"],T)[0] or "")) if T is not None else set();z["x"].update({"hnv":float(bool(hn&set(toks(z["v"])))),"hng":float(bool(hn&gl)),"hna":float(bool(hn)),"gq":sum(W[w] for w in qw if w in gl)/wt,"bias":1.0,"s0":math.log1p(10*z["raw"]),"share":z["raw"]/tot,"rk":1/(1+i),"tok":float(ty==0 and qt in ("person","place")),"tmiss":float(ty==1),"tbad":float(ty==2),"actqo":float(bool(ACT.search(q)) and z["x"]["qover"]>0),**occf(q,ent(z["v"],T)[0] if T is not None else None),"tchar":float(bool(T is not None and ACT.search(q) and re.search(r"(?i)character|fictional",(ent(z["v"],T)[0] or "")))),"gap":(z["raw"]-(R[1]["raw"] if i==0 and len(R)>1 else R[0]["raw"]))/max(z["raw"],1e-9)})
 return R
def gbx(R:list,m:dict=None)->np.ndarray:
 m=m or GBM;M=np.array([[z["x"].get(k,0.0) for k in m["feat"]] for z in R],np.float64);ri=[m["feat"].index(k) for k in m["rel"]];return np.hstack([M,M[:,ri]-M[:,ri].max(0)])
def gbp(X:np.ndarray,m:dict=None)->np.ndarray:
 m=m or GBM;n=np.broadcast_to(m["roots"],(len(X),len(m["roots"]))).copy();a=np.arange(len(X))[:,None]
 for _ in range(int(m["depth"])):f=m["f"][n];n=np.where(f<0,n,np.where(X[a,np.maximum(f,0)]<=m["t"][n],m["l"][n],m["r"][n]))
 return 1/(1+np.exp(-(float(m["base"])+m["v"][n].sum(1))))
def rank(qt:str,O:list,wts:dict=None,T=None,q:str="")->list:
 R=feats(qt,O,T,q=q);S=gbp(gbx(R)) if GBM is not None and not wts and R else [1/(1+math.exp(-sum((wts or WTS).get(k,0.0)*v for k,v in z["x"].items()))) for z in R]
 for z,sc in zip(R,S):z["s"]=float(sc);z["cov"]=z["x"]["cov"]
 return sorted(R,key=lambda z:-z["s"])
def candidates(q:str,k:int=None)->tuple:
 qt,O,P=occ(q,k);return qt,rank(qt,O,q=q),P
CF=("bias","p1","gap","p2","share","sup","cov","relhi","pas","pby","tok","tmiss","lnk","lead","q_date","q_num","q_person","q_place","q_other","ncand","top","qover","s0","occ","occna","q_act","q_won","fx","fxa","fxu","fw","kb","ko","nfx","fxr","q_rel","sbad","vorig","pstr","rstr")
CWTS=dict(zip(CF,json.loads(os.environ.get("AMNI_RKNOW_CWTS","null") or "null") or (GBM or {}).get("cw") or (-3.6662,0.3683,0.3651,0.0032,0.2542,0.2163,0.1386,0.3132,0.2674,0.1361,0.4505,-0.2512,0.2423,0.0822,0.2368,-0.0141,0.2428,-0.2956,-0.1706,0.0722,0.2527,-0.3748,0.2782,0.459,0.0569,-0.0053,-0.1548,0.3842,0.3024,-0.2298,-0.0303,0.0041,0.0145,-0.2638,0.3313,0.0243,-0.1219,-0.1215,0.2282,-0.0343)))
def qfeat(qt:str,R:list,q:str)->dict:
 c=R[0]["x"];p2=R[1]["s"] if len(R)>1 else 0.0;return {**{k:c.get(k,0.0) for k in CF},"bias":1.0,"p1":R[0]["s"],"gap":R[0]["s"]-p2,"p2":p2,"ncand":len(R)/10,"q_act":float(bool(ACT.search(q))),"nfx":math.log1p(sum(z["x"]["fx"]>=max(0.5,c["fx"]-0.05) for z in R[1:])),"vorig":float(c["fx"]>=0.5 and any(z["x"]["fx"]>=0.5 for z in R[1:]) and ((qparse(q)[4] and not c["yorig"]) or (bool(NEWQ.search(q)) and not c["ynew"]))),"fxr":c["fx"]-max([z["x"]["fx"] for z in R[1:]] or [0.0]),"q_rel":float(qparse(q)[0]>0),"pstr":float(qparse(q)[0]==1 and c["fx"]>=0.9 and c["fxa"]>0 and not any(z["x"]["fx"]>=0.5 for z in R[1:])),"rstr":float(qparse(q)[0]>1 and c["fx"]>=0.9 and c["fxa"]>0 and not any(z["x"]["fx"]>=0.5 for z in R[1:])),"q_won":float(bool(re.search(r"(?i)\b(?:won|win|wins|winner|champion\w*)\b",q)) and not re.search(r"\b(?:1[0-9]|20)\d\d\b",q))}
NEWQ=re.compile(r"(?i)\b(?:the\s+new|new\s+(?:movie|film|series|show|version|season|one|remake))\b(?!\s+(?:york|zealand|jersey|orleans|england|mexico|hampshire|delhi|testament|world|deal|south|brunswick|guinea|caledonia|amsterdam|hope|kids|girl|edition))")
def veto(e:dict,f:dict,q:str)->bool:return bool(e["x"].get("kb") or e["x"].get("sbad") or f.get("vorig") or f.get("fxr",0)<=-0.2 or (f.get("q_act") and e["x"].get("tchar")) or (qparse(q)[0]==1 and e["x"].get("fx",0)<0.8) or (qparse(q)[0]>0 and e["x"].get("fx",0)<float(os.environ.get("AMNI_RKNOW_VFX","0.5"))) or (f.get("q_person") and e["x"].get("tbad")) or (POSS.search(q) and not e["x"].get("fxa")))
def decide(qt:str,R:list,gate:dict=GATE,q:str="")->dict:
 if not R:return None
 e=R[0];f=qfeat(qt,R,q);c=1/(1+math.exp(-sum(CWTS.get(k,0.0)*v for k,v in f.items())));vt=veto(e,f,q);ok=c>=gate["c"] and not SOFT.search(e["v"]) and not vt
 return {"veto":vt,"span":e["v"],"title":e["t"],"url":KR.wurl(e["t"]),"evidence":e["y"],"qtype":qt,"conf":round(c,4),"score":round(e["s"],4),"sup":e["sup"],"cov":round(e["cov"],3),"ok":ok,"license":"CC BY-SA 4.0","reply":"%s (Source: Wikipedia, %s)."%(span(qt,e["v"]),e["t"] if not SOFT.search(e["t"]) else "article")}
TITQ=(re.compile(r"(?i)^\W*(?:who|what\s+\w+)\s+(?:\w+\s+){0,2}?(?:sings?|sang|sung|raps?|rapped|wrote|writes?|written|recorded|performed|made|produced|composed|scripted|said|says|did|does)\s+(?:the\s+(?:song|lyrics|theme)\s+(?:to\s+|for\s+|of\s+)?)?(?P<t>.+?)\??$"),re.compile(r"(?i)^\W*when\s+(?:did|was|were|does|is)\s+(?:the\s+(?:song|movie|film|album|show|book)\s+)?(?P<t>.+?)\s+(?:come\s+out|released|written|recorded|made|published|premiered?|air)\??$"),re.compile(r"(?i)^\W*who\s+(?:plays?|played|voices?|voiced|sings?|sang)\s+.+?\s+(?:in|on)\s+(?P<t>.+?)\??$"),re.compile(r"(?i)^\W*(?:where|what|when|who)\b.*?\b(?:term|phrase|saying|expression|quote|song|poem|book|novel|movie|film|show|series|album|hymn|slogan)\s+(?P<t>.+?)\s+(?:come\s+from|mean|originate|take\s+place|set|filmed|written|released|come\s+out)\??$"))
P1=re.compile(r"(?i)(?<!\.)(?<!\b[a-z] )\bi\b(?!\.|\s*-?\d)|\b(?:me|my|myself|i'm|im|i've|i'd|i'll|let's)\b");PW=re.compile(r"(?i)\b(?:we|our|ours|ourselves|us)\b");P2=re.compile(r"(?i)\b(?:you|your|yours|yourself|you're|youre|you've|you'll|you'd|u|ur)\b");ADAM=re.compile(r"(?i)\b(?:did|were)\s+(?:you|u)\b|(?:^|[.,!?;]\s*)(?:can|could|would|will|do|did|are|were|have|may)\s+(?:you|u)\b|\b(?:are|were|would|will|made|make|built|created|trained|love|like|hate|thank|meet|with|about|than|miss)\s+(?:you|u)\b(?!\s+(?:have\s+to|need\s+to|get|pronounce|say|spell|call|go|be|see|find|buy|use|wear|vote|serve|drink|carry|travel|run|graduate|have\s+a))|\b(?:you|u)\s+(?:are|were|think|want|like|love|feel|know|mean|remember|dream|prefer|doing|from|live|made|said|told|just|agree|believe|recommend|suggest|guys|enjoy|ever|yourself)\b|\b(?:you|u)\s+do\s*\??\s*$|\b(?:can|could)\s+(?:you|u)\s+(?:do|help|tell|give|show|find|search|explain|predict|try|suggest|recommend|make|write)\b|\byour(?:self)?\s+(?:name|favou?rite|hobbies|hobby|opinion|day|age|creator|family|thoughts|view|take|life|job|weekend|texts|own|team|answer|father|mother|feelings|purpose|goal)\b|^\W*(?:you|your|you're|u|ur)\b");CHAT=re.compile(r"(?i)\b(?:best|better|worst)\s+(?:\w+\s+){0,2}?(?:ever|to\s+(?:buy|learn|get|do|use|eat|watch|read|try|start)|way\s+to|for\s+(?:me|a\s+beginner|beginners))\b|\bwhich\s+is\s+(?:the\s+)?(?:best|better)\b|^\W*(?:what|which)(?:'s|\s+is|\s+are)?\s+(?:the\s+)?(?:best|better)\s+\w+(?:\s+\w+)?\s*\??\s*$|\bwho\s+would\s+win\b|\b(?:better|best)\b.*\bor\b|\bmeaning\s+of\s+life\b|\bwhat\s+time\s+is\s+it\b|\bfun\s+fact\b|\bweather\s+(?:like|today|tomorrow|tonight|now)\b|\b(?:stop|start)\s+raining\b|\bthe\s+point\s+of\s+(?:all|it|this|life)\b|^\W*who\s+cares\b|^\W*what\s+is\s+love\b|\bhow\s+far\s+is\s+too\s+far\b|\bget\s+over\s+(?:a|my|the)\s+breakup\b|\bstay\s+motivated\b|\bmake\s+friends\b|\ba\s+good\s+friend\b|\ba\s+person\s+need\b|^\W*what\s+a\b|\bany\s+good\b|\bgood\s+way\b|\bbest\s+way\b|\brecommend(?:ations?)?\b|\badvice\b|\btips\b|\bsuggest(?:ions?)?\b|\bin\s+your\s+opinion\b|\bshould\s+(?:i|we)\b|\b(?:can|could|would|will)\s+(?:i|we)\b|\bany\s+ideas\b|\bideas\s+for\b|\bgood\s+idea\b|\ba\s+good\s+(?:way|idea|hobby|name|gift|place|time|book|movie|show|restaurant|strategy)\b");NQS=re.compile(r"(?i)^\W*(?:who|whom|whose|who's|whos|what|what's|whats|when|where|where's|which|why|how|in\s+(?:what|which)|on\s+(?:what|which)|at\s+(?:what|which)|from\s+(?:what|which)|by\s+(?:what|which|whom)|name\s+(?:the|a|an|\d|some|two|three|four|five)|list\s+(?:the|of|all|some|\d|two|three|four|five)|number\s+of|total\s+number)\b");INW=re.compile(r"(?i)\b(?:who|whom|whose|what|which|when|where|how\s+many|how\s+much)\b");CHATD=re.compile(r"(?i)\b(?:today|tonight|tomorrow|yesterday|lately|right\s+now|this\s+(?:morning|week|weekend|evening|afternoon)|last\s+night|next\s+(?:week|month|weekend)|again|so\s+(?:much|tired|good|bad|bored|hard|happy|sad|excited|stressful)|really|amazing|awesome|terrible|hilarious|funny|stressful|stressed|loud|boring|bored|tired|lol|lmao|haha|omg|wow|cool|nice|okay|ok|sounds|yeah|yes|nope|sure|maybe|anyway|never\s+mind|oh|ooh|hmm|ugh|yay|damn|seriously|literally|kinda|gonna|gotta|crazy|fun|delicious|tasty|cute|annoying|excited|exhausted|hungry|too\s+(?:strong|hot|cold|late|much|early|long)|makes\s+sense|long\s+day|at\s+work|work\s+is)\b");CSTART=re.compile(r"(?i)^\W*(?:just|so|finally|still|feeling|thinking|planning|learning|going|trying|started|starting|got|getting|need|needs|want|wants|love|loving|hate|can't|cant|can|could|would|will|shall|may|should|it's|there's|such\s+a|well|really|wow|oh|ah|sure|good|bad|happy|sad|morning|night|hey|hi|hello|please|make|tell|give|show|help|let|say|quiz|write|explain|describe|compare|define|translate|summarize|imagine|pretend|guess|yes|no|yeah|nope|ok|okay|thanks|thank|sorry|great|nice|cool|anyway|best|worst)\b");YN=re.compile(r"(?i)^\W*(?:is|are|was|were|does|do|did|has|have|had)\s+");EVAL=re.compile(r"(?i)\b(?:okay|ok|weird|normal|bad|wrong|rude|fine|safe|healthy|worth|too\s+late|possible\s+to|good\s+idea|smart|real|alive|conscious|robot|smarter|better|worse)\b");WT=re.compile(r"(?i)\b(?:sings?|sang|sung|wrote|written|plays?|played|voiced?|starr?\w*|featured|shouts|sponsor|directed|produced|released|filmed|recorded|performed|made|cast|characters?|song|movie|film|show|album|series|episode|season|book|novel|poem|musical|lyrics)\b.*?\b(?:in|on|from|of|for|song|movie|film|show|album|sings?|sang|wrote|called|titled|named)\s+(?P<t>.+)$|^\W*(?P<u>.+?)\s+(?:original\s+artist|release\s+date|lyrics|cast|album|soundtrack|meaning|singer|who\s+sings|coming\s+out|come\s+out|came\s+out|first\s+(?:start|air|come\s+out)|filmed|become\s+our\s+motto|artist|number\s+of\s+episodes|how\s+many\s+(?:episodes|seasons))\b|^\W*(?:song|movie|film|album|book|novel|poem|tv\s+show)\s+(?P<v>.+)$");CTX=set("in on from of by sings sang sung sing song called titled named wrote written album movie film show series and play musical poem novel book is was does did".split());CHATW=re.compile(r"(?i)\b(?:tonight|tomorrow|this\s+weekend|dinner|lunch|breakfast|vacation|later|go\s+for|hang\s+out|talk\s+about)\b");TASKOK=re.compile(r"(?i)\W*(?:list|name|describe|write|state|give|explain|define|identify|mention)\s+(?!(?:me|us)\b|an?\s+(?:poem|story|joke|song|letter|essay|haiku|email|speech|paragraph|list|summary)\b)")
def istitle(ph:str)->bool:
 z=RL._kx()[1].z[0][1]["h"];h=np.uint64(KR.th64(ph));j=int(np.searchsorted(z,h));return j<len(z) and z[j]==h
def pr(x:str)->bool:return bool(P1.search(x) or P2.search(x) or PW.search(x))
def mask(q:str)->str:
 m=next((m for T in TITQ for m in [T.match(q)] if m and pr(m.group("t")) and len(m.group("t").split())>=2),None);s=(q[:m.start("t")]+" XTITLE "+q[m.end("t"):]) if m else q;m=WT.search(s);g=next(k for k in "tuv" if m.group(k)) if m else None;s=(s[:m.start(g)]+" XTITLE "+s[m.end(g):]) if m and not ADAM.search(m.group(g)) and (pr(m.group(g)) or not NQS.match(m.group(g))) else s;w=re.sub(r"[\"“][^\"”]{2,80}[\"”]"," XTITLE ",s).split();i=0;out=[]
 while i<len(w):n=next((n for n in range(min(8,len(w)-i),2,-1) if i>=1 and w[i-1].lower() in CTX and pr(" ".join(w[i:i+n])) and istitle(" ".join(w[i:i+n]))),0);out.append("XTITLE" if n else w[i]);i+=n or 1
 return " ".join(out)
def factq(q:str,wide:bool=False)->bool:
 from amni.compute.rchat import act
 s=q.strip();b=s.rstrip("?").strip()
 if len(b.split())<3 or re.search(r"[a-z]{3,}[.!;]\s+[A-Z]",s) or re.search(r"!\s*$",s) or (not wide and TEMP.search(q)):return False
 r=mask(s);a=act(r)
 if (a in ("greet","howru","bye") and not NQS.search(r)) or (a=="thanks" and re.search(r"(?i)thank|thx|appreciat",r)) or (a=="task" and not TASKOK.match(r)) or a in ("feel","advice","opinion") or P1.search(r) or ADAM.search(r) or CHAT.search(r) or RL.PERSONAL.search(r) or (istitle(b) and NQS.search(b)):return False
 if NQS.search(r) or INW.search(r):return not (PW.search(r) and CHATW.search(r))
 return (bool(re.search(r"\bor\b",b)) or not EVAL.search(r)) and not re.match(r"(?i)\W*\w+\s+(?:it|there\s+any|anyone|anybody|someone|something)\b",b) if YN.search(b) else (not CSTART.search(r) or bool(TASKOK.match(r))) and not CHATD.search(r)
def askq(q:str)->bool:return factq(q,True)
C2=float(os.environ.get("AMNI_RKNOW_C2",(GBM or {}).get("c2",0.3)));HEDGE={"high":"%s%s (Source: Wikipedia, %s).","medium":"%sI believe it's %s (Source: Wikipedia, %s).","low":"%sMy best guess is %s, but I'm unsure (Source: Wikipedia, %s)."}
def near(q:str,P:list)->tuple:
 qw=[w for w in dict.fromkeys(toks(q)) if w not in QSTOP];I=ix();W={w:I.idf(w) for w in qw};return max(((sum(W[w] for w in qw if w in set(toks(y)))/(sum(W.values()) or 1.0),y,t) for _,_,t,x in P[:3] for y in ss(x)[:6]),default=(0.0,"",""))
def ask(q:str,hedge:bool=True)->dict:
 if not askq(q):return None
 qt,R,P=candidates(q);r=decide(qt,R,q=q);tp="As of my Wikipedia snapshot, " if TEMP.search(q) else ""
 if r:k="high" if r["ok"] and not tp else "medium" if r["ok"] or r["conf"]>=C2 else "low";e=(span(qt,r["span"]),r["title"] if not SOFT.search(r["title"]) else "article");r.update(tier=k,reply=HEDGE[k]%(tp.rstrip(", ")+": " if tp and k=="high" else tp,*e))
 elif P and hedge:c,y,t=near(q,P);r={"span":None,"title":t,"url":KR.wurl(t),"evidence":y,"qtype":qt,"conf":0.0,"ok":False,"tier":"none","license":"CC BY-SA 4.0","reply":"I couldn't find a direct answer, but here's the closest fact I found: %s (Source: Wikipedia, %s)."%(y.rstrip(". "),t)} if y and c>=0.5 else None
 return r if r and (hedge or r["tier"]=="high") else None
