import os,re,json,math,unicodedata,collections
import numpy as np,pyarrow as pa
import amni.compute.ray_learn as RL
import amni.compute.ks_retrieve as KR
IXD=os.environ.get("AMNI_RKNOW_INDEX",os.path.join(KR.KS,"index_rknow","wiki_v2"));STOP=set("a an the of and or in on at to for by with from as is was are were be been it its that this which who he she his her they their has had have not but also into than then there".split());QSTOP=set("what when where which whom whose why how many much did doe do name called call".split())
K1,B,DFMAX,M64=0.9,0.4,300000,(1<<64)-1;LB=float(os.environ.get("AMNI_RKNOW_LB","3.0"));LIDF=float(os.environ.get("AMNI_RKNOW_LIDF","9.0"));QCI=6.0;TEMP=re.compile(r"(?i)\b(latest|current|currently|now|newest|next|recent|recently|today|this year|upcoming|so far|to date|as of)\b");BYV=re.compile(r"(?i)\b(?:played|portrayed|voiced|performed|sung|recorded|written|directed|composed|produced|created|founded|invented|discovered|built|designed|painted|developed|won|hosted|narrated|owned|led)\s+by\s*$");FACTQ=re.compile(r"(?i)^\W*(?:(?:and|so|but|ok|okay|hey|um|well)\W+)?(?:who|whom|whose|what|what's|when|where|which|how\s+(?:many|much|old|long|far|tall|big|high|deep|fast|often|large|wide)|(?:in|on|at|from|by|for|to)\s+(?:what|which|whom)|name\s+the)\b");YOU=re.compile(r"(?i)^\W*(?:\S+\s+){0,3}(?:you|your|you're|yours|yourself|u|ur)\b");OPIN=re.compile(r"(?i)\b(?:a good|any good|good way|best way|should|recommend\w*|advice|tips?|ideas?|suggest\w*|do you think|opinion|i can|can i|could i|would i|will i|myself)\b");ASV=re.compile(r"\s*(?:\(\s*)?(?:(?:stars?|starring|starred|plays?|played|portrays?|portrayed|voices?|voiced|appears?|appeared|cast)\s+)?as\s");POSS=re.compile(r"(?i)\b([\w.-]+(?:\s+[\w.-]+)?)['’]s\s+(?:real\s+|biological\s+|first\s+|second\s+|ex[- ]?)?(?:mother|father|mom|mum|dad|wife|husband|son|daughter|sister|brother|boyfriend|girlfriend|friend|best friend|boss|partner|uncle|aunt|cousin|niece|nephew|grandmother|grandfather|grandma|grandpa|baby|twin|fianc[eé]e?|lover|love interest|parents?|kids?|children|family|dog|cat)\b");ACT=re.compile(r"(?i)\b(?:play|plays|played|playing|portray\w*|voice[sd]?|star\w*|cast)\b");SOFT=re.compile(r"(?i)unknown|not sure|don'?t know|do not know|no idea|can'?t recall|not certain")
GATE={"c":float(os.environ.get("AMNI_RKNOW_C","0.5"))}
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
FEAT=("bias","s0","pw","r","cov","lc","d","sup","n","share","rk","relhi","rello","relna","tok","tmiss","tbad","qover","ntok","top","title","q_date","q_num","q_person","q_place","q_other","tchar","gap","actqo","pas","pby","ppar","lnk","lead","lnklead","occ","occna","pasx")
WTS=dict(zip(FEAT,json.loads(os.environ.get("AMNI_RKNOW_WTS","null") or "null") or (-2.6033,0.3956,-1.3234,-1.067,-0.1368,-0.4368,-0.3136,-0.1654,0.0773,0.4265,0.9631,0.6025,-0.3353,0.0917,0.4272,-0.5734,-0.637,-0.9201,0.2541,0.4588,0.2277,-0.3162,-0.0907,-0.0676,-0.7155,-0.6949,-0.2251,0.0344,-0.7633,0.5282,0.182,-0.0139,0.1959,0.2917,0.1563,0.6455,-0.3817,-0.099)))
_TY={}
def ents(y:str)->list:
 out=[]
 for m in ENT.finditer(y):
  tk=m.group(0).split();a=m.start()
  while tk and (tk[0] in RL.STOPC or tk[0].lower() in CONN):a=y.index(tk[0],a)+len(tk[0]);tk=tk[1:]
  while tk and tk[-1].lower() in CONN:tk=tk[:-1]
  tk and not re.fullmatch(r"[A-Z]\.|(?:Jr|Sr|St|Mt|Dr|Inc|Co|Ltd)\.",tk[-1]) and tk[-1].endswith(".") and tk.__setitem__(-1,tk[-1][:-1])
  v=" ".join(tk);a=y.find(tk[0],a) if tk else -1
  if not tk or not tk[0][0].isupper() or len(tk)>6 or a<0 or len(v)<2 or all(t in RL.STOPC for t in tk):continue
  e=y.find(tk[-1],a)+len(tk[-1]);out.append((y[a:e],a,e))
 return out
def gen(qt:str,y:str,q:str)->list:
 if qt in ("date","num"):return RL.cands(qt,y,q)
 E=ents(y);X=[(y[a:e2],a,e2) for (v,a,e),(v2,a2,e2) in zip(E,E[1:]) if qt=="place" and re.fullmatch(r",\s*",y[e:a2])]
 return [(v,AN_NORM(v),a,e) for v,a,e in E+X]
def AN_NORM(s:str)->str:return " ".join(re.sub(r"\b(a|an|the)\b"," ",re.sub(r"[^\w\s]"," ",fold(s).lower())).split())
def ent(v:str,T)->tuple:
 if v in _TY:return _TY[v]
 d=T.doc(v) if T is not None else None;g=KR.lead(*d)[2] or "?" if d else None;m=re.match(r"\s*([^()\n,;:\"]{3,80}?)\s*\(",d[1][:400]) if d else None;f=m.group(1).strip() if m else None;f=f if f and f!=v and len(f.split())<=6 and set(toks(v))<set(toks(f)) and f[:1].isupper() else None
 return _TY.setdefault(v,(g,f))
OCCS=((r"\b(?:play|plays|played|playing|portray\w*|star\w*|cast)\b",r"actor|actress|comedian|entertainer|performer|model|personality|presenter|host|singer|rapper"),(r"\bvoice[sd]?\b",r"actor|actress|voice|comedian|singer|performer"),(r"\b(?:sing|sings|sang|sung|perform\w*|record\w*)\b",r"singer|band|musician|rapper|group|duo|vocalist|songwriter|artist|trio|quartet|performer"),(r"\b(?:wrote|write|writes|written|author\w*|compos\w*|lyric\w*|pen\w*)\b",r"writer|author|novelist|poet|songwriter|composer|lyricist|playwright|screenwriter|journalist|band|singer|musician|rapper|producer"),(r"\bdirect\w*\b",r"director|filmmaker|producer|screenwriter"))
def occf(q:str,d)->dict:
 f=[o for a,o in OCCS if re.search(a,q,re.I)];return {"occ":float(bool(f and d and d!="?" and any(re.search(r"(?i)\b(?:%s)"%o,d) for o in f))),"occna":float(bool(f) and (not d or d=="?"))}
def _typeok(qt:str,v:str,T)->int:
 if qt not in ("person","place") or T is None:return 0
 d=ent(v,T)[0];return 1 if not d or d=="?" else 0 if (RL.PERSONY if qt=="person" else RL.PLACEY).search(d) and not RL.NEGD.search(d) else 2
def span(qt:str,v:str,T=None)->str:
 f=ent(v,T or RL._kx()[1])[1] if qt=="person" else None;return "%s (full name %s)"%(v,f) if f else v
def occ(q:str,k:int=None)->tuple:
 I=ix();qt=RL.qtype(q);h=I.search(q,k or int(os.environ.get("AMNI_RKNOW_K","30")));O=[];P=[]
 if not h:return qt,O,P
 qa=set(toks(q));qw=[w for w in dict.fromkeys(toks(q)) if w not in QSTOP];W={w:I.idf(w) for w in qw};tot=sum(W.values()) or 1.0;F=RL.fam(q);qn=set(RL.words(q.encode()));yrs=set(re.findall(r"\b(?:1[0-9]|20)\d\d\b",q));cache={};qi=[w[0] for w in re.findall(r"[a-z0-9]+",fold(q).lower())];ini={"".join(qi[i:j]) for i in range(len(qi)) for j in range(i+2,min(len(qi),i+6)+1)}
 qc={z for z in qw if W[z]>=QCI};pm=POSS.search(q);pt=set(toks(pm.group(1))) if pm else set();lks=getattr(I,"lk",set())
 for r,(sc,pid) in enumerate(h):
  t,x,g=I.text(pid,cache);tw=set(toks(t));P.append((sc,pid,t,x));fp=int(I.first[g])
  for y in RL.sents(x) or [x]:
   sw=set(toks(y));cov=(sum(W[z] for z in qw if z in sw)+0.5*sum(W[z] for z in qw if z in tw and z not in sw))/tot
   if cov<0.3 or (yrs and not all(z in y+" "+t for z in yrs)):continue
   yl=fold(y).lower();pos=[(m.start(),m.end(),z) for z in qw if z in sw for m in re.finditer(r"\b"+re.escape(z),yl)]
   for v,key,a,b in gen(qt,y,q):
    ct=set(toks(v));qo=1.0 if v.isupper() and 1<len(v)<=6 and v.lower() in ini else len(ct&qa)/max(1,len(ct))
    if not ct or qo>=1.0:continue
    d=min([a-p0 if p0<a else max(0,p0-b) for p0,_,_ in pos] or [200]);lc=sum(W[z] for z in {z for p0,p1,z in pos if p1>=a-80 and p0<=b+80})/tot
    qn2=lambda z:bool(set(toks(z))&(qc-tw));am=ASV.match(y[b:b+40]);ae=" ".join(v for v,_,_ in ents(y[b+am.end():b+am.end()+50])[:1]) if am else "";bm=BYV.search(y[max(0,a-40):a]);be=" ".join(v for v,_,_ in ents(y[max(0,a-110):a])[-1:]) if bm else "";O.append({"key":key,"v":v,"g":g,"t":t,"y":y,"r":r,"pw":sc/h[0][0],"cov":cov,"lc":lc,"d":d,"qo":qo,"rel":RL.rel(qt,y[max(0,a-100):b+100],a-max(0,a-100),b-max(0,a-100),F,qn) if F is not None else None,"pas":bool(am) and qn2(ae) and not set(toks(ae))&pt,"pasx":bool(pt and ((am and set(toks(ae))&pt) or (bm and set(toks(be))&pt))),"pby":bool(bm) and qn2(y[max(0,a-90):a]) and not (pt and set(toks(be))&pt),"ppar":y[max(0,a-2):a].strip()=="(" and qn2(y[max(0,a-50):a]),"lnk":g in lks,"lead":fp==pid})
 return qt,O,P
def feats(qt:str,O:list,T=None,n:int=10,q:str="")->list:
 C=collections.defaultdict(list)
 for o in O:C[o["key"]].append({**o,"s":o["pw"]**2*o["cov"]**2*(0.2+o["lc"])})
 for kk in sorted(C,key=len):
  sup=[j for j in C if j!=kk and set(kk.split())<set(j.split())];len(sup)==1 and C[sup[0]].extend(C.pop(kk))
 R=[]
 for kk,o in C.items():
  per={};[per.__setitem__(z["g"],max(per.get(z["g"],0.0),z["s"])) for z in o];f=collections.Counter(z["v"] for z in o);tf=max(f.values());v=max([z for z in f if f[z]>=max(2,tf/3)] or f,key=lambda z:(len(z),f[z]) if f[z]>=max(2,tf/3) else (f[z],len(z)));bz=max([z for z in o if z["v"]==v],key=lambda z:(bool(z.get("lnk")),bool(z.get("pas") or z.get("pby")),round(z["cov"],1),z["s"]));rl=[z["rel"] for z in o]
  R.append({"key":kk,"v":v,"t":bz["t"],"y":bz["y"],"raw":sum(per.values()),"sup":len(per),"x":{"pw":max(z["pw"] for z in o),"r":min(z["r"] for z in o)/30,"cov":max(z["cov"] for z in o),"lc":max(z["lc"] for z in o),"d":math.log1p(min(z["d"] for z in o))/5,"sup":math.log1p(len(per)),"n":math.log1p(len(o)),"relhi":float(any(x is not None and x>=1.0 for x in rl)),"rello":float(all(x is not None and x<0.4 for x in rl)),"relna":float(rl[0] is None),"qover":min(z["qo"] for z in o),"ntok":len(kk.split())/4,"top":float(any(z["r"]==0 for z in o)),"title":float(any(AN_NORM(z["t"]).startswith(kk) for z in o)),"pas":float(any(z.get("pas") for z in o)),"pby":float(any(z.get("pby") for z in o)),"ppar":float(any(z.get("ppar") for z in o)),"pasx":float(any(z.get("pasx") for z in o)),"lnk":float(any(z.get("lnk") for z in o)),"lead":float(any(z.get("lead") for z in o)),"lnklead":float(any(z.get("lnk") and z.get("lead") for z in o)),**{"q_"+x:float(qt==x) for x in ("date","num","person","place","other")}}})
 R=sorted(R,key=lambda z:-z["raw"])[:n];tot=sum(z["raw"] for z in R) or 1.0;T=T if T is not None or qt not in ("person","place") or not R else RL._kx()[1]
 for i,z in enumerate(R):ty=_typeok(qt,z["v"],T);z["x"].update({"bias":1.0,"s0":math.log1p(10*z["raw"]),"share":z["raw"]/tot,"rk":1/(1+i),"tok":float(ty==0 and qt in ("person","place")),"tmiss":float(ty==1),"tbad":float(ty==2),"actqo":float(bool(ACT.search(q)) and z["x"]["qover"]>0),**occf(q,ent(z["v"],T)[0] if T is not None else None),"tchar":float(bool(T is not None and ACT.search(q) and re.search(r"(?i)character|fictional",(ent(z["v"],T)[0] or "")))),"gap":(z["raw"]-(R[1]["raw"] if i==0 and len(R)>1 else R[0]["raw"]))/max(z["raw"],1e-9)})
 return R
def rank(qt:str,O:list,wts:dict=None,T=None,q:str="")->list:
 w=wts or WTS;R=feats(qt,O,T,q=q)
 for z in R:z["s"]=1/(1+math.exp(-sum(w.get(k,0.0)*v for k,v in z["x"].items())));z["cov"]=z["x"]["cov"]
 return sorted(R,key=lambda z:-z["s"])
def candidates(q:str,k:int=None)->tuple:
 qt,O,P=occ(q,k);return qt,rank(qt,O,q=q),P
CF=("bias","p1","gap","p2","share","sup","cov","relhi","pas","pby","tok","tmiss","lnk","lead","q_date","q_num","q_person","q_place","q_other","ncand","top","qover","s0","occ","occna","q_act","q_won")
CWTS=dict(zip(CF,json.loads(os.environ.get("AMNI_RKNOW_CWTS","null") or "null") or (-4.1953,0.7626,0.7338,0.0288,0.6721,0.0138,0.5731,0.4748,0.4543,0.1339,0.3915,-0.3585,0.2974,0.0667,0.2401,0.0193,0.358,-0.5249,-0.1287,-0.2489,0.3105,-0.756,0.3549,0.7686,0.1,-0.1359,-0.1392)))
def qfeat(qt:str,R:list,q:str)->dict:
 c=R[0]["x"];p2=R[1]["s"] if len(R)>1 else 0.0;return {**{k:c.get(k,0.0) for k in CF},"bias":1.0,"p1":R[0]["s"],"gap":R[0]["s"]-p2,"p2":p2,"ncand":len(R)/10,"q_act":float(bool(ACT.search(q))),"q_won":float(bool(re.search(r"(?i)\b(?:won|win|wins|winner|champion\w*)\b",q)) and not re.search(r"\b(?:1[0-9]|20)\d\d\b",q))}
def decide(qt:str,R:list,gate:dict=GATE,q:str="")->dict:
 if not R:return None
 e=R[0];f=qfeat(qt,R,q);c=1/(1+math.exp(-sum(CWTS.get(k,0.0)*v for k,v in f.items())));ok=c>=gate["c"] and not SOFT.search(e["v"])
 return {"span":e["v"],"title":e["t"],"url":KR.wurl(e["t"]),"evidence":e["y"],"qtype":qt,"conf":round(c,4),"score":round(e["s"],4),"sup":e["sup"],"cov":round(e["cov"],3),"ok":ok,"license":"CC BY-SA 4.0","reply":"%s (Source: Wikipedia, %s)."%(span(qt,e["v"]),e["t"] if not SOFT.search(e["t"]) else "article")}
def factq(q:str)->bool:
 from amni.compute.rchat import act
 return bool(FACTQ.search(q)) and not YOU.search(q) and not OPIN.search(q) and not RL.PERSONAL.search(q) and not TEMP.search(q) and act(q) in ("fact","other")
def ask(q:str,gate:bool=True)->dict:
 if not factq(q):return None
 qt,R,_=candidates(q);r=decide(qt,R,q=q);return r if r and (r["ok"] or not gate) else None
