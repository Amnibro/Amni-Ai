import os,re,sys,json,math,time,hashlib,argparse,unicodedata,functools
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DIR=Path(os.environ.get("AMNI_WIKIFACT_DIR",str(ROOT/"data"/"wikifact"/"v2"/"store")));THR=float(os.environ.get("AMNI_WIKIFACT_THR","0.5"));TEMPORAL=os.environ.get("AMNI_WIKIFACT_TEMPORAL","0")=="1";LIC="Wikidata CC0 1.0 (wikidata-20260921-all.json.gz)"
MON=["January","February","March","April","May","June","July","August","September","October","November","December"];STOPR={"the","a","an","of","and","s","mr","mrs","ms","dr"}
TYPES={"movie":(11424,29168811,24869,202866,506240),"film":(11424,29168811,24869,202866,506240),"song":(7366,134556,105543609),"single":(134556,7366),"book":(571,7725634,8261,47461344),"novel":(8261,7725634,571),"show":(5398426,581714,15416,1259759),"series":(5398426,581714,15416,1259759),"sitcom":(5398426,15416),"album":(482994,208569,169930),"band":(215380,2088357,5741069),"group":(215380,2088357),"musical":(2743,7725634),"play":(25379,7725634),"poem":(5185279,7725634),"game":(7889,)}
TPRE=re.compile(r"^(?:the )?(movie|film|song|single|book|novel|tv show|show|tv series|series|sitcom|album|band|group|musical|play|poem|video game|game) (?:called |titled |named )?");TSUF=re.compile(r" (?:the )?(movie|film|song|single|book|novel|tv show|show|tv series|series|sitcom|album|band|group|musical|play|poem|video game|game)$")
WHEN={"bear":[569],"birth":[569],"birthday":[569],"die":[570],"pass away":[570],"kill":[570],"assassinate":[570],"murder":[570],"execute":[570],"found":[571],"establish":[571],"form":[571],"create":[571,577],"incorporate":[571],"start":[580,571,577],"begin":[580,571],"open":[1619,571],"build":[1619,571],"construct":[1619,571],"complete":[1619,571],"invent":[575,571],"discover":[575,571],"write":[577,571],"publish":[577],"release":[577,580],"come out":[577,580],"premiere":[577,580],"air":[580,577],"debut":[577,580],"launch":[619,577,571],"sign":[585,571,577],"ratify":[585,571],"adopt":[585,571],"introduce":[571,577],"end":[582,576],"close":[576,582],"dissolve":[576],"abolish":[576],"disband":[576],"demolish":[576],"destroy":[576],"happen":[585,580],"take place":[585,580],"occur":[585,580],"fight":[585,580],"hold":[585,580],"declare":[585,571],"enact":[585,571],"pass":[585,571,577],"make":[577,571],"record":[577]}
WHO={"write":[50,86,676,58],"compose":[86,676],"pen":[50],"author":[50],"direct":[57],"produce":[162],"create":[170,61,112],"invent":[61],"discover":[61],"found":[112],"establish":[112],"design":[84,170],"build":[84,176],"develop":[178],"make":[176,178,170],"manufacture":[176],"own":[127],"publish":[123],"sing":[175],"perform":[175],"record":[175],"paint":[170],"sculpt":[170],"illustrate":[110],"win":[1346,991],"coach":[286],"distribute":[750],"narrate":[725]}
NOUN={"author":[50],"writer":[50,86,676,58],"director":[57],"producer":[162],"creator":[170,61,112],"inventor":[61],"discoverer":[61],"founder":[112],"designer":[84,170],"architect":[84],"developer":[178],"manufacturer":[176],"maker":[176,178],"owner":[127],"publisher":[123],"singer":[175],"performer":[175],"composer":[86],"lyricist":[676],"illustrator":[110],"winner":[1346,991],"coach":[286],"head coach":[286],"manager":[286],"ceo":[169],"chief executive officer":[169],"chairman":[488],"chairperson":[488],"president":[35,488],"prime minister":[6],"premier":[6],"chancellor":[6],"king":[35],"queen":[35],"monarch":[35],"emperor":[35],"mayor":[6],"governor":[6],"head of state":[35],"head of government":[6],"leader":[6,35,488],"chief minister":[6],"first minister":[6],"principal":[1037],"capital":[36],"capital city":[36],"population":[1082],"currency":[38],"national anthem":[85],"anthem":[85],"official language":[37],"language":[37],"headquarters":[159],"hq":[159],"continent":[30],"highest point":[610],"highest mountain":[610],"tallest mountain":[610],"highest peak":[610],"wife":[26],"husband":[26],"spouse":[26],"father":[22],"dad":[22],"mother":[25],"mom":[25],"son":[40],"daughter":[40],"children":[40],"kids":[40],"brother":[3373],"sister":[3373],"siblings":[3373],"birthplace":[19],"hometown":[19],"place of birth":[19],"nationality":[27],"real name":[1477],"birth name":[1477],"full name":[1477],"nickname":[1449],"atomic number":[1086],"chemical symbol":[246],"symbol":[246],"chemical formula":[274],"formula":[274],"network":[449],"channel":[449],"record label":[264],"label":[264],"sequel":[156],"prequel":[155],"cause of death":[509],"area":[2046],"height":[2048,2044],"elevation":[2044],"length":[2043,2047],"number of episodes":[1113],"number of seasons":[2437],"stadium":[115],"home stadium":[115],"arena":[115],"home ground":[115],"league":[118],"religion":[140],"party":[102],"political party":[102]}
LIGHT={"have","do","get","go","make","take","give","be","find","hit","keep","follow","enter","present","pass","deal","lot","off","execute","observe","notice","detect","remove","choke","struggle","contend","project","plan","instance","train","print","deliver","integrate","finish","stop","shut","set","put","run","come","see","let","hold","air","open","close","end","start","begin","form","create"}
CRIT={"season","seasons","episode","episodes","part","chapter","volume","sequel","prequel","first","last","next","new","latest","second","third","fourth","final","original","remake","sister","brother","wife","husband","son","daughter","mother","father","ii","iii","iv","junior","jr","sr","2nd","3rd","1st"}
GEN={"movie","film","song","single","show","tv","series","original","band","book","novel","album","us","usa","uk","america","american","british","english","world","version","cartoon","anime","musical","play","game","video","episode","character","group"}
CUR={35,6,169,488,286,127,54,1037,1082,115,118,159,102,26}
_MT={}
def fold(t:str)->str:
 if t.isascii():return t
 _MT or _MT.update({c:None for c in range(0x300,0x370)}|{c:None for c in range(0x1AB0,0x1B00)}|{c:None for c in range(0x1DC0,0x1E00)}|{c:None for c in range(0x20D0,0x2100)}|{c:None for c in range(0xFE20,0xFE30)})
 return unicodedata.normalize("NFKD",t).translate(_MT)
def nrm(s:str)->str:return " ".join(re.sub(r"[\W_]+"," ",re.sub(r"['’`´]","",fold(s or "").lower().replace("&"," and "))).split())
def h64(s:str)->int:return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),"little")
@functools.lru_cache(maxsize=1)
def _wn():
 try:
  import nltk;nltk.data.path.insert(0,str(ROOT/"data"/"concepts"/"nltk_data"));from nltk.corpus import wordnet as wn;wn.synsets("write","v");return wn
 except Exception:return None
@functools.lru_cache(maxsize=65536)
def lem(w:str)->str:
 wn=_wn();m=wn.morphy(w,"v") if wn else None
 return m or (w[:-3]+"y" if w.endswith("ies") else w[:-2] if w.endswith(("ed","es")) and len(w)>4 else w[:-1] if w.endswith("s") and len(w)>3 else w)
@functools.lru_cache(maxsize=1)
def tables()->tuple:
 wn=_wn();W,H=dict(WHEN),dict(WHO)
 for T in (W,H):
  for k,v in list(T.items()):
   for s in (wn.synsets(k,"v")[:1] if wn and " " not in k else []):
    [T.setdefault(l.name().lower(),v) for l in s.lemmas() if "_" not in l.name() and l.name().lower() not in LIGHT and wn.synsets(l.name(),"v")[:1]==[s]]
 return W,H
def years(x:str)->list:return [int(y) for y in re.findall(r"\b(1[5-9]\d\d|20[0-3]\d)\b",x)]
class Store:
 def __init__(s,d:Path=DIR):
  s.d=Path(d);L=lambda n:np.load(s.d/f"{n}.npy",mmap_mode="r");B=lambda n:np.memmap(s.d/n,np.uint8,"r") if (s.d/n).stat().st_size else np.zeros(0,np.uint8)
  for n in ("e_qid","e_sl","e_fs","e_fn","e_lo","e_to","f_s","f_p","f_r","f_k","f_v","f_x","f_qo","q_p","q_k","q_v","a_h","a_e","a_k","nm_o","r_v","r_f","rq_v","rq_f"):setattr(s,n,L(n))
  s.lab,s.tit,s.str,s.nm=B("e_lab.bin"),B("e_tit.bin"),B("str.bin"),B("nm.bin");s.man=json.loads((s.d/"manifest.json").read_text()) if (s.d/"manifest.json").exists() else {}
  s.cls={k:{i for i in (s.ix(q) for q in v) if i>=0} for k,v in TYPES.items()}
 def ix(s,q:int)->int:
  i=int(np.searchsorted(s.e_qid,q));return i if i<len(s.e_qid) and int(s.e_qid[i])==q else -1
 def label(s,e:int)->str:return bytes(s.lab[int(s.e_lo[e]):int(s.e_lo[e+1])]).decode("utf-8","replace") if 0<=e<len(s.e_qid) else ""
 def title(s,e:int)->str:return bytes(s.tit[int(s.e_to[e]):int(s.e_to[e+1])]).decode("utf-8","replace") if 0<=e<len(s.e_qid) else ""
 def name(s,e:int)->str:return s.label(e) or re.sub(r"\s*\([^)]*\)$","",s.title(e))
 def names(s,e:int)->list:return [x for x in bytes(s.nm[int(s.nm_o[e]):int(s.nm_o[e+1])]).decode("utf-8","replace").split("\n") if x] if 0<=e<len(s.e_qid) else []
 def text(s,v:int)->str:return bytes(s.str[v>>16:(v>>16)+(v&0xFFFF)]).decode("utf-8","replace")
 def ents(s,x:str)->list:
  h=np.uint64(h64(x));i,j=np.searchsorted(s.a_h,h,"left"),np.searchsorted(s.a_h,h,"right");return list(zip(s.a_e[i:j].tolist(),s.a_k[i:j].tolist()))
 def facts(s,e:int,props)->np.ndarray:
  a,n=int(s.e_fs[e]),int(s.e_fn[e]);P=np.asarray(s.f_p[a:a+n]);return a+np.flatnonzero(np.isin(P,list(props))) if n else np.zeros(0,np.int64)
 def quals(s,f:int)->list:
  a,b=int(s.f_qo[f]),int(s.f_qo[f+1]);return list(zip(s.q_p[a:b].tolist(),s.q_k[a:b].tolist(),s.q_v[a:b].tolist()))
 def rev(s,e:int,rv="r_v",rf="r_f")->np.ndarray:
  V=getattr(s,rv);i,j=np.searchsorted(V,e,"left"),np.searchsorted(V,e,"right");return np.asarray(getattr(s,rf)[i:j],np.int64)
 def inst(s,e:int)->set:return {int(s.f_v[f]) for f in s.facts(e,[31]) if int(s.f_k[f])==1}
 def year(s,e:int)->set:return {tu(int(s.f_v[f]))[0] for f in s.facts(e,[577,571,580,585,1191]) if int(s.f_k[f])==2}
@functools.lru_cache(maxsize=1)
def store():
 try:return Store(DIR) if (DIR/"a_h.npy").exists() else None
 except Exception:return None
def tu(v:int)->tuple:return ((v>>24)-(1<<39),(v>>16)&255,(v>>8)&255,(v&255)>>1)
def fdate(v:int,yo:bool=False)->list:
 y,m,d,p=tu(v);ys=f"{-y} BC" if y<=0 else str(y);ys=f"{y//10*10}s" if p==8 and y>0 else f"{(y-1)//100+1}th century" if p==7 and y>0 else ys
 return [ys] if yo or p<=9 or m==0 else [f"{MON[m-1]} {ys}"] if p==10 or d==0 else [f"{MON[m-1]} {d}, {ys}",f"{d} {MON[m-1]} {ys}",f"{y:04d}-{m:02d}-{d:02d}",f"{MON[m-1]} {d} {ys}"]
def fnum(x:float,u:str="")->list:
 v=int(round(x)) if abs(x-round(x))<1e-9 else round(x,2);a=f"{v:,}" if isinstance(v,int) and abs(v)>=10000 else str(v);return [a+(" "+u if u else ""),str(v)+(" "+u if u else ""),a]
def render(S:Store,f:int,yo:bool=False)->tuple:
 k,v=int(S.f_k[f]),int(S.f_v[f])
 if k==1:n=S.name(v);return (n,[n]+S.names(v),v) if n else (None,[],-1)
 if k==2:r=fdate(v,yo);return r[0],r,-1
 if k==3:x=int(S.f_x[f]);u=S.name(x) if x!=0xFFFFFFFF else {1113:"episodes",2437:"seasons"}.get(int(S.f_p[f]),"");r=fnum(float(np.uint64(v).view(np.float64)),u);return r[0],r,-1
 t=S.text(v);return t,[t],-1
def pick(S:Store,F:np.ndarray,mode:str)->list:
 if not len(F):return []
 F=[int(f) for f in F];pr=[f for f in F if int(S.f_r[f])==1] or F
 if mode=="earliest":T=[f for f in F if int(S.f_k[f])==2];return sorted(T,key=lambda f:int(S.f_v[f]))[:1] or pr[:1]
 if mode in ("cur","latest"):
  def key(f):
   Q=S.quals(f);end=any(p==582 for p,_,_ in Q);t=max([v for p,k,v in Q if p in (580,585) and k==2] or [0]);return (int(S.f_r[f]),not end if mode=="cur" else 0,t)
  return sorted(F,key=key,reverse=True)[:1]
 return pr
def variants(x:str)->list:
 out=[(x,1.0,None)];t=None;m=TPRE.match(x)
 m and out.append((x[m.end():],0.9,m.group(1).split()[-1]));m2=TSUF.search(x)
 m2 and out.append((x[:m2.start()],0.8,m2.group(1).split()[-1]))
 x.startswith("the ") and out.append((x[4:],0.97,None))
 for v,f,t in list(out):
  v.endswith("s") and len(v)>3 and out.append((v[:-1],f*0.95,t))
  yy=re.sub(r"\s*\b(1[5-9]\d\d|20[0-3]\d)\b\s*"," ",v).strip();yy!=v and yy and out.append((yy,f*0.9,t))
  " season " in v and out.append((re.sub(r" season \w+$","",v),f*0.6,t))
 seen=set();return [(v,f,t) for v,f,t in out if v and not (v in seen or seen.add(v))]
def link(S:Store,x:str,props,sub:bool=True,by:str|None=None,yr:list=()):
 C={};KW=(1.0,0.6,1.0,0.85)
 for v,f,t in variants(x):
  for e,k in S.ents(v):
   w=f*KW[min(k,3)];C[e]=max(C.get(e,(0,None)),(w,t))
 if not C and sub:
  tk=x.split();n=len([t for t in tk if t not in STOPR]) or 1
  for L in range(min(len(tk),10),0,-1):
   for i in range(len(tk)-L+1):
    sp=tk[i:i+L]
    if all(t in STOPR for t in sp) or (L==1 and len(sp[0])<4):continue
    rest=tk[:i]+tk[i+L:]
    if any(t.isdigit() or t in CRIT for t in rest):continue
    lo=[t for t in rest if t not in STOPR and t not in GEN and not t.isdigit()]
    for e,k in S.ents(" ".join(sp)):C[e]=max(C.get(e,(0,None)),((0.3+0.5*min(1,len([t for t in sp if t not in STOPR])/n))*0.45**len(lo)*KW[min(k,3)],None))
   if C and any(len(S.facts(e,props)) for e in C):break
 R=[]
 for e,(w,t) in C.items():
  if not len(S.facts(e,props)):continue
  tb=1.25 if t and S.cls.get(t) and S.inst(e)&S.cls[t] else 1.0;ye=S.year(e) if yr else set();yb=1.0 if not ye else 1.3 if any(abs(a-b)<=1 for a in ye for b in yr) else 0.0
  if by:
   B={e2 for vv,_,_ in variants(by) for e2,_ in S.ents(vv)}
   if not B or not any(int(S.f_v[g]) in B for g in S.facts(e,[175,50,86,676,57,170,162]) if int(S.f_k[g])==1):continue
  R.append((e,w*tb*yb,w,(int(S.e_sl[e])+1)))
 return R
def vote(S:Store,R:list,fn)->tuple:
 A={};tot=0.0
 for e,w,lf,sl in R:
  r=fn(e)
  if not r:continue
  z=w*w*math.log2(1+sl);tot+=z;k=nrm(r["answer"]);a=A.setdefault(k,{**r,"z":0.0,"lf":0.0});a["z"]+=z;a["lf"]=max(a["lf"],lf)
 if not A:return None
 b=max(A.values(),key=lambda a:a["z"]);b["share"]=b["z"]/max(tot,1e-9);return b
def ent_answer(S:Store,x:str,q:str,props:list,mode:str,base:float,yo:bool=False,by:str|None=None,rule:str="")->dict|None:
 R=link(S,x,props,by=by,yr=years(q))
 def fn(e):
  for p in props:
   F=pick(S,S.facts(e,[p]),mode)
   for f in F:
    a,forms,ve=render(S,f,yo)
    if a and not (int(S.f_k[f])==1 and nrm(a) and re.search(r"(?<!\S)"+re.escape(nrm(a))+r"(?!\S)",nrm(q))):return {"answer":a,"forms":forms,"e":e,"p":p,"f":f,"multi":len(S.facts(e,[p]))>1 and p not in (569,570,19,20,36,1082),"qid":int(S.e_qid[e])}
  return None
 b=vote(S,R,fn)
 if not b:return None
 cur=b["p"] in CUR;c=base*b["lf"]*b["share"]*min(1.0,math.log2(2+int(S.e_sl[b["e"]]))/4)*(0.9 if b["multi"] else 1.0)
 return {**b,"conf":round(min(c,0.99),4),"source":f"Wikidata Q{b['qid']} P{b['p']} ({LIC})","rule":rule,"temporal":cur}
def rmatch(rt:list,name:str)->float:
 nt=name.split();ok=lambda t:any(u==t or (len(t)>=3 and len(u)>=3 and (u.startswith(t) or t.startswith(u))) for u in nt)
 return sum(map(ok,rt))/max(1,len(rt))+0.01*sum(any(u==t for t in rt) for u in nt)/max(1,len(nt))
def cast(S:Store,role:str,work:str,q:str,base:float,rule:str="cast")->dict|None:
 rt=[t for t in role.split() if t not in STOPR]
 if not rt:return None
 RE={e for v,_,_ in variants(role) for e,_ in S.ents(v)};W=link(S,work,[161,725],sub=True,yr=years(q));best=[]
 for e,w,lf,sl in sorted(W,key=lambda z:-z[1]*math.log2(1+z[3]))[:6]:
  for f in S.facts(e,[161,725]):
   Q=S.quals(int(f));nm=[n for p,k,v in Q if p==453 and k==1 for n in [nrm(S.name(v))]+S.names(v)]+[nrm(S.text(v)) for p,k,v in Q if p==4633 and k in (4,5)]
   sc=max([1.02 if any(p==453 and v in RE for p,k,v in Q) else 0]+[rmatch(rt,n) for n in nm if n])
   sc>=1.0 and best.append((sc,w*math.log2(1+sl),lf,int(f),e))
 if not best:return None
 best.sort(key=lambda z:(-z[0],-z[1]));A={}
 for sc,z,lf,f,e in best:A.setdefault(int(S.f_v[f]),[]).append((sc,z,lf,f,e))
 top=max(A.items(),key=lambda kv:(max(x[0] for x in kv[1]),sum(x[1] for x in kv[1])));sc,z,lf,f,e=top[1][0];a=S.name(top[0])
 if not a:return None
 share=sum(x[1] for x in top[1])/sum(x[1] for v in A.values() for x in v);c=base*lf*share*(1.0 if sc>=1.0 else 0.8)
 return {"answer":a,"forms":[a]+S.names(top[0]),"e":e,"p":int(S.f_p[f]),"f":f,"qid":int(S.e_qid[e]),"conf":round(min(c,0.99),4),"source":f"Wikidata Q{int(S.e_qid[e])} P{int(S.f_p[f])}+P453 ({LIC})","rule":rule,"temporal":False,"share":share}
def rcast(S:Store,actor:str,work:str,q:str,base:float)->dict|None:
 AE={e for v,_,_ in variants(actor) for e,_ in S.ents(v)};W=link(S,work,[161,725],yr=years(q))
 for e,w,lf,sl in sorted(W,key=lambda z:-z[1]*math.log2(1+z[3]))[:6]:
  for f in S.facts(e,[161,725]):
   if int(S.f_v[f]) not in AE:continue
   for p,k,v in S.quals(int(f)):
    a=S.name(v) if p==453 and k==1 else S.text(v) if p==4633 and k in (4,5) else None
    if a:return {"answer":a,"forms":[a]+(S.names(v) if p==453 else []),"e":e,"p":int(S.f_p[f]),"f":int(f),"qid":int(S.e_qid[e]),"conf":round(base*lf,4),"source":f"Wikidata Q{int(S.e_qid[e])} P{int(S.f_p[f])}+P453 ({LIC})","rule":"rcast","temporal":False}
 return None
def role_only(S:Store,role:str,q:str,base:float)->dict|None:
 R=link(S,role,[175,725],sub=False)
 if R:
  r=ent_answer(S,role,q,[175,725],"",base*0.9,rule="role")
  if r:return r
 RE={e for v,_,_ in variants(role) for e,_ in S.ents(v)};V={}
 for e in RE:
  for g in S.rev(e,"rq_v","rq_f"):
   a=int(S.f_v[g]);int(S.f_p[g]) in (161,725) and V.__setitem__(a,V.get(a,0)+math.log2(2+int(S.e_sl[int(S.f_s[g])])))
 if not V:return None
 a,z=max(V.items(),key=lambda kv:kv[1]);n=S.name(a);share=z/sum(V.values())
 return {"answer":n,"forms":[n]+S.names(a),"e":a,"p":161,"f":-1,"qid":int(S.e_qid[a]),"conf":round(base*0.85*share,4),"source":f"Wikidata P161/P453 reverse ({LIC})","rule":"role_rev","temporal":False} if n else None
T=r"(?:the )?"
V1=r"(?:plays|played|portrays|portrayed|voices|voiced|is playing|was playing|will play|is the voice of|was the voice of|does the voice of|did the voice of|plays the voice of|played the voice of|played the role of|plays the role of|played the part of|plays the part of|starred as|stars as|acted as|is the actor that plays|is the actress that plays|is the actor who plays|is the actress who plays|is the actor who played|is the actress who played|was the actor who played|was the actress who played|played the character|plays the character|originally played|originally voiced)"
RULES=[(r"^who "+V1+r" (?P<r>.+?) (?:in|on|from|for) (?P<s>.+)$","cast",0.95),
 (r"^(?:who|what character|what role|which character|what part) (?:did|does|is|was) (?P<a>.+?) (?:play|playing|voice|voicing|portray) (?:in|on) (?P<s>.+)$","rcast",0.9),
 (r"^who "+V1+r" (?P<r>.+)$","role",0.85),
 (r"^who (?:is|was) (?P<r>.+?) (?:in|on) (?P<s>.+)$","cast",0.7),
 (r"^(?:when|what year|what date|what day|in what year|in which year|which year|when exactly) (?:was|were|did|is|does|do|will|are|has|had) (?:the )?(?P<s>.+?) (?P<v>\w+)(?: (?P<pt>place|out|away))?$","when",0.9),
 (r"^(?:when|what) is (?P<s>.+?) (?:birthday|date of birth|birth date)$","born",0.9),
 (r"^(?:what is |whats )?(?:the )?(?:date of birth|birthdate|birth date|birthday|dob) of (?P<s>.+)$","born",0.9),
 (r"^where (?:was|is|were) (?P<s>.+?) born$","p19",0.9),
 (r"^where did (?P<s>.+?) die$","p20",0.85),
 (r"^(?:how|what) did (?P<s>.+?) die$","p509",0.8),
 (r"^where (?:was|were|is|are|did they film|do they film) (?P<s>.+?) (?:filmed|shot)$","p915",0.8),
 (r"^where (?:does|did|is) (?P<s>.+?) (?:take place|takes place|set)$","p840",0.75),
 (r"^where does (?:the )?(?P<s>.+?) (?:start|begin|originate|come from|rise)$","p885",0.75),
 (r"^where does (?:the )?(?P<s>.+?) (?:end|flow into|empty into|drain into|flow to|empty|drain)$","p403",0.75),
 (r"^where (?:is|are) (?:the )?(?:headquarters|hq|head office|main office) (?:of|for) (?P<s>.+)$","p159",0.85),
 (r"^where (?:is|are) (?P<s>.+?) (?:headquartered|based|headquarters|hq)(?: located)?$","p159",0.85),
 (r"^where did (?P<s>.+?) (?:go to college|go to school|go to university|go to high school|study|graduate from|attend college|attend school|attend university)$","p69",0.7),
 (r"^(?:what|which) (?:country|nation) (?:is|was|are|were) (?P<s>.+?) (?:in|located in|from|part of)$","p17",0.8),
 (r"^(?:what|which) continent (?:is|was|are) (?P<s>.+?) (?:in|on|located in|part of)$","p30",0.8),
 (r"^(?:who|what) (?:is|was|are|were) (?P<s>.+?) named (?:after|for)$","p138",0.8),
 (r"^how many (?:people|residents|inhabitants|persons|citizens) (?:live|lived|living|are living|are there|reside|are) in (?P<s>.+)$","p1082",0.8),
 (r"^how many episodes (?:are |were |is )?(?:there )?(?:in |of )(?P<s>.+)$","p1113",0.8),
 (r"^how many episodes (?:does|did|do) (?P<s>.+?) (?:have|has)$","p1113",0.8),
 (r"^how many seasons (?:are |were |is )?(?:there )?(?:in |of )(?P<s>.+)$","p2437",0.8),
 (r"^how many seasons (?:does|did|do) (?P<s>.+?) (?:have|has)$","p2437",0.8),
 (r"^how tall (?:is|was|are) (?P<s>.+)$","p2048",0.6),
 (r"^how long is (?:the )?(?P<s>.+)$","p2043",0.55),
 (r"^what (?:is|was) the sequel to (?P<s>.+)$","p156",0.8),
 (r"^what (?:is|was) (?:the )?(?P<s>.+?) based on$","p144",0.7),
 (r"^what instrument (?:does|did) (?P<s>.+?) play$","p1303",0.75),
 (r"^what position (?:does|did) (?P<s>.+?) play$","p413",0.75),
 (r"^what (?:planet|star) (?:does|did) (?P<s>.+?) orbit$","p397",0.8),
 (r"^(?:what|which) (?:team|club) (?:does|did|is|was) (?P<s>.+?) (?:play for|playing for|play on|on|with|in)$","p54",0.7),
 (r"^(?:what|which) (?:channel|network|station|tv channel|tv network) (?:is|was|does|did) (?P<s>.+?) (?:on|air on|come on|shown on|broadcast on|aired on|air)$","p449",0.75),
 (r"^what (?:language|languages) (?:is|are|do they|do people) (?:spoken |speak )?(?:in )?(?P<s>.+?)$","p37",0.7),
 (r"^what currency (?:is used|does|do they use|is) (?:in )?(?P<s>.+?)(?: use)?$","p38",0.8),
 (r"^what (?:nationality|country) (?:is|was) (?P<s>.+?)(?: from)?$","p27",0.7),
 (r"^what (?:political )?party (?:is|was|does|did) (?P<s>.+?)(?: belong to| in| a member of| represent| from)?$","p102",0.7),
 (r"^what religion (?:is|was|does|did) (?P<s>.+?)(?: follow| practice| believe in)?$","p140",0.7),
 (r"^(?:what|which) (?:state|county|province|region|district|city) (?:is|was|are) (?P<s>.+?) (?:in|located in)$","p131",0.7),
 (r"^(?:(?:who|what|which) (?:is|was|are|were) |whats |what s )?(?:the )?(?:current |first |original |actual |real )?(?P<n>"+"|".join(sorted(NOUN,key=len,reverse=True))+r") (?:of|for|to|in) (?P<s>.+)$","noun",0.85),
 (r"^(?:what|whats) (?:is |was )?(?P<s>.+?)s? (?P<n>capital|capital city|population|currency|national anthem|official language|nickname|real name|birth name|full name|birthplace|hometown|nationality|atomic number|chemical symbol|highest point|area|religion)$","noun",0.8),
 (r"^who (?:is|was|are|were) (?P<s>.+?)s? (?P<n>wife|husband|spouse|father|dad|mother|mom|son|daughter|children|kids|brother|sister|siblings)$","noun",0.8),
 (r"^who (?:is|was) (?P<s>.+?) married to$","p26",0.8),
 (r"^who (?:did|does) (?P<s>.+?) marry$","p26",0.8),
 (r"^(?:the )?(?P<n>capital|capital city|population|currency|national anthem|official language) of (?P<s>.+)$","noun",0.8),
 (r"^(?P<s>.+?) (?:singer|sung by|is sung by|was sung by)$","p175",0.8),
 (r"^who (?:is|was|are|were) the (?:star|stars|main star|lead actor|lead actress|main actor|main actress|lead) (?:of|in) (?P<s>.+)$","p161",0.6),
 (r"^where (?:is|are|was|were) (?:the )?(?P<s>.+?) (?:located|situated|held|headquartered|based)(?: in (?:the )?(?:us|usa|united states|uk|world|country))?$","p276",0.6),
 (r"^who (?:originally |first |actually |really )?(?P<v>\w+)(?: the (?P<m>song|single|book|novel|movie|film|album|poem|play|story|lyrics to|lyrics for|lyrics of|music for|music to|music of|score for|score of|score to|screenplay for|screenplay of|script for|script of|theme song to|theme song for|theme song of|theme to|theme for|theme song|original song|original))? (?P<s>.+)$","who",0.9)]
RULES=[(re.compile(a),b,c) for a,b,c in RULES]
MODS={"lyrics":[676,86,50],"music":[86,676],"score":[86],"screenplay":[58],"script":[58],"theme":[86,175,676],"book":[50],"novel":[50],"poem":[50],"story":[50],"play":[50],"song":[86,676,50]}
def _answer(q:str)->dict|None:
 S=store()
 if S is None or not q:return None
 x=nrm(q);W,H=tables()
 for rx,kind,base in RULES:
  m=rx.match(x)
  if not m:continue
  g=m.groupdict();s=(g.get("s") or "").strip();r=None
  if kind=="cast":r=cast(S,g["r"],s,x,base)
  elif kind=="rcast":r=rcast(S,g["a"],s,x,base)
  elif kind=="role":r=role_only(S,g["r"],x,base)
  elif kind=="when":
   v=lem(g["v"]);key=f"{v} {g['pt']}" if g.get("pt") else v;P=W.get(key) or W.get(g["v"])
   r=P and ent_answer(S,s,x,P,"earliest" if P[0] in (577,580,1191,619) else "",base,yo=x.startswith(("what year","in what year","in which year","which year")),rule=f"when:{key}")
  elif kind=="born":r=ent_answer(S,s,x,[569],"",base,rule="born")
  elif kind=="noun":P=NOUN.get(g["n"]);r=P and ent_answer(S,s,x,P,"cur" if P[0] in CUR else "latest" if P[0]==1082 else "",base*(0.85 if P[0] in CUR and P[0]!=1082 else 1.0),rule=f"noun:{g['n']}")
  elif kind=="who":
   P=H.get(lem(g["v"]));mm=(g.get("m") or "").split(" ")[0];P=P and (MODS.get(mm,P) if P[0] in (50,86,676,58) else P);by=None
   if P and " by " in s:s,by=s.rsplit(" by ",1)
   r=P and ent_answer(S,s,x,P,"",base,by=by,rule=f"who:{lem(g['v'])}")
  else:
   p=int(kind[1:]);r=ent_answer(S,s,x,[p,*([2044] if p==2048 else [2047] if p==2043 else [159,131,17] if p==276 else [])],"cur" if p in CUR and p!=1082 else "latest" if p==1082 else "",base,rule=kind)
  if r:return {**r,"q":q}
 return None
def ask(q:str)->dict|None:
 try:return _answer((q or "")[:400])
 except Exception as e:return {"error":repr(e),"conf":0.0,"answer":None}
def answer(question:str)->tuple|None:
 r=ask(question);return (r["answer"],float(r["conf"]),r["source"]) if r and r.get("answer") and r["conf"]>=THR and (TEMPORAL or not r.get("temporal")) else None
def build(d:Path=DIR)->dict:
 t0=time.time();d=Path(d);ne=len(np.load(d/"e_qid.npy",mmap_mode="r"));O=np.zeros(ne+1,np.uint64);cur,buf,off,n=-1,[],0,0;CH,ch=[],[[],[]]
 def cut():
  ch[0] and CH.append((np.asarray(ch[0],np.uint64),np.asarray(ch[1],np.uint32)));ch[0],ch[1]=[],[]
 with open(d/"nm.bin.tmp","wb") as nb:
  def flush(buf,off):
   b=("\n".join(dict.fromkeys(buf))+"\n").encode() if buf else b"";nb.write(b);return off+len(b)
  for ln in open(d/"names.tsv",encoding="utf-8",errors="replace"):
   p=ln.rstrip("\n").split("\t",2);n+=1
   if len(p)<3:continue
   e,k,s=int(p[0]),int(p[1]),p[2];x=nrm(s)
   if e!=cur:
    off=flush(buf,off) if cur>=0 else off;O[cur+1:e+1]=off;cur,buf=e,[]
   if not x:continue
   buf.append(x)
   for v,kk in [(x,k)]+([(x[4:],k)] if x.startswith("the ") and len(x)>4 else [])+([(nrm(re.sub(r"\s*\([^)]*\)\s*$","",s)),3)] if k==2 and s.rstrip().endswith(")") else []):
    v and (ch[0].append(h64(v)),ch[1].append(e<<2|min(kk,3)))
   len(ch[0])>=1<<21 and cut()
  off=flush(buf,off) if cur>=0 else off;O[cur+1:]=off;cut()
 os.replace(d/"nm.bin.tmp",d/"nm.bin");np.save(d/"nm_o.npy",O);del O
 H=np.concatenate([c[0] for c in CH]) if CH else np.zeros(0,np.uint64);EK=np.concatenate([c[1] for c in CH]) if CH else np.zeros(0,np.uint32);CH.clear()
 o=np.argsort(H,kind="stable");H=H[o];EK=EK[o];del o;u=np.ones(len(H),bool);u[1:]=(H[1:]!=H[:-1])|((EK[1:]>>2)!=(EK[:-1]>>2))
 np.save(d/"a_h.npy",H[u]);del H;EK=EK[u];nk=int(u.sum());del u;np.save(d/"a_e.npy",EK>>2);np.save(d/"a_k.npy",(EK&3).astype(np.uint8));del EK
 fk=np.load(d/"f_k.npy",mmap_mode="r");fv=np.load(d/"f_v.npy",mmap_mode="r");m=np.flatnonzero(np.asarray(fk)==1).astype(np.uint32);v=np.asarray(fv[m]).astype(np.uint32);o=np.argsort(v,kind="stable");np.save(d/"r_v.npy",v[o]);np.save(d/"r_f.npy",m[o]);del m,v,o
 qo=np.load(d/"f_qo.npy",mmap_mode="r");qk=np.load(d/"q_k.npy",mmap_mode="r");qp=np.load(d/"q_p.npy",mmap_mode="r");m=np.flatnonzero((np.asarray(qk)==1)&(np.asarray(qp)==453));qf=(np.searchsorted(np.asarray(qo),m,side="right")-1).astype(np.uint32);v=np.asarray(np.load(d/"q_v.npy",mmap_mode="r")[m]).astype(np.uint32);o=np.argsort(v,kind="stable");np.save(d/"rq_v.npy",v[o]);np.save(d/"rq_f.npy",qf[o]);del m,v,o,qf
 pk=json.loads((d/"pack.json").read_text()) if (d/"pack.json").exists() else {};ex=json.loads((d.parent/"build"/"extract.json").read_text()) if (d.parent/"build"/"extract.json").exists() else {}
 man={"license":"CC0 1.0 (Wikidata); no attribution required, provenance kept","source":"https://dumps.wikimedia.org/wikidatawiki/entities/20260921/wikidata-20260921-all.json.gz","mirrors":["dumps.wikimedia.your.org","ftp.acc.umu.se","dumps.wikimedia.org"],"dump_md5_expected":"933a3fdb521eb2bdec761c021b6675e6","dump_md5_stream":ex.get("md5"),"dump_bytes":ex.get("gz_bytes"),"extract":ex,"pack":pk,"names":int(n),"alias_keys":nk,"entities":int(ne),"filter":"structured triples + short English/mul labels and aliases; neutral text filter not applicable; no PII contact properties kept (P968/P1329 excluded)","props":"crates/adam-wikifact/src/lib.rs KEEP/QKEEP","built":time.strftime("%Y-%m-%d %H:%M:%S"),"sec":round(time.time()-t0,1)}
 (d/"manifest.json").write_text(json.dumps(man,indent=1));return man
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--build",action="store_true");ap.add_argument("--dir",default=str(DIR));ap.add_argument("q",nargs="*");a=ap.parse_args()
 a.build and print(json.dumps(build(Path(a.dir)),indent=1));[print(json.dumps({k:v for k,v in (ask(" ".join(a.q)) or {}).items() if k!="forms"})) for _ in [0] if a.q]
