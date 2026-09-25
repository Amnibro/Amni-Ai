import functools,os,re,zlib
import numpy as np
from pathlib import Path
from amni.compute import tile_synth as TS,code_rank as CR,eval_lock
ROOT=Path(__file__).resolve().parents[2];MODEL=ROOT/"data"/"code_eval"/"x11_steer.npz"
def facts(sp):
 m=CR.memory(sp);ws=set(m.get("doc","words",[]))|set(CR.words(sp.name));return sorted({"b"}|{"w:"+TS._stem(w) for w in ws}|{f"#{x}" for w in ws for x in CR.syn(w)[:1]}|{"o:"+o for o in m.get("op","want",[])}|{f"a:{m.get(p,'type','')}" for p in sp.params}|{f"r:{m.get('ret','type','')}"}|{"y:"+t for t in m.ents("says")}|{"p:"+p for p in m.get("prop","want",[])}|{f"n:{len(sp.params)}"}|{"c:"+c for c in sp.cues}|(iof(sp) if os.environ.get("TS_STEER_IO","1")=="1" else set()))
def _ty(v):t=type(v).__name__;return t+("["+type(v[0]).__name__+"]" if t in ("list","tuple") and v and all(type(e) is type(v[0]) for e in v) else "")
def _sq(a,b):it=iter(b);return all(c in it for c in a)
def iof(sp):
 R=[(sp.rows[r],sp.T[r]) for r in sorted(sp.T) if r<len(sp.rows)];F={f"nx:{min(len(R),4)}"}
 if not R:return F
 O=[o for _,o in R];F|={"ot:"+_ty(o) for o in O};F|={f"it{k}:{_ty(x)}" for a,_ in R for k,x in enumerate(a[:3])};F|={"ob:mixed"} if all(type(o) is bool for o in O) and len(set(O))>1 else set()
 Q=lambda f:all(_safe(f,a,o) for a,o in R)
 for k in range(min(3,len(R[0][0]))):
  for nm,f in (("leneq",lambda a,o:hasattr(o,"__len__") and hasattr(a[k],"__len__") and len(o)==len(a[k])),("longer",lambda a,o:hasattr(o,"__len__") and hasattr(a[k],"__len__") and len(o)>len(a[k])),("shorter",lambda a,o:hasattr(o,"__len__") and hasattr(a[k],"__len__") and len(o)<len(a[k])),("substr",lambda a,o:type(o) is str and type(a[k]) is str and o in a[k]),("subseq",lambda a,o:type(o) in (str,list) and type(a[k]) is type(o) and _sq(o,a[k])),("perm",lambda a,o:type(o) in (str,list) and type(a[k]) is type(o) and sorted(map(repr,o))==sorted(map(repr,a[k]))),("srt",lambda a,o:type(o) is list and type(a[k]) is list and o==sorted(a[k])),("elem",lambda a,o:type(a[k]) in (list,str) and type(o) is not bool and o in a[k]),("chars",lambda a,o:type(o) is str and type(a[k]) is str and set(o)<=set(a[k])),("idx",lambda a,o:type(o) is int and type(o) is not bool and hasattr(a[k],"__len__") and 0<=o<len(a[k])),("lelen",lambda a,o:type(o) is int and hasattr(a[k],"__len__") and 0<=o<=len(a[k])),("same",lambda a,o:repr(o)==repr(a[k])),("inrng",lambda a,o:type(o) in (int,float) and type(a[k]) is list and a[k] and all(type(e) in (int,float) for e in a[k]) and min(a[k])<=o<=max(a[k])),("gt",lambda a,o:type(o) in (int,float) and type(a[k]) in (int,float) and o>a[k]),("lt",lambda a,o:type(o) in (int,float) and type(a[k]) in (int,float) and o<a[k]),("div",lambda a,o:type(o) is int and type(a[k]) is int and o and a[k]%o==0),("up",lambda a,o:type(o) is str and type(a[k]) is str and o.upper()==o!=o.lower()),("low",lambda a,o:type(o) is str and type(a[k]) is str and o.lower()==o!=o.upper())):F|={f"r{k}:{nm}"} if Q(f) else set()
 ws=[w for w in CR.words(sp.desc) if w not in TS.STOP];F|={"g:"+TS._stem(a)+"_"+TS._stem(b) for a,b in zip(ws,ws[1:])};return F
def _safe(f,a,o):
 try:return bool(f(a,o))
 except Exception:return False
@functools.lru_cache(None)
def targets():return tuple(f for f in TS.vocab() if f[0] in "cmops" or f in ("k:key","k:reverse"))
def labels(code):
 try:F=set(TS.feats(code))
 except Exception:return None
 return np.array([f in F for f in targets()],bool)
def keys(fs,T=None):T=T or targets();return np.array([[CR.hk(f+"|"+t) for t in T] for f in fs],np.int64).reshape(len(fs),len(T))
def train(X,Y,it=120,lr=0.3,l2=0.02,pw=3.0):
 T=len(targets());w=np.zeros(CR.M,np.float64);h=np.full(CR.M,1e-8);K=[keys(x) for x in X];P=np.concatenate(K,0);R=np.repeat(np.arange(len(K)),[len(k) for k in K]);Yf=np.asarray(Y,float)
 for _ in range(it):
  s=np.zeros((len(K),T));np.add.at(s,R,w[P]);p=1/(1+np.exp(-s));g=(p-Yf)*np.where(Yf>0,pw,1.0);gw=np.bincount(P.ravel(),g[R].ravel(),CR.M)+l2*w;h+=gw*gw;w-=lr*gw/np.sqrt(h)
 return CR.Model(w.astype(np.float32),{"kind":"x11_steer","targets":list(targets()),"it":it,"lr":lr,"l2":l2,"pw":pw})
def score(model,fs):return np.take(model.w,keys(fs,tuple(model.cfg["targets"]))).sum(0)
@functools.lru_cache(None)
def model(path=None):p=Path(os.environ.get("TS_STEER_MODEL",str(MODEL)) if path in (None,"","on","1") else path);return CR.Model.load(p) if p.exists() else None
def focus(sp,k=None,path=None):
 m=model(path)
 if m is None:return None
 s=score(m,facts(sp));T=m.cfg["targets"];o=np.argsort(-s);k=int(k or os.environ.get("TS_STEER_K","12"));return {T[j]:float(s[j]) for j in o[:k]},{T[j]:float(s[j]) for j in o}
def fit_rows(rows,prompt_of,checks_of):
 eval_lock.assert_trainable(rows,"the X11 steering table");X,Y,ids=[],[],[]
 for r in rows:
  y=labels(r.get("canonical") or "")
  try:sp=TS.Spec(prompt_of(r),checks_of(r))
  except Exception:continue
  (X.append(facts(sp)),Y.append(y),ids.append(r["id"])) if y is not None and y.any() else None
 return X,np.array(Y),ids
MEMP=ROOT/"data"/"code_eval"/"x11_memory.jsonl"
def shingles(t,n=13):w=re.findall(r"[a-z0-9]+",t.lower());return {zlib.crc32(" ".join(w[i:i+n]).encode()) for i in range(max(0,len(w)-n+1))}
def mwords(t):return [TS._stem(w) for w in CR.words(t) if w not in TS.STOP and len(w)>2]
@functools.lru_cache(None)
def memory(path=None):
 p=Path(os.environ.get("TS_MEMORY",str(MEMP)) if path is None else path)
 if not p.exists():return None
 import json;R=[json.loads(l) for l in open(p,encoding="utf-8")];df={};[df.__setitem__(w,df.get(w,0)+1) for r in R for w in set(r["words"])];N=len(R);avg=sum(len(r["words"]) for r in R)/max(1,N)
 return R,{w:np.log(1+(N-c+0.5)/(c+0.5)) for w,c in df.items()},avg,[set(r["g13"]) for r in R]
def recall(sp,k=12):
 M=memory()
 if M is None:return []
 R,idf,avg,G=M;q=set(mwords(sp.desc+" "+sp.name));g=shingles(sp.desc);fk=(sp.name,re.sub(r"\s+"," ",sp.desc)[:200]);sc=[]
 for j,r in enumerate(R):
  if G[j]&g or (r["fk"][0],r["fk"][1])==fk:continue
  tf={};[tf.__setitem__(w,tf.get(w,0)+1) for w in r["words"]];s=sum(idf.get(w,0)*tf[w]*2.2/(tf[w]+1.2*(0.25+0.75*len(r["words"])/avg)) for w in q if w in tf)
  (sc.append((s,j)) if s>0 else None)
 return [R[j] for s,j in sorted(sc,reverse=True)[:k]]
