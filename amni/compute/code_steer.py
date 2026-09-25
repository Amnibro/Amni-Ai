import functools,os,re,zlib
import numpy as np
from pathlib import Path
from amni.compute import tile_synth as TS,code_rank as CR,eval_lock
ROOT=Path(__file__).resolve().parents[2];MODEL=ROOT/"data"/"code_eval"/"x11_steer.npz"
def facts(sp):
 m=CR.memory(sp);ws=set(m.get("doc","words",[]))|set(CR.words(sp.name));return sorted({"b"}|{"w:"+TS._stem(w) for w in ws}|{f"#{x}" for w in ws for x in CR.syn(w)[:1]}|{"o:"+o for o in m.get("op","want",[])}|{f"a:{m.get(p,'type','')}" for p in sp.params}|{f"r:{m.get('ret','type','')}"}|{"y:"+t for t in m.ents("says")}|{"p:"+p for p in m.get("prop","want",[])}|{f"n:{len(sp.params)}"}|{"c:"+c for c in sp.cues})
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
