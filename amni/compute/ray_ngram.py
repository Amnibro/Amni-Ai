import os,sys,json
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCR=os.path.join(_H0,"scripts")
if _SCR not in sys.path:sys.path.insert(0,_SCR)
if _H0 not in sys.path:sys.path.insert(0,_H0)
from gf17_ray_t2s2 import CENTS,N,omega_byte,wrap,ray
KMAX=8
P=np.uint64(1099511628211)
M56=np.uint64((1<<56)-1)
PRUNE={1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:2}
DISCOUNT=0.75
TABLES=os.path.join(_H0,"exports","gf17_continuum","ray_ngram_k8.npz")
META=os.path.join(_H0,"exports","gf17_continuum","ray_ngram_meta.json")
def context_hashes(a:np.ndarray)->list:
 a=np.asarray(a,dtype=np.uint64)
 n=a.size
 out=[None]
 prev=np.zeros(n,dtype=np.uint64)
 for k in range(1,KMAX+1):
  h=np.zeros(n,dtype=np.uint64)
  with np.errstate(over="ignore"):
   h[k:]=prev[k-1:n-1]*P+(a[k-1:n-1]+np.uint64(1))
  out.append(h)
  prev=h
 return out
def build_tables(corpus:bytes,prune:bool=True)->dict:
 a=np.frombuffer(corpus,dtype=np.uint8)
 n=a.size
 hs=context_hashes(a)
 t={}
 for k in range(1,KMAX+1):
  keys=((hs[k][k:]&M56)<<np.uint64(8))|a[k:].astype(np.uint64)
  u,c=np.unique(keys,return_counts=True)
  keep=c>=(PRUNE[k] if prune else 1)
  u,c=u[keep],c[keep].astype(np.int64)
  ctx=u>>np.uint64(8)
  if u.size==0:
   t[k]={"keys":u,"counts":c,"ctx":ctx,"tot":np.zeros(0,dtype=np.int64),"nty":np.zeros(0,dtype=np.int64)}
   continue
  starts=np.flatnonzero(np.r_[True,ctx[1:]!=ctx[:-1]])
  tot=np.add.reduceat(c,starts)
  nty=np.diff(np.r_[starts,u.size])
  t[k]={"keys":u,"counts":c,"ctx":ctx[starts],"tot":tot,"nty":nty.astype(np.int64)}
 uni=np.bincount(a,minlength=256).astype(np.float64)+1.0
 t["uni"]=uni/uni.sum()
 t["n"]=int(n)
 return t
def save_tables(t:dict,path:str=TABLES):
 os.makedirs(os.path.dirname(path),exist_ok=True)
 arrs={"uni":t["uni"],"n":np.array([t["n"]])}
 for k in range(1,KMAX+1):
  for name,arr in t[k].items():arrs[f"k{k}_{name}"]=arr
 np.savez(path,**arrs)
def load_tables(path:str=TABLES)->dict:
 z=np.load(path)
 t={"uni":z["uni"],"n":int(z["n"][0])}
 for k in range(1,KMAX+1):
  t[k]={name:z[f"k{k}_{name}"] for name in ("keys","counts","ctx","tot","nty")}
 return t
def _buf_hashes(buf)->list:
 hs=[0]
 h=0
 L=len(buf)
 for k in range(1,KMAX+1):
  if k>L:hs.append(None);continue
  h=0
  for b in buf[L-k:]:h=(h*1099511628211+(int(b)&255)+1)&0xFFFFFFFFFFFFFFFF
  hs.append(h&((1<<56)-1))
 return hs
class RayNgram:
 def __init__(self,tables:dict=None,pack=None,path:str=TABLES,discount:float=DISCOUNT):
  self.t=tables if tables is not None else load_tables(path)
  self.pack=np.arange(N,dtype=np.int64) if pack is None else np.asarray(pack,dtype=np.int64)
  self.D=float(discount)
  self.kmax=max(k for k in range(1,KMAX+1) if k in self.t)
 def probs(self,buf)->np.ndarray:
  p=np.array(self.t["uni"],dtype=np.float64)
  hs=_buf_hashes(buf)
  for k in range(1,min(self.kmax,len(buf))+1):
   tk=self.t[k]
   h=np.uint64(hs[k])
   lo=int(np.searchsorted(tk["keys"],h<<np.uint64(8),side="left"))
   hi=int(np.searchsorted(tk["keys"],(h<<np.uint64(8))|np.uint64(255),side="right"))
   if hi<=lo:continue
   ci=int(np.searchsorted(tk["ctx"],h))
   T=float(tk["tot"][ci]);nt=float(tk["nty"][ci])
   nb=(tk["keys"][lo:hi]&np.uint64(255)).astype(np.int64)
   c=tk["counts"][lo:hi].astype(np.float64)
   q=np.zeros(256,dtype=np.float64)
   q[nb]=np.maximum(c-self.D,0.0)/T
   p=q+(self.D*nt/T)*p
  return p/p.sum()
 def scores(self,buf,v=None,tau_ray:float=0.6,rep_window:int=24,rep_pen:float=1.5)->np.ndarray:
  s=np.log(self.probs(buf)+1e-12)
  if v is not None and tau_ray>0:
   vn=np.asarray(v,dtype=np.float64).reshape(3)
   vn=vn/max(1e-12,float(np.linalg.norm(vn)))
   s=s+(CENTS[self.pack]@vn)/float(tau_ray)
  if rep_pen>0 and len(buf)>=8:
   rec=np.bincount(np.asarray(buf[-rep_window:],dtype=np.int64)&255,minlength=256)
   mask=np.ones(256,dtype=bool);mask[[32,10,44,46,9]]=False
   s=s-rep_pen*np.clip(rec-2,0,None)*mask
  return s
 def next_byte(self,buf,v=None,temp:float=0.8,rng=None,greedy:bool=False,**kw)->int:
  s=self.scores(buf,v,**kw)
  if greedy or temp<=0:return int(np.argmax(s))
  z=(s-s.max())/float(temp)
  p=np.exp(z);p=p/p.sum()
  r=np.random.default_rng() if rng is None else rng
  return int(r.choice(256,p=p))
 def generate(self,prefix:bytes,n:int=300,temp:float=0.8,seed:int=0,steer:bool=True,th:float=0.7,ph:float=0.3,**kw)->bytes:
  rng=np.random.default_rng(seed)
  buf=list(prefix)
  th,ph=wrap(th,ph)
  for _ in range(n):
   v=ray(th,ph) if steer else None
   b=self.next_byte(buf,v,temp=temp,rng=rng,**kw)
   buf.append(b)
   dth,dph=omega_byte(b)
   th,ph=wrap(th+dth,ph+dph)
  return bytes(buf)
def load_pack()->np.ndarray:
 p=os.path.join(_H0,"exports","gf17_continuum","adam_omni_model.json")
 if os.path.exists(p):
  try:return np.array(json.load(open(p,encoding="utf-8"))["pack"],dtype=np.int64)
  except Exception:pass
 return np.arange(N,dtype=np.int64)
