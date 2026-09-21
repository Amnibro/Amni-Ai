"""Resident 1.2T ptex + omni pack folded into T²×S² (pack, Λ, n-gram tables)."""
from __future__ import annotations
import json,os,sys
from collections import defaultdict,Counter
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCR=os.path.join(_H0,"scripts")
if _SCR not in sys.path:sys.path.insert(0,_SCR)
if _H0 not in sys.path:sys.path.insert(0,_H0)
from gf17_ray_t2s2 import CENTS,N,NBR,greedy_pack,inv_perm
import math
from amni.compute.ptex_1t_store import Ptex1TResidentStore,DOMAINS_1T,CODE_SUBRANGES_1T,TOTAL_PAGES_1T,PAGE_DIM,TOTAL_PARAMS_1T,default_1t_path
T5BIN=os.path.join(_H0,"exports","gf17_continuum","adam_1t_t5_1p2.bin")
T5_500=os.path.join(_H0,"exports","gf17_continuum","adam_500b_t5_1p2.bin")
_CACHE=None
OMNI=os.path.join(_H0,"exports","gf17_continuum","adam_omni_model.json")
PTEX=default_1t_path()
PTEX300=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
DOMS=tuple(DOMAINS_1T.keys())+tuple(CODE_SUBRANGES_1T.keys())

def _as_counter_map(raw)->dict:
 out=defaultdict(Counter)
 if not raw:return out
 for k,v in raw.items():
  kk=int(k)
  if isinstance(v,dict):
   out[kk].update({int(a):int(b) for a,b in v.items()})
  else:
   for i,b in enumerate(v):
    out[kk][int(b)]+=max(1,3-i)
 return out

def _lam_from_pnbr(p_nbr)->np.ndarray:
 arr=np.asarray(p_nbr,dtype=np.float64)
 lam=np.ones((N,7),dtype=np.float64)
 if arr.ndim==2 and arr.shape[0]==N:
  w=arr[:,:6] if arr.shape[1]>=6 else arr
  lam[:,1:1+w.shape[1]]=np.maximum(w,1e-6)
 lam=lam/np.maximum(1e-6,lam.sum(axis=1,keepdims=True))*7.0
 return lam

def _fit_lam(rows,pack)->np.ndarray:
 lam=np.zeros((N,7),dtype=np.float64)
 for r in rows:
  for a,b in zip(r,r[1:]):
   ca,cb=int(pack[int(a)&255]),int(pack[int(b)&255])
   nbrs=[ca]+[int(x) for x in NBR[ca]]
   if cb in nbrs:lam[ca,nbrs.index(cb)]+=2.0
 lam+=0.05
 return lam/np.maximum(1e-6,lam.sum(axis=1,keepdims=True))*7.0

def _page_rows(store,pages_per_dom=48,win=64,stride=16):
 rows=[]
 for dom in DOMS:
  r=store.get_domain_range(dom)
  for p in range(r[0],min(r[1],r[0]+pages_per_dom)):
   tile=store.read_tile_mmap(p,tile_size=getattr(store,"page_dim",PAGE_DIM))
   if not tile or len(set(tile[:256]))<3:continue
   for i in range(0,len(tile)-win,stride):
    rows.append(list(tile[i:i+win+1]))
 return rows

class PtexManifold:
 def __init__(self,store=None):
  self.store=store
  self.pack=np.arange(N,dtype=np.int64)
  self.inv=np.arange(N,dtype=np.int64)
  self.lam=np.ones((N,7),dtype=np.float64)
  self.g3=defaultdict(Counter)
  self.g2=defaultdict(Counter)
  self.g1=defaultdict(Counter)
  self.uni=list(range(32,127))
  self.source="identity"
  self.n_ptex=0
  self.t5=None
  self._eph_g3=defaultdict(Counter)
  self._eph_g2=defaultdict(Counter)
  self._eph_g1=defaultdict(Counter)
  self._eph_pages=[]
  self._eph_active=False
  self._load()
 def _open_store(self):
  if self.store is not None:return self.store
  if os.path.exists(PTEX):
   self.store=Ptex1TResidentStore(PTEX);return self.store
  if os.path.exists(PTEX300):
   from amni.compute.ptex_300b_store import Ptex300BResidentStore
   self.store=Ptex300BResidentStore(PTEX300)
  return self.store
 def _load(self):
  if os.path.exists(OMNI):
   raw=json.load(open(OMNI,encoding="utf-8"))
   self.pack=np.array(raw["pack"],dtype=np.int64)
   self.inv=np.array(raw.get("inv") or inv_perm(self.pack),dtype=np.int64)
   if "p_nbr" in raw:self.lam=_lam_from_pnbr(raw["p_nbr"])
   self.g3=_as_counter_map(raw.get("top_g3"))
   self.g2=_as_counter_map(raw.get("c2"))
   self.g1=_as_counter_map(raw.get("c1"))
   for _d,tbl in (raw.get("top_dom_g3") or {}).items():
    extra=_as_counter_map(tbl)
    for k,c in extra.items():self.g3[k].update(c)
   self.source="omni"
  st=self._open_store()
  if st is None:return
  n=0
  for dom in DOMS:
   try:t3,t2,t1,uni=st.read_domain_transitions(dom)
   except Exception:continue
   for k,vs in t3.items():
    self.g3[int(k)].update(int(x) for x in vs);n+=len(vs)
   for k,vs in t2.items():
    self.g2[int(k)].update(int(x) for x in vs)
   for k,vs in t1.items():
    self.g1[int(k)].update(int(x) for x in vs)
   for u in uni:
    if u not in self.uni:self.uni.append(u)
  self.n_ptex=n
  if self.source=="identity":
   rows=_page_rows(st)
   if len(rows)>=8:
    self.pack=greedy_pack(rows,len(rows))
    self.inv=inv_perm(self.pack)
    self.lam=_fit_lam(rows,self.pack)
    self.source="ptex_fit"
  elif np.allclose(self.lam,1.0):
   rows=_page_rows(st,pages_per_dom=16)
   if rows:self.lam=_fit_lam(rows,self.pack)
  if self.source=="omni" and self.n_ptex:self.source="omni+ptex"
  t5p=T5_500 if os.path.exists(T5_500) else (T5BIN if os.path.exists(T5BIN) else None)
  if t5p:
   from amni.compute.ptex_t5_1p2 import T5FoldStore
   try:self.t5=T5FoldStore(t5p)
   except Exception:self.t5=None
 def candidates(self,buf:list)->tuple:
  if len(buf)>=3:
   k3=((buf[-3]<<16)|(buf[-2]<<8)|buf[-1])&0xFFFFFF
   if self.g3[k3]:return list(self.g3[k3].keys()),self.g3[k3]
  if len(buf)>=2:
   k2=((buf[-2]<<8)|buf[-1])&0xFFFF
   if self.g2[k2]:return list(self.g2[k2].keys()),self.g2[k2]
  last=int(buf[-1])&255 if buf else 32
  if self.g1[last]:return list(self.g1[last].keys()),self.g1[last]
  return list(self.uni[:48]),Counter({u:1 for u in self.uni[:48]})
 def _ensure_eph(self):
  if not hasattr(self,"_eph_g3"):
   self._eph_g3=defaultdict(Counter)
   self._eph_g2=defaultdict(Counter)
   self._eph_g1=defaultdict(Counter)
   self._eph_pages=[]
   self._eph_active=False
 def prime_context(self,tiles,weight=140):
  self._ensure_eph()
  self.clear_ephemeral()
  w=max(1,int(weight))
  if not tiles:return
  self._eph_pages=list(getattr(self.store,"last_candidate_pages",[]) or [])
  self._eph_active=True
  added=0
  for tile in tiles:
   raw=bytes(tile) if not isinstance(tile,(bytes,bytearray)) else bytes(tile)
   n=len(raw)
   if n<2:continue
   added+=n
   for i in range(n-1):
    a=raw[i]&255;b=raw[i+1]&255
    self.g1[a][b]+=w;self._eph_g1[a][b]+=w
    if i+2<n:
     c=raw[i+2]&255
     k2=(a<<8)|b
     self.g2[k2][c]+=w;self._eph_g2[k2][c]+=w
     if i+3<n:
      d=raw[i+3]&255
      k3=(a<<16)|(b<<8)|c
      self.g3[k3][d]+=w;self._eph_g3[k3][d]+=w
      if d not in self.uni:self.uni.append(d)
  self.n_ptex=int(self.n_ptex)+added
 def clear_ephemeral(self):
  self._ensure_eph()
  if not getattr(self,"_eph_active",False):
   self._eph_pages=[]
   return
  for tbl,eph in ((self.g3,self._eph_g3),(self.g2,self._eph_g2),(self.g1,self._eph_g1)):
   for k,c in eph.items():
    for b,n in list(c.items()):
     tbl[k][b]-=n
     if tbl[k][b]<=0:del tbl[k][b]
    if k in tbl and not tbl[k]:del tbl[k]
  self._eph_g3=defaultdict(Counter)
  self._eph_g2=defaultdict(Counter)
  self._eph_g1=defaultdict(Counter)
  self._eph_pages=[]
  self._eph_active=False
 def pick_byte(self,buf,v,tau=0.45,creative=False,rng=None)->int:
  cands,dist=self.candidates(buf)
  if not cands:return 32
  cells=np.array([int(self.pack[int(b)&255]) for b in cands],dtype=np.int64)
  vn=np.asarray(v,dtype=np.float64).reshape(3)
  n=float(np.linalg.norm(vn));vn=vn/max(n,1e-12)
  dots=(CENTS[cells]*vn).sum(axis=1)
  counts=np.array([float(dist[b]) for b in cands],dtype=np.float64)
  scores=np.log(np.maximum(counts,1e-6))+dots/max(float(tau),1e-6)
  if len(buf)>=8:
   recent=buf[-12:]
   for i,b in enumerate(cands):
    if b not in (32,10,44,46) and recent.count(b)>2:scores[i]-=2.4
  if creative and rng is not None and len(cands)>1:
   p=np.exp(scores-float(np.max(scores)));p=p/p.sum()
   return int(cands[int(rng.choice(len(cands),p=p))])
  return int(cands[int(np.argmax(scores))])
 def walk_tile(self,th,ph,St)->int:
  st=self.store
  if st is None:return 32
  twopi=2.0*math.pi
  u=int((float(th)%twopi)*(1<<20))
  v=int((float(ph)%twopi)*(1<<20))
  pages=getattr(self,"_eph_pages",None) or []
  if pages:
   page=pages[(u+int(St))%len(pages)]
   tile=st.read_tile_mmap(int(page),tile_size=getattr(st,"page_dim",PAGE_DIM))
   if tile:return int(tile[v%len(tile)])
  virt=(u*104729+v*224737+int(St)*2654435761)&0xFFFFFFFFFFFFFFFF
  virt%=TOTAL_PARAMS_1T
  if self.t5 is not None:return self.t5.walk_virt(virt)
  if hasattr(st,"walk_virt"):return st.walk_virt(virt)
  page=int(u)%TOTAL_PAGES_1T
  tile=st.read_tile_mmap(page,tile_size=getattr(st,"page_dim",PAGE_DIM))
  if not tile:return 32
  return int(tile[v%len(tile)])

def load_manifold(store=None,force=False)->PtexManifold:
 global _CACHE
 if _CACHE is None or force or (store is not None and _CACHE.store is not store):
  _CACHE=PtexManifold(store=store)
 return _CACHE

def invalidate():
 global _CACHE
 _CACHE=None
