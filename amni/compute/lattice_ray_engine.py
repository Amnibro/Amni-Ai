import os,sys,math,re
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.ptex_300b_store import Ptex300BResidentStore
from amni.compute.ptex_1t_store import Ptex1TResidentStore,default_1t_path,PASSAGE_MISS
class LatticeGuidedRayEngine:
 def __init__(self,store=None,n_fib:int=256):
  p1t=default_1t_path()
  p300b=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
  def_store=Ptex1TResidentStore(p1t) if os.path.exists(p1t) else Ptex300BResidentStore(p300b)
  self.store=store if store is not None else def_store
  self.n_fib=n_fib
  idx=np.arange(n_fib,dtype=np.float32)
  phi=(1.0+math.sqrt(5.0))/2.0
  y=1.0-(2.0*idx+1.0)/float(n_fib)
  r=np.sqrt(np.maximum(0.0,1.0-y*y))
  theta=2.0*math.pi*idx/phi
  self.fib_pts=np.stack([r*np.cos(theta),y,r*np.sin(theta)],axis=1)
 def embed_query_ray(self,query:str)->tuple[np.ndarray,np.ndarray]:
  b=query.encode("utf-8",errors="ignore")
  h=sum((i+1)*b[i] for i in range(len(b)))%self.n_fib if len(b)>0 else 0
  pos=self.fib_pts[h].copy()
  vel=np.array([math.cos(h*0.05),math.sin(h*0.05),math.cos(h*0.02)],dtype=np.float32)
  vel=vel/max(1e-6,np.linalg.norm(vel))
  return pos,vel
 def compute_guidance_vector(self,stage_idx:int,total_stages:int)->np.ndarray:
  target_idx=int((stage_idx/max(1,total_stages))*self.n_fib)%self.n_fib
  target_pt=self.fib_pts[target_idx]
  return target_pt/max(1e-6,np.linalg.norm(target_pt))
 def generate_guided(self,domain:str,query:str)->str:
  pos,vel=self.embed_query_ray(query)
  from amni.compute.resident_code import coverage_ok,lang_domain,emit_verified
  from amni.compute.ptex_1t_store import CODE_SUBRANGES_1T,DOMAINS_1T
  kws=re.findall(r"[A-Za-z+]{3,}",query)
  lang=lang_domain(query,kws)
  known=set(CODE_SUBRANGES_1T)|set(DOMAINS_1T)|{"code"}
  code_intent=(domain in CODE_SUBRANGES_1T or domain=="code")
  if not code_intent:
   passage=self.store.extract_resident_passage(domain,kws,max_bytes=1024) if (self.store and hasattr(self.store,"extract_resident_passage")) else ""
   if passage:return passage
   return PASSAGE_MISS
  rec=emit_verified(query,self.store,anchors=kws)
  miss="No parseable resident function matched this query. I will not invent one."
  if rec.get("code"):
   res=rec["code"]
  else:
   return miss
  if domain in ("rust","cpp") or lang in ("rust","cpp","zig"):
   diff=res.count("{")-res.count("}")
   if diff>0:res+="\n}"*diff
   elif diff<0:
    for _ in range(-diff):
     idx=res.rfind("}")
     if idx>=0:res=res[:idx]+res[idx+1:]
  return res
