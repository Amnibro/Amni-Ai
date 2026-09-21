import os,sys,time
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
_SCR=os.path.join(_H0,"scripts")
if _SCR not in sys.path:sys.path.insert(0,_SCR)
from gf17_ray_t2s2 import CENTS,ray,step,lorentz_deflect
from amni.compute.ptex_1t_store import Ptex1TResidentStore
from amni.compute.ptex_300b_store import Ptex300BResidentStore
from amni.compute.ptex_manifold import load_manifold
from amni.compute.toroidal_steering import query_state,_unit,_B_delim,_B_indent,_stack_from,_push_delim
class EnergyTransitionSteerer:
 def __init__(self,store=None):
  from amni.compute.ptex_1t_store import default_1t_path
  p1t=default_1t_path()
  p300b=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
  if store is not None:self.store=store
  elif os.path.exists(p1t):self.store=Ptex1TResidentStore(p1t)
  else:self.store=Ptex300BResidentStore(p300b)
  self.manifold=load_manifold(store=self.store)
 def steer_continuation(self,domain:str,seed_prefix:str,target_keywords:list,max_tokens:int=60)->dict:
  t0=time.perf_counter()
  th,ph,St,v,seed=query_state(seed_prefix)
  out=list(seed)
  stack=_stack_from(seed)
  pack,lam=self.manifold.pack,self.manifold.lam
  for _ in range(int(max_tokens)):
   tb=int(self.manifold.walk_tile(th,ph,St))
   B=_B_delim(stack,pack)+_B_indent(out,pack)+0.35*CENTS[int(pack[tb])]
   v=_unit(lorentz_deflect(v,B,dt=0.12))
   bt,th,ph,St=step(th,ph,St,lam,3,17,greedy=True,use_magnetic=True,v_prev=v,dt=0.12)
   out.append(int(bt)&255)
   _push_delim(stack,bt)
   v=_unit(ray(th,ph))
   if chr(bt) in ".?!" and len(out)>=len(seed)+12 and not stack:break
  raw=bytes(out).decode("utf-8","replace")
  clean=" ".join(raw.split())
  dt=(time.perf_counter()-t0)*1000.0
  return {"text":clean,"latency_ms":round(dt,3),"tokens":len(out)-len(seed),"seam_score":78,"method":"energy_transition_steering","manifold":self.manifold.source}
