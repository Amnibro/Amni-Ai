import os,sys,time,math,concurrent.futures
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.lattice_ray_engine import LatticeGuidedRayEngine
from amni.compute.ptex_1t_store import Ptex1TResidentStore
from amni.compute.ptex_300b_store import Ptex300BResidentStore
class AsyncRayHarness:
 def __init__(self,engine:LatticeGuidedRayEngine=None,store=None):
  if engine is not None:self.engine=engine
  else:
   p1t=os.path.join(_H0,"exports","gf17_continuum","adam_1t_store.ptex")
   p300b=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
   s=store if store is not None else (Ptex1TResidentStore(p1t) if os.path.exists(p1t) else Ptex300BResidentStore(p300b))
   self.engine=LatticeGuidedRayEngine(store=s)
  self.store=self.engine.store
 def run_query_single(self,domain:str,query:str)->dict:
  t0=time.perf_counter()
  text=self.engine.generate_guided(domain,query)
  dt=time.perf_counter()-t0
  toks=len(text)/4.0
  return {"domain":domain,"query":query,"text":text,"length_chars":len(text),"tokens":toks,"latency_ms":dt*1000.0,"tokens_per_sec":toks/max(1e-9,dt)}
 def run_parallel_queries(self,query_list:list,num_workers:int=32)->dict:
  t0=time.perf_counter()
  results=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
   futs=[ex.submit(self.run_query_single,dom,q) for dom,q in query_list]
   for f in concurrent.futures.as_completed(futs):results.append(f.result())
  total_dt=time.perf_counter()-t0
  total_chars=sum(r["length_chars"] for r in results)
  total_tokens=total_chars/4.0
  return {"num_workers":num_workers,"total_queries":len(query_list),"total_time_ms":total_dt*1000.0,"total_chars":total_chars,"total_tokens":total_tokens,"chars_per_sec":total_chars/max(1e-9,total_dt),"tokens_per_sec":total_tokens/max(1e-9,total_dt),"avg_latency_ms":(total_dt*1000.0)/max(1,len(query_list)),"results":results}
 def run_batch_vectorized_bundle(self,domains:list,queries:list)->dict:
  t0=time.perf_counter()
  B=len(queries)
  n_fib=self.engine.n_fib
  fib_pts=self.engine.fib_pts
  pos_list,vel_list=[],[]
  for q in queries:
   b=q.encode("utf-8",errors="ignore")
   h=sum((i+1)*b[i] for i in range(len(b)))%n_fib if len(b)>0 else 0
   p=fib_pts[h].copy()
   v=np.array([math.cos(h*0.05),math.sin(h*0.05),math.cos(h*0.02)],dtype=np.float32)
   v=v/max(1e-6,float(np.linalg.norm(v)))
   pos_list.append(p)
   vel_list.append(v)
  P=np.stack(pos_list,axis=0)
  V=np.stack(vel_list,axis=0)
  results_text=[]
  for i in range(B):
   dom=domains[i]
   q=queries[i]
   results_text.append(self.engine.generate_guided(dom,q))
  total_dt=time.perf_counter()-t0
  total_chars=sum(len(t) for t in results_text)
  total_tokens=total_chars/4.0
  return {"batch_size":B,"total_time_ms":total_dt*1000.0,"total_chars":total_chars,"total_tokens":total_tokens,"chars_per_sec":total_chars/max(1e-9,total_dt),"tokens_per_sec":total_tokens/max(1e-9,total_dt),"results":results_text}