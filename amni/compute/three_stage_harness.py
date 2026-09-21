import time
from amni.compute.toroidal_steering import ToroidalParallelSteeringHarness
from amni.compute.lattice_ray_engine import LatticeGuidedRayEngine
class ThreeStageRayHarness:
 """Ingress/actor/critic collapsed onto T²×S² parallel steering threads."""
 def __init__(self,engine:LatticeGuidedRayEngine=None,store=None):
  self.engine=engine
  self.store=store
  self.steering=ToroidalParallelSteeringHarness(steps=64)
  self.history=[]
 def execute(self,user_msg:str,history:list=None,max_rearms:int=2)->dict:
  t0=time.perf_counter()
  hist=history if history is not None else self.history
  prior=b""
  if hist:
   last=hist[-1]
   chunk=(last.get("final_text") or last.get("text") or last.get("reply") or "")[-120:]
   prior=chunk.encode("utf-8","replace")
  q=user_msg if not prior else (prior.decode("utf-8","replace")+"\n"+user_msg)
  r=self.steering.run(q)
  r["history_turns"]=len(hist or [])
  r["rearms_triggered"]=int(max_rearms==0)
  r["total_latency_ms"]=round((time.perf_counter()-t0)*1000.0,3)
  self.history.append({"user":user_msg,"final_text":r["final_text"],"text":r["text"]})
  return r
