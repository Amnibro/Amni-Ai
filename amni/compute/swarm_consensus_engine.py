import os,sys,time
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.lattice_ray_engine import LatticeGuidedRayEngine
class SwarmConsensusEngine:
 def __init__(self,engine:LatticeGuidedRayEngine=None):
  self.engine=engine if engine is not None else LatticeGuidedRayEngine()
 def propose(self,domain:str,query:str)->dict:
  t0=time.perf_counter()
  raw=self.engine.generate_guided(domain,query)
  dt=(time.perf_counter()-t0)*1000.0
  return {"role":"proposer","text":raw,"latency_ms":round(dt,3)}
 def critique(self,domain:str,text:str)->dict:
  t0=time.perf_counter()
  flaws=[]
  if domain in ("rust","cpp"):
   open_b,close_b=text.count("{"),text.count("}")
   if open_b!=close_b:flaws.append(f"Brace mismatch: {open_b} open vs {close_b} close")
  if domain=="cpp" and "template" in text and "struct" not in text and "class" not in text and "void" not in text and "auto" not in text:flaws.append("Incomplete template definition")
  if domain=="fortran" and "subroutine" in text and "end subroutine" not in text:flaws.append("Missing end subroutine termination")
  dt=(time.perf_counter()-t0)*1000.0
  is_valid=len(flaws)==0
  return {"role":"critic","valid":is_valid,"flaws":flaws,"latency_ms":round(dt,3)}
 def synthesize(self,domain:str,proposal:str,critique_res:dict)->dict:
  t0=time.perf_counter()
  out=proposal
  if not critique_res["valid"]:
   for f in critique_res["flaws"]:
    if "Brace mismatch" in f:
     diff=out.count("{")-out.count("}")
     if diff>0:out+="\n}"*diff
     elif diff<0:
      for _ in range(-diff):
       idx=out.rfind("}")
       if idx>=0:out=out[:idx]+out[idx+1:]
    if "Missing end subroutine" in f:out+="\nend subroutine"
  dt=(time.perf_counter()-t0)*1000.0
  return {"role":"synthesizer","final_text":out,"was_repaired":not critique_res["valid"],"latency_ms":round(dt,3)}
 def run_consensus(self,domain:str,query:str)->dict:
  t0=time.perf_counter()
  p_res=self.propose(domain,query)
  c_res=self.critique(domain,p_res["text"])
  s_res=self.synthesize(domain,p_res["text"],c_res)
  total_dt=(time.perf_counter()-t0)*1000.0
  toks=len(s_res["final_text"])/4.0
  return {"domain":domain,"query":query,"final_text":s_res["final_text"],"total_latency_ms":round(total_dt,3),"proposer_ms":p_res["latency_ms"],"critic_ms":c_res["latency_ms"],"synthesizer_ms":s_res["latency_ms"],"consensus_valid":c_res["valid"],"repaired":s_res["was_repaired"],"tokens":toks,"tokens_per_sec":toks/max(1e-9,total_dt/1000.0)}