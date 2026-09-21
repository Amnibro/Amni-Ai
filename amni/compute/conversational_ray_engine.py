import os,sys,time,math
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.lattice_ray_engine import LatticeGuidedRayEngine
from amni.compute.three_stage_harness import ThreeStageRayHarness
from amni.compute.ptex_1t_store import Ptex1TResidentStore
from amni.compute.ptex_300b_store import Ptex300BResidentStore
class ConversationalRayEngine:
 def __init__(self,engine:LatticeGuidedRayEngine=None,store=None):
  if engine is not None:self.engine=engine
  else:
   from amni.compute.ptex_1t_store import default_1t_path
   p1t=default_1t_path()
   p300b=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
   s=store if store is not None else (Ptex1TResidentStore(p1t) if os.path.exists(p1t) else Ptex300BResidentStore(p300b))
   self.engine=LatticeGuidedRayEngine(store=s)
  self.store=self.engine.store
  self.three_stage=ThreeStageRayHarness(engine=self.engine,store=self.store)
  self.history=[]
  self.v_conv=np.array([1.0,0.0,0.0],dtype=np.float32)
 def chat(self,user_msg:str)->dict:
  t0=time.perf_counter()
  stage_res=self.three_stage.execute(user_msg,history=self.history)
  pos,vel=self.engine.embed_query_ray(user_msg)
  self.v_conv=0.8*self.v_conv+0.2*vel
  self.v_conv=self.v_conv/max(1e-6,float(np.linalg.norm(self.v_conv)))
  dt=(time.perf_counter()-t0)*1000.0
  toks=len(stage_res["final_text"])/4.0
  turn={
   "user":user_msg,
   "reply":stage_res["final_text"],
   "domain":stage_res["target_domain"],
   "intent":stage_res["implicit_goal"].lower(),
   "speech_act":stage_res["speech_act"],
   "latency_ms":round(dt,3),
   "tokens":toks,
   "congruent":stage_res["congruent"],
   "rearms":stage_res["rearms_triggered"],
   "cot_action":stage_res.get("cot_action"),
   "empirical_verified":stage_res.get("empirical_verified",False),
   "intent_questioning":stage_res.get("intent_questioning"),
   "critic_score":stage_res.get("critic_score"),
   "critic_verdict":stage_res.get("critic_verdict"),
   "critic_notes":stage_res.get("critic_notes"),
   "refinement_applied":stage_res.get("refinement_applied",False),
   "candidate_draft":stage_res.get("candidate_draft")
  }
  self.history.append(turn)
  return turn
 def reset_conversation(self):
  self.history=[]
  self.v_conv=np.array([1.0,0.0,0.0],dtype=np.float32)
  self.three_stage.history=[]