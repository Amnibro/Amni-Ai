import os,sys,re
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
class EgressSanitySentry:
 def __init__(self):
  self.total_audits=0
  self.total_rearms=0
 def audit(self,probe_res:dict,generated_text:str)->dict:
  self.total_audits+=1
  goal=probe_res["implicit_goal"]
  dom=probe_res["target_domain"]
  flaws=[]
  rearm_needed=False
  suggested_domain=dom
  if goal=="IDENTITY":
   if "Adam" not in generated_text:
    flaws.append("Output fails to establish Adam identity")
    rearm_needed=True
  elif goal=="CODE_SYNTHESIS":
   if dom in ("rust","cpp","fortran","code"):
    has_code=any(k in generated_text for k in ("fn ","def ","template","subroutine","struct ","import "))
    if not has_code:
     flaws.append(f"Expected {dom} code implementation but received non-code text")
     rearm_needed=True
    if "Theorem (Finite Field" in generated_text and dom!="math":
     flaws.append("Galois math theorem leaked into non-math code output")
     rearm_needed=True
    if dom in ("rust","cpp"):
     ob,cb=generated_text.count("{"),generated_text.count("}")
     if ob!=cb:flaws.append(f"Brace imbalance: {ob} open vs {cb} close")
  elif goal=="PROOF":
   if "Theorem" not in generated_text and "Q.E.D." not in generated_text:
    flaws.append("Expected formal proof delimiters")
  elif goal=="GREETING":
   if len(generated_text)>300:
    flaws.append("Greeting output is excessively verbose")
  if rearm_needed:self.total_rearms+=1
  return {
   "congruent":not rearm_needed,
   "rearm_needed":rearm_needed,
   "flaws":flaws,
   "suggested_domain":suggested_domain,
   "total_rearms_lifetime":self.total_rearms
  }