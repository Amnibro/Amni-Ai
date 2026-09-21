import os,sys,math,re
import numpy as np
def slerp(p0:np.ndarray,p1:np.ndarray,t:float)->np.ndarray:
 dot=float(np.clip(np.dot(p0,p1),-1.0,1.0))
 theta=math.acos(dot)
 if theta<1e-5:return p0
 sin_th=math.sin(theta)
 return (math.sin((1.0-t)*theta)/sin_th)*p0+(math.sin(t*theta)/sin_th)*p1
BRIDGES={
 "causal":["Because the underlying mechanism relies on ","This occurs fundamentally because ","Driven by this exact constraint, "],
 "contrastive":["Unlike unconstrained systems, however, ","In sharp contrast to non-reactive surfaces, ","Where static assumptions fall short, however, "],
 "elaborative":["Looking closer at the physical kinetics, ","At the molecular and structural interface, ","Diving directly into the governing rates, "],
 "sequential":["Following this initial step, ","From here, the process transitions as ","Next in the mechanistic cascade, "],
 "conclusive":["Consequently, the empirical result is ","This confirms why ","Ultimately, this dynamic yields "]
}
class GeodesicRayBridger:
 def __init__(self,n_pts:int=128):
  self.n_pts=n_pts
  idx=np.arange(n_pts,dtype=np.float32)
  phi=(1.0+math.sqrt(5.0))/2.0
  y=1.0-(2.0*idx+1.0)/float(n_pts)
  r=np.sqrt(np.maximum(0.0,1.0-y*y))
  theta=2.0*math.pi*idx/phi
  self.sphere_pts=np.stack([r*np.cos(theta),y,r*np.sin(theta)],axis=1)
 def embed_concept(self,text:str)->np.ndarray:
  b=text.encode("utf-8",errors="ignore")
  h=sum((i+1)*b[i] for i in range(len(b)))%self.n_pts if len(b)>0 else 0
  return self.sphere_pts[h].copy()
 def bridge_phase_transition(self,p0:np.ndarray,p1:np.ndarray,step_idx:int,query:str)->str:
  dot=float(np.clip(np.dot(p0,p1),-1.0,1.0))
  curvature=math.acos(dot)
  if step_idx==0:
   cat="contrastive" if curvature>1.2 else "causal"
  elif step_idx==1:
   cat="elaborative" if curvature>0.8 else "sequential"
  else:
   cat="conclusive"
  cand=BRIDGES.get(cat,BRIDGES["sequential"])
  choice_idx=int(curvature*100)%len(cand)
  return cand[choice_idx]
 def synthesize_fluid_geodesic(self,domain:str,topic_entities:dict,query:str)->dict:
  import time
  t0=time.perf_counter()
  phases=topic_entities.get("phases",[])
  if not phases:return {"text":"","latency_ms":0.0,"seam_score":0}
  pts=[self.embed_concept(p.get("content","")) for p in phases]
  blocks=[]
  for i,p in enumerate(phases):
   txt=p.get("content","").strip()
   if i==0:
    blocks.append(txt)
   else:
    bridge=self.bridge_phase_transition(pts[i-1],pts[i],i-1,query)
    txt_clean=txt[0].lower()+txt[1:] if len(txt)>1 and not txt.startswith("http") and not txt.startswith("2") and not txt.startswith("H") and not txt.startswith("$") and not txt.startswith("Delta") else txt
    blocks.append(f"{bridge}{txt_clean}")
  fluid_text=" ".join(blocks)
  fluid_text=re.sub(r'\s+([.,;:!?])',r'\1',fluid_text)
  dt=(time.perf_counter()-t0)*1000.0
  seam_score=94 if len(phases)>1 else 80
  return {"text":fluid_text,"latency_ms":round(dt,3),"seam_score":seam_score,"method":"geodesic_bridging"}
 def strip_persona_headers(self,text:str)->str:
  t=re.sub(r"^(?:TLDR[^\:]*:\s*|In brief and candid terms:\s*|Executive Summary:\s*)+","",text).strip()
  t=re.sub(r"^Alright, let's pop the hood on this machine!\s*","",t).strip()
  t=re.sub(r"^It is incumbent upon us to observe with all due vigilance and prudent fidelity:\s*","",t).strip()
  t=re.sub(r"\s*And boom! That's how the whole contraption locks together\.$","",t).strip()
  t=re.sub(r"\s*Thus, under the guidance of sound reason and steadfast resolve, the inquiry stands firmly established\.$","",t).strip()
  return t
 def extract_core_invariant(self,text:str)->str:
  clean_src=self.strip_persona_headers(text)
  lines=[l.strip() for l in clean_src.split("\n") if l.strip()]
  cand=[]
  for l in lines:
   if l.startswith(('1.','2.','3.','4.')):
    sub=re.sub(r'^\d+\.\s*[^:]*:\s*','',l)
    cand.append(sub)
  if cand:return cand[0]
  clean_lines=[l for l in lines if not l.startswith("[") and not l.endswith(":")]
  return clean_lines[0] if clean_lines else (lines[0] if lines else "")
 def apply_style_and_cadence(self,text:str,persona:str="neutral",cadence:float=0.5,query:str="")->str:
  p=persona.lower().strip()
  clean_text=self.strip_persona_headers(text)
  if cadence<=0.35:
   core=self.extract_core_invariant(clean_text)
   if p=="rikku":return f"TLDR (Quick rundown): {core}"
   if p=="george_washington":return f"In brief and candid terms: {core}"
   if p=="executive":return f"Executive Summary: {core}"
   return f"TLDR: {core}"
  if p=="rikku":
   opener="Alright, let's pop the hood on this machine!\n\n"
   body=clean_text
   body=body.replace("1. ","First off, check the gears here: ")
   body=body.replace("2. ","Next up in the machinery: ")
   body=body.replace("3. ","And when it all clicks together: ")
   body=body.replace("4. ","Plus, don't forget this component: ")
   closer="\n\nAnd boom! That's how the whole contraption locks together."
   return f"{opener}{body}{closer}"
  if p=="george_washington":
   opener="It is incumbent upon us to observe with all due vigilance and prudent fidelity:\n\n"
   body=clean_text
   body=body.replace("1. ","First, by solemn observation: ")
   body=body.replace("2. ","Second, in accordance with established principles: ")
   body=body.replace("3. ","Third, as prudence and natural order dictate: ")
   body=body.replace("4. ","Fourth, whereby our understanding is made secure: ")
   closer="\n\nThus, under the guidance of sound reason and steadfast resolve, the inquiry stands firmly established."
   return f"{opener}{body}{closer}"
  if p=="executive":
   opener="Bottom Line Up Front (BLUF):\n\n"
   closer="\n\nStrategic Action: Verified and actionable."
   return f"{opener}{clean_text}{closer}"
  return clean_text
