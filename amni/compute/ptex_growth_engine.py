import os,sys,time
from collections import defaultdict,Counter
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.ptex_1t_store import Ptex1TResidentStore
from amni.compute.ptex_300b_store import Ptex300BResidentStore
class PtexGrowthEngine:
 def __init__(self,store=None):
  if store is not None:self.store=store
  else:
   from amni.compute.ptex_1t_store import default_1t_path
   p1t=default_1t_path()
   p300b=os.path.join(_H0,"exports","gf17_continuum","adam_300b_store.ptex")
   self.store=Ptex1TResidentStore(p1t) if os.path.exists(p1t) else Ptex300BResidentStore(p300b)
  self.total_transitions_added=0
  self.pages_updated=set()
  self.growth_events=[]
 def extract_transitions(self,text:str)->list:
  b=text.encode("utf-8",errors="ignore")
  if len(b)<4:return []
  transitions=[]
  for a,b0,c0,d0 in zip(b,b[1:],b[2:],b[3:]):
   transitions.append((a,b0,c0,d0))
  return transitions
 def grow(self,domain:str,text:str)->dict:
  t0=time.perf_counter()
  trans=self.extract_transitions(text)
  if not trans:return {"domain":domain,"transitions_added":0,"latency_ms":0.0,"pages_updated":[]}
  r=self.store.get_domain_range(domain)
  target_page=r[0]+(self.total_transitions_added//1024)%(max(1,r[1]-r[0]))
  buf=bytearray()
  for a,b0,c0,d0 in trans[:1024]:buf.extend([a,b0,c0,d0])
  if len(buf)<4096:
   rem=4096-len(buf)
   seed=bytes(buf[:min(len(buf),rem)]) if len(buf)>0 else b"\x20\x20\x20\x20"
   while len(buf)<4096:buf.extend(seed[:min(len(seed),4096-len(buf))])
  self.store.write_page_data(target_page,bytes(buf[:4096]))
  self.pages_updated.add(target_page)
  self.total_transitions_added+=len(trans)
  dt=(time.perf_counter()-t0)*1000.0
  event={"domain":domain,"target_page":target_page,"transitions_added":len(trans),"latency_ms":round(dt,3)}
  self.growth_events.append(event)
  return event
 def get_growth_telemetry(self)->dict:
  return {"total_transitions_added":self.total_transitions_added,"unique_pages_updated":len(self.pages_updated),"total_events":len(self.growth_events),"history":self.growth_events[-5:]}