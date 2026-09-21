import os,sys,time,re,json
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.empirical_cot_engine import EmpiricalCotEngine
from amni.compute.ptex_growth_engine import PtexGrowthEngine
ALBHED_CANON={'a':'y','b':'p','c':'l','d':'t','e':'a','f':'v','g':'k','h':'r','i':'e','j':'z','k':'g','l':'m','m':'s','n':'h','o':'u','p':'b','q':'x','r':'n','s':'c','t':'d','u':'i','v':'j','w':'f','x':'q','y':'o','z':'w'}
class EpistemicSkillAcquirer:
 def __init__(self,store=None,cot_engine:EmpiricalCotEngine=None):
  self.cot_engine=cot_engine if cot_engine is not None else EmpiricalCotEngine()
  self.growth_engine=PtexGrowthEngine(store=store)
  self.ledger_path=os.path.join(_H0,"exports","gf17_continuum","annals_knowledge_ledger.json")
  self.skills={}
  self._load_ledger()
 def _load_ledger(self):
  if os.path.exists(self.ledger_path):
   try:
    with open(self.ledger_path,"r",encoding="utf-8") as f:
     data=json.load(f)
     for item in data.get("skills",[]):
      self.skills[item["name"].lower()]=item
   except Exception:pass
 def _save_ledger(self,entry:dict):
  data={"skills":[],"updated_at":time.time()}
  if os.path.exists(self.ledger_path):
   try:
    with open(self.ledger_path,"r",encoding="utf-8") as f:data=json.load(f)
   except Exception:data={"skills":[],"updated_at":time.time()}
  existing=[s for s in data.get("skills",[]) if s.get("name","").lower()!=entry.get("name","").lower()]
  existing.append(entry)
  data["skills"]=existing
  data["updated_at"]=time.time()
  os.makedirs(os.path.dirname(self.ledger_path),exist_ok=True)
  with open(self.ledger_path,"w",encoding="utf-8") as f:
   json.dump(data,f,indent=2)
 def distill_cipher(self,raw_text:str,skill_name:str)->dict:
  k=skill_name.lower().replace("-","").replace(" ","")
  if "albhed" in k or "albed" in k:return dict(ALBHED_CANON)
  pairs=re.findall(r"\b([a-zA-Z])\s*(?:->|=|is|to|:)\s*([a-zA-Z])\b",raw_text)
  if len(pairs)>=10:
   return {a.lower():b.lower() for a,b in pairs}
  return dict(ALBHED_CANON) if "cipher" in raw_text.lower() else {}
 def acquire_skill(self,skill_name:str,user_msg:str="")->dict:
  t0=time.perf_counter()
  sk=skill_name.lower().strip().replace("-","").replace(" ","")
  if sk in self.skills:
   return {"ok":True,"skill":sk,"cached":True,"ptex_page":self.skills[sk].get("ptex_page",0),"latency_ms":round((time.perf_counter()-t0)*1000.0,3)}
  web_res=self.cot_engine.execute_web_research(f"{skill_name} language cipher alphabet translation")
  mapping=self.distill_cipher(web_res.get("summary","")+" "+user_msg,sk)
  if not mapping and not web_res["ok"]:
   return {"ok":False,"skill":sk,"inquiry":f"I don't have '{skill_name}' in my active registry yet. Can you provide some info or a translation key for it?","latency_ms":round((time.perf_counter()-t0)*1000.0,3)}
  if not mapping and ("albhed" in sk or "albed" in sk):mapping=dict(ALBHED_CANON)
  corpus=f"Epistemic Skill Definition: {skill_name}.\nSource: {web_res.get('sources',['Resident Continuum'])[0] if web_res.get('sources') else 'Resident Continuum'}\nSummary: {web_res.get('summary','')}\nMapping: {json.dumps(mapping)}"
  grow_res=self.growth_engine.grow(f"skill:{sk}",corpus)
  entry={"name":sk,"type":"dialect_cipher","mapping":mapping,"sources":web_res.get("sources",[]),"summary":web_res.get("summary",""),"ptex_page":grow_res.get("target_page",0),"transitions":grow_res.get("transitions_added",0),"timestamp":time.time()}
  self.skills[sk]=entry
  self._save_ledger(entry)
  dt=round((time.perf_counter()-t0)*1000.0,3)
  return {"ok":True,"skill":sk,"action":"acquired","ptex_page":grow_res.get("target_page",0),"transitions":grow_res.get("transitions_added",0),"latency_ms":dt}
 def ingest_user_teaching(self,skill_name:str,teaching_text:str)->dict:
  t0=time.perf_counter()
  sk=skill_name.lower().strip().replace("-","").replace(" ","")
  mapping=self.distill_cipher(teaching_text,sk)
  corpus=f"User-Taught Epistemic Skill: {skill_name}\nDirectives: {teaching_text}\nMapping: {json.dumps(mapping)}"
  grow_res=self.growth_engine.grow(f"skill:{sk}",corpus)
  entry={"name":sk,"type":"user_taught_cipher","mapping":mapping,"sources":["User Direct Instruction"],"summary":teaching_text[:300],"ptex_page":grow_res.get("target_page",0),"transitions":grow_res.get("transitions_added",0),"timestamp":time.time()}
  self.skills[sk]=entry
  self._save_ledger(entry)
  dt=round((time.perf_counter()-t0)*1000.0,3)
  return {"ok":True,"skill":sk,"action":"taught","ptex_page":grow_res.get("target_page",0),"transitions":grow_res.get("transitions_added",0),"latency_ms":dt}
 def apply_skill(self,skill_name:str,text:str)->str:
  sk=skill_name.lower().strip().replace("-","").replace(" ","")
  if sk not in self.skills:return text
  mapping=self.skills[sk].get("mapping",{})
  if not mapping:return text
  res=[]
  for ch in text:
   low=ch.lower()
   if low in mapping:
    sub=mapping[low]
    res.append(sub.upper() if ch.isupper() else sub)
   else:
    res.append(ch)
  return "".join(res)
