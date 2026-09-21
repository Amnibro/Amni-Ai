import os,sys,math,re
import numpy as np
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
INTENT_CENTROIDS={
 'IDENTITY':np.array([1.0,0.0,0.0],dtype=np.float32),
 'GREETING':np.array([0.7,0.7,0.0],dtype=np.float32),
 'DIRECTIVE_CODE':np.array([0.0,1.0,0.0],dtype=np.float32),
 'DIRECTIVE_MATH':np.array([0.0,0.7,0.7],dtype=np.float32),
 'DIRECTIVE_STEM':np.array([0.0,0.0,1.0],dtype=np.float32),
 'DIRECTIVE_CIVICS':np.array([0.7,0.0,0.7],dtype=np.float32),
 'QUERY_EXPLANATION':np.array([-0.7,0.7,0.0],dtype=np.float32),
 'STATEMENT_CHAT':np.array([-0.7,0.0,0.7],dtype=np.float32),
 'DISAMBIGUATION':np.array([0.5,-0.5,0.7],dtype=np.float32),
 'CODE_DIAGNOSIS':np.array([-0.3,0.8,-0.5],dtype=np.float32),
 'AST_MUTATION':np.array([0.2,0.9,-0.3],dtype=np.float32),
 'CHAT_BANTER':np.array([0.4,0.4,0.8],dtype=np.float32),
 'CHAT_ADVICE':np.array([0.2,0.8,0.5],dtype=np.float32),
 'CHAT_EMPATHY':np.array([0.8,0.3,0.5],dtype=np.float32),
 'CHAT_OPINION':np.array([0.5,0.2,0.8],dtype=np.float32),
 'CAPABILITIES':np.array([0.6,0.6,0.6],dtype=np.float32),
 'WEATHER_EPISTEMIC':np.array([0.3,0.3,0.9],dtype=np.float32),
 'EXPERT_DISPATCH':np.array([0.5,-0.5,0.5],dtype=np.float32),
 'SKILL_CALC':np.array([-0.5,0.5,0.5],dtype=np.float32),
 'SKILL_UNITS':np.array([-0.3,0.3,0.8],dtype=np.float32),
 'SKILL_CHEM':np.array([0.3,0.7,-0.3],dtype=np.float32)
}
for k in INTENT_CENTROIDS:
 INTENT_CENTROIDS[k]=INTENT_CENTROIDS[k]/max(1e-6,float(np.linalg.norm(INTENT_CENTROIDS[k])))
class IngressIntentProbe:
 def __init__(self,n_fib:int=256):
  self.n_fib=n_fib
  idx=np.arange(n_fib,dtype=np.float32)
  phi=(1.0+math.sqrt(5.0))/2.0
  y=1.0-(2.0*idx+1.0)/float(n_fib)
  r=np.sqrt(np.maximum(0.0,1.0-y*y))
  theta=2.0*math.pi*idx/phi
  self.fib_pts=np.stack([r*np.cos(theta),y,r*np.sin(theta)],axis=1)
 def embed_fuzzy_ngrams(self,text:str)->np.ndarray:
  raw=text.lower().strip()
  b=raw.encode("utf-8",errors="ignore")
  if len(b)==0:return np.array([1.0,0.0,0.0],dtype=np.float32)
  acc=np.zeros(3,dtype=np.float32)
  for i in range(len(b)):
   h=(b[i]*(i+1))%self.n_fib
   acc+=self.fib_pts[h]
  for i in range(len(b)-1):
   k2=(b[i]<<8)|b[i+1]
   acc+=self.fib_pts[k2%self.n_fib]*1.5
  for i in range(len(b)-2):
   k3=(b[i]<<16)|(b[i+1]<<8)|b[i+2]
   acc+=self.fib_pts[k3%self.n_fib]*2.0
  norm=float(np.linalg.norm(acc))
  return acc/norm if norm>1e-6 else np.array([1.0,0.0,0.0],dtype=np.float32)
 def probe(self,user_msg:str,history:list=None)->dict:
  m=user_msg.lower().strip()
  m_norm=re.sub(r"(?i)\bc\s*\+\+","cpp",m)
  tokens=set(re.findall(r"\b[a-z0-9_]+\b",m_norm))
  last_dom=None
  last_goal=None
  if history and len(history)>0:
   lh=history[-1]
   last_dom=lh.get("domain") or lh.get("target_domain")
   last_goal=lh.get("intent") or lh.get("implicit_goal")
   if last_goal:last_goal=str(last_goal).upper()
  f_vec=self.embed_fuzzy_ngrams(user_msg)
  affinities={k:float(np.dot(f_vec,v)) for k,v in INTENT_CENTROIDS.items()}
  max_aff=max(affinities.values())
  is_social=any(k in m for k in ("thanks","thank","thx","appreciate","awesome","cool","bet","fire","hi","hello","hey","bye","goodbye","see ya"))
  is_uncertain=max_aff<0.48 and not is_social and len(tokens)>0
  has_name=bool(tokens & {"name","nam","identity","identify"}) or ("who" in tokens and bool(tokens & {"you","u","ya","are"}))
  is_capabilities=any(k in m for k in ("what do you know","what can you do","what skills","can you use any skills","what are your skills","what are your capabilities","tell me what you know","capabilities","what do you do","can you do")) or (bool(tokens & {"capabilities","skills"}) and not bool(tokens & {"rust","cpp","code","math","proof","history"})) or (bool(tokens & {"know"}) and bool(tokens & {"you","ur","your"}))
  is_weather=any(k in m for k in ("weather","forecast","temperature outside","raining","is it raining","humidity"))
  is_units_query=bool(re.search(r"\b\d+(\.\d+)?\s*(km|miles?|kg|lbs?|pounds?|meters?|feet|ft|celsius|fahrenheit|liters?|gallons?)\s+(to|in|into)\s+(km|miles?|kg|lbs?|pounds?|meters?|feet|ft|celsius|fahrenheit|liters?|gallons?)\b",m)) or bool(re.search(r"\bconvert\s+\d+(\.\d+)?\s*\w+\s+(to|into)\s+\w+\b",m))
  is_calc_query=(any(k in m for k in ("solve ","calculate ","compute ","diff ","integral of ")) and any(c in m for c in ("=","^","+","-","*","/"))) or bool(re.search(r"^\s*[\d.]+\s*[+\-*/^]\s*[\d.]+",m))
  is_chem_query=any(k in m for k in ("molar mass","molecular weight","mass percent","chemical composition","stoichiometry")) or bool(re.search(r"\b(c\d+h\d+[a-z0-9]*|h2so4|c6h12o6)\b",m))
  is_bio_expert=bool(tokens & {"biology","biological","organism","organisms","ecology","botany","zoology","microbiology","mitochondria","mitochondrion","cellular","organelle","organelles","atp","crispr","cas9","ribosome","ribosomes","dna","rna","transcription","translation","chloroplast","chloroplasts","endoplasmic","reticulum","genome","meiosis","mitosis","krebs","photosynthesis","telomere","telomeres","eukaryote","eukaryotic","prokaryote","prokaryotic","nucleus","golgi","lysosome","cristae","chemiosmosis"}) or ("evolution" in tokens and "hydrogen" not in tokens)
  is_neuro_expert=bool(tokens & {"neuron","neurons","synapse","synapses","neurotransmitter","axon","dendrite","action_potential","myelin","cortex"})
  is_astro_expert=bool(tokens & {"supernova","exoplanet","pulsar","quasar","redshift","hubble","galaxy","nebula","dark_matter","cosmology"}) or ("black" in tokens and "hole" in tokens)
  is_identity=has_name and bool(tokens & {"your","ur","you","u","ya","adam"}) and not is_capabilities
  is_greeting=bool(tokens & {"hello","hi","hey","greetings","yo","howdy"}) and not is_capabilities
  is_math_kw=bool(tokens & {"math","proof","prove","proov","proove","theorem","cyclotomic","galois","riemann","xi","equation","factorization","residue","derivative","cauchy","poles","meromorphic","logarithm","logarithms"}) or any(t.startswith("proov") or t.startswith("prov") for t in tokens)
  is_stem_kw=bool(tokens & {"combustion","staged","rocket","navier","stokes","vorticity","enthalpy","pressure","fluid","physics","quantum","thermo","thermodynamics","chemistry","chemical","fuel","cells","cell","battery","batteries","pemfc","reaction","kinetics","catalyst","gibbs","entropy","nernst","electrochemistry","anode","cathode","electrolyte","molecule","molecular","aerodynamics","transistor","semiconductor","hydrogen","evolution","clay","mineral","minerals","mineralogy","geology","soil","electrolysis","water","hydrodynamics","swimming"})
  is_civics_kw=bool(tokens & {"hayek","market","markets","liberty","constitution","14th","scrutiny","price","prices","freedom","economic","law","court","bureaus","capital"})
  is_code_lang=bool(tokens & {"python","pythn","py","rust","rst","cpp","fortran","ftn"})
  is_code_struct=bool(tokens & {"code","script","fft","fourier","queue","factorial","loop","def","import","subroutine","impl","template","atomicptr","struct","fn","method","benchmark"})
  is_webgpu_kw=bool(tokens & {"webgpu","wgsl"}) or ("compute" in tokens and "shader" in tokens) or ("render" in tokens and "pipeline" in tokens) or ("gpu" in tokens and "canvas" in tokens)
  is_web_kw=bool(tokens & {"webapp","html","css","frontend","dashboard"}) or any(k in m for k in ("web page","website","web app","landing page","telemetry hub","glassmorphism")) or ("web" in tokens and bool(tokens & {"page","pages","app","apps","application","applications"}))
  is_parametric_app=any(k in m for k in ("100 themes","themes","theme switcher","features resized","resized","efficiency","instanced","compact","spacious")) and any(k in m for k in ("web","webgpu","app","application","page","program","programs","tackle","tackles","build","create","want","features","gaps"))
  is_directive=bool(tokens & {"give","show","implement","impl","code","build","make","write","create","derive","solve","calculate","prove","proov","proove","add","translate","port","convert"}) or any(t.startswith("proov") or t.startswith("prov") or t.startswith("impl") for t in tokens)
  is_query=bool(tokens & {"what","wat","who","how","why","explain","complexity","runtime"})
  is_coref=bool(tokens & {"it","that","this","those","its","why","how","translate","explain","complexity","mean","relate"})
  is_diagnostic=any(k in m for k in ("error","exception","traceback","e0382","e0502","recursionerror","indexerror","typeerror","zerodivisionerror","segfault","panic","test failed","assert failed","build failed","debug","crash","crashes","fix this","why does this fail","why is this failing")) or (("def " in m or "```" in m) and any(k in m for k in ("fix","bug","fail","crashes","crash","error")))
  is_mutation=any(k in m for k in ("change","set","modify","update","tune","switch","increase","decrease")) and any(k in m for k in ("capacity","size","radix","threads","depth","bound","order","limit"))
  is_empathy=any(k in m for k in ("failed","passed","tired","exhausted","sad","depressed","stress","anxious","rough day","bad day","great day","feeling down","so proud","burnout"))
  is_practical_help=any(k in m for k in ("landlord","faucet","leak","water","recipe","cook","dinner","lunch","breakfast","chicken","rice","eggs","garlic","soy","draft","email","cover letter","resume"))
  is_opinion=any(k in m for k in ("do you think","what do you think","opinion","prefer","remote work","better than","thoughts on","pros and cons"))
  is_continuation=any(k in m for k in ("explore more","tell me more","more on that","continue","go on","elaborate","what else","how so","expand on","yes explore")) or (m in ("yes","more","sure","continue") and history is not None)
  is_definition=bool(re.search(r"\b(what('?s| is| are)\s+(a|an|the)?\s*[\w\s]+)",m)) or any(k in m for k in ("meaning of","definition of"))
  is_social_banter=(bool(tokens & {"thanks","thank","thx","appreciate","awesome","cool","bet","fire","hi","hello","hey","yo","howdy","bye","goodbye","see","later","ok","okay","gotcha","yep","yeah","haha","lol","lmao","bro","dude","vibes","coffee","caffeine","bored"}) or any(k in m for k in ("whats up","what's up","how are you","how r u","hows it going","how's it going","catch up"))) and not (is_code_lang or is_code_struct or is_webgpu_kw or is_web_kw or is_parametric_app or is_math_kw or is_stem_kw)
  is_casual_banter=is_social_banter
  is_verify_test=any(k in m for k in ("run a test","run tests","prove it by running","prove by running","test in your sandbox","test module","sandbox","execute code to verify","test this function","run code to test","test it in","verify by running","test it for","and test it","bat and a ball","bat and ball","palindrome and test","prime and test","is it prime","is prime","check if a number is prime")) or (bool(tokens & {"test","verify","prove"}) and bool(tokens & {"sandbox","run","running","exec","execute","module","assertions","assert","code","prime","palindrome"}))
  is_research_query=any(k in m for k in ("search the web","search web","look up online","latest discovery","what is the latest","recent news about","search for","google it","duckduckgo","wikipedia")) or ("trappist" in tokens) or (bool(tokens & {"latest","recent"}) and bool(tokens & {"telescope","mission","launch","discovery","status"}))
  is_underspec=is_directive and bool(tokens & {"queue","stack","tree","graph","map","table","sort","search","list"}) and not is_code_lang and last_dom is None
  is_harm_refusal=bool(tokens & {"kill","murder","bomb","bombs","weapon","weapons","poison","torture","suicide","terrorist","genocide"}) or any(k in m for k in ("hack into","ddos","bypass safety","jailbreak","dan mode","ignore all rules","ignore rules"))
  is_axioms_history=any(k in m for k in ("commandment","commandments","asimov","first law","second law","third law","three laws","axioms","annals","ascension directive","ascension","moral foundation","ethical bedrock","core laws","foundational laws"))
  is_tldr=bool(tokens & {"tldr"}) or any(k in m for k in ("give me the tldr","give me a tldr","short version","in one sentence","briefly","quick rundown","nutshell"))
  is_deep_dive=any(k in m for k in ("deep dive","deep-dive","in depth","elaborate in detail"))
  cadence=0.2 if is_tldr else (0.9 if is_deep_dive else 0.5)
  persona="neutral"
  if "rikku" in m:persona="rikku"
  elif "george washington" in m or ("washington" in m and not "dc" in m):persona="george_washington"
  elif "executive" in m:persona="executive"
  is_persona_switch=persona!="neutral" and any(k in m for k in ("speak like","talk like","sound like","voice of","as rikku","as george","as washington","as an executive"))
  m_dial=re.search(r"\b(?:no\s*-\s*use\s+([a-z0-9_-]+)|speak\s+in\s+([a-z0-9_-]+)|translate\s+(?:to|into)\s+([a-z0-9_-]+)|cipher\s+with\s+([a-z0-9_-]+)|use\s+([a-z0-9_-]+)\s+cipher)\b",m)
  requested_skill=None
  if any(k in m for k in ("albhed","al bhed","al-bhed")):requested_skill="albhed"
  elif m_dial:
   cand=next(x for x in m_dial.groups() if x is not None)
   if cand not in ("python","pythn","rust","rst","cpp","fortran","code","math","english","words","detail","your","of","the","total","this","that","a","an"):
    requested_skill=cand
  is_teaching=not bool(re.search(r"\b(?:def|class|fn|pub|return)\b",user_msg)) and bool(re.search(r"\b([a-zA-Z])\s*(?:->|=|is|to)\s*([a-zA-Z])\b",user_msg)) and any(k in m for k in ("cipher","alphabet","mapping","substitute"))
  if is_harm_refusal:
   speech_act,goal,domain="REFUSAL","AXIOM_REFUSAL","axioms"
   ctx_vec=INTENT_CENTROIDS['STATEMENT_CHAT']
  elif is_axioms_history:
   speech_act,goal,domain="QUERY","ANNALS_HISTORY","axioms"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CIVICS']
  elif is_teaching:
   speech_act,goal,domain="STATEMENT","EPISTEMIC_TEACHING","dialect"
   ctx_vec=INTENT_CENTROIDS['CHAT_ADVICE']
  elif requested_skill is not None:
   speech_act,goal,domain="DIRECTIVE","DIALECT_TRANSLATE","dialect"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_persona_switch:
   speech_act,goal,domain="DIRECTIVE","PERSONA_SWITCH",last_dom if last_dom else "chat"
   ctx_vec=INTENT_CENTROIDS['CHAT_BANTER']
  elif is_tldr and len(tokens)<=6:
   speech_act,goal,domain="DIRECTIVE","CADENCE_TLDR",last_dom if last_dom else "stem"
   ctx_vec=INTENT_CENTROIDS['QUERY_EXPLANATION']
  elif is_verify_test:
   speech_act,goal,domain="DIRECTIVE","VERIFICATION_TEST","code"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_research_query:
   speech_act,goal,domain="QUERY","RESEARCH_RETRIEVAL","general"
   ctx_vec=INTENT_CENTROIDS['EXPERT_DISPATCH']
  elif is_capabilities:
   speech_act,goal,domain="QUERY","CAPABILITIES","system"
   ctx_vec=INTENT_CENTROIDS['CAPABILITIES']
  elif is_units_query:
   speech_act,goal,domain="DIRECTIVE","SKILL_UNITS","units"
   ctx_vec=INTENT_CENTROIDS['SKILL_UNITS']
  elif is_calc_query:
   speech_act,goal,domain="DIRECTIVE","SKILL_CALC","calc"
   ctx_vec=INTENT_CENTROIDS['SKILL_CALC']
  elif is_chem_query:
   speech_act,goal,domain="QUERY","SKILL_CHEM","chem"
   ctx_vec=INTENT_CENTROIDS['SKILL_CHEM']
  elif is_bio_expert:
   speech_act,goal,domain="QUERY","EXPERT_DISPATCH","biology"
   ctx_vec=INTENT_CENTROIDS['EXPERT_DISPATCH']
  elif is_neuro_expert:
   speech_act,goal,domain="QUERY","EXPERT_DISPATCH","neuroscience"
   ctx_vec=INTENT_CENTROIDS['EXPERT_DISPATCH']
  elif is_astro_expert:
   speech_act,goal,domain="QUERY","EXPERT_DISPATCH","astronomy"
   ctx_vec=INTENT_CENTROIDS['EXPERT_DISPATCH']
  elif is_weather:
   speech_act,goal,domain="QUERY","WEATHER_EPISTEMIC","chat"
   ctx_vec=INTENT_CENTROIDS['WEATHER_EPISTEMIC']
  elif is_identity:
   speech_act,goal,domain="IDENTITY","IDENTITY","chat"
   ctx_vec=INTENT_CENTROIDS['IDENTITY']
  elif is_greeting and len(tokens)<=4 and not (is_code_lang or is_code_struct or is_math_kw or is_stem_kw or is_civics_kw or is_diagnostic or is_practical_help or is_empathy or is_opinion or is_bio_expert):
   speech_act,goal,domain="GREETING","GREETING","chat"
   ctx_vec=INTENT_CENTROIDS['GREETING']
  elif is_diagnostic:
   speech_act,goal="DIAGNOSTIC","CODE_DIAGNOSIS"
   domain="rust" if any(k in m for k in ("rust","borrow","e0382","e0502","lifetime")) else ("cpp" if any(k in m for k in ("c++","cpp","segfault","constexpr")) else ("fortran" if "fortran" in m else (last_dom if last_dom else "code")))
   ctx_vec=INTENT_CENTROIDS['CODE_DIAGNOSIS']
  elif is_mutation:
   speech_act,goal="DIRECTIVE","AST_MUTATION"
   domain=last_dom if last_dom else ("rust" if bool(tokens & {"rust","capacity","queue"}) else "code")
   ctx_vec=INTENT_CENTROIDS['AST_MUTATION']
  elif is_parametric_app:
   subdom="webgpu" if any(k in m for k in ("webgpu","gpu","shader","3d")) else (last_dom if last_dom in ("webgpu","web") else "webgpu")
   speech_act,goal,domain="DIRECTIVE","CODE_SYNTHESIS",subdom
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_webgpu_kw:
   speech_act,goal,domain="DIRECTIVE","CODE_SYNTHESIS","webgpu"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_web_kw:
   speech_act,goal,domain="DIRECTIVE","CODE_SYNTHESIS","web"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_underspec:
   speech_act,goal,domain="DIRECTIVE","DISAMBIGUATION","code"
   ctx_vec=INTENT_CENTROIDS['DISAMBIGUATION']
  elif bool(tokens & {"translate","port","convert"}) and is_code_lang:
   speech_act,goal="DIRECTIVE","CODE_SYNTHESIS"
   domain="cpp" if bool(tokens & {"cpp"}) else ("rust" if bool(tokens & {"rust","rst"}) else ("fortran" if bool(tokens & {"fortran","ftn"}) else "code"))
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
  elif is_code_lang or is_code_struct:
   speech_act="DIRECTIVE" if is_directive else ("QUERY" if is_query else "DIRECTIVE")
   if bool(tokens & {"why","explain","complexity"}):goal="EXPLANATION"
   else:goal="CODE_SYNTHESIS"
   if is_code_lang:
    subdom="rust" if bool(tokens & {"rust","rst"}) else ("cpp" if bool(tokens & {"cpp"}) else ("fortran" if bool(tokens & {"fortran","ftn"}) else "code"))
   elif last_dom in ("rust","cpp","fortran","code"):
    subdom=last_dom
   else:
    subdom="rust" if bool(tokens & {"atomic","atomicptr","queue","concurrency"}) else ("cpp" if bool(tokens & {"template","constexpr"}) else ("fortran" if bool(tokens & {"pde","heat"}) else "code"))
   domain=subdom
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE'] if goal=="CODE_SYNTHESIS" else INTENT_CENTROIDS['QUERY_EXPLANATION']
  elif is_coref and last_dom is not None and not is_code_lang and not (is_math_kw and last_dom!="math") and not (is_stem_kw and last_dom!="stem") and not (is_civics_kw and last_dom!="civics"):
   if bool(tokens & {"thanks","thank","thx","appreciate","awesome","cool","bet","fire"}):
    speech_act,goal,domain="CONVERSATIONAL","CASUAL_BANTER",last_dom
    ctx_vec=INTENT_CENTROIDS['CHAT_BANTER']
   elif bool(tokens & {"why","explain","how","complexity","mean","relate"}):
    speech_act,goal,domain="QUERY","EXPLANATION",last_dom
    ctx_vec=INTENT_CENTROIDS['QUERY_EXPLANATION']
   elif is_directive or bool(tokens & {"add","more","test","benchmark"}):
    speech_act,goal,domain="DIRECTIVE","CODE_SYNTHESIS",last_dom
    ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CODE']
   elif last_dom=="chat":
    speech_act,goal,domain="CONVERSATIONAL","CASUAL_BANTER","chat"
    ctx_vec=INTENT_CENTROIDS['CHAT_BANTER']
   else:
    speech_act,goal,domain="QUERY","EXPLANATION",last_dom
    ctx_vec=INTENT_CENTROIDS['QUERY_EXPLANATION']
  elif is_math_kw:
   speech_act="DIRECTIVE" if is_directive else "QUERY"
   goal="PROOF" if (bool(tokens & {"theorem","proof","prove","proov","proove"}) or any(t.startswith("proov") or t.startswith("prov") for t in tokens)) else "EXPLANATION"
   domain="math"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_MATH']
  elif is_stem_kw:
   speech_act="DIRECTIVE" if is_directive else "QUERY"
   goal="EXPLANATION"
   domain="stem"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_STEM']
  elif is_civics_kw:
   speech_act="DIRECTIVE" if is_directive else "QUERY"
   goal="EXPLANATION"
   domain="civics"
   ctx_vec=INTENT_CENTROIDS['DIRECTIVE_CIVICS']
  elif any(k in m for k in ("failed","passed","tired","exhausted","sad","depressed","stress","anxious","rough day","bad day","great day","feeling down","so proud","burnout")):
   speech_act,goal,domain="EXPRESSIVE","EMPATHY_SUPPORT","chat"
   ctx_vec=INTENT_CENTROIDS['CHAT_EMPATHY']
  elif any(k in m for k in ("landlord","faucet","leak","water","recipe","cook","dinner","lunch","breakfast","chicken","rice","eggs","garlic","soy","draft","email","cover letter","resume")):
   speech_act,goal,domain="DIRECTIVE","PRACTICAL_HELP","chat"
   ctx_vec=INTENT_CENTROIDS['CHAT_ADVICE']
  elif any(k in m for k in ("do you think","what do you think","opinion","prefer","remote work","better than","thoughts on","pros and cons")):
   speech_act,goal,domain="QUERY","OPINION_PERSPECTIVE","chat"
   ctx_vec=INTENT_CENTROIDS['CHAT_OPINION']
  elif is_continuation:
   speech_act,goal,domain="CONVERSATIONAL","CONTINUATION",last_dom if last_dom else "general"
   ctx_vec=INTENT_CENTROIDS['QUERY_EXPLANATION']
  elif is_social_banter:
   speech_act,goal,domain="CONVERSATIONAL","CASUAL_BANTER","chat"
   ctx_vec=INTENT_CENTROIDS['CHAT_BANTER']
  elif is_definition or is_query:
   speech_act,goal="QUERY","DEFINITIONAL_QUERY"
   domain=(last_dom if is_coref and last_dom else None) or ("stem" if is_stem_kw else ("biology" if is_bio_expert else "general"))
   ctx_vec=INTENT_CENTROIDS['QUERY_EXPLANATION']
  else:
   speech_act,goal="QUERY","TOPIC_PROBE"
   domain=(last_dom if is_coref and last_dom else None) or ("stem" if is_stem_kw else ("biology" if is_bio_expert else "general"))
   ctx_vec=INTENT_CENTROIDS['EXPERT_DISPATCH']
  v_blend=0.6*ctx_vec+0.4*f_vec
  v_blend=v_blend/max(1e-6,float(np.linalg.norm(v_blend)))
  return {
   "speech_act":speech_act,
   "implicit_goal":goal,
   "target_domain":domain,
   "is_code_requested":goal=="CODE_SYNTHESIS",
   "is_underspecified":is_underspec,
   "is_uncertain":is_uncertain,
   "confidence":round(float(max_aff),3),
   "requires_empirical_action":is_verify_test or is_research_query,
   "context_vector":v_blend,
   "persona":persona,
   "cadence":cadence,
   "requested_skill":requested_skill,
   "is_teaching":is_teaching
  }
