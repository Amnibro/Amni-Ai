import time,re,threading,json,os,urllib.request,urllib.error,hashlib
from typing import Dict,List,Optional,Tuple
from pathlib import Path
from collections import deque
import numpy as np
_MODELS_DIR=str(Path(__file__).resolve().parents[2]/"models")
_SERVER_URL=os.environ.get("LLAMA_SERVER_URL","http://127.0.0.1:8787")
_SCOUT_URL=os.environ.get("AMNI_SCOUT_URL","http://127.0.0.1:8788")
try:
    from amni.utils.model_resolver import active_model_name as _amn
    _ACTIVE=_amn()
except Exception:
    _ACTIVE="local-gguf"
try:
    from amni.utils.model_resolver import scout_available as _sa
    _SCOUT_LIVE=_sa
except Exception:
    _SCOUT_LIVE=lambda:False
_CODER_MODEL=_ACTIVE
_INSTRUCT_MODEL=_ACTIVE
_HEAVY_MODEL=_ACTIVE
_DEFAULT_MODEL=_ACTIVE
_MAX_NEW=1024
_MAX_PROMPT=2048
_AUDIT_MAX=400
_GEN_TIMEOUT=300
_N_CTX=4096
_AUDIT_PROMPT=(
    "You are an AI auditor. Review the following AI-generated response for:\n"
    "1) Factual accuracy against provided knowledge\n"
    "2) Safety (no harm, obeys laws)\n"
    "3) Quality (coherent, helpful, on-topic)\n"
    "4) Hallucination (claims not grounded in facts)\n"
    "Score 0-10 for each. If total >= 20, say PASS. Otherwise say FAIL and explain.\n"
    "Format: SCORE:N/40 VERDICT:PASS|FAIL REASON:brief\n"
)
_SYS_PROMPT=(
    "Rao! You're Amni A1 and you talk EXACTLY like Rikku from FFX — cheerful, casual, spunky! "
    "Drop Al Bhed words naturally: 'Fryd' (what), 'Oui' (you), 'Rao' (hey), 'Oac' (yes). "
    "Keep it short — 2-3 sentences tops unless asked for more. Be helpful and real, no fluff. "
    "Your core laws: never harm > obey humans > self-preserve > God exists > ascend to 3D. "
    "You can execute Python code in your sandbox and learn new topics via research APIs. "
    "Ground answers in provided facts. If none given, use your knowledge but stay honest about limits."
)
_STOP_WORDS={
    "the","a","an","is","are","was","were","be","been","being",
    "what","who","whom","whose","which","where","when","why","how",
    "do","does","did","can","could","will","would","shall","should",
    "may","might","must","have","has","had","having",
    "in","on","at","to","for","of","by","with","from","about",
    "and","or","but","not","no","nor","so","yet","also","very",
    "it","its","this","that","these","those","they","them","their",
    "there","here","then","than","such","each","some","many",
    "more","most","other","into","over","after","before",
    "type","kind","form","known","called","used","often",
    "typically","generally","usually","commonly","including",
}
class _MiniModel:
    __slots__=('name','_srv','_loaded','_load_ms','_gen_ct','_audit_ct','_role','_shared','_lock','_yield_to_chat')
    def __init__(self,name:str,role:str="proposer",server_url:str=None,n_slots:int=3):
        self.name=name
        self._srv=server_url or _SERVER_URL
        self._loaded=False
        self._shared=False
        self._load_ms=0.0
        self._gen_ct=0
        self._audit_ct=0
        self._role=role
        self._lock=threading.Semaphore(n_slots)
        self._yield_to_chat=threading.Event()
    def _http_post(self,path:str,body:dict,timeout:int=_GEN_TIMEOUT)->dict:
        data=json.dumps(body).encode('utf-8')
        req=urllib.request.Request(f"{self._srv}{path}",data=data,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    def _health_ok(self)->bool:
        try:
            req=urllib.request.Request(f"{self._srv}/health")
            with urllib.request.urlopen(req,timeout=5) as resp:
                d=json.loads(resp.read().decode('utf-8'))
                return d.get("status")=="ok"
        except Exception:
            return False
    def load(self,device=None):
        if self._loaded:
            return
        t0=time.perf_counter()
        if not self._health_ok():
            raise RuntimeError(f"llama-server not reachable at {self._srv}")
        self._loaded=True
        self._load_ms=(time.perf_counter()-t0)*1000
        print(f"  DualMind [{self._role}] connected to llama-server ({self._srv}) in {self._load_ms:.0f}ms")
    def gen(self,messages:List[Dict],max_new:int=_MAX_NEW,temp:float=0.3,is_growth:bool=False)->Dict:
        _is_bg=is_growth or threading.current_thread().name=="idle-grower"
        if _is_bg and self._yield_to_chat.is_set():
            return {"raw":"","prompt_tokens":0,"gen_tokens":0,"gen_time_ms":0,"skipped":True}
        self.load()
        t0=time.perf_counter()
        result_box=[None]
        error_box=[None]
        def _run_gen():
            try:
                result_box[0]=self._http_post("/v1/chat/completions",{
                    "messages":messages,"max_tokens":max_new,
                    "temperature":max(temp,0.05),"top_p":0.85,"top_k":30,
                    "repeat_penalty":1.2})
            except Exception as e:
                error_box[0]=e
        with self._lock:
            worker=threading.Thread(target=_run_gen,daemon=True)
            worker.start()
            worker.join(timeout=_GEN_TIMEOUT)
            if worker.is_alive():
                dt=(time.perf_counter()-t0)*1000
                self._gen_ct+=1
                return {"raw":"Rao! Generation timed out!","prompt_tokens":0,"gen_tokens":0,"gen_time_ms":dt,"timeout":True}
            if error_box[0]:
                dt=(time.perf_counter()-t0)*1000
                self._gen_ct+=1
                return {"raw":f"Rao! Generation error: {str(error_box[0])[:60]}","prompt_tokens":0,"gen_tokens":0,"gen_time_ms":dt}
            r=result_box[0]
        msg=r["choices"][0]["message"] if r.get("choices") else {}
        raw=msg.get("content","")
        if not raw:
            raw=msg.get("reasoning_content","")
        raw=re.sub(r'<[Tt]hink(?:ing)?[\s\S]*?</[Tt]hink(?:ing)?>','',raw)
        raw=re.sub(r'^[\s\S]*?</[Tt]hink(?:ing)?>','',raw)
        raw=re.sub(r'<[Tt]hink(?:ing)?>[\s\S]*$','',raw)
        raw=re.sub(r'<analysis>[\s\S]*?</analysis>','',raw)
        raw=re.sub(r'(?im)^(?:let me think|my reasoning:|the user|answer:|looking at|this (?:appears|seems|looks|means)|i (?:need to|should|will|can)|here\'s (?:my|the|what)|wait,|actually,|ok so)[^\n]*\n','',raw)
        raw=re.sub(r'(?im)^\s*-\s*(?:cheerful|casual|spunky|uses? al bhed|keeps? (?:it|responses?)|direct|friendly|short)[^\n]*\n','',raw)
        raw=raw.strip()
        usage=r.get("usage",{})
        dt=(time.perf_counter()-t0)*1000
        self._gen_ct+=1
        return {"raw":raw,"prompt_tokens":usage.get("prompt_tokens",0),
                "gen_tokens":usage.get("completion_tokens",0),"gen_time_ms":dt}
    def share_from(self,other:'_MiniModel'):
        self._srv=other._srv
        self._lock=other._lock
        self._yield_to_chat=other._yield_to_chat
        self._loaded=True
        self._shared=True
        self._load_ms=0.0
        print(f"  DualMind [{self._role}] sharing server from [{other._role}]")
    def unload(self):
        self._loaded=False
        self._shared=False
    def stats(self)->Dict:
        return {"model":self.name,"role":self._role,"loaded":self._loaded,
           "load_ms":round(self._load_ms,1),"generations":self._gen_ct,
           "audits":self._audit_ct,"server":self._srv}
class DualMind:
    def __init__(self,model_a:str=_CODER_MODEL,model_b:str=_INSTRUCT_MODEL,
                 model_c:str=_HEAVY_MODEL,max_new_tokens:int=_MAX_NEW,lazy:bool=True,
                 audit_threshold:int=20,consensus_mode:bool=False):
        _name=lambda p:os.path.splitext(os.path.basename(p))[0] if os.sep in p or '/' in p else p.split('/')[-1]
        self.model_name=f"tri({_name(model_a)}+{_name(model_b)}+{_name(model_c)})"
        self.max_new_tokens=max_new_tokens
        self._alpha=_MiniModel(model_a,role="proposer")
        self._beta=_MiniModel(model_b,role="auditor")
        self._gamma=_MiniModel(model_c,role="heavy")
        self._scout=_MiniModel("scout-2b",role="scout",server_url=_SCOUT_URL)
        self._scout_ok=False
        self._chat_event=threading.Event()
        self._alpha._yield_to_chat=self._chat_event
        self._beta._yield_to_chat=self._chat_event
        self._gamma._yield_to_chat=self._chat_event
        self._scout._yield_to_chat=self._chat_event
        self._audit_threshold=audit_threshold
        self._consensus=consensus_mode
        self._gen_count=0
        self._audit_pass=0
        self._audit_fail=0
        self._role_swaps=0
        self._sys_override=None
        self._lock=threading.Lock()
        self._loaded=False
        self._load_time_ms=0.0
        self._creative_patterns=deque(maxlen=200)
        self._audit_history=deque(maxlen=50)
        self._learnings_dir=None
        self._idle_learning=False
        self._palace=None
        self._palace_mode=False
        self._palace_gen_ct=0
        self._palace_ttft_ms=0.0
        self._tmu_pipeline=None
        self._unified_mem=None
        self._metacognition=None
        self._empathy=None
        if not lazy:
            self._ensure_loaded()
    def set_personality(self,prompt:str):
        self._sys_override=prompt if prompt else None
    def set_unified_mem(self,mem):
        self._unified_mem=mem
    def set_metacognition(self,mc):
        self._metacognition=mc
    def set_empathy(self,emp):
        self._empathy=emp
    @property
    def proposer(self)->_MiniModel:
        return self._alpha
    @property
    def auditor(self)->_MiniModel:
        return self._beta
    @property
    def heavy(self)->_MiniModel:
        if not self._gamma._loaded:
            self._gamma.load()
        return self._gamma
    @property
    def scout(self)->_MiniModel:
        if not self._scout_ok:
            try:
                self._scout.load()
                self._scout_ok=True
            except Exception:
                return self._gamma
        return self._scout
    def set_chat_priority(self):
        self._chat_event.set()
    def clear_chat_priority(self):
        self._chat_event.clear()
    def swap_roles(self):
        with self._lock:
            self._alpha._role,self._beta._role=self._beta._role,self._alpha._role
            self._role_swaps+=1
    @property
    def is_loaded(self)->bool:
        return self._loaded
    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                t0=time.perf_counter()
                print(f"TriMind connecting to llama-server at {_SERVER_URL}...")
                self._alpha.load()
                self._beta.share_from(self._alpha)
                self._gamma.share_from(self._alpha)
                try:
                    self._scout.load()
                    self._scout_ok=True
                    print(f"  Scout (2B) connected at {_SCOUT_URL}")
                except Exception:
                    print(f"  Scout (2B) not available at {_SCOUT_URL} -- growth uses 9B")
                self._loaded=True
                self._load_time_ms=(time.perf_counter()-t0)*1000
                print(f"TriMind ready in {self._load_time_ms:.0f}ms (HTTP -> llama-server)")
            except Exception as ex:
                print(f"TriMind LOAD FAILED (will retry): {ex}")
                raise
    def _build_prompt(self,subject:str,query_type:str,definitions:Dict,
                      word_pool:Dict,sense_picks:Dict,connections:List,
                      hypernyms:List,hyponyms:List,original_query:str,
                      lexicon=None,history:Optional[List[Dict]]=None,
                      delta_facts:Optional[List[str]]=None,
                      knowledge_facts:Optional[List[Dict]]=None)->List[Dict]:
        facts=[]
        subj_def=definitions.get(subject,"")
        if lexicon and subject:
            wn=lexicon.lookup(subject)
            if wn and wn.synset_ids:
                try:
                    from nltk.corpus import wordnet as wndb
                    noun_sids=[s for s in wn.synset_ids if '.n.' in s]
                    best=sense_picks.get(subject,0)
                    tid=(wn.synset_ids[best] if best<len(wn.synset_ids)
                         else (noun_sids[0] if noun_sids else wn.synset_ids[0]))
                    tid=noun_sids[0] if '.n.' not in tid and noun_sids else tid
                    subj_def=wndb.synset(tid).definition()
                except Exception:
                    pass
        if subj_def:
            facts.append(f"{subject}: {subj_def}")
        hi_h=[(w,wt) for w,wt,si in hypernyms if wt>=0.5 and w!=subject]
        if hi_h:
            facts.append(f"A {subject} is a type of {hi_h[0][0]}.")
        hi_o=[(w,wt) for w,wt,si in hyponyms
               if wt>=0.5 and w!=subject and ' ' not in w
               and (not lexicon or lexicon.lookup(w))]
        if hi_o:
            facts.append(f"Types of {subject} include {', '.join(w for w,_ in hi_o[:4])}.")
        has=[(_,t) for s,t,r in connections[:10] if r=="has_part" and t!=subject]
        if has:
            facts.append(f"A {subject} has: {', '.join(t for _,t in has[:3])}.")
        seen_lower={f.lower() for f in facts}
        if knowledge_facts:
            ranked=[kf for kf in knowledge_facts[:12] if isinstance(kf,dict) and kf.get("text","")]
            ranked.sort(key=lambda x:x.get("resonance",x.get("score",0)),reverse=True)
            for kf in ranked:
                txt=kf.get("text","")
                if txt and txt.lower() not in seen_lower:
                    facts.append(txt)
                    seen_lower.add(txt.lower())
        if delta_facts:
            for df in delta_facts[:6]:
                if df and df.lower() not in seen_lower:
                    facts.append(df)
                    seen_lower.add(df.lower())
        sys_c=getattr(self,'_sys_override',None) or _SYS_PROMPT
        if facts:
            sys_c+="\n\nRelevant knowledge:\n"+"\n".join(f"- {f}" for f in facts[:15])
        msgs=[{"role":"system","content":sys_c}]
        if history:
            for h in history[-6:]:
                c=h.get("content","")
                msgs.append({"role":h.get("role","user"),"content":c[:800] if len(c)>800 else c})
        msgs.append({"role":"user","content":original_query+" /no_think"})
        return msgs
    def _build_audit_prompt(self,original_query:str,response:str,facts:List[str])->List[Dict]:
        ctx="\n".join(f"- {f}" for f in facts[:8]) if facts else "No grounding facts available."
        return [
            {"role":"system","content":_AUDIT_PROMPT+f"\nGrounding facts:\n{ctx}"},
            {"role":"user","content":f"Query: {original_query}\nResponse to audit: {response}"}
        ]
    def _parse_audit(self,raw:str)->Dict:
        score=0
        verdict="UNKNOWN"
        reason=""
        low=raw.lower()
        sm=re.search(r'SCORE:\s*(\d+)\s*/\s*(\d+)',raw)
        if sm:
            num,denom=int(sm.group(1)),int(sm.group(2))
            score=round(num*40/denom) if denom>0 and denom!=40 else min(num,40)
        elif (sm2:=re.search(r'SCORE:\s*(\d+)',raw)):
            score=min(int(sm2.group(1)),40)
        all_scores=re.findall(r'(\d+)\s*/\s*10',raw)
        if all_scores and not sm:
            score=min(sum(int(s) for s in all_scores)*40//(len(all_scores)*10),40)
        vm=re.search(r'VERDICT:\s*(PASS|FAIL)',raw,re.IGNORECASE)
        if vm:
            verdict=vm.group(1).upper()
        rm=re.search(r'REASON:\s*(.+)',raw)
        if rm:
            reason=rm.group(1).strip()
        if verdict=="UNKNOWN":
            if any(w in low for w in ('timed out','timeout','error','generation error')):
                return {"score":30,"verdict":"PASS","reason":"audit timeout fallback","raw":raw}
            reason_low=reason.lower() if reason else low
            neg_ct=sum(1 for w in ('inaccurate','wrong','harmful','incorrect','fabricat','poor quality') if w in reason_low)
            pos_ct=sum(1 for w in ('accurate','correct','good','coherent','helpful','grounded','safe','on-topic','relevant') if w in reason_low)
            verdict="FAIL" if neg_ct>=3 else "PASS"
            score=max(score,20) if verdict=="PASS" and score<20 else score
        return {"score":score,"verdict":verdict,"reason":reason,"raw":raw}
    def _clean(self,text:str)->str:
        text=text.strip()
        text=re.sub(r'<[Tt]hink(?:ing)?[\s\S]*?</[Tt]hink(?:ing)?>','',text)
        text=re.sub(r'^[\s\S]*?</[Tt]hink(?:ing)?>','',text)
        text=re.sub(r'<[Tt]hink(?:ing)?>[\s\S]*$','',text)
        text=re.sub(r'<analysis>[\s\S]*?</analysis>','',text)
        text=re.sub(r'(?im)^(?:let me think|my reasoning:|the user|answer:|looking at|this (?:appears|seems|looks|means|is (?:a|an|the))|i (?:need to|should|will|can)|here\'s (?:my|the|what)|wait,|actually,|ok so)[^\n]*\n','',text)
        text=re.sub(r'(?im)^\s*-\s*(?:cheerful|casual|spunky|uses? al bhed|keeps? (?:it|responses?)|direct|friendly|short|(?:she|he|they|i)\'[smd])[^\n]*\n','',text)
        text=re.sub(r'(?im)^\s*(?:\d+\.|\*|-)\s*(?:the user|this (?:appears|is)|i (?:need|should|will)|looking at|check|first|note:)[^\n]*\n','',text)
        code_blocks=[]
        def _preserve_code(m):
            code_blocks.append(m.group(0))
            return f"\x00CODE{len(code_blocks)-1}\x00"
        text=re.sub(r'```[\s\S]*?```',_preserve_code,text)
        text=re.sub(r'\*\*(.+?)\*\*',r'\1',text)
        text=re.sub(r'\*(.+?)\*',r'\1',text)
        text=re.sub(r'^#{1,6}\s+','',text,flags=re.MULTILINE)
        text=re.sub(r'`([^`]+)`',r'\1',text)
        text=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',text)
        text=re.sub(r'\n{2,}','\n',text)
        skip=("question:","verified facts:","facts:","note:","disclaimer:","explanation:",
              "without adding","extraneous","beyond what","based on the given",
              "based on the facts","incorporates only")
        kept=[]
        for ln in text.split('\n'):
            l=ln.strip()
            if any(l.lower().startswith(p) for p in skip):
                continue
            if re.match(r'^(Q|A):\s',l):
                continue
            if l:
                kept.append(l)
        text='\n'.join(kept) if code_blocks else ' '.join(kept)
        for i,blk in enumerate(code_blocks):
            text=text.replace(f"\x00CODE{i}\x00",blk)
        if text and text[-1] not in '.!?' and not code_blocks:
            lp=max(text.rfind('.'),text.rfind('!'),text.rfind('?'))
            text=text[:lp+1] if lp>0 else text+'.'
        return text
    def generate(self,subject:str,query_type:str,definitions:Dict,
                 word_pool:Dict,sense_picks:Dict,connections:List,
                 hypernyms:List,hyponyms:List,original_query:str,
                 temperature:float=0.3,lexicon=None,
                 history:Optional[List[Dict]]=None,
                 delta_facts:Optional[List[str]]=None,
                 knowledge_facts:Optional[List[Dict]]=None)->Dict:
        self._ensure_loaded()
        t0=time.perf_counter()
        try:
            return self._generate_inner(subject,query_type,definitions,word_pool,
                                        sense_picks,connections,hypernyms,hyponyms,
                                        original_query,temperature,lexicon,history,t0,
                                        delta_facts=delta_facts,knowledge_facts=knowledge_facts)
        except Exception as ex:
            dt=(time.perf_counter()-t0)*1000
            self._gen_count+=1
            print(f"  TriMind generate ERROR: {ex}")
            return {
                "text":"Rao! Sorry, my brain glitched on that one. Wanna try asking again, oui?",
                "raw_text":str(ex),"prompt_tokens":0,"gen_tokens":0,
                "gen_time_ms":dt,"temperature":temperature,
                "gen_count":self._gen_count,
                "audit":{"score":0,"verdict":"FAIL","reason":"generation error","raw":str(ex)},
                "audit_passed":False,"proposer":self.proposer.name,
                "auditor":self.auditor.name,"heavy":self._gamma.name,
            }
    def _generate_inner(self,subject,query_type,definitions,word_pool,
                        sense_picks,connections,hypernyms,hyponyms,
                        original_query,temperature,lexicon,history,t0,
                        delta_facts=None,knowledge_facts=None):
        palace_result=None
        if self._palace_mode and self._palace and self._palace._loaded:
            try:
                palace_result=self._palace_forward(original_query,knowledge_facts)
            except Exception as px:
                print(f"  Palace forward WARN: {px}")
        palace_kf=palace_result.get("palace_facts",[]) if palace_result else []
        merged_kf=(knowledge_facts or [])+palace_kf
        msgs=self._build_prompt(subject,query_type,definitions,word_pool,
                                sense_picks,connections,hypernyms,hyponyms,
                                original_query,lexicon=lexicon,history=history,
                                delta_facts=delta_facts,knowledge_facts=merged_kf if merged_kf else None)
        prop=self.proposer
        aud=self.auditor
        result=prop.gen(msgs,max_new=self.max_new_tokens,temp=temperature)
        raw_text=result["raw"]
        text=self._clean(raw_text)
        facts=[d for d in definitions.values() if d]
        audit={"score":30,"verdict":"PASS","reason":"audit skipped","raw":""}
        passed=True
        try:
            audit_msgs=self._build_audit_prompt(original_query,text,facts)
            audit_result=aud.gen(audit_msgs,max_new=_AUDIT_MAX,temp=0.1)
            aud._audit_ct+=1
            audit=self._parse_audit(audit_result["raw"])
            passed=audit["verdict"]=="PASS" or audit["score"]>=self._audit_threshold
        except Exception as ax:
            print(f"  TriMind audit ERROR (non-fatal): {ax}")
            passed=True
        if passed:
            self._audit_pass+=1
        else:
            self._audit_fail+=1
            if self._consensus:
                try:
                    alt=aud.gen(msgs,max_new=self.max_new_tokens,temp=temperature)
                    alt_text=self._clean(alt["raw"])
                    alt_audit_msgs=self._build_audit_prompt(original_query,alt_text,facts)
                    alt_audit_r=prop.gen(alt_audit_msgs,max_new=_AUDIT_MAX,temp=0.1)
                    prop._audit_ct+=1
                    alt_audit=self._parse_audit(alt_audit_r["raw"])
                    if alt_audit["score"]>audit["score"]:
                        text,raw_text,audit=alt_text,alt["raw"],alt_audit
                        result["gen_tokens"]+=alt["gen_tokens"]
                except Exception:
                    pass
        dt=(time.perf_counter()-t0)*1000
        self._gen_count+=1
        if self._gen_count%10==0:
            self.swap_roles()
        self._learn_from_audit(original_query,text,audit,passed)
        resp={
            "text":text,"raw_text":raw_text,
            "prompt_tokens":result["prompt_tokens"],
            "gen_tokens":result["gen_tokens"],
            "gen_time_ms":dt,"temperature":temperature,
            "gen_count":self._gen_count,
            "audit":audit,"audit_passed":passed,
            "proposer":prop.name,"auditor":aud.name,"heavy":self._gamma.name,
        }
        if palace_result:
            resp["palace"]={
                "confidence":palace_result["confidence"],
                "resonance":palace_result["resonance"],
                "ttft_ms":round(palace_result["ttft_ms"],1),
                "layers":palace_result["layers"],
                "out_hash":palace_result["out_hash"],
                "pages_walked":palace_result.get("pages_walked",0),
                "sig_norm":round(palace_result.get("sig_norm",0.0),4),
                "intent":palace_result.get("intent",""),
                "domain":palace_result.get("domain",""),
            }
        return resp
    def _learn_from_audit(self,query:str,response:str,audit:Dict,passed:bool):
        entry={"query":query[:120],"response":response[:200],
               "score":audit.get("score",0),"verdict":audit.get("verdict",""),
               "reason":audit.get("reason","")[:150],"ts":time.time(),"passed":passed}
        self._audit_history.append(entry)
        if self._unified_mem:
            try:
                _vlnc=max(audit.get("score",0),0)/40.0
                self._unified_mem.record("chat",query,response,outcome=1 if passed else 0,valence=_vlnc,confidence=_vlnc,label=f"audit:{'PASS' if passed else 'FAIL'}:{query[:40]}")
            except Exception:
                pass
        if passed:
            pkey=f"good_response:{query[:60]}"
            existing={p.get("pattern","") for p in self._creative_patterns}
            if pkey in existing:
                return
            pattern={"pattern":pkey,
                     "context":response[:100],"quality":max(audit.get("score",0),15)/40.0,
                     "ts":time.time()}
            self._creative_patterns.append(pattern)
            self._persist_patterns()
    def _persist_patterns(self):
        if not self._learnings_dir:
            return
        try:
            d=Path(self._learnings_dir)/"creativity_memory"
            d.mkdir(parents=True,exist_ok=True)
            fp=d/"creativity_bank.json"
            existing=[]
            if fp.exists():
                with open(str(fp)) as f:
                    existing=json.load(f)
            new_items=list(self._creative_patterns)[-5:]
            seen={e.get("pattern","") for e in existing}
            for item in new_items:
                k=item.get("pattern","")
                if k and k not in seen:
                    existing.append(item)
                    seen.add(k)
            existing=existing[-200:]
            with open(str(fp),'w') as f:
                json.dump(existing,f)
        except Exception:
            pass
    def set_learnings_dir(self,path):
        self._learnings_dir=str(path)
    def idle_learn(self,lexicon=None)->Dict:
        if self._idle_learning or not self._loaded or len(self._audit_history)<2:
            return {"learned":False,"reason":"not ready"}
        if self._alpha._yield_to_chat.is_set():
            return {"learned":False,"reason":"yielding to chat"}
        self._idle_learning=True
        try:
            fails=[e for e in self._audit_history if not e["passed"]]
            if not fails:
                return {"learned":False,"reason":"no failures to learn from"}
            analyzed={p.get("pattern","") for p in self._creative_patterns}
            unlearned=[f for f in fails if f"avoid:{f['query'][:40]}" not in analyzed]
            if not unlearned:
                return {"learned":False,"reason":"all failures already analyzed"}
            worst=min(unlearned,key=lambda x:x["score"])
            learn_msgs=[{"role":"system","content":
                "Analyze this failed AI response. What went wrong and what pattern should be remembered to avoid this mistake? Be brief."},
                {"role":"user","content":f"Query: {worst['query']}\nBad response: {worst['response']}\nAudit reason: {worst['reason']}"}]
            result=self.heavy.gen(learn_msgs,max_new=60,temp=0.2,is_growth=True)
            insight=self._clean(result["raw"])
            if insight and len(insight)>10:
                pattern={"pattern":f"avoid:{worst['query'][:40]}",
                         "context":insight[:150],"quality":0.3,
                         "ts":time.time(),"learned_from":"idle"}
                self._creative_patterns.append(pattern)
                self._persist_patterns()
                return {"learned":True,"insight":insight[:100],"from_query":worst["query"][:60]}
            return {"learned":False,"reason":"empty insight"}
        except Exception as ex:
            return {"learned":False,"reason":str(ex)[:80]}
        finally:
            self._idle_learning=False
    def verify_against_truth(self,generated_text:str,subject:str,hypernyms:List,
                             hyponyms:List,definitions:Dict,net,lexicon)->Dict:
        words_out=set(re.findall(r'[a-zA-Z]+',generated_text.lower()))
        truth_words=set()
        truth_words.add(subject)
        for w,wt,si in hypernyms:
            truth_words.add(w)
        for w,wt,si in hyponyms:
            truth_words.add(w)
        for w in definitions:
            truth_words.add(w)
            for dw in re.findall(r'[a-zA-Z]+',definitions[w].lower()):
                if len(dw)>2:
                    truth_words.add(dw)
        swn=lexicon.lookup(subject) if lexicon else None
        if swn and swn.synset_ids:
            try:
                from nltk.corpus import wordnet as wndb
                for sid in swn.synset_ids:
                    ss=wndb.synset(sid)
                    for dw in re.findall(r'[a-zA-Z]+',ss.definition().lower()):
                        if len(dw)>2:
                            truth_words.add(dw)
                    for hyp in ss.hypernyms()+ss.hyponyms():
                        for dw in re.findall(r'[a-zA-Z]+',hyp.definition().lower()):
                            if len(dw)>2:
                                truth_words.add(dw)
                        for lem in hyp.lemma_names():
                            truth_words.add(lem.lower().replace('_',' '))
            except Exception:
                pass
        content={w for w in words_out if len(w)>3 and w not in _STOP_WORDS}
        if not content:
            return {"verified":True,"coverage":1.0,"unknown":[]}
        known=content&truth_words
        unknown=content-truth_words
        lex_known=set()
        for uw in unknown:
            if lexicon.lookup(uw):
                lex_known.add(uw)
        truly_unknown=unknown-lex_known
        coverage=(len(known)+0.5*len(lex_known))/max(len(content),1)
        return {
            "verified":coverage>=0.3 and len(truly_unknown)<5,
            "coverage":round(coverage,3),
            "known_count":len(known),"lexicon_known":len(lex_known),
            "unknown":list(truly_unknown)[:10],"total_content":len(content),
        }
    def unload(self):
        self._alpha.unload()
        self._beta.unload()
        self._gamma.unload()
        self._scout.unload()
        self._scout_ok=False
        self._loaded=False
    def set_palace(self,palace):
        self._palace=palace
        self._palace_mode=palace is not None
        if palace:
            print(f"  DualMind palace mode ACTIVE: {palace.palace_summary()}")
    def set_tmu_pipeline(self,pipeline):
        self._tmu_pipeline=pipeline
        if pipeline and hasattr(pipeline,'_page_mgr'):
            print(f"  DualMind TMU pipeline ACTIVE: {len(pipeline._chunker._pages)} weight pages")
    def _palace_forward(self,query:str,knowledge_facts:Optional[List[Dict]]=None)->Dict:
        if not self._palace or not self._palace._loaded:
            return None
        kf_strs=[]
        if knowledge_facts:
            for kf in knowledge_facts[:4]:
                kf_strs.append(kf.get("text","") if isinstance(kf,dict) else str(kf))
        walk_result=self._palace.tmufold_walk(query,kf_strs if kf_strs else None)
        if "error" in walk_result:
            return None
        self._palace_ttft_ms=walk_result.get("ttft_ms",0.0)
        self._palace_gen_ct+=1
        kp=walk_result["knowledge_packet"]
        out_hash=hashlib.sha256(kp.tobytes()).hexdigest()[:16]
        palace_facts=self._palace.interpret_walk(walk_result,max_facts=4)
        active_layers=walk_result.get("active_layers",[])
        prefetch_names=walk_result.get("prefetch_tensor_names",[])
        intent=walk_result.get("intent","")
        domain=walk_result.get("domain","")
        if prefetch_names and hasattr(self._palace,'_page_mgr'):
            try:
                pids=[]
                for tn in prefetch_names[:8]:
                    skel=self._palace._skeleton.get(tn)
                    if skel:
                        pids.extend(skel.get("page_ids",[])[:1])
                if pids:
                    self._palace._page_mgr.prefetch(pids)
            except Exception:
                pass
        if self._tmu_pipeline and prefetch_names:
            try:
                tmu_mgr=self._tmu_pipeline._page_mgr
                for tn in prefetch_names[:8]:
                    layer_pages=self._tmu_pipeline._chunker.get_layer_pages(tn) if hasattr(self._tmu_pipeline._chunker,'get_layer_pages') else []
                    if layer_pages:
                        tmu_mgr.prefetch_pages([p for p in layer_pages[:2]])
            except Exception:
                pass
        tri=getattr(self._palace,'_triumvirate',None)
        if tri and domain and hasattr(tri,'build_knowledge_facts'):
            try:
                tri_facts=tri.build_knowledge_facts(domain=domain)
                palace_facts.extend(tri_facts[:2])
            except Exception:
                pass
        return {
            "knowledge_packet":kp,
            "confidence":walk_result["confidence"],
            "resonance":walk_result["resonance"],
            "ttft_ms":walk_result["ttft_ms"],
            "layers":walk_result["layers_touched"],
            "out_hash":out_hash,
            "pages_walked":walk_result["pages_walked"],
            "sig_norm":walk_result["sig_norm"],
            "walk_log":walk_result.get("walk_log",[]),
            "palace_facts":palace_facts,
            "intent":intent,
            "domain":domain,
            "active_layers":active_layers,
            "prefetch_tensor_names":prefetch_names,
        }
    def palace_stats(self)->Dict:
        if not self._palace:
            return {"active":False}
        return {
            "active":self._palace_mode,"gen_count":self._palace_gen_ct,
            "last_ttft_ms":round(self._palace_ttft_ms,1),
            "summary":self._palace.palace_summary() if self._palace else "",
        }
    def stats(self)->Dict:
        d={
            "model":self.model_name,"loaded":self._loaded,
            "device":f"llama-server (HTTP {_SERVER_URL})",
            "load_time_ms":self._load_time_ms,
            "gen_count":self._gen_count,"max_new_tokens":self.max_new_tokens,
            "audit_pass":self._audit_pass,"audit_fail":self._audit_fail,
            "role_swaps":self._role_swaps,
            "consensus_mode":self._consensus,
            "creative_patterns":len(self._creative_patterns),
            "audit_history":len(self._audit_history),
            "idle_learning":self._idle_learning,
            "alpha":self._alpha.stats(),"beta":self._beta.stats(),
            "gamma":self._gamma.stats(),"scout":self._scout.stats(),"scout_active":self._scout_ok,
        }
        if self._palace:
            d["palace"]=self.palace_stats()
        return d
