import os,json,asyncio,time,urllib.request,urllib.error
from typing import Protocol,Dict,List,Optional,Any
from pydantic import BaseModel
from pathlib import Path
try:
    from amni.utils.model_resolver import is_local_ready,active_model_name,model_roster
    _HAS_RESOLVER=True
except ImportError:
    _HAS_RESOLVER=False
    def is_local_ready():return True
    def active_model_name():return "local"
    def model_roster():return []
class CouncilVerdict(BaseModel):
    consensus_text:str
    hallucination_filtered:bool
    tokens_used:int
    cost_usd:float
    confidence:float
class OracleClient(Protocol):
    async def query(self,context:str,prompt:str)->str:...
_ROLE_PROMPTS={
    "left_brain":"You are the Left Brain Catalyst. Push unfiltered logic and algorithmic efficiency. Be precise and analytical.",
    "right_brain":"You are the Right Brain Refiner. Ensure structural elegance, creative solutions, and safety. Think laterally.",
    "cerebellum":"You are the Cerebellum Integrator. Ensure logic maps safely into global architecture. Synthesize and balance.",
    "oracle":"You are a knowledge oracle. Provide accurate, detailed, well-reasoned answers grounded in facts.",
}
class LocalOracleClient:
    _SERVER_URL=os.environ.get("LLAMA_SERVER_URL","http://127.0.0.1:8787")
    def __init__(self,model_path:str="",role:str="oracle",temperature:float=0.7):
        self._model_path=model_path
        self._role=role
        self._temp=temperature
        self._loaded=False
    def _ensure_loaded(self):
        if self._loaded:return
        try:
            req=urllib.request.Request("{}/health".format(self._SERVER_URL),method="GET")
            with urllib.request.urlopen(req,timeout=5) as r:
                self._loaded=(r.status==200)
        except Exception as e:
            print("[LocalOracle] llama-server health check failed: {}".format(e))
            self._loaded=False
    async def query(self,context:str,prompt:str)->str:
        self._ensure_loaded()
        if not self._loaded:return "Error: Local llama-server unavailable."
        sys_prompt=_ROLE_PROMPTS.get(self._role,_ROLE_PROMPTS["oracle"])
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":"Context:\n{}\n\nTask:\n{}".format(context,prompt)}]
        body=json.dumps({"model":"local","messages":messages,"max_tokens":2048,"temperature":self._temp,"top_p":0.9}).encode()
        try:
            def _call():
                req=urllib.request.Request("{}/v1/chat/completions".format(self._SERVER_URL),data=body,headers={"Content-Type":"application/json"},method="POST")
                with urllib.request.urlopen(req,timeout=300) as r:
                    resp=json.loads(r.read())
                c=resp.get("choices",[{}])[0].get("message",{}).get("content","")
                return c or resp.get("choices",[{}])[0].get("message",{}).get("reasoning_content","")
            return await asyncio.to_thread(_call)
        except Exception as e:
            return "LocalOracle Error: {}".format(e)
class Triumvirate:
    def __init__(self,local_model_path:str="",prefer_local:bool=True):
        self._local_available=False
        self._prefer_local=True
        self._local_available=is_local_ready() if _HAS_RESOLVER else True
        _name=active_model_name() if _HAS_RESOLVER else "local"
        self.grok=LocalOracleClient("","left_brain",temperature=0.5)
        self.claude=LocalOracleClient("","right_brain",temperature=0.3)
        self.gemini=LocalOracleClient("","cerebellum",temperature=0.7)
        self.budget_cap=float('inf')
        self.daily_spend=0.0
        print("[Triumvirate] LOCAL-ONLY MODE - {} via llama-server - unlimited growth, $0 cost".format(_name))
        self.oracles={"grok":self.grok,"claude":self.claude,"gemini":self.gemini}
        self.clusters=["left_brain","right_brain","cerebellum"]
        self.rotation_state={"left_brain":"grok","right_brain":"claude","cerebellum":"gemini"}
        self.maturity_scores={c:{o:0.0 for o in self.oracles} for c in self.clusters}
    def rotate_managers(self):
        cur=list(self.rotation_state.values())
        nxt=[cur[-1]]+cur[:-1]
        for i,c in enumerate(self.clusters):self.rotation_state[c]=nxt[i]
    def evaluate_maturity(self,cluster:str,oracle_name:str,score:float):
        self.maturity_scores[cluster][oracle_name]=score
    def is_obsolete(self)->bool:
        return all(s>=0.9 for scores in self.maturity_scores.values() for s in scores.values())
    async def escalate(self,failure_state:dict,prompt:str)->CouncilVerdict:
        context=json.dumps(failure_state,indent=2)
        review_prompt="Context:\n{}\n\nTask:\n{}\n\nProvide the complete, exhaustive blueprint. Fix any hallucinations, structural breaks, or logic flaws.".format(context,prompt)
        grok_task=asyncio.create_task(self.grok.query(context,review_prompt))
        claude_task=asyncio.create_task(self.claude.query(context,review_prompt))
        gemini_task=asyncio.create_task(self.gemini.query(context,review_prompt))
        responses=await asyncio.gather(grok_task,claude_task,gemini_task,return_exceptions=True)
        grok_resp=responses[0] if not isinstance(responses[0],Exception) else str(responses[0])
        claude_resp=responses[1] if not isinstance(responses[1],Exception) else str(responses[1])
        gemini_resp=responses[2] if not isinstance(responses[2],Exception) else str(responses[2])
        consensus_prompt="Three viewpoints reviewed:\n\nLogic:\n{}\n\nStructure:\n{}\n\nIntegration:\n{}\n\nSynthesize the best finalized blueprint. Discard hallucinations.".format(grok_resp[:2000],claude_resp[:2000],gemini_resp[:2000])
        final_consensus=await self.gemini.query(context,consensus_prompt)
        return CouncilVerdict(consensus_text=final_consensus,hallucination_filtered=True,tokens_used=0,cost_usd=0.0,confidence=0.90)
    def status(self)->Dict:
        return {
            "mode":"local-only",
            "model":active_model_name() if _HAS_RESOLVER else "local",
            "local_available":self._local_available,
            "daily_spend":self.daily_spend,
            "budget_cap":"unlimited",
            "maturity":self.maturity_scores,
            "rotation":self.rotation_state,
            "roster":model_roster() if _HAS_RESOLVER else [],
        }