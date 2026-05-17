import numpy as np,torch,time,json,hashlib,urllib.request,urllib.error,threading
from typing import Dict,List,Tuple,Optional
from collections import deque,defaultdict
from pathlib import Path
try:
    import amni_kernels as _ak
    _HAS_AK=True
except ImportError:
    _HAS_AK=False
try:
    import torch_directml as _dml
    _DML_DEV=_dml.device()
except (ImportError,Exception):
    _dml,_DML_DEV=None,None
DEVICE=torch.device("cuda") if torch.cuda.is_available() else (_DML_DEV if _DML_DEV else torch.device("cpu"))
WDTYPE=torch.float16 if (DEVICE.type=="cuda" or _DML_DEV) else torch.float32
P=17
_STATE_DIM=64
_MAX_CACHE=512
_DECAY_ALPHA=14
_INPUT_BETA=3
class GF17RecurrentState:
    __slots__=('_dim','_state','_vmin','_vmax','_step_ct','_hit_ct','_miss_ct','_diffuse_ct')
    def __init__(self,dim:int=_STATE_DIM):
        self._dim=dim
        self._state=np.zeros(dim*dim,dtype=np.uint8) if not _HAS_AK else np.array(_ak.gf17_state_init(dim),dtype=np.uint8)
        self._vmin=0.0
        self._vmax=1.0
        self._step_ct=0
        self._hit_ct=0
        self._miss_ct=0
        self._diffuse_ct=0
    def update(self,x_f32:np.ndarray,v_f32:np.ndarray,alpha:int=_DECAY_ALPHA,beta:int=_INPUT_BETA)->None:
        if not _HAS_AK:
            return
        x_q,xmin,xmax=_ak.gf17_quantize_f32_to_state(x_f32[:self._dim].astype(np.float32))
        v_q,vmin,vmax=_ak.gf17_quantize_f32_to_state(v_f32[:self._dim].astype(np.float32))
        self._state=np.array(_ak.gf17_state_update(
            self._state,np.array(x_q,dtype=np.uint8),np.array(v_q,dtype=np.uint8),
            alpha%P,beta%P,self._dim),dtype=np.uint8)
        self._vmin=min(self._vmin,xmin,vmin)
        self._vmax=max(self._vmax,xmax,vmax)
        self._step_ct+=1
    def query(self,q_f32:np.ndarray)->np.ndarray:
        if not _HAS_AK:
            return q_f32[:self._dim]
        q_q,_,_=_ak.gf17_quantize_f32_to_state(q_f32[:self._dim].astype(np.float32))
        result_q=np.array(_ak.gf17_state_query(self._state,np.array(q_q,dtype=np.uint8),self._dim),dtype=np.uint8)
        return np.array(_ak.gf17_dequantize_state_to_f32(result_q,self._vmin,self._vmax),dtype=np.float32)
    def diffuse(self,rounds:int=2)->None:
        if not _HAS_AK:
            return
        self._state=np.array(_ak.gf17_state_diffuse(self._state,self._dim,rounds),dtype=np.uint8)
        self._diffuse_ct+=1
    def compress(self)->bytes:
        return (_ak.gf17_state_compress(self._state,self._dim) if _HAS_AK
                else self._state[:self._dim].tobytes())
    def energy(self)->float:
        return float(np.sum(self._state.astype(np.float32))/(max(len(self._state),1)*8.0))
    def stats(self)->Dict:
        return {"dim":self._dim,"steps":self._step_ct,"energy":round(self.energy(),4),
                "diffusions":self._diffuse_ct,"state_bytes":len(self._state),
                "vram_kb":round(len(self._state)/1024,2)}
class GF17StateCache:
    __slots__=('_cache','_max','_hits','_misses','_lock')
    def __init__(self,max_entries:int=_MAX_CACHE):
        self._cache={}
        self._max=max_entries
        self._hits=0
        self._misses=0
        self._lock=threading.Lock()
    def _fingerprint(self,text:str)->str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]
    def lookup(self,query:str)->Optional[Dict]:
        fp=self._fingerprint(query)
        with self._lock:
            entry=self._cache.get(fp)
            if entry:
                self._hits+=1
                entry["access_ct"]+=1
                entry["last_access"]=time.time()
                return entry
            self._misses+=1
            return None
    def store(self,query:str,response:str,state:GF17RecurrentState,tok_s:float=0,vram_mb:float=0)->None:
        fp=self._fingerprint(query)
        with self._lock:
            if len(self._cache)>=self._max:
                oldest=min(self._cache,key=lambda k:self._cache[k]["last_access"])
                del self._cache[oldest]
            self._cache[fp]={"query":query[:200],"response":response,"state_energy":state.energy(),
                "compressed_state":state.compress(),"tok_s":tok_s,"vram_mb":vram_mb,
                "access_ct":1,"last_access":time.time(),"created":time.time()}
    def stats(self)->Dict:
        total=self._hits+self._misses
        return {"entries":len(self._cache),"max":self._max,"hits":self._hits,
                "misses":self._misses,"hit_rate":round(self._hits/max(total,1),4)}
class GF17InferenceBackend:
    __slots__=('_srv','_state','_cache','_tandem','_lock','_gen_ct','_total_tok',
               '_total_ms','_domain_states','_active_domain')
    def __init__(self,server_url:str="http://127.0.0.1:8787",tandem_engine=None):
        self._srv=server_url
        self._state=GF17RecurrentState(_STATE_DIM)
        self._cache=GF17StateCache(_MAX_CACHE)
        self._tandem=tandem_engine
        self._lock=threading.Lock()
        self._gen_ct=0
        self._total_tok=0
        self._total_ms=0.0
        self._domain_states={}
        self._active_domain="general"
    def _detect_domain(self,query:str)->str:
        if not self._tandem or not self._tandem.is_loaded:
            return "general"
        words=query.lower().split()
        from amni.core.lexicon import DOMAIN_NAMES
        scores=defaultdict(float)
        for w in words:
            wn=getattr(self._tandem,'_layout',None)
            scores["general"]+=0.1
        return max(scores,key=scores.get) if scores else "general"
    def _get_domain_state(self,domain:str)->GF17RecurrentState:
        if domain not in self._domain_states:
            self._domain_states[domain]=GF17RecurrentState(_STATE_DIM)
        return self._domain_states[domain]
    def _http_gen(self,messages:List[Dict],max_tokens:int=512,temp:float=0.3)->Dict:
        body=json.dumps({"messages":messages,"max_tokens":max_tokens,
            "temperature":max(temp,0.05),"top_p":0.85,"top_k":30,
            "repeat_penalty":1.2}).encode('utf-8')
        req=urllib.request.Request(f"{self._srv}/v1/chat/completions",data=body,
            headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=300) as resp:
            return json.loads(resp.read().decode('utf-8'))
    def generate(self,query:str,system_prompt:str="",max_tokens:int=512,
                 temp:float=0.3,use_cache:bool=True)->Dict:
        t0=time.perf_counter()
        cached=self._cache.lookup(query) if use_cache else None
        if cached:
            dt=(time.perf_counter()-t0)*1000
            self._gen_ct+=1
            return {"raw":cached["response"],"cached":True,"tok_s":cached["tok_s"],
                    "gen_time_ms":dt,"state_energy":cached["state_energy"],
                    "backend":"gf17_recurrent","cache_hit":True}
        domain=self._detect_domain(query)
        self._active_domain=domain
        ds=self._get_domain_state(domain)
        q_vec=np.random.RandomState(abs(hash(query))%2**31).randn(_STATE_DIM).astype(np.float32)
        state_context=ds.query(q_vec)
        state_hint=(f"\n[GF17 State Energy: {ds.energy():.3f}, Domain: {domain}]"
                    if ds.energy()>0.01 else "")
        messages=[]
        if system_prompt:
            messages.append({"role":"system","content":system_prompt+state_hint})
        messages.append({"role":"user","content":query})
        try:
            r=self._http_gen(messages,max_tokens,temp)
        except Exception as e:
            dt=(time.perf_counter()-t0)*1000
            return {"raw":f"Error: {e}","cached":False,"tok_s":0,"gen_time_ms":dt,
                    "backend":"gf17_recurrent","error":True}
        msg=r.get("choices",[{}])[0].get("message",{})
        raw=msg.get("content","") or msg.get("reasoning_content","")
        usage=r.get("usage",{})
        comp_tok=usage.get("completion_tokens",0)
        dt=(time.perf_counter()-t0)*1000
        tok_s=comp_tok/max(dt/1000,0.01)
        resp_vec=np.random.RandomState(abs(hash(raw[:100]))%2**31).randn(_STATE_DIM).astype(np.float32)
        ds.update(q_vec,resp_vec,_DECAY_ALPHA,_INPUT_BETA)
        if ds._step_ct%5==0:
            ds.diffuse(2)
        vram_est=sum(s._dim*s._dim for s in self._domain_states.values())/(1024*1024)
        if use_cache:
            self._cache.store(query,raw,ds,tok_s,vram_est)
        self._gen_ct+=1
        self._total_tok+=comp_tok
        self._total_ms+=dt
        return {"raw":raw,"cached":False,"tok_s":round(tok_s,2),"gen_time_ms":round(dt,1),
                "prompt_tokens":usage.get("prompt_tokens",0),"gen_tokens":comp_tok,
                "state_energy":round(ds.energy(),4),"domain":domain,
                "backend":"gf17_recurrent","cache_hit":False,
                "vram_state_kb":round(vram_est*1024,2)}
    def stats(self)->Dict:
        avg_tok_s=self._total_tok/max(self._total_ms/1000,0.01) if self._total_ms>0 else 0
        return {"backend":"gf17_recurrent","generations":self._gen_ct,
                "total_tokens":self._total_tok,"avg_tok_s":round(avg_tok_s,2),
                "avg_latency_ms":round(self._total_ms/max(self._gen_ct,1),1),
                "domains":len(self._domain_states),"active_domain":self._active_domain,
                "global_state":self._state.stats(),"cache":self._cache.stats(),
                "domain_states":{d:s.stats() for d,s in self._domain_states.items()},
                "has_rust_kernels":_HAS_AK}
    def reset(self)->None:
        self._state=GF17RecurrentState(_STATE_DIM)
        self._domain_states.clear()
        self._cache=GF17StateCache(_MAX_CACHE)
        self._gen_ct=0
        self._total_tok=0
        self._total_ms=0.0
