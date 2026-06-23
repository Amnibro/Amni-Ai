"""Adam facade — single import surface for the deployable model.
Usage:
  from amni.adam import Adam
  from amni.bootstrap import load_config
  cfg = load_config()
  adam = Adam(bake=cfg['bake'], model=cfg['model'])
  result = adam.ask('What is 2 + 2?')
  print(result['answer'], result['tier'], result['tokens'])
Paths auto-detect via AMNI_HOME / ~/.amni-ai/config.json / candidate dirs. Set $AMNI_BAKE_PATHS or $AMNI_MODEL_PATHS to add custom locations.
Wraps AdamLoop with sensible defaults, persistent SemanticPTEXLUT, all tiers enabled.
"""
import os,json,time,hashlib
from pathlib import Path
from typing import Optional,Dict,Any,List,Tuple
class Adam:
    def __init__(self,bake:str,model:str,lessons_path:str='experiences/adam_lessons.npz',lut_root:str='experiences/adam_lut',budget_mb:int=8000,seed_lessons:Optional[list]=None,enable_crawler:bool=True,web_unrestricted:bool=True):
        self.bake=bake;self.model=model;self.lessons_path=Path(lessons_path);self.lut_root=lut_root
        self.web_unrestricted=web_unrestricted and not bool(os.environ.get('AMNI_WEB_RESTRICTED'))
        os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1')
        from amni.inference.streaming_chat import StreamingChatService
        from amni.inference.adam_loop import AdamLoop
        from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
        self._SemanticPTEXLUT=SemanticPTEXLUT
        t0=time.time()
        self.svc=None;self.svc_boot_s=0.0;self.runtime_error=None
        try:
            _mf=Path(bake)/'bake_manifest.json';_isnv=False
            try:_isnv=str(json.load(open(_mf)).get('format','')).startswith('nvfp4') if _mf.exists() else False
            except Exception:_isnv=False
            if _isnv:
                from amni.inference.nvfp4_atex_svc import Nvfp4AtexChatService
                self.svc=Nvfp4AtexChatService(bake=bake,tok_src=model);print(f'[Adam] NVFP4-ATEX server model loaded: {bake}',flush=True)
            else:self.svc=StreamingChatService(bake,model,budget_mb=budget_mb)
            self.svc_boot_s=time.time()-t0
        except Exception as e:
            self.runtime_error=str(e)
            print(f'[Adam] WARNING: StreamingChatService unavailable — chat generation will return runtime-error messages. Cached lesson-bank LUT hits, /healthz, /stats still work.\n  Reason: {e}',flush=True)
        self.crawler_plugin=None
        if enable_crawler:
            try:
                from amni.inference.web_crawler import CrawlerPlugin
                self.crawler_plugin=CrawlerPlugin(distiller_svc=self.svc,max_pages=2,distill_max_tokens=180,unrestricted=self.web_unrestricted)
                _mode='UNRESTRICTED (any domain)' if self.web_unrestricted else f'{len(self.crawler_plugin.crawler.allow)} trusted domains'
                print(f'[Adam] crawler plugin enabled ({_mode})',flush=True)
            except Exception as e:print(f'[Adam] crawler init failed (web-learn will fallback): {e}',flush=True)
        _routed=os.environ.get('AMNI_ROUTED_LESSONS','').lower() in ('1','true','yes')
        _base=str(self.lessons_path).removesuffix('.npz')
        if _routed:
            from amni.inference.routed_lessons import RoutedSemanticLUT
            _packroot=str(self.lessons_path.parent/'lesson_packs')
            try:
                if Path(_base+'.map.json').exists():
                    self.sem_lut=RoutedSemanticLUT.load(_base)
                elif self.lessons_path.exists():
                    _flat=SemanticPTEXLUT.load(_base)
                    print(f'[Adam] migrating {len(_flat._raw)} lessons -> routed map-PTEX store (one-time)...',flush=True)
                    self.sem_lut=RoutedSemanticLUT.from_flat(_flat,root=_packroot);self.sem_lut.save(_base)
                    print(f'[Adam] routed store ready: {self.sem_lut.stats()}',flush=True)
                else:self.sem_lut=RoutedSemanticLUT(grid=64,pca_dim=8,root=_packroot)
            except Exception as e:
                print(f'[Adam] routed lessons init failed, falling back to flat: {e}',flush=True)
                self.sem_lut=SemanticPTEXLUT(grid=64,pca_dim=8)
        elif self.lessons_path.exists():
            try:self.sem_lut=SemanticPTEXLUT.load(_base)
            except Exception as e:
                print(f'[Adam] failed to load lessons from {self.lessons_path}: {e}',flush=True)
                self.sem_lut=SemanticPTEXLUT(grid=64,pca_dim=8)
        else:
            self.sem_lut=SemanticPTEXLUT(grid=64,pca_dim=8)
        if seed_lessons and len(self.sem_lut._raw)==0:
            for q,a in seed_lessons:self.sem_lut.add(q,a)
            self.sem_lut.fit()
            self.save_lessons()
        self.adam=AdamLoop(self.svc,lut_root=self.lut_root,letter_only=False,tier3_cot_max_tokens=200,semantic_lut=self.sem_lut if len(self.sem_lut._raw)>0 else None,semantic_margin='auto',shape_sorter=True,chord_sampler=True,chord_n_frames=5,chord_min_conf=1.0,calc_tool=True,crawler_plugin=self.crawler_plugin)
        self._writeback_counter=0
        self._writeback_every=1
    def ask(self,query:str,writeback:bool=True)->Dict[str,Any]:
        t0=time.time()
        if self.svc is None:
            return {'answer':f'[Adam runtime not installed — generation requires the GF(17) streaming backend. The bake at `{self.bake}` could not be loaded. Run install.py to fetch from HF, or set AMNI_BAKE env var to point at an existing bake.]','error':f'runtime not installed: {(self.runtime_error or "")[:200]}','tier':'runtime_missing','tokens':0,'wall_s':round(time.time()-t0,3)}
        try:ans,tier,n=self.adam.answer(query,writeback=writeback)
        except Exception as e:return {'answer':None,'error':str(e),'tier':'ERROR','tokens':0,'wall_s':time.time()-t0}
        wall=time.time()-t0
        if writeback and tier in ('tier37_calc_tool','tier36_chord_sampler','tier35_shape_sorter','tier3_escalated','tier3_cold'):
            self._writeback_counter+=1
            if self._writeback_counter>=self._writeback_every:
                self._writeback_counter=0
                try:self.save_lessons()
                except Exception as e:print(f'[Adam] save_lessons failed: {e}',flush=True)
        return {'answer':ans,'tier':tier,'tokens':n,'wall_s':round(wall,3),'lessons_n':len(self.sem_lut._raw)}
    def save_lessons(self):
        if len(self.sem_lut._raw)==0:return
        self.lessons_path.parent.mkdir(parents=True,exist_ok=True)
        self.sem_lut.save(str(self.lessons_path).removesuffix('.npz'))
    def stats(self)->Dict[str,Any]:
        return {'lessons_n':len(self.sem_lut._raw),'auto_margin':self.sem_lut.auto_margin() if len(self.sem_lut._raw)>0 else None,'tier_counts':dict(self.adam._tier_counts),'token_counts':dict(self.adam._token_counts),'svc_boot_s':round(self.svc_boot_s,1)}
    def teach(self,q:str,a:str,source:str=''):
        self.sem_lut.add(q,a,source=source)
        try:self.sem_lut.fit();self.save_lessons()
        except Exception as _fe:print(f'[teach] fit/save deferred (lesson still queued): {_fe}',flush=True)
        return {'lessons_n':len(self.sem_lut._raw)}
    def _ensure_gated_bank(self):
        gb=getattr(self,'gated_bank',None)
        if gb is None:
            if self.svc is None or getattr(self.svc,'model',None) is None:raise RuntimeError('gated weight-learning needs the streaming model loaded')
            import os as _os
            from amni.learning.gated_pages import GatedPageBank
            self.gated_bank=GatedPageBank(self.svc.model,self.svc.tok,layers=list(range(16,40,2)),r=24,tau=0.28,sharp=22)
            rp=_os.path.join('bakes','reasoning_pages.pt')
            if _os.path.exists(rp):
                try:print(f'[gated] loaded {self.gated_bank.load(rp)} reasoning domains from {rp} — reasoning densification live',flush=True)
                except Exception as _re:print(f'[gated] reasoning-pages load skipped: {_re}',flush=True)
        return self.gated_bank
    def teach_weight(self,domain:str,facts:List[str],steps:int=420,lr:float=3e-4)->Dict[str,Any]:
        loss=self._ensure_gated_bank().add_domain(domain,list(facts),steps=steps,lr=lr)
        return {'domain':domain,'pages':len(facts),'final_loss':round(loss,3),'domains':list(self.gated_bank.domains)}
    def _persona_cache_key(self,system:str,message:str,history:Optional[List[Tuple[str,str]]],facts:Optional[List[str]])->str:
        h=hashlib.blake2b(digest_size=8)
        for u,a in (history or []):h.update(u.encode('utf-8','ignore'));h.update(b'\x1f');h.update(a.encode('utf-8','ignore'));h.update(b'\x1e')
        for f in (facts or []):h.update(f.encode('utf-8','ignore'));h.update(b'\x1d')
        return f'PERSONA::{system[:80]}::{h.hexdigest()}::{message}'
    def chat_persona_stream(self,message:str,system:str,max_new_tokens:int=120,do_sample:bool=True,history:Optional[List[Tuple[str,str]]]=None,facts:Optional[List[str]]=None,is_private:bool=False):
        bus=getattr(self,'bus',None)
        if bus is not None and not is_private:
            try:v,home,c=bus.recall(message)
            except Exception:v,home=None,''
            if v is not None and home=='tier0_atex_override':
                for chunk in [v[i:i+24] for i in range(0,len(v),24)]:yield chunk
                return
        ckey=self._persona_cache_key(system,message,history,facts)
        cached=self.adam.lut.lookup(ckey) if (hasattr(self.adam,'lut') and not is_private) else None
        if cached is not None and not (bus is not None and bus.is_suppressed(cached.get('a',''))):
            ans=cached.get('a','')
            for chunk in [ans[i:i+24] for i in range(0,len(ans),24)]:yield chunk
            return
        sl=getattr(self,'sem_lut',None)
        if sl is not None and not is_private and getattr(sl,'_raw',None):
            try:hit=sl.lookup_soft(message,k=1,cos_gate=float(os.environ.get('AMNI_RECALL_DIRECT_GATE','0.90')),margin=0.04)
            except Exception:hit=None
            if hit and not (bus is not None and bus.is_suppressed(hit)):
                for chunk in [hit[i:i+24] for i in range(0,len(hit),24)]:yield chunk
                return
        if self.svc is None:
            why=(self.runtime_error or 'StreamingChatService was None at server boot — check the server logs for the underlying exception')[:400]
            msg=f'[Adam streaming chat unavailable — the GF(17) backend failed to initialize at boot. Reason: {why}. Diagnostic: `python -c "from amni.runtime import fetch; fetch()"`. Most common cause: prebuilt amni_kernels .pyd is Python-version-specific (cp313 currently). Rebuild via `cd amni_kernels && pip install maturin && maturin develop --release`, then restart the server.]'
            for chunk in [msg[i:i+48] for i in range(0,len(msg),48)]:yield chunk
            return
        try:
            for chunk in self.svc.chat_stream(message,system=system,history=history,facts=facts,max_new_tokens=max_new_tokens,do_sample=do_sample,kb_top_k=int(os.environ.get('AMNI_PERSONA_KB_TOPK','4'))):yield chunk
        except Exception as e:yield f'[stream error: {e}]'
    def chat_persona(self,message:str,system:str,max_new_tokens:int=120,do_sample:bool=True,history:Optional[List[Tuple[str,str]]]=None,facts:Optional[List[str]]=None,is_private:bool=False)->Dict[str,Any]:
        t0=time.time()
        bus=getattr(self,'bus',None)
        if bus is not None and not is_private:
            try:
                v,home,c=bus.recall(message)
                if v is not None and home=='tier0_atex_override':return {'answer':v,'tier':home,'tokens':0,'wall_s':round(time.time()-t0,3)}
            except Exception:pass
        ckey=self._persona_cache_key(system,message,history,facts)
        cached=self.adam.lut.lookup(ckey) if (hasattr(self.adam,'lut') and not is_private) else None
        if cached is not None and not (bus is not None and bus.is_suppressed(cached.get('a',''))):return {'answer':cached.get('a'),'tier':'tier1_persona_lut','tokens':0,'wall_s':round(time.time()-t0,3)}
        sl=getattr(self,'sem_lut',None)
        if sl is not None and not is_private and getattr(sl,'_raw',None):
            try:
                hit=sl.lookup_soft(message,k=1,cos_gate=float(os.environ.get('AMNI_RECALL_DIRECT_GATE','0.90')),margin=0.04)
                if hit and not (bus is not None and bus.is_suppressed(hit)):return {'answer':hit,'tier':'tier1_sem_lut','tokens':0,'wall_s':round(time.time()-t0,3)}
            except Exception:pass
        if self.svc is None:return {'answer':None,'error':f'runtime not installed: {self.runtime_error}','tier':'runtime_missing','tokens':0,'wall_s':round(time.time()-t0,3)}
        try:resp,n=self.svc.chat(message,system=system,history=history,facts=facts,max_new_tokens=max_new_tokens,do_sample=do_sample,kb_top_k=int(os.environ.get('AMNI_PERSONA_KB_TOPK','4')))
        except Exception as e:return {'answer':None,'error':str(e),'tier':'persona_error','tokens':0,'wall_s':round(time.time()-t0,3)}
        ans=(resp or '').strip()
        try:
            if hasattr(self.adam,'lut') and ans and not is_private:self.adam.lut.store(ckey,ans,subject=None,source='persona',meta={'tokens':n,'system_len':len(system),'history_n':len(history or []),'is_private':False})
        except Exception:pass
        tier='tier_persona_hist' if history else 'tier_persona'
        return {'answer':ans,'tier':tier,'tokens':n,'wall_s':round(time.time()-t0,3),'is_private':is_private}
def select_model_bake(prefer='bakes/gemma4_12b_nvfp4_atex',min_free_gb=14.0):
    """Pick the NVFP4 12B as the server model only if it exists AND enough VRAM is free; else return (None,reason) so the caller keeps its lighter default bake. One server auto-fits the host: 16GB+ cards -> NVFP4 12B, smaller cards -> the light bake."""
    if not (Path(prefer)/'bake_manifest.json').exists():return None,'nvfp4 bake absent'
    try:
        import torch;free,_=torch.cuda.mem_get_info();fg=free/1e9
    except Exception as e:return None,f'no cuda ({str(e)[:40]})'
    return (prefer,f'{fg:.1f}GB free >= {min_free_gb}') if fg>=min_free_gb else (None,f'only {fg:.1f}GB free (< {min_free_gb})')
SEED_LESSONS=[
    ('What is 2 + 2?','4'),('What is 5 + 3?','8'),('What is 10 - 7?','3'),('What is 6 * 7?','42'),
    ('What is 9 * 8?','72'),('What is 100 / 4?','25'),('What is the square root of 81?','9'),
    ('What is the square root of 144?','12'),('What is 2 to the power 5?','32'),('What is 2 to the power 10?','1024'),
    ('What is the capital of France?','Paris'),('What is the capital of Japan?','Tokyo'),
    ('What is the capital of Italy?','Rome'),('What is the capital of Spain?','Madrid'),
    ('What is the capital of Germany?','Berlin'),('What is the capital of Russia?','Moscow'),
    ('What is the capital of China?','Beijing'),('What is the capital of Australia?','Canberra'),
    ('What is the capital of Canada?','Ottawa'),('What is the capital of Brazil?','Brasilia'),
    ('Who wrote Hamlet?','Shakespeare'),('Who wrote Romeo and Juliet?','Shakespeare'),
    ('Who painted the Mona Lisa?','Leonardo da Vinci'),('Who painted Starry Night?','Vincent van Gogh'),
    ('Who painted the Sistine Chapel?','Michelangelo'),('Who discovered penicillin?','Alexander Fleming'),
    ('Who proposed the theory of relativity?','Einstein'),('Who wrote 1984?','George Orwell'),
    ('Who is the author of Pride and Prejudice?','Jane Austen'),('Who is the first US president?','George Washington'),
    ('What is the chemical symbol for gold?','Au'),('What is the chemical symbol for silver?','Ag'),
    ('What is the chemical symbol for iron?','Fe'),('What is the chemical symbol for copper?','Cu'),
    ('What is the chemical symbol for water?','H2O'),('What is the chemical symbol for sodium?','Na'),
    ('What is the chemical symbol for potassium?','K'),('What is the chemical symbol for nitrogen?','N'),
    ('What is the chemical symbol for oxygen?','O'),('What is the chemical symbol for carbon?','C'),
    ('What is the largest planet in our solar system?','Jupiter'),
    ('What is the smallest planet in our solar system?','Mercury'),
    ('How many continents are there?','7'),('How many planets in the solar system?','8'),
    ('How many sides does a hexagon have?','6'),('How many sides does a pentagon have?','5'),
    ('How many sides does a triangle have?','3'),('How many sides does a square have?','4'),
    ('How many sides does an octagon have?','8'),('What is the smallest prime number?','2'),
    ('What is the speed of light in vacuum (m/s)?','299792458'),
    ('What is the boiling point of water in Celsius?','100'),
    ('What is the freezing point of water in Celsius?','0'),
    ('What color do you get by mixing red and blue?','Purple'),
    ('What color do you get by mixing yellow and blue?','Green'),
    ('What color do you get by mixing red and yellow?','Orange'),
]
