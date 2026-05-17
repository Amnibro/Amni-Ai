"""Adam facade — single import surface for the deployable model.
Usage:
  from amni.adam import Adam
  adam = Adam(bake='E:/Amni-Ai-Bakes/gemma4_e2b_it_gf17', model='E:/Amni-Ai-Models/gemma-4-E2B-it')
  result = adam.ask('What is 2 + 2?')
  print(result['answer'], result['tier'], result['tokens'])
Wraps AdamLoop with sensible defaults, persistent SemanticPTEXLUT, all tiers enabled.
"""
import os,json,time
from pathlib import Path
from typing import Optional,Dict,Any
class Adam:
    def __init__(self,bake:str,model:str,lessons_path:str='experiences/adam_lessons.npz',lut_root:str='experiences/adam_lut',budget_mb:int=8000,seed_lessons:Optional[list]=None,enable_crawler:bool=True):
        self.bake=bake;self.model=model;self.lessons_path=Path(lessons_path);self.lut_root=lut_root
        os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1')
        from amni.inference.streaming_chat import StreamingChatService
        from amni.inference.adam_loop import AdamLoop
        from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
        self._SemanticPTEXLUT=SemanticPTEXLUT
        t0=time.time()
        self.svc=StreamingChatService(bake,model,budget_mb=budget_mb)
        self.svc_boot_s=time.time()-t0
        self.crawler_plugin=None
        if enable_crawler:
            try:
                from amni.inference.web_crawler import CrawlerPlugin
                self.crawler_plugin=CrawlerPlugin(distiller_svc=self.svc,max_pages=2,distill_max_tokens=180)
                print(f'[Adam] crawler plugin enabled (CrawlerPlugin with {len(self.crawler_plugin.crawler.allow)} allowed domains)',flush=True)
            except Exception as e:print(f'[Adam] crawler init failed (web-learn will fallback): {e}',flush=True)
        if self.lessons_path.exists():
            try:self.sem_lut=SemanticPTEXLUT.load(str(self.lessons_path).removesuffix('.npz'))
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
    def teach(self,q:str,a:str):
        self.sem_lut.add(q,a)
        self.sem_lut.fit()
        self.save_lessons()
        return {'lessons_n':len(self.sem_lut._raw)}
    def chat_persona_stream(self,message:str,system:str,max_new_tokens:int=120,do_sample:bool=True):
        cached=self.adam.lut.lookup(f'PERSONA::{system[:80]}::{message}') if hasattr(self.adam,'lut') else None
        if cached is not None:
            ans=cached.get('a','')
            for chunk in [ans[i:i+24] for i in range(0,len(ans),24)]:yield chunk
            return
        try:
            for chunk in self.svc.chat_stream(message,system=system,max_new_tokens=max_new_tokens,do_sample=do_sample,kb_top_k=0):yield chunk
        except Exception as e:yield f'[stream error: {e}]'
    def chat_persona(self,message:str,system:str,max_new_tokens:int=120,do_sample:bool=True)->Dict[str,Any]:
        t0=time.time()
        cached=self.adam.lut.lookup(f'PERSONA::{system[:80]}::{message}') if hasattr(self.adam,'lut') else None
        if cached is not None:return {'answer':cached.get('a'),'tier':'tier1_persona_lut','tokens':0,'wall_s':round(time.time()-t0,3)}
        try:resp,n=self.svc.chat(message,system=system,max_new_tokens=max_new_tokens,do_sample=do_sample,kb_top_k=0)
        except Exception as e:return {'answer':None,'error':str(e),'tier':'persona_error','tokens':0,'wall_s':round(time.time()-t0,3)}
        ans=(resp or '').strip()
        try:
            if hasattr(self.adam,'lut') and ans:self.adam.lut.store(f'PERSONA::{system[:80]}::{message}',ans,subject=None,source='persona',meta={'tokens':n,'system_len':len(system)})
        except Exception:pass
        return {'answer':ans,'tier':'tier_persona','tokens':n,'wall_s':round(time.time()-t0,3)}
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
