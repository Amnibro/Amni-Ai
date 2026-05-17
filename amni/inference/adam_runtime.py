import os,time,json,numpy as np,torch
from pathlib import Path
from typing import Iterator,Tuple,Dict,Optional,Any
try:from amni.inference import _torch_dist_shim as _shim
except ImportError:_shim=None
try:
    from amni.inference import triton_sdpa_patch as _tsp
    if os.environ.get('AMNI_DISABLE_TRITON_SDPA','0')!='1':_tsp.enable()
except Exception as _e:print(f'[adam_runtime] triton_sdpa_patch unavailable: {_e}',flush=True)
_ROOT=Path(__file__).resolve().parents[2]
BACKBONES={
    'qwen2.5-0.5b':{'kind':'qwen_torch','snapshot':_ROOT/'downloaded_models'/'models--Qwen--Qwen2.5-0.5B'/'snapshots'/'060db6499f32faf8b98477b0a26969ef7d8b9987','label':'Qwen2.5-0.5B (fp16)','native_gf17':False,'precision':'fp16','desc':'fp16 torch baseline — small, fast'},
    'qwen3.5-9b-fp16':{'kind':'qwen_torch','snapshot':_ROOT/'models'/'Qwen3.5-9B','label':'Qwen 3.5-9B (fp16)','native_gf17':False,'precision':'fp16','desc':'Qwen 3.5-9B in fp16 — full Adam testbed (download via scripts/download_qwen35_9b.py)'},
    'adam-small-qwen':{'kind':'adam_native','config':'adam-small','tex_root':_ROOT/'models'/'adam-small-qwen-distilled','prism_dir':_ROOT/'models'/'adam-small-qwen-distilled'/'prism_out','tokenizer_dir':_ROOT/'downloaded_models'/'models--Qwen--Qwen2.5-0.5B'/'snapshots'/'060db6499f32faf8b98477b0a26969ef7d8b9987','vocab':151936,'label':'adam-small (GF17, distilled)','native_gf17':True,'precision':'gf17','desc':'Real GF(17)-native Adam, distilled from Qwen2.5-0.5B (256 hidden, 6 blocks)'},
    'adam-qwen35-9b':{'kind':'adam_native','config':'qwen35-9b','tex_root':_ROOT/'models'/'adam-qwen35-9b','prism_dir':_ROOT/'models'/'adam-qwen35-9b'/'prism_out','tokenizer_dir':_ROOT/'models'/'Qwen3.5-9B','vocab':248320,'label':'adam-qwen35-9b (GF17, target)','native_gf17':True,'precision':'gf17','desc':'GF(17)-native Adam sized for Qwen 3.5-9B (4096 hidden, 32 blocks, head_dim=256, vocab=248320) — distill from Jackrong/Qwen3.5-9B'},
    'qwen35-9b-rtier':{'kind':'qwen35_rtier','rtier_root':_ROOT/'hf_cache'/'qwen35_9b_rtier','tokenizer_dir':_ROOT/'models'/'Qwen3.5-9B','label':'Qwen 3.5-9B (lossless rtier reconstructed)','native_gf17':False,'precision':'gf17_lossless','desc':'Lossless GF(17) Reffelt 4-tier baked weights → fp16 reconstruction → standard transformers inference (proves bake correctness, cos=1.0 vs raw fp16)'},
}
def _adam_weights_present(info:dict)->bool:
    pd=info.get('prism_dir');tr=info.get('tex_root')
    if pd and pd.exists() and (pd/'model_gf17.npz').exists():return True
    if pd and pd.exists() and (pd/'embed.npy').exists() and (pd/'head.npy').exists():return True
    if tr and tr.exists():
        if (tr/(info['config']+'.npz')).exists():return True
        if (tr/'meta.json').exists():return True
    return False
def list_backbones()->list:
    out=[]
    for name,info in BACKBONES.items():
        st='ready'
        if info['kind']=='qwen_torch':st='ready' if info['snapshot'].exists() else 'missing_weights'
        elif info['kind']=='adam_native':st='ready' if _adam_weights_present(info) else 'needs_distill'
        elif info['kind']=='qwen35_rtier':
            rt=info['rtier_root']
            n=len(list(rt.glob('*/meta.json'))) if rt.exists() else 0
            st='ready' if n>=300 else f'incomplete_bake:{n}'
        out.append({'name':name,'label':info['label'],'desc':info['desc'],'native_gf17':info['native_gf17'],'status':st,'kind':info['kind']})
    return out
class AdamRuntime:
    __slots__=('name','info','model','tok','kind','status','_tok_cls','_loaded_at')
    def __init__(self,name:str):
        if name not in BACKBONES:raise ValueError(f'unknown backbone: {name}. options: {list(BACKBONES.keys())}')
        self.name=name;self.info=BACKBONES[name];self.kind=self.info['kind'];self.model=None;self.tok=None;self.status='lazy';self._loaded_at=0.0
    def load(self)->Tuple[bool,str]:
        if self.model is not None:return True,'already_loaded'
        try:
            from transformers import AutoTokenizer
            if self.kind=='qwen35_rtier':
                from amni.inference.qwen35_rtier_runtime import Qwen35RtierRuntime
                rt=Qwen35RtierRuntime(rtier_root=self.info['rtier_root'],tok_dir=self.info['tokenizer_dir'])
                ok,st=rt.load()
                if ok:self.model=rt;self.tok=rt.tok;self.status=st;self._loaded_at=time.time();return True,st
                self.status=st;return False,st
            if self.kind=='qwen_torch':
                from transformers import AutoModelForCausalLM
                snap=self.info['snapshot']
                if not snap.exists():self.status=f'missing_weights:{snap}';return False,self.status
                self.tok=AutoTokenizer.from_pretrained(str(snap))
                self.model=AutoModelForCausalLM.from_pretrained(str(snap),torch_dtype=torch.float16,low_cpu_mem_usage=True).cuda().eval()
                self.status='ready';self._loaded_at=time.time();return True,'ready'
            from amni.model.adam import AdamModel,ADAM_CONFIGS
            cfg=self.info['config']
            if cfg not in ADAM_CONFIGS:self.status=f'unknown_config:{cfg}';return False,self.status
            tdir=self.info['tokenizer_dir']
            self.tok=AutoTokenizer.from_pretrained(str(tdir)) if tdir.exists() else None
            vocab=self.info.get('vocab',ADAM_CONFIGS[cfg].get('vocab',17))
            self.model=AdamModel(config_name=cfg,vocab=vocab,tex_root=self.info['tex_root'],auto_load=False)
            loaded_path=self._load_distilled_weights()
            if loaded_path:self.model._loaded_from=loaded_path
            self.status='ready' if self.model._loaded_from else 'loaded_random_init'
            self._loaded_at=time.time();return True,self.status
        except Exception as e:
            import traceback;traceback.print_exc()
            self.status=f'load_failed:{type(e).__name__}:{str(e)[:200]}';return False,self.status
    def _load_distilled_weights(self)->Optional[str]:
        pd=self.info.get('prism_dir')
        if not pd or not pd.exists():return None
        npz=pd/'model_gf17.npz'
        if npz.exists():
            try:self.model.load_gf17(str(npz));return str(npz)
            except Exception as e:print(f'[adam_runtime] load_gf17 failed: {e}',flush=True)
        emb=pd/'embed.npy';hd=pd/'head.npy'
        if emb.exists() and hd.exists():
            try:
                e=np.load(str(emb));h=np.load(str(hd))
                if e.shape==self.model.embed.shape:self.model.embed[:]=e
                if h.shape==self.model.head.shape:self.model.head[:]=h
                for li in range(self.model.n_blocks):
                    bdir=pd/f'block_{li:02d}'
                    if not bdir.exists():continue
                    blk=self.model.blocks[li]
                    for fn,attr in [('q','q_proj'),('k','k_proj'),('v','v_proj'),('o','o_proj')]:
                        p=bdir/f'{fn}.npy'
                        if p.exists():
                            w=np.load(str(p));tw=getattr(blk.attn,attr).w
                            if tw.shape==w.shape:tw[:]=w
                    for fn,attr in [('g','gate'),('u','up'),('d','down')]:
                        p=bdir/f'{fn}.npy'
                        if p.exists():
                            w=np.load(str(p));tw=getattr(blk.mlp,attr).w
                            if tw.shape==w.shape:tw[:]=w
                return str(pd)
            except Exception as e:print(f'[adam_runtime] direct npy load failed: {e}',flush=True)
        return None
    def unload(self):
        self.model=None;self.tok=None;self.status='unloaded'
        try:torch.cuda.empty_cache()
        except Exception:pass
    def is_native_gf17(self)->bool:return self.info['native_gf17']
    def param_summary(self)->Dict[str,Any]:
        if self.kind=='qwen35_rtier':
            return self.model.param_summary() if self.model is not None else {'kind':'qwen35_rtier','loaded':False,'status':self.status}
        if self.kind=='qwen_torch':
            if self.model is None:return {'kind':'qwen_torch','loaded':False}
            c=self.model.config
            return {'kind':'qwen_torch','loaded':True,'hidden':c.hidden_size,'n_layers':c.num_hidden_layers,'n_heads':c.num_attention_heads,'n_kv_heads':getattr(c,'num_key_value_heads',c.num_attention_heads),'vocab':c.vocab_size,'native_gf17':False,'mode':'fp16','status':self.status}
        if self.model is None:return {'kind':'adam_native','loaded':False,'status':self.status}
        m=self.model;total=m.vocab*m.hidden*2+sum(m._block_param_count(i) for i in range(m.n_blocks))
        return {'kind':'adam_native','loaded':True,'config':m.config_name,'hidden':m.hidden,'n_layers':m.n_blocks,'n_heads':m.n_heads,'n_kv_heads':m.n_kv_heads,'inter':m.inter,'vocab':m.vocab,'head_dim':m._head_dim,'total_params':int(total),'native_gf17':True,'mode':m._mode,'streaming':m._streaming,'loaded_from':m._loaded_from,'status':self.status}
    def generate_iter(self,prompt:str,max_new:int=32,max_prompt:int=64,greedy:bool=True)->Iterator[Tuple[str,Dict]]:
        if self.model is None:
            ok,_=self.load()
            if not ok:yield('error',{'msg':self.status});return
        if self.tok is None:yield('error',{'msg':'no tokenizer available for backbone'});return
        if self.kind=='qwen35_rtier':yield from self.model.generate_iter(prompt,max_new=max_new,max_prompt=max_prompt,greedy=greedy);return
        if self.kind=='qwen_torch':yield from self._gen_qwen(prompt,max_new,max_prompt,greedy);return
        yield from self._gen_adam(prompt,max_new,max_prompt,greedy)
    def _gen_qwen(self,prompt,max_new,max_prompt,greedy):
        ids=self.tok(prompt,return_tensors='pt').input_ids
        if ids.shape[1]>max_prompt:ids=ids[:,:max_prompt]
        ids=ids.cuda();T_in=int(ids.shape[1])
        yield('prompt_done',{'t_in':T_in,'decoded':self.tok.decode(ids[0],skip_special_tokens=False)})
        gen=[];kv=None;cur=ids;eos=self.tok.eos_token_id
        with torch.no_grad():
            for step in range(max_new):
                t0=time.perf_counter()
                out=self.model(input_ids=cur,past_key_values=kv,use_cache=True);kv=out.past_key_values
                logits=out.logits[:,-1,:]
                cur=(logits.argmax(dim=-1,keepdim=True).long() if greedy else torch.multinomial(torch.softmax(logits,dim=-1),1))
                nid=int(cur.item());gen.append(nid)
                dt=time.perf_counter()-t0
                yield('token',{'step':step,'token_id':nid,'text':self.tok.decode([nid],skip_special_tokens=False),'wall_s':dt})
                if eos is not None and nid==eos:break
        yield('done',{'gen_ids':gen,'gen_text':self.tok.decode(gen,skip_special_tokens=False)})
    def _gen_adam(self,prompt,max_new,max_prompt,greedy):
        if self.model._streaming and not self.model._loaded_from:
            yield('error',{'msg':f'adam-{self.info["config"]} requires distilled weights — call /api/adam/distill_from_qwen'});return
        ids=self.tok(prompt,return_tensors='pt').input_ids[0].numpy().astype(np.int64)
        if len(ids)>max_prompt:ids=ids[:max_prompt]
        T_in=int(len(ids))
        yield('prompt_done',{'t_in':T_in,'decoded':self.tok.decode(ids.tolist(),skip_special_tokens=False)})
        gen=[]
        try:
            self.model.clear_cache()
            t0=time.perf_counter()
            logits=self.model.forward_cached(ids[np.newaxis])
            nid=int(np.argmax(logits[0]))
            gen.append(nid)
            yield('token',{'step':0,'token_id':nid,'text':self.tok.decode([nid],skip_special_tokens=False),'wall_s':time.perf_counter()-t0})
            for step in range(1,max_new):
                t0=time.perf_counter();nid=self.model.forward_next_cached(gen[-1]);gen.append(nid)
                yield('token',{'step':step,'token_id':nid,'text':self.tok.decode([nid],skip_special_tokens=False),'wall_s':time.perf_counter()-t0})
                if self.tok.eos_token_id is not None and nid==self.tok.eos_token_id:break
            self.model.clear_cache()
        except Exception as e:
            import traceback;traceback.print_exc()
            yield('error',{'msg':f'{type(e).__name__}:{str(e)[:200]}'});return
        yield('done',{'gen_ids':gen,'gen_text':self.tok.decode(gen,skip_special_tokens=False)})
