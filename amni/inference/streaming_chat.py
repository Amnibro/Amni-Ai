import os,sys,torch
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1')
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(_ROOT))
try:
    from amni.inference.streaming_linear import TensorRegistry,swap_modules,materialize_remaining_params,install_prefetch_chain
    _RUNTIME_AVAILABLE=True
except ImportError:
    TensorRegistry=swap_modules=materialize_remaining_params=install_prefetch_chain=None
    _RUNTIME_AVAILABLE=False
from accelerate import init_empty_weights
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
class _RuntimeBlobMissing(RuntimeError):
    """Raised when StreamingChatService is instantiated without the Reffelt runtime blob.
    Fix: install the runtime via `from amni.runtime import fetch; fetch(license_key='free-noncommercial')`
    See https://amni-scient.com/amni-ai for the encrypted-blob install path."""
_GDN_ARCHS=('Qwen3_5ForCausalLM','Qwen3_5MoeForCausalLM','MiniMaxText01ForCausalLM','Qwen3CoderNextForCausalLM')
_MULTIMODAL_PREFIXES=('model.language_model.',)
class StreamingChatService:
    def __init__(self,bake_dir,model_path,budget_mb=4000,device=None,lmhead_full=True,enable_prefetch=True,prefetch_horizon=6,pin_embed=True):
        if not _RUNTIME_AVAILABLE:
            raise _RuntimeBlobMissing("Reffelt runtime blob not installed. The public Amni-Ai source skeleton ships without the GF(17) streaming backend. To enable chat:\n  python -c \"from amni.runtime import fetch; fetch(license_key='free-noncommercial')\"\n\nSee https://github.com/Amnibro/Amni-Ai/blob/main/docs/INSTALL.md for current install paths. Until the runtime is fetched, only /healthz + /stats + cached lesson-bank LUT hits work.")
        if device is None:
            if torch.cuda.is_available():device='cuda'
            elif hasattr(torch,'xpu') and torch.xpu.is_available():device='xpu'
            else:device='cpu';print('[StreamingChatService] WARNING: no GPU detected, falling back to CPU (~1 tok/s). Install ROCm (AMD) or CUDA (NVIDIA) torch wheel: see https://pytorch.org/get-started/locally/',flush=True)
        self.tok=AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:self.tok.pad_token=self.tok.eos_token
        cfg=AutoConfig.from_pretrained(model_path)
        archs=tuple(getattr(cfg,'architectures',None) or [])
        is_gdn=any(a in _GDN_ARCHS for a in archs)
        if is_gdn:
            from amni.inference import triton_gdn_patch
            triton_gdn_patch.apply()
        text_cfg=cfg.text_config if hasattr(cfg,'text_config') else cfg
        try:text_cfg._attn_implementation='eager'
        except Exception:pass
        with init_empty_weights():m=AutoModelForCausalLM.from_config(text_cfg,attn_implementation='eager')
        m=m.to(dtype=torch.bfloat16)
        if is_gdn:
            from amni.inference import triton_gdn_patch
            triton_gdn_patch.reattach_to_model(m)
        self.registry=TensorRegistry(bake_dir,budget_mb*1024*1024,device,enable_prefetch=enable_prefetch)
        ts=self.registry.manifest['tensors']
        for o in list(ts.keys()):
            for pfx in _MULTIMODAL_PREFIXES:
                if o.startswith(pfx):
                    a='model.'+o[len(pfx):]
                    if a not in ts:ts[a]=ts[o]
        if getattr(text_cfg,'tie_word_embeddings',False):
            for src in ('model.embed_tokens.weight','model.language_model.embed_tokens.weight'):
                if src in ts and 'lm_head.weight' not in ts:ts['lm_head.weight']=ts[src];break
        swap_modules(m,self.registry,lmhead_tile_rows=999999 if lmhead_full else 4096)
        materialize_remaining_params(m,self.registry,device)
        _rope_reinit_done=set()
        for name,buf in list(m.named_buffers()):
            if buf.is_meta or buf.device!=torch.device(device):
                pp,_,leaf=name.rpartition('.')
                parent=m.get_submodule(pp) if pp else m
                handled=False
                if name in self.registry.manifest['tensors']:
                    try:
                        t=self.registry.get_full(name).to(device=device,dtype=buf.dtype)
                        parent.register_buffer(leaf,t,persistent=False);handled=True
                    except Exception:pass
                if not handled and buf.is_meta and 'inv_freq' in leaf and hasattr(parent,'rope_init_fn') and hasattr(parent,'config'):
                    try:
                        inv,scale=parent.rope_init_fn(parent.config,device)
                        parent.register_buffer('inv_freq',inv.to(device),persistent=False)
                        parent.attention_scaling=scale;handled=True
                    except Exception:pass
                if not handled and buf.is_meta and 'inv_freq' in leaf and pp not in _rope_reinit_done:
                    try:
                        cls=type(parent);rcfg=getattr(parent,'config',text_cfg)
                        fresh=cls(rcfg)
                        for bn,bv in fresh.named_buffers():
                            try:parent.register_buffer(bn,bv.to(device=device,dtype=torch.float32),persistent=False)
                            except Exception:setattr(parent,bn,bv.to(device=device,dtype=torch.float32))
                        for an in ('full_attention_attention_scaling','sliding_attention_attention_scaling','attention_scaling'):
                            if hasattr(fresh,an):setattr(parent,an,getattr(fresh,an))
                        _rope_reinit_done.add(pp);handled=True
                    except Exception as ex:print(f'  rope-reinit failed for {pp}: {ex}',flush=True)
                if not handled:
                    b2=getattr(parent,leaf)
                    if b2.is_meta:b2=torch.zeros(buf.shape,dtype=buf.dtype,device=device)
                    else:b2=b2.to(device)
                    try:parent.register_buffer(leaf,b2,persistent=False)
                    except Exception:setattr(parent,leaf,b2)
        if is_gdn:
            from amni.inference import triton_gdn_patch
            triton_gdn_patch.reattach_to_model(m)
        n_chained=install_prefetch_chain(m,horizon=prefetch_horizon) if enable_prefetch else 0
        if pin_embed:
            embed_key='model.embed_tokens.weight'
            if embed_key in self.registry.manifest['tensors']:
                self.registry.pin(embed_key)
                self.registry.get_full(embed_key)
        m.eval();self.model=m;self.device=device;self.n_chained=n_chained;self.is_gdn=is_gdn;self.architectures=archs
    def generate_text(self,prompt,max_new_tokens=80,do_sample=False,temperature=1.0):
        enc=self.tok(prompt,return_tensors='pt').to(self.device)
        with torch.no_grad():
            ids=self.model.generate(input_ids=enc.input_ids,attention_mask=enc.attention_mask,max_new_tokens=max_new_tokens,do_sample=do_sample,temperature=temperature if do_sample else 1.0,pad_token_id=self.tok.pad_token_id)
        new_ids=ids[0,enc.input_ids.shape[1]:]
        return self.tok.decode(new_ids,skip_special_tokens=True),int(new_ids.shape[0])
    def generate_stream(self,prompt,max_new_tokens=80,do_sample=False,temperature=1.0):
        from transformers import TextIteratorStreamer
        from threading import Thread
        enc=self.tok(prompt,return_tensors='pt').to(self.device)
        streamer=TextIteratorStreamer(self.tok,skip_prompt=True,skip_special_tokens=True,timeout=300.0)
        gen_kw=dict(input_ids=enc.input_ids,attention_mask=enc.attention_mask,max_new_tokens=max_new_tokens,do_sample=do_sample,temperature=temperature if do_sample else 1.0,pad_token_id=self.tok.pad_token_id,streamer=streamer)
        t=Thread(target=self._safe_generate,args=(gen_kw,))
        t.daemon=True;t.start()
        for chunk in streamer:
            if chunk:yield chunk
        t.join(timeout=5.0)
    def _safe_generate(self,gen_kw):
        try:
            with torch.no_grad():self.model.generate(**gen_kw)
        except Exception as e:print(f'[streaming_chat] generate_stream worker error: {e}',flush=True)
    def chat_stream(self,user_msg,system=None,history=None,facts=None,max_new_tokens=120,do_sample=False,kb_top_k=0,kb_max_chars_per=600):
        kb_block=self._kb_context(user_msg,kb_top_k,kb_max_chars_per) if kb_top_k>0 else None
        prompt=self._build_prompt(user_msg,system,history,facts,kb_block=kb_block)
        yield from self.generate_stream(prompt,max_new_tokens=max_new_tokens,do_sample=do_sample)
    def _resolve_subject(self,user_msg,subject):
        if subject!='auto':return subject
        if not hasattr(self,'_classifier'):
            from amni.learning.subject_classifier import SubjectClassifier
            self._classifier=SubjectClassifier()
        chosen,score,scores=self._classifier.classify_with_confidence(user_msg)
        return chosen
    def _make_retriever(self,kb_root):
        from amni.inference.kb_retriever import KBRetriever
        use_emb=os.environ.get('AMNI_RETRIEVER','').lower()=='embedding'
        if use_emb:
            try:
                from amni.inference.embedding_cosine_retriever import EmbeddingCosineRetriever
                return EmbeddingCosineRetriever(kb_root,device=os.environ.get('AMNI_RETRIEVER_DEVICE','cpu'))
            except Exception as e:print(f'  [warn] embedding retriever failed for {kb_root}: {e}; falling back',flush=True)
        if (Path(kb_root)/'nonce_index.ptex').exists():
            from amni.inference.nonce_kb_retriever import NonceKBRetriever
            return NonceKBRetriever(kb_root)
        return KBRetriever(kb_root)
    def attach_kb(self,kb_root,skip_subjects=None):
        self._kb=self._make_retriever(kb_root)
        self._kb_skip_subjects=set(skip_subjects or ())
        return self._kb.stats()
    def attach_subject_kbs(self,subject_to_kb_root,skip_subjects=None):
        self._subject_kbs={s:self._make_retriever(r) for s,r in subject_to_kb_root.items() if r}
        self._kb_skip_subjects=set(skip_subjects or ())
        return {s:k.kb.stats() for s,k in self._subject_kbs.items()}
    def _route_kb_for_query(self,user_msg,subject_min_score=0):
        if not hasattr(self,'_classifier'):
            from amni.learning.subject_classifier import SubjectClassifier
            self._classifier=SubjectClassifier()
        subj,score,_=self._classifier.classify_with_confidence(user_msg)
        if subject_min_score>0 and score<subject_min_score:return None,subj
        skips=getattr(self,'_kb_skip_subjects',set())
        if subj in skips:return None,subj
        if hasattr(self,'_subject_kbs') and self._subject_kbs:
            kb=self._subject_kbs.get(subj) or self._subject_kbs.get('global')
            return kb,subj
        return getattr(self,'_kb',None),subj
    def _kb_context(self,user_msg,kb_top_k,kb_max_chars_per,kb_min_top_score=0,subject_min_score=0):
        kb,subj=self._route_kb_for_query(user_msg,subject_min_score=subject_min_score) if (hasattr(self,'_kb_skip_subjects') or hasattr(self,'_subject_kbs')) else (getattr(self,'_kb',None),None)
        if kb is None:return None
        results=kb.retrieve(user_msg,k=kb_top_k,max_chars_per=kb_max_chars_per)
        if not results:return None
        top=max((r[2] for r in results),default=0)
        return kb.format_as_context(results) if top>=kb_min_top_score else None
    def _build_prompt(self,user_msg,system,history,facts,kb_block):
        sys_parts=[system] if system else []
        if kb_block:sys_parts.append(kb_block)
        if facts:
            sys_parts.append('\nRelevant facts (authoritative):')
            for f in facts:sys_parts.append(f'- {f.strip()[:240]}')
        sys_text='\n'.join(sys_parts) if sys_parts else None
        msgs=[]
        if sys_text:msgs.append({'role':'system','content':sys_text})
        for u,a in (history or [])[-4:]:
            msgs.append({'role':'user','content':u})
            msgs.append({'role':'assistant','content':a})
        msgs.append({'role':'user','content':user_msg})
        return self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    def _next_token_top_prob(self,prompt):
        enc=self.tok(prompt,return_tensors='pt').to(self.device)
        with torch.no_grad():
            out=self.model(input_ids=enc.input_ids,attention_mask=enc.attention_mask)
        logits=out.logits[0,-1].float()
        probs=torch.softmax(logits,dim=-1)
        return float(probs.max().item())
    def attach_session_writer(self,session_root,confidence_threshold=0.6):
        from amni.learning.session_atex_writer import SessionATEXWriter
        self._session_writer=SessionATEXWriter(session_root,confidence_threshold=confidence_threshold)
        return self._session_writer
    def chat(self,user_msg,system=None,history=None,facts=None,max_new_tokens=80,do_sample=False,subject=None,kb_top_k=3,kb_max_chars_per=600,kb_min_top_score=0,subject_min_score=0,kb_skip_if_conf=0.0,cache_writeback=True,writeback_min_conf=0.6):
        if subject is not None:
            resolved=self._resolve_subject(user_msg,subject)
            self.registry.set_active_subjects((resolved,) if resolved!='global' else ('global',))
        skip_kb=False
        top_prob=None
        if kb_skip_if_conf>0 and kb_top_k>0:
            prompt_no_kb=self._build_prompt(user_msg,system,history,facts,kb_block=None)
            top_prob=self._next_token_top_prob(prompt_no_kb)
            if top_prob>=kb_skip_if_conf:skip_kb=True
        kb_block=None if skip_kb else self._kb_context(user_msg,kb_top_k,kb_max_chars_per,kb_min_top_score=kb_min_top_score,subject_min_score=subject_min_score)
        prompt=self._build_prompt(user_msg,system,history,facts,kb_block=kb_block)
        reply,n_tok=self.generate_text(prompt,max_new_tokens=max_new_tokens,do_sample=do_sample)
        if cache_writeback and getattr(self,'_session_writer',None) is not None and reply:
            if top_prob is None:
                try:top_prob=self._next_token_top_prob(prompt)
                except Exception:top_prob=writeback_min_conf
            if top_prob>=writeback_min_conf:
                try:self._session_writer.write(user_msg,reply,float(top_prob),meta={'kind':'chat_auto','tokens':n_tok})
                except Exception:pass
        return reply,n_tok
