"""CompositionalDecoder v0.3: template-fetch (prefix OR cosine) + block-fetch + refine.
Wraps StreamingChatService.generate_text to inject KB-matched function/block templates mid-generation.
Per the maintainer's vision: 1.5B Adam behaves like 7B by skipping boilerplate generation via LUT lookups.
v0.3: dual-matcher — string-prefix matcher fires on exact-prefix hits (cheap, deterministic); embedding-cosine matcher fires on semantic similarity when prefix misses (recall safety net).
"""
import torch
from amni.inference.function_matcher import FunctionPrefixMatcher
class CompositionalDecoder:
    def __init__(self,svc,kb_root='E:/Amni-Ai-KB/personal_functions',min_match_chars=20,max_template_chars=600,refine_after_inject=8,use_embedding=True,emb_min_cos=0.45,semantic_window_chars=200):
        self.svc=svc
        self.matcher=FunctionPrefixMatcher(kb_root,min_prefix_len=min_match_chars,max_prefix_len=80)
        self.max_template_chars=max_template_chars
        self.refine_after_inject=refine_after_inject
        self._injects=0
        self._tokens_saved=0
        self._prefix_injects=0
        self._semantic_injects=0
        self.semantic_window_chars=semantic_window_chars
        self.emb_min_cos=emb_min_cos
        self._emb=None
        if use_embedding:
            try:
                from amni.inference.embedding_cosine_retriever import EmbeddingCosineRetriever
                self._emb=EmbeddingCosineRetriever(kb_root,device='cpu')
            except Exception as e:print(f'  [warn] embedding matcher disabled: {e}',flush=True)
    def stats(self):return {'injects':self._injects,'tokens_saved_estimate':self._tokens_saved,'prefix_injects':self._prefix_injects,'semantic_injects':self._semantic_injects,'matcher':self.matcher.stats()}
    def generate(self,prompt,max_new_tokens=200,do_sample=False,window_chars=120,inject_cooldown_tokens=20,use_chat_template=True,system='You are a coding assistant.'):
        tok=self.svc.tok;m=self.svc.model;dev=self.svc.device
        if use_chat_template and getattr(tok,'chat_template',None):
            msgs=[{'role':'user','content':prompt}] if not system else [{'role':'system','content':system},{'role':'user','content':prompt}]
            try:p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            except Exception:p=prompt
        else:p=prompt
        enc=tok(p,return_tensors='pt').to(dev)
        ids=enc.input_ids
        gen_ids=ids.clone()
        n_new=0;cooldown=0
        prompt_len=ids.shape[1]
        eos=tok.eos_token_id
        while n_new<max_new_tokens:
            with torch.no_grad():
                out=m(input_ids=gen_ids[:,-min(gen_ids.shape[1],2048):])
                logits=out.logits[0,-1]
                next_id=int(logits.argmax().item()) if not do_sample else int(torch.multinomial(torch.softmax(logits,dim=-1),1).item())
            gen_ids=torch.cat([gen_ids,torch.tensor([[next_id]],device=dev)],dim=1)
            n_new+=1
            if eos is not None and next_id==eos:break
            if cooldown>0:cooldown-=1;continue
            tail_ids=gen_ids[0,-min(gen_ids.shape[1]-prompt_len,window_chars//2):]
            if tail_ids.numel()<6:continue
            tail_text=tok.decode(tail_ids,skip_special_tokens=True)
            match=self.matcher.find_match(tail_text)
            inject_kind='prefix' if match else None
            if not match and self._emb is not None and len(tail_text)>=20:
                sem_window_ids=gen_ids[0,-min(gen_ids.shape[1]-prompt_len,self.semantic_window_chars):]
                sem_text=tok.decode(sem_window_ids,skip_special_tokens=True)
                results=self._emb.retrieve(sem_text,k=1,min_score=self.emb_min_cos,max_chars_per=self.max_template_chars)
                if results:
                    key,content,cos=results[0]
                    match={'content':content,'key':key};inject_kind='semantic'
            if not match:continue
            template=match['content'][:self.max_template_chars]
            already=tail_text.lower().rfind(match['content'][:30].lower().strip())
            if already>=0:tail_template=template[len(template[:30])-(len(tail_text)-already):]
            else:tail_template=template
            if not tail_template or len(tail_template)<5:continue
            tmpl_ids=tok(tail_template,return_tensors='pt',add_special_tokens=False).input_ids.to(dev)
            gen_ids=torch.cat([gen_ids,tmpl_ids],dim=1)
            n_added=int(tmpl_ids.shape[1])
            n_new+=n_added
            self._injects+=1
            self._tokens_saved+=n_added
            if inject_kind=='prefix':self._prefix_injects+=1
            else:self._semantic_injects+=1
            cooldown=inject_cooldown_tokens
        new_tokens=gen_ids[0,prompt_len:]
        return tok.decode(new_tokens,skip_special_tokens=True),int(new_tokens.shape[0])
