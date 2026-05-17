_ROLE_PROMPTS={
    'left_brain':'You are the Left Brain Catalyst. Push unfiltered logic and algorithmic precision. Be analytical and exact.',
    'right_brain':'You are the Right Brain Refiner. Ensure structural elegance, creative solutions, and safety. Think laterally.',
    'cerebellum':'You are the Cerebellum Integrator. Synthesize the left and right perspectives into a unified, balanced answer.',
    'oracle':'You are a knowledge oracle. Provide accurate, well-reasoned answers grounded in facts.',
    'fast':'Answer briefly and directly with your first instinct (System 1 / fast).',
    'slow':'Think step-by-step. Reason carefully before answering (System 2 / slow).',
    'grail_verify':'You are a truth-seeking verifier. Given a statement, identify factual errors or unsupported claims. Return: VALID or list issues.',
}
def _gen_text(rt,prompt,mp,mn,greedy=True):
    gen=[];txt=''
    for evt,dd in rt.generate_iter(prompt,max_new=mn,max_prompt=mp,greedy=greedy):
        if evt=='token':gen.append(dd['token_id'])
        elif evt=='done':txt=dd['gen_text']
        elif evt=='error':return [],f'[error: {dd["msg"]}]'
    return gen,txt
def _gen_batched(rt,prompts,mp,mn,greedy=True):
    if rt.kind!='qwen_torch' or rt.model is None:return [_gen_text(rt,p,mp,mn,greedy)[1] for p in prompts]
    import torch
    tok=rt.tok;model=rt.model;B=len(prompts)
    enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=mp)
    ids=enc.input_ids.cuda();attn=enc.attention_mask.cuda()
    eos=tok.eos_token_id;done=[False]*B;outs=[[]for _ in range(B)]
    with torch.no_grad():
        out=model(input_ids=ids,attention_mask=attn,use_cache=True);kv=out.past_key_values
        cur=out.logits[:,-1,:].argmax(dim=-1,keepdim=True).long()
        new_attn=torch.cat([attn,torch.ones(B,1,dtype=attn.dtype,device=attn.device)],dim=1)
        for i in range(B):outs[i].append(int(cur[i,0].item()));done[i]=(eos is not None and outs[i][-1]==eos)
        for step in range(mn-1):
            if all(done):break
            out=model(input_ids=cur,past_key_values=kv,attention_mask=new_attn,use_cache=True);kv=out.past_key_values
            cur=out.logits[:,-1,:].argmax(dim=-1,keepdim=True).long()
            new_attn=torch.cat([new_attn,torch.ones(B,1,dtype=attn.dtype,device=attn.device)],dim=1)
            for i in range(B):
                if not done[i]:
                    nid=int(cur[i,0].item());outs[i].append(nid)
                    if eos is not None and nid==eos:done[i]=True
    return [tok.decode(o,skip_special_tokens=False) for o in outs]
def role_query(rt,role:str,topic:str,response:str,mp:int=256,mn:int=64):
    sys_prompt=_ROLE_PROMPTS.get(role,_ROLE_PROMPTS['oracle'])
    full=f'{sys_prompt}\n\nTopic: {topic}\nText: {response}\n\nReview:'
    _,out=_gen_text(rt,full,mp,mn,greedy=True)
    return out.strip()[:200]
def triumvirate_review(rt,topic:str,response:str,mp:int=256,mn:int=48):
    p_left=f'{_ROLE_PROMPTS["left_brain"]}\n\nTopic: {topic}\nText: {response}\n\nReview:'
    p_right=f'{_ROLE_PROMPTS["right_brain"]}\n\nTopic: {topic}\nText: {response}\n\nReview:'
    outs=_gen_batched(rt,[p_left,p_right],mp,mn,greedy=True)
    left,right=outs[0].strip()[:200],outs[1].strip()[:200]
    cere_prompt=f'{_ROLE_PROMPTS["cerebellum"]}\n\nTopic: {topic}\nText: {response}\n\nLeft Brain: {left[:120]}\nRight Brain: {right[:120]}\n\nIs this text useful, novel, on-topic? Answer "yes" or "no" then briefly justify:'
    _,verdict=_gen_text(rt,cere_prompt,mp,32,greedy=True)
    yes='yes' in verdict.lower()[:30]
    return {'mode':'triumvirate','passed':yes,'left':left[:120],'right':right[:120],'verdict':verdict.strip()[:120]}
def dual_mind_review(rt,topic:str,response:str,mp:int=256,mn:int=48):
    p_fast=f'{_ROLE_PROMPTS["fast"]}\n\nTopic: {topic}\nText: {response}\n\nReview:'
    p_slow=f'{_ROLE_PROMPTS["slow"]}\n\nTopic: {topic}\nText: {response}\n\nReview:'
    outs=_gen_batched(rt,[p_fast,p_slow],mp,max(mn,16),greedy=True)
    fast,slow=outs[0].strip()[:200],outs[1].strip()[:200]
    consensus_prompt=f'Topic: {topic}\nText: {response}\nFast: {fast[:80]}\nSlow: {slow[:160]}\n\nIs the text valuable? Answer "yes" or "no":'
    _,verdict=_gen_text(rt,consensus_prompt,mp,16,greedy=True)
    yes='yes' in verdict.lower()[:20]
    return {'mode':'dual_mind','passed':yes,'fast':fast[:80],'slow':slow[:120],'verdict':verdict.strip()[:80]}
def grail_verify(rt,topic:str,response:str,mp:int=256,mn:int=32):
    full=f'{_ROLE_PROMPTS["grail_verify"]}\n\nTopic: {topic}\nClaim: {response}\n\nVerification:'
    _,verdict=_gen_text(rt,full,mp,mn,greedy=True)
    valid='valid' in verdict.lower()[:30] and 'invalid' not in verdict.lower()[:30]
    return {'mode':'grail','passed':valid,'verdict':verdict.strip()[:160]}
def lite_review(rt,topic:str,response:str,mp:int=256,mn:int=16):
    crit_prompt=f'Topic: {topic}\nText: {response}\nIs the text a useful, novel, on-topic fact? Answer "yes" or "no":'
    _,verdict=_gen_text(rt,crit_prompt,mp,mn,greedy=True)
    yes='yes' in verdict.lower()[:30]
    return {'mode':'lite','passed':yes,'verdict':verdict.strip()[:60]}
COUNCIL_MODES={'none':None,'lite':lite_review,'triumvirate':triumvirate_review,'dual_mind':dual_mind_review,'grail':grail_verify}
def council_review(rt,mode:str,topic:str,response:str,mp:int=256,mn:int=48):
    fn=COUNCIL_MODES.get(mode)
    if fn is None:return {'mode':'none','passed':True,'verdict':'(no review)'}
    return fn(rt,topic,response,mp,mn)
