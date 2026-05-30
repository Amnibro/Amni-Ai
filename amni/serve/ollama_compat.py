"""Ollama-compatible endpoints: /api/tags, /api/show, /api/generate, /api/chat (non-streaming + SSE).
Shape-matches Ollama spec so Open WebUI / Continue.dev / LangChain Ollama clients plug in unchanged.
Adam is exposed as 'adam:granite-gf17' plus aliases for common names so clients with hardcoded model strings still resolve."""
import time,json
from typing import List,Dict,Any,Optional,AsyncIterator
_DIGEST='sha256:e2b-gf17-reffelt-4tier-asimov-locked-amni-ai-v6-0-0-deployable-surface-2026-05-15'
_MODEL_NAMES=['adam:granite-gf17','adam:e2b-gf17','adam:latest','llama3:latest','qwen2.5:latest']
def _model_record(name:str,adam_stats:Optional[Dict]=None)->Dict[str,Any]:
    n=adam_stats.get('lessons_n',0) if adam_stats else 0
    return {'name':name,'model':name,'modified_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'size':5_000_000_000,'digest':_DIGEST,'details':{'parent_model':'','format':'ptex-gf17','family':'adam','families':['adam','granite','reffelt'],'parameter_size':'2B','quantization_level':'GF17-lossless'},'lessons_n':n}
def tags_response(agent)->Dict[str,Any]:
    stats=agent.stats() if hasattr(agent,'stats') else {}
    return {'models':[_model_record(n,stats) for n in _MODEL_NAMES]}
def show_response(name:str,agent)->Dict[str,Any]:
    stats=agent.stats() if hasattr(agent,'stats') else {}
    return {'modelfile':f'# Adam — GF(17) texture-native, 5 Asimov laws, persistent lessons (N={stats.get("lessons_n",0)})\nFROM adam:granite-gf17','parameters':'temperature 0\nnum_ctx 8192','template':'{{ .Prompt }}','details':_model_record(name,stats)['details'],'model_info':{'general.architecture':'adam','general.parameter_count':2_000_000_000}}
def generate_response(agent,prompt:str,session_id:Optional[str]=None,model:str='adam:granite-gf17')->Dict[str,Any]:
    r=agent.chat(prompt,session_id=session_id,use_skills=True,writeback=True)
    return {'model':model,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'response':r.get('answer',''),'done':True,'done_reason':'stop','context':[],'total_duration':int(r.get('wall_s',0)*1e9),'load_duration':0,'prompt_eval_count':len(prompt.split()),'prompt_eval_duration':0,'eval_count':r.get('tokens',0),'eval_duration':int(r.get('wall_s',0)*1e9),'amni_tier':r.get('tier'),'amni_skill_calls':r.get('skill_calls',[]),'session_id':r.get('session_id')}
def chat_response(agent,messages:List[Dict[str,str]],session_id:Optional[str]=None,model:str='adam:granite-gf17')->Dict[str,Any]:
    user_msgs=[m['content'] for m in messages if m.get('role')=='user']
    if not user_msgs:return {'error':'no user message'}
    msg=user_msgs[-1]
    r=agent.chat(msg,session_id=session_id,use_skills=True,writeback=True)
    return {'model':model,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':{'role':'assistant','content':r.get('answer','')},'done':True,'done_reason':'stop','total_duration':int(r.get('wall_s',0)*1e9),'load_duration':0,'prompt_eval_count':sum(len(m['content'].split()) for m in messages),'prompt_eval_duration':0,'eval_count':r.get('tokens',0),'eval_duration':int(r.get('wall_s',0)*1e9),'amni_tier':r.get('tier'),'amni_skill_calls':r.get('skill_calls',[]),'session_id':r.get('session_id')}
async def chat_stream(agent,messages:List[Dict[str,str]],session_id:Optional[str]=None,model:str='adam:granite-gf17')->AsyncIterator[str]:
    user_msgs=[m['content'] for m in messages if m.get('role')=='user']
    if not user_msgs:yield json.dumps({'error':'no user message'})+'\n';return
    msg=user_msgs[-1]
    r=agent.chat(msg,session_id=session_id,use_skills=True,writeback=True)
    answer=r.get('answer','') or ''
    ts=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    chunk_size=max(1,len(answer)//8 or 1)
    for i in range(0,len(answer),chunk_size):
        yield json.dumps({'model':model,'created_at':ts,'message':{'role':'assistant','content':answer[i:i+chunk_size]},'done':False})+'\n'
    yield json.dumps({'model':model,'created_at':ts,'message':{'role':'assistant','content':''},'done':True,'done_reason':'stop','total_duration':int(r.get('wall_s',0)*1e9),'eval_count':r.get('tokens',0),'amni_tier':r.get('tier'),'amni_skill_calls':r.get('skill_calls',[]),'session_id':r.get('session_id')})+'\n'
def mount(app,agent):
    from fastapi import Request,HTTPException
    from fastapi.responses import StreamingResponse,JSONResponse
    from amni.serve.rate_limit import from_env as _rlf,client_key as _rlk
    import os as _os
    _RL=_rlf('ollama',60);_MAXC=int(_os.environ.get('AMNI_MAX_INPUT_CHARS','100000'));_MAXEMB=int(_os.environ.get('AMNI_MAX_EMBED_BATCH','512'))
    def _guard(req,text):
        ok,info=_RL.allow(_rlk(req))
        if not ok:raise HTTPException(status_code=429,detail=f"rate limit {info['limit']}/{int(info['window_s'])}s — retry in {info['retry_after_s']}s")
        if len(text or '')>_MAXC:raise HTTPException(status_code=413,detail=f'input too large (>{_MAXC} chars)')
    @app.get('/api/tags')
    def tags():return tags_response(agent)
    @app.post('/api/show')
    async def show(req:Request):
        body=await req.json()
        return show_response(body.get('name','adam:granite-gf17'),agent)
    @app.post('/api/generate')
    async def generate(req:Request):
        body=await req.json()
        stream=bool(body.get('stream',False))
        prompt=body.get('prompt','')
        _guard(req,prompt)
        sid=body.get('session_id') or body.get('context_id')
        model=body.get('model','adam:granite-gf17')
        if not stream:return generate_response(agent,prompt,session_id=sid,model=model)
        async def gen():
            r=agent.chat(prompt,session_id=sid,use_skills=True,writeback=True)
            ans=r.get('answer','') or '';ts=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            chunk=max(1,len(ans)//8 or 1)
            for i in range(0,len(ans),chunk):yield json.dumps({'model':model,'created_at':ts,'response':ans[i:i+chunk],'done':False})+'\n'
            yield json.dumps({'model':model,'created_at':ts,'response':'','done':True,'done_reason':'stop','eval_count':r.get('tokens',0),'amni_tier':r.get('tier'),'amni_skill_calls':r.get('skill_calls',[]),'session_id':r.get('session_id')})+'\n'
        return StreamingResponse(gen(),media_type='application/x-ndjson')
    @app.post('/api/chat')
    async def chat(req:Request):
        body=await req.json()
        stream=bool(body.get('stream',False))
        messages=body.get('messages',[])
        _guard(req,''.join(str(m.get('content') or '') for m in messages if isinstance(m,dict)))
        sid=body.get('session_id') or body.get('context_id')
        model=body.get('model','adam:granite-gf17')
        if not stream:return chat_response(agent,messages,session_id=sid,model=model)
        return StreamingResponse(chat_stream(agent,messages,session_id=sid,model=model),media_type='application/x-ndjson')
    @app.get('/api/version')
    def version():return {'version':'amni-ai-6.0.0'}
    @app.post('/api/embed')
    async def embed(req:Request):
        body=await req.json()
        text=body.get('input') or body.get('prompt') or ''
        if isinstance(text,str):text=[text]
        if not isinstance(text,list):raise HTTPException(status_code=400,detail='input must be a string or list')
        if len(text)>_MAXEMB:raise HTTPException(status_code=413,detail=f'too many embedding inputs ({len(text)} > {_MAXEMB})')
        _guard(req,''.join(str(t) for t in text))
        try:
            if agent.adam.sem_lut is not None and hasattr(agent.adam.sem_lut,'_encoder'):
                enc=agent.adam.sem_lut._encoder
                embs=enc.encode(text,normalize_embeddings=True,convert_to_numpy=True).tolist()
                return {'model':body.get('model','adam:granite-gf17'),'embeddings':embs}
        except Exception as e:return JSONResponse(status_code=500,content={'error':str(e)})
        return JSONResponse(status_code=503,content={'error':'embedding not initialized'})
