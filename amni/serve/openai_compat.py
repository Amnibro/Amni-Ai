"""OpenAI /v1/* compat — turns Adam into a drop-in OpenAI tool-calling backend for Amni-Code, Continue.dev, Cline, Aider, etc.
Endpoints:
  GET  /v1/models                 — exposes adam:e2b-gf17 + aliases
  POST /v1/chat/completions       — messages[] + tools[] + stream → tool_calls or final text
Wire shape exactly mirrors OpenAI's spec. Tool-call grammar is taught via system prompt (tool_protocol.build_system_prompt). Adam emits ```tool_call fenced JSON, the parser lifts them into the OpenAI tool_calls[] shape. role:"tool" messages get flattened into Adam's history as OBSERVATION blocks.
Persistent autonomy: every completion records (intent → tool sequence) into the CodeAtlas via cell-address LUT, so future similar coding tasks get a `hint_for_prompt` injected up front."""
import time,json,uuid,asyncio
from typing import List,Dict,Any,Optional,AsyncIterator
from amni.serve.tool_protocol import parse_tool_calls,strip_tool_calls,build_system_prompt,flatten_history,tools_digest,build_openai_tool_calls,openai_finish_reason,now_unix
from amni.serve.widget_protocol import parse_widgets,strip_widgets,build_system_prompt_addendum as _widget_sys_addendum
_MODEL_NAME='adam:e2b-gf17'
_MODEL_ALIASES=['adam','adam:latest','adam-e2b-gf17','amni-a1','amni-ai','adam-gf17','gpt-3.5-turbo','gpt-4','gpt-4o-mini','claude-3-sonnet','llama3.1','qwen2.5','gemma2']
def _model_card(name:str)->Dict[str,Any]:return {'id':name,'object':'model','created':1715000000,'owned_by':'amnibro','permission':[],'root':_MODEL_NAME,'parent':None}
def models_list()->Dict[str,Any]:return {'object':'list','data':[_model_card(_MODEL_NAME)]+[_model_card(a) for a in _MODEL_ALIASES]}
def _completion_id()->str:return f'chatcmpl-{uuid.uuid4().hex[:24]}'
def _call_adam(adam,agent,system:str,pairs:List,user_msg:str,tools:Optional[List],is_stream:bool,max_tokens:int,do_sample:bool,code_atlas,session_id:str):
    facts=[]
    try:
        if agent is not None and hasattr(agent,'_extract_user_facts'):
            conv=agent.store.get(session_id) if hasattr(agent,'store') else None
            if conv is not None:facts=agent._extract_user_facts(conv) or []
    except Exception:pass
    is_private=False
    try:
        from amni.serve.conversation import detect_personal as _dp
        is_private=bool(_dp(user_msg))
    except Exception:pass
    if code_atlas is not None and tools:
        try:
            hint=code_atlas.hint_for_prompt(user_msg,session_id=session_id,k=2)
            if hint:system=(system+'\n\n'+hint).strip()
        except Exception:pass
    if is_stream:return adam.chat_persona_stream(user_msg,system=system,history=pairs,facts=facts,is_private=is_private,max_new_tokens=max_tokens,do_sample=do_sample)
    return adam.chat_persona(user_msg,system=system,history=pairs,facts=facts,is_private=is_private,max_new_tokens=max_tokens,do_sample=do_sample)
def _make_completion_response(comp_id:str,model:str,full_text:str,tool_calls_oa:List[Dict[str,Any]],tokens:int,wall_s:float,widgets:Optional[List[Dict[str,Any]]]=None)->Dict[str,Any]:
    visible=strip_widgets(strip_tool_calls(full_text)) if (tool_calls_oa or widgets) else (full_text or '')
    msg={'role':'assistant','content':visible if visible else None}
    if tool_calls_oa:msg['tool_calls']=tool_calls_oa
    if widgets:msg['amni_widgets']=widgets
    out={'id':comp_id,'object':'chat.completion','created':now_unix(),'model':model,'choices':[{'index':0,'message':msg,'finish_reason':openai_finish_reason(tool_calls_oa)}],'usage':{'prompt_tokens':0,'completion_tokens':int(tokens or 0),'total_tokens':int(tokens or 0)},'amni_wall_s':round(wall_s,3),'amni_tier':'tier_tool_agent' if tool_calls_oa else ('tier_widget' if widgets else 'tier_persona')}
    if widgets:out['amni_widgets']=widgets
    return out
def _sse(data:Any)->str:return f'data: {json.dumps(data)}\n\n'
def _stream_chunks(comp_id:str,model:str,gen_iter,session_id:str,code_atlas,intent:str)->AsyncIterator[str]:
    async def _agen():
        ts=now_unix();first=_sse({'id':comp_id,'object':'chat.completion.chunk','created':ts,'model':model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]});yield first
        full=''
        try:
            for chunk in gen_iter:
                if not chunk:continue
                full+=chunk
                if '```tool_call' in chunk or '```tool_call' in full[-32:]:continue
                yield _sse({'id':comp_id,'object':'chat.completion.chunk','created':ts,'model':model,'choices':[{'index':0,'delta':{'content':chunk},'finish_reason':None}]})
                await asyncio.sleep(0)
        except Exception as e:yield _sse({'error':{'message':f'stream error: {e}','type':'inference_error'}});return
        parsed=parse_tool_calls(full)
        if parsed:
            tcs=build_openai_tool_calls(parsed)
            yield _sse({'id':comp_id,'object':'chat.completion.chunk','created':ts,'model':model,'choices':[{'index':0,'delta':{'tool_calls':tcs},'finish_reason':None}]})
            yield _sse({'id':comp_id,'object':'chat.completion.chunk','created':ts,'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})
        else:
            visible=strip_tool_calls(full)
            yield _sse({'id':comp_id,'object':'chat.completion.chunk','created':ts,'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})
            if code_atlas is not None and intent and visible:
                try:code_atlas.record(session_id,intent,parsed,outcome=visible[:600])
                except Exception:pass
        yield 'data: [DONE]\n\n'
    return _agen()
def mount(app,adam,agent,code_atlas=None):
    from fastapi import Request,HTTPException
    from fastapi.responses import StreamingResponse,JSONResponse
    @app.get('/v1/models')
    def v1_models():return models_list()
    @app.get('/v1/models/{model_id}')
    def v1_model(model_id:str):
        if model_id==_MODEL_NAME or model_id in _MODEL_ALIASES:return _model_card(model_id)
        raise HTTPException(status_code=404,detail=f'unknown model {model_id}')
    @app.post('/v1/chat/completions')
    async def v1_chat(req:Request):
        body=await req.json()
        messages=body.get('messages',[]) or []
        tools=body.get('tools') or []
        is_stream=bool(body.get('stream',False))
        model=body.get('model',_MODEL_NAME)
        max_tokens=int(body.get('max_tokens') or body.get('max_completion_tokens') or 1024)
        temperature=float(body.get('temperature') or 0.7)
        do_sample=temperature>0.01
        session_id=body.get('user') or body.get('session_id') or 'oai_default'
        pairs,user_msg,client_system=flatten_history(messages)
        if not user_msg and pairs:user_msg=pairs[-1][0];pairs=pairs[:-1]
        if not user_msg:user_msg='(empty user message)'
        persona_sys=None
        try:
            if agent is not None and getattr(agent,'use_persona',False) and hasattr(agent,'personas'):
                p=agent.personas.for_session(session_id);persona_sys=p.system_prompt(user_msg) if p else None
        except Exception:persona_sys=None
        base_persona=(client_system+('\n\n'+persona_sys if persona_sys else '')).strip() if client_system or persona_sys else None
        widget_addendum=_widget_sys_addendum(['weather','system','time','news','code','file','error','info'])
        system=build_system_prompt(tools,cwd=None,custom=None,base_persona=base_persona)
        if widget_addendum:system=(system+'\n\n'+widget_addendum).strip() if system else widget_addendum
        comp_id=_completion_id();t0=time.time()
        if is_stream:
            gen=_call_adam(adam,agent,system,pairs,user_msg,tools,True,max_tokens,do_sample,code_atlas,session_id)
            return StreamingResponse(_stream_chunks(comp_id,model,gen,session_id,code_atlas,user_msg),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
        r=_call_adam(adam,agent,system,pairs,user_msg,tools,False,max_tokens,do_sample,code_atlas,session_id)
        full=(r.get('answer') or '') if isinstance(r,dict) else ''
        tokens=(r.get('tokens') or 0) if isinstance(r,dict) else 0
        parsed=parse_tool_calls(full)
        widgets=parse_widgets(full)
        tcs=build_openai_tool_calls(parsed)
        if code_atlas is not None and user_msg:
            try:code_atlas.record(session_id,user_msg,parsed,outcome=strip_widgets(strip_tool_calls(full))[:600] if not parsed else '')
            except Exception:pass
        return _make_completion_response(comp_id,model,full,tcs,tokens,time.time()-t0,widgets=widgets)
    @app.get('/v1/healthz')
    def v1_health():return {'ok':True,'model':_MODEL_NAME,'tools':'openai_v1_chat_completions'}
