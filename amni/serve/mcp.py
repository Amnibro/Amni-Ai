"""Minimal MCP-style HTTP transport — exposes Adam as a tool source for any MCP client (Claude Code, Cursor, IDE plugins).
JSON-RPC 2.0 over POST /mcp. Implements: initialize, tools/list, tools/call, resources/list, prompts/list.
Tools exposed: ask_adam, scan_directory, mem_search, file_read, calc, time, plus every registered skill."""
import time,os
from amni import APP_VERSION
_MCP_MAX_ARG_CHARS=int(os.environ.get('AMNI_MAX_INPUT_CHARS','100000'))
def _tool_defs(agent):
    base=[{'name':'ask_adam','description':'Ask Adam a question. Routes through full tier pipeline (LUT cache, semantic match, tier3 cold-solve). Persistent learning enabled.','inputSchema':{'type':'object','properties':{'question':{'type':'string'},'persona':{'type':'string','description':'Optional persona name (rikku, yoda, mentor, etc, or any custom name — Adam will web-learn unknowns).'}},'required':['question']}},
          {'name':'mem_search','description':"Search Adam's persistent lesson bank with semantic + flat-cosine fallback. Returns top-K (question, answer, score) hits.",'inputSchema':{'type':'object','properties':{'query':{'type':'string'},'k':{'type':'integer','default':3}},'required':['query']}},
          {'name':'scan_directory','description':'Walk a directory, chunk text files, ingest each chunk into Adam lesson bank. Persistent.','inputSchema':{'type':'object','properties':{'path':{'type':'string'},'glob':{'type':'string','default':'**/*'},'max_files':{'type':'integer','default':20},'distill':{'type':'boolean','default':False}},'required':['path']}},
          {'name':'list_personas','description':'List known personas (presets + previously web-learned).','inputSchema':{'type':'object','properties':{}}},
          {'name':'set_persona','description':'Switch active persona. Unknown names trigger web-learn.','inputSchema':{'type':'object','properties':{'name':{'type':'string'},'description':{'type':'string','description':'Optional manual description (skips web-learn)'}},'required':['name']}}]
    for s in agent.list_skills():
        if s['name'] in ('mem','scan'):continue
        base.append({'name':f'skill_{s["name"]}','description':f'(direct skill) {s["desc"]}','inputSchema':{'type':'object','properties':{k:{'type':'string'} for k in (s.get('schema') or {})},'additionalProperties':True}})
    return base
def _call_tool(agent,name,args):
    if name=='ask_adam':
        sid=None
        if args.get('persona'):
            try:agent.personas.assign_session('mcp_session',args['persona'])
            except Exception:pass
            sid='mcp_session'
        r=agent.chat(args['question'],session_id=sid)
        return {'answer':r.get('answer'),'tier':r.get('tier'),'tokens':r.get('tokens'),'persona':r.get('persona'),'category':r.get('category')}
    if name=='mem_search':
        r=agent.skills.call('mem',{'query':args['query'],'k':int(args.get('k',3))},ctx={'adam':agent.adam})
        return r.to_dict()
    if name=='scan_directory':
        r=agent.skills.call('scan',{'path':args['path'],'glob':args.get('glob','**/*'),'max_files':int(args.get('max_files',20)),'distill':bool(args.get('distill',False))},ctx={'adam':agent.adam})
        return r.to_dict()
    if name=='list_personas':return {'default':agent.personas._default,'known':[p.to_dict() for p in agent.personas.list_known()]}
    if name=='set_persona':
        n=args['name']
        if not agent.personas.has(n):p=agent.personas.learn(n,user_description=args.get('description'))
        else:p=agent.personas.get(n)
        return {'persona':p.to_dict()}
    if name.startswith('skill_'):
        skill=name[6:]
        r=agent.skills.call(skill,args,ctx={'adam':agent.adam})
        return r.to_dict()
    return {'error':f'unknown tool: {name}'}
def mount(app,agent):
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from amni.serve.rate_limit import from_env as _rl_from_env,client_key as _rl_key
    _RL_MCP=_rl_from_env('mcp',120)
    @app.post('/mcp')
    async def mcp(req:Request):
        try:body=await req.json()
        except Exception:return JSONResponse(status_code=400,content={'jsonrpc':'2.0','error':{'code':-32700,'message':'parse error'},'id':None})
        method=body.get('method');rpc_id=body.get('id');params=body.get('params') or {}
        if method=='initialize':
            return {'jsonrpc':'2.0','id':rpc_id,'result':{'protocolVersion':'2025-06-18','serverInfo':{'name':'amni-ai-adam','version':APP_VERSION},'capabilities':{'tools':{'listChanged':False},'resources':{'subscribe':False,'listChanged':False},'prompts':{'listChanged':False}}}}
        if method=='tools/list':return {'jsonrpc':'2.0','id':rpc_id,'result':{'tools':_tool_defs(agent)}}
        if method=='tools/call':
            _ok,_rl=_RL_MCP.allow(_rl_key(req))
            if not _ok:return {'jsonrpc':'2.0','id':rpc_id,'result':{'content':[{'type':'text','text':f"error: rate limit {_rl['limit']}/{int(_rl['window_s'])}s — retry in {_rl['retry_after_s']}s"}],'isError':True}}
            name=params.get('name','');args=params.get('arguments',{})
            if len(str(args))>_MCP_MAX_ARG_CHARS:return {'jsonrpc':'2.0','id':rpc_id,'result':{'content':[{'type':'text','text':f'error: arguments too large (>{_MCP_MAX_ARG_CHARS} chars)'}],'isError':True}}
            try:out=_call_tool(agent,name,args);return {'jsonrpc':'2.0','id':rpc_id,'result':{'content':[{'type':'text','text':str(out)}],'isError':False,'_raw':out}}
            except Exception as e:return {'jsonrpc':'2.0','id':rpc_id,'result':{'content':[{'type':'text','text':f'error: {e}'}],'isError':True}}
        if method=='resources/list':
            sessions=agent.store.list_sessions()
            return {'jsonrpc':'2.0','id':rpc_id,'result':{'resources':[{'uri':f'amni://session/{s["session_id"]}','name':f'Session {s["session_id"]}','mimeType':'application/jsonl'} for s in sessions[:20]]}}
        if method=='prompts/list':return {'jsonrpc':'2.0','id':rpc_id,'result':{'prompts':[{'name':'introspect','description':'Adam describes its own capabilities','arguments':[]},{'name':'persona_intro','description':'Introduce yourself in current persona','arguments':[]}]}}
        if method=='ping':return {'jsonrpc':'2.0','id':rpc_id,'result':{}}
        return {'jsonrpc':'2.0','id':rpc_id,'error':{'code':-32601,'message':f'method not found: {method}'}}
    @app.get('/mcp')
    def mcp_info():return {'name':'amni-ai-adam','version':APP_VERSION,'transport':'http-jsonrpc','endpoint':'/mcp','client_config_example':{'mcpServers':{'amni-ai':{'url':'http://localhost:11434/mcp','transport':'http'}}}}
