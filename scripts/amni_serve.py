"""v6.0.0 deployable Adam server.
Routes:
  GET  /                  — simple chat UI
  POST /chat              — agent chat (multi-turn, skill dispatch)
  POST /ask               — single-shot Adam (no skills, no memory)
  POST /teach             — add (q,a) to lesson bank
  GET  /stats             — Adam + agent stats
  GET  /healthz           — liveness
  GET  /skills            — list registered skills
  POST /skills/{name}     — direct skill invocation
  GET  /sessions          — list conversation sessions
  DELETE /sessions/{id}   — delete a session
  /api/tags /api/show /api/generate /api/chat /api/embed /api/version  — Ollama compat
Usage:
  python scripts/amni_serve.py --seed
  Then point Open WebUI at http://localhost:8001 or open http://localhost:8001 in a browser.
"""
import os,sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bake',default='E:/Amni-Ai-Bakes/gemma4_e2b_it_gf17')
    ap.add_argument('--model',default='E:/Amni-Ai-Models/gemma-4-E2B-it')
    ap.add_argument('--port',type=int,default=8001)
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--lessons',default='experiences/adam_lessons.npz')
    ap.add_argument('--lut-root',default='experiences/adam_lut')
    ap.add_argument('--conv-root',default='experiences/conversations')
    ap.add_argument('--audit-log',default='logs/agent_skill_calls.jsonl')
    ap.add_argument('--workdir',default=None,help='Primary skill workdir (defaults to cwd). Use --root for additional roots.')
    ap.add_argument('--root',action='append',default=[],help='Additional allowed root for file_*/code_edit/scan/shell. Repeatable.')
    ap.add_argument('--unrestricted-files',action='store_true',help='Drop workdir gating and allow file ops on ANY drive. Use with care.')
    ap.add_argument('--seed',action='store_true',help='seed lessons with default bank if file missing')
    ap.add_argument('--cors',action='store_true',help='enable permissive CORS for dev')
    ap.add_argument('--persona-bank',default='experiences/personas.json',help='Persona store path (learned personas + per-session map)')
    ap.add_argument('--default-persona',default=None,help='Default persona name (preset or learned). e.g. rikku, yoda, neutral')
    ap.add_argument('--no-persona',action='store_true',help='Disable persona layer entirely (raw Adam responses)')
    args=ap.parse_args()
    try:
        from fastapi import FastAPI,Request,HTTPException
        from pydantic import BaseModel
        from typing import Optional
        import uvicorn
    except ImportError:
        print('[amni_serve] missing fastapi/uvicorn. Install: pip install fastapi uvicorn pydantic',flush=True)
        sys.exit(1)
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore,PersonaStore
    from amni.serve.skills import default_registry
    from amni.serve import ollama_compat,web,mcp
    print(f'[amni_serve] booting Adam with bake={args.bake}',flush=True)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    print(f'[amni_serve] Adam ready: {adam.stats()}',flush=True)
    skills=default_registry(workdir=args.workdir,roots=args.root,audit_log=args.audit_log,unrestricted=args.unrestricted_files)
    store=ConversationStore(root=args.conv_root)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.default_persona:personas.set_default(args.default_persona)
    agent=AmniAgent(adam=adam,skills=skills,store=store,workdir=args.workdir,personas=personas,use_persona=not args.no_persona)
    print(f'[amni_serve] Persona: default={personas._default} known={[p.name for p in personas.list_known()]}',flush=True)
    scope='UNRESTRICTED (all drives)' if args.unrestricted_files else f'roots={[str(r) for r in skills.roots]}'
    print(f'[amni_serve] Agent ready: skills={[s["name"] for s in agent.list_skills()]} {scope} sessions={len(store.list_sessions())}',flush=True)
    app=FastAPI(title='Amni-Ai Adam',version='6.0.0')
    if args.cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
    class ChatRequest(BaseModel):
        message:str
        session_id:Optional[str]=None
        use_skills:bool=True
        writeback:bool=True
    class AskRequest(BaseModel):
        query:str
        writeback:bool=True
    class TeachRequest(BaseModel):
        question:str
        answer:str
    class SkillRequest(BaseModel):
        args:dict={}
    @app.post('/chat')
    def chat(req:ChatRequest):return agent.chat(req.message,session_id=req.session_id,use_skills=req.use_skills,writeback=req.writeback)
    @app.post('/chat/stream')
    async def chat_stream(req:ChatRequest,request:Request):
        from fastapi.responses import StreamingResponse
        import json as _json
        from amni.serve.agent import _needs_cot,_pick_cot
        from amni.serve import tone_atlas
        def gen():
            t0=time.time()
            conv=store.get(req.session_id)
            conv.append('user',req.message)
            persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
            persona_name=persona.name if persona else 'Adam'
            yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":persona_name})}\n\n'
            try:
                from amni.a1.semantic_intent import screen as _sem_screen
                _blk,_cat,_cos,_refmsg=_sem_screen(req.message)
                if _blk:
                    yield f'event: meta\ndata: {_json.dumps({"blocked":True,"category":_cat,"cos":round(_cos,3)})}\n\n'
                    for ch in [_refmsg[i:i+24] for i in range(0,len(_refmsg),24)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                    conv.append('assistant',_refmsg,{'tier':f'tier_intent_block_{_cat}','blocked':True,'cos':round(_cos,3)})
                    yield f'event: done\ndata: {_json.dumps({"tier":f"tier_intent_block_{_cat}","wall_s":round(time.time()-t0,3),"blocked":True})}\n\n'
                    return
            except Exception:pass
            category=tone_atlas.classify_intent(req.message)
            apply_cot=_needs_cot(category,req.message) and persona and persona.name!='Adam'
            sl=getattr(adam,'sem_lut',None)
            try:
                eff=sl.auto_margin() if sl and hasattr(sl,'auto_margin') else 0.08
                hit=sl.lookup_soft(req.message,margin=eff) if sl and hasattr(sl,'lookup_soft') else None
            except Exception:hit=None
            if hit:
                for ch in [hit[i:i+24] for i in range(0,len(hit),24)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                conv.append('assistant',hit,{'tier':'tier1_5_semantic_lesson','persona':persona_name,'category':category})
                yield f'event: done\ndata: {_json.dumps({"tier":"tier1_5_semantic_lesson","wall_s":round(time.time()-t0,3)})}\n\n';return
            if apply_cot:scaffold=_pick_cot(category,req.message);sys_p=persona.system_prompt(req.message)+'\n\n'+scaffold
            elif persona:sys_p=persona.system_prompt(req.message)
            else:sys_p='You are a helpful assistant.'
            yield f'event: meta\ndata: {_json.dumps({"cot":apply_cot,"category":category})}\n\n'
            max_new=int(80+200*(persona.length if persona else 0.5))+(700 if (apply_cot and category=="code") else (450 if apply_cot else 0))
            full=[]
            try:
                for chunk in adam.chat_persona_stream(req.message,system=sys_p,max_new_tokens=max_new,do_sample=True):
                    full.append(chunk)
                    yield f'event: token\ndata: {_json.dumps(chunk)}\n\n'
            except Exception as e:yield f'event: error\ndata: {_json.dumps(str(e))}\n\n';return
            final=''.join(full).strip()
            tier_final='tier_persona_cot' if apply_cot else 'tier_persona'
            if apply_cot and category=='code' and final and skills.has('run_python'):
                from amni.serve.agent import _extract_python_blocks,_validate_python,_perturb_retry,_extract_asserts,_run_with_tests
                blocks=_extract_python_blocks(final)
                if blocks:
                    bad=_validate_python(blocks)
                    if bad:yield f'event: validate\ndata: {_json.dumps({"syntax_errors":len(bad)})}\n\n'
                    runnable=[b for b in blocks if ('print(' in b or 'if __name__' in b)]
                    if runnable:
                        snippet=('\n\n'.join(blocks) if len(blocks)>1 else runnable[-1])
                        if len(blocks)>1:yield f'event: multi_block\ndata: {_json.dumps({"blocks":len(blocks),"runnable":len(runnable),"stitched_chars":len(snippet)})}\n\n'
                        try:
                            run_r=skills.call('run_python',{'code':snippet,'timeout':8},ctx={'adam':adam})
                            if run_r.ok and not run_r.output.get('error'):
                                so=(run_r.output.get('stdout') or '').strip()
                                se=(run_r.output.get('stderr') or '').strip()
                                rc=run_r.output.get('returncode')
                                yield f'event: exec\ndata: {_json.dumps({"stdout":so[:1500],"stderr":se[:600],"returncode":rc,"timed_out":run_r.output.get("timed_out",False)})}\n\n'
                                exec_md=f'\n\n**[Sandbox execution — exit {rc}{"  (timed out)" if run_r.output.get("timed_out") else ""}]**\n'
                                if so:exec_md+=f'```\n{so[:1500]}\n```\n'
                                if se:exec_md+=f'_stderr:_\n```\n{se[:600]}\n```'
                                final+=exec_md
                                tier_final+='_run'
                                test_failed=False;test_err=''
                                if rc==0 and not se:
                                    asserts=_extract_asserts(final)
                                    if asserts:
                                        from amni.serve.agent import _assert_diversity
                                        div_score,div_info=_assert_diversity(asserts)
                                        passed,terr,tinfo=_run_with_tests(skills,adam,snippet,asserts)
                                        yield f'event: test_run\ndata: {_json.dumps({"asserts_n":len(asserts),"passed":passed,"diversity":round(div_score,2),"info":tinfo,"div":div_info})}\n\n'
                                        div_tag='_tests_thin' if div_score<0.5 else ('_tests_diverse' if div_score>=0.75 else '_tests_ok')
                                        if passed:
                                            final+=f'\n\n**[Self-tests — {len(asserts)}/{len(asserts)} passed · diversity={div_score:.2f}]**'
                                            tier_final+=div_tag
                                            try:
                                                promo_ans=final[:2000]
                                                tr=adam.teach(req.message,promo_ans)
                                                tier_final+='_promoted'
                                                yield f'event: promoted\ndata: {_json.dumps({"lessons_n":tr.get("lessons_n",0)})}\n\n'
                                            except Exception as _pe:yield f'event: promoted\ndata: {_json.dumps({"error":str(_pe)[:120]})}\n\n'
                                        else:
                                            final+=f'\n\n**[Self-tests FAILED — {terr[:200]}]**'
                                            test_failed=True;test_err=terr
                                if rc!=0 or se or test_failed:
                                    perturb_events=[]
                                    emit_fn=lambda d:perturb_events.append(d)
                                    err_signal=test_err if test_failed else (se or f'exit code {rc}')
                                    perturb_asserts=_extract_asserts(final) if test_failed else None
                                    pr=_perturb_retry(adam,skills,sys_p,snippet,err_signal,req.message,max_steps=3,emit=emit_fn,asserts=perturb_asserts)
                                    for ev in perturb_events:yield f'event: perturb\ndata: {_json.dumps(ev)}\n\n'
                                    if pr.get('success'):
                                        final+=f'\n\n**[Trial-and-error fixed it — {pr["magnitude"]} perturbation]**\n```python\n{pr["code"]}\n```\n```\n{pr["stdout"][:1500]}\n```'
                                        tier_final+=f'_perturb_{pr["magnitude"].lower()}'
                                        yield f'event: perturb\ndata: {_json.dumps({"final":True,"magnitude":pr["magnitude"],"success":True})}\n\n'
                                    else:
                                        final+=f'\n\n**[Trial-and-error exhausted SMALL/MEDIUM/LARGE — code still failing]**'
                                        tier_final+='_perturb_failed'
                                        yield f'event: perturb\ndata: {_json.dumps({"final":True,"success":False,"steps":len(pr.get("history",[]))})}\n\n'
                            elif run_r.ok and run_r.output.get('error'):
                                yield f'event: exec\ndata: {_json.dumps({"error":run_r.output["error"]})}\n\n'
                        except Exception as e:yield f'event: exec\ndata: {_json.dumps({"error":str(e)})}\n\n'
            conv.append('assistant',final,{'tier':tier_final,'persona':persona_name,'category':category})
            yield f'event: done\ndata: {_json.dumps({"tier":tier_final,"wall_s":round(time.time()-t0,3),"persona":persona_name,"category":category})}\n\n'
        return StreamingResponse(gen(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
    @app.post('/ask')
    def ask(req:AskRequest):return adam.ask(req.query,writeback=req.writeback)
    @app.post('/teach')
    def teach(req:TeachRequest):return adam.teach(req.question,req.answer)
    @app.get('/stats')
    def stats():return agent.stats()
    @app.get('/healthz')
    def health():return {'status':'ok','lessons_n':len(adam.sem_lut._raw),'skills_n':len(skills.list_skills()),'version':'6.0.0'}
    @app.get('/skills')
    def list_skills():return {'skills':skills.list_skills()}
    @app.post('/skills/{name}')
    def call_skill(name:str,req:SkillRequest):
        r=skills.call(name,req.args,ctx={'adam':adam})
        if not r.ok:raise HTTPException(status_code=400,detail=r.to_dict())
        return r.to_dict()
    @app.get('/sessions')
    def sessions():return {'sessions':store.list_sessions()}
    @app.delete('/sessions/{sid}')
    def del_session(sid:str):return {'deleted':store.delete(sid)}
    @app.get('/lessons')
    def lessons(q:str='',offset:int=0,limit:int=50):
        sl=getattr(adam,'sem_lut',None)
        raw=getattr(sl,'_raw',[]) if sl is not None else []
        items=[(i,k,v) for i,(k,v) in enumerate(raw)]
        if q:items=[(i,k,v) for i,k,v in items if q.lower() in k.lower() or q.lower() in v.lower()]
        total=len(items)
        page=items[offset:offset+limit]
        return {'total':total,'lessons_n':len(raw),'offset':offset,'limit':limit,'items':[{'idx':i,'q':k[:300],'a':v[:600]} for i,k,v in page]}
    class PersonaReq(BaseModel):
        name:str
        session_id:Optional[str]=None
        description:Optional[str]=None
        learn_via_web:bool=True
    @app.get('/personas')
    def list_personas():return {'default':personas._default,'known':[p.to_dict() for p in personas.list_known()]}
    @app.get('/persona/{name}')
    def get_persona(name:str):
        if not personas.has(name):
            p=personas.learn(name)
            return {'persona':p.to_dict(),'learned_now':True}
        return {'persona':personas.get(name).to_dict(),'learned_now':False}
    @app.post('/persona')
    def set_persona(req:PersonaReq):
        if not personas.has(req.name):
            if not req.learn_via_web and not req.description:raise HTTPException(status_code=404,detail=f'unknown persona "{req.name}" — pass description= or learn_via_web=true')
            p=personas.learn(req.name,user_description=req.description)
        else:p=personas.get(req.name)
        if req.session_id:personas.assign_session(req.session_id,req.name)
        else:personas.set_default(req.name)
        return {'persona':p.to_dict(),'session_id':req.session_id,'made_default':not req.session_id}
    class ReflectReq(BaseModel):
        max_n:int=3
        min_age_sec:int=0
    @app.post('/reflect')
    def trigger_reflect(req:ReflectReq):
        try:from amni.serve.reflection import reflect_once
        except Exception as e:raise HTTPException(status_code=500,detail=f'reflect import failed: {e}')
        return reflect_once(adam,max_n=req.max_n,min_age_sec=req.min_age_sec)
    @app.get('/project')
    def project_info():
        import os as _os
        root=_os.environ.get('AMNI_PROJECT_ROOT') or args.workdir or _os.getcwd()
        ptype=_os.environ.get('AMNI_PROJECT_TYPE','unknown')
        code_mode=_os.environ.get('AMNI_CODE_MODE')=='1'
        return {'root':root,'type':ptype,'code_mode':code_mode,'cwd':_os.getcwd()}
    @app.get('/project/tree')
    def project_tree(path:str='',depth:int=2,limit:int=200):
        import os as _os
        root=_os.environ.get('AMNI_PROJECT_ROOT') or args.workdir or _os.getcwd()
        target=Path(root)/path if path else Path(root)
        target=target.resolve()
        try:
            common=_os.path.commonpath([str(target),str(Path(root).resolve())])
            if common!=str(Path(root).resolve()):raise HTTPException(status_code=400,detail='outside project root')
        except Exception:raise HTTPException(status_code=400,detail='invalid path')
        if not target.exists():raise HTTPException(status_code=404,detail='not found')
        skip={'.git','node_modules','__pycache__','.venv','venv','.pytest_cache','dist','build','.next','target','.idea','.vscode'}
        items=[]
        def walk(p:Path,d:int):
            if d>depth or len(items)>=limit:return
            try:
                for e in sorted(p.iterdir(),key=lambda x:(not x.is_dir(),x.name.lower())):
                    if e.name.startswith('.') and e.name not in ('.gitignore','.env.example'):continue
                    if e.name in skip:continue
                    rel=str(e.relative_to(Path(root))).replace('\\','/')
                    items.append({'path':rel,'name':e.name,'is_dir':e.is_dir(),'size':e.stat().st_size if e.is_file() else 0,'depth':d})
                    if e.is_dir() and d<depth:walk(e,d+1)
                    if len(items)>=limit:break
            except PermissionError:pass
        walk(target,0)
        return {'root':str(Path(root)),'cwd':str(target),'items':items[:limit],'truncated':len(items)>=limit}
    @app.delete('/lessons/{idx}')
    def del_lesson(idx:int):
        sl=getattr(adam,'sem_lut',None)
        if sl is None or not hasattr(sl,'_raw') or idx<0 or idx>=len(sl._raw):raise HTTPException(status_code=404,detail='lesson idx out of range')
        removed=sl._raw.pop(idx)
        try:sl.fit()
        except Exception:pass
        try:adam.save_lessons()
        except Exception:pass
        return {'deleted':{'q':removed[0][:200],'a':removed[1][:200]},'lessons_n':len(sl._raw)}
    ollama_compat.mount(app,agent)
    mcp.mount(app,agent)
    web.mount(app)
    print(f'[amni_serve] serving on http://{args.host}:{args.port}',flush=True)
    print(f'[amni_serve]   browser UI:    http://{args.host}:{args.port}/',flush=True)
    print(f'[amni_serve]   Ollama compat: http://{args.host}:{args.port}/api/tags',flush=True)
    print(f'[amni_serve]   MCP server:    http://{args.host}:{args.port}/mcp',flush=True)
    print(f'[amni_serve]   Personas:      http://{args.host}:{args.port}/personas',flush=True)
    uvicorn.run(app,host=args.host,port=args.port,log_level='info')
if __name__=='__main__':main()
