"""Memory inspector HTTP endpoints — surface what Adam knows across all atlases for the /jarvis memory panel.
Read-only by default; explicit POST /memory/forget for destructive ops with strict scoping.
mount(app, agent) wires:
  GET  /memory/snapshot        — top-level stats across all atlases
  GET  /memory/profile         — PersonalAtlas facts + pending clarifications
  GET  /memory/kg              — KG top subjects/predicates + stats
  GET  /memory/coach           — coach topics + mastery
  GET  /memory/daemon          — LearningDaemon stats
  POST /memory/forget          — atlas-scoped delete {atlas, pattern?, fact?, id?, topic?, confirm:true}
  POST /memory/confirm         — pending clarification confirm {id, is_confidential}"""
def mount(app,agent):
    from fastapi import Request,HTTPException
    from fastapi.responses import JSONResponse
    @app.get('/memory/snapshot')
    def snapshot():
        sl=getattr(agent.adam,'sem_lut',None)
        lessons_n=len(getattr(sl,'_raw',[]) or []) if sl is not None else 0
        return {'lesson_bank':{'n':lessons_n},'personal_atlas':agent.personal_atlas.stats() if getattr(agent,'personal_atlas',None) is not None else None,'coach_atlas':{'topics':agent.coach_atlas.list_topics() if getattr(agent,'coach_atlas',None) is not None else []},'knowledge_graph':agent.knowledge_graph.stats() if getattr(agent,'knowledge_graph',None) is not None else None,'learning_daemon':agent.learning_daemon.stats() if getattr(agent,'learning_daemon',None) is not None else None,'conversation_atlas':agent.atlas.stats() if getattr(agent,'atlas',None) is not None else None,'scheduler':agent.scheduler.atlas.stats() if getattr(agent,'scheduler',None) is not None else None}
    @app.get('/memory/profile')
    def profile(limit:int=100,include_confidential:bool=True):
        if getattr(agent,'personal_atlas',None) is None:return {'facts':[],'pending':[],'stats':{}}
        return {'facts':agent.personal_atlas.list_facts(include_confidential=include_confidential,limit=limit),'pending':agent.personal_atlas.pending_clarifications(limit=10),'stats':agent.personal_atlas.stats()}
    @app.get('/memory/kg')
    def kg(limit:int=20):
        if getattr(agent,'knowledge_graph',None) is None:return {'stats':{},'top_subjects':[]}
        kg=agent.knowledge_graph
        with kg._lock:
            by_deg=sorted(((s,len(ks)) for s,ks in kg._by_subject.items()),key=lambda x:-x[1])[:limit]
            top_preds=sorted(((p,len(ks)) for p,ks in kg._by_predicate.items()),key=lambda x:-x[1])[:limit]
        return {'stats':kg.stats(),'top_subjects':[{'subject':s,'edges_out':n} for s,n in by_deg],'top_predicates':[{'predicate':p,'count':n} for p,n in top_preds]}
    @app.get('/memory/coach')
    def coach(limit:int=50):
        if getattr(agent,'coach_atlas',None) is None:return {'topics':[],'streak':{}}
        atlas=agent.coach_atlas
        return {'topics':atlas.list_topics()[:limit],'streak':atlas.streak_stats() if hasattr(atlas,'streak_stats') else {}}
    @app.get('/memory/self-improvement')
    def self_improvement_list(status:str='',category:str='',limit:int=50,include_history:bool=False):
        from amni.serve.self_improvement import list_proposals,stats
        return {'proposals':list_proposals(status=status or None,category=category or None,limit=limit,include_history=include_history),'stats':stats()}
    @app.post('/memory/self-improvement')
    async def self_improvement_propose(req:Request):
        from amni.serve.self_improvement import propose
        body=await req.json()
        return propose(title=body.get('title',''),rationale=body.get('rationale',''),planned_change=body.get('planned_change',''),files_touched=body.get('files_touched',[]),category=body.get('category','enhancement'),author=body.get('author','user'))
    @app.post('/memory/self-improvement/{pid}/status')
    async def self_improvement_status(pid:str,req:Request):
        from amni.serve.self_improvement import transition
        body=await req.json()
        new_status=(body.get('status') or '').strip().lower()
        if not new_status:raise HTTPException(400,'need status')
        return transition(pid,new_status,notes=body.get('notes',''),author=body.get('author','user'))
    @app.get('/memory/self-reflection')
    def self_reflection_status():
        from amni.serve.self_reflection import status as _sr_status
        return _sr_status()
    @app.post('/memory/self-reflection/run')
    async def self_reflection_run(req:Request):
        from amni.serve.self_reflection import run_cycle
        body={}
        try:body=await req.json()
        except Exception:pass
        return run_cycle(force=bool(body.get('force',False)),dry_run=bool(body.get('dry_run',False)),notify=bool(body.get('notify',True)))
    @app.post('/memory/self-reflection/toggle')
    async def self_reflection_toggle(req:Request):
        from amni.serve.self_reflection import set_enabled
        body=await req.json();return set_enabled(bool(body.get('enabled',True)))
    @app.get('/memory/proposal-attempt/handlers')
    def proposal_attempt_handlers():
        from amni.serve.proposal_attempter import list_handlers
        return {'handlers':list_handlers()}
    @app.post('/memory/proposal-attempt/{pid}')
    async def proposal_attempt_one(pid:str,req:Request):
        from amni.serve.proposal_attempter import attempt
        body={}
        try:body=await req.json()
        except Exception:pass
        return attempt(pid,dry_run=bool(body.get('dry_run',False)),notify=bool(body.get('notify',True)))
    @app.post('/memory/proposal-attempt/next')
    async def proposal_attempt_next(req:Request):
        from amni.serve.proposal_attempter import attempt_next_eligible
        body={}
        try:body=await req.json()
        except Exception:pass
        return attempt_next_eligible(max_attempts=int(body.get('max',1)),dry_run=bool(body.get('dry_run',False)))
    @app.get('/memory/coach/reviews')
    def coach_reviews(topic:str='',limit:int=20):
        if getattr(agent,'coach_atlas',None) is None:return {'reviews':[]}
        atlas=agent.coach_atlas
        if not hasattr(atlas,'due_reviews'):return {'reviews':[]}
        return {'reviews':atlas.due_reviews(topic=(topic or None),limit=limit)}
    @app.get('/memory/coach/topic/{topic}/export.{fmt}')
    def coach_export(topic:str,fmt:str):
        from fastapi.responses import PlainTextResponse
        if getattr(agent,'coach_atlas',None) is None:raise HTTPException(404,'CoachAtlas not initialized')
        if fmt not in ('md','txt','json'):raise HTTPException(400,f'unknown fmt {fmt!r}; use md|txt|json')
        atlas=agent.coach_atlas
        if not hasattr(atlas,'export_topic'):raise HTTPException(501,'export_topic not implemented in this build')
        actual_fmt='anki' if fmt=='txt' else fmt
        out=atlas.export_topic(topic,fmt=actual_fmt)
        if out['count']==0:raise HTTPException(404,f'no practice history for topic {topic!r}')
        ct={'md':'text/markdown; charset=utf-8','txt':'text/plain; charset=utf-8','json':'application/json'}[fmt]
        return PlainTextResponse(content=out['content'],media_type=ct,headers={'Content-Disposition':f'attachment; filename="{out["filename"]}"'})
    @app.get('/memory/daemon')
    def daemon():
        if getattr(agent,'learning_daemon',None) is None:return {'enabled':False,'reason':'no daemon'}
        return agent.learning_daemon.stats()
    @app.get('/memory/needs-testing')
    def needs_testing(limit:int=50,include_done:bool=False):
        from amni.serve.edit_verifier import list_needs_testing
        items=list_needs_testing(limit=limit,include_done=include_done)
        return {'items':items,'count':len(items),'pending':len([i for i in items if i.get('status')=='pending'])}
    @app.post('/memory/needs-testing/done')
    async def needs_testing_done(req:Request):
        from amni.serve.edit_verifier import mark_needs_testing_done
        body=await req.json();sub=(body.get('path_substring') or '').strip()
        if not sub:raise HTTPException(400,'need path_substring')
        n=mark_needs_testing_done(sub);return {'marked_done':n,'path_substring':sub}
    @app.get('/memory/shell-history')
    def shell_history(limit:int=50,errors_only:bool=False,kind:str=''):
        from amni.serve.shell_audit import list_shell_history,shell_history_stats
        items=list_shell_history(limit=limit,errors_only=errors_only,kind=kind or None)
        return {'items':items,'count':len(items),'stats':shell_history_stats()}
    @app.post('/memory/forget')
    async def forget(req:Request):
        body=await req.json()
        atlas=(body.get('atlas') or '').strip().lower()
        if not body.get('confirm'):return JSONResponse(status_code=400,content={'error':'must include confirm:true to delete'})
        if atlas=='personal':
            if getattr(agent,'personal_atlas',None) is None:raise HTTPException(404,'PersonalAtlas not initialized')
            if body.get('forget_all'):return agent.personal_atlas.forget(forget_all=True)
            return agent.personal_atlas.forget(fact_pattern=body.get('pattern'))
        if atlas=='kg':
            if getattr(agent,'knowledge_graph',None) is None:raise HTTPException(404,'KnowledgeGraph not initialized')
            n=agent.knowledge_graph.forget(subject=body.get('subject'),predicate=body.get('predicate'),object_=body.get('object'))
            return {'atlas':'kg','forgot':n}
        if atlas=='coach':
            if getattr(agent,'coach_atlas',None) is None:raise HTTPException(404,'CoachAtlas not initialized')
            ok=agent.coach_atlas.forget(body.get('topic',''))
            return {'atlas':'coach','forgot':bool(ok)}
        if atlas=='conversation':
            if getattr(agent,'atlas',None) is None:raise HTTPException(404,'ConversationAtlas not initialized')
            sid=body.get('session_id','')
            if not sid:raise HTTPException(400,'need session_id')
            return {'atlas':'conversation','forgot':agent.atlas.forget_session(sid)}
        raise HTTPException(400,f'unknown atlas {atlas}; valid: personal|kg|coach|conversation')
    @app.post('/memory/confirm')
    async def confirm(req:Request):
        body=await req.json()
        if getattr(agent,'personal_atlas',None) is None:raise HTTPException(404,'PersonalAtlas not initialized')
        fid=(body.get('id') or '').strip()
        is_conf=bool(body.get('is_confidential'))
        if not fid:raise HTTPException(400,'need id')
        return agent.personal_atlas.confirm_clarification(fid,is_conf)
