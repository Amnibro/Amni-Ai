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
    @app.get('/memory/skill-failures')
    def skill_failures_list(limit:int=20,skill:str=''):
        from amni.serve.skill_failures import recent,stats
        return {'failures':recent(limit=limit,skill_filter=skill or None),'stats':stats()}
    @app.get('/memory/reminders')
    def reminders_list(limit:int=50,session_id:str=''):
        from amni.serve.reminders import list_active,list_due,stats
        return {'reminders':list_active(session_id=session_id or None,limit=limit),'due':list_due(),'stats':stats()}
    @app.post('/memory/reminders')
    async def reminders_add(req:Request):
        from amni.serve.reminders import add
        body=await req.json()
        return add(text=body.get('text',''),due_at=body.get('due_at'),session_id=body.get('session_id',''))
    @app.post('/memory/reminders/{rid}/dismiss')
    def reminders_dismiss(rid:str):
        from amni.serve.reminders import dismiss
        return dismiss(rid)
    @app.get('/memory/bookmarks')
    def bookmarks_list(limit:int=20,session_id:str='',search:str=''):
        from amni.serve.bookmarks import list_recent,stats
        return {'bookmarks':list_recent(limit=limit,session_id=session_id or None,search=search or ''),'stats':stats()}
    @app.post('/memory/bookmarks')
    async def bookmarks_add(req:Request):
        from amni.serve.bookmarks import add
        body=await req.json()
        return add(session_id=body.get('session_id',''),user_msg=body.get('user_msg',''),bot_msg=body.get('bot_msg',''),note=body.get('note',''),tier=body.get('tier',''),persona=body.get('persona',''))
    @app.delete('/memory/bookmarks/{bid}')
    def bookmarks_delete(bid:str):
        from amni.serve.bookmarks import delete
        return delete(bid)
    @app.get('/memory/notes')
    def notes_list(limit:int=50,tag:str='',search:str='',session_id:str='',session_only:bool=False):
        from amni.serve.notes import list_recent,stats,all_tags
        return {'notes':list_recent(limit=limit,tag=tag or None,search=search or '',session_id=(session_id or None) if session_only else None),'tags':all_tags(),'stats':stats()}
    @app.post('/memory/notes')
    async def notes_add(req:Request):
        from amni.serve.notes import add
        body=await req.json()
        return add(text=body.get('text',''),tags=body.get('tags'),session_id=body.get('session_id',''))
    @app.delete('/memory/notes/{nid}')
    def notes_delete(nid:str):
        from amni.serve.notes import delete
        return delete(nid)
    @app.get('/memory/pii-egress')
    def pii_egress_audit(limit:int=50):
        from amni.serve.pii_egress import audit_stats
        return audit_stats(limit=limit)
    @app.get('/memory/thinking-leaks')
    def thinking_leaks(limit:int=30):
        from amni.serve.leak_ledger import stats
        return stats(limit=limit)
    @app.get('/memory/pc-actions')
    def pc_actions_audit(limit:int=30):
        from amni.serve.pc_actions import audit_recent,list_pending
        return {**audit_recent(limit=limit),'pending':list_pending().get('pending',[])}
    @app.get('/memory/coding-attempts')
    def coding_attempts(task:str='',limit:int=10):
        from amni.serve.coding_ledger import stats,recall
        return {**stats(),'recall':(recall(task,k=limit) if task else [])}
    @app.post('/memory/coding-run/prepare')
    async def coding_run_prepare(req:Request):
        from amni.serve.coding_runner import prepare
        body=await req.json()
        if not body.get('task'):raise HTTPException(400,'need task')
        return prepare(body['task'],agent=agent,max_attempts=int(body.get('max_attempts',3)))
    @app.get('/memory/coding-federation')
    def coding_federation_export(limit:int=200,only_success:bool=True):
        from amni.serve.coding_ledger import federation_export
        return federation_export(limit=limit,only_success=bool(only_success))
    @app.post('/memory/coding-federation')
    async def coding_federation_import(req:Request):
        from amni.serve.coding_ledger import federation_import
        body=await req.json()
        entries=body.get('federable') or body.get('entries') or []
        if not isinstance(entries,list):raise HTTPException(400,'need federable/entries list')
        return federation_import(entries,source=str(body.get('source') or 'http-peer'))
    @app.post('/memory/coding-federation/pull')
    async def coding_federation_pull(req:Request):
        import urllib.request,json as _json
        from amni.serve.coding_ledger import federation_import
        body=await req.json()
        url=(body.get('url') or '').strip()
        if not url:raise HTTPException(400,'need url of a peer Adam (its /memory/coding-federation)')
        peer=url.rstrip('/')+('/memory/coding-federation' if not url.rstrip('/').endswith('/memory/coding-federation') else '')
        try:
            with urllib.request.urlopen(peer,timeout=8) as resp:data=_json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:raise HTTPException(502,f'peer unreachable: {e}')
        return federation_import(data.get('federable') or [],source=str(body.get('source') or peer))
    @app.get('/memory/se-dashboard')
    def se_dashboard():
        from amni.serve.code_index import stats as _ci_stats
        from amni.serve.coding_ledger import stats as _cl_stats
        from amni.serve.coding_runner import list_runs as _cr_runs
        ci=_ci_stats();cl=_cl_stats()
        rate=round(100.0*cl.get('succeeded',0)/cl['total'],1) if cl.get('total') else 0.0
        return {'code_index':ci,'coding':{**cl,'success_rate_pct':rate},'open_runs':_cr_runs().get('open',[])}
    @app.post('/memory/coding-run/complete')
    async def coding_run_complete(req:Request):
        from amni.serve.coding_runner import complete
        body=await req.json()
        if not body.get('run_id'):raise HTTPException(400,'need run_id')
        return complete(body['run_id'],success=bool(body.get('success',False)),outcome=body.get('outcome',''),errors=body.get('errors'),lesson=body.get('lesson',''),approach=body.get('approach',''),files=body.get('files'),agent=agent)
    @app.get('/memory/review')
    def pre_response_review(q:str=''):
        from amni.serve.pre_response_review import review
        if not q:raise HTTPException(400,'need q (the message to review for)')
        return review(q,agent=agent)
    @app.post('/memory/thinking-leaks/commit')
    async def thinking_leaks_commit(req:Request):
        from amni.serve.leak_ledger import commit_to_ptex
        body={}
        try:body=await req.json()
        except Exception:pass
        return commit_to_ptex(adam=agent.adam if getattr(agent,'adam',None) is not None else None,save=bool(body.get('save',True)))
    @app.post('/memory/skill-failures/ack')
    def skill_failures_ack():
        from amni.serve.skill_failures import ack_all
        return ack_all()
    @app.get('/memory/metrics')
    def metrics_status():
        from amni.serve.metrics_snapshot import status as _s
        return _s()
    @app.get('/memory/metrics/history')
    def metrics_history(limit:int=30):
        from amni.serve.metrics_snapshot import history as _h
        return {'history':_h(limit=limit)}
    @app.get('/memory/metrics/trend')
    def metrics_trend(days:int=7):
        from amni.serve.metrics_snapshot import trend as _t
        return _t(days=days)
    @app.post('/memory/metrics/snapshot')
    async def metrics_snapshot_now(req:Request):
        from amni.serve.metrics_snapshot import snapshot as _snap
        body={}
        try:body=await req.json()
        except Exception:pass
        return _snap(force=bool(body.get('force',False)),notify=bool(body.get('notify',False)))
    @app.post('/memory/metrics/toggle')
    async def metrics_toggle(req:Request):
        from amni.serve.metrics_snapshot import set_enabled as _se
        body=await req.json();return _se(bool(body.get('enabled',True)))
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
