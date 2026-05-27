"""coding_runner — the conductor of Adam's software-engineer loop.
prepare(task) bundles the REVIEW (prior attempts from coding_ledger) + LOCATE (relevant files from code_index)
into one work order with the attempt number; the agent then EDITs/TESTs via the existing gated skills; complete()
records the outcome to coding_ledger and tells you whether a retry is warranted (and hands back the lesson so
attempt N+1 starts smarter). The runner never edits the disk itself — writes go through the propose->confirm PC rails."""
import time,uuid,threading
from pathlib import Path
from typing import Dict,Any,List,Optional
from amni.serve import reffelt_tag as _rt
_RUNS:Dict[str,Dict[str,Any]]={}
_LOCK=threading.Lock()
_TTL=3600.0
def _runs_log()->Path:
    p=Path(__file__).resolve().parents[2]/'data'/'coding_runs.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _log(rec:Dict[str,Any]):
    try:
        import json
        with _runs_log().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception:pass
def _locate(task:str,limit:int=6)->List[Dict[str,Any]]:
    try:from amni.serve import code_index as _ci
    except Exception:return []
    tags=_rt.salient_tags(task)
    hits={};order=[]
    for term in tags[:8]:
        try:q=_ci.query(term,limit=4)
        except Exception:continue
        for h in (q.get('symbols') or []):
            key=h.get('path')
            if key and key not in hits:hits[key]={'path':key,'via':h.get('symbol'),'lang':h.get('lang')};order.append(key)
        for h in (q.get('files') or []):
            key=h.get('path')
            if key and key not in hits:hits[key]={'path':key,'via':'path','lang':h.get('lang')};order.append(key)
        if len(order)>=limit:break
    return [hits[k] for k in order[:limit]]
def prepare(task:str,agent=None,max_attempts:int=3)->Dict[str,Any]:
    task=(task or '').strip()
    if not task:return {'error':'task required'}
    tags=_rt.salient_tags(task);nonce=_rt.nonce(tags)
    prior=[];attempt_n=1
    try:
        from amni.serve import coding_ledger as _cl
        prior=_cl.recall(task,k=3);attempt_n=_cl.attempts_for(task)+1
    except Exception:pass
    located=_locate(task)
    ctx_lines=[f'TASK (attempt #{attempt_n} of up to {max_attempts}): {task}']
    if prior:
        ctx_lines.append('PRIOR ATTEMPTS — do better than these:')
        for h in prior:
            st='✓' if h.get('success') else ('✗' if h.get('success') is False else '?')
            seg=f"  - #{h.get('attempt','?')} {st}"
            if h.get('approach'):seg+=f" approach: {h['approach'][:100]}"
            if h.get('errors'):seg+=f" | errors: {'; '.join(h['errors'][:2])[:140]}"
            if h.get('lesson'):seg+=f" | LESSON: {h['lesson'][:140]}"
            ctx_lines.append(seg)
    if located:
        ctx_lines.append('RELEVANT FILES (from the code map):')
        for f in located:ctx_lines.append(f"  - {f['path']} (via {f.get('via')})")
    else:
        ctx_lines.append('No code map yet — run code_index build first to locate relevant files.')
    run_id='cr_'+uuid.uuid4().hex[:12]
    rec={'run_id':run_id,'task':task[:500],'attempt':attempt_n,'max_attempts':max_attempts,'nonce':nonce,'started':time.time(),'status':'open'}
    with _LOCK:_RUNS[run_id]=rec
    _log({**rec,'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'event':'prepare','located':[f['path'] for f in located]})
    return {'run_id':run_id,'task':task,'attempt':attempt_n,'max_attempts':max_attempts,'prior_attempts':prior,'located_files':located,'context':'\n'.join(ctx_lines)}
def complete(run_id:str,success:bool,outcome:str='',errors:Optional[List[str]]=None,lesson:str='',approach:str='',files:Optional[List[str]]=None,agent=None)->Dict[str,Any]:
    with _LOCK:run=_RUNS.pop(run_id,None)
    if run is None:return {'error':f'no open run {run_id!r} (expired or already completed)'}
    rec_out={}
    try:
        from amni.serve import coding_ledger as _cl
        rec_out=_cl.record(task=run['task'],outcome=outcome,approach=approach,errors=errors,lesson=lesson,success=bool(success),files=files)
    except Exception as e:rec_out={'recorded':False,'error':str(e)}
    will_retry=(not success) and run['attempt']<run['max_attempts']
    next_hint=''
    if will_retry:
        try:
            from amni.serve import coding_ledger as _cl
            next_hint=_cl.brief(run['task'])
        except Exception:pass
    _log({'run_id':run_id,'task':run['task'][:300],'attempt':run['attempt'],'status':'success' if success else 'failed','iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'event':'complete','will_retry':will_retry})
    return {'completed':True,'run_id':run_id,'attempt':run['attempt'],'success':bool(success),'recorded':rec_out.get('recorded',False),'will_retry':will_retry,'next_hint':next_hint}
def _errors_from_test(tr:Dict[str,Any])->List[str]:
    import re as _re
    if tr.get('error'):return [str(tr['error'])[:300]]
    out=((tr.get('stdout','') or '')+'\n'+(tr.get('stderr','') or ''))
    errs=[]
    for ln in out.splitlines():
        s=ln.strip()
        if len(s)>5 and _re.search(r'(?i)(?:\bFAILED\b|AssertionError|\bError:|error\[|panicked|\bFAIL\b|traceback|\bE\s{2,}|✗)',s):
            errs.append(s[:200])
        if len(errs)>=6:break
    return errs
def complete_from_test(run_id:str,test_result:Dict[str,Any],lesson:str='',approach:str='',files:Optional[List[str]]=None,agent=None)->Dict[str,Any]:
    """Objective completion: success is decided by the tests, not self-assessment. tests pass -> success; failures -> extracted as the errors to learn from."""
    tr=test_result or {}
    success=bool(tr.get('passed')) and not tr.get('error')
    errors=[] if success else _errors_from_test(tr)
    outcome=('tests passed' if success else 'tests failed')+(f" rc={tr.get('returncode')}" if tr.get('returncode') is not None else '')+(f" via {tr.get('flavor')}" if tr.get('flavor') else '')
    if not lesson and not success and errors:lesson=f'address first failure: {errors[0][:140]}'
    out=complete(run_id,success=success,outcome=outcome,errors=errors,lesson=lesson,approach=approach,files=files,agent=agent)
    return {**out,'objective':True,'tests_passed':success}
def status(run_id:str)->Dict[str,Any]:
    with _LOCK:run=_RUNS.get(run_id)
    return dict(run) if run else {'error':f'no open run {run_id!r}'}
def list_runs()->Dict[str,Any]:
    now=time.time()
    with _LOCK:
        for k in [k for k,v in _RUNS.items() if now-v['started']>_TTL]:_RUNS.pop(k,None)
        return {'open':[{'run_id':v['run_id'],'task':v['task'][:120],'attempt':v['attempt'],'age_s':round(now-v['started'],1)} for v in _RUNS.values()]}
