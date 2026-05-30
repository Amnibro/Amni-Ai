"""coding_ledger — so Adam's SECOND attempt at a task does better than the first.
Every coding attempt records task -> approach -> outcome -> errors/debug -> lesson, append-only to
data/coding_attempts.jsonl AND committed to a PTEX file (lessons/coding_attempts_ptex). Before a retry,
recall() surfaces prior attempts on the same task (Reffelt-nonce + tag match) so Adam sees exactly what failed
last time and why — then changes approach. This is the self-learning loop for the software-engineer north star."""
import time,json,uuid,threading,re,os
from pathlib import Path
from typing import Dict,Any,List,Optional
from amni.serve import reffelt_tag as _rt
_LOCK=threading.Lock()
_FED_MAX_ENTRIES=int(os.environ.get('AMNI_FED_MAX_ENTRIES','500'))
_FED_MAX_LESSON_BYTES=int(os.environ.get('AMNI_FED_MAX_LESSON_BYTES','20000'))
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _ledger_path()->Path:
    p=_repo_root()/'data'/'coding_attempts.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _ptex_path()->Path:
    p=_repo_root()/'lessons'/'coding_attempts_ptex';p.parent.mkdir(parents=True,exist_ok=True);return p
def _commit_state_path()->Path:return _repo_root()/'data'/'coding_commit_state.json'
def _read_all()->List[Dict[str,Any]]:
    p=_ledger_path()
    if not p.exists():return []
    out=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:out.append(json.loads(ln))
            except Exception:continue
    except Exception:return []
    return out
def record(task:str,outcome:str='',approach:str='',files:Optional[List[str]]=None,errors:Optional[List[str]]=None,lesson:str='',success:Optional[bool]=None,session_id:str='')->Dict[str,Any]:
    task=(task or '').strip()
    if not task:return {'recorded':False,'reason':'task required'}
    tags=_rt.salient_tags(task);nonce=_rt.nonce(tags)
    prior=[r for r in _read_all() if r.get('nonce')==nonce]
    attempt_n=len(prior)+1
    rec={'id':'ca_'+uuid.uuid4().hex[:12],'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'task':task[:500],'attempt':attempt_n,'approach':(approach or '')[:600],'outcome':(outcome or '')[:600],'errors':[str(e)[:300] for e in (errors or [])][:8],'lesson':(lesson or '')[:500],'success':(None if success is None else bool(success)),'files':[str(f)[:200] for f in (files or [])][:20],'tags':tags,'nonce':nonce,'session_id':(session_id or '')[:64]}
    try:
        with _LOCK:
            with _ledger_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
    except Exception as e:return {'recorded':False,'error':str(e)}
    return {'recorded':True,'id':rec['id'],'attempt':attempt_n,'prior_attempts':len(prior)}
def recall(task:str,k:int=3)->List[Dict[str,Any]]:
    task=(task or '').strip()
    if not task:return []
    qtags=_rt.salient_tags(task);qn=_rt.nonce(qtags)
    scored=[]
    for r in _read_all():
        rel=_rt.relevance(qtags,qn,r.get('tags') or [],r.get('nonce') if isinstance(r.get('nonce'),int) else _rt.nonce(r.get('tags') or []))
        if rel>0.05:scored.append((rel,r))
    scored.sort(key=lambda x:(-x[0],-(x[1].get('ts') or 0)))
    return [{'task':r['task'],'attempt':r.get('attempt'),'success':r.get('success'),'outcome':r.get('outcome',''),'errors':r.get('errors',[]),'lesson':r.get('lesson',''),'approach':r.get('approach',''),'federated':bool(r.get('federated')),'rel':round(rel,3)} for rel,r in scored[:k]]
def synthesize(task:str,k:int=10)->str:
    """When Adam has failed the same kind of task repeatedly, don't just echo the last error — read ALL prior
    attempts and INFER what's actually required: distinct failure modes, approaches already burned, recorded lessons.
    The model then forms a hypothesis from these notes and changes approach. This is the 'take notes' step for when
    the mission shifts and a single retry won't cut it."""
    hits=[h for h in recall(task,k=k) if h.get('success') is False]
    if not hits:return ''
    errs=[];seen=set()
    for h in hits:
        for e in (h.get('errors') or []):
            sig=re.sub(r'\d+','N',str(e))[:70].lower()
            if sig not in seen:seen.add(sig);errs.append(str(e)[:140])
    approaches=[h['approach'] for h in hits if h.get('approach')]
    lessons=[h['lesson'] for h in hits if h.get('lesson')]
    lines=[f'NOTES from {len(hits)} prior FAILED attempt(s) — you have failed this repeatedly, so INFER the real requirement from these memories and change approach fundamentally:']
    if errs:lines.append('  distinct failures seen: '+' | '.join(errs[:5]))
    if approaches:lines.append('  approaches already tried (do something DIFFERENT): '+'; '.join(a[:60] for a in approaches[:4]))
    if lessons:lines.append('  lessons recorded: '+'; '.join(l[:80] for l in lessons[:4]))
    lines.append('  Form a hypothesis about what is actually being asked, then solve it a new way that addresses every failure above.')
    return '\n'.join(lines)
def attempts_for(task:str)->int:
    qn=_rt.nonce(_rt.salient_tags(task or ''))
    return sum(1 for r in _read_all() if r.get('nonce')==qn and not r.get('federated'))
def brief(task:str,k:int=3)->str:
    hits=recall(task,k=k)
    if not hits:return ''
    lines=[f'PRIOR ATTEMPTS on a similar task ({len(hits)}) — learn from these, change approach if one failed:']
    for h in hits:
        st='✓ worked' if h.get('success') else ('✗ failed' if h.get('success') is False else '? unknown')
        bit=f"- attempt #{h.get('attempt','?')} {st}"
        if h.get('approach'):bit+=f" · approach: {h['approach'][:120]}"
        if h.get('errors'):bit+=f" · errors: {'; '.join(h['errors'][:2])[:160]}"
        if h.get('lesson'):bit+=f" · LESSON: {h['lesson'][:160]}"
        lines.append(bit)
    return '\n'.join(lines)
def commit_to_ptex(adam=None,save:bool=True,max_pairs:int=400)->Dict[str,Any]:
    rows=_read_all()
    if not rows:return {'committed':0,'reason':'no attempts'}
    pairs=[(r.get('task',''),(r.get('lesson') or r.get('outcome') or '').strip() or '[no lesson recorded]') for r in rows[-int(max_pairs):] if r.get('task')]
    if not pairs:return {'committed':0,'reason':'no pairs'}
    try:
        from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
        enc=getattr(getattr(adam,'sem_lut',None),'encoder',None) if adam is not None else None
        if enc is None:return {'committed':0,'reason':'no encoder (boot Adam to commit coding-ledger PTEX)'}
        lut=SemanticPTEXLUT(grid=48,pca_dim=6,encoder=enc)
    except Exception as e:return {'committed':0,'error':f'lut unavailable: {e}'}
    for q,a in pairs:
        try:lut.add(q,a)
        except Exception:pass
    if save:
        try:lut.fit();lut.save(str(_ptex_path()))
        except Exception as e:return {'committed':len(pairs),'fit_or_save_error':str(e)[:160]}
    return {'committed':len(pairs),'ptex':str(_ptex_path())}
def maybe_commit_to_ptex(adam=None,min_new:int=3)->Dict[str,Any]:
    rows=_read_all();total=len(rows)
    last=0;sp=_commit_state_path()
    if sp.exists():
        try:last=int(json.loads(sp.read_text(encoding='utf-8')).get('committed_total',0))
        except Exception:last=0
    if total-last<int(min_new):return {'committed':0,'reason':f'only {total-last} new (<{min_new})','total':total}
    r=commit_to_ptex(adam=adam,save=True)
    if r.get('committed',0)>0 and 'fit_or_save_error' not in r:
        try:sp.write_text(json.dumps({'committed_total':total,'ts':time.time()}),encoding='utf-8')
        except Exception:pass
    return {**r,'total':total,'new_since_last':total-last}
def _defederate_paths(s:str)->str:
    s=re.sub(r'(?i)[A-Za-z]:\\Users\\[^\\\s]+','<path>',s)
    s=re.sub(r'/(?:home|Users)/[^/\s]+','<path>',s)
    return s
def federation_export(limit:int=200,only_success:bool=True)->Dict[str,Any]:
    """Share WHAT WORKED across Adam instances without leaking. Exports only the scrubbed LESSON + generic tags +
    Reffelt nonce — never the raw task, file paths, or error traces (which can carry proprietary code / PII).
    Successful lessons only by default. Double-scrubbed: pii_egress patterns + user-home path stripping."""
    try:from amni.serve.pii_egress import scrub as _scrub
    except Exception:_scrub=lambda t,**k:t
    out=[];seen=set()
    for r in _read_all():
        if only_success and r.get('success') is not True:continue
        lesson=(r.get('lesson') or '').strip()
        if not lesson:continue
        scrubbed=_defederate_paths(_scrub(lesson,source='federation'))
        key=(r.get('nonce'),scrubbed.lower())
        if not scrubbed or key in seen:continue
        seen.add(key)
        out.append({'tags':(r.get('tags') or [])[:8],'nonce':r.get('nonce'),'lesson':scrubbed[:300],'success':True})
        if len(out)>=int(limit):break
    return {'federable':out,'n':len(out),'source':'coding_ledger','note':'lessons only; raw tasks/paths/errors never exported'}
def federation_import(entries:List[Dict[str,Any]],source:str='peer')->Dict[str,Any]:
    """Receive a peer's scrubbed successful lessons into the local ledger, marked federated (so they help recall but
    never count as first-party attempts). Re-scrubbed on the way in — peers aren't trusted blindly. Idempotent (nonce+lesson dedupe)."""
    try:from amni.serve.pii_egress import scrub as _scrub
    except Exception:_scrub=lambda t,**k:t
    try:from amni.serve.federated import verify_incoming as _verify
    except Exception:_verify=None
    existing=set((r.get('nonce'),(r.get('lesson') or '').lower()) for r in _read_all())
    added=0;skipped=0;rejected_unsafe=0;oversize=0;lines=[]
    _all=entries or [];capped=max(0,len(_all)-_FED_MAX_ENTRIES) if isinstance(_all,list) else 0
    for e in (_all[:_FED_MAX_ENTRIES] if isinstance(_all,list) else []):
        if not isinstance(e,dict):skipped+=1;continue
        lesson=(e.get('lesson') or '').strip();tags=(e.get('tags') if isinstance(e.get('tags'),list) else [])[:16]
        if not lesson:skipped+=1;continue
        if len(lesson)>_FED_MAX_LESSON_BYTES:oversize+=1;continue
        if _verify is not None:
            _ok,_sq,_sa,_reasons=_verify('federated lesson',lesson)
            if not _ok:rejected_unsafe+=1;continue
            lesson=(_sa or '').strip()
            if not lesson:skipped+=1;continue
        lesson=_defederate_paths(_scrub(lesson,source='federation_import'))[:300]
        nonce=e.get('nonce') if isinstance(e.get('nonce'),int) else _rt.nonce(tags or _rt.salient_tags(lesson))
        key=(nonce,lesson.lower())
        if not lesson or key in existing:skipped+=1;continue
        existing.add(key)
        rec={'id':'cf_'+uuid.uuid4().hex[:12],'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'task':'[federated] '+', '.join((tags or [])[:6]),'attempt':0,'approach':'','outcome':'federated lesson','errors':[],'lesson':lesson,'success':True,'files':[],'tags':(tags or [])[:16],'nonce':nonce,'federated':True,'source':str(source)[:40],'session_id':''}
        lines.append(rec);added+=1
    if lines:
        try:
            with _LOCK:
                with _ledger_path().open('a',encoding='utf-8') as fh:
                    for r in lines:fh.write(json.dumps(r,ensure_ascii=False)+'\n')
        except Exception as ex:return {'imported':0,'skipped':skipped,'rejected_unsafe':rejected_unsafe,'oversize':oversize,'capped':capped,'error':str(ex)}
    return {'imported':added,'skipped':skipped,'rejected_unsafe':rejected_unsafe,'oversize':oversize,'capped':capped,'source':source}
def stats()->Dict[str,Any]:
    rows=_read_all()
    fed=sum(1 for r in rows if r.get('federated'));own=[r for r in rows if not r.get('federated')]
    succ=sum(1 for r in own if r.get('success') is True);fail=sum(1 for r in own if r.get('success') is False)
    retried=len([r for r in own if (r.get('attempt') or 1)>1])
    return {'total':len(own),'succeeded':succ,'failed':fail,'unknown':len(own)-succ-fail,'retried_attempts':retried,'federated':fed,'last_iso':rows[-1].get('iso') if rows else None}
