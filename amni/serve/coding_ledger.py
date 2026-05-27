"""coding_ledger — so Adam's SECOND attempt at a task does better than the first.
Every coding attempt records task -> approach -> outcome -> errors/debug -> lesson, append-only to
data/coding_attempts.jsonl AND committed to a PTEX file (lessons/coding_attempts_ptex). Before a retry,
recall() surfaces prior attempts on the same task (Reffelt-nonce + tag match) so Adam sees exactly what failed
last time and why — then changes approach. This is the self-learning loop for the software-engineer north star."""
import time,json,uuid,threading
from pathlib import Path
from typing import Dict,Any,List,Optional
from amni.serve import reffelt_tag as _rt
_LOCK=threading.Lock()
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
    return [{'task':r['task'],'attempt':r.get('attempt'),'success':r.get('success'),'outcome':r.get('outcome',''),'errors':r.get('errors',[]),'lesson':r.get('lesson',''),'approach':r.get('approach',''),'rel':round(rel,3)} for rel,r in scored[:k]]
def attempts_for(task:str)->int:
    qn=_rt.nonce(_rt.salient_tags(task or ''))
    return sum(1 for r in _read_all() if r.get('nonce')==qn)
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
def stats()->Dict[str,Any]:
    rows=_read_all()
    succ=sum(1 for r in rows if r.get('success') is True);fail=sum(1 for r in rows if r.get('success') is False)
    retried=len([r for r in rows if (r.get('attempt') or 1)>1])
    return {'total':len(rows),'succeeded':succ,'failed':fail,'unknown':len(rows)-succ-fail,'retried_attempts':retried,'last_iso':rows[-1].get('iso') if rows else None}
