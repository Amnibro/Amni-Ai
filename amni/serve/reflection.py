"""Self-reflection daemon — Adam picks low-confidence / stale lessons, re-researches via web,
cross-source verifies, updates lesson if improved, leaves alone if regression. No PII, no external sharing.
Run as `amni reflect` (interval) or `amni reflect --once`."""
import time,json,re,os
from pathlib import Path
from typing import Dict,Any,Optional,List,Tuple
_LOG=Path('logs/reflection.jsonl')
def _audit(rec:Dict[str,Any]):
    _LOG.parent.mkdir(parents=True,exist_ok=True)
    try:
        with open(_LOG,'a',encoding='utf-8') as f:f.write(json.dumps(rec,default=str)+'\n')
    except Exception:pass
def _pick_targets(adam,max_n:int=5,min_age_sec:int=86400)->List[Tuple[int,str,str]]:
    sl=getattr(adam,'sem_lut',None)
    raw=getattr(sl,'_raw',[]) if sl is not None else []
    if not raw:return []
    lut=getattr(getattr(adam,'adam',None),'lut',None)
    cutoff=time.time()-min_age_sec
    cands=[]
    for i,(q,a) in enumerate(raw):
        if not q or not a or len(a)<5:continue
        if q.startswith('PERSONA::') or q.startswith('What does ') or q.startswith('Who is the persona '):continue
        if 'mock' in q.lower() or '<EMAIL>' in a or '<PATH>' in a:continue
        meta=None
        try:
            if lut is not None and hasattr(lut,'lookup'):
                m=lut.lookup(q)
                meta=m.get('meta') if isinstance(m,dict) else None
        except Exception:pass
        ts=(meta or {}).get('ts',0)
        conf=(meta or {}).get('conf',0.5)
        score=(1-conf)*0.7+(0.3 if ts==0 or ts<cutoff else 0)
        cands.append((score,i,q,a))
    cands.sort(key=lambda x:-x[0])
    return [(i,q,a) for _,i,q,a in cands[:max_n]]
def _research(adam,q:str,timeout_s:float=60.0)->Optional[str]:
    crawler=getattr(getattr(adam,'adam',None),'crawler',None)
    if crawler is None:return None
    try:ans,sources,_=crawler.crawl_and_distill(q,subject=None,letter_only=False)
    except Exception:return None
    return (ans or '').strip() or None
def _judge(adam,q:str,old_a:str,new_a:str)->Tuple[str,str]:
    svc=getattr(getattr(adam,'adam',None),'svc',None)
    if svc is None or not new_a:return ('keep_old','no svc')
    if not old_a:return ('use_new','old empty')
    if old_a.strip()==new_a.strip():return ('keep_old','identical')
    sys_p='You are an answer-quality judge. Given a question and two candidate answers, output ONE of: BETTER_NEW (new is more accurate/complete), BETTER_OLD (old is more accurate/complete), or AGREE (functionally equivalent).'
    prompt=f'Question: {q}\n\nOLD answer:\n{old_a[:500]}\n\nNEW answer:\n{new_a[:500]}\n\nVerdict (single word):'
    try:resp,_=svc.chat(prompt,system=sys_p,max_new_tokens=8,do_sample=False,kb_top_k=0)
    except Exception as e:return ('keep_old',f'judge error: {e}')
    v=(resp or '').strip().upper()
    if 'BETTER_NEW' in v or v.startswith('NEW'):return ('use_new','judge: new is better')
    if 'BETTER_OLD' in v or v.startswith('OLD'):return ('keep_old','judge: old is better')
    if 'AGREE' in v:return ('keep_old','agree')
    return ('keep_old',f'inconclusive: {v[:40]}')
def reflect_once(adam,max_n:int=5,min_age_sec:int=86400)->Dict[str,Any]:
    t0=time.time()
    targets=_pick_targets(adam,max_n=max_n,min_age_sec=min_age_sec)
    sl=adam.sem_lut
    actions=[];n_updated=0;n_kept=0;n_failed=0
    for idx,q,old_a in targets:
        new_a=_research(adam,q)
        if not new_a:n_failed+=1;actions.append({'q':q[:80],'action':'skip','reason':'no web result'});continue
        verdict,reason=_judge(adam,q,old_a,new_a)
        rec={'q':q[:120],'old_a':old_a[:120],'new_a':new_a[:120],'verdict':verdict,'reason':reason,'ts':time.time()}
        if verdict=='use_new':
            try:sl._raw[idx]=(q,new_a);n_updated+=1;rec['action']='updated'
            except Exception as e:rec['action']='update_failed';rec['err']=str(e)
        else:n_kept+=1;rec['action']='kept'
        actions.append(rec);_audit(rec)
    if n_updated>0:
        try:sl.fit()
        except Exception:pass
        try:adam.save_lessons()
        except Exception:pass
    return {'cycle_wall_s':round(time.time()-t0,2),'targets':len(targets),'updated':n_updated,'kept':n_kept,'failed':n_failed,'actions':actions}
def reflect_loop(adam,interval_sec:int=300,max_per_cycle:int=5,min_age_sec:int=86400,one_shot:bool=False):
    print(f'[reflect] starting (interval={interval_sec}s, max_per_cycle={max_per_cycle}, min_age={min_age_sec}s, one_shot={one_shot})',flush=True)
    cycle=0
    while True:
        cycle+=1
        r=reflect_once(adam,max_n=max_per_cycle,min_age_sec=min_age_sec)
        print(f'[reflect cycle {cycle}] {r["targets"]} targets, {r["updated"]} updated, {r["kept"]} kept, {r["failed"]} failed in {r["cycle_wall_s"]}s',flush=True)
        for a in r['actions']:print(f'  {a["action"]:<8} q={a["q"]!r}',flush=True)
        if one_shot:break
        try:time.sleep(interval_sec)
        except KeyboardInterrupt:print('[reflect] interrupted, exiting.',flush=True);break
