"""pre_response_review — the single pass Adam runs BEFORE every response.
Gathers the whole reviewable substrate — learned leaks (avoid), past errors, relevant lessons, known facts —
addresses each by its Reffelt context-nonce, ranks by contextual-tag overlap (recall-safe), and assembles a
compact directive brief that is injected into the system prompt pre-generation. "Everything is a self-learning
loop": each response is shaped by what Adam has already learned, leaked, and gotten wrong near this context."""
from typing import Dict,Any,List,Optional
from amni.serve import reffelt_tag as _rt
def _safe(fn,default):
    try:return fn()
    except Exception:return default
def review(message:str,agent=None,max_errors:int=3,max_lessons:int=1)->Dict[str,Any]:
    qtags=_rt.salient_tags(message);qn=_rt.nonce(qtags)
    items={'leaks':{},'errors':[],'lessons':[],'knowledge':[]}
    leak_stats=_safe(lambda:__import__('amni.serve.leak_ledger',fromlist=['stats']).stats(limit=1),{'total':0,'distinct_signatures':0})
    items['leaks']={'total':leak_stats.get('total',0),'signatures':leak_stats.get('distinct_signatures',0)}
    def _errs():
        from amni.serve.skill_failures import recent
        rows=recent(limit=30) or []
        scored=[]
        for r in rows:
            itags=r.get('tags') or _rt.salient_tags((r.get('skill','')+' '+r.get('message',''))+' '+(r.get('error','') or '')[:60])
            inonce=r.get('nonce') if isinstance(r.get('nonce'),int) else _rt.nonce(itags)
            rel=_rt.relevance(qtags,qn,itags,inonce)
            if rel>0.05:scored.append((rel,r))
        scored.sort(key=lambda x:-x[0])
        return [{'skill':r.get('skill','?'),'error':(r.get('error') or '')[:90],'rel':round(rel,3)} for rel,r in scored[:max_errors]]
    items['errors']=_safe(_errs,[])
    def _lessons():
        sl=getattr(getattr(agent,'adam',None),'sem_lut',None) if agent is not None else None
        if sl is None or not getattr(sl,'_raw',None):return []
        hit=sl.lookup_soft(message,margin='auto')
        return [{'preview':(hit or '')[:120]}] if hit else []
    items['lessons']=_safe(_lessons,[])
    def _know():
        pa=getattr(agent,'personal_atlas',None) if agent is not None else None
        if pa is None or not hasattr(pa,'recall'):return []
        hits=pa.recall(message,k=3,include_confidential=True) or []
        return [{'fact':(h.get('fact') or '')[:80],'confidential':bool(h.get('is_confidential'))} for h in hits]
    items['knowledge']=_safe(_know,[])
    brief=build_brief(qn,qtags,items)
    return {'nonce':qn,'digits':_rt.decompose(qn),'tags':qtags,'brief':brief,'items':items}
def build_brief(qn:int,qtags:List[str],items:Dict[str,Any])->str:
    digits=_rt.decompose(qn)
    lines=[]
    lk=items.get('leaks') or {}
    if lk.get('total',0)>0:
        lines.append(f"AVOID: emit only the final answer — no internal reasoning, 'Thinking Process', step labels, or [bracketed] tool narration ({lk.get('total')} past leak(s) on record).")
    errs=items.get('errors') or []
    if errs:
        es='; '.join(f"{e['skill']}→{e['error'][:50]}" for e in errs[:3])
        lines.append(f"PAST ERRORS near this context: {es}. Don't repeat these failure modes.")
    les=items.get('lessons') or []
    if les and les[0].get('preview'):
        lines.append(f"RELEVANT LESSON on file: {les[0]['preview']}")
    kn=items.get('knowledge') or []
    if kn:
        lines.append(f"KNOWN about the user ({len(kn)} fact(s)) — use naturally, never echo confidential items to external tools.")
    if not lines:return ''
    head=f"[PRE-RESPONSE REVIEW · reffelt-nonce {'.'.join(str(d) for d in digits)}]"
    return head+"\n"+"\n".join(lines)
