"""consensus — multi-source verification layer. When ≥N sources agree on a fact, promote to verified; when sources disagree, flag debated.
Uses LearningAtlas's per-cell metadata. Sits in the ingest path: every Q-A pair from a source first checks if a near-match exists in the LUT, and either reinforces it (confidence bump) or records a debate."""
import hashlib,re
from typing import List,Dict,Any,Optional
_WORD_RE=re.compile(r'\w+')
def _norm(s:str)->str:return ' '.join(_WORD_RE.findall((s or '').lower()))[:300]
def _jaccard(a:str,b:str)->float:
    sa=set(_WORD_RE.findall(a.lower()));sb=set(_WORD_RE.findall(b.lower()))
    if not sa and not sb:return 1.0
    if not sa or not sb:return 0.0
    return len(sa&sb)/len(sa|sb)
def find_match(sem_lut,question:str,min_jaccard:float=0.55)->Optional[Dict[str,Any]]:
    if sem_lut is None or not hasattr(sem_lut,'_raw') or not sem_lut._raw:return None
    qn=_norm(question);best=None;best_score=0.0
    for q,a in sem_lut._raw[-2000:]:
        s=_jaccard(qn,_norm(q))
        if s>best_score and s>=min_jaccard:best=(q,a);best_score=s
    if best is None:return None
    return {'q':best[0],'a':best[1],'jaccard':round(best_score,3)}
def ingest_with_consensus(adam,question:str,answer:str,source:str,learning_atlas,sem_lut=None)->Dict[str,Any]:
    if sem_lut is None and adam is not None:sem_lut=getattr(adam,'sem_lut',None)
    existing=find_match(sem_lut,question) if sem_lut is not None else None
    if existing is None:
        if adam is not None and hasattr(adam,'teach'):
            try:adam.teach(question,answer)
            except Exception as e:return {'outcome':'teach_failed','error':str(e)[:200]}
        if learning_atlas is not None:meta=learning_atlas.record(question,answer,source=source,kind='qa_extract')
        else:meta=None
        return {'outcome':'new','q':question[:120],'meta':meta}
    a_match=_jaccard(existing['a'],answer)
    if a_match>=0.5:
        if learning_atlas is not None:meta=learning_atlas.record(existing['q'],existing['a'],source=source,kind='reinforce')
        else:meta=None
        return {'outcome':'reinforced','q':existing['q'][:120],'jaccard_q':existing['jaccard'],'jaccard_a':round(a_match,3),'meta':meta}
    if learning_atlas is not None:
        meta=learning_atlas.mark_debated(existing['q'],existing['a'],answer,source=source)
        return {'outcome':'debated','q':existing['q'][:120],'jaccard_q':existing['jaccard'],'jaccard_a':round(a_match,3),'original_a':existing['a'][:120],'new_a':answer[:120],'meta':meta}
    return {'outcome':'conflict_no_atlas','q':existing['q'][:120]}
def ingest_qa_pairs_with_consensus(adam,pairs:List[Dict[str,Any]],source:str,learning_atlas)->Dict[str,Any]:
    counts={'new':0,'reinforced':0,'debated':0,'teach_failed':0,'other':0}
    samples=[]
    for p in pairs:
        out=ingest_with_consensus(adam,p['q'],p['a'],source,learning_atlas)
        kind=out.get('outcome','other')
        counts[kind if kind in counts else 'other']=counts.get(kind if kind in counts else 'other',0)+1
        if len(samples)<5:samples.append(out)
    return {'source':source,'pairs_n':len(pairs),'counts':counts,'samples':samples}
