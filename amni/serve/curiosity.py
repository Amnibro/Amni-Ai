"""curiosity — find embedding-space gaps in Adam's knowledge and generate gap topics.
Cells in sparse regions of the PTEX grid indicate under-represented topics. Cells with low confidence in LearningAtlas indicate unverified knowledge. Topics with low mastery in CoachAtlas indicate weak skill.
Returns ranked list of gap topics ready to feed into build_curriculum."""
import math,random,re
from typing import List,Dict,Any,Optional
def _find_sparse_neighborhoods(sem_lut,sample_n:int=200,radius:int=3)->List[Dict[str,Any]]:
    if sem_lut is None or not hasattr(sem_lut,'_cells') or not sem_lut._cells:return []
    cells=list(sem_lut._cells.items())
    if len(cells)<5:return []
    sample=random.sample(cells,min(sample_n,len(cells)))
    sparse=[]
    for cell_key,cell_val in sample:
        if not isinstance(cell_key,tuple):continue
        neighbors_in_radius=0
        for other_key in sem_lut._cells.keys():
            if other_key==cell_key:continue
            try:d=sum(abs(a-b) for a,b in zip(cell_key,other_key))
            except Exception:continue
            if d<=radius:neighbors_in_radius+=1
        if neighbors_in_radius<=1:
            q=cell_val.get('q','') if isinstance(cell_val,dict) else ''
            sparse.append({'cell':cell_key,'q':q[:200],'neighbors':neighbors_in_radius})
    sparse.sort(key=lambda x:x['neighbors'])
    return sparse[:20]
def _extract_topic_candidates(qs:List[str])->List[str]:
    topics=[]
    for q in qs:
        if not q:continue
        m=re.search(r"\babout\s+([a-zA-Z][\w\s\-']{3,40})",q,re.IGNORECASE)
        if m:topics.append(m.group(1).strip(' ?.,!'));continue
        words=re.findall(r"[A-Z][a-z]{3,}\s+[A-Z][a-z]{3,}",q)
        if words:topics.extend(words);continue
        nouns=re.findall(r"\b[a-z]{6,}\b",q.lower())
        if nouns:topics.append(' '.join(nouns[:2]))
    seen=set();out=[]
    for t in topics:
        t=t.strip()
        if t and t.lower() not in seen and len(t)>=4:seen.add(t.lower());out.append(t)
    return out[:10]
def find_gaps(adam=None,learning_atlas=None,coach_atlas=None,sem_lut=None,limit:int=10)->List[Dict[str,Any]]:
    if sem_lut is None and adam is not None:sem_lut=getattr(adam,'sem_lut',None)
    gaps=[]
    if sem_lut is not None:
        sparse=_find_sparse_neighborhoods(sem_lut)
        sparse_qs=[s['q'] for s in sparse if s.get('q')]
        for t in _extract_topic_candidates(sparse_qs):
            gaps.append({'topic':t,'kind':'sparse_region','priority':0.9,'reason':'few neighbors in PTEX cell-space — Adam has thin coverage here'})
    if learning_atlas is not None:
        try:
            debated=learning_atlas.debated_facts(limit=20)
            for d in debated[:5]:
                qs=d.get('q','')
                topic_words=re.findall(r"[A-Z][a-z]{3,}|[a-z]{6,}",qs)
                if topic_words:gaps.append({'topic':' '.join(topic_words[:3]),'kind':'debated','priority':0.85,'reason':'debated fact — sources disagree, needs verification'})
            low_conf=[r for r in (learning_atlas._meta.values()) if float(r.get('confidence',1.0))<0.5 and not r.get('verified')]
            random.shuffle(low_conf)
            for lc in low_conf[:5]:
                topic_words=re.findall(r"[A-Z][a-z]{3,}|[a-z]{6,}",lc.get('q',''))
                if topic_words:gaps.append({'topic':' '.join(topic_words[:3]),'kind':'low_confidence','priority':0.7,'reason':f'confidence={lc.get("confidence",0)} — needs more sources'})
        except Exception as e:print(f'[curiosity] learning_atlas scan failed: {e}',flush=True)
    if coach_atlas is not None:
        try:
            topics=coach_atlas.list_topics()
            weak=sorted([t for t in topics if t.get('mastery_pct',100)<60],key=lambda t:t.get('mastery_pct',100))
            for t in weak[:5]:gaps.append({'topic':t.get('topic',''),'kind':'low_mastery','priority':0.95,'reason':f'coach mastery {t.get("mastery_pct")}% — Adam tutoring weakness'})
        except Exception as e:print(f'[curiosity] coach_atlas scan failed: {e}',flush=True)
    seen=set();uniq=[]
    for g in gaps:
        t=g.get('topic','').lower().strip()
        if not t or t in seen:continue
        seen.add(t);uniq.append(g)
    uniq.sort(key=lambda g:-g.get('priority',0))
    return uniq[:limit]
def pick_next_gap(adam=None,learning_atlas=None,coach_atlas=None,sem_lut=None)->Optional[Dict[str,Any]]:
    gaps=find_gaps(adam=adam,learning_atlas=learning_atlas,coach_atlas=coach_atlas,sem_lut=sem_lut,limit=20)
    if not gaps:return None
    if random.random()<0.7:return gaps[0]
    return random.choice(gaps[:5])
