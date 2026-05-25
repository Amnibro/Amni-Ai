"""sleep_consolidator — Adam's REM cycle. Groups nearby cells in PTEX embedding space, asks Adam to synthesize a higher-order summary cell, writes it back. Compresses redundancy, builds abstractions.
Runs during low-activity windows so it doesn't compete with user chat for Adam cycles."""
import json,re,time,random
from typing import List,Dict,Any,Optional
_JSON_OBJ_RE=re.compile(r'\{(?:[^{}]|\{[^{}]*\})*\}',re.DOTALL)
_CONSOLIDATE_PROMPT=(
    'Several related facts are listed below. Synthesize them into ONE higher-order summary fact.\n'
    'Output ONLY this JSON (no prose):\n'
    '{"q":"<unified question covering all", "a":"<concise synthesis 2-3 sentences>", "topic_tag":"<one short topic label>"}\n\n'
    'RELATED FACTS:\n{FACTS}\n\nJSON:'
)
def _cluster_cells(sem_lut,min_cluster:int=3,max_radius:int=2,max_clusters:int=12)->List[List[tuple]]:
    if sem_lut is None or not hasattr(sem_lut,'_cells') or len(sem_lut._cells)<min_cluster:return []
    cells=list(sem_lut._cells.keys())
    visited=set();clusters=[]
    for seed in cells:
        if seed in visited:continue
        cluster=[seed];visited.add(seed)
        for other in cells:
            if other in visited:continue
            try:d=sum(abs(a-b) for a,b in zip(seed,other))
            except Exception:continue
            if d<=max_radius:cluster.append(other);visited.add(other)
        if len(cluster)>=min_cluster:clusters.append(cluster)
        if len(clusters)>=max_clusters:break
    return clusters
def _extract_json_obj(text:str)->Optional[Dict[str,Any]]:
    if not text:return None
    m=_JSON_OBJ_RE.search(text);raw=m.group(0) if m else text.strip()
    try:return json.loads(raw)
    except Exception:
        try:return json.loads(raw.replace("'",'"'))
        except Exception:return None
def consolidate_cluster(adam,cluster:List[tuple],sem_lut)->Optional[Dict[str,Any]]:
    if adam is None or sem_lut is None:return None
    facts=[]
    for cell in cluster:
        v=sem_lut._cells.get(cell)
        if v and isinstance(v,dict):
            q=v.get('q','');a=v.get('a','')
            if q and a:facts.append(f'Q: {q[:140]}\nA: {a[:240]}')
    if len(facts)<2:return None
    prompt=_CONSOLIDATE_PROMPT.replace('{FACTS}','\n\n'.join(facts[:6]))
    try:r=adam.chat_persona(prompt,system='You are a strict JSON synthesizer. Output ONE JSON object.',max_new_tokens=320,do_sample=False)
    except Exception:return None
    ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
    obj=_extract_json_obj(ans)
    if not obj or not obj.get('q') or not obj.get('a'):return None
    return {'q':str(obj.get('q','')).strip()[:300],'a':str(obj.get('a','')).strip()[:600],'topic_tag':str(obj.get('topic_tag','')).strip()[:60],'source_cluster_size':len(cluster),'source_facts_n':len(facts)}
def sleep_pass(adam,sem_lut=None,learning_atlas=None,max_clusters:int=5)->Dict[str,Any]:
    if sem_lut is None and adam is not None:sem_lut=getattr(adam,'sem_lut',None)
    if sem_lut is None:return {'consolidated':0,'reason':'no sem_lut'}
    clusters=_cluster_cells(sem_lut,max_clusters=max_clusters)
    if not clusters:return {'consolidated':0,'reason':'no clusters found','clusters_examined':0}
    consolidated=[];failures=0
    for cluster in clusters[:max_clusters]:
        summary=consolidate_cluster(adam,cluster,sem_lut)
        if summary is None:failures+=1;continue
        try:
            adam.teach(summary['q'],summary['a'])
            if learning_atlas is not None:learning_atlas.record(summary['q'],summary['a'],source='sleep_consolidation',confidence=0.7,kind='consolidation')
            consolidated.append({'q':summary['q'][:120],'topic_tag':summary['topic_tag'],'source_cluster_size':summary['source_cluster_size']})
        except Exception as e:failures+=1;print(f'[sleep] consolidate teach failed: {e}',flush=True)
    return {'consolidated':len(consolidated),'clusters_examined':len(clusters[:max_clusters]),'failures':failures,'samples':consolidated[:3],'ts':time.time()}
