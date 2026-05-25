"""kg_extractor — turn Q-A pairs into (subject, predicate, object) triples via Adam JSON-mode.
Each Q-A yields 1-3 triples. Mainstream LLMs do this only at training time; Adam does it on every ingest and writes to a queryable graph that survives across sessions."""
import json,re
from typing import List,Dict,Any,Optional,Tuple
_JSON_ARR_RE=re.compile(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]',re.DOTALL)
_EXTRACT_PROMPT=(
    'Extract 1-3 atomic subject-predicate-object triples from this fact. Each triple captures ONE relationship.\n'
    'Rules:\n'
    '- subject: the main thing/concept (1-4 words, no determiners like "the/a")\n'
    '- predicate: a short relation slug, snake_case (e.g. is_a, has_part, occurs_in, defined_by, located_in, depends_on, causes, opposite_of, member_of)\n'
    '- object: the related thing (1-4 words)\n'
    '- Pick concrete relations, not generic "related_to".\n'
    '- Skip if the fact is opinionated/subjective.\n\n'
    'Output ONLY a JSON array (no prose):\n'
    '[{"s":"<subject>","p":"<predicate_slug>","o":"<object>"}]\n\n'
    'QUESTION: {Q}\nANSWER: {A}\n\n'
    'JSON array:'
)
def _extract_json_array(text:str)->List[Dict[str,Any]]:
    if not text:return []
    m=_JSON_ARR_RE.search(text);raw=m.group(0) if m else text.strip()
    try:parsed=json.loads(raw)
    except Exception:
        try:parsed=json.loads(raw.replace("'",'"'))
        except Exception:return []
    if not isinstance(parsed,list):return []
    out=[]
    for p in parsed:
        if not isinstance(p,dict):continue
        s=str(p.get('s','') or p.get('subject','')).strip()
        pr=str(p.get('p','') or p.get('predicate','')).strip()
        o=str(p.get('o','') or p.get('object','')).strip()
        if s and pr and o and len(s)>=2 and len(o)>=2:out.append({'s':s,'p':pr,'o':o})
    return out
def extract_triples(adam,question:str,answer:str,max_triples:int=3)->List[Dict[str,Any]]:
    if adam is None or not hasattr(adam,'chat_persona'):return []
    if not question or not answer:return []
    prompt=_EXTRACT_PROMPT.replace('{Q}',question[:300].replace('"','\\"')).replace('{A}',answer[:600].replace('"','\\"'))
    try:r=adam.chat_persona(prompt,system='You are a strict JSON triple extractor. Output ONLY a JSON array.',max_new_tokens=260,do_sample=False)
    except Exception:return []
    ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
    return _extract_json_array(ans)[:max_triples]
def extract_and_store(adam,kg,question:str,answer:str,source:str='',confidence:Optional[float]=None,max_triples:int=3)->Dict[str,Any]:
    if kg is None:return {'stored':0,'reason':'no_kg'}
    triples=extract_triples(adam,question,answer,max_triples=max_triples)
    if not triples:return {'stored':0,'extracted':0,'reason':'no_triples_extracted'}
    n_stored=0;samples=[]
    for t in triples:
        added=kg.add(t['s'],t['p'],t['o'],source=source,confidence=confidence,kind='kg_extract')
        if added is not None:n_stored+=1
        if len(samples)<5:samples.append(t)
    return {'stored':n_stored,'extracted':len(triples),'samples':samples}
