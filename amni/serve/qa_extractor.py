"""qa_extractor — turns raw text chunks into atomic (question, answer) Q-A pairs via Adam JSON-mode.
~5× density per byte vs raw-chunk teach. Each chunk yields 3-5 atomic facts that are independently searchable + verifiable.
Used by ingest pipeline (replaces _teach_chunks raw mode when extractor available)."""
import json,re
from typing import List,Dict,Any,Optional
_JSON_ARR_RE=re.compile(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]',re.DOTALL)
_EXTRACT_PROMPT=(
    'Extract atomic factual question-answer pairs from this text. Each pair captures ONE specific fact.\n'
    'Rules:\n'
    '- 3-5 pairs per chunk if dense; fewer if thin.\n'
    '- Question must be self-contained (no "this", "above", "the author" — name the subject).\n'
    '- Answer must be 1-2 sentences, concrete, specific.\n'
    '- Skip generic filler ("the article discusses X"). Capture FACTS.\n\n'
    'Output ONLY a JSON array (no prose):\n'
    '[{"q":"<self-contained question>", "a":"<concrete 1-2 sentence answer>"}]\n\n'
    'TEXT: """{TEXT}"""\n\n'
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
        q=str(p.get('q','') or p.get('question','')).strip()
        a=str(p.get('a','') or p.get('answer','')).strip()
        if q and a and len(q)>=4 and len(a)>=4:out.append({'q':q[:300],'a':a[:600]})
    return out
def extract_qa_pairs(adam,chunk:str,max_pairs:int=5)->List[Dict[str,Any]]:
    if adam is None or not hasattr(adam,'chat_persona'):return []
    if not chunk or len(chunk)<80:return []
    prompt=_EXTRACT_PROMPT.replace('{TEXT}',chunk[:1400].replace('"','\\"'))
    try:r=adam.chat_persona(prompt,system='You are a strict JSON fact extractor. Output ONLY a JSON array.',max_new_tokens=520,do_sample=False)
    except Exception:return []
    ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
    pairs=_extract_json_array(ans)[:max_pairs]
    return pairs
def extract_qa_batch(adam,chunks:List[str],max_pairs_per_chunk:int=5)->List[Dict[str,Any]]:
    out=[]
    for c in chunks:
        pairs=extract_qa_pairs(adam,c,max_pairs=max_pairs_per_chunk)
        out.extend(pairs)
    return out
def teach_qa_pairs(adam,pairs:List[Dict[str,Any]],source:str='',learning_atlas=None)->Dict[str,Any]:
    if adam is None or not hasattr(adam,'teach'):return {'taught':0,'verified_after':0}
    taught=0;verified=0
    for p in pairs:
        q=p['q'];a=p['a']
        try:adam.teach(q,a);taught+=1
        except Exception:continue
        if learning_atlas is not None:
            try:
                meta=learning_atlas.record(q,a,source=source,kind='qa_extract')
                if meta.get('verified'):verified+=1
            except Exception:pass
    return {'taught':taught,'verified_after':verified,'pairs_n':len(pairs)}
