"""reffelt_tag — contextual Reffelt-nonce addressing for the pre-response review substrate.
Every reviewable item (lesson, error, leak, fact) is tagged with (1) clear context tags — the salient
keywords + intent — and (2) a Reffelt nonce: a base-17 4-digit address (RGBA digits, REFFELT_K4=[1,17,289,4913])
where each digit is a hash projection of one slice of the tag set, so contexts that share tags share digits.
Per the paradigm: the nonce IS the address IS the meaning. Retrieval matches by tag overlap (recall-safe — can't be
missed) with the nonce as the coarse bucket / tie-break, so the right items surface before every response."""
import re,hashlib
from typing import List,Set,Dict,Any,Tuple,Optional
REFFELT_K4=(1,17,289,4913)
_STOP={'the','and','for','with','this','that','from','your','have','about','what','when','where','which','there','their','would','could','should','will','your','you','are','was','were','has','had','can','does','did','how','why','who','whom','our','out','its','into','than','then','them','they','his','her','she','him','not','but','all','any','get','got','use','one','two','use','via','per','off','set'}
def salient_tags(text:str,max_tags:int=12)->List[str]:
    if not text:return []
    toks=re.findall(r"[a-zA-Z][a-zA-Z0-9_+\-]{2,}",text.lower())
    seen=set();out=[]
    for t in toks:
        if t in _STOP or len(t)<3:continue
        if t in seen:continue
        seen.add(t);out.append(t)
    return out[:max_tags]
def _digit(parts:List[str],salt:bytes)->int:
    key=('|'.join(sorted(parts)) or '_').encode('utf-8','ignore')
    return int.from_bytes(hashlib.blake2b(key,digest_size=6,person=salt).digest(),'big')%17
def nonce(tags)->int:
    ts=sorted(set(str(t).lower() for t in (tags or []) if str(t).strip()))
    if not ts:return 0
    n=len(ts)
    slices=[ts,ts[:max(1,n//2)],ts[::2] or ts,ts[-max(1,n//3):]]
    d=[_digit(slices[i],s) for i,s in enumerate((b'reff0d0!',b'reff1d1!',b'reff2d2!',b'reff3d3!'))]
    return d[0]*REFFELT_K4[0]+d[1]*REFFELT_K4[1]+d[2]*REFFELT_K4[2]+d[3]*REFFELT_K4[3]
def decompose(n:int)->List[int]:
    n=int(n);return [n%17,(n//17)%17,(n//289)%17,(n//4913)%17]
def recompose(digits)->int:
    d=list(digits)+[0,0,0,0];return d[0]+d[1]*17+d[2]*289+d[3]*4913
def nonce_for_text(text:str)->int:
    return nonce(salient_tags(text))
def nonce_distance(a:int,b:int)->int:
    da,db=decompose(a),decompose(b);return sum(1 for i in range(4) if da[i]!=db[i])
def tag_overlap(query_tags,item_tags)->float:
    q=set(str(t).lower() for t in (query_tags or []));i=set(str(t).lower() for t in (item_tags or []))
    if not q or not i:return 0.0
    inter=len(q&i);union=len(q|i)
    return inter/union if union else 0.0
def relevance(query_tags,query_nonce:int,item_tags,item_nonce:int)->float:
    j=tag_overlap(query_tags,item_tags)
    nd=nonce_distance(query_nonce,item_nonce)
    return j+0.12*(4-nd)/4.0
def tag_record(text:str,extra_tags=None)->Dict[str,Any]:
    tags=salient_tags(text)
    if extra_tags:
        for t in extra_tags:
            tl=str(t).lower().strip()
            if tl and tl not in tags:tags.append(tl)
    return {'tags':tags[:16],'nonce':nonce(tags)}
