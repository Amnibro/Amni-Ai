"""pii_egress — single choke-point that scrubs personal info from ANY text leaving the box (web search, crawl, news widget, ingest).
Defense-in-depth: pattern scrub (email/phone/SSN/card/IP/address/ZIP+4) PLUS removal of every token drawn from the user's PersonalAtlas (their real name, city, email, etc.). "No leaks ever" — over-scrubbing a PII-shaped token is acceptable; a leak is not.
Audit trail logs categories + counts ONLY (never the raw values) to gitignored data/pii_egress_audit.jsonl so the log itself can never become a PII store."""
import re,time,json,hashlib,threading
from pathlib import Path
from typing import Dict,Any,List,Optional,Tuple
_STOP={'the','and','for','with','this','that','from','your','have','about','what','when','where','which','there','their','would','could','should'}
_PATTERNS=[
 ('email',re.compile(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b')),
 ('ssn',re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
 ('phone_intl',re.compile(r'\+\d{1,3}[ \-.]?\(?\d{2,4}\)?[ \-.]?\d{3}[ \-.]?\d{3,4}\b')),
 ('phone_us',re.compile(r'\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b')),
 ('credit_card',re.compile(r'\b(?:\d[ -]?){13,16}\b')),
 ('ipv4',re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')),
 ('zip_plus4',re.compile(r'\b\d{5}-\d{4}\b')),
 ('street_address',re.compile(r'\b\d{1,6}\s+(?:[A-Z][a-zA-Z]+\.?\s+){1,4}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place|Ter|Terrace|Cir|Circle|Hwy|Highway)\b',re.IGNORECASE)),
]
_LOCK=threading.Lock()
def _audit_path()->Path:
    p=Path(__file__).resolve().parents[2]/'data'/'pii_egress_audit.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _atlas_tokens(atlas)->List[str]:
    if atlas is None or not hasattr(atlas,'list_facts'):return []
    toks=set()
    try:
        for f in (atlas.list_facts(include_confidential=True,limit=300) or []):
            v=str(f.get('value') or f.get('fact') or '').strip()
            if not v:continue
            for tok in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}",v):
                tl=tok.lower()
                if tl in _STOP or len(tl)<3:continue
                toks.add(tok)
            for num in re.findall(r'\b\d{4,}\b',v):toks.add(num)
    except Exception:return []
    return sorted(toks,key=len,reverse=True)
def scrub_text(text:str,atlas=None)->Tuple[str,Dict[str,Any]]:
    if not text:return text,{'scrubbed':False,'categories':[],'removed':0}
    cleaned=text;cats=[];removed=0
    for name,pat in _PATTERNS:
        new,n=pat.subn(' ',cleaned)
        if n:cats.append(name);removed+=n;cleaned=new
    atok=0
    for tok in _atlas_tokens(atlas):
        new,n=re.subn(r'(?i)\b'+re.escape(tok)+r'\b',' ',cleaned)
        if n:atok+=n;cleaned=new
    if atok:cats.append('atlas_token');removed+=atok
    cleaned=re.sub(r'\s+',' ',cleaned).strip(' ,.;:!?-')
    if not cleaned:cleaned=text
    return cleaned,{'scrubbed':bool(cats),'categories':sorted(set(cats)),'removed':removed}
def _resolve_atlas(agent=None,atlas=None):
    if atlas is not None:return atlas
    return getattr(agent,'personal_atlas',None) if agent is not None else None
def scrub(text:str,agent=None,atlas=None,source:str='web',audit:bool=True)->str:
    cleaned,report=scrub_text(text,atlas=_resolve_atlas(agent,atlas))
    if audit and report['scrubbed']:
        try:
            rec={'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'source':source,'categories':report['categories'],'removed':report['removed'],'in_hash':hashlib.sha256((text or '').encode('utf-8','ignore')).hexdigest()[:12]}
            with _LOCK:
                with _audit_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec)+'\n')
        except Exception:pass
    return cleaned
def audit_stats(limit:int=50)->Dict[str,Any]:
    p=_audit_path()
    if not p.exists():return {'total':0,'by_category':{},'by_source':{},'recent':[]}
    rows=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:rows.append(json.loads(ln))
            except Exception:continue
    except Exception:return {'total':0,'by_category':{},'by_source':{},'recent':[]}
    bycat={};bysrc={}
    for r in rows:
        for c in r.get('categories',[]):bycat[c]=bycat.get(c,0)+1
        s=r.get('source','?');bysrc[s]=bysrc.get(s,0)+1
    rows.sort(key=lambda r:-float(r.get('ts') or 0))
    return {'total':len(rows),'by_category':dict(sorted(bycat.items(),key=lambda kv:-kv[1])),'by_source':dict(sorted(bysrc.items(),key=lambda kv:-kv[1])),'recent':rows[:int(max(1,limit))]}
