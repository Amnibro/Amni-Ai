"""Federated learning bridge — push Adam's lessons to Amni-Prism (HF-bound), pull community lessons back.
PII filtering is mandatory before publish: emails, phones, IPs, file paths, names, API keys all stripped or skipped.
Quality gates: confidence >= 0.8, dedup via content_hash, min length, max length, no script tags."""
import re,hashlib,time,json,os
from pathlib import Path
from typing import Dict,List,Optional,Any,Tuple
_PII_EMAIL=re.compile(r'\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b')
_PII_PHONE=re.compile(r'\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b')
_PII_IP=re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_PII_PATH_WIN=re.compile(r'[A-Za-z]:[\\/](?:[\w .\\\-]+)+')
_PII_PATH_NIX=re.compile(r'/(?:home|Users|root|var|tmp|opt|etc)/[\w./\-]+')
_PII_API_KEY=re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|xox[bopa]-[A-Za-z0-9-]{20,}|AIza[\w-]{35}|AKIA[A-Z0-9]{16})\b')
_PII_UUID=re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',re.IGNORECASE)
_PII_SSN=re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_PII_CC=re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b')
_PII_NAME_HINTS=re.compile(r'\b(?:my\s+name\s+is|i\s+am|call\s+me|name:|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',re.IGNORECASE)
_PII_HOMEDIR=re.compile(r'\bC:[/\\]Users[/\\][^/\\\s]+',re.IGNORECASE)
_SCRIPT_TAG=re.compile(r'<script[^>]*>|<iframe[^>]*>',re.IGNORECASE)
_MIN_LEN=10
_MAX_LEN=4000
def scrub_pii(text:str)->Tuple[str,List[str]]:
    if not text:return ('',[])
    flags=[];out=text
    for name,pat,repl in [('email',_PII_EMAIL,'<EMAIL>'),('phone',_PII_PHONE,'<PHONE>'),('ip',_PII_IP,'<IP>'),('homedir',_PII_HOMEDIR,'<HOMEDIR>'),('path_win',_PII_PATH_WIN,'<PATH>'),('path_nix',_PII_PATH_NIX,'<PATH>'),('api_key',_PII_API_KEY,'<APIKEY>'),('uuid',_PII_UUID,'<UUID>'),('ssn',_PII_SSN,'<SSN>'),('cc',_PII_CC,'<CC>')]:
        if pat.search(out):flags.append(name);out=pat.sub(repl,out)
    for m in _PII_NAME_HINTS.finditer(out):
        flags.append('name_hint');out=out.replace(m.group(0),f'{m.group(0).split()[0]} <NAME>')
    return (out,flags)
_CHAT_PERSONAL_PATTERNS=('my name','my email','my phone','my address','my birthday','my favorite','my password','my company','my employer','my salary','my wife','my husband','my kid','my son','my daughter','my partner','i live at','i work at','call me','i am called','i am named',"whats my","what's my",'whos my',"who's my",'contact:','reach me at','email me at','phone:','ssn','social security')
def is_publishable(q:str,a:str,confidence:float=1.0,min_confidence:float=0.8,is_personal:bool=False)->Tuple[bool,str]:
    if is_personal:return (False,'flagged is_personal at intake')
    if not q or not a:return (False,'empty q or a')
    try:from amni.serve.conversation import detect_personal as _dp
    except Exception:_dp=None
    if _dp is not None and (_dp(q) or _dp(a)):return (False,'detect_personal positive (raw PII in q or a)')
    ql=q.lower();al=a.lower()
    if any(k in ql or k in al for k in _CHAT_PERSONAL_PATTERNS):return (False,'conversation-style personal pattern')
    if q.startswith('PERSONA::') or q.startswith('ATLAS::'):return (False,'cache/atlas key')
    if confidence<min_confidence:return (False,f'confidence {confidence:.2f} < {min_confidence}')
    if len(q)<_MIN_LEN or len(a)<_MIN_LEN:return (False,f'too short (q={len(q)} a={len(a)})')
    if len(q)>_MAX_LEN or len(a)>_MAX_LEN:return (False,f'too long (q={len(q)} a={len(a)})')
    if _SCRIPT_TAG.search(q+' '+a):return (False,'contains script/iframe tag')
    if q.startswith('What does ') and ' say about ' in q:return (False,'scan-synthetic key (low quality for sharing)')
    if 'mock' in ql or 'mock' in al:return (False,'mock content')
    return (True,'ok')
_DOMAIN_KEYWORDS={'math':['math','algebra','geometry','calculus','arithmetic','equation','theorem','sqrt','sin','cos','tan','log','exp','derivative','integral'],'physics':['physics','force','velocity','acceleration','mass','energy','quantum','electron','photon','wavelength','newton','joule','watt','momentum'],'chemistry':['chemistry','molecule','atom','ion','bond','reaction','element','periodic','solution','acid','base','catalyst','enzyme','protein'],'biology':['biology','cell','dna','rna','gene','organism','species','evolution','ecosystem','photosynthesis','mitosis','genome'],'history':['history','war','battle','empire','dynasty','revolution','treaty','century','BC','AD','medieval','renaissance','ancient'],'geography':['capital','country','continent','ocean','river','mountain','city','population','geography'],'literature':['novel','poem','poet','author','wrote','book','character','protagonist','shakespeare','dickens','tolstoy'],'computer_science':['python','javascript','algorithm','data structure','function','class','code','programming','recursion','complexity','sort'],'general':[]}
def _detect_domain(q:str,a:str)->str:
    text=(q+' '+a).lower()
    best='general';best_n=0
    for d,kws in _DOMAIN_KEYWORDS.items():
        n=sum(1 for k in kws if k in text)
        if n>best_n:best=d;best_n=n
    return best
def filter_lessons(adam,min_confidence:float=0.8,domain:Optional[str]=None,limit:int=100)->List[Dict[str,Any]]:
    sl=getattr(adam,'sem_lut',None)
    raw=getattr(sl,'_raw',[]) if sl is not None else []
    out=[]
    for q,a in raw:
        ok,reason=is_publishable(q,a,confidence=1.0,min_confidence=min_confidence)
        if not ok:continue
        scrubbed_q,fq=scrub_pii(q);scrubbed_a,fa=scrub_pii(a)
        if not scrubbed_q.strip() or not scrubbed_a.strip():continue
        d=_detect_domain(scrubbed_q,scrubbed_a)
        if domain and d!=domain:continue
        out.append({'q':scrubbed_q,'a':scrubbed_a,'domain':d,'pii_flags':fq+fa,'original_len':(len(q),len(a))})
        if len(out)>=limit:break
    return out
def filter_atlas(atlas,min_confidence:float=0.8,domain:Optional[str]=None,limit:int=200)->List[Dict[str,Any]]:
    if atlas is None or not hasattr(atlas,'federation_pull'):return []
    out=[]
    for entry in atlas.federation_pull(min_confidence=min_confidence,limit=limit*3):
        q=entry.get('q','');a=entry.get('a','')
        ok,reason=is_publishable(q,a,confidence=entry.get('conf',1.0),min_confidence=min_confidence,is_personal=False)
        if not ok:continue
        sq,fq=scrub_pii(q);sa,fa=scrub_pii(a)
        if not sq.strip() or not sa.strip():continue
        if fq or fa:continue
        d=_detect_domain(sq,sa)
        if domain and d!=domain:continue
        out.append({'q':sq,'a':sa,'domain':d,'pii_flags':[],'source':'atlas','original_len':(len(q),len(a))})
        if len(out)>=limit:break
    return out
def publish_from_atlas(adam,atlas,codex_dir:str='./codex',contributor_id:str='amni-ai-anonymous',min_confidence:float=0.8,domain:Optional[str]=None,limit:int=100,dry_run:bool=False)->Dict[str,Any]:
    lessons=filter_atlas(atlas,min_confidence=min_confidence,domain=domain,limit=limit)
    if dry_run:return {'dry_run':True,'eligible':len(lessons),'source':'atlas','preview':lessons[:3],'pii_summary':_pii_summary(lessons)}
    try:from prism.contribute import contribute_text
    except ImportError:return {'error':'amni-prism not installed. pip install amni-prism','eligible':len(lessons),'source':'atlas'}
    Path(codex_dir).mkdir(parents=True,exist_ok=True)
    added=0;duplicates=0;errors=[]
    for L in lessons:
        try:
            payload=f'Q: {L["q"]}\nA: {L["a"]}'
            r=contribute_text(codex_dir,payload,domain=L['domain'],contributor_id=contributor_id,source='amni-ai/v6.5-atlas',confidence=1.0,verified=True)
            st=r.get('status')
            if st=='added':added+=1
            elif st=='duplicate':duplicates+=1
            else:errors.append(r)
        except Exception as e:errors.append({'error':str(e)})
    return {'codex_dir':codex_dir,'eligible':len(lessons),'added':added,'duplicates':duplicates,'errors':errors[:5],'source':'atlas','next_step':'Run `prism push` or sync codex_dir to your HF repo manually.'}
def publish_lessons(adam,codex_dir:str='./codex',contributor_id:str='amni-ai-anonymous',min_confidence:float=0.8,domain:Optional[str]=None,limit:int=100,dry_run:bool=False)->Dict[str,Any]:
    lessons=filter_lessons(adam,min_confidence=min_confidence,domain=domain,limit=limit)
    if dry_run:
        return {'dry_run':True,'eligible':len(lessons),'preview':lessons[:3],'pii_summary':_pii_summary(lessons)}
    try:from prism.contribute import contribute_text
    except ImportError:return {'error':'amni-prism not installed. pip install amni-prism','eligible':len(lessons)}
    Path(codex_dir).mkdir(parents=True,exist_ok=True)
    added=0;duplicates=0;errors=[]
    for L in lessons:
        try:
            payload=f'Q: {L["q"]}\nA: {L["a"]}'
            r=contribute_text(codex_dir,payload,domain=L['domain'],contributor_id=contributor_id,source='amni-ai/v6.4',confidence=1.0,verified=True)
            if r.get('status')=='added':added+=1
            elif r.get('status')=='duplicate':duplicates+=1
            else:errors.append(r)
        except Exception as e:errors.append({'error':str(e)})
    return {'codex_dir':codex_dir,'eligible':len(lessons),'added':added,'duplicates':duplicates,'errors':errors[:5],'pii_summary':_pii_summary(lessons),'next_step':'Run `prism push` (if HF push wired) or sync codex_dir to your HF repo manually.'}
def pull_lessons(adam,codex_dir:str='./codex',domain:Optional[str]=None,limit:int=200,dry_run:bool=False)->Dict[str,Any]:
    try:from prism.contribute import _load_ndjson
    except ImportError:return {'error':'amni-prism not installed'}
    manifest=os.path.join(codex_dir,'manifest.ndjson')
    entries=_load_ndjson(manifest)
    if domain:entries=[e for e in entries if e.get('domain')==domain]
    if not entries:return {'codex_dir':codex_dir,'available':0,'note':'no manifest entries; pull codex from HF first'}
    sl=getattr(adam,'sem_lut',None)
    existing_qs=set(q for q,_ in (getattr(sl,'_raw',[]) if sl else []))
    fresh=[];added=0
    for e in entries[:limit]:
        f=os.path.join(codex_dir,e.get('file',''))
        if not os.path.exists(f):continue
        try:txt=Path(f).read_text(encoding='utf-8')
        except Exception:continue
        if '\nA:' not in txt or not txt.startswith('Q:'):continue
        try:q,a=txt.split('\nA:',1);q=q[2:].strip();a=a.strip()
        except Exception:continue
        if q in existing_qs:continue
        scrubbed_q,_=scrub_pii(q);scrubbed_a,_=scrub_pii(a)
        if not scrubbed_q or not scrubbed_a:continue
        fresh.append((scrubbed_q,scrubbed_a,e.get('domain','general')))
    if dry_run:return {'codex_dir':codex_dir,'available':len(fresh),'preview':[{'q':q[:80],'a':a[:80],'domain':d} for q,a,d in fresh[:5]]}
    if sl is not None and fresh:
        for q,a,d in fresh:sl.add(q,a);added+=1
        try:sl.fit()
        except Exception:pass
        try:adam.save_lessons()
        except Exception:pass
    return {'codex_dir':codex_dir,'available':len(fresh),'added':added,'lessons_total_now':len(getattr(sl,'_raw',[]))}
def _pii_summary(lessons:List[Dict])->Dict[str,int]:
    out={}
    for L in lessons:
        for f in L.get('pii_flags',[]):out[f]=out.get(f,0)+1
    return out
