import hashlib,re
from typing import Any,Dict,List,Optional,Sequence,Tuple
_DEV_PATTERNS:List[Tuple[str,re.Pattern]]=[
('syntax',re.compile(r'(?i)\bsyntaxerror\b|unexpected (?:token|indent|eof)|invalid syntax|indent(?:ation)?error|taberror')),
('name',re.compile(r"(?i)\bnameerror\b|name '(\w+)' is not defined|undefined (?:name|variable|identifier)")),
('type',re.compile(r'(?i)\btypeerror\b|unsupported operand|not (?:callable|subscriptable|iterable)|cannot unpack')),
('attr',re.compile(r'(?i)\battributeerror\b|has no attribute')),
('index',re.compile(r'(?i)\bindexerror\b|list index out of range|string index out of range')),
('key',re.compile(r'(?i)\bkeyerror\b|key (?:error|not found)')),
('assert',re.compile(r'(?i)\bassertionerror\b|\bassert\b|assert equal|expected .+ got')),
('import',re.compile(r'(?i)\bimporterror\b|modulenotfounderror|no module named')),
('timeout',re.compile(r'(?i)\btimeout\b|timed? out|deadline exceeded')),
('io',re.compile(r'(?i)\bfilenotfounderror\b|permissionerror|is a directory|no such file')),
('algo',re.compile(r'(?i)\bwrong (?:answer|output|result)\b|failed test|test failed|incorrect|mismatch|did not match')),
('empty',re.compile(r'(?i)empty (?:code|output)|produced no code|no code')),
('runtime',re.compile(r'(?i)\btraceback\b|\bexception\b|\berror\b|panicked|segfault|oom|memory')),
]
_STOP=set('a an the to of for and or in on with by from as is are was were be been being this that those these it its into over under not no yes if then else when while do does did so than then just only also very more most less least via use using used make makes made add adds write writes fix fixes'.split())
def classify_deviation(errors:Optional[Sequence[Any]]=None,outcome:str='')->Dict[str,Any]:
    blob=' '.join(str(e) for e in (errors or []))+' '+str(outcome or '')
    if not blob.strip():return {'class':'unknown','why':'no error signal','evidence':''}
    hits=[]
    for name,pat in _DEV_PATTERNS:
        m=pat.search(blob)
        if m:hits.append((name,m.group(0)[:120]))
    if not hits:return {'class':'unknown','why':blob.strip()[:160],'evidence':blob.strip()[:200]}
    primary=hits[0][0];ev=hits[0][1]
    why={'syntax':'parse/indent failure — rewrite structure carefully','name':'undefined identifier — align names or define missing symbols','type':'type/arity mismatch — check call shapes and returns','attr':'missing attribute — wrong object or incomplete API','index':'bounds error — guard lengths and off-by-one','key':'missing key — use defaults / validate schema','assert':'spec mismatch — re-read required outputs and edge cases','import':'missing dependency — pure-stdlib rewrite or correct import','timeout':'non-termination — bound loops / avoid infinite recursion','io':'filesystem path issue — pure in-memory logic only','algo':'wrong algorithm/result — change approach fundamentally','empty':'no code emitted — emit a full fenced file','runtime':'runtime fault — simplify and harden control flow'}.get(primary,blob[:120])
    return {'class':primary,'why':why,'evidence':ev,'all_classes':[h[0] for h in hits]}
def strategy_tokens(text:str)->set:
    toks=re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}',(text or '').lower())
    return {t for t in toks if t not in _STOP and not t.isdigit()}
def strategy_distance(a:str,b:str)->float:
    ta,tb=strategy_tokens(a),strategy_tokens(b)
    if not ta and not tb:return 0.0
    if not ta or not tb:return 1.0
    inter=len(ta&tb);union=len(ta|tb)
    return round(1.0-(inter/union if union else 0.0),4)
def method_id(approach:str,deviation_class:str='')->str:
    raw=(approach or '').strip().lower()+'|'+(deviation_class or '')
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
def kind_for(success:Optional[bool],attempt_n:int=1,deviation_class:str='')->str:
    if success is True:return 'repair_patch' if attempt_n>1 else 'memorized_fact'
    if success is False:return 'new_method' if (deviation_class in ('algo','assert') or attempt_n>=2) else 'repair_patch'
    return 'unknown'
def burned_methods(rows:Sequence[Dict[str,Any]])->List[str]:
    out=[];seen=set()
    for r in rows or []:
        if r.get('success') is not False:continue
        m=(r.get('method') or r.get('approach') or '').strip()
        if m and m.lower() not in seen:seen.add(m.lower());out.append(m[:80])
    return out
def burned_deviations(rows:Sequence[Dict[str,Any]])->List[str]:
    out=[];seen=set()
    for r in rows or []:
        if r.get('success') is not False:continue
        c=r.get('deviation_class')
        if not c:
            d=r.get('deviation');c=(d.get('class') if isinstance(d,dict) else d) or ''
        c=str(c or '').strip().lower()
        if c and c not in seen and c!='unknown':seen.add(c);out.append(c)
    return out
def diversity_steer(prior_approaches:Sequence[str],min_distance:float=0.35)->str:
    burned=burned_methods([{'success':False,'approach':a} for a in prior_approaches])
    if not burned:return ''
    lines=['METHOD DIVERSITY REQUIRED — previous approaches burned (do not rephrase them):']
    for a in burned[:6]:lines.append(f'  - {a}')
    lines.append(f'Pick a strategy with token-set distance ≥{min_distance} from all of the above. Change algorithm/data-structure/control-flow, not wording.')
    return '\n'.join(lines)
def pack_deviation(errors:Optional[Sequence[Any]]=None,outcome:str='',explicit:Any=None)->Dict[str,Any]:
    if isinstance(explicit,dict) and explicit.get('class'):return {'class':str(explicit.get('class'))[:40],'why':str(explicit.get('why') or '')[:200],'evidence':str(explicit.get('evidence') or '')[:200]}
    if isinstance(explicit,str) and explicit.strip():
        base=classify_deviation(errors,outcome);base['why']=explicit.strip()[:200];return base
    return classify_deviation(errors,outcome)
