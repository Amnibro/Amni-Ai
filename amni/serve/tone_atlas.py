"""Categorical PTEX-style tone atlas — multi-dim opener/closer phrase store.
Coordinates: (intent_category, warmth_bin, formality_bin, excitement_bin) → list of phrase variants.
Sampling is deterministic-ish with light entropy from message hash so the same query gives the same opener but different queries vary organically. Acts like a tiny PTEX read where the cell address is (category × tone-dims) and the cell payload is a phrase template."""
import hashlib,re
from typing import List,Tuple,Optional
_CATS={'greeting','factual','code','reasoning','calc_result','time_result','file_result','scan_result','error','introspect','personal','creative','unknown'}
_BANK={('greeting','warm','casual','high'):['Rao!','Hey there!','Hi!','Yo!','Hiya!'],
       ('greeting','warm','casual','med'):['Hey,','Hi,','Hello!'],
       ('greeting','cool','formal','low'):['Hello.','Greetings.','Good day.'],
       ('greeting','warm','formal','med'):['Hello there.','Greetings, friend.'],
       ('factual','warm','casual','med'):['','So,','Right,','Easy —'],
       ('factual','cool','formal','low'):['','Indeed,','To answer:'],
       ('factual','warm','casual','high'):['Oac!','Easy one!','Got it —','Quick one:'],
       ('calc_result','warm','casual','high'):['','Easy!','Boom —','Got it:'],
       ('calc_result','cool','formal','low'):['','Result:','Computed:'],
       ('calc_result','warm','casual','med'):['','That gives','Comes out to','Equals'],
       ('time_result','warm','casual','med'):['','Right now,',"It's"],
       ('reasoning','warm','casual','med'):['Let me think...','Okay so,','Walking through this:','Here we go:'],
       ('reasoning','cool','formal','low'):['Consider:','To reason through:','Step by step:'],
       ('code','warm','casual','med'):['','Sure thing:','Try this:','Here:'],
       ('code','cool','formal','low'):['','Implementation:','Code:'],
       ('error','warm','casual','high'):['Oops —','Hmm,','Aaack —','Welp,'],
       ('error','warm','casual','med'):['Hmm,','Sorry,','That broke:'],
       ('error','cool','formal','low'):['Error:','Failure:','Issue:'],
       ('introspect','warm','casual','high'):['Rao!','Hey, glad you asked!','Sure thing!'],
       ('introspect','warm','casual','med'):['Sure,','Glad you asked.','OK so,'],
       ('introspect','cool','formal','low'):['Capabilities:','Overview:'],
       ('personal','warm','casual','high'):['Aww,','Got it!','For sure!','Noted!'],
       ('personal','warm','casual','med'):['Got it.','Noted.','Sure.'],
       ('personal','cool','formal','low'):['Acknowledged.','Understood.','Noted.'],
       ('scan_result','warm','casual','high'):['','Done!','Boom —','Sweet!'],
       ('scan_result','cool','formal','low'):['','Scan complete.','Ingested:'],
       ('file_result','warm','casual','med'):['','Here:','Got it:'],
       ('file_result','cool','formal','low'):['','File contents:','Read:'],
       ('creative','warm','casual','high'):['Ooh!','Fun one!','Let me try —','Okay here goes:'],
       ('creative','warm','casual','med'):['Hmm,','Let me try:','Okay:'],
       ('unknown','warm','casual','med'):['','Hmm,','Let me see —','Okay,']}
_CLOSERS={('factual','warm','casual','high'):['','— easy peasy!','done!'],
          ('factual','warm','casual','med'):['','.'],
          ('factual','cool','formal','low'):['.'],
          ('reasoning','warm','casual','med'):['','— make sense?','— there ya go.'],
          ('error','warm','casual','high'):['— wanna try again?','— hmm.'],
          ('introspect','warm','casual','high'):['Try me!','— what should we tackle first?'],
          ('introspect','warm','casual','med'):['','— what would you like to do?']}
def _bin(v:float,low_th:float=0.35,high_th:float=0.7)->str:
    return 'low' if v<low_th else ('high' if v>high_th else 'med')
def _bin_warmth(v:float)->str:return 'warm' if v>=0.5 else 'cool'
def _bin_formality(v:float)->str:return 'casual' if v<=0.5 else 'formal'
def _bin_excitement(v:float)->str:return _bin(v,0.35,0.7)
_INTENT_PATTERNS=[(re.compile(r'^\s*(?:hi|hey|hello|yo|howdy|sup|good\s+(?:morning|afternoon|evening))\b',re.IGNORECASE),'greeting'),
                  (re.compile(r'\b(?:what\s+can\s+you|who\s+are\s+you|what\s+are\s+you|capabilities|introduce\s+yourself|help)\b',re.IGNORECASE),'introspect'),
                  (re.compile(r'\b(?:write|generate|create)\s+(?:a\s+)?(?:python|javascript|js|ts|html|code|function|class|script)\b',re.IGNORECASE),'code'),
                  (re.compile(r'\b(?:why|how\s+come|explain|reason|step\s+by\s+step|walk\s+me\s+through)\b',re.IGNORECASE),'reasoning'),
                  (re.compile(r'\b(?:i\s+(?:am|like|love|hate|prefer|feel)|my\s+(?:name|favorite|pet|family|kid|wife))\b',re.IGNORECASE),'personal'),
                  (re.compile(r'\b(?:write|tell|make)\s+(?:me\s+)?(?:a\s+)?(?:poem|story|joke|haiku|song|tale)\b',re.IGNORECASE),'creative'),
                  (re.compile(r'\b(?:what\s+is|capital\s+of|who\s+wrote|chemical\s+symbol|how\s+many\s+(?:planets|continents|sides))\b',re.IGNORECASE),'factual')]
def classify_intent(message:str,skill_used:Optional[str]=None,had_error:bool=False)->str:
    if had_error:return 'error'
    if skill_used:
        return {'time':'time_result','calc':'calc_result','file_read':'file_result','file_write':'file_result','code_edit':'file_result','scan':'scan_result','mem':'factual','web':'factual','shell':'file_result'}.get(skill_used,'factual')
    for pat,cat in _INTENT_PATTERNS:
        if pat.search(message):return cat
    return 'unknown'
def _hash_idx(seed:str,n:int)->int:
    return int(hashlib.md5(seed.encode('utf-8')).hexdigest(),16)%max(1,n)
def sample_opener(category:str,warmth:float,formality:float,excitement:float,seed:str='')->str:
    cat=category if category in _CATS else 'unknown'
    key=(cat,_bin_warmth(warmth),_bin_formality(formality),_bin_excitement(excitement))
    bucket=_BANK.get(key)
    if not bucket:
        for fb in [(cat,'warm','casual','med'),(cat,'cool','formal','low'),('unknown','warm','casual','med')]:
            bucket=_BANK.get(fb)
            if bucket:break
    if not bucket:return ''
    return bucket[_hash_idx(seed or category,len(bucket))]
def sample_closer(category:str,warmth:float,formality:float,excitement:float,seed:str='')->str:
    cat=category if category in _CATS else 'unknown'
    key=(cat,_bin_warmth(warmth),_bin_formality(formality),_bin_excitement(excitement))
    bucket=_CLOSERS.get(key)
    if not bucket:return ''
    return bucket[_hash_idx(seed or category+'_close',len(bucket))]
def _strip_thinking_process(text:str)->str:
    """Remove meta-reasoning + tool-narration leakage. The tool_discipline prompt bans these
    but models leak them anyway; this is the server-side safety net. Detected leaks feed the
    self-learning leak_ledger (commit-to-PTEX + learned-signature consult) so repeats the regex
    never anticipated still get caught."""
    if not text:return text
    import re as _re
    orig=text;leak_frag=''
    t=_re.sub(r'(?is)<\s*(think|thinking|thought|reasoning|scratchpad|reflection|analysis|monologue)\s*>.*?<\s*/\s*\1\s*>',' ',text)
    t=_re.sub(r'(?is)<\s*/?\s*(?:think|thinking|thought|reasoning|scratchpad|reflection|analysis|monologue)\s*>',' ',t)
    t=_re.sub(r'(?is)\n*(?:thinking\s*process|thought\s*process|thought)\s*:?\s*\n.*?(?=\n\s*\n[^\d\s*•\-]|\Z)','',t)
    hdr=_re.search(r'(?i)(?:thought\s+)?(?:thinking|thought)\s*process\s*:?|\bchain[\s-]of[\s-]thought\b|(?:^|\n)\s*(?:my\s+)?(?:reasoning|internal\s+monologue|scratchpad)\s*:',t)
    if hdr:
        p=hdr.start()
        fin=list(_re.finditer(r'(?i)\b(?:final\s+(?:output|answer|response)\s*:|final\s*:)\s*',t[p:]))
        if fin:leak_frag=t[p:p+fin[-1].end()];t=t[p+fin[-1].end():]
        else:
            before=t[:p].strip()
            if len(before)>=12:leak_frag=t[p:];t=before
    t=_re.sub(r'(?i)(?:(?<=\s)|^)\d+\.\s+(?:analyze\s+the\s+request|check\s+(?:context|tools|the\s+context)|determine\s+(?:the\s+)?strategy|formulate\s+(?:the\s+)?\w+|self[- ]correction|refinement|recall\s+(?:the\s+)?persona|apply\s+(?:the\s+)?persona|final\s+output)\b[^\n]*?(?=\s+\d+\.\s|\n|$)',' ',t)
    t=_re.sub(r'(?im)^\s*\d+\.\s+(?:analyze|check|determine|formulate|apply|final\s+output|self-correction|refinement)\b.*$\n?','',t)
    t=_re.sub(r'(?im)^\s*\*\s*(?:restate|knowns|approach|first\s+shot|critique|refine)\s*:.*$\n?','',t)
    t=_re.sub(r'(?im)^\s*-?\s*(?:restate|knowns|approach|first\s+shot|critique|refine)\s*:.*$\n?','',t)
    t=_re.sub(r'(?im)^\s*\[(?:looked|looked\s+it\s+up|search(?:\s+performed)?|searching|searched|presenting|current\s+(?:weather|system|time|news|data)\s*(?:data|info|results?)?|inserting[^\]]*|result\s+of[^\]]*|the\s+system\s+(?:returns?|outputs?)[^\]]*|output|outputs|simulating|waiting[^\]]*|assuming[^\]]*)\][:.]?\s*\n?',' ',t)
    t=_re.sub(r'(?im)\s*\[(?:looked|looked\s+it\s+up|search(?:\s+performed)?|searching|searched|presenting|current\s+(?:weather|system|time|news|data)\s*(?:data|info|results?)?|simulating|output)\]\s*',' ',t)
    t=_re.sub(r'(?im)^\s*\((?:outputting[^)]*|search\s+returns?[^)]*|simulating[^)]*|waiting[^)]*|assuming[^)]*)\)\s*\n?',' ',t)
    t=_re.sub(r'(?m)[ \t]+$','',t)
    t=_re.sub(r'\n{3,}','\n\n',t)
    t=t.strip()
    try:
        from amni.serve import leak_ledger as _ll
        if leak_frag and len(_ll._norm(leak_frag))>=16:_ll.record(leak_frag,clean=t,categories=['thinking_process'],source='wrap')
        t2=_ll.scrub_learned(t)
        if t2:t=t2
    except Exception:pass
    return t or text
def wrap(answer:str,category:str,persona,seed:str='')->str:
    answer=_strip_thinking_process(answer)
    op=sample_opener(category,persona.warmth,persona.formality,persona.excitement,seed=seed)
    cl=sample_closer(category,persona.warmth,persona.formality,persona.excitement,seed=seed)
    parts=[]
    if op and not answer.lower().startswith(op.lower()[:6]):parts.append(op)
    parts.append(answer.strip())
    if cl and not answer.rstrip().endswith(cl.rstrip()):parts.append(cl)
    return ' '.join(p for p in parts if p).strip()
