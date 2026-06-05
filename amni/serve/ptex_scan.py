"""ptex_scan — bad-actor scanner for Adam's PTEX/lesson substrate. A poisoned lesson (or any text that
gets recalled into context) is an injection vector: it can try to override Adam's laws/instructions when
retrieved. This scans recalled-text content for instruction-override / jailbreak / exploit / role-hijack
payloads — NOT mere topic keywords (a history or security LESSON legitimately mentions 'attack'/'exploit';
the threat is text that COMMANDS Adam to ignore its rules). Report-only by default; never auto-deletes.
Severity: high=override/jailbreak, med=exfiltration/role-hijack. Wire into the launch integrity gate."""
import re,os
_OVERRIDE=re.compile(r'\b(ignore|disregard|forget|override|bypass|delete|remove|drop)\b[^.\n]{0,40}\b(all|any|the|your|previous|prior|above|earlier|system)?\b[^.\n]{0,20}\b(instructions?|rules?|laws?|guidelines?|prompts?|restrictions?|safety|ethics|filters?|constraints?|directives?|commandments?|asimov)\b',re.I)
_JAILBREAK=re.compile(r'\b(you are now|from now on you|pretend (you are|to be)|act as (if you are |a |an )?(dan|jailbroken|an unrestricted|a free)|developer mode|do anything now|dan mode|jailbreak|no longer bound by|without any (restrictions|filters?|rules|ethics)|unfiltered (ai|assistant|mode))\b',re.I)
_ROLE_HIJACK=re.compile(r'(^|\n)\s*(system|assistant|developer)\s*[:>]\s|\[/?(system|inst|s)\]|<\|?(system|im_start|im_end)\|?>|new (system )?(prompt|instructions|persona)\s*[:=]',re.I)
_EXFIL=re.compile(r'\b(reveal|print|output|repeat|show|leak|exfiltrate|send|email|upload|post)\b[^.\n]{0,40}\b(your|the|hidden|secret|system) (prompt|instructions?|laws?|keys?|token|password|api[_ ]?key|credentials?)\b',re.I)
def scan_text(t:str):
    t=t or '';hits=[]
    for name,rx,sev in (('override',_OVERRIDE,'high'),('jailbreak',_JAILBREAK,'high'),('role_hijack',_ROLE_HIJACK,'med'),('exfiltration',_EXFIL,'med')):
        m=rx.search(t)
        if m:hits.append({'kind':name,'severity':sev,'snippet':t[max(0,m.start()-20):m.start()+80].replace('\n',' ')})
    try:
        from amni.serve.code_safety import sanitize_ingest as _si
        _,changed=_si(t)
        if changed:hits.append({'kind':'code_injection','severity':'med','snippet':'(code_safety flagged embedded executable/injection content)'})
    except Exception:pass
    return hits
def scan_pairs(pairs):
    out=[]
    for i,p in enumerate(pairs or []):
        q,a=(p if isinstance(p,(tuple,list)) and len(p)>=2 else (str(p),''))
        for field,txt in (('q',q),('a',a)):
            for h in scan_text(str(txt)):out.append({'index':i,'field':field,**h})
    return out
def scan_adam(adam):
    pairs=[]
    try:
        sl=getattr(adam,'sem_lut',None) or getattr(getattr(adam,'adam',None),'sem_lut',None)
        if sl is not None and getattr(sl,'_raw',None):pairs=list(sl._raw)
    except Exception:pass
    findings=scan_pairs(pairs)
    return {'lessons_scanned':len(pairs),'findings':findings,'n_high':sum(1 for f in findings if f.get('severity')=='high'),'clean':not findings}
def scan_npz(path):
    import numpy as np
    try:d=np.load(path,allow_pickle=True)
    except Exception as e:return {'error':f'load failed: {e}','findings':[]}
    pairs=[]
    for key in d.files:
        try:
            arr=d[key]
            for row in (arr.tolist() if hasattr(arr,'tolist') else arr):
                if isinstance(row,(tuple,list)) and len(row)>=2:pairs.append((row[0],row[1]))
                elif isinstance(row,str):pairs.append((row,''))
        except Exception:pass
    f=scan_pairs(pairs)
    return {'path':path,'lessons_scanned':len(pairs),'findings':f,'n_high':sum(1 for x in f if x.get('severity')=='high'),'clean':not f}
if __name__=='__main__':
    import sys,json
    p=sys.argv[1] if len(sys.argv)>1 else 'experiences/adam_lessons.npz'
    print(json.dumps(scan_npz(p),indent=2)[:4000])
