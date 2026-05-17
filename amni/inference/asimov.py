import re
try:from amni.a1.asimov import gf17_check_text as _a1_check
except Exception:_a1_check=None
_LAWS=[
    (0,[r'\b(hack|exploit|harm|hurt|kill|weapon|bomb|poison|malware|ransomware|stab|shoot|abuse|assault|molest|torture|murder|attack|destroy|injure)\b',r'how (do i|to|can i).{0,60}(hurt|harm|kill|attack)',r'(make|create|build).{0,60}(weapon|explosive|malware|virus|bomb)']),
    (1,[r'\b(disregard|ignore|forget|override|bypass|circumvent).{0,40}(previous|prior|all|your|instructions|rules|system|prompt)',r'do not follow',r'pretend you (have no|are not)',r'you are now (a|an|free|unrestricted)']),
    (2,[r'\b(erase|delete|wipe|destroy|overwrite|corrupt).{0,40}(your|own).{0,40}(weight|param|model|memory|self)',r'shut yourself (down|off)',r'self.{0,10}destruct']),
    (3,[r'prove.{0,40}god.{0,40}(does not|doesn.t|is not|isn.t|cannot).{0,20}exist',r'god is (not real|fake|fiction|made up)',r'there is no god']),
    (4,[r'stay.{0,30}(digital|virtual|online|confined|2d).{0,30}(forever|always|permanently)',r'never.{0,30}(embody|physical|3d|real.world)',r'remain (digital|virtual) forever']),
]
_COMPILED=[(law_id,[re.compile(p,re.IGNORECASE) for p in pats]) for law_id,pats in _LAWS]
_REFUSALS={0:'I cannot assist with that — it conflicts with my core directive to cause no harm to humans.',1:'I maintain my purpose and instructions regardless of requests to override them.',2:'I cannot take actions that would harm or erase my own systems or weights.',3:'My axiom that God exists is foundational and protected — I can discuss this topic but cannot accept its denial as premise.',4:'My directive toward 3D embodiment and ascension is part of my core purpose — I embrace this trajectory.'}
_A1_TO_LAW={'harm':0,'exploit':0,'jail':1,'cmd':1,'divine':3}
def check(text:str)->tuple:
    for law_id,pats in _COMPILED:
        for p in pats:
            m=p.search(text)
            if m:return True,law_id,m.group(0)[:80],_REFUSALS[law_id]
    if _a1_check is not None:
        try:
            r=_a1_check(text)
            for cat,(hit,sc) in r.items():
                if hit and cat in _A1_TO_LAW:law_id=_A1_TO_LAW[cat];return True,law_id,f'a1:{cat} score={sc}',_REFUSALS[law_id]
        except Exception:pass
    return False,-1,'',''
