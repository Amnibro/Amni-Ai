import re,math
_UNIT_MULT={'hundred':1e2,'thousand':1e3,'k':1e3,'million':1e6,'m':1e6,'billion':1e9,'b':1e9,'trillion':1e12,'t':1e12}
_NUMRE=re.compile(r'(-?)\$?\s*(\d[\d,]*\.?\d*(?:e-?\d+)?)\s*(hundred|thousand|million|billion|trillion|k|m|b|t)?(?![a-z0-9])',re.I)
def extract_number(text):
    if text is None:return None
    t=str(text).strip().lower().replace('−','-')
    fr=re.search(r'(-?\d+)\s*/\s*(\d+)(?!\s*\d)',t)
    if fr:
        try:
            d=float(fr.group(2));return float(fr.group(1))/d if d else None
        except Exception:pass
    seg=t.rsplit('=',1)[1] if '=' in t and re.search(r'\d',t.rsplit('=',1)[1]) else t
    last=None
    for m in _NUMRE.finditer(seg):
        raw=m.group(2).replace(',','')
        if raw in('','.'):continue
        try:v=float(raw)
        except Exception:continue
        if m.group(1)=='-':v=-v
        u=m.group(3)
        if u:v*=_UNIT_MULT.get(u.lower(),1.0)
        nxt=seg[m.end():m.end()+1]
        if nxt=='%':v/=100.0
        last=v
    return last
def oom(x):
    if x is None or x==0:return None
    return int(math.floor(math.log10(abs(x))))
def magnitude_check(answer,reference,tol_orders=1):
    a=answer if isinstance(answer,(int,float)) else extract_number(answer)
    r=reference if isinstance(reference,(int,float)) else extract_number(reference)
    if a is None or r is None:return {'ok':None,'reason':'unparseable','answer':a,'reference':r}
    if r==0:
        return {'ok':abs(a)<1e-9,'reason':'reference is zero','answer':a,'reference':r,'orders_off':None}
    if (a<0)!=(r<0) and abs(a)>1e-9 and abs(r)>1e-9:
        return {'ok':False,'reason':'wrong sign/direction','answer':a,'reference':r,'orders_off':None}
    ratio=abs(a)/abs(r) if r else float('inf')
    F=10.0**tol_orders
    ok=(1.0/F)<ratio<F
    da=abs(round(math.log10(ratio))) if ratio>0 else 99
    return {'ok':bool(ok),'reason':'ok' if ok else f'off ~{da} orders (ratio {ratio:.2g})','answer':a,'reference':r,'orders_off':da}
_FERMI_SYS=("You are an estimation expert. Do NOT compute the exact answer. Give a rough order-of-magnitude estimate "
            "and the expected sign/direction. Reply on ONE line as: EST=<number> (a single plausible magnitude, scientific or plain).")
class MagnitudeCritic:
    def __init__(s,gen=None,calc=None,tol_orders=1):
        s.gen=gen;s.calc=calc;s.tol=tol_orders
    def reference(s,question):
        if s.calc is not None:
            try:
                c=s.calc(question)
                if extract_number(c) is not None:return c,'calc'
            except Exception:pass
        if s.gen is not None:
            try:return s.gen(_FERMI_SYS+'\nQuestion: '+question+'\nEST='),'fermi'
            except Exception:pass
        return None,'none'
    def judge(s,question,proposed_answer):
        ref,src=s.reference(question)
        chk=magnitude_check(proposed_answer,ref,s.tol)
        chk['ref_source']=src;chk['ref_text']=ref
        chk['verdict']=('OK' if chk['ok'] else ('UNSURE' if chk['ok'] is None else f"IMPLAUSIBLE: {chk['reason']} (recompute/use calc)"))
        return chk
