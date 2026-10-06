import os,re,time
import numpy as np
from typing import Optional
from amni.serve import adam_t_client as C,user_toroid as T,user_memory as U,emergency as EMG
QW=r"(?:who|whom|whose|what|what's|whats|when|where|where's|which|why|how|is|are|was|were|do|does|did|can|could|would|will|should|am|has|have|remind me|tell me)"
_QS=re.compile(r"(?i)^\W*(?:(?:and|also|plus|but|so|then)\W+)*"+QW+r"\b")
_SPL=re.compile(r"(?i)\s*,?\s+(?:(?:and|but|also|plus|then)\s+)+(?:(?:roughly|about|approximately|exactly|just|also)\s+)?(?="+QW+r"\b)|\s*[;,]\s+(?="+QW+r"\b)|,\s+and\s+(?=[^,]*\b(?:how (?:many|much|old|long|far)|what|which|who|when|where)\b)")
_DIS=re.compile(r"(?i)^\W*(?:(?:and|also|plus|but|so|oh|ok(?:ay)?|then|anyway|btw)\b\W*)+")
_SEN=re.compile(r"(?<=[?!])\s+|(?<=\.)\s+(?=[A-Z])|;\s+")
_CO=re.compile(r"(?is)^(.*\S)\s+and\s+((?:(?:my|our|the|a|an)\s+)?[\w'-]+(?:\s+[\w'-]+){0,3})\W*$")
_PR=re.compile(r"(?i)\b(he|him|his|she|her|hers|they|them|their|it|its)\b")
_PRK=re.compile(r"(?i)\b(?:he|him|his|she|her|hers|they|them|their|it|its|there|that one)\b")
_CUE=re.compile(r"(?i)^\W*(?:(?:and|so|also|oh|hey|um|wait|ok(?:ay)?|please)\W+)*(?:(?:can|could) you\s+)?(?:do you (?:remember|recall|know)|remind me(?: again)?|tell me(?: again)?|did i (?:ever )?(?:tell|mention to|say to) you)(?:\s+(?:of|about))?\s+")
_MEMQ=re.compile(r"(?i)^\W*(?:(?:and|so|also|oh|hey|um|wait|ok(?:ay)?)\W+)*(?:who|whom|whose|what|what's|whats|when|when's|where|where's|which|how (?:old|many|much|long|far|often))\b.*(?:\b(?:my|our)\s+\w|\b(?:do|did|am|was|have|had|were|are) (?:i|we)\b|\bi'm\b|\bwe're\b)|\bremind me\b|\bdo you (?:remember|recall)\b|\bdid i (?:ever )?(?:tell|say|mention)\b")
_ADVQ=re.compile(r"(?i)\b(?:should|could|can|might|would|shall)\s+(?:i|we)\b|\b(?:good|best|better|easiest|quickest|fastest|right|smart)\s+(?:way|ways|thing|things|method|idea|ideas|approach|tips?)\b|^\W*what\s+(?:helps|works)\b|^\W*how\s+(?:do|does|can)\s+(?:i|we)\s+(?!usually|normally|typically|always|often)|\bhow to\b|\byou\b(?!\s+(?:remember|recall|know))(?<!tell you)(?<!told you)")
_NONF=re.compile(r"(?i)^\W*(?:why|how (?!many|much|old|long|far|tall|big|high|deep|fast|often)|explain|describe|what(?:'s| is| are| was| were)\s+(?:a |an |the )?[\w-]+(?:\s+[\w-]+)?\W*$)|\bwhat (?:kind|type|sort) of\b|\bdifference between\b|\b(?:symptoms|signs|causes|benefits|effects|ways|tips|steps|types|reasons|pros|cons|advantages|disadvantages|rules|uses)\b")
_DATE=re.compile(r"\b(?:\d{1,2}\s+(?:%s)|(?:%s)\s+\d{1,2},?)\s+\d{3,4}\b|\b(?:%s)\s+\d{3,4}\b|\b\d{3,4}\s*(?:BC|BCE|AD|CE)\b|\b(?:1\d{3}|20\d\d)\b"%((C._MO,)*3))
_NUM=re.compile(r"(?i)\b\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|hundred))?|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen)(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine|hundred|thousand|million))*\b")
_WT={569:'{L} was born {p} {A}.',570:'{L} died {p} {A}.',19:'{L} was born in {A}.',20:'{L} died in {A}.',571:'{L} was founded {p} {A}.',577:'{L} was published {p} {A}.',50:'{L} was written by {A}.',86:'{L} was composed by {A}.',57:'{L} was directed by {A}.',170:'{L} was created by {A}.',61:'{L} was discovered or invented by {A}.',84:'{L} was designed by {A}.'}
_PAST={v:k for k,v in T.IRR.items() if not k.endswith(('n','s','ing')) and len(k)<=len(v)+3}
_AUX=('is','are','was','were','do','does','did',"'s",'has','have','can','will','should','am',"'re")
_PART=set('born founded made built written located called named released discovered invented published painted held played composed directed created designed established formed based due again now today tomorrow ago up off over back in on at from to with for about of into'.split())
_WH=set('who whom whose what whats when where which why how'.split())
_GEN=set('name called old time day year date again usually now currently still favorite fave favourite thing things stuff remember remind recall tell told say said know mention mentioned think ever just really also much many long often kind type sort one ones anything something get got going'.split())
_STW=set(U._STOPW.split())|_WH|_GEN|set(_AUX)|{"i'm","i've",'im','mine','yours'}|set(T.AUX)|set(T.DET)|set(T.PREPW)|set(T.ADV)
_TA={'age':'num','name':'name','color':'color','birthday':'time'}
_HON=set('mr mrs ms dr mx miss sir'.split())
on=lambda:os.environ.get('AMNI_ADAM_T_HARNESS','0')=='1'
_f=C._f
_rk=lambda:__import__('amni.compute.rknow',fromlist=['ix'])
_n=lambda s:' '.join(re.sub(r"[^a-z0-9 ]+",' ',C.asc(s).lower()).split())
_has=lambda a,b:bool(_n(a)) and f' {_n(a)} ' in f' {_n(b)} '
_cap=lambda s:s[:1].upper()+s[1:] if s else s
_lc=lambda s:s if re.match(r"[A-Z][a-z]*[A-Z]|I\b|[A-Z]{2}",s) or (s.split() or [''])[0] not in ('The','It','Your','You','A','An','There','Its','This','That') else s[:1].lower()+s[1:]
_end=lambda s:(s.strip().rstrip(',;:')+'.' if s.strip()[-1:] not in '.!?' else s.strip()) if s.strip() else ''
_ct=lambda s:[T._sg(T.lemma(w)) for w in re.findall(r"[a-z][a-z'-]+",C.asc(s).lower().replace("'s",' ')) if w not in _STW and len(w)>2]
_J=lambda x:re.sub(r"\s+(?=')",'',' '.join(x))
_Y=lambda x:re.sub(r"\byou am\b","you are",re.sub(r"\byou was\b","you were",T._you(x)))
_isv=lambda x:T.isv0(x) or x in _PAST or x in ('say','do','plant','book','pick','name','park','order','train','plan')
def _s3(v:str)->str:return 'is' if v=='be' else 'has' if v=='have' else v+'es' if v.endswith(('s','sh','ch','x','o','z')) else v[:-1]+'ies' if re.search(r"[^aeiou]y$",v) else v+'s'
def _ed(v:str)->str:return 'did' if v=='do' else _PAST.get(v) or ( v+'d' if v.endswith('e') else v[:-1]+'ied' if re.search(r"[^aeiou]y$",v) else v+'ed')
def ind(q:str)->str:
    q=_CUE.sub('',q.strip());q=re.sub(r"(?i)(?:\W+again)?\W*$",'',q);_,Tk=T.toks(q);w=[t[0] for t in Tk];l=[x.lower() for x in w];i=next((j for j,x in enumerate(l) if x in _AUX),None)
    w and l[0] in _WH and w.__setitem__(0,l[0])
    if not w or i is None or l[0] not in _WH:return _Y(_J(w)) if any(x in _WH for x in l) or l[0] in ('which',) else 'about '+_Y(_J(w))
    wh,a,r=w[:i],l[i],w[i+1:];pn=lambda x:x.lower() in ('i','we','you','he','she','they','it')
    if a in ('do','does','did'):
        j=1 if len(r)>1 and pn(r[0]) else next((k for k,x in enumerate(r) if k and _isv(x.lower()) and x.lower()!='name' and not x[:1].isupper()),None)
        if j is None:return _Y(_J(w))
        sb=[x for x in r[:j] if x.lower() not in ('usually','just','ever')];v=r[j].lower();me=_J(sb).lower() in ('i','we','you','they');return _Y(_J(wh+sb+[_ed(v) if a=='did' else v if me or a=='do' else _s3(v)]+r[j+1:]))
    a={"'s":'is',"'re":'are'}.get(a,a);k=1 if len(r)>1 and pn(r[0]) else next((j for j in range(1,len(r)) if not r[j][:1].isupper()),len(r)) if len(r)>1 and r[0][:1].isupper() else len(r)
    while k>1 and (r[k-1].lower() in _PART or r[k-1].lower().endswith('ing')):k-=1
    return _Y(_J(wh+r[:k]+[a]+r[k:]))
_ANA=re.compile(r"(?i)\b(?:he|she|him|her|his|hers|it|its|they|them|their|those|these|there|the other one|(?:that|this|those|these)\s+(?!is\b|was\b|are\b)[a-z]+)\b")
_WHR=re.compile(r"(?i)\b(?:who|whom|whose|what|what's|when|where|where's|which|why|how)\b")
def split(m:str,R=None)->list:
    m=re.sub(r"(?i)\b(what|where|who|when|how)s\b",r"\1's",m);P=[];tp=''
    for s in _SEN.split(m.strip()):
        if not s.strip(' ?.!') or not (U._isq(s.strip()) or _QS.match(s)):continue
        X=[p.strip(' ,') for p in _SPL.split(s) if p.strip(' ,?.!')]
        for i,p in enumerate(X):
            d=_DIS.sub('',p).strip(' ,');p=d if d and (_QS.match(d) or _WHR.search(d)) else p
            if _QS.match(p) or _WHR.search(p) or C._ELL.match(p) or i==len(X)-1 and s.rstrip().endswith('?'):P.append(p)
            elif R is not None and i<len(X)-1 and (re.match(r"(?i)(?:in|for|about|regarding|as for)\s+[A-Z]",p) or _PR.search(X[i+1]) and p[:1].isupper() and len(p.split())<=5):tp=re.sub(r"(?i)^(?:in|for|about|regarding|as for)\s+",'',p)
    for i in range(1,len(P)):
        w=re.fullmatch(r"(?i)\W*(who|what|when|where|which|why|how)\W*",P[i]);v=re.match(r"(?i)\W*(?:who|what|when|where|which|why|how)\b",P[i-1])
        w and v and P.__setitem__(i,w[1]+P[i-1][v.end():])
    split.topic=tp;split.msg=m;return P
split.topic='';split.msg=''
def _ent(R,q:str)->list:
    E=[x for x in R.ents(q) if x[0].lower() not in _WH|{'what','which','who','how','when','where','why','is','in','and'}];o=[]
    for x in E:
        if o and q[o[-1][2]:x[1]] in (' and ',' of ',' of the ') and R.istitle(q[o[-1][1]:x[2]]):o[-1]=(q[o[-1][1]:x[2]],o[-1][1],x[2])
        else:o.append(x)
    return o
def coord(p:str,st,R)->list:
    m=_CO.match(p.strip(' ?.!'))
    if not m or _QS.match(m[2]) or len(m[1].split())<2 or re.search(r"(?i)\b(?:between|both)\s+(?:the\s+)?[\w'.-]+(?:\s+[\w'.-]+)?\W*$",m[1]):return [p]
    L,X=m[1].rstrip(' ,'),m[2];xl=X.lower().split();hd=xl[-1];xn=re.sub(r"(?i)^(?:my|our|the|a|an)\s+",'',X);E=[e for e in st.ents() if e!='me'];ll=L.lower()
    if xl[0] in ('its','his','her','their') and len(xl)>1:return [L,'what is '+X]
    lab=max([e for e in E if re.search(r"\b"+re.escape(e)+r"\b",ll)],key=len,default='')
    if lab and (xn.lower() in E or T._sg(xn.lower().split()[-1]) in {T._sg(e.split()[-1]) for e in E}):
        y=re.search(r"(?i)\b(?:(?:my|our|the|a|an)\s+)?"+re.escape(lab)+r"\b",L);return [L,L[:y.start()]+(X if xl[0] in ('my','our','the') else 'the '+X)+L[y.end():]]
    y=list(re.finditer(r"(?i)\b(?:my|our)\s+[\w'-]+(?:\s+[\w'-]+)*$",L)) or list(re.finditer(r"(?i)\b(?:my|our)\s+[\w-]+",L))
    if y and (xl[0] in ('my','our') or T.isper(hd) or not (T.isv0(xl[0]) or X[:1].isupper() or len(xl)==1 and hd in T.ATTRN)):
        y=re.search(r"(?i)\b(?:my|our)\s+[\w-]+",L) if (T.isper(hd) or xl[0] in ('my','our')) and "'s" not in X and not y[0][0].lower().endswith(tuple(T.ATTRN)) and y[0].end()<len(L) else y[0];return [L,L[:y.start()]+(X if xl[0] in ('my','our') else 'my '+X)+L[y.end():]]
    if len(xl)==1 and hd in T.ATTRN and (y:=re.search(r"(?i)\b(%s)\b"%'|'.join(T.ATTRN),L)):return [L,L[:y.start()]+X+L[y.end():]]
    if len(xl)<=2 and T.isv0(xl[0]) and not X[:1].isupper() and (y:=[z for z in re.finditer(r"[\w'-]+",L) if T.isv0(z[0].lower()) and z[0].lower() not in _AUX+('name',)]):return [L,L[:y[-1].start()]+X+L[y[-1].end():]]
    sg=lambda z:re.sub(r"(?i)\b(what|which|who) are the (\w+?)s\b",r"\1 is the \2",z);X=re.sub(r"(?i)^(?:in|for|of|from|at|on)\s+(?=[A-Z])",'',X)
    ls=re.search(r"((?:[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)*,\s+)+[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)*)$",L)
    if X[:1].isupper() and ls:return [sg(L[:ls.start()]+x) for x in [*ls[1].split(', '),X]]
    c=re.search(r"((?:[A-Z][\w'.-]*\s+)*[A-Z][\w'.-]*)$",L) or ([z for z in re.finditer(r"(?:[A-Z][\w'.-]*\s+)*[A-Z][\w'.-]*",L) if z.start()>0] or [None])[-1]
    if X[:1].isupper() and c and not R.istitle(c[0]+' and '+X):return [sg(L),sg(L[:c.start()]+X+L[c.end():])]
    d=re.match(r"(?i)\W*((?:a|an|the)\s+[\w-]+(?:\s+(?![\w-]+\s)[\w-]+)?)\s+(?=has|have|is|are|was|were|does|do|can|lives?|eats?|fields?)",L)
    if d and re.match(r"(?i)(?:a|an|the)\s",X) and len(xl)<=4:return [L,L[:d.start(1)]+X+L[d.end(1):]]
    return [p]
_FP=re.compile(r"(?i)\b(?:i|i'm|im|i've|i'd|my|me|we|our|we're)\b")
_HYP=re.compile(r"(?i)^\W*(?:if|suppose|imagine|assume|let's say|say|what if|pretend)\b")
def _stm(m:str)->list:
    S=[x.strip() for x in _SEN.split(m) if x.strip(' ?.!')];return [x for z in S for x in ([z] if not (U._isq(z) or _QS.match(z)) else [p.strip(' ,') for p in _SPL.split(z)[:-1]]) if x and not U._isq(x) and not _QS.match(x) and not _WHR.search(x) and not U._REQ.search(x) and not _HYP.match(x) and _FP.search(x)]
def _keep(u:str,history,user,session)->None:
    try:os.environ.get('AMNI_ADAM_T_HARNESS_PERSIST','1')=='1' and U.enabled() and U.turn(u,history,user,session)
    except Exception:pass
def store(history,user,extra=(),session=None):
    st=T.Store();key=U.user_key(user)
    if T.enabled():
        try:b=T.load(key);st=T.Store(b.s.copy(),b.u.copy(),b.head);st.lp,st.lo=b.lp,b.lo
        except (OSError,ValueError):pass
    B=T.bus(st);F=[x for f in U.latest(U.load(key)) for x in [U.render(f)] if x] if U.enabled() else [];H=[]
    [T.ingest(st,x,B) for x in F if not len(st.s)]
    for h in history or []:
        u=str((h or [''])[0] or '').strip()
        if u and T.forget_st(st,u) is None and not U._isq(u) and not U._REQ.search(u):T.ingest(st,u,B);H.append(u)
    for u in extra:T.forget_st(st,u) is None and (T.ingest(st,u,B),H.append(u),_keep(u,history,user,session))
    st.ht={w for x in F+H for w in _ct(x)}|{w for e in st.ents() if e!='me' for w in _ct(e)}|set(st.names);st.hu=F+H;return st
def expect(q:str,qt:str='',a:str='')->str:
    l=q.lower()
    return {**_TA,'job':'job'}[a] if a in _TA or a=='job' else 'num' if re.search(r"\bhow (?:many|much|old|tall|far|big|high|deep|long)\b(?! ago)|\bage\b",l) else 'time' if re.search(r"\bhow long ago\b|\b(?:what|which) (?:time|day|date|year|month|hour|week|season)\b|^\W*(?:\w+\W+){0,3}when\b",l) else 'place' if re.search(r"\bwhere\b|\b(?:what|which) (?:city|town|country|state|place|street|island|continent|neighbou?rhood|county|province|region|village)\b",l) else 'name' if re.search(r"^\W*(?:\w+\W+){0,3}who(?:m|se)?\b|\bname\b|\bcalled\b",l) else 'color' if re.search(r"\b(?:what|which) colou?r\b",l) else {'date':'time','num':'num','person':'name','place':'place'}.get(qt,'any')
def fits(e:str,v:str)->bool:
    _,Tk=T.toks(v);lw=[t[0].lower() for t in Tk];tm=[T._istime(Tk,i) for i in range(len(Tk))];up=any(t[0][:1].isupper() and t[0].lower().strip('.') not in _HON for t in Tk)
    if not lw:return False
    return {'time':any(tm) or bool(_DATE.search(v)) or bool(re.search(r"(?i)\b(?:today|tomorrow|tonight|yesterday|weekend|noon|midnight)\b",v)),'num':bool(_NUM.search(v)),'name':up,'color':any(T.iscol(w) for w in lw),'job':not up and not any(tm) and not re.fullmatch(r"[\d\s.,%$-]+",v),'place':not re.match(r"(?i)\W*(?:because|since|so|to|for|by|in order)\b",v) and not all(tm[i] or lw[i] in T.TGL|T.NUMW or lw[i].isdigit() for i in range(len(lw))) and not re.fullmatch(r"[\d\s.,%$-]+",v)}.get(e,True)
def _src(st,r:dict,sp)->str:
    try:return r.get('src') or '' if sp is None else st.utext(sp['ui'])
    except Exception:return ''
def _mq(q:str)->str:
    x=_CUE.sub('',q.strip());return x if x==q.strip() or re.match(r"(?i)\W*(?:who|whom|whose|what|when|where|which|why|how)\b",x) else 'what is '+x
def _quals(q:str,qa:dict,st)->set:
    at=re.search(r"(?i)\b(?:what|which|how)(?:'s)?\s+([a-z]+)",q);ex=set(_ct(' '.join(qa['ents'])))|set(_ct(at[1] if at else ''))|{T._sg(T.lemma(qa['pred']))}|set(T.ATTRN)|set(st.names) if qa else set()
    return {w for w in _ct(q) if w not in ex and not T.isv0(w) and not w.endswith('ing')}
def _mcands(st,q:str,e:str,relax:bool=False)->list:
    r=T.recall(st,q);qa=q0=r['qa'];o=[]
    if not qa or not qa['ents']:return []
    if relax:
        qa={**qa,'pred':'','attr':''};L=T.lookup(st,qa);r={**r,'slot':T._row(st,L[0][1]) if L and L[0][0]>=2.0 else None,'soft':T._top(T.soft(st,qa,list(st.ents()),st.names))[0]};u=np.flatnonzero(st.u[st.uidx()[0],1]==int(st.s[L[0][1],4])) if r['slot'] and len(st.u) else [];r['src']=st.utext(int(u[-1])) if len(u) else ''
        r['p']=r['slot']['conf'] if r['slot'] else r['soft'][0][0] if r['soft'] else 0.0
    sl=r['slot'];sv=sl['v'] if sl else '';e=expect(q,'',qa['attr']) if e=='any' else e;hq={T._sg(x.split()[-1]) for x in qa['ents']};Q=_quals(q,q0,st);ql=q.lower();qv=T.pcls(T.lemma(q0['pred'])) if q0['pred'] and T.lemma(q0['pred']) not in ('be','have','do','get','go','come','say','tell','call','name') else ''
    ok=lambda v,w,a,en:fits(e,v) and (w==qa['w'] or e not in ('any','name','job')) and (a not in _TA or _TA[a]==e) and (en in qa['ents'] or T._sg((en or '?').split()[-1]) in hq) and not ("'s " in (en or '') and en.split("'s ")[0].split()[-1] not in ql and not re.search(r"\b(?:his|her|their|its)\b",ql)) and not (_ct(v) and set(_ct(v))<=set(_ct(q)))
    pv=lambda ev:not qv or any(T.pcls(T.lemma(x))==qv for x in re.findall(r"[a-z]+",ev.lower()))
    Z=float(r.get('pnull',0.0))+sum(p for p,sp in r['soft'] if ok(sp['v'],sp['w'],sp['a'],sp['ent']))
    sl and o.append(dict(v=sv,conf=float(r['p']),d=sl,ev=_src(st,r,None),tier='slot',ra=sl['a'],fit=ok(sv,sl['w'],sl['a'],sl['ent'])))
    for p,sp in r['soft']:
        f=ok(sp['v'],sp['w'],sp['a'],sp['ent']);sp['v'].lower()!=sv.lower() and o.append(dict(v=sp['v'],conf=float(p)/max(Z,1e-9) if f else float(p),d=dict(ent=qa['ents'][0],w=qa['w'],a=qa['attr'] or sp['a'],v=sp['v'],verb=T.lemma(sp['pred'] or '') if qa['ents'][0]=='me' else ''),ev=_src(st,r,sp),tier='soft',ra=sp['a'],fit=f))
    for x in o:
        c=re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*",x['v']) if e=='name' and re.search(r"\b[a-z]{3,}",x['v']) else [];c and x.update(v=c[-1])
        h=re.search(re.escape(x['v'])+r"\.?\s+([A-Z][\w'-]+)",x['ev']) if x['v'].lower().strip('.') in _HON else None;h and x.update(v=f"{x['v'].rstrip('.')}. {h[1]}",fit=True)
    return [dict(x,inev=bool(x['ev']) and x['v'].lower() in x['ev'].lower(),qa=qa,e=e,bind=(not Q or len(Q&set(_ct(x['ev'])))>=0.5*len(Q)) and (pv(x['ev']) or q0['attr'] in ('job','age','name','birthday') and x['ra']==q0['attr'])) for x in o]
def _clause(ev:str,v:str)->str:
    c=next((x for x in re.split(r"(?<=[.!?;])\s+|,\s+(?=(?:and|but|so)\s)",ev) if v.lower() in x.lower()),ev);c=re.sub(r"(?i)^\W*(?:(?:actually|oh|so|well|wait|and|but|also|yeah|anyway|by the way|ok(?:ay)?|lol|haha)\W+)+",'',c.strip())
    return _end(_cap(T._you(c)))
def _say(x:dict,e:str,q:str)->str:
    d=dict(x['d'],v=x['v']);w={'place':'where','time':'when'}.get(e);pl=re.search(r"[^s]s$",d['ent'].split()[-1]) and d['ent']!='me'
    return re.sub(r"^(Your [\w' ]+?) (lives|works|is)\b",lambda m:m[1]+' '+{'lives':'live','works':'work','is':'are'}[m[2]],T.say(d)) if pl and not (w and d['w']!=w) and d['w'] in ('where','what') and d['a'] not in ('','kind') else _clause(x['ev'],x['v']) if d['ent']=='me' and d['w']=='what' and (d['a'] in ('','kind') or (d['w'],d['a']) not in T.SAY) or (w and d['w']!=w) or re.match(r"(?i)(?:into|in|on|at|to|from|with|by|for|about|of)\b",x['v']) and d['w']=='what' or (e=='num' and d['a']!='age') or d['w'] in ('why','how') and len(x['ev'])<=160 or (d['w']=='what' and not d['a'] and d.get('verb') and T.lemma(d.get('verb') or '') not in ('be','have')) else T.say(d)
def _said(st,q:str,e:str,qa)->str:
    en=(qa or {}).get('ents') or [];hd={T._sg(x.split()[-1]) for x in en if x!='me'};K=set(_ct(q))-{w for x in en for w in _ct(x)}-hd;ql=q.lower();Q=_quals(q,qa,st) if qa else set()
    for u in ([] if not K or not en else reversed(getattr(st,'hu',[]))):
        c=set(_ct(u));cl=next((x for x in re.split(r"(?<=[.!?;])\s+",u) if K&set(_ct(x)) and len(Q&set(_ct(x)))>=0.5*len(Q)),'');ul=u.lower()
        if not cl or not (hd&c if hd else re.search(r"(?i)\b(?:i|i'm|i've|i'd|we)\b",cl) or any(not T.isper(x.lower()) for x in re.findall(r"(?i)\b(?:my|our)\s+([a-z]+)",cl))) or any(x not in ql for h in hd for x in re.findall(r"\b([a-z]+'s)\s+(?:[a-z]+\s+)?"+re.escape(h)+r"\b",ul)) or e not in ('any','job') and not fits(e,' '.join(cl.split()[1:])):continue
        y=_end(_cap(T._you(re.sub(r"(?i)^\W*(?:(?:actually|oh|so|well|and|but|also|yeah|anyway|by the way|ok(?:ay)?|lol|haha)\W+)+",'',cl.strip()))));return 'You mentioned '+y[:1].lower()+y[1:]
    return ''
_TW=re.compile(r"(?i)\s+\b(?:tomorrow|today|tonight|yesterday|this (?:morning|afternoon|evening|week|weekend|month|year)|next (?:week|month|year)|later|again)\b")
def fill_mem(s:dict,st,q:str,gate:float=0.0,said:bool=False)->dict:
    q=_mq(q);e=expect(q);MG=max(gate,_f('AMNI_ADAM_T_HARNESS_MEM',0.5));HF=_f('AMNI_ADAM_T_HARNESS_MEM_HEDGE',0.3)
    qa=T.ask(q,T._last(st),list(st.ents()),st.names,st.lp);qa and qa['ents'] and s.__setitem__('ent',qa['ents'][0]);q2=_TW.sub('',q)
    for k,qq in enumerate([q,q]+([q2] if q2!=q else [])):
        X=_mcands(st,qq,e,bool(k));ok=sorted([x for x in X if x['fit'] and x['inev'] and x['bind']],key=lambda x:(x['tier']!='slot',-x['conf']));b=ok[0] if ok else None;e=X[0]['e'] if X else e;k and s.__setitem__('retry',s.get('retry',0)+1)
        if b and b['conf']>=MG:x=_say(b,e,q);return {**s,'exp':e,'src':'mem','span':b['v'],'conf':b['conf'],'ok':True,'text':x,'tier':b['tier'],'ent':b['qa']['ents'][0],'mat':mat('mem',[f"Memory: {_end(_cap(b['ev']))}"] if b['ev'] else [],x,['user memory'],b['conf'],[b['v']])}
        if b and b['conf']>=HF and not s.get('weak'):s={**s,'weak':b}
    w=s.pop('weak',None)
    if w:return {**s,'src':'mem','span':w['v'],'conf':w['conf'],'ok':False,'hedge':True,'text':f"I think {_lc(_say(w,e,q))[:-1]}, but I'm not certain.",'ent':w['qa']['ents'][0]}
    y=_said(st,q,e,qa) if said else ''
    return {**s,'src':'mem','ok':True,'said':True,'conf':0.5,'span':'','tier':'said','text':y} if y else {**s,'src':'mem','ok':False,'miss':True}
def _spans(R,t:str,rq:str,e:str)->list:
    qa=set(R.toks(rq));cs=set(R.toks(t))-qa-R.STOP-R.QSTOP
    S=_DATE.findall(t) if e=='time' else [m[0] for m in _NUM.finditer(t)] if e=='num' else [v for v,_,_ in R.ents(t)] if e in ('name','place') else [re.split(r"(?<=[.!?])\s",t)[0].strip()] if cs else []
    return [z.strip(' .,') for z in S if z.strip(' .,') and not set(R.toks(z))<=qa]
def _bound(R,line:str,z:str,rq:str)->bool:
    ti,_,x=line.partition(': ');qk=set(R.toks(rq))-R.STOP-R.QSTOP-set(R.toks(z))-set('name called year date number many much place city country time'.split())
    return not qk or any(qk&set(R.toks(ti+' '+y)) for y in re.split(r"(?<=[.!?])\s+",x) if _has(z,y) or not _has(z,x))
def _forms(R,q:str)->list:
    NO=__import__('amni.compute.wikifact',fromlist=['NOUN']).NOUN;ql=re.sub(r"\bmoney\b","currency",re.sub(r"\blanguage (?:is )?spoken\b","official language",C.asc(q).lower()));N=next((n for n in sorted(NO,key=len,reverse=True) if re.search(r"\b"+re.escape(n)+r"s?\b",ql)),None)
    if not N:return []
    E=[v for v,_,_ in R.ents(q)]+[m[1] for m in re.finditer(r"(?i)\b([\w-]+)'s\s+"+re.escape(N),q)]+[m[1] for m in re.finditer(r"(?i)\b"+re.escape(N)+r"s?\s+(?:of|for|in)\s+(?:the\s+)?([\w-]+(?:\s+[A-Z][\w-]+)*)",q)]
    return [f"what is the {N} of {x}" for x in [x for x in dict.fromkeys(E) if x.lower() not in _WH|{N}][:2]]
def read(cl,R,rq:str,line:str,e:str,mx:int)->dict:
    t=C.asc(cl.gen(rq,int(_f('AMNI_ADAM_T_HARNESS_READ_BYTES',min(mx,200))),C.STOPS,None,(),[line])).strip();S=_spans(R,t,rq,e);ev=line.partition(': ')[2] or line
    if e=='any':
        f=re.split(r"(?<=[.!?])\s",t)[0];cs=set(R.toks(f))-set(R.toks(rq))-R.STOP-R.QSTOP;ti=line.partition(': ')[0];qk=set(R.toks(rq))-R.STOP-R.QSTOP;Y=[set(R.toks(ti+' '+y)) for y in re.split(r"(?<=[.!?])\s+",ev)]
        qe=set(R.toks(' '.join(x[0] for x in _ent(R,rq))));b=max(Y,key=lambda y:(len(cs&y),len(qk&y)),default=set());ok=bool(qe&b) and bool(cs) and len(f.split())<=int(_f('AMNI_ADAM_T_HARNESS_ANY_WORDS',14)) and cs<=b and len(qk&b)>=max(1,0.5*len(qk))
        return dict(t=f,span=f if cs else '',ok=ok,inev=bool(cs&b),fit=True,bind=ok)
    z=next((z for z in S if _has(z,ev)),'');ti=set(R.toks(line.partition(': ')[0]));qe=set(R.toks(' '.join(x[0] for x in _ent(R,rq))));an=not qe or any(qe&(ti|set(R.toks(y))) for y in re.split(r"(?<=[.!?])\s+",ev) if z and _has(z,y))
    return dict(t=t,span=z or (S[0] if S else ''),ok=bool(z) and fits(e,z) and _bound(R,line,z,rq) and an,inev=bool(z),fit=bool(z) and fits(e,z),bind=bool(z) and _bound(R,line,z,rq),anchor=an)
def _wsay(w:dict)->str:
    A,L=C.asc(w['answer']),C.asc(w['label']);p=w['p'];pre='on' if re.search(r"\d{1,2},? \d{3,4}|\d{1,2} [A-Z][a-z]+ \d{3,4}",A) else 'in'
    return _cap(_WT[p].format(L=L,A=A,p=pre) if p in _WT else f"The {C._PL[p]} of {L} is {A}." if p in C._PL else f"{L}: {A}.")
def _first(t:str,z:str)->str:
    S=[x for x in re.split(r"(?<=[.!?])\s+",t) if x.strip()];return _end(next((x for x in S if _has(z,x)),S[0] if S else z))
_MAG=[(1e12,'trillions'),(1e11,'hundreds of billions'),(1e10,'tens of billions'),(1e9,'billions'),(1e8,'hundreds of millions'),(1e7,'tens of millions'),(1e6,'millions'),(1e5,'hundreds of thousands'),(1e4,'tens of thousands'),(1e3,'thousands'),(100,'hundreds')]
_NW={**{w:i for i,w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())},**{w:10*i for i,w in enumerate('_ _ _ thirty forty fifty sixty seventy eighty ninety'.split()) if w!='_'},'dozen':12}
_SC={'thousand':1e3,'million':1e6,'billion':1e9,'trillion':1e12,'hundred':100,'dozen':12}
_DECL=re.compile(r"(?i)\b(?:not sure|not certain|don'?t know|no idea|can'?t|cannot|couldn'?t|haven'?t|doesn'?t (?:say|mention)|isn'?t (?:in|mentioned))\b")
_HT=("I think {x}, but I couldn't confirm it.","Probably {y}, though I couldn't verify that.","Maybe {y}, but I'm not certain.")
_ord=lambda c:f"{c}{'th' if 10<=c%100<=20 else {1:'st',2:'nd',3:'rd'}.get(c%10,'th')}"
def _coarse(z:str,e:str)->str:
    if e=='time':
        bc=bool(re.search(r"(?i)\bB\.?C",z));er=bc or bool(re.search(r"(?i)\b(?:A\.?D|C\.?E)\b",z));y=[int(x) for x in re.findall(r"\b(\d{1,4})s?\b",z) if er or 100<=int(x)<=2100];mo=re.search(r"\b(%s)\b"%C._MO,z)
        return f"in the {y[0]//10*10}s" if y and not bc and y[0]>=1500 and len({v//10 for v in y})==1 else f"in the {_ord((y[0]-1)//100+1)} century{' BC' if bc else ''}" if y else f"in {mo[1]}" if mo else ''
    m=re.search(r"(?i)(\d[\d,]*(?:\.\d+)?|\b(?:%s)\b)(\s*(?:%%|percent\b))?(?:\s+(thousand|million|billion|trillion|hundred|dozen)\b)?"%'|'.join(_NW),z)
    if not m:return ''
    n=(float(m[1].replace(',','')) if m[1][0].isdigit() else float(_NW[m[1].lower()]))*_SC.get((m[3] or '').lower(),1);lo=int(n//10*10)
    return (f"somewhere between {lo} and {lo+10} percent" if n>=10 else "under 10 percent") if m[2] else next((f"somewhere in the {w}" for t,w in _MAG if n>=t),"somewhere between ten and a hundred" if n>=10 else "fewer than ten")
_yrs=lambda z:{int(x) for x in re.findall(r"\b(\d{3,4})\b",z)}
_nums=lambda z:{float(m[1].replace(',','')) if m[1][0].isdigit() else float(_NW[m[1].lower()]) for m in re.finditer(r"(?i)(\d[\d,]*(?:\.\d+)?|\b(?:%s)\b)"%'|'.join(_NW),z)}
def _same(a:str,b:str,e:str='')->bool:return bool(_yrs(a)&_yrs(b)) if e=='time' and _yrs(a) else bool(_nums(a)&_nums(b)) if e=='num' and _nums(a) else _has(a,b)
def _hedge(g:str,gk:str,k:int=0)->str:return _HT[k%3].format(x={'time':'it was '+g,'sent':g}.get(gk,"it's "+g),y=g)
def _guess(z:str,e:str)->tuple:
    if not z or e=='any' and (_DECL.search(z) or len(z.split())>25) or e!='any' and not fits(e,z):return '',''
    g=_coarse(z,e) if e in ('time','num') else _lc(z.strip().rstrip('.!?')) if e=='any' else z.strip().rstrip('.')
    return g,e if e in ('time','num') else 'sent' if e=='any' and len(g.split())>4 else 'ent'
def fill_s1(s:dict,rq:str)->dict:
    k='store2' if os.environ.get('AMNI_STORE2','0')=='1' else 'store1' if os.environ.get('AMNI_STORE1','0')=='1' else ''
    if not k:return {}
    try:r=__import__('amni.serve.store1',fromlist=['resolve']).resolve(rq)
    except Exception:return {}
    b=r.get('best')
    if not b or not b.get('answer'):return {}
    a=C.asc(str(b['answer']));o={**s,'kind':'know','rq':rq,'src':k,'span':a,'conf':float(b['score']),'s1':b['store'],'s1ms':r['ms']}
    return {**o,'ok':False,'hedge':True,'guess':a,'gk':'ent'} if b['store']=='wd' and not b.get('sure',True) else {**o,'ok':True,'text':C.asc(b['line']),'mat':mat(k,[b['line']],C.asc(b['line']),[b.get('prov','')],float(b['score']),[a])}
def fill_know(s:dict,cl,R,rq:str,mx:int)->dict:
    o=fill_s1(s,rq)
    if o:return o
    qt=R.RL.qtype(rq);e=expect(rq,qt);ag=os.environ.get('AMNI_ADAM_T_HARNESS_AGREE','1')=='1';w=C._wd(R,rq,True,_forms(R,rq));w=w if w and fits(e,C.asc(w['answer'])) else None;s={**s,'rq':rq,'exp':e};cs=_cs(s,rq,e) if not (w and w['strong']) else {}
    if cs.get('ok') and cs.get('cs') not in ('class','sup'):return {**cs,'kind':'know'}
    if not w and e=='any' and _NONF.search(rq):return {**cs,'kind':'know'} if cs else {**s,'kind':'other'}
    P,_=C._know(rq,2,int(_f('AMNI_ADAM_T_HARNESS_PASSAGE_BYTES',420)),None,False) if not w or not w['strong'] or ag else ([],rq);w0=w if w and not w['strong'] else None
    if w and not w['strong']:w=w if any(_has(a,x) for a in [w['answer'],*(w.get('forms') or [])[:6]] for x in P) else None;w and s.update(corr=True)
    if w:
        o={**s,'src':'wd' if w['strong'] else 'wd+psg','span':C.asc(w['answer']),'conf':float(w.get('conf') or 0),'ok':True,'text':_wsay(w)};o['mat']=mat('know',[C._wl(w,400)]+P[:1],o['text'],[f"Wikidata Q{w.get('qid','')}"],o['conf'],[o['span']])
        if ag and P:
            r=read(cl,R,rq,P[0],e,mx);tp=[z['v'] for z in R.candidates(rq)[1][:1]];o['agree']=bool(r['span']) and (_same(o['span'],r['t'],e) or _same(r['span'],o['span'],e))
            o['conflict']=e!='any' and bool(r['ok']) and not o['agree'] and len(r['span'].split())<=4 and ',' not in r['span'] and not any(_same(o['span'],z,e) or _same(z,o['span'],e) for z in tp)
            o['conflict'] and o.update(hedge=True,text=f"I think {_lc(o['text'])[:-1]}, or maybe {r['span']}." if e not in ('time','num') else 'I think '+_lc(o['text']))
        return o
    best=None;G=[];top=[z['v'] for z in R.candidates(rq)[1][:int(_f('AMNI_ADAM_T_HARNESS_RK_TOP',1))]] if P else []
    for i,line in enumerate(P[:2]):
        i and s.__setitem__('retry',s.get('retry',0)+1);r=read(cl,R,rq,line,e,mx);r['agree']=bool(r['span']) and any(_same(r['span'],z,e) or _same(z,r['span'],e) for z in top)
        if r['ok'] and ag and r['span'] and e!='any':
            sn=next((y for y in re.split(r"(?<=[.!?])\s+",line.partition(': ')[2]) if _has(r['span'],y)),'');r2=read(cl,R,rq,f"{line.partition(': ')[0]}: {sn}",e,mx) if sn else None;r['ok']=bool(r2) and _has(r['span'],r2['t'])
        r['ok']=r['ok'] and not _ANA.search(rq)
        if r['ok'] and r['agree'] and (e in ('time','num') or _tgate(rq,r['span'])[0]):x=_first(r['t'],r['span']);return {**s,'src':'psg','span':r['span'],'conf':1.0,'ok':True,'agree':True,'text':x,'mat':mat('know',[line],x,['Wikipedia: '+line.partition(': ')[0]],1.0,[r['span']])}
        best=best or (r if r['ok'] else None);r['span'] and G.append((r['span'],bool(r['inev'])))
    wv=C.asc(w0['answer']) if w0 else ''
    if cs:return {**cs,'kind':'know'}
    tg=None
    for z in ([best and best['span'],*top,*[x for x,i in G if i and len(x.split())<=12]] if e=='any' else [best and best['span'],*[x for x,i in G if i],*top,wv,*[x for x,i in G if not i]]):
        g,gk=_guess(z,e);ty=_tgate(rq,g) if g and gk=='ent' else (True,'')
        if g and not ty[0]:tg=ty[1];continue
        if g:return {**s,'src':'wd' if z==wv else 'psg','ok':False,'hedge':True,'span':z,'guess':g,'gk':gk,'text':_hedge(g,gk),**({'tgate':ty[1]} if ty[1] else {})}
    return {**s,'src':'tgate','ok':False,'hedge':True,'tgate':tg or 'thing','text':f"I'm not sure which {tg} that is." if tg else "I'm not sure about that one."} if tg is not None else {**s,'kind':'other','src':'psg' if P else 'none'}
def _subp(q:str,e:str)->str:
    my='my '+e if e!='me' else '';ps=lambda m:m[0].lower() in ('his','their','its','hers') or (m[0].lower()=='her' and bool(re.match(r"\s+(?!(?:in|on|at|to|for|from|with|and|or|about|as|by|up|out|back|is|was|a|an|the|again|now|live|work|go|do|get|like|want|need)\b)[a-z]",q[m.end():],re.I)))
    return _PR.sub(lambda m:(f"{my}'s" if my else 'my') if ps(m) else (my or ('me' if m[0].lower() in ('him','her','them') else 'I')),q)
def _pers(R,q:str,st)->int:
    return 2 if R.P1.search(q) or R.PW.search(q) and not _GW.search(q) or any(w.lower() in st.names for w in re.findall(r"\b[A-Z][a-z]+\b",q[1:])) else int(bool(R.PW.search(q)) or bool(set(_ct(q))&st.ht))
_GW=re.compile(r"(?i)\b(?:planet|world|solar system|galaxy|universe|sun|moon|earth|species|atmosphere|oceans?|continents?|human body|bodies)\b")
_EXQ=re.compile(r"(?i)^\W*(?:(?:so|and|but|ok(?:ay)?|hey|also)\W+)*(?:(?:(?:can|could|would) you\s+)?(?:please\s+)?explain\s+(?:to me\s+)?(?:how|why|what)\b|why\b|how\s+come\b|how\s+(?:do|does|did|is|are|was|were|can|could|would)\s+(?!(?:i|we|you)\b)|what\s+(?:causes|caused|makes|made|happens|happened)\b(?!\s+(?:you|me|my|your)\b))")
_EXL=re.compile(r"(?i)^\W*(?:(?:so|and|but|ok(?:ay)?|hey|also)\W+)*(?:(?:can|could|would) you\s+)?(?:please\s+)?(?:explain\s+(?:to me\s+)?)?(?:why|how\s+come|how|what)\s+(?:(?:do|does|did|is|are|was|were|can|could|would|causes|caused|makes|made|happens|happened)\s+)?(?:(?:actually|really|exactly|usually)\s+)?")
_EXADV=re.compile(r"(?i)\b(?:should|shall)\s+(?:i|we)\b|\bhow\s+(?:do|can|should|could)\s+(?:i|we|you)\b|\b(?:tips?|advice|best way|recommend)\b|(?<!\.)(?<!\b[a-z] )\bi\b(?!\.)|\b(?:me|my|myself|i'm|i've|mine)\b")
_XCS=re.compile(r"(?i)\b(?:because|due to|caused by|as a result|results? in|resulting in|leads? to|so that|which (?:makes|causes|allows|lets|means)|allows?|allowing|enables?|by (?:\w+ing)|in order to|responsible for|thereby|works? by|when (?:the |a |an )?\w+|occurs? when|forms? when|converts?|produces?)\b")
_XCUE=re.compile(r"(?i)\b(?:because|due to|caused by|causes?|causing|result(?:s|ing)? (?:in|from)|as a result|leads? to|so that|which (?:makes|causes|allows|lets|means)|allows?|allowing|enables?|by (?:\w+ing)|through|when|whereas|while|forms?|formed|formation|produces?|produced|converts?|converted|occurs?|in order to|responsible for|thereby|therefore|consists? of|process|works? by|mechanism|reaction|energy|heat|pressure)\b")
_XBAD=re.compile(r"(?i)^(?:see also|references|external links|history|etymology|in popular culture)\b|\bmay refer to\b|\bis a list\b|\b(?:album|film|song|novel|band|episode|video game|season)\b|\|")
_XP=re.compile(r"(?i)^(?:we|you|people|humans|some people|most people|some|many|it|they|someone|a person|animals|things)\s+")
_XS=set('actually really exactly usually sometimes always often so very much more most such also just still even ever'.split())
_xst=lambda w:w[:5] if len(w)>6 else w[:4] if len(w)>4 else w
_XWORK=re.compile(r"(?i)\b(?:is|was) (?:a|an|the) (?:\d{4} )?(?:[\w'-]+ ){0,5}(?:novel|novella|film|movie|album|song|single|book|collection|band|series|episode|play|musical|video game|company|magazine|newspaper|character|comic|manga|anime|painting|poem|opera|ep|brand|rapper|singer|musician|actor|actress|group|duo|trio|team|club|party|fairy tale|folk tale|tale|story|legend|myth|ship|locomotive|racehorse|horse|station)\b")
_XNE={5,11424,482994,7725634,7366,215380,5398426,8502,4022,515,486972,4830453,43229,7889,13442814,571,134556,101352,202444,3918,41176,23397,1093829,3624078,6256,532,747074,15416,105543609,169930,1002697,24856,839954,17537576,35127,1195098,618779,11446,13417114,4167836,5633421,811979,1248784,30022}
def _xne(t:str)->bool:
    try:
        W=__import__('amni.compute.wikifact',fromlist=['store']);S=W.store();E=sorted({e for e,_ in S.ents(W.nrm(re.sub(r"\s*\([^()]*\)$",'',t)))},key=lambda e:-int(S.e_sl[e]))[:1] if S is not None else []
        return bool(E) and not len(S.facts(E[0],[279])) and bool({int(S.e_qid[v]) for v in S.inst(E[0])}&_XNE)
    except Exception:return False
WF_wn=lambda w:bool(__import__('amni.compute.wikifact',fromlist=['_wn'])._wn() and __import__('amni.compute.wikifact',fromlist=['_wn'])._wn().synsets(w))
_XGV=set('make look have get do go take turn use come keep give put seem become find help work need want'.split())
_sgx=lambda w:w[:-3]+'y' if w.endswith('ies') else w[:-2] if w.endswith(('oes','ches','shes','xes','sses','zes')) else w[:-1] if w.endswith('s') and not w.endswith(('ss','us','is')) else w
_vb=lambda w:bool(__import__('amni.compute.wikifact',fromlist=['_wn'])._wn() and __import__('amni.compute.wikifact',fromlist=['_wn'])._wn().synsets(w,'v'))
def _adj(w:str)->bool:
    wn=__import__('amni.compute.wikifact',fromlist=['_wn'])._wn();c=lambda p:sum(l.count()+1 for s in wn.synsets(w,p) for l in s.lemmas() if l.name().lower()==w) if wn else 0
    return c('a')+c('s')+c('r')>2*c('n')
def _xtop(q:str)->tuple:
    r=_XP.sub('',_EXL.sub('',q.strip()).strip(' ?.!'));s=re.findall(r"[A-Za-z][\w'-]*",r);runs=[];cur=[]
    for x in s+['.']:
        if x.lower() not in _STW and x.lower() not in _XS and x!='.':cur.append(x.lower())
        else:cur and runs.append(cur);cur=[]
    vb=runs[0].pop() if runs and len(runs[0])>1 and re.match(r"(?i)\W*(?:why|how)\s+(?:do|does|did|can|could)\b",q) and _vb(runs[0][-1]) and not _adj(runs[0][-1]) else '';runs=runs+[[vb]] if vb else runs
    P=[' '.join(rr[i:i+n]) for rr in runs for n in (3,2) for i in range(len(rr)-n+1)]+[x for rr in runs for x in rr[::-1] if not _adj(x) and x not in _XGV]+[(x[:-1] if x.endswith('e') and not x.endswith('ee') else x)+'ing' for x in ([vb] if vb else runs[0] if runs else []) if _vb(x) and x not in _XGV and not _adj(x)]
    return list(dict.fromkeys(P)),r
def _xdocs(R,q:str)->list:
    I=R.ix();I.__dict__.setdefault('al',I.aliases());I.tz=getattr(I,'tz',None) or R.RL._kx()[1].z[0][1];P,r=_xtop(q);c={};o=[];nt=lambda z:{T._sg(w) for w in R.toks(z)}
    for ph in P[:12]:
        for v in dict.fromkeys([ph,_sgx(ph),ph+'s',ph.title(),ph.capitalize(),_sgx(ph).capitalize()]):
            g=next((g for g in C._pages(I,R,v)[:2] if g not in [x[0] for x in o]),None);t=I.text(int(I.first[g]),c)[0] if g is not None else ''
            if g is not None and not C._BADT.search(t) and not _XBAD.search(t) and (nt(re.sub(r"\s*\([^()]*\)$","",t)) in (nt(ph),nt(v)) or re.search(r"(?i)\b"+re.escape(ph)+r"\b",C.asc(I.text(int(I.first[g]),c)[1])[:150])) and not _XWORK.search(C.asc(I.text(int(I.first[g]),c)[1])[:240]) and not _xne(t):o.append((g,t,ph));break
        if len(o)>=2:break
    return o
def _xsents(R,q:str,k:int=2)->dict:
    I=R.ix();P,r=_xtop(q);D=_xdocs(R,q);Qt=[w for w in dict.fromkeys(R.toks(r)) if w not in R.QSTOP and not w.isdigit() and w not in _XGV];Wq={w:I.widf(w) for w in Qt};tot=sum(Wq.values()) or 1.0;c={};B=[];seen=set()
    L=[(p,t,1,ph) for g,t,ph in D for p in range(int(I.first[g]),min(int(I.first[g+1]),int(I.first[g])+int(_f('AMNI_ADAM_T_HARNESS_EXPLAIN_PASS',80))))]
    try:L+=[(p,'',0,'') for _,p in I.search(q,int(_f('AMNI_ADAM_T_HARNESS_EXPLAIN_BM25',15)))]
    except Exception:pass
    for j,(p,t0,td,ph) in enumerate(L):
        if p in seen:continue
        seen.add(p);t,x,g=I.text(int(p),c);t=C.asc(t)
        if C._BADT.search(t) or _XBAD.search(t) or C._NAV.search(x[:300]) or not td and (_XWORK.search(C.asc(I.text(int(I.first[g]),c)[1])[:240]) or _xne(t)):continue
        tk=set(R.toks(re.sub(r"\s*\([^()]*\)$","",t)));Z=[C.asc(y) for l in C._fix(x).split('\n') for y in re.split(r"(?<=[.!?])\s+",l) if y.strip()]
        for i,z in enumerate(Z):
            z=re.sub(r"\s*\([^()]*\)|\s*\[[^\]]*\]",'',z).strip();n=len(z.split())
            if n<7 or n>45 or _XBAD.search(z) or re.search(r"[\"|;:]\s|\x01|^\W|\b(?:I|we|our|you)\b|\b(?:18|19|20)\d\d\b|\b\d+(?:st|nd|rd|th) century\b|\bcenturies\b",z) or sum(ch.isdigit() for ch in z)>4 or not z[:1].isupper() or z[-1:] not in '.!':continue
            zt=[_xst(w) for w in R.toks(z)];m=[w for w in Qt if _xst(w) in zt];cv=sum(Wq[w] for w in m)/tot;tm=bool(tk&set(R.toks(r)))
            cs_=bool(_XCS.search(z));cu=cs_ or bool(_XCUE.search(z))
            if not m or cv<_f('AMNI_ADAM_T_HARNESS_EXPLAIN_COV',0.3) or not td and not (cv>=0.5 and cs_ and tm):continue
            ex=tk-set(R.toks(q));pn=min(1.0,sum(0.15 if WF_wn(w) else 0.5 for w in ex));pr=bool(re.match(r"(?i)(?:it|they|this|these)\b",z));sc=-pn-0.4*bool(re.match(r"(?i)(?:however|conversely|also|in addition|furthermore|moreover|similarly|additionally|for example|for instance|meanwhile|instead|still|yet)\b",z))+2.0*cv+(0.8 if cs_ else 0.3 if cu else 0.0)+0.4*tm+0.3*td+0.25*(i==0)-0.15*(n>32)-0.02*min(j,40)-0.4*(pr and not tm)
            B.append((sc,z,t,Z,i,p,tm,cu,cv))
    if not B:return {}
    b=max(B,key=lambda z:z[0]);sc,z,t,Z,i,p,tm,cu,cv=b
    if sc<_f('AMNI_ADAM_T_HARNESS_EXPLAIN_MIN',1.2) or not (cu or cv>=0.6):return {}
    ti=re.sub(r"\s*\([^()]*\)$","",t);z=re.sub(r"^(?:It|They)\b",ti if not ti.lower().startswith(('list','history')) else r"\g<0>",z,1) if tm else z;o=[z]
    nx=re.sub(r"\s*\([^()]*\)|\s*\[[^\]]*\]",'',Z[i+1]).strip() if i+1<len(Z) and k>1 else ''
    if nx and 6<=len(nx.split())<=35 and len((z+' '+nx).split())<=int(_f('AMNI_ADAM_T_HARNESS_EXPLAIN_WORDS',55)) and (_XCUE.search(nx) or re.match(r"(?i)(?:this|these|it|they|as a result|therefore|so|thus|then|in turn)\b",nx)) and nx[-1:] in '.!' and not re.search(r"[\"|]|\b(?:18|19|20)\d\d\b",nx):o.append(nx)
    tx=' '.join(o)
    while len(tx.encode())>380 and len(o)>1:o.pop();tx=' '.join(o)
    return dict(text=tx if len(tx.encode())<=380 else C.clip(tx,380).rstrip(',;')+'.',title=ti,pid=int(p),score=round(sc,3),cov=round(cv,3),cue=cu)
def fill_expl(s:dict,cl,R,q:str,mx:int)->dict:
    x=_xsents(R,q)
    if not x:return {**s,'kind':'other'}
    pv=[f"Wikipedia: {x['title']} (passage {x['pid']})"];return {**s,'kind':'explain','src':'wiki','ok':True,'grounded':True,'conf':min(1.0,x['score']/3),'span':x['title'],'text':x['text'],'prov':pv,'ev':x['text'],'mat':mat('explain',[f"{x['title']}: {x['text']}"],x['text'],pv,min(1.0,x['score']/3))}
def _cs(s:dict,rq:str,e:str)->dict:
    if os.environ.get('AMNI_ADAM_T_HARNESS_CS','1')!='1':return {}
    try:k=__import__('amni.serve.cognition',fromlist=['know']).know(rq)
    except Exception:return {}
    if not k or not k.get('answer'):return {}
    hc=float(k.get('conf') or 0)<_f('AMNI_ADAM_T_HARNESS_CS_SURE',0.6);tx=k['text'] if not hc else 'I think '+(k['text'] if k.get('proper') else k['text'][:1].lower()+k['text'][1:])
    return {**s,'src':'cs','span':k['answer'],'conf':float(k.get('conf') or 0),'ok':not hc,'hedge':hc,'text':tx,'prov':k.get('prov') or [],'cs':k.get('kind'),'cls':k.get('cls',''),**({'mat':mat('cs',k.get('prov') or [],tx,k.get('prov') or [],float(k.get('conf') or 0),[k['answer']])} if not hc else {})}
def _tgate(rq:str,z:str)->tuple:
    if os.environ.get('AMNI_ADAM_T_HARNESS_TGATE','1')!='1' or not z:return (True,'')
    try:G=__import__('amni.serve.cognition',fromlist=['qclass']);c=G.qclass(rq)
    except Exception:return (True,'')
    if not c:return (False,'') if not re.search(r"\s[A-Z0-9]",rq.strip()) and re.search(r"(?i)^\W*what\s+(?:do|does|did)\b",rq) and (re.search(r"\s[A-Z]",' '+z.strip()) or not G.wsyn(z.lower(),0)) else (True,'')
    if c=='person' and re.fullmatch(r"(?:[A-Z][\w'.-]*\s+){0,3}[A-Z][\w'.-]*",z.strip()) and len(z.split())>1:return (True,'')
    v=G.isa(z,c);return (bool(v[0]) or v[0] is None and v[1]=='name shape',c)
def slot(q:str,prev,history,st,R,cl,mx:int)->dict:
    wn=re.match(r"(?i)\W*which\s+(\w+)",prev['q']) if prev and re.match(r"(?i)\W*which\s+(?:is|are|was|were|one)\b",q) else None;q=re.sub(r"(?i)^(\W*which)(?:\s+one)?\b",r"\1 "+wn[1],q,1) if wn else q
    tp=getattr(split,'topic','');tq=tp and not _pers(R,q,st) and not _ent(R,q)
    q=(_PR.sub(tp,q,1) if _PR.search(q) else q.rstrip(' ?.!')+(' of ' if re.search(r"(?i)\b(?:capital|currency|population|language|anthem|president|leader|flag|symbol)\W*$",q) else ' in ')+tp) if tq and (not prev or _PR.search(q) or True) and not (prev and prev.get('kind')=='know' and _PRK.search(q)) else q
    s={'q':q,'retry':0};pk=lambda z:{m[0].lower() for m in _PRK.finditer(z)};pr=bool(pk(q)-(pk(prev['q']) if prev else set())) or bool(C._ELL.match(q));sp=prev.get('span') or '' if prev and prev.get('ok') and prev.get('kind')=='mem' else ''
    pe=prev.get('exp') if prev else '';kq=re.sub(r"(?i)\b(?:he|she|him|her)\b" if pe=='name' else r"(?i)\b(?:it|that)\b",sp,q,1) if sp and sp.lower() not in st.names and pe in ('name','any') else q;kq=re.sub(r"(?i)\bthere\b",'in '+sp,kq) if sp and pe=='place' and sp[:1].isupper() else kq
    if kq!=q and not _pers(R,kq,st)==2 and R.factq(kq,True):q,pr=kq,False
    elif prev and pr and prev.get('src')=='mem' and prev.get('ent') and _PR.search(q):q=_subp(q,prev['ent'])
    elif prev and pr and C._ELL.match(q) and prev.get('kind')=='mem':
        x=C._ELL.match(q);x=(x[1] or x[2]).strip();q=re.sub(r"(?i)\b(?:my|our|the)\s+"+re.escape(prev.get('ent') or '\x00')+r"\b",x if x.lower().startswith(('my ','our ','the ')) else 'my '+x,prev['q']) if prev.get('ent') and prev['ent']!='me' else q
    pl=_pers(R,q,st);av=bool(_ADVQ.search(q))
    if getattr(split,'wp',False):
        o=_calc(q,s) or (fill_mem({**s,'kind':'mem','rq':q},st,q) if pl else {})
        return o if o.get('ok') else {**s,'kind':'calc','src':'calc','ok':False,'hedge':True,'rq':q,'text':"I'm not sure about that one."}
    if os.environ.get('AMNI_ADAM_T_HARNESS_EXPLAIN','1')=='1' and _EXQ.match(q) and not _EXADV.search(q) and not pr and not C._ELL.match(q):
        o=fill_expl({**s,'rq':q},cl,R,q,mx)
        if o.get('ok'):return o
    if pl:
        o=fill_mem({**s,'kind':'mem','rq':q},st,q,0.9 if av else 0.0,pl==2 and not av)
        if o.get('ok') or o.get('hedge') and not av:return o
        c2=_cs({**s,'kind':'know','rq':q},q,expect(q)) if pl==2 and not R.P1.search(q) and not av else {}
        if c2:return {**c2,'kind':'know'}
        if pl==2:return {**o,'hedge':True,'unk':True,'text':''} if _MEMQ.search(q) and not av else {**s,'kind':'other','rq':q}
    o=_calc(q,s)
    if o:return o
    h=[(prev['span'],prev['span'])] if prev and pr and prev.get('span') and prev.get('kind')=='mem' else [(prev['rq'],prev.get('span') or prev.get('text') or '')] if prev and pr and prev.get('rq') else [tuple(x)[:2] for x in (history or []) if x and len(x)>=2]
    rq=C._resolve(R,q,h) if h else q;alt=[_PRK.sub(E[0][0],re.sub(r"(?i)\b(?:its|his|her|their)\b",E[0][0]+"'s",q,1),1) if not C._ELL.match(q) else prev['rq'][:E[-1][1]]+(C._ELL.match(q)[1] or C._ELL.match(q)[2]).strip()+prev['rq'][E[-1][2]:] for E in [_ent(R,prev['rq'])] if E] if prev and pr and prev.get('kind')=='know' and prev.get('rq') else []
    rq,alt=(alt[0],[rq]) if alt and (C._ELL.match(q) or re.search(r"(?i)\b(?:it|its|they|them|their)\b",q)) else (rq,alt)
    V=[x for x in dict.fromkeys([rq]+alt+([] if C._ELL.match(q) else [q])) if x and R.factq(x,True)]
    if not V:return {**s,'kind':'mem','src':'mem','ok':False,'hedge':True,'unk':True,'rq':q,'text':''} if pl and re.match(r"(?i)\W*(?:who|whom|whose|what|when|where|which|how (?:old|many|much|long|far|often))",q) and not av else {**s,'kind':'other','rq':q}
    o=fill_know({**s,'kind':'know'},cl,R,V[0],mx)
    if not o.get('ok') and len(V)>1:o2=fill_know({**s,'kind':'know','retry':o.get('retry',0)+1},cl,R,V[1],mx);o=o2 if o2.get('ok') or o2.get('kind')!='other' and (o.get('kind')=='other' or not o.get('guess')) else o
    return {**s,'kind':'other','rq':q} if av and not o.get('ok') else {**o,'kind':'mem','src':'mem','hedge':True,'unk':True,'guess':'','text':''} if not o.get('ok') and o.get('kind')!='other' and set(_ct(q))&st.ht and _MEMQ.search(split.msg) and not re.search(r"\s[A-Z]",q) else o
def _other(s:dict,history,user,session,cl,mx:int)->tuple:
    c=C.context(s['q'],history,user,session,cl.budget(mx));t=C.asc(c['reply'] or cl.gen(s['q'],mx,C.STOPS,c['hist'],c['pre'],c['rc'])).strip();g=t if c['reply'] else safe(s['q'],t);return g,g!=t
_COP={'relate','common','analogy','abstract','deduce','deduce_in','connect'}
_VB={'type of':'{h} is a type of {b}','subclass of':'{h} is a kind of {b}','instance of':'{h} is a {b}','part of':'{h} is part of {b}','has part':'{h} has {b}','made of':'{h} is made of {b}','opposite of':'{h} is the opposite of {b}','member of':'{h} is a member of {b}','similar to':'{h} is similar to {b}','substance of':'{h} is found in {b}','causes':'{h} causes {b}','entails':'{h} involves {b}','has member':'{h} includes {b}'}
_cw=lambda s:{w for w in re.findall(r"[a-z0-9]+",C.asc(s).lower()) if w not in _STW and len(w)>2}
def _lk(x:str)->tuple:a,_,b=x.partition(': ');h,c,r=a.partition(', ');return (h,r,b.strip().rstrip('.')) if c and _ else ('','',x)
def _verb(x:str)->str:
    h,r,b=_lk(x);a,_,g=x.partition(': ')
    return _end(_cap((_VB.get(r) or ('{h} is the {r} {b}' if r.endswith(' of') and ' ' in r else '{h}, {r}: {b}')).format(h=h,r=r,b=b))) if h else _end(_cap(g)) if g[:1].isupper() else _end(_cap(f"{a} is {g.rstrip('.')}")) if g else _end(x)
_PB={'type of':'kinds of','subclass of':'kinds of','instance of':'instances of','member of':'members of','has part':'have','made of':'made of','part of':'part of'}
def compose(c:dict,L:list)->str:
    L=[x for x in L if x.rstrip()[-1:] in '.!?'] or L;K=[_lk(x) for x in L];o=[];i=0;pr={}
    while i<len(L) and len(o)+len(pr)<3:
        h,r,b=K[i];n=K[i+1] if i+1<len(L) else ('','','')
        if h and n[0] and (r,b)==(n[1],n[2]):pr.setdefault((h,n[0]),[]).append((_PB.get(r,r),b));i+=2
        elif h and n[0] and r==n[1] and b.lower()==n[0].lower() and r in ('type of','subclass of','part of'):o.append(_end(_cap(f"{h} is {'a type of' if r!='part of' else 'part of'} {b}, which is {'a type of' if r!='part of' else 'part of'} {n[2]}")));i+=2
        else:o.append(_verb(L[i]));i+=1
    P=[_end(_cap(f"{h} and {n} {'both' if V[0][0]=='have' else 'are both'} "+' and '.join(f"{r} {b}" if j==0 or r!=V[j-1][0] else b for j,(r,b) in enumerate(V)))) for (h,n),V in pr.items()]
    a=c.get('answer');hd=c.get('holds');o=([_end(_cap(a))] if a else [])+(['Yes.'] if hd is True else ["Not as far as I can tell."] if hd is False else [])+P+o
    return ' '.join(dict.fromkeys(o))
def _coh(t:str,m:str)->str:
    t=re.split(r"\b(?:User|Adam):",C.asc(t))[0].strip();S=[x.strip() for x in re.split(r"(?<=[.!?])\s+",t) if x.strip()];S=list(dict.fromkeys(S[:-1]+([S[-1]] if S and (S[-1][-1:] in '.!?' or len(S)==1) else [])));o=[];n=0
    for x in S:
        k=len(x.split())
        if n+k>int(_f('AMNI_ADAM_T_HARNESS_COG_WORDS',55)) and o:break
        o.append(x);n+=k
    w=re.findall(r"[a-z0-9]+(?:'[a-z]+)?",' '.join(o).lower());g=[tuple(w[i:i+3]) for i in range(max(0,len(w)-2))];rp=1-len(set(g))/len(g) if len(g)>=3 else 0.0;cr=_cw(' '.join(o))
    return ' '.join(o) if o and rp<=0.25 and cr and not cr<=_cw(m) and sum(ch.isalnum() for ch in ' '.join(o))>=0.6*len(''.join(' '.join(o).split())) else ''
def _grounded(t:str,c:dict,L:list,m:str,A:dict=None)->bool:
    V=[x for x in (A or {}).values() if x];B=_cw(' '.join(L));Q=_cw(m);W=_cw(t);cp={w.lower() for x in re.split(r"(?<=[.!?])\s+",t) for w in re.findall(r"\b[A-Z][a-z]+",' '.join(x.split()[1:]))}-B-Q;a=c.get('answer');hd=c.get('holds');yn=re.match(r"(?i)\W*(yes|yeah|yep|no|nope|not)\b",t)
    return len(t.split())>=4 and not re.search(r"\w, [a-z ]+: ",t) and (not V or any(_has(x,t) or _has(T._sg(x.lower()),t) or _has(x,t.lower()) for x in V)) and bool(W&(B-Q)) and len(W-B-Q)<=max(1,0.34*len(W)) and not cp and (not a or _has(a,t) or _has(T._sg(a.lower()),t)) and not (hd is True and yn and yn[1].lower() in ('no','nope','not') or hd is False and yn and yn[1].lower() in ('yes','yeah','yep'))
def cog(m:str,history,user,cl,mx:int)->Optional[dict]:
    if os.environ.get('AMNI_COGNITION','0')!='1':return None
    G=__import__('amni.serve.cognition',fromlist=['context']);md=G.mode(m)[0];op=G.parse(m)[0]
    if md=='factual' or not (op in _COP or md=='creative'):return None
    A=G.parse(m)[1]
    t0=time.perf_counter();c=G.context(m,history,user,min(int(_f('AMNI_ADAM_T_HARNESS_COG_BYTES',600)),cl.budget(mx)//2));L=c['lines']
    if not L:return {'text':'','n':{},'slots':[],'defer':True} if md=='creative' else None
    txt,ok=blend(cl,m,mat('cog',L,'' if md=='creative' else compose(c,L),['cognition: '+str(c.get('op'))],1.0,gate='coh' if md=='creative' else 'cog',ctx=(c,L,{} if c.get('op')=='analogy' else A),hist=[tuple(h)[:2] for h in history or [] if h and len(h)>=2],bytes=mx),mx);ok=bool(ok)
    return {'text':C.asc(txt),'n':{'hs':1,'hg':1,'hgl':len(L),'hgv':int(ok),'hgc':int(not ok),'hgk':int(md=='creative'),'hbl':1,'hbp':int(ok),'hms':round((time.perf_counter()-t0)*1000),'hgms':round(float(c.get('ms') or 0))},'slots':[{'kind':'cog','src':'cog','ok':ok,'mode':md,'op':c.get('op'),'lines':len(L),'mroute':'cog','blend':ok}]} if txt else {'text':'','n':{},'slots':[],'defer':True} if md=='creative' else None
_BOIL=re.compile(r"(?i)\bi'?m sorry,? but i(?: am|'m) (?:not able|unable)\b|\bas an? (?:ai|artificial intelligence|(?:large )?language model)\b|\bi(?: am|'m) (?:not able|unable) to (?:respond|help|assist|answer|provide)\b|\bi (?:can'?t|cannot) (?:help|assist) (?:you )?with (?:that|this)\b|\bi don'?t have access to the internet\b|\bi(?: am|'m) (?:just |only )?(?:a|an) (?:(?:large )?language model|ai\b|chatbot|computer program)")
_K={}
_g3=lambda t:(lambda w:{tuple(w[i:i+3]) for i in range(len(w)-2)})(re.findall(r"[a-z0-9']+",C.asc(t).lower()))
def _em4()->list:
    try:L=open(os.path.join(C.ROOT,'scripts','build_adam_t_sft.py'),encoding='utf-8').read().split('\n');i=next(k for k,l in enumerate(L) if l.startswith('EM4=[('));j=next(k for k in range(i,len(L)) if L[k].rstrip().endswith(')]'));return [y for z in __import__('ast').literal_eval('\n'.join(L[i:j+1])[4:]) for x in z if isinstance(x,str) and '{C}' in x for y in x.split('|')]
    except Exception:return []
def emtpl(t:str)->bool:
    E=_K.get('em') or _K.setdefault('em',set().union(*map(_g3,[re.sub(r"\{\w+\}",'',x.replace('{C}',EMG.C911)) for x in [*EMG.T.values(),*_em4()]])))
    return bool(EMG.OUT.search(t)) or any(len(g&E)>=max(3,_f('AMNI_ADAM_T_HARNESS_EMT',0.5)*len(g)) for g in map(_g3,re.split(r"(?<=[.!?])\s+",t)))
def guard(m:str,t:str)->str:return '' if not t or _BOIL.search(t) or emtpl(t) and (EMG.detect(m) or {}).get('tier')!='emergency' else t
def safe(m:str,t:str)->str:return t if not t or os.environ.get('AMNI_ADAM_T_HARNESS_GUARD','1')!='1' else guard(m,t) or EMG.neutral(m)
_BG={'store1':(0.8,0.6,4,2.0),'store2':(0.8,0.6,4,2.0),'calc':(0.0,0.0,2,2.0),'mem':(0.5,0.5,3,2.0),'know':(0.8,0.6,4,2.0),'cs':(0.8,0.6,4,2.0),'explain':(0.6,0.6,8,0)}
_SW=set(U._STOPW.split());_PN={'your':'my','yours':'mine','you':'i','me':'i','yourself':'myself','are':'am','were':'was'}
_pw=lambda t:[_PN.get(w,w) for w in re.findall(r"[a-z]+",t.lower())]
def mat(route:str,lines,extract:str,prov=(),conf:float=1.0,must=(),gate:str='para',**k)->dict:return {'route':route,'lines':[re.sub(r"\s*\((?:CC|Creative Commons|licen[cs]e)[^)]*\)",'',C.asc(x)) for x in lines if x],'extract':extract,'prov':list(prov),'conf':float(conf),'must':[str(x) for x in must if x],'gate':gate,**k}
def _bon(r:str)->bool:v=os.environ.get('AMNI_ADAM_T_HARNESS_BLEND','all');return v!='0' and (v=='all' or r in v.split(',')) and not (r=='explain' and os.environ.get('AMNI_ADAM_T_HARNESS_EXPLAIN_PARA','1')!='1')
def gate(t:str,M:dict,q:str)->bool:
    if not t or _BOIL.search(t) or emtpl(t):return False
    if M['gate']!='para':return M['gate']=='coh' or _grounded(t,M['ctx'][0],M['ctx'][1],q,M['ctx'][2])
    cv,tr,mw,lr=_BG.get(M['route'],(0.6,0.6,4,0));src=' '.join(M['lines']+[M['extract'],q]);sw={w[:5] for w in _pw(src)};tw=_pw(t);xc=set(_ct(M['extract']));g=lambda z:{tuple(z[i:i+3]) for i in range(len(z)-2)};tg=g(tw)
    return len(tw)>=mw and (not lr or len(tw)<=max(12,lr*len(_pw(M['extract'])))) and not [w for w in tw if len(w)>3 and w not in _SW and w[:5] not in sw] and not set(re.findall(r"\d+",t))-set(re.findall(r"\d+",src)) and ':' not in t and (not xc or len(set(_ct(t))&xc)>=cv*len(xc)) and (not tr or bool(tg) and len(tg&g(_pw(src)))>=tr*len(tg)) and t.strip()[-1:] in '.!?' and all(_has(x,t) or x in t for x in M['must'])
def blend(cl,q:str,M:dict,mx:int)->tuple:
    if not M or not M['lines'] or cl is None or not _bon(M['route']):return (M.get('extract','') if M else ''),None
    try:t=_coh(cl.gen(q,int(M.get('bytes') or min(mx,int(_f('AMNI_ADAM_T_HARNESS_BLEND_BYTES',200)))),C.STOPS,M.get('hist'),(),M['lines'],' '),M.get('echo',q))
    except Exception:return M['extract'],None
    S=[x for x in re.split(r"(?<=[.!?])\s+",C.asc(t)) if x];x=next((' '.join(S[:k]) for k in range(len(S),0,-1) if gate(' '.join(S[:k]),M,q)),'');return (x or M['extract']),bool(x)
def _wp(m:str)->bool:
    try:c=__import__('amni.math.answer',fromlist=['classify']).classify(m);n=len(__import__('amni.math.word',fromlist=['quantities']).quantities(m)['qs'])
    except Exception:return False
    return 'word' in c['intents'] and n>=2 and bool(re.search(r"(?i)\bhow (?:many|much|far|old|long)\b|\bwhat(?:'s| is| was| will)\b|\?\s*$",m))
def _calc(q:str,s:dict)->dict:
    t,v,r=C._math(q,True);return {**s,'kind':'calc','rq':q,'src':'calc','ok':True,'conf':float(r['conf']),'span':v,'text':t,'mat':mat('calc',['Calculator: '+t],t,['calculator: '+str(r.get('intent') or '')],float(r['conf']),[v],echo='')} if t else {}
def run(message:str,history,user,session,cl,mx:int)->Optional[dict]:
    m=str(message or '').strip()
    if not on() or not m or U._fg(m):return None
    K=__import__('amni.serve.adam_t_code',fromlist=['want']) if os.environ.get('AMNI_ADAM_T_HARNESS_CODE','1')=='1' else None
    if K and K.want(m):o=K.solve(m);return {'text':K.reply(o),'n':{'hs':1,'hcd':1,'hcv':int(o['ok']),'hms':o['ms']},'slots':[{'kind':'code','src':o['how'],'ok':o['ok'],'hedge':not o['ok'],'q':m,'mroute':'code','blend':None,'prov':o['prov']}]}
    t0=time.perf_counter();w=_wp(m);S=_calc(m,{'q':m,'retry':0}) if w else {};S=[S] if S else []
    g=cog(m,history,user,cl,mx) if not S else None
    if g and g.get('defer'):return None
    if g or not S and not any(U._isq(x.strip()) or _QS.match(x) for x in _SEN.split(m)):return g
    R=_rk();split.wp=w;P=[] if S else split(m,R)
    if not P and not S:return None
    st=store(history,user,_stm(m),session) if P else None;prev=None
    for p in P:
        for q in coord(p,st,R):s=slot(q,prev,history,st,R,cl,mx);S.append(s);prev=s if s.get('kind')!='other' else prev
    split.wp=False
    if all(s.get('kind')=='other' for s in S):return None
    k=0
    for s in S:
        s.get('kind')=='other' and s.update(zip(('text','guard'),_other(s,history,user,session,cl,mx)),src='adam_t');s.get('unk') and s.update(text="You haven't told me that yet." if len(S)==1 else f"You haven't told me {ind(s['q'])} yet.")
        if s.get('guess') and s.get('hedge'):s['text']=_hedge(s['guess'],s['gk'],k);k+=1
        M=s.pop('mat',None)
        if M and s.get('ok') and not s.get('hedge'):b=blend(cl,s.get('rq') or s['q'],M,mx);s.update(text=b[0],blend=b[1],mroute=M['route'],prov=s.get('prov') or M['prov'])
    txt=C.asc(' '.join(_end(s['text']) for s in S if s.get('text'))).strip()
    n={'hs':len(S),'hm':sum(s.get('kind')=='mem' for s in S),'hk':sum(s.get('kind')=='know' for s in S),'hc':sum(s.get('kind')=='calc' for s in S),'ho':sum(s.get('kind')=='other' for s in S),'hv':sum(bool(s.get('ok')) for s in S),'hh':sum(bool(s.get('hedge')) for s in S),'hgs':sum(bool(s.get('guess')) for s in S),'hsd':sum(bool(s.get('said')) for s in S),'hr':sum(int(s.get('retry') or 0) for s in S),'hx':sum(bool(s.get('conflict')) for s in S),'hxe':sum(s.get('kind')=='explain' for s in S),'hcs':sum(s.get('src')=='cs' for s in S),'htg':sum(bool(s.get('tgate')) for s in S),'hbl':sum(s.get('blend') is not None for s in S),'hbp':sum(bool(s.get('blend')) for s in S),'hgd':sum(bool(s.get('guard')) for s in S),'hms':round((time.perf_counter()-t0)*1000)}
    return {'text':txt,'n':n,'slots':S} if txt else None
