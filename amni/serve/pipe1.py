import os,re,json,time,contextvars,threading
from amni.serve import adam_t_client as C,adam_t_harness as H,user_memory as U,emergency as EMG,at_rest as AR
from amni.compute.gf17_overlay import Ov,nonce
on=lambda:os.environ.get('AMNI_PIPE1','0')=='1' and os.environ.get('AMNI_ADAM_T_HARNESS','0')=='1'
CUR=contextvars.ContextVar('pipe1_user',default=(None,None));_lk=threading.RLock();_cache={}
FAM={'store2':'fact','store1':'fact','know':'fact','calc':'fact','mem':'fact','cs':'fact','p1g':'fact','explain':'explain','chat':'chat','other':'chat'}
DEF={'fact':0.0,'store2':0.0,'explain':0.5,'chat':1.0}
TPL={131:'{L} is in {A}.',17:'{L} is in {A}.',30:'{L} is in {A}.',276:'{L} is located in {A}.',159:"{L}'s headquarters are in {A}.",27:'{L} is a citizen of {A}.',112:'{L} was founded by {A}.',175:'{L} was performed by {A}.',176:'{L} is made by {A}.',178:'{L} was developed by {A}.',162:'{L} was produced by {A}.',58:'{L} was written by {A}.',676:'The lyrics of {L} were written by {A}.',123:'{L} is published by {A}.',127:'{L} is owned by {A}.',403:'{L} flows into {A}.',69:'{L} studied at {A}.',108:'{L} worked for {A}.',54:'{L} played for {A}.',1303:'{L} played the {A}.',361:'{L} is part of {A}.',138:'{L} is named after {A}.',264:'{L} recorded for {A}.',463:'{L} is a member of {A}.',102:'{L} is a member of the {A}.',140:"{L}'s religion is {A}.",1412:'{L} spoke {A}.',103:"{L}'s native language is {A}.",2044:'The elevation of {L} is {A}.',1376:'{L} is the capital of {A}.',150:'{L} contains {A}.',47:'{L} borders {A}.',800:'{L} is known for {A}.',166:'{L} received the {A}.',1346:'{L} won the {A}.',39:'{L} held the position of {A}.',106:'{L} worked as a {A}.',413:'{L} played as a {A}.',641:'{L} competed in {A}.'}
_IMP=re.compile(r"(?i)^\W*(?:(?:please|can you|could you|would you)\s+)*(?:name|tell me|give me|identify)\s+(?!about\b|a\b|an\b|some\b|something\b|anything\b|more\b|one\b|two\b|three\b|five\b|ten\b|me\b)(?:the\s+)?(\S+(?:\s+\S+)+)$")
def imp(m:str)->str:g=_IMP.match((m or '').strip());x=re.sub(r"(?i)[\s,]*please\W*$",'',g[1]).rstrip(' .,!?') if g else '';return f"What is the {x}?" if g and len(x.split())>=2 and not U._isq(m.strip()) else m
TASK=('Task: Rewrite the draft as a natural reply and keep every fact.','Task: Answer naturally from these notes and keep every fact.')
def strength(r:str)->float:
    v=os.environ.get('AMNI_PIPE1_S','').strip();f=FAM.get(r,'fact');d=json.loads(v) if v[:1]=='{' else {}
    return 0.0 if r=='p1u' else float(d.get(r,d.get(f,DEF.get(r,DEF[f])))) if d else float(v) if v else float(DEF.get(r,DEF[f]))
def th(s:float)->dict:return {'cov':0.9-0.6*s,'tri':0.7*(1-s),'nw':0.0 if s<=0.25 else 0.4*(s-0.25),'cap':2+2*s}
def dec(s:float)->dict:T=0.0 if s<=0.25 else round(1.2*s-0.3,4);return {'temperature':T,'top_k':0 if T==0 else int(round(8+32*s)),'top_p':1.0 if T==0 else round(0.6+0.35*s,4),'seed':17}
lower=lambda s:s/2 if s/2>=0.1 else 0.0
def plines(M:dict)->list:s=M['s'];t=[] if os.environ.get('AMNI_PIPE1_NOTASK')=='1' else [TASK[s>=0.5]];return ['Draft: '+M['extract']]+([] if s<0.5 else M.get('sup',[]))+t
def newcap(t:str,src:str)->bool:sw={w.lower() for w in re.findall(r"[A-Za-z][\w'-]*",src)};return any(w.lower() not in sw for x in re.split(r"(?<=[.!?])\s+",t) for w in re.findall(r"\b[A-Z][\w'-]*",' '.join(x.split()[1:])) if w!='I')
def assemble(M:dict)->dict:
    if not M or M.get('gate')!='para':return M
    x=M['extract'];return {**M,'sup':list(M['lines']),'lines':['Draft: '+x],'must':[z for z in dict.fromkeys(list(M['must'])+re.findall(r"\d[\d,.]*\d|\d",x)) if z],'s':strength(M['route'])}
_RF=threading.local()
def refine(cl,q:str,M:dict,mx:int)->tuple:
    s=M['s'];n=0;_RF.on=True
    try:
        while s>0 and n<2:
            t,ok=H.blend(cl,q,{**M,'s':s},mx);n+=1
            if ok:return t,True,s,n
            s=lower(s)
        return M['extract'],(False if n else None),0.0,n
    finally:_RF.on=False
def chat(cl,message:str,c:dict,mx:int)->tuple:
    s=strength('chat');n=0;_RF.on=True
    try:
        while s>0 and n<2:
            t=C.asc(cl.gen(message,mx,C.STOPS,c['hist'],c['pre'],c['rc'],dec=dec(s))).strip();n+=1;g=H._coh(t,message) if t else '';src=' '.join(list(c['rc'])+list(c['pre'])+[message]+[str(x) for h in c['hist'] or [] for x in list(h)[:2]])
            if g and H.guard(message,g) and not (U._isq(message) and (newcap(g,src) or set(re.findall(r"\d+",g))-set(re.findall(r"\d+",src)))):return g,s,n
            s=lower(s)
        return EMG.neutral(message),0.0,n
    finally:_RF.on=False
_S2=lambda:__import__('amni.serve.store1',fromlist=['resolve'])
def qkey(q:str):c=sorted(set(H._ct(q)));return nonce('q|'+' '.join(c)) if c else None
def ckey(b:dict):return nonce(f"c|{b['key'][0]}|{b['key'][1]}") if b and b.get('store')=='wd' and b.get('key') else None
VU=AR.Vault('p1u',b'AMP1');VG=AR.Vault('p1g',b'AMPG');VQ=AR.Vault('p1q',b'AMPQ');JK=nonce('pipe1|journal')
_hx=lambda k:re.sub(r'[^0-9a-f]','',k)[:32]
def ukey()->str:u,_=CUR.get();return U.user_key(u)
def gdir()->str:return os.environ.get('AMNI_PIPE1_DIR') or os.path.join(U.root(),'pipe1')
def _ld(V,p:str,kp:str,name:str,n:int)->Ov:
    with _lk:
        st=os.stat(p).st_mtime_ns if os.path.exists(p) else None;c=_cache.get(p)
        if c and c[0]==st:return Ov.loads(c[1])
        if not st:return Ov(n)
        inner=V.read(p,kp,name)[0];_cache[p]=(st,inner);return Ov.loads(inner)
def _sv(V,p:str,kp:str,name:str,o:Ov,rotate:bool=False):
    with _lk:
        os.makedirs(os.path.dirname(p),mode=0o700,exist_ok=True);_cache.pop(p,None)
        V.wipe(p,kp) if not o.items() else V.write(p,kp,name,o.dumps(),rotate)
def _up(key:str)->tuple:return os.path.join(U.root(),_hx(key)+'.p1ov'),os.path.join(U.kdir(),_hx(key)+'.p1ov.key'),key
def pov(key:str=None)->Ov:return _ld(VU,*_up(key or ukey()),1) if U.enabled() else Ov(1)
def psave(o:Ov,key:str=None,rotate:bool=False):U.enabled() and _sv(VU,*_up(key or ukey()),o,rotate)
def _gp(f:str)->tuple:p=os.path.join(gdir(),f);kp,b=VG.fkey(p);return p,kp,b
def gov()->Ov:return _ld(VG,*_gp('global.p1ov'),16)
def gsave(o:Ov,rotate:bool=False):len(o.pend)>=int(os.environ.get('AMNI_PIPE1_COMPACT','32')) and o.compact();_sv(VG,*_gp('global.p1ov'),o,rotate)
def qload()->list:
    p=os.path.join(gdir(),'quarantine.p1q')
    with _lk:return [json.loads(x) for x in VQ.get(p).decode().split('\n') if x] if os.path.exists(p) else []
def qsave(L:list):
    p=os.path.join(gdir(),'quarantine.p1q')
    with _lk:os.makedirs(gdir(),mode=0o700,exist_ok=True);VQ.drop(p) if not L else VQ.put(p,'\n'.join(json.dumps(x,sort_keys=True) for x in L).encode(),True)
def _pay(b)->dict:
    try:return json.loads(b.decode()) if b else {}
    except ValueError:return {}
def _gb(R_,S,v:dict)->dict:
    if 'lit' in v:return {'store':'p1g','answer':v['lit'],'line':f"Correction (admin endorsed): {v['lit']}.",'prov':'admin-endorsed correction','sure':True,'score':9.0}
    e=R_.D.ix(int(v['qid']));V=R_.D.values(e,int(v['pid']));z=V[min(int(v['idx']),len(V)-1)] if V else None
    if z is None:return {}
    a,f=R_.render(z[0],z[1],z[2],int(v['pid']));pl=S.props().get(int(v['pid']),{}).get('label',f"P{v['pid']}")
    return {'store':'wd','e':e,'p':int(v['pid']),'answer':a,'forms':f,'line':f"{R_.D.label(e)}, {pl}: {a}.",'prov':f"Wikidata Q{v['qid']} P{v['pid']} (verified correction)",'sure':True,'score':9.0,'key':(int(v['qid']),int(v['pid']))}
def overlay(q:str,b=None)->dict:
    K=[k for k in (qkey(q),ckey(b)) if k];o=pov()
    for k in K:
        v=_pay(o.get(k))
        if v.get('v'):return {'store':'p1u','answer':v['v'],'old':v.get('old') or '','line':f"Correction from you: {v['v']}.",'prov':'your correction (unverified)','sure':True,'score':9.0}
    v=_pay(gov().get(K[0])) if K and K[0]==qkey(q) else {}
    if not v:return {}
    try:S=_S2();return _gb(S.resolver(),S,v)
    except Exception:return {'store':'p1g','answer':v['lit'],'line':f"Correction (admin endorsed): {v['lit']}.",'prov':'admin-endorsed correction','sure':True,'score':9.0} if 'lit' in v else {}
def lut(rq:str)->dict:
    S=_S2();r=S.resolve(rq);b=r.get('best');o=overlay(rq,b);return {**r,'best':o,'base':b} if o else r
def dtext(b:dict,rq:str)->str:
    a=C.asc(str(b['answer']))
    if b['store']=='p1u':o=C.asc(b.get('old') or '');return f"You told me it's {a}."+(f" My reference says {o}." if o and WF().nrm(o)!=WF().nrm(a) else '')
    if b['store']=='wd':S=_S2();R_=S.resolver();L=C.asc(R_.D.label(b['e']));return H._cap(TPL[b['p']].format(L=L,A=a)) if b['p'] in TPL else H._wsay({'answer':a,'label':L,'p':b['p'],'pl':S.props().get(b['p'],{}).get('label','')})
    if b['store']=='concepts':
        try:k=__import__('amni.serve.cognition',fromlist=['know']).know(rq)
        except Exception:k=None
        return C.asc(k['text']) if k and k.get('text') and H._has(a,k['text']) else f"The answer is {a}."
    return f"The answer is {a}." if b['store']=='calc' else f"It's {a}."
WF=lambda:__import__('amni.compute.wikifact',fromlist=['nrm'])
def support(rq:str,b:dict)->list:
    P,_=C._know(rq,2,int(C._f('AMNI_ADAM_T_HARNESS_PASSAGE_BYTES',420)),None,False);F=[str(b['answer']),*(b.get('forms') or [])[:6]]
    return [x for x in P if any(H._has(f,x) for f in F)][:1]
def ovslot(s:dict,q:str)->dict:
    try:b=overlay(q)
    except Exception:return {}
    if not b:return {}
    a=C.asc(str(b['answer']));tx=dtext(b,q);k=b['store'] if b['store'] in ('p1u','p1g') else 'p1g'
    return {**s,'kind':'know','rq':q,'src':k,'span':a,'conf':9.0,'ok':True,'text':tx,'mat':H.mat(k,[b['line']],tx,[b.get('prov','')],9.0,[a])}
_tk=lambda x:re.findall(r"[a-z0-9]+",C.asc(x).lower().replace("'",''))
_nz=lambda x:C.asc(x).replace("'",'')
def _cut(rx,x:str)->tuple:
    g=rx.match(_nz(x))
    if not g:return None,x
    n=len(_tk(g[0]));W=list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*",C.asc(x)));e=W[n-1].end() if n and len(W)>=n else 0
    return g,re.sub(r"^[\s,.;:!-]+",'',C.asc(x)[e:])
def _rx(L:list)->str:return '|'.join(r'\W+'.join(map(re.escape,_tk(x))) for x in sorted(L,key=lambda z:-len(z)))
SC=['no','nope','nah','wrong','incorrect','not quite','not really','not true','not correct','thats wrong','that is wrong','thats not right','that is not right','thats incorrect','that is incorrect','thats not true','thats not correct','that was wrong','wrong answer','you are wrong','youre wrong','you got that wrong','you got it wrong','youve got that wrong','youve got it wrong','you have that wrong','you have it wrong','you are mistaken','youre mistaken','correction','not exactly','close but']
WC=['actually','well actually','i think youll find','im pretty sure','i am pretty sure','pretty sure','i believe','i think','hmm','um','uh','oh','wait','no wait','sorry']
UN=['undo','undo that','undo that correction','undo my correction','undo the correction','scratch that','scratch my correction','scratch what i said','forget my correction','forget that correction','forget the correction','cancel that correction','cancel my correction','revert that','revert my correction','revert the correction','take that back','i take that back','never mind you were right','nevermind you were right','you were right','you were right after all','ignore my correction','ignore that correction','disregard my correction','disregard that correction','withdraw my correction','i was wrong','my mistake you were right']
VL=['it is actually','it was actually','its actually','it is','its','it was','it should be','it should have been','the answer is','the correct answer is','the right answer is','that would be','thats','that is','they are','theyre','he is','hes','she is','shes','the real answer is']
NC=['no worries','no problem','no problemo','no thanks','no thank you','no way','no idea','no rush','no biggie','no big deal','nope not yet','no not yet','no i dont','no i didnt','no im not','no im good','no im fine','actually yes','no comment']
SAFE=set('dose dosage doses overdose overdosed poison poisonous poisoning toxic toxin lethal deadly fatal emergency 911 999 112 suicide suicidal medication medications medicine medicines pill pills tablet tablets drug drugs allergic allergy allergies anaphylaxis bleach ammonia carbon monoxide choking cpr bleeding stroke seizure antidote venom venomous overdosing insulin dangerous unsafe hazardous explosive flammable'.split())|{'safe'}
SAFEP={2175,2176,780,1050,2293,4044,636,2240,2789,3489,1995,923,1060,2710,2717,2718,3780,4954,1193,231,2329}
_CUE=re.compile(r"(?i)^\W*(?:(?:"+_rx(SC+WC)+r")\b[\s,.;:!-]*|(?:its|it is|it was|thats|that is)\s+not\s+[^,;]+[,;]\s*(?:but\s+)?)+")
_NCR=re.compile(r"(?i)^\W*(?:"+_rx(NC)+r")\b")
_PRO=re.compile(r"(?i)\b(?:i|me|my|mine|you|your|yours|we|us|our|i'm|im|i've|ive|you're|youre)\b")
_SCR=re.compile(r"(?i)\b(?:"+_rx(SC)+r")\b")
_UNR=re.compile(r"(?i)^\W*(?:(?:ok(?:ay)?|please|actually|sorry|oh|hey|adam)\W+)*(?:(?:"+_rx(UN)+r")\b\W*){1,3}(?:please|thanks|thank you|ok(?:ay)?)?\W*$")
_VLR=re.compile(r"(?i)^\W*(?:"+_rx(VL)+r")\b\W*")
_NOT=re.compile(r"(?i)(?:,\s*|\s+)(?:not|rather than|instead of)\s+[^,.;!?]+[.!?]?\s*$|^\W*not\s+[^,;]+,\s*(?:but\s+)?(?:it'?s\s+)?")
_SPL=re.compile(r"(?i)\bspel(?:led|t)\b\s*|[,;]?\s+(?:with|w/)\s+(?:an?|two|double|one|capital)\b.*$|\s*\([^)]*\)")
_COP=re.compile(r"(?i)\s+(?:is|was|are|were)\s+|'s\s+")
_FPS=re.compile(r"(?i)^\W*(?:my|our|i|i'm|im|we|we're)\b")
_PERQ=re.compile(r"(?i)\b(?:my|mine|our|ours|i'm|am i|was i|have i|did i (?:tell|say|mention))\b|\bwhere (?:do|did) (?:i|we) (?:live|work|grow up)\b|\bhow old am i\b")
_isundo=lambda m:bool(_UNR.match(_nz(m)))
def _uh(m:str)->int:return nonce('u|'+' '.join(_tk(m)))
def _xval(r:str,qp:str)->str:
    r=_cut(_VLR,_NOT.sub('',_cut(_VLR,r.strip())[1]))[1].strip(' .!,;:"\'-');qc=set(H._ct(qp))
    if len(r.split())>=3 and _COP.search(r):a,b=_COP.split(r,1)[0],_COP.split(r,1)[1];r=min((b,a),key=lambda z:len(set(H._ct(z))&qc)).strip()
    r=re.sub(r"(?i)\s*,?\s*(?:i think|i believe|i'm pretty sure|for sure|actually|by the way)\W*$",'',_cut(_VLR,r)[1]).strip(' .!,;:"\'-')
    return r if 1<=len(r.split())<=8 else ''
CUEX=SC+WC+['fix','small fix','mixed up','mixed it up','mix up','mix those','misspoke','typo','i mean','i meant','last i checked','i read','i heard','i always heard','not what you said','not whatever you said','thats off','its off','more like','spells','spelling','spelled','spelt','renamed','instead','in fact','should be','my mistake','thinking of','certain','im sure','call him','call her','we call','goes by','go by','its','it is','it was','his name is','her name is','its called','its named','got promoted','i misspoke']
VC=['it is','its','it was','it is actually','is','was','are','were','i mean','i meant','called','named','name is','spells it','spelling','spelled','renamed him','renamed her','renamed it','call him','call her','call it','go by','goes by','more like','reaches','count was','uses the','uses','use','should be','actually','been']
FILL=set('hi hey ha hmm hm um uh oh ok okay well so wait ugh actually no nope yes yeah yep sorry small typo wrong not that thats its it i im ive id we he she they his her their the a an you your youre youve my our last pretty sure please cool nah lol huh whoa spelling fix let lets nevermind never this these those there theres everyone everybody ill hes shes theyre whats'.split())
UNV=['undo','un correct','uncorrect','revert','retract','withdraw','disregard','ignore','scrap','scratch','cancel','drop','forget','never mind','nevermind','take back','go back','go with the first','go with that','go with the original','stands','was correct','was right','had it right','you were right','you were fine','switched it back','back to the old','first spelling','first number','first answer','your answer','original']
_CUEX=re.compile(r"(?i)\b(?:"+_rx(CUEX)+r")\b");_VCX=re.compile(r"(?i)\b(?:"+_rx(VC)+r"|\w+'s|\w+'re)\b");_UNX=re.compile(r"(?i)\b(?:"+_rx(UNV)+r")\b")
_CAPS=re.compile(r"[A-Z][\w'.-]*(?:\s+(?:(?:of|the|de|del|da|di|du|la|le|von|van|der|den|and|y)\s+)?[A-Z][\w'.-]*)*")
_NUMX=re.compile(r"(?i)\b(?:(?:about|around|roughly|approximately|nearly|almost|over|under|exactly|more than|less than|just over|just under)\s+)?(?:the\s+)?\d[\d,]*(?:\.\d+)?(?::\d\d)?(?:\s*(?:am|pm)\b)?(?:\s+(?!(?:and|or|but|not|i|you|he|she|it|they|we|in|on|at|to|for|of|the|a|an|is|was|are|these|those|this|that|ago|with|by|from|as|so|because|which|now|today|days|then|last|every|each|not|no)\b)[a-z]{2,}){0,2}")
_MON=set('january february march april may june july august september october november december'.split())
undoish=lambda m:bool(_UNX.search(_nz(m))) and len(str(m).split())<=20 and not U._isq(str(m).strip())
def _spans(m:str,ex:set,exd:set)->list:
    S=[]
    for g in _CAPS.finditer(m):
        w=[re.sub(r"(?:'s|')$","",x).strip('.') for x in g[0].split()];o=g.start()
        while w and (not _tk(w[0]) or _tk(w[0])[0] in FILL):o+=len(g[0][o-g.start():].split(' ',1)[0])+1;w=w[1:]
        while w and w[-1].lower() in ('of','the','de','and','y'):w=w[:-1]
        s=' '.join(w);tk=set(_tk(s))
        s and tk and not tk<=ex|FILL and S.append((o,s,'cap'))
    for g in _NUMX.finditer(m):s=g[0].strip(' .,');d=set(re.findall(r"\d+",s));s and d and not d<=exd and S.append((g.start(),s,'num'))
    return sorted(S)
def xval2(m:str,qp:str,ap:str,e:str)->str:
    m=C.asc(m);ex=set(_tk(qp));exd=set(re.findall(r"\d+",qp));S=_spans(m,ex,exd);V=[g.end() for g in _VCX.finditer(m)];ng=[g.end() for g in re.finditer(r"(?i)\b(?:not|instead of|rather than)\s+",m)]
    ok=lambda z:z[2]=='num' if e=='num' else (z[2]=='num' or z[1].lower() in _MON) if e=='time' else z[2]=='cap' if e in ('name','place') else True
    S=sorted([z for z in S if ok(z) and z[0] not in ng],key=lambda z:(not any(0<=z[0]-v<=24 for v in V),z[2]!='cap' if e not in ('num','time') else z[2]!='num',z[0]))
    if S:return '' if H._has(S[0][1],ap) or set(_tk(S[0][1]))<=set(_tk(ap)) else S[0][1]
    for c in re.split(r"[,;:.!?]|\s-\s|\s+but\s+",m):
        g=list(_VCX.finditer(c));r=re.sub(r"(?i)^(?:a|an|the|some)\s+",'',c[g[-1].end():].strip()) if g else ''
        if r and 1<=len(r.split())<=4 and not _PRO.search(r) and not set(_tk(r))<=ex|FILL:return r
    return ''
def detect(m:str,history=None,J=None)->dict:
    m=C.asc(str(m or '')).strip();pv=[tuple(h)[:2] for h in (history or []) if h and len(h)>=2];qp,ap=(str(pv[-1][0] or ''),str(pv[-1][1] or '')) if pv else ('','')
    if not m:return {}
    if _isundo(m) or J and undoish(m):return {'kind':'undo'}
    if U._isq(m) or _NCR.match(_nz(m)) or undoish(m):return {}
    r=_cut(_CUE,m)[1];sp=bool(re.search(r"(?i)\bspel(?:led|t)\b",m)) and bool(_FPS.search(r));st=_end(_SPL.sub('',_NOT.sub('',r)).strip(' ,'))
    if not qp or not U._isq(qp.strip()):return {'kind':'personal','stmt':st,'qp':'','x':''} if sp else {}
    if not (_CUEX.search(_nz(m)) or _CUE.match(_nz(m))):return {}
    pq=bool(_PERQ.search(qp));x=xval2(m,qp,ap,H.expect(qp))
    return {'kind':'personal' if pq else 'world','x':x,'qp':qp,'ap':ap,'stmt':st if pq and _FPS.search(r) else ''} if x and not H._has(x,ap) and (pq or len(x.split())<=6 and not _PRO.search(x)) else {}
_end=lambda s:H._end(H._cap(s.strip())) if s.strip() else ''
def safety(*xs)->bool:return any(w in SAFE for x in xs for w in _tk(x)) or any((EMG.detect(x) or {}).get('tier')=='emergency' for x in xs if x)
def _jl(o:Ov)->list:return _pay(o.get(JK)).get('j',[])
def _js(o:Ov,J:list):o.forget(JK) if not J else None;J and o.learn(JK,json.dumps({'j':J[-32:]}).encode())
def _qrec(rec:dict)->dict:L=qload();L.append(rec);qsave(L);return rec
def verify(r:dict,x:str,base)->tuple:
    S=_S2();R_=S.resolver();w=r.get('wd') or {};E=list(w.get('ents') or [])[:5];P=list(dict.fromkeys(([base['p']] if base and base.get('p') else [])+list(w.get('props') or [])[:6]));nx=WF().nrm(x)
    if not nx:return 'unverified',None
    sure=bool(base) and base.get('store')=='wd' and base.get('sure',False);mt=lambda a,f:any(nx==WF().nrm(y) or f' {nx} ' in f' {WF().nrm(y)} ' for y in [a,*f] if y)
    def vals(e,p):
        try:return R_.D.values(e,p) or []
        except Exception:return []
    def ended(e,p,i,z):
        try:return len(z)>4 and bool(z[4]) and any(q in (582,) for q,_,_ in R_.D.quals(e,p,i,z[4]))
        except Exception:return True
    if base and mt(str(base.get('answer','')),list(base.get('forms') or [])):return 'same',None
    if sure:return ('unverified' if any(mt(*R_.render(z[0],z[1],z[2],base['p'])) for z in vals(base['e'],base['p'])) else 'rejected'),None
    for e in ([base['e']] if base and base.get('e') is not None else [])+E:
        for p in P:
            for i,z in enumerate(vals(e,p)):
                if mt(*R_.render(z[0],z[1],z[2],p)) and not ended(e,p,i,z):return 'verified',{'qid':int(R_.D.qid(e)),'pid':int(p),'idx':i}
    return 'unverified',None
ACK={'personal':"Got it, I'll remember that for you.",'world':"Got it, I'll remember that for you, and I've noted it for review.",'rejected':"Got it, I'll remember that you said so, but my reference says {old}, so I've noted it for review.",'verified':"Thanks, I checked that against my reference and updated it.",'safety':"I can't change safety information from a correction, but I've noted it for review.",'undo':"Okay, I've undone that correction.",'same':"That matches what I have."}
def _out(t:str,sc:str,ms:float)->dict:return {'text':t,'n':{'hs':1,'p1c':1,'hms':round(ms)},'slots':[{'kind':'corr','src':'pipe1','ok':True,'scope':sc,'text':t,'mroute':'corr','blend':None}]}
def wipe_all(user=None)->None:
    k=U.user_key(user)
    with _lk:p,kp,_=_up(k);_cache.pop(p,None);VU.wipe(p,kp)
def correct(m:str,history=None,user=None,session=None)->dict:
    t0=time.perf_counter();f=U._fg(str(m or '').strip())
    if f and f[1]:wipe_all(user);return {}
    k=U.user_key(user);o=pov(k);J=_jl(o);ms=lambda:(time.perf_counter()-t0)*1000;d=detect(m,history,J if J and J[-1].get('h') in {_uh(str((h or [''])[0] or '')) for h in list(history or [])[-6:]} else None)
    if not d or any((EMG.detect(z) or {}).get('tier')=='emergency' for z in (m,d.get('qp') or '') if z) or d.get('kind')!='undo' and H.emtpl(d.get('ap') or ''):return {}
    if d['kind']=='undo':
        if not J:return {}
        j=J.pop();[o.forget(z) for z in j.get('keys',[])];_js(o,J);o.compact();psave(o,k,True)
        j['k']=='mem' and U.forget(k,j['ent'],j['rel'],j['clk']);j.get('rid') and qsave([x for x in qload() if x['id']!=j['rid'] or x['tier'] in ('verified','endorsed')])
        return _out(ACK['undo'],'undo',ms())
    mf=[];pk=d['kind']=='personal'
    if pk:
        if d.get('x') and d.get('qp'):p=U._P();p.turn(U._enc(d['qp']));mf=[[p.qent,p.qrel,U._enc(d['x'])[:24],1,p.names.get(p.qent,'')]] if p.qent and p.qrel else []
        mf=mf or (U.parse(d['stmt'])[1] if d.get('stmt') else [])
        if not U.enabled() or not mf and not d.get('x'):return {}
    if pk and mf:
        U.remember(k,mf,session);fs=U.load(k);e,r_=mf[-1][0],mf[-1][1];c=max([z.clk for z in fs if z.ent==e and z.rel==r_],default=0);nf=next((z for z in fs if z.clk==c),None)
        J.append({'k':'mem','ent':e,'rel':r_,'clk':c,'h':_uh(m),'stmt':(U.render(nf) if nf else '') or d.get('stmt') or ''});_js(o,J);o.compact();psave(o,k);return _out(ACK['personal'],'personal',ms())
    qp,x=d['qp'],d['x']
    try:S=_S2();r=S.resolve(qp);b=r.get('best') if r.get('best') and r['best'].get('store')=='wd' else None
    except Exception:r,b={},None
    old=C.asc(str(b['answer'])) if b else '';sf=safety(qp,m) or bool(b and int(b.get('p') or 0) in SAFEP)
    rid=format(nonce(f"{k}|{qp}|{x}|{time.time_ns()}"),'016x')[:12];rec={'id':rid,'u':_hx(k)[:16],'keys':[z for z in (qkey(qp),ckey(b)) if z],'coord':list(b['key']) if b else None,'v':x,'old':old,'tier':'unverified','ts':int(time.time()),'safety':sf}
    if sf:_qrec(rec);return _out(ACK['safety'],'safety',ms())
    tv,cd=verify(r,x,b) if r and not pk else ('unverified',None)
    if tv=='same':return _out(ACK['same'],'same',ms())
    rec['tier']=tv;pk or _qrec(rec);K=rec['keys'] if tv!='verified' else []
    if tv=='verified':g=gov();g.learn(qkey(qp),json.dumps(cd,sort_keys=True).encode());gsave(g)
    [o.learn(z,json.dumps({'v':x,'old':old}).encode()) for z in K];J.append({'k':'world','keys':K,'h':_uh(m),'rid':'' if pk else rid});_js(o,J);o.compact();psave(o,k)
    if pk:return _out(ACK['personal'],'personal',ms())
    return _out(ACK['rejected'].format(old=old) if tv=='rejected' else ACK['verified'] if tv=='verified' else ACK['world'],tv if tv!='unverified' else 'world',ms())
def hist(history,user=None)->list:
    J=_jl(pov(U.user_key(user)));act={j['h']:j.get('stmt','') for j in J if j['k']=='mem'};dk={(j['ent'],j['rel']) for j in J if j['k']=='mem'};o=[]
    for h in history or []:
        u=str((h or [''])[0] or '');d=detect(u,o) if u else {}
        if d and act.get(_uh(u)):o.append([act[_uh(u)],*list(h)[1:]])
        elif not d and not undoish(u) and not (dk and u and not U._isq(u.strip()) and any((f[0],f[1]) in dk for f in U.parse(u)[1])):o.append(h)
    return o
def promote(rid:str,by:str='admin')->dict:
    L=qload();x=next((z for z in L if z['id']==rid),None)
    if not x or x.get('safety') or x['tier'] in ('verified','endorsed'):return {'ok':False,'why':'missing' if not x else 'safety' if x.get('safety') else 'already'}
    g=gov();k=x['keys'][0] if x['keys'] else None
    if not k:return {'ok':False,'why':'no key'}
    g.learn(k,json.dumps({'lit':x['v']}).encode());gsave(g);x.update(tier='endorsed',by=by,pts=int(time.time()));qsave(L);return {'ok':True,'id':rid}
def reject(rid:str)->dict:L=qload();[z.update(tier='rejected') for z in L if z['id']==rid];qsave(L);return {'ok':any(z['id']==rid for z in L)}
def gforget(key:int)->bool:g=gov();r=g.forget(key,True);gsave(g,True);return r
def gcompact()->list:g=gov();T=g.compact();_sv(VG,*_gp('global.p1ov'),g);return T
