import os,re,json,time,shlex,base64,queue,ctypes,logging,threading,subprocess,unicodedata
from pathlib import Path
from typing import Iterable,List,Optional,Sequence,Tuple
B64=lambda b:base64.b64encode(b).decode();U64=lambda s:base64.b64decode(s)
LOG=logging.getLogger('amni.serve.adam_t')
ROOT=Path(__file__).resolve().parents[2]
STOPS=('\n\n','\nUser:','\nAdam:')
ACK='Got it, thanks for telling me.'
try:_PRCTL=ctypes.CDLL(None,use_errno=True).prctl if os.name=='posix' else None
except Exception:_PRCTL=None
_pdeath=lambda pp:lambda:(_PRCTL and _PRCTL(1,9),os.getppid()!=pp and os._exit(1))
_SQ=queue.Queue();_ST=[None];_SL=threading.Lock()
def _spawner():
    while True:
        a,k,r=_SQ.get()
        try:r.put((1,subprocess.Popen(*a,**k)))
        except BaseException as e:r.put((0,e))
def spawn(*a,**k)->subprocess.Popen:
    with _SL:_ST[0] and _ST[0].is_alive() or (_ST.__setitem__(0,threading.Thread(target=_spawner,daemon=True,name='adam-t-spawn')),_ST[0].start())
    r=queue.Queue(1);_SQ.put((a,{**k,**({'preexec_fn':_pdeath(os.getpid())} if os.name=='posix' else {})},r));ok,v=r.get()
    if not ok:raise v
    return v
class ProtoError(RuntimeError):pass
class Proto:
    def __init__(s,cmd,timeout:Optional[float]=None,env=None,stderr=None,load_timeout:Optional[float]=None):
        s.cmd=cmd;s.timeout=timeout;s.lock=threading.RLock();s.q=queue.Queue()
        s.p=spawn(shlex.split(cmd) if isinstance(cmd,str) else list(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,env=env,start_new_session=True)
        threading.Thread(target=s._pump,args=(s.p,s.q),daemon=True).start()
        try:s.info={'cmd':cmd if isinstance(cmd,str) else shlex.join(cmd),**s.rpc(timeout=load_timeout or timeout,op='hello')}
        except BaseException:s.kill();raise
    @staticmethod
    def _pump(p,q):
        try:[q.put(l) for l in iter(p.stdout.readline,'')]
        except Exception:pass
        q.put(None)
    def alive(s)->bool:return s.p.poll() is None
    def kill(s)->None:
        try:s.p.kill();s.p.wait(5)
        except Exception:pass
    def rpc(s,timeout:Optional[float]=None,**k)->dict:
        with s.lock:
            t=timeout or s.timeout;end=time.monotonic()+t if t else None
            if not s.alive():raise ProtoError(f'server exited rc={s.p.poll()}')
            try:s.p.stdin.write(json.dumps(k)+'\n');s.p.stdin.flush()
            except (OSError,ValueError) as e:s.kill();raise ProtoError(f'write failed: {e}')
            while True:
                try:l=s.q.get(timeout=max(0.0,end-time.monotonic()) if end else None)
                except queue.Empty:s.kill();raise ProtoError(f"timeout after {t}s on {k.get('op')}")
                if l is None:s.kill();raise ProtoError(f'server exited rc={s.p.poll()}')
                try:r=json.loads(l)
                except ValueError:continue
                if not isinstance(r,dict):continue
                if 'error' in r:raise ProtoError(str(r['error']))
                return r
    def text(s,prompt:str,max_bytes:int,stops:Iterable[str]=(),timeout:Optional[float]=None,dec:Optional[dict]=None)->str:
        return U64(s.rpc(timeout=timeout,op='gen',prompt=B64(prompt.encode('utf-8')),max=int(max_bytes),stop=[B64(x.encode('utf-8')) for x in stops],**(dec or {}))['text']).decode('utf-8','replace')
    def lp(s,P:Sequence[Tuple[bytes,bytes]],chunk:int=64)->List[float]:
        return [float(x) for i in range(0,len(P),chunk) for x in s.rpc(op='lp',items=[[B64(a),B64(b)] for a,b in P[i:i+chunk]])['lp']]
    def close(s)->None:
        try:s.alive() and (s.rpc(timeout=10,op='bye'),s.p.wait(30))
        except Exception:s.kill()
_nl=lambda x:re.sub(r'\n\s*\n+','\n',str(x or '').strip())
_SYM=str.maketrans({'\u00a3':'GBP ','\u20ac':'EUR ','\u00a5':'JPY ','\u00b0':' degrees','\u2044':'/','\u2212':'-','\u00d7':'x','\u00f7':'/','\u221a':'sqrt','\u00bd':' 1/2','\u00bc':' 1/4','\u00be':' 3/4','\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2013':'-','\u2014':'-','\u2026':'...'})
asc=lambda t:re.sub(r'\s+',' ',re.sub(r'\[(?:\d+|[a-z]|citation needed|note \d+)\]','',unicodedata.normalize('NFKD',str(t or '').translate(_SYM)).encode('ascii','ignore').decode())).strip()
clip=lambda t,n:t if len(t.encode('utf-8'))<=n else (lambda c:c.rsplit(' ',1)[0] if ' ' in c else c)(t.encode('utf-8')[:max(0,n)].decode('utf-8','ignore'))
def _fit(T:list,n:int,budget:int)->Tuple[list,int]:
    k=len(T)
    while k and budget>0 and n+len(T[k-1].encode('utf-8'))<=budget:n+=len(T[k-1].encode('utf-8'));k-=1
    return (T[k:] if budget>0 else T),n
def template(message:str,history=None,budget:int=0,pre:Sequence[str]=(),rc:Sequence[str]=(),sep:str=' ')->str:
    R=[x for x in rc if x];cur=lambda R:f"User: {sep.join(R+[_nl(message)])}\nAdam:"
    while R and budget>0 and len(cur(R).encode('utf-8'))>budget:R.pop()
    T=[f'User: {_nl(u)}\nAdam: {_nl(a)}\n\n' for u,a in [tuple(h)[:2] for h in (history or []) if h and len(h)>=2]];P,n=_fit(list(pre),len(cur(R).encode('utf-8')),budget);H,_=_fit(T,n,budget)
    return ''.join(P)+''.join(H)+cur(R)
def _f(k:str,d:float)->float:
    try:return float(os.environ.get(k) or d)
    except ValueError:return d
def _mem(message:str,history,user,session)->dict:
    from amni.serve.user_memory import turn
    m=turn(message,history,user,session);return {'reply':m['reply'],'hist':m['hist'],'route':bool(m['route']),'pre':[t.replace('\nAdam: Got it.\n\n',f'\nAdam: {ACK}\n\n') for t in re.findall(r'User: [^\n]*\nAdam: [^\n]*\n\n',m['pre'] or '')]}
def _tor(message:str,history,user,session,c:dict)->dict:
    try:
        T=__import__('amni.serve.user_toroid',fromlist=['turn']);t=T.turn(message,history,user,session) if T.enabled() else {'lines':[],'forgot':None}
        t['forgot'] and t['forgot'][1] and 'forgotten' not in (c['reply'] or '') and c.update(reply="Okay, I've forgotten that.");t.get('direct') and not c['reply'] and c.update(reply=t['direct']);return t
    except Exception as e:c['n']['tor_err']=type(e).__name__;return {'lines':[],'forgot':None}
def _math(message:str,full:bool=False):
    A=__import__('amni.math.answer',fromlist=['solve_math']);r=A.solve_math(message) if os.environ.get('AMNI_MATH','1')=='1' else None;r=r if r and float(r['conf'])>=_f('AMNI_MATH_MIN_CONF',0.6) else None;t=asc(r['reply']) if r else '';t=t+('' if not t or t[-1] in '.!?' else '.')
    return (t,A.fmt(r['value']) if r else '',r) if full else t
_MO='January|February|March|April|May|June|July|August|September|October|November|December'
_PRON=re.compile(r"(?i)\b(?:he|him|his|she|her|hers|it|its|they|them|their|there|that(?=\s*(?:in|on|at|from|of|by|for|located|\?|$)))\b")
_ELL=re.compile(r"(?i)^\W*(?:(?:and|but|so|ok|okay)[\s,]+)?(?:what|how)\s+about\s+(.+?)[\s?.!]*$|^\W*and\s+(?!(?:what|who|whom|when|where|which|why|how|is|are|was|were|do|does|did|can|could|would|will|should|you|i|we)\b)(.+?)[\s?.!]*$")
_QW=set('who whom whose what when where which why how name is are was were do does did in on at the a an and of'.split())
_TYPEW=set('year years country countries city cities capital color colour colors name number type kind people person place language animal element planet state continent ocean river mountain company part word called used main first largest biggest smallest most famous world many much long old one ones thing things day days home'.split())
_NOTSUP=set('forest interest contest harvest honest modest request protest suggest invest arrest guest chest crest nest pest rest test west'.split())
_BADT=re.compile(r"(?i)^(?:lists? of|index of|outline of|timeline of|glossary of|bibliography of|history of the name)\b|\(disambiguation\)|\b(?:by (?:country|year|city|state|region)|videography|discography|filmography|bibliography|in popular culture|awards and nominations)\b")
_NAV=re.compile(r"(?i)\bmay (?:also )?refer to\b|\bis a list of\b")
_HOLE=re.compile(r"(?i)\b(?:approximately|about|around|roughly|nearly|almost|over|some|measures|measuring|elevation|height|area|length|population|covers|covering|of|is|was|at|to|from|had)\s{2,}(?=[a-z(,.;])|\b(?:the|at|in|of)\s+,|(?<!\s)\s{2,}(?=(?:long|high|tall|wide|deep|away|above|below|across|in (?:length|diameter|area|size))\b)")
_NUMW=re.compile(r"(?i)\b(?:\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|dozen)\b")
_PL={36:'capital',50:'author',86:'composer',57:'director',170:'creator',61:'discoverer or inventor',112:'founder',175:'performer',569:'born',570:'died',19:'place of birth',20:'place of death',17:'country',30:'continent',131:'located in',276:'location',159:'headquarters',38:'currency',37:'official language',1082:'population',571:'founded',577:'published',580:'start',582:'end',585:'date',6:'head of government',35:'head of state',26:'spouse',22:'father',25:'mother',40:'child',2048:'height',2043:'length',1086:'atomic number',246:'symbol',138:'named after',84:'architect',176:'manufacturer',178:'developer',162:'producer',676:'lyricist',58:'screenwriter',127:'owner',123:'publisher',885:'source',403:'mouth',610:'highest point',27:'citizenship',69:'educated at',102:'party',140:'religion',840:'setting',915:'filming location',509:'cause of death',1449:'nickname',1477:'birth name'}
def _fix(x:str)->str:
    x=re.sub(r"\b(\d{1,2} (?:%s) \d{3,4})(?=\d{1,2} (?:%s) \d{3,4})"%(_MO,_MO),r"\1 - ",x);x=re.sub(r"\b((?:%s) \d{1,2}, \d{3,4})(?=(?:%s) \d{1,2}, \d{3,4})"%(_MO,_MO),r"\1 - ",x);x=re.sub(r"\((1\d{3}|20\d\d)(1\d{3}|20\d\d)\)",lambda m:f"({m[1]}-{m[2]})" if m[2]>=m[1] else m[0],x)
    return re.sub(r"\(\s*[,;]?\s*\)","",_HOLE.sub(lambda m:m[0]+'\x01',x))
def _trim(S:list,sc:list,cap:int)->str:
    i=max(range(len(S)),key=lambda j:(sc[j],-j));k=set();n=-1
    for j in sorted(range(len(S)),key=lambda j:(abs(j-i),j<i)):n+len(S[j])+1<=cap and (k.add(j),n:=n+len(S[j])+1)
    return ' '.join(S[j] for j in sorted(k)) or clip(S[i],cap)
def _ents(R,q:str)->list:
    return [v for v in (' '.join(w for w in re.split(r'\s+',v) if w.lower() not in _QW or v.split()[-1]!=w) for v,_,_ in R.ents(q)) if v and not set(v.lower().split())<=_QW]
def _human(name:str)->Optional[bool]:
    WF=__import__('amni.compute.wikifact',fromlist=['store']);S=WF.store();E=sorted({e for e,_ in S.ents(WF.nrm(name))},key=lambda e:-int(S.e_sl[e]))[:1] if S is not None else []
    return (S.ix(5) in S.inst(E[0])) if E else None
def _aent(a:str)->list:
    a=re.split(r'(?<=[a-z0-9)])[.!?;]\s|,\s',re.sub(r"(?i)\s+(?:wrote|painted|composed|directed|invented|discovered|founded|created|built|made|did|is|was)\s+(?:it|them|this|that)\b.*$",'',str(a or '').strip()))[0].strip(' .!?"')
    a=re.sub(r"(?i)^(?:it(?:'s| is| was)|that(?:'s| is| was)|in|the)\s+",'',a);return [a] if a and len(a.split())<=6 and not re.fullmatch(r'[\d\s.,-]+',a) and a.lower() not in _QW else []
def _resolve(R,message:str,history)->str:
    h=[tuple(x)[:2] for x in (history or []) if x and len(x)>=2]
    if not h:return message
    u,a=(str(z or '') for z in h[-1]);pers=bool(R.P1.search(u) or R.PW.search(u));e=_ELL.match(message);x=e and (e[1] or e[2]).strip()
    if x and len(x.split())<=6:
        if pers:return ''
        x=re.sub(r"(?i)^(?:of|for|in|with|on|at)\s+",'',x);yr=re.search(r"\b(?:1\d{3}|20\d\d)\b",x);sp=next((w for w in re.findall(r"[a-z]+",x.lower()) if len(w)>5 and w.endswith('est') and w not in _NOTSUP),None) if not _ents(R,x) else None;m=(yr or sp) and re.search(r"(?i)\b(?:first|last|latest|second|third|1\d{3}|20\d\d|\w{3,}est)\b",u)
        if m:return u[:m.start()]+(yr[0] if yr else sp)+u[m.end():]
        E0=_ents(R,u);I=R.ix();c=[w for w in re.findall(r"[\w'-]+",u) if w.lower() not in _QW|R.STOP|R.QSTOP|_TYPEW and len(w)>2]
        return re.sub(r"(?i)(?:\bthe\s+)?"+re.escape(E0[-1]),x,u,1) if E0 else re.sub(r"(?i)\b"+re.escape(max(c,key=lambda w:I.widf(R.toks(w)[0]) if R.toks(w) else 0))+r"\b",re.sub(r"(?i)^(?:a|an|the)\s+",'',x),u,1) if c else ''
    if not _PRON.search(message):return '' if pers and (set(R.toks(message))-R.QSTOP-_TYPEW)&set(R.toks(u)) else message
    if pers:return ''
    C=list(dict.fromkeys(_aent(a)+_ents(R,u)));hu={z:_human(z) for z in C};wh=bool(re.match(r"(?i)\W*who\b",u))
    rep={'p':next((z for z in C if hu[z]),(_aent(a) or [None])[0] if wh else None),'n':next((z for z in C if hu[z] is False or (hu[z] is None and not (wh and z in _aent(a)))),None),'t':(C or [None])[0]}
    k=lambda p:'p' if p in ('he','him','his','she','her','hers') else 't' if p in ('they','them','their') else 'n'
    if any(rep[k(m[0].lower())] is None for m in _PRON.finditer(message)):return ''
    ps=lambda m:m[0].lower() in ('his','its','their','hers') or (m[0].lower()=='her' and bool(re.match(r"\s+(?!(?:in|on|at|to|for|from|with|and|or|about|as|by|up|out|back|is|was|a|an|the)\b)[a-z]",message[m.end():],re.I)))
    return _PRON.sub(lambda m:(f"{rep[k(m[0].lower())]}'s" if ps(m) else f"in {rep['n']}" if m[0].lower()=='there' else rep[k(m[0].lower())]),message)
def _tfit(R,qt:str,y:str,qa:set)->bool:
    return bool(re.search(r"\b(?:1\d{3}|20\d\d)\b",y)) if qt=='date' else any(m[0].lower() not in qa for m in _NUMW.finditer(y)) if qt=='num' else any(not set(R.toks(v))<=qa for v,_,_ in R.ents(y)) if qt in ('person','place') else True
def _pages(I,R,ph:str)->list:
    import numpy as np;from amni.compute import ks_retrieve as KR
    al=getattr(I,'al',None);h=np.uint64(R.fnv(R.akey(ph).encode()));a,b=(int(np.searchsorted(al[0],h,z)) for z in ('left','right')) if al is not None else (0,0);th=np.uint64(KR.th64(ph));tz=I.tz;j=int(np.searchsorted(tz['h'],th))
    return list(dict.fromkeys([int(al[1][x]) for x in sorted((x for x in range(a,b) if int(al[2][x])>>4==0),key=lambda x:-int(al[4][x]))]+[int(I.db[int(tz['seg'][x])])+int(tz['doc'][x]) for x in range(j,int(np.searchsorted(tz['h'],th,'right')))]))
def _keys(I,R,q:str,qa:set,cache:dict,k:int=2)->Tuple[list,set]:
    E=_ents(R,q);w=[z for z in re.findall(r"[\w'-]+",q)];runs=[];r=[];tt=lambda g:I.text(int(I.first[g]),cache)[0];kt=set(R.toks(E[0])) if E else set()
    P=[g for g in dict.fromkeys(getattr(I,'lk',None) or []) if not _BADT.search(tt(g)) and set(R.toks(re.sub(r"\s*\([^()]*\)$","",tt(g))))<=qa][:k]
    for z in w+['.']:r=r+[z] if z.lower() not in _QW|R.STOP|R.QSTOP and z!='.' else (runs.append(r),[])[1]
    N=[' '.join(r[i:i+n]) for n in (3,2) for r in runs for i in range(len(r)-n+1)]+sorted({z for r in runs for z in r if z.lower() not in _TYPEW and R.toks(z) and I.widf(R.toks(z)[0])>=5},key=lambda z:-I.widf(R.toks(z)[0]))
    for ph in (list(dict.fromkeys(sorted(E,key=lambda v:-len(v))+N))[:12] if len(P)<k else []):
        g=next((g for v in dict.fromkeys([ph,' '.join(R.toks(ph))]) for g in _pages(I,R,v)[:2] if not _BADT.search(tt(g))),None)
        g is not None and g not in P and P.append(g)
        if len(P)>=k:break
    return P,kt
def _wd(R,q:str,weak:bool=False,forms:Sequence[str]=()):
    if os.environ.get('AMNI_RKNOW_WF','1')!='1':return None
    WF=__import__('amni.compute.wikifact',fromlist=['ask']);S=WF.store();qn=set(WF.nrm(' '.join([q,*forms])).split());W=[]
    for x in dict.fromkeys([q,re.sub(r"(?i)^\W*(?:please\s+)?(?:name|tell me|give me|do you know|can you tell me)\s+(?:the\s+)?","what is the ",q),*forms]):
        w=WF.ask(x);g=bool(S is not None and w and w.get('answer') and (WF.TEMPORAL or not w.get('temporal')));st=g and float(w.get('conf') or 0)>=WF.THR and float(w.get('lf') or 0)>=_f('AMNI_ADAM_T_WD_LF',0.85);ok=st or weak and g
        lb=S.name(w['e']) if ok else '';an=set(WF.nrm(w['answer']).split()) if ok else set()
        if ok and WF.nrm(lb)!=WF.nrm(w['answer']) and not an<=qn and any(set(WF.nrm(z).split())-WF.STOPR<=qn and set(WF.nrm(z).split())-WF.STOPR for z in [lb]+S.names(w['e'])):W.append({**w,'label':lb,'strong':st})
        if W and W[-1]['strong']:break
    return next((w for w in W if w['strong']),W[0] if W else None)
_wl=lambda w,cap:clip(f"{asc(w['label'])}{', '+_PL[w['p']] if w['p'] in _PL else ''}: {asc(w['answer'])}.",cap)
_DEF=re.compile(r"(?i)\bstands? for\b|\bshort for\b|\babbreviat|\bacronym\b|^\W*what\s+(?:is|are)\s+(?:a|an)\s+[\w-]+\W*$|\bwhat (?:kind|type|sort) of\b")
_KW=dict(defn=1.0,key=0.3,key2=0.15,lead=0.2,tf=0.25,ntf=0.35,rr=0.3,cs=2.0,wa=0.8,hole=0.3,nc=5,cmin=0.05,scs=1.0,stf=0.5,swa=2.0)
def _know(message:str,n:int,cap:int,history=None,w=None)->Tuple[list,str]:
    from amni.compute import rknow as R
    rq=_resolve(R,message,history)
    if not rq or not (R.factq(rq,True) or re.match(r"(?i)\W*what\s+does\s+(?:the\s+)?[A-Z][A-Z0-9.&-]+\s+stand\s+for\W*$",rq)):return [],rq
    I=R.ix();K={**_KW,**json.loads(os.environ.get("AMNI_ADAM_T_KW") or "{}")};qt=R.RL.qtype(rq);qa=set(R.toks(rq));W={w:I.widf(w) for w in dict.fromkeys(R.toks(rq)) if w not in R.QSTOP};tot=sum(W.values()) or 1.0;cov=_f('AMNI_ADAM_T_KNOW_COV',0.5);w=_wd(R,rq) if w is None else w or None;out=[];seen=set()
    w and out.append(_wl(w,cap));wa=set(R.toks(w['answer'])) if w else set()
    _,Rc,P=R.candidates(rq);cache=getattr(I,'rc',None) or {};T=[(a,z['s']) for z in Rc[:int(K['nc'])] if z['s']>=K['cmin'] for a in [set(R.toks(z['v']))-qa] if a];cs=lambda sw:max([s for a,s in T if a<=sw] or [0.0])
    KP,kt=_keys(I,R,rq,qa,cache);C={p:r for r,(_,p,_,_) in enumerate(P)};[C.setdefault(p,None) for g in KP for p in range(int(I.first[g]),min(int(I.first[g+1]),int(I.first[g])+int(_f('AMNI_ADAM_T_KEY_PASS',30))))];X=[]
    for p,r in C.items():
        t,x,g=I.text(p,cache)
        if _BADT.search(t) or _NAV.search(x[:300]):continue
        t,x=asc(t),asc(_fix(x));sw=set(R.toks(t+' '+x));c=sum(v for q,v in W.items() if q in sw)/tot;kd=KP.index(g) if g in KP else -1;tf=_tfit(R,qt,x,qa)
        if c<(cov*0.6 if kd>=0 else cov) or (kt and kd<0 and not kt<=sw):continue
        X.append((c+(K['key'] if kd==0 else K['key2'] if kd>0 else 0)+((K['lead']+(K['defn'] if kd==0 and _DEF.search(rq) else 0)) if kd>=0 and p==int(I.first[g]) else 0)+(K['tf'] if tf else -K['ntf'] if qt in ('date','num','person','place') else 0)+(K['rr']/(1+r)**0.5 if r is not None else 0)+K['cs']*cs(sw)+(K['wa'] if wa and wa<=sw else 0)-(K['hole'] if qt=='num' and '\x01' in x else 0),p,t,x,g))
    for s,p,t,x,g in sorted(X,key=lambda z:-z[0]):
        if g in seen:continue
        S=[z for z in re.split(r'(?<=[.!?])\s+',x) if z];seen.add(g);out.append(re.sub(r'\s*\x01\s*',' ',f'{t}: '+_trim(S,[sum(v for q,v in W.items() if q in set(R.toks(z)))/tot+K['scs']*cs(set(R.toks(z)))+(K['stf'] if _tfit(R,qt,z,qa) else 0)+(K['swa'] if wa and wa<=set(R.toks(z)) else 0)-('\x01' in z) for z in S],cap-len(t)-2)).strip())
        if len(out)>=(1 if w else n)+bool(w):break
    return out,rq
def docs(message:str,session=None,n:int=2)->list:
    if os.environ.get('AMNI_ADAM_T_DOCS','1')!='1':return []
    try:
        dr=__import__('amni.serve.docread',fromlist=['ask']);sid=session or __import__('amni.serve.user_memory',fromlist=['_CV'])._CV.get()[1]
        return dr.passages(sid,message,n) if sid else []
    except Exception:return []
def doclines(H:list,cap:int)->list:return [clip(asc(', '.join(x for x in (h.get('file'),__import__('amni.serve.docread',fromlist=['_loc'])._loc(h),h.get('head')) if x)+': '+str(h.get('text') or '')),cap) for h in H]
def context(message:str,history=None,user=None,session=None,budget:int=0)->dict:
    t0=time.perf_counter();c={'reply':None,'hist':list(history or []),'pre':[],'rc':[],'n':{}}
    if os.environ.get('AMNI_ADAM_T_CONTEXT','1')!='1':return {**c,'ms':0.0}
    mb,ab,kb,np_,pb=(int(_f(k,d)) for k,d in (('AMNI_ADAM_T_MEM_BYTES',640),('AMNI_ADAM_T_MATH_BYTES',240),('AMNI_ADAM_T_KNOW_BYTES',1100),('AMNI_ADAM_T_PASSAGES',2),('AMNI_ADAM_T_PASSAGE_BYTES',560)));m={'route':False};K=[]
    B=lambda:budget-len(template(message,None,0,(),c['rc']).encode('utf-8')) if budget>0 else 1<<30
    try:m=_mem(message,history,user,session);c.update(reply=m['reply'],hist=m['hist'])
    except Exception as e:c['n']['mem_err']=type(e).__name__
    tr=_tor(message,history,user,session,c);tl=tr['lines']
    try:ma=(lambda r,n:clip('Calculator: '+r,n) if r and n>=32 else '')(_math(message),min(ab,B()-1)) if ab>0 and not c['reply'] and not tl else '';c['rc']=[ma] if ma else []
    except Exception as e:c['n']['math_err']=type(e).__name__
    kq=min(kb,B()-np_)
    D=doclines(docs(message,session,max(1,np_)),min(pb,kq//max(1,np_))) if kq>=64 and not c['reply'] else [];c['n']['doc']=len(D)
    try:K,c['rq']=(D,'') if D else _know(message,np_,min(pb,kq//max(1,np_)),history) if kq>=64 and np_>0 and not (c['reply'] or c['rc'] or tl) else ([],'');K=_fit(K[::-1],0,kq)[0][::-1] if not m['route'] or c['rq'] not in ('',message) else []
    except Exception as e:c['n']['know_err']=type(e).__name__
    c['rc']+=K+tl;q=min(mb,B());c['pre']=_fit(m['pre'],0,q)[0] if m['route'] and q>0 and not (tl and os.environ.get('AMNI_UMEM_TOROID_PRE','0')!='1') else []
    c['n'].update(mem=len(c['pre']),know=len(K),math=int(len(c['rc'])>len(K)+len(tl)),tor=len(tl));c['ms']=(time.perf_counter()-t0)*1000;return c
class AdamT:
    def __init__(s,run:str='',bin:str='',timeout:float=0,load_timeout:float=0):
        s.run=run or os.environ.get('AMNI_ADAM_T','').strip();s.bin=bin or os.environ.get('AMNI_ADAM_T_BIN','').strip() or str(ROOT/'crates'/'target'/'release'/'adam-t')
        s.timeout=timeout or _f('AMNI_ADAM_T_TIMEOUT',30);s.load_timeout=load_timeout or _f('AMNI_ADAM_T_LOAD_TIMEOUT',180);s.lock=threading.RLock();s.p=None;s.fails=0;s.until=0.0
    def target(s)->str:
        g=os.path.join(s.run,'model.gf17');return s.run if s.run.endswith('.gf17') or not os.path.isfile(g) else g
    def env(s)->dict:
        return {**os.environ,'BACKEND':os.environ.get('AMNI_ADAM_T_BACKEND') or 'ndarray','RAYON_NUM_THREADS':os.environ.get('AMNI_ADAM_T_THREADS') or '2'}
    def proto(s)->Proto:
        if s.p and s.p.alive():return s.p
        if time.monotonic()<s.until:raise ProtoError(f'restart backoff {s.until-time.monotonic():.1f}s')
        s.p=None;s.p=Proto([s.bin,'serve',s.target()],s.timeout,s.env(),subprocess.DEVNULL,s.load_timeout);return s.p
    def _fail(s)->None:s.fails+=1;s.until=time.monotonic()+min(300.0,2.0**s.fails);s.p and not s.p.alive() and setattr(s,'p',None)
    def _budget(s,max_tokens:int)->int:return max(64,int(_f('AMNI_ADAM_T_CTX',s.proto().info.get('ctx') or 1024))-int(max_tokens))
    def budget(s,max_tokens:int=400)->int:
        with s.lock:
            try:return s._budget(max_tokens)
            except BaseException:s._fail();raise
    def gen(s,prompt:str,max_tokens:int=400,stops:Iterable[str]=STOPS,history=None,pre:Sequence[str]=(),rc:Sequence[str]=(),sep:str=' ',dec:Optional[dict]=None)->str:
        with s.lock:
            try:b=s._budget(max_tokens);t=s.p.text(template(prompt,history,b,pre,rc,sep),int(max_tokens),stops,**({'dec':dec} if dec else {})).strip();s.fails=0;return t
            except BaseException:s._fail();raise
    def status(s)->dict:
        p=s.p;return {'run':s.run,'alive':bool(p and p.alive()),'fails':s.fails,**({k:p.info.get(k) for k in ('model','step','fmt','ctx')} if p else {})}
    def close(s)->None:
        with s.lock:s.p and s.p.close();s.p=None
_C={}
_L=threading.Lock()
def on()->bool:return bool(os.environ.get('AMNI_ADAM_T','').strip())
def warm()->None:
    try:_know('who wrote the novel moby dick?',1,64)
    except Exception:pass
def client()->AdamT:
    r=os.environ.get('AMNI_ADAM_T','').strip()
    if os.environ.get('AMNI_BLENDER','').startswith('atex:'):return __import__('amni.serve.blender_atex',fromlist=['blender']).blender()
    with _L:
        c=_C.get(r)
        if c is None:[x.close() for x in _C.values()];_C.clear();c=_C[r]=AdamT(r);os.environ.get('AMNI_ADAM_T_CONTEXT','1')=='1' and os.environ.get('AMNI_ADAM_T_WARM','1')=='1' and threading.Thread(target=warm,daemon=True,name='adam-t-warm').start()
        return c
_safe=lambda m,t:__import__('amni.serve.adam_t_harness',fromlist=['safe']).safe(m,t) if os.environ.get('AMNI_ADAM_T_HARNESS','0')=='1' else t
def _har(message:str,history,user,session,cl,mx:int,c:dict)->Optional[dict]:
    if os.environ.get('AMNI_ADAM_T_HARNESS','0')!='1':return None
    try:return __import__('amni.serve.adam_t_harness',fromlist=['run']).run(message,history,user,session,cl,mx)
    except ProtoError:raise
    except Exception as e:c['n']['har_err']=type(e).__name__;return None
def answer(message:str,history=None,max_new_tokens:int=512,user=None,session=None)->Optional[str]:
    if not on():return None
    t0=time.monotonic();c={'ms':0.0,'n':{}}
    try:
        cl=client()
        if not (cl.p and cl.p.alive()) and time.monotonic()<cl.until:raise ProtoError('restart backoff')
        mx=min(int(_f('AMNI_ADAM_T_MAX_BYTES',400)),max(64,4*int(max_new_tokens or 0)));h=_har(message,history,user,session,cl,mx,c);p1=os.environ.get('AMNI_PIPE1','0')=='1' and os.environ.get('AMNI_ADAM_T_HARNESS','0')=='1';c=c if h else (lambda x:(x['n'].update(c['n']),x)[1])(context(message,__import__('amni.serve.pipe1',fromlist=['hist']).hist(history,user) if p1 else history,user,session,cl.budget(mx)));t=h['text'] if h else c['reply'] or (__import__('amni.serve.pipe1',fromlist=['chat']).chat(cl,message,c,mx)[0] if p1 else _safe(message,cl.gen(message,mx,STOPS,c['hist'],c['pre'],c['rc'])));why='' if t else 'empty';h and c['n'].update(h['n'])
    except Exception as e:t,why,h='',type(e).__name__,None
    LOG.debug('route=%s ms=%d ctx_ms=%.1f ctx=%s',why and f'fallback:{why}' or ('harness' if h else 'memory_action' if c.get('reply') else 'adam_t'),int((time.monotonic()-t0)*1000),c.get('ms',0.0),','.join(f'{k}:{v}' for k,v in sorted(c.get('n',{}).items())));return t or None
