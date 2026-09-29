import os,re,sys,json,math,time,functools,threading
import numpy as np
from pathlib import Path
import amni.compute.wikifact as WF
ROOT=Path(__file__).resolve().parents[2];CD=Path(os.environ.get('AMNI_CONCEPTS_DIR',str(ROOT/'data'/'concepts')))
DL=float(os.environ.get('AMNI_COG_DELTA','0.5'));D0=float(os.environ.get('AMNI_COG_D0','2.5'))
CF=dict(wd=0.95,wn=0.95,us=0.9,gl=0.85,cn=0.8,tx=0.8,br=0.7,co=0.5);STRICT={'wd','wn','us','gl'};NL=dict(factual=2,analytical=4,creative=4)
PN={6:'head of government',17:'country',19:'place of birth',20:'place of death',21:'sex',22:'father',25:'mother',26:'spouse',27:'citizenship',30:'continent',31:'instance of',35:'head of state',36:'capital',37:'official language',38:'currency',39:'position held',40:'child',50:'author',54:'team',57:'director',58:'screenwriter',61:'discoverer or inventor',69:'educated at',84:'architect',85:'anthem',86:'composer',101:'field of work',102:'political party',106:'occupation',108:'employer',110:'illustrator',112:'founded by',115:'home venue',118:'league',122:'form of government',123:'publisher',127:'owned by',131:'located in',136:'genre',138:'named after',140:'religion',144:'based on',155:'follows',156:'followed by',159:'headquarters',161:'cast member',162:'producer',166:'award received',169:'chief executive',170:'creator',175:'performer',176:'manufacturer',178:'developer',179:'part of the series',206:'next to body of water',264:'record label',272:'production company',276:'location',279:'subclass of',286:'head coach',361:'part of',364:'original language',397:'orbits',403:'flows into',407:'language',413:'position played',449:'broadcaster',451:'partner',452:'industry',463:'member of',488:'chairperson',495:'country of origin',509:'cause of death',527:'has part',641:'sport',664:'organizer',674:'characters',676:'lyrics by',710:'participant',725:'voice actor',740:'formed in',749:'parent organization',750:'distributor',800:'notable work',840:'set in',885:'source of the watercourse',915:'filming location',991:'successful candidate',1037:'director or manager',1056:'product',1080:'fictional universe',1303:'instrument',1308:'officeholder',1346:'winner',1365:'replaces',1366:'replaced by',1376:'capital of',1431:'executive producer',1433:'published in',1441:'present in work',1923:'participating team',3373:'sibling',3450:'season of',4552:'mountain range',4584:'first appearance',4908:'season',171:'parent taxon',183:'endemic to',186:'made from',366:'has use',425:'field of this occupation',460:'said to be the same as',461:'opposite of',462:'color',516:'powered by',703:'found in taxon',737:'influenced by',780:'symptoms',828:'has cause',921:'main subject',941:'inspired by',1034:'main food source',1060:'transmitted by',1269:'facet of',1478:'has immediate cause',1479:'has contributing factor',1535:'used by',1536:'immediate cause of',1537:'contributing factor of',1542:'has effect',1552:'has characteristic',1557:'manifestation of',1582:'natural product of',1672:'source of',1995:'health specialty',2012:'cuisine',2079:'fabrication method',2175:'treats',2176:'treated with',2283:'uses',2578:'studies',2579:'studied by',2670:'has parts of the class',2974:'habitat',2975:'host',3094:'develops from',3095:'practiced by',4330:'contains',5642:'risk factor',9714:'taxon range'}
WR=['synonym','type of','has type','has part','part of','has member','member of','made of','substance of','similar to','see also','attribute','entails','causes','related form','pertains to']
JQ=(4167410,13406463,4167836,11266439,101352,202444,12308941,11879590,17362920,3331189,22808320,13442814,21286738);KQ=(16521,55983715)
ISA=(31,279);TX=(31,279,171);LOC=(131,17,30,361,706,276,1376);PW={17:0.8,31:0.8,279:0.85,361:0.85,527:0.85,131:0.85,276:0.85,495:0.85,407:0.8,364:0.8,27:0.85,166:0.7,1441:0.7,179:0.8,101:0.9,1269:0.8,2670:0.85,460:0.8,9714:0.7,462:0.75,1552:0.85,921:0.85,737:0.85,941:0.85,703:0.8};RV=re.compile(r"(?i)\b(?:made|makes?|produc\w*|used?|uses|from|caus\w*|transmit\w*|spread\w*|native|grow[ns]?|grown|lives?|lived|eats?|feeds?|form(?:s|ed)?|consist\w*|contain\w*|derived|home|origin\w*|invent\w*|discover\w*|founded|founder|wrote|written|composed|built|located|named|pollinat\w*|depend\w*)\b");XS={166,1441,1080,4584,179,155,156,1365,1366,40,21,407,364,674,161,725,1923,710,737,941,460,9714};BADT=re.compile(r"(?i)^(?:lists?|index|outline|timeline|glossary|bibliography|discography|filmography) of\b|\(disambiguation\)|\b(?:by (?:country|year|city|state)|in popular culture)\b")
BSTOP=set("also used known called named include including includes such many most more other some several which where while although however first second third one two three four five six seven ten century year years time times form forms part parts type types kind kinds species genus family different various common commonly often usually found make made well like large small high low new old same number area areas group groups term terms word words name names use uses system way ways world state states around within between during among along across based particular certain given main major general total present important early late modern".split())
G17=[pow(3,k,17) for k in range(16)];L17={v:k for k,v in enumerate(G17)}
def rz(n:int)->tuple:return (n>>4,G17[n&15])
def rn(c:tuple)->int:return (c[0]<<4)+L17[c[1]]
def boost(c:tuple,k:int)->tuple:l=L17[c[1]]+(k&15);return (c[0]+(k>>4)+(l>>4),G17[l&15])
def ldist(na:float,nb:float,cs:float)->float:a,b=DL*na,DL*nb;return float(math.acosh(max(1.0,math.cosh(a)*math.cosh(b)-math.sinh(a)*math.sinh(b)*cs)))
def band(D)->float:return 0.6 if D is None else (D/D0)*math.exp(1-D/D0)
def enabled()->bool:return os.environ.get('AMNI_COGNITION','0')=='1'
@functools.lru_cache(maxsize=1)
def _g()->dict:
    G=np.load(CD/'wordnet_graph.npz',allow_pickle=True);Z=np.load(CD/'sphere.npz',allow_pickle=True);nm=[str(x) for x in G['node_names']];N=len(nm)
    s,d,t=G['edge_src'].astype(np.int64),G['edge_dst'].astype(np.int64),G['edge_type'].astype(np.int64);o=np.lexsort((d,s));s,d,t=s[o],d[o],t[o];h=t==1;hs,hd=s[h],d[h];dep=np.full(N,-1,np.int64);r=np.ones(N,bool);r[hs]=False;dep[r]=0
    for k in range(1,64):
        m=(dep[hd]==k-1)&(dep[hs]<0);dep[hs[m]]=k
        if not m.any():break
    xyz=np.asarray(Z['xyz'][:N],np.float64);xyz/=np.sqrt((xyz*xyz).sum(1,keepdims=True))+1e-12;wp=G['word_ptr']
    return dict(nm=nm,ix={n:i for i,n in enumerate(nm)},pos=G['node_pos'].astype(np.int64),d=d,t=t,ptr=np.searchsorted(s,np.arange(N+1)),dep=np.maximum(dep,0),xyz=xyz,W={str(w):(int(wp[i]),int(wp[i+1])) for i,w in enumerate(G['word_list'])},ws=G['word_syn'])
def hd(i:int)->str:return _g()['nm'][i].rsplit('.',2)[0].replace('_',' ')
def lab(i:int)->str:return _gi()[2][i] if _gi.cache_info().currsize else hd(i)
def dep(i:int)->int:return int(_g()['dep'][i])
def out(i:int,T=None)->list:
    G=_g();a,b=int(G['ptr'][i]),int(G['ptr'][i+1]);d,t=G['d'][a:b],G['t'][a:b];m=np.ones(len(d),bool) if T is None else np.isin(t,list(T));return list(zip(d[m].tolist(),t[m].tolist()))
@functools.lru_cache(maxsize=16384)
def wsyn(w:str,pos:int=-1)->tuple:
    G=_g();w=' '.join(str(w).lower().replace('-',' ').replace('_',' ').split());O=[];sn=lambda i,v:(hd(i)!=v,int(G['nm'][i].rsplit('.',1)[1]) if hd(i)==v else 99)
    for v in dict.fromkeys([w,w[:-3]+'y' if w.endswith('ies') else w,w[:-2] if w.endswith(('ches','shes','sses','xes','zes','oes')) else w,w[:-1] if w.endswith('s') and not w.endswith(('ss','us','is')) else w]):
        r=G['W'].get(v);O+=sorted((int(x) for x in G['ws'][r[0]:r[1]]),key=lambda i:sn(i,v)) if r else []
    m=None if O else (lambda wn:wn and wn.morphy(w.replace(' ','_'),'n'))(WF._wn());r=G['W'].get(m.replace('_',' ')) if m else None;O+=sorted((int(x) for x in G['ws'][r[0]:r[1]]),key=lambda i:sn(i,m.replace('_',' '))) if r else []
    return tuple(i for i in dict.fromkeys(O) if pos<0 or int(G['pos'][i])==pos)
@functools.lru_cache(maxsize=65536)
def anc(i:int,T:tuple=(1,),k:int=24)->dict:
    A={i:(0,-1)};F=[i]
    for h in range(1,k):
        F=[A.setdefault(y,(h,x)) and y for x in F for y,_ in out(x,T) if y not in A]
        if not F:break
    return A
def chain(A:dict,z:int)->list:
    c=[z]
    while A[c[-1]][1]>=0:c.append(A[c[-1]][1])
    return c[::-1]
def lca(a:int,b:int)->tuple:
    A,B=anc(a),anc(b);C=set(A)&set(B);return (max(C,key=lambda z:(dep(z),-(A[z][0]+B[z][0]))) if C else None),A,B
def sdist(i:int,j:int):
    if i<0 or j<0:return None
    G=_g();l=lca(i,j)[0];nl=dep(l) if l is not None else -1;return ldist(dep(i)-nl,dep(j)-nl,float((G['xyz'][i]*G['xyz'][j]).sum()))
def coord(i:int)->dict:
    G=_g();n=dep(i);return dict(node=G['nm'][i],gf17=[(i//k)%17 for k in (1,17,289,4913,83521)],n=n,rap=rz(n),dir=[round(float(x),4) for x in G['xyz'][i]])
def S():return WF.store()
@functools.lru_cache(maxsize=1)
def _jk()->tuple:s=S();return frozenset(s.ix(q) for q in JQ)-{-1},frozenset(s.ix(q) for q in KQ)-{-1}
def wname(e:int)->str:return S().name(e)
def qid(e:int)->int:return int(S().e_qid[e])
@functools.lru_cache(maxsize=65536)
def wf(e:int)->tuple:
    s=S();a,n=int(s.e_fs[e]),int(s.e_fn[e]);P=np.asarray(s.f_p[a:a+n]).astype(np.int64);V=np.asarray(s.f_v[a:a+n]).astype(np.int64);m=np.asarray(s.f_k[a:a+n])==1;return P[m],V[m],a+np.flatnonzero(m),np.asarray(s.f_r[a:a+n])[m]
def wdeg(e:int)->int:s=S();k=np.uint32(e);return int(np.searchsorted(s.r_v,k,'right')-np.searchsorted(s.r_v,k,'left'))
def wrev(e:int,cap:int=5000)->tuple:
    s=S();k_=np.uint32(e);i,j=int(np.searchsorted(s.r_v,k_,'left')),int(np.searchsorted(s.r_v,k_,'right'));F=np.asarray(s.r_f[i:j]).astype(np.int64) if j-i<=cap else np.zeros(0,np.int64)
    return np.asarray(s.f_p[F]).astype(np.int64),np.asarray(s.f_s[F]).astype(np.int64),F
def hub(e:int)->float:return 1/(1+math.log1p(wdeg(e)/100))
@functools.lru_cache(maxsize=8192)
def went(t:str,cls:bool)->tuple:
    s=S()
    if s is None:return ()
    C={};J,K=_jk()
    for v,f,_ in WF.variants(WF.nrm(t)):
        for e,k in s.ents(v):C[e]=max(C.get(e,0),f*(1.0,0.6,1.0,0.85)[min(k,3)])
    R=[(w*math.log2(2+int(s.e_sl[e]))**2*(1.6 if cls and ((P==279).any() or bool(set(V[P==31].tolist())&K)) else 1.0),e) for e,w in C.items() for P,V,_,_ in [wf(e)] if not set(V[P==31].tolist())&J]
    return tuple(e for _,e in sorted(R,reverse=True)[:4])
@functools.lru_cache(maxsize=65536)
def wanc(e:int)->int:
    n=wsyn(WF.nrm(wname(e)),0)
    if n:return n[0]
    P,V,_,_=wf(e);return next((m[0]+0 for v in V[np.isin(P,ISA)][:3].tolist() for m in [wsyn(WF.nrm(wname(v)),0)] if m),-1)
@functools.lru_cache(maxsize=1)
def _ix()->tuple:
    try:import amni.compute.rknow as R;I=R.ix();I.__dict__.setdefault('al',I.aliases());return R,I
    except Exception:return None,None
@functools.lru_cache(maxsize=1)
def _co()->dict:
    C=np.load(CD/'cooc_graph.npz',allow_pickle=True);V=[str(x) for x in C['vocab']];s,d,w=C['edge_src'].astype(np.int64),C['edge_dst'].astype(np.int64),C['edge_w'].astype(np.float64);o=np.argsort(d,kind='stable');n=np.arange(len(V)+1)
    return dict(V=V,ix={x:i for i,x in enumerate(V)},cnt=C['counts'],p=np.searchsorted(s,n),d=d,w=w,rp=np.searchsorted(d[o],n),rs=s[o],rw=w[o])
def _sg(w:str)->str:return w[:-3]+'y' if w.endswith('ies') and len(w)>4 else w[:-1] if w.endswith('s') and not w.endswith(('ss','us','is')) and len(w)>3 else w
def cnb(w:str,rev:bool=False,minc:int=300)->list:
    C=_co();R={};w=w.lower()
    for f in dict.fromkeys([w,w+'s',w+'es',_sg(w)]):
        i=C['ix'].get(f);a,b=(int((C['rp'] if rev else C['p'])[i]),int((C['rp'] if rev else C['p'])[i+1])) if i is not None else (0,0)
        for j,x in zip((C['rs'] if rev else C['d'])[a:b].tolist(),(C['rw'] if rev else C['w'])[a:b].tolist()):
            v=C['V'][j];u=_sg(v);v.isalpha() and int(C['cnt'][j])>=minc and u!=_sg(w) and u not in BSTOP and R.__setitem__(u,max(R.get(u,0.0),x))
    return sorted(R.items(),key=lambda z:-z[1])
ART=re.compile(r"(?i)^(?:the|a|an|some|my|your|our)\s+")
def _urows(user,ent:str)->list:
    try:
        UT=__import__('amni.serve.user_toroid',fromlist=['load']);st=user if hasattr(user,'s') and hasattr(user,'text') else UT.load(__import__('amni.serve.user_memory',fromlist=['user_key']).user_key(user)) if user is not None and UT.enabled() else None
        return [r for r in (UT._row(st,i) for i in range(len(st.s))) if r['ent'].lower()==ent.lower()] if st is not None else []
    except Exception:return []
def resolve(term,user=None)->dict:
    if isinstance(term,dict):return term
    raw=' '.join(str(term).strip(' ?.!,;:"\'').split());t=ART.sub('',ART.sub('',raw));my=bool(re.match(r"(?i)(?:my|our)\s",raw));u=_urows(user,t) if my else [];pr=bool(re.search(r"(?:^|\s)[A-Z]",raw)) and not my
    n=wsyn(t);n0=[i for i in n if _g()['pos'][i]==0] or list(n);n0=[i for i in n0 if pr or not lab(i)[:1].isupper()] or n0;w=went(t,not pr and bool(n0))
    return dict(t=t,raw=raw,n=n0[:4],na=list(n[:4]),w=list(w[:3]),u=u,my=my,pr=pr,lab=f"your {t}" if my else t)
def E(a,r,b,src,ref='',ka=-1,kb=-1,c=None,line=None)->dict:return dict(a=a,r=r,b=b,src=src,ref=ref,ka=ka,kb=kb,c=CF[src] if c is None else c,line=line or f"{a}, {r}: {b}.")
def P(e:list,sp:float=1.0,kind:str='')->dict:return dict(e=e,sp=sp,kind=kind)
def wE(s,p,o)->dict:return E(wname(s),PN.get(p,f"P{p}"),wname(o),'wd',f"Wikidata Q{qid(s)} P{p} Q{qid(o)}",wanc(s),wanc(o))
def nE(i,t,j)->dict:return E(lab(i),WR[t],lab(j),'wn',f"WordNet {_g()['nm'][i]} {WR[t]} {_g()['nm'][j]}",i,j)
def iE(i,l,k=None)->dict:return E(lab(i),'type of',lab(l),'wn',f"WordNet {_g()['nm'][i]} hypernym^{k if k is not None else anc(i)[l][0]} {_g()['nm'][l]}",i,l)
def _wdp(X,Y)->list:
    R=[]
    for ex in X['w'][:2]:
        Px,Vx,_,_=wf(ex)
        for ey in Y['w'][:2]:
            if ex==ey:continue
            Py,Vy,_,_=wf(ey);R+=[P([wE(ex,int(p),ey)],PW.get(int(p),1.0),'wd1') for p in dict.fromkeys(Px[Vx==ey].tolist())]+[P([wE(ey,int(p),ex)],PW.get(int(p),1.0),'wd1') for p in dict.fromkeys(Py[Vy==ex].tolist())]
            for m in np.intersect1d(Vx,Vy)[:12].tolist():m not in (ex,ey) and R.append(P([wE(ex,int(Px[Vx==m][0]),m),wE(ey,int(Py[Vy==m][0]),m)],hub(m),'wdshare'))
            for a,Pa,Va,b in ((ex,Px,Vx,ey),(ey,Py,Vy,ex)):
                M=sorted(set(Va.tolist())-{a,b},key=wdeg)[:60]
                for m in M:
                    Pm,Vm,_,_=wf(m);h=Pm[Vm==b]
                    len(h) and R.append(P([wE(a,int(Pa[Va==m][0]),m),wE(m,int(h[0]),b)],hub(m),'wd2'))
            Rx,Sx,_=wrev(ex);Ry,Sy,_=wrev(ey)
            for m in sorted(np.intersect1d(Sx,Sy).tolist(),key=lambda z:-int(S().e_sl[z]))[:4]:R.append(P([wE(m,int(Rx[Sx==m][0]),ex),wE(m,int(Ry[Sy==m][0]),ey)],0.9,'wdco'))
    return R
def _pair(X,Y)->tuple:
    C=[((dep(l) if l is not None else -1)-2.5*(i+j),x,y,l) for i,x in enumerate(X['n'][:3]) for j,y in enumerate(Y['n'][:3]) for l in [lca(x,y)[0]]]
    return max(C,key=lambda z:z[0])[1:] if C else (-1,-1,None)
def _align(X,x:int)->dict:
    k=lambda e:(0 if wanc(e)==x else 1 if (l:=lca(wanc(e),x)[0] if wanc(e)>=0 else None) is not None and dep(l)>=dep(x)-2 else 2)
    return {**X,'w':sorted(X['w'],key=k)} if x>=0 and X['w'] else X
def _pre(X,Y)->tuple:x,y,_=_pair(X,Y);return _align(X,x),_align(Y,y)
def _wnp(X,Y)->list:
    x,y,l=_pair(X,Y);R=[]
    if x<0 or x==y:return R
    R+=[P([nE(x,t,y)],1.0,'wn1') for d,t in out(x) if d==y]
    Ox={d:t for d,t in out(x) if t not in (1,2)};Oy={d:t for d,t in out(y) if t not in (1,2)}
    R+=[P([nE(x,Ox[m],m),nE(y,Oy[m],m)],min(1.0,(dep(m)+1)/10),'wnshare') for m in list(set(Ox)&set(Oy))[:6]]
    if l is not None:R.append(P([iE(x if l!=x else y,l)],1.0,'wnisa') if l in (x,y) else P([iE(x,l),iE(y,l)],float(np.clip((dep(l)+1)/12,0.05,1.0)),'wnlca'))
    return R
GSTOP=set("the a an of and or in on at to for by with from as is was are were be been it its that this which who whom whose has had have not but into than then there any all each esp especially usually often something someone ones etc being having how what when where so very".split())
@functools.lru_cache(maxsize=1)
def _gi()->tuple:
    wn=WF._wn();nm=_g()['nm'];D={};gl=[];LN=[]
    for i,s in enumerate(wn.all_synsets() if wn else []):
        ok=i<len(nm) and s.name()==nm[i];g=s.definition() if ok else '';gl.append(g);LN.append(s.lemmas()[0].name().replace('_',' ') if ok else hd(i) if i<len(nm) else '')
        for w in {_sg(w) for w in re.findall(r"[a-z]+",g.lower())}:len(w)>2 and w not in BSTOP and w not in GSTOP and D.setdefault(w,[]).append(i)
    return {w:np.asarray(v,np.int32) for w,v in D.items()},gl,LN
@functools.lru_cache(maxsize=65536)
def gloss(i:int)->str:
    gl=_gi()[1];return gl[i] if 0<=i<len(gl) else ''
def gwords(i:int)->list:return [w for w in dict.fromkeys(_sg(w) for w in re.findall(r"[a-z]+",gloss(i).lower())) if len(w)>2 and w not in BSTOP and w not in GSTOP]
def gspec(w:str)->float:n=len(_gi()[0].get(w,()));return 0.0 if not n else min(1.0,math.log(len(_gi()[1])/n)/8)
def _mre(T:list)->list:return [re.compile(r"(?i)\b(?:"+re.escape(w)+r"(?:s|es|'s)?"+(r"|"+re.escape(w[:-1])+r"ies" if w.endswith("y") else "")+r")\b") for w in T]
def _glp(X,Y)->list:
    R=[];tk=lambda t:[w for w in re.findall(r"[a-z]+",_sg(t.lower())) if w not in BSTOP and len(w)>2]
    for A,B in ((X,Y),(Y,X)):
        rb=_mre(tk(B['t']))
        for i in A['n'][:2]:
            g=gloss(i)
            if rb and g and all(r.search(g) for r in rb):R.append(P([E(lab(i),'defined as',g,'gl',f"WordNet gloss {_g()['nm'][i]}",i,(B['n'] or [-1])[0],line=f"{lab(i)}: {g}.")],1.0,'gl'));break
    return R
def _tdocs(R,I,t:str)->set:
    al=I.al
    if al is None:return set()
    h=np.uint64(R.fnv(R.akey(t).encode()));a,b=int(np.searchsorted(al[0],h,'left')),int(np.searchsorted(al[0],h,'right'));return {int(al[1][x]) for x in range(a,min(b,a+8)) if int(al[2][x])>>4==0}
def _plist(I,T:list):
    J=[I.lookup(w) for w in T]
    if not J or min(J)<0:return None
    J=sorted(J,key=lambda j:int(I.df[j]));p=I.post(J[0])[0]
    for j in J[1:]:p=np.intersect1d(p,I.post(j)[0],assume_unique=True)
    return p
@functools.lru_cache(maxsize=4096)
def text(a:str,b:str,k:int=24):
    R,I=_ix()
    if I is None:return None
    ta,tb=R.toks(a),R.toks(b)
    if not ta or not tb or set(ta)&set(tb):return None
    pa,pb=_plist(I,ta),_plist(I,tb)
    if pa is None or pb is None:return None
    c=np.intersect1d(pa,pb,assume_unique=True);D=_tdocs(R,I,a)|_tdocs(R,I,b);g=np.searchsorted(I.first,c.astype(np.uint32),'right')-1;tm=np.isin(g,list(D)) if D else np.zeros(len(c),bool);L=np.asarray(I.ntok[c]).astype(np.int64);o=np.lexsort((L,~tm))[:k];ma,mb=_mre(ta),_mre(tb);cache={};best=None;cnt={};ex=set(ta)|set(tb)|R.STOP
    for p in o.tolist():
        t,x,_=I.text(int(c[p]),cache);seen=set()
        for z in re.split(r"(?<=[.!?])\s+|\n+",x):
            if len(z)>320 or len(z.split())<5 or not all(r.search(z) for r in ma+mb) or re.search(r"^\s*[-*|]|\||:\s|\"",z):continue
            A=[m.start() for r in ma for m in [r.search(z)]];B_=[m.start() for r in mb for m in [r.search(z)]];gap=len(z[min(A[0],B_[0]):max(A[0],B_[0])].split());lead=min(A[0],B_[0])<=5 or bool(re.match(r"(?i)\W*(?:the|an?|most|many|some|all)\s",z[:12]) and min(A[0],B_[0])<=12)
            sc=(1.0 if tm[p] else 0.0)+1/(1+len(z.split())/20)-0.4*(z.count(',')>4)-1.0*bool(BADT.search(t))+0.4*lead+0.3*(gap<=8)+0.3*bool(RV.search(z));best=max(best or (sc,t,z.strip(),bool(tm[p])),(sc,t,z.strip(),bool(tm[p])))
            seen|={w for w in R.toks(z) if w.isalpha() and len(w)>3 and w not in ex and w not in BSTOP}
        for w in seen:cnt[w]=cnt.get(w,0)+1
    br=sorted(((n*I.widf(w),w,n) for w,n in cnt.items() if n>=2 and I.widf(w)>=4.0),reverse=True)[:3]
    return dict(n=int(len(c)),sent=best,br=[(w,n,round(s,2)) for s,w,n in br])
def _txp(X,Y)->list:
    r=text(X['t'],Y['t']);R=[];ax,ay=(X['n'] or [-1])[0],(Y['n'] or [-1])[0]
    if not r:return R
    s=r['sent'];s and s[0]>0.8 and R.append(P([E(X['lab'],'mentioned with',Y['lab'],'tx',f"Wikipedia: {s[1]}",ax,ay,CF['tx']*(1.0625 if s[3] else 0.9),f"{s[1]}: {s[2][:240]}")],1.0,'tx'))
    for w,n,_ in r['br'][:2]:m=(wsyn(w) or [-1])[0];c=CF['br']*min(1.0,0.5+n/10);R.append(P([E(X['lab'],'linked with',w,'br',f"Wikipedia co-mention ({n} passages)",ax,m,c),E(Y['lab'],'linked with',w,'br',f"Wikipedia co-mention ({n} passages)",ay,m,c)],1.0,'br'))
    return R
def _cop(X,Y)->list:
    try:A,B=dict(cnb(X['t'])),dict(cnb(Y['t']))
    except Exception:return []
    ax,ay=(X['n'] or [-1])[0],(Y['n'] or [-1])[0];cc=lambda z:CF['co']*min(1.0,z/8);R=[]
    yt,xt=_sg(Y['t'].lower()),_sg(X['t'].lower());(yt in A or xt in B) and R.append(P([E(X['lab'],'associated with',Y['lab'],'co','corpus co-occurrence (PMI)',ax,ay,cc(max(A.get(yt,0),B.get(xt,0))))],1.0,'co1'))
    for m in sorted(set(A)&set(B),key=lambda z:-A[z]-B[z])[:3]:mi=(wsyn(m) or [-1])[0];R.append(P([E(X['lab'],'associated with',m,'co','corpus co-occurrence (PMI)',ax,mi,cc(A[m])),E(Y['lab'],'associated with',m,'co','corpus co-occurrence (PMI)',ay,mi,cc(B[m]))],1.0,'co2'))
    return R
def _upre(X)->list:
    if not X['my']:return []
    a=(X['n'] or [-1])[0];return [E(X['lab'],'type of',X['t'],'us','user memory',a,a)]+[E(X['lab'],r['a'] or r['w'],r['v'],'us','user memory',a,-1) for r in X['u'][:2] if r['v']]
def _rank(R:list,mode:str,X=None,Y=None)->list:
    for p in R:
        p['v']=float(np.prod([e['c'] for e in p['e']]));hp=0.85**(len(p['e'])-1);ks=[k for e in p['e'] for k in (e['ka'],e['kb'])]
        L=[d for e in p['e'] for d in [sdist(e['ka'],e['kb'])] if d is not None];p['D']=float(np.mean(L)) if L else sdist(X['n'][0],Y['n'][0]) if X and Y and X['n'] and Y['n'] else None
        p['q']=round(p['v']*p['sp']*hp,4);p['cr']=round(p['v']*hp*band(p['D']),4)
    R=[p for p in R if mode!='factual' or all(e['src'] in STRICT or (e['src']=='tx' and e['c']>CF['tx']) for e in p['e'])]
    return sorted(R,key=lambda p:-(p['cr'] if mode=='creative' else p['q']))
def _out(op:str,mode:str,R:list,t0:float,k:int=3,pre:list=(),**kw)->dict:
    R=R[:k];sk='cr' if mode=='creative' else 'q';T=R[:1]+[p for p in R[1:2] if mode!='factual' and p.get(sk,0)>=0.5*R[0].get(sk,0) and (op!='analogy' or p.get('ans')==R[0].get('ans'))];L=list(dict.fromkeys(e['line'] for e in list(pre)+[e for p in T for e in p['e']]))[:NL.get(mode,4)+len(pre)]
    return dict(op=op,mode=mode,ok=bool(R),paths=[{k2:v for k2,v in p.items()} for p in R],lines=L,ms=round((time.perf_counter()-t0)*1000,2),**kw)
def relate(x,y,mode:str='analytical',user=None,k:int=3)->dict:
    t0=time.perf_counter();X,Y=_pre(resolve(x,user),resolve(y,user));R=_rank(_wdp(X,Y)+_wnp(X,Y)+_glp(X,Y)+_cnp(X,Y)+_txp(X,Y)+(_cop(X,Y) if mode!='factual' else []),mode,X,Y)
    return _out('relate',mode,R,t0,k,_upre(X)+_upre(Y),x=X['lab'],y=Y['lab'])
def wup(e:int,T=TX,k:int=4,w:int=6)->dict:
    A={e:(0,-1,0)};F=[e]
    for h in range(1,k):
        N=[]
        for x in F:
            Px,Vx,_,Rx=wf(x);o=np.argsort(-(Rx.astype(np.int64)),kind='stable')
            for p,v in list(zip(Px[o].tolist(),Vx[o].tolist()))[:64]:
                p in T and v not in A and len(N)<w*len(F) and (A.__setitem__(v,(h,x,p)),N.append(v))
        F=N
        if not F:break
    return A
def wch(A:dict,m:int)->list:
    c=[m]
    while A[c[-1]][1]>=0:c.append(A[c[-1]][1])
    return c[::-1]
def wI(e:int,A:dict,m:int)->dict:return wE(e,A[m][2],m) if A[m][0]==1 else E(wname(e),'type of',wname(m),'wd',"Wikidata "+" > ".join(f"Q{qid(z)}" for z in wch(A,m)),wanc(e),wanc(m))
def common(x,y,mode:str='analytical',user=None,k:int=4)->dict:
    t0=time.perf_counter();X,Y=_pre(resolve(x,user),resolve(y,user));R=[p for p in _wnp(X,Y) if p['kind'] in ('wnlca','wnshare','wnisa')];XI=list(ISA)+list(XS)
    for ex in X['w'][:2]:
        for ey in Y['w'][:2]:
            if ex==ey:continue
            A,B=wup(ex),wup(ey);C=sorted(set(A)&set(B)-{ex,ey},key=lambda z:(A[z][0]+B[z][0],wdeg(z)))[:2]
            R+=[P([wI(ex,A,m),wI(ey,B,m)],min(hub(z) for z in wch(A,m)[1:]+wch(B,m)[1:])*0.9**(A[m][0]+B[m][0]-2),'wdcls') for m in C]
            Px,Vx,_,_=wf(ex);Py,Vy,_,_=wf(ey);R+=[P([wE(ex,int(Px[Vx==m][0]),m),wE(ey,int(Py[Vy==m][0]),m)],hub(m),'wdshare') for m in np.intersect1d(Vx[~np.isin(Px,XI)],Vy[~np.isin(Py,XI)])[:8].tolist()]
    R+=[p for p in _cnp(X,Y) if p['kind']=='cnshare']+[p for p in _txp(X,Y) if p['kind']=='br']+([p for p in _cop(X,Y) if p['kind']=='co2'] if mode!='factual' else [])
    R=_rank(R,mode,X,Y);return _out('common',mode,R,t0,k,_upre(X)+_upre(Y),x=X['lab'],y=Y['lab'])
def abstract(x,mode:str='analytical',user=None,k:int=3)->dict:
    t0=time.perf_counter();X=resolve(x,user);k=1 if mode=='factual' else k;e=[];nodes=[]
    N0=[i for i in X['n'] if _g()['pos'][i]==0 and out(i,(1,))]
    if N0 and (not X['pr'] or hd(N0[0])==X['t'].lower()):
        i=N0[0];c=rz(dep(i))
        for _ in range(k):
            H=[d for d,t in out(i,(1,))]
            if not H:break
            j=max(H,key=lambda z:(dep(z),-z));c=boost(c,dep(j)-rn(c));e.append(iE(i,j,1));nodes.append(dict(node=_g()['nm'][j],n=rn(c),rap=c));i=j
    if not e and X['w']:
        x0=X['w'][0]
        for _ in range(k):
            Px,Vx,_,Rx=wf(x0);o=[(p,v) for p,v,r in sorted(zip(Px.tolist(),Vx.tolist(),Rx.tolist()),key=lambda z:(z[0]!=(31 if not e else 279),-z[2])) if p in ISA and v!=x0 and v not in _jk()[0]]
            if not o:break
            p,v=o[0];e.append(wE(x0,p,v));nodes.append(dict(node=f"Q{qid(v)}",n=None));x0=v
    R=_rank([P(e,1.0,'up')],mode) if e else [];return _out('abstract',mode,R,t0,1,_upre(X),x=X['lab'],chain=nodes)
def _ants(i:int)->list:
    wn=WF._wn()
    try:return [(a.synset().name(),a.name().replace('_',' ')) for l in wn.synset(_g()['nm'][i]).lemmas() for a in l.antonyms()] if wn else []
    except Exception:return []
def _gpat(i:int,b:str):
    g=gloss(i).lower();m=re.search(r"(\w+)\s+"+re.escape(_sg(b.lower()))+r"\w{0,3}\b",g);return m.group(1) if m and m.group(1) not in ('a','an','the','of','or','and','to') else None
def _gapply(i:int,w:str):
    g=gloss(i).lower();m=re.search(r"\b"+re.escape(w)+r"\s+([\w\s-]+)",g)
    if not m:return None
    T=m.group(1).split()[:4]
    while T and T[0] in GSTOP:T=T[1:]
    T=T[:next((j for j,t in enumerate(T) if t in GSTOP),len(T))]
    return next(((wsyn(v,0)[0],v) for n in range(len(T),0,-1) for v in [' '.join(T[len(T)-n:])] if wsyn(v,0) and v not in BSTOP),None)
def analogy(a,b,c,mode:str='analytical',user=None,k:int=3)->dict:
    t0=time.perf_counter();A,B,Cx=resolve(a,user),resolve(b,user),resolve(c,user);V={};why={}
    def vote(key,lb,sc,ea,ec,kd):key=_sg(lb.lower());V[key]=V.get(key,0.0)+sc;(key not in why or sc>why[key][0]) and why.__setitem__(key,(sc,lb,ea,ec,kd))
    ag=lambda d,y:1.5 if (l:=lca(d,y)[0]) is not None and dep(l)>=max(dep(d),dep(y))-3 else 1.0
    tc=lambda e:set(wf(e)[1][np.isin(wf(e)[0],ISA)].tolist())
    for ea in A['w'][:2]:
        Pa,Va,_,_=wf(ea)
        for eb in B['w'][:2]:
            Pb,Vb,_,_=wf(eb);cb=tc(eb);T=[(int(p),1) for p in dict.fromkeys(Pa[Va==eb].tolist())]+[(int(p),-1) for p in dict.fromkeys(Pb[Vb==ea].tolist())]
            for p,dr in T:
                for r,ec in enumerate(Cx['w'][:2]):
                    if dr>0:Pc,Vc,_,Rc=wf(ec);D=sorted([(v,rr) for pp,v,rr in zip(Pc.tolist(),Vc.tolist(),Rc.tolist()) if pp==p],key=lambda z:-z[1])
                    else:Pr,Sr,Fr=wrev(ec,20000);D=[(s_,1) for pp,s_ in zip(Pr.tolist(),Sr.tolist()) if pp==p];D=sorted(D,key=lambda z:-int(S().e_sl[z[0]]))[:8]
                    for q,(d,rr) in enumerate(D[:8]):d!=ec and vote(('w',d),wname(d),(1.0 if rr==1 else 0.6)*(1.0 if dr>0 else 0.7)*(1.5 if cb&tc(d) else 1.0)/(1+q)/(1+r),(ea,p,eb,dr),(ec,p,d,dr),'wd')
    for i,x in enumerate(A['na'][:6]):
        if any(nm_.lower()==B['t'].lower() for _,nm_ in _ants(x)):[vote(('t',n2.lower()),n2,1.0/(1+q)/(1+r),(x,'opposite of',-1),(z,'opposite of',n2),'ant') for r,z in enumerate(Cx['na'][:6]) for q,(s2,n2) in enumerate(_ants(z))];break
    for i,x in enumerate(A['n'][:2]):
        Z=sorted(Cx['n'][:3],key=lambda z:-(dep(lca(z,x)[0]) if lca(z,x)[0] is not None else -1))[:2]
        for y in B['n'][:2]:
            ts=[t for d,t in out(x) if d==y];Ax=anc(x)
            for t in ts:
                for r,z in enumerate(Z):
                    for q,(d,tt) in enumerate(sorted([(d,tt) for d,tt in out(z,(t,))],key=lambda u:(abs((dep(u[0])-dep(z))-(dep(y)-dep(x))),u[0]))[:4]):vote(('n',d),lab(d),0.9*ag(d,y)/(1+q)/(1+r)/(1+i),(x,t,y),(z,t,d),'wn')
            if y in Ax and Ax[y][0]>1 or (y in Ax and not ts):
                for r,z in enumerate(Z):
                    Az=anc(z);C2=sorted([d for d in Az if d not in (z,y) and lca(d,y)[0] not in (None,d)],key=lambda d:(abs((dep(d)-dep(lca(d,y)[0]))-(dep(y)-dep(lca(d,y)[0]))),-dep(lca(d,y)[0]),abs((dep(d)-dep(z))-(dep(y)-dep(x)))))[:3]
                    for q,d in enumerate(C2):d!=y and vote(('n',d),lab(d),0.7/(1+q)/(1+r)/(1+i),(x,-Ax[y][0],y),(z,-Az[d][0],d),'up')
            w=_gpat(x,B['t']) if y not in Ax else None
            if w:[vote(('n',g2[0]),g2[1],0.5/(1+r),(x,f"gloss '{w} _'",y),(z,f"gloss '{w} _'",g2[0]),'gloss') for r,z in enumerate(Z) for g2 in [_gapply(z,w)] if g2]
    if not V:return _out('analogy',mode,[],t0,k,x=A['lab'],y=B['lab'],z=Cx['lab'],answer=None)
    R=[]
    for key,sc in sorted(V.items(),key=lambda z:-z[1])[:k]:
        _,lb,src,dst,kd=why[key]
        if kd=='wd':e=[wE(src[0],src[1],src[2]) if src[3]>0 else wE(src[2],src[1],src[0]),wE(dst[0],dst[1],dst[2]) if dst[3]>0 else wE(dst[2],dst[1],dst[0])]
        elif kd=='wn':e=[nE(*src),nE(*dst)]
        elif kd=='up':e=[iE(src[0],src[2]),iE(dst[0],dst[2])]
        elif kd=='ant':e=[E(lab(src[0]),'opposite of',B['t'],'wn',f"WordNet antonym {_g()['nm'][src[0]]}",src[0],src[2] if isinstance(src[2],int) else -1),E(lab(dst[0]),'opposite of',lb,'wn',f"WordNet antonym {_g()['nm'][dst[0]]}",dst[0],-1)]
        else:e=[E(lab(src[0]),'defined as',gloss(src[0]),'gl',f"WordNet gloss {_g()['nm'][src[0]]}",src[0],src[2],line=f"{lab(src[0])}: {gloss(src[0])}."),E(lab(dst[0]),'defined as',gloss(dst[0]),'gl',f"WordNet gloss {_g()['nm'][dst[0]]}",dst[0],dst[2],line=f"{lab(dst[0])}: {gloss(dst[0])}.")]
        R.append(dict(P(e,1.0,kd),ans=lb,vote=round(sc,4)))
    R=_rank(R,mode);R.sort(key=lambda p:-p['vote']);return _out('analogy',mode,R,t0,k,x=A['lab'],y=B['lab'],z=Cx['lab'],answer=R[0]['ans'] if R else None)
def deduce(x,y,rel:str='isa',mode:str='analytical',user=None)->dict:
    t0=time.perf_counter();X,Y=resolve(x,user),resolve(y,user);TN=(1,) if rel=='isa' else (1,4);R=[];yn=set(Y['n'][:1]);yw=set(Y['w'][:1])
    for i in X['n'][:3]:
        A=anc(i,TN);h=[z for z in A if z in yn]
        if h:z=min(h,key=lambda z:A[z][0]);c=chain(A,z);R.append(P([nE(c[q],next(t for d,t in out(c[q]) if d==c[q+1] and t in TN),c[q+1]) for q in range(len(c)-1)] if rel!='isa' else [iE(c[0],z)],1.0,'chain'));break
    for ex in X['w'][:2]:
        A=wup(ex,TX if rel=='isa' else ISA+LOC,6,8);h=[z for z in A if z in yw]
        if h:
            z=min(h,key=lambda z:A[z][0]);c=[z]
            while A[c[-1]][1]>=0:c.append(A[c[-1]][1])
            c=c[::-1];R.append(P([wE(c[q],A[c[q+1]][2],c[q+1]) for q in range(len(c)-1)],1.0,'wdchain'));break
    R=_rank(R,mode);return _out('deduce',mode,R,t0,1,x=X['lab'],y=Y['lab'],holds=bool(R))
def connect(seed,mode:str='creative',user=None,k:int=5,full:bool=False)->dict:
    t0=time.perf_counter();X=resolve(seed,user);sx=(X['n'] or [-1])[0];st=_sg(X['t'].lower());C={};GI=_gi()[0];near=set(anc(sx)) if sx>=0 else set()
    near|={d for d,_ in out(sx)} if sx>=0 else set()
    def add(y,sy,e,v,kind,via):
        v=v*(0.7 if y[:1].isupper() else 1.0);D=sdist(sx,sy);cr=v*band(D);(sy not in near and y!=st and (y not in C or cr>C[y]['cr'])) and C.__setitem__(y,dict(e=e,sp=1.0,kind=kind,v=round(v,4),D=None if D is None else round(D,3),q=round(v,4),cr=round(cr,4),target=y,via=via))
    for j in GI.get(st,np.zeros(0,np.int32))[:300].tolist():_g()['pos'][j]==0 and add(lab(j),j,[E(lab(j),'defined with',X['t'],'gl',f"WordNet gloss {_g()['nm'][j]}",j,sx,line=f"{lab(j)}: {gloss(j)}.")],CF['gl'],'defby',X['t'])
    for i in X['n'][:2]:
        for m in gwords(i)[:12]:
            sp=gspec(m);P_=GI.get(m,np.zeros(0,np.int32))
            if sp<0.45 or m==st or not wsyn(m,0):continue
            for j in P_[:150].tolist():j!=i and _g()['pos'][j]==0 and add(lab(j),j,[E(lab(i),'defined with',m,'gl',f"WordNet gloss {_g()['nm'][i]}",i,j,line=f"{lab(i)}: {gloss(i)}."),E(lab(j),'defined with',m,'gl',f"WordNet gloss {_g()['nm'][j]}",j,i,line=f"{lab(j)}: {gloss(j)}.")],CF['gl']**2*sp,'gloss2',m)
    try:props=[(m,w) for m,w in cnb(X['t'],minc=1000) if wsyn(m)][:8]
    except Exception:props=[]
    for m,wm in props:
        for y,wy in cnb(m,rev=True,minc=1000)[:30]:
            if y==m or y in BSTOP or not wsyn(y,0):continue
            mi=(wsyn(m) or [-1])[0];sy=wsyn(y,0)[0];add(y,sy,[E(X['lab'],'associated with',m,'co','corpus co-occurrence (PMI)',sx,mi,CF['co']*min(1.0,wm/8)),E(y,'associated with',m,'co','corpus co-occurrence (PMI)',sy,mi,CF['co']*min(1.0,wy/8))],CF['co']*min(1.0,wm/8)*CF['co']*min(1.0,wy/8),'assoc',m)
    R=sorted(C.values(),key=lambda p:-p['cr']);U={};R2=[p for p in R if U.__setitem__(p['via'],U.get(p['via'],0)+1) or U[p['via']]<=1];o=_out('connect',mode,R2+[p for p in R if p not in R2],t0,k,x=X['lab']);full and o.update(all=[dict(target=p['target'],via=p['via'],v=p['v'],D=p['D'],cr=p['cr'],kind=p['kind']) for p in R]);return o
CN=Path(os.environ.get('AMNI_CONCEPTNET_DIR',str(CD/'conceptnet')))
CNV=dict(IsA='type of',UsedFor='used for',CapableOf='can',PartOf='part of',HasA='has',MadeOf='made of',AtLocation='found at',Causes='causes',Desires='wants',ReceivesAction='can be',HasProperty='is',Antonym='opposite of',CreatedBy='created by',HasPrerequisite='needs',CausesDesire='makes you want',MotivatedByGoal='done for',DefinedAs='defined as',LocatedNear='near')
@functools.lru_cache(maxsize=1)
def _cn():
    try:L=lambda n:np.load(CN/f"{n}.npy",mmap_mode='r');m=json.loads((CN/'manifest.json').read_text());o=np.asarray(L('n_o')).astype(np.int64).tolist();b=(CN/'n_txt.bin').read_bytes();return dict(**{k:L(k) for k in ('n_h','e_p','e_r','e_d','e_w','r_p','r_e')},T=[b[o[i]:o[i+1]].decode('utf-8','replace') for i in range(len(o)-1)],R=m['filter']['relations'],m=m)
    except Exception:return None
def cnt(i:int)->str:return _cn()['T'][i]
def cnid(t:str)->int:
    C=_cn();k=' '.join(re.sub(r"[^a-z0-9' ]+",' ',str(t).lower().replace('_',' ')).split())
    if C is None or not k:return -1
    h=np.uint64(WF.h64(k));j=int(np.searchsorted(C['n_h'],h));return j if j<len(C['n_h']) and C['n_h'][j]==h else -1
def cnv(t:str)->list:
    t=ART.sub('',str(t).lower().strip(' .?!'));w=t.split();p=' '.join(w[:-1]+[_sg(w[-1])]) if w else t;return [i for i in dict.fromkeys(cnid(x) for x in (t,p,p+'s',p+'es')) if i>=0]
def cne(i:int,R=None,rev:bool=False)->list:
    C=_cn()
    if C is None or i<0:return []
    p=C['r_p' if rev else 'e_p'];a,b=int(p[i]),int(p[i+1]);X=np.asarray(C['r_e'][a:b]).astype(np.int64) if rev else np.arange(a,b,dtype=np.int64);X=X[np.isin(np.asarray(C['e_r'][X]),[C['R'].index(r) for r in R if r in C['R']])] if R is not None else X;O=np.searchsorted(C['e_p'],X,'right')-1 if rev else np.asarray(C['e_d'][X]).astype(np.int64)
    return [(C['R'][int(r)],int(o),float(w)/100) for r,o,w in zip(np.asarray(C['e_r'][X]).tolist(),O.tolist(),np.asarray(C['e_w'][X]).tolist())]
@functools.lru_cache(maxsize=200000)
def _sk(w:str)->str:
    wn=WF._wn();w=w.lower()
    if not wn:return _sg(w)
    return (wn.morphy(w,'v') or wn.morphy(w,'n') or w) if w.endswith(('ing','ed')) else (wn.morphy(w,'n') or wn.morphy(w,'v') or w)
@functools.lru_cache(maxsize=1)
def _cnx()->dict:
    C=_cn();X={}
    if C is None:return X
    R=np.asarray(C['e_r']);keep=np.flatnonzero(np.isin(R,[C['R'].index(r) for r in ('UsedFor','CapableOf','DefinedAs','ReceivesAction','HasProperty','IsA','AtLocation','Desires','HasA','MadeOf','PartOf')]));D=np.asarray(C['e_d'])[keep].tolist();T=C['T']
    for e,d in zip(keep.tolist(),D):
        for w in set(T[d].split()):len(w)>2 and w not in GSTOP and X.setdefault(_sk(w),[]).append(e)
    return {w:np.asarray(v,np.int64) for w,v in X.items()}
def cnsrc(e:int)->int:return int(np.searchsorted(_cn()['e_p'],e,'right')-1)
def cE(i:int,r:str,j:int,w:float=1.0)->dict:a,b=cnt(i),cnt(j);return E(a,CNV.get(r,r),b,'cn',f"ConceptNet 5.7 /r/{r} /c/en/{a.replace(' ','_')} /c/en/{b.replace(' ','_')} (CC BY-SA 4.0)",(wsyn(a) or [-1])[0],(wsyn(b) or [-1])[0],CF['cn']*min(1.0,0.7+0.1*w))
def _cnp(X,Y)->list:
    if _cn() is None:return []
    R=[];hb=lambda m:1/(1+math.log1p((int(_cn()['r_p'][m+1])-int(_cn()['r_p'][m]))/50))
    for a in cnv(X['t'])[:2]:
        Ea=[z for z in cne(a) if z[0]!='IsA' or z[2]>=1.0];Da={}
        [Da.setdefault(j,(r,w)) for r,j,w in Ea]
        for b in cnv(Y['t'])[:2]:
            Eb=[z for z in cne(b) if z[0]!='IsA' or z[2]>=1.0];Db={}
            [Db.setdefault(j,(r,w)) for r,j,w in Eb]
            R+=[P([cE(a,r,b,w)],1.0,'cn1') for r,j,w in Ea if j==b]+[P([cE(b,r,a,w)],1.0,'cn1') for r,j,w in Eb if j==a]+[P([cE(a,Da[m][0],m,Da[m][1]),cE(b,Db[m][0],m,Db[m][1])],hb(m),'cnshare') for m in sorted(set(Da)&set(Db)-{a,b},key=lambda m:-hb(m))[:6]]
    return R
def csyn(c:str)->tuple:
    c=ART.sub('',str(c).lower().strip());return (tuple(wsyn(c,0)[:4]) or tuple(wsyn(c.split()[-1],0)[:3])) if c else ()
@functools.lru_cache(maxsize=16384)
def isa(t:str,c:str)->tuple:
    K=set(csyn(c));t=ART.sub('',str(t).strip(' .,!?').lower());nm=_g()['nm'];cl=_sg(str(c).lower().split()[-1]) if str(c).split() else ''
    if not K or not t:return (None,'')
    N=list(wsyn(t,0)[:4]);hk=lambda Z:next(((z,k) for z in Z for k in K if k in anc(z)),None);h=hk(N)
    if h:return (True,f"WordNet {nm[h[0]]} is a kind of {nm[h[1]]}")
    J=cnv(t)[:2]
    for j in J:
        for r,d,w in cne(j,('IsA',))[:40]:
            x=cnt(d)
            if _sg(x.split()[-1])==cl or hk(wsyn(x,0)[:2]):return (True,f"ConceptNet 5.7 {cnt(j)} IsA {x} (CC BY-SA 4.0)")
    W=went(WF.nrm(t),False)[:2] if S() is not None else ()
    for e in W:
        P_,V_,_,_=wf(e)
        for p,v in zip(P_.tolist(),V_.tolist()):
            x=WF.nrm(wname(v)) if p in (31,279,106,171) else ''
            if x and (_sg(x.split()[-1])==cl or hk(wsyn('person' if x=='human' else x,0)[:2])):return (True,f"Wikidata Q{qid(e)} P{p} Q{qid(v)} ({x})")
    pc=bool(wsyn('person',0)) and any(wsyn('person',0)[0] in anc(k) for k in K)
    return (None,'name shape') if pc and not N and re.fullmatch(r"(?:[a-z][\w'.-]*\s+){1,3}[a-z][\w'.-]*",t) else (False if N or J or W else None,'')
QCL=re.compile(r"(?i)^\W*(?:what|which)\s+(?:kind of |type of |sort of )?((?:[a-z]+\s+)??[a-z]+?)\s+(?:(?:in|of|on|inside|from)\s+(?:the\s+|a\s+|an\s+|your\s+|our\s+)?[a-z]+\s+)?(?:is|are|was|were|do|does|did|can|could|has|have|lies|lie|[a-z]+s)\b")
QOF=re.compile(r"(?i)^\W*(?:what|which)\s+(?:is|are|was)\s+the\s+(?:[a-z]+\s+)?([a-z]+)\s+(?:of|for|in|used)\b")
QSUP=re.compile(r"(?i)\b(?:the\s+)?([a-z]+est|most\s+(?!often\b|commonly\b|likely\b|frequently\b|usually\b|of\b|people\b|popular\b)[a-z]+|innermost|outermost|first|last|main)\s+(?:(?:kind|type|sort|species) of\s+)?((?:(?!(?:of|in|on|that|which|to|for|from|with|by|at|is|are|was|and|or)\b)[a-z]+\s+)?(?!(?:of|in|on|that|which|to|for|from|with|by|at|is|are|was|and|or)\b)[a-z]+)\b")
QGEN=set("thing things one ones kind type sort word words name number year years time way ways part parts percent percentage amount do does did is are was were can could will would should has have had".split())
NOSUP=set("best worst rest west test interest forest honest modest nest chest guest harvest earnest protest request suggest invest arrest pest crest".split())
QEND=re.compile(r"(?i)\b(?:which|what)\s+([a-z]+(?:\s+[a-z]+)?)\s*$")
def qclass(q:str)->str:
    q=re.sub(r"\s+",' ',str(q).lower().strip(' ?.!'));m=QCL.match(q);s=QSUP.search(q);e=QEND.search(q);o=QOF.match(q);z=[x for x in ([m[1]] if m else [])+([s[2]] if s and s[1] not in NOSUP else [])+([e[1]] if e else [])+([o[1]] if o else []) if x]
    for x in z:
        w=[y for y in x.split() if y not in QGEN]
        if not w:continue
        for v in ([' '.join(w[:-1]+[_sg(w[-1])]),_sg(w[-1])] if len(w)>1 else [_sg(w[-1])]):
            if v.split()[-1] not in QGEN and wsyn(v,0):return v
    return 'person' if re.match(r"(?i)\W*who\b",q) else ''
NUMW={w:i for i,w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())};NUMW.update({'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'dozen':12})
ORD={'hundredth':100,'thousandth':1000,'tenth':10,'twelfth':12,'sixtieth':60,'twentieth':20}
NUMR=r"(\d+|%s)"%'|'.join(sorted(NUMW,key=len,reverse=True))
KST=set("what which who whom whose where when how is are was were do does did can could will would should shall may might the a an of in on at to for by with from as that this these those it its their there here most mostly usually often typically commonly generally really very called known name named kind type sort one ones thing things some any our your you we they us them people someone something anything also just only get gets got have has had like likes typical normally".split())
GENV={_x for _x in "use make have get give come go take keep put call find know help need want do see say tell become seem".split()}
VSYN={'eat':('eat','eats','eating','feed on','feeds on','feeding on','feeds','live on','lives on','consume','consumes'),'drink':('drink','drinks','drinking'),'make':('make','makes','making','produce','produces','producing','secrete','secretes','yield','yields','spin','spins'),'produce':('produce','produces','make','makes','secrete','yield'),'build':('build','builds','building','construct','constructs'),'spin':('spin','spins','spinning','weave','weaves','make','makes'),'give':('give','gives','provide','provides','produce','produces','yield','yields','supply'),'lay':('lay','lays','laying'),'store':('store','stores','hoard','hoards','cache','bury','buries','gather','collect'),'collect':('collect','collects','gather','gathers'),'release':('release','releases','give off','gives off','emit','emits','produce','produces'),'grow':('grow into','grows into','turn into','turns into','become','becomes','develop into','develops into'),'turn':('turn into','turns into','become','becomes','grow into','develop into'),'become':('become','becomes','turn into','grow into'),'spray':('spray','sprays'),'breathe':('breathe','breathes','exhale'),'hunt':('hunt','hunts','eat','eats','prey on','preys on'),'catch':('catch','catches','eat','eats')}
PPART={'eat':'eaten','drink':'drunk','make':'made','build':'built','spin':'spun','give':'given','lay':'laid','store':'stored','collect':'collected','release':'released','produce':'produced','write':'written','use':'used','grow':'grown','hunt':'hunted','catch':'caught'}
WDV={'eat':(1034,),'make':(1056,1672),'produce':(1056,1672),'give':(1056,1672),'use':(2283,366)}
def _went(t:str)->list:
    if S() is None:return []
    n={WF.nrm(t),WF.nrm(_sg(t)),WF.nrm(_sg(t)+'s')};return [e for e in went(WF.nrm(_sg(t)),True)[:3] if WF.nrm(wname(e)) in n or n&set(S().names(e)[:30])][:2]
def _cw2(t:str)->list:return [_sk(w) for w in re.findall(r"[a-z]+",str(t).lower()) if w not in KST and w not in GSTOP and len(w)>1]
def _art(a:str)->str:return a if a[:1].isupper() or re.match(r"(?i)(?:the|a|an)\s",a) else ('an ' if a[:1].lower() in 'aeiou' else 'a ')+a
def _plu(w:str)->tuple:
    wn=WF._wn();ex=[k for k,v in (wn._exception_map.get('n',{}).items() if wn and hasattr(wn,'_exception_map') else []) if w in v and k!=w]
    return (ex[0],True) if ex else (w[:-1]+'ies' if re.search(r"[^aeiou]y$",w) else w+'es' if w.endswith(('s','x','z','ch','sh')) else w+'s',False)
def _mass(a:str)->bool:
    Z=wsyn(a,0)[:1];M=set(wsyn('substance',0)[:2])|set(wsyn('matter',0)[:3])|set(wsyn('chemical element',0)[:1])|set(wsyn('food',0)[:2])
    return a[:1].isupper() or bool(Z) and any(k in anc(Z[0]) for k in M) and not a.endswith('s')
def _gls(i:int)->str:return f"WordNet {_g()['nm'][i]}: {gloss(i)}"
def _vote(V:dict,a:str,sc:float,pv:str)->None:
    a=' '.join(a.split()).strip(" ,.;:'\"");k=_sg(a.lower())
    if not a or k in KST or k in GSTOP or len(k)<2:return
    x=V.setdefault(k,{'a':a,'s':0.0,'p':[]});x['s']+=sc;len(x['p'])<3 and pv not in x['p'] and x['p'].append(pv);a!=a.lower() and x['a']==x['a'].lower() and x.update(a=a)
def _np(t:str,k:int=3)->str:
    w=re.findall(r"[a-z][a-z'-]*",t.lower())[:8]
    while w and w[0] in ('a','an','the','some','its','their','his','her','of','other','small','certain','various','many','mostly','mainly','chiefly','especially','usually','only','on'):w=w[1:]
    o=[]
    for z in w:
        if z in GSTOP or z in ('and','or','but','to','that','which','for','with','in','on','at','from','by','as','into','while','when'):break
        o.append(z)
    return ' '.join(o[:k])
def _head(a:str)->str:
    w=a.split();return next((' '.join(w[i:]) for i in range(len(w)) if wsyn(' '.join(w[i:]),0)),'')
def _pick(V:dict,mn:float=1.0)->tuple:
    R=sorted(V.values(),key=lambda x:-x['s']);return (R[0],R[1]['s'] if len(R)>1 else 0.0) if R and R[0]['s']>=mn else (None,0.0)
@functools.lru_cache(maxsize=4096)
def desc(k:int,cap:int=800)->frozenset:
    S_={k};F=[k]
    while F and len(S_)<=cap:F=[d for x in F for d,_ in out(x,(2,)) if d not in S_ and not S_.add(d)]
    return frozenset(S_) if len(S_)<=cap else frozenset()
@functools.lru_cache(maxsize=65536)
def wfreq(i:int)->float:
    wn=WF._wn()
    try:return math.log1p(sum(l.count() for l in wn.synset(_g()['nm'][i]).lemmas())) if wn else 0.0
    except Exception:return 0.0
def _rx(w:str)->re.Pattern:return re.compile(r"\b"+re.escape(w[:-1] if w.endswith('e') and len(w)>4 else w)+r"[a-z]{0,4}\b")
def _base(s:str)->str:
    b=s[:-3] if s.endswith('est') else s;b=b[:-1]+'y' if s.endswith('iest') else b[:-1] if len(b)>2 and b[-1]==b[-2] and b[-1] not in 'ls' else b;wn=WF._wn()
    return b+'e' if wn and not wn.synsets(b,'a') and wn.synsets(b+'e','a') else b
@functools.lru_cache(maxsize=65536)
def qw(w:str)->float:
    wn=WF._wn();n,v=(bool(wn.synsets(w,'n')),bool(wn.synsets(w,'v'))) if wn else (True,False);return gspec(w)*(1.4 if n and not v else 0.7 if v else 1.0)
def _supok(i:int,sp:list,cls:str,G_:str)->bool:
    wn=WF._wn();K=csyn(cls) if cls else ();A={_sg(cls.split()[-1])} if cls else set()
    for k in K:
        try:A|={l.name().lower().replace('_',' ').split()[-1] for s in [wn.synset(_g()['nm'][k])]+wn.synset(_g()['nm'][k]).hypernyms() for l in s.lemmas()}
        except Exception:pass
    for s in sp:
        for m in re.finditer(r"\b"+re.escape(s)+r"\s+((?:[a-z-]+\s+){0,2}?[a-z-]+)",G_.lower()):
            if any(_sg(x) in A for x in m[1].split()) and not re.match(r"\s*(?:in|on|of|off|along)\s+(?:the\s+)?(?!world\b|earth\b|solar system\b|universe\b)[A-Za-z]",G_[m.end():m.end()+40]):return True
    return False
def _incls(i:int,K:set,cls:str)->float:return 1.0 if any(k in anc(i) for k in K) else 0.7 if cls and re.search(r"\b"+re.escape(cls.split()[-1])+r"s?\b",gloss(i).lower()) else 0.0
def _cand_cls(K:set,Q:list,sup:list,V:dict,cls:str='')->None:
    GI=_gi()[0];Qs=[w for w in dict.fromkeys(Q) if w in GI];Z={};Kx=set().union(*[set(anc(k)) for k in K]) if K else set();top=max(Qs,key=qw) if Qs else None;sb=[_base(sup[0])] if sup and sup[0].endswith('est') else []
    for w in Qs+[s for s in sup+sb if s in GI]:
        for i in GI.get(w,np.zeros(0,np.int32))[:4000].tolist():Z.setdefault(i,set()).add(w)
    D=set().union(*[desc(k) for k in K]) if K else set();[Z.setdefault(i,set()) for i in D]
    for i,M in Z.items():
        if _g()['pos'][i]!=0 or i in Kx:continue
        f=_incls(i,K,cls)
        if not f:continue
        G_=gloss(i);g=G_.lower();ms=[w for w in Qs if w in M or _rx(w).search(g)];ss=[s for s in sup if (s in M or re.search(r"\b"+re.escape(s)+r"\b",g)) and _supok(i,[s],cls,G_)];bs=[b for b in sb if not Qs and i in D and re.search(r"\b"+re.escape(b)+r"(?:er|est|ly)?\b",g)];neg=bool(re.search(r"\b(?:one of the|second|third|fourth)\s+(?:\w+\s+)?(?:%s)\b"%'|'.join(map(re.escape,sup+sb)),g)) if sup else False
        if sup and not (ss or bs) or neg or not sup and top not in ms or sup and Qs and not ms:continue
        _vote(V,lab(i),f*(sum(qw(w) for w in ms)+(1.2 if ss else 0.5 if bs else 0.0)+0.05*wfreq(i)),_gls(i))
    for w in Q:
        if gspec(w)<0.45:continue
        for i in wsyn(w,0)[:2]:
            for p,z in enumerate(dict.fromkeys(_sg(x) for x in re.findall(r"[a-z]+",gloss(i).lower()))):
                if z in Q or z in GSTOP or z in BSTOP:continue
                j=next((j for j in wsyn(z,0)[:2] if j!=i and j not in Kx and any(k in anc(j) for k in K)),None)
                j is not None and _vote(V,lab(j),(0.9 if p<4 else 0.4)*gspec(w)+0.3,_gls(i))
        for j in cnv(w)[:1]:
            for r,d,wt in cne(j,('HasProperty','IsA','MadeOf','DefinedAs')):
                x=cnt(d);Zx=wsyn(x,0)[:1];' ' not in x and Zx and Zx[0] not in Kx and any(k in anc(Zx[0]) for k in K) and _vote(V,x,0.5+0.1*min(wt,4),f"ConceptNet 5.7 {cnt(j)} {r} {x} (CC BY-SA 4.0)")
def _cand_cn(K:set,Q:list,V:dict)->None:
    X=_cnx();C=_cn();tq=max(Q,key=qw) if Q else ''
    if not X or C is None or _sk(tq) not in X:return
    Qs=[w for w in dict.fromkeys(Q) if w in X];top=_sk(tq);E_=np.concatenate([X[w] for w in Qs]);u,n=np.unique(E_,return_counts=True);u=u[(n>=2)|np.isin(u,X[top])][:4000];Kx=set().union(*[set(anc(k)) for k in K]) if K else set();B={}
    for e in u.tolist():
        s=cnsrc(e);a=cnt(s);Z=wsyn(a,0)[:2]
        if not Z or Z[0] in Kx or K and not any(k in anc(z) for z in Z for k in K):continue
        d=cnt(int(C['e_d'][e]));r=C['R'][int(C['e_r'][e])];sc=0.7*sum(qw(w) for w in set(_cw2(d))&set(Qs))/max(1e-6,sum(qw(w) for w in Qs))+0.2*min(1.0,float(C['e_w'][e])/200);o=B.get(a);B[a]=(max(o[0],sc) if o else sc,(o[1] if o else 0)+1,o[2] if o and o[0]>=sc else f"ConceptNet 5.7 {a} {r} {d} (CC BY-SA 4.0)")
    [_vote(V,a,s+0.1*min(n_-1,3),p) for a,(s,n_,p) in B.items()]
def _sup_cn(sup:str,cls:str,K:set,V:dict,Q:list=())->None:
    X=_cnx();C=_cn();k=_sk(sup)
    if not X or C is None or k not in X:return
    for e in X[k][:6000].tolist():
        r=C['R'][int(C['e_r'][e])]
        if r not in ('DefinedAs','IsA','HasProperty'):continue
        d=cnt(int(C['e_d'][e]));dw=d.split()
        if sup not in dw or re.search(r"\b(?:one of|second|third|fourth|fifth|not)\b",' '.join(dw[:dw.index(sup)])) or not any(_sg(x)==_sg(cls.split()[-1]) or any(k2 in anc(z) for z in wsyn(x,0)[:1] for k2 in K) for x in dw[dw.index(sup)+1:]):continue
        s=cnt(cnsrc(e));Z=wsyn(s,0)[:2]
        if not Z or K and not any(k2 in anc(z) for z in Z for k2 in K):continue
        _vote(V,s,1.3 if not Q or set(Q)&set(_cw2(d)) else 0.5,f"ConceptNet 5.7 {s} {r} {d} (CC BY-SA 4.0)")
SUPS={'fastest':('fastest','swiftest','quickest'),'largest':('largest','biggest'),'biggest':('biggest','largest'),'tallest':('tallest','highest'),'highest':('highest','tallest'),'smallest':('smallest','tiniest'),'innermost':('innermost','nearest','closest'),'closest':('closest','nearest'),'nearest':('nearest','closest')}
def _sup_wiki(sup:str,cls:str,Q:list,K:set,V:dict)->None:
    R,I=_ix();cw=_sg(cls.split()[-1]) if cls else ''
    if I is None or not cw:return
    pat=re.compile(r"(?i)\b(?:%s)\s+(?:[a-z-]+\s+){0,2}?%s(?:e?s)?\b"%('|'.join(map(re.escape,SUPS.get(sup,(sup,)))),re.escape(cw)))
    try:H=I.search(' '.join([sup,cls]+Q[:3]),int(os.environ.get('AMNI_COG_SUPW_K','20')))
    except Exception:return
    c={};seen=set()
    for rk,(sc,p) in enumerate(H):
        t,x,g=I.text(int(p),c);t=re.sub(r"\s*\([^()]*\)$","",t)
        if g in seen or BADT.search(t):continue
        for z in re.split(r"(?<=[.!?])\s+",x):
            m=pat.search(z);pre=z[:m.start()] if m else '';aft=z[m.end():m.end()+50] if m else '';tk=[w for w in re.findall(r"[a-z]+",t.lower()) if w not in GSTOP];nx=re.match(r"\s+([a-z]+)",aft);sj=re.match(r"\s*((?:[A-Z][\w'-]*\s+){0,3}[A-Z][\w'-]*)(?:\s*\([^)]*\))?\s*,?[^.;]{0,60}?\b(?:is|was|are|remains|became)\b(?:\s+[\w']+){0,3}?\s*$",pre)
            if not m or re.search(r"(?i)\b(?:second|third|fourth|fifth|one of|among|after|than|not|nearly|almost)\b",pre[-30:]) or re.match(r"\s+(?:range\s+|chain\s+)?(?:in|of|on)\s+(?!the world\b|the solar system\b|earth\b|our solar system\b|the universe\b)(?:the\s+)?[A-Z]",aft) or nx and nx[1] not in KST and nx[1] not in GSTOP and nx[1] not in ('by','ever','known','yet','found','recorded','alive','living') and wsyn(nx[1],0) and not WF._wn().synsets(nx[1],'v'):continue
            C_=[a for a in ([t] if tk and all(w in pre.lower() for w in tk[:2]) else [])+([sj[1]] if sj else []) if _sg(a.lower())!=cw]
            a=next((a for a in C_ if isa(a,cls)[0] or ' ' in a and isa(a.split()[-1],cls)[0]),None)
            if a:seen.add(g);_vote(V,a,0.6/(1+0.15*rk),f"Wikipedia: {t}: {z.strip()[:160]}");break
def _count(N:str,Sb:str,V:dict)->None:
    ns=_sg(N);st=ns[:max(3,len(ns)-2)];rx=re.compile(r"\b"+NUMR+r"(?:[\s-]+[a-z]+){0,2}?[\s-]+"+re.escape(st)+r"[a-z]*\b");fr=re.compile(r"(?:1/(\d+)(?:st|nd|rd|th)?|one ("+'|'.join(ORD)+r"))\s+of\s+(?:an?\s+|one\s+)?"+re.escape(_sg(Sb.split()[-1])))
    for lv,i in [(0,i) for i in wsyn(Sb,0)[:3]]+[(1,h) for i in wsyn(Sb,0)[:1] for h,_ in out(i,(1,))[:2]]:
        for m in rx.finditer(gloss(i).lower()):_vote(V,m[1],1.4/(1+lv),_gls(i))
    for i in wsyn(ns,0)[:2]:
        g=gloss(i).lower();m=fr.search(g);m and _vote(V,m[1] or str(ORD[m[2]]),1.4,_gls(i));m=re.match(r"(?:either|each|one) of (?:the )?"+NUMR+r"\b",g);m and _vote(V,m[1],1.2,_gls(i))
    J=cnv(Sb)[:2];H=[d for j in J for r,d,w in cne(j,('IsA',))[:6]]
    for lv,j in [(0,j) for j in J]+[(1,h) for h in H]:
        for r,d,w in cne(j,('HasA','PartOf','DefinedAs','HasProperty')):
            m=rx.search(cnt(d));m and _vote(V,m[1],(1.0+0.1*w)/(1+lv),f"ConceptNet 5.7 {cnt(j)} {r} {cnt(d)} (CC BY-SA 4.0)")
    R_,I=_ix() if not V else (None,None)
    if I is None:return
    c={};n=0
    for g in list(_tdocs(R_,I,Sb))[:1]:
        for p in range(int(I.first[g]),min(int(I.first[g+1]),int(I.first[g])+int(os.environ.get('AMNI_COG_COUNT_PASS','40')))):
            t,x,_=I.text(p,c)
            for z in re.split(r"(?<=[.!?])\s+",x):
                m=rx.search(z.lower())
                if m and n<4 and re.search(r"\b(?:has|have|had|with|its|their|possess|possesses|into|of|comprises|comprise|contains|contain)\s+(?:[a-z-]+\s+){0,2}$",z.lower()[:m.start()][-40:]) and not re.search(r"\b(?:up to|over|more than|about|around|some|only|species|genera|million|thousand)\s+$",z.lower()[:m.start()][-16:]):_vote(V,m[1],0.6,f"Wikipedia: {t}: {z.strip()[:160]}");n+=1
def _objs(Sb:str,v:str,pr:str,V:dict)->None:
    Vf=VSYN.get(v,(v,_s3(v),v+'ing'));Vf=tuple(dict.fromkeys(f"{x} {pr}" if pr and pr not in x.split() else x for x in Vf));rx=re.compile(r"\b(?:%s)\s+([a-z][a-z' -]+)"%'|'.join(re.escape(x) for x in Vf));sl=_sg(Sb.split()[-1]);J=cnv(Sb)[:2]
    for j in J:
        for r,d,w in cne(j,('CapableOf','Desires','UsedFor','HasA','MadeOf','CreatedBy','HasProperty')):
            m=rx.match(cnt(d));a=_np(m[1]) if m else ''
            a and _head(a) and _vote(V,a,1.2+0.15*min(w,4),f"ConceptNet 5.7 {cnt(j)} {r} {cnt(d)} (CC BY-SA 4.0)")
    pp=tuple(dict.fromkeys([PPART.get(v,v+'ed')]+[PPART[x.split()[0]] for x in Vf if x.split()[0] in PPART]));rr=re.compile(r"^(?:%s)\s+by\s+(?:[a-z]+\s+){0,2}?(?:%s)\b"%('|'.join(pp),re.escape(sl)));X=_cnx();C=_cn()
    for e in (X.get(_sk(sl),np.zeros(0,np.int64)) if X else np.zeros(0,np.int64))[:8000].tolist():
        if C['R'][int(C['e_r'][e])]!='ReceivesAction':continue
        d=cnt(int(C['e_d'][e]));rr.match(d) and _vote(V,cnt(cnsrc(e)),1.2,f"ConceptNet 5.7 {cnt(cnsrc(e))} ReceivesAction {d} (CC BY-SA 4.0)")
    for i in wsyn(Sb,0)[:2]:
        m=rx.search(gloss(i).lower());a=_np(m[1]) if m else ''
        a and _head(a) and _vote(V,a,1.3,_gls(i))
    GI=_gi()[0];ps=re.compile(r"\b(?:%s)\s+by\s+(?:[a-z]+\s+){0,2}?%s"%('|'.join(pp),re.escape(sl)))
    for i in GI.get(sl,np.zeros(0,np.int32))[:4000].tolist():
        _g()['pos'][i]==0 and ps.search(gloss(i).lower()) and _vote(V,lab(i),1.3,_gls(i))
    for e in (_went(Sb) if WDV.get(v) else ()):
        P_,V_,_,_=wf(e)
        for p,x in zip(P_.tolist(),V_.tolist()):p in WDV[v] and _vote(V,wname(x),1.6 if wsyn(WF.nrm(wname(x)),0) or cnv(wname(x)) else 0.3,f"Wikidata Q{qid(e)} P{p} Q{qid(x)} ({wname(x)})")
    if V:return False
    for i in wsyn(Sb,0)[:1]:
        for h,(lv,_) in sorted(anc(i).items(),key=lambda z:z[1][0]):
            if not 0<lv<=7:continue
            U=[k for k in cnv(lab(h))[:1]]+[k for c,_ in out(h,(2,))[:30] for k in cnv(lab(c))[:1]]+[k for c,_ in out(h,(2,))[:12] for c2,_ in out(c,(2,))[:8] for k in cnv(lab(c2))[:1]];sn=set()
            for j in dict.fromkeys(U):
                for r,d,w in cne(j,('CapableOf','Desires')):
                    m=rx.match(cnt(d));a=_np(m[1]) if m else ''
                    a and _head(a) and (j,a) not in sn and not sn.add((j,a)) and _vote(V,a,0.3/(1+0.25*lv),f"ConceptNet 5.7 {cnt(j)} {r} {cnt(d)} (CC BY-SA 4.0); WordNet {_g()['nm'][i]} is a kind of {_g()['nm'][h]}")
    return True
def _made(Sb:str,V:dict)->None:
    rx=re.compile(r"\b(?:made (?:from|of|with|by \w+ing)|prepared from|derived from|juice of|juice \(?of|consisting (?:mostly |mainly )?of|dried|pulp of|obtained from|extracted from)\s+(?:the\s+)?(?:(?:\w+ed|ground|roasted|mashed|pressed|fine|certain|fresh)\s+){0,2}([a-z][a-z -]+)")
    for i in wsyn(Sb,0)[:2]:
        for m in rx.finditer(gloss(i).lower()):
            a=_np(m[1],2);a and _head(a) and _vote(V,a,1.3,_gls(i))
    for j in cnv(Sb)[:2]:
        for r,d,w in cne(j,('MadeOf','ReceivesAction')):
            x=cnt(d);m=re.match(r"(?:made (?:from|of|with) )?([a-z][a-z -]+)",x) if r=='MadeOf' else re.match(r"made (?:from|of|with|out of)\s+([a-z][a-z -]+)",x);a=_np(m[1],2) if m else ''
            a and _vote(V,a,1.1+0.1*w,f"ConceptNet 5.7 {cnt(j)} {r} {x} (CC BY-SA 4.0)")
    for e in _went(Sb):
        P_,V_,_,_=wf(e)
        for p,x in zip(P_.tolist(),V_.tolist()):p==186 and _vote(V,wname(x),1.5,f"Wikidata Q{qid(e)} P186 Q{qid(x)} ({wname(x)})")
def _where(Sb:str,V:dict)->None:
    for j in cnv(Sb)[:2]:
        for r,d,w in cne(j,('AtLocation',)):_vote(V,cnt(d),0.4+0.15*min(w,5),f"ConceptNet 5.7 {cnt(j)} AtLocation {cnt(d)} (CC BY-SA 4.0)")
    L=set(wsyn('location',0)[:1])|set(wsyn('region',0)[:2])|set(wsyn('geographical area',0)[:1])|set(wsyn('habitat',0)[:1])|set(wsyn('body of water',0)[:1])
    for i in wsyn(Sb,0)[:2]:
        for m in re.finditer(r"\b(?:of|in|from|on|off)\s+(?:the\s+)?((?:[A-Z][a-z]+\s*)+|[a-z]+(?:\s+[a-z]+)?)",gloss(i)):
            a=m[1].strip();Z=wsyn(a,0)[:2]
            Z and (a[:1].isupper() or any(k in anc(z) for z in Z for k in L)) and _vote(V,a,1.3,_gls(i))
    for e in _went(Sb):
        P_,V_,_,_=wf(e)
        for p,x in zip(P_.tolist(),V_.tolist()):p in (2974,183) and _vote(V,wname(x),1.4,f"Wikidata Q{qid(e)} P{p} Q{qid(x)} ({wname(x)})")
def _lex(q:str)->dict:
    wn=WF._wn();m=re.search(r"\bopposite of (?:the word )?['\"]?([a-z]+)",q);V={}
    if m and wn:
        x=m[1];[_vote(V,a.name().replace('_',' '),2.0 if s.pos() in 'as' else 1.0,f"WordNet {s.name()} antonym {a.synset().name()}") for s in wn.synsets(x) for l in s.lemmas() if l.name().lower()==x for a in l.antonyms()];[_vote(V,cnt(d),0.8,f"ConceptNet 5.7 {x} Antonym {cnt(d)} (CC BY-SA 4.0)") for j in cnv(x)[:1] for r,d,w in cne(j,('Antonym',)) if ' ' not in cnt(d)]
        b,s2=_pick(V,0.8);return dict(answer=b['a'],text=f"The opposite of {x} is {b['a']}.",conf=min(0.95,0.6+0.1*(b['s']-s2)),prov=b['p'],kind='antonym') if b else None
    m=re.search(r"\b(?:same as|synonym (?:of|for)|another word for)\s+['\"]?([a-z]+)",q)
    if m and wn:
        x=m[1];[_vote(V,l.name().replace('_',' '),1.0/(1+k),f"WordNet {s.name()} synonym") for k,s in enumerate(wn.synsets(x)[:3]) for l in s.lemmas() if l.name().lower()!=x and '_' not in l.name()];b,_=_pick(V,0.9);return dict(answer=b['a'],text=f"{b['a'].capitalize()} means about the same as {x}.",conf=0.75,prov=b['p'],kind='synonym') if b else None
    m=re.search(r"\bplural of ['\"]?([a-z]+)",q)
    if m and wn:x=m[1];p,ex=_plu(x);return dict(answer=p,text=f"The plural of {x} is {p}.",conf=0.9 if ex else 0.7,prov=[f"WordNet noun exception list {p} -> {x}" if ex else "English plural rule"],kind='plural')
    m=re.search(r"\b(?:baby|young)\s+([a-z]+)\s+(?:is\s+)?called\b|\bcalled\b.*\b(?:baby|young)\s+([a-z]+)",q)
    if m:
        x=_sg(m[1] or m[2]);H=[x]+[lab(h).lower() for i in wsyn(x,0)[:1] for h,_ in out(i,(1,))[:2]];GI=_gi()[0]
        for i in GI.get('young',np.zeros(0,np.int32)).tolist():
            g=gloss(i).lower();_g()['pos'][i]==0 and any(re.match(r"(?:an? |the )?(?:young|baby)\s+(?:[a-z]+\s+)?(?:of\s+)?(?:an? |the )?(?:domestic\s+)?"+re.escape(h)+r"\b",g) for h in H) and _vote(V,lab(i),1.5 if re.search(r"\b"+re.escape(x)+r"\b",g) else 1.0,_gls(i))
        b,_=_pick(V,1.0);return dict(answer=b['a'],text=f"A baby {x} is called {_art(b['a'])}.",conf=0.8,prov=b['p'],kind='young') if b else None
    m=re.search(r"\bgroup of ([a-z]+)\s+(?:is\s+)?called\b",q)
    if m:
        x=m[1];GI=_gi()[0]
        for i in GI.get(_sg(x),np.zeros(0,np.int32)).tolist():_g()['pos'][i]==0 and re.match(r"(?:an? |the )?(?:large |small )?(?:group|collection|company|flock|herd) of (?:[a-z]+\s+){0,2}"+re.escape(x)+r"\b",gloss(i).lower()) and _vote(V,lab(i),1.2,_gls(i))
        b,_=_pick(V,1.0);return dict(answer=b['a'],text=f"A group of {x} is called {_art(b['a'])}.",conf=0.75,prov=b['p'],kind='group') if b else None
    m=re.search(r"\bwhat (?:sound|noise) (?:does|do) (?:an? |the )?([a-z]+) make\b",q)
    if m:
        x=_sg(m[1]);GI=_gi()[0]
        for i in GI.get(x,np.zeros(0,np.int32)).tolist():_g()['pos'][i]==0 and re.search(r"\b(?:sound|noise|cry|call)\b[^;]*\b"+re.escape(x)+r"\b",gloss(i).lower()) and _vote(V,lab(i),1.2,_gls(i))
        b,_=_pick(V,1.0);return dict(answer=b['a'],text=f"{_art(x).capitalize()} makes a {b['a']} sound.",conf=0.75,prov=b['p'],kind='sound') if b else None
    return None
def _s3(v:str)->str:return v+'es' if v.endswith(('s','sh','ch','x','o','z')) else v[:-1]+'ies' if re.search(r"[^aeiou]y$",v) else v+'s'
def know(q:str)->dict:
    t0=time.perf_counter();qc=re.sub(r"\s+",' ',re.sub(r"[?!.]+$",'',str(q or '').strip())).strip();qc=re.sub(r"(?i)^(?:name|tell me|give me)\s+(?:the\s+)?",'what is the ',qc);qc=re.sub(r"(?i)^(.+?)\s+(?:is|are)\s+called\s+what$",r"what is \1 called",qc);qc=re.sub(r"(?i)^what (?:is|are) the (?:main|primary|chief|basic) ingredients? (?:in|of) (?:an? |the )?(.+)$",r"what is \1 made from",qc);qc=re.sub(r"(?i)^what do (?:we|you|people) call (?:an? |the )?([a-z]+) (?:who|that|which) ",r"which \1 ",qc);ql=qc.lower()
    fin=lambda r:dict(r,q=str(q),ms=round((time.perf_counter()-t0)*1000,1),proper=bool(r.get('answer')) and str(r.get('text','')).startswith(str(r.get('answer'))) and str(r.get('answer'))[:1].isupper()) if r else None
    try:
        r=_lex(ql)
        if r:return fin(r)
        V={};m=re.match(r"^(?:how many (?P<n>[a-z]+(?: [a-z]+)?) (?:does|do|did|can|will) (?:an? |the |most |typical |one )*(?P<s>[a-z ]+?) (?:usually |normally |typically )?(?:have|has|get)|how many (?P<n2>[a-z]+) (?:are|is) (?:there )?in (?:an? |one |the )?(?P<s2>[a-z ]+?)|how many (?P<n4>[a-z]+) make up (?:an? |one |the )?(?P<s4>[a-z ]+?)|(?:an? |the )?(?P<s3>[a-z ]+?) (?:has|have|lasts|is divided into|contains) how many (?P<n3>[a-z]+))(?: (?:on|in|at|with) .+)?$",ql)
        if m:
            N,Sb=(m['n'] or m['n2'] or m['n3'] or m['n4']).split()[-1],ART.sub('',(m['s'] or m['s2'] or m['s3'] or m['s4']).strip());_count(N,Sb,V);not V and ' ' in Sb and _count(N,Sb.split()[-1],V);b,s2=_pick(V,1.0)
            return fin(dict(answer=b['a'],text=f"There are {b['a']} {N} in {_art(Sb)}." if m['n2'] or m['n4'] else f"{_art(Sb)[:1].upper()+_art(Sb)[1:]} has {b['a']} {N}.",conf=min(0.9,0.55+0.15*(b['s']-s2)),prov=b['p'],kind='count')) if b else None
        m=re.match(r"^what (?:is|are) (?:an? |the )?(?P<s>[a-z ]+?) (?:mainly |mostly |usually |generally |typically )?(?:made (?:from|of|out of)|produced from)$",ql) or re.match(r"^(?:what|which) (?P<c>[a-z]+) (?:is|are) (?:an? |the )?(?P<s>[a-z ]+?) (?:mainly |mostly |usually )?(?:made (?:from|of|out of)|produced from)$",ql) or re.match(r"^(?:an? |the )?(?P<s>[a-z ]+?) (?:is|are) (?:mainly |mostly |usually |generally )?(?:made (?:from|of|out of)|produced from) (?:which|what) (?P<c>[a-z]+)$",ql)
        if m:
            Sb=m['s'];_made(Sb,V);mc=m.groupdict().get('c') or '';[V.pop(k) for k in [k for k,x in V.items() if mc and isa(x['a'],mc)[0] is False]];b,s2=_pick(V,1.0);pl=Sb.endswith('s') and not Sb.endswith(('ss','us','is'))
            return fin(dict(answer=b['a'],text=f"{Sb[:1].upper()+Sb[1:]} {'are' if pl else 'is'} made from {b['a']}.",conf=min(0.9,0.55+0.15*(b['s']-s2)),prov=b['p'],kind='made')) if b else None
        m=re.match(r"^where (?:do|does|did) (?:an? |the |wild )?(?P<s>[a-z ]+?) (?:live|come from|grow|nest)(?: in the wild)?$|^where (?:is|are) (?:an? |the )?(?P<s2>[a-z ]+?)(?: found| located)?$|^where would you find (?:an? |the |wild )?(?P<s3>[a-z ]+?)(?: living)?$",ql)
        if m:
            Sb=m['s'] or m['s2'] or m['s3'];_where(Sb,V);b,s2=_pick(V,1.2);pl=Sb.endswith('s') and not Sb.endswith(('ss','us','is'));A=b['a'] if b else '';A=A if A[:1].isupper() else 'the '+A;su=Sb[:1].upper()+Sb[1:] if pl else 'The '+Sb
            return fin(dict(answer=b['a'],text=f"{su} {('live' if pl else 'lives') if not m['s2'] else ('are' if pl else 'is')} in {A}.",conf=min(0.85,(0.62 if any(x.startswith(('WordNet','Wikidata')) for x in b['p']) else 0.5)+0.12*(b['s']-s2)),prov=b['p'],kind='where')) if b else None
        m=re.match(r"^what (?:is|are) (?:an? |the )?(?P<x>[a-z ]+?) called$",ql)
        if m:
            Q=[w for w in re.findall(r"[a-z]+",m['x']) if w not in KST and w not in GSTOP];GI=_gi()[0];Z=[set(GI.get(w,np.zeros(0,np.int32)).tolist())|set(GI.get(_sg(w),np.zeros(0,np.int32)).tolist()) for w in Q if w in GI or _sg(w) in GI]
            for i in (set.intersection(*Z) if Z and len(Z)==len(Q) else set()):_g()['pos'][i]==0 and _vote(V,lab(i),1.0+2.0/(1+len(gloss(i).split()))+0.05*wfreq(i),_gls(i))
            b,s2=_pick(V,1.0);return fin(dict(answer=b['a'],text=f"{m['x'][:1].upper()+m['x'][1:]} {'are' if m['x'].endswith('s') else 'is'} called {b['a'] if _mass(b['a']) else _art(b['a'])}.",conf=min(0.85,0.55+0.2*(b['s']-s2)),prov=b['p'],kind='define')) if b else None
        cls=qclass(ql);K=set(csyn(cls)) if cls else set();sm=QSUP.search(ql) or re.search(r"\bthe\s+([a-z]+est)\s*$",ql);sup=sm[1] if sm and sm[1] not in NOSUP else ''
        m=re.match(r"^what (?:do|does|did) (?:an? |the |most )?(?P<s>[a-z ]+?) (?:mostly |mainly |usually |often |normally |typically |like to |love to |need to )*(?P<v>[a-z]+)(?: (?P<r>.+))?$",ql)
        if m and not cls and m['v'] not in ('have','do','look','mean','say','call','cost','weigh','use','like','need','want','know','think'):
            Sb=m['s'];v=WF.lem(m['v']);pr=(m['r'] or '').split()[0] if (m['r'] or '').split()[:1] in (['into'],['on'],['up'],['out']) else '';fb=_objs(Sb,v,pr,V);b,s2=_pick(V,0.45 if fb else 1.0)
            if not b:return None
            a=b['a'];pl=Sb.endswith('s') and not Sb.endswith(('ss','us','is'));vv=m['v'] if pl else _s3(m['v']);ob=_plu(a)[0] if pl and ' ' not in a and not a.endswith('s') and v in ('grow','turn','become','lay','build','spin') else a
            return fin(dict(answer=a,text=f"{Sb[:1].upper()+Sb[1:] if pl else _art(Sb)[:1].upper()+_art(Sb)[1:]} {vv}{' '+pr if pr else ''} {ob}.",conf=min(0.5 if fb else 0.9,0.5+0.15*(b['s']-s2)),prov=b['p'],kind='object'+('~' if fb else '')))
        if not K or re.match(r"\W*who\b",ql) or re.search(r"\b(?:mix|mixing|mixed|combine|combining|add|adding|plus|minus|times)\b",ql):return None
        Q=[w for w in _cw2(ql) if w not in _cw2(cls) and w!=_sk(sup) and not w.endswith('est') and w not in GENV];ew={_sg(w) for w in re.findall(r"[a-z]+",ql) if w not in KST and w not in GSTOP}-{_sg(x) for x in cls.split()}
        sq=re.search(r"\b(after|before|following|preceding)\s+([a-z]+)",ql)
        if sq:
            y=sq[2];[_vote(V,lab(i),1.2,_gls(i)) for k in K for i in desc(k) if re.search(r"\b(?:%s)\s+%s\b"%('following|after' if sq[1] in ('after','following') else 'preceding|before',re.escape(y)),gloss(i).lower())]
        elif sup:
            [_sup_cn(sp,cls,K,V,[w for w in Q if w not in ('world','earth','solar','system','live','exist')]) for sp in SUPS.get(sup,(sup,))];_cand_cls(K,[w for w in Q if w not in ('world','earth','solar','system','planet')],list(SUPS.get(sup,(sup,))),V,cls);_sup_wiki(sup,cls,[w for w in Q if w not in ('world','earth')],K,V)
        else:_cand_cls(K,Q,[],V,cls);_cand_cn(K,Q,V)
        [V.pop(k) for k in [k for k,x in V.items() if {_sg(w) for w in re.findall(r"[a-z]+",x['a'].lower())}&ew]]
        b,s2=_pick(V,float(os.environ.get('AMNI_COG_KNOW_MIN','0.8' if sup else '1.0')));R_=sorted(V.values(),key=lambda x:-x['s']);z1,z2=(wsyn(R_[0]['a'],0)[:1],wsyn(R_[1]['a'],0)[:1]) if len(R_)>1 else ([],[])
        if b and z1 and z2 and z2[0]!=z1[0] and z2[0] in anc(z1[0]) and anc(z1[0])[z2[0]][0]<=2 and R_[1]['s']>=0.5*R_[0]['s'] and re.search(r"\b"+re.escape(R_[1]['a'].lower()),gloss(z1[0]).lower()):b,s2=R_[1],0.5*R_[1]['s']
        if not b:return None
        ok=isa(b['a'],cls);a=b['a'];z=wsyn(a,0)[:1];a=lab(z[0]) if z and lab(z[0]).lower()==a.lower() else a
        if ok[0] is False:return None
        an=a if _mass(a) else 'the '+a;An=an[:1].upper()+an[1:];m1=re.match(r"(?i)^(?:what|which) (?:kind of |type of )?(?:[a-z]+ )?"+re.escape(cls.split()[-1])+r"s?\s+(?!(?:do|does|did|is|are|was|were)\b)(?P<r>.+)$",qc);m2=re.match(r"(?i)^(?:what|which) (?:is|are|was) the (?P<r>.+)$",qc);m3=re.match(r"(?i)^(?:what|which) [a-z]+ (?:is|are|do|does) (?P<x>.+?) (?P<v>made (?:from|of)|come from|comes from)$",qc);m4=re.match(r"(?i)^(?:what|which) (?:[a-z]+ )?"+re.escape(cls.split()[-1])+r"s? (?P<v>is|are|was) (?P<r>.+)$",qc)
        txt=f"{An} {m1['r']}." if m1 else f"{An} is the {m2['r']}." if m2 and sup else f"{m3['x'][:1].upper()+m3['x'][1:]} {'are ' if m3['v'].startswith('made') and m3['x'].endswith('s') else 'is ' if m3['v'].startswith('made') else ''}{m3['v']} {_plu(a)[0] if not _mass(a) and m3['x'].endswith('s') and ' ' not in a else an}." if m3 else f"{An} {m4['v']} {m4['r']}." if m4 else f"It's {an}."
        return fin(dict(answer=a,text=txt,conf=min(0.9,0.5+0.12*(b['s']-s2)+(0.1 if ok[0] else 0)),prov=b['p']+([ok[1]] if ok[1] else []),kind='sup' if sup else 'class',cls=cls))
    except Exception as e:return fin(dict(err=type(e).__name__,msg=str(e)[:200])) if os.environ.get('AMNI_COG_DEBUG')=='1' else None
CUES=[(r"\b(?:poems?|poetry|haikus?|limericks?|sonnets?|stor(?:y|ies)|tales?|fables?|lyrics|song about|metaphors?|similes?|imagine|imagining|pretend|what if|what would happen if|brainstorm|invent|make up|dream up|come up with|creative(?:ly)?|whimsical|playful|surreal|fiction(?:al)?|riddles?|slogans?|puns?|jokes?|myths?|personif\w+|as if (?:it|they|you) were|write (?:me )?(?:a|an|some)|fantasy|daydream)\b",'creative',2.0),
 (r"\b(?:how (?:are|is|were|was|do|does|did) .+? (?:connected|related|linked|tied|alike|similar|different)|connection|relationship|relation between|link between|what links|what connects|in common|similarit(?:y|ies)|differences?|compare|comparison|contrast|versus|vs\.?|analog(?:y|ous)|is to .+? as|why (?:do|does|did|is|are|was|were|would|can)|explain|reason(?:s|ing)?|cause[sd]?|because|implication|consequence|deduce|infer|therefore|categor(?:y|ize)|classify|what (?:kind|type|sort) of (?:thing|animal|object|category|concept)|example of|broader|generali[sz]e|abstract|pattern|trade-?offs?|pros and cons|advantages?|what makes|how (?:does|do|did|can|could|would|will) .+? (?:affect|influence|impact|shape|change|lead to|contribute to)|(?:is|are) (?:an? |the )?[\w-]+ (?:an? )?(?:kind|type|sort|form|member) of)\b",'analytical',1.5),
 (r"^\W*(?:who|when|where|which|whose|how (?:many|much|old|tall|long|far|big|high|deep|fast))\b|\b(?:capital of|population of|born|died|founded|invented by|discovered by|date of|what year|stand for|spell|how do you spell|atomic number|currency of|president of|author of|who wrote|who painted|who directed|what is the (?:name|capital|population|currency|height|length|area))\b",'factual',1.2),
 (r"^\W*what (?:is|are|was|were) (?:the |an? )?[\w' -]+\W*$|^\W*(?:define|definition of|meaning of)\b",'factual',0.8)]
CUES=[(re.compile(a,re.I),m,w) for a,m,w in CUES]
OPS=[(r"^(?:if\s+)?(?P<a>[\w' -]+?)\s+is\s+to\s+(?:an?\s+|the\s+)?(?P<b>[\w' -]+?)\s*,?\s*(?:as|then)\s+(?:an?\s+|the\s+)?(?P<c>[\w' -]+?)\s+is\s+to\b",'analogy'),
 (r"(?P<a>[\w' .-]+?)\s*:\s*(?P<b>[\w' .-]+?)\s*::\s*(?P<c>[\w' .-]+?)\s*:",'analogy'),
 (r"what (?:do|does|did) (?P<a>.+?) and (?P<b>.+?) (?:have in common|share|both have)\b",'common'),
 (r"(?:similarit(?:y|ies)|common ground|things? in common|what(?:'s| is) common) (?:between|to|of) (?P<a>.+?) and (?P<b>.+?)$",'common'),
 (r"how (?:are|is|were|was) (?P<a>.+?) and (?P<b>.+?) (?:alike|similar)",'common'),
 (r"how (?:are|is|were|was|do|does|did) (?P<a>.+?) (?:and|&) (?P<b>.+?) (?:connected|related|linked|tied|associated|connect|relate|link)\b",'relate'),
 (r"(?:connection|relationship|relation|link|tie|bond)s? between (?P<a>.+?) and (?P<b>.+?)$",'relate'),
 (r"(?:what|how) (?:connects|links|ties|relates) (?P<a>.+?) (?:and|to|with) (?P<b>.+?)$",'relate'),
 (r"how (?:is|are|was|were) (?P<a>.+?) (?:connected|related|linked|tied) (?:to|with) (?P<b>.+?)$",'relate'),
 (r"(?:^|\b)connect (?:the )?(?P<a>.+?) (?:and|to|with) (?P<b>.+?)$",'relate'),
 (r"^(?:is|are|was|were) (?:an? |the )?(?P<a>[\w' -]+?) (?:an? |a kind of |a type of |a sort of |a form of |kinds of |types of )(?P<b>[\w' -]+?)$",'deduce'),
 (r"^(?:is|are) (?:an? |the )?(?P<a>[\w' -]+?) (?:in|part of|located in|inside) (?:an? |the )?(?P<b>[\w' -]+?)$",'deduce_in'),
 (r"what (?:kind|type|sort) of (?:thing|object|entity|animal|creature|plant|place|food|concept|category|item)?\s*(?:is|are|was) (?:an? |the )?(?P<a>.+?)$",'abstract'),
 (r"what (?:is|are|was) (?:an? |the )?(?P<a>.+?) an? (?:example|instance|kind|type|sort|form) of",'abstract'),
 (r"what (?:broader |larger |general )?(?:category|class|group|family|kind of thing) (?:does|do|is|are) (?:an? |the )?(?P<a>.+?) (?:belong to|fall under|fit into|in|part of)",'abstract'),
 (r"(?:generali[sz]e|zoom out (?:from|on)|(?:broader|more general|higher) (?:category|concept|term|class) (?:for|of|than)) (?:an? |the )?(?P<a>.+?)$",'abstract'),
 (r"(?:metaphors?|similes?|analog(?:y|ies)|brainstorm(?:ing)?(?: ideas)?|associations?|poems?|stor(?:y|ies)|haikus?|songs?|riddles?) (?:for|about|on|with|around|involving|linking|connecting) (?:an? |the )?(?P<a>[\w' -]+?)(?:\s+(?:and|with|to)\s+(?:an? |the )?(?P<b>[\w' -]+?))?$",'connect'),
 (r"(?:brainstorm|associate|riff on|free-?associate(?: on)?) (?:with |on |about )?(?:an? |the )?(?P<a>[\w' -]+?)$",'connect')]
OPS=[(re.compile(a,re.I),o) for a,o in OPS]
def mode(message:str)->tuple:
    m=str(message or '');sc={'factual':0.3,'analytical':0.0,'creative':0.0};op=parse(m)[0];op and sc.__setitem__('creative' if op=='connect' else 'analytical',sc['creative' if op=='connect' else 'analytical']+1.5)
    for rx,md,w in CUES:sc[md]+=w*min(2,len(rx.findall(m)))
    b=max(sc,key=lambda z:(sc[z],z=='factual'));return b,round(sc[b]/(sum(sc.values()) or 1),3)
def parse(message:str)->tuple:
    m=re.sub(r"\s+",' ',str(message or '').strip()).rstrip(' ?.!');m=re.sub(r"(?i)^(?:hey|hi|so|ok(?:ay)?|adam|please|quick question)[,:\s]+|^(?:can|could|would) you (?:please )?(?:tell me|explain|show me|help me (?:see|understand))\s*|^(?:tell me|explain|show me|i wonder|i'm curious)\s*",'',m);m=re.sub(r"(?i)^(?:write|give|make|compose|craft)(?: me)? (?:an? |some |three |five |a few )?(?:short |little |quick |fun )?",'',m)
    for rx,op in OPS:
        g=rx.search(m)
        if g:d={k:re.sub(r"(?i)^(?:the|an?)\s+",'',v.strip(" ,'\"")) for k,v in g.groupdict().items() if v};return (op,d) if all(len(v.split())<=5 for v in d.values()) else (None,{})
    return None,{}
def run(message:str,user=None,mode_:str='')->dict:
    t0=time.perf_counter();md,cf=mode(message);md=mode_ or md;op,a=parse(message)
    o=(relate(a['a'],a['b'],md,user) if op=='relate' else common(a['a'],a['b'],md,user) if op=='common' else analogy(a['a'],a['b'],a['c'],md,user) if op=='analogy' else abstract(a['a'],md,user) if op=='abstract' else deduce(a['a'],a['b'],'isa' if op=='deduce' else 'in',md,user) if op in ('deduce','deduce_in') else (relate(a['a'],a['b'],'creative',user) if a.get('b') else connect(a['a'],'creative',user)) if op=='connect' else dict(op=None,ok=False,paths=[],lines=[]))
    return {**o,'mode':md,'mconf':cf,'args':a,'ms':round((time.perf_counter()-t0)*1000,2)}
def context(message:str,history=None,user=None,budget:int=0)->dict:
    if not enabled():return dict(mode=None,op=None,lines=[],ms=0.0)
    try:o=run(message,user)
    except Exception as e:return dict(mode=mode(message)[0],op=None,lines=[],ms=0.0,err=type(e).__name__)
    ac=__import__('amni.serve.adam_t_client',fromlist=['asc']);L=[ac.clip(ac.asc(x),int(os.environ.get('AMNI_COG_LINE_BYTES','200'))) for x in o['lines']];n=0;K=[]
    for x in L:
        if budget>0 and n+len(x)+1>budget:break
        K.append(x);n+=len(x)+1
    return dict(mode=o['mode'],op=o.get('op'),lines=K,ms=o['ms'],answer=o.get('answer'),holds=o.get('holds'))
def warm()->None:
    try:[f() for f in (_g,_co,_ix,WF.store,WF._wn,_gi,_cn,_cnx)];relate('bee','almond')
    except Exception:pass
enabled() and os.environ.get('AMNI_COG_WARM','1')=='1' and threading.Thread(target=warm,daemon=True,name='cognition-warm').start()
if __name__=='__main__':
    t=time.perf_counter();warm();print(f"warm {time.perf_counter()-t:.2f}s",file=sys.stderr);[print(json.dumps({k:v for k,v in run(q).items() if k!='paths'},default=str)) for q in (sys.argv[1:] or ["How are bees and almonds connected?"])]
