import os,re,json,math,time,functools
import numpy as np
from pathlib import Path
from amni.compute import wikifact as WF,store1 as D1
ROOT=Path(__file__).resolve().parents[2];PJ=Path(os.environ.get('AMNI_STORE1_PROPS',str(ROOT/'data'/'store1'/'props_en.json')))
on=lambda:os.environ.get('AMNI_STORE1','0')=='1'
CF=dict(tau=float(os.environ.get('AMNI_STORE1_TAU','2.5')),wmin=float(os.environ.get('AMNI_STORE1_WMIN','0.5')),prior=0.35,cap=0.25,kind=(0.12,0.0,0.12,0.06),tfit=0.45,tbad=0.9,hand=1.0,alias=0.6,ncand=40,cs=0.55)
ABL=set(filter(None,os.environ.get('AMNI_STORE1_ABL','').split(',')))
STOP=set("a an the of in on at to for by with from as is are was were be been being am do does did done has have had having will would shall should can could may might must what which who whom whose when where why how that this these those it its they them their there here i you he she we me my your his her our us and or but not no yes if then than so very just also about into onto over under up down out off again ever any some all each every much many more most other such only own same few name named called call tell give know please one ones s".split())
TH=set("film movie novel book song album band series show painting opera play game company team river mountain city island planet moon poem magazine newspaper statue building ship comic sculpture symphony musical".split())
GEN=set("year date day time place country city people person number name thing things kind type sort part member first last current original famous known".split())
HAND="""who|paint=170;who|sculpt=170;who|draw=170;artist=170;painter=170;sculptor=170;who|create=170 61 112;creator=170 61
who|write=50 86 676 58;author=50;writer=50 58;who|compose=86 676;composer=86;who|direct=57;director=57;who|produce=162;producer=162
who|found=112;founder=112;who|establish=112;who|invent=61;inventor=61;who|discover=61;discoverer=61;who|design=84 170;architect=84;who|build=84 176
who|develop=178;developer=178;who|make=176 178 170;who|manufacture=176;manufacturer=176;who|publish=123;publisher=123;who|own=127;owner=127
who|sing=175;singer=175;who|perform=175;who|record=175;who|star=161;who|act=161;who|voice=725;who|illustrate=110;illustrator=110;who|coach=286;coach=286
who|marry=26;wife=26;husband=26;spouse=26;father=22;dad=22;mother=25;mom=25;son=40;daughter=40;child=40;brother=3373;sister=3373;sibling=3373
who|succeed=1366;succeed=1366;successor=1366;replace=1365 1366;who|precede=1365;predecessor=1365;who|win=1346;winner=1346
who|lead=6 35 488;leader=6 35 488;president=35 488;king=35;queen=35;monarch=35;emperor=35;prime minister=6;head of state=35;head of government=6;mayor=6;governor=6;ceo=169;chairman=488;chairperson=488
who|name=138;name after=138;name for=138;who|influence=737;influence=737;who|inspire=941;inspire=941;who|lyric=676;lyricist=676;lyric=676;screenplay=58;screenwriter=58
when|bear=569;birthday=569;bear=569 19;when|die=570;die=570 20;death=570 20 509;when|found=571;when|establish=571;when|form=571 740;when|create=571 577;when|start=580 571;when|begin=580 571
when|open=1619 571;when|build=1619 571;when|construct=1619 571;when|complete=1619 571;when|invent=575 571;when|discover=575 571;when|write=577 571;when|publish=577;when|release=577;when|premiere=577 1191;when|launch=619 577 571
when|end=582 576;when|close=576 582;when|dissolve=576;when|abolish=576;when|demolish=576;when|happen=585 580;when|take place=585 580;when|occur=585 580;when|fight=585 580;when|sign=585 571;when|first perform=1191;when|perform=1191 577
where|bear=19;birthplace=19;hometown=19;where|die=20;where|locate=131 276 17 30 159;where|base=159 131;where|headquarter=159;headquarters=159;where|find=276 131 17 2974 183
where|live=2974 183 131;where|be=131 276 17 30 159;where|flow=403 206;where|rise=885;where|originate=885;where|film=915;where|set=840;where|study=69;where|educate=69;where|attend=69;where|work=108;where|form=740
how tall=2048 2044;how high=2044 2048;elevation=2044;height=2048 2044;how long=2043 2047;length=2043;how big=2046;how large=2046;area=2046;how many people=1082;population=1082;how many resident=1082;how many inhabitant=1082
how many episode=1113;how many season=2437;atomic number=1086;element symbol=246;chemical symbol=246;symbol=246;chemical formula=274;formula=274;nickname=1449;birth name=1477;real name=1477;short name=1813;abbreviation=1813;pseudonym=742;pen name=742
capital=36 1376;capital of=1376 36;currency=38;official language=37;language=37 407 364;anthem=85;national anthem=85;continent=30;country=17 27 495;nationality=27;citizen=27;state=131;province=131;county=131;region=131;located=131 276 17
flow into=403;empty into=403;mouth=403;source=885;mountain range=4552;orbit=397;highest point=610;tallest mountain=610;highest mountain=610;lowest point=1589;river=206 403;lake=206;body of water=206;sea=206
study=69;university=69;college=69;school=69;attend=69;employer=108;work for=108;occupation=106;job=106;profession=106;party=102;political party=102;religion=140;team=54;club=54;play for=54;position=413;instrument=1303;sport=641;league=118
stadium=115;home ground=115;venue=115;arena=115;award=166;prize=166;notable work=800;best know=800;famous for=800;field=101;member of=463;part of=361;make of=186;make from=186;material=186;use for=366;eat=1034;feed on=1034;diet=1034;habitat=2974;endemic=183;color=462;colour=462;opposite=461
genus=171;family=171;parent taxon=171;record label=264;label=264;network=449;channel=449;air on=449;broadcaster=449;genre=136;base on=144;adapt from=144;sequel=156;follow by=156;prequel=155;series=179;set in=840;film in=915;character=674;original language=364
production company=272;distributor=750;distribute=750;executive producer=1431;product=1056;industry=452;organizer=664;parent company=749;parent organization=749;cause of death=509;how|die=509;symptom=780;treat=2176 2175;subject=921;about=921"""
HAND2="""money=38;cash=38;come into the world=569 19;office=39;position held=39;hold=39;illustrate=110;put out=123;release by=123 264;style=136;style of music=136;music style=136;who|play=161 175;play=161 1303 641 54;star as=161;spot=61;who|spot=61;who|find=61;come up with=61;who|come up with=61;identify=61;moon of=397;satellite of=397;circle=397;orbit=397;revolve around=397;summit=610;peak=610;band=463 361;join=463 54;who|start=112;start=112 580 571;shoot=915;where|shoot=915;film in=915;produce=1056 162;sell=1056;mine=1056;display=276;keep=276;house=276;museum=276;discharge=403;drain=403;empty=403;flow=403 206;where|begin=885;where|start=885;begin=885 580 571;live=2974 183;carve=186;carve from=186;belong to=361 279 171 463;type of=279 31;kind of=279 31;hero=1441;hero of=1441;appear in=1441;character in=1441;protagonist=1441;famous=800;best know work=800;rule=35 6;govern=6;date back=571;set up=571 112;go down=582 576;sink=582 576;go by=1477 742 1449;nickname=1449 742;know as=1449 742 1813;epithet=1449 742;take its name from=138;name from=138;derive name from=138;studio=272 750;production studio=272"""
SOUNDV=re.compile(r"\b(?:sound|noise|cry|call|utter|vocali)")
@functools.lru_cache(maxsize=200000)
def lem(w:str)->str:
 if len(w)<=3:return w
 wn=WF._wn();return ((wn.morphy(w,'v') or wn.morphy(w,'n')) if wn else None) or w
@functools.lru_cache(maxsize=200000)
def vinf(w:str)->bool:
 wn=WF._wn();v=wn.morphy(w,'v') if wn and len(w)>3 else None;return bool(v) and v!=w
@functools.lru_cache(maxsize=1)
def props()->dict:
 J=json.loads(PJ.read_text())['props'];return {int(k[1:]):v for k,v in J.items()}
@functools.lru_cache(maxsize=1)
def ptab()->dict:
 T={}
 def add(k,ps,w,src):
  for p in ps:T.setdefault(k,{});T[k][p]=max(T[k].get(p,(0,''))[0],w),src if w>=T[k].get(p,(0,''))[0] else T[k][p][1]
 if 'alias' not in ABL:
  A={}
  for p,v in props().items():
   for i,a in enumerate([v['label']]+v['aliases']):
    t=[lem(x) for x in WF.nrm(re.sub(r"\([^)]*\)","",a)).split()]
    if t and not all(x in STOP for x in t) and len(t)<=4:A.setdefault(' '.join(t),set()).add((p,1.0 if i==0 else 0.8))
  for k,S in A.items():
   n=len({p for p,_ in S});[add(k,[p],CF['alias']*w*(0.6 if ' ' not in k else 1.0)/n,'alias') for p,w in S]
 if 'hand' not in ABL:
  for it in re.split(r"[;\n]",HAND):
   k,_,v=it.strip().partition('=');wh,_,g=k.rpartition('|');g=' '.join(lem(x) for x in g.split());ps=[int(x) for x in v.split()];add((wh+'|' if wh else '')+g,ps,CF['hand'],'hand');wh and add(g,ps,0.7*CF['hand'],'hand')
  for it in HAND2.split(';'):k,_,v=it.strip().partition('=');wh,_,g=k.rpartition('|');g=' '.join(lem(x) for x in g.split());add((wh+'|' if wh else '')+g,[int(x) for x in v.split()],0.75*CF['hand'],'hand')
  for d,wh in ((WF.WHEN,'when'),(WF.WHO,'who')):
   for k,v in d.items():add(wh+'|'+' '.join(lem(x) for x in k.split()),v,0.9*CF['hand'],'hand')
  for k,v in WF.NOUN.items():add(' '.join(lem(x) for x in k.split()),v,0.9*CF['hand'],'hand')
 return {WF.h64(k):v for k,v in T.items()}
CONTR={'what','who','where','when','how','that','there','here','it','he','she','which'}
def toks(q:str,brk:bool=False)->tuple:
 W=list(re.finditer(r"[^\W_]+(?:['’`´][^\W_]+)*",F:=WF.fold(q or '').replace("&"," and ")));o=[];ps=set();B=set();pe=0
 for mm in W:
  w=mm[0];re.search(r"[,;:()\"!?.]",F[pe:mm.start()]) and B.add(len(o));pe=mm.end();m=re.fullmatch(r"(.+)['’`´][sS]",w);c=bool(m) and m[1].lower() in CONTR;m and not c and ps.add(len(o));o.append(re.sub(r"['’`´]","",m[1] if m else w));c and o.append('is')
 return (o,[x.lower() for x in o],ps,B) if brk else (o,[x.lower() for x in o],ps)
def cphrase(t:list,O:list=None)->set:
 for i,x in enumerate(t):
  if x in ('which','what'):
   j=i+1;j+=2 if t[j:j+2] in (['kind','of'],['type','of'],['sort','of']) else 0;S=set()
   while j<len(t) and t[j] not in STOP and j-i<=3 and not vinf(t[j]) and not (O and O[j][:1].isupper()):S.add(j);j+=1
   return S
 return set()
@functools.lru_cache(maxsize=8192)
def nclass(w:str)->str:
 wn=WF._wn();S=wn.synsets(w,'n')[:1] if wn else []
 if not S:return ''
 H={h.name() for h in S[0].closure(lambda z:z.hypernyms()+z.instance_hypernyms())}
 return 'who' if 'person.n.01' in H else 'where' if H&{'location.n.01','region.n.03','geographical_area.n.01','body_of_water.n.01','building.n.01','structure.n.01'} and w not in ('river','lake','sea','ocean','mountain','museum','building','stadium') else ''
def whq(t:list,o:list=None)->str:
 s=' '.join(t)
 if re.search(r"\b(?:what|which) (?:year|date|day|century|decade|month)\b|\bwhen\b",s):return 'when'
 if re.search(r"\bhow (?:many|much|tall|long|high|big|large|old|far|deep|heavy|wide)\b",s):return 'how'
 m=re.search(r"\b(who|whom|whose|where|which|what)\b",s)
 if not m:return 'what'
 w={'whom':'who','whose':'who'}.get(m[1],m[1]);cp=sorted(cphrase(t,o))
 return (nclass(t[cp[-1]]) or w) if w in ('which','what') and cp and WF.h64(lem(t[cp[-1]])) not in ptab() else w
def etype(q:str,t:list,o:list=None)->str:
 s=' '.join(t);w=whq(t,o)
 if w=='when':return 'time'
 if w=='how':return 'number'
 if w=='where':return 'place'
 if w=='who':return 'person'
 if 'tgate' in ABL:return 'any'
 c=_qclass(q);return 'class:'+c if c and c not in ('person',) else 'any'
@functools.lru_cache(maxsize=4096)
def _qclass(q:str)->str:
 try:return __import__('amni.serve.cognition',fromlist=['qclass']).qclass(q)
 except Exception:return ''
def cues(t:list,o:list=None)->tuple:
 lt=[lem(x) for x in t];w=whq(t);T=ptab();M=[];pos=set()
 for n in range(1,5):
  for i in range(len(lt)-n+1):
   g=' '.join(lt[i:i+n])
   if n==1 and (g in STOP or len(g)<3):continue
   for k in (g,w+'|'+g):
    for p,(sc,_) in T.get(WF.h64(k),{}).items():
     sc=sc*(1.15 if '|' in k else 1.0)*(1+0.1*(n-1))*(0.3 if o and all(o[x][:1].isupper() and x>0 for x in range(i,i+n)) else 1.0);M.append((p,sc,i,i+n));sc>=0.3 and pos.update(range(i,i+n))
 return M,pos
class R:
 def __init__(s,fd=None):s.D=D1.drive(D1.FD if fd is None else fd);s.hum=s.D.ix(5) if s.D else -1
 def link(s,o:list,t:list,pos:set,ps:set=frozenset(),cp:set=frozenset(),brk:set=frozenset())->tuple:
  pos=pos|cp;SP=[]
  for i in range(len(t)):
   for j in range(i+1,min(len(t),i+8)+1):
    c=[k for k in range(i,j) if t[k] not in STOP]
    if not c or set(c)<=cp or j-i==1 and (len(t[i])<3 and not (any(ch.isdigit() for ch in t[i]) and not t[i].isdigit() or o[i].isupper() and len(o[i])>1) or t[i] in GEN or t[i].isdigit()):continue
    SP.append((i,j,' '.join(t[i:j])))
  ex=[(i,j,' '.join(t[k]+('s' if k in ps else '') for k in range(i,j))) for i,j,_ in SP if ps&set(range(i,j))]+[(i,j,x[:-1]) for i,j,x in SP if x.endswith('s') and len(x)>4]
  H=s.D.ents_many([WF.h64(x) for _,_,x in SP+ex]);E={};lk=set();cov=set()
  for (i,j,x),L in zip(SP+ex,H):
   L=[z for z in L if s.D.vfy(x,z[0])] if s.D.fd==0 else L
   if not L:continue
   lk.update(range(i,j));cov.add((i,j));c=[k for k in range(i,j) if t[k] not in STOP];nc=len([k for k in c if k not in pos]);cap=sum(o[k][:1].isupper() for k in range(i,j) if k>0 or t[0] not in STOP)/max(1,j-i-(i==0 and t[0] in STOP))
   loc=-0.15 if i>0 and t[i-1] in ('in','on','at','during','near') else 0.0
   for e,k in L:
    sc=1.0+0.25*min(len(c),3)-0.1*(len(c)-nc)-(0.4 if nc==0 else 0.0)+CF['cap']*cap+CF['kind'][min(k,3)]+loc
    if sc>E.get(e,(-9,))[0]:E[e]=(sc,i,j,k)
  th={x for x in t if x in TH or x.rstrip('s') in TH}
  for e in list(E):
   sl=s.D.sitelinks(e) if 'prior' not in ABL else 0;E[e]=(E[e][0]+CF['prior']*min(1.0,math.log2(1+sl)/9.0),)+E[e][1:]+(sl,)
  if th and 'hint' not in ABL:
   for e in sorted(E,key=lambda z:-E[z][0])[:CF['ncand']]:
    L={w for _,v,*_ in s.D.values(e,31)[:5] for w in WF.nrm(s.D.label(v)).split()}
    if {x.rstrip('s') for x in th}&{w.rstrip('s') for w in L}:E[e]=(E[e][0]+0.3,)+E[e][1:]
  pn={k for k in range(1,len(o)) if o[k][:1].isupper() and t[k] not in STOP and not t[k].isdigit()};cu=pn|{k for k in range(1,len(o)-1) if t[k] in ('of','the','de','for','and','at','a') and k-1 in pn and k+1 in pn};run={}
  for k in sorted(cu):run[k]=run.get(k-1,k) if k not in brk else k
  E=dict(sorted(E.items(),key=lambda z:-z[1][0])[:CF['ncand']])
  for e,(sc,i,j,k,sl) in list(E.items()):
   rs={run[x] for x in range(i,j) if x in run};mis=len([x for x in pn if run.get(x) in rs and not i<=x<j]);lw=(i>0 or o[0][:1].islower()) and not any(o[x][:1].isupper() for x in range(i,j)) and 'case' not in ABL;lb=s.D.label(e) if lw else '';lc=lw and bool(lb) and lb[:1].isupper() and not lb[:1].lower()==lb[:1]
   E[e]=(sc-0.5*mis-0.6*lc,i,j,k,sl,mis)
  R2={}
  for k,r in run.items():R2.setdefault(r,[]).append(k)
  ur=[v for v in R2.values() if len([x for x in v if x in pn])>=2 and not any(a<=min(v) and max(v)<b for a,b in cov)]
  return dict(sorted(E.items(),key=lambda z:-z[1][0])),((pn-lk)-pos)|{-1-i for i,_ in enumerate(ur)},ur
 def ety(s,e:int)->str:
  G={p for p,_ in s.D.groups(e)}
  if 21 in G or 569 in G or any(v==s.hum for _,v,*_ in s.D.values(e,31)[:4]) or 106 in G:return 'person'
  if G&{17,131,30,36,1082,2044,206,47,1376,610,4552}:return 'place'
  return 'org' if G&{159,112,452,1128,169,749} else 'thing'
 def cls(s,e:int,c:str)->float:
  w=WF.nrm(c).split()[-1:] or ['']
  for p in (31,106,279):
   for _,v,*_ in s.D.values(e,p)[:6]:
    n=WF.nrm(s.D.label(v))
    if n and (w[0] in n.split() or w[0].rstrip('s') in n.split()):return 1.0
  try:r=__import__('amni.serve.cognition',fromlist=['isa']).isa(s.D.label(e),c)[0]
  except Exception:r=None
  return 1.0 if r else -1.0 if r is False else 0.0
 def tfit(s,et:str,k:int,v:int)->float:
  if et=='time':return CF['tfit'] if k==2 else -CF['tbad']
  if et=='number':return CF['tfit'] if k==3 else -CF['tbad']
  if k!=1:return -CF['tbad'] if et in ('person','place') or et.startswith('class:') else 0.0
  if et in ('person','place'):y=s.ety(v);return CF['tfit'] if y==et or et=='person' and y=='org' and False else 0.5*CF['tfit'] if et=='person' and y=='org' else -CF['tbad']*0.6
  if et.startswith('class:'):return CF['tfit']*s.cls(v,et[6:])
  return 0.0
 def qmatch(s,e:int,p:int,i:int,nq:int,Y:set,TS:set)->int:
  if not nq:return 0
  c=0;st=en=None
  for qp,k,v in s.D.quals(e,p,i,nq):
   if k==2:
    y=WF.tu(v)[0];c+=2*(qp in (585,577,580) and y in Y);st=y if qp==580 else st;en=y if qp==582 else en
   elif k==1:l=WF.nrm(s.D.label(v)).split();c+=bool(l) and set(l)<=TS
   elif k>=4:l=WF.nrm(s.D.string(v)).split();c+=bool(l) and set(l)<=TS
  return c+(not c and st is not None and any(st<=y<=(en or 9999) for y in Y))
 UC={'foot':('metre',0.3048),'mile':('kilometre',1.609344),'inch':('centimetre',2.54),'square mile':('square kilometre',2.589988),'pound':('kilogram',0.45359237),'yard':('metre',0.9144),'acre':('hectare',0.40468564)}
 def render(s,k:int,v:int,x:int,p:int)->tuple:
  if k==1:n=s.D.name(v);return n,[n]+s.D.names(v)
  if k==2:r=WF.fdate(v);return r[0],r
  if k==3:
   u=s.D.name(x) if x!=0xFFFFFFFF else {1113:'episodes',2437:'seasons'}.get(p,'');f=float(np.uint64(v).view(np.float64));r=WF.fnum(f,u);c=s.UC.get(u)
   if c:r2=WF.fnum(round(f*c[1],1),c[0]);return f"{r[0]} ({r2[0]})",r+r2
   return r[0],r
  t=s.D.string(v);return t,[t]
 def wd(s,q:str)->dict:
  o,t,ps,bk=toks(q,True);M,pos=cues(t,o);et=etype(q,t,o);cp=cphrase(t,o);E,un,ur=s.link(o,t,pos,ps,cp,bk);B=[];Y={int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b",' '.join(t))};TS=set(t);top=set(list(E)[:5]);cpp={m[0] for m in M if set(range(m[2],m[3]))<=cp}
  for e,(sc,i,j,k,sl,mis) in E.items():
   P={}
   for p,w,a,b in M:
    if b<=i or a>=j:P[p]=max(P.get(p,0.0),w)
   if not P:continue
   pen=0.6*len(un-set(range(i,j)))+0.2*len([v for v in ur if not set(v)&set(range(i,j))])
   for p,_ in s.D.groups(e):
    w=P.get(p,0.0)
    if not w:continue
    V=s.D.values(e,p)
    if not V:continue
    qm=[s.qmatch(e,p,i_,z[4],Y,TS) if len(V)>1 and e in top and 'qual' not in ABL else 0 for i_,z in enumerate(V)];o_=sorted(range(len(V)),key=lambda i_:(-qm[i_],-V[i_][3],i_));kk,v,x,_,_=V[o_[0]];tf=0.0 if 'tgate' in ABL or et.startswith('class:') and p in cpp else s.tfit(et,kk,v)
    B.append((sc+w+tf+0.3*min(qm[o_[0]],2)-pen+(0.3 if p in cpp else 0.0),e,p,kk,v,x,sc,w,tf,pen+0.5*mis,[V[i_] for i_ in o_]))
  B.sort(key=lambda z:-z[0])
  ents=list(dict.fromkeys([b[1] for b in B]+list(E)));PP=sorted({m[0] for m in M},key=lambda p:-max(m[1] for m in M if m[0]==p))
  if not B:return {'store':'wd','score':0.0,'absent':True,'ents':ents,'props':PP,'et':et}
  b=B[0];a,forms=s.render(b[3],b[4],b[5],b[2])
  if b[3]==3 and len(b[10])>1:
   for z in b[10][1:4]:a2,f2=s.render(z[0],z[1],z[2],b[2]);forms+=f2;a+=f" ({a2})" if z[2]!=b[5] else ''
  pl=props().get(b[2],{}).get('label',f'P{b[2]}')
  return {'store':'wd','score':b[0],'absent':b[9]>0 or b[7]<CF['wmin'] and b[8]<=0,'e':b[1],'p':b[2],'answer':a,'forms':forms,'ents':ents,'props':list(dict.fromkeys([x[2] for x in B]+PP)),'et':et,'parts':[round(x,3) for x in (b[6],b[7],b[8],b[9])],'line':f"{s.D.label(b[1])}, {pl}: {a}.",'prov':f"Wikidata Q{s.D.qid(b[1])} P{b[2]}",'key':(int(s.D.qid(b[1])),b[2])}
CSR=[('antonym',r"\b(?:opposite|antonym|reverse)\b"),('synonym',r"\b(?:synonym|another word|other word|word that means|same thing as|means the same|similar meaning)\b"),('young',r"\b(?:baby|babies|young|offspring|newborn)\b"),('group',r"\b(?:group|collective|bunch|gathering) of\b"),('sound',r"\b(?:sound|noise|call|cry)\b.*\bmake\b|\bmakes? what (?:sound|noise)\b"),
 ('count',r"\bhow many\b"),('made',r"\bmade (?:of|from|out of|with)\b|\bmaterial\b|\b(?:formed|produced|carved|crafted) from\b|\bingredient"),('used_for',r"\bused (?:for|to)\b|\buse (?:an? |the )?[a-z ]+ (?:for|to)\b|\bpurpose of\b|\bwhat (?:is|are) (?:an? |the )?[a-z ]+ for\b"),('color',r"\bcolou?r\b"),('where',r"\blives?\b|\bhabitat\b|\bfound in the wild\b|\bnests?\b"),
 ('eat',r"\b(?:eat|eats|feed|feeds|graze|hunt|diet)\b"),('drink',r"\bdrinks?\b"),('part_of',r"\bpart of\b"),('can_do',r"\bwhat can\b.*\bdo\b")]
CSR=[(k,re.compile(r)) for k,r in CSR]
CSW=set("opposite antonym reverse synonym another other word means same similar meaning baby babies young offspring newborn group collective bunch gathering sound noise call cry make makes many made out material formed produced carved crafted ingredient ingredients used use purpose colour color live lives living habitat found wild nest nests eat eats feed feeds graze hunt diet drink drinks part can mostly mainly usually typically normally traditional typical called known kind animal animals thing things".split())
COLORS=set("red orange yellow green blue purple violet pink brown black white gray grey gold silver indigo turquoise tan beige crimson maroon".split())
def _cog():return __import__('amni.serve.cognition',fromlist=['know'])
def _concept(G,t:list,o:list,cp:set,skip:set=frozenset(),left:bool=False)->str:
 c=[i for i,x in enumerate(t) if x not in STOP and x not in CSW and i not in cp and i not in skip and not x.isdigit()]
 for n in (3,2,1):
  for a in (range(len(c)-n+1) if left else range(len(c)-n,-1,-1)):
   ix=c[a:a+n]
   if ix[-1]-ix[0]==n-1:
    s=' '.join(t[k] for k in ix)
    if G.cnv(s) or G.wsyn(s,0):return s
 return t[c[-1]] if c else ''
def csrel(ql:str)->str:return next((k for k,rx in CSR if rx.search(ql)),'')
def cs(R_,q:str)->dict:
 G=_cog();o,t,_=toks(q);ql=' '.join(t);rel=csrel(ql);cp=cphrase(t);V={}
 if not rel:return {'store':'concepts','rel':'','score':0.0}
 mN=re.search(r"\bhow many ([a-z]+)",ql);mG=re.search(r"\b(?:group|collective|bunch|gathering) of (\w+)",ql) if rel=='group' else re.search(r"\b(?:baby|young|newborn) (\w+)",ql) if rel=='young' else None;c=mG[1] if mG else _concept(G,t,o,cp,{t.index(mN[1])} if mN else set(),rel=='count');sg=G._sg(c) if c else ''
 def vote(a,sc,pv,key):a=a.strip();k=G._sg(a.lower());(k and k!=sg and k not in STOP and k not in CSW) and V.setdefault(k,{'a':a,'s':0.0,'p':pv,'key':key}).__setitem__('s',V[k]['s']+sc)
 wn=WF._wn()
 if rel in ('synonym','young','sound') and c and wn:
  if rel=='synonym':[vote(l.name().replace('_',' '),(2.0 if sy.pos() in 'as' else 1.0)/(1+k)+0.05*l.count(),f"WordNet {sy.name()} synonym",('wn','synonym',sy.name())) for k,sy in enumerate(wn.synsets(c)[:4]) for l in sy.lemmas() if l.name().lower()!=c and '_' not in l.name()]
  else:
   rx=re.compile(r"^(?:an? |the )?(?:characteristic |typical |usual |loud |harsh |high-pitched |low )?(?:sound|cry|call|noise|vocalization)s? (?:made by|of|uttered by|produced by|characteristic of) (?:an? |the )?(?:[a-z]+ )?"+re.escape(sg)+r"s?\b" if rel=='sound' else r"^(?:an? |the )?(?:young|baby|larval|immature|juvenile|newborn)(?: stage of (?:an? )?)?\s*(?:[a-z]+ )?(?:of (?:an? |the )?)?(?:[a-z]+ )?"+re.escape(sg)+r"s?\b")
   for i in G._gi()[0].get(sg,[])[:3000]:
    g=G.gloss(int(i)).lower();m=rx.search(g);m and vote(G.lab(int(i)),2.0+(1.0 if len(g)<=m.end()+2 else 0.0),f"WordNet {G._g()['nm'][int(i)]}: {G.gloss(int(i))}",('wn','gloss',G._g()['nm'][int(i)]))
  if V:b=max(V.values(),key=lambda z:z['s']);return {'store':'concepts','rel':rel,'concept':c,'answer':b['a'],'forms':[b['a']],'score':min(0.95,0.5+0.15*b['s']),'line':b['p'],'prov':b['p'],'key':b['key']}
 if rel in ('where','made','count') and c:
  LOC={'location.n.01','region.n.03','geographical_area.n.01','body_of_water.n.01','habitat.n.01','environment.n.01','land.n.04','vegetation.n.01'}
  for j in G.cnv(c)[:2]:
   for r,d,w in G.cne(j,{'where':('AtLocation',),'made':('MadeOf',),'count':('HasA','PartOf','DefinedAs','HasProperty')}[rel]):
    x=G.cnt(d)
    if rel=='count':
     m=re.search(r"\b"+G.NUMR+r"(?:[\s-]+[a-z]+){0,2}?[\s-]+"+re.escape(G._sg(mN[1])[:max(3,len(mN[1])-2)]),x) if mN else None;m and vote(m[1],1.2+0.1*min(w,4),f"ConceptNet 5.7 {G.cnt(j)} {r} {x} (CC BY-SA 4.0)",('cn',r,G.cnt(j)))
    else:vote(re.sub(r"^(?:made (?:from|of|with) )","",x),(0.6+0.15*min(w,5)) if rel=='made' else 0.4+0.05*min(w,5),f"ConceptNet 5.7 {G.cnt(j)} {r} {x} (CC BY-SA 4.0)",('cn',r,G.cnt(j)))
  for sy in (wn.synsets(c.replace(' ','_'),'n')[:2] if wn else []):
   g=sy.definition().lower()
   if rel=='count' and mN:[vote(m[1],1.4,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name())) for m in re.finditer(r"\b"+G.NUMR+r"(?:[\s-]+[a-z]+){0,2}?[\s-]+"+re.escape(G._sg(mN[1])[:max(3,len(mN[1])-2)]),g)]
   if rel=='made':[vote(G._np(m[1],2) or m[1],1.3,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name())) for m in re.finditer(r"\b(?:made (?:from|of|with)|prepared from|derived from|consisting (?:mostly |mainly )?of|obtained from|extracted from)\s+(?:the\s+)?([a-z][a-z -]+)",g)]
   if rel=='where':[vote(m[1],1.2 if any(h.name() in LOC for h in (wn.synsets(m[1].replace(' ','_'),'n')[:1] or [sy])[0].closure(lambda z:z.hypernyms())) else 0.4,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name())) for m in re.finditer(r"\b(?:in|of|native to|found in|inhabiting|living in|lives in)\s+(?:the\s+)?([a-z]+(?: [a-z]+)?)",g)]
  if R_ and R_.D and rel in ('where','made'):
   for e,_ in R_.D.ents(WF.nrm(c))[:3]+R_.D.ents(WF.nrm(sg))[:3]:[vote(R_.D.name(v),1.7,f"Wikidata Q{R_.D.qid(e)} P{pp}",('wd',R_.D.qid(e),pp)) for pp in ((2974,183) if rel=='where' else (186,)) for _,v,*_ in R_.D.values(e,pp)[:3]]
  if V:b=max(V.values(),key=lambda z:z['s']);return {'store':'concepts','rel':rel,'concept':c,'answer':b['a'],'forms':[b['a']],'score':min(0.95,0.5+0.15*b['s']),'line':b['p'],'prov':b['p'],'key':b['key']}
 if rel in ('eat','drink') and c:
  vb=G.VSYN.get(rel,(rel,));rx=re.compile(r"^(?:%s)\s+(?:mostly |mainly |only )?([a-z][a-z -]+)$"%'|'.join(map(re.escape,vb)))
  for j in G.cnv(c)[:2]:
   for r,d,w in G.cne(j,('CapableOf','Desires','UsedFor','HasA')):
    m=rx.match(G.cnt(d));m and vote(m[1],1.0+0.2*min(w,5),f"ConceptNet 5.7 {G.cnt(j)} {r} {G.cnt(d)} (CC BY-SA 4.0)",('cn',r,G.cnt(j)))
  for sy in (wn.synsets(c.replace(' ','_'),'n')[:2] if wn else []):
   m=re.search(r"\b(?:%s)\s+(?:on\s+)?(?:mainly\s+|chiefly\s+|mostly\s+)?([a-z][a-z -]+?)(?:;|,|$| and )"%'|'.join(map(re.escape,vb+('feeds','feeding','lives on'))),sy.definition().lower());m and vote(m[1],1.4,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name()))
  if R_ and R_.D:
   for e,_ in R_.D.ents(WF.nrm(c))[:3]+R_.D.ents(WF.nrm(sg))[:3]:[vote(R_.D.name(v),1.7,f"Wikidata Q{R_.D.qid(e)} P1034",('wd',R_.D.qid(e),1034)) for _,v,*_ in R_.D.values(e,1034)[:3]]
  if V:b=max(V.values(),key=lambda z:z['s']);return {'store':'concepts','rel':rel,'concept':c,'answer':b['a'],'forms':[b['a']],'score':min(0.95,0.5+0.15*b['s']),'line':b['p'],'prov':b['p'],'key':b['key']}
  return {'store':'concepts','rel':rel,'concept':c,'score':0.0}
 if rel in ('used_for','color','part_of','can_do','sound') and c:
  RR={'used_for':('UsedFor',),'color':('HasProperty',),'part_of':('PartOf',),'can_do':('CapableOf',),'sound':('CapableOf',)}[rel]
  for j in G.cnv(c)[:2]:
   for r,d,w in G.cne(j,RR):
    x=G.cnt(d)
    if rel=='color' and x not in COLORS:continue
    if rel=='sound' and not (wn and any(SOUNDV.search(s_.definition()) for s_ in wn.synsets(x.split()[0],'v')[:2])):continue
    vote(x,1.0+0.2*min(w,5),f"ConceptNet 5.7 {G.cnt(j)} {r} {x} (CC BY-SA 4.0)",('cn',r,G.cnt(j)))
  for sy in (wn.synsets(c.replace(' ','_'),'n')[:2] if wn else []):
   if rel=='part_of':[vote(l.name().replace('_',' '),1.2,f"WordNet {sy.name()} holonym {h.name()}",('wn','holonym',sy.name())) for h in sy.part_holonyms()+sy.member_holonyms() for l in h.lemmas()[:1]]
   if rel=='color':[vote(w_,0.9,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name())) for w_ in re.findall(r"[a-z]+",sy.definition().lower()) if w_ in COLORS]
   if rel=='used_for':m=re.search(r"\b(?:used (?:for|to|in)|for) ([a-z ]+?)(?:;|$|,)",sy.definition().lower());m and vote(m[1],1.6,f"WordNet {sy.name()}: {sy.definition()}",('wn','gloss',sy.name()))
  if R_ and R_.D and rel in ('used_for','color','part_of'):
   pp={'used_for':366,'color':462,'part_of':361}[rel]
   for e,_ in R_.D.ents(WF.nrm(c))[:3]:[vote(R_.D.name(v),1.7,f"Wikidata Q{R_.D.qid(e)} P{pp}",('wd',R_.D.qid(e),pp)) for _,v,*_ in R_.D.values(e,pp)[:3]]
  if V:b=max(V.values(),key=lambda z:z['s']);return {'store':'concepts','rel':rel,'concept':c,'answer':b['a'],'forms':[b['a']],'score':min(0.95,0.5+0.15*b['s']),'line':b['p'],'prov':b['p'],'key':b['key']}
 wn=WF._wn();pl=c if c.endswith('s') or wn and wn.morphy(c,'n') not in (None,c) else c+'es' if c.endswith(('sh','ch','x','o')) else c+'s'
 F={'young':[f"what is a baby {sg} called"],'group':[f"a group of {pl} is called"],'sound':[f"what sound does a {sg} make"],'antonym':[f"what is the opposite of {c}"],'synonym':[f"another word for {c}"]}.get(rel,[])
 for qq in F+([q] if rel in ('antonym','young','group','synonym','sound') else []):
  try:r=G.know(qq)
  except Exception:r=None
  if r and r.get('answer') and any(str(p).startswith(('WordNet','ConceptNet','Wikidata','English plural')) for p in r.get('prov') or []):
   pv=next(str(p) for p in r['prov'] if str(p).startswith(('WordNet','ConceptNet','Wikidata','English plural')));st='wn' if pv.startswith('WordNet') else 'cn' if pv.startswith('ConceptNet') else 'wd' if pv.startswith('Wikidata') else 'rule'
   return {'store':'concepts','rel':r.get('kind') or rel,'concept':c,'answer':r['answer'],'forms':[r['answer']],'score':float(r.get('conf') or 0.5),'line':r.get('text') or r['answer'],'prov':pv,'key':(st,r.get('kind'),c)}
 return {'store':'concepts','rel':rel,'concept':c,'score':0.0}
PRIME=re.compile(r"\bprimes?\b")
def isp(n:int)->bool:return n>1 and all(n%d for d in range(2,int(n**0.5)+1))
def _num(w:str):
 w=w.replace(',','');return int(w) if w.isdigit() else {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10}.get(w)
def prime(q:str):
 s=re.sub(r"(?<=\d),(?=\d)",'',(q or '').lower());n=[int(x) for x in re.findall(r"\d+",s)]
 if not PRIME.search(s):return None
 if len(n)==1 and re.search(r"^\W*(?:is|was|can|could|does|would|will)\b|\bis \d+ (?:an? )?(?:prime|composite)\b|\bwhether\b",s):return 'yes' if isp(n[0]) else 'no'
 m=re.search(r"\b(\d+)(?:st|nd|rd|th) prime\b|\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) prime\b",s)
 if re.search(r"\bsmallest prime\b|\blowest prime\b",s) and not n:return '2'
 if re.search(r"\b(?:smallest|first|lowest) (?:odd )?prime\b",s) and 'odd' in s:return '3'
 if re.search(r"\bbetween\b",s) and len(n)>=2:
  P=[x for x in range(min(n[:2])+1,max(n[:2])) if isp(x)];return str(len(P)) if re.search(r"\bhow many\b",s) else ', '.join(map(str,P)) if P else None
 if re.search(r"\b(?:next|after|following|greater than|bigger than|larger than|above)\b",s) and n:
  x=n[0]+1
  while not isp(x):x+=1
  return str(x)
 if re.search(r"\b(?:before|below|less than|smaller than|under)\b",s) and n:
  if re.search(r"\bhow many\b",s):return str(sum(isp(x) for x in range(2,n[0])))
  x=n[0]-1
  while x>1 and not isp(x):x-=1
  return str(x) if x>1 else None
 if m:
  k=int(m[1]) if m[1] else _num(m[2]);x,c=1,0
  while c<k:x+=1;c+=isp(x)
  return str(x)
 return None
AR=[(r"\bdivide (\d+(?:\.\d+)?) by (\d+(?:\.\d+)?)",lambda a,b:a/b),(r"(\d+(?:\.\d+)?) divided by (\d+(?:\.\d+)?)",lambda a,b:a/b),(r"\bmultiply (\d+(?:\.\d+)?) by (\d+(?:\.\d+)?)",lambda a,b:a*b),(r"(\d+(?:\.\d+)?) (?:times|multiplied by|x) (\d+(?:\.\d+)?)",lambda a,b:a*b),(r"\bproduct of (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)",lambda a,b:a*b),(r"\bsum of (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)",lambda a,b:a+b),(r"(\d+(?:\.\d+)?) (?:plus|added to|and) (\d+(?:\.\d+)?)",lambda a,b:a+b),(r"\bsubtract (\d+(?:\.\d+)?) from (\d+(?:\.\d+)?)",lambda a,b:b-a),(r"\btake (\d+(?:\.\d+)?) (?:away )?from (\d+(?:\.\d+)?)",lambda a,b:b-a),(r"(\d+(?:\.\d+)?) minus (\d+(?:\.\d+)?)",lambda a,b:a-b),(r"\bdifference between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)",lambda a,b:abs(a-b))]
AR=[(re.compile(a),f) for a,f in AR]
def arith(q:str):
 s=re.sub(r"(?<=\d),(?=\d)",'',(q or '').lower())
 for rx,f in AR:
  m=rx.search(s)
  if m and len(re.findall(r"\d+(?:\.\d+)?",s))==2:
   try:v=f(float(m[1]),float(m[2]))
   except ZeroDivisionError:return None
   return str(int(v)) if float(v).is_integer() else f"{v:.6g}"
 return None
def calc(q:str)->dict:
 p=prime(q) or arith(q)
 if p:return {'store':'calc','answer':p,'forms':[p],'score':0.95,'line':f"Calculator: {p}.",'prov':'calculator','key':('arith',)}
 try:r=__import__('amni.math.answer',fromlist=['solve_math']).solve_math(q)
 except Exception:r=None
 if not r or float(r.get('conf') or 0)<0.6:return {'store':'calc','score':0.0}
 A=__import__('amni.math.answer',fromlist=['fmt']);v=A.fmt(r['value']) if r.get('value') is not None else str(r.get('reply'));return {'store':'calc','answer':v,'forms':[v,str(r.get('reply') or '')],'score':float(r['conf']),'line':'Calculator: '+str(r.get('reply')),'prov':'calculator: '+str(r.get('intent')),'key':(r.get('intent'),)}
_R={}
def resolver(fd=None)->R:
 k=D1.FD if fd is None else fd
 if k not in _R:_R[k]=R(k)
 return _R[k]
def resolve(q:str,fd=None)->dict:
 t0=time.perf_counter();Rs=resolver(fd);o,t,_=toks(q);ql=' '.join(t);C=[]
 c=calc(q) if 'calc' not in ABL else {'score':0.0};c['score']>0 and C.append(c)
 rel=csrel(ql) if 'cs' not in ABL else '';csm=bool(rel) and not any(o[k][:1].isupper() for k in range(1,len(o)))
 w=Rs.wd(q) if Rs.D and not c['score'] else {'score':0.0,'ents':[],'props':[]}
 w.get('answer') and C.append(w)
 if rel and (csm or w['score']<CF['tau']) and not c['score']:
  z=cs(Rs,q);z['score']>0 and C.append({**z,'score':z['score']+(CF['cs'] if csm else 0.0)})
 C.sort(key=lambda z:-(z['score']+(10 if z['store']=='calc' else 0)+(10 if z['store']=='concepts' and csm else 0)))
 best=C[0] if C else None;ok=bool(best) and (best['store']=='calc' or best['store']=='concepts' and best['score']>=CF['cs'] or best['store']=='wd' and not best['absent'])
 best and best['store']=='wd' and best.update(sure=best['score']>=CF['tau'])
 return {'q':q,'abstain':not ok,'best':best if ok else None,'cands':C,'wd':w,'ms':round((time.perf_counter()-t0)*1000,3)}
def material(q:str)->dict:
 r=resolve(q);b=r['best'];return {'route':'store1','lines':[b['line']] if b else [],'extract':(b['line'] if b else ''),'prov':[b.get('prov','')] if b else [],'conf':float(b['score']) if b else 0.0,'must':[b['answer']] if b else [],'abstain':r['abstain'],'ms':r['ms']}
