import re,itertools
from fractions import Fraction as Fr
from amni.math.word import quantities,lem,_f,TOK,STOP as STOPW
_COMMON=set("if the on in how what a an he she they it there this that these those each every after before when then his her their its at for of to and but so all some one two three four five six seven eight nine ten eleven twelve twenty thirty during while since by with as from yesterday today tomorrow last next first finally also later now we i you my our your mr mrs ms dr miss given find calculate determine half both twice only once out over because although however instead meanwhile overall total altogether together again still just monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december here several many most few other another no not do does did is are was were will would can could should must let suppose assume consider use using round note which who whose where why christmas halloween easter thanksgiving valentine new year years english spanish french american dollars usd am pm tv ok hi hello mom dad grandma grandpa mother father sister brother uncle aunt unfortunately fortunately luckily currently initially originally recently usually afterward afterwards eventually additionally furthermore moreover therefore thus hence besides either neither nor yet whether though upon onto into within without between among through throughout across around about above below under behind beside near far up down off per than more less fewer much any such own same very too quite rather almost nearly exactly approximately roughly even ever never always often sometimes already soon according as science math art history music".split())
_RATK={'HALF','TWICE','TRIPLE','QUAD','THIRD','QUARTER','FIFTH','DOZEN'}
_MULT={'twice':Fr(2),'double':Fr(2),'thrice':Fr(3),'triple':Fr(3),'half':Fr(1,2),'quadruple':Fr(4)}
_FRW={'half':Fr(1,2),'third':Fr(1,3),'quarter':Fr(1,4),'fourth':Fr(1,4),'fifth':Fr(1,5),'tenth':Fr(1,10)}
_PLUS=r"(?:more|longer|taller|older|heavier|bigger|larger|higher|greater|further|farther|extra|faster|later)";_MINUS=r"(?:fewer|less|shorter|younger|lighter|smaller|lower|cheaper|slower|earlier)"
_VL=[('buy','G','buys bought buy buying'),('get','G','gets got get getting gotten'),('receive','G','receives received receive'),('find','G','finds found find'),('collect','G','collects collected collect'),('pick','G','picks picked pick'),('earn','G','earns earned earn'),('make','G','makes made make'),('bake','G','bakes baked bake'),('add','G','adds added add'),('win','G','wins won win'),('gain','G','gains gained gain'),('catch','G','catches caught catch'),('grow','G','grows grew grow'),('save','G','saves saved save'),('borrow','G','borrows borrowed borrow'),('bring','G','brings brought bring'),('gather','G','gathers gathered gather'),('harvest','G','harvests harvested harvest'),('produce','G','produces produced produce'),('score','G','scores scored score'),('lay','G','lays laid lay'),('give','L','gives gave give given'),('sell','L','sells sold sell'),('eat','L','eats ate eat eaten'),('use','L','uses used use'),('lose','L','loses lost lose'),('spend','L','spends spent spend'),('donate','L','donates donated donate'),('break','L','breaks broke break broken'),('throw','L','throws threw throw thrown'),('pay','L','pays paid pay'),('drink','L','drinks drank drink'),('share','L','shares shared share'),('lend','L','lends lent lend'),('discard','L','discards discarded discard'),('remove','L','removes removed remove'),('drop','L','drops dropped drop'),('burn','L','burns burned burnt burn'),('waste','L','wastes wasted waste'),('return','L','returns returned return'),('mail','L','mails mailed mail'),('send','L','sends sent send'),('trade','L','trades traded trade'),('read','A','reads read'),('walk','A','walks walked walk'),('run','A','runs ran run'),('drive','A','drives drove drive driven'),('swim','A','swims swam swim'),('write','A','writes wrote write written'),('jump','A','jumps jumped jump'),('ride','A','rides rode ride'),('travel','A','travels traveled travelled travel'),('practice','A','practices practiced practice'),('play','A','plays played play'),('work','A','works worked work'),('type','A','types typed type'),('watch','A','watches watched watch'),('climb','A','climbs climbed climb'),('paint','A','paints painted paint'),('plant','A','plants planted plant'),('cook','A','cooks cooked cook'),('wash','A','washes washed wash'),('knit','A','knits knitted knit'),('sew','A','sews sewed sew'),('fold','A','folds folded fold'),('do','A','does did do done'),('complete','A','completes completed complete'),('finish','A','finishes finished finish'),('solve','A','solves solved solve'),('answer','A','answers answered answer'),('hike','A','hikes hiked hike'),('jog','A','jogs jogged jog'),('bike','A','bikes biked bike'),('pack','A','packs packed pack'),('deliver','A','delivers delivered deliver'),('clean','A','cleans cleaned clean'),('build','A','builds built build'),('fill','A','fills filled fill'),('visit','A','visits visited visit'),('take','A','takes took take taken'),('post','A','posts posted post'),('sing','A','sings sang sing'),('draw','A','draws drew draw'),('shoot','A','shoots shot shoot'),('kick','A','kicks kicked kick'),('sleep','A','sleeps slept sleep'),('study','A','studies studied study'),('teach','A','teaches taught teach'),('see','A','sees saw see seen'),('count','A','counts counted count'),('order','G','orders ordered order'),('rent','A','rents rented rent'),('hire','A','hires hired hire'),('invite','A','invites invited invite'),('put','A','puts put'),('deposit','G','deposits deposited deposit'),('withdraw','L','withdraws withdrew withdraw withdrawn')]
VERB={f:(l,c) for l,c,fs in _VL for f in fs.split()}
_VRE=r"(?:%s)"%'|'.join(sorted(VERB,key=len,reverse=True))
_HAVE=r"(?:has|have|had|owns|owned|own|is|are|was|were|weighs|weighed|holds|contains|contained|costs|cost|measures|needs|need|wants|want|there)"
_VERBISH=re.compile(r"\b(?:%s|%s|will|can|could|would|should|must|does|did|decides|decided|plans|planned|hopes|takes|took|goes|went|comes|came|keeps|kept|leaves|left|stays|becomes|became|starts|started|begins|began|gets|got|charges|charged|equals|totals|totaled|totalled)\b"%(_VRE,_HAVE))
_CUE=re.compile(r"#\d+|\banother\s(?:set|round|batch|order|one|time|trip|bag|box)\b|\bagain\b|\ba\s(?:pound|dollar|kilogram|gallon|liter|cup|bag|box|pack|pair|quart|few|couple)\b|\b(?:twice|half|double|doubled|triple|tripled|thrice|third|thirds|quarter|quarters|fourth|fourths|fifth|fifths|times|more|less|fewer|percent|each|per|every|remaining|rest|remainder|left|dozen|same|as many|as much)\b")
_K=r"(#\d+|twice|double|thrice|triple|quadruple|half)"
_VGL='|'.join(sorted([f for f,(l,c) in VERB.items() if c in 'GL'],key=len,reverse=True))
class Lin(dict):
 def __add__(s,o):
  r=Lin(s)
  for k,v in (o.items() if isinstance(o,dict) else [('',Fr(o))]):r[k]=r.get(k,Fr(0))+v
  return Lin({k:v for k,v in r.items() if v!=0})
 def __rmul__(s,c):return Lin({k:v*Fr(c) for k,v in s.items() if v*Fr(c)!=0})
 def __neg__(s):return (-1)*s
 def __sub__(s,o):return s+(-(o if isinstance(o,Lin) else Lin({'':Fr(o)})))
def V(name):return Lin({name:Fr(1)})
def C(v):return Lin({'':Fr(v)}) if v else Lin()
def solve_lin(eqs:list):
 rows=[dict(e) for e in eqs];vs=sorted({k for e in rows for k in e if k});piv={}
 for x in vs:
  r=next((i for i,e in enumerate(rows) if e.get(x,0)!=0 and i not in piv.values()),None)
  if r is None:continue
  e=rows[r];c=e[x];e={k:v/c for k,v in e.items()};rows[r]=e;piv[x]=r
  for i in range(len(rows)):
   if i!=r and rows[i].get(x,0)!=0:f=rows[i][x];rows[i]={k:v for k,v in ((k,rows[i].get(k,0)-f*e.get(k,0)) for k in set(rows[i])|set(e)) if v!=0}
 if any(set(e)=={''} for e in rows):return None
 return {x:-rows[r].get('',Fr(0)) for x,r in piv.items() if not any(k and k!=x for k in rows[r])}
def ev(l:Lin,sol:dict):return None if any(k and k not in sol for k in l) else sum((v*(sol[k] if k else 1) for k,v in l.items()),Fr(0))
def normalize(text:str):
 text=re.sub(r"(?i)\bwhat's\b","what is",text);text=re.sub(r"(?i)\b(one|two|three|four|five|six|seven|eight|nine)-(half|halves|thirds?|fourths?|quarters?|fifths?|sixths?|eighths?|tenths?)\b",r"\1 \2",text);text=re.sub(r"\b([A-Z][a-z]+),?\s(?:who|which)\s(?=is|are|was|were|has|had)",r"\1. \1 ",text)
 info=quantities(text);qs=info['qs'];pos={q['t0']:k for k,q in enumerate(qs) if q['kind'] not in _RATK};skip={j for q in qs if q['kind'] not in _RATK for j in range(q['t0']+1,q['tok']+1)};toks=info['toks'];out=[];names=[];last=None;cur=-1
 for i,(t,si) in enumerate(toks):
  if si!=cur:cur=si;out.append([])
  if i in skip:continue
  lw=t.lower();nm=re.sub(r"'s$",'',lw)
  if i in pos:out[-1].append(f"#{pos[i]}")
  elif t[:1].isupper() and nm not in _COMMON and re.fullmatch(r"[A-Za-z]+(?:'s)?",t) and nm not in VERB:
   if out[-1] and out[-1][-1].startswith('@') and not out[-1][-1].endswith("'s"):out[-1][-1]='@'+nm+("'s" if lw.endswith("'s") else '');names[-1:]=[nm];last=nm;continue
   names.append(nm) if nm not in names else None;last=nm;out[-1].append('@'+nm+("'s" if lw.endswith("'s") else ''))
  elif lw in ('i','me'):out[-1].append('@me');names.append('me') if 'me' not in names else None
  elif lw=='my' and i+1<len(toks) and toks[i+1][0].lower() in ('money','savings','allowance','salary','paycheck','budget','wallet','account','collection','total','earnings','income'):out[-1].append("@me's")
  elif lw in ('he','she') and last:out[-1].append('@'+last)
  elif lw in ('his','her') and last and i+1<len(toks) and toks[i+1][0].lower() in ('money','savings','allowance','salary','paycheck','budget','wallet','account','collection','total','earnings','income'):out[-1].append('@'+last+"'s")
  else:out[-1].append(lw)
 return info,[' '.join(s) for s in out],names
def _clauses(sent:str)->list:
 parts=re.split(r"(\s(?:,\s)?(?:and then|and|but|while|whereas|so|then|after that)\s|\s[,;:]\s)",sent);out=[];pend=''
 for i in range(0,len(parts),2):
  x=(pend+' '+parts[i]).strip() if pend else parts[i].strip();sep=parts[i-1].strip(' ,') if i else ''
  if not x:continue
  if not _VERBISH.search(x) and i+1<len(parts) and not out:pend=x+' '+parts[i+1].strip();continue
  vp=re.match(r"^(.*?\b(?:%s))\s(?:away\s)?#\d+\s(?:on|for|to|at)\b"%_VGL,out[-1]) if out and not _VERBISH.search(x) and re.match(r"^(?:another\s)?#\d+\s(?:on|for|to|at)\s",x) else None
  pend='';(out.append(x) if _VERBISH.search(x) or not out else out.append(vp.group(1)+' '+x) if vp else out.__setitem__(-1,out[-1]+' '+(sep+' ' if sep else '')+x))
 return out+([pend] if pend else [])
class Prob:
 def __init__(s,text):
  s.info,s.ss,s.names=normalize(text);s.qs=s.info['qs'];s.qi=s.info['qi'];s.eqs=[];s.used=set();s.ver={};s.trace=[];s.vars=[];s.start={};s.base={};s.goal=None;s.pertot=None;s.group=[];s.subj=None;s.unparsed=[];s.events=[];s.lastamt=None;s.uk={}
 def cur(s,n):n=re.sub(r"'s$",'',n);s.vars.append(n) if n not in s.vars else None;return V(f"{n}@{s.ver.get(n,0)}")
 def bump(s,n):
  n=re.sub(r"'s$",'',n);s.vars.append(n) if n not in s.vars else None;s.start.setdefault(n,V(f"{n}@0"));s.base.setdefault(n,V(f"{n}@0"));s.ver[n]=s.ver.get(n,0)+1;return V(f"{n}@{s.ver[n]}")
 def use(s,*toks):s.used|={t for t in toks if t and t.startswith('#')}
 def val(s,t):return qs_val(t,s.qs)
 def eq(s,l,r,msg):s.eqs.append(l-r);s.trace.append(msg)
 def ref(s,n,c=''):
  n=re.sub(r"'s$",'',n);mv=re.search(r"\b(%s)\b"%_VRE,c) if c else None;ev_=[e[2] for e in s.events if mv and e[0]==n and e[1]==VERB[mv.group(1)][0] and VERB[mv.group(1)][1]=='L']
  return sum(ev_,Lin()) if ev_ else s.base[n] if n in s.base else s.cur(n)
 def kconf(s,n,q):
  k='m' if q['kind']=='MONEY' else 'c' if q['kind'] in ('NUM','WORD') else None;o=s.uk.setdefault(n,k) if k else None;return bool(k and o and o!=k)
 def known_set(s,n):return any(t.startswith(n+' = ') for t in s.trace)
 def known(s,n):return n in s.ver or any(k.startswith(n+'@') for e in s.eqs for k in e)
 def event(s,n,verb,amt):s.events.append((n,verb,amt));s.lastamt=amt
def qs_val(t,qs):return qs[int(t[1:])]['v'] if t and t.startswith('#') else _MULT.get(t) if t in _MULT else _FRW.get(t.split()[-1]) if t else None
def _pct(t,qs):return bool(t) and t.startswith('#') and qs[int(t[1:])]['kind']=='PCT'
def _frac(t,p):return p.val(t)/100 if _pct(t,p.qs) else p.val(t)
_FRACP=r"(half|a\sthird|one\sthird|a\squarter|one\squarter|a\sfourth|one\sfourth|a\sfifth|one\sfifth|a\stenth|#\d+)"
def frame(c:str,p:Prob)->bool:
 c=re.sub(r"^(?:(?:if|since|given that|suppose|now|so|then|also|and|but|after that|in addition|additionally|meanwhile|finally|later|when|however|unfortunately|luckily|in total|overall)\s(?:,\s)?)+",'',c.strip());c=re.sub(r"^(?:on|in|at|during|by|over|for|after|before)\s(?:the\s)?(?:[a-z]+\s){0,2}?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|night|day|week|month|year|weekend|first|second|third|last|next|end|beginning)\s?(?:[a-z]+\s)?,?\s",'',c)
 m=re.match(r"^(?:(?:then|also|only|later|now|finally|still|both|all|so|and)\s)?(?:(?:his|her|their|the|a|an|my|our)\s(?:[a-z]+\s){0,2})?@(\w+(?:'s)?)",c);S=re.sub(r"'s$",'',m.group(1)) if m else p.subj if re.match(r"^(?:he|she|it|they|then|also|only|later|%s|%s)\b"%(_VRE,_HAVE),c) and not c.startswith('there') else None;p.subj=S or p.subj;nums=re.findall(r"#\d+",c);qs=p.qs
 if re.search(r"\b(?:all but|product|ratio|consecutive|digits?|squared?|average)\b|times\s(?:more|less|fewer|smaller|bigger|larger|greater|higher|lower)\s(?:[a-z@']+\s){0,4}?than|#\d+\s-\s[a-z]",c):return False
 mg=re.search(r"(?:^|\s)(gives|gave|give|lends|lent|sends|sent|hands|handed|passes|passed)\s@(\w+)\s(?:(?:another|also)\s)?(#\d+|twice|double|triple|thrice|half)\s(?:times\s)?(?:the\snumber\sof\s|as\smany\s|as\smuch\s)?",c)
 if mg and S and len(nums)<=1:
  x=mg.group(2);amt=C(p.val(mg.group(3))) if mg.group(3).startswith('#') else (p.val(mg.group(3))*p.lastamt if p.lastamt is not None else None)
  if amt is None:return False
  ox=p.cur(x);p.eq(p.bump(x),ox+amt,f"{x} + {_f(p.val(mg.group(3)))}{'' if mg.group(3).startswith('#') else ' × previous'}");p.use(mg.group(3));p.lastamt=amt;p.events.append((x,'receive',amt))
  if p.known(S):os_=p.cur(S);p.eq(p.bump(S),os_-amt,f"{S} - given")
  return True
 if len(nums)==1 and re.search(r"(?:for|in|about|over|during|after)\s(?:about\s)?%s\s(?:hours?|minutes?|days?|weeks?|months?|years?|seconds?)\b"%nums[0],c):return False
 m=re.search(r"^((?:@\w+\s(?:,\s)?)+)(?:and\s)(@\w+)\s(each\s)?(?:[a-z]+\s){0,3}?(#\d+)",c)
 if m and len(nums)==1 and not re.search(r"\b(?:than|times|twice|half|as many|as much)\b",c):
  ws=re.findall(r"@(\w+)",m.group(1)+m.group(2));n=p.val(m.group(4));p.group=ws;p.use(m.group(4))
  if re.search(r"\b(?:need|needs|want|wants|goal|hope|hopes|plan|plans)\b",c):p.goal=(ws,n);p.trace.append(f"goal {' + '.join(ws)} = {_f(n)}");return True
  if m.group(3) or re.search(r"\beach\b",c):[p.eq(p.cur(w),C(n),f"{w} = {_f(n)}") for w in ws];return True
  p.eq(sum((p.cur(w) for w in ws),Lin()),C(n),f"{' + '.join(ws)} = {_f(n)}");return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,6}?(#\d+)\s(?:[a-z\-]+\s){0,2}?(%s|%s)\s(?:[a-z\-]+\s){0,3}?than\s%s\s(?:times\s)?(?:as\s(?:many|much)|the\snumber\sof|the\samount\sof|what|that\sof)?\s?(?:[a-z\-]+\s){0,3}?@(\w+)"%(_PLUS,_MINUS,_K),c)
 if m and m.group(1)!=m.group(5):
  sg=1 if re.fullmatch(_PLUS,m.group(3)) else -1;k=p.val(m.group(4));n=p.val(m.group(2));p.eq(p.cur(m.group(1)),k*p.ref(m.group(5),c)+C(sg*n),f"{m.group(1)} = {_f(k)} × {m.group(5)} {'+' if sg>0 else '-'} {_f(n)}");p.use(m.group(2),m.group(4));return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,5}?(#\d+)\s(?:[a-z\-]+\s){0,3}?(%s|%s)\s(?:[a-z\-]+\s){0,3}?than\s(?:the\s)?(?:[a-z\-]+\s){0,2}?((?:@\w+(?:'s)?\s(?:[a-z\-]+\s){0,2}?(?:,\s)?(?:and\s)?){2,4})(?:[a-z\-]+\s){0,2}?(?:combined|together|put together|total)"%(_PLUS,_MINUS),c)
 if m:
  ws=[re.sub(r"'s$",'',x) for x in re.findall(r"@(\w+(?:'s)?)",m.group(4))];sg=1 if re.fullmatch(_PLUS,m.group(3)) else -1;n=p.val(m.group(2));p.eq(p.cur(m.group(1)),sum((p.cur(w) for w in ws),Lin())+C(sg*n),f"{m.group(1)} = {' + '.join(ws)} {'+' if sg>0 else '-'} {_f(n)}");p.use(m.group(2));return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,5}?(#\d+)\stimes\s(%s|%s)\s(?:[a-z\-]+\s){0,3}?than\s(?:[a-z\-]+\s){0,3}?@(\w+)"%(_PLUS,_MINUS),c)
 if m and m.group(1)!=m.group(4):k=p.val(m.group(2));p.eq(p.cur(m.group(1)),k*p.cur(m.group(4)),f"{m.group(1)} = {_f(k)} × {m.group(4)}");p.use(m.group(2));return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,5}?(#\d+)\s(?:[a-z\-]+\s){0,3}?(%s|%s)\s(?:[a-z\-]+\s){0,3}?than\s(?:[a-z\-]+\s){0,3}?(@\w+|he|she)\b"%(_PLUS,_MINUS),c)
 if m:
  o=p.subj if m.group(4) in ('he','she') else m.group(4)[1:];sg=1 if re.fullmatch(_PLUS,m.group(3)) else -1;n=qs[int(m.group(2)[1:])];pct=n['kind']=='PCT'
  if o and o!=m.group(1):p.eq(p.cur(m.group(1)),(1+sg*n['v']/100)*p.ref(o,c) if pct else p.ref(o,c)+C(sg*n['v']),f"{m.group(1)} = {o} {'+' if sg>0 else '-'} {_f(n['v'])}{'%' if pct else ''}");p.use(m.group(2));return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,5}?%s\s(?:times\s)?(?:as\s(?:many|much|long|far|old|big|tall|heavy|fast|large|expensive)|the\s(?:number|amount|price|cost|size|age|length|weight)\sof|what|that\sof|the)\s(?:[a-z\-]+\s){0,4}?(?:as\s|than\s)?@(\w+)"%_K,c)
 if m and m.group(1)!=m.group(3):k=p.val(m.group(2));p.eq(p.cur(m.group(1)),k*p.ref(m.group(3),c),f"{m.group(1)} = {_f(k)} × {m.group(3)}");p.use(m.group(2));return True
 m=re.search(r"@(\w+)(?:'s)?\s(?:[a-z#@']+\s){0,5}?%s\s(?:of\s)(?:the\s)?(?:[a-z\-]+\s){0,3}?(?:as\s|that\s|what\s)?@(\w+)"%_FRACP,c)
 if m and m.group(1)!=m.group(3) and not re.search(r"\b(?:%s)\b|\bto\s@"%_VRE,c[m.start():m.end()]) and (not m.group(2).startswith('#') or qs[int(m.group(2)[1:])]['kind'] in ('PCT','FRAC')):
  f=_frac(m.group(2),p);p.eq(p.cur(m.group(1)),f*p.ref(m.group(3),c),f"{m.group(1)} = {_f(f)} × {m.group(3)}");p.use(m.group(2));return True
 m=re.search(r"(?:a\s)?(?:total|combined|together|altogether|in all|sum)\s(?:[a-z@'\-,]+\s){0,9}?(?:of\s|is\s|are\s|was\s|were\s|equals?\s|equal\sto\s)?(#\d+)|(#\d+)\s(?:[a-z@'\-]+\s){0,3}?(?:in total|altogether|combined|together|in all|total)\b",c)
 if m and len(nums)==1:
  n=m.group(1) or m.group(2);ws=[re.sub(r"'s$",'',x) for x in re.findall(r"@(\w+(?:'s)?)",c)];p.group=list(dict.fromkeys(ws)) if len(set(ws))>=2 else p.group;p.pertot=p.val(n);p.use(n);p.trace.append(f"total = {_f(p.val(n))}");return True
 m=re.search(r"(?:^|\s)(%s)\s(?:away\s)?(?:%s\s(?:of\s)(?:the\s)?(rest|remaining|remainder|what\sis\sleft|what\swas\sleft|what\s(?:@\w+|she|he|they)\s(?:has|had|have)\s(?:left)?|(?:money|amount|cash|[a-z]+)\s(?:@\w+|he|she|they)\s(?:had|has|have)\s(?:left|remaining)|[a-z@'\s]*?)|(the\srest|the\sremaining|the\sremainder|all\sthe\srest|everything\selse))(?:\s|$)"%(_VRE,_FRACP),c)
 if m and S and len(nums)<=1 and VERB[m.group(1)][1] in 'GL' and (not m.group(2) or not m.group(2).startswith('#') or qs[int(m.group(2)[1:])]['kind'] in ('PCT','FRAC')):
  lem_,cls=VERB[m.group(1)];rest=bool(m.group(4)) or bool(m.group(3) and re.match(r"rest|remain|what",m.group(3)));f=Fr(1) if m.group(4) else _frac(m.group(2),p)
  old=p.cur(S);ref=old if rest else p.base.get(S,old);amt=f*ref;new=p.bump(S);p.eq(new,old+(amt if cls=='G' else -amt),f"{S} {'+' if cls=='G' else '-'} {_f(f)} of {'what is left' if rest else 'it'}");p.event(S,lem_,amt);p.use(m.group(2))
  for m2 in re.finditer(r"(?:,\s|and\s)%s\s(?:of\s)(?:the\s)?(rest|remaining|remainder|what\sis\sleft|it|them)\b"%_FRACP,c[m.end():]):
   f2=_frac(m2.group(1),p);old=p.cur(S);ref=old if re.match(r"rest|remain|what",m2.group(2)) else p.base.get(S,old);amt=f2*ref;p.eq(p.bump(S),old+(amt if cls=='G' else -amt),f"{S} {'+' if cls=='G' else '-'} {_f(f2)} of {m2.group(2)}");p.event(S,lem_,amt);p.use(m2.group(1))
  return True
 m=re.search(r"(?:wants|want|needs|need|hopes|plans|aims|goal\sis|trying|has\sto|have\sto)\s(?:to\s)?(?!pay|spend|give|donate|use|lose)(?:[a-z]+\s){0,4}?(#\d+)(?!\s(?:[a-z]+\s)?of\b)",c)
 if m and len(nums)==1 and S and qs[int(m.group(1)[1:])]['kind'] not in ('PCT','FRAC') and not re.search(r"\b(?:each|per|every|than|times|more)\b",c):p.goal=([S],p.val(m.group(1)));p.use(m.group(1));p.trace.append(f"goal {S} = {_f(p.val(m.group(1)))}");return True
 m=re.search(r"(?:took|takes|take|stole|steals|steal|borrowed|borrows|ate|eats)\s(?:away\s)?(#\d+)\s(?:[a-z\-]+\s){0,3}?from\s@(\w+)",c)
 if m and len(nums)==1 and re.sub(r"'s$",'',m.group(2))!=S:x=m.group(2);old=p.cur(x);p.eq(p.bump(x),old-C(p.val(m.group(1))),f"{x} - {_f(p.val(m.group(1)))}");p.use(m.group(1));return True
 if re.search(r"\b(?:all but|times (?:more|less|fewer) than)\b",c):return False
 m=re.search(r"(?:^|\s)(%s)\s(?:away\s|out\s|off\s|up\s|back\s)?(?:(?:only|another|an\sadditional|additional|a\stotal\sof)\s)?(?:[a-z@]+\s){0,2}?(#\d+)\b"%_VRE,c)
 if m and S and len(nums)==1 and not re.search(r"\b(?:each|per|every|than|times|twice|half)\b",c[m.end():]) and not re.search(r"\b(?:each|per|every|a\sday|a\sweek|an\shour|a\smonth|a\syear)\b",c):
  lem_,cls=VERB[m.group(1)];nq=qs[int(m.group(2)[1:])];cls='L' if lem_ in ('buy','order','rent') and nq['kind']=='MONEY' and p.known(S) else cls
  if (nq['kind'] in ('PCT','FRAC') and cls=='A') or re.match(r"^@\w+'s\s(?:[a-z]+\s)?(?:coach|teacher|mom|dad|mother|father|brother|sister|friend|friends|boss|parents|wife|husband|son|daughter|grandma|grandpa|grandmother|grandfather|uncle|aunt|cousin|neighbor|team|class|family|dog|cat|boyfriend|girlfriend|kids|children|classmates|roommate|partner)\b",c) or p.kconf(S,{**nq,'kind':'MONEY'} if lem_ in ('spend','pay','earn','save','donate','lend','borrow') and nq['kind']=='NUM' else nq):return False
  if nq['kind'] in ('PCT','FRAC') and cls in 'GL':
   if not p.known(S):return False
   if re.match(r"\s(?:[a-z]+\s)?(?:more|less|fewer|higher|lower)\b",c[m.end():]):return False
   old=p.cur(S);f=nq['v']/100 if nq['kind']=='PCT' else nq['v'];amt=f*(old if cls=='G' else p.base.get(S,old));p.eq(p.bump(S),old+(amt if cls=='G' else -amt),f"{S} {'+' if cls=='G' else '-'} {_f(f)} of it");p.event(S,lem_,amt);p.use(m.group(2));return True
  if re.search(r"\bfor\s(?:a|an|one|each|every)\b",c[m.end():]):return False
  n=p.val(m.group(2));mc=re.match(r"\s(?:[a-z]+\s)?(more|fewer|less)\b",c[m.end():])
  if mc and not re.search(r"\bthan\b",c[m.end():]):
   prev=next((e for e in reversed(p.events) if e[0]!=S),None) if mc.group(1)!='more' or not p.known(S) else None
   if prev is not None and not p.known(S):p.eq(p.cur(S),prev[2]+C(n if mc.group(1)=='more' else -n),f"{S} = {prev[0]} {'+' if mc.group(1)=='more' else '-'} {_f(n)}");p.use(m.group(2));p.event(S,lem_,prev[2]+C(n if mc.group(1)=='more' else -n));return True
   if mc.group(1)!='more' or not p.known(S):return False
  p.use(m.group(2));p.event(S,lem_,C(n))
  mt=re.search(r"\bto\s@(\w+)",c[m.end():]) if lem_ in ('give','lend','send','sell','mail','donate','pass') else None
  if mt and mt.group(1)!=S and p.known(mt.group(1)):x=mt.group(1);ox=p.cur(x);p.eq(p.bump(x),ox+C(n),f"{x} + {_f(n)}")
  if cls in 'GA' and p.known(S):old=p.cur(S);p.eq(p.bump(S),old+C(n),f"{S} + {_f(n)}")
  elif cls in 'GA':p.eq(p.cur(S),C(n),f"{S} = {_f(n)}");p.base[S]=p.cur(S)
  else:old=p.cur(S);p.eq(p.bump(S),old-C(n),f"{S} - {_f(n)}")
  return True
 m=re.search(r"(?:^|\s)(%s)\s(?:in\s)?(?:[a-z]+\s){0,3}?(twice|double|thrice|triple|half|#\d+\stimes)\s(?:as\smany|as\smuch|that\snumber|that\samount|the\snumber|the\samount)\b"%_VRE,c)
 if m and S and p.lastamt is not None and not re.search(r"@\w+",c[m.end():]):
  lem_,cls=VERB[m.group(1)];cls='G' if re.search(r"\b(?:throws|throw|threw|puts|put|drops|dropped)\sin\b",c) else cls;k=p.val(m.group(2).split()[0]);amt=k*p.lastamt;p.use(m.group(2).split()[0])
  prev=p.events[-1][0] if p.events else None
  if prev!=S and not p.known(S):p.eq(p.cur(S),amt,f"{S} = {_f(k)} × {prev}")
  elif p.known(S):old=p.cur(S);p.eq(p.bump(S),old+(-amt if cls=='L' else amt),f"{S} {'-' if cls=='L' else '+'} {_f(k)} × previous")
  else:p.eq(p.cur(S),amt,f"{S} = {_f(k)} × previous")
  p.event(S,lem_,amt);return True
 m=re.search(r"(?:^|\s)%s\s(?:only\s)?(?:[a-z]+\s){0,3}?(#\d+)\b"%_HAVE,c) or re.search(r"^@\w+(?:'s)?\s(?:[a-z]+\s){0,2}?(#\d+)\b",c)
 if m and S and len(nums)==1 and re.search(r"\b(?:before|at first|originally|initially|at the start|in the beginning|to begin with|to start with)\b",c) and not re.search(r"\b(?:each|per|every|than|times|twice|half|%s|%s)\b"%(_PLUS,_MINUS),c):
  p.start.setdefault(S,V(f"{S}@0"));p.eq(p.start[S],C(p.val(m.group(1))),f"{S} started with {_f(p.val(m.group(1)))}");p.use(m.group(1));p.vars.append(S) if S not in p.vars else None;return True
 if m and S and len(nums)==1 and not re.search(r"\b(?:each|per|every|than|times|twice|half|left|weekdays?|daily|weekly|monthly|yearly|hourly|%s|%s)\b"%(_PLUS,_MINUS),c) and not re.search(r"\b(?:a|an)\s(?:day|week|month|year|hour|minute)\b",c) and not re.match(r"^@\w+'s\s(?:[a-z]+\s)?(?:coach|teacher|mom|dad|mother|father|brother|sister|friend|friends|boss|parents|wife|husband|son|daughter|grandma|grandpa|grandmother|grandfather|uncle|aunt|cousin|neighbor|team|class|family|dog|cat|boyfriend|girlfriend|kids|children|classmates|roommate|partner)\b",c) and not p.kconf(S,qs[int(m.group(1)[1:])]):
  p.eq(p.cur(S),C(p.val(m.group(1))),f"{S} = {_f(p.val(m.group(1)))}");p.base[S]=p.cur(S);p.use(m.group(1));return True
 m=re.search(r"(?:^|\s)(?:left\s)?(?:with\s)?(#\d+)\s(?:[a-z\-]+\s){0,3}?left\b|(?:has|have|had|is|are)\s(?:only\s)?(#\d+)\s(?:[a-z\-]+\s){0,3}?(?:left|remaining)\b",c)
 if m and S and len(nums)==1:n=m.group(1) or m.group(2);p.eq(p.cur(S),C(p.val(n)),f"{S} ends with {_f(p.val(n))}");p.use(n);return True
 return False
def _ask(p):
 q0=p.ss[p.qi] if p.ss else '';mq=re.search(r"\b(?:how|what|find|calculate|determine|compute|figure out)\b",q0);return (q0[:mq.start()],q0[mq.start():]) if mq else ('',q0)
def _target(p:Prob):
 pre,q=_ask(p)
 for c in [y for x in re.split(r"\s,\s",re.sub(r"^(?:if|in total|altogether|given that|assuming)\s,?\s?",'',pre)) for y in _clauses(x) if _CUE.search(y)]:frame(c,p) or p.unparsed.append(c)
 ws=list(dict.fromkeys(re.sub(r"'s$",'',x) for x in re.findall(r"@(\w+(?:'s)?)",q)));vw=[w for w in ws if w in p.vars]
 if ws and not vw:return None,None
 sh=re.search(r"\bnext year\b",q) and Fr(1) or (lambda mm:mm and (p.val(mm.group(1)) if mm.group(2) is None else -p.val(mm.group(1))))(re.search(r"(?:in\s(#\d+)\syears?|(#\d+)\syears?\s(?:from now|later))(?!\s)|(#\d+)\syears?\s(ago)",q)) if re.search(r"\bhow old\b|\bage\b",q) else None
 if sh is not None and vw and not re.search(r"\b(?:more|less|fewer|than|total|sum|combined|together|twice|times|half)\b",q):
  mm=re.search(r"#\d+",q);p.use(mm.group(0)) if mm else None;return p.cur(vw[0])+C(sh),f"{vw[0]} {'+' if sh>=0 else '-'} {_f(abs(sh))} years"
 if re.search(r"#\d+|\b(?:twice|double|triple|thrice|third|quarter|fifth|percent|times|each|per|average|ratio|percentage|required|tie|tying|chance|if|change|profit|loss|discount|tip|tax|interest|commission|rate|speed)\b",q) or re.search(r"\bhalf\b(?! as)",q):return None,None
 m=re.search(r"how (?:many|much)\s(more|fewer|less)\s(?:[a-z\-]+\s){0,4}?(?:does|did|do|will|would|can|could|is|are|was|were|has|had)\s@(\w+)(?:'s)?\s(?:[a-z\-]+\s){0,6}?(?:than|compared\sto|compared\swith)\s@(\w+)",q) or re.search(r"how (?:many|much)\s(more|fewer|less)\s(?:[a-z\-]+\s){0,4}?than\s@(\w+)\s(?:[a-z\-]+\s){0,4}?(?:does|did|do|will|would|can|could|is|are|was|were|has|had)\s@(\w+)",q)
 if m:
  a_,b_=(m.group(2),m.group(3)) if 'than @'+m.group(2) not in q else (m.group(3),m.group(2));a_,b_=(a_,b_) if m.group(1)=='more' else (b_,a_);return p.cur(a_)-p.cur(b_),f"{a_} - {b_}"
 m=re.search(r"as\s(?:large|many|much|big|tall|old|long)\sas\s@(\w+)",q)
 if m and re.search(r"\b(?:need|needs|have to|has to|must)\b",q) and vw and vw[0]!=m.group(1):return p.cur(m.group(1))-p.cur(vw[0]),f"{m.group(1)} - {vw[0]}"
 if re.search(r"\bhow (?:many|much)\s(?:more|fewer|less)\b|\bdifference\b|\b(?:faster|slower|longer|shorter|older|younger|taller|heavier|lighter|cheaper|farther|further|earlier|later|bigger|smaller|larger)\b",q):
  if p.goal and re.search(r"\b(?:need|needs|should|must|have to|has to|still|left to|remain)\b",q):who=p.goal[0];return C(p.goal[1])-sum((p.cur(w) for w in who),Lin()),f"goal - ({' + '.join(who)})"
  return None,None
 if re.search(r"\b(?:start|started|begin|began|beginning|initially|originally|at first|to start with|in the beginning)\b",q):return (p.start[vw[0]],f"{vw[0]} at start") if vw and vw[0] in p.start else (None,None)
 m=re.search(r"\b(?:share|shared|split|divide|divided|distribute|distributed)\s(?:them\s|it\s|the\s[a-z]+\s)?(?:equally\s|evenly\s)?(?:between|among|amongst)\s(?:themselves|them|the\s(?:two|three|four)|both)",q)
 if m and len(p.vars)>=2:return Fr(1,len(p.vars))*sum((p.cur(w) for w in p.vars),Lin()),f"({' + '.join(p.vars)}) ÷ {len(p.vars)}"
 mv=re.search(r"\b(%s)\b"%_VRE,q);ev_=[e for e in p.events if mv and e[1]==VERB[mv.group(1)][0] and (not vw or e[0]==vw[0])] if mv else []
 if mv and VERB[mv.group(1)][1]=='G' and len(vw)==1 and vw[0] in p.start and not any(e[0]==vw[0] and e[1]==VERB[mv.group(1)][0] for e in p.events) and not p.known_set(vw[0]):return p.start[vw[0]],f"{vw[0]} at start"
 if ev_ and len(vw)==1 and (len(ev_)>=2 or VERB[mv.group(1)][1]=='A') and not re.search(r"\bleft\b|\bremain|\b(?:they|altogether|total|combined|together|in all|both)\b",q):return sum((e[2] for e in ev_),Lin()),f"sum of {VERB[mv.group(1)][0]} events"
 conj=re.findall(r"((?:@\w+(?:'s)?\s(?:,\s)?)+)(?:and\s)@(\w+)",q)
 if conj:who=[w for w in dict.fromkeys([re.sub(r"'s$",'',x) for a_,b_ in conj for x in re.findall(r"@(\w+(?:'s)?)",a_)+[b_]]) if w in p.vars];return (sum((p.cur(w) for w in who),Lin()),' + '.join(who)) if len(who)>=2 else (None,None)
 if re.search(r"\b(?:in total|altogether|combined|together|in all|total|both|all|sum|they|everyone|the friends)\b",q) and p.pertot is None:
  who=vw if len(vw)>=2 else p.group if len(p.group)>=2 and not vw else list(p.vars) if not vw else vw
  return (sum((p.cur(w) for w in who),Lin()),' + '.join(who)) if who else (None,None)
 if vw:return p.cur(vw[0]),vw[0]
 m=re.search(r"how (?:many|much)\s(?:[a-z\-]+\s){0,3}?(?:does|did|do|is|are|was|were|will)\s(?:the\s)?(?:[a-z]+\s)?(\w+)",q)
 if m and m.group(1) in p.vars:return p.cur(m.group(1)),m.group(1)
 return (p.cur(p.vars[0]),p.vars[0]) if not ws and len(p.vars)==1 and re.search(r"\b(?:left|remain|remaining|now|have|has|in the end|at the end)\b",q) else (None,None)
_CONV={('minute','hour'):Fr(1,60),('hour','minute'):Fr(60),('cent','dollar'):Fr(1,100),('dollar','cent'):Fr(100),('inch','foot'):Fr(1,12),('foot','inch'):Fr(12),('day','week'):Fr(1,7),('week','day'):Fr(7),('month','year'):Fr(1,12),('year','month'):Fr(12),('ounce','pound'):Fr(1,16),('pound','ounce'):Fr(16),('second','minute'):Fr(1,60),('minute','second'):Fr(60),('hour','day'):Fr(1,24),('day','hour'):Fr(24)}
_UNITW={'minute':'minute','minutes':'minute','hour':'hour','hours':'hour','cent':'cent','cents':'cent','dollar':'dollar','dollars':'dollar','inch':'inch','inches':'inch','foot':'foot','feet':'foot','day':'day','days':'day','week':'week','weeks':'week','month':'month','months':'month','year':'year','years':'year','ounce':'ounce','ounces':'ounce','pound':'pound','pounds':'pound','second':'second','seconds':'second','mile':'mile','miles':'mile','yard':'yard','yards':'yard','meter':'meter','meters':'meter','kilometer':'km','kilometers':'km','gallon':'gallon','gallons':'gallon','quart':'quart','quarts':'quart','cup':'cup','cups':'cup'}
_GEN=set('do does did are is was were will can could would should have has had centimeter meter kilometer inch foot yard mile gram kilogram pound ounce liter gallon cup cm km kg mm lb oz more fewer less much many times piece item thing person people dollar cent time year month week day hour minute second unit total point object they them it one member student'.split())
def ulem2(w):return {'people':'person','children':'child','men':'man','women':'woman','feet':'foot','teeth':'tooth'}.get(w) or lem(w)
def compile_problem(text:str):
 p=Prob(text)
 if not p.ss or len(p.qs)<1:return None
 for i,s in enumerate(p.ss):
  if i==p.qi:continue
  for c in _clauses(s):frame(c,p) or (p.unparsed.append(c) if _CUE.search(c) or (re.search(r"\b(?:%s)\s(?:some|several|a\sfew|many|more|the\srest|his|her|their)\b"%_VGL,c) and not re.search(r"#\d+",c)) else None)
 if p.pertot is not None:
  who=p.group if len(p.group)>=2 else list(p.vars)
  if not who:return None
  p.eqs.append(sum((p.cur(w) for w in who),Lin())-C(p.pertot))
 t,tq=_target(p)
 if t is None:return None
 _,q=_ask(p);mt=re.search(r"how many\s(?:more\s|fewer\s)?()(?!(?:do|does|did|are|is|were|was|will|can|could|would|should|have|has|had)\b)([a-z]+)\b",q)
 body=' '.join(x for i,x in enumerate(p.ss) if i!=p.qi)+' '+_ask(p)[0]
 if mt and ulem2(mt.group(2)) not in _GEN and ulem2(mt.group(2)) not in {ulem2(w) for w in re.findall(r"[a-z]+",body)}:p.unparsed.append(f"target noun {mt.group(2)} not in body")
 _,q=_ask(p);mu=re.search(r"how (?:many|much)\s(?:more\s)?(\w+)",q);qu=_UNITW.get(mu.group(1)) if mu else None
 nu={_UNITW.get(x['r'][0]) for x in p.qs if x['r'] and x['kind'] not in _RATK}-{None};f=_CONV.get((next(iter(nu)),qu),Fr(1)) if qu and len(nu)==1 and qu not in nu else Fr(1)
 if qu and nu and qu not in nu and f==1:p.unparsed.append(f"unit {sorted(nu)}->{qu}")
 if len(nu)>1 and qu:p.unparsed.append(f"mixed units {sorted(nu)}")
 sol=solve_lin(p.eqs) if p.eqs else {}
 if sol is None:return None
 v=ev(t,sol)
 if v is None:return None
 v=v*f;nums={f"#{i}" for i,q_ in enumerate(p.qs) if q_['kind'] not in _RATK}
 return {'value':v,'eqs':p.trace+([f"convert × {_f(f)}"] if f!=1 else []),'target':tq,'sol':{k:_f(x) for k,x in sol.items()},'all_used':nums<=p.used,'unused':sorted(nums-p.used),'verified':all(ev(e,sol)==0 for e in p.eqs if all(k=='' or k in sol for k in e)),'neg':any(x<0 for x in sol.values()),'unparsed':p.unparsed,'n_eq':len(p.eqs),'norm':p.ss}
_POSTV={'away':'L','off':'L','popped':'L','taken':'L','burst':'L','broke':'L','died':'L','escaped':'L','melted':'L','on':'G','joined':'G','arrived':'G','back':'G','withdrawal':'L','withdrawals':'L','deposit':'G','deposits':'G'}
_CMPW={'older':1,'more':1,'longer':1,'taller':1,'heavier':1,'extra':1,'younger':-1,'fewer':-1,'less':-1,'shorter':-1,'lighter':-1}
_HAVEW=set('has have had holds hold held contains contain contained is are was were owe owes owed needs need there started start starts'.split());_AUXW=set('do does did are is were was will can could would should have has had'.split());_OBJ=set('her him them me us'.split());_RATK2={'HALF','THIRD','QUARTER','FIFTH'};_DIVW=set('split share shared sharing divide divided among between evenly equally'.split());_CUT={'and',',','.','?','!','then','but'};_MOV=set('get gets got flew fly flies ran went go goes came come comes hopped hop climbed jumped walked swam rode sailed moved wandered'.split())
def _rr(x):return list(itertools.takewhile(lambda w:w not in _CUT,x['r']))
def _sg(x):
 l=x['l'];j=next((k for k in range(len(l)-1,-1,-1) if l[k] in VERB),None);pre=VERB[l[j]][1] if j is not None else None;rx=_rr(x)[:3];po=next((_POSTV[w] for k,w in enumerate(rx) if w in _POSTV and (w not in ('away','off','on','back') or k and (rx[k-1] in VERB or rx[k-1] in _MOV))),None) if pre is None or VERB[l[j]][0] in ('get','go','come') else None;pre='G' if pre=='L' and VERB[l[j]][0] in ('give','send','lend','mail','pass') and j+1<len(l) and l[j+1] in _OBJ else pre
 return po or ('G' if pre=='A' else pre)
def _per(x,Q):
 l,r=x['l'],x['r']
 if 'each' in l:j=len(l)-1-l[::-1].index('each');nx=l[j+1] if j+1<len(l) else '';return (lem(nx) if nx and nx not in VERB and nx not in _HAVEW else lem(l[j-1]) if j else ''),None
 if r[:1]==['each']:return next((lem(w) for w in reversed(l) if w.isalpha() and w not in ('at','for','cost','costs','are','is','each','a','an','the')),''),None
 n=next((y for y in Q if y['i']==x['i']+1 and y['v']==1 and y['s']==x['s'] and y['l'][-1:]==['for']),None);k=next((k for k in range(len(r)-1) if lem(r[k+1])==x['per'] and r[k] in ('per','each','every','a','an')),None)
 return (n['unit'],n['i']) if n else (x['per'] if x['per'] and (x['per']!=x['unit'] or x['kind']=='MONEY') and (x['per'] not in VERB or k and r[k-1] in ('in','on','for','at','of','per') or k+2>=len(r) or not r[k+2].isalpha()) and k is not None and (r[k] in ('per','each','every') or x['per'] in _UNITW or any(y['unit']==x['per'] for y in Q)) else '',None)
def _seg(l):k=max([i for i,w in enumerate(l) if w in ('and',',','then',';')]+[-1]);return l[k+1:],k>=0
def _vb(l):
 g,_=_seg(l);w=next((w for w in reversed(g) if w in VERB or w in _HAVEW),None);return (w,'H') if w in _HAVEW and w not in VERB else (w,VERB[w][1]) if w else (None,None)
def _un(x):return {lem(w) for w in x['r'][:2] if w.isalpha() and w not in STOPW|_CUT|_MOV|_HAVEW|_POSTV.keys()|_CMPW.keys()|{'my','your','our','each','per','times','another','more','left'} and w not in VERB}
def _seq(text:str):
 I=quantities(text);ss=I['sents'];qi=I['qi'];Q=[dict(x,i=k) for k,x in enumerate(I['qs'])]
 nm={w.lower() for w in re.findall(r"\b[A-Z][a-z']+",text) if w.lower() not in _COMMON and w.lower() not in VERB and w.lower() not in STOPW}
 if not ss or not Q or len(Q)>4 or len(text.split())>32 or len(nm)>2 or re.search(r"(?i)\bthan\b|\b(?:half|third|quarter) as (?:many|much)\b|\bhow (?:many|much) (?:more|fewer|less)\b|\bpercent|%|\baverage\b|\bboth\b|\bfor every\b|\bwill be\b|\b(?:had|has|have|needs?|wants?) to\b|\bother\b|\d-|/|\bhead start\b|\btimes (?:that|the)\b|[^\x00-\x7f$]|\bone of\b|\bper \w+ per\b",text) or any(x['kind'] in ('PCT','FRAC') for x in Q):return None
 qs=ss[qi];m=re.search(r"(?i)\b(?:how (?:many|much|far|old|long)|what(?:'s| is| was| will)\b)",qs)
 if not m or re.search(r"\b(?:would|instead|if)\b",qs[m.start():].lower()) or any(y['t0']==x['tok']+1 or x['r'][:2]==['of','them'] and x['kind'] not in _RATK2 or x['kind']=='MONEY' and x['r'][:1] and x['r'][0] in _CMPW or 'each' in x['l'] and x['per'] for x in Q for y in Q):return None
 wq=sum(1 for _,s in I['toks'] if s<qi)+len(TOK.findall(qs[:m.start()]));q=qs[m.start():].lower();B=[x for x in Q if x['tok']<wq];A=[x for x in Q if x['tok']>=wq];mt=re.search(r"how many (?:more |fewer |less )?([a-z]+)",q);tu=lem(mt[1]) if mt and mt[1] not in _AUXW else 'year' if q.startswith('how old') else None
 bl={lem(w) for w in re.findall(r"[a-z]+",(' '.join(ss[:qi])+' '+qs[:m.start()]).lower())};tot=bool(re.search(r"\b(?:total|altogether|in all|combined|together)\b",q));each=bool(re.search(r"\beach\b|\bper (?:person|kid|child|friend|student)\b",q));used=set();st=[];pd=re.search(r"\b(?:in|per|a|each|every) (day|week|month|year|hour|minute)\b",q)
 for x in Q:
  if x['i'] in used or x['kind'] in _RATK2:continue
  p,n=_per(x,Q);ofp=next((y for y in B if y['i']==x['i']-1 and y['r'][1:2]==['of'] and x['l'][-1:]==['of']),None)
  if ofp and not p:used|={x['i'],ofp['i']};st.append(dict(x,v=x['v']*ofp['v'],vb=_vb(ofp['l']),why=f"{_f(ofp['v'])} × {_f(x['v'])}"));continue
  if not p:continue
  n is not None and used.add(n);cc=[y for y in A+B if y['i'] not in used|{x['i']} and y['unit']==p and not _per(y,Q)[0]];c=cc[0] if len(cc)==1 else None
  if len(cc)>1:return None
  if c and any(y['i']!=c['i'] and y['i'] not in used and y['s']==c['s'] and y['tok']>c['tok'] and y['kind']==c['kind'] and not _un(y) for y in Q if y['i']!=x['i']):return None
  if c:used|={x['i'],c['i']};st.append(dict(x,tok=max(x['tok'],c['tok']),s=max(x['s'],c['s']),v=x['v']*c['v'],vb=next((v for v in (_vb(c['l']),_vb(x['l'])) if v[1] in ('G','L')),_vb(x['l']) if _vb(x['l'])[0] else _vb(c['l'])),ei='each' in x['l'],why=f"{_f(c['v'])} × {_f(x['v'])}"));continue
  f=_CONV.get((pd[1],p)) if pd else None
  if f:used.add(x['i']);st.append(dict(x,v=x['v']*f,vb=_vb(x['l']),why=f"{_f(x['v'])} × {_f(f)}"));continue
  d=next((y for y in A if y['i'] not in used and y['unit']==x['unit']),None) if tu==p else None
  if not d or len(Q)!=len(used|{x['i'],d['i']}):return None
  v=d['v']/x['v'];return v,[f"{_f(d['v'])} ÷ {_f(x['v'])} = {_f(v)}"]
 if len(st)>1:return None
 V=None;mk=False;ls=None;W=[];dv=2 if each and re.search(r"\b(?:split|share|shared|divide|divided)\b[^.?!]*\bwith (?:her|his|my|their|a|an|the) [a-z]+",text.lower()) else None;fr=False
 for x in sorted([x for x in B if x['i'] not in used]+st,key=lambda z:z['tok']):
  r=_rr(x);l=x['l'];vw,vc=x['vb'] if 'vb' in x else _vb(l);g,jn=_seg(l);cm=next((_CMPW[w] for w in r[:3] if w in _CMPW),None);po=next((_POSTV[w] if w in _POSTV else VERB[w][1] for k,w in enumerate(r[:3]) if w in _POSTV and (w not in ('away','off','on','back') or k and (r[k-1] in VERB or r[k-1] in _MOV)) or w in VERB and VERB[w][1] in 'GL' and k and r[k-1] in ('were','was','are','is','got','get')),None) if vc in (None,'H') or VERB.get(vw,('',''))[0] in ('get','go','come') else None
  s=po or ('G' if vc=='A' and VERB[vw][0] in ('plant','cook','build','paint','draw','write','knit','sew') else 'G' if vc=='L' and VERB[vw][0] in ('give','send','lend','mail','pass') and l[l.index(vw)+1:l.index(vw)+2]!=[] and l[l.index(vw)+1] in _OBJ else vc if vc in ('G','L') else None)
  if x['kind'] in _RATK2:
   pu=next((lem(w) for w in r if w.isalpha() and w not in ('of','them','it','are','is','were','was','the')),'');cp=bool(tu and pu and (tu!=pu and tu not in bl or re.search(r"\bnot "+re.escape(pu),q)) and re.search(r"\bof (?:them|those|these|the \w+) (?:are|were|is|was)\b",' '.join(r)))
   if V is None or fr or not (s=='L' or cp):return None
   pt=V/x['v'];V=V-pt;fr=True;ls=s if s=='L' else None;W.append(f"{_f(V+pt)} - 1/{_f(x['v'])} of it = {_f(V)}");continue
  if x['kind'] in ('TWICE','TRIPLE','QUAD') or r[:1]==['times'] and 'as' in r:
   wn=[w for w in re.findall(r"\b[A-Z][a-z]+",ss[x['s']]) if w.lower() in nm]
   if V is None or r.count('as')!=1 or not tot and not (wn and wn[-1] in qs[m.start():]):return None
   nv=V*x['v'];W.append(f"{_f(V)} × {_f(x['v'])} = {_f(nv)}");V=V+nv if tot else nv;ls=None;continue
  if V is not None and each and any(w in _DIVW for w in l[-3:]+r[:4]):dv=x['v']+(l[-1:]==['with'] or l[-2:-1]==['with']);continue
  if tu and tu in bl and _un(x) and tu not in _un(x) or V is not None and (x['kind']=='MONEY')!=mk and _un(x):return None
  if V is None:
   if x['v']==1 and not x.get('why') and x['kind']=='WORD':return None
   V=x['v'];ps=x['s'];bc=vc;bu=_un(x);mk=x['kind']=='MONEY';ls='G' if vc in ('A','G','H') and not x.get('ei') else None;W.append(f"{x['why']} = {_f(V)}" if x.get('why') else f"start {_f(V)}");continue
  if cm is not None and s is None:
   if vc not in (None,'H') or cm<0 and vc is None:return None
   nv=V+cm*x['v'];W.append(f"{_f(V)} {'+' if cm>0 else '-'} {_f(x['v'])} = {_f(nv)}");V=V+nv if tot else nv;ls=None;continue
  if s=='G' and bc=='H' and mk and VERB.get(vw,('',''))[0] in ('buy','order','rent','pay'):s='L'
  if s is None and vc=='H':return None
  if s is None and not (jn and x['s']==ps and ls and (not _un(x) or not bu or _un(x)&(bu|{tu}))):return None
  if _un(x) and bu and not _un(x)&(bu|{tu}) and not x.get('why'):return None
  s=s or ls;ls=s;ps=x['s'];V=V+x['v'] if s=='G' else V-x['v'];W.append((x['why']+" = "+_f(x['v'])+", " if x.get('why') else '')+f"{'+' if s=='G' else '-'} {_f(x['v'])} = {_f(V)}")
  if V<0:return None
 if V is None or len(W)<2 and not st and not dv or any(x['i'] not in used for x in A) or each and not dv or not each and re.search(r"\b(?:split|splits|share|shares|shared|divide|divides|divided)\b",text.lower()):return None
 if dv:V=V/dv;W.append(f"{_f(V*dv)} ÷ {_f(dv)} = {_f(V)}")
 return None if tu and tu not in bl and not fr else (V,W)
def solve_word(text:str):
 try:r=compile_problem(text)
 except (ValueError,ZeroDivisionError,KeyError,IndexError,TypeError,AttributeError,RecursionError):r=None
 if not (r and r['all_used'] and not r['unparsed'] and r['verified'] and not r['neg'] and r['value']>=0 and (r['value']*100).denominator==1):
  try:z=_seq(text)
  except (ValueError,ZeroDivisionError,KeyError,IndexError,TypeError,AttributeError,RecursionError):z=None
  return {'value':z[0],'reply':f"The answer is {_f(z[0])}. Working: {'; '.join(z[1])}.",'conf':0.75,'path':'seq','trace':{'eqs':z[1],'target':'answer','sol':{}}} if z and z[0]>=0 and (z[0]*100).denominator==1 else None
 v=r['value'];conf=0.85 if v.denominator==1 else 0.65
 return {'value':v,'reply':f"The answer is {_f(v)}. Working: {'; '.join(r['eqs'])}; so {r['target']} = {_f(v)}.",'conf':conf,'path':'relate','trace':{'eqs':r['eqs'],'target':r['target'],'sol':r['sol']}}
