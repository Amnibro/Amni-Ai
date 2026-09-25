import re,math,datetime as dt,calendar
from fractions import Fraction as Fr
from amni.serve.code_safety import safe_arith
from amni.serve.skills import _UNITS,_UAL,_temp_to
from amni.math.sympy_tool import job
from amni.math.relate import solve_word
_ONES={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19}
_TENS={'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
_SCALE={'hundred':100,'thousand':1000,'million':10**6,'billion':10**9,'trillion':10**12}
_NW=re.compile(r"\b(?:(?:%s)(?:[\s-]+(?:and[\s-]+)?(?:%s))*)\b"%(('|'.join([*_ONES,*_TENS,*_SCALE]),)*2),re.I)
_NUM=r"-?\d+(?:\.\d+)?"
_N=r"\$?\s*(-?\d+(?:\.\d+)?)"
_LEAD=re.compile(r"^(?:(?:hey|hi|ok|okay|so|adam|please|quick question|quick|real quick)[,!:]?\s+)*(?:(?:can|could|would|will) you\s+(?:please\s+)?(?:tell me|help me|work out|figure out|calculate|compute|evaluate|solve|find|do|convert|check)?\s*|(?:please\s+)?(?:tell me|i need|i want to know|do you know)\s+|let'?s\s+(?:see|say)\s+)?(?:(?:what(?:'s| is| are|s)|whats|how much is|how much are|how much|calculate|compute|evaluate|work out|figure out|find|determine|give me|do|convert|simplify)\s+)?",re.I)
_FNS=('sqrt','cbrt','factorial','abs','log','ln','log2','exp','sin','cos','tan','floor','ceil','round','pi')
_ARITH_OK=re.compile(r"^(?:[\d\s.+\-*/%%()]|\b(?:%s)\b|\be\b)+$"%'|'.join(_FNS))
_FACTUAL=re.compile(r"(?i)\b(population|born|died|founded|invented|capital|president|king|queen|century|war|located|elected|album|movie|film|song|novel|season|episode|olympics?|world cup|super bowl|in the year|how old is|how tall is|height of|distance (?:from|between) [a-z]+ (?:and|to) [a-z]+|who|when did|where|moons?|planets?|chromosomes?|bones|teeth|legs does|calories in|players on|countries|states in|continents|oceans)\b")
def words_to_num(t:str)->str:
 def cv(m):
  tot=cur=0;ws=[w for w in re.split(r"[\s-]+",m.group(0).lower()) if w and w!='and']
  if len(ws)==1 and ws[0] in ('hundred','thousand','million','billion','trillion'):return m.group(0)
  for w in ws:
   cur=cur+_ONES[w] if w in _ONES else cur+_TENS[w] if w in _TENS else cur*_SCALE[w] if w=='hundred' else 0
   if w in _SCALE and w!='hundred':tot,cur=tot+max(cur,1)*_SCALE[w],0
  return str(tot+cur)
 return _NW.sub(cv,t)
def _fr(v)->Fr:return v if isinstance(v,Fr) else Fr(v) if isinstance(v,int) else Fr(str(v))
def fmt(v,sig:int=6)->str:
 if isinstance(v,float) and (v!=v or v in (float('inf'),float('-inf'))):return str(v)
 if isinstance(v,float) and abs(v-round(v))<1e-9*max(1,abs(v)) and abs(v)<1e15:v=Fr(round(v))
 if isinstance(v,(int,Fr)):
  f=_fr(v)
  if f.denominator==1:return f"{f.numerator:,}" if abs(f.numerator)>=10000 else str(f.numerator)
  d=f.denominator
  for p in (2,5):
   while d%p==0:d//=p
  return (str(float(f)) if len(str(float(f)))<16 else f"{float(f):.10g}") if d==1 and len(str(f.denominator))<=8 else f"{f.numerator}/{f.denominator} ≈ {float(f):.{sig}g}" if f.denominator<1000 and abs(f.numerator)<10**6 else f"{float(f):.{sig}g}"
 s=f"{v:.{sig}g}";return s if 'e' not in s else f"{v:.{sig}e}"
def num(v):return float(v) if isinstance(v,Fr) else v
def _sq(x,k=2):
 x=_fr(x) if isinstance(x,(int,Fr)) else x
 if isinstance(x,Fr) and x>=0:
  a,b=round(x.numerator**(1/k)),round(x.denominator**(1/k))
  for aa in (a-1,a,a+1):
   for bb in (b-1,b,b+1):
    if aa>=0 and bb>0 and aa**k==x.numerator and bb**k==x.denominator:return Fr(aa,bb)
 if float(x)<0 and k==2:raise ValueError('complex root')
 return math.copysign(abs(float(x))**(1/k),float(x))
def _fact(x):
 if _fr(x).denominator!=1 or not 0<=x<=1000:raise ValueError('factorial domain')
 return math.factorial(int(x))
def _fns(deg=False):
 tr=lambda f:(lambda x:f(math.radians(float(x)) if deg else float(x)))
 return {'sqrt':_sq,'cbrt':lambda x:_sq(x,3),'factorial':_fact,'abs':abs,'log':lambda x,b=10:math.log(float(x),float(b)),'ln':lambda x:math.log(float(x)),'log2':lambda x:math.log2(float(x)),'exp':lambda x:math.exp(min(float(x),700)),'sin':tr(math.sin),'cos':tr(math.cos),'tan':tr(math.tan),'floor':lambda x:math.floor(x),'ceil':lambda x:math.ceil(x),'round':lambda x,n=0:round(x,int(n)) if n else round(x),'pi':math.pi,'e':math.e}
def evaluate(expr:str,deg=False,bind=None):
 v=safe_arith(expr,num=lambda c:Fr(str(c)),funcs={**_fns(deg),**(bind or {})})
 return Fr(v.real) if isinstance(v,complex) and not v.imag else v
def _clean(t:str)->str:
 t=t.replace('×','*').replace('⋅','*').replace('·','*').replace('÷','/').replace('−','-').replace('–','-').replace('²','^2').replace('³','^3').replace('√','sqrt ').replace('π','pi').replace('’',"'")
 t=re.sub(r"(?<=\d),(?=\d{3}\b)","",t);return words_to_num(t)
def _strip(t:str)->str:
 t=_LEAD.sub('',t.strip()).strip()
 t=re.sub(r"(?i)^(?:the\s+)?(?:answer|result|value)\s+(?:to|of|for)\s+",'',t);t=re.sub(r"(?i)\s*(?:equals?|=|is)?\s*\??\s*(?:please|thanks|thank you)?(?:[\s?.]|(?<!\d)!)*$",'',t);return t.strip()
_OPW=[(r"\bto the power of\b|\braised to(?: the power of)?\b|\bto the\b(?=\s*\d+\s*(?:st|nd|rd|th)?(?:\s+power)?)","^"),(r"(?<=\d)(?:st|nd|rd|th)\s+power\b|(?<=\d)(?:st|nd|rd|th)\b(?=\s*$)",""),(r"\bmultiplied by\b|\btimes\b|(?<=\d)\s*[x]\s*(?=[\d(])","*"),(r"\bdivided by\b|\bover\b","/"),(r"\bplus\b|\badded to\b","+"),(r"\bminus\b|\bless\b","-"),(r"\bmod(?:ulo)?\b","%"),(r"\bsquared\b","^2"),(r"\bcubed\b","^3")]
def to_expr(t:str):
 e=_strip(_clean(t)).lower()
 e=re.sub(r"(?:the\s+)?square root of\s*",'sqrt ',e);e=re.sub(r"(?:the\s+)?cube root of\s*",'cbrt ',e)
 e=re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+",r"\1/100*",e);e=re.sub(r"(\d+(?:\.\d+)?)\s*%",r"(\1/100)",e)
 e=re.sub(r"\bhalf of\s+",'1/2*',e);e=re.sub(r"\b(?:a|one) third of\s+",'1/3*',e);e=re.sub(r"\b(?:a|one) quarter of\s+",'1/4*',e);e=re.sub(r"\b(?:twice|double)\s+(?=\d)",'2*',e);e=re.sub(r"\btriple\s+(?=\d)",'3*',e)
 m=re.fullmatch(r"(?:the\s+)?(sum|product|difference|quotient)\s+(?:of|between)\s+(%s)\s+and\s+(%s)"%(_NUM,_NUM),e)
 e={'sum':f"{m.group(2)}+{m.group(3)}",'product':f"{m.group(2)}*{m.group(3)}",'difference':f"{m.group(2)}-{m.group(3)}",'quotient':f"{m.group(2)}/{m.group(3)}"}[m.group(1)] if m else e
 m=re.fullmatch(r"(add|subtract|take away|multiply|divide)\s+(%s)\s+(?:and|to|from|by|into)\s+(%s)"%(_NUM,_NUM),e)
 e={'add':f"{m.group(2)}+{m.group(3)}",'subtract':f"{m.group(3)}-{m.group(2)}",'take away':f"{m.group(3)}-{m.group(2)}",'multiply':f"{m.group(2)}*{m.group(3)}",'divide':f"{m.group(2)}/{m.group(3)}"}[m.group(1)] if m else e
 for p,r in _OPW:e=re.sub(p,r,e)
 e=re.sub(r"(\d+|\))\s*!",r"factorial(\1)",e);e=re.sub(r"(\d+)\s+factorial\b",r"factorial(\1)",e)
 e=re.sub(r"\b(sqrt|cbrt|ln|log2|log|exp|sin|cos|tan|abs)\s+(\d+(?:\.\d+)?)",r"\1(\2)",e);e=re.sub(r"\b(sqrt|cbrt|ln|log2|log|exp|sin|cos|tan|abs)\s+\(",r"\1(",e)
 e=re.sub(r"\s*(?:degrees|degree|deg|°)",'',e) if re.search(r"\b(?:sin|cos|tan)\b",e) else e
 e=re.sub(r"(\d)\s*(\(|pi\b|sqrt|cbrt)",r"\1*\2",e);e=re.sub(r"\)\s*(\(|\d)",r")*\1",e);e=e.replace('^','**').strip(' =?.')
 return e if e and re.search(r"\d",e) and _ARITH_OK.match(e) and (re.search(r"[+\-*/%]",e.lstrip('-')) or re.search(r"\b(?:%s)\b"%'|'.join(_FNS),e)) else None
def _pretty(e:str)->str:return re.sub(r"\s*\^\s*",'^',re.sub(r"\s+",' ',e.replace('**','^').replace('*',' × ').replace('/',' ÷ ').replace('+',' + ').replace(' -',' - ')).replace('( ','(').replace(' )',')')).strip()
def solve_arith(t:str):
 e=None if re.fullmatch(r"(?i)\s*(?:what(?:'s| is)\s+)?(?:9/11|24/7|7/11|911|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})\s*\??\s*",t) else to_expr(t)
 if e is None:return None
 try:v=evaluate(e,deg=bool(re.search(r"(?i)degree|°|\bdeg\b",t)))
 except OverflowError:return {'kind':'arith','value':None,'reply':"That number is too large for me to compute exactly (it goes past my 8,192-bit size guard).",'conf':0.9}
 except (ValueError,ZeroDivisionError,TypeError,SyntaxError,ArithmeticError) as x:return {'kind':'arith','value':None,'reply':"That's undefined (division by zero)." if isinstance(x,ZeroDivisionError) else None,'conf':0.9} if isinstance(x,ZeroDivisionError) else None
 if isinstance(v,complex):return None
 return {'kind':'arith','value':v,'reply':f"{_pretty(e)} = {fmt(v)}",'conf':0.99}
_PCT=[(r"(?:what\s+)?(?:percent(?:age)?|%%)\s+of\s+%s\s+is\s+%s"%(_N,_N),lambda a,b:(b/a*100,'pct',f"{fmt(b)} is {{v}}% of {fmt(a)} ({fmt(b)} ÷ {fmt(a)} × 100)")),
 (r"%s\s+is\s+what\s+(?:percent(?:age)?|%%)\s+of\s+%s"%(_N,_N),lambda a,b:(a/b*100,'pct',f"{fmt(a)} is {{v}}% of {fmt(b)} ({fmt(a)} ÷ {fmt(b)} × 100)")),
 (r"(?:what\s+(?:percent(?:age)?|%%)\s+is\s+)?%s\s*(?:out of|/)\s*%s(?:\s+(?:as|in)\s+(?:a\s+)?percent(?:age)?)?"%(_N,_N),lambda a,b:(a/b*100,'pct',f"{fmt(a)} out of {fmt(b)} is {{v}}% ({fmt(a)} ÷ {fmt(b)} × 100)")),
 (r"%s\s+is\s+%s\s*(?:%%|percent)\s+of\s+what(?:\s+number)?"%(_N,_N),lambda a,b:(a/(b/100),'num',f"{fmt(a)} is {fmt(b)}% of {{v}} ({fmt(a)} ÷ {fmt(b/100)})")),
 (r"(?:the\s+)?percent(?:age)?\s+(?:change|increase|decrease|difference|growth|drop)\s+from\s+%s\s+to\s+%s"%(_N,_N),lambda a,b:((b-a)/a*100,'pct',f"from {fmt(a)} to {fmt(b)} is a {{v}}% {'increase' if b>=a else 'change'} (({fmt(b)} - {fmt(a)}) ÷ {fmt(a)} × 100)")),
 (r"(?:increase|raise|mark up|grow)\s+%s\s+by\s+%s\s*(?:%%|percent)"%(_N,_N),lambda a,b:(a*(1+b/100),'num',f"{fmt(a)} increased by {fmt(b)}% is {{v}} ({fmt(a)} × {fmt(1+b/100)})")),
 (r"(?:decrease|reduce|lower|mark down|discount|cut)\s+%s\s+by\s+%s\s*(?:%%|percent)"%(_N,_N),lambda a,b:(a*(1-b/100),'num',f"{fmt(a)} decreased by {fmt(b)}% is {{v}} ({fmt(a)} × {fmt(1-b/100)})")),
 (r"%s\s*(?:%%|percent)\s+off\s+(?:of\s+)?(?:a\s+)?%s"%(_N,_N),lambda a,b:(b*(1-a/100),'num',f"{fmt(a)}% off {fmt(b)} is {{v}} (you save {fmt(b*a/100)}: {fmt(b)} × {fmt(1-a/100)})")),
 (r"%s\s*(?:\+|plus)\s*%s\s*(?:%%|percent)"%(_N,_N),lambda a,b:(a*(1+b/100),'num',f"{fmt(a)} plus {fmt(b)}% is {{v}} ({fmt(a)} × {fmt(1+b/100)})")),
 (r"%s\s*(?:-|minus)\s*%s\s*(?:%%|percent)"%(_N,_N),lambda a,b:(a*(1-b/100),'num',f"{fmt(a)} minus {fmt(b)}% is {{v}} ({fmt(a)} × {fmt(1-b/100)})")),
 (r"(?:a\s+)?%s\s*(?:%%|percent)\s+(tip|tax|gratuity|commission|interest|discount|deposit)\s+(?:on|of|for)\s+(?:a\s+)?%s"%(_N,_N),None),
 (r"%s\s*(?:%%|percent)\s+of\s+%s"%(_N,_N),lambda a,b:(a/100*b,'num',f"{fmt(a)}% of {fmt(b)} is {{v}} ({fmt(a/100)} × {fmt(b)})"))]
def solve_percent(t:str):
 s=_strip(_clean(t)).lower().replace(',','')
 if not re.search(r"%|percent",s) and not re.search(r"out of",s):return None
 for p,f in _PCT:
  m=re.fullmatch(r"(?:what\s+is\s+|how much is\s+|calculate\s+)?"+p+r"(?:\s*(?:dollars|usd|bucks))?",s)
  if not m:continue
  if f is None:
   a,kind,b=Fr(m.group(1)),m.group(2),Fr(m.group(3));v=a/100*b;tot=b-v if kind=='discount' else b+v
   return {'kind':'percent','value':v,'reply':f"A {fmt(a)}% {kind} on {fmt(b)} is {fmt(v)} ({fmt(a/100)} × {fmt(b)}); the total comes to {fmt(tot)}.",'conf':0.97}
  try:v,k,msg=f(Fr(m.group(1)),Fr(m.group(2)))
  except ZeroDivisionError:return None
  return {'kind':'percent','value':v,'reply':msg.replace('{v}',fmt(v)),'conf':0.98}
 return None
_UX={'degrees fahrenheit':'f','degree fahrenheit':'f','°f':'f','deg f':'f','degrees celsius':'c','degree celsius':'c','°c':'c','deg c':'c','degrees kelvin':'k','fl oz':'floz','fluid ounce':'floz','fluid ounces':'floz','square feet':'ft2','square foot':'ft2','sq ft':'ft2','square meters':'m2','square metres':'m2','square meter':'m2','sq m':'m2','square kilometers':'km2','square miles':'mi2','square mile':'mi2','square inches':'in2','acres':'acre','hectares':'ha','hectare':'ha','km/h':'kmh','km/hr':'kmh','kilometers per hour':'kmh','kilometres per hour':'kmh','miles per hour':'mph','meters per second':'mps','metres per second':'mps','m/s':'mps','feet per second':'fps','knots':'kn','tablespoons':'tbsp','tablespoon':'tbsp','teaspoons':'tsp','teaspoon':'tsp','cups':'cup','pints':'pt','quarts':'qt','gallons':'gal','tons':'ton','tonnes':'tonne','stones':'st','stone':'st','nautical miles':'nmi','nautical mile':'nmi','light years':'ly','light year':'ly','light-years':'ly','milligrams':'mg','milligram':'mg','micrograms':'ug','micrometers':'um','nanometers':'nm','millimetres':'mm','centimetres':'cm','metres':'m','metre':'m','kilometre':'km','millilitres':'ml','millilitre':'ml','milliseconds':'ms','millisecond':'ms','microseconds':'us','hrs':'h','hr':'h','kilobyte':'kb','megabyte':'mb','gigabyte':'gb','terabyte':'tb','petabytes':'pb','kibibytes':'kib','mebibytes':'mib','gibibytes':'gib','byte':'b','bit':'bit','kilojoules':'kj','kilocalories':'kcal','kcals':'kcal','watt hours':'wh','kilowatt hours':'kwh','kilowatts':'kw','horsepower':'hp','megawatts':'mw','atmospheres':'atm','bars':'bar','kilopascals':'kpa','kilonewtons':'kn','pounds force':'lbf','kilohertz':'khz','megahertz':'mhz','gigahertz':'ghz','inch':'in','week':'week','weeks':'week','month':'month','months':'month','day':'day','decade':'decade','decades':'decade','century':'century','centuries':'century','ton':'ton','kilometers':'km','fortnight':'fortnight','fortnights':'fortnight'}
_TIMEX={'month':2629800,'decade':315576000,'century':3155760000,'fortnight':1209600}
def unit_key(u:str):
 u=re.sub(r"\s+",' ',(u or '').strip().lower().rstrip('.?!'));u=re.sub(r"^(?:a|an|one|the)\s+",'',u)
 for c in (u,u[:-1] if u.endswith('s') else u,u[:-2] if u.endswith('es') else u):
  k=_UX.get(c) or _UAL.get(c) or c
  if k in ('c','f','k') or k in _TIMEX or any(k in tb for tb in _UNITS.values()):return k
 return None
_EXACTU={'kmh':Fr(5,18),'kn':Fr(1852,3600),'gal':Fr('3.785411784'),'qt':Fr('0.946352946'),'pt':Fr('0.473176473'),'cup':Fr('0.2365882365'),'floz':Fr('0.0295735295625'),'tbsp':Fr('0.01478676478125'),'tsp':Fr('0.00492892159375'),'st':Fr('6.35029318'),'oz':Fr('0.028349523125'),'mi2':Fr('2589988.110336'),'acre':Fr('4046.8564224'),'ft2':Fr('0.09290304'),'ly':Fr(9460730472580800),'au':Fr(149597870700),'hp':Fr('745.69987158227022'),'psi':Fr('6894.757293168'),'mmhg':Fr('133.322387415'),'torr':Fr(101325,760),'lbf':Fr('4.4482216152605'),'btu':Fr('1055.05585262'),'m3':Fr(1000),'cm3':Fr(1,1000),'year':Fr(31557600)}
def _factor(fr_,to):
 for cat,tb in _UNITS.items():
  T={**tb,**(_TIMEX if cat=='time' else {}),**{k:v for k,v in _EXACTU.items() if k in tb}}
  if fr_ in T and to in T:return cat,_fr(T[fr_])/_fr(T[to])
 return None,None
_UW=r"((?:(?!(?:are|is|in|to|into|there|as|make|makes|equal|equals|per)\b)[a-z°/]+\s?){1,3}?)"
def solve_units(t:str):
 s=_strip(_clean(t)).lower().replace(',','')
 m=re.fullmatch(r"(?:convert\s+)?%s\s*%s\s+(?:to|in|into|in to|as|=|equals how many|is how many|are how many)\s+%s"%(_N,_UW,_UW),s)
 m2=None if m else re.fullmatch(r"how many\s+%s\s+(?:are\s+)?(?:there\s+)?(?:in|is|are|make up|equals?|per)\s+(?:(a|an|one|%s)\s*)?%s"%(_UW,_NUM,_UW),s)
 v,fu,tu=(Fr(m.group(1)),m.group(2),m.group(3)) if m else (Fr(1) if m2.group(2) in (None,'a','an','one') else Fr(m2.group(2)),m2.group(3),m2.group(1)) if m2 else (None,None,None)
 a,b=unit_key(fu),unit_key(tu)
 if v is None or not a or not b:return None
 if a in ('c','f','k') or b in ('c','f','k'):
  r=_temp_to(float(v),a,b) if a in ('c','f','k') and b in ('c','f','k') else None
  L={'c':'°C','f':'°F','k':'K'};w={('f','c'):f"({fmt(v)} - 32) × 5/9",('c','f'):f"{fmt(v)} × 9/5 + 32",('c','k'):f"{fmt(v)} + 273.15",('k','c'):f"{fmt(v)} - 273.15"}.get((a,b),'via kelvin')
  return {'kind':'units','value':r,'reply':f"{fmt(v)} {L[a]} is {fmt(round(r,6))} {L[b]} ({w})",'conf':0.98} if r is not None else None
 cat,k=_factor(a,b)
 if cat is None:return None
 r=v*k;note=' (using 365.25-day years)' if 'year' in (a,b) or a in ('decade','century') or b in ('decade','century') else ' (using average-length months)' if 'month' in (a,b) else ''
 return {'kind':'units','value':r,'reply':f"{fmt(v)} {fu.strip()} is {fmt(r)} {tu.strip()} ({fmt(v)} × {fmt(k)}){note}",'conf':0.97}
_MON={m.lower():i for i,m in enumerate(calendar.month_name) if m}|{m.lower():i for i,m in enumerate(calendar.month_abbr) if m}|{'sept':9}
_MONR=r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_HOL=r"(?:christmas(?:\s+day|\s+eve)?|new\s+year'?s?(?:\s+day|\s+eve)?|halloween|valentine'?s?(?:\s+day)?|independence\s+day|the\s+fourth\s+of\s+july|july\s+4(?:th)?|thanksgiving|the\s+end\s+of\s+the\s+year|the\s+new\s+year)"
_DATE=r"(today|tomorrow|yesterday|now|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|%s\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?%s(?:,?\s+\d{4})?|%s\s+\d{4}|%s)"%(_MONR,_MONR,_MONR,_HOL)
def _thanks(y):d=dt.date(y,11,1);return d+dt.timedelta(days=(3-d.weekday())%7+21)
def parse_date(s:str,today:dt.date,future=False):
 s=re.sub(r"\s+",' ',s.strip().lower().replace(',',''));y0=today.year
 if s in ('today','now'):return today
 if s in ('tomorrow','yesterday'):return today+dt.timedelta(days=1 if s=='tomorrow' else -1)
 h=[(r"christmas eve",lambda y:dt.date(y,12,24)),(r"christmas",lambda y:dt.date(y,12,25)),(r"new year'?s? eve|the end of the year",lambda y:dt.date(y,12,31)),(r"new year|the new year",lambda y:dt.date(y+(1 if future else 0),1,1)),(r"halloween",lambda y:dt.date(y,10,31)),(r"valentine",lambda y:dt.date(y,2,14)),(r"independence|fourth of july|july 4",lambda y:dt.date(y,7,4)),(r"thanksgiving",_thanks)]
 for p,f in h:
  if re.fullmatch(r"(?:%s)(?:'?s)?(?:\s+day)?"%p,s):d=f(y0);return f(y0+1) if future and d<today and 'new year' not in p else d
 m=re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})",s) or None
 if m:return dt.date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
 m=re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?",s)
 if m:y=int(m.group(3)) if m.group(3) else None;y=y+2000 if y is not None and y<100 else y;d=dt.date(y or y0,int(m.group(1)),int(m.group(2)));return d.replace(year=y0+1) if y is None and future and d<today else d
 m=re.fullmatch(r"(%s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?"%_MONR,s) or re.fullmatch(r"(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(%s)(?:\s+(\d{4}))?"%_MONR,s)
 if m:
  a,b=(m.group(1),m.group(2)) if not m.group(1).isdigit() else (m.group(2),m.group(1));mo=_MON.get(a[:3] if a[:4]!='sept' else 'sept') or _MON.get(a);y=int(m.group(3)) if m.group(3) else None
  d=dt.date(y or y0,mo,int(b));return d.replace(year=y0+1) if y is None and future and d<today else d
 m=re.fullmatch(r"(%s)\s+(\d{4})"%_MONR,s)
 return dt.date(int(m.group(2)),_MON.get(m.group(1)[:3]),1) if m else None
def _dstr(d:dt.date)->str:return f"{calendar.day_name[d.weekday()]}, {calendar.month_name[d.month]} {d.day}, {d.year}"
def _addm(d:dt.date,months:int)->dt.date:
 y,m=divmod(d.month-1+months,12);y+=d.year;return dt.date(y,m+1,min(d.day,calendar.monthrange(y,m+1)[1]))
def solve_date(t:str,today:dt.date|None=None):
 today=today or dt.date.today();s=_strip(_clean(t)).lower().replace(',','');s=re.sub(r"\s+",' ',s)
 m=re.search(r"is (\d{4}) a leap year|was (\d{4}) a leap year|will (\d{4}) be a leap year",s)
 if m:y=int(next(g for g in m.groups() if g));L=calendar.isleap(y);return {'kind':'date','value':int(L),'reply':f"{'Yes' if L else 'No'}, {y} {'is' if L else 'is not'} a leap year ({'divisible by 4' if y%100 else 'divisible by 400' if L else 'a century year not divisible by 400'}{'' if L or y%4==0 else ' fails: not divisible by 4'}).",'conf':0.99}
 m=re.search(r"how many (days|weeks|months|years|hours|minutes|seconds) (?:are there |is it |are left |left |remain )?(?:between|from) %s (?:and|to|until|till) %s"%(_DATE,_DATE),s)
 if m:
  a,b=parse_date(m.group(2),today),parse_date(m.group(3),today)
  if a and b:
   n=(b-a).days;u=m.group(1);v={'days':Fr(n),'weeks':Fr(n,7),'hours':Fr(n*24),'minutes':Fr(n*1440),'seconds':Fr(n*86400)}.get(u)
   v=Fr((b.year-a.year)*12+b.month-a.month-(1 if b.day<a.day else 0)) if u=='months' else Fr(b.year-a.year-(1 if (b.month,b.day)<(a.month,a.day) else 0)) if u=='years' else v
   return {'kind':'date','value':abs(v),'reply':f"There are {fmt(abs(v))} {u} from {_dstr(min(a,b))} to {_dstr(max(a,b))} ({abs(n)} days apart).",'conf':0.98}
 m=re.search(r"how (?:many|much) (days|weeks|hours|sleeps|months)(?: (?:are|is))?(?: there| it| left| remain)?(?: is it)? (?:until|till|til|before|to) %s"%_DATE,s) or re.search(r"(?:how long|how many days)(?: is it)? (?:until|till|before) %s"%_DATE,s)
 if m:
  g=m.groups();u=g[0] if len(g)>1 else 'days';d=parse_date(g[-1],today,True)
  if d:n=(d-today).days;v=Fr(n,7) if u=='weeks' else Fr(n*24) if u=='hours' else Fr(n);return {'kind':'date','value':v,'reply':f"{fmt(v)} {'days' if u=='sleeps' else u} until {_dstr(d)} (counting from today, {_dstr(today)}).",'conf':0.97}
 m=re.search(r"how many (days|weeks|years|months) (?:has it been |have passed |ago was |since )?(?:since|ago was) %s"%_DATE,s) or re.search(r"how (?:long|many days) (?:has it been |ago was )?since %s"%_DATE,s)
 if m:
  g=m.groups();u=g[0] if len(g)>1 else 'days';d=parse_date(g[-1],today)
  if d:n=(today-d).days;v=Fr(n,7) if u=='weeks' else Fr(n) if u=='days' else Fr((today.year-d.year)*12+today.month-d.month-(1 if today.day<d.day else 0)) if u=='months' else Fr(today.year-d.year-(1 if (today.month,today.day)<(d.month,d.day) else 0));return {'kind':'date','value':v,'reply':f"{fmt(v)} {u} since {_dstr(d)} (as of today, {_dstr(today)}).",'conf':0.97}
 m=re.search(r"(?:(\d+) (days?|weeks?|months?|years?) (from|after|before|ago|since|past) ?%s?|in (\d+) (days?|weeks?|months?|years?))"%_DATE,s)
 if m and not s.startswith('how many') and len(s.split())<=14 and re.search(r"\bwhat(?:'s| is| will| was| would)? (?:the )?(?:date|day)\b|\bwhich (?:date|day)\b|\bwhen\b|\bdate\b|\bwhat day\b",s):
  n,u,dr,ds=(int(m.group(1)),m.group(2),m.group(3),m.group(4)) if m.group(1) else (int(m.group(5)),m.group(6),'from',None)
  b=parse_date(ds or 'today',today);sg=-1 if dr in ('before','ago') else 1
  if b and n<=100000:
   u=u.rstrip('s');r=_addm(b,sg*n*(12 if u=='year' else 1)) if u in ('month','year') else b+dt.timedelta(days=sg*n*(7 if u=='week' else 1))
   return {'kind':'date','value':Fr(r.toordinal()),'reply':f"{n} {u}{'s' if n!=1 else ''} {'before' if sg<0 else 'after'} {_dstr(b)} is {_dstr(r)}.",'conf':0.97}
 m=re.search(r"what day(?: of the week)? (?:is|was|will be|will it be|falls on|did|does) %s|%s (?:is|was|will be|falls on) (?:a|on a|what day)"%(_DATE,_DATE),s)
 if m:
  d=parse_date(m.group(1) or m.group(2),today)
  if d:return {'kind':'date','value':Fr(d.weekday()),'reply':f"{_dstr(d)} is a {calendar.day_name[d.weekday()]}.",'conf':0.98}
 return None
_ALG_VERB=re.compile(r"(?i)\b(solve|factor(?:ise|ize)?|expand|simplify|differentiate|derivative|integrate|integral|antiderivative|roots?|zeros?)\b")
_ALG_OK=re.compile(r"^(?:[\d\s.+\-*/^()=,]|\b(?:sqrt|sin|cos|tan|log|ln|exp|pi|abs)\b|\b[a-z]\b|(?<=\d)[a-z]\b|[a-z](?=\^|\())+$")
def _impl(e:str)->str:
 e=e.replace('^','**');e=re.sub(r"(\d)\s*([a-z(])",r"\1*\2",e);e=re.sub(r"\)\s*([a-z\d(])",r")*\1",e)
 return re.sub(r"\b([a-z])\s*(\()",lambda m:m.group(1)+'*'+m.group(2),e)
def _alg_text(t:str):
 s=_strip(_clean(t)).lower();vb=_ALG_VERB.search(s);verb=None
 if vb:
  w=vb.group(1);verb='solve' if w in ('solve','root','roots','zero','zeros') else 'factor' if w.startswith('factor') else 'diff' if w in ('differentiate','derivative') else 'integrate' if w in ('integrate','integral','antiderivative') else w
 s=re.sub(r"(?i)^.*?\b(?:solve|factor(?:ise|ize)?|expand|simplify|differentiate|find the derivative of|the derivative of|derivative of|integrate|the integral of|integral of|antiderivative of|(?:the\s+)?(?:roots?|zeros?) of)\b\s*(?:for\s+[a-z]\s*)?(?:the\s+equation\s+)?[:,]?\s*",'',s) if vb else s
 s=re.sub(r"(?i)\s*(?:,\s*)?(?:with respect to|wrt|d/d)\s*([a-z])\s*$",'',s);s=re.sub(r"(?i)[,;]?\s*(?:for|find|what is|what's)\s+[a-z](?:\s+and\s+[a-z])?\s*\??$",'',s).strip(' ?.:')
 s=re.sub(r"\s+and\s+",', ',s);s=re.sub(r"(?i)^(?:if|when|given)\s+",'',s)
 return verb,s
def solve_algebra(t:str):
 verb,s=_alg_text(t)
 if not s or len(s)>300 or not _ALG_OK.match(s) or '__' in s:return None
 vs=sorted(set(re.findall(r"(?<![a-z])([a-z])(?![a-z])",re.sub(r"\b(?:sqrt|sin|cos|tan|log|ln|exp|pi|abs)\b",' ',s)))-{'e'})
 if not vs or (verb is None and '=' not in s) or len(vs)>3:return None
 verb=verb or 'solve';parts=[p.strip() for p in re.split(r"[,;]",s) if p.strip()]
 if verb=='solve':
  eqs=[p.split('=',1) if '=' in p else [p,'0'] for p in parts]
  if any('=' in b for _,b in eqs) or not all(a.strip() and b.strip() for a,b in eqs):return None
  if len(vs)==1 and len(eqs)==1:
   f=lambda x:evaluate(_impl(eqs[0][0]),bind={vs[0]:x})-evaluate(_impl(eqs[0][1]),bind={vs[0]:x})
   try:
    f0,f1,f2=f(Fr(0)),f(Fr(1)),f(Fr(3))
    if all(isinstance(z,Fr) for z in (f0,f1,f2)) and f1-f0!=0 and f2-f0==3*(f1-f0):
     x=-f0/(f1-f0)
     if f(x)==0:return {'kind':'algebra','value':x,'reply':f"{vs[0]} = {fmt(x)} (solving {eqs[0][0].strip()} = {eqs[0][1].strip()}; check: both sides equal {fmt(evaluate(_impl(eqs[0][0]),bind={vs[0]:x}))} at {vs[0]} = {fmt(x)})",'conf':0.99}
   except Exception:pass
  d=job('solve',vs,eqs=[[a.strip(),b.strip()] for a,b in eqs]);sols=d.get('sols') if d else None
  if not sols:return None if sols is None else {'kind':'algebra','value':None,'reply':f"{'; '.join(a.strip()+' = '+b.strip() for a,b in eqs)} has no solution.",'conf':0.8}
  def chk(d):
   try:
    b={k:complex(*d[k][1]) if d[k][1][1] else d[k][1][0] for k in d}
    if any(isinstance(z,complex) for z in b.values()):return True
    return all(abs(float(evaluate(_impl(a),bind=b))-float(evaluate(_impl(c),bind=b)))<=1e-6*max(1,abs(float(evaluate(_impl(c),bind=b)))) for a,c in eqs)
   except Exception:return False
  good=[d for d in sols if all(k in d for k in vs) and chk(d)]
  if not good:return None
  real=[d for d in good if all(d[k][1] and abs(d[k][1][1])<1e-12 for k in vs)] or good
  sh=lambda d:', '.join(f"{k} = {d[k][0].replace('**','^').replace('*','')}"+(f" ≈ {d[k][1][0]:.6g}" if d[k][1] and not re.fullmatch(r"-?\d+(?:/\d+)?",d[k][0]) and not d[k][1][1] else '') for k in vs)
  v=Fr(str(real[0][vs[0]][0])) if len(real)==1 and re.fullmatch(r"-?\d+(?:/\d+)?",real[0][vs[0]][0]) else (real[0][vs[0]][1][0] if real[0][vs[0]][1] else None)
  return {'kind':'algebra','value':v,'reply':' or '.join(sh(d) for d in real)+f" (solving {'; '.join(a.strip()+' = '+b.strip() for a,b in eqs)}; {'checked by substitution' if len(real)==1 else 'each checked by substitution'})",'conf':0.97,'all':[sh(d) for d in real]}
 if len(parts)!=1 or '=' in parts[0]:return None
 d=job(verb,vs[:1] if verb in ('diff','integrate') else vs,body=parts[0])
 if not d or not d.get('ok'):return None
 lbl={'diff':f"d/d{vs[0]} of {parts[0]}",'integrate':f"∫ {parts[0]} d{vs[0]}",'factor':f"{parts[0]} factors as",'expand':f"{parts[0]} expands to",'simplify':f"{parts[0]} simplifies to"}[verb]
 res=d['res']+(' + C' if verb=='integrate' else '')
 return {'kind':'algebra','value':None,'reply':f"{lbl} {res}" if verb in ('factor','expand','simplify') else f"{lbl} = {res}",'conf':0.95,'expr':res}
def solve_misc(t:str):
 s=_strip(_clean(t)).lower().replace(',',' ');s=re.sub(r"\s+",' ',s)
 m=re.fullmatch(r"(?:the\s+)?(?:average|mean|arithmetic mean)\s+(?:of\s+)?((?:%s\s*(?:and\s+)?)+)"%_NUM,s)
 if m:
  xs=[Fr(x) for x in re.findall(_NUM,m.group(1))]
  if len(xs)>=2:v=sum(xs)/len(xs);return {'kind':'arith','value':v,'reply':f"The average is {fmt(v)} (({' + '.join(fmt(x) for x in xs)}) ÷ {len(xs)})",'conf':0.99}
 m=re.fullmatch(r"(?:the\s+)?(gcd|gcf|hcf|greatest common (?:divisor|factor)|highest common factor|lcm|least common multiple|lowest common multiple)\s+(?:of\s+)?((?:\d+\s*(?:and\s+)?)+)",s)
 if m:
  xs=[int(x) for x in re.findall(r"\d+",m.group(2))];g=m.group(1).startswith(('g','h'))
  if len(xs)>=2 and max(xs)<10**15:v=math.gcd(*xs) if g else math.lcm(*xs);return {'kind':'arith','value':Fr(v),'reply':f"The {'greatest common divisor' if g else 'least common multiple'} of {', '.join(map(str,xs))} is {v}.",'conf':0.99}
 m=re.fullmatch(r"(?:is\s+)?(\d+)\s+(?:a\s+)?prime(?:\s+number)?|is (\d+) prime",s)
 if m:
  n=int(m.group(1) or m.group(2))
  if n<10**12:f=next((p for p in range(2,math.isqrt(n)+1) if n%p==0),None) if n>1 else 1;return {'kind':'arith','value':Fr(int(f is None and n>1)),'reply':f"Yes, {n} is prime (no divisor up to √{n} ≈ {math.isqrt(n)})." if f is None and n>1 else f"No, {n} is not prime"+(f" ({n} = {f} × {n//f})." if n>1 else " (primes start at 2)."),'conf':0.99}
 m=re.fullmatch(r"(?:the\s+)?prime factori[sz]ation of (\d+)|(?:the\s+)?prime factors of (\d+)|factori[sz]e (\d+)",s)
 if m:
  n=int(next(g for g in m.groups() if g));fs=[];k=n;p=2
  if 1<n<10**12:
   while p*p<=k:
    while k%p==0:fs.append(p);k//=p
    p+=1
   fs+=[k] if k>1 else [];return {'kind':'arith','value':Fr(n),'reply':f"{n} = {' × '.join(map(str,fs))}",'conf':0.99}
 m=re.fullmatch(r"(?:the\s+)?sum of (?:all\s+)?(?:the\s+)?(?:integers|numbers|whole numbers)?\s*(?:from\s+)?(\d+) (?:to|through|thru) (\d+)",s)
 if m:
  a,b=int(m.group(1)),int(m.group(2))
  if a<=b<10**12:v=(a+b)*(b-a+1)//2;return {'kind':'arith','value':Fr(v),'reply':f"The sum from {a} to {b} is {v:,} (({a} + {b}) × {b-a+1} ÷ 2)",'conf':0.99}
 return None
_LUT={'each':1.2,'per':1.0,'total':1.1,'altogether':1.4,'left':1.1,'remaining':1.0,'more':0.6,'than':0.4,'twice':1.0,'half':0.8,'times':0.5,'buys':1.2,'bought':1.2,'sells':1.1,'sold':1.1,'costs':1.2,'cost':0.9,'spent':1.2,'spends':1.2,'pays':1.1,'paid':1.0,'earns':1.2,'earned':1.1,'saves':1.0,'has':0.7,'had':0.6,'gave':1.0,'gives':1.0,'if':0.5,'then':0.5,'every':0.6,'how':0.4,'many':0.3,'much':0.3,'dollars':0.8,'$':1.0,'apples':0.8,'cookies':0.8,'candies':0.8,'pencils':0.8,'marbles':0.9,'books':0.5,'she':0.5,'he':0.5,'they':0.3,'his':0.4,'her':0.4,'week':0.3,'hour':0.3,'minutes':0.3,'rest':0.6,'remainder':0.8,'shared':0.8,'split':0.6,'equally':0.9,'among':0.6,'population':-3,'born':-3,'died':-3,'founded':-3,'president':-3,'century':-2,'war':-2,'capital':-3,'moons':-3,'planet':-2,'chromosomes':-3,'bones':-2,'legs':-1,'calories':-1,'country':-2,'countries':-2,'olympics':-3,'album':-3,'movie':-3,'season':-2,'episode':-3,'who':-2,'when':-1,'where':-2,'why':-2,'year':-0.6,'invented':-3,'discovered':-3,'elected':-3,'built':-1,'tall':-1,'height':-1}
def classify(t:str)->dict:
 s=_clean(t or '').lower();toks=re.findall(r"\$|[a-z]+",s);nums=[x for x in re.findall(r"\d+(?:\.\d+)?",s) if not re.fullmatch(r"(?:1[5-9]|20)\d\d",x)]
 sc=sum(_LUT.get(w,0) for w in toks)+(1.5 if len(nums)>=2 else 0.5 if nums else -4)+(1 if re.search(r"\bhow (?:many|much)\b|\bwhat is the (?:total|cost|number|difference|sum)\b|\bfind\b|\bcalculate\b",s) else -1)-(2.5 if _FACTUAL.search(s) else 0)
 intents=[k for k,p in (('date',r"\b(?:days?|weeks?|months?|years?|hours?)\b.*\b(?:between|until|till|since|from|after|before|ago)\b|\bwhat day\b|\bday of the week\b|\bleap year\b|\bhow long (?:until|till|since)\b|\bin \d+ (?:days|weeks|months|years)\b"),('units',r"\b(?:convert|to|in|into)\b|\bhow many\b"),('percent',r"%|\bpercent|\bout of\b"),('algebra',r"\b(?:solve|factor|expand|simplify|derivative|differentiate|integrate|integral|roots?|zeros?)\b|(?:\d|\b)[a-z]\b\s*(?:\^|\*\*|[+\-=])|="),('misc',r"\b(?:average|mean|gcd|gcf|hcf|lcm|common (?:divisor|factor|multiple)|prime|factori[sz]|sum of)\b"),('arith',r"\d")) if re.search(p,s)]
 return {'intents':intents+(['word'] if sc>=1.0 and len(nums)>=1 else []),'word_score':round(sc,2),'nums':len(nums)}
def solve_word_text(t:str):
 r=solve_word(t)
 return None if r is None else {'kind':'word','value':r['value'],'reply':r['reply'],'conf':r['conf'],'trace':r.get('trace')}
_ENG={'date':solve_date,'units':solve_units,'percent':solve_percent,'algebra':solve_algebra,'misc':solve_misc,'arith':solve_arith,'word':solve_word_text}
def solve_math(text:str,today:dt.date|None=None):
 t=(text or '').strip()
 if not t or len(t)>2000:return None
 c=classify(t)
 for k in c['intents']:
  try:r=_ENG[k](t,today) if k=='date' else _ENG[k](t)
  except (ValueError,ZeroDivisionError,OverflowError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError,RecursionError):r=None
  if r and r.get('reply'):return {**r,'intent':k,'word_score':c['word_score']}
 return None
def answer(text:str,today:dt.date|None=None):
 r=solve_math(text,today);return (r['reply'],float(r['conf'])) if r else None
