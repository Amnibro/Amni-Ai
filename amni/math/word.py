import re
from fractions import Fraction as Fr
_W1={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'million':1000000}
_RAT={'half':('HALF',Fr(2)),'twice':('TWICE',Fr(2)),'double':('TWICE',Fr(2)),'doubled':('TWICE',Fr(2)),'triple':('TRIPLE',Fr(3)),'tripled':('TRIPLE',Fr(3)),'thrice':('TRIPLE',Fr(3)),'quadruple':('QUAD',Fr(4)),'dozen':('DOZEN',Fr(12)),'third':('THIRD',Fr(3)),'quarter':('QUARTER',Fr(4)),'fourth':('QUARTER',Fr(4)),'fifth':('FIFTH',Fr(5))}
_FRW={'half':2,'halves':2,'third':3,'thirds':3,'fourth':4,'fourths':4,'quarter':4,'quarters':4,'fifth':5,'fifths':5,'sixth':6,'sixths':6,'eighth':8,'eighths':8,'tenth':10,'tenths':10}
TOK=re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?|\$?\.\d+%?|[A-Za-z]+(?:'[a-z]+)?|[.?!;:,]|-")
STOP={'the','a','an','of','to','and','in','on','for','at','is','was','are','were','be','that','this','it','his','her','their','its','with','as','by','if','he','she','they','them','him','there','has','had','have','does','did','do','from','each','per','every','than','more','less','will','would','can','could','then','so','but','or','how','many','much','what'}
def lem(w:str)->str:
 w=w.lower().strip("'")
 return w[:-3]+'y' if w.endswith('ies') and len(w)>4 else w[:-2] if w.endswith(('ches','shes','xes','sses')) else w[:-1] if w.endswith('s') and not w.endswith(('ss','us','is')) and len(w)>3 else w
def sentences(text:str)->list:
 t=re.sub(r"(?<=\d)\.(?=\d)",'<DOT>',text.strip().replace('’',"'").replace('‘',"'"));t=re.sub(r"\b(Mr|Mrs|Ms|Dr|St|vs)\.",r"\1",t)
 return [s.replace('<DOT>','.').strip() for s in re.split(r"(?<=[.?!])\s+",t) if s.strip()]
def _val(tok:str):
 s=tok.strip('$%').replace(',','')
 try:return Fr(s) if '/' not in s else Fr(int(s.split('/')[0]),int(s.split('/')[1]))
 except (ValueError,ZeroDivisionError):return None
def quantities(text:str)->dict:
 ss=sentences(text);qs=[];toks=[];qi=max(range(len(ss)),key=lambda i:(('?' in ss[i]) or bool(re.match(r"(?i)(how|what|find|calculate|determine)\b",ss[i])),i)) if ss else 0
 for si,s in enumerate(ss):
  tk=[m.group() for m in TOK.finditer(s)];base=len(toks);toks+=[(t,si) for t in tk];i=0
  while i<len(tk):
   t=tk[i];lw=t.lower();nx=tk[i+1].lower() if i+1<len(tk) else '';pv=tk[i-1].lower() if i else '';i0=i
   v=_val(t) if re.match(r"\$?[\d.]",t) else None;kind=None;alts=None
   if v is not None and pv in ('in','since','from','of') and re.fullmatch(r"(?:1[6-9]|20)\d\d",t):v=None
   if v is not None:
    kind='PCT' if t.endswith('%') or nx in ('percent','%') else 'MONEY' if t.startswith('$') or nx in ('dollars','dollar','bucks','usd') else 'FRAC' if '/' in t else 'NUM'
    if kind=='NUM' and nx in _FRW and v<=20:v=v/_FRW[nx];kind='FRAC';i+=1
    elif kind in ('NUM','MONEY') and nx in ('dozen','dozens'):v=v*12;i+=1
   elif lw in _W1 and not (lw=='one' and nx in ('of','day','time','another','more','other','each','by')):
    v=Fr(_W1[lw]);kind='WORD';j=i+1
    while j<len(tk) and tk[j].lower() in _W1|{'-':0,'and':0} and not (tk[j]=='-' and j+1<len(tk) and tk[j+1].lower() not in _W1):
     w=tk[j].lower();v=v*_W1[w] if w in ('hundred','thousand','million') else v+_W1[w] if w in _W1 else v;j+=1
    nx2=tk[j].lower() if j<len(tk) else ''
    if nx2 in _FRW and v<=20:v=v/_FRW[nx2];kind='FRAC';j+=1
    if nx2=='percent':kind='PCT'
    elif nx2 in ('dozen','dozens'):v=v*12;j+=1
    i=j-1;nx=nx2
   elif lw in ('dozen',) and pv in ('a','one'):kind,v='NUM',Fr(12)
   elif lw in _RAT and lw!='dozen':
    kind,v=_RAT[lw]
    if lw in ('third','fourth','fifth') and pv not in ('a','one','every'):v=None
   if v is not None and kind:
    alts=[(v,'r')]+([(v/100,'f'),(1-v/100,'c'),(1+v/100,'p')] if kind=='PCT' else [(1/v,'f')] if kind in ('HALF','THIRD','QUARTER','FIFTH') else [(1-v,'c')] if kind=='FRAC' and v<1 else [])
    L=max(0,i-4);ctx=[x.lower() for x in tk[L:i] if not _val(x)];rc=[x.lower() for x in tk[i+1:i+6] if not re.match(r"[\d$]",x)]
    unit=next((lem(x) for x in rc if x.isalpha() and x not in STOP and x not in _W1),'')
    per=next((lem(rc[k+1]) for k in range(len(rc)-1) if rc[k] in ('per','each','every','an','a') and rc[k+1].isalpha() and rc[k+1] not in STOP),'')
    qs.append({'v':v,'alts':alts,'kind':kind,'s':si,'tok':base+i,'t0':base+i0,'txt':t,'unit':unit,'per':per,'q':si==qi,'l':ctx,'r':rc})
   i+=1
 return {'sents':ss,'qs':qs,'qi':qi,'toks':toks}
def _f(v:Fr)->str:return str(v.numerator) if v.denominator==1 else (f"{float(v):g}" if len(f"{float(v):g}")<9 else f"{v.numerator}/{v.denominator}")
