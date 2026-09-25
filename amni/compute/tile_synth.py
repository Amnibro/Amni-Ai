import ast,builtins,collections,copy,functools,gc,itertools,json,math,operator as op,os,random,re,resource,signal,string,textwrap,threading,time,types,typing,zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from amni.compute import ray_field
ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT/"data"/"code_eval"/"learned_tiles.jsonl"
CAP,ROWCAP,PAIRCAP,RMAX,MAXC,BIG,LMAX=2000,600,800,2000,18,1<<128,24
EMPTY=frozenset()
COVW=int(os.environ.get("TS_COVW","1"))
CONC=ROOT/"data"/"concepts"
SEMM=os.environ.get("TS_SEM","wn")
PEN={k:float(v) for k,v in (x.split("=") for x in ("intent=5,cidx=2,fconst=2,crash=1,redund=2,edge=0.5,rel=0,"+os.environ.get("TS_PEN","")).split(",") if x)}
AFT,REL_T,SPHT,COVT=0.75,(1,2,9,14,15),float(os.environ.get("TS_SPHT","0.99999")),float(os.environ.get("TS_COVT","0.75"))
TOPN,XLV,XS=int(os.environ.get("TS_TOPN","16")),int(os.environ.get("TS_XLV","2")),float(os.environ.get("TS_XS","3"))
NATP=ray_field.BASE_PACK
RANK=dict(x.split("=") for x in ("nat=off,w=0,pw=1,lw=0,tb=0,af=off,at=-99,anl=0,ap=0,pv=0,pz=0,pm=99,cons=off,cn=99,cw=1,ct=1,cx=d,ci=a,cz=2,cm=x,cp=0,ca=0,cab=x,sx=0,sb=0,sk=20,sm=sub,sw=1,sl=0,so=0,sp=0,x10=on,"+os.environ.get("TS_RANK","")).split(",") if x)
PRIORP=os.environ.get("TS_PRIOR",str(ROOT/"data"/"pycode"/"x5_prior.npz"))
PCFG=dict(x.split("=") for x in ("b=20,idf=1,gw=1,ga=1,gv=1,gr=1,norm=1,agg=sum,mode=pmi,fc=cmkops,"+os.environ.get("TS_PCFG","")).split(",") if x)
SKC=dict(x.split("=") for x in ("s=4,b=3,n=4,e=1,eb=1.5,en=3,k=20,"+os.environ.get("TS_SKC","")).split(",") if x)
PBIAS=dict(x.split("=") for x in ("t=99,dn=-99,rs=0,pat=1,"+os.environ.get("TS_PB","")).split(",") if x)
class Unsafe(Exception):pass
class Timeout(Exception):pass
class Stop(BaseException):pass
SAFE_B={k:getattr(builtins,k) for k in "abs all any bin bool chr dict divmod enumerate filter float frozenset hex int isinstance iter len list map max min next oct ord pow range repr reversed round set slice sorted str sum tuple zip ValueError TypeError IndexError KeyError ZeroDivisionError StopIteration Exception ArithmeticError AssertionError".split()}
BAN_ATTR={"format","format_map","mro","register"}
BAN_NAME={"eval","exec","open","getattr","setattr","delattr","globals","locals","vars","compile","breakpoint","input","help","type","object","super","memoryview","dir","id","classmethod","staticmethod","property"}
BAN_NODE=(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.AsyncFunctionDef,ast.Await,ast.AsyncFor,ast.AsyncWith,ast.With,ast.Yield,ast.YieldFrom)
def _ns(m,ks=None):return types.SimpleNamespace(**{k:getattr(m,k) for k in (ks or dir(m)) if not k.startswith("_")})
MODS={"math":_ns(math),"itertools":_ns(itertools),"functools":_ns(functools,["reduce","cmp_to_key","lru_cache","partial"])}
XMODS={"re":_ns(re,["compile","findall","finditer","fullmatch","match","search","split","sub","subn","escape","IGNORECASE","I","M","S"]),"collections":_ns(collections,["Counter","defaultdict","deque","OrderedDict"]),"string":_ns(string,["ascii_lowercase","ascii_uppercase","ascii_letters","digits","hexdigits","octdigits","punctuation","whitespace","printable"])}
TYPING={k:getattr(typing,k) for k in "List Tuple Optional Any Dict Union Set Callable Iterable Sequence".split()}
def _bad(n):return isinstance(n,BAN_NODE) or (isinstance(n,ast.Attribute) and (n.attr.startswith("_") or n.attr in BAN_ATTR)) or (isinstance(n,ast.Name) and (n.id.startswith("__") or n.id in BAN_NAME))
def safe_check(src,mode="exec"):
 bad=[n for n in ast.walk(ast.parse(src,mode=mode)) if _bad(n)]
 if bad:raise Unsafe(f"blocked {type(bad[0]).__name__} {getattr(bad[0],'attr',getattr(bad[0],'id',''))}")
 return src
def _env(env):return {"__builtins__":SAFE_B,**MODS,**(env or {})}
def safe_exec(src,env=None):g=_env(env);exec(compile(safe_check(src),"<tile>","exec"),g);return g
def safe_eval(src,env=None):return eval(compile(safe_check(src,"eval"),"<tile>","eval"),_env(env))
def _alarm(*_):raise Timeout("call timeout")
def timed(fn,*a,t=0.1):
 if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]>0:return fn(*a)
 old=signal.signal(signal.SIGALRM,_alarm);signal.setitimer(signal.ITIMER_REAL,t)
 try:return fn(*a)
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
def _x0eq(a,b):
 try:return a==b or abs(a-b)<=1e-6*max(1.0,abs(b))
 except BaseException:return False
def _eqv(a,b):
 try:return (type(a) is bool)==(type(b) is bool) and bool(a==b or (type(a) in (int,float) and type(b) in (int,float) and abs(a-b)<=1e-6*max(1.0,abs(b))))
 except Exception:return False
N,I,F,B,S_,C,Q,L,T,SN,SS,SQ,D,E,X,O,SB=(1<<k for k in range(17))
XB,YB,IB,X2B,I2B=1<<20,1<<21,1<<22,1<<23,1<<24
PR=dict(n=N,i=I,b=B,s=S_,c=C,q=Q,l=Q|L,t=Q|T,N=Q|SN,S=Q|SS,Q=Q|SQ,B=Q|SB,d=D,x=X,a=0)
def _ek(v):ts={type(e) for e in v};return (SN if ts<={int,float} else 0)|(SS if ts<={str} else 0)|(SQ if ts<={list,tuple} else 0)|(SB if ts<={bool} else 0)
def _sz(v):return sum(len(e) if type(e) in (list,tuple,str) else 1 for e in v)<=4*CAP
def _kv(v):
 t=type(v)
 return B if t is bool else (N|I if -BIG<v<BIG else 0) if t is int else (N|F if v==v and -1e300<v<1e300 else 0) if t is float else (S_|X|(C if len(v)==1 else 0) if len(v)<=CAP else 0) if t is str else (Q|X|(L if t is list else T)|_ek(v) if len(v)<=CAP and _sz(v) else 0) if t is list or t is tuple else (E|X|_ek(v) if len(v)<=CAP else 0) if t is set or t is frozenset else (D if len(v)<=CAP else 0) if t is dict else O if v is None else 0
def _no(*a):raise ValueError("guard")
def _pw(a,b):return a**b if abs(b)<=64 and not (a==0 and b<0) and (type(a) is float or type(b) is float or abs(a)<=10**6) else _no()
def _isp(n):return (n>1 and all(n%d for d in range(2,int(n**0.5)+1))) if n<10**10 else _no()
def _fib(n):return functools.reduce(lambda p,_:(p[1],p[0]+p[1]),range(n),(0,1))[0] if 0<=n<=300 else _no()
def _rng(a,b):return list(range(a,b)) if b-a<=RMAX else _no()
def _rep(s,n):return s*n if 0<=n<=64 and len(s)*n<=CAP else _no()
def _med(a):s=sorted(a);return (s[len(a)//2]+s[(len(a)-1)//2])/2
HELP={"_dflat":"def _dflat(x):\n    return [z for y in x for z in (_dflat(y) if isinstance(y, (list, tuple)) else [y])]\n","_droot":"def _droot(n):\n    while n > 9:\n        n = sum(int(d) for d in str(n))\n    return n\n","_vmult":"def _vmult(n, k):\n    c = 0\n    while n and k > 1 and n % k == 0:\n        n //= k\n        c += 1\n    return c\n","_tobase":"def _tobase(n, b):\n    s = ''\n    while n > 0 and b > 1:\n        s = str(n % b) + s\n        n //= b\n    return s or '0'\n"}
HFN={k:safe_exec(v)[k] for k,v in HELP.items()}
TILE_DEFS=[
("add","nn","({0} + {1})",op.add,2,"sum add plus total more increase added","C"),
("sub","nn","({0} - {1})",op.sub,2,"difference subtract minus less decrease between",""),
("mul","nn","({0} * {1})",op.mul,2,"product multiply times area volume","C"),
("div","nn","({0} / {1})",op.truediv,2,"divide ratio average mean per fraction",""),
("fdiv","nn","({0} // {1})",op.floordiv,2,"divide quotient integer floor",""),
("mod","nn","({0} % {1})",op.mod,2,"remainder modulo divisible mod",""),
("pow","nn","({0} ** {1})",_pw,2,"power exponent raise",""),
("min2","nn","min({0}, {1})",min,2,"minimum smaller smallest least min","C"),
("max2","nn","max({0}, {1})",max,2,"maximum larger largest greater max","C"),
("adiff","nn","abs({0} - {1})",lambda a,b:abs(a-b),2,"difference distance absolute close closer","C"),
("lt","nn","({0} < {1})",op.lt,2,"less smaller below under than closer",""),
("le","nn","({0} <= {1})",op.le,2,"less most within",""),
("eq","nn|ss|qq","({0} == {1})",op.eq,2,"equal same identical","C"),
("ne","nn|ss","({0} != {1})",op.ne,3,"different distinct","C"),
("divs","ii","({0} % {1} == 0)",lambda a,b:a%b==0,2,"divisible divides multiple factor divisor",""),
("bxor","ii","({0} ^ {1})",op.xor,3,"xor bitwise exclusive","C"),
("band","ii","({0} & {1})",op.and_,3,"bitwise and mask","C"),
("bor","ii","({0} | {1})",op.or_,3,"bitwise or","C"),
("shl","ii","({0} << {1})",lambda a,b:a<<b if 0<=b<=64 else _no(),3,"shift left bitwise power",""),
("shr","ii","({0} >> {1})",op.rshift,3,"shift right bitwise",""),
("gcd","ii","math.gcd({0}, {1})",math.gcd,2,"gcd common divisor greatest","Cm"),
("lcm","ii","({0} * {1} // math.gcd({0}, {1}))",lambda a,b:a*b//math.gcd(a,b),3,"lcm multiple least common","Cm"),
("rnd","ni","round({0}, {1})",lambda a,b:round(a,b) if 0<=b<=12 else _no(),3,"round decimal places",""),
("neg","n","(-{0})",op.neg,2,"opposite negate",""),
("abs","n","abs({0})",abs,2,"absolute magnitude distance",""),
("inc","n","({0} + 1)",lambda a:a+1,2,"next increment successor plus one",""),
("dec","n","({0} - 1)",lambda a:a-1,2,"previous decrement minus one smaller closest",""),
("dbl","n","({0} * 2)",lambda a:a*2,2,"double twice two",""),
("half","n","({0} // 2)",lambda a:a//2,3,"half middle",""),
("halff","n","({0} / 2)",lambda a:a/2,3,"half average",""),
("even","n","({0} % 2 == 0)",lambda a:a%2==0,2,"even",""),
("odd","n","({0} % 2 == 1)",lambda a:a%2==1,2,"odd",""),
("pos","n","({0} > 0)",lambda a:a>0,2,"positive greater",""),
("negv","n","({0} < 0)",lambda a:a<0,2,"negative below",""),
("isz","n","({0} == 0)",lambda a:a==0,3,"zero",""),
("sq","n","({0} ** 2)",lambda a:a**2,2,"square squared",""),
("cube","n","({0} ** 3)",lambda a:a**3,3,"cube cubed",""),
("sqrt","n","({0} ** 0.5)",lambda a:a**0.5 if a>=0 else _no(),3,"root sqrt square",""),
("int","n","int({0})",int,3,"integer truncate floor whole",""),
("flt","n","float({0})",float,3,"float decimal",""),
("round","n","round({0})",round,3,"round nearest",""),
("strn","n","str({0})",str,3,"string digit digits convert text",""),
("mod10","i","({0} % 10)",lambda a:a%10,3,"last digit unit",""),
("div10","i","({0} // 10)",lambda a:a//10,3,"digit remove",""),
("fact","i","math.factorial({0})",lambda a:math.factorial(a) if 0<=a<=30 else _no(),3,"factorial","m"),
("prime","i","({0} > 1 and all({0} % _d for _d in range(2, int({0} ** 0.5) + 1)))",_isp,2,"prime primes primality",""),
("digsum","i","sum(int(_d) for _d in str(abs({0})))",lambda a:sum(int(d) for d in str(abs(a))),3,"digit digits sum",""),
("digits","i","[int(_d) for _d in str(abs({0}))]",lambda a:[int(d) for d in str(abs(a))],3,"digit digits",""),
("revn","i","int(str({0})[::-1])",lambda a:int(str(a)[::-1]),3,"reverse reversed palindrome",""),
("bin","i","bin({0})[2:]",lambda a:bin(a)[2:],3,"binary bits base",""),
("pop","i","bin({0}).count('1')",lambda a:bin(a).count('1'),3,"bit bits set binary ones count",""),
("issq","i","(int({0} ** 0.5) ** 2 == {0})",lambda a:int(a**0.5)**2==a if a>=0 else _no(),3,"perfect square",""),
("fib","i","functools.reduce(lambda _p, _q: (_p[1], _p[0] + _p[1]), range({0}), (0, 1))[0]",_fib,3,"fibonacci fib","f"),
("rng","i","list(range({0}))",lambda a:_rng(0,a),3,"range numbers up to less below",""),
("rng1","i","list(range(1, {0} + 1))",lambda a:_rng(1,a+1),3,"range natural first inclusive",""),
("rng0i","i","list(range({0} + 1))",lambda a:_rng(0,a+1),3,"range inclusive upto",""),
("rng2","ii","list(range({0}, {1}))",_rng,3,"range between from",""),
("rng2i","ii","list(range({0}, {1} + 1))",lambda a,b:_rng(a,b+1),3,"range between inclusive",""),
("chr","i","chr({0})",lambda a:chr(a) if 0<=a<0x110000 else _no(),3,"character ascii letter char",""),
("ord","c","ord({0})",ord,3,"ascii code value unicode",""),
("divsr","i","[_d for _d in range(1, {0} + 1) if {0} % _d == 0]",lambda a:[d for d in range(1,a+1) if a%d==0] if a<=10**5 else _no(),3,"divisor divisors divides divide factor factors",""),
("tri","i","({0} * ({0} + 1) // 2)",lambda a:a*(a+1)//2,3,"triangular natural",""),
("not","b","(not {0})",op.not_,2,"not negate false",""),
("and","bb","({0} and {1})",lambda a,b:a and b,2,"and both","C"),
("or","bb","({0} or {1})",lambda a,b:a or b,2,"or either","C"),
("xor","bb","({0} != {1})",op.ne,3,"xor exclusive","C"),
("b2i","b","int({0})",int,3,"count number",""),
("len","x|d","len({0})",len,2,"length len size count number long many",""),
("lower","s","{0}.lower()",str.lower,2,"lower lowercase small case insensitive",""),
("upper","s","{0}.upper()",str.upper,2,"upper uppercase capital case",""),
("swap","s","{0}.swapcase()",str.swapcase,2,"swap flip toggle case",""),
("title","s","{0}.title()",str.title,3,"title capitalize word",""),
("cap","s","{0}.capitalize()",str.capitalize,3,"capitalize capital first",""),
("strip","s","{0}.strip()",str.strip,3,"strip trim whitespace",""),
("rev","x","{0}[::-1]",lambda a:a[::-1],2,"reverse reversed backward",""),
("pal","x","({0} == {0}[::-1])",lambda a:a==a[::-1],2,"palindrome backward same",""),
("split","s","{0}.split()",str.split,2,"word words split sentence space",""),
("splitc","ss","{0}.split({1})",lambda a,b:a.split(b) if b else _no(),3,"split separate delimiter separated",""),
("lst","s|q","list({0})",list,3,"character characters list letters",""),
("set","x","set({0})",set,3,"unique distinct set",""),
("srt","x","sorted({0})",sorted,2,"sort sorted order ascending alphabetical increasing",""),
("srtr","x","sorted({0}, reverse=True)",lambda a:sorted(a,reverse=True),3,"descending decreasing reverse sort largest",""),
("jsrt","s","''.join(sorted({0}))",lambda a:"".join(sorted(a)),3,"sort anagram alphabetical",""),
("join","S","''.join({0})","".join,2,"join concatenate combine string",""),
("joinsp","S","' '.join({0})"," ".join,2,"join sentence space words",""),
("joinc","Ss","{1}.join({0})",lambda a,b:b.join(a),3,"join separator delimiter",""),
("isdig","s","{0}.isdigit()",str.isdigit,3,"digit number numeric integer",""),
("isal","s","{0}.isalpha()",str.isalpha,3,"letter alphabetic alpha",""),
("isup","s","{0}.isupper()",str.isupper,3,"upper uppercase capital",""),
("islo","s","{0}.islower()",str.islower,3,"lower lowercase",""),
("issp","s","{0}.isspace()",str.isspace,3,"space whitespace",""),
("isan","s","{0}.isalnum()",str.isalnum,3,"alphanumeric",""),
("ints","s","int({0})",int,3,"integer number convert parse",""),
("flts","s","float({0})",float,3,"float number convert parse",""),
("cnt","ss|Nn|Ss|Bb","{0}.count({1})",lambda a,b:a.count(b) if b!="" else _no(),2,"count occurrence occurrences number times frequency",""),
("repl","sss","{0}.replace({1}, {2})",lambda a,b,c:a.replace(b,c) if b else _no(),2,"replace substitute change",""),
("rmv","ss","{0}.replace({1}, '')",lambda a,b:a.replace(b,"") if b else _no(),2,"remove delete",""),
("sw","ss","{0}.startswith({1})",str.startswith,2,"start starts begin begins prefix",""),
("ew","ss","{0}.endswith({1})",str.endswith,2,"end ends suffix",""),
("sws","ssi","{0}.startswith({1}, {2})",str.startswith,3,"occurrence overlap overlapping substring",""),
("isin","ss|nN|sS|qQ|nd|sd","({0} in {1})",lambda a,b:a in b,2,"in contain contains present substring member include exist appear",""),
("nin","ss|nN|sS","({0} not in {1})",lambda a,b:a not in b,3,"not absent missing remove without",""),
("find","ss","{0}.find({1})",str.find,3,"find index position first",""),
("idx","Nn|Ss","{0}.index({1})",lambda a,b:a.index(b),3,"index position",""),
("cat","ss|ll|tt","({0} + {1})",op.add,2,"concatenate join append combine add merge",""),
("reps","si","({0} * {1})",_rep,3,"repeat times",""),
("at","xi","{0}[{1}]",lambda a,b:a[b],2,"index element position nth kth at",""),
("first","x","{0}[0]",lambda a:a[0],2,"first head start begin",""),
("last","x","{0}[-1]",lambda a:a[-1],2,"last end final",""),
("second","x","{0}[1]",lambda a:a[1],3,"second",""),
("tail","x","{0}[1:]",lambda a:a[1:],3,"rest tail except remove first",""),
("init","x","{0}[:-1]",lambda a:a[:-1],3,"except last remove",""),
("evi","x","{0}[::2]",lambda a:a[::2],3,"even index alternate every other",""),
("odi","x","{0}[1::2]",lambda a:a[1::2],3,"odd index alternate",""),
("take","xi","{0}[:{1}]",lambda a,b:a[:b],3,"first prefix take",""),
("drop","xi","{0}[{1}:]",lambda a,b:a[b:],3,"after drop skip",""),
("takel","xi","{0}[-{1}:]",lambda a,b:a[-b:],3,"last suffix",""),
("slc","xii","{0}[{1}:{2}]",lambda a,b,c:a[b:c],3,"range between slice",""),
("rot","xi","({0}[{1}:] + {0}[:{1}])",lambda a,b:a[b:]+a[:b],3,"rotate shift cyclic left",""),
("sum","N|B","sum({0})",sum,2,"sum total add count",""),
("max","x","max({0})",max,2,"max maximum largest biggest greatest highest",""),
("min","x","min({0})",min,2,"min minimum smallest least lowest",""),
("mean","N","(sum({0}) / len({0}))",lambda a:sum(a)/len(a),2,"average mean",""),
("spread","N","(max({0}) - min({0}))",lambda a:max(a)-min(a),3,"range difference spread",""),
("prod","N","math.prod({0})",math.prod,2,"product multiply multiplied","m"),
("any","q","any({0})",any,2,"any exist some contain",""),
("all","q","all({0})",all,2,"all every each",""),
("nodup","x","(len({0}) == len(set({0})))",lambda a:len(a)==len(set(a)),3,"unique distinct duplicate duplicates",""),
("issrt","x","(list({0}) == sorted({0}))",lambda a:list(a)==sorted(a),3,"sorted order ascending increasing monotonic",""),
("issrtr","x","(list({0}) == sorted({0}, reverse=True))",lambda a:list(a)==sorted(a,reverse=True),3,"descending decreasing",""),
("uniq","x","sorted(set({0}))",lambda a:sorted(set(a)),2,"unique distinct sorted",""),
("dedup","q|s","list(dict.fromkeys({0}))",lambda a:list(dict.fromkeys(a)),3,"unique distinct duplicate remove order",""),
("nuniq","x","len(set({0}))",lambda a:len(set(a)),3,"unique distinct different",""),
("median","N","((sorted({0})[len({0}) // 2] + sorted({0})[(len({0}) - 1) // 2]) / 2)",_med,3,"median middle",""),
("mode","q|s","max({0}, key={0}.count)",lambda a:max(a,key=a.count),3,"frequent frequency common mode occurrence most max maximum",""),
("freq","q|s","{_v: {0}.count(_v) for _v in {0}}",lambda a:{v:a.count(v) for v in a},3,"frequency count occurrence dictionary each",""),
("flat","Q","[_y for _x in {0} for _y in _x]",lambda a:[y for x in a for y in x],3,"flatten nested combine merge",""),
("zip","qq","list(zip({0}, {1}))",lambda a,b:list(zip(a,b)),3,"pair zip combine corresponding",""),
("tup","x","tuple({0})",tuple,3,"tuple convert",""),
("maxl","q","max({0}, key=len)",lambda a:max(a,key=len),3,"longest max length",""),
("minl","q","min({0}, key=len)",lambda a:min(a,key=len),3,"shortest min length",""),
("srtl","q","sorted({0}, key=len)",lambda a:sorted(a,key=len),3,"sort length",""),
("imax","x","{0}.index(max({0}))",lambda a:a.index(max(a)),3,"index position largest",""),
("imin","x","{0}.index(min({0}))",lambda a:a.index(min(a)),3,"index position smallest",""),
("sand","xx","sorted(set({0}) & set({1}))",lambda a,b:sorted(set(a)&set(b)),3,"common intersection shared both","C"),
("sor","xx","sorted(set({0}) | set({1}))",lambda a,b:sorted(set(a)|set(b)),3,"union merge combine","C"),
("ssub","xx","sorted(set({0}) - set({1}))",lambda a,b:sorted(set(a)-set(b)),3,"difference remove exclude",""),
("sxor","xx","sorted(set({0}) ^ set({1}))",lambda a,b:sorted(set(a)^set(b)),3,"symmetric difference","C"),
("issub","xx","(set({0}) <= set({1}))",lambda a,b:set(a)<=set(b),3,"subset contain",""),
("app","Nn|Ss","(list({0}) + [{1}])",lambda a,b:list(a)+[b],3,"append add insert end",""),
("pre","Nn|Ss","([{1}] + list({0}))",lambda a,b:[b]+list(a),3,"prepend insert beginning front",""),
("pair","nn|ss|qq","({0}, {1})",lambda a,b:(a,b),3,"tuple pair both",""),
("lpair","nn","[{0}, {1}]",lambda a,b:[a,b],3,"list both",""),
("dkeys","d","list({0})",list,3,"keys key",""),
("dvals","d","list({0}.values())",lambda a:list(a.values()),3,"values value",""),
("ditems","d","list({0}.items())",lambda a:list(a.items()),3,"items pairs",""),
("dsum","d","sum({0}.values())",lambda a:sum(a.values()),3,"sum total",""),
("dmaxk","d","max({0}, key={0}.get)",lambda a:max(a,key=a.get),3,"max largest highest most",""),
("dmink","d","min({0}, key={0}.get)",lambda a:min(a,key=a.get),3,"min smallest lowest least",""),
("dget","dn|ds","{0}.get({1})",lambda a,b:a.get(b),3,"get value lookup key",""),
("dmerge","dd","{**{0}, **{1}}",lambda a,b:{**a,**b},3,"merge combine union",""),
("dinv","d","{_v: _k for _k, _v in {0}.items()}",lambda a:{v:k for k,v in a.items()},3,"invert swap reverse",""),
("dsrt","d","dict(sorted({0}.items()))",lambda a:dict(sorted(a.items())),3,"sort sorted",""),
("dzip","qq","dict(zip({0}, {1}))",lambda a,b:dict(zip(a,b)),3,"dictionary map mapping",""),
("dsrtv","d","sorted({0}.items(), key=lambda _t: _t[1])",lambda a:sorted(a.items(),key=lambda t:t[1]),3,"sort value",""),
("isint","a","isinstance({0}, int)",lambda a:isinstance(a,int),3,"integer int integers type",""),
("isstr","a","isinstance({0}, str)",lambda a:isinstance(a,str),3,"string str strings type",""),
("isflt","a","isinstance({0}, float)",lambda a:isinstance(a,float),3,"float type",""),
("isseq","a","isinstance({0}, (list, tuple))",lambda a:isinstance(a,(list,tuple)),3,"list nested type",""),
("isvow","s","({0}.lower() in 'aeiou')",lambda a:a.lower() in "aeiou",3,"vowel vowels",""),
("dflat","q","_dflat({0})",HFN["_dflat"],3,"flatten nested recursive recursively deep","h"),
("droot","i","_droot({0})",HFN["_droot"],3,"digital root repeatedly single digit until","h"),
("vmult","ii","_vmult({0}, {1})",HFN["_vmult"],3,"times divisible divide multiplicity power exponent","h"),
("tobase","ii","_tobase({0}, {1})",HFN["_tobase"],3,"base convert conversion representation","h"),
("polyg","ii","((({0} - 2) * {1} * {1} - ({0} - 4) * {1}) // 2)",lambda a,b:((a-2)*b*b-(a-4)*b)//2 if 2<a<=100 else _no(),3,"polygonal nth number",""),
("zip3","qqq","list(zip({0}, {1}, {2}))",lambda a,b,c:list(zip(a,b,c)),3,"three corresponding elementwise zip combine",""),
]
PATS=(("map","single",1,"each every convert apply"),("filter","single",1,"only filter remove keep select those which contain"),("sortby","single",2,"sort sorted order according"),("sortbyr","single",3,"descending reverse"),("maxby","single",2,"max maximum largest longest most highest"),("minby","single",2,"min minimum smallest shortest least lowest"),("firstidx","single",2,"first index position find"),("dictc","single",3,"dictionary dict mapping"),("flatmap","single",2,"flatten"),("map","pair",2,"pair pairs two close closer distinct"),("map","adj",2,"consecutive adjacent monotonic neighbor next increasing decreasing"),("map","zip",2,"corresponding pairwise elementwise position both"))
FOLDS=(("_a * 10 + _x",lambda a,x:a*10+x,"0",0,"digit digits integer number convert combine join",0),("_a * 2 + _x",lambda a,x:a*2+x,"0",0,"binary bits decimal convert",0),("_a * 10 + int(_x)",lambda a,x:a*10+int(x),"0",0,"digit digits integer number convert combine join",0),("_a * 2 + int(_x)",lambda a,x:a*2+int(x),"0",0,"binary bits decimal convert",0),("_x + _a",lambda a,x:x+a,"''","","reverse reversed backward",0),("_a + _x",op.add,"",None,"sum total add",0),("_a * _x",op.mul,"",None,"product multiply",0),("max(_a, _x)",max,"",None,"max maximum largest greatest",0),("min(_a, _x)",min,"",None,"min minimum smallest least",0),("_a - _x",op.sub,"",None,"subtract difference minus",0),("_a + _x",op.add,"",None,"sum total add running cumulative balance",1),("max(_a, _x)",max,"",None,"max maximum largest greatest rolling highest",1),("min(_a, _x)",min,"",None,"min minimum smallest least lowest",1),("_a * _x",op.mul,"",None,"product multiply",1))
DIGW="['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']"
KWC={"vowel":["'aeiou'","'aeiouAEIOU'"],"consonant":["'aeiou'","'aeiouAEIOU'"],"digit":["'0123456789'"],"space":["' '"],"whitespace":["' '"],"comma":["','"],"alphabet":["'abcdefghijklmnopqrstuvwxyz'"],"letter":["'abcdefghijklmnopqrstuvwxyz'"],"underscore":["'_'"],"hyphen":["'-'"],"dash":["'-'"],"parenthes":["'('","')'"],"bracket":["'('","')'","'['","']'"],"pi":["math.pi"],"circle":["math.pi"],"sphere":["math.pi"],"circumference":["math.pi"],"cylinder":["math.pi"],"cone":["math.pi"],"radius":["math.pi"],"two":["2"],"three":["3"],"four":["4"],"five":["5"],"six":["6"],"seven":["7"],"eight":["8"],"nine":["9"],"ten":["10"],"hundred":["100"],"thousand":["1000"],"twice":["2"],"double":["2"],"triple":["3"],"cube":["3"],"binary":["2"],"decimal":["10"],"percent":["100"],"triangle":["3"],"triangular":["3"],"square":["4"],"pentagon":["5"],"pentagonal":["5"],"hexagon":["6"],"hexagonal":["6"],"heptagon":["7"],"octagon":["8"],"octagonal":["8"],"decagon":["10"],"none":["None"],"numeral":[DIGW],"numberal":[DIGW]}
GENC=(("0",2),("1",2),("2",2),("''",3),("' '",3),("-1",3),("10",3),("None",3))
STRONG=set("divisor divide divides sum product multiply sort sorted reverse reversed maximum max largest greatest minimum min smallest count average mean even odd prime square cube vowel unique distinct duplicate palindrome frequency absolute divisible digit upper uppercase lower lowercase length longest shortest binary factorial fibonacci gcd lcm power replace concatenate join split common intersection union flatten positive negative difference ascii median occurrence kth nth index swap rotate ascending descending root remainder".split())
STOP=set("a an the of to in and or is are be for return returns given write function python find that which with from by this it as if not on its each all element elements value values input output check whether integer integers number numbers list lists string strings array get set true false".split())
KREQ=set("pentagon pentagonal hexagon hexagonal heptagon octagon octagonal decagon triangular percent hundred thousand".split())
NUMW={"2":set("two twice double half pair even odd square binary second".split()),"10":set("ten decimal digit".split()),"-1":set("minus negative".split()),"None":{"none"}}
CUES={"pair":(r"\b(any|every|each|all) (two|pairs?)\b(?! consecutive| adjacent)|\beach other\b|\bpairs? of (numbers|elements|integers|items|values)\b|\bnumber of pairs\b",()),"acc":(r"\bat any point\b|\bbalance\b|\brunning\b|\bso far\b|\bcumulative\b|\brolling\b",()),"all":(r"\b(every|all)\b",(bool,)),"count":(r"\bcount\b|\bhow many\b|\bnumber of\b",(int,)),"index":(r"\bindex\b|\bindices\b|\bpositions?\b",(int,list))}
CREQ={"pair":r"\[\w+ \+ 1:\]|zip\(sorted|combinations","acc":r"accumulate|\[:\w+ \+ 1\]","all":r"\ball\(|\bany\(|\bfor |set\(|sorted\(","count":r"\blen\(|\.count\(|\bsum\(","index":r"enumerate|\.index\(|\.find\(|range\(|zip\("}
QRX=re.compile(r"\b(any|all|every|each|elements|items|at any point|so far|pairs?)\b");PRX=re.compile(r"\b(first|last|second|third|nth|kth|index|indices|position|head|tail|beginning|end|start|middle)\b")
IDEM={"sorted","set","abs","list","tuple","str","int","float","round","lower","upper","strip","dedup"}
def _fn(n):return n.func.id if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) else n.func.attr if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and not n.args else None
def _redund(code):
 try:return sum(1 for n in ast.walk(ast.parse(code,mode="eval")) if _fn(n)=="len" and n.args and isinstance(n.args[0],ast.ListComp) and not any(g.ifs for g in n.args[0].generators) or _fn(n) in IDEM and ((n.args and _fn(n.args[0])==_fn(n)) or (isinstance(n.func,ast.Attribute) and _fn(n.func.value)==_fn(n))) or isinstance(n,ast.Subscript) and isinstance(n.value,ast.Subscript) and ast.unparse(n.slice)==ast.unparse(n.value.slice)=="::-1")
 except Exception:return 0
def _lem(w,wl):return [q for q in dict.fromkeys([w]+[] if len(w)<5 else [w]+([w[:-3]+"y"] if w.endswith("ies") else [])+([w[:-2]] if w.endswith(("es","ed","er")) else [])+([w[:-1]] if w.endswith(("s","d","r")) else [])+([w[:-3],w[:-3]+"e",w[:-4]] if w.endswith(("ing","est")) else [])+([w[:-2]] if w.endswith("ly") else [])) if q in wl][:2]
class Sem:
 def __init__(s):
  z=np.load(CONC/"wordnet_graph.npz",allow_pickle=True);s.x=np.load(CONC/"sphere.npz")["xyz"][:len(z["node_names"])].astype(np.float32);s.wl={w:k for k,w in enumerate(z["word_list"].tolist())};s.wp,s.ws=z["word_ptr"],z["word_syn"]
  m=np.isin(z["edge_type"],REL_T);a,b=np.concatenate([z["edge_src"][m],z["edge_dst"][m]]),np.concatenate([z["edge_dst"][m],z["edge_src"][m]]);o=np.argsort(a,kind="stable");s.nd,s.nptr=b[o],np.searchsorted(a[o],np.arange(len(s.x)+1))
  s.K=sorted({k for kw in [t[5] for t in TILE_DEFS]+[t[3] for t in PATS]+[f[4] for f in FOLDS]+list(STRONG) for k in kw.split()});s.l2k,s.s2k,s.r2k,ks={},{},{},{k:s.sn(k) for k in s.K}
  for k in s.K:
   [s.l2k.setdefault(l,set()).add(k) for l in [k]+_lem(k,s.wl)];[s.s2k.setdefault(q,set()).add(k) for q in ks[k][:1]];[s.r2k.setdefault(q,set()).add(k) for q in s.nb(ks[k][:1])]
  s.kk=[k for k in s.K if ks[k]];s.kx=s.x[[ks[k][0] for k in s.kk]]
 def sn(s,w):return list(dict.fromkeys(int(q) for l in _lem(w,s.wl) for q in s.ws[s.wp[s.wl[l]]:s.wp[s.wl[l]+1]][:2]))[:3]
 def nb(s,ss):return {int(q) for a in ss for q in s.nd[s.nptr[a]:s.nptr[a+1]]}
 def aff(s,w):
  S,out=s.sn(w),{}
  put=lambda ks,v:[out.__setitem__(k,max(v,out.get(k,0))) for k in ks]
  (put([k for k,c in zip(s.kk,s.kx@s.x[S[0]]) if c>=SPHT],0.4) if S and SEMM=="full" else None);[put(s.r2k.get(q,()),0.6) for q in S[:1]];[put(s.s2k.get(q,()),0.6) for q in s.nb(S[:1])];[put(s.s2k.get(q,()),0.8) for q in S[:1]];[put(s.l2k.get(l,()),1.0) for l in [w]+([w[:-1]] if len(w)>3 and w[-1]=="s" else [])+_lem(w,s.wl)]
  return out
@functools.lru_cache(None)
def sem():return Sem() if SEMM!="off" and (CONC/"wordnet_graph.npz").exists() else None
FMOD={"math","itertools","functools","collections","string","re","operator","statistics","heapq"}
FCMP={"Gt":"Lt","GtE":"LtE","Is":"Eq","IsNot":"NotEq"}
TYV={bool:"bool",int:"int",float:"float",str:"str",list:"list",tuple:"tuple",dict:"dict",set:"set",frozenset:"set",type(None):"none"}
TYA={"integer":"int","int":"int","long":"int","string":"str","str":"str","unicode":"str","text":"str","bool":"bool","boolean":"bool","float":"float","number":"float","double":"float","list":"list","sequence":"list","iterable":"list","array":"list","tuple":"tuple","dict":"dict","dictionary":"dict","mapping":"dict","set":"set","frozenset":"set","none":"none"}
PSHAPE={"map":"[x for x in a]","filter":"[x for x in a if b]","sortby":"sorted(a, key=lambda x: b)","sortbyr":"sorted(a, key=lambda x: b, reverse=True)","maxby":"max(a, key=lambda x: b)","minby":"min(a, key=lambda x: b)","firstidx":"next((i for i, x in enumerate(a) if b), -1)","dictc":"{x: b for x in a}","flatmap":"[y for x in a for y in b]","pair":"[b for i, x in enumerate(a) for y in a[i + 1:]]","adj":"[b for x, y in zip(a, a[1:])]","zip":"[b for x, y in zip(a, c)]","fold":"functools.reduce(lambda _a, _x: _a + _x, a)","scan":"list(itertools.accumulate(a, lambda _a, _x: _a + _x))","tern":"(a if b else c)"}
def _cv(x):return x.value if isinstance(x,ast.Constant) else -x.operand.value if isinstance(x,ast.UnaryOp) and isinstance(x.op,ast.USub) and isinstance(x.operand,ast.Constant) and type(x.operand.value) in (int,float) else None
def _fk(v):return "k:bool" if type(v) is bool else "k:None" if v is None else ("k:"+repr(v) if v in (0,1,2,10,-1) else "k:int") if type(v) is int else "k:float" if type(v) is float else ("k:"+repr(v) if v in (""," ") else "k:str") if type(v) is str else None
def _i1(x):return isinstance(x,ast.BinOp) and isinstance(x.op,ast.Add) and _cv(x.right)==1 and not isinstance(x.left,ast.Constant)
def _fcall(n):f=n.func;a=f.id if isinstance(f,ast.Name) else f.attr if isinstance(f,ast.Attribute) else "";return ([("c:" if isinstance(f,ast.Name) or isinstance(f.value,ast.Name) and f.value.id in FMOD else "m:")+a] if a and a[0]!="_" else [])+["k:"+k.arg for k in n.keywords if k.arg in ("key","reverse")]+(["p:adj"] if a=="zip" and any(isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Slice) and _cv(x.slice.lower)==1 for x in n.args) else [])+(["p:pair"] if a=="range" and n.args and _i1(n.args[0]) else [])+(["p:acc"] if a=="accumulate" else [])
def _fsub(s):c=_cv(s);return ((["s:rev"] if _cv(s.step)==-1 else ["s:step"] if s.step else ["s:slice"] if s.lower and s.upper else ["s:drop"] if s.lower else ["s:take"] if s.upper else [])+(["p:pair"] if _i1(s.lower) else [])) if isinstance(s,ast.Slice) else ["s:"+str(c)] if type(c) is int and c in (0,1,-1) else ["s:idx"]
def _fcomp(n):return ["p:iter"]+(["p:filter"] if any(g.ifs for g in n.generators) else [])+(["p:nest"] if len(n.generators)>1 else [])+(["p:dictc"] if isinstance(n,ast.DictComp) else [])
FX={ast.Call:_fcall,ast.BinOp:lambda n:["o:"+type(n.op).__name__],ast.Compare:lambda n:["o:"+FCMP.get(type(o).__name__,type(o).__name__) for o in n.ops],ast.UnaryOp:lambda n:["o:Not"] if isinstance(n.op,ast.Not) else [_fk(_cv(n))] if _cv(n) is not None else ["o:USub"] if isinstance(n.op,ast.USub) else [],ast.BoolOp:lambda n:["o:"+type(n.op).__name__],ast.Subscript:lambda n:_fsub(n.slice),**{c:_fcomp for c in (ast.ListComp,ast.GeneratorExp,ast.SetComp,ast.DictComp)},ast.For:lambda n:["p:iter"]+(["p:filter"] if any(isinstance(b,ast.If) for b in n.body) else [])+(["p:nest"] if any(isinstance(b,ast.For) for b in n.body) else []),ast.While:lambda n:["p:while"],ast.Lambda:lambda n:["p:lambda"],ast.IfExp:lambda n:["p:cond"],ast.If:lambda n:["p:cond"],ast.AugAssign:lambda n:["p:acc"],ast.Constant:lambda n:[_fk(n.value)],ast.Attribute:lambda n:["k:pi"] if n.attr=="pi" else []}
def _nil(n):return ()
def feats(src):
 t=src if isinstance(src,ast.AST) else ast.parse(textwrap.dedent(src).strip());fd=isinstance(t,(ast.FunctionDef,ast.AsyncFunctionDef))
 return {x for s in (t.body[1:] if fd and ast.get_docstring(t) is not None else t.body if fd or isinstance(t,ast.Module) else [t]) for n in ast.walk(s) for x in FX.get(type(n),_nil)(n) if x}
@functools.lru_cache(None)
def tfeats():return {**{n:frozenset(feats(_rd(_parts(tp),["a","b","c"]))) for n,pr,tp,*_ in TILE_DEFS},**{"@"+k:frozenset(feats(v)) for k,v in PSHAPE.items()}}
@functools.lru_cache(None)
def vocab():return tuple(sorted(set().union(*tfeats().values())|{_fk(v) for v in (0,1,2,10,-1,""," ",None,True,3,0.5,"x")}|{"k:pi","p:acc","p:while","p:nest","p:cond"}))
def _h(t):b=t.encode();return zlib.crc32(b)|(zlib.adler32(b)&0xFFFFFF)<<32
def _stem(w):return w[:-3]+"y" if len(w)>4 and w.endswith("ies") else w[:-1] if len(w)>3 and w[-1]=="s" and w[-2]!="s" else w
def _at(s):m=re.findall(r"[a-z]+",re.sub(r"typing\.|optional\[","",str(s or "").lower()));b=TYA.get(m[0]) if m else None;e=(TYA.get(m[2]) if len(m)>2 and m[1]=="of" else TYA.get(m[1]) if len(m)>1 else None) if b in ("list","tuple","set") else None;return ([b]+([f"{b}[{e}]"] if e else [])) if b else []
def _vt(v):b=TYV.get(type(v));e=[TYV.get(type(x)) for x in v] if b in ("list","tuple","set") and v else [];return ([b]+([f"{b}[{e[0]}]"] if e and len(set(e))==1 and e[0] else [])) if b else []
def ptoks(doc,name,params,at,rt):return {"w:"+_stem(t) for t in re.findall(r"[a-z]+",(doc+" "+re.sub(r"([a-z])([A-Z])",r"\1 \2",name).replace("_"," ")).lower()) if len(t)>1}|{"v:"+re.sub(r"\d+$","",p.lower()) for p in params if p not in ("self","cls")}|{"a:"+x for a in at for x in a}|{"r:"+x for x in rt}
class Prior:
 def __init__(s,path):z=np.load(path);s.pk,s.pc,s.tk,s.tc,s.fc,s.N,s.V=z["pk"],z["pc"],z["tk"],z["tc"],z["fc"].astype(np.float64),float(z["n"]),[str(x) for x in z["vocab"]];s.fi={f:j for j,f in enumerate(s.V)};s.hd={n:[s.fi[f] for f in fs if f in s.fi and f[0] in "cmp"] or [s.fi[f] for f in fs if f in s.fi] for n,fs in tfeats().items()}
 def _get(s,k,tab,cnt):i=np.minimum(np.searchsorted(tab,k),len(tab)-1);return np.where(tab[i]==k,cnt[i],0).astype(np.float64)
 def fvec(s,toks,cfg=None):
  c={**PCFG,**(cfg or {})};T=sorted(toks);h=np.array([_h(t) for t in T],np.uint64).reshape(-1);ct=s._get(h,s.tk,s.tc);C=s._get((h[:,None]<<np.uint64(8))|np.arange(len(s.V),dtype=np.uint64)[None,:],s.pk,s.pc);b=float(c["b"]);pf=(s.fc+0.5)/(s.N+1);L=np.log((C+b*pf)/((ct[:,None]+b)*pf));w=np.where(ct>0,np.log((s.N+1)/(ct+1)),0.0)**float(c["idf"]);g=np.array([{"a":"a","v":"v","r":"r"}.get(t[0],"w") for t in T])
  a=w*np.array([float(c["g"+x]) for x in g]) if T else w;q=np.clip(((a[:,None]*(C+b*pf)/(ct[:,None]+b)).sum(0)+1e-3*pf)/(a.sum()+1e-3),1e-9,1-1e-9);lo=np.log(q/(1-q))-(np.log(pf/(1-pf)) if c["mode"]=="lr" else 0)
  return (lo if c["mode"] in ("lo","lr") else sum(float(c["g"+k])*(w[g==k,None]*L[g==k]).sum(0)/(max(1e-9,w[g==k].sum()) if c["norm"]=="1" else 1) for k in "wavr") if T else np.zeros(len(s.V)))*np.array([f[0] in c["fc"] for f in s.V])
 def cand(s,fv,expr,agg=None):
  try:j=[s.fi[f] for f in feats(expr) if f in s.fi]
  except Exception:j=[]
  a=agg or PCFG["agg"];return float(fv[j].max() if a=="max" and j else fv[j].sum()/(max(1,len(j)) if a=="mean" else 1))
 def tile(s,fv,name):j=s.hd.get(name);return float(fv[j].max()) if j else 0.0
@functools.lru_cache(None)
def prior(path=None):p=Path(path or PRIORP);return Prior(p) if p.exists() and os.environ.get("TS_PRIOR_OFF")!="1" else None
class Tm:
 __slots__=("c","v","k","s","u","d","w","f")
 def __init__(s,c,v,k,z,u,d,w,f=0):s.c,s.v,s.k,s.s,s.u,s.d,s.w,s.f=c,v,k,z,u,d,w,f
class Tl:
 __slots__=("n","p","parts","fn","cost","comm","dep","hit","ar","fam","mc","sfn")
 def __init__(s,n,p,parts,fn,cost,comm=False,dep=EMPTY,hit=0,mc=99):s.n,s.p,s.parts,s.fn,s.cost,s.comm,s.dep,s.hit,s.ar,s.fam,s.mc,s.sfn=n,p,parts,fn,cost,comm,dep,hit,len(p),FAM.get(n,0),mc,functools.partial(_try,fn)
def _parts(t):return re.split(r"\{(\d)\}",t)
def _rd(parts,a):return "".join(p if k%2==0 else a[int(p)] for k,p in enumerate(parts))
def _call(name,n):return _parts(name+"("+", ".join("{%d}"%k for k in range(n))+")")
DEPF={"m":"math","f":"functools","i":"itertools"}
MIXED={"isint","isstr","isflt","isseq"}
class _Err:
 __slots__=()
 def __repr__(s):return "<ERR>"
ERR=_Err()
def _try(fn,*a):
 try:return fn(*a)
 except Exception:return ERR
def _mut(v,g,x=0):
 t=type(v);k=g.randrange(7 if x else 4)
 return (g.random()<0.5) if t is bool else g.choice((0,1,-1,2,-v,v+1,v-1,2*v+1,v*10)) if t is int and k>=4 else (max(0,v+g.choice((-2,-1,1,2,3,5,7))) if v>=0 else v-g.choice((1,2,3,5))) if t is int else (round(-v,3) if k==4 else float(g.choice((0,1,-1,2))) if k==5 else round(v*g.uniform(0.5,1.5)+g.uniform(-1,1),3)) if t is float else _smut(v,g,k) if t is str else _lmut(v,g,k,x) if t is list else tuple(_mut(e,g,x) if g.random()<0.4 else e for e in v) if t is tuple else (dict(g.sample(list(v.items()),max(1,len(v)-1))) if k<4 else {a:_mut(b,g,x) for a,b in v.items()}) if t is dict and v else v
def _smut(v,g,k):
 j=g.randrange(len(v)) if v else 0
 return g.choice("abcxyz ") if not v else "".join(g.sample(v,len(v))) if k==0 else v[:j]+v[j+1:] if k==1 else v[:j]+v[j]+v[j:] if k==2 else v[::-1] if k==4 else v+v if k==5 else v[:j]+g.choice("aZ09 ,.-!")+v[j:] if k==6 else v[:j]+v[j].swapcase()+v[j+1:] if v[j].swapcase()!=v[j] else v[:j]+chr(ord(v[j])+1)+v[j+1:]
def _lmut(v,g,k,x=0):
 j=g.randrange(len(v)) if v else 0;w=g.sample(v,len(v)) if k==0 else v[:j]+v[j+1:] if k==1 and len(v)>1 else v+[_mut(v[j],g,x)] if k==2 and v else v+[v[j]] if k==4 and v else [-e if type(e) in (int,float) else e for e in v] if k==5 else v[j:j+1] if k==6 else v
 return [_mut(e,g,x) if g.random()<(0.9 if k==3 else 0.4 if k<4 else 0) else e for e in w]
def _probes(rows,K,x=0):
 out,seen=[],{repr(r) for r in rows}
 for k in range(K*4 if rows else 0):
  g=random.Random(zlib.crc32((repr(rows[k%len(rows)])+str(k)).encode()));r=tuple(_mut(a,g,x) for a in rows[k%len(rows)])
  (out.append(r),seen.add(repr(r))) if repr(r) not in seen and len(out)<K else None
 return out
def _ev(v):t=type(v);nm=t is list and all(type(e) in (int,float) for e in v);return [] if t is bool else [0,1,-1,2,-7,100] if t is int else [0.0,1.0,-1.5,0.5] if t is float else ["",v[:1] or "a",v+v,v.swapcase(),v[::-1]," "] if t is str else [[],v[:1],v+v,v[::-1],[v[0]]*3 if v else []]+([[-e for e in v],[0]*len(v)] if nm and v else []) if t is list else [v[::-1]] if t is tuple else [{}] if t is dict else []
def _edges(rows):
 out,seen=[],{repr(r) for r in rows}
 for p in range(len(rows[0]) if rows else 0):
  for b in rows[:3]:
   for e in _ev(b[p]):r=b[:p]+(e,)+b[p+1:];(out.append(r),seen.add(repr(r))) if repr(r) not in seen else None
 return out
FAM={"add":1,"sub":1,"inc":1,"dec":1,"mul":2,"dbl":2,"div":2,"fdiv":2,"half":2,"halff":2}
def _bc(sp,n,kw,c):d=max(1,c-(sp is not None and sp.aff(kw)>=AFT));e=max(1,d-(sp.pb(n) if sp is not None else 0));return e,({("b:" if e<d else "B:")+n} if e!=d and PBIAS["rs"]=="1" else EMPTY)
def library(sp=None):return sorted([Tl(n,tuple(PR[ch] for ch in pv),_parts(tp),fn,_bc(sp,n,kw,c)[0],"C" in fl,frozenset(DEPF[ch] for ch in fl if ch in DEPF)|({"h:"+tp.split("(")[0]} if "h" in fl else EMPTY)|_bc(sp,n,kw,c)[1],sp.hm(kw) if sp else 0) for n,pr,tp,fn,c,kw,fl in TILE_DEFS for pv in pr.split("|")],key=lambda t:(-bool(t.hit),t.cost))
def _strip_imports(src):return re.sub(r"(?m)^[ \t]*(?:import [\w., ]+|from [\w.]+ import [^\n]+)$","",src)
def _wrap(fn,cp=True):return (lambda *a:timed(fn,*copy.deepcopy(a),t=0.1)) if cp else (lambda *a:timed(fn,*a,t=0.05))
def _annp(a):s=(ast.unparse(a) if a is not None else "").lower();return S_ if "str" in s else N if ("int" in s or "float" in s) else Q if ("list" in s or "tuple" in s) else B if "bool" in s else 0
def _major(k):e=k&(SN|SS|SQ|SB);return (k&(N|B|S_|Q|D))|(e if e!=(SN|SS|SQ|SB) else 0)
def _lit(code):return (math.pi,frozenset({"math"})) if code=="math.pi" else (ast.literal_eval(code),EMPTY)
class Spec:
 def __init__(s,prompt,checks):
  s.prompt=prompt;fs=[n for n in ast.parse(prompt).body if isinstance(n,ast.FunctionDef)];f=fs[-1];s.name=f.name;s.params=[a.arg for a in f.args.args];s.ann,s.ret=[a.annotation for a in f.args.args],f.returns;s.hnodes=fs[:-1];dv=f.args.defaults;s.dflt={}
  for p,d in zip(s.params[len(s.params)-len(dv):],dv):
   try:s.dflt[p]=ast.literal_eval(d)
   except Exception:pass
  s.doc=ast.get_docstring(f) or "";ls=s.doc.splitlines()
  ex=re.compile(rf"\b{re.escape(s.name)}\s*\(|^\s*assert\b|>>>");s.desc=" ".join(l for k,l in enumerate(ls) if not ex.search(l) and not (k and ">>>" in ls[k-1]))
  toks=re.findall(r"[a-z]+",(s.desc+" "+re.sub(r"([a-z])([A-Z])",r"\1 \2",s.name).replace("_"," ")).lower())
  s.words=frozenset(w for t in toks for w in (t,t[:-1] if len(t)>3 and t.endswith("s") else t,t[:-2] if len(t)>4 and t.endswith("es") else t))
  nd=[f"_x0eq(({m.group(1)}),(None))" for m in re.finditer(r"^\s*>>>\s*("+re.escape(s.name)+r"\(.*\))\s*\n(?=\s*(?:>>>|$))",s.doc+"\n",re.M)];checks=list(checks)+[c for c in nd if c not in checks]
  s.rows,s.keys,s.T,s.raw,s.rawc,s.used,s.n_checks=[],[],{},[],[],[],len(checks)
  for ch in checks:s._parse(ch)
  S=sem();s.consts=s._consts();s.wa={t:(S.aff(t) if S else {k:1.0 for k in (t,t[:-1] if len(t)>3 and t[-1]=="s" else t,t[:-2] if len(t)>4 and t.endswith("es") else t)}) for t in dict.fromkeys(toks) if t not in STOP and len(t)>2};s.ka={};sw={}
  [s.ka.__setitem__(k,max(v,s.ka.get(k,0))) for a in s.wa.values() for k,v in a.items()];[sw.setdefault(frozenset(k for k,v in a.items() if v>=0.99) or t,t) for t,a in s.wa.items() if any(a.get(w,0)>=0.8 for w in STRONG)]
  s.strong=(list(sw.values())+[c[4] for c in s.consts if c[4]])[:30];s.sb={w:1<<k for k,w in enumerate(s.strong)};lc=s.desc.lower();ot={type(v) for v in s.T.values()}
  s.cues={k for k,(rx,ty) in CUES.items() if re.search(rx,lc) and (not ty or ot&set(ty))};s.quant=bool(QRX.search(lc)) and not PRX.search(lc);s.cix=re.compile(r"(?<![\w\].)])(?:"+"|".join(map(re.escape,s.params))+r")\[-?\d+\]") if s.params else None
  P=prior();s.fv=P.fvec(s.toks()) if P else None
 def toks(s):return ptoks(s.desc,s.name,s.params,[(_at(ast.unparse(a)) if a is not None else []) or (_vt(s.rows[0][k]) if s.rows else []) for k,a in enumerate(s.ann)],(_at(ast.unparse(s.ret)) if s.ret is not None else []) or (_vt(s.T[min(s.T)]) if s.T else []))
 def pri(s,expr):P=prior();return P.cand(s.fv,expr) if P and s.fv is not None else 0.0
 def pb(s,name):P=prior();v=P.tile(s.fv,name) if P and s.fv is not None else 0.0;return int(v>=float(PBIAS["t"]))-int(v<=float(PBIAS["dn"]))
 def aff(s,kw):return max((s.ka.get(k,0) for k in kw.split()),default=0)
 def hm(s,kw):ks=kw.split() if isinstance(kw,str) else kw;return functools.reduce(op.or_,(b for t,b in s.sb.items() if t in s.wa and max((s.wa[t].get(k,0) for k in ks),default=0)>=COVT),0)
 def bind(s,a,k):
  if len(a)>len(s.params) or set(k)-set(s.params):raise TypeError("bind")
  d={**s.dflt,**dict(zip(s.params,a)),**k};return tuple(d[p] for p in s.params)
 def _row(s,c):
  if any(isinstance(a,ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords):raise ValueError("star")
  v=s.bind(tuple(safe_eval(ast.unparse(x)) for x in c.args),{k.arg:safe_eval(ast.unparse(k.value)) for k in c.keywords});key=repr(v)
  (s.keys.append(key),s.rows.append(v)) if key not in s.keys else None
  return s.keys.index(key)
 def _parse(s,ch):
  try:
   e=ast.parse(ch,mode="eval").body;calls=[n for n in ast.walk(e) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==s.name]
   if not calls:return
   rs=[s._row(c) for c in calls];lr=e.args if isinstance(e,ast.Call) and isinstance(e.func,ast.Name) and e.func.id=="_x0eq" and len(e.args)==2 else (e.left,e.comparators[0]) if isinstance(e,ast.Compare) and len(e.ops)==1 and isinstance(e.ops[0],ast.Eq) else (None,None)
   if len(calls)==1 and lr[0] is calls[0]:s.T.setdefault(rs[0],safe_eval(ast.unparse(lr[1])))
   else:s.rawc.append(compile(safe_check(ch,"eval"),"<chk>","eval"));s.raw.append(ch)
   s.used.append(ch)
  except Exception:return
 def _consts(s):
  out={}
  req={c for w in s.words&KREQ for c in KWC[w]};lits=[m.group(0) for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)",s.desc)][:8]
  for code in lits+[repr(a or b) for a,b in re.findall(r"'([^'\n]{1,12})'|\"([^\"\n]{1,12})\"",s.desc)][:6]+[c for w in sorted(s.words) for c in KWC.get(w,())]+([DIGW] if len(s.words&set(ast.literal_eval(DIGW)))>=3 else []):
   try:v,d=_lit(code);out.setdefault(repr(v),(code,v,1,d,"#"+code if (code in lits or code in req) and code not in ("0","1") else ""))
   except Exception:pass
  for code,c in GENC:v,d=_lit(code);out.setdefault(repr(v),(code,v,c,d|({"k:"+code} if code in NUMW and not re.search(r"(?<![\w.])"+re.escape(code)+r"(?![\w.])",s.doc) and not NUMW[code]&s.words else EMPTY),""))
  return list(out.values())
 def helper_tiles(s):
  if not s.hnodes:return []
  try:g=safe_exec(_strip_imports(s.prompt),TYPING)
  except Exception:return []
  return [Tl(h.name,tuple(_annp(p.annotation) for p in h.args.args[:len(h.args.args)-len(h.args.defaults)]),_call(h.name,len(h.args.args)-len(h.args.defaults)),_wrap(g[h.name]),1,False,frozenset({"p:"+h.name}),0) for h in s.hnodes if callable(g.get(h.name)) and 1<=len(h.args.args)-len(h.args.defaults)<=3]
def load_store(path):p=Path(path);return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if path and p.exists() else []
def learned_tiles(recs,W,sp=None):
 out,src=[],{}
 for ov,r in sorted(((len((set(r["words"])-STOP)&(W-STOP)),r) for r in recs),key=lambda x:-x[0])[:LMAX]:
  try:fn=safe_exec(_strip_imports(r["src"]),XMODS)[r["name"]] if ov else None
  except Exception:fn=None
  (out.append(Tl(r["name"],tuple(_major(k) for k in r["kinds"]),_call(r["name"],len(r["kinds"])),_wrap(fn,False),1 if ov>=2 else 2,False,frozenset({"h:"+r["name"]}),sp.hm(r["words"]) if sp else 0,3)),src.setdefault(r["name"],r["src"])) if fn else None
 return out,src
class Ctx:
 def __init__(s,Sr,kind,R,R0,xc=0,pu=0,pd=EMPTY,owner=None,spans=None,xv=None,clause=None,et=None,mode=None,xs="",depth=0,lb=0,xbit=0,xyb=0):
  s.S,s.kind,s.R,s.R0,s.xc,s.pu,s.pd,s.owner,s.spans,s.xv,s.clause,s.et,s.mode,s.xs,s.depth,s.lb,s.xbit,s.xyb=Sr,kind,R,R0,xc,pu,pd,owner,spans,xv,clause,et,mode,xs,depth,lb,xbit,xyb
  s.seen,s.lv,s.idx,s.level,s.sd,s.mm,s.cm,s.kids,s.spq={},{},{m:{} for m in Sr.pm},0,{},{},{},[],[];s.etk=functools.reduce(op.and_,map(_kv,et)) if et else 0
 def seed(s,c,v,code,u=0,d=EMPTY,w=0):s.sd.setdefault(c,[]).append((v,code,u,d,w))
 def grow(s,c):
  while s.level<c:
   s.level+=1;c0=s.level
   for v,code,u,d,w in s.sd.pop(c0,()):s.add(v,c0,u,d,w,code)
   for t in s.S.tiles:s.apply(t,c0)
   (s.S.patterns(s,c0) if s.kids or not s.depth else None);s.S.split(s);sq,s.spq=s.spq,[]
   for t in sq:s.S.spawn(s,t)
 def add(s,v,c,u,d,w,code,f=0):
  Sr=s.S;Sr.tick();R0=s.R0
  try:key=repr(v)
  except ValueError:return None
  if key in s.seen or len(key)>200000:return (Sr.on_outer(Tm(code() if callable(code) else code,v,s.seen[key].k,c,u,d,w,f),key,Sr.hk[key],True) if not s.depth and key in Sr.hk and s.seen[key] is not None else None)
  ks=[_kv(x) for x in v[:R0]];pk=[x is ERR or bool(_kv(x)) for x in v[R0:]];s.seen[key]=None
  if not all(ks):return None
  v,k=(v if all(pk) else v[:R0]+tuple(x if ok else ERR for x,ok in zip(v[R0:],pk))),functools.reduce(op.and_,ks);t=Tm(code() if callable(code) else code,v,k,c,u,d,w,f);s.seen[key]=t;s.seen[repr(v)]=t;s.lv.setdefault(c,[]).append(t)
  for m in Sr.pm:(s.idx[m].setdefault(c,[]).append(t) if k&m==m else None)
  Sr.on_outer(t,key) if not s.depth else Sr.on_elem(s,t)
  return t
 def apply(s,t,c):
  r=c-t.cost;ix=s.idx
  if r<t.ar:return
  if t.ar==1:
   for a in ix[t.p[0]].get(r,()) if r<=t.mc else ():s.mk(t,(a,),c)
  elif t.ar==2:
   for c1 in range(max(1,r-t.mc),min(r,t.mc+1)):
    A,Bs=ix[t.p[0]].get(c1),ix[t.p[1]].get(r-c1)
    if A and Bs and not (t.comm and c1>r-c1):
     for q,a in enumerate(A):
      for b in (Bs[q+1:] if t.comm and c1==r-c1 and t.p[0]==t.p[1] else Bs):s.mk(t,(a,b),c)
  else:
   for c1 in range(1,min(r-1,t.mc+1)):
    for c2 in range(max(1,r-c1-t.mc),min(r-c1,t.mc+1)):
     A,Bs,Cs=ix[t.p[0]].get(c1),ix[t.p[1]].get(c2),ix[t.p[2]].get(r-c1-c2)
     for a in (A or ()) if Bs and Cs else ():
      for b in Bs:
       for x in Cs:s.mk(t,(a,b,x),c)
 def mk(s,t,a,c):
  s.S.tick();R0=s.R0;u,d,w=0,t.dep,t.hit;cf=t.fam if t.fam and t.ar==1 else 0
  for x in a:u,w,d=u|x.u,w|x.w,(d|x.d if x.d else d)
  if not u or (cf and any(x.f==cf for x in a if x.u)) or (t.ar>1 and len({id(x) for x in a})<t.ar) or (t.n in MIXED and a[0].k&(N|B|S_|Q|D|E|O)):return
  try:v=tuple(map(t.fn,*(x.v[:R0] for x in a)))
  except TypeError:return
  except Exception:return s.S.partial(s,t,a,c,u,d,w) if not s.depth and s.S.split_on else None
  s.add(v+tuple(map(t.sfn,*(x.v[R0:] for x in a))),c,u,d,w,lambda:_rd(t.parts,[x.c for x in a]),cf)
class Search:
 def __init__(z,sp,budget=10.0,recs=(),maxctx=28,only=None):
  z.sp,z.dl,z.n,z.hits,z.ctxs,z.iters,z.maxctx,z.tried,z.bods,z.c0=sp,time.perf_counter()+budget,0,[],[],[],maxctx,set(),set(),None
  z.rows=sp.rows+_probes(sp.rows,int(os.environ.get("TS_PROBES","6")));z.RE=len(z.rows);eg=tuple(type(a)() if type(a) in (list,str,tuple,dict) else a for a in sp.rows[0]) if sp.rows else ();z.rows+=[eg] if eg!=sp.rows[0] and repr(eg) not in {repr(r) for r in z.rows} else [];z.R0,z.R=len(sp.rows),len(z.rows)
  z.ex=sorted(sp.T);z.T=sp.T;z.full=(1<<len(z.ex))-1;z.abits=(1<<len(sp.params))-1;z.split_on=len(z.ex)>=3;W=sp.words
  lt,hs=learned_tiles(recs,W,sp);z.hsrc={**HELP,**hs};z.n_learned=len(lt);z.tiles=[t for t in sorted(library(sp)+sp.helper_tiles()+lt,key=lambda t:(-bool(t.hit),t.cost)) if only is None or t.n in only[0] or "p:"+t.n in t.dep];z.pm=sorted({m for t in z.tiles for m in t.p}|{0});z.smask=(1<<len(sp.strong))-1;z.cmask=sum(b for w,b in sp.sb.items() if w[0]=="#")
  z.pats={k:sorted([(n,max(1,c-(sp.aff(kw)>=AFT)-(kd in ("pair","adj") and "pair" in sp.cues)-sp.pb("@"+(n if kd=="single" else kd))*(PBIAS["pat"]=="1"))) for n,kd,c,kw in PATS if kd==k and (only is None or (n if kd=="single" else kd) in only[1])],key=lambda x:x[1]) for k in ("single","pair","adj","zip")};z.pw={(kd,n):sp.hm(kw) for n,kd,c,kw in PATS}
  z.fc=2 if W&{"fold","accumulate","running","cumulative","reduce","rolling"} or "acc" in sp.cues else 3;z.loo,z.nex,z.pe,z.hk,z.be,z.vfail={},len(z.ex),None,{},1e9,set();z.fm=[(z.fc+(sp.aff(f[4])<AFT),sp.hm(f[4])) for f in FOLDS if only is None or only[2]];sg={}
  [sg.__setitem__(m.group(0),max(sg.get(m.group(0),0),sp.aff(tl[5]))) for tl in TILE_DEFS for m in [re.search(r"(?<![\w.])[\w.]+\(|\.\w+\(",tl[2])] if m];z.sig=[re.compile(r"(?<![\w.])"+re.escape(k)) for k,a in sg.items() if a<AFT]
  taken=set(sp.params)|{h.name for h in sp.hnodes}|{sp.name};z.X,z.Y,z.I=(next(n for n in ns if n not in taken) for ns in (("x","e","v","w","z"),("y","u","t","o"),("i","j","k","m","p")));taken|={z.X,z.Y,z.I};z.X2,z.I2=(next(n for n in ns if n not in taken) for ns in (("e","v","w","z","q","r"),("j","k","m","p","h")));z.vo=re.compile(rf"\b{z.Y}\b.*\b{z.X}\b")
  z.outer=Ctx(z,"outer",z.R,z.R0)
  for k,p in enumerate(sp.params):z.outer.seed(1,tuple(r[k] for r in z.rows),p,1<<k)
  for code,v,c,d,key in sp.consts:z.outer.seed(c,(v,)*z.R,code,0,d,sp.sb.get(key,0))
 def tick(z):
  z.n+=1
  if not z.n&511 and time.perf_counter()>z.dl:raise Stop
 def rowmap(z,f,it,R0=None):
  out,R0=[],z.R0 if R0 is None else R0
  for r,x in enumerate(it):
   try:out.append(f(x))
   except Exception:
    if r<R0:return None
    out.append(ERR)
  return tuple(out)
 def raw_ok(z,t):
  if not z.sp.raw:return True
  tab=dict(zip(z.sp.keys,t.v));env={"__builtins__":SAFE_B,**MODS,z.sp.name:lambda *a,**k:tab[repr(z.sp.bind(a,k))],"_x0eq":_x0eq}
  try:return all(bool(timed(eval,c,env,t=0.5)) for c in z.sp.rawc)
  except Exception:return False
 def score(z,t):
  sp,pr,ty=z.sp,t.v[z.R0:z.RE],{type(z.T[r]) for r in z.ex};pv=[repr(x) for x in t.v if x is not ERR]
  if ty and any(x is not ERR and x is not None and type(x) not in ty and not (type(x) in (int,float) and ty<={int,float}) or "count" in sp.cues and x is not ERR and (type(x) is not int or x<0) for x in pr):return None
  return t.s+5*(len(sp.params)-bin(t.u&z.abits).count("1"))+COVW*bin(z.smask&~t.w&~z.cmask).count("1")+2*bin(z.cmask&~t.w).count("1")+(3 if len(set(pv))==1 and len(pv)-z.R0>=2 and not t.k&B else 0)+(3 if t.c in sp.params else 0)+PEN["intent"]*sum(not re.search(CREQ[k],t.c) for k in sp.cues)+(PEN["cidx"]*min(2,len(sp.cix.findall(t.c))) if sp.quant and sp.cix else 0)+PEN["fconst"]*sum(x[:2]=="k:" for x in t.d)+PEN["crash"]*sum(x is ERR for x in pr)+PEN["redund"]*_redund(t.c)+PEN["edge"]*(z.RE<z.R and t.v[z.RE] is ERR)+PEN["rel"]*min(3,sum(1 for rx in z.sig if rx.search(t.c)))+(sum(x[:2]=="b:" for x in t.d)-sum(x[:2]=="B:" for x in t.d) if PBIAS["rs"]=="1" else 0)+(0.5 if z.vo.search(t.c.split(" for ")[0]) and not re.search(rf"\b{z.X}\b.*\b{z.Y}\b",t.c.split(" for ")[0]) else 0)
 def on_outer(z,t,key=None,m=None,dup=False):
  m=sum(1<<q for q,r in enumerate(z.ex) if _eqv(t.v[r],z.T[r])) if m is None else m;q=(z.full^m).bit_length()-1;nr=z.nex>=2 and m and bin(m).count("1")==z.nex-1;(z.hk.__setitem__(key,m) if key and (m==z.full or nr) else None)
  if t.u&z.abits and m==z.full and (not dup or t.s<=z.be) and z.raw_ok(t):e=z.score(t);(z.hits.append((e,t)),setattr(z,"be",min(z.be,e))) if e is not None else None
  elif t.u&z.abits and nr and t.s<z.loo.get(q,1e9):e=z.score(t);(z.loo.__setitem__(q,min(e,z.loo.get(q,1e9))) if e is not None else None)
  if dup:return
  (z.outer.spq.append(t) if t.s<=4 and t.k&(X|D) else None)
  if z.split_on:
   (z.outer.mm.setdefault(m,t) if m else None)
   (z.outer.cm.setdefault(sum(1<<q for q,r in enumerate(z.ex) if t.v[r]),t) if t.k&B and t.s<=6 and t.u&z.abits else None)
 def on_elem(z,cx,t):
  (cx.spq.append(t) if cx.depth==1 and t.s<=2 and t.u&cx.lb and t.k&(X|D) else None)
  if cx.et is None or not t.u&cx.lb or t.s>4:return
  (cx.cm.setdefault(sum(1<<j for j,x in enumerate(t.v[:cx.R0]) if x),t) if t.k&B else None)
  if t.k&cx.etk&(N|S_|B|Q|D|O):m=sum(1<<j for j,(x,y) in enumerate(zip(t.v,cx.et)) if _eqv(x,y));(cx.mm.setdefault(m,t) if m else None)
 def partial(z,cx,t,a,c,u,d,w):
  v=tuple(_try(t.fn,*(x.v[j] for x in a)) for j in range(cx.R));m=sum(1<<q for q,r in enumerate(z.ex) if v[r] is not ERR and _eqv(v[r],z.T[r]))
  (cx.mm.setdefault(m,Tm(_rd(t.parts,[x.c for x in a]),v,0,c,u,d,w)) if m else None)
 def split(z,cx):
  if not cx.cm or not cx.mm:return
  full=z.full if not cx.depth else (1<<cx.R0)-1;ms=sorted(cx.mm.items(),key=lambda kv:kv[1].s)
  for tm,Ct in sorted(cx.cm.items(),key=lambda kv:kv[1].s):
   z.tick();nb=full^tm
   if tm==0 or nb==0:continue
   A=next((t for m,t in ms if m&tm==tm),None);Bt=next((t for m,t in ms if m&nb==nb),None)
   if A is None or Bt is None or (id(A),id(Bt),id(Ct)) in z.tried:continue
   z.tried.add((id(A),id(Bt),id(Ct)));z.tern(cx,A,Bt,Ct)
 def tern(z,cx,A,Bt,Ct):
  u,d,w,code=A.u|Bt.u|Ct.u,A.d|Bt.d|Ct.d,A.w|Bt.w|Ct.w,f"({A.c} if {Ct.c} else {Bt.c})";ev=tuple(a if c else b for a,b,c in zip(A.v,Bt.v,Ct.v))
  if not cx.depth:return z.push(A.s+Bt.s+Ct.s+2,ev,code,u,d,w)
  j=cx.mode=="join";v=z.rowmap((lambda ab:"".join(ev[ab[0]:ab[1]])) if j else (lambda ab:list(ev[ab[0]:ab[1]])),cx.spans) if u&cx.lb else None;v=tuple(ERR if r in cx.bad else x for r,x in enumerate(v)) if v and cx.bad else v
  (z.push(cx.xc+A.s+Bt.s+Ct.s+3+j,v,("''.join(" if j else "")+f"[{code} {cx.clause(u)}]"+(")" if j else ""),(u&~cx.lb)|cx.pu,d|cx.pd,w) if v else None)
 def push(z,c,v,code,u,d,w):(z.outer.seed(c,v,code,u,d,w) if c>z.outer.level else z.outer.add(v,c,u,d,w,code)) if c<=MAXC else None
 def spawn(z,host,t):
  vs,R0=t.v,host.R0;ok=lambda v:type(v) in (str,list,tuple,dict)
  if len(z.ctxs)>=z.maxctx or (host.depth and len(host.kids)>=3) or not all(ok(v) for v in vs[:R0]):return
  rows=[list(v) if ok(v) else [] for v in vs];ln=[len(r) for r in rows];isd=any(type(v) is dict for v in vs[:R0])
  (z.ectx(host,"single",(t,),[[(j,j) for j in range(n)] for n in ln],rows,rows) if sum(ln[:R0]) and sum(ln)<=ROWCAP else None)
  if t.s>2 or isd or host.depth:return
  pr=[[(a,b) for a in range(n) for b in range(a+1,n)] for n in ln];(z.ectx(host,"pair",(t,),pr,rows,rows) if sum(map(len,pr[:R0])) and sum(map(len,pr))<=PAIRCAP else None)
  aj=[[(j,j+1) for j in range(n-1)] for n in ln];(z.ectx(host,"adj",(t,),aj,rows,rows) if sum(map(len,aj[:R0])) and sum(ln)<=ROWCAP else None)
  for o in z.iters:
   ro=[list(v) if ok(v) else [] for v in o.v];lz=[min(len(a),len(b)) for a,b in zip(ro,rows)];(z.ectx(host,"zip",(o,t),[[(j,j) for j in range(n)] for n in lz],ro,rows) if lz[:R0]==ln[:R0]==[len(r) for r in ro[:R0]] and repr(ro)!=repr(rows) and sum(lz[:R0]) and sum(lz)<=ROWCAP else None)
  z.iters.append(t)
 def ectx(z,host,kind,par,slots,ra,rb):
  if len(z.ctxs)>=z.maxctx:return
  dp=host.depth+1;X_,Y_,I_=(z.X,z.Y,z.I) if dp==1 else (z.X2,None,z.I2);xb,yb,ib=(XB,YB,IB) if dp==1 else (X2B,0,I2B);R0=host.R0;xs=par[0].c;owner=[r for r,sl in enumerate(slots) for _ in sl];flat=[p for sl in slots for p in sl];sp,a0=[],0;E0=sum(len(sl) for sl in slots[:R0])
  for sl in slots:sp.append((a0,a0+len(sl)));a0+=len(sl)
  cl={"single":lambda u:f"for {I_}, {X_} in enumerate({xs})" if u&ib else f"for {X_} in {xs}","pair":lambda u:f"for {I_}, {X_} in enumerate({xs}) for {Y_} in {xs}[{I_} + 1:]","adj":lambda u:f"for {X_}, {Y_} in zip({xs}, {xs}[1:])","zip":lambda u:f"for {X_}, {Y_} in zip({xs}, {par[-1].c})"}[kind]
  xv=[ra[r][p[0]] for r,p in zip(owner,flat)];yv=[rb[r][p[1]] for r,p in zip(owner,flat)];et=mode=None
  if dp==1 and len(z.ex)==R0 and E0>=4:
   et,mode=([y for r in range(R0) for y in z.T[r]],"map") if all(type(z.T[r]) is list and len(z.T[r])==len(slots[r]) for r in range(R0)) else ([y for r in range(R0) for y in z.T[r]],"join") if kind=="single" and all(type(z.T[r]) is str and type(par[0].v[r]) is str and len(z.T[r])==len(slots[r]) for r in range(R0)) else (None,None)
  cx=Ctx(z,kind,len(flat),E0,sum(p.s for p in par),functools.reduce(op.or_,(p.u for p in par)),functools.reduce(op.or_,(p.d for p in par)),owner,sp,xv,cl,et,mode,xs,dp,xb|yb|ib,xb,xb|yb);z.ctxs.append(cx);host.kids.append(cx);cx.bad={r for p in par for r,x in enumerate(p.v) if type(x) not in (str,list,tuple,dict)}
  cx.seed(1,tuple(xv),X_,xb);(cx.seed(1,tuple(yv),Y_,yb) if kind!="single" else cx.seed(2,tuple(p[0] for p in flat),I_,ib))
  if kind=="single":
   vs=par[0].v
   if any(type(v) is dict for v in vs[:R0]):cx.seed(2,tuple(vs[r][x] for r,x in zip(owner,xv)),f"{xs}[{X_}]",par[0].u|xb,par[0].d)
   else:
    for c,code,f in ((2,f"{xs}[:{I_} + 1]",lambda v,j:v[:j+1]),(3,f"{xs}[:{I_}]",lambda v,j:v[:j]),(3,f"{xs}[{I_} + 1:]",lambda v,j:v[j+1:])):cx.seed(c,tuple(f(vs[r],p[0]) for r,p in zip(owner,flat)),code,par[0].u|ib,par[0].d)
  for bt in [t for t in host.lv.get(1,()) if t.u]+([t for t in host.lv.get(2,()) if t.u and t.k&(N|B|S_)][:40] if dp==1 else []):cx.seed(bt.s,tuple(bt.v[r] for r in owner),bt.c,bt.u,bt.d,bt.w)
  for code,v,c,d,key in z.sp.consts:cx.seed(c,(v,)*len(flat),code,0,d,z.sp.sb.get(key,0))
 def patterns(z,host,c):
  for cx in list(host.kids):
   for nm,pc in z.pats[cx.kind]:
    b=c-cx.xc-pc
    if b<1:continue
    cx.grow(b)
    for Et in list(cx.lv.get(b,())):(z.emit(host,cx,nm,Et,c) if (Et.u&cx.lb if cx.kind=="single" else Et.u&cx.xyb==cx.xyb) else None)
  (z.fold(c) if not host.depth else None)
 def emit(z,host,cx,nm,Et,c):
  z.tick();xv,ev,X_,I_,xs=cx.xv,Et.v,(z.X if cx.depth==1 else z.X2),(z.I if cx.depth==1 else z.I2),cx.xs;k=Et.k;cl=cx.clause(Et.u);lam=not Et.u&(cx.lb&~cx.xbit);rv=nm=="sortbyr";mx=max if nm=="maxby" else min
  rf,code=((lambda a,b:list(ev[a:b])),f"[{Et.c} {cl}]") if nm=="map" else ((lambda a,b:[xv[j] for j in range(a,b) if ev[j]]),f"[{X_} {cl} if {Et.c}]") if nm=="filter" and k&B else ((lambda a,b:[xv[j] for j in sorted(range(a,b),key=ev.__getitem__,reverse=rv)]),f"sorted({xs}, key=lambda {X_}: {Et.c}"+(", reverse=True)" if rv else ")")) if nm in ("sortby","sortbyr") and lam and k&(N|S_) else ((lambda a,b:xv[mx(range(a,b),key=ev.__getitem__)]),f"{nm[:3]}({xs}, key=lambda {X_}: {Et.c})") if nm in ("maxby","minby") and lam and k&(N|S_) else ((lambda a,b:next((j-a for j in range(a,b) if ev[j]),-1)),f"next(({I_} for {I_}, {X_} in enumerate({xs}) if {Et.c}), -1)") if nm=="firstidx" and k&B else ((lambda a,b:{xv[j]:ev[j] for j in range(a,b)}),"{"+f"{X_}: {Et.c} {cl}"+"}") if nm=="dictc" else ((lambda a,b:[y for j in range(a,b) for y in ev[j]]),f"[_y {cl} for _y in {Et.c}]") if nm=="flatmap" and k&X and not k&E and cx.depth==1 else (None,None)
  v=z.rowmap(lambda ab:rf(*ab),cx.spans,host.R0) if rf else None;v=tuple(ERR if r in cx.bad else x for r,x in enumerate(v)) if v and cx.bad else v
  (host.add(v,c,(Et.u&~cx.lb)|cx.pu,Et.d|cx.pd,Et.w|z.pw[(cx.kind,nm)],code) if v else None)
 def fold(z,c):
  for (body,fn,ic,iv,kw,sc),(k,hw) in zip(FOLDS,z.fm):
   for t in (z.outer.lv.get(c-k,()) if c>k else ()):
    if not all(type(v) in (list,tuple,str) for v in t.v[:z.R0]):continue
    v=z.rowmap((lambda x:list(itertools.accumulate(x,fn))) if sc else (lambda x:functools.reduce(fn,x,iv)) if ic else (lambda x:functools.reduce(fn,x)),t.v);(z.outer.add(v,c,t.u,t.d|{"itertools" if sc else "functools"},t.w|hw,f"list(itertools.accumulate({t.c}, lambda _a, _x: {body}))" if sc else f"functools.reduce(lambda _a, _x: {body}, {t.c}"+(f", {ic})" if ic else ")")) if v else None)
 def body(z,t):return "".join(f"    import {m}\n" for m in sorted(x for x in t.d if ":" not in x))+"".join(textwrap.indent(z.hsrc[x[2:]],"    ") for x in sorted(t.d) if x.startswith("h:"))+f"    return {t.c}\n"
 def verify(z,body):
  sp=z.sp
  try:
   env={k:v for k,v in _load(sp,body).items() if k!="__builtins__"};env["_x0eq"]=_x0eq
   return bool(sp.used) and all(bool(timed(safe_eval,ch,env,t=2.0)) for ch in sp.used)
  except Exception:return False
 def pick(z,c,n=1):
  out=[]
  for e,t in sorted([h for h in z.hits if h[0]<=c],key=lambda h:(h[0],-bin(h[1].w).count("1"),h[1].s))[:30]:
   if len(out)>=n:break
   b=z.body(t)
   if b in z.bods:continue
   z.bods.add(b);(out.append((e,t,b)) if z.verify(b) else z.vfail.update(x[2:] for x in t.d if x.startswith("h:t_")))
  z.pe=out[0][0] if out and z.pe is None else z.pe;z.hits=[h for h in z.hits if h[0]>c];return out
 def run(z,n=1):
  out=[]
  try:
   for c in range(1,MAXC+1):
    z.outer.grow(c);out+=z.pick(c,n-len(out)) if z.hits else [];z.dl,z.c0=(min(z.dl,time.perf_counter()+XS),c) if out and z.c0 is None else (z.dl,z.c0)
    if out and (len(out)>=n or c>=z.c0+XLV):break
  except Stop:pass
  return out+(z.pick(1<<30,n-len(out)) if z.hits and len(out)<n else [])
 def stats(z):return {"terms":sum(len(l) for c in [z.outer]+z.ctxs for l in c.lv.values()),"level":z.outer.level,"n_ctx":len(z.ctxs),"n_learned_tiles":z.n_learned,"n_tiles":len(z.tiles)}
 def edit(z,nb,dl):
  sp,out=z.sp,[]
  try:F0=ast.parse(nb["src"]).body[0];ps=[a.arg for a in F0.args.args];req=len(ps)-len(F0.args.defaults)
  except Exception:return out
  hm=[(h,c) for h,c in ((t.hit,_core(tfeats().get(t.n,()))) for t in z.tiles) if h and c];kc={repr(v) for _,v,*_ in sp.consts}
  for V,how in (_variants(F0,sp) if 1<=req<=len(sp.params) else ()):
   for pm in itertools.permutations(sp.params,req):
    if time.perf_counter()>dl or len(out)>=int(SKC["en"]):return out
    b,c,u=_render(V,dict(zip(ps,pm)),ps[req:],sp)
    if b in z.bods:continue
    z.bods.add(b)
    try:f=_load(sp,b)[sp.name];v=z.rowmap(lambda r:timed(f,*copy.deepcopy(r),t=0.05),z.rows)
    except Exception:continue
    if v is None or not all(_eqv(v[q],z.T[q]) for q in z.ex):continue
    k=functools.reduce(op.and_,(_kv(x) for x in v[:z.R0]));cf=_cf(c);t=Tm(c,v,k,_ecost(V),sum(1<<j for j,p in enumerate(sp.params) if p in u),frozenset("k:"+x for x in _kx(V) if x not in kc),functools.reduce(op.or_,(h for h,q in hm if q<=cf),0));e=z.score(t) if k else None
    (out.append((e,t,b,{"sk":2,"nb":[nb["i"]],"how":how+"|"+",".join(pm)})) if e is not None and z.verify(b) else None)
  return out
def _load(sp,body):
 try:return safe_exec(_strip_imports(sp.prompt+body),{**TYPING,**(XMODS if "def x5_" in body or re.search(r"\b(?:collections|string|re)\.",body) else {})})
 except Unsafe:return safe_exec(f"def {sp.name}({', '.join(sp.params)}):\n"+_strip_imports(body))
def _cn(v):t=type(v);return repr(int(v)) if t in (int,bool) or t is float and v==v and abs(v)<1e15 and v==int(v) else repr(round(v,6)) if t is float else repr(v) if t in (str,type(None)) else "["+",".join(map(_cn,v))+"]" if t is list else "("+",".join(map(_cn,v))+")" if t is tuple else "{"+",".join(sorted(map(_cn,v)))+"}" if t in (set,frozenset) else "{"+",".join(sorted(_cn(a)+":"+_cn(b) for a,b in v.items()))+"}" if t is dict else "<"+t.__name__+">"
def _bcall(f,r):
 try:o=timed(f,*copy.deepcopy(r),t=0.05);return _cn(o),o
 except Timeout:return -2,ERR
 except Exception:return -1,ERR
def _pok(fn,r,o,f):
 try:return o is ERR or bool(fn(r,o,f))
 except Exception:return True
PROPS={"srt":(r"\bascending\b|\bincreasing order\b|\bsorted (list|order|array)\b|\bsort(s|ed)? (the |a |given )?(list|array|elements|numbers)\b(?!.*\b(by|key|according|descending|reverse|length|based)\b)",lambda r,o,f:type(o) is not list or all(a<=b for a,b in zip(o,o[1:]))),"srtr":(r"\bdescending\b|\bdecreasing order\b",lambda r,o,f:type(o) is not list or all(a>=b for a,b in zip(o,o[1:]))),"uniq":(r"\b(unique|distinct)\b|\bremove (all )?(the )?duplicates?\b|\bwithout duplicates?\b",lambda r,o,f:type(o) not in (list,tuple) or len({_cn(e) for e in o})==len(o)),"sub":(r"\b(remove|filter|filters|keep only|extract|delete)\b",lambda r,o,f:type(o) not in (list,str,tuple) or not any(type(a) is type(o) for a in r) or {_cn(e) for e in o}<={_cn(e) for a in r if type(a) is type(o) for e in a}),"len":(r"\b(each|every) (element|number|item|value|integer|word|string)\b",lambda r,o,f:type(o) is not list or not any(type(a) in (list,str) for a in r) or len(o)==len(next(a for a in r if type(a) in (list,str)))),"idem":(r"\b(sort|sorted|unique|distinct|duplicates?|strip|trim|normali[sz]e|flatten)\b",lambda r,o,f:len(r)!=1 or type(o) is not type(r[0]) or _bcall(f,(o,))[0]==_cn(o)),"nneg":(r"\bcount\b|\bhow many\b|\bnumber of\b",lambda r,o,f:type(o) is not int or o>=0)}
def _neg(v):t=type(v);return v<0 if t in (int,float) else any(map(_neg,v)) if t in (list,tuple,set,frozenset) else any(map(_neg,v.values())) if t is dict else False
def _ood(rows,r):return any((_neg(a) and not any(_neg(x[p]) for x in rows)) or (type(a) in (list,tuple,str,dict,set) and not a and all(type(x[p]) is type(a) and x[p] for x in rows)) for p,a in enumerate(r))
def behave(sp,bodies,K=None,budget=None):
 K,dl,t0=int(os.environ.get("TS_BEHK","40")) if K is None else K,time.perf_counter()+(float(os.environ.get("TS_BEHS","3")) if budget is None else budget),time.perf_counter();ins,sn=[],set()
 for r,k in [(r,"n") for r in _probes(sp.rows,K//2)]+[(r,"m") for r in _probes(sp.rows,K,1)]+[(r,"e") for r in _edges(sp.rows)]:(ins.append((r,k.upper() if _ood(sp.rows,r) else k)),sn.add(repr(r))) if repr(r) not in sn else None
 fs=[]
 for b in bodies:
  try:fs.append(_load(sp,b).get(sp.name))
  except Exception:fs.append(None)
 pr=[k for k,(rx,_) in PROPS.items() if re.search(rx,sp.desc.lower())];cols,kd,pv=[],"",[{k:0 for k in pr} for _ in fs]
 for r,k in ins:
  if time.perf_counter()>dl:break
  col=[_bcall(f,r) if f else (-1,ERR) for f in fs];m={};cols.append([x if type(x) is int else m.setdefault(x,len(m)) for x,_ in col]);kd+=k;[d.__setitem__(q,d[q]+(not _pok(PROPS[q][1],r,o,f))) for d,(_,o),f in zip(pv,col,fs) if f for q in pr]
 return {"beh":[list(c) for c in zip(*cols)] if cols else [[] for _ in fs],"kind":kd,"inputs":[repr(r)[:80] for r,_ in ins[:len(cols)]],"props":pr,"prop":pv,"time":time.perf_counter()-t0}
def _gen(src,params):
 tr,mp=ast.parse(src),{p:f"a{k}" for k,p in enumerate(params)}
 for n in ast.walk(tr):(setattr(n,"id",mp[n.id]) if isinstance(n,ast.Name) and n.id in mp else setattr(n,"arg",mp[n.arg]) if isinstance(n,ast.arg) and n.arg in mp else None)
 return ast.unparse(tr)+"\n"
def _now():return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def qpath(store):p=Path(store);return p.with_name(p.stem+"_quarantine.jsonl")
def save_store(path,recs):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text("".join(json.dumps(r)+"\n" for r in recs),encoding="utf-8")
def learn_store(path,sp,t,body,origin="",status="quarantine",reason="",loo=None):
 src=_gen(f"def _t({', '.join(sp.params)}):\n{body}",sp.params);name="t_"+format(zlib.crc32(src.encode()),"08x");src=src.replace("def _t(",f"def {name}(",1);Path(path).parent.mkdir(parents=True,exist_ok=True);Q=load_store(path)
 rec={"name":name,"params":[f"a{k}" for k in range(len(sp.params))],"kinds":[functools.reduce(op.and_,(_kv(r[k]) for r in sp.rows)) for k in range(len(sp.params))],"ret":t.k,"src":src,"words":sorted(sp.words),"from":origin or sp.name,"examples":list(sp.used),"date":_now(),"expr":_gen(t.c,sp.params).strip(),"cost":t.s,"status":status,"reason":reason,"loo":loo,"strikes":0}
 (open(path,"a",encoding="utf-8").write(json.dumps(rec)+"\n") if 1<=len(sp.params)<=3 and name not in {r["name"] for r in Q} and rec["expr"] not in {r["expr"] for r in Q} else None)
 return rec
def probation(sp,budget,store,origin="",ref=None):
 try:
  qp=qpath(store);Q=load_store(qp);me=origin or sp.name;cand=[r for r in Q if r.get("status")=="quarantine" and r.get("from")!=me];nt=len(learned_tiles(cand,sp.words,sp)[0]) if cand else 0
  if not nt:return {"pg_tried":0}
  Sr=Search(sp,budget,load_store(store)+cand);r=(Sr.run() or [None])[0];byn={q["name"]:q for q in cand};lf=sum(v<Sr.pe for v in Sr.loo.values()) if r and Sr.nex>=2 else 0;hu={x[2:] for x in r[1].d if x.startswith("h:t_")}&set(byn) if r else set();dis=hu if ref is not None and r and repr(r[1].v)!=repr(ref) else set();used=hu-dis if not lf else set();bad=((Sr.vfail&set(byn))|dis)-used
  for q in Q:(q.update(status="promoted",promoted_by=me,date_promoted=_now()) if q["name"] in used else q.update(strikes=q.get("strikes",0)+1,**({"status":"demoted","reason":"probation_disagrees_with_trusted" if q["name"] in dis else "probation_verify_fail","demoted_by":me} if q.get("strikes",0)>=1 else {})) if q["name"] in bad else None)
  (save_store(qp,Q),[open(store,"a",encoding="utf-8").write(json.dumps(q)+"\n") for q in Q if q["name"] in used]) if used or bad else None
  return {"pg_tried":nt,"pg_promoted":sorted(used),"pg_demoted":sorted(n for n in bad if byn[n]["status"]=="demoted"),"pg_expr":r[1].c if r else ""}
 except Exception as e:return {"pg_error":f"{type(e).__name__}: {e}"[:200]}
SWAP=[{"add","sub","mul"},{"lt","le","eq","ne"},{"min","max"},{"min2","max2"},{"srt","srtr"},{"even","odd"},{"first","last"},{"maxl","minl"},{"imax","imin"},{"any","all"},{"lower","upper"},{"sw","ew"},{"isin","nin"},{"take","drop","takel"},{"rng","rng1","rng0i"},{"rng2","rng2i"},{"div","fdiv"},{"sand","sor","ssub"},{"dmaxk","dmink"},{"issrt","issrtr"}]
PK={"map":{"p:iter"},"filter":{"p:filter"},"sortby":{"c:sorted","k:key"},"sortbyr":{"c:sorted","k:reverse"},"maxby":{"c:max","k:key"},"minby":{"c:min","k:key"},"firstidx":{"c:enumerate"},"dictc":{"p:dictc"},"flatmap":{"p:nest"},"pair":{"p:pair"},"adj":{"p:adj"},"zip":{"c:zip"}}
SWO={ast.Lt:(ast.LtE,ast.Gt,ast.GtE),ast.LtE:(ast.Lt,ast.GtE),ast.Gt:(ast.GtE,ast.Lt,ast.LtE),ast.GtE:(ast.Gt,ast.LtE),ast.Eq:(ast.NotEq,),ast.NotEq:(ast.Eq,),ast.Add:(ast.Sub,),ast.Sub:(ast.Add,),ast.Mult:(ast.FloorDiv,),ast.Div:(ast.FloorDiv,),ast.FloorDiv:(ast.Div,),ast.And:(ast.Or,),ast.Or:(ast.And,)}
SWN={"min":"max","max":"min","any":"all","all":"any"}
SKMOD=("math","itertools","functools","collections","re","string")
def _core(fs):return frozenset(f for f in fs if f[0] in "cmops")
def _cf(c):
 try:return _core(feats(c))
 except Exception:return EMPTY
def _nodes(c):
 try:return sum(1 for _ in ast.walk(ast.parse(c,mode="eval")))
 except SyntaxError:return sum(1 for _ in ast.walk(ast.parse(c)))
def sk_only(fs):fs=set(fs);c=_core(fs);tf=tfeats();t={n for n,*_ in TILE_DEFS if _core(tf[n]) and _core(tf[n])<=c};return frozenset(t|{x for g in SWAP if g&t for x in g}|({n for n,*_,fl in TILE_DEFS if not _core(tf[n]) and "h" not in fl} if t else set())),frozenset(p for p,k in PK.items() if k<=fs),bool({"p:acc","c:reduce","c:accumulate"}&fs)
def _holes(F):return [n for n in ast.walk(F) if isinstance(n,ast.Constant) and type(n.value) in (int,float,str) and len(str(n.value))<=12][:4],[(n,j) for n in ast.walk(F) for j,o in enumerate(n.ops if isinstance(n,ast.Compare) else [n.op] if isinstance(n,(ast.BinOp,ast.BoolOp)) else []) if type(o) in SWO][:6]+[(n,-1) for n in ast.walk(F) if isinstance(n,ast.Name) and n.id in SWN][:3]
def _variants(F,sp):
 yield F,"orig"
 K,O=_holes(F);pool=[(code,v) for code,v,c,d,key in sp.consts if c<=2 and type(v) is not bool and v is not None]
 for k,n in enumerate(K):
  for code,v in pool:
   if v!=n.value and (type(v) is str)==(type(n.value) is str):G=copy.deepcopy(F);_holes(G)[0][k].value=v;yield G,f"c{k}:{n.value!r}->{code}"
 for k,(n,j) in enumerate(O):
  for a in (SWO[type(n.ops[j] if isinstance(n,ast.Compare) else n.op)] if j>=0 else [SWN[n.id]]):G=copy.deepcopy(F);m,_=_holes(G)[1][k];(m.ops.__setitem__(j,a()) if isinstance(m,ast.Compare) else setattr(m,"op",a()) if j>=0 else setattr(m,"id",a));yield G,f"o{k}->{a if isinstance(a,str) else a.__name__}"
def _render(V,mp,opt,sp):
 G=copy.deepcopy(V);nm=G.name;[setattr(n,"id",mp[n.id] if n.id in mp else sp.name) for n in ast.walk(G) if isinstance(n,ast.Name) and (n.id in mp or n.id==nm)];dv=dict(zip([a.arg for a in G.args.args][len(G.args.args)-len(G.args.defaults):],G.args.defaults));st=[ast.Assign(targets=[ast.Name(id=p,ctx=ast.Store())],value=dv[p]) for p in opt]+G.body;src=ast.unparse(ast.fix_missing_locations(ast.Module(body=st,type_ignores=[])));ns={n.id for n in ast.walk(G) if isinstance(n,ast.Name)}
 return "".join(f"    import {m}\n" for m in SKMOD if m in ns)+textwrap.indent(src,"    ")+"\n",(ast.unparse(st[0].value) if len(st)==1 and isinstance(st[0],ast.Return) and st[0].value is not None else src),ns
def _ecost(V):return max(1,sum(1 for s in V.body for n in ast.walk(s) if not isinstance(n,(ast.expr_context,ast.operator,ast.cmpop,ast.boolop,ast.unaryop)))//2)
def _kx(V):return [repr(n.value) for n in ast.walk(V) if isinstance(n,ast.Constant) and type(n.value) in (int,float,str) and n.value not in (0,1,"")]
def sketch(z,nbs):
 sp,out,g=z.sp,[],{}
 for x in nbs.get("nb",[])[:int(SKC["k"])]:o=sk_only(x["f"]);(g.setdefault(o,[]).append(x["i"]) if o[0] or o[1] else None)
 ks=list(g)[:int(SKC["s"])];per=float(SKC["b"])/max(1,len(ks))
 for o in ks:
  try:out+=[(e,t,b,{"sk":1,"nb":g[o]}) for e,t,b in Search(sp,per,(),only=o).run(int(SKC["n"]))]
  except Exception:pass
 dl=time.perf_counter()+float(SKC["eb"])
 for x in nbs.get("pe",[]) if SKC["e"]=="1" else ():out+=z.edit(x,dl)
 return out
def _skmerge(rs,ks):
 eb={b:[] for _,_,b in rs};sd={}
 for e,t,b,m in ks:(eb[b].extend(m["nb"]) if b in eb else sd.setdefault(b,(e,t,b,{**m,"nb":[]}))[3]["nb"].extend(m["nb"]))
 return [(e,t,b,{"sk":0,"nb":eb[b]}) for e,t,b in rs]+sorted(sd.values(),key=lambda x:x[0])
def support(cs,nb,k):
 nb=nb[:int(k["sk"])];F=[(set(x["f"]),_core(x["f"])) for x in nb];w=np.array([1.0 if k["sw"]=="1" else x["s"] for x in nb],float);w=w/max(1e-9,w.max()) if len(w) else w;W=w.sum();C=[set(c.get("cf") or ()) for c in cs]
 m=np.array([[len(a&g)/max(1,len(a|g)) if k["sm"]=="jac" else float(a<=f) for f,g in F] for a in C],float).reshape(len(cs),len(F));M=(m*w[None,:]).sum(1);br=np.array([max(1e-6,c.get("br",1.0)) for c in cs])
 return np.log((M+br)/((W+1)*br)) if k["sl"]=="1" else M/max(W,1e-9)
def solve(prompt,checks,budget=None,store=None,learn=False,origin="",topn=None,nbs=None):
 t0=time.perf_counter();budget=float(os.environ.get("TS_BUDGET","10")) if budget is None else budget;base={"found":False,"body":"","n_checks":len(checks),"time":0.0,"size":0,"cost":0,"expr":""}
 try:sp=Spec(prompt,checks)
 except Exception as e:return {**base,"reason":f"spec_error {type(e).__name__}","time":time.perf_counter()-t0}
 base.update(n_ex=len(sp.T),n_raw=len(sp.raw))
 if not sp.used:return {**base,"reason":"no_examples","time":time.perf_counter()-t0}
 try:Sr=Search(sp,budget,load_store(store) if store else ());rs=Sr.run(TOPN if topn is None else topn);t1=time.perf_counter()-t0;rs=_skmerge(rs,sketch(Sr,nbs) if nbs else []);r=rs[0] if rs else None;st=Sr.stats();pe,nex,cm=Sr.pe if Sr.pe is not None or not r else r[0],Sr.nex,Sr.cmask;lo=lambda e:sum(v<e for v in Sr.loo.values()) if nex>=2 else None;lf=lo(pe) if r else None;V=set(vocab());cs=[{"body":b,"expr":t.c,"cost":t.s,"score":e,"cov":bin(t.w).count("1"),"loo_fail":lo(e),"pri":sp.pri(t.c),"sk":m["sk"],"nb":list(dict.fromkeys(m["nb"])),"cf":sorted(_cf(t.c)&V),**({"how":m["how"]} if "how" in m else {})} for e,t,b,m in rs];Sr=None;gc.collect();tk=time.perf_counter()-t0-t1;bh=behave(sp,[x[2] for x in rs]) if len(rs)>1 and os.environ.get("TS_BEH","1")=="1" else None;[c.update(beh=v,prop=q) for c,v,q in zip(cs,bh["beh"],bh["prop"])] if bh else None;pg=probation(sp,budget,store,origin,r[1].v if r else None) if learn and store else {}
 except Exception as e:return {**base,"reason":f"search_error {type(e).__name__}: {e}"[:200],"time":time.perf_counter()-t0}
 tm={"time":t1,"time_total":time.perf_counter()-t0,**({"sk_time":tk,"sk_nb":nbs.get("nb",[]),"sk_pe":[x["i"] for x in nbs.get("pe",[])],"sk_guard":nbs.get("guard",{}),"sk_n":sum(bool(c["sk"]) for c in cs)} if nbs else {})}
 if not r:return {**base,**st,**pg,"reason":"no_program",**tm}
 _,t,body,_=r;gate="helper" if any(x.startswith("p:") for x in t.d) else "reuse" if any(x.startswith("h:t_") for x in t.d) else "n_ex<2" if nex<2 else "loo" if lf else "const" if any(x[:2]=="k:" for x in t.d) or t.w&cm else ""
 (learn_store(qpath(store),sp,t,body,origin,"demoted" if gate else "quarantine",gate,{"n":nex,"fail":lf}) if learn and store and gate not in ("helper","reuse") else None)
 return {**base,**st,**pg,"found":True,"reason":"found","body":body,"expr":t.c,"cost":t.s,"score":pe,"loo_n":nex,"loo_fail":lf,"learn_gate":gate or "ok","size":_nodes(t.c),"used_learned":any(x.startswith("h:t_") for x in t.d),"cands":cs,**({"beh_kind":bh["kind"],"beh_in":bh["inputs"],"beh_props":bh["props"],"beh_time":bh["time"]} if bh else {}),**tm}
def _clean(b):
 try:return textwrap.indent(ast.unparse(ast.parse(textwrap.dedent(b))),"    ")+"\n"
 except Exception:return b
def natural(prompt,cs,path=None):
 bs=[c["body"].encode() for c in cs];cb=[_clean(c["body"]).encode() for c in cs];n=len(bs);m=list(re.finditer(r"(?m)^def [^\n]*:[ \t]*\n",prompt));P,S=prompt.encode(),prompt[:m[-1].end()].encode() if m else b""
 lp=ray_field.engine(path or os.environ.get("TS_NAT_PACK",NATP)).score([P]*n+[b""]*n+[S]*n+[P]*n,bs+bs+bs+cb) if n else []
 [c.update(nat_lp=float(a),nat_lp0=float(b),nat_lps=float(q),nat_lpc=float(w),nat_n=len(x),nat_nc=len(y)) for c,a,b,q,w,x,y in zip(cs,lp[:n],lp[n:2*n],lp[2*n:3*n],lp[3*n:],bs,cb)];return cs
def _nf(c,m):return -c["score"] if m=="score" else 0.0 if m=="off" or "nat_lp" not in c else c["nat_lp"]/max(1,c["nat_n"]) if m=="byte" else c["nat_lpc"]/max(1,c["nat_nc"]) if m=="cbyte" else c["nat_lp"]-c["nat_lps"] if m=="pmis" else (c["nat_lp"]-c["nat_lp0"])/(max(1,c["nat_n"]) if m=="pmib" else 1)
def consensus(cs,k,kind=""):
 n=min(len(cs),int(k["cn"]));B=[c.get("beh") or [] for c in cs[:n]];L=min(map(len,B),default=0);ok=np.array([not any((c.get("prop") or {}).values()) for c in cs[:n]]) if k["cp"]=="1" else np.ones(n,bool);ok=ok if ok.any() else np.ones(n,bool)
 A=np.array([b[:L] for b in B]).reshape(n,L);A=A[:,np.array([k["ci"]=="a" or kind[j:j+1] in k["ci"] for j in range(L)],bool)&((A<0).sum(0)<float(k["cz"])*n)] if L else A
 if k["cons"]=="off" or n<2 or not A.shape[1]:j=int(np.argmax(ok));return j,1.0,[i==j for i in range(n)]
 E=(A[:,None,:]==A[None,:,:])|(((A[:,None,:]<0)|(A[None,:,:]<0))&(k["cx"]=="w"));G=E.all(-1) if k["cm"]=="x" else E.mean(-1) if k["cm"]=="a" else E.mean(-1)>=float(k["cm"]);sc=np.array([c["score"] for c in cs[:n]],float)
 w=(np.ones(n) if k["cw"]=="1" else np.exp(-(sc-sc[0])/float(k["ct"])) if k["cw"]=="s" else 1/np.maximum(1,np.array([c["cost"] for c in cs[:n]],float)) if k["cw"]=="c" else 1/(1+np.arange(n)))*ok;S=np.round((G*w[None,:]).sum(1)*ok,9);j=int(np.argmax(S))
 return j,float(S[j]/max(1e-12,w.sum())),(G[j]>=1-1e-9).tolist()
def rerank(r,prompt=None,cfg=None,checks=None):
 k={**RANK,**(cfg or {})};cs=r.get("cands") or [];(natural(prompt,cs) if prompt is not None and cs and "nat_lp" not in cs[0] else None)
 tb=float(k["tb"]);pr=np.array([c.get("pri",0.0) for c in cs]);pr=(pr-pr.mean())/(pr.std()+1e-9) if k["pz"]=="1" and len(cs)>1 else pr;su=support(cs,r.get("sk_nb") or [],k) if cs and float(k["sb"]) else np.zeros(len(cs));kk=[bool(c.get("sk")) for c in cs];ne=all(kk);el=[not q or k["sx"]=="1" or ne for q in kk];ks=k["sp"]=="1" and any(a and b for a,b in zip(el,kk));el=[a and (b or not ks) for a,b in zip(el,kk)];j=min(range(len(cs)),key=lambda i:(not el[i],cs[i]["cost"]-float(k["sb"])*su[i]+float(k["pw"])*(cs[i]["score"]-cs[i]["cost"])+float(k["lw"])*(cs[i]["loo_fail"] or 0)-float(k["w"])*_nf(cs[i],k["nat"])-float(k["pv"])*pr[i]*(cs[i]["score"]<=cs[0]["score"]+float(k["pm"])),-tb*cs[i].get("cov",0),-(k["so"]=="1")*kk[i],tb*cs[i]["cost"],-tb*_nf(cs[i],k["nat"]),i)) if cs else 0;c=cs[j] if cs else {}
 j,sh,_=consensus(cs,k,r.get("beh_kind","")) if cs and (k["cons"]!="off" or k["cp"]=="1") else (j,1.0,None);c=cs[j] if cs else {}
 x9,x10=j,(__import__("amni.compute.code_rank",fromlist=["x"]).rank(prompt,checks,cs,j,k["x10"]) if cs and prompt is not None and checks is not None and k["x10"]!="off" else None);j=x10["pick"] if x10 else j
 ax=bool(cs) and (k["ap"]=="1" or (c["loo_fail"]>0 if c["loo_fail"] is not None else k["anl"]=="1")) and _nf(c,k["af"])<float(k["at"]);ac=bool(cs) and sh<float(k["ca"]);ab={"x":ax,"c":ac,"o":ax or ac,"a":ax and ac}[k["cab"]]
 return {**r,"cshare":sh,**({"body":cs[j].get("body",""),"expr":cs[j]["expr"],"cost":cs[j]["cost"],"score":cs[j]["score"],"loo_fail":cs[j]["loo_fail"],"size":_nodes(cs[j]["expr"])} if cs else {}),"pick":j,"x9_pick":x9,**({"x10":x10} if x10 else {}),"abstain":ab,"nat":_nf(c,k["nat"]) if cs else None,"rank":k}
def _limit():
 vs=int(open("/proc/self/statm").read().split()[0])*os.sysconf("SC_PAGE_SIZE");resource.setrlimit(resource.RLIMIT_AS,(vs+int(float(os.environ.get("TS_MEM_GB","1.5"))*2**30),resource.RLIM_INFINITY))
def pool(jobs=1):return ProcessPoolExecutor(max(1,jobs),mp_context=get_context("spawn"),initializer=_limit)
def _solve_item(item,**kw):return solve(item[0],item[1],**{**kw,**(item[2] if len(item)>2 else {})})
def solve_many(ex,items,**kw):
 try:return list(ex.map(functools.partial(_solve_item,**kw),items))
 except BrokenProcessPool:
  if len(items)==1:return [{"found":False,"body":"","n_checks":len(items[0][1]),"time":0.0,"size":0,"cost":0,"expr":"","reason":"worker_crash"}]
  h=len(items)//2;n=max(1,getattr(ex,"_max_workers",1));return [r for part in (items[:h],items[h:]) for r in (lambda q:(lambda o:(q.shutdown(cancel_futures=True),o)[1])(solve_many(q,part,**kw)))(pool(n))]
