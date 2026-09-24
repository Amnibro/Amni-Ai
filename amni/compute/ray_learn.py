import os,re,json,math,collections,time,fcntl,hashlib,threading,urllib.error,urllib.parse,urllib.request
from concurrent.futures import ThreadPoolExecutor
from amni.compute.ray_qa import words,hard,STOP
try:from ddgs import DDGS
except Exception:DDGS=None
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE=os.environ.get("AMNI_RAY_LEARNED",os.path.join(_H0,"data","learned","qa_learned.jsonl"));UA={"User-Agent":"AmniAdam/1.0 (https://amni-scient.com/; learn-on-miss bot)"}
CACHE=os.environ.get("AMNI_WIKI_CACHE",os.path.join(_H0,"data","learned","wiki_cache"));API="https://en.wikipedia.org/w/api.php?";_LK=collections.defaultdict(threading.Lock)
PERSONAL=re.compile(r"(?i)\b(my|our)\s+(sister|brother|mom|mother|dad|father|wife|husband|partner|spouse|son|daughter|kids?|children|friend|boss|coworker|neighbou?r|roommate|aunt|uncle|cousin|niece|nephew|grand\w+|family|dog|cat|pet|name|age|birthday|job|salary|car|address|phone|email|password|account|bank|doctor|diagnosis|health|house|home|school|teacher)\b|\b(i\s+(am|was|live|work|have|had|drive|study|studied)|i'm|about\s+me|do\s+i|did\s+i|am\s+i|should\s+i)\b")
def _pace(h:str):
 os.makedirs(CACHE,exist_ok=True)
 with _LK[h],open(os.path.join(CACHE,".pace_"+h),"a+") as f:fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);t=f.read().strip();time.sleep(max(0.0,float(t or 0)+float(os.environ.get("AMNI_WIKI_GAP","1.05"))-time.time()));f.seek(0);f.truncate();f.write(repr(time.time()));f.flush();fcntl.flock(f,fcntl.LOCK_UN)
def _cached(key:str,fn):
 f=os.path.join(CACHE,hashlib.sha1(key.encode()).hexdigest()+".json")
 if os.path.exists(f):return json.load(open(f))
 x=fn();os.makedirs(CACHE,exist_ok=True);open(f+".tmp","w").write(json.dumps(x));os.replace(f+".tmp",f);return x
def _fetch(url:str,timeout:float,tries:int)->dict:
 for i in range(tries):
  _pace("wiki")
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=timeout) as r:return json.loads(r.read().decode("utf-8","replace"))
  except urllib.error.HTTPError as e:
   if e.code!=429 or i+1>=tries:raise
   time.sleep(min(60.0,float(e.headers.get("Retry-After") or 2**(i+1))))
def _get(url:str,timeout:float=8.0,tries:int=4)->dict:return _cached(url,lambda:_fetch(url,timeout,tries))
def _ddg(q:str,k:int,tries:int=3)->list:
 for i in range(tries):
  _pace("ddg")
  try:
   with DDGS() as d:return [{"u":r.get("href") or "","b":r.get("body") or "","t":r.get("title") or ""} for r in d.text(q,max_results=k,backend=os.environ.get("AMNI_DDG_BACKEND","auto"))]
  except Exception:
   if i+1>=tries:raise
   time.sleep(3.0*(i+1))
def ddg(q:str,k:int=10)->list:
 try:return _cached("ddg:%d:%s"%(k,q),lambda:_ddg(q,k)) if DDGS else []
 except Exception:return []
def _scrub(q:str)->str:
 try:
  from amni.serve.pii_egress import scrub
  return scrub(q,source="ray_learn") or q
 except Exception:return q
def search(q:str,k:int=5)->list:return [p["title"] for p in (_get(API+"action=query&list=search&srlimit=%d&srprop=&srsearch=%s&format=json"%(k,urllib.parse.quote(q))).get("query") or {}).get("search",[])]
def page(t:str,cap:int=20000)->tuple:
 p=list((_get(API+"action=query&prop=extracts|description&explaintext=1&redirects=1&titles=%s&format=json"%urllib.parse.quote(t)).get("query") or {}).get("pages",{}).values()) or [{}]
 return p[0].get("title",t),(p[0].get("extract") or "")[:cap],p[0].get("description") or "","https://en.wikipedia.org/wiki/"+urllib.parse.quote(p[0].get("title",t).replace(" ","_"))
def descs(ts:list)->dict:
 x=(_get(API+"action=query&prop=description&redirects=1&titles=%s&format=json"%urllib.parse.quote("|".join(sorted(set(ts))[:50]))) if ts else {}).get("query") or {};m={r["from"]:r["to"] for r in x.get("normalized",[])+x.get("redirects",[])}
 d={p["title"]:(p.get("description") or ("" if "missing" in p else "?")) for p in x.get("pages",{}).values()};return {t:d.get(m.get(m.get(t,t),m.get(t,t)),None) for t in ts}
_WC={}
def _wc()->dict:
 if not _WC:
  for l in open(os.path.join(_H0,"data","big2","wordcounts.txt"),"rb"):
   a=l.rsplit(None,1);_WC[a[0].decode("utf-8","replace")]=int(a[1]) if len(a)==2 and a[1].isdigit() else 0
  _WC[""]=sum(_WC.values())
 return _WC
def idf(w:bytes)->float:
 c=_wc();t=w.decode("utf-8","replace");return min(12.0,math.log((c[""]+1)/(c.get(t,0)+c.get(t.capitalize(),0)+1)))
MON="January|February|March|April|May|June|July|August|September|October|November|December"
DATE=re.compile(r"\b(?:(?:\d{1,2} (?:%s)|(?:%s) \d{1,2}),? )?(?:(?:early |mid-?|late )?(?:%s) )?((?:1[0-9]|20)\d\d)(?:s\b)?(?!\d)"%(MON,MON,MON))
NUMW="one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|dozen"
NUM=re.compile(r"(?i)(?:US)?\$?\b(?:\d[\d,]*(?:\.\d+)?|(?:%s)(?:[- ](?:%s))?)(?:\s?(?:%%|percent|million|billion|thousand|trillion))?\b"%(NUMW,NUMW))
_U="A-ZÀ-ÖØ-Þ";_L="a-zß-öø-ÿĀ-ž";_TK=r"(?:[%s]{2,6}(?![\w])|[%s][%s]*(?:[%s][%s]+)*(?:['’-][%s]?[%s]+)*(?![\w'’-])|[%s]\.)"%(_U,_U,_L,_U,_L,_U,_L,_U);NAME=re.compile(r"(?<![\w'’-])%s(?:\s+(?:(?:de|van|von|da|del|der|la|le|du|di|bin|al|of|the)\s+)?%s)*"%(_TK,_TK))
STOPC=set(("American British English Canadian Australian Irish Scottish Welsh French German Italian Spanish Norwegian Swedish Danish Dutch Japanese Chinese Korean Indian Mexican Brazilian Russian Belgian Polish Swiss Austrian Greek Jamaican Nigerian DJ MC Singer Actor Actress Director Producer Rapper Songwriter I A An The He She It They We You This That These Those There Here His Her Its Their Our My Your In On At By For From After Before During When While However Although Since Following According With As Both Also Then But And Or If So Some Many Most Other Such Each Every All No Not One Two Three First Second Last New Early Late Jan Feb Mar Apr May Jun Jul Aug Sep Sept Oct Nov Dec Monday Tuesday Wednesday Thursday Friday Saturday Sunday "+MON.replace("|"," ")).split())
JUNK=re.compile(r"%5B|%3A|Wikipedia:|\{\{|\}\}|displaystyle|\\|https?:|Template:|&nbsp;")
TEMPORAL=re.compile(r"(?i)\b(last|latest|current|currently|now|new|newest|next|recent|recently|today|this year|upcoming|so far|to date|as of|still|will|going to)\b|\bhow many\b.*\b(does|do|has|have)\b")
PERSONY=re.compile(r"(?i)\b(actor|actress|singer|musician|songwriter|rapper|band|group|duo|politician|writer|author|novelist|poet|playwright|screenwriter|composer|director|producer|player|footballer|cricketer|athlete|businessman|businesswoman|executive|engineer|scientist|physicist|chemist|mathematician|philosopher|painter|artist|sculptor|architect|comedian|presenter|host|journalist|king|queen|prince|princess|emperor|president|monarch|saint|bishop|pope|lawyer|judge|general|officer|soldier|explorer|inventor|designer|dancer|model|wrestler|boxer|coach|manager|character|born|died|\d{4}\s*[–-]\s*\d{4}|team|club|people|person|family|orchestra|choir|entrepreneur|economist|historian|priest|activist|diplomat|lyricist|guitarist|drummer|pianist|vocalist|voice)\b")
PLACEY=re.compile(r"(?i)\b(city|town|village|country|state|province|region|county|capital|island|islands|river|lake|sea|ocean|mountain|range|area|district|borough|municipality|neighbou?rhood|continent|peninsula|desert|territory|place|suburb|commune|metropolitan|valley|bay|coast|park|republic|kingdom|nation|archipelago|community|settlement|census|prefecture|oblast|emirate|capital city|in [A-Z])\b")
FAM=[(r"writ|wrote|pen|author|compos|lyric",r"wrote|written|writer|writers|penned|co-wrote|co-written|authored|author|composed|composer|lyrics|lyricist|songwriters?"),(r"sing|sang|sung|perform|record",r"sang|sung|sings|singer|recorded|performed|performer|vocals|vocalist|song|single|duet"),(r"play|portray|star|cast|actor|actress",r"played|plays|play|portrayed|portrays|portraying|starring|stars|starred|cast|role|as"),(r"voice",r"voiced|voices|voice|voicing"),(r"direct",r"directed|director|directs"),(r"invent|discover|found|creat|propos|commission|built|build|design|develop|coin|introduc|establish|formulat|credit|responsib|origin",r"invented|inventor|discovered|discoverer|founded|founder|founders|created|creator|proposed|commissioned|built|designed|designer|developed|developer|coined|introduced|established|formulated|credited|responsible|originated|first"),(r"\bw[io]ns?\b|winner|champion|beat",r"won|win|wins|winner|winners|champion|champions|defeated|defeating|beat|beating|title"),(r"marr|wife|husband",r"married|marries|marry|wife|husband|spouse|wed"),(r"releas|come out|came out|start|begin|began|premier|air|publish|debut|open|launch|written|made|creat",r"released|release|premiered|premiere|aired|debuted|published|began|begins|started|starts|launched|opened|written|wrote|made|created|produced|founded|established|formed|recorded"),(r"\bdie|\bdied|death|kill",r"died|death|killed|dies|murdered"),(r"\bborn\b",r"born|birth"),(r"enter|join",r"entered|joined|declared|entry"),(r"locat|found|situat|held|film|shot|\bset\b|take place|based|headquart|\bliv|come from|originat",r"located|found|situated|lies|based|held|filmed|shot|set|headquartered|lived|lives|live|born|originated|originates|native|takes place|took place|hosted")]
def fam(q:str)->re.Pattern:
 l=q.lower();f=[b for a,b in FAM if re.search(a,l)]+[w[:5]+r"\w*" for w in re.findall(r"\b[a-z]{5,}ed\b",l)];return re.compile(r"(?i)\b(?:%s)\b"%"|".join(f)) if f else None
def rel(qt:str,x:str,a:int,b:int,F,qn:set=frozenset())->float:
 if F is None:return 0.0
 ms=list(F.finditer(x));by=any(re.fullmatch(r"(?:\W+[\w'-]+){0,3}?\W+(?:%s)(?:\W+[\w'-]+){0,5}?\W+"%("in|at|near|from|on" if qt=="place" else "by")," "+x[m.end():a],re.I) for m in ms if m.end()<=a);act=any(re.fullmatch(r"(?:\W+[\w'-]+){0,2}?\W+",x[b:m.start()]+" ") and (m.group(0).lower()!="as" or any(stem_(w.lower()) in qn for w in re.findall(r"[A-Z]\w+",x[m.end():m.end()+25]))) for m in ms if m.start()>=b)
 near=any(abs((m.start()+m.end())/2-(a+b)/2)<70 for m in ms);return 1.0 if (by or act or (qt=="date" and near) or (qt=="num" and near) or re.search(r"\(\s*$",x[:a]) and qt=="person") else (0.4 if ms else 0.1)
NEGD=re.compile(r"(?i)same term|ethnic|racial|language|nationality|demonym|television|\bfilm\b|album|\bsong\b|genre|award|ceremony|season|episode|franchise|video game|novel|magazine|newspaper|website|software|brand|\bship\b|\bspecies\b")
def qtype(q:str)->str:
 l=q.lower().strip();return "date" if re.search(r"^(?:\w+ )?(?:when|what year|which year|in what year|in which year|what date)\b|\b(?:what|which) year\b",l) else ("num" if re.search(r"\bhow (?:many|much|old|long|far|tall|big|high|deep|fast)\b",l) else ("person" if re.search(r"^(?:\w+ ){0,2}who(?:m|se)?\b|^(?:what|which)(?: is| was| are| were)?(?: the)? (?:\w+'s )?(?:name|actor|actress|singer|band|group|player|president|author|writer|king|queen|team|composer|director|artist|rapper)\b|^what(?:'s| is| was) (?:[\w']+ ){0,3}name\b",l) else ("place" if re.search(r"^(?:\w+ )?where\b|\b(?:what|which) (?:country|city|state|county|continent|province|island|river|region|town)\b",l) else "other")))
def sents(txt:str)->list:
 out=[]
 for para in re.split(r"\n+",re.sub(r"\n=+[^=\n]+=+\n","\n",txt)):
  for x in re.split(r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bvs\.)(?<!\bNo\.)(?<!\bSr\.)(?<!\bCo\.)(?<!\bInc\.)\s+(?=[A-Z0-9\"“(])",para.strip()):
   x=re.sub(r"\s*\(\s*[;,]?\s*\)","",x.strip());out.append(x) if 20<len(x)<420 and not JUNK.search(x) and x[-1:] in ".!?\"”)" else None
 return out
def cands(qt:str,s:str,q:str,loc:tuple=None)->list:
 qn=set(words(q.encode()));ql=q.lower()
 if qt=="date":return [(m.group(0),m.group(1),m.start(),m.end()) for m in DATE.finditer(s) if m.group(1) not in ql]
 if qt=="num":
  hq=re.search(r"\bhow (?:many|much|old|long|far|tall|big|high|deep|fast)\s+(\w+)?",ql);head=stem_(hq.group(1)) if hq and hq.group(1) else "";unit=re.compile(r"(?i)\b(years?|months?|days?|hours?|minutes?|seconds?|centur(?:y|ies)|decades?|weeks?)\b") if "long" in hq.group(0) or "old" in hq.group(0) else (re.compile(r"(?i)\$|\bdollars?\b|\b(?:pounds?|euros?|€|£)|\bUS\$") if "much" in hq.group(0) and re.search(r"\b(cost|price|worth|pay|paid|earn|make|spend|budget|salary|value)\b",ql) else None)
  out=[]
  for m in NUM.finditer(s):
   tail=s[m.end():m.end()+40];v=m.group(0)
   ok=(bool(unit.search(s[max(0,m.start()-3):m.end()+14])) if unit else (any(stem_(w)==head for w in re.findall(r"[a-z]+",tail.lower())[:4]) if head and "many" in hq.group(0) else True)) and v.lower() not in ql and not re.fullmatch(r"(?:1[0-9]|20)\d\d",v)
   out.append((v+(" "+re.match(r"\s*(\S+(?: \S+)?)",tail).group(1) if unit and not unit.search(v) and re.match(r"\s*\S",tail) else ""),v.lower().replace(",",""),m.start(),m.end())) if ok else None
  return out
 out=[]
 for m in NAME.finditer(s):
  tk=re.sub(r"['’]s$","",m.group(0)).split();a=m.start()
  while tk and _common(tk[0],loc):a=s.index(tk[0],a)+len(tk[0]);tk=tk[1:]
  while tk and _common(tk[-1],loc):tk=tk[:-1]
  v=" ".join(tk);a=s.find(v,a) if v else -1
  if not tk or len(tk)>4 or a<0 or any(stem_(t.lower().strip(".")) in qn or t.lower() in ql.split() for t in tk) or all(_common(t,loc) for t in tk) or len(v)<3:continue
  out.append((v,v.lower(),a,a+len(v)))
 return out
def _common(t:str,loc:tuple=None)->bool:return t in STOPC or bool(loc and loc[0].get(t.lower(),0)>loc[1].get(t,0))
def stem_(w:str)->bytes:
 r=words(w.encode());return r[0] if r else w.encode()
MIRROR=re.compile(r"wikipedia\.org|wikiwand|grokipedia|explained\.today|wiki2\.org|wikimili|alchetron|dbpedia|infogalactic|justapedia|wikizero|wikiless|kiddle|wikiwiki|en-academic|wikibrief|thereaderwiki|wiki\.alquds|dictionary\.sensagent")
def _sk(u:str)->str:
 p=urllib.parse.urlparse(u);return "w:"+urllib.parse.unquote(p.path.rstrip("/").rsplit("/",1)[-1]).replace("_"," ").lower() if MIRROR.search(p.netloc) else p.netloc.replace("www.","")
def gather(q:str)->tuple:
 with ThreadPoolExecutor(4) as ex:
  f=[ex.submit(ddg,z,10) for z in (q+" wikipedia",q," ".join(w for w in re.findall(r"[\w'-]+",q.lower()) if w not in STOP))]+[ex.submit(search,q,5)];e=f[0].result()+f[1].result()+f[2].result()
  w=[urllib.parse.unquote(x["u"].split("/wiki/",1)[1].split("#")[0]).replace("_"," ") for x in f[0].result() if "en.wikipedia.org/wiki/" in x["u"]]
  return [x for i,x in enumerate(e) if x["u"] not in {y["u"] for y in e[:i]}],[page(x) for x in list(dict.fromkeys(w[:3]+f[3].result()[:3]))[:5]]
def evidence(q:str)->tuple:
 qt=qtype(q);F=fam(q);d,P=gather(q);qw=set(words(q.encode()));fw={w for w in qw if any(re.search(a,w.decode()) for a,_ in FAM)};W={w:idf(w) for w in qw};tot=sum(W.values()) or 1.0;yrs=set(re.findall(r"\b(?:1[0-9]|20)\d\d\b",q));V={}
 ps=[(t,u,x,set(words(t.encode()))) for t,txt,_,u in P for x in sents(txt)]+[(urllib.parse.urlparse(x["u"]).netloc,x["u"],y,set()) for x in d if "wikipedia.org" not in x["u"] for y in sents(re.sub(r"^(?:\w{3} \d{1,2}, \d{4}|\d+ \w+ ago)\s*[·—–-]+\s*","",x["b"]))]
 nz=lambda y:re.sub(r"[^a-z0-9]","",y.lower())[:50];dup={nz(x):"w:"+t.lower() for t,txt,_,u in P for x in sents(txt)};ps=[(t,u,x,tw,"w:"+t.lower() if "wikipedia.org" in u else dup.get(nz(x),_sk(u))) for t,u,x,tw in ps]
 al=" ".join(t for _,t,_,_ in P);loc=(collections.Counter(re.findall(r"\b[a-zß-öø-ÿ]+\b",al)),collections.Counter(re.findall(r"(?<![.!?]\s)\b[A-ZÀ-ÖØ-Þ]\w+",al)))
 for t,u,x,tw,sk in ps:
  sw=set(words(x.encode()))|(fw if F and F.search(x) else set());r=(sum(W[w] for w in qw&sw)+0.5*sum(W[w] for w in (qw&tw)-sw))/tot
  if r<0.3 or (yrs and not all(y in x+" "+t for y in yrs)):continue
  xl=x.lower();pos=[m.start() for w in qw&sw for m in re.finditer(r"\b"+re.escape(w.decode()),xl)]
  for v,k,a,b in cands(qt,x,q,loc):dist=min([a-p0 if p0<a else max(0,p0-b) for p0 in pos] or [200]);V.setdefault(k,[]).append((r/(1+dist/80.0),r,v,x,t,u,rel(qt,x,a,b,F,qw),sk))
 for k in (sorted(V,key=len) if qt in ("person","place") else []):
  sup=[j for j in V if j!=k and set(k.split())<set(j.split())];len(sup)==1 and V[sup[0]].extend(V.pop(k))
 R=[]
 for k,h in V.items():
  f=collections.Counter(z[2] for z in h);tf=max(f.values());v=max([z for z in f if f[z]>=max(2,tf/3)] or f,key=lambda z:(len(z),f[z]) if f[z]>=max(2,tf/3) else (f[z],len(z)));hb=max([z for z in h if v in z[3]] or h,key=lambda z:(z[6],"wikipedia" in z[5],z[0]+0.05*len(re.findall(r"\b(?:%s)\b|\b\d{1,2}\b"%MON,z[2]))))
  R.append((k,{"v":v,"mx":max(z[0] for z in h),"rs":sum(max([z[1]*z[6] for z in h if z[7]==t]) for t in {z[7] for z in h}),"ns1":len({z[7] for z in h if z[6]>=1.0 and z[1]>=0.6}),"ev":sorted({z[5] for z in h if z[6]>=1.0 and z[1]>=0.6}),"span":hb[2] if v in hb[2] else v,"best":(hb[3],hb[4],hb[5])}))
 R=sorted(R,key=lambda kv:-(kv[1]["rs"]+0.1*kv[1]["mx"]))[:12]
 if qt in ("person","place") and R:
  ds=descs([e["v"] for _,e in R])
  for _,e in R:e["desc"]=ds.get(e["v"]);e["ok"]=bool(e["desc"] and e["desc"]!="?" and (PERSONY if qt=="person" else PLACEY).search(e["desc"]) and not NEGD.search(e["desc"]))
 return qt,R,P,d
class Learned:
 def __init__(self,path:str=STORE):
  self.path=path;self.rows=[json.loads(l) for l in open(path)] if os.path.exists(path) else []
 def find(self,q:str)->dict:
  qw=set(words(q.encode()));b=None;bs=0.0
  for r in self.rows:
   rw=set(words(r["q"].encode()));i=len(qw&rw);rec=i/max(1,len(qw));pre=i/max(1,len(rw))
   if rec>=0.8 and pre>=0.6 and hard(q,r["q"],r["a"].encode()) and rec+pre>bs:bs,b=rec+pre,r
  return b
 def add(self,q:str,a:str,title:str,url:str,**kw)->dict:
  r={"q":q,"a":a,"title":title,"src":url,"t":time.strftime("%Y-%m-%dT%H:%M:%S"),**kw};os.makedirs(os.path.dirname(self.path) or ".",exist_ok=True);open(self.path,"a").write(json.dumps(r)+"\n");self.rows.append(r);return r
def decide(q:str)->dict:
 qt,R,P,d=evidence(q) if qtype(q)!="other" else ("other",[],[],[]);R=[e for _,e in R if e.get("ok",True)]
 if not R:return None
 e=R[0];e2=R[1] if len(R)>1 else {"rs":0.0,"ns1":0};g=e["ns1"]-e2["ns1"];m=e["rs"]/(e["rs"]+e2["rs"]+1e-9)
 return {"a":e["best"][0],"title":e["best"][1],"src":e["best"][2],"span":e["span"],"confidence":round(m*(1-0.5**g),3),"evidence":e["ev"],"qtype":qt} if e["ns1"]>=2 and m>=0.65 and (g>=3 or e2["ns1"]==0) else None
def learn(q:str,store:Learned)->dict:
 if PERSONAL.search(q) or TEMPORAL.search(q):return None
 try:h=decide(_scrub(q))
 except Exception:return None
 return store.add(q,h.pop("a"),h.pop("title"),h.pop("src"),**h) if h else None
