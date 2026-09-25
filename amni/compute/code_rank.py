import ast,functools,json,math,os,re,sys,zlib
import numpy as np
from pathlib import Path
from amni.compute import tile_synth as TS
ROOT=Path(__file__).resolve().parents[2];SYNP=ROOT/"data"/"clean2"/"synsets.txt";MODEL=ROOT/"data"/"code_eval"/"x10_rank.npz"
M,K4,NR=1<<18,np.array([1,17,289,4913,83521,1419857,24137569,410338673],np.uint64),64
OPC={"sort":("sort sorted order ascending arrange alphabetical alphabetically increasing","c:sorted m:sort"),"desc":("descending decreasing","k:reverse s:rev c:reversed"),"count":("count counts number many occurrence occurrences frequency times","m:count c:len c:sum c:Counter"),"uniq":("unique distinct duplicate duplicates different repeated","c:set m:fromkeys"),"sum":("sum total add plus added","c:sum o:Add"),"prod":("product multiply multiplied times","o:Mult c:prod"),"diff":("difference subtract minus","o:Sub"),"div":("divide divided ratio quotient average mean half","o:Div o:FloorDiv"),"mod":("even odd divisible remainder multiple modulo","o:Mod"),"max":("maximum largest biggest greatest highest longest most max","c:max"),"min":("minimum smallest least lowest shortest min fewest","c:min"),"rev":("reverse reversed backward backwards","s:rev c:reversed"),"first":("first beginning start starts leading head","s:0 m:startswith s:take"),"last":("last end ends ending final trailing","s:-1 m:endswith s:drop"),"index":("index indices position positions","c:enumerate m:index m:find c:range"),"len":("length long size","c:len"),"filter":("remove filter keep only exclude without delete","p:filter m:replace"),"all":("all every","c:all"),"any":("any exist exists contain contains some present","c:any o:In"),"pair":("pair pairs two","p:pair c:zip"),"adj":("consecutive adjacent neighbor neighbour neighbours","p:adj c:zip"),"pow":("power square squared cube exponent","o:Pow"),"abs":("absolute magnitude","c:abs"),"digit":("digit digits","c:str c:int"),"bin":("binary bit bits bitwise","c:bin o:BitAnd o:BitXor o:RShift o:LShift o:BitOr"),"upper":("upper uppercase capital capitalize capitalized","m:upper m:isupper m:capitalize m:title"),"lower":("lower lowercase","m:lower m:islower"),"word":("word words sentence","m:split m:join"),"join":("join concatenate combine merge","m:join o:Add"),"replace":("replace substitute change","m:replace m:swapcase"),"cmp":("greater larger less smaller more fewer exceed above below","o:Lt o:LtE"),"eq":("equal same identical match","o:Eq o:NotEq"),"acc":("running cumulative accumulate balance","p:acc c:accumulate"),"loop":("each every","p:iter"),"round":("round rounded nearest decimal","c:round"),"char":("character characters letter letters","c:ord c:chr c:list")}
TYW={"bool":r"\b(return|returns|output)s? (true|false)\b|\b(true|false) if\b|\bcheck (whether|if)\b|\bdetermine (whether|if)\b|\bwhether\b","int":r"\bnumber of\b|\bhow many\b|\bcount\b|\bindex of\b","list":r"\b(return|returns)( a| the)? (new )?(list|array)\b|\blist of\b","str":r"\b(return|returns)( a| the)? (new )?string\b"}
EDGE=re.compile(r"(?:return|returns|output|should return|should be|must return)\s+(-?\d+(?:\.\d+)?|none|null|true|false|an? empty (?:list|string|array)|\[\]|''|\"\"|0)\s*(?:,\s*)?(?:if|when|in case)\s+([^.;\n]{0,80})|(?:if|when)\s+([^,.;\n]{0,80}?),?\s*(?:then\s+)?(?:return|output|the (?:result|answer|output) (?:is|should be))\s+(-?\d+(?:\.\d+)?|none|null|true|false|an? empty (?:list|string|array)|\[\]|''|\"\"|0)")
PROPW={"srt":r"\bascending\b|\bincreasing order\b|\bsorted\b|\bsort(s|ed)? (the |a |given )?(list|array|elements|numbers|characters|letters|words)\b","srtr":r"\bdescending\b|\bdecreasing order\b","uniq":r"\b(unique|distinct)\b|\bremove (all )?(the )?duplicates?\b|\bwithout duplicates?\b","sub":r"\b(remove|filter|filters|keep only|extract|delete)\b","nneg":r"\bcount\b|\bhow many\b|\bnumber of\b"}
STOPS=set("a an the of to in and or is are be for this it as that which with from by on its given write function python".split())
PHR={"count":r"\bnumber of\b|\bhow many\b","max":r"\bthe most\b","min":r"\bthe least\b","filter":r"\bonly\b.*\b(that|which|whose)\b"}
@functools.lru_cache(None)
def lex():
 d={}
 try:
  for l in open(SYNP,encoding="utf-8"):p=l.split();(d.__setitem__(p[0],(int(p[1]),int(p[2]))) if len(p)==3 else None)
 except OSError:pass
 return d
def syn(w):L=lex();return L.get(w) or L.get(TS._stem(w)) or ()
@functools.lru_cache(None)
def opx():
 w2o={};o2f={}
 for o,(ws,fs) in OPC.items():
  o2f[o]=set(fs.split())
  for w in ws.split():[w2o.setdefault(k,set()).add(o) for k in (w,TS._stem(w))+tuple(f"#{s}" for s in syn(w)[:1])]
 return w2o,o2f
def words(t):return [w for w in re.findall(r"[a-z]+",re.sub(r"([a-z])([A-Z])",r"\1 \2",t).replace("_"," ").lower()) if len(w)>1]
def hk(s):return zlib.crc32(s.encode())%M
def _val(v):
 v=v.strip().lower();return None if v in ("none","null") else True if v=="true" else False if v=="false" else [] if v in ("[]",) or "empty list" in v or "empty array" in v else "" if v in ("''",'""') or "empty string" in v else float(v) if "." in v else int(v)
class Mem:
 __slots__=("ent","rel","val","n","tab")
 def __init__(s):s.ent,s.rel,s.val,s.n,s.tab=np.zeros(NR,np.uint32),np.zeros(NR,np.uint32),[None]*NR,0,{}
 def put(s,e,r,v=True):k=(hk(e),hk(r));j=s.tab.get(k,s.n if s.n<NR else None);(None if j is None else (s.ent.__setitem__(j,k[0]),s.rel.__setitem__(j,k[1]),s.val.__setitem__(j,v),s.tab.__setitem__(k,j),setattr(s,"n",max(s.n,j+1))));return s
 def get(s,e,r,d=None):j=s.tab.get((hk(e),hk(r)));return d if j is None else s.val[j]
 def ents(s,r):h=hk(r);return [s.val[j] for j in np.flatnonzero(s.rel[:s.n]==h)]
def memory(sp):
 m=Mem();d=sp.desc;lc=d.lower();w2o,_=opx();ws=[w for w in words(d) if w not in STOPS];m.put("doc","words",ws)
 for k,p in enumerate(sp.params):
  a=ast.unparse(sp.ann[k]) if k<len(sp.ann) and sp.ann[k] is not None else "";t=(TS._at(a) or (TS._vt(sp.rows[0][k]) if sp.rows else []))[:1];m.put(p,"type",t[0] if t else "").put(p,"is","param").put("param","list",m.get("param","list",[])+[p])
 rt=(TS._at(ast.unparse(sp.ret)) if sp.ret is not None else [])[:1] or (TS._vt(sp.T[min(sp.T)])[:1] if sp.T else []);m.put("ret","type",rt[0] if rt else "")
 [m.put("ret","says",t) for t,rx in TYW.items() if re.search(rx,lc)]
 ops=sorted({o for o,rx in PHR.items() if re.search(rx,lc)}|{o for w in ws for o in w2o.get(w,set())|w2o.get(TS._stem(w),set())|set().union(*[w2o.get(f"#{x}",set()) for x in syn(w)[:1]])});m.put("op","want",ops)
 ed=[]
 for g in EDGE.finditer(lc):
  v,c=(g.group(1),g.group(2)) if g.group(1) else (g.group(4),g.group(3))
  try:ed.append((_val(v),"empty" if re.search(r"\bempty\b|\bno (elements|items|numbers)\b|\blength (is )?(0|zero)\b",c) else "neg" if re.search(r"\bnegative\b",c) else "zero" if re.search(r"\bzero\b|\b0\b",c) else "other"))
  except Exception:pass
 m.put("edge","facts",ed).put("edge","empty",bool(re.search(r"\bempty\b",lc))).put("prop","want",[k for k,rx in PROPW.items() if re.search(rx,lc)])
 m.put("const","mention",{c[0] for c in sp.consts if c[2]==1});return m
JAIL=r'''
import json,signal,copy,zlib,math,itertools,functools,collections,re,string,sys,time
from typing import *
sys.setrecursionlimit(3000)
def _cn(v):t=type(v);return repr(int(v)) if t in (int,bool) or t is float and v==v and abs(v)<1e15 and v==int(v) else repr(round(v,6)) if t is float else repr(v) if t in (str,type(None)) else "["+",".join(map(_cn,v))+"]" if t is list else "("+",".join(map(_cn,v))+")" if t is tuple else "{"+",".join(sorted(map(_cn,v)))+"}" if t in (set,frozenset) else "{"+",".join(sorted(_cn(a)+":"+_cn(b) for a,b in v.items()))+"}" if t is dict else "<"+t.__name__+">"
def _fl(o,x):
 L=type(o) in (list,tuple);f=0
 try:f|=1 if L and all(a<=b for a,b in zip(o,o[1:])) else 0;f|=2 if L and all(a>=b for a,b in zip(o,o[1:])) else 0
 except Exception:pass
 try:f|=4 if L and len(set(map(_cn,o)))==len(o) else 0
 except Exception:pass
 try:s={_cn(e) for a in x if type(a) is type(o) and type(a) in (list,str,tuple) for e in a};f|=8 if type(o) in (list,str,tuple) and s and {_cn(e) for e in o}<=s else 0
 except Exception:pass
 try:f|=16 if type(o) in (list,str) and any(type(a) in (list,str) and len(a)==len(o) for a in x) else 0
 except Exception:pass
 f|=32 if type(o) in (int,float) and not type(o) is bool and o>=0 else 0;f|=64 if any(_cn(o)==_cn(a) for a in x) else 0
 return f
def _al(*a):raise TimeoutError
D=json.loads(sys.stdin.read());P,N,B=D["p"],D["n"],D["b"];inf=float("inf");nan=float("nan");I=[eval(x) for x in D["i"]];signal.signal(signal.SIGALRM,_al);T0=time.time();out=[]
for b in B:
 g={};f=None
 try:exec(P+b,g);f=g.get(N)
 except BaseException:f=None
 row=[]
 for x in I:
  if f is None or time.time()-T0>D["t"]:row.append([-3 if f else -1,"",0,""]);continue
  try:
   signal.setitimer(signal.ITIMER_REAL,0.05);o=f(*copy.deepcopy(x));signal.setitimer(signal.ITIMER_REAL,0);c=_cn(o);row.append([_fl(o,x),type(o).__name__,zlib.crc32(c.encode()),c[:40]])
  except TimeoutError:row.append([-2,"",0,""])
  except BaseException:signal.setitimer(signal.ITIMER_REAL,0);row.append([-1,"",0,""])
 out.append(row)
print("X10J"+json.dumps(out))
'''
def inputs(sp,K=16):
 ins,sn=[],set()
 for r,k in [(r,"x") for r in sp.rows]+[(r,"n") for r in TS._probes(sp.rows,K//2)]+[(r,"m") for r in TS._probes(sp.rows,K,1)]+[(r,"z") for r in _empties(sp)]+[(r,"e") for r in TS._edges(sp.rows)]:
  try:q=repr(r);_=ast.literal_eval(q.replace("inf","0").replace("nan","0"))
  except Exception:continue
  (ins.append((r,k.upper() if k in "nme" and TS._ood(sp.rows,r) else k)),sn.add(q)) if q not in sn and len(q)<20000 else None
 return ins
def _empties(sp):
 if not sp.rows:return []
 r=sp.rows[0];e=lambda v:type(v)() if type(v) in (list,str,tuple,dict,set) else v;out=[tuple(e(v) if j==k else v for j,v in enumerate(r)) for k in range(len(r)) if type(r[k]) in (list,str,tuple,dict,set)]
 return out+([tuple(e(v) for v in r)] if len(out)>1 else [])
def jail(prompt,name,bodies,ins,budget=6.0):
 sys.path.insert(0,str(ROOT/"scripts")) if str(ROOT/"scripts") not in sys.path else None
 import subprocess,tempfile,shutil
 from x0_code_eval import _bwrap_cmd,PY,TMP
 TMP.mkdir(parents=True,exist_ok=True);wd=tempfile.mkdtemp(prefix="x10_",dir=str(TMP))
 try:
  (Path(wd)/"j.py").write_text(JAIL,encoding="utf-8");p=subprocess.run(_bwrap_cmd(wd,[PY,"j.py"],cpu=int(budget)+6,mem=1<<30),input=json.dumps({"p":prompt,"n":name,"b":bodies,"i":[repr(r) for r,_ in ins],"t":budget}),text=True,capture_output=True,timeout=budget+15,env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"});m=re.search(r"X10J(.*)",p.stdout or "")
  return json.loads(m.group(1)) if m else [[[-1,"",0,""]]*len(ins) for _ in bodies]
 except subprocess.TimeoutExpired:return [[[-2,"",0,""]]*len(ins) for _ in bodies]
 finally:shutil.rmtree(wd,ignore_errors=True)
def _b(x,cuts):return next((k for k,c in enumerate(cuts) if x<=c),len(cuts))
TYN={"int":"int","float":"float","bool":"bool","str":"str","list":"list","tuple":"tuple","dict":"dict","set":"set","NoneType":"none","frozenset":"set"}
def cfeats(sp,m,cands,beh,kinds):
 n=len(cands);_,o2f=opx();want=m.get("op","want",[]);ws=m.get("doc","words",[]);sy=[f"#{x}" for w in ws for x in syn(w)[:1]];rt=m.get("ret","type","");says=set(m.ents("says"));ed=m.get("edge","facts",[]);props=m.get("prop","want",[]);ment=m.get("const","mention",set())
 sc=np.array([c.get("score",0.0) for c in cands],float);co=np.array([c.get("cost",0) for c in cands],float);ind=[j for j,k in enumerate(kinds) if k in "xnmez"];inn=[j for j,k in enumerate(kinds) if k in "nm"];emp=[j for j,k in enumerate(kinds) if k=="z"]
 sig=[tuple(beh[i][j][2] if beh[i][j][0]>=0 else -1 for j in inn) for i in range(n)];cl={s:sig.count(s) for s in sig};out=[]
 for i,c in enumerate(cands):
  try:F=TS.feats(c.get("expr") or "") if "\n" not in (c.get("expr") or "") else TS.feats(c.get("body") or "")
  except Exception:F=set()
  F=set(F);co_=[o for o,fs in o2f.items() if fs&F];B=beh[i];ok=[B[j] for j in ind if B[j][0]>=0];crash=[j for j in inn if B[j][0]<0];ty={TYN.get(x[1],"obj") for x in ok}
  f={"b:rank":_b(i,(0,1,2,4,8)),"b:dsc":_b(sc[i]-sc.min(),(0,0.5,1,2,4)),"b:dco":_b(co[i]-co.min(),(0,1,2,4)),"b:sk":int(bool(c.get("sk"))),"b:loo":_b(c.get("loo_fail") or 0,(0,1)),"b:cov":_b(c.get("cov",0),(0,1,2,3)),"b:nod":_b(TS._nodes(c.get("expr") or "0") if "\n" not in (c.get("expr") or "") else 30,(4,7,10,15,25))}
  f["op:conf"]=_b(len(set(want)&set(co_)),(0,1,2));f["op:miss"]=_b(len(set(want)-set(co_)),(0,1,2,3));f["op:extra"]=_b(len(set(co_)-set(want)),(0,1,2,3))
  f["ty:ret"]=(rt or "?")+"|"+("+".join(sorted(ty)) or "none");f["ty:mis"]=int(bool(rt) and any(t!=rt and not (t=="int" and rt=="float") for t in ty));f["ty:none"]=int("none" in ty);[f.__setitem__("ty:says:"+t,int(t in ty)) for t in says]
  f["cr:in"]=_b(len(crash)/max(1,len(inn)),(0,0.1,0.3,0.6));f["cr:to"]=int(any(B[j][0]==-2 for j in ind));f["cl"]=_b(cl.get(sig[i],0)/max(1,n),(0.15,0.3,0.5,0.8));f["deg"]=int(len({B[j][2] for j in inn if B[j][0]>=0})<=1 and len(inn)>2);f["ident"]=_b(sum(1 for j in inn if B[j][0]>=0 and B[j][0]&64)/max(1,len(inn)),(0,0.5,0.99))
  for v,cond in ed:
   cv=zlib.crc32(TS._cn(v).encode());hit=any(B[j][0]>=0 and B[j][2]==cv for j in ind)
   f["e:val:"+cond]=int(hit);(f.__setitem__("e:emp",int(all(B[j][0]>=0 and B[j][2]==cv for j in emp))) if cond=="empty" and emp else None)
  (f.__setitem__("e:empcr",int(any(B[j][0]<0 for j in emp))) if m.get("edge","empty") and emp else None)
  for p in props:bit={"srt":1,"srtr":2,"uniq":4,"sub":8,"nneg":32}[p];vs=[x for x in (B[j] for j in inn) if x[0]>=0 and x[1] in (("list","tuple") if p in ("srt","srtr","uniq") else ("list","str","tuple") if p=="sub" else ("int",))];f["pr:"+p]=_b(sum(1 for x in vs if not x[0]&bit)/max(1,len(vs)),(0,0.2,0.6)) if vs else -1
  ks=[x[2:] for x in F if x[:2]=="k:" and x not in ("k:0","k:1","k:bool","k:int","k:str","k:float","k:None")];f["k:for"]=_b(len([k for k in ks if k not in {repr(x) for x in ment}]),(0,1,2));f["k:use"]=_b(len([x for x in ment if repr(x) in (c.get("expr") or "")]),(0,1))
  X=[f"{k}={v}" for k,v in f.items()]+[f"F|{x}" for x in F]+[f"O|{o}" for o in co_]+[f"W|{w}|{x}" for w in set(ws) for x in F]+[f"S|{s}|{x}" for s in set(sy) for x in F]+[f"WO|{w}|{o}" for w in set(ws) for o in co_]+[f"R|{rt}|{x}" for x in F]+[f"OC|{o}|{'+' if o in co_ else '-'}" for o in want]+[f"OC|{o}|x" for o in co_ if o not in want]
  out.append(X)
 return out
def v19keys(prompt,cands,path=None):
 from amni.compute import ray_field as RF
 n=len(cands);bs=[(c.get("body") or "").encode() for c in cands];lp=RF.engine(path or RF.DEFAULT_PACK).score([prompt.encode()]*n+[b""]*n,bs+bs) if n else np.zeros(0);v=(lp[:n]-lp[n:])/np.maximum(1,[len(b) for b in bs]);z=(v-v.mean())/(v.std()+1e-9) if n>1 else np.zeros(n);r=np.argsort(np.argsort(-v)) if n else v
 return [[f"V:r={_b(int(r[i]),(0,1,3))}",f"V:z={_b(float(z[i]),(-1,-0.3,0.3,1))}"] for i in range(n)]
def prep(prompt,checks,cands,budget=6.0):
 sp=TS.Spec(prompt,checks);m=memory(sp);ins=inputs(sp);beh=jail(prompt,sp.name,[c["body"] for c in cands],ins,budget) if cands else [];return cfeats(sp,m,cands,beh,"".join(k for _,k in ins)) if cands else []
class Model:
 def __init__(s,w=None,cfg=None):s.w=np.zeros(M,np.float32) if w is None else w;s.cfg=cfg or {}
 def keys(s,X):gs=s.cfg.get("groups");return [k for k in X if not gs or k.split("|")[0].split("=")[0].split(":")[0] in gs]
 def score(s,feats):return np.array([float(np.take(s.w,[hk(k) for k in s.keys(X)]).sum()) for X in feats]) if feats else np.zeros(0)
 def save(s,path):u=s.w.view(np.uint32).astype(np.uint64);Path(path).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,page=((u[:,None]//K4[None,:])%np.uint64(17)).astype(np.uint8).reshape(-1,2,4),cfg=np.frombuffer(json.dumps(s.cfg).encode(),np.uint8))
 @staticmethod
 def load(path):z=np.load(path);d=z["page"].reshape(-1,8).astype(np.uint64);return Model(((d*K4[None,:]).sum(1)).astype(np.uint32).view(np.float32),json.loads(z["cfg"].tobytes()))
XG=("W|","S|","WO|","R|")
def compile_(probs,model):
 K,XK,C,G,Y=[],set(),[],[],[];c=0
 for g,p in enumerate(probs):
  for i,X in enumerate(p["feats"]):ks=model.keys(X);K+=[hk(k) for k in ks];XK|={hk(k) for k in ks if k.startswith(XG)};C+=[c]*len(ks);G.append(g);Y.append(bool(p["oks"][i]));c+=1
 return np.array(K,np.int64),np.array(sorted(XK),np.int64),np.array(C,np.int64),np.array(G,np.int64),np.array(Y,bool)
def train(probs,cfg):
 nb=int(cfg.get("bag",0))
 if nb:g=np.random.default_rng(int(cfg.get("seed",17)));ms=[train([probs[i] for i in g.integers(0,len(probs),len(probs))],{**cfg,"bag":0}) for _ in range(nb)];return Model(np.mean([m.w for m in ms],0).astype(np.float32),cfg)
 model=Model(cfg=cfg);K,XK,C,G,Y=compile_(probs,model);nc=len(G);ng=int(G.max())+1 if nc else 0;mix=np.bincount(G,Y,ng);tot=np.bincount(G,minlength=ng);use=((mix>0)&(mix<tot))[G] if nc else np.zeros(0,bool);w=model.w.astype(np.float64);h=np.full(M,1e-8);lr,it=float(cfg.get("lr",0.5)),int(cfg.get("it",150));lam=np.full(M,float(cfg.get("l2",0.01)));lam[XK]=float(cfg.get("l2x",cfg.get("l2",0.01)))
 st=np.r_[0,np.flatnonzero(np.diff(G))+1] if nc else np.zeros(0,np.int64);run=np.cumsum(np.r_[0,np.diff(G)!=0]) if nc else np.zeros(0,np.int64)
 for _ in range(it if nc else 0):
  s=np.bincount(C,w[K],nc);e=np.exp(s-np.maximum.reduceat(s,st)[run]);Z=np.bincount(G,e,ng)[G];zr=np.bincount(G,e*Y,ng)[G];gs=(e/Z-np.where(Y,e/np.maximum(zr,1e-300),0))*use;gw=np.bincount(K,gs[C],M)+lam*w;h+=gw*gw;w-=lr*gw/np.sqrt(h)
 model.w=w.astype(np.float32);return model
def pick(model,feats,x9=0):s=model.score(feats);j=int(np.lexsort((np.arange(len(s)),-np.round(s,6)))[0]) if len(s) else x9;return j if len(s) and s[j]-s[min(x9,len(s)-1)]>float(model.cfg.get("tau",0)) else x9
@functools.lru_cache(None)
def model(path=None):p=Path(MODEL if path in (None,"","on","1") else path);return Model.load(p) if p.exists() else None
def rank(prompt,checks,cands,x9=0,path=None):
 t=__import__("time").perf_counter();m=model(path)
 try:F=prep(prompt,checks,cands) if m is not None else [];j=pick(m,F,x9) if F else x9;return {"pick":j,"x9":x9,"flip":j!=x9,"time":round(__import__("time").perf_counter()-t,3),"model":str(path or MODEL)}
 except Exception as e:return {"pick":x9,"x9":x9,"flip":False,"error":f"{type(e).__name__}: {e}"[:200]}
