import os,re,json,time,random,hashlib,argparse,resource,collections,multiprocessing as mp
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view as swv
ROOT=Path(__file__).resolve().parents[2];EVD=ROOT/"data"/"concepts"/"eval";CACHE=EVD/".decontam_cache";VER=3;SEED=17;KEEP=4;MIN_DISTINCT=4
PATS=("data/concepts/eval/**/*.json*","data/rchat/eval/conv_*.jsonl","data/webeval/*","data/code_eval/*.jsonl","data/chat*/hold_eval.jsonl","data/concepts/eval/v20_m13/*","data/concepts/eval/v20_m15/**/*","data/math/gsm8k/test*.jsonl","data/math/svamp/*.json","data/adam_t/eval/**/*.json*")
SKIP=re.compile(r"(?i)report|nightly|_stub|harvested_tiles|prereg_lock|(^|_)meta\.json$|\.(log|pid|md|tmp|tgz|gz|zip|npz|npy|bin|pt|txt)$")
TOK=re.compile(r"[^\W_]+");_MEMO={}
class Index:
 __slots__=("n","V","R","H","M","groups","files","key","stats")
 def __repr__(s):return f"Index(n={s.n}, files={len(s.files)}, grams={len(s.H)}, vocab={len(s.V)}, key={s.key[:12]})"
def rel(p)->str:
 p=Path(p).resolve();return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def group(p)->str:
 s=rel(p).split("/");return "m13" if "v20_m13" in s else "m15" if "v20_m15" in s else ("concepts" if s[1]=="concepts" else s[2] if s[1]=="math" and len(s)>3 else s[1]) if s[0]=="data" and len(s)>2 else Path(p).parent.name
def eval_sources()->list:return sorted({p.resolve() for pat in PATS for p in ROOT.glob(pat) if p.is_file() and not SKIP.search(p.name) and ".decontam_cache" not in p.parts})
def load(p):
 t=Path(p).read_text(encoding="utf-8",errors="replace")
 try:return json.loads(t)
 except ValueError:pass
 try:return [json.loads(l) for l in t.splitlines() if l.strip()]
 except ValueError:return t
def leaves(o)->list:return [o] if isinstance(o,str) else [y for k,v in o.items() for y in leaves(k)+leaves(v)] if isinstance(o,dict) else [y for v in o for y in leaves(v)] if isinstance(o,(list,tuple)) else []
def words(t:str)->list:return TOK.findall(t.lower())
def _mul(n:int)->np.ndarray:return np.random.PCG64(1300+n).random_raw(n).astype(np.uint64)|np.uint64(1)
def _roll(i:np.ndarray,R:np.ndarray,n:int,dig:np.ndarray=None)->tuple:
 L=len(i)-n+1
 if L<=0:return np.zeros(0,np.uint64),np.zeros(0,bool)
 ok=i>=0;j0=np.where(ok,i,0);r=np.where(ok,R[j0] if len(R) else 0,0).astype(np.uint64);c=np.concatenate(([0],np.cumsum(~ok)));v=(c[n:]-c[:-n])==0;M=_mul(n);h=np.zeros(L,np.uint64)
 d=np.concatenate(([0],np.cumsum(dig[j0]&ok))) if dig is not None and len(dig) else None;v=v&((d[n:]-d[:-n])<n) if d is not None else v
 for j in range(n):h+=r[j:j+L]*M[j]
 return h,v
def _key(fs:list,n:int)->str:
 h=hashlib.sha256(f"{VER}|{n}".encode());[h.update(f"|{rel(f)}|{st.st_mtime_ns}|{st.st_size}".encode()) for f in fs for st in [os.stat(f)]];return h.hexdigest()
def _build(fs:list,n:int,k:str)->Index:
 t0=time.time();V={};per=[];st=collections.Counter()
 for f in fs:
  ids=[]
  for s in leaves(load(f)):ids.extend(V.setdefault(w,len(V)) for w in words(s));ids.append(-1);st["leaves"]+=1
  per.append((f,np.array(ids,np.int64)));st["words"]+=len(ids)
 ix=Index();ix.n,ix.V,ix.key,ix.files=n,V,k,[rel(f) for f in fs];ix.groups=sorted({group(f) for f in fs});gi={g:j for j,g in enumerate(ix.groups)};assert len(ix.groups)<=32,ix.groups
 ix.R=np.random.PCG64(SEED).random_raw(len(V)).astype(np.uint64);dig=np.fromiter((w.isdigit() or len(w)==1 for w in V),bool,len(V));H=[np.zeros(0,np.uint64)];G=[np.zeros(0,np.uint32)];pf={}
 for f,i in per:h,v=_roll(i,ix.R,n,dig);u=np.sort(swv(i,n),axis=1) if len(v) else np.zeros((0,n),np.int64);v=v&(1+(u[:,1:]!=u[:,:-1]).sum(1)>=MIN_DISTINCT);H.append(h[v]);G.append(np.full(int(v.sum()),1<<gi[group(f)],np.uint32));pf[rel(f)]=int(v.sum())
 H=np.concatenate(H);G=np.concatenate(G);o=np.argsort(H,kind="stable");H=H[o];G=G[o];s=np.flatnonzero(np.r_[True,H[1:]!=H[:-1]]) if len(H) else np.zeros(0,np.int64)
 ix.H=H[s];ix.M=np.bitwise_or.reduceat(G,s) if len(s) else G;ix.stats={"files":len(fs),"leaves":st["leaves"],"words":st["words"],"vocab":len(V),"grams_total":int(len(H)),"grams_unique":int(len(ix.H)),"per_file":pf,"build_s":round(time.time()-t0,2),"built":time.strftime("%Y-%m-%dT%H:%M:%S")};return ix
def _savec(ix:Index,p:Path):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.stem+".part.npz");np.savez(tmp,H=ix.H,M=ix.M,R=ix.R,vocab=np.frombuffer("\n".join(ix.V).encode(),np.uint8),meta=np.frombuffer(json.dumps({"n":ix.n,"key":ix.key,"groups":ix.groups,"files":ix.files,"stats":ix.stats}).encode(),np.uint8));os.replace(tmp,p)
 [q.unlink() for q in sorted(p.parent.glob("idx*_*.npz"),key=os.path.getmtime)[:-KEEP] if q!=p]
def _loadc(p:Path)->Index:
 z=np.load(p);m=json.loads(z["meta"].tobytes());vs=z["vocab"].tobytes().decode();ix=Index();ix.V={w:j for j,w in enumerate(vs.split("\n"))} if vs else {}
 ix.H,ix.M,ix.R=z["H"],z["M"],z["R"];ix.n,ix.key,ix.groups,ix.files,ix.stats=m["n"],m["key"],m["groups"],m["files"],{**m["stats"],"cache":str(p)};return ix
def index(n:int=13,files=None,cache=CACHE,rebuild:bool=False)->Index:
 fs=[Path(f).resolve() for f in (eval_sources() if files is None else files)];k=_key(fs,n);cp=Path(cache)/f"idx{n}_{k[:24]}.npz" if cache else None
 ix=_MEMO[k] if k in _MEMO and not rebuild else _loadc(cp) if cp and cp.exists() and not rebuild else _build(fs,n,k)
 cp and (rebuild or not cp.exists()) and _savec(ix,cp);_MEMO[k]=ix;files is None and n==13 and _MEMO.__setitem__("default",ix);return ix
def _ix(ix=None)->Index:return ix or _MEMO.get("default") or index()
def _hits(ws:list,ix:Index)->tuple:
 i=np.fromiter((ix.V.get(w,-1) for w in ws),np.int64,len(ws));h,v=_roll(i,ix.R,ix.n)
 if not len(ix.H) or not len(h):return len(h),np.zeros(0,np.int64),np.zeros(0,np.uint32)
 k=np.searchsorted(ix.H,h);k[k==len(ix.H)]=0;p=np.flatnonzero(v&(ix.H[k]==h));return len(h),p,ix.M[k[p]]
def gram_hashes(text:str,ix=None)->np.ndarray:
 ix=_ix(ix);ws=words(text);h,v=_roll(np.fromiter((ix.V.get(w,-1) for w in ws),np.int64,len(ws)),ix.R,ix.n);return np.unique(h[v])
def dirty(text:str,frac:float=None,ix=None)->list:
 ix=_ix(ix);ws=words(text);L,p,_=_hits(ws,ix);out=list(dict.fromkeys(" ".join(ws[j:j+ix.n]) for j in p))
 return out if frac is None or (L and len(p)/L>=frac) else []
def filter_iter(docs,key=None,frac:float=None,ix=None,stats:collections.Counter=None):
 f=key if callable(key) else (lambda d:d[key]) if key is not None else (lambda d:d);ix=_ix(ix)
 for d in docs:
  h=dirty(f(d),frac,ix);stats is not None and stats.update(["dirty" if h else "clean"])
  if not h:yield d
def _ctx(u:str,g:str,w:int=110)->str:
 m=re.search(r"[\W_]+".join(map(re.escape,g.split())),u,re.I);a=m.start() if m else 0;return re.sub(r"\s+"," ",u[max(0,a-w//3):a+w]).strip()
def _scan_units(units:list,ix:Index,ex:int,cross:bool)->dict:
 W=[words(u) for u in units];sep=[] if cross else ["\x00"];ws=[w for x in W for w in x+sep];cnt=np.fromiter((len(x)+len(sep) for x in W),np.int64,len(W));L,p,m=_hits(ws,ix)
 uid=np.repeat(np.arange(len(W)),cnt)[p] if len(p) else np.zeros(0,np.int64);du=np.unique(uid);ch=np.fromiter(map(len,units),np.int64,len(units));G={};E=[];top=collections.defaultdict(collections.Counter)
 for b,g in enumerate(ix.groups):
  s=((m>>np.uint32(b))&np.uint32(1)).astype(bool)
  if not s.any():continue
  G[g]={"hit_grams":int(s.sum()),"dirty_units":int(np.unique(uid[s]).size)};gp=p[s];gu=uid[s];tg=[" ".join(ws[j:j+ix.n]) for j in gp[:4000]];top[g].update(tg)
  E+=[{"group":g,"gram":t,"ctx":_ctx(units[u],t)} for t,u in list({t:int(u) for t,u in zip(tg,gu[:4000])}.items())[:ex]]
 return {"units":len(units),"dirty_units":int(du.size),"chars":int(ch.sum()),"dirty_chars":int(ch[du].sum()) if du.size else 0,"grams":int(L),"hit_grams":int(len(p)),"by_group":G,"examples":E,"top":{g:dict(c.most_common(25)) for g,c in top.items()}}
def _seg(a:tuple)->dict:
 path,off,ln,ex=a;rd=max(0,off-2)
 with open(path,"rb") as f:f.seek(rd);b=f.read(off-rd+ln+(1<<20))
 B=off-rd+ln;s=0 if off==0 else (lambda x:x+2 if x>=0 else len(b))(b.find(b"\n\n"));e=b.find(b"\n\n",max(B-2,s));e=len(b) if e<0 else e
 return _scan_units(b[s:e].decode("utf-8","replace").split("\n\n") if s<B and s<len(b) else [],_ix(),ex,True)
def _merge(rs:list,ex:int)->dict:
 o={k:sum(r[k] for r in rs) for k in ("units","dirty_units","chars","dirty_chars","grams","hit_grams")};G=collections.defaultdict(collections.Counter);T=collections.defaultdict(collections.Counter);E=collections.defaultdict(list)
 for r in rs:
  [G[g].update(v) for g,v in r["by_group"].items()];[T[g].update(v) for g,v in r["top"].items()];[E[x["group"]].append(x) for x in r["examples"] if len(E[x["group"]])<ex and x["gram"] not in {y["gram"] for y in E[x["group"]]}]
 return {**o,"unit_rate":o["dirty_units"]/max(1,o["units"]),"char_rate":o["dirty_chars"]/max(1,o["chars"]),"gram_rate":o["hit_grams"]/max(1,o["grams"]),"by_group":{g:{**dict(v),"unit_rate":v["dirty_units"]/max(1,o["units"])} for g,v in sorted(G.items())},"examples":{g:E[g] for g in sorted(E)},"top":{g:dict(c.most_common(15)) for g,c in sorted(T.items())}}
def scan_file(path,sample_mb:float=0,strata:int=256,workers:int=6,ex:int=5,seed:int=SEED,seg_mb:float=4)->dict:
 p=Path(path);t0=time.time();ix=_ix();sz=p.stat().st_size
 if p.suffix in (".json",".jsonl"):
  o=load(p);rs=[_scan_units(["\n\n".join(leaves(r)) for r in (o if isinstance(o,list) else [o])],ix,ex,False)];mode="records"
 else:
  sm=int(sample_mb*2**20);R=random.Random(seed);k=max(1,strata);w=sm//k;sl=sz//k
  segs=[(str(p),j*sl+R.randrange(max(1,sl-w)),w,ex) for j in range(k)] if sm and sz>sm else [(str(p),o,int(seg_mb*2**20),ex) for o in range(0,sz,int(seg_mb*2**20))];mode=f"sample {k}x{w/2**20:.2f}MiB stratified" if sm and sz>sm else "full"
  with mp.get_context("fork").Pool(max(1,workers)) as pool:rs=pool.map(_seg,segs,chunksize=1)
 return {"path":rel(p),"size":sz,"mode":mode,**_merge(rs,ex),"secs":round(time.time()-t0,1)}
def rss_mb()->dict:return {"self_max_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),"children_max_mb":round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,1)}
def main(argv=None):
 ap=argparse.ArgumentParser(prog="python -m amni.compute.decontam");sp=ap.add_subparsers(dest="cmd",required=True);b=sp.add_parser("build");b.add_argument("--rebuild",action="store_true");sp.add_parser("sources")
 s=sp.add_parser("scan");s.add_argument("paths",nargs="+");s.add_argument("--sample-mb",type=float,default=0);s.add_argument("--strata",type=int,default=256);s.add_argument("--workers",type=int,default=6);s.add_argument("--examples",type=int,default=5);s.add_argument("--out");a=ap.parse_args(argv)
 if a.cmd=="sources":[print(group(f),rel(f)) for f in eval_sources()];return
 t0=time.time();ix=index(rebuild=getattr(a,"rebuild",False));print(json.dumps({"index":repr(ix),"load_or_build_s":round(time.time()-t0,2),"groups":ix.groups,**{k:v for k,v in ix.stats.items() if k!="per_file"},**rss_mb()}),flush=True)
 if a.cmd=="build":return
 fs=[f for q in a.paths for f in ([q] if Path(q).is_file() else sorted(x for x in Path(q).iterdir() if x.is_file() and x.suffix in (".bin",".json",".jsonl",".txt")))];out=[]
 for f in fs:
  r=scan_file(f,a.sample_mb,a.strata,a.workers,a.examples);out.append(r)
  print(f"{r['path']}: {r['mode']}, units {r['units']}, dirty {r['dirty_units']} ({100*r['unit_rate']:.3f}%), chars {100*r['char_rate']:.3f}%, grams {r['hit_grams']}/{r['grams']} ({1e6*r['gram_rate']:.1f} ppm), {r['secs']}s",flush=True)
  [print(f"  {g}: hit_grams {v['hit_grams']}, dirty_units {v['dirty_units']} ({100*v['unit_rate']:.3f}%)") for g,v in r["by_group"].items()]
 a.out and json.dump({"index":{"key":ix.key,"files":len(ix.files),"grams":int(len(ix.H))},"results":out,**rss_mb()},open(a.out,"w"),indent=1);print(json.dumps(rss_mb()))
if __name__=="__main__":main()
