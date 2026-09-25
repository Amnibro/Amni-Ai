import argparse,ast,json,re,sys,textwrap,time,warnings,zlib
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from amni.compute import tile_synth as TS,code_rank as CR,decontam,eval_lock
from x5_mine import blocks,pure,norm,bench_exact,CH,W,E
from x0_code_eval import DATA,run_sandbox
TRAIN=ROOT/"data"/"pycode2"/"train.bin";RAW=DATA/"x11_mined_raw.json";OUT=DATA/"x11_mined_tiles.jsonl";STATS=DATA/"x11_mine_stats.json"
KIND={"i":"int","s":"str","N":"list[int]","S":"list[str]"}
def _tpl(t):
 ps=[a.arg for a in t.args.args];e=copy_expr(t.body[-1].value);mp={p:f"_A{k}" for k,p in enumerate(ps)};cv={};ck=["."]*len(ps)
 for n in ast.walk(e):(setattr(n,"id",mp[n.id]) if isinstance(n,ast.Name) and n.id in mp else None)
 class Rw(ast.NodeTransformer):
  def visit_Constant(s,n):
   v=n.value;ok=type(v) is str and 1<=len(v)<=3 or type(v) is int and 2<=abs(v)<=100
   if not ok:return n
   (cv.__setitem__((type(v),v),f"_A{len(ps)+len(cv)}"),ck.append("s" if type(v) is str else "i")) if (type(v),v) not in cv else None;return ast.copy_location(ast.Name(id=cv[(type(v),v)],ctx=ast.Load()),n)
 e=Rw().visit(e) if not any(isinstance(n,ast.JoinedStr) for n in ast.walk(e)) else ast.Constant(value="\x00");s=ast.unparse(e);bad=[n for n in ast.walk(e) if isinstance(n,ast.Constant) and not (n.value in ("",None,True,False) or type(n.value) is int and abs(n.value)<2 or type(n.value) is float and n.value in (0.0,0.5,1.0,2.0))]
 return (re.sub(r"_A(\d)",r"{\1}",s),"".join(ck)) if len(ck)<=3 and not bad and all(f"_A{k}" in s for k in range(len(ps))) else (None,"")
def copy_expr(e):return ast.parse(ast.unparse(e),mode="eval").body
def mine_chunk(a):
 warnings.simplefilter("ignore");st=Counter();out={};b=min(a+CH,TRAIN.stat().st_size);lo=max(0,a-W)
 with open(TRAIN,"rb") as fh:fh.seek(lo);buf=fh.read(b-lo+E)
 for m,blk in blocks(buf,a-lo,b-lo):
  st["defs"]+=1
  if blk is None or blk.count(b"\n")>12:st["skip_block"]+=1;continue
  try:src=textwrap.dedent(blk.decode("utf-8"));t=ast.parse(src).body[0];assert isinstance(t,ast.FunctionDef)
  except Exception:st["parse_fail"]+=1;continue
  bd=t.body[1:] if ast.get_docstring(t) is not None else t.body
  if len(bd)!=1 or not isinstance(bd[0],ast.Return) or bd[0].value is None:st["not_single_return"]+=1;continue
  why,_=pure(t)
  if why:st["impure"]+=1;continue
  tp,ck=_tpl(t)
  if not tp or len(tp)>120 or "{0}" not in tp or t.name in tp:st["tpl_fail"]+=1;continue
  try:TS.safe_check(TS._rd(TS._parts(tp),["a","b","c"]),"eval")
  except Exception:st["unsafe"]+=1;continue
  st["kept"]+=1;r=out.setdefault(tp,{"n":0,"ex":src[:600],"off":lo+m.start(),"nh":norm(t),"ck":ck,"words":Counter()});r["n"]+=1;r["words"].update(w for w in CR.words(t.name+" "+(ast.get_docstring(t) or "")[:300]) if len(w)>2 and w not in TS.STOP)
 return st,{k:{**v,"words":dict(v["words"].most_common(8))} for k,v in out.items()}
def cmd_mine(a):
 eval_lock.assert_trainable([str(TRAIN)],"tile mining");t0=time.time();sz=TRAIN.stat().st_size;offs=list(range(0,sz,CH))[:a.limit or None];st=Counter();agg={}
 with get_context("spawn").Pool(a.jobs) as pool:
  for k,(s,o) in enumerate(pool.imap_unordered(mine_chunk,offs)):
   st.update(s)
   for tp,v in o.items():
    r=agg.setdefault(tp,{"n":0,"ex":v["ex"],"off":v["off"],"nh":v["nh"],"ck":v["ck"],"words":Counter()});r["n"]+=v["n"];r["words"].update(v["words"])
   (print(k,len(offs),dict(st),len(agg),f"{time.time()-t0:.0f}s",flush=True) if k%40==0 else None)
 keep={tp:{**v,"words":dict(v["words"].most_common(10))} for tp,v in agg.items() if v["n"]>=a.minf};RAW.write_text(json.dumps({"stats":dict(st),"n_templates":len(agg),"kept":keep}));print(dict(st),len(agg),len(keep),f"{time.time()-t0:.0f}s")
FUZZ=r'''import json,random,signal,copy,math,itertools,functools,collections,re,string
def _al(*_):raise TimeoutError
signal.signal(signal.SIGALRM,_al);g=random.Random(17);D=json.loads(open("q.json").read());out={}
def gen(k):
 return g.choice([0,1,2,3,-1,5,10,g.randint(-50,50),g.randint(0,500)]) if k=="i" else "".join(g.choice("abcdeXYZ 019,.aeio") for _ in range(g.randint(0,12))) if k=="s" else [gen("i") for _ in range(g.randint(0,7))] if k=="N" else [gen("s") for _ in range(g.randint(0,6))]
for tp,combos in D.items():
 try:f=eval("lambda _a0=None,_a1=None,_a2=None: "+tp.replace("{0}","_a0").replace("{1}","_a1").replace("{2}","_a2"))
 except Exception:continue
 ok=[]
 for c in combos:
  n=good=bad=0;outs=set()
  for i in range(24):
   a=[gen(k) for k in c];a0=copy.deepcopy(a)
   try:
    signal.setitimer(signal.ITIMER_REAL,0.2);r1=f(*copy.deepcopy(a));r2=f(*copy.deepcopy(a));signal.setitimer(signal.ITIMER_REAL,0)
    t=type(r1);good+=repr(r1)==repr(r2) and a==a0 and t in (int,float,bool,str,list,tuple,dict,set) and len(repr(r1))<5000;bad+=repr(r1)!=repr(r2) or a!=a0;outs.add(repr(r1)[:200])
   except TimeoutError:bad+=1
   except BaseException:signal.setitimer(signal.ITIMER_REAL,0)
   n+=1
  (ok.append(c) if good>=0.6*n and not bad and len(outs)>=3 else None)
 out[tp]=ok
print("X11FUZZ"+json.dumps(out))
'''
def cmd_select(a):
 import itertools as it
 raw=json.loads(RAW.read_text())["kept"];ix=decontam.index();ex=bench_exact();have={re.sub(r"\s+","",TS._rd(TS._parts(tp),["{0}","{1}","{2}"])) for _,_,tp,*_ in TS.TILE_DEFS+TS.X11_TILES};_,o2f=CR.opx();fw=json.loads(Path(a.opw).read_text()) if a.opw else {};st=Counter();cand=[]
 for tp,v in raw.items():
  ar=len(set(re.findall(r"\{(\d)\}",tp)))
  if re.sub(r"\s+","",tp) in have:st["existing_tile"]+=1;continue
  if v["nh"] in ex:st["bench_exact"]+=1;continue
  if decontam.dirty(v["ex"],ix=ix):st["decontam_13gram"]+=1;continue
  try:F=set(TS.feats(TS._rd(TS._parts(tp),["a","b","c"])))
  except Exception:st["feat_fail"]+=1;continue
  ops=[o for o,fs in o2f.items() if fs&F];sc=v["n"]*(1+sum(fw.get(o,0) for o in ops))
  if not ops or ar>3 or len(TS._core(F))<2 or len(v["ck"])!=ar or re.search(r"\b(re|math)\.",tp):st["no_op_or_trivial"]+=1;continue
  cand.append((sc,tp,v,ops,ar))
 cand=sorted(cand,key=lambda x:-x[0])[:a.top*3];q={tp:["".join(c) for c in it.product(*[("isNS" if k=="." else k) for k in v["ck"]])] for _,tp,v,_,ar in cand};wd=DATA/"tmp";wd.mkdir(parents=True,exist_ok=True)
 r=run_sandbox("import json\nopen('q.json','w').write("+repr(json.dumps(q))+")\n"+FUZZ,timeout=900.0,cpu=900,mem=2<<30);m=re.search(r"X11FUZZ(\{.*\})",r.get("stdout") or "");ok=json.loads(m.group(1)) if m else {}
 out=[]
 for sc,tp,v,ops,ar in cand:
  cs=ok.get(tp) or []
  if not cs:st["fuzz_fail"]+=1;continue
  if len(out)>=a.top:break
  out.append({"name":"m_"+format(zlib.crc32(tp.encode()),"08x"),"prims":"|".join(cs[:6]),"tpl":tp,"freq":v["n"],"ops":ops,"kw":" ".join(sorted(v["words"],key=lambda w:-v["words"][w])[:8]),"example":v["ex"],"source":f"data/pycode2/train.bin@{v['off']}","licence":"permissive-only corpus (apache-2.0/bsd/mit/isc/cc0/unlicense/own)"})
 OUT.write_text("".join(json.dumps(o)+"\n" for o in out));st["selected"]=len(out);STATS.write_text(json.dumps({"select":dict(st),"fuzz_ok":r.get("ok"),"top":a.top,"decontam_index":ix.key},indent=1));print(dict(st))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("cmd",choices=["mine","select"]);ap.add_argument("--jobs",type=int,default=4);ap.add_argument("--limit",type=int,default=0);ap.add_argument("--minf",type=int,default=3);ap.add_argument("--top",type=int,default=60);ap.add_argument("--opw",default="");a=ap.parse_args()
 cmd_mine(a) if a.cmd=="mine" else cmd_select(a)
if __name__=="__main__":main()
