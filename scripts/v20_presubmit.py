import argparse,ast,gzip,importlib,json,os,re,shlex,shutil,socket,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
VPY=ROOT/".venv-linux"/"bin"/"python";PY=str(VPY) if VPY.exists() else sys.executable
OUTD=ROOT/"docs"/"tasks"/"v20"/"presubmit";STATUSD=ROOT/"docs"/"tasks"/"v20"/"status"
CODE={".py",".rs",".sh",".js",".mjs",".ts",".go"};MAXB=4<<20;SKIPD={"target","__pycache__",".git","node_modules",".pytest_cache",".decontam_cache",".venv",".venv-linux"}
JAILS={"amni/serve/self_debug.py","scripts/x0_code_eval.py","amni/serve/code_safety.py"};FILTERS={"scripts/neutral_filter.py","amni/compute/decontam.py"};SHARED={"amni/compute/decontam.py","amni/compute/eval_lock.py"};SELF={"scripts/v20_presubmit.py","tests/test_v20_presubmit.py"}
BOX=re.compile(r"(?<![\w.~-])/workspace(?![\w-])|(?<![\w.-])/home/box(?![\w-])|\b[A-Za-z]:(?:\\\\|\\|/)+Users\b",re.I)
HOMEU=re.compile(r"(?<![\w.-])/home/(?!box(?![\w-]))[a-z_][\w.-]*");ENVD=re.compile(r"environ\.get|getenv|env::var|\$\{\w+:-|setdefault|unwrap_or|environ\[")
TAINT=[(k,re.compile(v,re.I)) for k,v in (("big2/big3",r"(?<![a-z0-9])big[23](?![a-z0-9])"),("chat1-chat4e",r"data/chat(?:[1-3]|4[a-e]?)(?![a-z0-9])|(?<![a-z0-9])chat(?:[1-3]|4[a-e]?)[/_.-][\w./-]*?(?:train|hold|eval|sft|mix|corpus|pack)"),("grokipedia",r"(?<![a-z0-9])grokipedia(?![a-z0-9])"),("openhermes",r"(?<![a-z0-9])open[-_ ]?hermes(?![a-z0-9])"),("yahoo",r"(?<![a-z0-9])yahoo(?![a-z0-9])"))]
DENY=re.compile(r"(?i)\b(no|not|never|zero|ban\w*|block\w*|deny\w*|refus\w*|purg\w*|exclud\w*|forbid\w*|taint\w*|reject\w*|drop\w*|delet\w*|remov\w*|without|absent|avoid\w*)\b")
INTERP=re.compile(r"^(python[\d.]*|pypy[\d.]*|sh|bash|zsh|dash|node|nodejs|deno|bun|go|rustc|ruby|perl|php|lua)$");SCRIPTI=re.compile(r"^(python[\d.]*|pypy[\d.]*|sh|bash|zsh|dash|node|nodejs|deno|bun|ruby|perl|php|lua)$")
WRAP={"timeout","nice","env","prlimit","setsid","stdbuf","ionice","taskset","nohup","sudo","doas"};EXECF={"-c","-e","--eval","-p","--print","-"};PYC=re.compile(r"python[\d.]*\s+-c\b");SPF={"run","call","check_call","check_output","Popen","getoutput","getstatusoutput"}
JAILDEF={"_bwrap_cmd","run_sandbox","jail_run","_bwrap_run","run_in_sandbox"};SCRIPTARG=re.compile(r"\.(py|js|mjs|sh)\b|ROOT|__file__|_H0|HERE|BASE|SCRIPT")
EVALDIR=re.compile(r"(?i)(^|/)(concepts/eval|webeval|code_eval|hold_eval(\.jsonl)?|gsm8k|svamp|humaneval|mbpp|registry\.json)(/|$|\.|_)|^evals?$")
DCL=re.compile(r"(?i)contam|ngram|n_gram|(^|_)grams?\d*($|_)|disjoint|evalset|eval_files|eval_set")
LOCKRE=re.compile(r"(?i)related\.jsonl|humaneval|mbpp|hold_eval|locked_test|(^|[/_-])test[\w-]*\.(jsonl?|parquet|csv|tsv)$|_test[\w-]*\.jsonl?$")
TUNE=re.compile(r"(?i)(^|_)(fit|tune|tuning|tuned|grid|gridsearch|sweep|hill|hillclimb|optimi[sz]e|calibrat\w*|search_params|hparams?|hyperparams?|best_(params|thr|threshold|alpha|beta|cfg|k))($|_|\d)")
NONTRAIN=re.compile(r"(?i)(^|[/_.-])(eval|evals|hold|holdout|test|tests|dev|gate|val|valid|locked|burned)([/_.-]|$)")
CORS=re.compile(r"CorsLayer::(very_)?permissive\(\)|allow_origin\(\s*(Any|AllowOrigin::any\(\)|\"\*\")|allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]|\bCORS\(\s*\w+\s*\)|Access-Control-Allow-Origin['\"]?\s*[:,]\s*['\"]\*")
ROUTE=re.compile(r"\.route\(\s*\"|@\w+\.(get|post|put|delete|patch|route|websocket|api_route)\(|add_url_rule\(|add_api_route\(|router\.add_(get|post|put|route)\(|\bweb\.(get|post|put|delete)\(|def do_(GET|POST|PUT|DELETE)\(")
AUTH=re.compile(r"(?i)\bauth(?!or)|authorization|bearer|api[_-]?key|x-[\w-]*token|(check|require|verify)_token|compare_digest|ValidateRequestHeaderLayer")
TB=re.compile(r"\.(tgz|tar|tar\.gz|tar\.xz|tar\.bz2|tbz2|txz)$",re.I);BAK=re.compile(r"\.bak(\.|$)|\.orig$|~$|\.rej$");SEV={"PASS":0,"WARN":1,"FAIL":2}
def rel(p)->str:
 p=Path(p).resolve();return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def fnd(sev,f,line,rule,ev,ctx="")->dict:return {"sev":sev,"file":rel(f) if f else "","line":line,"rule":rule,"evidence":str(ev)[:240],**({"context":ctx.strip()[:240]} if ctx else {})}
def dedup(fs)->list:return list({(f["sev"],f["file"],f["line"],f["rule"],f["evidence"]):f for f in fs}.values())
def worst(fs)->str:return max((f["sev"] for f in fs),key=SEV.get,default="PASS")
def env4()->dict:return {**os.environ,"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","MKL_NUM_THREADS":"4","RAYON_NUM_THREADS":"4","CARGO_BUILD_JOBS":"4","PYO3_PYTHON":PY,"PYTHONDONTWRITEBYTECODE":"1"}
def run(cmd,cwd=ROOT,timeout=1800,env=None)->dict:
 t0=time.time()
 try:r=subprocess.run(cmd,cwd=str(cwd),env=env or env4(),capture_output=True,text=True,timeout=timeout,preexec_fn=lambda:os.nice(10));return {"rc":r.returncode,"out":r.stdout,"err":r.stderr,"secs":round(time.time()-t0,1)}
 except subprocess.TimeoutExpired as e:return {"rc":None,"out":e.stdout if isinstance(e.stdout,str) else "","err":f"TIMEOUT after {timeout}s","secs":round(time.time()-t0,1)}
 except OSError as e:return {"rc":None,"out":"","err":str(e),"secs":0}
def expand(paths)->tuple:
 fs,miss,dirs=set(),[],[]
 for q in paths:
  p=Path(q) if Path(q).exists() else ROOT/q
  miss.append(q) if not p.exists() else dirs.append(p.resolve()) if p.is_dir() else fs.add(p.resolve())
 fs|={f.resolve() for d in dirs for f in d.rglob("*") if f.is_file() and not SKIPD&set(f.relative_to(d).parts)};return sorted(fs),miss,dirs
def text(p):
 try:b=p.read_bytes() if p.suffix in CODE or p.stat().st_size<=MAXB else None
 except OSError:b=None
 return None if b is None or b"\0" in b[:8192] else b.decode("utf-8","replace")
def lines(T):return [(p,i,l) for p,t in T.items() if t is not None for i,l in enumerate(t.splitlines(),1)]
def c_inputs(miss,fs)->list:return [fnd("FAIL",None,0,"declared file missing on Antman",q) for q in miss]+([fnd("FAIL",None,0,"no files","--files expanded to nothing")] if not fs else [])
def _doc(p,src)->set:
 t=_parse(src) if p.suffix==".py" and src else None;return {l for n in ast.walk(t) if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str) for l in range(n.body[0].lineno,n.body[0].end_lineno+1)} if t else set()
def _cmt(p,l,i,pos,doc)->bool:
 pre=l[:pos];return i in doc or (p.suffix in (".py",".sh") and bool(re.search(r"(^|\s)#",pre)) and pre.split("#")[0].count('"')%2==0 and pre.split("#")[0].count("'")%2==0) or (p.suffix in (".rs",".js",".mjs",".ts",".go") and "//" in pre and pre.split("//")[0].count('"')%2==0)
def c_paths(T)->list:
 D={p:_doc(p,t) for p,t in T.items()};hit=[(p,i,l,m,"box path") for p,i,l in lines(T) if rel(p) not in SELF for m in BOX.finditer(l)]+[(p,i,l,m,"absolute /home/<user> path outside an env default") for p,i,l in lines(T) if rel(p) not in SELF and not ENVD.search(l) for m in HOMEU.finditer(l)]
 return [fnd("WARN",p,i,r+" (comment/docstring only)",m.group(0),l) if _cmt(p,l,i,m.start(),D[p]) else fnd("FAIL",p,i,r,m.group(0),l) for p,i,l,m,r in hit]
def c_taint(T)->list:return [fnd("WARN" if DENY.search(l) else "FAIL",p,i,f"tainted source reference ({k})"+(" in a deny/guard context" if DENY.search(l) else ""),m.group(0),l) for p,i,l in lines(T) if rel(p) not in SELF for k,rx in TAINT for m in rx.finditer(l) if not (rel(p) in FILTERS and "#" in l[:m.start()])]
def _assigns(tree)->dict:
 A={}
 for n in ast.walk(tree):[A.setdefault(t.id,[]).append((n.lineno,v)) for t,v in ([(t,n.value) for t in n.targets] if isinstance(n,ast.Assign) else [(n.target,n.value)] if isinstance(n,ast.AnnAssign) and n.value is not None else []) if isinstance(t,ast.Name)]
 return A
def _last(A,k,ln):c=[v for l,v in A.get(k,[]) if l<=ln] or [v for l,v in A.get(k,[])];return c[-1] if c else None
def _atom(e,A,ln,d=0)->tuple:
 v=_last(A,e.id,ln) if isinstance(e,ast.Name) and d<4 else None
 if isinstance(e,ast.Constant):return ("const",str(e.value))
 if v is not None:return _atom(v,A,ln,d+1)
 s=ast.unparse(e);return ("bwrap",s) if "bwrap" in s.lower() else ("py",s) if re.search(r"(?i)sys\.executable|python|\bPYV?\b",s) else ("dyn",s)
def _atoms(n,A,ln,d=0):
 if d>4:return None
 if isinstance(n,(ast.List,ast.Tuple)):return [a for e in n.elts for a in (((_atoms(e.value,A,ln,d+1) or [("dyn",ast.unparse(e))]) if isinstance(e,ast.Starred) else [_atom(e,A,ln)]))]
 if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):l=_atoms(n.left,A,ln,d+1);return None if l is None else l+(_atoms(n.right,A,ln,d+1) or [("dyn","+")])
 if isinstance(n,ast.Name):v=_last(A,n.id,ln);return None if v is None else _atoms(v,A,ln,d+1)
 if isinstance(n,ast.Call) and re.search(r"(?i)bwrap|jail|sandbox",ast.unparse(n.func)):return [("bwrap",ast.unparse(n.func))]
 return [("const",x) for x in n.value.split()] if isinstance(n,ast.Constant) and isinstance(n.value,str) else None
def classify(at,stdin=False):
 i=0
 while i<len(at) and at[i][0]=="const" and Path(at[i][1]).name in WRAP:
  i+=1
  while i<len(at) and at[i][0]=="const" and (at[i][1].startswith("-") or "=" in at[i][1] or re.fullmatch(r"[\d.]+[smhd]?",at[i][1])):i+=1
 if i>=len(at):return None
 k,v=at[i];prog=Path(v).name if k=="const" else "";rest=at[i+1:];lead=[x for _,x in rest[:next((j for j,(kk,x) in enumerate(rest) if kk!="const" or not x.startswith("-")),len(rest))]]
 hit=next((x for x in lead if x in EXECF),None) or ("run" if prog in ("go","deno","bun") and rest[:1]==[("const","run")] else None);interp=k=="py" or bool(INTERP.match(prog));sint=k=="py" or bool(SCRIPTI.match(prog))
 return None if k=="bwrap" or prog=="bwrap" else ("FAIL",f"host interpreter `{prog or v}` with `{hit}` runs code outside jail_run/run_sandbox") if interp and hit else ("FAIL",f"host interpreter `{prog or v}` reads code from stdin outside the jail") if sint and stdin and not rest else ("WARN",f"host interpreter `{prog or v}` runs a non-constant script `{rest[0][1]}`") if sint and rest and rest[0][0]!="const" and not SCRIPTARG.search(rest[0][1]) else ("WARN",f"subprocess with a non-constant program `{v}`") if k=="dyn" else None
def py_exec(p,src)->list:
 try:t=ast.parse(src)
 except SyntaxError as e:return [fnd("FAIL",p,e.lineno or 0,"python does not parse",e.msg)]
 A=_assigns(t);mods={"subprocess"}|{a.asname or a.name for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names if a.name=="subprocess"};fns={a.asname or a.name:a.name for n in ast.walk(t) if isinstance(n,ast.ImportFrom) and n.module=="subprocess" for a in n.names};out=[]
 for n in ast.walk(t):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in JAILDEF:out.append(fnd("WARN",p,n.lineno,"private jail implementation",f"def {n.name}: use amni.serve.self_debug.jail_run or scripts/x0_code_eval.run_sandbox"))
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and PYC.search(n.value):out.append(fnd("FAIL",p,n.lineno,"`python -c` command string",n.value[:120]))
  if not isinstance(n,ast.Call):continue
  d=ast.unparse(n.func);kw={k.arg:k.value for k in n.keywords if k.arg};a0=n.args[0] if n.args else kw.get("args") or kw.get("cmd");nc=a0 is not None and not isinstance(a0,ast.Constant);ln=n.lineno
  sub=(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in mods and n.func.attr in SPF) or (isinstance(n.func,ast.Name) and fns.get(n.func.id) in SPF);what=n.func.attr if isinstance(n.func,ast.Attribute) else fns.get(d,d)
  sh=isinstance(kw.get("shell"),ast.Constant) and kw["shell"].value is True;stdin=kw.get("input") is not None and not isinstance(kw.get("input"),ast.Constant)
  out+=[fnd("FAIL",p,ln,"shell=True",d)] if sub and sh else []
  out+=[fnd("FAIL",p,ln,f"{d} runs a non-constant shell string",ast.unparse(a0)[:120])] if (sub and what in ("getoutput","getstatusoutput") or d in ("os.system","os.popen","asyncio.create_subprocess_shell")) and nc else []
  out+=[fnd("WARN",p,ln,f"{d} spawns a process",ast.unparse(n)[:120])] if re.fullmatch(r"os\.(exec|spawn)\w*",d) else []
  out+=[fnd("FAIL",p,ln,f"builtin {d}() on non-constant input",ast.unparse(a0)[:120])] if isinstance(n.func,ast.Name) and d in ("exec","eval") and nc else []
  at=_atoms(a0,A,ln) if (sub and what not in ("getoutput","getstatusoutput") and not sh or d=="asyncio.create_subprocess_exec") and a0 is not None else None
  at=_atoms(ast.List(elts=list(n.args)),A,ln) if d=="asyncio.create_subprocess_exec" else at;c=classify(at,stdin) if at else None
  out+=[fnd(c[0],p,ln,c[1],ast.unparse(n)[:200])] if c else [fnd("WARN",p,ln,"subprocess argv not statically resolvable",ast.unparse(n)[:200])] if sub and not sh and at is None and what not in ("getoutput","getstatusoutput") and a0 is not None else []
 return out
def rs_exec(p,src)->list:
 out=[]
 for m in re.finditer(r"Command::new\(\s*([^;]*?)\s*\)(?=[\s.;])",src):
  a=m.group(1);lit=re.fullmatch(r"\"([^\"]*)\"",a);prog=Path(lit.group(1)).name if lit else "";st=src[m.end():src.find(";",m.end()) if src.find(";",m.end())>0 else len(src)];ln=src.count("\n",0,m.start())+1;ls=[Path(x).name for x in re.findall(r"\"([^\"]*)\"",st)]
  sev,msg=(("FAIL",f"Command::new(\"{prog}\") runs {prog} on the host, not inside bwrap"+(" with -c" if "-c" in ls else "")) if INTERP.match(prog) else ("FAIL",f"Command::new(\"{prog}\") wraps {next(x for x in ls if INTERP.match(x))} on the host, not inside bwrap") if prog in WRAP and any(INTERP.match(x) for x in ls) else ("WARN",f"Command::new({a}) with a non-constant program") if not lit and "bwrap" not in a.lower() else (None,None))
  out+=[fnd(sev,p,ln,msg,src[m.start():m.start()+160].split("\n")[0])] if sev else []
 return out
def c_exec(T)->list:return [f for p,t in T.items() if t is not None and rel(p) not in JAILS|SELF for f in (py_exec(p,t) if p.suffix==".py" else rs_exec(p,t) if p.suffix==".rs" else [fnd("WARN",p,i,"`python -c` in a shell script",l.strip()) for i,l in enumerate(t.splitlines(),1) if PYC.search(l)] if p.suffix==".sh" else [])]
def istest(p)->bool:return p.suffix==".py" and (p.name.startswith("test_") or p.name.endswith("_test.py"))
def _strs(t):return [(n.lineno,n.value) for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
def _idents(t,calls=True):return [(n.lineno,x) for n in ast.walk(t) for x in ([n.name] if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) else [n.id] if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) else [n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id] if calls and isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name)) else [])]
def _imports_decontam(t,src)->bool:return "amni.compute.decontam" in src or any(isinstance(n,ast.ImportFrom) and n.module=="amni.compute" and any(a.name=="decontam" for a in n.names) for n in ast.walk(t))
def _parse(src):
 try:return ast.parse(src)
 except SyntaxError:return None
def c_decontam_static(T)->list:
 out=[]
 for p,src in T.items():
  t=_parse(src) if src is not None and p.suffix==".py" and not istest(p) and rel(p) not in SHARED|SELF else None
  if t is None:continue
  ev=[(l,s) for l,s in _strs(t) if EVALDIR.search(s)];dl=[(l,x) for l,x in _idents(t,False) if DCL.search(x) and not x.startswith("test")]
  out+=[fnd("FAIL",p,dl[0][0],"private eval list / decontamination (does not import amni.compute.decontam)",f"decontam logic {sorted({x for _,x in dl})[:6]} over eval paths {sorted({s for _,s in ev})[:6]}")] if ev and dl and not _imports_decontam(t,src) else []
 return out
def _materialize(p,tmp):
 if p.suffix in (".jsonl",".json",".txt",".bin"):return p
 if p.suffix==".gz":o=tmp/f"{len(list(tmp.iterdir()))}_{p.stem}";o.write_bytes(gzip.open(p).read());return o
 if p.suffix!=".parquet":return None
 pq=importlib.import_module("pyarrow.parquet");tb=pq.read_table(p);col=next((c for c in ("text","content","body","markdown","abstract") if c in tb.column_names),None);o=tmp/f"{len(list(tmp.iterdir()))}_{p.stem}.jsonl"
 col and o.write_text("".join(json.dumps({"text":x})+"\n" for x in tb.column(col).to_pylist() if x),encoding="utf-8");return o if col else None
def c_decontam_data(data,sample_mb,timeout)->tuple:
 fs,miss,_=expand(data);out=[fnd("FAIL",None,0,"--data path missing",q) for q in miss];notes=[]
 if not fs:return out,notes
 with tempfile.TemporaryDirectory(prefix="presub_dc_") as td:
  tmp=Path(td);mat=[(p,_materialize(p,tmp)) for p in fs];out+=[fnd("WARN",p,0,"data file not scannable (need .jsonl/.json/.txt/.bin/.gz/.parquet with a text column)",rel(p)) for p,m in mat if m is None];mat=[(p,m) for p,m in mat if m is not None]
  r=run([PY,"-m","amni.compute.decontam","scan",*[str(m) for _,m in mat],"--workers","4","--sample-mb",str(sample_mb),"--out",str(tmp/"scan.json")],timeout=timeout) if mat else None
  res=json.loads((tmp/"scan.json").read_text())["results"] if r and (tmp/"scan.json").exists() else None
  out+=[fnd("FAIL",None,0,"decontam scan failed",(r or {}).get("err","")[-300:])] if mat and res is None else []
  for (p,_),x in zip(mat,res or []):
   tr=not NONTRAIN.search(rel(p));notes.append({"file":rel(p),"units":x["units"],"dirty_units":x["dirty_units"],"hit_grams":x["hit_grams"],"grams":x["grams"],"training_split":tr,"by_group":x["by_group"]})
   out+=[fnd("FAIL" if tr else "WARN",p,0,"eval 13-gram hit in a training split" if tr else "eval 13-gram hit in a non-training split",f"{x['dirty_units']}/{x['units']} dirty units; groups {sorted(x['by_group'])}; e.g. {next(iter(e['gram'] for v in x['examples'].values() for e in v),'')}")] if x["dirty_units"] else []
 return out,notes
def locked_names()->set:
 ms=[p for pat in ("data/**/*manifest*.json","docs/**/*manifest*.json","data/**/*lock*.json","docs/**/*lock*.json","data/concepts/eval/REGISTRY.json") for p in ROOT.glob(pat) if p.is_file() and p.stat().st_size<(4<<20)];nm=set()
 for p in ms:
  try:d=json.loads(p.read_text())
  except (ValueError,OSError):continue
  nm|={Path(k).name for k in (d.get("splits") or {}) if isinstance(d,dict) and isinstance(d.get("splits"),dict)} if isinstance(d,dict) and "splits" in d else set()
  nm|={Path(e.get("path","")).name for e in d.get("entries",[]) if isinstance(e,dict) and e.get("role") in ("locked","hold","burned")} if isinstance(d,dict) and isinstance(d.get("entries"),list) else set()
 return {n for n in nm if len(n)>=8 and not re.search(r"(?i)train|(^|_)dev(\.|_)|REGISTRY",n)}
def c_tuning(T)->list:
 L=locked_names();out=[]
 for p,src in T.items():
  t=_parse(src) if src is not None and p.suffix==".py" and not istest(p) and rel(p) not in SELF else None
  if t is None:continue
  ev=[(l,s) for l,s in _strs(t) if LOCKRE.search(s) or any(n in s for n in L)];tu=[(l,x) for l,x in _idents(t) if TUNE.search(x)]
  out+=[fnd("WARN",p,tu[0][0],"possible tuning on a locked eval split",f"reads {sorted({s[:60] for _,s in ev})[:4]} (line {ev[0][0]}) and tunes/fits via {sorted({x for _,x in tu})[:5]}")] if ev and tu else []
 return out
def taut_py(src)->tuple:
 t=_parse(src)
 if t is None:return 0,0,[]
 rd=lambda n:isinstance(n,ast.Call) and (ast.unparse(n.func).endswith("getsource") or (isinstance(n.func,ast.Attribute) and n.func.attr in ("read_text","read","read_bytes") and re.search(r"__file__\)?$|\.(py|rs)['\"]",ast.unparse(n.func.value))))
 srcn={x.id for n in ast.walk(t) if isinstance(n,ast.Assign) and any(rd(c) for c in ast.walk(n.value)) for x in n.targets if isinstance(x,ast.Name)}
 const=lambda e:not any(isinstance(x,(ast.Name,ast.Call,ast.Attribute,ast.Subscript,ast.Starred,ast.NamedExpr)) for x in ast.walk(e))
 srct=lambda e:isinstance(e,ast.Compare) and all(isinstance(o,(ast.In,ast.NotIn)) for o in e.ops) and isinstance(e.left,ast.Constant) and all(const(c) or rd(c) or (isinstance(c,ast.Name) and c.id in srcn) for c in e.comparators)
 triv=lambda e:const(e) or srct(e) or (isinstance(e,ast.BoolOp) and all(triv(v) for v in e.values))
 fns=[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith("test")];res=[]
 for f in fns:
  ex=[n.test for n in ast.walk(f) if isinstance(n,ast.Assert)]+[a for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith("assert") for a in n.args[:2]];res.append((f.name,len(ex),sum(map(triv,ex))))
 return sum(r[1] for r in res),sum(r[2] for r in res),[r[0] for r in res if r[1] and r[1]==r[2]]
def taut_rs(src)->tuple:
 ax=re.findall(r"assert(?:_eq|_ne)?!\((.*?)\);",src,re.S);inc=bool(re.search(r"include_str!\([^)]*\.rs\"",src))
 tv=[a for a in ax if not re.search(r"[A-Za-z_]\w*",re.sub(r"\"(?:[^\"\\]|\\.)*\"|\b(true|false)\b|\b\d[\w.]*","",a)) or (inc and "contains" in a)];return len(ax),len(tv)
def c_tests(fs,T,crates,timeout)->tuple:
 out=[];notes={}
 for c in crates:
  for kind,cmd in (("build",["cargo","build","--release","-j","4","--locked","-p",c]),("test",["cargo","test","-j","4","--locked","-p",c])):
   r=run(cmd,ROOT/"crates",timeout);tr=[tuple(map(int,x)) for x in re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed",r["out"])];np,nf=sum(x[0] for x in tr),sum(x[1] for x in tr);notes[f"cargo {kind} -p {c}"]={"cmd":shlex.join(cmd),"rc":r["rc"],"secs":r["secs"],**({"passed":np,"failed":nf} if kind=="test" else {})}
   out+=[fnd("FAIL",None,0,f"cargo {kind} -p {c} failed",f"rc={r['rc']}: "+" | ".join([l for l in (r["err"]+r["out"]).splitlines() if re.search(r"^error|FAILED|panicked|TIMEOUT|failures:",l)][:6] or (r["err"]+r["out"]).splitlines()[-3:]))] if r["rc"]!=0 else [fnd("WARN",None,0,f"cargo test -p {c} ran 0 tests","crate has no tests")] if kind=="test" and np+nf==0 else []
 tp=[p for p in fs if istest(p)]
 if tp:
  r=run([PY,"-m","pytest","-q","-p","no:cacheprovider","--continue-on-collection-errors",*[str(p) for p in tp]],ROOT,timeout);s=(r["out"].strip().splitlines() or [""])[-1];notes["pytest"]={"files":[rel(p) for p in tp],"rc":r["rc"],"summary":s,"secs":r["secs"]}
  out+=[fnd("FAIL",None,0,"pytest red",s or r["err"][-300:])] if r["rc"] not in (0,5) else [fnd("WARN",None,0,"pytest collected no tests",s)] if r["rc"]==5 else []
 for p in tp:
  n,k,fns=taut_py(T.get(p) or "");out+=[fnd("WARN",p,0,"tautological test",f"all {n} asserts check constants or the module's own source text")] if n and n==k else [fnd("WARN",p,0,"tautological test functions",f"{fns}")] if fns else []
 for p in fs:
  s=T.get(p) or "";n,k=taut_rs(s) if p.suffix==".rs" and "#[test]" in s else (0,0);out+=[fnd("WARN",p,0,"tautological test",f"all {n} Rust asserts compare literals or source text")] if n and n==k else []
 return out,notes
def c_servers(T)->list:
 out=[]
 for p,src in T.items():
  if src is None or p.suffix not in (".py",".rs",".js",".mjs",".ts") or rel(p) in SELF:continue
  L=src.splitlines();out+=[fnd("FAIL",p,i,"open CORS",m.group(0),l) for i,l in enumerate(L,1) for m in CORS.finditer(l)];rt=[(i,l) for i,l in enumerate(L,1) if ROUTE.search(l)]
  out+=[fnd("FAIL",p,rt[0][0],"HTTP routes without an auth check",f"{len(rt)} routes, e.g. "+"; ".join(l.strip()[:60] for _,l in rt[:4]))] if rt and not AUTH.search(src) else []
 return out
def _num(s):
 try:return float(s)
 except (TypeError,ValueError):return None
def _dig(d,k):
 for x in k.split("."):d=d[int(x)] if isinstance(d,list) else d.get(x) if isinstance(d,dict) else None
 return d
def c_claims(path,fs,rerun,tol,timeout)->tuple:
 if not path:return [fnd("WARN",None,0,"no --claims","numbers in the status are unverifiable by this gate; list {metric,value,command,artifact} per number")],[]
 try:cl=json.loads(Path(path).read_text());cl=cl if isinstance(cl,list) else cl.get("claims",[])
 except (OSError,ValueError) as e:return [fnd("FAIL",None,0,"claims file unreadable",str(e))],[]
 out,notes=[],[];newest=max((p.stat().st_mtime for p in fs if p.suffix in CODE),default=0);sd=importlib.import_module("amni.serve.self_debug") if rerun else None
 for j,c in enumerate(cl):
  miss=[k for k in ("metric","value","command","artifact") if k not in c];nm=c.get("metric",f"#{j}")
  if miss:out.append(fnd("FAIL",None,0,f"claim {nm} missing keys",miss));continue
  art=Path(c["artifact"]) if Path(c["artifact"]).is_absolute() else ROOT/c["artifact"];argv=shlex.split(c["command"]);n={"metric":nm,"value":c["value"],"command":c["command"],"artifact":rel(art),"artifact_exists":art.exists()}
  out+=[fnd("FAIL",None,0,f"claim {nm}: box path",m.group(0)) for m in BOX.finditer(c["command"]+" "+c["artifact"])]
  out+=[fnd("FAIL",None,0,f"claim {nm}: artifact not on Antman",c["artifact"])] if not art.exists() or not art.resolve().is_relative_to(ROOT) else []
  out+=[fnd("FAIL",None,0,f"claim {nm}: command not runnable here",f"`{argv[0] if argv else ''}` not found")] if not argv or not (shutil.which(argv[0]) or (ROOT/argv[0]).exists()) else []
  out+=[fnd("FAIL",None,0,f"claim {nm}: command references a missing path",x) for x in argv[1:] if "/" in x and not x.startswith("-") and "{out}" not in x and "=" not in x and not ((ROOT/x).exists() or Path(x).exists())]
  age=[fnd("WARN",None,0,f"claim {nm}: artifact is older than the declared code",f"{rel(art)} mtime {time.strftime('%H:%M',time.localtime(art.stat().st_mtime))} < newest code {time.strftime('%H:%M',time.localtime(newest))}")] if art.exists() and art.stat().st_mtime<newest else []
  out+=[fnd("WARN",None,0,f"claim {nm}: perfect score",f"{nm}={c['value']}: show the gold labels are independent of the system (circular-metric check)")] if re.search(r"(?i)prec|recall|acc|f1|pass",nm) and _num(c["value"]) in (0.0,1.0) else []
  try:ad=json.loads(art.read_text()) if art.exists() and art.suffix==".json" else {}
  except ValueError:ad={}
  el=ad.get("eval_lock") if isinstance(ad,dict) else None;elh=importlib.import_module("amni.compute.eval_lock").code_hash([ROOT/k for k in el["code_files"]]) if isinstance(el,dict) and el.get("code_files") and all((ROOT/k).exists() for k in el["code_files"]) else None
  out+=[fnd("FAIL",None,0,f"claim {nm}: artifact code_hash differs from the code on Antman",f"artifact {el['code_hash'][:12]} vs now {elh[:12]}")] if elh and elh!=el.get("code_hash") else []
  if rerun and argv:
   with tempfile.TemporaryDirectory(prefix="presub_rr_") as od:
    av=[x.replace("{out}",od) for x in argv];jl=c.get("jail",True);r=sd.jail_run(av,od,timeout=c.get("timeout",timeout),mem_mb=c.get("mem_mb",2048),nproc=c.get("nproc",64),extra=["--ro-bind",str(ROOT),str(ROOT),"--chdir",str(ROOT)],max_output=4<<20) if jl else {**run(av,ROOT,c.get("timeout",timeout)),"jailed":False}
    so=(r.get("stdout") or r.get("out") or "")+"\n"+(r.get("stderr") or r.get("err") or "");jf=c.get("json");fp=Path(jf["file"].replace("{out}",od)) if jf else None
    fp=fp if fp is None or fp.is_absolute() else ROOT/fp
    try:got=_num(_dig(json.loads(fp.read_text()),jf["key"])) if jf else None
    except (OSError,ValueError,KeyError,TypeError):got=None
    m=None if jf else re.search(c.get("regex") or rf"{re.escape(nm)}\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",so);got=_num(m.group(1)) if m else got;rc=r.get("returncode",r.get("rc"));t=float(c.get("tol",tol))
    n.update({"rerun_rc":rc,"rerun_value":got,"jailed":r.get("jailed"),"tol":t})
    out+=[fnd("WARN",None,0,f"claim {nm}: rerun outside the jail","set jail:true unless the command runs no mission code")] if not jl else []
    out+=[fnd("FAIL",None,0,f"claim {nm}: rerun failed",f"rc={rc} killed={r.get('killed')} {so.strip()[-200:]}")] if rc!=0 else [fnd("FAIL",None,0,f"claim {nm}: value not found in rerun output","set regex or json {file,key}")] if got is None else [fnd("FAIL",None,0,f"claim {nm}: rerun differs",f"claimed {c['value']} rerun {got} tol {t}")] if abs(got-float(c["value"]))>t else []
    age=[] if rc==0 and got is not None and abs(got-float(c["value"]))<=t else age
  out+=age;notes.append(n)
 return out,notes
def _git(*a,text_=True):return subprocess.run(["git","-C",str(ROOT),*a],capture_output=True,text=text_).stdout
def c_git(mid,fs,dirs)->list:
 decl={rel(p) for p in fs};drel=[rel(d) for d in dirs];inside=lambda r:any(r==d or r.startswith(d+"/") for d in drel);foot={str(Path(r).parent) for r in decl}|set(drel)
 mod=sorted(set(_git("diff","--name-only","HEAD").split())|set(_git("ls-files","-d").split()));tracked=set(_git("ls-files").splitlines());out=[]
 out+=[fnd("FAIL" if str(Path(r).parent) in foot else "WARN",ROOT/r,0,"tracked file modified outside --files"+("" if str(Path(r).parent) in foot else " (shared tree: confirm it is not this mission's)"),r) for r in mod if r not in decl and not inside(r)]
 out+=[fnd("WARN",p,0,"backup file inside a declared directory","move to backups/") for p in fs if BAK.search(p.name)]
 tid=re.compile(rf"(?i)(?<![a-z0-9]){re.escape(mid)}(?![0-9])")
 out+=[fnd("FAIL" if tid.search(p.name) else "WARN",p,0,"tarball in docs/tasks/v20/status/","deliver individual files, not tarballs") for p in (sorted(STATUSD.iterdir()) if STATUSD.exists() else []) if TB.search(p.name)]
 out+=[fnd("FAIL",p,0,"tarball declared as a mission file","deliver individual files") for p in fs if TB.search(p.name)]
 cand={p.resolve() for p in [*ROOT.glob("*"),*[q for d in ROOT.glob("_inbox_*") for q in d.rglob("*")],*((ROOT/"data"/mid.lower()).rglob("*") if (ROOT/"data"/mid.lower()).is_dir() else [])] if p.is_file() and TB.search(p.name) and tid.search(rel(p))}
 for tp in sorted(cand):
  out.append(fnd("WARN",tp,0,"mission tarball in the repo","tarballs are not an accepted delivery form"))
  try:tf=tarfile.open(tp)
  except (tarfile.TarError,OSError) as e:out.append(fnd("WARN",tp,0,"unreadable tarball",str(e)));continue
  with tf:
   for m in tf.getmembers():
    if not m.isfile() or m.size>(16<<20):continue
    pt=Path(m.name).parts;cs={"/".join(pt[k:]) for k in range(min(3,len(pt))) if k==0 or len(pt)-k>=2};cs|={"crates/"+x for x in cs};ht=cs&tracked;hd=(cs&decl)-ht
    if not ht and not hd:continue
    b=tf.extractfile(m).read()
    for r in sorted(ht):
     wt=(ROOT/r).read_bytes() if (ROOT/r).is_file() else None;hb=subprocess.run(["git","-C",str(ROOT),"show",f"HEAD:{r}"],capture_output=True).stdout
     out.append(fnd("FAIL",ROOT/r,0,"tarball extracted over a tracked file",f"{rel(tp)}:{m.name} == working tree != HEAD") if wt==b and wt!=hb else fnd("WARN",ROOT/r,0,"tarball carries a tracked path",f"{rel(tp)}:{m.name}"))
    for r in sorted(hd):
     wt=(ROOT/r).read_bytes();new=m.mtime>(ROOT/r).stat().st_mtime
     out+=[fnd("FAIL" if new else "WARN",ROOT/r,0,"Antman file differs from the newer delivered tarball: the measured code is not the code on Antman" if new else "stale tarball copy differs from the Antman file",f"{rel(tp)}:{m.name} ({len(b)} B, {time.strftime('%m-%d %H:%M',time.localtime(m.mtime))}) vs Antman ({len(wt)} B, {time.strftime('%m-%d %H:%M',time.localtime((ROOT/r).stat().st_mtime))})")] if wt!=b else []
 return out
def report_md(rep)->str:
 g=rep["git"];L=[f"# {rep['mission']} pre-submit: **{rep['verdict']}**",f"{rep['generated']} · host {rep['host']} · HEAD `{g['head'][:10]}` ({g['branch']}, dirty tracked: {g['dirty_tracked']}) · code_hash `{rep['code_hash'][:16]}` · {len(rep['code_files'])} files","","| check | status | findings |","|---|---|---|"]
 L+=[f"| {k} | {v['status']} | {len(v['findings'])} |" for k,v in rep["checks"].items()]
 for k,v in rep["checks"].items():
  bad=[f for f in v["findings"] if f["sev"]!="PASS"];L+=([f"## {k}: {v['status']}"]+[f"- {f['sev']} `{f['file']}{':'+str(f['line']) if f['line'] else ''}` {f['rule']}: `{f['evidence'].replace('`',chr(39))}`" for f in bad[:12]]+([f"- … {len(bad)-12} more in the JSON"] if len(bad)>12 else [])) if bad else []
 n=rep["checks"].get("6_build_test",{}).get("notes") or {};L+=["## Builds and tests"]+[f"- `{k}`: rc {v.get('rc')}"+(f", {v['passed']} passed / {v['failed']} failed" if "passed" in v else "")+(f", {v['summary']}" if "summary" in v else "")+f" ({v.get('secs')} s)" for k,v in n.items()] if n else []
 return "\n".join(L)+"\n"
def main(argv=None)->int:
 ap=argparse.ArgumentParser(prog="scripts/v20_presubmit.py",description="v20 pre-submission gate: run on Antman before a status goes to QA");ap.add_argument("--mission",required=True);ap.add_argument("--files",nargs="+",required=True);ap.add_argument("--crates",nargs="*",default=[]);ap.add_argument("--data",nargs="*",default=[]);ap.add_argument("--claims");ap.add_argument("--rerun",action="store_true");ap.add_argument("--tol",type=float,default=5e-4);ap.add_argument("--timeout",type=int,default=1800);ap.add_argument("--sample-mb",type=float,default=0);ap.add_argument("--out-dir",default=str(OUTD));ap.add_argument("--no-status",action="store_true")
 a=ap.parse_args(argv);mid=a.mission.upper();fs,miss,dirs=expand(a.files);st=STATUSD/f"{mid}.md";scan=sorted(set(fs)|({st.resolve()} if st.exists() and not a.no_status else set()));T={p:text(p) for p in scan};TC={p:T[p] for p in fs}
 dd,dn=c_decontam_data(a.data,a.sample_mb,a.timeout) if a.data else ([],[]);tf,tn=c_tests(fs,TC,a.crates,a.timeout);cf,cn=c_claims(a.claims,fs,a.rerun,a.tol,a.timeout)
 ch={"0_inputs":(c_inputs(miss,fs),None),"1_paths":(c_paths(T),None),"2_tainted_data":(c_taint(T),None),"3_exec_safety":(c_exec(TC),None),"4_decontam":(c_decontam_static(TC)+dd,dn or None),"5_gate_tuning":(c_tuning(TC),None),"6_build_test":(tf,tn or None),"7_servers":(c_servers(TC),None),"8_claims":(cf,cn or None),"9_git_delivery":(c_git(mid,fs,dirs),None)}
 checks={k:{"status":worst(f),"findings":dedup(f),**({"notes":n} if n else {})} for k,(f,n) in ch.items()};el=importlib.import_module("amni.compute.eval_lock");verdict=worst([{"sev":v["status"]} for v in checks.values()])
 rep={"mission":mid,"verdict":verdict,"generated":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"host":socket.gethostname(),"argv":shlex.join([sys.executable,*sys.argv]) if argv is None else shlex.join(argv),"git":el.git_head(),"code_hash":el.code_hash(fs),"code_files":el.file_hashes(fs),"status_file":rel(st) if st.exists() and not a.no_status else None,"crates":a.crates,"data":a.data,"claims":a.claims,"checks":checks}
 od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);(od/f"{mid}_presubmit.json").write_text(json.dumps(rep,indent=1));(od/f"{mid}_presubmit.md").write_text(report_md(rep))
 print(f"{mid}: {verdict}  "+"  ".join(f"{k}={v['status']}" for k,v in checks.items())+f"\n-> {rel(od/(mid+'_presubmit.json'))}");return 0 if verdict!="FAIL" else 1
if __name__=="__main__":sys.exit(main())
