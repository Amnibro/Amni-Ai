import argparse,ast,base64,builtins,json,os,re,shutil,subprocess,sys,tempfile,time,warnings,zlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"code_eval"
TMP=DATA/"tmp"
PY=str(ROOT/".venv-linux"/"bin"/"python");PYV=PY
BWRAP="/usr/bin/bwrap"
PRLIMIT="/usr/bin/prlimit"
TMPFS_SIZE=64<<20
sys.path.insert(0,str(ROOT))
def _now():return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def extract_code(text:str)->str:
 t=(text or "").strip();ms=re.findall(r"```(?:python|py)?\s*\n(.*?)```",t,re.S|re.I)
 return (ms[-1] if ms else t).strip()
RO=[d for d in {sys.prefix,sys.base_prefix,str(Path.home()/".local"/"share"/"uv"/"python"),str(ROOT/".venv-linux"),str(ROOT/"data"/"ks"/"reference"/"python"/"venvs")} if os.path.isdir(d)]
def _bwrap_cmd(workdir:str,argv:list,cpu:int=10,mem:int=1<<30)->list:
 binds=["--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64","--ro-bind","/bin","/bin","--ro-bind","/sbin","/sbin"]+[x for d in RO for x in ("--ro-bind",d,d)]
 return [BWRAP,"--unshare-user","--disable-userns","--unshare-net","--unshare-pid","--unshare-ipc","--unshare-uts","--hostname","sandbox","--cap-drop","ALL","--new-session","--clearenv","--setenv","PATH","/usr/bin:/bin","--setenv","HOME","/tmp","--setenv","LANG","C.UTF-8","--setenv","OPENBLAS_NUM_THREADS","1","--setenv","OMP_NUM_THREADS","1",*binds,"--proc","/proc","--dev","/dev","--size",str(TMPFS_SIZE),"--tmpfs","/tmp","--bind",workdir,workdir,"--chdir",workdir,"--die-with-parent",PRLIMIT,f"--cpu={cpu}",f"--as={mem}",*argv]
def run_sandbox(code:str,stdin:str="",cpu:int=10,mem:int=1<<30,timeout:float=12.0,filename:str="main.py",py:str=PY)->dict:
 TMP.mkdir(parents=True,exist_ok=True)
 wd=tempfile.mkdtemp(prefix="x0_",dir=str(TMP))
 try:
  (Path(wd)/filename).write_text(code,encoding="utf-8")
  cmd=_bwrap_cmd(wd,[py,filename],cpu=cpu,mem=mem)
  t0=time.perf_counter()
  try:
   p=subprocess.run(cmd,input=stdin or "",text=True,capture_output=True,timeout=timeout,env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"})
   wall=time.perf_counter()-t0
   return {"ok":p.returncode==0,"rc":p.returncode,"stdout":p.stdout or "","stderr":p.stderr or "","wall":wall,"timeout":False}
  except subprocess.TimeoutExpired as e:
   return {"ok":False,"rc":-1,"stdout":(e.stdout or "") if isinstance(e.stdout,str) else "","stderr":"TIMEOUT","wall":time.perf_counter()-t0,"timeout":True}
 finally:
  shutil.rmtree(wd,ignore_errors=True)
def he_harness(prompt:str,solution:str,test:str,entry:str)->str:
 return f"{prompt}{solution}\n{test}\n\ncheck({entry})\n"
def mbpp_harness(code:str,tests:list,imports:list|None=None)->str:
 im="\n".join(imports or []);body="\n".join(tests or [])
 return f"{im}\n{code}\n{body}\n" if im else f"{code}\n{body}\n"
def lcb_harness(code:str)->str:return code if "def " in code or "class " in code or "import " in code else code
def save_jsonl(name:str,rows:list)->Path:
 DATA.mkdir(parents=True,exist_ok=True);p=DATA/f"{name}.jsonl"
 with p.open("w",encoding="utf-8") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 return p
X11DEV=("mbpp_holdout","apps_fn_dev","lcb_easy_fn","apps_fn_lock")
def load_ds(name:str)->list:
 import random as _r
 rs=[r for f in X11DEV for r in load_jsonl(f)] if name=="x11_dev" else load_jsonl(name);sm=os.environ.get("X0_SAMPLE","")
 q=(sm+"::").split(":");return sorted(_r.Random(int(q[1] or 11)).sample(rs,min(len(rs),int(q[0])))[:int(q[2] or 10**9)],key=rs.index) if sm and name=="x11_dev" else rs
def load_jsonl(name:str)->list:
 p=DATA/f"{name}.jsonl"
 return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
def prepare_datasets(force:bool=False)->dict:
 meta={"prepared_at":_now(),"datasets":{}}
 if force or not (DATA/"humaneval.jsonl").exists():
  from datasets import load_dataset
  he=load_dataset("openai/openai_humaneval",split="test")
  rows=[{"id":r["task_id"],"prompt":r["prompt"],"canonical":r["canonical_solution"],"test":r["test"],"entry_point":r["entry_point"],"kind":"humaneval"} for r in he]
  save_jsonl("humaneval",rows);meta["datasets"]["humaneval"]={"source":"openai/openai_humaneval","split":"test","count":len(rows),"date":_now()}
 else:
  rows=load_jsonl("humaneval");meta["datasets"]["humaneval"]={"source":"openai/openai_humaneval","split":"test","count":len(rows),"date":"cached"}
 if force or not (DATA/"mbpp.jsonl").exists():
  from datasets import load_dataset
  mb=load_dataset("google-research-datasets/mbpp","sanitized",split="test")
  rows=[{"id":f"mbpp/{r['task_id']}","prompt":r["prompt"],"canonical":r["code"],"test_list":list(r["test_list"]),"test_imports":list(r.get("test_imports") or []),"kind":"mbpp"} for r in mb]
  save_jsonl("mbpp",rows);meta["datasets"]["mbpp"]={"source":"google-research-datasets/mbpp","config":"sanitized","split":"test","count":len(rows),"date":_now()}
 else:
  rows=load_jsonl("mbpp");meta["datasets"]["mbpp"]={"source":"google-research-datasets/mbpp","config":"sanitized","split":"test","count":len(rows),"date":"cached"}
 if force or not (DATA/"mbpp_holdout.jsonl").exists():
  from datasets import load_dataset
  full=load_dataset("google-research-datasets/mbpp","full");san={int(r["id"].split("/")[1]) for r in load_jsonl("mbpp")};rows=[{"id":f"mbpp_full/{r['task_id']}","prompt":r["text"],"canonical":r["code"],"test_list":list(r["test_list"]),"test_imports":[],"test_setup":r.get("test_setup_code") or "","kind":"mbpp","split":sp} for sp in ("train","validation") for r in full[sp]]
  assert san and not san&{int(r["id"].split("/")[1]) for r in rows},"mbpp_holdout overlaps sanitized test"
  save_jsonl("mbpp_holdout",rows);meta["datasets"]["mbpp_holdout"]={"source":"google-research-datasets/mbpp","config":"full","split":"train+validation","count":len(rows),"disjoint_from":"mbpp sanitized test task_ids (asserted)","date":_now()}
 else:
  rows=load_jsonl("mbpp_holdout");meta["datasets"]["mbpp_holdout"]={"source":"google-research-datasets/mbpp","config":"full","split":"train+validation","count":len(rows),"date":"cached"}
 tid=lambda f:{int(r["id"].split("/")[1]) for r in load_jsonl(f)}
 if force or not (DATA/"mbpp_fresh.jsonl").exists():
  from datasets import load_dataset
  sz=load_dataset("google-research-datasets/mbpp","sanitized");san=tid("mbpp")|{r["task_id"] for sp in sz for r in sz[sp]};ho=tid("mbpp_holdout");rows=[{"id":f"mbpp_full/{r['task_id']}","prompt":r["text"],"canonical":r["code"],"test_list":list(r["test_list"]),"test_imports":[],"test_setup":r.get("test_setup_code") or "","kind":"mbpp","split":"test"} for r in load_dataset("google-research-datasets/mbpp","full",split="test") if r["task_id"] not in san|ho];save_jsonl("mbpp_fresh",rows)
 san,ho,fr=tid("mbpp"),tid("mbpp_holdout"),tid("mbpp_fresh");assert san and ho and fr and not fr&san and not fr&ho and len(fr)==len(load_jsonl("mbpp_fresh")),"mbpp_fresh overlaps sanitized test or mbpp_holdout";print(f"mbpp_fresh n={len(fr)} disjoint from sanitized test ({len(san)}) and mbpp_holdout ({len(ho)}) task_ids",flush=True)
 meta["datasets"]["mbpp_fresh"]={"source":"google-research-datasets/mbpp","config":"full","split":"test","count":len(fr),"excludes":"every task_id in the sanitized config (all splits) and in mbpp_holdout","disjoint_from":"mbpp sanitized test and mbpp_holdout task_ids (asserted)","date":_now()}
 if force or not (DATA/"humanevalplus.jsonl").exists():
  from datasets import load_dataset
  rows=[{"id":r["task_id"],"test":r["test"],"entry_point":r["entry_point"]} for r in load_dataset("evalplus/humanevalplus",split="test")];save_jsonl("humanevalplus",rows);meta["datasets"]["humanevalplus"]={"source":"evalplus/humanevalplus","split":"test","count":len(rows),"use":"extra hidden tests only","date":_now()}
 if force or not (DATA/"livecodebench.jsonl").exists():
  from datasets import load_dataset
  ds=load_dataset("marianna13/livecodebench_code_generation_lite_parquet",split="test")
  rows=[];skipped_old=0;skipped_fn=0;max_date=""
  for r in ds:
   d=str(r.get("contest_date") or "")[:10];max_date=max(max_date,d)
   if d<="2025-06-01":skipped_old+=1;continue
   try:cases=json.loads(r["public_test_cases"]) if isinstance(r["public_test_cases"],str) else r["public_test_cases"]
   except Exception:cases=[]
   tt=(cases[0].get("testtype") if cases and isinstance(cases[0],dict) else None)
   if tt=="functional":skipped_fn+=1;continue
   if not cases or not any((c.get("input") or "") for c in cases if isinstance(c,dict)):skipped_fn+=1;continue
   priv=[]
   try:
    raw=base64.b64decode(r["private_test_cases"]);data=zlib.decompress(raw);priv=json.loads(data)
   except Exception:priv=[]
   pub=[{"input":c.get("input",""),"output":c.get("output","")} for c in cases if isinstance(c,dict)]
   priv_io=[{"input":c.get("input",""),"output":c.get("output","")} for c in (priv if isinstance(priv,list) else []) if isinstance(c,dict) and (c.get("input") is not None)]
   rows.append({"id":str(r.get("question_id") or r.get("question_title")),"prompt":r.get("question_content") or "","starter_code":r.get("starter_code") or "","canonical":"","tests":(pub+priv_io)[:20],"contest_date":d,"kind":"livecodebench","platform":r.get("platform")})
   if len(rows)>=100:break
  save_jsonl("livecodebench",rows)
  meta["datasets"]["livecodebench"]={"source":"marianna13/livecodebench_code_generation_lite_parquet","split":"test","filter":"contest_date>2025-06-01 stdin/stdout cap100","count":len(rows),"date":_now(),"note":f"official LCB lite max contest_date={max_date}; after filter n={len(rows)}; skipped_old={skipped_old} skipped_non_io={skipped_fn}"}
 else:
  rows=load_jsonl("livecodebench");meta["datasets"]["livecodebench"]={"source":"marianna13/livecodebench_code_generation_lite_parquet","split":"test","filter":"contest_date>2025-06-01 stdin/stdout cap100","count":len(rows),"date":"cached"}
 (DATA/"meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8");return meta
def eval_problem(prob:dict,code:str,plus:dict|None=None)->dict:
 kind=prob["kind"];t0=time.perf_counter()
 if kind=="humaneval":
  src=he_harness(prob["prompt"],code,(plus or prob)["test"],prob["entry_point"]);res=run_sandbox(src,cpu=30 if plus else 10,timeout=40.0 if plus else 12.0,mem=(2<<30) if plus else 1<<30,py=PYV if plus else PY)
 elif kind=="mbpp":
  src=mbpp_harness(code,([prob["test_setup"]] if (prob.get("test_setup") or "").strip() else [])+(prob.get("test_list") or []),prob.get("test_imports") or []);res=run_sandbox(src)
 elif kind=="livecodebench":
  fails=[];ok_all=True;walls=[]
  for i,tc in enumerate(prob.get("tests") or []):
   res=run_sandbox(code,stdin=tc.get("input") or "")
   walls.append(res["wall"]);exp=(tc.get("output") or "").strip();got=(res.get("stdout") or "").strip()
   if (not res["ok"]) or got!=exp:ok_all=False;fails.append({"i":i,"rc":res["rc"],"stderr":(res.get("stderr") or "")[:300],"got":got[:200],"exp":exp[:200]});break
  return {"ok":ok_all,"wall":sum(walls) if walls else time.perf_counter()-t0,"error":(fails[0] if fails else {}), "stderr":str(fails[0]) if fails else ""}
 else:
  res={"ok":False,"wall":0,"stderr":"unknown kind","rc":-2,"timeout":False}
 err=(res.get("stderr") or "")[:400]
 return {"ok":bool(res.get("ok")),"wall":float(res.get("wall") or 0),"error":{"rc":res.get("rc"),"stderr":err,"timeout":res.get("timeout")},"stderr":err}
def make_oracle_gen(probs_by_id:dict):
 def gen(prompt:str)->str:
  for p in probs_by_id.values():
   if p.get("prompt") and p["prompt"] in prompt:return p.get("canonical") or ""
   if p.get("id") and str(p["id"]) in prompt:return p.get("canonical") or ""
  return ""
 return gen
def make_adam_gen(pack:str|None=None,max_bytes:int=600):
 from amni.compute.ray_field import RayField
 from amni.compute.ray_qa import QA,guide
 pack=pack or str(ROOT/"exports"/"gf17_continuum"/"rayhash_clean1v2_base_v18_s100k_sid.npy")
 e=RayField(path=pack,arch=os.environ.get("AMNI_RAY_ARCH","cpu")).load();qa=QA()
 def gen(prompt:str)->str:
  q=prompt;formatted=prompt if prompt.rstrip().endswith("Adam:") else f"User: {q}\nAdam:"
  g=guide(e,qa,q) if qa else None
  out="".join(e.stream(formatted,max_bytes=max_bytes,guide=(b" "+g) if g else None,beta=20.0,stop=("\n\n","\nUser:","\nAdam:"),seed=11)).strip()
  return extract_code(out) if out else ""
 return gen
def make_ref_gen(model_id:str="Qwen/Qwen2.5-Coder-0.5B-Instruct",max_new:int=256):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model_id);model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.float32).to("cpu");model.eval()
 def gen(prompt:str)->str:
  msgs=[{"role":"user","content":prompt}]
  text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True) if hasattr(tok,"apply_chat_template") else prompt
  inputs=tok(text,return_tensors="pt")
  with torch.no_grad():out=model.generate(**inputs,max_new_tokens=max_new,do_sample=False)
  gen_ids=out[0][inputs["input_ids"].shape[-1]:];reply=tok.decode(gen_ids,skip_special_tokens=True)
  return extract_code(reply)
 return gen
def compiles(src:str)->bool:
 try:
  with warnings.catch_warnings():warnings.simplefilter("ignore");compile(src,"<x1>","exec")
  return True
 except Exception:return False
def _expr_ok(s:str)->bool:
 try:
  with warnings.catch_warnings():warnings.simplefilter("ignore");compile(s,"<x1c>","eval")
  return True
 except Exception:return False
def _close(s:str,i:int)->int:
 d=0
 for k in range(i,len(s)):
  d+=(s[k]=="(")-(s[k]==")")
  if d==0:return k+1
 return -1
def _free_ok(c:str,allow:set)->bool:
 try:return _expr_ok(c) and all(n.id in allow for n in ast.walk(ast.parse(c,mode="eval")) if isinstance(n,ast.Name))
 except Exception:return False
def _rv(r:str)->list:return [r.rstrip(".,;: "),r,re.split(r"[.;,]\s|\s(?:and|because|since|if|when|as)\s|\s#",r)[0].rstrip(".,;: "),r.split("=")[-1].strip().rstrip(".,;: ")]
def _kwcall(f:str,params:list,txt:str)->str:
 kv=[(m.group(1),m.group(2).strip().rstrip(".,;: ")) for m in re.finditer(r"(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|,?\s+the\b|,?\s+#|$)",txt.strip())]
 return f"{f}({', '.join(f'{k}={v}' for k,v in kv)})" if kv and all(k in params for k,_ in kv) else (f"{f}({txt.strip()})" if txt.strip() and not kv else "")
def selfchecks(prompt:str)->list:
 ds=list(re.finditer(r"^def (\w+)\s*\(",prompt,re.M));f=ds[-1].group(1) if ds else "\0";ls=prompt[ds[-1].end():].splitlines() if ds else [];out=[];allow={f,"_x0eq","math","re"}|set(dir(builtins))
 try:params=[a.arg for n in ast.walk(ast.parse(prompt+"    pass\n")) if isinstance(n,ast.FunctionDef) and n.name==f for a in n.args.args+n.args.kwonlyargs]
 except SyntaxError:params=[]
 for i,l in enumerate(ls):
  s=l.strip();dt=s.startswith(">>>");a=bool(re.match(r"\s*assert\b",l));s=re.sub(r"^(?:>>>|assert\b|for example[:,]?|examples?[:,]?|e\.g\.,?|[-*•])\s*","",s,flags=re.I).strip();mj=re.search(rf"\b{re.escape(f)}\s*\(",s);j=mj.start() if mj else -1;k=_close(s,mj.end()-1) if mj else -1;nxt=ls[i+1].strip() if i+1<len(ls) else "";m=re.match(r"(?:#\s*)?(==>|==|=>|➞|->|→|=|should return|returns|return|is)\s*(.+)$",s[k:].strip()) if k>0 else None;fo=re.match(r"for (.+?)[, ]+the (?:output|result|return value|answer) (?:should|will) be (.+)$",s,re.I);io=re.match(r"input:\s*(.+)$",s,re.I);oo=re.match(r"output:\s*(.+)$",nxt,re.I)
  prs=([(s,[nxt])] if dt and nxt and not nxt.startswith((">>>",'"""',"'''")) else [])+([(s[j:k],_rv(m.group(2)))] if m else [])+([(_kwcall(f,params,fo.group(1)),_rv(fo.group(2)))] if fo else [])+([(_kwcall(f,params,io.group(1)),_rv(oo.group(1)))] if io and oo else [])
  c=next((x for lh,rs in prs for x in (f"_x0eq(({lh}),({r}))" for r in rs if lh and r) if _free_ok(x,allow)),None) or (s if j>=0 and ("==" in s or a) and _free_ok(s,allow) else None)
  (out.append(c) if c and c not in out else None)
 return out
def _argname(a,i:int)->str:
 return ({str:"s",int:"n",float:"x",bool:"flag"}.get(type(a.value),"a") if isinstance(a,ast.Constant) else {ast.List:"lst",ast.Tuple:"tup",ast.Dict:"d",ast.Set:"items"}.get(type(a),"a"))+str(i)
def mbpp_code_prompt(prob:dict)->str:
 t=((prob.get("test_list") or [""])[0]).strip()
 try:cs=[n for n in ast.walk(ast.parse(t)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 except SyntaxError:cs=[]
 c=next((n for n in cs if n.func.id not in dir(builtins)),cs[0] if cs else None);ar=[_argname(a,i+1) for i,a in enumerate(c.args)]+[k.arg for k in c.keywords if k.arg] if c else ["*args"];desc=" ".join((prob.get("prompt") or "").split());q="'''" if '"""' in desc+t else '"""';r="r" if "\\" in desc+t else ""
 return f"import math\nimport re\nimport heapq\nimport collections\nimport itertools\n\n\ndef {c.func.id if c else 'solution'}({', '.join(ar)}):\n    {r}{q}{desc}\n    {t}\n    {q}\n"
def selfcheck_src(program:str,checks:list)->str:
 return program+"\nimport math,re,collections,itertools,heapq\ndef _x0eq(a,b):\n try:return a==b or abs(a-b)<=1e-6*max(1.0,abs(b))\n except BaseException:return False\n"+"".join(f"try:print('X0SC_7f3a',bool({c}),flush=True)\nexcept BaseException:print('X0SC_7f3a',False,flush=True)\n" for c in checks)
def sample_bodies(e,prompt:str,n:int,max_bytes:int,temp:float,topk:int,seed:int)->list:
 g=np.random.default_rng(seed);lg=e.prime(prompt.encode(),nb=n);buf=np.zeros((n,max_bytes),np.uint8);prev=np.full(n,10);run=np.zeros(n,int);ln=np.zeros(n,int);live=np.ones(n,bool);ar=np.arange(n)
 for t in range(max_bytes):
  z=np.where(e.ok,lg/temp,-1e9);ix=np.argpartition(-z,topk,axis=1)[:,:topk];zz=np.take_along_axis(z,ix,1);p=np.exp(zz-zz.max(1,keepdims=True));p/=p.sum(1,keepdims=True);c=ix[ar,np.minimum((p.cumsum(1)<g.random((n,1))).sum(1),topk-1)]
  run=np.where(c==10,run+1,0);live&=~(((prev==10)&(c!=10)&(c!=32)&(c!=9))|(run>=3));ln=np.where(live,t+1,ln);buf[:,t]=c;prev=c
  if not live.any():break
  lg=e._advance(ar.tolist(),c.tolist(),True)
 return [bytes(buf[i,:ln[i]]).decode("utf-8","replace").rstrip()+"\n" for i in range(n)]
def trim_body(prompt:str,body:str)->str:
 ls=body.splitlines();return next(("\n".join(ls[:k])+"\n" for k in range(len(ls),0,-1) if "\n".join(ls[:k]).strip() and compiles(prompt+"\n".join(ls[:k])+"\n")),body)
def runnable(prompt:str,body:str)->bool:return any(l.strip() and not l.strip().startswith("#") for l in body.splitlines()) and compiles(prompt+body)
def select_sample(prompt:str,bodies:list,jobs:int=8)->dict:
 checks=selfchecks(prompt);comp=[runnable(prompt,b) for b in bodies];uniq=list(dict.fromkeys(b for b,ok in zip(bodies,comp) if ok and checks))
 with ThreadPoolExecutor(jobs) as ex:sc=dict(zip(uniq,ex.map(lambda b:(run_sandbox(selfcheck_src(prompt+b,checks),timeout=6.0).get("stdout") or "").count("X0SC_7f3a True"),uniq)))
 scores=[sc.get(b,0) for b in bodies];return {"samples":bodies,"pick":max(range(len(bodies)),key=lambda i:(scores[i],comp[i],-i)),"n_checks":len(checks),"scores":scores,"compiles":comp,"checks":checks}
def make_adam_code_gen(pack:str|None=None):
 from amni.compute.ray_field import RayField
 E=os.environ.get;e=RayField(path=pack,arch=E("AMNI_RAY_ARCH","cpu")).load();n,temp,topk,mb,trim=int(E("X1_N","16")),float(E("X1_TEMP","0.6")),int(E("X1_TOPK","12")),int(E("X1_BYTES","500")),E("X1_TRIM","1")=="1";retry=int(E("X1_RETRY","0"))
 def gen(prompt:str)->dict:
  sd=zlib.crc32(prompt.encode());bs=[]
  for r in range(retry+1):
   bs+=[trim_body(prompt,b) if trim else b for b in sample_bodies(e,prompt,n,mb,temp,topk,sd+r)]
   if any(runnable(prompt,b) for b in bs):break
  return select_sample(prompt,bs)
 gen.mode="code";gen.cfg={"pack":e.path,"arch":e.arch,"n":n,"temp":temp,"topk":topk,"max_bytes":mb,"trim":trim,"retry":retry};return gen
def make_adam_tiles_gen():
 from amni.compute import tile_synth as TS
 E=os.environ.get;[os.environ.setdefault(k,v) for k,v in (("AMNI_RAY_IMPL","rust"),("AMNI_RAY_ARCH","cpu"))];mode=E("TS_MODE","fresh");cur=mode=="curriculum";store=E("TS_STORE",str(DATA/"learned_tiles.jsonl"));jobs=1 if cur else int(E("TS_JOBS","8"));budget=float(E("TS_BUDGET","10"));kw={"budget":budget,"store":store if cur else E("TS_XSTORE") or None,"learn":cur};cache={};X=__import__("x8_sketch").Index() if E("TS_SK","off")!="off" else None;po=X.post if X else (lambda r:r);it=lambda p,c:(p,c,{"nbs":X.nbs(p,c,k=int(E("TS_SKK","20")),kp=int(E("TS_SKP","8")))}) if X else (p,c)
 ([Path(x).unlink(missing_ok=True) for x in (store,TS.qpath(store))] if cur and E("TS_RESET")=="1" else None);ex=TS.pool(jobs)
 def prefetch(prompts):
  if jobs<=1:return
  rs=TS.solve_many(ex,[it(p,selfchecks(p)) for p in prompts],**kw)
  with ThreadPoolExecutor(int(E("X0_RRJ","3"))) as tp:cache.update(zip(prompts,tp.map(lambda pr:TS.rerank(po(pr[1]),pr[0],checks=selfchecks(pr[0])),zip(prompts,rs))))
 def gen(prompt:str)->dict:
  r=cache.pop(prompt,None) or TS.rerank(po(TS.solve_many(ex,[it(prompt,selfchecks(prompt))],**kw)[0]),prompt,checks=selfchecks(prompt));cs=r.get("cands") or [];bs=[c["body"] for c in cs] or [r["body"]]
  return {"samples":bs,"pick":r["pick"],"n_checks":r["n_checks"],"scores":[r["n_checks"] if r["found"] else 0]*len(bs),"compiles":[bool(b) for b in bs],"meta":{**{f"ts_{k}":v for k,v in r.items() if k not in ("body","cands")},"ts_cands":[{k:v for k,v in c.items() if k!="body"} for c in cs],**({"ts_bodies":bs,"ts_checks":selfchecks(prompt)} if E("X0_KEEP_BODIES")=="1" else {})}}
 gen.mode="code";gen.cfg={"generator":"adam_tiles","mode":mode,"budget_s":budget,"jobs":jobs,"store":kw["store"],"quarantine":str(TS.qpath(store)) if cur else None,"proving_ground":"learned programs enter quarantine with provenance; normal search uses only promoted tiles; promotion needs LOO pass on source examples plus a verified (prompt-example) probation use on a different problem; failures are demoted and kept with reasons" if cur else None,"n_library_tiles":len(TS.library()),"selection":"lowest penalized score (cost + coverage, intent-cue, constant, redundancy, probe-crash and edge penalties) among programs passing all prompt examples; mutated-input type inconsistency rejects; no examples or no passing program -> empty body; X4: up to TS_TOPN verified candidates kept, reranked by TS_RANK (cost, penalty weight, LOO weight, pycode-LM naturalness), abstain flag for variant (b) = LOO fail (or no LOO) and low naturalness; submitted program is always the (a) pick","probes":int(E("TS_PROBES","6")),"sem":E("TS_SEM","wn"),"topn":TS.TOPN,"extra_levels":TS.XLV,"extra_s":TS.XS,"rank":TS.RANK,"prior":str(TS.PRIORP) if TS.prior() else None,"pcfg":TS.PCFG,"pbias":TS.PBIAS,"xstore":None if cur else E("TS_XSTORE"),"nat_pack":E("TS_NAT_PACK",TS.NATP),"ray_impl":E("AMNI_RAY_IMPL"),"sketch":{"on":bool(X),"skc":TS.SKC,"index":str(X.fh.name) if X else None,"retrieval":"BM25 (k1=1.2,b=0.75) over docstring+name words x (1+0.25 arity match+0.15 return-type match); top-k corpus neighbours + top pure neighbours; 13-gram + exact-renamed-body contamination guard at retrieval time"}};gen.prefetch=prefetch;return gen
def tiles_stats(results:list)->dict:
 ts=[r for r in results if "ts_reason" in r];fd=[r for r in ts if r["ts_found"]];sz=sorted(r["ts_size"] for r in fd);he=lambda lo,hi:[r for r in ts if str(r["id"]).startswith("HumanEval/") and lo<=int(str(r["id"]).split("/")[1])<hi]
 part=lambda rs:{"n":len(rs),"n_pass":sum(r["ok"] for r in rs),"pass_at_1":sum(r["ok"] for r in rs)/max(1,len(rs)),"n_found":sum(r["ts_found"] for r in rs),"n_found_but_hidden_fail":sum(r["ts_found"] and not r["ok"] for r in rs),"n_no_examples":sum(r["ts_reason"]=="no_examples" for r in rs),"n_answered":sum(r["ts_found"] and not r.get("ts_abstain") for r in rs),"n_right_answered":sum(r["ok"] and not r.get("ts_abstain") for r in rs),**({"n_pass_plus":sum(bool(r.get("ok_plus")) for r in rs),"n_right_answered_plus":sum(bool(r.get("ok_plus")) and not r.get("ts_abstain") for r in rs)} if any("ok_plus" in r for r in rs) else {})};lo=lambda f:[r for r in fd if r.get("ts_loo_fail") is not None and f(r["ts_loo_fail"])]
 return {"tiles":{**part(ts),"precision_when_found":sum(r["ok"] for r in fd)/max(1,len(fd)),"mean_search_s":sum(r["ts_time"] for r in ts)/max(1,len(ts)),"mean_search_s_found":sum(r["ts_time"] for r in fd)/max(1,len(fd)),"size_ast_nodes":{"min":sz[0],"median":sz[len(sz)//2],"mean":sum(sz)/len(sz),"max":sz[-1]} if sz else {},"cost":{"mean":sum(r["ts_cost"] for r in fd)/max(1,len(fd)),"max":max((r["ts_cost"] for r in fd),default=0)},"n_used_learned":sum(bool(r.get("ts_used_learned")) for r in fd),"loo":{"n_scored":len(lo(lambda x:True)),"n_loo_pass":len(lo(lambda x:x==0)),"precision_loo_pass":sum(r["ok"] for r in lo(lambda x:x==0))/max(1,len(lo(lambda x:x==0))),"n_loo_fail":len(lo(lambda x:x>0)),"precision_loo_fail":sum(r["ok"] for r in lo(lambda x:x>0))/max(1,len(lo(lambda x:x>0)))},"learn_gate":{k:sum(r.get("ts_learn_gate")==k for r in fd) for k in sorted({str(r.get("ts_learn_gate")) for r in fd})},**({"proving_ground":{"n_probation_searches":sum(bool(r.get("ts_pg_tried")) for r in ts),"n_promoted":sum(len(r.get("ts_pg_promoted") or []) for r in ts),"n_demoted_in_probation":sum(len(r.get("ts_pg_demoted") or []) for r in ts),"n_pg_errors":sum("ts_pg_error" in r for r in ts)}} if any("ts_pg_tried" in r for r in ts) else {}),"reasons":{k:sum(r["ts_reason"]==k for r in ts) for k in sorted({r["ts_reason"] for r in ts})},**({"he_0_39":part(he(0,40)),"he_40_163":part(he(40,164))} if he(0,164) else {}),**({"mbpp_dev_0_99":part(ts[:100]),"mbpp_rest":part(ts[100:])} if ts and str(ts[0]["id"]).startswith("mbpp/") else {})}} if ts else {}
def multi_eval(prob:dict,prompt:str,d:dict,jobs:int=int(os.environ.get("X0_EVJ","8")),plus:dict|None=None)->dict:
 cs=[_join(prob,prompt,b) for b in d["samples"]];sub=os.environ.get("X0_EVSUB")=="1";uniq=list(dict.fromkeys([cs[j] for j in {d["pick"],d.get("meta",{}).get("ts_x9_pick",d["pick"])} if j is not None and j<len(cs)] if sub else cs))
 with ThreadPoolExecutor(jobs) as ex:ok=dict(zip(uniq,ex.map(lambda c:bool(eval_problem(prob,c).get("ok")),uniq)));okp=dict(zip([c for c in uniq if ok[c]],ex.map(lambda c:bool(eval_problem(prob,c,plus).get("ok")),[c for c in uniq if ok[c]]))) if plus else {}
 oks=[ok.get(c,False) for c in cs];return {"oks":oks,**({"evsub":True} if sub else {}),**({"oks_plus":[okp.get(c,False) for c in cs]} if plus else {}),"first_ok":oks[0],"any_ok":any(oks),"frac_ok":sum(oks)/len(oks),"pick":d["pick"],"n_checks":d["n_checks"],"checks_passed":d["scores"][d["pick"]],"n_compile":sum(d["compiles"]),"n_drawn":len(d["samples"]),"first_head":d["samples"][0][:240],**d.get("meta",{})}
def _join(prob:dict,prompt:str,body:str)->str:return body if prob["kind"]=="humaneval" else prompt+body
def problem_prompt(prob:dict,mode:str="")->str:
 if mode=="code" and prob["kind"] in ("humaneval","mbpp"):return prob["prompt"] if prob["kind"]=="humaneval" else mbpp_code_prompt(prob)
 if prob["kind"]=="humaneval":return prob["prompt"]
 if prob["kind"]=="mbpp":return f"Write a Python solution.\n\n{prob['prompt']}\n\nTests:\n"+"\n".join(prob.get("test_list") or [])
 sc=prob.get("starter_code") or ""
 return f"Solve this programming problem. Read from stdin and write to stdout.\n\n{prob.get('prompt','')}\n\n{('Starter code:\n'+sc) if sc else ''}"
def run_eval(dataset:str,gen_name:str,gen,limit:int=0,tag:str="",plus:bool=False)->dict:
 probs=load_ds(dataset);probs=probs[:limit] if limit else probs;hp={r["id"]:r for r in load_jsonl("humanevalplus")} if plus and dataset=="humaneval" else {}
 results=[];fails=[];walls=[];n_pass=0;(gen.prefetch([problem_prompt(p,getattr(gen,"mode","")) for p in probs]) if hasattr(gen,"prefetch") else None)
 for i,prob in enumerate(probs):
  prompt=problem_prompt(prob,getattr(gen,"mode",""));t0=time.perf_counter();d=None
  try:
   code=gen(prompt) if gen_name!="oracle_check" else (prob.get("canonical") or "");d=code if isinstance(code,dict) else None;code=_join(prob,prompt,d["samples"][d["pick"]]) if d else code
  except Exception as e:
   code="";err=f"GEN_ERROR {type(e).__name__}: {e}";res={"ok":False,"wall":time.perf_counter()-t0,"error":{"stderr":err},"stderr":err};results.append({"id":prob["id"],**res,"code_head":"","prompt_head":prompt[:200]});walls.append(res["wall"]);fails.append({"id":prob["id"],"prompt_head":prompt[:200],"code_head":"","error_head":err[:300]});continue
  if gen_name=="oracle_check" and prob["kind"]=="livecodebench" and not (prob.get("canonical") or "").strip():
   res={"ok":False,"wall":0,"error":{"stderr":"no_canonical"},"stderr":"no_canonical"};results.append({"id":prob["id"],**res,"code_head":"","prompt_head":prompt[:200]});walls.append(0);fails.append({"id":prob["id"],"prompt_head":prompt[:200],"code_head":"","error_head":"no_canonical"});continue
  code=extract_code(code) if gen_name!="oracle_check" and not d else code
  if prob["kind"]=="humaneval" and gen_name!="oracle_check":
   code=code if code.strip().startswith(("def ","class ","from ","import ")) or prob["prompt"].rstrip().endswith(":") else code
  res=eval_problem(prob,code if gen_name!="oracle_check" or prob["kind"]!="humaneval" else code)
  if gen_name=="oracle_check" and prob["kind"]=="humaneval":
   res=eval_problem(prob,code)
  wall=time.perf_counter()-t0;walls.append(wall);n_pass+=int(bool(res.get("ok")))
  item={"id":prob["id"],"ok":bool(res.get("ok")),**({"ok_plus":bool(res.get("ok")) and bool(eval_problem(prob,code,hp[prob["id"]]).get("ok"))} if prob["id"] in hp else {}),"wall":wall,"code_head":(code or "")[:240],"prompt_head":prompt[:200],"error_head":str(res.get("stderr") or res.get("error") or "")[:300],**(multi_eval(prob,prompt,d,plus=hp.get(prob["id"])) if d else {})}
  results.append(item)
  (fails.append({"id":prob["id"],"prompt_head":item["prompt_head"],"code_head":item["code_head"],"error_head":item["error_head"]}) if not item["ok"] and len(fails)<5 else None)
  print(f"[{gen_name}/{dataset}] {i+1}/{len(probs)} ok={item['ok']} wall={wall:.3f}"+(f" first={item['first_ok']} any={item['any_ok']} checks={item['checks_passed']}/{item['n_checks']} compile={item['n_compile']}" if d else ""),flush=True)
 n=len(probs);npl=sum(bool(r.get("ok_plus")) for r in results);rep={"dataset":dataset,"generator":gen_name,"n":n,"pass_at_1":(n_pass/n if n else 0.0),"n_pass":n_pass,**({"n_pass_plus":npl,"pass_at_1_plus":npl/n,"plus_note":"HumanEval+ (evalplus/humanevalplus) hidden tests run on the same picked program (original HumanEval prompt + body); counted only if the base tests also pass; sandbox cpu 30 s"} if hp else {}),"mean_wall":(sum(walls)/len(walls) if walls else 0.0),"failures":fails[:5],"tag":tag or gen_name,"ts":_now()};ms=[r for r in results if "first_ok" in r];nm=max(1,len(ms))
 rep.update({"pass_at_1_first_sample":sum(r["first_ok"] for r in ms)/nm,"pass_at_1_self_verified":n_pass/n if n else 0.0,"pass_at_n":sum(r["any_ok"] for r in ms)/nm,"pass_at_1_mean_over_samples":sum(r["frac_ok"] for r in ms)/nm,"mean_self_checks":sum(r["n_checks"] for r in ms)/nm,"frac_with_self_checks":sum(r["n_checks"]>0 for r in ms)/nm,"mean_compiling_samples":sum(r["n_compile"] for r in ms)/nm,"mean_samples_drawn":sum(r["n_drawn"] for r in ms)/nm,"cfg":getattr(gen,"cfg",{}),**tiles_stats(results),"note":"pass_at_1 = pass_at_1_self_verified: one submitted program per problem, chosen only by self-checks parsed from the prompt (HumanEval docstring examples; MBPP: first test assertion only, rendered into the prompt). Hidden tests never used for selection. pass_at_n is informational (oracle over all samples). mean_wall = generation + self-check selection + hidden eval of the picked program; excludes the pass@N evaluation of the other samples."} if ms else {})
 out=DATA/f"report_{tag or gen_name}_{dataset}.json";out.write_text(json.dumps({**rep,"items":results},indent=2),encoding="utf-8");print(json.dumps({k:rep[k] for k in rep if k!='failures'},indent=1));return rep
def free_available_gb()->float:
 out=subprocess.check_output(["free","-g"],text=True)
 for line in out.splitlines():
  if line.lower().startswith("mem:"):
   parts=line.split();return float(parts[-1])
 return 0.0
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--prepare",action="store_true");ap.add_argument("--force-prepare",action="store_true")
 ap.add_argument("--gen",choices=["oracle_check","adam_ray","adam_code","adam_tiles","reference_small","all"],default="oracle_check")
 ap.add_argument("--dataset",choices=["humaneval","mbpp","mbpp_holdout","mbpp_fresh","livecodebench","lcb_easy_fn","apps_fn_lock","apps_fn_dev","apps_x11_lock","lcd_easy_lock","x11_dev","all"],default="all");ap.add_argument("--plus",action="store_true");ap.add_argument("--limit",type=int,default=0);ap.add_argument("--tag",default="");ap.add_argument("--pack",default=None)
 ap.add_argument("--skip-ref-if-slow",action="store_true",default=True);ap.add_argument("--ref-probe",type=int,default=2)
 args=ap.parse_args();TMP.mkdir(parents=True,exist_ok=True)
 if args.prepare or args.force_prepare or not (DATA/"meta.json").exists():prepare_datasets(force=args.force_prepare)
 ds_list=["humaneval","mbpp","livecodebench"] if args.dataset=="all" else [args.dataset]
 gens=["oracle_check","adam_ray","reference_small"] if args.gen=="all" else [args.gen]
 summaries=[]
 for gname in gens:
  if gname=="oracle_check":
   for ds in ds_list:
    probs=load_jsonl(ds)
    if ds=="livecodebench" and (not probs or all(not (p.get("canonical") or "").strip() for p in probs)):
     rep={"dataset":ds,"generator":gname,"n":len(probs),"pass_at_1":None,"n_pass":0,"mean_wall":0.0,"failures":[],"tag":args.tag or gname,"ts":_now(),"note":"no canonical solutions in LCB mirror; oracle skipped"}
     (DATA/f"report_{(args.tag or gname)}_{ds}.json").write_text(json.dumps(rep,indent=2),encoding="utf-8");summaries.append(rep);continue
    summaries.append(run_eval(ds,gname,lambda p:"",limit=args.limit,tag=args.tag or gname,plus=args.plus))
  elif gname=="adam_ray":
   gen=make_adam_gen()
   for ds in ds_list:summaries.append(run_eval(ds,gname,gen,limit=args.limit,tag=args.tag or gname))
  elif gname in ("adam_code","adam_tiles"):
   gen=make_adam_code_gen(args.pack) if gname=="adam_code" else make_adam_tiles_gen()
   for ds in [x for x in ds_list if x!="livecodebench"]:summaries.append(run_eval(ds,gname,gen,limit=args.limit,tag=args.tag or gname,plus=args.plus))
  elif gname=="reference_small":
   avail=free_available_gb();print("free_available_gb",avail,flush=True)
   if avail<12:
    summaries.append({"generator":gname,"skipped":True,"reason":f"free -g available={avail} < 12"});continue
   t_probe=time.perf_counter();gen=None
   try:gen=make_ref_gen()
   except Exception as e:
    summaries.append({"generator":gname,"skipped":True,"reason":f"load_failed {type(e).__name__}: {e}"});continue
   load_t=time.perf_counter()-t_probe
   probe_n=max(1,args.ref_probe);probe_probs=load_jsonl("humaneval")[:probe_n];pt0=time.perf_counter()
   for p in probe_probs:
    _=gen(problem_prompt(p))
   per= (time.perf_counter()-pt0)/max(1,len(probe_probs))
   total_n=sum(len(load_jsonl(ds)) if not args.limit else min(args.limit,len(load_jsonl(ds))) for ds in ds_list)
   est=load_t+per*total_n
   print(f"ref_probe per={per:.2f}s est_total={est/60:.1f}min n={total_n}",flush=True)
   if est>float(os.environ.get("X0_REF_BUDGET_MIN","30"))*60:
    summaries.append({"generator":gname,"skipped":True,"reason":f"estimated {est/60:.1f} min > 30 min (per={per:.2f}s, n={total_n}, load={load_t:.1f}s)","free_available_gb":avail});del gen;continue
   for ds in ds_list:summaries.append(run_eval(ds,gname,gen,limit=args.limit,tag=args.tag or gname))
   del gen
 summ_path=DATA/f"summary_{args.tag or args.gen}.json";summ_path.write_text(json.dumps(summaries,indent=2),encoding="utf-8");print("wrote",summ_path)
if __name__=="__main__":main()
