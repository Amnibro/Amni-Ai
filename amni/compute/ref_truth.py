from __future__ import annotations
import ast,importlib,json,os,shutil,subprocess,sys,warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/"data"/"ks"/"reference"/"python"
VERSIONS=("3.8","3.9","3.10","3.11","3.12","3.13")
PRLIMIT="/usr/bin/prlimit"
LIT={"Constant":None,"List":"list","ListComp":"list","Dict":"dict","DictComp":"dict","Set":"set","SetComp":"set","JoinedStr":"str","Tuple":"tuple"}
def imports_of(tree)->dict:
    a={}
    for n in ast.walk(tree):
        [a.__setitem__(al.asname,al.name) if al.asname else a.__setitem__(al.name.split(".")[0],al.name.split(".")[0]) for al in n.names] if isinstance(n,ast.Import) else None
        [a.__setitem__(al.asname or al.name,n.module+"."+al.name) for al in n.names if al.name!="*"] if isinstance(n,ast.ImportFrom) and not n.level and n.module else None
    return a
def bound_names(tree)->set:
    b=set()
    for n in ast.walk(tree):
        b.add(n.id) if isinstance(n,ast.Name) and not isinstance(n.ctx,ast.Load) else None
        b.add(n.name) if type(n).__name__ in ("FunctionDef","AsyncFunctionDef","ClassDef","ExceptHandler","MatchAs","MatchStar") and isinstance(getattr(n,"name",None),str) else None
        b.add(n.arg) if isinstance(n,ast.arg) else None
        b.add(n.rest) if type(n).__name__=="MatchMapping" and n.rest else None
        b.update(n.names) if isinstance(n,(ast.Global,ast.Nonlocal)) else None
    return b
def _base(n,aliases,bound):
    t=type(n).__name__
    return (aliases.get(n.id) or ("builtins."+n.id if n.id not in bound else None)) if t=="Name" else (("builtins."+type(n.value).__name__) if t=="Constant" and isinstance(n.value,(str,bytes)) else (("builtins."+LIT[t]) if LIT.get(t) else None))
def refs(tree,aliases=None,bound=None)->list:
    aliases=imports_of(tree) if aliases is None else aliases;bound=bound_names(tree) if bound is None else bound
    nodes=list(ast.walk(tree));calls={id(n.func):n for n in nodes if isinstance(n,ast.Call)};inner={id(n.value) for n in nodes if isinstance(n,ast.Attribute)};out=[]
    for n in nodes:
        out.extend([[al.name,n.lineno,None] for al in n.names] if isinstance(n,ast.Import) else ([[n.module,n.lineno,None]]+[[n.module+"."+al.name,n.lineno,None] for al in n.names if al.name!="*"] if isinstance(n,ast.ImportFrom) and not n.level and n.module else []))
        if id(n) in inner or not isinstance(n,(ast.Attribute,ast.Name)) or (isinstance(n,ast.Name) and not isinstance(n.ctx,ast.Load)):
            continue
        ch=[];b=n
        while isinstance(b,ast.Attribute):
            ch.insert(0,b.attr);b=b.value
        root=_base(b,aliases,bound);c=calls.get(id(n))
        out.append([".".join([root]+ch),n.lineno,[sum(1 for x in c.args if not isinstance(x,ast.Starred)),[k.arg for k in c.keywords if k.arg],int(any(isinstance(x,ast.Starred) for x in c.args)),int(any(k.arg is None for k in c.keywords)),int(not isinstance(b,ast.Name) and len(ch)==1)] if c else None]) if root else None
    return out
def probe(code:str)->dict:
    r={"version":"%d.%d.%d"%sys.version_info[:3]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            compile(code,"<snippet>","exec",dont_inherit=True);r.update(parses=True,warnings=[str(x.message)[:200] for x in w])
        except (SyntaxError,ValueError,MemoryError,RecursionError,OverflowError) as e:
            r.update(parses=False,error=type(e).__name__+": "+str(getattr(e,"msg",e))[:300],line=getattr(e,"lineno",None))
    r.update(refs=refs(ast.parse(code))) if r["parses"] else None
    return r
@lru_cache(maxsize=None)
def interpreters()->dict:
    p=REF/"interpreters.json";d=json.loads(p.read_text()) if p.exists() else {}
    fb={v:subprocess.run(["uv","python","find","--no-project",v],capture_output=True,text=True).stdout.strip() for v in VERSIONS if v not in d} if shutil.which("uv") else {}
    d.update({v:{"path":os.path.realpath(x)} for v,x in fb.items() if x})
    return d
@lru_cache(maxsize=None)
def index()->dict:
    return json.loads((REF/"api_index.json").read_text())
def lookup(q:str,ver:str)->tuple:
    nm=index()["names"];vb=1<<VERSIONS.index(ver);parts=q.split(".");cur=parts[0];chain=[cur]
    if cur not in nm:
        return "unknown",q,chain
    for p in parts[1:]:
        e=nm[cur]
        if not e["v"]&vb:
            return "missing",cur,chain
        base=e["t"] if e["k"]=="module" and e.get("t") in nm else cur
        if e["k"] not in ("module","class","exception"):
            return "ok",cur,chain
        nxt=base+"."+p
        if nxt not in nm or not nm[nxt]["v"]&vb:
            return ("unverifiable" if e.get("ga",0)&vb or nm.get(base,{}).get("ga",0)&vb else "missing"),nxt,chain
        cur=nxt;chain.append(cur)
    return ("ok" if nm[cur]["v"]&vb else "missing"),cur,chain
def bind_error(params:list,call:list,kind:str=""):
    npos,kws,st,ds=call[:4];npos+=int(len(call)>4 and call[4] and kind not in ("classmethod","staticmethod"));pos=[p for p in params if p[1] in (0,1)];varp=any(p[1]==2 for p in params);vark=any(p[1]==4 for p in params);got=set(p[0] for p in pos[:npos]);errs=[]
    errs.append("takes %d positional but %d given"%(len(pos),npos)) if npos>len(pos) and not varp else None
    for k in kws:
        m=[p for p in params if p[0]==k and p[1] in (1,3)]
        errs.append("unexpected keyword %r"%k) if not m and not vark else (errs.append("multiple values for %r"%k) if m and k in got else got.add(k))
    miss=[p[0] for p in params if p[1] in (0,1,3) and not p[2] and p[0] not in got]
    errs.append("missing required %s"%miss) if miss and not st and not ds else None
    return "; ".join(errs) or None
def _vt(s:str)->tuple:
    return tuple(int(x) for x in s.split(".")[:2] if x.isdigit())
def _deps(chain:list,ver:str)->list:
    nm=index()["names"];ds=[[c,nm[c]["dep"][ver]] for c in chain if c in nm and ver in nm[c].get("dep",{})]
    return [[c,i] for c,i in ds if i.get("import_warn") or i.get("runtime___deprecated__") or i.get("doc_deprecated")=="yes" or _vt(i.get("doc_deprecated") or "99")<=_vt(ver)]
def _run_probe(ver:str,code:str,timeout:float=20.0)->dict:
    py=interpreters().get(ver,{}).get("path")
    if not py or not os.path.exists(py):
        return {"parses":None,"error":"interpreter %s not installed"%ver}
    try:
        p=subprocess.run([PRLIMIT,"--cpu=15","--as=%d"%(1<<31),py,"-I","-S",str(Path(__file__).resolve()),"--probe"],input=code,capture_output=True,text=True,timeout=timeout,env={"PATH":"/usr/bin","LANG":"C.UTF-8","PYTHONIOENCODING":"utf-8"})
        return json.loads(p.stdout) if p.returncode==0 else {"parses":None,"error":"probe failed rc=%d %s"%(p.returncode,p.stderr[-300:])}
    except subprocess.TimeoutExpired:
        return {"parses":None,"error":"probe timeout"}
def check_snippet(code:str,target_versions=None)->dict:
    vs=[v for v in VERSIONS if v in (target_versions or VERSIONS)];nm=index()["names"]
    with ThreadPoolExecutor(len(vs) or 1) as ex:
        pr=dict(zip(vs,ex.map(lambda v:_run_probe(v,code),vs)))
    rf=next((pr[v]["refs"] for v in reversed(vs) if pr[v].get("parses")),[]);used={};third=set()
    for q,line,call in rf:
        top=q.split(".")[0];std=top in nm
        used.setdefault(q,{"lines":[],"calls":[]}) if std else third.add(top)
        (used[q]["lines"].append(line),used[q]["calls"].append(call) if call else None) if std else None
    used={q:u for q,u in used.items() if not (q.startswith("builtins.") and all(lookup(".".join(q.split(".")[:2]),v)[0]!="ok" for v in VERSIONS))}
    per={}
    for v in vs:
        p=pr[v];r={"parses":p.get("parses"),"error":p.get("error"),"line":p.get("line"),"syntax_warnings":p.get("warnings",[]),"missing":[],"unverifiable":[],"sig_mismatch":[],"deprecated":[],"reasons":[]}
        r["reasons"].append("syntax: "+str(p.get("error"))+(" (line %s)"%p.get("line") if p.get("line") else "")) if not p.get("parses") else None
        for q,u in (used.items() if p.get("parses") else []):
            st,res,chain=lookup(q,v);e=nm.get(res,{})
            others=[x+("?" if lookup(q,x)[0]=="unverifiable" else "") for x in VERSIONS if x!=v and lookup(q,x)[0] in ("ok","unverifiable")]
            r["missing"].append(q) if st=="missing" else (r["unverifiable"].append(q) if st=="unverifiable" else None)
            r["reasons"].append("%s not in %s stdlib (exists in: %s; ?=lazy module __getattr__, unverifiable)"%(q,v,",".join(others) or "none")) if st=="missing" else None
            sg=[s for s in e.get("sg",[]) if s[2]>>VERSIONS.index(v)&1 and s[1] is not None] if st=="ok" and len(chain)==q.count(".")+1 else []
            bad=[(c,[bind_error(s[1],c,e.get("k","")) for s in sg]) for c in u["calls"]] if sg else []
            [(r["sig_mismatch"].append({"name":q,"signature":sg[0][0],"error":errs[0]}),r["reasons"].append("%s%s: %s"%(q,sg[0][0],errs[0]))) for c,errs in bad if all(errs)]
            dp=[(c,i) for c,i in _deps(chain,v) if c not in {d["via"] for d in r["deprecated"]}];r["deprecated"].extend([{"name":q,"via":c,"info":i} for c,i in dp]);r["reasons"].extend(["%s deprecated in %s (%s)"%(c,v,json.dumps(i,sort_keys=True)) for c,i in dp])
        r["valid"]=bool(p.get("parses")) and not r["missing"] and not r["sig_mismatch"];per[v]=r
    ok=[v for v in vs if per[v]["valid"]]
    return {"executed":False,"versions":per,"stdlib_names":sorted(used),"third_party":sorted(third),"verdict":{"valid_on":ok,"invalid_on":[v for v in vs if v not in ok],"clean_on":[v for v in ok if not per[v]["deprecated"] and not per[v]["unverifiable"]],"summary":("valid on "+", ".join(ok) if ok else "valid on no pinned version")}}
def run_on_version(code:str,version:str,timeout:float=12.0,cpu:int=10,mem:int=1<<30,stdin:str="")->dict:
    sys.path.insert(0,str(ROOT/"scripts")) if str(ROOT/"scripts") not in sys.path else None
    run_sandbox=importlib.import_module("x0_code_eval").run_sandbox;py=interpreters().get(version,{}).get("path")
    return dict(run_sandbox(code,stdin=stdin,cpu=cpu,mem=mem,timeout=timeout,py=py),version=version,interpreter=py) if py and os.path.exists(py) else {"ok":False,"rc":None,"stdout":"","stderr":"interpreter %s not installed"%version,"version":version,"timeout":False}
if __name__=="__main__" and sys.argv[1:]==["--probe"]:
    sys.stdout.write(json.dumps(probe(sys.stdin.read())))
