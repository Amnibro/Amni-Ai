import os,re,sys,json,glob,time,shlex,socket,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EVD=ROOT/"data"/"concepts"/"eval";KINDS=("prod","dry","stub");NONPROD=re.compile(r"_(stub|dry)(\.|$)")
def rel(p)->str:
 p=Path(p).resolve();return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def sha256(p)->str:
 h=hashlib.sha256()
 with open(p,"rb") as f:[h.update(b) for b in iter(lambda:f.read(1<<20),b"")]
 return h.hexdigest()
def file_hashes(files)->dict:return {rel(f):sha256(f) for f in sorted({str(Path(f).resolve()) for f in files})}
def code_hash(files)->str:return hashlib.sha256(json.dumps(file_hashes(files),sort_keys=True).encode()).hexdigest()
def lock(split_files,out=None)->dict:
 s=file_hashes(split_files);m={"splits":s,"splits_hash":hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),"n":len(s),"locked":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
 out and (Path(out).parent.mkdir(parents=True,exist_ok=True),Path(out).write_text(json.dumps(m,indent=1)));return m
def verify(manifest)->dict:
 m=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest;return {f:{"locked":h,"now":sha256(ROOT/f) if (ROOT/f).exists() else None} for f,h in m["splits"].items() if not (ROOT/f).exists() or sha256(ROOT/f)!=h}
def git_head()->dict:
 g=lambda *a:subprocess.run(["git","-C",str(ROOT),*a],capture_output=True,text=True).stdout.strip();return {"head":g("rev-parse","HEAD"),"branch":g("rev-parse","--abbrev-ref","HEAD"),"dirty_tracked":bool(g("status","--porcelain","--untracked-files=no"))}
def record(report_path,metrics:dict,cmd=None,code_files=(),splits=(),kind:str="prod",extra:dict=None)->dict:
 assert kind in KINDS,f"kind must be one of {KINDS}";sm=splits if isinstance(splits,dict) else lock(splits);bad=verify(sm) if isinstance(splits,dict) else {}
 assert not (bad and kind=="prod"),f"locked splits changed: {bad}";assert not (kind=="prod" and NONPROD.search(Path(report_path).name)),"prod report cannot use a _stub/_dry name"
 prov={"kind":kind,"cmd":cmd or shlex.join([sys.executable]+sys.argv),"cwd":os.getcwd(),"git":git_head(),"code_hash":code_hash(code_files),"code_files":file_hashes(code_files),"splits":sm["splits"],"splits_hash":sm["splits_hash"],"splits_changed":bad,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"host":socket.gethostname(),**(extra or {})}
 rep={**metrics,"kind":kind,"eval_lock":prov};p=Path(report_path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(rep,indent=1));os.replace(tmp,p);return rep
def report_kind(d)->str:
 dc=d.get("decode") if isinstance(d,dict) and isinstance(d.get("decode"),dict) else {};return "other" if not isinstance(d,dict) else d["kind"] if d.get("kind") in KINDS else "stub" if d.get("engine")=="stub" or dc.get("stub") is True or d.get("stub") is True else "dry" if dc.get("dry_run") is True or d.get("dry_run") is True else "prod"
def _kind(p)->str:
 try:m=NONPROD.search(Path(p).name);return m.group(1) if m else report_kind(json.load(open(p)))
 except (OSError,ValueError):return "unreadable"
def prod_reports(pattern=None)->list:return [p for p in sorted(glob.glob(str(pattern or EVD/"report_*.json")),key=os.path.getmtime) if _kind(p)=="prod"]
if __name__=="__main__":
 a=sys.argv[1:];print(json.dumps(lock(a[2:],a[1]) if a[:1]==["lock"] and len(a)>2 else {p:_kind(p) for p in sorted(glob.glob(str(a[1] if len(a)>1 else EVD/"report_*.json")))} if a[:1]==["kinds"] else {"usage":"python -m amni.compute.eval_lock lock <manifest.json> <split files...> | kinds [glob]"},indent=1))
