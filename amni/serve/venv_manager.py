"""venv_manager — sandboxed Python virtual environments for Adam's self-improvement experiments.
All venvs live under <workdir>/.adam-venvs/<name>/. Names sanitized to [a-z0-9_-]{1,32}. Operations
audited via shell_audit (v6.10.22). Hard cap of 8 concurrent experimental venvs."""
import re,subprocess,sys,time,shutil,os
from pathlib import Path
from typing import Dict,Any,List,Optional
_NAME_RE=re.compile(r'^[a-z0-9_-]{1,32}$')
_MAX_VENVS=8
_TIMEOUT_CREATE=120
_TIMEOUT_INSTALL=180
_TIMEOUT_RUN=300
def _root(workdir:Path)->Path:
    p=workdir/'.adam-venvs';p.mkdir(parents=True,exist_ok=True);return p
def _venv_path(workdir:Path,name:str)->Optional[Path]:
    if not _NAME_RE.match(name or ''):return None
    return _root(workdir)/name
def _interp(venv_dir:Path)->Path:
    return venv_dir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
def list_venvs(workdir:Path)->List[Dict[str,Any]]:
    r=_root(workdir);out=[]
    if not r.exists():return out
    for d in sorted(r.iterdir()):
        if not d.is_dir():continue
        interp=_interp(d);created=None
        try:created=d.stat().st_mtime
        except Exception:pass
        out.append({'name':d.name,'path':str(d),'interpreter':str(interp),'exists':interp.exists(),'created_at':created})
    return out
def create(workdir:Path,name:str)->Dict[str,Any]:
    from amni.serve.shell_audit import log_shell_run
    if not _NAME_RE.match(name or ''):return {'error':f'venv name must match [a-z0-9_-]{{1,32}}; got {name!r}'}
    if len(list_venvs(workdir))>=_MAX_VENVS:return {'error':f'venv cap reached ({_MAX_VENVS}); remove an existing experiment first'}
    vd=_venv_path(workdir,name)
    if vd.exists():return {'error':f'venv {name!r} already exists at {vd}'}
    t0=time.time()
    try:r=subprocess.run([sys.executable,'-m','venv',str(vd)],capture_output=True,text=True,timeout=_TIMEOUT_CREATE)
    except subprocess.TimeoutExpired:return {'error':f'venv creation timed out after {_TIMEOUT_CREATE}s'}
    dur=round(time.time()-t0,2);log_shell_run('venv',f'create {name}',r.returncode,r.stdout,r.stderr,str(vd),dur)
    if r.returncode!=0:return {'error':f'venv create failed (rc {r.returncode}): {r.stderr[:400]}','returncode':r.returncode}
    return {'name':name,'path':str(vd),'interpreter':str(_interp(vd)),'created':True,'duration_s':dur}
def install(workdir:Path,name:str,packages:List[str])->Dict[str,Any]:
    from amni.serve.shell_audit import log_shell_run
    vd=_venv_path(workdir,name)
    if not vd or not vd.exists():return {'error':f'venv {name!r} not found; create it first'}
    if not packages or not isinstance(packages,list):return {'error':'packages must be a non-empty list of strings'}
    safe=[p for p in packages if isinstance(p,str) and re.match(r'^[A-Za-z0-9_.\[\]<>=!~+\-]+$',p)]
    if len(safe)!=len(packages):return {'error':'one or more package specs failed validation; allowed: name with optional version constraint'}
    interp=_interp(vd)
    if not interp.exists():return {'error':f'venv interpreter missing at {interp}'}
    t0=time.time()
    try:r=subprocess.run([str(interp),'-m','pip','install','--disable-pip-version-check','--no-input',*safe],capture_output=True,text=True,timeout=_TIMEOUT_INSTALL)
    except subprocess.TimeoutExpired:return {'error':f'pip install timed out after {_TIMEOUT_INSTALL}s'}
    dur=round(time.time()-t0,2);log_shell_run('venv',f'pip install {" ".join(safe)} (in {name})',r.returncode,r.stdout,r.stderr,str(vd),dur)
    return {'name':name,'packages':safe,'returncode':r.returncode,'stdout_tail':r.stdout[-1200:],'stderr_tail':r.stderr[-600:],'duration_s':dur,'ok':r.returncode==0}
def run(workdir:Path,name:str,cmd:str,timeout:int=_TIMEOUT_RUN)->Dict[str,Any]:
    from amni.serve.shell_audit import log_shell_run
    vd=_venv_path(workdir,name)
    if not vd or not vd.exists():return {'error':f'venv {name!r} not found'}
    interp=_interp(vd)
    if not interp.exists():return {'error':f'venv interpreter missing at {interp}'}
    if not (cmd or '').strip():return {'error':'cmd required'}
    if re.search(r'(?:^|[;&|`$])\s*(?:rm\s+-rf|del\s+/[sf]|format\s+|shutdown)',cmd,re.IGNORECASE):return {'error':'cmd appears destructive (rm -rf / format / shutdown); refuse'}
    env={**os.environ,'VIRTUAL_ENV':str(vd),'PATH':str(vd/('Scripts' if os.name=='nt' else 'bin'))+os.pathsep+os.environ.get('PATH','')}
    t0=time.time();timeout=int(min(max(5,timeout),_TIMEOUT_RUN))
    try:r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(workdir),env=env)
    except subprocess.TimeoutExpired:return {'error':f'cmd timed out after {timeout}s','timeout':True}
    dur=round(time.time()-t0,2);log_shell_run('venv',f'{name}: {cmd}',r.returncode,r.stdout,r.stderr,str(workdir),dur)
    return {'name':name,'cmd':cmd,'returncode':r.returncode,'stdout':r.stdout[-8000:],'stderr':r.stderr[-4000:],'duration_s':dur,'ok':r.returncode==0}
def remove(workdir:Path,name:str)->Dict[str,Any]:
    from amni.serve.shell_audit import log_shell_run
    vd=_venv_path(workdir,name)
    if not vd or not vd.exists():return {'error':f'venv {name!r} not found'}
    try:shutil.rmtree(vd);log_shell_run('venv',f'remove {name}',0,'','',str(vd),0);return {'name':name,'removed':True,'path':str(vd)}
    except Exception as e:return {'error':f'remove failed: {e}'}
