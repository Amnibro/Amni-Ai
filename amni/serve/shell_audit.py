"""shell_audit — append-only audit log of every shell command Adam runs.
Every `_skill_shell` and `_skill_git` invocation flushes one JSONL row with timestamp, command, exit code, and stdout/stderr tails. Exposed via /memory/shell-history for the /jarvis SHELL pill so the user can see exactly what Adam has been doing on their machine."""
import json,time
from pathlib import Path
from typing import Dict,Any,List,Optional
def _data_dir()->Path:
    base=Path(__file__).resolve().parents[2]/'data';base.mkdir(parents=True,exist_ok=True);return base
def _log_path()->Path:return _data_dir()/'shell_history.jsonl'
def log_shell_run(kind:str,cmd:str,returncode:int,stdout:str='',stderr:str='',cwd:str='',duration_s:Optional[float]=None,extras:Optional[Dict[str,Any]]=None):
    try:
        try:from amni.serve.code_safety import scrub_egress as _se
        except Exception:_se=lambda x:x
        entry={'ts':time.time(),'kind':kind,'cmd':_se(cmd or ''),'returncode':returncode,'cwd':_se(cwd or ''),'stdout_tail':_se((stdout or '')[-1200:]),'stderr_tail':_se((stderr or '')[-600:]),'duration_s':duration_s}
        if extras:entry.update(_se(extras))
        with open(_log_path(),'a',encoding='utf-8') as f:f.write(json.dumps(entry,default=str)+'\n')
        if returncode!=0:
            try:
                from amni.serve.notifications import queue_notification
                err_tail=_se((stderr or '').strip()).splitlines()
                err_msg=err_tail[-1][:200] if err_tail else 'no stderr'
                _scmd=_se(cmd or '')
                queue_notification('warn',kind,f'{kind} command failed (rc {returncode})',f'$ {_scmd[:80]}{"…" if len(_scmd)>80 else ""}\n{err_msg}',ttl_s=300.0,cmd=_scmd,returncode=returncode)
            except Exception:pass
    except Exception:pass
def list_shell_history(limit:int=50,errors_only:bool=False,kind:Optional[str]=None)->List[Dict[str,Any]]:
    p=_log_path()
    if not p.exists():return []
    out=[]
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:obj=json.loads(line)
            except Exception:continue
            if errors_only and (obj.get('returncode') or 0)==0:continue
            if kind and obj.get('kind')!=kind:continue
            out.append(obj)
    except Exception:return []
    return out[-limit:][::-1]
def shell_history_stats()->Dict[str,Any]:
    p=_log_path()
    if not p.exists():return {'n_total':0,'n_errors':0,'last_ts':None,'by_kind':{}}
    total=0;errors=0;last_ts=0.0;by_kind={}
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:obj=json.loads(line)
            except Exception:continue
            total+=1
            if (obj.get('returncode') or 0)!=0:errors+=1
            k=obj.get('kind','?');by_kind[k]=by_kind.get(k,0)+1
            ts=float(obj.get('ts') or 0);last_ts=max(last_ts,ts)
    except Exception:pass
    return {'n_total':total,'n_errors':errors,'last_ts':last_ts or None,'by_kind':by_kind}
