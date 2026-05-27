"""pc_actions — Tier-5b safe PC operation: Adam acts on the machine only through a propose→confirm gate.
Every action is PROPOSED first (returns a token + plain-language description + risk); nothing touches the OS until
the owner CONFIRMS that token. Clearly-destructive patterns are refused outright (even with confirm). Everything —
proposed, confirmed, executed, refused, cancelled — is appended to logs/pc_actions.jsonl. This is the human-in-the-loop
rail behind "do anything on a PC": capable, never unsupervised. Executors are a swappable dict so tests never touch the real OS."""
import time,uuid,threading,json,re,os,subprocess,webbrowser
from pathlib import Path
from typing import Dict,Any,Optional
_PENDING:Dict[str,Dict[str,Any]]={}
_LOCK=threading.Lock()
_TTL=300.0
_DENY=re.compile(r'(?i)(?:\brm\s+-rf\b|\brmdir\s+/s\b|\bdel\s+/[sq]\b|\bformat\s+[a-z]:|\bmkfs\b|\bdd\s+if=|\bshutdown\b|\breboot\b|:\(\)\s*\{|\bfork\s*bomb\b|\breg\s+delete\b|\bdiskpart\b|\bdrop\s+(?:table|database)\b|>\s*/dev/sd)')
_RISK={'echo':'low','notify':'low','open_url':'low','open_path':'medium','launch_app':'medium','run':'high'}
def _audit_path()->Path:
    p=Path(__file__).resolve().parents[2]/'logs'/'pc_actions.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _audit(rec:Dict[str,Any]):
    try:
        with _audit_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception:pass
def is_destructive(target:str)->bool:
    return bool(_DENY.search(target or ''))
def _exec_open_url(t,a):
    webbrowser.open(t);return {'opened_url':t}
def _exec_open_path(t,a):
    p=Path(t)
    if not p.exists():return {'error':f'path not found: {t}'}
    if hasattr(os,'startfile'):os.startfile(str(p))
    else:subprocess.Popen(['xdg-open' if os.name=='posix' else 'open',str(p)])
    return {'opened_path':str(p)}
def _exec_launch(t,a):
    args=[t]+(list(a.get('argv',[])) if isinstance(a,dict) else [])
    proc=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return {'launched':t,'pid':proc.pid}
def _exec_run(t,a):
    timeout=int((a or {}).get('timeout',20)) if isinstance(a,dict) else 20
    r=subprocess.run(t,shell=True,capture_output=True,text=True,timeout=timeout)
    return {'returncode':r.returncode,'stdout':(r.stdout or '')[:4000],'stderr':(r.stderr or '')[:2000]}
def _exec_notify(t,a):
    return {'notified':t[:200]}
def _exec_echo(t,a):
    return {'echoed':t}
_EXECUTORS={'echo':_exec_echo,'notify':_exec_notify,'open_url':_exec_open_url,'open_path':_exec_open_path,'launch_app':_exec_launch,'run':_exec_run}
def _describe(action,target,risk):
    verb={'echo':'echo','notify':'show a desktop notification','open_url':'open in your browser','open_path':'open with the default app','launch_app':'launch the program','run':'run the shell command'}.get(action,action)
    return f'[{risk.upper()} RISK] Adam wants to {verb}: {target[:160]}'
def propose(action:str,target:str,args:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    action=(action or '').lower().strip()
    target=(target or '').strip()
    if action not in _RISK:return {'error':f'unknown action {action!r}; valid: {", ".join(_RISK)}'}
    if not target:return {'error':'target required'}
    if is_destructive(target):
        _audit({'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'action':action,'target':target[:300],'status':'refused','reason':'destructive_pattern'})
        return {'error':'refused: matches a destructive/irreversible pattern and will not be run','refused':True,'action':action,'target':target}
    token='pca_'+uuid.uuid4().hex[:12];risk=_RISK[action];now=time.time()
    rec={'token':token,'action':action,'target':target[:1000],'args':args or {},'risk':risk,'ts':now,'status':'proposed'}
    with _LOCK:_PENDING[token]=rec
    _audit({**{k:rec[k] for k in ('token','action','target','risk','status')},'iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
    return {'token':token,'action':action,'target':rec['target'],'risk':risk,'requires_confirm':True,'description':_describe(action,target,risk),'expires_in_s':int(_TTL)}
def confirm(token:str)->Dict[str,Any]:
    with _LOCK:rec=_PENDING.pop(token,None)
    if rec is None:return {'error':f'no pending action {token!r} (expired or already used)'}
    if time.time()-rec['ts']>_TTL:
        _audit({'token':token,'action':rec['action'],'status':'expired','iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
        return {'error':'action expired; re-propose'}
    if is_destructive(rec['target']):
        _audit({'token':token,'action':rec['action'],'status':'refused','reason':'destructive_on_confirm','iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
        return {'error':'refused: destructive pattern','refused':True}
    ex=_EXECUTORS.get(rec['action'])
    if ex is None:return {'error':f'no executor for {rec["action"]!r}'}
    try:out=ex(rec['target'],rec.get('args') or {})
    except Exception as e:
        _audit({'token':token,'action':rec['action'],'target':rec['target'][:300],'status':'error','error':str(e)[:300],'iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
        return {'error':f'execution failed: {e}','action':rec['action']}
    _audit({'token':token,'action':rec['action'],'target':rec['target'][:300],'status':'executed','iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
    return {'executed':True,'action':rec['action'],'target':rec['target'],'result':out}
def cancel(token:str)->Dict[str,Any]:
    with _LOCK:rec=_PENDING.pop(token,None)
    if rec is None:return {'error':f'no pending action {token!r}'}
    _audit({'token':token,'action':rec['action'],'status':'cancelled','iso':time.strftime('%Y-%m-%dT%H:%M:%S')})
    return {'cancelled':token}
def list_pending()->Dict[str,Any]:
    now=time.time()
    with _LOCK:
        for k in [k for k,v in _PENDING.items() if now-v['ts']>_TTL]:_PENDING.pop(k,None)
        items=[{'token':v['token'],'action':v['action'],'target':v['target'][:160],'risk':v['risk'],'age_s':round(now-v['ts'],1)} for v in _PENDING.values()]
    return {'pending':items,'n':len(items)}
def audit_recent(limit:int=30)->Dict[str,Any]:
    p=_audit_path()
    if not p.exists():return {'total':0,'recent':[],'by_status':{}}
    rows=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:rows.append(json.loads(ln))
            except Exception:continue
    except Exception:return {'total':0,'recent':[],'by_status':{}}
    by={}
    for r in rows:by[r.get('status','?')]=by.get(r.get('status','?'),0)+1
    return {'total':len(rows),'by_status':by,'recent':rows[-int(max(1,limit)):][::-1]}
