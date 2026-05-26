"""skill_failures — append-only diagnostic log for skill invocations that returned error.
Lets us root-cause transient skill misfires (e.g. system_stats refusing once after working fine).
Surfaces via GET /memory/skill-failures?limit=N."""
import json,time,os
from pathlib import Path
from typing import Dict,Any,List,Optional
_LOG_REL='logs/skill_failures.jsonl'
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _log_path()->Path:
    p=_repo_root()/_LOG_REL;p.parent.mkdir(parents=True,exist_ok=True);return p
def record(skill:str,message:str,args:Dict[str,Any],error:str,tb:Optional[str]=None,extra:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    rec={'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime()),'skill':skill,'message':(message or '')[:400],'args':args,'error':(error or '')[:600]}
    if tb:rec['tb']=(tb or '')[:1500]
    if extra:rec['extra']=extra
    try:
        with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception as e:print(f'[skill_failures] write failed: {e}',flush=True)
    return rec
def recent(limit:int=20,skill_filter:Optional[str]=None)->List[Dict[str,Any]]:
    p=_log_path()
    if not p.exists():return []
    out=[]
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:rec=json.loads(line)
            except Exception:continue
            if skill_filter and rec.get('skill')!=skill_filter:continue
            out.append(rec)
    except Exception:return []
    return out[-int(max(1,limit)):]
def stats()->Dict[str,Any]:
    p=_log_path()
    if not p.exists():return {'total':0,'by_skill':{},'last_iso':None}
    counts:Dict[str,int]={};total=0;last_iso=None
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:rec=json.loads(line)
            except Exception:continue
            sk=rec.get('skill') or '?';counts[sk]=counts.get(sk,0)+1;total+=1;last_iso=rec.get('iso') or last_iso
    except Exception:pass
    return {'total':total,'by_skill':dict(sorted(counts.items(),key=lambda kv:-kv[1])[:20]),'last_iso':last_iso}
