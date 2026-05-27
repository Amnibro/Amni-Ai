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
    try:
        from amni.serve.reffelt_tag import tag_record
        _t=tag_record((skill or '')+' '+(message or ''),extra_tags=[skill] if skill else None);rec['tags']=_t['tags'];rec['nonce']=_t['nonce']
    except Exception:pass
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
def _ack_path()->Path:return _log_path().parent/'skill_failures_ack.json'
def _load_ack()->Dict[str,Any]:
    p=_ack_path()
    if not p.exists():return {'ack_count':0,'ack_iso':None}
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return {'ack_count':0,'ack_iso':None}
def _save_ack(d:Dict[str,Any])->None:
    try:_ack_path().write_text(json.dumps(d,default=str,indent=2),encoding='utf-8')
    except Exception as e:print(f'[skill_failures] ack save failed: {e}',flush=True)
def ack_all()->Dict[str,Any]:
    """Acknowledge the current failure count as cleared — future failures will count as new."""
    s=stats();total=int(s.get('total') or 0)
    d={'ack_count':total,'ack_iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime())}
    _save_ack(d);return {'acked':total,'ack_iso':d['ack_iso']}
def unacked_count()->int:
    total=int((stats().get('total') or 0));ack=int((_load_ack().get('ack_count') or 0))
    return max(0,total-ack)
def stats()->Dict[str,Any]:
    p=_log_path()
    if not p.exists():return {'total':0,'by_skill':{},'last_iso':None,'ack_count':int((_load_ack().get('ack_count') or 0)),'unacked':0}
    counts:Dict[str,int]={};total=0;last_iso=None
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:rec=json.loads(line)
            except Exception:continue
            sk=rec.get('skill') or '?';counts[sk]=counts.get(sk,0)+1;total+=1;last_iso=rec.get('iso') or last_iso
    except Exception:pass
    ack=int((_load_ack().get('ack_count') or 0))
    return {'total':total,'by_skill':dict(sorted(counts.items(),key=lambda kv:-kv[1])[:20]),'last_iso':last_iso,'ack_count':ack,'unacked':max(0,total-ack)}
