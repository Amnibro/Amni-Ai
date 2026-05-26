"""self_improvement — Adam's proposal log + state machine for self-directed evolution.
Adam (or the user) writes proposals describing intended changes. Each proposal moves through states:
  proposed -> attempted -> validated -> deployed   (happy path)
  proposed -> declined                              (user vetoed or Adam realized infeasible)
  attempted -> reverted                             (didn't validate, rolled back)
Every transition is appended to data/self_improvement.jsonl (no overwrites — full audit trail). The
log doubles as Adam's memory of what's been tried: future inspections read it and avoid re-proposing
the same thing.

This is intentionally NOT autonomous execution — Adam writes proposals, but actual code edits flow
through the existing file_write+code_edit+verify+pytest pipeline (v6.10.16/19). The log is a notebook,
not a robot arm."""
import json,time,uuid
from pathlib import Path
from typing import Dict,Any,List,Optional
_VALID_STATUSES={'proposed','attempted','validated','deployed','declined','reverted'}
_VALID_CATEGORIES={'enhancement','bug-fix','refactor','documentation','performance','security','experiment'}
def _data_dir()->Path:
    base=Path(__file__).resolve().parents[2]/'data';base.mkdir(parents=True,exist_ok=True);return base
def _log_path()->Path:return _data_dir()/'self_improvement.jsonl'
def propose(title:str,rationale:str,planned_change:str,files_touched:Optional[List[str]]=None,category:str='enhancement',author:str='adam')->Dict[str,Any]:
    """Record a new proposal. Returns the proposal dict (with id)."""
    if not (title or '').strip():return {'error':'title required'}
    if not (rationale or '').strip():return {'error':'rationale required'}
    if category not in _VALID_CATEGORIES:return {'error':f'category must be one of: {sorted(_VALID_CATEGORIES)}'}
    pid=f'si_{uuid.uuid4().hex[:12]}';now=time.time()
    rec={'id':pid,'ts':now,'event':'propose','title':title.strip()[:200],'rationale':rationale.strip()[:2000],'planned_change':planned_change.strip()[:4000],'files_touched':list(files_touched or [])[:30],'category':category,'author':author,'status':'proposed'}
    _append(rec);return rec
def transition(proposal_id:str,new_status:str,notes:str='',author:str='adam')->Dict[str,Any]:
    if new_status not in _VALID_STATUSES:return {'error':f'status must be one of: {sorted(_VALID_STATUSES)}'}
    cur=get_proposal(proposal_id)
    if cur is None:return {'error':f'unknown proposal {proposal_id!r}'}
    rec={'id':proposal_id,'ts':time.time(),'event':'transition','status':new_status,'notes':(notes or '').strip()[:1000],'author':author,'prev_status':cur.get('status')}
    _append(rec);return rec
def list_proposals(status:Optional[str]=None,category:Optional[str]=None,limit:int=50,include_history:bool=False)->List[Dict[str,Any]]:
    proposals=_replay()
    if status:proposals=[p for p in proposals if p.get('status')==status]
    if category:proposals=[p for p in proposals if p.get('category')==category]
    proposals.sort(key=lambda p:-float(p.get('ts') or 0))
    if not include_history:
        for p in proposals:p.pop('history',None)
    return proposals[:limit]
def get_proposal(proposal_id:str)->Optional[Dict[str,Any]]:
    for p in _replay():
        if p.get('id')==proposal_id:return p
    return None
def stats()->Dict[str,Any]:
    by_status={};by_category={};total=0
    for p in _replay():
        total+=1;s=p.get('status','?');c=p.get('category','?')
        by_status[s]=by_status.get(s,0)+1;by_category[c]=by_category.get(c,0)+1
    return {'total':total,'by_status':by_status,'by_category':by_category,'open':sum(by_status.get(s,0) for s in ('proposed','attempted'))}
def _append(rec:Dict[str,Any]):
    try:
        with open(_log_path(),'a',encoding='utf-8') as f:f.write(json.dumps(rec,default=str)+'\n')
    except Exception as e:print(f'[self_improvement] append failed: {e}',flush=True)
def _replay()->List[Dict[str,Any]]:
    """Replay the jsonl into the current state of every proposal."""
    p=_log_path()
    if not p.exists():return []
    proposals:Dict[str,Dict[str,Any]]={}
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:rec=json.loads(ln)
            except Exception:continue
            pid=rec.get('id');
            if not pid:continue
            if rec.get('event')=='propose':
                cur=dict(rec);cur.pop('event',None);cur['history']=[{'ts':rec['ts'],'status':'proposed','notes':''}]
                proposals[pid]=cur
            elif rec.get('event')=='transition' and pid in proposals:
                proposals[pid]['status']=rec.get('status',proposals[pid].get('status'))
                proposals[pid]['history'].append({'ts':rec.get('ts'),'status':rec.get('status'),'notes':rec.get('notes','')})
    except Exception as e:print(f'[self_improvement] replay failed: {e}',flush=True)
    return list(proposals.values())
