"""reminders — simple time-aware todo store.
data/reminders.jsonl: append-only event log of {ts, event:added|dismissed, id, text, due_at?, ...}.
Active set is reconstructed by replay. Due reminders surface via list_due() for proactive notification."""
import json,time,uuid,re
from pathlib import Path
from typing import Dict,Any,List,Optional
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _log_path()->Path:
    p=_repo_root()/'data'/'reminders.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def _append(rec:Dict[str,Any])->None:
    try:
        with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception as e:print(f'[reminders] write failed: {e}',flush=True)
def _read_events()->List[Dict[str,Any]]:
    p=_log_path()
    if not p.exists():return []
    out=[]
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:out.append(json.loads(line))
            except Exception:continue
    except Exception:return []
    return out
def _replay_active()->Dict[str,Dict[str,Any]]:
    state:Dict[str,Dict[str,Any]]={}
    for ev in _read_events():
        if ev.get('event')=='added':state[ev.get('id')]={'id':ev.get('id'),'ts_added':ev.get('ts'),'iso_added':ev.get('iso'),'text':ev.get('text',''),'due_at':ev.get('due_at'),'due_iso':ev.get('due_iso'),'session_id':ev.get('session_id','')}
        elif ev.get('event') in ('dismissed','deleted'):state.pop(ev.get('id'),None)
    return state
_DUE_PATTERNS=[
    (re.compile(r'\bin\s+(\d+)\s*(?:m|min|mins|minute|minutes)\b',re.IGNORECASE),lambda m,now:now+int(m.group(1))*60),
    (re.compile(r'\bin\s+(\d+)\s*(?:h|hr|hrs|hour|hours)\b',re.IGNORECASE),lambda m,now:now+int(m.group(1))*3600),
    (re.compile(r'\bin\s+(\d+)\s*(?:d|day|days)\b',re.IGNORECASE),lambda m,now:now+int(m.group(1))*86400),
    (re.compile(r'\bin\s+a\s+(?:half\s+)?hour\b',re.IGNORECASE),lambda m,now:now+(1800 if 'half' in m.group(0).lower() else 3600)),
    (re.compile(r'\btomorrow\b',re.IGNORECASE),lambda m,now:_at_time(now+86400,9,0)),
    (re.compile(r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b',re.IGNORECASE),lambda m,now:_parse_at(m,now)),
]
def _at_time(ts:float,hour:int,minute:int)->float:
    import datetime as _dt
    dt=_dt.datetime.fromtimestamp(ts).replace(hour=hour,minute=minute,second=0,microsecond=0)
    return dt.timestamp()
def _parse_at(m,now:float)->float:
    hour=int(m.group(1));minute=int(m.group(2) or 0);ampm=(m.group(3) or '').lower()
    if ampm=='pm' and hour<12:hour+=12
    if ampm=='am' and hour==12:hour=0
    today=_at_time(now,hour,minute)
    if today<=now+30:today+=86400
    return today
def parse_due_at(text:str,now:Optional[float]=None)->Optional[float]:
    """Parse a natural-language due time out of the reminder text. Returns unix ts or None."""
    if not text:return None
    now=now if now is not None else time.time()
    for pat,fn in _DUE_PATTERNS:
        m=pat.search(text)
        if m:
            try:return float(fn(m,now))
            except Exception:continue
    return None
def add(text:str,due_at:Optional[float]=None,session_id:str='')->Dict[str,Any]:
    if not (text or '').strip():return {'error':'text required'}
    rid='rm_'+uuid.uuid4().hex[:12];now=time.time()
    if due_at is None:due_at=parse_due_at(text,now=now)
    rec={'event':'added','id':rid,'ts':now,'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(now)),'text':text.strip()[:500],'due_at':due_at,'due_iso':(time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(due_at)) if due_at else None),'session_id':(session_id or '')[:64]}
    _append(rec);return {'id':rid,'text':rec['text'],'due_at':due_at,'due_iso':rec['due_iso'],'iso_added':rec['iso']}
def dismiss(rid:str)->Dict[str,Any]:
    if rid not in _replay_active():return {'error':f'unknown reminder {rid!r}'}
    _append({'event':'dismissed','id':rid,'ts':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime())})
    return {'dismissed':rid}
def list_active(session_id:Optional[str]=None,limit:int=50)->List[Dict[str,Any]]:
    state=list(_replay_active().values())
    if session_id:state=[r for r in state if r.get('session_id')==session_id]
    state.sort(key=lambda r:(float(r.get('due_at') or 9e18),-float(r.get('ts_added') or 0)))
    return state[:int(max(1,limit))]
def list_due(now:Optional[float]=None,grace_s:int=60)->List[Dict[str,Any]]:
    now=now if now is not None else time.time()
    return [r for r in _replay_active().values() if r.get('due_at') and float(r['due_at'])<=now+grace_s]
def stats()->Dict[str,Any]:
    active=_replay_active();due=list_due()
    return {'active':len(active),'due_now':len(due),'with_due':sum(1 for r in active.values() if r.get('due_at'))}
