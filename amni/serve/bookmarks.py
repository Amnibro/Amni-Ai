"""bookmarks — star important bot replies for later retrieval.
Append-only data/bookmarks.jsonl. Each record: {id, ts, iso, session_id, user_msg, bot_msg, note?, tier?, persona?}.
DELETE rewrites the file without the matching id (rare op, fine for an interactive bookmark list)."""
import json,time,uuid,re
from pathlib import Path
from typing import Dict,Any,List,Optional
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _log_path()->Path:
    p=_repo_root()/'data'/'bookmarks.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def add(session_id:str,user_msg:str,bot_msg:str,note:str='',tier:str='',persona:str='')->Dict[str,Any]:
    bid='bm_'+uuid.uuid4().hex[:12];now=time.time()
    rec={'id':bid,'ts':now,'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(now)),'session_id':(session_id or '')[:64],'user_msg':(user_msg or '').strip()[:600],'bot_msg':(bot_msg or '').strip()[:2400],'note':(note or '').strip()[:400],'tier':(tier or '')[:60],'persona':(persona or '')[:32]}
    try:
        with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception as e:return {'error':f'write failed: {e}'}
    return rec
def _read_all()->List[Dict[str,Any]]:
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
def list_recent(limit:int=20,session_id:Optional[str]=None,search:str='')->List[Dict[str,Any]]:
    items=_read_all()
    if session_id:items=[r for r in items if r.get('session_id')==session_id]
    if search:
        pat=re.compile(re.escape(search),re.IGNORECASE)
        items=[r for r in items if pat.search(r.get('user_msg','')) or pat.search(r.get('bot_msg','')) or pat.search(r.get('note',''))]
    items.sort(key=lambda r:-float(r.get('ts') or 0))
    return items[:int(max(1,limit))]
def delete(bid:str)->Dict[str,Any]:
    p=_log_path()
    if not p.exists():return {'error':'no bookmarks file'}
    items=_read_all();n_before=len(items)
    items=[r for r in items if r.get('id')!=bid]
    if len(items)==n_before:return {'error':f'no bookmark with id {bid!r}'}
    try:
        with p.open('w',encoding='utf-8') as fh:
            for r in items:fh.write(json.dumps(r,default=str)+'\n')
    except Exception as e:return {'error':f'rewrite failed: {e}'}
    return {'deleted':bid,'remaining':len(items)}
def stats()->Dict[str,Any]:
    items=_read_all()
    by_persona:Dict[str,int]={}
    for r in items:
        p=r.get('persona') or 'unknown';by_persona[p]=by_persona.get(p,0)+1
    last_iso=items[-1].get('iso') if items else None
    return {'total':len(items),'by_persona':by_persona,'last_iso':last_iso}
