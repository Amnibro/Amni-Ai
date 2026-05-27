"""notes — quick text capture without ceremony.
Unlike bookmarks (require a bot reply) or reminders (time-aware), notes are bare snippets.
data/notes.jsonl append-only; deletions rewrite the file."""
import json,time,uuid,re
from pathlib import Path
from typing import Dict,Any,List,Optional
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _log_path()->Path:
    p=_repo_root()/'data'/'notes.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
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
def _extract_tags(text:str)->List[str]:
    return sorted(set(m.lower() for m in re.findall(r'(?:^|\s)#([a-z][a-z0-9_-]{0,30})\b',text,re.IGNORECASE)))
def add(text:str,tags:Optional[List[str]]=None,session_id:str='')->Dict[str,Any]:
    if not (text or '').strip():return {'error':'text required'}
    nid='nt_'+uuid.uuid4().hex[:12];now=time.time()
    auto_tags=_extract_tags(text)
    if tags:auto_tags=sorted(set(auto_tags+[t.lower().lstrip('#').strip() for t in tags if str(t).strip()]))
    rec={'id':nid,'ts':now,'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(now)),'text':text.strip()[:2000],'tags':auto_tags,'session_id':(session_id or '')[:64]}
    try:
        with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
    except Exception as e:return {'error':f'write failed: {e}'}
    return rec
def list_recent(limit:int=50,tag:Optional[str]=None,search:str='',session_id:Optional[str]=None)->List[Dict[str,Any]]:
    items=_read_all()
    if session_id:items=[r for r in items if r.get('session_id')==session_id]
    if tag:t=tag.lower().lstrip('#');items=[r for r in items if t in (r.get('tags') or [])]
    if search:pat=re.compile(re.escape(search),re.IGNORECASE);items=[r for r in items if pat.search(r.get('text',''))]
    items.sort(key=lambda r:-float(r.get('ts') or 0))
    return items[:int(max(1,limit))]
def delete(nid:str)->Dict[str,Any]:
    p=_log_path()
    if not p.exists():return {'error':'no notes file'}
    items=_read_all();n_before=len(items);items=[r for r in items if r.get('id')!=nid]
    if len(items)==n_before:return {'error':f'no note with id {nid!r}'}
    try:
        with p.open('w',encoding='utf-8') as fh:
            for r in items:fh.write(json.dumps(r,default=str)+'\n')
    except Exception as e:return {'error':f'rewrite failed: {e}'}
    return {'deleted':nid,'remaining':len(items)}
def stats()->Dict[str,Any]:
    items=_read_all();tag_counts:Dict[str,int]={}
    for r in items:
        for t in r.get('tags') or []:tag_counts[t]=tag_counts.get(t,0)+1
    return {'total':len(items),'tag_counts':dict(sorted(tag_counts.items(),key=lambda kv:-kv[1])[:20]),'last_iso':items[-1].get('iso') if items else None}
def all_tags()->List[str]:
    return sorted(set(t for r in _read_all() for t in (r.get('tags') or [])))
