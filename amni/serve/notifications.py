"""notifications — proactive toast queue for /jarvis.
Sources (LearningDaemon, edit_verifier, shell_audit, etc.) call queue_notification(). /jarvis polls
/notifications every 10s and slides toasts in at bottom-right. In-memory ring buffer (50-entry cap)
so the queue can never grow unbounded; oldest get evicted. Each notification has a TTL after which
it auto-disappears from list_active even if unread."""
import time,threading,uuid
from typing import Dict,Any,List,Optional
_LOCK=threading.RLock()
_QUEUE:List[Dict[str,Any]]=[]
_MAX_ENTRIES=50
_RECENT_DEDUP_S=60.0
def queue_notification(level:str,source:str,title:str,body:str='',ttl_s:float=300.0,**extras)->Optional[str]:
    """Push a toast onto the queue. Returns the notification id, or None if a near-duplicate within ttl was already queued."""
    now=time.time();nid=f'n_{uuid.uuid4().hex[:12]}'
    with _LOCK:
        for old in _QUEUE[-12:]:
            if old.get('source')==source and old.get('title')==title and (now-old.get('ts',0))<_RECENT_DEDUP_S:return None
        entry={'id':nid,'ts':now,'level':(level or 'info').lower(),'source':source,'title':title,'body':body or '','ttl_s':float(ttl_s),'read':False,'extras':dict(extras) if extras else {}}
        _QUEUE.append(entry)
        if len(_QUEUE)>_MAX_ENTRIES:del _QUEUE[:len(_QUEUE)-_MAX_ENTRIES]
    return nid
def list_active(limit:int=20,include_read:bool=False)->List[Dict[str,Any]]:
    now=time.time();out=[]
    with _LOCK:
        for n in reversed(_QUEUE):
            if not include_read and n.get('read'):continue
            age=now-n.get('ts',0);ttl=n.get('ttl_s',300)
            if age>ttl:continue
            entry=dict(n);entry['age_s']=round(age,1)
            out.append(entry)
            if len(out)>=limit:break
    return out
def mark_read(notif_id:str)->bool:
    with _LOCK:
        for n in _QUEUE:
            if n.get('id')==notif_id:n['read']=True;return True
    return False
def mark_all_read()->int:
    with _LOCK:
        n=0
        for x in _QUEUE:
            if not x.get('read'):x['read']=True;n+=1
        return n
def stats()->Dict[str,Any]:
    now=time.time();total=0;unread=0;active=0;by_source={};by_level={}
    with _LOCK:
        for x in _QUEUE:
            total+=1;age=now-x.get('ts',0);ttl=x.get('ttl_s',300)
            if age>ttl:continue
            active+=1
            if not x.get('read'):unread+=1
            s=x.get('source','?');by_source[s]=by_source.get(s,0)+1
            lv=x.get('level','info');by_level[lv]=by_level.get(lv,0)+1
    return {'total':total,'active':active,'unread':unread,'by_source':by_source,'by_level':by_level}
def clear():
    with _LOCK:_QUEUE.clear()
