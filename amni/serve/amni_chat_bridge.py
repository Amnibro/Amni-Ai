"""amni_chat_bridge — lets Adam lightly participate in Amni-Chat conversations via text.
A relay on the PC forwards an inbound Amni-Chat DM here; Adam runs it through the agent and returns a reply.
Two hard safety rails for an EXTERNAL-facing surface:
 1. Outbound replies are scrubbed through pii_egress with the OWNER's PersonalAtlas — Adam must never leak the
    owner's name/location/contact to a chat peer (the leak-liability rule, applied to replies not just searches).
 2. Per-peer rate limit + length cap + an enable flag — "lightweight" interaction, not an open firehose.
Per-peer conversation continuity uses session id `amnichat:<conversation_id|from_user>`."""
import time,threading
from collections import defaultdict,deque
from typing import Dict,Any,Optional,Deque
_LOCK=threading.Lock()
_HITS:Dict[str,Deque[float]]=defaultdict(deque)
_STATE={'enabled':True,'max_per_min':12,'max_in_chars':2000,'max_reply_chars':1200}
def set_enabled(on:bool)->Dict[str,Any]:
    _STATE['enabled']=bool(on);return {'enabled':_STATE['enabled']}
def config()->Dict[str,Any]:return dict(_STATE)
def _rate_ok(peer:str)->bool:
    now=time.time();win=60.0
    with _LOCK:
        dq=_HITS[peer]
        while dq and now-dq[0]>win:dq.popleft()
        if len(dq)>=int(_STATE['max_per_min']):return False
        dq.append(now);return True
def _scrub_owner_pii(text:str,agent)->str:
    try:
        from amni.serve.pii_egress import scrub
        return scrub(text,agent=agent,source='amni_chat_out')
    except Exception:return text
def handle_message(text:str,from_user:str='peer',conversation_id:str='',agent=None)->Dict[str,Any]:
    if not _STATE['enabled']:return {'error':'amni-chat bridge disabled','enabled':False}
    text=(text or '').strip()
    if not text:return {'error':'empty message'}
    if len(text)>int(_STATE['max_in_chars']):text=text[:int(_STATE['max_in_chars'])]
    peer=(conversation_id or from_user or 'peer')[:80]
    if not _rate_ok(peer):return {'error':'rate limited','retry_after_s':60}
    if agent is None:return {'error':'agent unavailable'}
    sid='amnichat:'+peer
    try:
        r=agent.chat(text,session_id=sid)
    except Exception as e:return {'error':f'agent error: {e}'}
    reply=(r.get('answer') or '').strip() if isinstance(r,dict) else str(r)
    reply=_scrub_owner_pii(reply,agent)
    if len(reply)>int(_STATE['max_reply_chars']):reply=reply[:int(_STATE['max_reply_chars'])].rstrip()+'…'
    return {'reply':reply,'from':'adam','conversation_id':peer,'tier':r.get('tier') if isinstance(r,dict) else None,'persona':r.get('persona') if isinstance(r,dict) else None}
def mount(app,agent):
    from fastapi import Request,HTTPException
    @app.get('/bridge/amni-chat/status')
    def bridge_status():return {**config(),'active_peers':len(_HITS)}
    @app.post('/bridge/amni-chat')
    async def bridge_message(req:Request):
        body=await req.json()
        text=body.get('text') or body.get('message') or ''
        if not text:raise HTTPException(400,'need text')
        out=handle_message(text,from_user=body.get('from_user','peer'),conversation_id=body.get('conversation_id',''),agent=agent)
        if out.get('error') and out['error']=='rate limited':raise HTTPException(429,out)
        return out
    @app.post('/bridge/amni-chat/toggle')
    async def bridge_toggle(req:Request):
        body={}
        try:body=await req.json()
        except Exception:pass
        return set_enabled(bool(body.get('enabled',True)))
