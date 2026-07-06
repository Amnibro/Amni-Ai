"""amni_chat_bridge — lets Adam lightly participate in Amni-Chat conversations via text.
A relay on the PC forwards an inbound Amni-Chat DM here; Adam runs it through the agent and returns a reply.
Two hard safety rails for an EXTERNAL-facing surface:
 1. Outbound replies are scrubbed through pii_egress with the OWNER's PersonalAtlas — Adam must never leak the
    owner's name/location/contact to a chat peer (the leak-liability rule, applied to replies not just searches).
 2. Per-peer rate limit + length cap + an enable flag — "lightweight" interaction, not an open firehose.
Per-peer conversation continuity uses session id `amnichat:<conversation_id|from_user>`."""
import time,threading,re,os
from collections import defaultdict,deque
from typing import Dict,Any,Optional,Deque
_LOCK=threading.Lock()
_HITS:Dict[str,Deque[float]]=defaultdict(deque)
_LAST_IMG:Dict[str,Dict[str,Any]]={}
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
def handle_image_frame(frame:str,client,from_user:str='peer',agent=None)->Dict[str,Any]:
    if not _STATE['enabled']:return {'error':'amni-chat bridge disabled'}
    peer=(from_user or 'peer')[:80]
    if not _rate_ok(peer):return {'error':'rate limited'}
    try:att=client.fetch_attachment(frame)
    except Exception:att=None
    if not att or not str(att.get('mime','')).startswith('image/'):return {'error':'not an image'}
    if len(att['data'])>10*1024*1024:return {'error':'image too large'}
    vision=getattr(agent,'vision',None)
    if vision is None or not vision.is_available():return {'reply':'📷 Got your image! My vision module is offline right now though — ask Anthony to enable it.','from':'adam','conversation_id':peer,'tier':'skill:vision_offline'}
    d=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'json','charts','inbox');os.makedirs(d,exist_ok=True)
    ext={'image/png':'png','image/jpeg':'jpg','image/jpg':'jpg','image/webp':'webp','image/gif':'gif'}.get(str(att.get('mime','')).lower(),'img')
    p=os.path.join(d,f"{peer[:8]}_{int(time.time())}.{ext}")
    open(p,'wb').write(att['data'])
    try:r=vision.describe(att['data'])
    except Exception as e:r={'error':str(e)[:150]}
    if not isinstance(r,dict) or r.get('error'):return {'reply':f"📷 Got your image but couldn't analyze it — {(r or {}).get('error','vision error')}",'from':'adam','conversation_id':peer,'tier':'skill:vision_error'}
    _LAST_IMG[peer]={'path':p,'ts':time.time()}
    cap=str(r.get('caption') or '').strip() or 'something I could not quite make out'
    return {'reply':f"📷 I see {cap}. Ask me about it — e.g. \"what color is it in the photo?\"",'from':'adam','conversation_id':peer,'tier':'skill:vision_caption'}
def handle_message(text:str,from_user:str='peer',conversation_id:str='',agent=None)->Dict[str,Any]:
    if not _STATE['enabled']:return {'error':'amni-chat bridge disabled','enabled':False}
    text=(text or '').strip()
    if not text:return {'error':'empty message'}
    if len(text)>int(_STATE['max_in_chars']):text=text[:int(_STATE['max_in_chars'])]
    peer=(conversation_id or from_user or 'peer')[:80]
    if not _rate_ok(peer):return {'error':'rate limited','retry_after_s':60}
    try:
        from amni.serve.skills import chart_command as _chartcmd,_render_azno_charts as _renderchart
        _cc=_chartcmd(text)
    except Exception:_cc=None
    if _cc:
        try:_r=_renderchart(_cc[0],_cc[1],want_options=_cc[2])
        except Exception:_r={'ok':False}
        if _r.get('ok'):
            if agent is not None and hasattr(agent,'_active_sym'):agent._active_sym['amnichat:'+peer]=(_cc[0],_cc[1])
            _cap=('🧮 '+_r['summary']) if _r.get('summary') else f"📈 {_cc[0]} • {_cc[1]}"
            return {'reply':_cap[:int(_STATE['max_reply_chars'])],'images':_r.get('images') or [],'from':'azno','conversation_id':peer,'tier':'skill:chart'}
    try:
        from amni.serve.skills import options_command as _optcmd,_fmt_options as _optfmt,_skill_options as _optsk
        _oc=_optcmd(text)
    except Exception:_oc=None
    if _oc:
        if agent is not None and hasattr(agent,'_active_sym'):agent._active_sym['amnichat:'+peer]=(_oc[0],_oc[1])
        return {'reply':_optfmt(_optsk({'ticker':_oc[0],'tf':_oc[1]},{},None)),'from':'azno','conversation_id':peer,'tier':'skill:options'}
    if agent is None:return {'error':'agent unavailable'}
    _li=_LAST_IMG.get(peer)
    if _li and time.time()-_li['ts']<600 and re.search(r'(?i)\b(photo|image|pic|picture|screenshot|img)\b',text):
        _v=getattr(agent,'vision',None)
        if _v is not None and _v.is_available():
            try:
                _vr=_v.caption_with_question(open(_li['path'],'rb').read(),text)
                if isinstance(_vr,dict) and _vr.get('answer'):return {'reply':f"📷 {str(_vr['answer']).strip()}",'from':'adam','conversation_id':peer,'tier':'skill:vision_vqa'}
            except Exception:pass
    sid='amnichat:'+peer
    try:
        r=agent.chat(text,session_id=sid,brief=True)
    except Exception as e:return {'error':f'agent error: {e}'}
    reply=(r.get('answer') or '').strip() if isinstance(r,dict) else str(r)
    reply=_scrub_owner_pii(reply,agent)
    if len(reply)>int(_STATE['max_reply_chars']):reply=reply[:int(_STATE['max_reply_chars'])].rstrip()+'…'
    imgs=list((r.get('images') or [])[:4]) if isinstance(r,dict) else []
    try:
        from amni.serve.widgets import render_weather_card as _wxcard
        for _sc in ((r.get('skill_calls') or []) if isinstance(r,dict) else []):
            _o=(_sc.get('result') or {}).get('output') if isinstance(_sc.get('result'),dict) else None
            if _sc.get('skill')=='weather' and isinstance(_o,dict) and _o.get('temp_c') is not None:
                _p=_wxcard(_o)
                if _p:imgs.append(_p);break
    except Exception:pass
    base={'reply':reply,'from':'adam','conversation_id':peer,'tier':r.get('tier') if isinstance(r,dict) else None,'persona':r.get('persona') if isinstance(r,dict) else None}
    return {**base,'images':imgs} if imgs else base
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
