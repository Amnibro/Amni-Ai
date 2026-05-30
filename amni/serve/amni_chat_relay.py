"""amni_chat_relay — connects Adam to a live Amni-Chat server (Tier 5a wiring).
Registers an identity, polls /inbox/drain, routes each decrypted DM through the existing amni_chat_bridge.handle_message
(PII-scrubbed, rate-limited, persona-aware), and sends the reply back end-to-end-encrypted. Runs in a daemon thread.
Crypto/protocol live in the vendored amni_chat_client (X25519+HKDF+ChaCha20-Poly1305, wire-compatible with the app)."""
import threading,os
from amni.serve import amni_chat_bridge as bridge
from amni.serve.amni_chat_client import Identity,AmniChatClient,run_relay
_STATE={'thread':None,'stop':False,'client':None,'ed':None,'x':None,'err':None,'connected':False,'server':None}
def _default_id_path():
    d=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'json');os.makedirs(d,exist_ok=True);return os.path.join(d,'amni_chat_identity.json')
def status():
    return {'running':bool(_STATE['thread'] and _STATE['thread'].is_alive()),'connected':_STATE['connected'],'ed':_STATE['ed'],'server':_STATE['server'],'error':_STATE['err'],**bridge.config()}
def profile(identity_path=None):
    idn=Identity.load_or_create(identity_path or _default_id_path());return {'ed':idn.ed_hex,'x':idn.x_hex,'link':f'amni://contact?p=adam&ed={idn.ed_hex}&x={idn.x_hex}&fp=&np='}
def _reply_for(agent,item):
    out=bridge.handle_message(item.get('text') or '',from_user=item.get('from_ed','peer'),conversation_id=item.get('from_ed',''),agent=agent)
    return None if (not isinstance(out,dict) or out.get('error')) else out.get('reply')
def start(agent,server_url,identity_path=None,push_token='',interval=3.0):
    if _STATE['thread'] and _STATE['thread'].is_alive():return status()
    idn=Identity.load_or_create(identity_path or _default_id_path());_STATE.update({'ed':idn.ed_hex,'x':idn.x_hex,'server':server_url,'stop':False,'err':None})
    c=AmniChatClient(server_url,idn);_STATE['client']=c
    def _run():
        try:c.register(platform='amni-ai',push_token=push_token);_STATE['connected']=True
        except Exception as e:_STATE['err']=f'register failed: {e}';_STATE['connected']=False
        run_relay(c,lambda it:_reply_for(agent,it),interval=interval,should_stop=lambda:_STATE['stop'],on_error=lambda e:_STATE.__setitem__('err',str(e)))
    t=threading.Thread(target=_run,daemon=True,name='amni-chat-relay');t.start();_STATE['thread']=t;return status()
def stop():
    _STATE['stop']=True;return {'stopping':True}
def mount(app,agent):
    from fastapi import Request
    @app.get('/bridge/amni-chat/relay/status')
    def relay_status():return status()
    @app.get('/bridge/amni-chat/relay/profile')
    def relay_profile():return profile()
    @app.post('/bridge/amni-chat/relay/start')
    async def relay_start(req:Request):
        body={}
        try:body=await req.json()
        except Exception:pass
        url=body.get('server_url') or os.environ.get('AMNI_CHAT_SERVER') or 'https://chat.example.com'
        return start(agent,url,push_token=body.get('push_token',''),interval=float(body.get('interval',3.0)))
    @app.post('/bridge/amni-chat/relay/stop')
    async def relay_stop():return stop()
