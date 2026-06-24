"""Agent operating modes — plan / edit / autonomous (bypass), like Claude Code. Gates what Adam may DO with a proposed action by (action_type, risk): PLAN = describe-only (read freely, never mutate); EDIT = act but require confirmation on writes/shell/network/destructive; AUTONOMOUS = bypass confirmations (still guards truly destructive deletes unless force). A ModeGate.decide() returns 'execute' | 'confirm' | 'describe' | 'block'. Process-global current mode + a mount() to GET/POST /mode. Reused by the agent before any side-effecting action."""
import threading
_MODES=('plan','edit','autonomous')
_RISK_ORDER={'low':0,'medium':1,'high':2,'critical':3}
_MATRIX={
 'plan':{'read':'execute','default':'describe'},
 'edit':{'read':'execute','write':'confirm','edit':'confirm','shell':'confirm','network':'confirm','install':'confirm','delete':'confirm','default':'confirm'},
 'autonomous':{'read':'execute','write':'execute','edit':'execute','shell':'execute','network':'execute','install':'execute','delete':'confirm','default':'execute'},
}
_DESC={'plan':'Plan — propose a plan, never mutate. Reads allowed; all changes are described, not executed.','edit':'Edit — act on changes but confirm writes/shell/network/deletes first.','autonomous':'Autonomous (bypass) — execute freely; only truly destructive deletes still confirm (use force to bypass).'}
_STATE={'mode':'edit'}
_LOCK=threading.Lock()
def set_mode(mode:str)->dict:
    mode=(mode or '').strip().lower()
    if mode not in _MODES:return {'error':f'unknown mode {mode!r}; valid: {list(_MODES)}','mode':_STATE['mode']}
    with _LOCK:_STATE['mode']=mode
    return {'mode':mode,'description':_DESC[mode]}
def get_mode()->dict:return {'mode':_STATE['mode'],'description':_DESC[_STATE['mode']],'modes':[{'name':m,'description':_DESC[m]} for m in _MODES]}
class ModeGate:
    def __init__(s,mode:str=None):s.mode=(mode or _STATE['mode'])
    def decide(s,action_type:str,risk:str='low',force:bool=False)->str:
        at=(action_type or 'default').strip().lower();m=_MATRIX.get(s.mode,_MATRIX['edit'])
        d=m.get(at,m['default'])
        if d=='describe':return d
        if force and d=='confirm':return 'execute'
        if _RISK_ORDER.get((risk or 'low').lower(),0)>=2 and d=='execute' and s.mode!='autonomous':d='confirm'
        return d
    def allowed(s,action_type:str,risk:str='low',force:bool=False)->bool:return s.decide(action_type,risk,force)=='execute'
    def gate(s,action_type:str,risk:str='low',force:bool=False)->dict:
        d=s.decide(action_type,risk,force)
        return {'mode':s.mode,'action':action_type,'risk':risk,'decision':d,'will_execute':d=='execute','needs_confirm':d=='confirm','plan_only':d=='describe','blocked':d=='block'}
def mount(app,agent=None):
    from fastapi import Request
    @app.get('/agent/mode')
    def _get():return get_mode()
    @app.post('/agent/mode')
    async def _set(req:Request):
        b=await req.json();return set_mode(b.get('mode',''))
    @app.post('/agent/mode/gate')
    async def _gate(req:Request):
        b=await req.json();return ModeGate(b.get('mode')).gate(b.get('action','default'),b.get('risk','low'),bool(b.get('force')))
    return app
