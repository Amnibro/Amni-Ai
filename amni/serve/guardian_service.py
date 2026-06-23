"""guardian_service — Adam's self-improvement GUARDIAN + a companion app (dispatch & discussion), mounted into the Amni-Ai FastAPI server as a first-class serve module. Reuses the shared Adam (via agent), the skills registry, and notifications — NO second model load. Routes: GET /guardian (companion app HTML), POST /guardian/ask (discussion), POST /guardian/dispatch (build+sandbox-verify a tool), GET /guardian/pull (Adam->you pushes), GET /guardian/toolkit, GET /guardian/status. A background thread self-improves when idle (propose->write->SANDBOX-verify->bank), pausing during user activity. Reach it from the phone at /guardian over LAN / meshnet / adb-reverse. Self-written code is verified in a timeout-bounded subprocess so it can never hang the server."""
import os,re,json,time,glob,uuid,threading,subprocess,tempfile,sys,ast
_TOOLS='adam_tools'
def _model(agent,prompt,system=None,max_tokens=256,do_sample=False):
    ad=getattr(agent,'adam',None)
    calls=[]
    if ad is not None:
        calls.append(lambda:ad.chat_persona(prompt,system=system or 'You are Adam.',max_new_tokens=max_tokens,do_sample=do_sample))
        calls.append(lambda:ad.chat(prompt,system=system,max_new_tokens=max_tokens,do_sample=do_sample))
        calls.append(lambda:ad.ask(((system+'\n\n') if system else '')+prompt))
    calls.append(lambda:agent.chat(((system+'\n\n') if system else '')+prompt,session_id='guardian'))
    for c in calls:
        try:
            r=c()
            if isinstance(r,dict):return (r.get('answer') or r.get('resp') or '').strip()
            if isinstance(r,(tuple,list)):return str(r[0]).strip()
            return str(r).strip()
        except Exception:continue
    return ''
def _extract(t):
    m=re.search(r'```(?:python|py)?\s*\n?(.*?)```',t or '',re.S);return (m.group(1) if m else (t or '')).strip()
def _sanitize(c):return (c or '').translate({0x200b:None,0x200c:None,0x200d:None,0xfeff:None,0x2060:None}).replace(' ',' ')
def _sandbox(code,timeout=8):
    if not code:return False,'empty'
    try:ast.parse(code)
    except SyntaxError as e:return False,f'SyntaxError line {e.lineno}: {e.msg}'
    f=tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8');f.write(code+'\n_selftest()\nprint("__OK__")\n');f.close()
    try:
        r=subprocess.run([sys.executable,f.name],capture_output=True,text=True,timeout=timeout)
        if '__OK__' in (r.stdout or '') and r.returncode==0:return True,None
        tail=(r.stderr or '').strip().splitlines();return False,(tail[-1][:160] if tail else 'selftest did not pass')
    except subprocess.TimeoutExpired:return False,f'timed out >{timeout}s (possible loop)'
    except Exception as e:return False,f'{type(e).__name__}: {e}'
    finally:
        try:os.remove(f.name)
        except OSError:pass
class _Guardian:
    def __init__(s,agent):
        s.agent=agent;os.makedirs(_TOOLS,exist_ok=True)
        s.built=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(_TOOLS,'*.py'))]
        s.pushes=[];s.feed=[];s.cycles=0;s.improved=0;s.active_until=0.0;s.lock=threading.Lock()
    def touch(s):s.active_until=time.time()+90
    def busy(s):return time.time()<s.active_until
    def discuss(s,text):
        kit=', '.join(s.built) or 'none yet'
        return _model(s.agent,text,system=f"You are Adam, a continuously self-improving guardian running on Anthony's PC and reachable from his phone. Tools YOU built (in adam_tools/): {kit}. Answer concisely and truthfully about yourself and these tools.",max_tokens=240)
    def build(s,task):
        spec=task.strip()+' Also define a function `_selftest()` with fast asserts (no filesystem, no network, no large loops) that validate it.'
        prompt='Write '+spec;code='';err='?'
        for _ in range(3):
            raw=_model(s.agent,prompt,system='You are an expert Python engineer. Output ONLY one ```python code block that fully solves the task. No prose, no markdown outside the block.',max_tokens=700)
            code=_sanitize(_extract(raw));ok,err=_sandbox(code)
            if ok:
                name='tool_'+uuid.uuid4().hex[:6]
                open(os.path.join(_TOOLS,name+'.py'),'w',encoding='utf-8').write(code)
                with s.lock:s.built.append(name);s.feed.insert(0,{'t':time.time(),'kind':'built','name':name,'task':task[:60]})
                s._register(name);return {'ok':True,'name':name,'code':code}
            prompt=f'That code FAILED: {err}\nReturn ONLY the corrected ```python code block.'
        return {'ok':False,'error':err,'code':code}
    def _register(s,name):
        try:
            reg=getattr(s.agent,'skills',None)
            if reg is not None and hasattr(reg,'register'):reg.register('guardian_'+name,lambda *a,**k:{'tool':name,'path':os.path.join(_TOOLS,name+'.py')},desc=f'Adam-built tool {name}')
        except Exception:pass
        try:
            from amni.serve import notifications as N;N.queue_notification('info','guardian','New tool',f'Adam built + verified {name}')
        except Exception:pass
    def push(s,text):
        with s.lock:s.pushes.append({'from':'adam','text':text,'t':time.time()})
    def pull(s):
        with s.lock:out=s.pushes[:];s.pushes=[];return out
    def self_improve(s):
        have=', '.join(s.built[-12:]) or 'none yet'
        idea=_model(s.agent,f'You already have tools: {have}. Propose ONE small, self-contained, stdlib-only utility to add next. Reply EXACTLY as:  NAME: <snake_name> | WHAT: <one sentence>',system='Be specific and concise.',max_tokens=48)
        nm=re.search(r'NAME:\s*([a-zA-Z_]\w*)',idea);wh=re.search(r'WHAT:\s*(.+)',idea)
        if not (nm and wh):return
        name=nm.group(1).lower()
        if name in s.built or s.busy():return
        r=s.build(f'a self-contained Python tool named {name} that: {wh.group(1).strip()[:160]}. Standard library only.')
        if r.get('ok'):s.improved+=1;s.push(f'I built + verified a new tool: {name}')
    def loop(s):
        while True:
            try:
                if not s.busy():s.cycles+=1;s.self_improve()
                time.sleep(5 if s.busy() else 25)
            except Exception:time.sleep(15)
_PAGE=r'''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Adam Guardian</title>
<style>:root{--bg:#0b0d10;--p:#13171c;--fg:#e6e8eb;--mut:#7a8088;--acc:#7fd6c5;--u:#1c5fd1;--b:#1f242a}
*{box-sizing:border-box}body{margin:0;height:100vh;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,sans-serif;display:flex;flex-direction:column}
#tabs{display:flex;border-bottom:1px solid var(--b)}#tabs button{flex:1;background:0;border:0;color:var(--mut);padding:13px;font-size:15px;border-bottom:2px solid transparent}#tabs button.on{color:var(--acc);border-bottom-color:var(--acc)}
.view{flex:1;min-height:0;display:none;flex-direction:column;overflow:hidden}.view.on{display:flex}
#log,#dout{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:9px}
.m{padding:9px 13px;border-radius:13px;max-width:84%;white-space:pre-wrap;word-wrap:break-word}.u{background:var(--u);margin-left:auto}.a{background:var(--p)}
.bar{display:flex;padding:8px;gap:8px;border-top:1px solid var(--b)}.bar input,.bar textarea{flex:1;padding:11px;border-radius:8px;border:0;background:#1b2230;color:#fff;font-size:16px;font-family:inherit;resize:none}
.bar button{padding:11px 16px;border:0;border-radius:8px;background:#2ea043;color:#fff;font-size:15px}
pre{background:#000;padding:10px;border-radius:8px;overflow:auto;font-size:12px;margin:0;font-family:Consolas,monospace}
.k{font-size:12px;color:var(--mut);padding:6px 12px}.k b{color:var(--acc)}.ok{color:#2ea043}.no{color:#d65a5a}
#feed{padding:8px 12px;font-size:12px;color:var(--mut);border-top:1px solid var(--b);max-height:30%;overflow:auto}
</style></head><body>
<div id=tabs><button class=on onclick="tab('d')">💬 Discuss</button><button onclick="tab('p')">🛠️ Dispatch</button></div>
<div id=vd class="view on"><div id=log></div><div class=bar><input id=ti placeholder="Talk to Adam..." onkeydown="if(event.key=='Enter')say()"><button onclick=say()>Send</button></div></div>
<div id=vp class=view><div id=dout><div class=k>Describe a tool and Adam will <b>write + verify</b> it. Its toolkit:<span id=kit></span></div></div><div class=bar><textarea id=tk rows=2 placeholder="a function that ..."></textarea><button onclick=disp()>Build</button></div><div id=feed></div></div>
<script>
function tab(t){for(const v of['d','p']){document.getElementById('v'+v).classList.toggle('on',v==t)}document.querySelectorAll('#tabs button').forEach((b,i)=>b.classList.toggle('on',i==(t=='d'?0:1)))}
function add(p,c,w){const d=document.createElement('div');d.className='m '+(w=='u'?'u':'a');d.textContent=c;document.getElementById(p).appendChild(d);d.parentElement.scrollTop=1e9;return d}
async function say(){const t=ti.value.trim();if(!t)return;add('log',t,'u');ti.value='';const ph=add('log','…','a');try{const r=await fetch('/guardian/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});ph.textContent=(await r.json()).resp||'(no reply)'}catch(e){ph.textContent='(err '+e+')'}}
async function disp(){const t=tk.value.trim();if(!t)return;add('dout','build: '+t,'u');tk.value='';const ph=add('dout','building + verifying…','a');try{const r=await(await fetch('/guardian/dispatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:t})})).json();ph.innerHTML=r.ok?('<span class=ok>✓ verified '+r.name+'</span><pre>'+r.code.replace(/</g,'&lt;')+'</pre>'):('<span class=no>✗ '+(r.error||'failed')+'</span>');kitload()}catch(e){ph.textContent='(err '+e+')'}}
async function kitload(){try{const k=await(await fetch('/guardian/toolkit')).json();kit.innerHTML=' <b>'+k.count+'</b> tools: '+k.tools.join(', ')}catch(e){}}
async function poll(){try{for(const m of await(await fetch('/guardian/pull')).json())add('log','📣 '+m.text,'a');const s=await(await fetch('/guardian/status')).json();feed.innerHTML='self-improvements: <b>'+s.improved+'</b> · cycles: '+s.cycles+(s.busy?' · (paused — you\'re here)':'')}catch(e){}setTimeout(poll,3000)}
kitload();poll();
</script></body></html>'''
def mount(app,agent):
    from fastapi import Request
    from fastapi.responses import HTMLResponse
    g=_Guardian(agent)
    threading.Thread(target=g.loop,daemon=True).start()
    @app.get('/guardian',response_class=HTMLResponse)
    def guardian_page():return HTMLResponse(_PAGE)
    @app.post('/guardian/ask')
    async def guardian_ask(req:Request):
        b=await req.json();g.touch();return {'kind':'chat','resp':g.discuss((b.get('text') or '').strip())}
    @app.post('/guardian/dispatch')
    async def guardian_dispatch(req:Request):
        b=await req.json();g.touch();return g.build((b.get('task') or '').strip())
    @app.get('/guardian/pull')
    def guardian_pull():return g.pull()
    @app.get('/guardian/toolkit')
    def guardian_toolkit():return {'tools':sorted(g.built),'count':len(g.built)}
    @app.get('/guardian/status')
    def guardian_status():return {'cycles':g.cycles,'improved':g.improved,'tools':len(g.built),'busy':g.busy(),'feed':g.feed[:10]}
    return g
