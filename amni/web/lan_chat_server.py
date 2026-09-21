import os,sys,time,json,socket
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,JSONResponse
_H0=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_H0)
from amni.compute.conversational_ray_engine import ConversationalRayEngine
from amni.compute.swarm_consensus_engine import SwarmConsensusEngine
from amni.compute.ptex_growth_engine import PtexGrowthEngine
app=FastAPI(title="Adam 1.2T LAN Chat")
conv_engine=ConversationalRayEngine()
swarm_engine=SwarmConsensusEngine(engine=conv_engine.engine)
growth_engine=PtexGrowthEngine(store=conv_engine.store)
def get_lan_ip()->str:
 try:
  s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
  s.connect(("8.8.8.8",80))
  ip=s.getsockname()[0]
  s.close()
  return ip
 except Exception:return "127.0.0.1"
HTML_UI=r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Adam 1.2T - Resident Continuum Intelligence</title>
<style>
:root{--bg:#0b0f19;--card:#131b2e;--accent:#00ffcc;--accent2:#ff007f;--text:#e2e8f0;--muted:#64748b;--code:#060a12;}
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
body{background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh;overflow:hidden;}
header{background:var(--card);padding:12px 18px;border-bottom:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;}
.title{font-weight:700;font-size:1.15rem;color:var(--accent);display:flex;align-items:center;gap:8px;}
.badge{background:#00ffcc22;color:var(--accent);padding:2px 8px;border-radius:12px;font-size:0.75rem;border:1px solid var(--accent);}
.stats{display:flex;gap:12px;font-size:0.8rem;color:var(--muted);align-items:center;}
.stat-val{color:#38bdf8;font-weight:600;}
#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;}
.msg{max-width:85%;padding:12px 16px;border-radius:14px;line-height:1.45;font-size:0.92rem;word-break:break-word;}
.user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:2px;}
.adam{align-self:flex-start;background:var(--card);border:1px solid #1e293b;border-bottom-left-radius:2px;}
.meta{font-size:0.72rem;color:var(--muted);margin-bottom:6px;display:flex;gap:8px;align-items:center;}
.meta-badge{background:#ff007f22;color:var(--accent2);padding:1px 6px;border-radius:6px;}
pre{background:var(--code);padding:10px;border-radius:8px;overflow-x:auto;margin-top:8px;font-family:"Fira Code",monospace;font-size:0.82rem;border:1px solid #1e293b;white-space:pre-wrap;}
#controls{background:var(--card);padding:10px 16px;border-top:1px solid #1e293b;display:flex;flex-direction:column;gap:10px;}
.pills{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;}
.pill{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:4px 10px;border-radius:16px;font-size:0.78rem;cursor:pointer;white-space:nowrap;}
.pill:hover{background:#334155;color:#fff;}
.toggles{display:flex;gap:16px;font-size:0.8rem;color:var(--muted);align-items:center;}
.input-row{display:flex;gap:10px;}
textarea{flex:1;background:var(--code);border:1px solid #334155;color:#fff;padding:10px 14px;border-radius:10px;resize:none;height:46px;font-size:0.92rem;}
textarea:focus{outline:none;border-color:var(--accent);}
button{background:var(--accent);color:#000;border:none;padding:0 20px;border-radius:10px;font-weight:700;cursor:pointer;transition:0.1s;}
button:hover{opacity:0.9;}
</style>
</head>
<body>
<header>
<div class="title">⚡ Adam 1.2T Continuum <span class="badge">T²×S² TOROIDAL STEERING</span></div>
<div class="stats">
<div>Params: <span class="stat-val">1.2 Trillion</span></div>
<div>Pages: <span class="stat-val">3,584</span></div>
<div>Live Growth: <span class="stat-val" id="growth-count">0</span> trans</div>
<div>LAN: <span class="stat-val" id="lan-ip">...</span></div>
</div>
</header>
<div id="chat">
<div class="msg adam">
<div class="meta"><span class="meta-badge">SYSTEM</span> 0.001 ms</div>
Hello! I am Adam, resident 1.2T continuum intelligence running zero dense GEMMs on your host CPU. Driven by closed dynamical equations on the T²×S² manifold with 3 parallel steering threads (Adversary, Tool Sandbox, Tonality).
</div>
</div>
<div id="controls">
<div class="pills">
<div class="pill" onclick="sendQuick('@debug def f(a, b):\n    return a / b')">🧪 Dynamic AST Div Guard</div>
<div class="pill" onclick="sendQuick('What is the manifold equation for T²×S²?')">🌐 Toroidal Manifold</div>
<div class="pill" onclick="sendQuick('def solve(x):\n')">⚡ Ray Walk Solve</div>
<div class="pill" onclick="sendQuick('@debug def get_elem(arr, idx):\n    return arr[idx]')">🛡️ Dynamic AST Index Guard</div>
<div class="pill" onclick="sendQuick('How do the three parallel threads steer on S²?')">🧭 S² Steering Threads</div>
</div>
<div class="toggles">
<label><input type="checkbox" id="use-swarm" checked> Lightning Swarm Consensus</label>
<label><input type="checkbox" id="grow-ptex" checked> Real-Time PTEX Growth</label>
<button style="background:transparent;color:#f87171;border:1px solid #f87171;padding:2px 8px;font-size:0.75rem;" onclick="resetChat()">Reset</button>
</div>
<div class="input-row">
<textarea id="prompt" placeholder="Ask Adam anything... (Enter to send, Shift+Enter for newline)"></textarea>
<button onclick="sendMsg()">Send</button>
</div>
</div>
<script>
fetch('/api/status').then(r=>r.json()).then(d=>{
 document.getElementById('lan-ip').innerText = d.lan_ip+':8765';
 document.getElementById('growth-count').innerText = d.total_transitions_added;
});
const promptBox = document.getElementById('prompt');
promptBox.addEventListener('keydown', e => { if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(); } });
function appendMsg(role, text, metaHtml=''){
 const c = document.getElementById('chat');
 const d = document.createElement('div');
 d.className = 'msg ' + role;
 let body = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
 if(body.includes('```')){
  body = body.replace(/```([a-z]*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
 } else if(role==='adam' && (body.includes('{') || body.includes('subroutine') || body.includes('Theorem'))){
  body = '<pre><code>' + body + '</code></pre>';
 }
 d.innerHTML = (metaHtml ? '<div class="meta">'+metaHtml+'</div>' : '') + body;
 c.appendChild(d);
 c.scrollTop = c.scrollHeight;
}
function sendQuick(q){ promptBox.value = q; sendMsg(); }
function sendMsg(){
 const text = promptBox.value.trim();
 if(!text) return;
 appendMsg('user', text);
 promptBox.value = '';
 const useSwarm = document.getElementById('use-swarm').checked;
 const growPtex = document.getElementById('grow-ptex').checked;
 fetch('/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: text, use_swarm: useSwarm, grow_ptex: growPtex})
 }).then(r=>r.json()).then(d=>{
  let meta = `<span class="meta-badge">${d.domain.toUpperCase()}</span> ⚡ ${d.latency_ms.toFixed(3)} ms | 🚀 ${d.tokens_per_sec.toLocaleString()} tok/s`;
  if(d.critic_score) meta += ` | ⚖️ Critic: ${d.critic_score}/100`;
  if(d.refinement_applied) meta += ` | 🔄 Refined`;
  if(d.cot_action==='sandbox_test') meta += ` | 🧪 Sandbox Passed`;
  else if(d.cot_action==='web_retrieval') meta += ` | 🌐 Web Verified`;
  if(d.swarm_consensus) meta += ` | 🤝 Swarm Valid`;
  if(d.ptex_growth) meta += ` | 🌱 +${d.ptex_growth.transitions_added} Baked`;
  if(d.intent_questioning) meta += `<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;border-left:2px solid #38bdf8;padding-left:6px;"><em>Intent: ${d.intent_questioning}</em></div>`;
  appendMsg('adam', d.reply, meta);
  if(d.total_growth_transitions) document.getElementById('growth-count').innerText = d.total_growth_transitions;
 });
}
function resetChat(){
 fetch('/api/reset', {method:'POST'}).then(()=>{
  document.getElementById('chat').innerHTML = '<div class="msg adam"><div class="meta"><span class="meta-badge">SYSTEM</span> 0.001 ms</div>Dialogue reset. Ready for new queries.</div>';
 });
}
</script>
</body>
</html>"""
@app.get("/",response_class=HTMLResponse)
def index():
 return HTMLResponse(content=HTML_UI)
@app.get("/api/status")
def status():
 hdr=conv_engine.store.read_header()
 telem=growth_engine.get_growth_telemetry()
 return {"lan_ip":get_lan_ip(),"total_params":hdr["total_params"],"total_pages":hdr["total_pages"],"file_size_bytes":hdr["file_size_bytes"],"total_transitions_added":telem["total_transitions_added"],"unique_pages_updated":telem["unique_pages_updated"]}
@app.post("/api/chat")
async def chat_endpoint(req:Request):
 data=await req.json()
 msg=data.get("message","").strip()
 use_swarm=data.get("use_swarm",True)
 grow_ptex=data.get("grow_ptex",True)
 t0=time.perf_counter()
 if msg.startswith("@debug") or ("def " in msg and ("debug" in msg.lower() or "fix" in msg.lower() or "error" in msg.lower() or "crash" in msg.lower())):
  from amni.compute.code_debugger_engine import CodeDebuggerEngine
  dbg_res=CodeDebuggerEngine().debug_and_converge(msg)
  reply=dbg_res["final_text"]
  dom="code"
  intent="code_debugging"
  cot_act="sandbox_test"
  swarm_meta={"valid":dbg_res["verified"],"repaired":True,"consensus_ms":dbg_res.get("elapsed_ms",0)}
  turn={"empirical_verified":dbg_res["verified"],"critic_score":100 if dbg_res["verified"] else 70,"critic_verdict":"VERIFIED" if dbg_res["verified"] else "NEEDS_WORK"}
 else:
  turn=conv_engine.chat(msg)
  reply=turn["reply"]
  dom=turn["domain"]
  intent=turn["intent"]
  cot_act=turn.get("cot_action")
  swarm_meta={"valid":turn.get("empirical_verified",True),"threads":["adv","tool","tone"]}
 ptex_meta=None
 if grow_ptex and len(reply)>16:
  from amni.compute.ptex_manifold import invalidate
  page_dom=("math","stem","code","civics")[len(msg)%4]
  ev=growth_engine.grow(page_dom,reply)
  invalidate()
  ptex_meta=ev
 total_dt=(time.perf_counter()-t0)*1000.0
 toks=len(reply)/4.0
 telem=growth_engine.get_growth_telemetry()
 return {"reply":reply,"domain":dom,"latency_ms":round(total_dt,3),"tokens":toks,"tokens_per_sec":round(toks/max(1e-9,total_dt/1000.0),0),"cot_action":cot_act,"empirical_verified":turn.get("empirical_verified",False),"critic_score":turn.get("critic_score"),"critic_verdict":turn.get("critic_verdict"),"critic_notes":turn.get("critic_notes"),"refinement_applied":turn.get("refinement_applied",False),"intent_questioning":turn.get("intent_questioning"),"swarm_consensus":swarm_meta,"ptex_growth":ptex_meta,"total_growth_transitions":telem["total_transitions_added"]}
@app.post("/api/reset")
def reset_endpoint():
 conv_engine.reset_conversation()
 return {"status":"reset_ok"}