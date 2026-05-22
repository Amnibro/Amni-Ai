"""Jarvis-themed Adam UI at /jarvis — neural-net particle background, neon glow, inline widget cards.
Calls /v1/chat/completions (widget-aware OpenAI compat) so amni_widgets[] cards render inline next to chat bubbles. Voice in (Web Speech API) + voice out (speechSynthesis). Tactical-overlay aesthetic.
Mount: jarvis_web.mount(app) — adds GET /jarvis."""
_HTML=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adam — Jarvis Mode</title>
<style>
:root{--bg:#040711;--bg2:#0a1224;--glass:rgba(10,18,36,.55);--cyan:#00e5ff;--cyan2:#00b8d4;--magenta:#ff2bd6;--gold:#ffd770;--ok:#00ff9d;--err:#ff5577;--fg:#dff6ff;--mute:#5e7a99;font-family:"JetBrains Mono","SF Mono",Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--fg);overflow:hidden}
#nebula{position:fixed;inset:0;z-index:-2;background:radial-gradient(ellipse at 20% 30%,rgba(0,229,255,.12),transparent 50%),radial-gradient(ellipse at 80% 70%,rgba(255,43,214,.08),transparent 50%),var(--bg)}
#netcanvas{position:fixed;inset:0;z-index:-1;width:100vw;height:100vh}
#frame{position:fixed;inset:12px;border:1px solid rgba(0,229,255,.18);border-radius:6px;pointer-events:none;z-index:5;box-shadow:inset 0 0 30px rgba(0,229,255,.05)}
#frame::before,#frame::after{content:'';position:absolute;width:24px;height:24px;border:2px solid var(--cyan);box-shadow:0 0 12px var(--cyan)}
#frame::before{top:-2px;left:-2px;border-right:none;border-bottom:none}
#frame::after{bottom:-2px;right:-2px;border-left:none;border-top:none}
#corner-tr,#corner-bl{position:fixed;width:24px;height:24px;border:2px solid var(--cyan);box-shadow:0 0 12px var(--cyan);pointer-events:none;z-index:5}
#corner-tr{top:10px;right:10px;border-left:none;border-bottom:none}
#corner-bl{bottom:10px;left:10px;border-right:none;border-top:none}
#scanline{position:fixed;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);box-shadow:0 0 12px var(--cyan);pointer-events:none;animation:scan 6s linear infinite;opacity:.4;z-index:4}
@keyframes scan{0%{top:0}100%{top:100vh}}
#app{position:relative;z-index:10;display:grid;grid-template-rows:auto 1fr auto;height:100vh;padding:28px 32px;gap:18px}
header{display:flex;align-items:center;gap:14px;font-size:13px}
.title{font-size:20px;font-weight:700;letter-spacing:.18em;color:var(--cyan);text-shadow:0 0 10px var(--cyan),0 0 20px rgba(0,229,255,.4)}
.status{display:flex;gap:14px;margin-left:auto;font-size:11px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase}
.status .pill{padding:3px 10px;border:1px solid rgba(0,229,255,.3);border-radius:99px;background:rgba(0,229,255,.05);color:var(--cyan);box-shadow:inset 0 0 8px rgba(0,229,255,.1)}
.status .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);margin-right:6px;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
#chat-wrap{position:relative;overflow:hidden;border:1px solid rgba(0,229,255,.15);border-radius:4px;background:var(--glass);backdrop-filter:blur(8px);box-shadow:inset 0 0 30px rgba(0,229,255,.04)}
#log{height:100%;overflow-y:auto;padding:24px 28px;display:flex;flex-direction:column;gap:18px;scroll-behavior:smooth}
#log::-webkit-scrollbar{width:6px}
#log::-webkit-scrollbar-track{background:transparent}
#log::-webkit-scrollbar-thumb{background:rgba(0,229,255,.25);border-radius:3px}
.welcome{text-align:center;padding:48px 24px;color:var(--mute)}
.welcome h2{font-size:14px;letter-spacing:.3em;color:var(--cyan);margin-bottom:18px;text-shadow:0 0 8px var(--cyan)}
.welcome .examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:24px;max-width:760px;margin-left:auto;margin-right:auto}
.welcome .ex{background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.18);border-radius:3px;padding:12px 14px;cursor:pointer;font-family:inherit;color:var(--fg);font-size:11px;text-align:left;transition:all .2s}
.welcome .ex:hover{border-color:var(--cyan);box-shadow:0 0 14px rgba(0,229,255,.4);transform:translateY(-1px)}
.welcome .ex .lbl{color:var(--cyan);font-size:9px;text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;display:block}
.msg{display:flex;flex-direction:column;gap:6px;max-width:84%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.bot{align-self:flex-start;align-items:flex-start}
.bubble{padding:12px 16px;border-radius:4px;font-size:13px;line-height:1.6;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word}
.msg.user .bubble{background:linear-gradient(135deg,rgba(0,229,255,.18),rgba(0,184,212,.12));border:1px solid rgba(0,229,255,.4);color:var(--fg);box-shadow:0 0 14px rgba(0,229,255,.15)}
.msg.bot .bubble{background:rgba(8,14,28,.7);border:1px solid rgba(0,229,255,.2);color:var(--fg);box-shadow:inset 0 0 12px rgba(0,229,255,.05)}
.bubble code{background:rgba(0,0,0,.6);padding:1px 6px;border-radius:2px;font-size:11px;color:var(--cyan)}
.bubble pre{background:rgba(0,0,0,.7);border:1px solid rgba(0,229,255,.15);padding:10px 12px;border-radius:3px;overflow-x:auto;font-size:11px;margin:6px 0;color:var(--cyan)}
.bubble pre code{background:none;padding:0;color:inherit}
.bubble strong{color:var(--cyan);text-shadow:0 0 4px rgba(0,229,255,.5)}
.meta{font-size:9px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;display:flex;gap:8px;align-items:center}
.meta .badge{padding:2px 8px;border:1px solid rgba(0,229,255,.3);border-radius:99px;color:var(--cyan);background:rgba(0,229,255,.05)}
.meta .badge.persona{border-color:rgba(255,43,214,.5);color:var(--magenta);background:rgba(255,43,214,.05)}
.thinking{color:var(--mute);font-style:italic;letter-spacing:.1em}
.thinking::after{content:'\2589';animation:blink .8s steps(2) infinite;margin-left:4px;color:var(--cyan)}
@keyframes blink{50%{opacity:0}}
.widgets{display:flex;flex-direction:column;gap:10px;width:100%;margin-top:4px}
.widget{border:1px solid rgba(0,229,255,.35);border-radius:4px;background:rgba(8,14,28,.85);padding:14px 16px;position:relative;overflow:hidden;box-shadow:0 0 18px rgba(0,229,255,.12),inset 0 0 24px rgba(0,229,255,.04)}
.widget::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.6}
.widget .w-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan)}
.widget .w-icon{font-size:14px;filter:drop-shadow(0 0 4px var(--cyan))}
.widget .w-body{font-size:12px;color:var(--fg)}
.widget.weather .w-temp{font-size:36px;font-weight:300;color:var(--cyan);text-shadow:0 0 12px var(--cyan);letter-spacing:-.02em;display:flex;align-items:baseline;gap:6px}
.widget.weather .w-temp small{font-size:14px;color:var(--mute)}
.widget.weather .w-loc{font-size:11px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px}
.widget.weather .w-desc{font-size:13px;color:var(--fg);margin:4px 0 8px;text-transform:capitalize}
.widget.weather .w-stats{display:flex;gap:14px;font-size:10px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;border-top:1px solid rgba(0,229,255,.12);padding-top:8px;margin-top:8px}
.widget.weather .w-stats .v{color:var(--cyan);font-size:12px;margin-left:4px}
.widget.system{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.widget.system .sys-card{border:1px solid rgba(0,229,255,.15);padding:8px 10px;border-radius:3px;background:rgba(0,229,255,.02)}
.widget.system .sys-card .lbl{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute)}
.widget.system .sys-card .val{font-size:18px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);margin-top:4px}
.widget.system .sys-card .bar{height:3px;background:rgba(0,229,255,.1);border-radius:2px;margin-top:6px;overflow:hidden}
.widget.system .sys-card .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 8px var(--cyan);transition:width .3s}
.widget.time .w-clock{font-size:32px;color:var(--cyan);text-shadow:0 0 12px var(--cyan);letter-spacing:.1em;font-weight:300}
.widget.time .w-date{font-size:11px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-top:4px}
.widget.time .w-tz{font-size:9px;color:var(--mute);margin-top:2px}
.widget.error{border-color:rgba(255,85,119,.5);box-shadow:0 0 18px rgba(255,85,119,.2)}
.widget.error .w-head{color:var(--err)}
.widget.info{border-color:rgba(255,215,112,.4);box-shadow:0 0 14px rgba(255,215,112,.15)}
.widget.info .w-head{color:var(--gold)}
.widget.code .w-body pre{background:rgba(0,0,0,.6);padding:10px;border-radius:2px;font-size:11px;color:var(--cyan);overflow-x:auto}
#composer{display:flex;gap:10px;align-items:center}
#mic-shell{position:relative;width:46px;height:46px;border:1px solid rgba(0,229,255,.4);border-radius:50%;background:rgba(0,229,255,.05);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:18px;color:var(--cyan);transition:all .2s;flex-shrink:0}
#mic-shell:hover{box-shadow:0 0 14px var(--cyan);border-color:var(--cyan)}
#mic-shell.listening{background:rgba(255,43,214,.15);border-color:var(--magenta);color:var(--magenta);box-shadow:0 0 18px var(--magenta);animation:pulseMic 1.2s ease-in-out infinite}
@keyframes pulseMic{0%,100%{box-shadow:0 0 18px var(--magenta)}50%{box-shadow:0 0 28px var(--magenta)}}
#input-shell{flex:1;position:relative;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);border-radius:4px;display:flex;align-items:center;transition:all .2s}
#input-shell:focus-within{border-color:var(--cyan);box-shadow:0 0 14px rgba(0,229,255,.2)}
#input-shell::before{content:'>';color:var(--cyan);font-size:14px;padding-left:14px;text-shadow:0 0 4px var(--cyan)}
#input{flex:1;background:transparent;border:0;color:var(--fg);padding:13px 14px;font-family:inherit;font-size:13px;resize:none;outline:none;line-height:1.4;min-height:46px;max-height:160px}
#input::placeholder{color:var(--mute);font-style:italic}
#send{padding:0 22px;height:46px;border:1px solid var(--cyan);background:rgba(0,229,255,.1);color:var(--cyan);font-family:inherit;font-size:11px;letter-spacing:.25em;text-transform:uppercase;cursor:pointer;border-radius:4px;transition:all .2s;text-shadow:0 0 4px var(--cyan)}
#send:hover:not(:disabled){background:var(--cyan);color:var(--bg);box-shadow:0 0 18px var(--cyan)}
#send:disabled{opacity:.3;cursor:wait}
#voiceout-toggle{padding:0 14px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#voiceout-toggle.on{color:var(--gold);border-color:var(--gold);background:rgba(255,215,112,.08)}
.sidehint{position:fixed;bottom:16px;right:36px;font-size:9px;color:var(--mute);letter-spacing:.2em;z-index:6}
@media(max-width:760px){.status .pill{display:none}.title{font-size:16px}#app{padding:18px 16px}}
</style></head><body>
<div id="nebula"></div>
<canvas id="netcanvas"></canvas>
<div id="frame"></div><div id="corner-tr"></div><div id="corner-bl"></div><div id="scanline"></div>
<div id="app">
  <header>
    <div class="title">A D A M ▸ JARVIS</div>
    <div class="status">
      <span class="pill"><span class="dot"></span>GF(17) online</span>
      <span class="pill" id="lesson-pill">lessons —</span>
      <span class="pill" id="persona-pill">persona —</span>
    </div>
  </header>
  <div id="chat-wrap"><div id="log">
    <div class="welcome">
      <h2>NEURAL INTERFACE READY</h2>
      <div style="color:var(--mute);font-size:11px;letter-spacing:.1em">Ask anything. Live data renders inline as glowing cards.</div>
      <div class="examples">
        <button class="ex" onclick="quick('What is the weather in Boston?')"><span class="lbl">live data</span>What is the weather in Boston?</button>
        <button class="ex" onclick="quick('Show me current system stats')"><span class="lbl">live data</span>Show me current system stats</button>
        <button class="ex" onclick="quick('What time is it in Tokyo?')"><span class="lbl">live data</span>What time is it in Tokyo?</button>
        <button class="ex" onclick="quick('Write a python function to reverse a string')"><span class="lbl">code</span>Write a python function to reverse a string</button>
        <button class="ex" onclick="quick('What is 17 * 23?')"><span class="lbl">math</span>What is 17 * 23?</button>
        <button class="ex" onclick="quick('Tell me a haiku about AI')"><span class="lbl">creative</span>Tell me a haiku about AI</button>
      </div>
    </div>
  </div></div>
  <div id="composer">
    <button id="mic-shell" type="button" onclick="toggleMic()" title="Voice input">⏵</button>
    <div id="input-shell"><textarea id="input" placeholder="Speak or type..." autofocus></textarea></div>
    <button id="voiceout-toggle" type="button" onclick="toggleVoiceOut()" title="Speak responses">VOICE</button>
    <button id="send" onclick="send()">TRANSMIT</button>
  </div>
</div>
<div class="sidehint">Adam • Amni-Ai • Local • GF(17)</div>
<script>
const SKEY='amni_jarvis_session',VKEY='amni_jarvis_voiceout';
let sid=localStorage.getItem(SKEY)||'';
let voiceOut=localStorage.getItem(VKEY)==='1';
let recog=null,recoOn=false;
const log=document.getElementById('log'),input=document.getElementById('input'),send_btn=document.getElementById('send'),lessonPill=document.getElementById('lesson-pill'),personaPill=document.getElementById('persona-pill');
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function md(src){
  src=esc(src);
  src=src.replace(/```([\w-]*)\n([\s\S]*?)```/g,(_,l,c)=>`<pre><code>${c.replace(/\n$/,'')}</code></pre>`);
  src=src.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  src=src.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  src=src.replace(/^###\s+(.+)$/gm,'<strong>$1</strong>').replace(/^##\s+(.+)$/gm,'<strong>$1</strong>');
  src=src.replace(/\n\n+/g,'<br><br>').replace(/\n/g,'<br>');
  return src;
}
function bubble(role,text,meta){
  const w=document.querySelector('.welcome');if(w)w.remove();
  const m=document.createElement('div');m.className='msg '+role;
  const b=document.createElement('div');b.className='bubble';
  if(role==='bot')b.innerHTML=md(text||'');else b.textContent=text||'';
  m.appendChild(b);
  if(meta){const mt=document.createElement('div');mt.className='meta';mt.innerHTML=meta;m.appendChild(mt)}
  log.appendChild(m);log.scrollTop=log.scrollHeight;
  return {msg:m,bubble:b};
}
function renderWidget(w){
  const t=w.type;const d=w.data||{};const el=document.createElement('div');el.className='widget '+t;
  const icon=esc(w.icon||({weather:'☁',system:'⚙',time:'◷',error:'!',info:'i',code:'<>'}[t]||'◆'));
  const head=`<div class="w-head"><span class="w-icon">${icon}</span><span>${esc(w.title||t)}</span></div>`;
  let body='';
  if(t==='weather'){
    const temp=d.temp_c!=null?Math.round(d.temp_c):'?';
    body=`<div class="w-loc">${esc(d.location||'?')}</div><div class="w-temp">${temp}<small>°C</small></div><div class="w-desc">${esc(d.description||'')}</div><div class="w-stats"><div>HI<span class="v">${d.high_c!=null?Math.round(d.high_c):'?'}°</span></div><div>LO<span class="v">${d.low_c!=null?Math.round(d.low_c):'?'}°</span></div><div>HUMID<span class="v">${d.humidity_pct||'?'}%</span></div><div>WIND<span class="v">${d.wind_kmh||'?'} km/h</span></div></div>`;
  }else if(t==='system'){
    const cards=[];
    if(d.cpu_pct!=null)cards.push(`<div class="sys-card"><div class="lbl">CPU</div><div class="val">${d.cpu_pct}%</div><div class="bar"><div class="bar-fill" style="width:${d.cpu_pct}%"></div></div></div>`);
    if(d.mem_pct!=null)cards.push(`<div class="sys-card"><div class="lbl">MEMORY</div><div class="val">${d.mem_pct}%</div><div class="bar"><div class="bar-fill" style="width:${d.mem_pct}%"></div></div></div>`);
    if(d.disk_pct!=null)cards.push(`<div class="sys-card"><div class="lbl">DISK</div><div class="val">${d.disk_pct}%</div><div class="bar"><div class="bar-fill" style="width:${d.disk_pct}%"></div></div></div>`);
    if(d.gpu_name)cards.push(`<div class="sys-card"><div class="lbl">GPU</div><div class="val" style="font-size:11px;letter-spacing:.05em">${esc(d.gpu_name.slice(0,28))}</div>${d.gpu_vram_total_gb?'<div class="lbl" style="margin-top:6px">VRAM '+d.gpu_vram_total_gb+' GB</div>':''}</div>`);
    body=cards.join('');
  }else if(t==='time'){
    body=`<div class="w-clock">${esc(d.time_human||d.iso||'?')}</div><div class="w-date">${esc(d.weekday||'')} · ${esc(d.date_human||'')}</div><div class="w-tz">${esc(d.tz||'local')}</div>`;
  }else if(t==='code'){
    body=`<pre><code>${esc(d.code||'')}</code></pre>`;
  }else if(t==='error'||t==='info'){
    body=esc(d.message||'');
  }else{
    body='<pre>'+esc(JSON.stringify(d,null,2)).slice(0,800)+'</pre>';
  }
  el.innerHTML=head+'<div class="w-body">'+body+'</div>';
  return el;
}
function appendWidgets(msgEl,widgets){
  if(!widgets||!widgets.length)return;
  const wrap=document.createElement('div');wrap.className='widgets';
  for(const w of widgets){wrap.appendChild(renderWidget(w))}
  msgEl.appendChild(wrap);log.scrollTop=log.scrollHeight;
}
function quick(t){input.value=t;send()}
async function send(){
  const text=input.value.trim();if(!text)return;
  input.value='';input.style.height='auto';send_btn.disabled=true;
  bubble('user',text);
  const bot=bubble('bot','...');bot.bubble.classList.add('thinking');
  try{
    const body={model:'adam:e2b-gf17',messages:[{role:'user',content:text}],stream:false};
    if(sid)body.user=sid;
    const resp=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await resp.json();
    const msg=j.choices&&j.choices[0]&&j.choices[0].message||{};
    const content=msg.content||'';
    const widgets=msg.amni_widgets||j.amni_widgets||[];
    bot.bubble.classList.remove('thinking');
    bot.bubble.innerHTML=md(content);
    if(widgets.length){appendWidgets(bot.msg,widgets)}
    const tier=j.amni_tier||'?';const wall=j.amni_wall_s||'';
    const metaEl=document.createElement('div');metaEl.className='meta';
    metaEl.innerHTML=`<span class="badge">${esc(tier)}</span>${wall?`<span>${wall}s</span>`:''}${widgets.length?`<span class="badge">${widgets.length} widget(s)</span>`:''}`;
    bot.msg.appendChild(metaEl);
    if(voiceOut&&content)speak(content);
  }catch(err){bot.bubble.classList.remove('thinking');bot.bubble.textContent='Error: '+err.message}
  send_btn.disabled=false;input.focus();log.scrollTop=log.scrollHeight;
}
function toggleVoiceOut(){voiceOut=!voiceOut;localStorage.setItem(VKEY,voiceOut?'1':'0');const el=document.getElementById('voiceout-toggle');el.classList.toggle('on',voiceOut)}
function speak(text){if(!voiceOut||!('speechSynthesis' in window))return;try{const u=new SpeechSynthesisUtterance(text.replace(/```[\s\S]*?```/g,'(code)').replace(/[*_`#<>]/g,'').slice(0,800));u.rate=1;u.pitch=1;speechSynthesis.cancel();speechSynthesis.speak(u)}catch{}}
function toggleMic(){
  if(!('webkitSpeechRecognition' in window||'SpeechRecognition' in window)){alert('Voice input needs Chrome/Edge.');return}
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const m=document.getElementById('mic-shell');
  if(recoOn&&recog){recog.stop();return}
  recog=new SR();recog.lang='en-US';recog.interimResults=false;recog.continuous=false;
  recoOn=true;m.classList.add('listening');
  recog.onresult=e=>{input.value=e.results[0][0].transcript;send()};
  recog.onerror=()=>{recoOn=false;m.classList.remove('listening')};
  recog.onend=()=>{recoOn=false;m.classList.remove('listening')};
  recog.start();
}
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px'});
async function refreshStats(){try{const r=await fetch('/stats');const j=await r.json();lessonPill.textContent='lessons '+(j.lessons_n||0);}catch{}}
if(voiceOut)document.getElementById('voiceout-toggle').classList.add('on');
refreshStats();setInterval(refreshStats,30000);
const canvas=document.getElementById('netcanvas'),ctx=canvas.getContext('2d');
let W=0,H=0,nodes=[];
function resize(){W=canvas.width=window.innerWidth*window.devicePixelRatio;H=canvas.height=window.innerHeight*window.devicePixelRatio;canvas.style.width=window.innerWidth+'px';canvas.style.height=window.innerHeight+'px';ctx.scale(window.devicePixelRatio,window.devicePixelRatio);initNodes()}
function initNodes(){nodes=[];const N=Math.min(60,Math.floor((W*H)/(window.devicePixelRatio*window.devicePixelRatio)/24000));for(let i=0;i<N;i++)nodes.push({x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25,r:1+Math.random()*1.4})}
function tick(){
  ctx.clearRect(0,0,W,H);
  for(const n of nodes){
    n.x+=n.vx;n.y+=n.vy;
    if(n.x<0||n.x>window.innerWidth)n.vx*=-1;
    if(n.y<0||n.y>window.innerHeight)n.vy*=-1;
  }
  ctx.lineWidth=.5;
  for(let i=0;i<nodes.length;i++){
    for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j];const dx=a.x-b.x,dy=a.y-b.y;const d=Math.hypot(dx,dy);
      if(d<160){const alpha=(1-d/160)*.35;ctx.strokeStyle=`rgba(0,229,255,${alpha})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    }
  }
  for(const n of nodes){
    ctx.fillStyle='rgba(0,229,255,.85)';
    ctx.shadowBlur=8;ctx.shadowColor='rgba(0,229,255,.7)';
    ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fill();
    ctx.shadowBlur=0;
  }
  requestAnimationFrame(tick);
}
window.addEventListener('resize',resize);resize();tick();
</script></body></html>"""
def mount(app):
    from fastapi.responses import HTMLResponse
    @app.get('/jarvis',response_class=HTMLResponse)
    def jarvis():return HTMLResponse(content=_HTML)
