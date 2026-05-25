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
#mem-panel{position:fixed;top:60px;right:-440px;width:420px;height:calc(100vh - 140px);z-index:9;border:1px solid rgba(0,229,255,.35);border-radius:4px;background:rgba(8,14,28,.92);box-shadow:0 0 28px rgba(0,229,255,.2);overflow-y:auto;transition:right .25s ease-out;backdrop-filter:blur(8px)}
#mem-panel.show{right:24px}
#mem-panel::-webkit-scrollbar{width:5px}
#mem-panel::-webkit-scrollbar-thumb{background:rgba(0,229,255,.3);border-radius:3px}
#mem-panel .mem-head{padding:10px 14px;border-bottom:1px solid rgba(0,229,255,.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan);display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:rgba(8,14,28,.95);z-index:2}
#mem-panel .mem-head .close{cursor:pointer;color:var(--mute);padding:2px 8px;border:1px solid rgba(0,229,255,.2);border-radius:3px;font-size:11px}
#mem-panel .mem-head .close:hover{color:var(--err);border-color:var(--err)}
#mem-panel .mem-section{padding:12px 14px;border-bottom:1px solid rgba(0,229,255,.08)}
#mem-panel .mem-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);margin-bottom:8px;display:flex;align-items:center;gap:6px}
#mem-panel .mem-section h3 .count{margin-left:auto;color:var(--mute);font-size:9px;letter-spacing:.1em}
#mem-panel .mem-row{padding:6px 8px;font-size:11px;border:1px solid rgba(0,229,255,.08);border-radius:3px;margin-bottom:5px;background:rgba(0,229,255,.02);display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
#mem-panel .mem-row:hover{border-color:rgba(0,229,255,.3);background:rgba(0,229,255,.05)}
#mem-panel .mem-row .body{flex:1;line-height:1.4;word-break:break-word}
#mem-panel .mem-row .lbl{font-size:8px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-bottom:3px}
#mem-panel .mem-row .confid{color:var(--magenta)}
#mem-panel .mem-row .actions{display:flex;flex-direction:column;gap:3px}
#mem-panel .mem-row button{background:transparent;border:1px solid rgba(255,85,119,.4);color:var(--err);font-family:inherit;font-size:9px;padding:2px 6px;border-radius:2px;cursor:pointer;letter-spacing:.1em}
#mem-panel .mem-row button:hover{background:rgba(255,85,119,.1)}
#mem-panel .mem-row button.cyan{border-color:rgba(0,229,255,.4);color:var(--cyan)}
#mem-panel .mem-row button.cyan:hover{background:rgba(0,229,255,.1)}
#mem-panel .mem-stat{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:6px}
#mem-panel .mem-stat .stat{padding:6px 8px;border:1px solid rgba(0,229,255,.12);border-radius:3px;background:rgba(0,229,255,.02)}
#mem-panel .mem-stat .stat .v{font-size:18px;color:var(--cyan);text-shadow:0 0 6px var(--cyan)}
#mem-panel .mem-stat .stat .lbl{font-size:8px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute)}
#mem-panel .mastery-bar{height:4px;background:rgba(0,229,255,.1);border-radius:2px;overflow:hidden;margin-top:4px}
#mem-panel .mastery-bar .fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 6px var(--cyan)}
#mem-panel .empty{color:var(--mute);font-style:italic;text-align:center;padding:14px;font-size:10px}
#mem-toggle{padding:0 12px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#mem-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(0,229,255,.08);box-shadow:0 0 12px rgba(0,229,255,.3)}
#cam-panel{position:fixed;top:60px;right:24px;width:200px;z-index:8;display:none;border:1px solid rgba(0,229,255,.4);border-radius:4px;background:rgba(8,14,28,.85);box-shadow:0 0 18px rgba(0,229,255,.18);overflow:hidden}
#cam-panel.show{display:block}
#cam-panel .cam-head{padding:5px 10px;background:rgba(0,229,255,.08);font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;align-items:center;justify-content:space-between}
#cam-panel .cam-head .dot{width:6px;height:6px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);animation:pulse 1.6s ease-in-out infinite}
#cam-stage{position:relative;width:100%;aspect-ratio:4/3;background:#000}
#cam-video,#cam-landmarks{position:absolute;inset:0;width:100%;height:100%}
#cam-video{transform:scaleX(-1);object-fit:cover}
#cam-panel .gesture-readout{padding:6px 10px;border-top:1px solid rgba(0,229,255,.18);font-size:10px;letter-spacing:.15em;color:var(--cyan);text-shadow:0 0 4px var(--cyan);text-align:center;min-height:22px}
#cam-panel.idle .gesture-readout{color:var(--mute);text-shadow:none}
#gesture-toggle{padding:0 12px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#gesture-toggle.on{color:var(--magenta);border-color:var(--magenta);background:rgba(255,43,214,.08);box-shadow:0 0 12px rgba(255,43,214,.3)}
.gesture-flash{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);font-size:22px;letter-spacing:.3em;color:var(--magenta);text-shadow:0 0 24px var(--magenta);pointer-events:none;z-index:9;opacity:0;transition:opacity .3s}
.gesture-flash.show{opacity:1}
@media(max-width:760px){.status .pill{display:none}.title{font-size:16px}#app{padding:18px 16px}#cam-panel{width:140px;top:50px;right:14px}}
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
    <button id="gesture-toggle" type="button" onclick="toggleGesture()" title="Hand gesture control (webcam)">GESTURE</button>
    <button id="mem-toggle" type="button" onclick="toggleMem()" title="Inspect what Adam knows">MEMORY</button>
    <button id="send" onclick="send()">TRANSMIT</button>
  </div>
</div>
<div id="mem-panel">
  <div class="mem-head"><span>◆ MEMORY INSPECTOR</span><span class="close" onclick="toggleMem()">CLOSE</span></div>
  <div class="mem-section" id="mem-stats"><h3>SUBSTRATE OVERVIEW <span class="count" id="mem-uptime">—</span></h3><div id="mem-stat-grid">loading...</div></div>
  <div class="mem-section" id="mem-profile-sec"><h3>WHAT I KNOW ABOUT YOU <span class="count" id="mem-profile-n">—</span></h3><div id="mem-pending"></div><div id="mem-profile">loading...</div></div>
  <div class="mem-section" id="mem-kg-sec"><h3>KNOWLEDGE GRAPH <span class="count" id="mem-kg-n">—</span></h3><div id="mem-kg">loading...</div></div>
  <div class="mem-section" id="mem-coach-sec"><h3>COACH MASTERY <span class="count" id="mem-coach-n">—</span></h3><div id="mem-coach">loading...</div></div>
  <div class="mem-section" id="mem-daemon-sec"><h3>LEARNING DAEMON <span class="count" id="mem-daemon-status">—</span></h3><div id="mem-daemon">loading...</div></div>
</div>
<div id="cam-panel">
  <div class="cam-head"><span><span class="dot"></span>HAND TRACK</span><span id="cam-fps">— fps</span></div>
  <div id="cam-stage">
    <video id="cam-video" autoplay playsinline muted></video>
    <canvas id="cam-landmarks"></canvas>
  </div>
  <div class="gesture-readout" id="gesture-readout">—</div>
</div>
<div id="gesture-flash" class="gesture-flash"></div>
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
let gestureOn=false,hands=null,camStream=null,lastGesture='',lastGestureAt=0,frameTimes=[],camRAF=null;
const GKEY='amni_jarvis_gesture';
const GESTURE_COOLDOWN_MS=900;
const _flash=document.getElementById('gesture-flash'),_readout=document.getElementById('gesture-readout'),_camPanel=document.getElementById('cam-panel'),_camVideo=document.getElementById('cam-video'),_camLm=document.getElementById('cam-landmarks'),_gToggle=document.getElementById('gesture-toggle');
function _dist(a,b){const dx=a.x-b.x,dy=a.y-b.y,dz=(a.z||0)-(b.z||0);return Math.sqrt(dx*dx+dy*dy+dz*dz)}
function _fingerExtended(lm,tipIdx,pipIdx,mcpIdx){return _dist(lm[tipIdx],lm[0])>_dist(lm[pipIdx],lm[0])&&_dist(lm[tipIdx],lm[mcpIdx])>0.06}
function classifyGesture(lm){
  if(!lm||lm.length<21)return 'unknown';
  const t=lm[4],i=lm[8],m=lm[12],r=lm[16],p=lm[20];
  const iPip=lm[6],mPip=lm[10],rPip=lm[14],pPip=lm[18];
  const iMcp=lm[5],mMcp=lm[9],rMcp=lm[13],pMcp=lm[17];
  const ext={t:_dist(t,lm[0])>_dist(lm[2],lm[0]),i:_fingerExtended(lm,8,6,5),m:_fingerExtended(lm,12,10,9),r:_fingerExtended(lm,16,14,13),p:_fingerExtended(lm,20,18,17)};
  const pinch=_dist(t,i);
  if(pinch<0.05&&!ext.m&&!ext.r&&!ext.p)return 'pinch';
  if(!ext.i&&!ext.m&&!ext.r&&!ext.p)return 'fist';
  if(ext.i&&ext.m&&ext.r&&ext.p)return 'open_palm';
  if(ext.i&&ext.m&&!ext.r&&!ext.p)return 'peace';
  if(ext.i&&!ext.m&&!ext.r&&!ext.p)return 'point';
  if(ext.t&&!ext.i&&!ext.m&&!ext.r&&!ext.p)return 'thumb_up';
  return 'unknown';
}
const GESTURE_ACTIONS={pinch:'toggle voice',fist:'clear chat',open_palm:'system check',peace:'cycle theme',point:'next question',thumb_up:'submit input'};
function _flashGesture(name){
  _flash.textContent=name.replace('_',' ').toUpperCase();_flash.classList.add('show');
  setTimeout(()=>_flash.classList.remove('show'),650);
}
const _THEMES=[{cyan:'#00e5ff',magenta:'#ff2bd6'},{cyan:'#ffd770',magenta:'#ff5577'},{cyan:'#00ff9d',magenta:'#7fd6c5'},{cyan:'#c5a3ff',magenta:'#00e5ff'}];
let _themeIdx=0;
function _cycleTheme(){
  _themeIdx=(_themeIdx+1)%_THEMES.length;const t=_THEMES[_themeIdx];
  document.documentElement.style.setProperty('--cyan',t.cyan);document.documentElement.style.setProperty('--magenta',t.magenta);
}
function applyGestureAction(g){
  if(g==='pinch')toggleVoiceOut();
  else if(g==='fist'){log.innerHTML='';bubble('bot','(chat cleared by gesture)')}
  else if(g==='open_palm')quick('Show me current system stats');
  else if(g==='peace')_cycleTheme();
  else if(g==='point'){const last=log.querySelectorAll('.msg.bot .meta .badge');if(last.length)quick('Tell me more about that')}
  else if(g==='thumb_up'){const t=input.value.trim();if(t)send()}
}
function _onHandsResults(res){
  const ctx=_camLm.getContext('2d');_camLm.width=_camLm.clientWidth*window.devicePixelRatio;_camLm.height=_camLm.clientHeight*window.devicePixelRatio;
  ctx.clearRect(0,0,_camLm.width,_camLm.height);
  const lms=(res.multiHandLandmarks||[])[0];
  if(!lms){_readout.textContent='—';_camPanel.classList.add('idle');return}
  _camPanel.classList.remove('idle');
  ctx.fillStyle='rgba(0,229,255,.95)';ctx.shadowBlur=6;ctx.shadowColor='rgba(0,229,255,.7)';
  for(const lm of lms){const x=(1-lm.x)*_camLm.width,y=lm.y*_camLm.height;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}
  ctx.shadowBlur=0;ctx.strokeStyle='rgba(0,229,255,.45)';ctx.lineWidth=1.4;
  const conns=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  for(const [a,b] of conns){ctx.beginPath();ctx.moveTo((1-lms[a].x)*_camLm.width,lms[a].y*_camLm.height);ctx.lineTo((1-lms[b].x)*_camLm.width,lms[b].y*_camLm.height);ctx.stroke()}
  const g=classifyGesture(lms);_readout.textContent=g==='unknown'?'—':g.replace('_',' ').toUpperCase();
  const now=performance.now();frameTimes.push(now);if(frameTimes.length>30)frameTimes.shift();
  if(frameTimes.length>=2){const fps=Math.round(1000*(frameTimes.length-1)/(frameTimes[frameTimes.length-1]-frameTimes[0]));document.getElementById('cam-fps').textContent=fps+' fps'}
  if(g!=='unknown'&&g!==lastGesture&&(now-lastGestureAt)>GESTURE_COOLDOWN_MS){
    lastGesture=g;lastGestureAt=now;_flashGesture(g);applyGestureAction(g);
  }else if(g==='unknown'){lastGesture=''}
}
async function _loadMediaPipe(){
  if(window.Hands)return true;
  for(const src of ['https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/hands.js','https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3/camera_utils.js']){
    await new Promise((res,rej)=>{const s=document.createElement('script');s.src=src;s.crossOrigin='anonymous';s.onload=res;s.onerror=rej;document.head.appendChild(s)}).catch(e=>console.warn('mediapipe load fail',src,e));
  }
  return !!window.Hands;
}
async function startGesture(){
  const ok=await _loadMediaPipe();
  if(!ok){bubble('bot','Could not load MediaPipe Hands from CDN. Check your network or a content blocker.','<span class="badge err">gesture</span>');gestureOn=false;_gToggle.classList.remove('on');return}
  try{camStream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240,facingMode:'user'},audio:false})}
  catch(e){bubble('bot','Webcam permission denied or unavailable: '+e.message,'<span class="badge err">gesture</span>');gestureOn=false;_gToggle.classList.remove('on');return}
  _camVideo.srcObject=camStream;await _camVideo.play().catch(()=>{});
  hands=new window.Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${f}`});
  hands.setOptions({maxNumHands:1,modelComplexity:0,minDetectionConfidence:.6,minTrackingConfidence:.5});
  hands.onResults(_onHandsResults);
  const loop=async()=>{if(!gestureOn)return;if(_camVideo.readyState>=2){try{await hands.send({image:_camVideo})}catch{}};camRAF=requestAnimationFrame(loop)};
  camRAF=requestAnimationFrame(loop);
  _camPanel.classList.add('show');
}
function stopGesture(){
  if(camRAF){cancelAnimationFrame(camRAF);camRAF=null}
  if(camStream){camStream.getTracks().forEach(t=>t.stop());camStream=null}
  _camPanel.classList.remove('show');_readout.textContent='—';frameTimes=[];
  if(hands){try{hands.close()}catch{};hands=null}
}
function toggleGesture(){
  gestureOn=!gestureOn;localStorage.setItem(GKEY,gestureOn?'1':'0');_gToggle.classList.toggle('on',gestureOn);
  if(gestureOn)startGesture();else stopGesture();
}
if(localStorage.getItem(GKEY)==='1'){setTimeout(()=>{gestureOn=true;_gToggle.classList.add('on');startGesture()},800)}
let memOpen=false;
const _MEM_PANEL=document.getElementById('mem-panel'),_MEM_TOG=document.getElementById('mem-toggle');
function toggleMem(){memOpen=!memOpen;_MEM_PANEL.classList.toggle('show',memOpen);_MEM_TOG.classList.toggle('on',memOpen);if(memOpen)refreshMemory()}
async function refreshMemory(){
  try{
    const snap=await(await fetch('/memory/snapshot')).json();
    const lessons=(snap.lesson_bank&&snap.lesson_bank.n)||0;
    const pa=snap.personal_atlas||{};const la=snap.learning_daemon||{};const kg=snap.knowledge_graph||{};
    document.getElementById('mem-stat-grid').innerHTML=`<div class="mem-stat"><div class="stat"><div class="lbl">Lessons</div><div class="v">${lessons}</div></div><div class="stat"><div class="lbl">Triples</div><div class="v">${kg.triples||0}</div></div><div class="stat"><div class="lbl">Verified</div><div class="v">${(la.atlas&&la.atlas.verified)||0}</div></div><div class="stat"><div class="lbl">Facts/hr</div><div class="v">${la.facts_per_hour||0}</div></div></div>`;
    document.getElementById('mem-uptime').textContent=la.uptime_hours?`${la.uptime_hours}h up`:'';
  }catch(e){document.getElementById('mem-stat-grid').innerHTML=`<div class="empty">snapshot failed: ${esc(e.message)}</div>`}
  try{
    const pr=await(await fetch('/memory/profile?limit=50')).json();
    const facts=pr.facts||[];const pending=pr.pending||[];
    document.getElementById('mem-profile-n').textContent=`${facts.length} facts · ${(pr.stats&&pr.stats.confidential)||0} confid`;
    const pendingHtml=pending.map(p=>`<div class="mem-row"><div class="body"><div class="lbl">awaiting your call</div>${esc(p.fact||'')}</div><div class="actions"><button class="cyan" onclick="memConfirmFact('${esc(p.id)}',true)">CONFID</button><button class="cyan" onclick="memConfirmFact('${esc(p.id)}',false)">PUBLIC</button></div></div>`).join('');
    document.getElementById('mem-pending').innerHTML=pending.length?`<div class="lbl" style="margin-bottom:6px">PENDING CLARIFICATIONS</div>${pendingHtml}<div style="height:8px"></div>`:'';
    document.getElementById('mem-profile').innerHTML=facts.length?facts.map(f=>`<div class="mem-row"><div class="body">${f.is_confidential?'<span class="confid lbl">CONFIDENTIAL</span>':'<span class="lbl">public</span>'}${esc(f.fact)}</div><div class="actions"><button onclick="memForgetProfile('${esc(f.fact).replace(/'/g,"\\\\'")}')">FORGET</button></div></div>`).join(''):'<div class="empty">no profile facts yet</div>';
  }catch(e){document.getElementById('mem-profile').innerHTML=`<div class="empty">profile failed: ${esc(e.message)}</div>`}
  try{
    const kg=await(await fetch('/memory/kg?limit=10')).json();
    document.getElementById('mem-kg-n').textContent=`${(kg.stats&&kg.stats.triples)||0} triples`;
    const subs=(kg.top_subjects||[]).slice(0,8);const preds=(kg.top_predicates||[]).slice(0,6);
    const subHtml=subs.length?subs.map(s=>`<div class="mem-row"><div class="body"><span class="lbl">${s.edges_out} edges</span>${esc(s.subject)}</div><div class="actions"><button class="cyan" onclick="quick('show me what you know about ${esc(s.subject).replace(/'/g,"\\\\'")}')">EXPLORE</button></div></div>`).join(''):'<div class="empty">graph empty</div>';
    const predHtml=preds.length?`<div class="lbl" style="margin-top:6px">TOP PREDICATES</div>`+preds.map(p=>`<div class="mem-row"><div class="body">${esc(p.predicate)}<span class="lbl" style="margin-left:8px">${p.count}</span></div></div>`).join(''):'';
    document.getElementById('mem-kg').innerHTML=subHtml+predHtml;
  }catch(e){document.getElementById('mem-kg').innerHTML=`<div class="empty">kg failed: ${esc(e.message)}</div>`}
  try{
    const co=await(await fetch('/memory/coach')).json();
    const topics=co.topics||[];
    document.getElementById('mem-coach-n').textContent=`${topics.length} topics`;
    document.getElementById('mem-coach').innerHTML=topics.length?topics.map(t=>`<div class="mem-row"><div class="body">${esc(t.topic)}<div class="lbl">${t.n_questions} asked · ${t.mastery_pct}%</div><div class="mastery-bar"><div class="fill" style="width:${t.mastery_pct}%"></div></div></div><div class="actions"><button class="cyan" onclick="quick('coach me on ${esc(t.topic).replace(/'/g,"\\\\'")}')">RESUME</button></div></div>`).join(''):'<div class="empty">no coach sessions yet</div>';
  }catch(e){document.getElementById('mem-coach').innerHTML=`<div class="empty">coach failed: ${esc(e.message)}</div>`}
  try{
    const d=await(await fetch('/memory/daemon')).json();
    document.getElementById('mem-daemon-status').textContent=d.enabled?'ACTIVE':'paused';
    const c=d.counters||{};
    document.getElementById('mem-daemon').innerHTML=`<div class="mem-stat"><div class="stat"><div class="lbl">Curiosity</div><div class="v">${c.curiosity_ticks||0}</div></div><div class="stat"><div class="lbl">Sleep passes</div><div class="v">${c.sleep_passes||0}</div></div><div class="stat"><div class="lbl">New facts</div><div class="v">${c.qa_pairs_new||0}</div></div><div class="stat"><div class="lbl">Queue</div><div class="v">${d.queue_depth||0}</div></div></div><div class="mem-row" style="margin-top:6px"><div class="body"><span class="lbl">${d.user_active_recently?'yielding to you':'running freely'}</span>Adam's 24/7 learning loop. Use the curiosity_tick / sleep_pass / pause skills to control it.</div></div>`;
  }catch(e){document.getElementById('mem-daemon').innerHTML=`<div class="empty">daemon failed: ${esc(e.message)}</div>`}
}
async function memConfirmFact(id,isConf){
  try{await fetch('/memory/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id,is_confidential:isConf})});refreshMemory()}
  catch(e){console.warn('confirm fail',e)}
}
async function memForgetProfile(factSnippet){
  if(!confirm('Forget facts matching: '+factSnippet.slice(0,80)+'?'))return;
  try{
    const pat=factSnippet.replace(/[.*+?^${}()|[\]\\]/g,'\\\\$&').slice(0,60);
    await fetch('/memory/forget',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({atlas:'personal',pattern:pat,confirm:true})});
    refreshMemory();
  }catch(e){console.warn('forget fail',e)}
}
</script></body></html>"""
def mount(app):
    from fastapi.responses import HTMLResponse
    @app.get('/jarvis',response_class=HTMLResponse)
    def jarvis():return HTMLResponse(content=_HTML)
