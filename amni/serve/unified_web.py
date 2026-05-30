"""unified_web — ONE fresh UI over the proven jarvis_web engine. Reuses jarvis_web._HTML verbatim (all 187 element IDs + the 2814-line script + 7 themes + CSS vars stay intact) and progressively enhances it: a left hamburger rail (JARVIS/Code/CLI/HUD + voice/sessions/skills), a bottom status bar (lessons/skills/tts/gpu/version), frosted-glass + stray-button fixes, persona list prepopulated on load. Served at /unified during build; /  flips to it once polished. mount(app) — adds GET /unified."""
from amni.serve import jarvis_web as _j
_SKIN=r"""<style id="unified-skin">
:root{--u-rail:64px;--u-bar:30px}
body.u-on #app{margin-left:var(--u-rail);margin-bottom:var(--u-bar);transition:margin .25s ease}
body.u-on header>.surfnav{display:none}
#u-rail{position:fixed;top:0;left:0;width:var(--u-rail);height:100vh;z-index:60;display:flex;flex-direction:column;align-items:center;gap:6px;padding:10px 0;background:linear-gradient(180deg,rgba(var(--panel-rgb,10,18,36),.92),rgba(var(--panel-rgb,10,18,36),.78));border-right:1px solid rgba(var(--c-rgb,0,229,255),.18);backdrop-filter:blur(8px)}
#u-rail .u-logo{font:700 16px/1 'JetBrains Mono',monospace;color:var(--cyan,#00e5ff);text-shadow:0 0 10px rgba(var(--c-rgb,0,229,255),.6);margin-bottom:8px;letter-spacing:.05em}
.u-nav{width:46px;height:46px;border-radius:12px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;border:1px solid rgba(var(--c-rgb,0,229,255),.16);background:rgba(var(--panel-rgb,10,18,36),.5);color:var(--fg,#dff6ff);font:600 8px/1 'JetBrains Mono',monospace;letter-spacing:.04em;transition:all .15s ease;text-decoration:none}
.u-nav:hover{border-color:var(--cyan,#00e5ff);box-shadow:0 0 12px rgba(var(--c-rgb,0,229,255),.35);transform:translateY(-1px)}
.u-nav.active{background:rgba(var(--c-rgb,0,229,255),.14);border-color:var(--cyan,#00e5ff)}
.u-nav .ic{font-size:17px;line-height:1}
.u-nav .lb{opacity:.85}
#u-rail .u-spacer{flex:1}
#u-statusbar{position:fixed;left:var(--u-rail);right:0;bottom:0;height:var(--u-bar);z-index:55;display:flex;align-items:center;gap:0;padding:0 12px;font:600 10px/1 'JetBrains Mono',monospace;color:var(--mute,#5e7a99);background:linear-gradient(0deg,rgba(var(--panel-rgb,10,18,36),.95),rgba(var(--panel-rgb,10,18,36),.8));border-top:1px solid rgba(var(--c-rgb,0,229,255),.16);backdrop-filter:blur(8px);letter-spacing:.05em}
#u-statusbar .u-st{display:flex;align-items:center;gap:5px;padding:0 12px;border-right:1px solid rgba(var(--c-rgb,0,229,255),.1);white-space:nowrap}
#u-statusbar .u-st:last-child{border-right:0;margin-left:auto}
#u-statusbar .u-st b{color:var(--fg,#dff6ff);font-weight:600}
#u-statusbar .u-st .led{width:6px;height:6px;border-radius:50%;background:var(--ok,#00ff9d);box-shadow:0 0 6px var(--ok,#00ff9d)}
body.u-on #nebula,body.u-on #scanline,body.u-on #frame,body.u-on #corner-tr,body.u-on #corner-bl{pointer-events:none}
body.u-on #nebula{opacity:.42}
body.u-on #scanline{opacity:.28}
body.u-on #composer{position:relative;z-index:42;gap:9px;padding:11px 16px;align-items:center}
body.u-on #input-shell{background:rgba(var(--panel-rgb,10,18,36),.9);backdrop-filter:blur(4px);border:1px solid rgba(var(--c-rgb,0,229,255),.3);flex:1}
body.u-on #input{color:var(--fg,#dff6ff);background:transparent}
body.u-on #input::placeholder{color:var(--mute,#5e7a99)}
body.u-on #mic-shell{font-size:0;width:42px;height:42px;border-radius:50%;background:rgba(var(--panel-rgb,10,18,36),.7);border:1px solid rgba(var(--c-rgb,0,229,255),.3);display:flex;align-items:center;justify-content:center;flex:0 0 auto}
body.u-on #mic-shell::before{content:'\1F3A4';font-size:17px}
body.u-on #mic-shell.rec,body.u-on #mic-shell.listening{border-color:var(--err,#ff5577);box-shadow:0 0 14px rgba(255,85,119,.5)}
body.u-on #jarvis-toggle,body.u-on #tools-toggle{flex:0 0 auto;opacity:.9}
body.u-on #send{flex:0 0 auto;background:linear-gradient(135deg,rgba(var(--c-rgb,0,229,255),.95),rgba(var(--c-rgb,0,229,255),.62));color:#04121a;font-weight:700;border:0;text-shadow:none}
body.u-on #send:hover{box-shadow:0 0 16px rgba(var(--c-rgb,0,229,255),.5)}
body.u-on .bubble{background:rgba(var(--panel-rgb,10,18,36),.9)!important}
body.u-on #log{position:relative;z-index:5}
#u-load{position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px;background:radial-gradient(ellipse at center,#0a1224 0%,#040711 70%);transition:opacity .5s ease}
#u-load.gone{opacity:0;pointer-events:none}
#u-load .ulg{font:700 44px/1 'JetBrains Mono',monospace;color:#00e5ff;text-shadow:0 0 24px rgba(0,229,255,.55);letter-spacing:.18em}
#u-load .ulr{width:240px;height:3px;border-radius:3px;background:rgba(0,229,255,.12);overflow:hidden}
#u-load .ulr i{display:block;height:100%;width:35%;border-radius:3px;background:linear-gradient(90deg,transparent,#00e5ff,transparent);animation:ulsweep 1.1s ease-in-out infinite}
@keyframes ulsweep{0%{transform:translateX(-120%)}100%{transform:translateX(380%)}}
#u-load .uls{font:600 11px/1.5 'JetBrains Mono',monospace;color:#5e7a99;letter-spacing:.16em;text-transform:uppercase;text-align:center;min-height:32px}
#u-load .uls b{color:#00ff9d}
#u-load .ulspin{width:54px;height:54px;border-radius:50%;border:2px solid rgba(0,229,255,.15);border-top-color:#00e5ff;animation:ulspin .8s linear infinite}
@keyframes ulspin{to{transform:rotate(360deg)}}
</style>"""
_BOOT=r"""<script id="unified-boot">(function(){try{
var B=document.body;B.classList.add('u-on');
function el(t,a,h){var e=document.createElement(t);if(a)for(var k in a)e.setAttribute(k,a[k]);if(h!=null)e.innerHTML=h;return e;}
var rail=el('div',{id:'u-rail'});
rail.appendChild(el('div',{class:'u-logo'},'A'));
var navs=[['JARVIS','⚡',function(){var i=document.getElementById('input');i&&i.focus();}],['CODE','⌨',function(){try{_openPeer();}catch(e){window.open('http://127.0.0.1:3000','_blank');}}],['CLI','▤',function(){try{_showCliInfo();}catch(e){}}],['HUD','◈',function(){location.href='/hud';}],['VOICE','🔊',function(){try{toggleToolsDrawer();}catch(e){}}],['CHATS','☷',function(){try{toggleSessionsPanel();}catch(e){try{toggleStatusPanel();}catch(_){}}}],['SKILLS','⚙',function(){try{toggleStatusPanel();}catch(e){}}],['WEATHER','🌤',function(){try{(typeof _qcAsk==='function'?_qcAsk:quick)("what's my local weather and forecast for today and this week?");}catch(e){}}],['MENU','≡',function(){try{toggleCmdMenu();}catch(e){}}]];
navs.forEach(function(n,i){var b=el('div',{class:'u-nav'+(i===0?' active':''),title:n[0]},'<span class="ic">'+n[1]+'</span><span class="lb">'+n[0]+'</span>');b.onclick=function(){rail.querySelectorAll('.u-nav').forEach(function(x){x.classList.remove('active');});b.classList.add('active');n[2]();};rail.appendChild(b);});
rail.appendChild(el('div',{class:'u-spacer'}));
var theme=el('div',{class:'u-nav',title:'Persona + theme'},'<span class="ic">◐</span><span class="lb">THEME</span>');theme.onclick=function(){try{togglePersonaPanel();}catch(e){}};rail.appendChild(theme);
B.appendChild(rail);
var bar=el('div',{id:'u-statusbar'});
bar.innerHTML='<span class="u-st"><span class="led"></span>GF(17) <b id="u-ver">online</b></span><span class="u-st">lessons <b id="u-lessons">—</b></span><span class="u-st">skills <b id="u-skills">—</b></span><span class="u-st">tts <b id="u-tts">—</b></span><span class="u-st">gpu <b id="u-gpu">—</b></span><span class="u-st">recall <b id="u-recall">—</b></span>';
B.appendChild(bar);
function set(id,v){var e=document.getElementById(id);if(e&&v!=null&&v!=='')e.textContent=v;}
function pull(){fetch('/healthz').then(function(r){return r.json();}).then(function(d){var ln=(d.lessons_n!=null)?d.lessons_n:(d.adam&&d.adam.lessons_n);set('u-lessons',ln!=null?ln:'—');var sk=(d.skills_n!=null)?d.skills_n:(d.skills&&d.skills.count);set('u-skills',sk!=null?sk:'—');if(d.version)set('u-ver','v'+d.version);}).catch(function(){});fetch('/voice/status').then(function(r){return r.json();}).then(function(d){set('u-tts',(d&&(d.backend||d.voice||(d.available?'on':'off')))||'—');}).catch(function(){});fetch('/stats').then(function(r){return r.json();}).then(function(d){var mb=d&&d.memory_bus;if(mb)set('u-recall',(mb.recall_hits||0)+'/'+(mb.recall_total||0));}).catch(function(){});fetch('/health').then(function(r){return r.json();}).then(function(d){var g=(d&&d.gpu)||{};set('u-gpu',g.device_name||(g.cuda_or_rocm?'gpu':'cpu'));}).catch(function(){});}
pull();setInterval(pull,7000);
setTimeout(function(){try{if(typeof _loadPersonas==='function'){Promise.resolve(_loadPersonas()).then(function(){try{typeof _renderPersonaPanel==='function'&&_renderPersonaPanel();}catch(e){}});}if(typeof _loadVoices==='function'){try{_loadVoices();}catch(e){}}}catch(e){}},500);
(function(){var ld=document.getElementById('u-load');if(!ld)return;var uls=ld.querySelector('.uls');var t0=Date.now();var done=false;
function reveal(msg){if(done)return;done=true;if(uls)uls.innerHTML='<b>'+(msg||'ready')+'</b>';setTimeout(function(){ld.classList.add('gone');setTimeout(function(){ld&&ld.parentNode&&ld.parentNode.removeChild(ld);},600);},250);}
function poll(){fetch('/warmup').then(function(r){return r.json();}).then(function(d){var w=(d&&d.warmup)||{};var el=Math.round((Date.now()-t0)/1000);if(w.done){reveal(w.error?'ready (warmup note: '+String(w.error).slice(0,40)+')':'Adam ready'+(w.wall_s?' · warmed in '+w.wall_s+'s':''));}else{if(uls)uls.textContent='loading Adam — GF(17) weights warming · '+el+'s';setTimeout(poll,900);}}).catch(function(){var el=Math.round((Date.now()-t0)/1000);if(el>240){reveal('ready');return;}if(uls)uls.textContent='starting server · '+el+'s';setTimeout(poll,1200);});}
poll();}());
}catch(e){console&&console.warn&&console.warn('[unified-boot]',e);}})();</script>
<script id="unified-loader-fallback">setTimeout(function(){var l=document.getElementById('u-load');if(l&&!l.classList.contains('gone')){l.classList.add('gone');}},300000);</script>"""
_LOADER=r"""<div id="u-load"><div class="ulg">A D A M</div><div class="ulspin"></div><div class="ulr"><i></i></div><div class="uls">loading Adam — GF(17) weights warming…</div></div>"""
def page()->str:
    import re as _re
    html=getattr(_j,'_HTML','') or ''
    if '</body>' in html:html=html.replace('</body>',_SKIN+_BOOT+'</body>',1)
    else:html=html+_SKIN+_BOOT
    m=_re.search(r'<body[^>]*>',html,_re.IGNORECASE)
    if m:html=html[:m.end()]+_LOADER+html[m.end():]
    else:html=_LOADER+html
    return html
def mount(app):
    from fastapi.responses import HTMLResponse
    @app.get('/unified',response_class=HTMLResponse)
    def unified():return HTMLResponse(content=page())
