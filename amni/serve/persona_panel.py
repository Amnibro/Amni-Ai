"""persona_panel — minimal self-contained HTML side-panel for IDE integrations (Amni-Code, VS Code WebView, etc).
Polls /persona/observe every 5s. ~320px wide; designed for narrow IDE sidebars. No external deps.
Embed via iframe: <iframe src="http://localhost:7700/persona/panel" width="320" height="600">"""
PERSONA_PANEL_HTML=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adam — Persona Mirror</title>
<style>
:root{--bg:#040711;--bg2:#0a1224;--cyan:#00e5ff;--cyan2:#00b8d4;--magenta:#ff2bd6;--gold:#ffd770;--err:#ff5577;--fg:#dff6ff;--mute:#5e7a99;--tint:#00e5ff}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--fg);font-family:"JetBrains Mono","SF Mono",Consolas,monospace;font-size:12px;overflow-y:auto}
#root{padding:14px 14px 20px;max-width:420px;margin:0 auto}
.head{display:flex;align-items:center;gap:8px;padding-bottom:10px;border-bottom:1px solid rgba(0,229,255,.15);margin-bottom:12px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--tint);box-shadow:0 0 8px var(--tint);transition:background .3s,box-shadow .3s}
.title{font-size:10px;letter-spacing:.22em;color:var(--mute);text-transform:uppercase}
.name{font-size:18px;color:var(--tint);font-weight:600;letter-spacing:.03em;text-shadow:0 0 6px var(--tint);transition:color .3s,text-shadow .3s}
.src{font-size:9px;color:var(--mute);letter-spacing:.18em;text-transform:uppercase;margin-top:2px}
.desc{font-size:11px;color:var(--fg);line-height:1.5;padding:10px;background:rgba(0,0,0,.35);border-left:2px solid var(--tint);border-radius:0 3px 3px 0;margin:8px 0;transition:border-color .3s}
.section{margin-top:12px}
.section-label{font-size:9px;letter-spacing:.22em;color:var(--mute);text-transform:uppercase;margin-bottom:6px}
.dims{display:grid;grid-template-columns:1fr;gap:5px;font-size:10px}
.dim{display:grid;grid-template-columns:80px 1fr 36px;gap:6px;align-items:center;color:var(--mute);letter-spacing:.08em;text-transform:uppercase}
.bar{height:4px;background:rgba(0,229,255,.08);border-radius:2px;overflow:hidden;border:1px solid rgba(0,229,255,.1)}
.bar-fill{height:100%;background:var(--tint);transition:width .35s ease-out,background .3s;box-shadow:0 0 4px var(--tint)}
.dim-val{color:var(--tint);text-align:right;font-family:inherit;font-size:10px;transition:color .3s}
.hints{font-size:10px;color:var(--mute);line-height:1.6;padding:8px 10px;background:rgba(0,0,0,.25);border:1px solid rgba(0,229,255,.08);border-radius:3px}
.hints .h{padding:2px 0}
.hints .h::before{content:'·  ';color:var(--tint)}
.samples{display:flex;flex-direction:column;gap:5px}
.sample{font-size:10.5px;color:var(--fg);padding:7px 10px;background:rgba(0,229,255,.04);border-left:2px solid rgba(0,229,255,.25);border-radius:0 3px 3px 0;line-height:1.5;font-style:italic;transition:border-color .3s}
.meta{font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:14px;padding-top:10px;border-top:1px dotted rgba(0,229,255,.1);display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}
.meta .k{color:var(--mute)}
.meta .v{color:var(--cyan2)}
.err{color:var(--err);font-size:11px;padding:10px;border:1px solid rgba(255,87,119,.3);border-radius:3px;background:rgba(255,87,119,.05)}
.tint-warm .dot,.tint-warm .bar-fill{background:#ffb547;box-shadow:0 0 8px #ffb547}
.tint-warm .name,.tint-warm .dim-val{color:#ffb547;text-shadow:0 0 6px #ffb547}
.tint-warm .desc,.tint-warm .sample{border-left-color:#ffb547}
.tint-spirited .dot,.tint-spirited .bar-fill{background:#ff2bd6;box-shadow:0 0 8px #ff2bd6}
.tint-spirited .name,.tint-spirited .dim-val{color:#ff2bd6;text-shadow:0 0 6px #ff2bd6}
.tint-spirited .desc,.tint-spirited .sample{border-left-color:#ff2bd6}
.tint-formal .dot,.tint-formal .bar-fill{background:#9fb8c8;box-shadow:0 0 8px #9fb8c8}
.tint-formal .name,.tint-formal .dim-val{color:#9fb8c8;text-shadow:0 0 6px #9fb8c8}
.tint-formal .desc,.tint-formal .sample{border-left-color:#9fb8c8}
</style>
</head><body>
<div id="root">
  <div class="head">
    <span class="dot" id="tint-dot"></span>
    <div>
      <div class="title">Adam · persona mirror</div>
      <div class="name" id="p-name">loading…</div>
      <div class="src" id="p-src"></div>
    </div>
  </div>
  <div class="desc" id="p-desc">Connecting to Adam server…</div>
  <div class="section">
    <div class="section-label">VOICE DIMENSIONS</div>
    <div class="dims" id="dims"></div>
  </div>
  <div class="section">
    <div class="section-label">VOICE HINTS</div>
    <div class="hints" id="hints"></div>
  </div>
  <div class="section">
    <div class="section-label">SAMPLE SENTENCES</div>
    <div class="samples" id="samples"></div>
  </div>
  <div class="meta" id="meta"></div>
</div>
<script>
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function bar(v){const pct=Math.max(0,Math.min(1,Number(v||0)))*100;return `<div class="bar"><div class="bar-fill" style="width:${pct.toFixed(0)}%"></div></div>`}
async function refresh(){
  try{
    const r=await fetch('/persona/observe');
    if(!r.ok){document.getElementById('p-desc').innerHTML='<span class="err">Server returned '+r.status+'</span>';return}
    const j=await r.json();const p=j.active||{};const tint=j.tint||null;
    document.body.className=tint?('tint-'+tint.name):'';
    document.getElementById('p-name').textContent=p.name||'?';
    document.getElementById('p-src').textContent=(p.source||'preset')+(tint?(' · '+tint.name):'');
    document.getElementById('p-desc').textContent=p.description||'(no description)';
    const dims=[['warmth',p.warmth],['formality',p.formality],['excitement',p.excitement],['length',p.length]];
    document.getElementById('dims').innerHTML=dims.map(([k,v])=>`<div class="dim"><span>${k}</span>${bar(v)}<span class="dim-val">${Number(v||0).toFixed(2)}</span></div>`).join('');
    const hints=(p.voice_hints||[]);
    document.getElementById('hints').innerHTML=hints.length?hints.map(h=>`<div class="h">${esc(h)}</div>`).join(''):'<div style="opacity:.5">(no hints)</div>';
    const samples=(j.samples||[]);
    document.getElementById('samples').innerHTML=samples.length?samples.map(s=>`<div class="sample">"${esc(s)}"</div>`).join(''):'<div style="opacity:.5;font-size:10px">(no samples)</div>';
    document.getElementById('meta').innerHTML=`<span><span class="k">default:</span> <span class="v">${esc(j.default||'?')}</span></span><span><span class="k">known:</span> <span class="v">${j.known_count||0}</span></span><span><span class="k">tts:</span> <span class="v">${esc(p.tts_voice||'?')}</span></span>`;
  }catch(e){document.getElementById('p-desc').innerHTML='<span class="err">Cannot reach Adam server: '+esc(String(e))+'</span>'}
}
refresh();setInterval(refresh,5000);
</script>
</body></html>"""
