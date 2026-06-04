"""Jarvis-themed Adam UI at /jarvis — neural-net particle background, neon glow, inline widget cards.
Calls /v1/chat/completions (widget-aware OpenAI compat) so amni_widgets[] cards render inline next to chat bubbles. Voice in (Web Speech API) + voice out (speechSynthesis). Tactical-overlay aesthetic.
Mount: jarvis_web.mount(app) — adds GET /jarvis."""
_HTML=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><link rel="manifest" href="/manifest.webmanifest"><meta name="theme-color" content="#040711"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"><meta name="apple-mobile-web-app-title" content="Adam"><link rel="apple-touch-icon" href="/assets/icons/adam-192.png">
<title>Adam — Jarvis Mode</title>
<link rel="stylesheet" href="/assets/katex/katex.min.css" onerror="this.onerror=null;this.href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'">
<script defer src="/assets/katex/katex.min.js" onload="window._katexReady=true;window._rerenderPendingMath&&window._rerenderPendingMath()" onerror="this.onerror=null;var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js';s.defer=true;s.onload=function(){window._katexReady=true;window._rerenderPendingMath&&window._rerenderPendingMath()};document.head.appendChild(s)"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js" onload="window._prismCoreReady=true"></script>
<script defer src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js" onload="if(window.Prism&&window.Prism.plugins&&window.Prism.plugins.autoloader){window.Prism.plugins.autoloader.languages_path='https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/';}window._prismReady=true"></script>
<script type="module">
  try{const m=await import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs');m.default.initialize({startOnLoad:false,theme:'dark',themeVariables:{primaryColor:'#00e5ff',primaryTextColor:'#dff6ff',primaryBorderColor:'#00e5ff',lineColor:'#00b8d4',secondaryColor:'#ff2bd6',tertiaryColor:'#0a1224',background:'#040711',mainBkg:'#0a1224',nodeBorder:'#00e5ff',edgeLabelBackground:'#040711',clusterBkg:'#0a1224',clusterBorder:'#00e5ff',titleColor:'#00e5ff',fontFamily:'JetBrains Mono, monospace'},securityLevel:'loose'});window._mermaid=m.default;window._mermaidReady=true}catch(e){console.debug('mermaid load failed:',e)}
</script>
<style>
:root{--bg:#040711;--bg2:#0a1224;--glass:rgba(10,18,36,.55);--cyan:#00e5ff;--cyan2:#00b8d4;--magenta:#ff2bd6;--gold:#ffd770;--ok:#00ff9d;--err:#ff5577;--fg:#dff6ff;--mute:#5e7a99;--c-rgb:0,229,255;--m-rgb:255,43,214;--g-rgb:255,224,102;font-family:"JetBrains Mono","SF Mono",Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--fg);overflow:hidden}
#nebula{position:fixed;inset:0;z-index:-2;background:radial-gradient(ellipse at 20% 30%,rgba(var(--c-rgb),.12),transparent 50%),radial-gradient(ellipse at 80% 70%,rgba(var(--m-rgb),.08),transparent 50%),var(--bg)}
#netcanvas{position:fixed;inset:0;z-index:-1;width:100vw;height:100vh}
#frame{position:fixed;inset:12px;border:1px solid rgba(var(--c-rgb),.18);border-radius:6px;pointer-events:none;z-index:5;box-shadow:inset 0 0 30px rgba(var(--c-rgb),.05)}
#frame::before,#frame::after{content:'';position:absolute;width:14px;height:14px;border:1.5px solid var(--cyan);box-shadow:0 0 8px rgba(var(--c-rgb),.5)}
#frame::before{top:-2px;left:-2px;border-right:none;border-bottom:none}
#frame::after{bottom:-2px;right:-2px;border-left:none;border-top:none}
#corner-tr,#corner-bl{position:fixed;width:14px;height:14px;border:1.5px solid var(--cyan);box-shadow:0 0 8px rgba(var(--c-rgb),.5);pointer-events:none;z-index:5}
#corner-tr{top:10px;right:10px;border-left:none;border-bottom:none}
#corner-bl{bottom:10px;left:10px;border-right:none;border-top:none}
#scanline{position:fixed;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);box-shadow:0 0 12px var(--cyan);pointer-events:none;animation:scan 6s linear infinite;opacity:.4;z-index:4}
@keyframes scan{0%{top:0}100%{top:100vh}}
#app{position:relative;z-index:10;display:grid;grid-template-rows:auto 1fr auto;height:100vh;padding:28px 32px;gap:18px}
header{display:flex;align-items:center;gap:14px;font-size:13px}
.title{font-size:20px;font-weight:700;letter-spacing:.18em;color:var(--cyan);text-shadow:0 0 10px var(--cyan),0 0 20px rgba(var(--c-rgb),.4)}
.status{display:flex;gap:14px;margin-left:auto;font-size:11px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase}
.status .pill{padding:3px 10px;border:1px solid rgba(var(--c-rgb),.3);border-radius:99px;background:rgba(var(--c-rgb),.05);color:var(--cyan);box-shadow:inset 0 0 8px rgba(var(--c-rgb),.1)}
.status .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);margin-right:6px;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
#chat-wrap{position:relative;overflow:hidden;border:1px solid rgba(var(--c-rgb),.15);border-radius:4px;background:var(--glass);backdrop-filter:blur(8px);box-shadow:inset 0 0 30px rgba(var(--c-rgb),.04)}
#log{height:100%;overflow-y:auto;padding:24px 28px;display:flex;flex-direction:column;gap:18px;scroll-behavior:smooth}
#log::-webkit-scrollbar{width:6px}
#log::-webkit-scrollbar-track{background:transparent}
#log::-webkit-scrollbar-thumb{background:rgba(var(--c-rgb),.25);border-radius:3px}
.welcome{text-align:center;padding:48px 24px;color:var(--mute)}
.welcome h2{font-size:14px;letter-spacing:.3em;color:var(--cyan);margin-bottom:18px;text-shadow:0 0 8px var(--cyan)}
.welcome .examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:24px;max-width:760px;margin-left:auto;margin-right:auto}
.welcome .ex{background:rgba(var(--c-rgb),.04);border:1px solid rgba(var(--c-rgb),.18);border-radius:3px;padding:12px 14px;cursor:pointer;font-family:inherit;color:var(--fg);font-size:11px;text-align:left;transition:all .2s}
.welcome .ex:hover{border-color:var(--cyan);box-shadow:0 0 14px rgba(var(--c-rgb),.4);transform:translateY(-1px)}
.welcome .ex .lbl{color:var(--cyan);font-size:9px;text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;display:block}
.msg{display:flex;flex-direction:column;gap:6px;max-width:84%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.bot{align-self:flex-start;align-items:flex-start}
.bubble{padding:12px 16px;border-radius:4px;font-size:13px;line-height:1.6;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word}
.msg.user .bubble{background:linear-gradient(135deg,rgba(var(--c-rgb),.18),rgba(0,184,212,.12));border:1px solid rgba(var(--c-rgb),.4);color:var(--fg);box-shadow:0 0 14px rgba(var(--c-rgb),.15)}
.msg.bot .bubble{background:rgba(var(--panel-rgb,8,14,28),.7);border:1px solid rgba(var(--c-rgb),.2);color:var(--fg);box-shadow:inset 0 0 12px rgba(var(--c-rgb),.05)}
.bubble code{background:rgba(0,0,0,.6);padding:1px 6px;border-radius:2px;font-size:11px;color:var(--cyan)}
.bubble pre{background:rgba(0,0,0,.7);border:1px solid rgba(var(--c-rgb),.15);padding:10px 12px;border-radius:3px;overflow-x:auto;font-size:11px;margin:6px 0;color:var(--cyan)}
.bubble pre code{background:none;padding:0;color:inherit}
.bubble .md-link{color:var(--cyan);text-decoration:none;border-bottom:1px dotted rgba(var(--c-rgb),.45);padding-bottom:1px;transition:color .12s,border-color .12s}
.bubble .md-link:hover{color:#dff6ff;border-bottom-color:var(--cyan);text-shadow:0 0 6px rgba(var(--c-rgb),.6)}
.bubble .md-link:visited{color:var(--cyan2)}
.bubble pre,.widget pre{position:relative}
.bubble pre .code-copy,.widget pre .code-copy{position:absolute;top:4px;right:4px;padding:2px 8px;background:rgba(var(--panel-rgb,8,14,28),.85);border:1px solid rgba(var(--c-rgb),.25);color:var(--mute);font-family:JetBrains Mono,monospace;font-size:8.5px;letter-spacing:.18em;text-transform:uppercase;cursor:pointer;border-radius:2px;opacity:0;transition:opacity .15s,color .15s,background .15s,border-color .15s;z-index:2}
.bubble pre:hover .code-copy,.widget pre:hover .code-copy,.bubble pre .code-copy.ok,.widget pre .code-copy.ok,.bubble pre .code-copy.err,.widget pre .code-copy.err{opacity:.95}
.bubble pre .code-copy:hover,.widget pre .code-copy:hover{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.1)}
.bubble pre .code-copy.ok,.widget pre .code-copy.ok{color:#00ff9c;border-color:rgba(0,255,156,.5);background:rgba(0,255,156,.08)}
.bubble pre .code-copy.err,.widget pre .code-copy.err{color:#ff7b7b;border-color:rgba(255,123,123,.5);background:rgba(255,123,123,.08)}
.bubble pre[class*="language-"]{background:rgba(0,0,0,.78);border:1px solid rgba(var(--c-rgb),.22);box-shadow:inset 0 0 12px rgba(var(--c-rgb),.05)}
.bubble pre[class*="language-"] code{font-family:JetBrains Mono,SF Mono,Consolas,monospace;font-size:11px;text-shadow:none}
.bubble .token.comment,.bubble .token.prolog,.bubble .token.doctype,.bubble .token.cdata{color:#5e7a99}
.bubble .token.punctuation{color:#7a98b8}
.bubble .token.namespace{opacity:.7}
.bubble .token.property,.bubble .token.tag,.bubble .token.boolean,.bubble .token.number,.bubble .token.constant,.bubble .token.symbol,.bubble .token.deleted{color:#ff2bd6}
.bubble .token.selector,.bubble .token.attr-name,.bubble .token.string,.bubble .token.char,.bubble .token.builtin,.bubble .token.inserted{color:#00ff9d}
.bubble .token.operator,.bubble .token.entity,.bubble .token.url,.bubble .language-css .token.string,.bubble .style .token.string{color:#ffd770}
.bubble .token.atrule,.bubble .token.attr-value,.bubble .token.keyword{color:#00e5ff;text-shadow:0 0 4px rgba(var(--c-rgb),.35)}
.bubble .token.function,.bubble .token.class-name{color:#00b8d4}
.bubble .token.regex,.bubble .token.important,.bubble .token.variable{color:#ff5577}
.bubble .token.important,.bubble .token.bold{font-weight:700}
.bubble .token.italic{font-style:italic}
.bubble .token.entity{cursor:help}
.bubble strong{color:var(--cyan);text-shadow:0 0 4px rgba(var(--c-rgb),.5)}
.meta{font-size:9px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;display:flex;gap:8px;align-items:center}
.meta .badge{padding:2px 8px;border:1px solid rgba(var(--c-rgb),.3);border-radius:99px;color:var(--cyan);background:rgba(var(--c-rgb),.05)}
.meta .badge.persona{border-color:rgba(var(--m-rgb),.5);color:var(--magenta);background:rgba(var(--m-rgb),.05)}
.meta .badge.tok{border-color:rgba(122,214,255,.3);color:#7ad6ff;background:rgba(122,214,255,.04)}
.meta .badge.err{border-color:rgba(255,91,91,.55);color:var(--err);background:rgba(255,91,91,.08)}
.meta .badge.warn{border-color:rgba(255,200,80,.5);color:#ffc850;background:rgba(255,200,80,.07)}
.meta .badge.ok{border-color:rgba(80,230,140,.5);color:#50e68c;background:rgba(80,230,140,.07)}
.surfnav{display:flex;gap:4px;margin-left:18px}
.surfnav .surfbtn{padding:4px 12px;border:1px solid rgba(var(--c-rgb),.25);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:JetBrains Mono,monospace;font-size:9.5px;letter-spacing:.18em;text-transform:uppercase;cursor:pointer;border-radius:3px;transition:all .15s}
.surfnav .surfbtn:hover{color:var(--cyan);border-color:rgba(var(--c-rgb),.5);background:rgba(var(--c-rgb),.08)}
.surfnav .surfbtn.active{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.1);text-shadow:0 0 6px var(--cyan)}
.thinking{color:var(--mute);font-style:italic;letter-spacing:.1em}
.thinking::after{content:'\2589';animation:blink .8s steps(2) infinite;margin-left:4px;color:var(--cyan)}
@keyframes blink{50%{opacity:0}}
.widgets{display:flex;flex-direction:column;gap:10px;width:100%;margin-top:4px}
.widget{border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.85);padding:14px 16px;position:relative;overflow:hidden;box-shadow:0 0 18px rgba(var(--c-rgb),.12),inset 0 0 24px rgba(var(--c-rgb),.04)}
.widget::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.6}
.widget .w-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan)}
.widget .w-icon{font-size:14px;filter:drop-shadow(0 0 4px var(--cyan))}
.widget .w-body{font-size:12px;color:var(--fg)}
.widget.weather .w-temp{font-size:36px;font-weight:300;color:var(--cyan);text-shadow:0 0 12px var(--cyan);letter-spacing:-.02em;display:flex;align-items:baseline;gap:6px}
.widget.weather .w-temp small{font-size:14px;color:var(--mute)}
.widget.weather .w-loc{font-size:11px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px}
.widget.weather .w-desc{font-size:13px;color:var(--fg);margin:4px 0 8px;text-transform:capitalize}
.widget.weather .w-stats{display:flex;gap:14px;font-size:10px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;border-top:1px solid rgba(var(--c-rgb),.12);padding-top:8px;margin-top:8px}
.widget.weather .w-stats .v{color:var(--cyan);font-size:12px;margin-left:4px}
.widget.system{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.widget.system .sys-card{border:1px solid rgba(var(--c-rgb),.15);padding:8px 10px;border-radius:3px;background:rgba(var(--c-rgb),.02)}
.widget.system .sys-card .lbl{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute)}
.widget.system .sys-card .val{font-size:18px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);margin-top:4px}
.widget.system .sys-card .bar{height:3px;background:rgba(var(--c-rgb),.1);border-radius:2px;margin-top:6px;overflow:hidden}
.widget.system .sys-card .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 8px var(--cyan);transition:width .3s}
.widget.file_change .fc-head{display:flex;gap:8px;align-items:baseline;font-size:13px;flex-wrap:wrap}
.widget.file_change .fc-op{font-size:9px;letter-spacing:.18em;padding:2px 6px;border-radius:2px;text-transform:uppercase;font-weight:bold}
.widget.file_change .fc-op.op-create{background:rgba(0,255,156,.15);color:#00ff9c;border:1px solid rgba(0,255,156,.3)}
.widget.file_change .fc-op.op-edit{background:rgba(var(--c-rgb),.12);color:var(--cyan);border:1px solid rgba(var(--c-rgb),.3)}
.widget.file_change .fc-op.op-overwrite{background:rgba(255,181,71,.12);color:#ffb547;border:1px solid rgba(255,181,71,.3)}
.widget.file_change .fc-bn{color:var(--fg);font-weight:600;font-family:JetBrains Mono,monospace}
.widget.file_change .fc-ext{color:var(--mute);font-size:11px;font-family:JetBrains Mono,monospace}
.widget.file_change .fc-folder{font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:4px;font-family:JetBrains Mono,monospace;word-break:break-all}
.widget.file_change .fc-stats{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;letter-spacing:.1em;color:var(--mute);margin:8px 0;border-top:1px solid rgba(var(--c-rgb),.08);padding-top:6px}
.widget.file_change .fc-add{color:#00ff9c;font-weight:bold}
.widget.file_change .fc-rem{color:#ff7b7b;font-weight:bold}
.widget.file_change .fc-repl{color:var(--cyan)}
.widget.file_change .fc-size{margin-left:auto;color:var(--mute)}
.widget.file_change .fc-preview{font-size:10px;font-family:JetBrains Mono,monospace;background:rgba(0,0,0,.35);border:1px solid rgba(var(--c-rgb),.1);border-radius:3px;padding:6px 8px;max-height:160px;overflow:auto;color:var(--fg);white-space:pre;line-height:1.4;margin:4px 0}
.widget.skill_error{border-color:rgba(255,91,91,.4);box-shadow:0 0 12px rgba(255,91,91,.2)}
.widget.skill_error .w-head{color:#ff7b7b;border-color:rgba(255,91,91,.3)}
.widget.skill_error .se-head{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.widget.skill_error .se-skill{font-size:13px;color:#ff7b7b;font-weight:600;letter-spacing:.05em;font-family:JetBrains Mono,monospace}
.widget.skill_error .se-status{font-size:9px;letter-spacing:.22em;color:#ff5b5b;padding:2px 6px;border:1px solid rgba(255,91,91,.4);border-radius:2px;background:rgba(255,91,91,.08)}
.widget.skill_error .se-msg{font-size:10.5px;color:var(--mute);font-style:italic;margin:5px 0;padding:4px 8px;border-left:2px solid rgba(255,91,91,.3);background:rgba(0,0,0,.2)}
.widget.skill_error .se-err{font-size:10.5px;color:#ffb7b7;background:rgba(255,91,91,.06);border:1px solid rgba(255,91,91,.2);border-radius:3px;padding:6px 8px;font-family:JetBrains Mono,monospace;white-space:pre-wrap;word-break:break-all;line-height:1.45;margin:5px 0}
.widget.skill_error .se-args{font-size:9px;color:var(--mute);letter-spacing:.04em;font-family:JetBrains Mono,monospace;margin-top:5px;opacity:.7}
.widget.skill_error .se-actions{display:flex;gap:6px;margin-top:8px}
.widget.skill_error .se-btn{padding:5px 10px;background:rgba(255,91,91,.06);border:1px solid rgba(255,91,91,.35);color:#ffb7b7;font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px;text-transform:uppercase}
.widget.skill_error .se-btn:hover{background:rgba(255,91,91,.14);border-color:#ff5b5b;color:#ff7b7b}
.widget.pc_confirm{border-color:rgba(var(--g-rgb),.45);box-shadow:0 0 14px rgba(var(--g-rgb),.18)}
.widget.pc_confirm .w-head{color:#ffe066;border-color:rgba(var(--g-rgb),.3)}
.widget.pc_confirm .pcw-risk{font-size:9px;letter-spacing:.2em;text-transform:uppercase;padding:3px 8px;border-radius:3px;display:inline-block;margin-bottom:7px;border:1px solid}
.widget.pc_confirm .pcw-low{color:var(--ok);border-color:var(--ok)}
.widget.pc_confirm .pcw-medium{color:#ffe066;border-color:#ffe066}
.widget.pc_confirm .pcw-high{color:var(--err);border-color:var(--err);text-shadow:0 0 4px var(--err)}
.widget.pc_confirm .pcw-target{font-family:JetBrains Mono,monospace;font-size:11px;color:var(--fg);background:rgba(0,0,0,.3);border:1px solid rgba(var(--g-rgb),.15);border-radius:3px;padding:6px 8px;word-break:break-all;white-space:pre-wrap;line-height:1.4;margin-bottom:6px}
.widget.pc_confirm .pcw-note{font-size:9px;color:var(--mute);letter-spacing:.03em;margin-bottom:8px}
.widget.pc_confirm .pcw-actions{display:flex;gap:8px}
.widget.pc_confirm .pcw-btn{flex:1;padding:7px 12px;font-family:inherit;font-size:10px;letter-spacing:.18em;cursor:pointer;border-radius:3px;text-transform:uppercase}
.widget.pc_confirm .pcw-btn>*{pointer-events:none}
.widget.pc_confirm .pcw-btn.confirm{border:1px solid var(--ok);background:rgba(0,255,140,.08);color:var(--ok)}
.widget.pc_confirm .pcw-btn.confirm:hover{background:rgba(0,255,140,.2)}
.widget.pc_confirm .pcw-btn.cancel{border:1px solid rgba(255,91,91,.4);background:rgba(255,91,91,.06);color:#ffb7b7}
.widget.pc_confirm .pcw-btn.cancel:hover{background:rgba(255,91,91,.16)}
.widget.pc_confirm.resolved{opacity:.55}
.widget.pc_confirm.resolved .pcw-actions{display:none}
.widget.file_change .fc-diff{font-size:10px;font-family:JetBrains Mono,monospace;background:rgba(0,0,0,.45);border:1px solid rgba(var(--c-rgb),.12);border-radius:3px;max-height:200px;overflow:auto;line-height:1.45;margin:4px 0}
.widget.file_change .fc-diff .dl{display:block;padding:1px 8px;white-space:pre;word-break:break-all}
.widget.file_change .fc-diff .dl.add{background:rgba(0,255,156,.08);color:#9eff9c;border-left:2px solid #00ff9c}
.widget.file_change .fc-diff .dl.rem{background:rgba(255,123,123,.08);color:#ffb7b7;border-left:2px solid #ff7b7b}
.widget.file_change .fc-diff .dl.hunk{color:var(--mute);background:rgba(var(--c-rgb),.05);border-left:2px solid rgba(var(--c-rgb),.3);font-style:italic}
.widget.file_change .fc-diff .dl.ctx{color:var(--mute)}
.widget.file_change .fc-viewtoggle{display:flex;gap:0;margin:4px 0 2px;font-size:9px;letter-spacing:.18em}
.widget.file_change .fc-viewtoggle button{flex:0 0 auto;padding:3px 9px;background:rgba(0,0,0,.4);border:1px solid rgba(var(--c-rgb),.18);color:var(--mute);font-family:inherit;cursor:pointer}
.widget.file_change .fc-viewtoggle button:first-child{border-radius:2px 0 0 2px}
.widget.file_change .fc-viewtoggle button:last-child{border-radius:0 2px 2px 0}
.widget.file_change .fc-viewtoggle button:not(:last-child){border-right:none}
.widget.file_change .fc-viewtoggle button.on{background:rgba(var(--c-rgb),.15);color:var(--cyan);border-color:var(--cyan)}
.widget.file_change .fc-viewtoggle button:hover:not(.on){background:rgba(var(--c-rgb),.06);color:var(--fg)}
.widget.file_change .fc-actions{display:flex;gap:6px;margin-top:6px}
.widget.file_change .fc-btn{flex:0 0 auto;padding:4px 10px;background:rgba(var(--c-rgb),.06);border:1px solid rgba(var(--c-rgb),.25);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
.widget.file_change .fc-btn:hover{background:rgba(var(--c-rgb),.14);border-color:var(--cyan)}
.widget.file_change .fc-verify{margin-left:auto;font-size:9px;letter-spacing:.18em;padding:2px 6px;border-radius:2px;font-weight:bold;cursor:help}
.widget.file_change .fc-verify.pass{background:rgba(0,255,156,.15);color:#00ff9c;border:1px solid rgba(0,255,156,.3);text-shadow:0 0 4px rgba(0,255,156,.5)}
.widget.file_change .fc-verify.fail{background:rgba(255,91,91,.15);color:#ff5b5b;border:1px solid rgba(255,91,91,.4);text-shadow:0 0 4px rgba(255,91,91,.5);animation:fcShake .35s ease-out}
.widget.file_change .fc-verify.manual{background:rgba(255,181,71,.1);color:#ffb547;border:1px solid rgba(255,181,71,.3)}
@keyframes fcShake{0%,100%{transform:translateX(0)}25%{transform:translateX(-2px)}75%{transform:translateX(2px)}}
.widget.file_change .fc-issues{background:rgba(255,91,91,.06);border:1px solid rgba(255,91,91,.25);border-radius:3px;padding:6px 8px;margin:6px 0;font-size:10px;color:#ff7b7b;font-family:JetBrains Mono,monospace}
.widget.file_change .fc-issue{margin:2px 0}
.widget.file_change .fc-suggested{font-size:10px;color:#ffb547;letter-spacing:.05em;margin:6px 0;padding:4px 8px;background:rgba(255,181,71,.05);border-left:2px solid rgba(255,181,71,.3);border-radius:0 3px 3px 0}
.widget.file_change .fc-suggested code{background:transparent;color:var(--cyan);font-size:10px}
.widget.file_change .fc-test-run{margin:6px 0;padding:6px 10px;border-radius:3px;font-size:10px;font-family:JetBrains Mono,monospace;border-left:2px solid}
.widget.file_change .fc-test-run.pass{background:rgba(0,255,156,.06);border-left-color:#00ff9c;color:#a0ffd0}
.widget.file_change .fc-test-run.fail{background:rgba(255,91,91,.06);border-left-color:#ff5b5b;color:#ffb0b0}
.widget.file_change .fc-test-run.timeout{background:rgba(255,181,71,.06);border-left-color:#ffb547;color:#ffd699}
.widget.file_change .fc-test-head{font-size:10px;letter-spacing:.12em;font-weight:bold;text-transform:uppercase}
.widget.file_change .fc-test-fails{margin-top:5px;font-family:JetBrains Mono,monospace;font-size:9px}
.widget.file_change .fc-test-fail{padding:3px 0;border-top:1px solid rgba(255,91,91,.1)}
.widget.file_change .fc-test-fail .t{color:#ff7b7b;font-weight:bold}
.widget.file_change .fc-test-fail .m{color:var(--mute);display:block;margin-left:8px;font-size:9px;line-height:1.3}
.widget.news .w-body{display:flex;flex-direction:column;gap:6px;max-height:280px;overflow-y:auto}
.widget.news .news-item{padding:6px 8px;border:1px solid rgba(var(--c-rgb),.08);border-radius:3px;background:rgba(var(--c-rgb),.02);text-decoration:none;color:var(--fg);display:block;transition:all .15s}
.widget.news .news-item:hover{border-color:rgba(var(--c-rgb),.4);background:rgba(var(--c-rgb),.05);box-shadow:0 0 8px rgba(var(--c-rgb),.15)}
.widget.news .news-item .title{font-size:12px;line-height:1.3}
.widget.news .news-item .src{font-size:9px;color:var(--cyan);letter-spacing:.15em;text-transform:uppercase;margin-top:3px}
.widget.stock .quotes{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.widget.stock .quote{padding:10px;border:1px solid rgba(var(--c-rgb),.15);border-radius:3px;background:rgba(var(--c-rgb),.02)}
.widget.stock .quote .sym{font-size:14px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);letter-spacing:.1em;font-weight:600}
.widget.stock .quote .name{font-size:9px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
.widget.stock .quote .price{font-size:22px;color:var(--fg);text-shadow:0 0 4px rgba(var(--c-rgb),.4)}
.widget.stock .quote .chg{font-size:11px;margin-top:3px}
.widget.stock .quote .chg.up{color:var(--ok);text-shadow:0 0 4px var(--ok)}
.widget.stock .quote .chg.down{color:var(--err);text-shadow:0 0 4px var(--err)}
.widget.stock .quote .meta{font-size:9px;color:var(--mute);margin-top:6px;border-top:1px solid rgba(var(--c-rgb),.08);padding-top:4px}
.widget.file .file-meta{display:flex;gap:12px;font-size:9px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px}
.widget.file .file-meta .v{color:var(--cyan);margin-left:4px}
.widget.file pre{background:rgba(0,0,0,.6);padding:10px;border-radius:2px;font-size:10px;color:var(--cyan);max-height:300px;overflow:auto;line-height:1.4}
.widget.disk .partitions{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}
.widget.disk .part{padding:8px 10px;border:1px solid rgba(var(--c-rgb),.12);border-radius:3px;background:rgba(var(--c-rgb),.02)}
.widget.disk .part .mount{font-size:11px;color:var(--cyan);text-shadow:0 0 4px var(--cyan);letter-spacing:.1em;font-family:Consolas,monospace}
.widget.disk .part .stat{font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:var(--mute);margin-top:4px}
.widget.disk .part .stat .v{color:var(--fg);margin-left:4px}
.widget.disk .part .bar{height:3px;background:rgba(var(--c-rgb),.1);border-radius:2px;margin-top:6px;overflow:hidden}
.widget.disk .part .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 4px var(--cyan)}
.widget.git .git-branch{font-size:14px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);letter-spacing:.1em}
.widget.git .git-stats{display:flex;gap:14px;font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--mute);margin:6px 0 10px;padding-bottom:6px;border-bottom:1px solid rgba(var(--c-rgb),.08)}
.widget.git .git-stats .v{color:var(--cyan);margin-left:4px}
.widget.git .git-stats .v.dirty{color:var(--gold)}
.widget.git .commits{font-size:10px;color:var(--fg);font-family:Consolas,monospace;line-height:1.5}
.widget.git .commits .row{padding:2px 0;border-bottom:1px solid rgba(var(--c-rgb),.05)}
.widget.git .commits .row:last-child{border-bottom:none}
.widget.git .commits .row .sha{color:var(--cyan);margin-right:8px}
.widget.git .dirty-files{margin-top:8px;font-size:10px;color:var(--gold);font-family:Consolas,monospace}
.widget.git .dirty-files .lbl{color:var(--mute);font-size:9px;letter-spacing:.15em;text-transform:uppercase;margin-bottom:3px}
.widget.watch .watch-head{font-size:11px;color:var(--cyan);letter-spacing:.1em;font-family:Consolas,monospace;margin-bottom:8px;border-bottom:1px solid rgba(var(--c-rgb),.1);padding-bottom:6px;display:flex;justify-content:space-between}
.widget.watch .watch-head .id{color:var(--mute);font-size:9px;letter-spacing:.2em}
.widget.watch .events{display:flex;flex-direction:column;gap:3px;font-family:Consolas,monospace;font-size:10px;max-height:260px;overflow-y:auto}
.widget.watch .event{padding:3px 6px;border-radius:2px;background:rgba(var(--c-rgb),.02);display:grid;grid-template-columns:60px 1fr auto;gap:8px;align-items:center}
.widget.watch .event .kind{font-size:8px;letter-spacing:.2em;text-transform:uppercase;text-align:right}
.widget.watch .event.created .kind{color:var(--ok);text-shadow:0 0 4px var(--ok)}
.widget.watch .event.modified .kind{color:var(--cyan);text-shadow:0 0 4px var(--cyan)}
.widget.watch .event.deleted .kind{color:var(--err);text-shadow:0 0 4px var(--err)}
.widget.watch .event .path{color:var(--fg);word-break:break-all;line-height:1.3}
.widget.watch .event .size{color:var(--mute);font-size:9px}
.widget.time .w-clock{font-size:32px;color:var(--cyan);text-shadow:0 0 12px var(--cyan);letter-spacing:.1em;font-weight:300}
.widget.time .w-date{font-size:11px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-top:4px}
.widget.time .w-tz{font-size:9px;color:var(--mute);margin-top:2px}
.widget.error{border-color:rgba(255,85,119,.5);box-shadow:0 0 18px rgba(255,85,119,.2)}
.widget.error .w-head{color:var(--err)}
.widget.info{border-color:rgba(255,215,112,.4);box-shadow:0 0 14px rgba(255,215,112,.15)}
.widget.info .w-head{color:var(--gold)}
.widget.code .w-body pre{background:rgba(0,0,0,.6);padding:10px;border-radius:2px;font-size:11px;color:var(--cyan);overflow-x:auto}
#composer{display:flex;gap:10px;align-items:center}
#jarvis-toggle{padding:8px 14px;background:rgba(var(--c-rgb),.06);border:1px solid rgba(var(--c-rgb),.4);color:var(--cyan);font-family:inherit;font-size:11px;letter-spacing:.25em;cursor:pointer;border-radius:3px;transition:all .18s;text-shadow:0 0 4px rgba(var(--c-rgb),.5);font-weight:600}
#jarvis-toggle:hover{background:rgba(var(--c-rgb),.14);box-shadow:0 0 12px rgba(var(--c-rgb),.35)}
#jarvis-toggle.on{background:linear-gradient(180deg,rgba(var(--c-rgb),.22),rgba(var(--c-rgb),.08));border-color:var(--cyan);color:#dff6ff;box-shadow:0 0 18px rgba(var(--c-rgb),.6),inset 0 0 12px rgba(var(--c-rgb),.18);animation:jarvisPulse 2.4s ease-in-out infinite}
@keyframes jarvisPulse{0%,100%{box-shadow:0 0 18px rgba(var(--c-rgb),.55),inset 0 0 12px rgba(var(--c-rgb),.15)}50%{box-shadow:0 0 24px rgba(var(--c-rgb),.85),inset 0 0 16px rgba(var(--c-rgb),.28)}}
#status-pill .sp-led{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--mute);margin-right:6px;transition:background .25s,box-shadow .25s;vertical-align:middle}
#status-pill.attention .sp-led{background:#ffb547;box-shadow:0 0 6px #ffb547}
#status-pill.error .sp-led{background:#ff5b5b;box-shadow:0 0 6px #ff5b5b}
#status-pill.ok .sp-led{background:#00ff9c;box-shadow:0 0 5px #00ff9c}
#status-panel{position:fixed;top:60px;right:24px;width:300px;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;box-shadow:0 0 24px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);padding:12px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#status-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#status-panel.td-hidden{display:block}
#status-panel .sp-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:8px;font-size:9.5px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;text-shadow:0 0 4px var(--cyan)}
#status-panel .sp-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#status-panel .sp-close:hover{color:var(--err);border-color:var(--err)}
#status-panel .sp-row{display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:3px;font-size:11px;color:var(--fg);cursor:pointer;transition:background .12s;font-family:JetBrains Mono,monospace;letter-spacing:.04em}
#status-panel .sp-row:hover{background:rgba(var(--c-rgb),.08)}
#status-panel .sp-name{flex:1;color:var(--fg)}
#status-panel .sp-val{color:var(--cyan2);font-size:10.5px;letter-spacing:.05em}
#status-panel .sp-arrow{color:var(--mute);font-size:13px}
#status-panel .sp-led-inline{width:6px;height:6px;border-radius:50%;background:var(--mute);box-shadow:0 0 4px transparent;transition:background .2s,box-shadow .2s}
#bookmarks-panel{position:fixed;top:60px;right:24px;width:380px;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--g-rgb),.4);border-radius:4px;box-shadow:0 0 24px rgba(var(--g-rgb),.18);backdrop-filter:blur(8px);padding:12px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#bookmarks-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#bookmarks-panel.td-hidden{display:block}
#bookmarks-panel .sp-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--g-rgb),.2);margin-bottom:8px;font-size:9.5px;letter-spacing:.3em;color:#ffe066;text-transform:uppercase;text-shadow:0 0 4px #ffe066}
#bookmarks-panel .sp-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--g-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#bookmarks-panel .sp-close:hover{color:var(--err);border-color:var(--err)}
.bm-row{margin:6px 0;padding:7px 9px;background:rgba(var(--g-rgb),.04);border-left:2px solid rgba(var(--g-rgb),.4);border-radius:0 3px 3px 0;font-size:10.5px;line-height:1.5;position:relative;font-family:JetBrains Mono,monospace}
.bm-row .bm-head{display:flex;align-items:center;gap:8px;margin-bottom:3px}
.bm-row .bm-ts{font-size:9px;letter-spacing:.15em;color:#ffd770;flex:1}
.bm-row .bm-pers{font-size:8.5px;letter-spacing:.15em;color:var(--magenta);padding:1px 6px;border:1px solid rgba(var(--m-rgb),.3);border-radius:2px}
.bm-row .bm-del{background:none;border:1px solid rgba(255,91,91,.25);color:var(--mute);font-size:9px;padding:1px 6px;border-radius:2px;cursor:pointer;font-family:inherit}
.bm-row .bm-del:hover{color:#ff5b5b;border-color:#ff5b5b;background:rgba(255,91,91,.1)}
.bm-row .bm-q{color:var(--mute);font-style:italic;font-size:10px}
.bm-row .bm-a{color:var(--fg);margin-top:3px}
.bm-row .bm-note{color:#ffd770;margin-top:3px;font-size:10px}
#reminders-panel{position:fixed;top:60px;right:24px;width:380px;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;box-shadow:0 0 24px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);padding:12px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#reminders-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#reminders-panel.td-hidden{display:block}
#reminders-panel .sp-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:8px;font-size:9.5px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;text-shadow:0 0 4px var(--cyan)}
#reminders-panel .sp-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#reminders-panel .sp-close:hover{color:var(--err);border-color:var(--err)}
.rm-row{margin:6px 0;padding:7px 9px;background:rgba(var(--c-rgb),.04);border-left:2px solid rgba(var(--c-rgb),.4);border-radius:0 3px 3px 0;font-size:10.5px;line-height:1.5;font-family:JetBrains Mono,monospace}
.rm-row.rm-due{background:rgba(255,91,91,.06);border-left-color:#ff5b5b;animation:rmDuePulse 1.8s ease-in-out infinite}
@keyframes rmDuePulse{0%,100%{box-shadow:0 0 4px rgba(255,91,91,.2)}50%{box-shadow:0 0 12px rgba(255,91,91,.5)}}
.rm-row .rm-head{display:flex;align-items:center;gap:8px;margin-bottom:3px}
.rm-row .rm-icon{font-size:11px;color:var(--cyan);width:14px;text-align:center}
.rm-row.rm-due .rm-icon{color:#ff7b7b;font-weight:bold}
.rm-row .rm-due-iso{font-size:9.5px;letter-spacing:.1em;color:var(--mute);flex:1}
.rm-row.rm-due .rm-due-iso{color:#ffb7b7}
.rm-row .rm-del{background:rgba(0,255,156,.05);border:1px solid rgba(0,255,156,.25);color:#00ff9c;font-size:11px;padding:1px 8px;border-radius:2px;cursor:pointer;font-family:inherit;line-height:1}
.rm-row .rm-del:hover{background:rgba(0,255,156,.18);border-color:#00ff9c;box-shadow:0 0 8px rgba(0,255,156,.3)}
.rm-row .rm-text{color:var(--fg);font-style:italic}
#tools-toggle{padding:8px 12px;background:rgba(var(--c-rgb),.04);border:1px solid rgba(var(--c-rgb),.22);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.22em;cursor:pointer;border-radius:3px;transition:all .15s}
#tools-toggle:hover{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.1)}
#tools-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.12)}
#tools-drawer{position:fixed;bottom:84px;right:24px;width:340px;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;box-shadow:0 0 28px rgba(var(--c-rgb),.22);backdrop-filter:blur(10px);padding:14px;transform:translateY(10px);opacity:0;pointer-events:none;transition:opacity .2s ease-out, transform .25s ease-out}
#tools-drawer.show{transform:translateY(0);opacity:1;pointer-events:auto}
#tools-drawer.td-hidden{display:block}
#tools-drawer .td-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:10px;font-size:10px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;text-shadow:0 0 4px var(--cyan)}
#tools-drawer .td-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#tools-drawer .td-close:hover{color:var(--err);border-color:var(--err)}
#tools-drawer .td-section{margin-bottom:14px}
#tools-drawer .td-section:last-child{margin-bottom:0}
#tools-drawer .td-label{font-size:8.5px;letter-spacing:.28em;color:var(--mute);text-transform:uppercase;margin-bottom:6px}
#tools-drawer .td-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
#tools-drawer .td-btn{padding:7px 10px;background:rgba(var(--c-rgb),.04);border:1px solid rgba(var(--c-rgb),.22);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:3px;transition:all .15s;text-transform:uppercase;text-align:center}
#tools-drawer .td-btn:hover{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.1)}
#tools-drawer .td-btn.on{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.14);box-shadow:0 0 8px rgba(var(--c-rgb),.25);text-shadow:0 0 4px var(--cyan)}
#tools-drawer .td-btn .convo-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--mute);margin-right:5px;transition:background .2s}
#tools-drawer .td-btn.on .convo-dot{background:var(--cyan);box-shadow:0 0 4px var(--cyan)}
#quick-bar{display:flex;gap:6px;padding:6px 12px 0;flex-wrap:wrap;align-items:center;font-family:JetBrains Mono,monospace;border-top:1px solid rgba(var(--c-rgb),.05)}
.qchip{padding:5px 11px;border:1px solid rgba(var(--c-rgb),.2);background:rgba(var(--c-rgb),.04);color:var(--mute);font-size:10px;letter-spacing:.12em;text-transform:uppercase;cursor:pointer;border-radius:99px;font-family:inherit;transition:all .15s;display:inline-flex;align-items:center;gap:6px;line-height:1}
.qchip:hover{background:rgba(var(--c-rgb),.14);border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 8px rgba(var(--c-rgb),.25)}
.qchip .ico{font-size:13px}
.qchip.qc-coach:hover{background:rgba(255,77,200,.14);border-color:var(--magenta);color:var(--magenta);box-shadow:0 0 8px rgba(255,77,200,.25)}
.qchip.qc-export:hover{background:rgba(0,255,156,.14);border-color:#00ff9c;color:#00ff9c;box-shadow:0 0 8px rgba(0,255,156,.25)}
.qchip.qc-learn:hover{background:rgba(255,181,71,.12);border-color:#ffb547;color:#ffb547;box-shadow:0 0 8px rgba(255,181,71,.22)}
.qchip-hint{font-size:9px;color:var(--mute);letter-spacing:.15em;margin-left:auto;opacity:.6;text-transform:uppercase}
#quick-bar.qb-collapsed .qb-item{display:none}
#quick-bar:not(.qb-collapsed) .qb-toggle{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.12)}
.qb-toggle .ico{font-size:14px;letter-spacing:0}
#mic-shell{position:relative;width:46px;height:46px;border:1px solid rgba(var(--c-rgb),.4);border-radius:50%;background:rgba(var(--c-rgb),.05);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:18px;color:var(--cyan);transition:all .2s;flex-shrink:0}
#mic-shell:hover{box-shadow:0 0 14px var(--cyan);border-color:var(--cyan)}
#mic-shell.listening{background:rgba(var(--m-rgb),.15);border-color:var(--magenta);color:var(--magenta);box-shadow:0 0 18px var(--magenta);animation:pulseMic 1.2s ease-in-out infinite}
@keyframes pulseMic{0%,100%{box-shadow:0 0 18px var(--magenta)}50%{box-shadow:0 0 28px var(--magenta)}}
#input-shell{flex:1;position:relative;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);border-radius:4px;display:flex;align-items:center;transition:all .2s}
#input-shell:focus-within{border-color:var(--cyan);box-shadow:0 0 14px rgba(var(--c-rgb),.2)}
#input-shell::before{content:'>';color:var(--cyan);font-size:14px;padding-left:14px;text-shadow:0 0 4px var(--cyan)}
#input{flex:1;background:transparent;border:0;color:var(--fg);padding:13px 14px;font-family:inherit;font-size:13px;resize:none;outline:none;line-height:1.4;min-height:46px;max-height:160px}
#input::placeholder{color:var(--mute);font-style:italic}
#send{padding:0 22px;height:46px;border:1px solid var(--cyan);background:rgba(var(--c-rgb),.1);color:var(--cyan);font-family:inherit;font-size:11px;letter-spacing:.25em;text-transform:uppercase;cursor:pointer;border-radius:4px;transition:all .2s;text-shadow:0 0 4px var(--cyan)}
#send:hover:not(:disabled){background:var(--cyan);color:var(--bg);box-shadow:0 0 18px var(--cyan)}
#send.stop-mode{border-color:#ff5577;color:#ff8aa2;background:rgba(255,85,119,.1);text-shadow:0 0 4px rgba(255,85,119,.6);animation:stopPulse 1.6s ease-in-out infinite}
#send.stop-mode:hover{background:#ff5577;color:var(--bg);box-shadow:0 0 18px rgba(255,85,119,.7);animation:none}
@keyframes stopPulse{0%,100%{box-shadow:0 0 6px rgba(255,85,119,.4)}50%{box-shadow:0 0 18px rgba(255,85,119,.85)}}
#send:disabled{opacity:.3;cursor:wait}
#voiceout-toggle{padding:0 14px;height:46px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#voiceout-toggle.on{color:var(--gold);border-color:var(--gold);background:rgba(255,215,112,.08)}
.md-table{border-collapse:collapse;margin:9px 0;font-size:12px;max-width:100%;display:block;overflow-x:auto}
.md-table th,.md-table td{border:1px solid rgba(var(--c-rgb,0,229,255),.22);padding:5px 10px;text-align:left;vertical-align:top}
.md-table th{background:rgba(var(--c-rgb,0,229,255),.10);font-weight:600;white-space:nowrap}
.md-table tbody tr:nth-child(even) td{background:rgba(var(--c-rgb,0,229,255),.045)}
.sidehint{position:fixed;bottom:14px;left:36px;font-size:9px;color:var(--mute);letter-spacing:.2em;z-index:3;pointer-events:none;opacity:.6}
#wd-pill{position:fixed;bottom:14px;right:36px;font-size:9px;color:var(--mute);letter-spacing:.18em;text-transform:uppercase;font-family:JetBrains Mono,monospace;z-index:5;padding:3px 9px;border:1px solid rgba(var(--c-rgb),.18);background:rgba(var(--panel-rgb,8,14,28),.65);border-radius:99px;cursor:pointer;backdrop-filter:blur(6px);transition:all .15s;display:none}
#wd-pill.show{display:inline-block}
#wd-pill:hover{color:var(--cyan);border-color:rgba(var(--c-rgb),.45);box-shadow:0 0 10px rgba(var(--c-rgb),.2)}
#wd-pill .wd-lbl{color:var(--cyan);margin-right:6px}
#wd-pill.unrestricted{border-color:rgba(255,181,71,.35)}
#wd-pill.unrestricted .wd-lbl{color:#ffb547}
#wd-panel{position:fixed;bottom:50px;right:24px;width:380px;max-height:60vh;z-index:11;border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);display:none;overflow:hidden;flex-direction:column}
#wd-panel.show{display:flex}
#wd-panel .wp-head{padding:10px 14px;border-bottom:1px solid rgba(var(--c-rgb),.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center}
#wd-panel .wp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:10px}
#wd-panel .wp-head .close:hover{color:var(--err);border-color:var(--err)}
#wd-panel .wp-base{padding:6px 14px;font-size:9px;color:var(--mute);font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(var(--c-rgb),.08);word-break:break-all;letter-spacing:.02em}
#wd-panel .wp-list{flex:1;overflow-y:auto;padding:6px 8px}
#wd-panel .wp-row{display:flex;align-items:center;gap:6px;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:11px;font-family:JetBrains Mono,monospace;transition:background .12s}
#wd-panel .wp-row:hover{background:rgba(var(--c-rgb),.08)}
#wd-panel .wp-row .wp-icon{width:18px;text-align:center;color:var(--mute);font-size:12px}
#wd-panel .wp-row.dir .wp-icon{color:var(--cyan)}
#wd-panel .wp-row.dir .wp-name{color:var(--cyan)}
#wd-panel .wp-row .wp-name{flex:1;color:var(--fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#wd-panel .wp-row .wp-size{color:var(--mute);font-size:9px;letter-spacing:.05em;flex-shrink:0}
#wd-panel .wp-empty{padding:18px;text-align:center;color:var(--mute);font-size:10px;font-style:italic}
#wd-panel .wp-trunc{padding:6px 14px;font-size:9px;color:#ffb547;border-top:1px solid rgba(255,181,71,.2);text-align:center;letter-spacing:.05em}
.status .pill.clickable{cursor:pointer}
.status .pill.clickable:hover{border-color:var(--cyan);background:rgba(var(--c-rgb),.12)}
#persona-panel{position:fixed;top:60px;right:24px;width:300px;z-index:11;border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(var(--c-rgb),.22);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#persona-panel.show{display:block}
#persona-panel .pp-head{padding:10px 14px;border-bottom:1px solid rgba(var(--c-rgb),.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#persona-panel .pp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.2);border-radius:3px;font-size:10px}
#persona-panel .pp-head .close:hover{color:var(--err);border-color:var(--err)}
#persona-panel .pp-section{padding:12px 14px;border-bottom:1px solid rgba(var(--c-rgb),.08)}
#persona-panel .pp-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
#persona-panel .pp-row{padding:6px 8px;border:1px solid rgba(var(--c-rgb),.12);border-radius:3px;margin-bottom:4px;cursor:pointer;font-size:11px;display:flex;justify-content:space-between;align-items:center;transition:all .15s;background:rgba(var(--c-rgb),.02)}
#persona-panel .pp-row:hover{border-color:rgba(var(--c-rgb),.4);background:rgba(var(--c-rgb),.08)}
#persona-panel .pp-row.active{border-color:var(--cyan);background:rgba(var(--c-rgb),.14);box-shadow:inset 0 0 6px rgba(var(--c-rgb),.2)}
#persona-panel .pp-row .nm{color:var(--fg);text-transform:capitalize}
#persona-panel .pp-row.active .nm{color:var(--cyan);text-shadow:0 0 4px var(--cyan)}
#persona-panel .pp-row .voice{font-size:9px;color:var(--mute);letter-spacing:.1em}
#persona-panel .pp-empty{font-size:10px;color:var(--mute);text-align:center;padding:10px;font-style:italic}
#persona-panel .pe-desc{font-size:10px;color:var(--fg);background:rgba(0,0,0,.45);border:1px solid rgba(var(--c-rgb),.15);border-radius:3px;padding:6px 8px;line-height:1.45;margin-bottom:8px;max-height:120px;overflow:auto;font-family:JetBrains Mono,monospace}
#persona-panel .pe-desc[contenteditable=true]{outline:1px solid rgba(var(--c-rgb),.35);background:rgba(var(--c-rgb),.04)}
#persona-panel .pe-row{display:flex;align-items:center;gap:8px;font-size:10px;margin:5px 0;color:var(--mute);letter-spacing:.1em}
#persona-panel .pe-row label{flex:0 0 80px;text-transform:uppercase}
#persona-panel .pe-row input[type=range]{flex:1;accent-color:var(--cyan);height:14px}
#persona-panel .pe-row .pe-val{flex:0 0 38px;color:var(--cyan);text-align:right;font-family:JetBrains Mono,monospace}
#persona-panel .pe-hints{font-size:9.5px;color:var(--mute);background:rgba(0,0,0,.35);border:1px solid rgba(var(--c-rgb),.1);border-radius:3px;padding:6px 8px;margin:6px 0;line-height:1.5;letter-spacing:.04em;max-height:90px;overflow:auto;font-family:JetBrains Mono,monospace;white-space:pre-wrap}
#persona-panel .pe-source{font-size:8.5px;color:var(--mute);letter-spacing:.18em;margin:4px 0 6px}
#persona-panel .pe-actions{display:flex;gap:6px;margin-top:8px}
#persona-panel .pe-btn{flex:1;padding:6px 8px;background:rgba(var(--c-rgb),.06);border:1px solid rgba(var(--c-rgb),.3);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px;text-transform:uppercase}
#persona-panel .pe-btn:hover{background:rgba(var(--c-rgb),.16);border-color:var(--cyan)}
#persona-panel .pe-btn.danger{color:#ff7b7b;border-color:rgba(255,123,123,.35)}
#persona-panel .pe-btn.danger:hover{background:rgba(255,123,123,.12);border-color:#ff7b7b}
#persona-panel .pp-input-row{display:flex;gap:6px}
#persona-panel input[type=text]{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(var(--c-rgb),.2);color:var(--fg);padding:5px 8px;border-radius:3px;font-family:inherit;font-size:11px}
#persona-panel input[type=text]:focus{outline:none;border-color:var(--cyan)}
#persona-panel button.act{padding:5px 10px;border:1px solid rgba(var(--c-rgb),.4);background:rgba(var(--c-rgb),.04);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#persona-panel button.act:hover{background:rgba(var(--c-rgb),.14)}
.ld-led{display:inline-block;width:6px;height:6px;border-radius:50%;background:#4a5568;box-shadow:0 0 4px #4a5568;margin-right:6px;vertical-align:middle;transition:background .25s, box-shadow .25s}
.ld-led.idle{background:#4a5568;box-shadow:0 0 4px #4a5568}
.ld-led.active{background:#00ff9c;box-shadow:0 0 8px #00ff9c, 0 0 14px rgba(0,255,156,.5);animation:ldPulse 1.4s ease-in-out infinite}
.ld-led.paused{background:#ffb547;box-shadow:0 0 6px #ffb547}
.ld-led.error{background:#ff5b5b;box-shadow:0 0 6px #ff5b5b}
.tp-led{display:inline-block;width:6px;height:6px;border-radius:50%;background:#4a5568;box-shadow:0 0 4px #4a5568;margin-right:6px;vertical-align:middle;transition:background .25s, box-shadow .25s}
.tp-led.empty{background:#00ff9c;box-shadow:0 0 6px #00ff9c}
.tp-led.pending{background:#ffb547;box-shadow:0 0 8px #ffb547, 0 0 14px rgba(255,181,71,.4);animation:tpPulse 1.6s ease-in-out infinite}
.tp-led.failed{background:#ff5b5b;box-shadow:0 0 8px #ff5b5b, 0 0 14px rgba(255,91,91,.5);animation:tpPulse 1.0s ease-in-out infinite}
.tp-led.error{background:#4a5568;box-shadow:0 0 4px #4a5568}
@keyframes tpPulse{0%,100%{opacity:1}50%{opacity:.5}}
#tests-panel{position:fixed;top:60px;right:24px;width:380px;z-index:11;border:1px solid rgba(255,181,71,.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(255,181,71,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#tests-panel.show{display:block}
#tests-panel .tp-head{padding:10px 14px;border-bottom:1px solid rgba(255,181,71,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:#ffb547;text-shadow:0 0 4px #ffb547;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#tests-panel .tp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(255,181,71,.22);border-radius:3px;font-size:10px}
#tests-panel .tp-head .close:hover{color:var(--err);border-color:var(--err)}
#tests-panel .tp-section{padding:12px 14px;border-bottom:1px solid rgba(255,181,71,.08)}
#tests-panel .tp-item{padding:8px 10px;border:1px solid rgba(255,181,71,.18);border-radius:3px;background:rgba(255,181,71,.04);margin-bottom:6px}
#tests-panel .tp-item .path{font-family:JetBrains Mono,monospace;font-size:11px;color:var(--fg);word-break:break-all}
#tests-panel .tp-item .meta{font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:4px;text-transform:uppercase}
#tests-panel .tp-item .reason{font-size:10px;color:#ffb547;margin-top:4px;font-style:italic}
#tests-panel .tp-item .row{display:flex;gap:6px;margin-top:6px;align-items:center}
#tests-panel .tp-item .op{display:inline-block;font-size:8px;letter-spacing:.15em;padding:1px 5px;border-radius:2px;background:rgba(255,181,71,.15);color:#ffb547;text-transform:uppercase;font-weight:bold}
#tests-panel .tp-item .age{font-size:9px;color:var(--mute);margin-left:auto}
#tests-panel .tp-item button.act{padding:4px 9px;border:1px solid rgba(0,255,156,.4);background:rgba(0,255,156,.04);color:#00ff9c;font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px}
#tests-panel .tp-item button.act:hover{background:rgba(0,255,156,.14)}
#tests-panel .tp-empty{padding:20px 14px;text-align:center;color:var(--mute);font-size:11px;font-style:italic}
#tests-panel .tp-toolbar{display:flex;gap:6px;padding:8px 14px;border-bottom:1px solid rgba(255,181,71,.08);font-size:10px}
#tests-panel .tp-toolbar button{padding:4px 8px;border:1px solid rgba(255,181,71,.3);background:rgba(255,181,71,.04);color:#ffb547;font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px}
#tests-panel .tp-toolbar button:hover{background:rgba(255,181,71,.12)}
#tests-panel .tp-summary{font-size:10px;color:var(--mute);letter-spacing:.05em;margin-left:auto;align-self:center}
#sessions-panel{position:fixed;top:60px;right:24px;width:440px;z-index:11;border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#sessions-panel.show{display:block}
#sessions-panel .sp-head{padding:10px 14px;border-bottom:1px solid rgba(var(--c-rgb),.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#sessions-panel .sp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:10px}
#sessions-panel .sp-head .close:hover{color:var(--err);border-color:var(--err)}
#sessions-panel .sp-toolbar{display:flex;gap:6px;padding:8px 14px;border-bottom:1px solid rgba(var(--c-rgb),.08);font-size:10px}
#sessions-panel .sp-toolbar button{padding:4px 8px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.04);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px}
#sessions-panel .sp-toolbar button:hover{background:rgba(var(--c-rgb),.12)}
#sessions-panel .sp-toolbar .sp-count{margin-left:auto;align-self:center;font-size:10px;color:var(--mute);letter-spacing:.05em}
#sessions-panel .sp-list{padding:8px 12px}
#sessions-panel .sp-item{display:flex;flex-direction:column;gap:4px;padding:9px 11px;border-left:2px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);margin-bottom:6px;border-radius:0 3px 3px 0;cursor:pointer;transition:all .15s;position:relative}
#sessions-panel .sp-item:hover{background:rgba(var(--c-rgb),.1);border-left-color:var(--cyan)}
#sessions-panel .sp-item.current{border-left-color:#00ff9c;background:rgba(0,255,156,.05)}
#sessions-panel .sp-item.current::after{content:'CURRENT';position:absolute;top:8px;right:32px;font-size:8px;letter-spacing:.18em;color:#00ff9c;font-family:JetBrains Mono,monospace;font-weight:bold}
#sessions-panel .sp-item .sp-row1{display:flex;gap:8px;align-items:baseline;font-size:11px}
#sessions-panel .sp-item .sp-sid{font-family:JetBrains Mono,monospace;color:var(--cyan);font-size:10px;font-weight:bold;letter-spacing:.08em}
#sessions-panel .sp-item .sp-turns{font-size:9px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase}
#sessions-panel .sp-item .sp-age{margin-left:auto;font-size:9px;color:var(--mute)}
#sessions-panel .sp-item .sp-first{font-size:11px;color:var(--fg);line-height:1.35;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
#sessions-panel .sp-item .sp-del{position:absolute;top:8px;right:10px;width:18px;height:18px;border-radius:50%;background:rgba(255,91,91,.1);border:1px solid rgba(255,91,91,.2);color:#ff7b7b;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1;padding:0;opacity:0;transition:opacity .15s}
#sessions-panel .sp-item:hover .sp-del{opacity:1}
#sessions-panel .sp-item .sp-del:hover{background:rgba(255,91,91,.3);color:#fff}
#sessions-panel .sp-empty{padding:24px;text-align:center;color:var(--mute);font-size:10px;font-style:italic}
.sh-led{display:inline-block;width:6px;height:6px;border-radius:50%;background:#4a5568;box-shadow:0 0 4px #4a5568;margin-right:6px;vertical-align:middle;transition:background .25s}
.sh-led.clean{background:#00e5ff;box-shadow:0 0 6px #00e5ff}
.sh-led.dirty{background:#ff7b7b;box-shadow:0 0 8px #ff7b7b, 0 0 14px rgba(255,123,123,.4);animation:shPulse 1.4s ease-in-out infinite}
.sh-led.error{background:#4a5568}
@keyframes shPulse{0%,100%{opacity:1}50%{opacity:.55}}
#shell-panel{position:fixed;top:60px;right:24px;width:480px;z-index:11;border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#shell-panel.show{display:block}
#shell-panel .sh-head{padding:10px 14px;border-bottom:1px solid rgba(var(--c-rgb),.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#shell-panel .sh-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:10px}
#shell-panel .sh-head .close:hover{color:var(--err);border-color:var(--err)}
#shell-panel .sh-toolbar{display:flex;gap:6px;padding:8px 14px;border-bottom:1px solid rgba(var(--c-rgb),.08);font-size:10px}
#shell-panel .sh-toolbar button{padding:4px 8px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.04);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px}
#shell-panel .sh-toolbar button:hover{background:rgba(var(--c-rgb),.12)}
#shell-panel .sh-toolbar button.on{background:rgba(255,123,123,.12);border-color:#ff7b7b;color:#ff7b7b}
#shell-panel .sh-summary{font-size:10px;color:var(--mute);letter-spacing:.05em;margin-left:auto;align-self:center}
#shell-panel .sh-section{padding:8px 14px}
#shell-panel .sh-item{padding:8px 10px;border-left:2px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);margin-bottom:6px;font-family:JetBrains Mono,monospace}
#shell-panel .sh-item.fail{border-left-color:#ff7b7b;background:rgba(255,123,123,.05)}
#shell-panel .sh-item .cmd{font-size:11px;color:var(--fg);word-break:break-all;line-height:1.4}
#shell-panel .sh-item .meta{display:flex;gap:8px;font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:4px;text-transform:uppercase;flex-wrap:wrap}
#shell-panel .sh-item .kind{display:inline-block;padding:1px 5px;border-radius:2px;background:rgba(var(--c-rgb),.12);color:var(--cyan);font-size:8px;font-weight:bold}
#shell-panel .sh-item.fail .kind{background:rgba(255,123,123,.15);color:#ff7b7b}
#shell-panel .sh-item .rc{padding:1px 5px;border-radius:2px;font-size:8px;font-weight:bold}
#shell-panel .sh-item .rc.ok{background:rgba(0,255,156,.12);color:#00ff9c}
#shell-panel .sh-item .rc.bad{background:rgba(255,91,91,.12);color:#ff7b7b}
#shell-panel .sh-item .age{margin-left:auto;color:var(--mute);font-size:9px}
#shell-panel .sh-item .toggle{cursor:pointer;color:var(--cyan);font-size:9px;letter-spacing:.15em;margin-top:6px;display:inline-block;text-transform:uppercase}
#shell-panel .sh-item pre{margin:6px 0 0 0;font-size:9.5px;background:rgba(0,0,0,.4);padding:6px 8px;border-radius:2px;max-height:160px;overflow:auto;color:var(--fg);line-height:1.35;display:none;white-space:pre-wrap;word-break:break-all}
#shell-panel .sh-item pre.show{display:block}
#shell-panel .sh-empty{padding:24px;text-align:center;color:var(--mute);font-size:10px;font-style:italic}
#chat-search{position:fixed;top:60px;left:50%;transform:translateX(-50%) translateY(-20px);width:min(560px,90vw);z-index:13;background:rgba(var(--panel-rgb,8,14,28),.97);border:1px solid rgba(var(--c-rgb),.5);border-radius:4px;padding:10px 14px;box-shadow:0 0 28px rgba(var(--c-rgb),.28);backdrop-filter:blur(8px);display:none;opacity:0;transition:opacity .2s, transform .2s}
#chat-search.show{display:block;opacity:1;transform:translateX(-50%) translateY(0)}
#chat-search .cs-row{display:flex;gap:8px;align-items:center}
#chat-search input{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(var(--c-rgb),.25);color:var(--fg);padding:7px 11px;border-radius:3px;font-family:inherit;font-size:13px;letter-spacing:.02em}
#chat-search input:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 8px rgba(var(--c-rgb),.4)}
#chat-search .cs-count{font-size:10px;color:var(--cyan);letter-spacing:.15em;min-width:60px;text-align:right}
#chat-search .cs-btn{padding:4px 8px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.06);color:var(--cyan);font-family:inherit;font-size:11px;cursor:pointer;border-radius:3px}
#chat-search .cs-btn:hover{background:rgba(var(--c-rgb),.16)}
#chat-search .cs-help{font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace}
.msg.cs-hidden{display:none}
mark.cs-hit{background:rgba(var(--g-rgb),.32);color:var(--fg);padding:0 2px;border-radius:2px;box-shadow:0 0 6px rgba(var(--g-rgb),.4)}
mark.cs-hit.current{background:rgba(0,255,156,.4);box-shadow:0 0 8px rgba(0,255,156,.6);color:#fff}
.restore-banner{display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border:1px dashed rgba(var(--c-rgb),.22);background:rgba(var(--c-rgb),.03);border-radius:99px;font-size:8.5px;color:var(--mute);letter-spacing:.16em;text-transform:uppercase;margin:4px 0 10px;font-family:JetBrains Mono,monospace;max-width:fit-content;opacity:1;transition:opacity .6s ease-out}
.restore-banner.rb-fade{opacity:.35}
.restore-banner.rb-fade:hover{opacity:1}
.restore-banner .rb-close{margin-left:6px;cursor:pointer;color:var(--cyan);padding:1px 5px;border:1px solid rgba(var(--c-rgb),.3);border-radius:2px;font-size:8px}
.restore-banner .rb-close:hover{background:rgba(var(--c-rgb),.12)}
.msg.restored{opacity:.78}
.msg.restored .bubble{border-left:2px solid rgba(255,255,255,.12)}
.msg.bot{position:relative}
.msg-retry{position:absolute;top:8px;right:8px;background:rgba(255,77,200,.06);border:1px solid rgba(255,77,200,.18);color:var(--magenta);font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;padding:3px 8px;border-radius:3px;cursor:pointer;opacity:0;transition:opacity .15s,background .15s,box-shadow .15s}
.msg.bot:hover .msg-retry{opacity:.6}
.msg-retry:hover{opacity:1!important;background:rgba(255,77,200,.18);box-shadow:0 0 8px rgba(255,77,200,.25)}
.msg-star{position:absolute;top:8px;right:72px;background:rgba(var(--g-rgb),.06);border:1px solid rgba(var(--g-rgb),.22);color:#ffe066;font-family:inherit;font-size:13px;line-height:1;padding:2px 7px;border-radius:3px;cursor:pointer;opacity:0;transition:opacity .15s,background .15s,box-shadow .15s,color .15s}
.msg.bot:hover .msg-star{opacity:.6}
.msg-star:hover{opacity:1!important;background:rgba(var(--g-rgb),.18);box-shadow:0 0 8px rgba(var(--g-rgb),.3)}
.msg-star.starred{opacity:1;color:#ffd770;background:rgba(255,215,112,.18);border-color:#ffd770;text-shadow:0 0 6px rgba(255,215,112,.7)}
.msg.restored .bubble::before{content:'';display:none}
.tok-meter{font-size:9px;letter-spacing:.18em;color:var(--mute);font-family:JetBrains Mono,monospace;margin-top:4px;padding:2px 7px;border:1px solid rgba(var(--c-rgb),.18);background:rgba(var(--c-rgb),.04);border-radius:2px;display:inline-block;text-transform:uppercase;transition:opacity .8s, color .25s}
.tok-meter.done{color:var(--cyan);border-color:rgba(var(--c-rgb),.35);background:rgba(var(--c-rgb),.08);text-shadow:0 0 3px rgba(var(--c-rgb),.4)}
.tok-meter.fade{opacity:0}
.briefing{font-family:JetBrains Mono,monospace;font-size:11px;border:1px solid rgba(var(--c-rgb),.25);background:rgba(var(--c-rgb),.04);border-radius:4px;padding:10px 12px;box-shadow:0 0 14px rgba(var(--c-rgb),.12)}
.briefing .b-head{font-size:11px;letter-spacing:.3em;color:var(--cyan);text-shadow:0 0 4px var(--cyan);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(var(--c-rgb),.18);display:flex;align-items:center;gap:8px}
.briefing .b-icon{font-size:14px;color:var(--cyan)}
.briefing .b-section{padding:5px 0;font-size:11px;color:var(--fg)}
.briefing .b-section+.b-section{border-top:1px dashed rgba(var(--c-rgb),.08)}
.briefing .b-lbl{display:inline-block;width:80px;font-size:9px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase}
.briefing .b-tags{margin-top:4px;display:flex;flex-wrap:wrap;gap:4px}
.briefing .b-tag{display:inline-block;padding:1px 6px;font-size:9px;background:rgba(var(--c-rgb),.08);color:var(--cyan);border:1px solid rgba(var(--c-rgb),.18);border-radius:2px;letter-spacing:.05em;text-transform:capitalize}
.briefing .b-mute{color:var(--mute);font-style:italic;font-size:10px}
.bubble .katex{color:var(--cyan);text-shadow:0 0 4px rgba(var(--c-rgb),.3)}
.bubble .katex-display{margin:8px 0;padding:6px 12px;background:rgba(var(--c-rgb),.04);border-left:2px solid rgba(var(--c-rgb),.3);border-radius:0 3px 3px 0;overflow-x:auto}
.math-pending{font-family:JetBrains Mono,monospace;color:var(--mute);font-style:italic}
.math-pending.math-display{display:block;padding:6px 10px;margin:6px 0;background:rgba(var(--c-rgb),.03);border-left:2px solid rgba(var(--c-rgb),.2)}
.reasoning{margin:0 0 6px 0;border-left:2px solid rgba(94,122,153,.4);background:rgba(94,122,153,.06);border-radius:0 4px 4px 0;overflow:hidden}
.reasoning-head{display:flex;align-items:center;gap:6px;padding:4px 10px;color:var(--mute);cursor:pointer;user-select:none;letter-spacing:.08em;text-transform:uppercase;font-size:9px}
.reasoning-toggle{margin-left:auto;transition:transform .2s}
.reasoning.open .reasoning-toggle{transform:rotate(180deg)}
.reasoning-dot{width:6px;height:6px;border-radius:50%;background:var(--mute);animation:rpulse 1.2s ease-in-out infinite}
.reasoning.done .reasoning-dot{animation:none;background:var(--cyan2)}
@keyframes rpulse{0%,100%{opacity:.3}50%{opacity:1}}
.reasoning-body{display:none;padding:0 10px 8px 10px;max-height:200px;overflow-y:auto;white-space:pre-wrap;color:var(--mute);font-family:Consolas,monospace;font-size:11px;line-height:1.45}
.reasoning.open .reasoning-body{display:block}
.pp-themes{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.th-sw{display:flex;align-items:center;gap:5px;padding:7px 9px;border:1px solid;border-radius:5px;cursor:pointer;transition:transform .12s,box-shadow .12s;overflow:hidden}
.th-sw:hover{transform:translateY(-1px)}
.th-sw.active{box-shadow:0 0 0 1px rgba(var(--c-rgb),.9),0 0 10px rgba(var(--c-rgb),.4)}
.th-dot{width:8px;height:8px;border-radius:50%;flex:none}
.th-name{font-size:10px;letter-spacing:.04em;margin-left:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
body.theme-min #scanline{display:none}
body.theme-min #netcanvas{opacity:.22}
body.theme-min #nebula{opacity:.4}
body.theme-min #frame,body.theme-min #corner-tr,body.theme-min #corner-bl{display:none}
body.theme-min #adam-core{opacity:.6}
#cmd-menu{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;background:rgba(2,5,12,.6);backdrop-filter:blur(3px)}
#cmd-menu.show{display:flex}
.cm-box{width:min(520px,92vw);max-height:72vh;display:flex;flex-direction:column;background:var(--glass);border:1px solid rgba(var(--c-rgb),.35);border-radius:8px;box-shadow:0 0 40px rgba(var(--c-rgb),.18);overflow:hidden;backdrop-filter:blur(10px)}
.cm-head{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;border-bottom:1px solid rgba(var(--c-rgb),.18);font-size:11px;letter-spacing:.12em;color:var(--cyan)}
.cm-close{cursor:pointer;color:var(--mute);font-size:10px;letter-spacing:.1em}
.cm-close:hover{color:var(--cyan)}
.cm-list{overflow-y:auto;padding:6px}
.cm-group{font-size:8.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--mute);padding:10px 10px 4px}
.cm-row{display:flex;align-items:baseline;gap:10px;padding:8px 10px;border-radius:5px;cursor:pointer;transition:background .12s}
.cm-row:hover{background:rgba(var(--c-rgb),.1)}
.cm-cmd{color:var(--fg);font-size:12px;min-width:130px;font-weight:500}
.cm-hint{color:var(--mute);font-size:10px}
.qc-menu{border-color:rgba(var(--c-rgb),.5)!important}
.math-fail{color:var(--err);font-family:JetBrains Mono,monospace;text-decoration:underline dotted}
.mermaid-pending{margin:10px 0;padding:10px 14px;background:rgba(var(--c-rgb),.04);border:1px solid rgba(var(--c-rgb),.2);border-radius:4px;font-family:JetBrains Mono,monospace;font-size:10px;color:var(--mute);white-space:pre-wrap;overflow:auto;max-width:100%}
.mermaid-pending.mermaid-rendered{padding:14px;background:rgba(var(--c-rgb),.05);border-color:var(--cyan);box-shadow:0 0 14px rgba(var(--c-rgb),.18)}
.mermaid-pending.mermaid-rendered svg{max-width:100%;height:auto;display:block;margin:0 auto}
.mermaid-pending.mermaid-ok::before{content:'';display:none}
.mermaid-fail{color:var(--err);font-family:JetBrains Mono,monospace;font-size:10px;padding:4px 0;border-bottom:1px dotted rgba(255,85,119,.4);margin-bottom:6px}
#kbd-overlay{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:min(500px,92vw);z-index:17;background:rgba(var(--panel-rgb,8,14,28),.97);border:1px solid var(--cyan);border-radius:6px;padding:18px 20px;box-shadow:0 0 36px rgba(var(--c-rgb),.4);display:none;font-family:inherit;max-height:88vh;overflow-y:auto}
#kbd-overlay.show{display:block}
#kbd-overlay .kbd-head{display:flex;justify-content:space-between;align-items:center;font-size:11px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan);padding-bottom:10px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:12px}
#kbd-overlay .kbd-close{cursor:pointer;color:var(--mute);padding:1px 8px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:10px;letter-spacing:.2em}
#kbd-overlay .kbd-close:hover{color:var(--err);border-color:var(--err)}
#kbd-overlay .kbd-grid{display:grid;grid-template-columns:1fr;gap:4px}
#kbd-overlay .kbd-row{display:flex;align-items:center;gap:12px;padding:6px 8px;border-radius:3px;font-size:11px}
#kbd-overlay .kbd-row:hover{background:rgba(var(--c-rgb),.04)}
#kbd-overlay .kbd-row span{color:var(--fg);flex:1}
#kbd-overlay kbd{display:inline-block;padding:2px 8px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.06);color:var(--cyan);border-radius:3px;font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:.1em;min-width:90px;text-align:center}
#kbd-overlay .kbd-foot{padding-top:10px;margin-top:12px;border-top:1px solid rgba(var(--c-rgb),.12);font-size:9px;color:var(--mute);letter-spacing:.05em;text-align:center}
#kbd-overlay .kbd-foot kbd{min-width:auto;padding:1px 5px;font-size:9px}
#gesture-tour{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:min(440px,86vw);z-index:16;background:rgba(var(--panel-rgb,8,14,28),.97);border:1px solid var(--cyan);border-radius:6px;padding:16px;box-shadow:0 0 36px rgba(var(--c-rgb),.32);display:none;font-family:inherit;max-height:78vh;overflow-y:auto}
#gesture-tour.show{display:block}
#gesture-tour h3{font-size:12px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan);margin:0 0 14px;text-align:center}
#gesture-tour .tour-intro{font-size:11px;color:var(--mute);text-align:center;margin-bottom:18px;line-height:1.5}
#gesture-tour .gesture-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:18px}
#gesture-tour .g-card{padding:10px 12px;border:1px solid rgba(var(--c-rgb),.22);background:rgba(var(--c-rgb),.04);border-radius:4px;display:flex;flex-direction:column;gap:4px;transition:all .15s}
#gesture-tour .g-card:hover{border-color:var(--cyan);background:rgba(var(--c-rgb),.1);box-shadow:0 0 10px rgba(var(--c-rgb),.18)}
#gesture-tour .g-card .g-emoji{font-size:28px;line-height:1;text-align:center;text-shadow:0 0 6px var(--cyan);filter:drop-shadow(0 0 4px rgba(var(--c-rgb),.6))}
#gesture-tour .g-card .g-name{font-size:10px;letter-spacing:.18em;color:var(--cyan);text-transform:uppercase;text-align:center;font-family:JetBrains Mono,monospace}
#gesture-tour .g-card .g-action{font-size:10px;color:var(--fg);text-align:center;line-height:1.4}
#gesture-tour .tour-actions{display:flex;gap:8px;margin-top:6px;justify-content:space-between;align-items:center}
#gesture-tour button.gt-act{padding:8px 14px;border:1px solid rgba(var(--c-rgb),.4);background:rgba(var(--c-rgb),.06);color:var(--cyan);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#gesture-tour button.gt-act:hover{background:rgba(var(--c-rgb),.18)}
#gesture-tour button.gt-act.primary{border-color:var(--cyan);background:rgba(var(--c-rgb),.15)}
#gesture-tour .tour-hint{font-size:10px;color:var(--mute);letter-spacing:.05em;flex:1;text-align:center;font-style:italic}
#adam-core{position:fixed;top:18px;right:24px;width:60px;height:60px;z-index:8;cursor:pointer;opacity:.85;transition:opacity .25s, transform .25s}
#adam-core:hover{opacity:1;transform:scale(1.08)}
#adam-core.hidden{display:none}
#adam-core.collapsed{transform:scale(0.5);opacity:.55}
#toast-stack{position:fixed;bottom:140px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:14;max-width:340px;pointer-events:none}
.toast{pointer-events:auto;padding:10px 12px;border:1px solid rgba(var(--c-rgb),.35);background:rgba(var(--panel-rgb,8,14,28),.94);border-left:3px solid var(--cyan);border-radius:3px;font-size:11px;color:var(--fg);box-shadow:0 0 14px rgba(var(--c-rgb),.18);backdrop-filter:blur(6px);transform:translateX(60px);opacity:0;transition:transform .25s ease-out, opacity .25s ease-out;cursor:pointer;font-family:inherit}
.toast.show{transform:translateX(0);opacity:1}
.toast.dismiss{transform:translateX(60px);opacity:0;transition:transform .35s ease-in, opacity .35s ease-in}
.toast.info{border-left-color:var(--cyan)}
.toast.warn{border-left-color:#ffb547;border-color:rgba(255,181,71,.35);box-shadow:0 0 12px rgba(255,181,71,.18)}
.toast.error{border-left-color:#ff5b5b;border-color:rgba(255,91,91,.35);box-shadow:0 0 12px rgba(255,91,91,.18)}
.toast.success{border-left-color:#00ff9c;border-color:rgba(0,255,156,.35);box-shadow:0 0 12px rgba(0,255,156,.18)}
.toast .t-head{display:flex;align-items:center;gap:6px;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:var(--mute)}
.toast .t-src{color:var(--cyan)}
.toast.warn .t-src{color:#ffb547}
.toast.error .t-src{color:#ff5b5b}
.toast.success .t-src{color:#00ff9c}
.toast .t-title{font-size:12px;color:var(--fg);font-weight:600;margin-top:3px;line-height:1.35}
.toast .t-body{font-size:10px;color:var(--mute);margin-top:3px;line-height:1.4;white-space:pre-wrap;word-break:break-word;max-height:80px;overflow:hidden}
.toast .t-age{margin-left:auto;font-size:9px;color:var(--mute)}
.toast .t-close{font-size:9px;color:var(--mute);cursor:pointer;padding:0 4px}
.toast .t-close:hover{color:var(--err)}
#coach-toggle.on{color:var(--magenta);border-color:var(--magenta);background:rgba(255,77,200,.08);box-shadow:0 0 10px rgba(255,77,200,.3)}
#coach-panel{position:fixed;top:60px;right:24px;width:440px;z-index:11;border:1px solid rgba(255,77,200,.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(255,77,200,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#coach-panel.show{display:block}
#coach-panel .cp-head{padding:10px 14px;border-bottom:1px solid rgba(255,77,200,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--magenta);text-shadow:0 0 4px var(--magenta);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#coach-panel .cp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(255,77,200,.22);border-radius:3px;font-size:10px}
#coach-panel .cp-head .close:hover{color:var(--err);border-color:var(--err)}
#coach-panel .cp-streak-badge{margin-left:auto;padding:2px 8px;font-size:10px;font-weight:bold;letter-spacing:.05em;border-radius:99px;background:rgba(255,77,200,.08);border:1px solid rgba(255,77,200,.25);color:var(--magenta);text-shadow:none;font-family:JetBrains Mono,monospace;cursor:help;display:none}
#coach-panel .cp-streak-badge.active{display:inline-block}
#coach-panel .cp-streak-badge.fire{background:rgba(255,140,60,.15);border-color:rgba(255,140,60,.45);color:#ffa460;text-shadow:0 0 6px rgba(255,140,60,.6);animation:streakPulse 2.4s ease-in-out infinite}
#coach-panel .cp-streak-badge.elite{background:rgba(0,255,156,.15);border-color:rgba(0,255,156,.5);color:#00ff9c;text-shadow:0 0 6px rgba(0,255,156,.6);animation:streakPulse 1.8s ease-in-out infinite}
@keyframes streakPulse{0%,100%{opacity:1}50%{opacity:.7}}
#coach-panel .cp-section{padding:12px 14px;border-bottom:1px solid rgba(255,77,200,.08)}
#coach-panel .cp-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
#coach-panel .cp-topic-row{display:flex;gap:6px;align-items:center}
#coach-panel input[type=text],#coach-panel textarea{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(255,77,200,.2);color:var(--fg);padding:6px 9px;border-radius:3px;font-family:inherit;font-size:12px;resize:vertical}
#coach-panel input[type=text]:focus,#coach-panel textarea:focus{outline:none;border-color:var(--magenta);box-shadow:0 0 6px rgba(255,77,200,.3)}
#coach-panel select{background:rgba(0,0,0,.4);border:1px solid rgba(255,77,200,.2);color:var(--fg);padding:5px 8px;border-radius:3px;font-family:inherit;font-size:11px}
#coach-panel button.cp-act{padding:6px 12px;border:1px solid rgba(255,77,200,.4);background:rgba(255,77,200,.06);color:var(--magenta);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px;transition:all .15s}
#coach-panel button.cp-act:hover{background:rgba(255,77,200,.16);border-color:var(--magenta)}
#coach-panel button.cp-act.on{color:var(--gold);border-color:var(--gold);background:rgba(255,215,112,.1);box-shadow:0 0 8px rgba(255,215,112,.25)}
#coach-panel .cp-topic-card{display:flex;align-items:center;gap:8px;padding:6px 10px;border:1px solid rgba(255,77,200,.15);border-radius:3px;background:rgba(255,77,200,.03);margin-bottom:4px;cursor:pointer;transition:all .15s}
#coach-panel .cp-topic-card:hover{border-color:var(--magenta);background:rgba(255,77,200,.1)}
#coach-panel .cp-topic-card .name{flex:1;font-size:11px;color:var(--fg);text-transform:capitalize}
#coach-panel .cp-topic-card .n{font-size:9px;color:var(--mute);letter-spacing:.1em}
#coach-panel .cp-topic-card .pct{font-size:11px;font-weight:bold;min-width:38px;text-align:right;font-family:JetBrains Mono,monospace}
#coach-panel .cp-topic-card.lvl-master .pct{color:#00ff9c;text-shadow:0 0 4px #00ff9c}
#coach-panel .cp-topic-card.lvl-good .pct{color:#ffe066;text-shadow:0 0 3px #ffe066}
#coach-panel .cp-topic-card.lvl-fair .pct{color:#ffb547}
#coach-panel .cp-topic-card.lvl-novice .pct{color:#ff7b7b}
#coach-panel .cp-topic-card .mini-bar{flex:1;max-width:80px;height:4px;background:rgba(255,77,200,.08);border-radius:2px;overflow:hidden;border:1px solid rgba(255,77,200,.12)}
#coach-panel .cp-topic-card .mini-bar-fill{height:100%;background:linear-gradient(90deg,#ff4dc8,#ffe066);transition:width .3s}
#coach-panel .cp-topic-card.lvl-master .mini-bar-fill{background:linear-gradient(90deg,#00ff9c,#ffe066)}
#coach-panel .cp-review-card{padding:6px 10px;border-left:2px solid var(--magenta);background:rgba(255,77,200,.05);margin-bottom:4px;border-radius:0 3px 3px 0;cursor:pointer;font-size:11px;transition:all .15s}
#coach-panel .cp-review-card:hover{background:rgba(255,77,200,.14);box-shadow:0 0 8px rgba(255,77,200,.18)}
#coach-panel .cp-review-card .rv-topic{font-size:9px;letter-spacing:.18em;color:var(--magenta);text-transform:uppercase;font-family:JetBrains Mono,monospace}
#coach-panel .cp-review-card .rv-q{color:var(--fg);margin-top:3px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.3}
#coach-panel .cp-review-card .rv-meta{display:flex;gap:8px;font-size:9px;color:var(--mute);margin-top:4px;letter-spacing:.05em;text-transform:uppercase}
#coach-panel .cp-review-card .rv-overdue{margin-left:auto;color:#ff7b7b}
#coach-panel .cp-review-card.urgent{border-left-color:#ff5b5b;background:rgba(255,91,91,.06)}
#coach-panel .cp-review-card.urgent .rv-overdue{color:#ff5b5b;font-weight:bold}
#coach-panel .cp-topic-card .tc-export{margin-left:6px;width:20px;height:20px;border-radius:3px;background:rgba(0,255,156,.06);border:1px solid rgba(0,255,156,.18);color:#00ff9c;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1;padding:0;opacity:0;transition:opacity .15s}
#coach-panel .cp-topic-card:hover .tc-export{opacity:1}
#coach-panel .cp-topic-card .tc-export:hover{background:rgba(0,255,156,.18);color:#fff}
#coach-panel button.cp-act:disabled{opacity:.4;cursor:not-allowed}
#coach-panel button.cp-act.danger{border-color:rgba(255,91,91,.4);color:#ff7b7b}
#coach-panel button.cp-act.danger:hover{background:rgba(255,91,91,.12);border-color:#ff5b5b}
#coach-panel .cp-q{font-size:14px;color:var(--fg);line-height:1.5;padding:14px;background:rgba(255,77,200,.05);border-left:3px solid var(--magenta);border-radius:0 4px 4px 0;text-shadow:0 0 2px rgba(255,255,255,.05);min-height:60px}
#coach-panel .cp-q.empty{color:var(--mute);font-style:italic;border-left-color:rgba(255,77,200,.15)}
#coach-panel .cp-meta{display:flex;gap:14px;font-size:9px;letter-spacing:.15em;color:var(--mute);margin-top:8px;text-transform:uppercase;flex-wrap:wrap}
#coach-panel .cp-meta .v{color:var(--magenta);font-size:11px;margin-left:4px}
#coach-panel .cp-mastery-bar{height:6px;background:rgba(255,77,200,.08);border-radius:3px;overflow:hidden;margin-top:8px;border:1px solid rgba(255,77,200,.15)}
#coach-panel .cp-mastery-fill{height:100%;background:linear-gradient(90deg,#ff4dc8,#ffe066);box-shadow:0 0 6px var(--magenta);transition:width .4s ease-out}
#coach-panel .cp-grade{padding:10px 12px;border-radius:3px;font-size:11px;margin-top:8px;line-height:1.5}
#coach-panel .cp-grade.good{background:rgba(0,255,156,.06);border:1px solid rgba(0,255,156,.3);color:#a0ffd0}
#coach-panel .cp-grade.ok{background:rgba(var(--g-rgb),.06);border:1px solid rgba(var(--g-rgb),.3);color:#ffe6a0}
#coach-panel .cp-grade.bad{background:rgba(255,91,91,.06);border:1px solid rgba(255,91,91,.3);color:#ffb0b0}
#coach-panel .cp-grade .score{font-size:20px;font-weight:bold;display:block;margin-bottom:4px}
#coach-panel .cp-grade.good .score{color:#00ff9c}
#coach-panel .cp-grade.ok .score{color:#ffe066}
#coach-panel .cp-grade.bad .score{color:#ff7b7b}
#coach-panel .cp-grade ul{margin:6px 0 0 14px;padding:0;font-size:10px;color:var(--mute)}
#coach-panel .cp-grade li{margin:2px 0}
#coach-panel .cp-streak{display:inline-block;padding:2px 6px;border-radius:2px;font-size:9px;letter-spacing:.15em;font-weight:bold;text-transform:uppercase}
#coach-panel .cp-streak.correct{background:rgba(0,255,156,.12);color:#00ff9c;border:1px solid rgba(0,255,156,.25)}
#coach-panel .cp-streak.wrong{background:rgba(255,91,91,.12);color:#ff7b7b;border:1px solid rgba(255,91,91,.25)}
#coach-panel .cp-btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
#coach-panel .cp-hint{font-size:11px;color:#ffe066;background:rgba(var(--g-rgb),.05);border-left:2px solid #ffe066;padding:6px 10px;border-radius:0 3px 3px 0;margin-top:8px;font-style:italic}
@keyframes ldPulse{0%,100%{opacity:1}50%{opacity:.45}}
#learn-panel{position:fixed;top:60px;right:24px;width:340px;z-index:11;border:1px solid rgba(0,255,156,.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.96);box-shadow:0 0 28px rgba(0,255,156,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#learn-panel.show{display:block}
#learn-panel .lp-head{padding:10px 14px;border-bottom:1px solid rgba(0,255,156,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:#00ff9c;text-shadow:0 0 4px #00ff9c;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.98)}
#learn-panel .lp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(0,255,156,.22);border-radius:3px;font-size:10px}
#learn-panel .lp-head .close:hover{color:var(--err);border-color:var(--err)}
#learn-panel .lp-section{padding:12px 14px;border-bottom:1px solid rgba(0,255,156,.08)}
#learn-panel .lp-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
#learn-panel .lp-now{padding:10px 12px;border:1px solid rgba(0,255,156,.3);border-radius:3px;background:rgba(0,255,156,.05);font-size:11px}
#learn-panel .lp-now .topic{color:#00ff9c;text-shadow:0 0 4px #00ff9c;font-weight:bold;font-size:12px;word-break:break-word}
#learn-panel .lp-now .phase{color:var(--mute);font-size:9px;letter-spacing:.2em;margin-top:4px;text-transform:uppercase}
#learn-panel .lp-now.idle{border-color:rgba(74,85,104,.4);background:rgba(74,85,104,.05)}
#learn-panel .lp-now.idle .topic{color:var(--mute);text-shadow:none;font-weight:normal;font-style:italic}
#learn-panel .lp-stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:10px}
#learn-panel .lp-stat{padding:6px 8px;border:1px solid rgba(0,255,156,.12);border-radius:3px;background:rgba(0,255,156,.02)}
#learn-panel .lp-stat .v{color:#00ff9c;font-size:13px;font-weight:bold;display:block}
#learn-panel .lp-stat .k{color:var(--mute);font-size:8px;letter-spacing:.18em;text-transform:uppercase;display:block;margin-top:2px}
#learn-panel .lp-recent{font-size:10px;color:var(--fg);padding:5px 8px;border-left:2px solid rgba(0,255,156,.3);margin-bottom:4px;background:rgba(0,255,156,.02)}
#learn-panel .lp-recent .t{color:#00ff9c}
#learn-panel .lp-recent .meta{color:var(--mute);font-size:8px;letter-spacing:.1em;margin-top:2px}
#learn-panel .lp-btn-row{display:flex;gap:6px;margin-top:8px}
#learn-panel button.lp-act{flex:1;padding:6px 10px;border:1px solid rgba(0,255,156,.4);background:rgba(0,255,156,.04);color:#00ff9c;font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#learn-panel button.lp-act:hover{background:rgba(0,255,156,.14)}
#learn-panel button.lp-act:disabled{opacity:.4;cursor:not-allowed}
#learn-panel .lp-queue-row{display:flex;gap:6px;margin-top:6px}
#learn-panel .lp-queue-row input{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(0,255,156,.2);color:var(--fg);padding:5px 8px;border-radius:3px;font-family:inherit;font-size:11px}
#learn-panel .lp-queue-row input:focus{outline:none;border-color:#00ff9c}
#task-tray{position:fixed;left:50%;bottom:88px;transform:translateX(-50%) translateY(140%);width:min(560px,92vw);z-index:8;border:1px solid rgba(var(--c-rgb),.3);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.95);box-shadow:0 0 24px rgba(var(--c-rgb),.18);transition:transform .25s ease-out;backdrop-filter:blur(6px);max-height:30vh;overflow-y:auto}
#task-tray.show{transform:translateX(-50%) translateY(0)}
#task-tray .tray-head{padding:6px 12px;font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);border-bottom:1px solid rgba(var(--c-rgb),.15);display:flex;align-items:center;gap:8px;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.95)}
#task-tray .tray-head .dot{width:5px;height:5px;border-radius:50%;background:var(--ok);box-shadow:0 0 5px var(--ok);animation:pulse 1.6s ease-in-out infinite}
#task-tray .tray-head .ct{margin-left:auto;color:var(--mute);font-size:9px}
.task-row{padding:6px 12px;border-bottom:1px solid rgba(var(--c-rgb),.06);display:grid;grid-template-columns:1fr auto;gap:8px;font-size:10px;align-items:center}
.task-row:last-child{border-bottom:none}
.task-row .label{color:var(--fg);letter-spacing:.05em;line-height:1.3}
.task-row .label .kind{color:var(--mute);font-size:8px;letter-spacing:.2em;text-transform:uppercase;margin-right:6px}
.task-row .label .msg{color:var(--mute);font-size:9px;margin-top:1px}
.task-row .pbar{height:3px;background:rgba(var(--c-rgb),.1);border-radius:2px;overflow:hidden;margin-top:4px}
.task-row .pbar .fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 4px var(--cyan);transition:width .3s}
.task-row .cancel{background:transparent;border:1px solid rgba(255,85,119,.4);color:var(--err);font-family:inherit;font-size:9px;padding:3px 8px;border-radius:2px;cursor:pointer;letter-spacing:.15em}
.task-row .cancel:hover{background:rgba(255,85,119,.1)}
.task-row.cancelling .cancel{opacity:.4;cursor:wait}
#mem-panel{position:fixed;top:60px;right:-440px;width:420px;height:calc(100vh - 140px);z-index:9;border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.92);box-shadow:0 0 28px rgba(var(--c-rgb),.2);overflow-y:auto;transition:right .25s ease-out;backdrop-filter:blur(8px)}
#mem-panel.show{right:24px}
#mem-panel::-webkit-scrollbar{width:5px}
#mem-panel::-webkit-scrollbar-thumb{background:rgba(var(--c-rgb),.3);border-radius:3px}
#mem-panel .mem-head{padding:10px 14px;border-bottom:1px solid rgba(var(--c-rgb),.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 6px var(--cyan);display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:rgba(var(--panel-rgb,8,14,28),.95);z-index:2}
#mem-panel .mem-head .close{cursor:pointer;color:var(--mute);padding:2px 8px;border:1px solid rgba(var(--c-rgb),.2);border-radius:3px;font-size:11px}
#mem-panel .mem-head .close:hover{color:var(--err);border-color:var(--err)}
#mem-panel .mem-section{padding:12px 14px;border-bottom:1px solid rgba(var(--c-rgb),.08)}
#mem-panel .mem-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);margin-bottom:8px;display:flex;align-items:center;gap:6px}
#mem-panel .mem-section h3 .count{margin-left:auto;color:var(--mute);font-size:9px;letter-spacing:.1em}
#mem-panel .mem-row{padding:6px 8px;font-size:11px;border:1px solid rgba(var(--c-rgb),.08);border-radius:3px;margin-bottom:5px;background:rgba(var(--c-rgb),.02);display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
#mem-panel .mem-row:hover{border-color:rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.05)}
#mem-panel .mem-row .body{flex:1;line-height:1.4;word-break:break-word}
#mem-panel .mem-row .lbl{font-size:8px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-bottom:3px}
#mem-panel .mem-row .confid{color:var(--magenta)}
#mem-panel .mem-row .actions{display:flex;flex-direction:column;gap:3px}
#mem-panel .mem-row button{background:transparent;border:1px solid rgba(255,85,119,.4);color:var(--err);font-family:inherit;font-size:9px;padding:2px 6px;border-radius:2px;cursor:pointer;letter-spacing:.1em}
#mem-panel .mem-row button:hover{background:rgba(255,85,119,.1)}
#mem-panel .mem-row button.cyan{border-color:rgba(var(--c-rgb),.4);color:var(--cyan)}
#mem-panel .mem-row button.cyan:hover{background:rgba(var(--c-rgb),.1)}
#mem-panel .mem-stat{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:6px}
#mem-panel .mem-stat .stat{padding:6px 8px;border:1px solid rgba(var(--c-rgb),.12);border-radius:3px;background:rgba(var(--c-rgb),.02)}
#mem-panel .mem-stat .stat .v{font-size:18px;color:var(--cyan);text-shadow:0 0 6px var(--cyan)}
#mem-panel .mem-stat .stat .lbl{font-size:8px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute)}
#mem-panel .mastery-bar{height:4px;background:rgba(var(--c-rgb),.1);border-radius:2px;overflow:hidden;margin-top:4px}
#mem-panel .mastery-bar .fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 6px var(--cyan)}
#mem-panel .empty{color:var(--mute);font-style:italic;text-align:center;padding:14px;font-size:10px}
#mem-toggle{padding:0 12px;height:46px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#mem-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.08);box-shadow:0 0 12px rgba(var(--c-rgb),.3)}
#cam-panel{position:fixed;top:60px;right:24px;width:200px;z-index:8;display:none;border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.85);box-shadow:0 0 18px rgba(var(--c-rgb),.18);overflow:hidden}
#cam-panel.show{display:block}
#cam-panel .cam-head{padding:5px 10px;background:rgba(var(--c-rgb),.08);font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;align-items:center;justify-content:space-between}
#cam-panel .cam-head .dot{width:6px;height:6px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);animation:pulse 1.6s ease-in-out infinite}
#cam-stage{position:relative;width:100%;aspect-ratio:4/3;background:#000}
#cam-video,#cam-landmarks{position:absolute;inset:0;width:100%;height:100%}
#cam-video{transform:scaleX(-1);object-fit:cover}
#cam-panel .gesture-readout{padding:6px 10px;border-top:1px solid rgba(var(--c-rgb),.18);font-size:10px;letter-spacing:.15em;color:var(--cyan);text-shadow:0 0 4px var(--cyan);text-align:center;min-height:22px}
#cam-panel.idle .gesture-readout{color:var(--mute);text-shadow:none}
#cam-panel .cam-train-btn{font-size:9px;padding:2px 6px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.04);color:var(--cyan);cursor:pointer;border-radius:2px;letter-spacing:.15em;font-family:inherit}
#cam-panel .cam-train-btn:hover{background:rgba(var(--c-rgb),.18)}
#cam-panel .cam-train-btn.recording{background:rgba(255,77,200,.18);border-color:var(--magenta);color:var(--magenta);animation:pulse 0.6s ease-in-out infinite}
#cam-panel .custom-list{border-top:1px solid rgba(var(--c-rgb),.12);padding:4px 8px;font-size:9px;max-height:80px;overflow-y:auto}
#cam-panel .custom-row{display:flex;align-items:center;gap:6px;padding:2px 0}
#cam-panel .custom-row .cg-name{color:var(--magenta);text-shadow:0 0 3px var(--magenta);flex:1;text-transform:uppercase;letter-spacing:.1em}
#cam-panel .custom-row .cg-act{color:var(--mute);font-size:8px;letter-spacing:.05em}
#cam-panel .custom-row .cg-del{color:var(--err);cursor:pointer;padding:0 4px;border-radius:2px;border:1px solid transparent}
#cam-panel .custom-row .cg-del:hover{border-color:var(--err);background:rgba(255,91,91,.1)}
#pose-panel{position:fixed;top:60px;right:24px;width:268px;z-index:9;display:none;border:1px solid rgba(var(--g-rgb),.45);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.92);box-shadow:0 0 22px rgba(var(--g-rgb),.2);overflow:hidden;resize:both;min-width:220px;min-height:280px;max-width:92vw;max-height:90vh}
#pose-panel.show{display:flex;flex-direction:column}
#pose-panel .pc-head{padding:5px 10px;background:rgba(var(--g-rgb),.09);font-size:9px;letter-spacing:.22em;text-transform:uppercase;color:#ffe066;text-shadow:0 0 4px #ffe066;display:flex;align-items:center;justify-content:space-between}
#pose-panel .pc-head .dot{width:6px;height:6px;border-radius:50%;background:var(--mute);transition:background .2s}
#pose-panel.live .pc-head .dot{background:var(--ok);box-shadow:0 0 6px var(--ok);animation:pulse 1.6s ease-in-out infinite}
#pose-panel .pc-close{cursor:pointer;color:var(--mute);padding:0 6px;border:1px solid rgba(var(--g-rgb),.25);border-radius:2px;font-size:9px}
#pose-panel .pc-close:hover{color:var(--err);border-color:var(--err)}
#pose-stage{position:relative;width:100%;flex:1;min-height:200px;background:#03060d}
#pose-video,#pose-landmarks{position:absolute;inset:0;width:100%;height:100%}
#pose-video{transform:scaleX(-1);object-fit:cover}
#pose-panel .pc-controls{padding:6px 8px;display:flex;gap:6px;align-items:center;border-top:1px solid rgba(var(--g-rgb),.16)}
#pose-panel .pc-controls select{flex:1;background:rgba(0,0,0,.5);border:1px solid rgba(var(--g-rgb),.28);color:var(--fg);font-family:inherit;font-size:10px;padding:5px 6px;border-radius:3px}
#pose-panel .pc-go{padding:5px 11px;border:1px solid rgba(var(--g-rgb),.4);background:rgba(var(--g-rgb),.08);color:#ffe066;font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px;text-transform:uppercase}
#pose-panel .pc-go:hover{background:rgba(var(--g-rgb),.2)}
#pose-panel .pc-go.live{border-color:var(--err);color:var(--err);background:rgba(255,91,91,.08)}
#pose-panel .pc-stats{display:flex;justify-content:space-around;padding:6px 8px;border-top:1px solid rgba(var(--g-rgb),.16);font-family:JetBrains Mono,monospace}
#pose-panel .pc-stat{text-align:center}
#pose-panel .pc-stat .v{font-size:18px;color:#ffe066;text-shadow:0 0 6px #ffe066;line-height:1}
#pose-panel .pc-stat .l{font-size:7.5px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-top:3px}
#pose-panel .pc-fb{padding:7px 10px;border-top:1px solid rgba(var(--g-rgb),.16);font-size:10px;line-height:1.4;color:var(--fg);min-height:20px;text-align:center}
#pose-panel .pc-fb.warn{color:#ffb347}
#pose-panel .pc-fb.good{color:var(--ok)}
#pose-panel .pc-fps{font-size:8px;color:var(--mute);letter-spacing:.1em}
#perms-panel{position:fixed;top:60px;right:24px;width:308px;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;box-shadow:0 0 24px rgba(var(--c-rgb),.18);backdrop-filter:blur(8px);padding:12px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#perms-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#perms-panel.td-hidden{display:block}
#perms-panel .pm-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:10px;font-size:9.5px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;text-shadow:0 0 4px var(--cyan)}
#perms-panel .pm-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#perms-panel .pm-close:hover{color:var(--err);border-color:var(--err)}
#perms-panel .pm-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 4px;border-bottom:1px solid rgba(var(--c-rgb),.08)}
#perms-panel .pm-row:last-of-type{border-bottom:none}
#perms-panel .pm-info{display:flex;flex-direction:column;gap:2px;flex:1;min-width:0}
#perms-panel .pm-name{font-size:11px;color:var(--fg);letter-spacing:.05em}
#perms-panel .pm-why{font-size:8.5px;color:var(--mute);letter-spacing:.02em}
#perms-panel .pm-state{font-size:7.5px;letter-spacing:.16em;text-transform:uppercase;padding:2px 7px;border-radius:10px;border:1px solid var(--mute);color:var(--mute);white-space:nowrap}
#perms-panel .pm-state.granted{color:var(--ok);border-color:var(--ok);text-shadow:0 0 4px var(--ok)}
#perms-panel .pm-state.denied{color:var(--err);border-color:var(--err)}
#perms-panel .pm-state.prompt{color:#ffe066;border-color:#ffe066}
#perms-panel .pm-grant{font-size:9px;padding:4px 10px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.05);color:var(--cyan);cursor:pointer;border-radius:3px;letter-spacing:.12em;font-family:inherit}
#perms-panel .pm-grant:hover{background:rgba(var(--c-rgb),.16)}
#perms-panel .pm-foot{font-size:8.5px;color:var(--mute);margin-top:9px;line-height:1.45;letter-spacing:.02em}
#pclog-panel{position:fixed;top:60px;right:24px;width:360px;max-height:74vh;overflow-y:auto;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--g-rgb),.35);border-radius:4px;box-shadow:0 0 24px rgba(var(--g-rgb),.16);backdrop-filter:blur(8px);padding:12px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#pclog-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#pclog-panel.td-hidden{display:block}
#pclog-panel .pl-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--g-rgb),.18);margin-bottom:8px;font-size:9.5px;letter-spacing:.3em;color:#ffe066;text-transform:uppercase;text-shadow:0 0 4px #ffe066}
#pclog-panel .pl-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--g-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#pclog-panel .pl-close:hover{color:var(--err);border-color:var(--err)}
#pclog-panel .pl-summary{font-size:9px;color:var(--mute);letter-spacing:.05em;margin-bottom:8px}
#pclog-panel .pl-row{display:flex;align-items:center;gap:8px;padding:5px 4px;border-bottom:1px solid rgba(var(--g-rgb),.07);font-size:10px}
#pclog-panel .pl-row:last-child{border-bottom:none}
#pclog-panel .pl-when{color:var(--mute);font-size:8px;font-family:JetBrains Mono,monospace;white-space:nowrap}
#pclog-panel .pl-act{color:var(--fg);font-family:JetBrains Mono,monospace;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#pclog-panel .pl-st{font-size:7px;letter-spacing:.14em;text-transform:uppercase;padding:2px 6px;border-radius:9px;border:1px solid var(--mute);color:var(--mute);white-space:nowrap}
#pclog-panel .pl-st.executed{color:var(--ok);border-color:var(--ok)}
#pclog-panel .pl-st.refused{color:var(--err);border-color:var(--err)}
#pclog-panel .pl-st.cancelled,#pclog-panel .pl-st.expired{color:#ffb347;border-color:#ffb347}
#pclog-panel .pl-st.proposed{color:var(--cyan);border-color:var(--cyan)}
#pclog-panel .pl-pending{margin:8px 0 4px;font-size:8.5px;letter-spacing:.2em;color:#ffe066;text-transform:uppercase}
#pclog-panel .pl-empty{color:var(--mute);font-size:10px;text-align:center;padding:10px}
#se-panel{position:fixed;top:60px;right:24px;width:340px;max-height:74vh;overflow-y:auto;z-index:12;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid rgba(var(--c-rgb),.4);border-radius:4px;box-shadow:0 0 24px rgba(var(--c-rgb),.2);backdrop-filter:blur(8px);padding:13px;transform:translateY(-8px);opacity:0;pointer-events:none;transition:opacity .18s ease-out,transform .22s ease-out}
#se-panel.show{transform:translateY(0);opacity:1;pointer-events:auto}
#se-panel.td-hidden{display:block}
#se-panel .se-head{display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:1px solid rgba(var(--c-rgb),.18);margin-bottom:10px;font-size:9.5px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;text-shadow:0 0 4px var(--cyan)}
#se-panel .se-close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.22);border-radius:3px;font-size:9px;letter-spacing:.18em}
#se-panel .se-close:hover{color:var(--err);border-color:var(--err)}
#se-panel .se-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:10px}
#se-panel .se-stat{background:rgba(var(--c-rgb),.04);border:1px solid rgba(var(--c-rgb),.16);border-radius:4px;padding:8px;text-align:center}
#se-panel .se-stat .v{font-size:19px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);line-height:1;font-family:JetBrains Mono,monospace}
#se-panel .se-stat .l{font-size:7.5px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-top:4px}
#se-panel .se-sec{font-size:8.5px;letter-spacing:.22em;color:var(--mute);text-transform:uppercase;margin:6px 0 5px;border-top:1px solid rgba(var(--c-rgb),.08);padding-top:7px}
#se-panel .se-bar{height:6px;background:rgba(0,0,0,.35);border-radius:3px;overflow:hidden;margin-top:4px}
#se-panel .se-bar .fill{height:100%;background:linear-gradient(90deg,var(--ok),var(--cyan));box-shadow:0 0 6px var(--cyan)}
#se-panel .se-run{font-size:10px;color:var(--fg);padding:4px 4px;border-bottom:1px solid rgba(var(--c-rgb),.07);font-family:JetBrains Mono,monospace;display:flex;gap:8px}
#se-panel .se-run .id{color:var(--cyan)}
#se-panel .se-langs{font-size:9px;color:var(--mute);letter-spacing:.04em;margin-top:4px}
#se-panel .se-empty{color:var(--mute);font-size:10px;text-align:center;padding:8px}
#train-modal{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:340px;z-index:15;background:rgba(var(--panel-rgb,8,14,28),.97);border:1px solid var(--magenta);border-radius:6px;padding:18px;box-shadow:0 0 40px rgba(255,77,200,.4);display:none;font-family:inherit}
#train-modal.show{display:block}
#train-modal h3{font-size:11px;letter-spacing:.3em;text-transform:uppercase;color:var(--magenta);text-shadow:0 0 6px var(--magenta);margin:0 0 14px;text-align:center}
#train-modal label{display:block;font-size:9px;color:var(--mute);letter-spacing:.18em;text-transform:uppercase;margin:8px 0 4px}
#train-modal input,#train-modal select{width:100%;padding:7px 10px;background:rgba(0,0,0,.5);border:1px solid rgba(255,77,200,.3);color:var(--fg);font-family:inherit;font-size:12px;border-radius:3px}
#train-modal input:focus,#train-modal select:focus{outline:none;border-color:var(--magenta)}
#train-modal .tm-actions{display:flex;gap:8px;margin-top:14px}
#train-modal button.tm-act{flex:1;padding:8px 12px;border:1px solid rgba(255,77,200,.4);background:rgba(255,77,200,.06);color:var(--magenta);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#train-modal button.tm-act.primary{background:rgba(255,77,200,.18);border-color:var(--magenta)}
#train-modal button.tm-act:hover{background:rgba(255,77,200,.25)}
#train-modal .tm-status{font-size:10px;color:var(--mute);letter-spacing:.05em;margin-top:10px;min-height:14px;text-align:center}
#train-modal .tm-prompt-row{display:flex;gap:6px;margin-top:4px}
#train-modal .tm-prompt-row input{flex:1}
#train-modal .tm-countdown{font-size:36px;color:var(--magenta);text-shadow:0 0 12px var(--magenta);text-align:center;font-family:JetBrains Mono,monospace;font-weight:bold;letter-spacing:.05em}
#convo-toggle{padding:0 12px;height:46px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px;position:relative}
#convo-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.08);box-shadow:0 0 12px rgba(var(--c-rgb),.3)}
#wake-toggle.on{color:#ffe066;border-color:#ffe066;background:rgba(var(--g-rgb),.08);box-shadow:0 0 10px rgba(var(--g-rgb),.28)}
#wake-toggle{position:relative}
#wake-toggle.fired{animation:wakeFiredPulse .9s ease-out}
#wake-toggle.fired::after{content:'';position:absolute;inset:-6px;border-radius:6px;border:2px solid #ffe066;pointer-events:none;animation:wakeFiredRing .9s ease-out forwards;opacity:0;box-sizing:border-box}
.wake-ambient.ok{border-color:#9cffb7;color:#9cffb7;text-shadow:0 0 4px rgba(0,255,156,.55)}
@keyframes wakeFiredPulse{0%{box-shadow:0 0 6px rgba(var(--g-rgb),.4)}50%{box-shadow:0 0 22px rgba(var(--g-rgb),.85),inset 0 0 12px rgba(var(--g-rgb),.3)}100%{box-shadow:0 0 10px rgba(var(--g-rgb),.28)}}
@keyframes wakeFiredRing{0%{opacity:1;transform:scale(.9)}80%{opacity:.4;transform:scale(1.5)}100%{opacity:0;transform:scale(1.7)}}
.wake-ambient{position:fixed;bottom:140px;left:50%;transform:translateX(-50%);background:rgba(var(--panel-rgb,8,14,28),.85);border:1px solid rgba(var(--g-rgb),.35);color:#ffe066;font-size:10px;letter-spacing:.18em;padding:6px 14px;border-radius:3px;backdrop-filter:blur(6px);z-index:6;opacity:0;pointer-events:none;transition:opacity .25s, transform .25s;text-transform:uppercase;font-family:inherit;text-shadow:0 0 4px rgba(var(--g-rgb),.6)}
.persona-flash{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(.95);background:rgba(var(--panel-rgb,8,14,28),.92);border:1px solid var(--cyan);color:var(--cyan);font-size:18px;letter-spacing:.22em;padding:14px 28px;border-radius:4px;backdrop-filter:blur(8px);z-index:7;opacity:0;pointer-events:none;transition:opacity .15s ease-out, transform .25s ease-out;text-transform:uppercase;font-family:JetBrains Mono,monospace;text-shadow:0 0 8px rgba(var(--c-rgb),.7);box-shadow:0 0 24px rgba(var(--c-rgb),.35),inset 0 0 14px rgba(var(--c-rgb),.08)}
.jtb-pill{position:fixed;bottom:96px;right:50%;transform:translate(50%,12px);background:rgba(var(--panel-rgb,8,14,28),.92);border:1px solid var(--cyan);color:var(--cyan);font-size:10px;letter-spacing:.2em;padding:5px 14px;border-radius:99px;backdrop-filter:blur(6px);z-index:6;opacity:0;pointer-events:none;transition:opacity .2s ease-out,transform .25s ease-out;text-transform:uppercase;font-family:JetBrains Mono,monospace;cursor:pointer;box-shadow:0 0 14px rgba(var(--c-rgb),.25)}
.slash-ac{position:fixed;background:rgba(var(--panel-rgb,8,14,28),.96);border:1px solid var(--cyan);border-radius:4px;padding:4px;z-index:15;box-shadow:0 0 22px rgba(var(--c-rgb),.22);backdrop-filter:blur(8px);font-family:JetBrains Mono,monospace;display:none;max-height:240px;overflow-y:auto}
.slash-ac.show{display:block}
.slash-ac .sl-row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 10px;border-radius:3px;cursor:pointer;font-size:11px;color:var(--fg);transition:background .12s}
.slash-ac .sl-row:hover,.slash-ac .sl-row.active{background:rgba(var(--c-rgb),.14)}
.slash-ac .sl-cmd{color:var(--cyan);font-weight:600;letter-spacing:.04em}
.slash-ac .sl-hint{color:var(--mute);font-size:9.5px;letter-spacing:.03em}
.jtb-pill.show{opacity:.95;transform:translate(50%,0);pointer-events:auto}
.jtb-pill:hover{background:var(--cyan);color:var(--bg);box-shadow:0 0 22px rgba(var(--c-rgb),.55)}
.persona-flash.show{opacity:.95;transform:translate(-50%,-50%) scale(1)}
.wake-ambient.show{opacity:.85}
#convo-toggle .convo-dot{position:absolute;top:6px;right:6px;width:6px;height:6px;border-radius:50%;background:var(--mute);transition:all .2s}
#convo-toggle.on .convo-dot{background:var(--cyan);box-shadow:0 0 6px var(--cyan);animation:convoPulse 1.4s ease-in-out infinite}
#convo-toggle.state-recording .convo-dot{background:var(--cyan);box-shadow:0 0 10px var(--cyan);animation:convoPulse .8s ease-in-out infinite}
#convo-toggle.state-thinking .convo-dot{background:var(--gold);box-shadow:0 0 10px var(--gold);animation:convoSpin 1s linear infinite}
#convo-toggle.state-speaking .convo-dot{background:var(--magenta);box-shadow:0 0 10px var(--magenta);animation:convoPulse 1s ease-in-out infinite}
#convo-toggle.state-error .convo-dot{background:var(--err);box-shadow:0 0 8px var(--err)}
@keyframes convoPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.4)}}
@keyframes convoSpin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
#vad-panel{position:fixed;top:60px;left:24px;width:300px;z-index:10;border:1px solid rgba(var(--c-rgb),.35);border-radius:4px;background:rgba(var(--panel-rgb,8,14,28),.94);box-shadow:0 0 24px rgba(var(--c-rgb),.22);overflow:hidden;backdrop-filter:blur(8px);display:none}
#vad-panel.show{display:block}
#vad-panel .vp-head{padding:8px 12px;border-bottom:1px solid rgba(var(--c-rgb),.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;align-items:center;justify-content:space-between;background:rgba(var(--panel-rgb,8,14,28),.98)}
#vad-panel .vp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(var(--c-rgb),.2);border-radius:3px;font-size:10px}
#vad-panel .vp-head .close:hover{color:var(--err);border-color:var(--err)}
#vad-panel .vp-body{padding:10px 12px}
#vad-panel .vp-row{margin-bottom:10px}
#vad-panel .vp-row .vp-lbl{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute);display:flex;justify-content:space-between;align-items:baseline}
#vad-panel .vp-row .vp-lbl .v{color:var(--cyan);font-size:11px}
#vad-panel input[type=range]{width:100%;-webkit-appearance:none;appearance:none;background:transparent;margin-top:5px;cursor:pointer}
#vad-panel input[type=range]::-webkit-slider-runnable-track{height:3px;background:rgba(var(--c-rgb),.15);border-radius:2px}
#vad-panel input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);margin-top:-6px;cursor:grab}
#vad-panel input[type=range]::-moz-range-track{height:3px;background:rgba(var(--c-rgb),.15);border-radius:2px}
#vad-panel input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);border:0;cursor:grab}
#vad-panel .vp-meter{padding:8px 0;border-top:1px solid rgba(var(--c-rgb),.08);margin-top:8px}
#vad-panel .vp-meter .lbl{font-size:9px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-bottom:4px}
#vad-panel .vp-meter .bar-track{height:8px;background:rgba(var(--c-rgb),.05);border-radius:2px;position:relative;border:1px solid rgba(var(--c-rgb),.15)}
#vad-panel .vp-meter .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));transition:width .06s;border-radius:2px}
#vad-panel .vp-meter .threshold-line{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--gold);box-shadow:0 0 4px var(--gold)}
#vad-panel .vp-meter .barge-line{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--magenta);box-shadow:0 0 4px var(--magenta)}
#vad-panel .vp-row.act{display:flex;gap:6px;margin-top:12px;border-top:1px solid rgba(var(--c-rgb),.08);padding-top:10px}
#vad-panel .vp-row.act button{flex:1;padding:5px 8px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#vad-panel .vp-row.act button:hover{background:rgba(var(--c-rgb),.1);border-color:var(--cyan)}
#vad-panel .hint{font-size:9px;color:var(--mute);letter-spacing:.05em;margin-top:6px;font-style:italic}
#vad-toggle{padding:0 10px;height:46px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#vad-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(var(--c-rgb),.08)}
#convo-banner{position:fixed;left:50%;top:64px;transform:translateX(-50%);z-index:7;padding:6px 18px;border:1px solid var(--cyan);background:rgba(var(--c-rgb),.08);border-radius:99px;color:var(--cyan);font-size:10px;letter-spacing:.3em;text-transform:uppercase;text-shadow:0 0 6px var(--cyan);box-shadow:0 0 14px rgba(var(--c-rgb),.25);display:none;align-items:center;gap:8px}
#convo-banner.show{display:flex}
#convo-banner .level{height:14px;width:60px;background:rgba(var(--c-rgb),.1);border-radius:2px;overflow:hidden;border:1px solid rgba(var(--c-rgb),.2)}
#convo-banner .level .bar{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));transition:width .08s}
#voice-wave{position:fixed;left:50%;top:106px;transform:translateX(-50%);z-index:7;width:280px;height:46px;display:none;border:1px solid rgba(var(--c-rgb),.18);background:rgba(var(--c-rgb),.025);border-radius:3px;backdrop-filter:blur(4px);box-shadow:0 0 12px rgba(var(--c-rgb),.12)}
#voice-wave.show{display:block}
#voice-wave.state-listening{border-color:rgba(var(--c-rgb),.3);box-shadow:0 0 14px rgba(var(--c-rgb),.2)}
#voice-wave.state-recording{border-color:var(--cyan);box-shadow:0 0 18px rgba(var(--c-rgb),.4)}
#voice-wave.state-thinking{border-color:var(--magenta);box-shadow:0 0 18px rgba(255,77,200,.35)}
#voice-wave.state-speaking{border-color:var(--gold);box-shadow:0 0 18px rgba(255,215,112,.4)}
#gesture-toggle{padding:0 12px;height:46px;border:1px solid rgba(var(--c-rgb),.3);background:rgba(var(--c-rgb),.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#gesture-toggle.on{color:var(--magenta);border-color:var(--magenta);background:rgba(var(--m-rgb),.08);box-shadow:0 0 12px rgba(var(--m-rgb),.3)}
.msg .img-attach{max-width:280px;max-height:200px;border:1px solid rgba(var(--c-rgb),.4);border-radius:3px;margin-top:6px;box-shadow:0 0 10px rgba(var(--c-rgb),.18)}
.drop-overlay{position:fixed;inset:0;background:rgba(var(--c-rgb),.08);border:2px dashed var(--cyan);box-shadow:inset 0 0 60px rgba(var(--c-rgb),.2);z-index:50;display:none;align-items:center;justify-content:center;pointer-events:none}
.drop-overlay.show{display:flex}
.drop-overlay .label{font-size:24px;letter-spacing:.3em;color:var(--cyan);text-shadow:0 0 16px var(--cyan)}
.gesture-flash{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);font-size:22px;letter-spacing:.3em;color:var(--magenta);text-shadow:0 0 24px var(--magenta);pointer-events:none;z-index:9;opacity:0;transition:opacity .3s}
.gesture-flash.show{opacity:1}
@media(max-width:760px){.status .pill{display:none}.title{font-size:16px}#app{padding:18px 16px}#cam-panel{width:140px;top:50px;right:14px}}
</style></head><body>
<div id="nebula"></div>
<canvas id="netcanvas"></canvas>
<div id="frame"></div><div id="corner-tr"></div><div id="corner-bl"></div><div id="scanline"></div>
<div id="app">
  <header>
    <div class="title">A D A M</div>
    <div class="surfnav">
      <button class="surfbtn active" onclick="document.getElementById('input')&&document.getElementById('input').focus()" title="Jarvis HUD — you are here">⚡ Jarvis</button>
      <button class="surfbtn" onclick="_openPeer()" title="Open the Amni-Code IDE (:3000)">⌨ Code</button>
      <button class="surfbtn" onclick="_showCliInfo()" title="Adam terminal CLI">▤ CLI</button>
    </div>
    <div class="status">
      <span class="pill"><span class="dot"></span>GF(17) online</span>
      <span class="pill clickable" id="persona-pill" onclick="togglePersonaPanel()" title="Click to change persona + voice">persona —</span>
      <span class="pill clickable" id="status-pill" onclick="toggleStatusPanel()" title="System status: lessons, learning daemon, pending tests, shell history"><span class="sp-led" id="sp-led"></span><span id="sp-text">status —</span></span>
    </div>
    <div id="status-panel" class="td-hidden">
      <div class="sp-head"><span>◆ SYSTEM STATUS</span><span class="sp-close" onclick="toggleStatusPanel()">CLOSE</span></div>
      <div class="sp-row" onclick="toggleStatusPanel();togglePersonaPanel()"><span class="sp-name">lessons</span><span class="sp-val" id="sp-lessons">—</span></div>
      <div class="sp-row" id="sp-learn-row" onclick="toggleStatusPanel();toggleLearnPanel()"><span class="sp-led-inline" id="ld-led"></span><span class="sp-name" id="ld-text">learning —</span><span class="sp-arrow">›</span></div>
      <div class="sp-row" id="sp-tests-row" onclick="toggleStatusPanel();toggleTestsPanel()"><span class="sp-led-inline" id="tp-led"></span><span class="sp-name" id="tp-text">tests —</span><span class="sp-arrow">›</span></div>
      <div class="sp-row" id="sp-shell-row" onclick="toggleStatusPanel();toggleShellPanel()"><span class="sp-led-inline" id="sh-led"></span><span class="sp-name" id="sh-text">shell —</span><span class="sp-arrow">›</span></div>
      <div class="sp-row" id="sp-skill-failures-row" onclick="toggleStatusPanel();_skillFailuresShow()"><span class="sp-led-inline" id="sfl-led"></span><span class="sp-name" id="sfl-text">skill failures —</span><span class="sp-arrow">›</span></div>
      <div class="sp-row" id="sp-bookmarks-row" onclick="toggleStatusPanel();toggleBookmarksPanel()"><span class="sp-led-inline"></span><span class="sp-name">bookmarks ★</span><span class="sp-arrow">›</span></div>
      <div class="sp-row" id="sp-reminders-row" onclick="toggleStatusPanel();toggleRemindersPanel()"><span class="sp-led-inline" id="rm-led"></span><span class="sp-name" id="rm-text">reminders ⏰</span><span class="sp-arrow">›</span></div>
    </div>
    <div id="reminders-panel" class="td-hidden">
      <div class="sp-head"><span>⏰ REMINDERS</span><span class="sp-close" onclick="toggleRemindersPanel()">CLOSE</span></div>
      <div style="display:flex;gap:6px;margin-bottom:8px">
        <input type="text" id="rmp-new" placeholder="remind me to..." style="flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(var(--c-rgb),.2);color:var(--fg);padding:5px 9px;border-radius:3px;font-family:inherit;font-size:11px" onkeydown="if(event.key==='Enter')_remindersAddFromPanel()">
        <button class="pe-btn" onclick="_remindersAddFromPanel()" style="padding:5px 12px">ADD</button>
      </div>
      <div id="rm-list" style="max-height:60vh;overflow-y:auto"></div>
    </div>
    <div id="bookmarks-panel" class="td-hidden">
      <div class="sp-head"><span>★ BOOKMARKS</span><span class="sp-close" onclick="toggleBookmarksPanel()">CLOSE</span></div>
      <input type="text" id="bm-search" placeholder="filter…" oninput="_bookmarksPanelRender()" style="width:100%;background:rgba(0,0,0,.4);border:1px solid rgba(var(--c-rgb),.2);color:var(--fg);padding:5px 9px;border-radius:3px;font-family:inherit;font-size:11px;margin-bottom:8px">
      <div id="bm-list" style="max-height:60vh;overflow-y:auto"></div>
    </div>
    <span class="pill" id="lesson-pill" style="display:none">lessons —</span>
  </header>
  <div id="chat-wrap"><div id="log">
    <div class="welcome">
      <h2 id="welcome-heading">NEURAL INTERFACE READY</h2>
      <div id="welcome-tagline" style="color:var(--mute);font-size:11px;letter-spacing:.1em">Ask anything. Live data renders inline as glowing cards.</div>
      <div id="welcome-persona-stamp" style="margin-top:8px;font-size:10px;color:var(--mute);letter-spacing:.18em;text-transform:uppercase;font-family:JetBrains Mono,monospace"></div>
      <div class="examples">
        <button class="ex" onclick="quick(&quot;what's my local weather and forecast for the day today and this week?&quot;)"><span class="lbl">live data</span>What's my local weather and forecast?</button>
        <button class="ex" onclick="quick('Show me current system stats')"><span class="lbl">live data</span>Show me current system stats</button>
        <button class="ex" onclick="quick('What time is it in Tokyo?')"><span class="lbl">live data</span>What time is it in Tokyo?</button>
        <button class="ex" onclick="quick('Write a python function to reverse a string')"><span class="lbl">code</span>Write a python function to reverse a string</button>
        <button class="ex" onclick="quick('What is 17 * 23?')"><span class="lbl">math</span>What is 17 * 23?</button>
        <button class="ex" onclick="quick('Tell me a haiku about AI')"><span class="lbl">creative</span>Tell me a haiku about AI</button>
      </div>
    </div>
  </div></div>
  <div id="quick-bar" class="qb-collapsed">
    <button class="qchip qb-toggle" onclick="toggleQuickBar()" title="Show example prompts"><span class="ico">⋯</span><span id="qb-toggle-label">EXAMPLES</span></button>
    <button class="qchip qb-item qc-menu" onclick="toggleCmdMenu()" title="All modes & commands — click, no typing needed"><span class="ico">≡</span>MENU</button>
    <button class="qchip qb-item" onclick="cycleTheme()" oncontextmenu="event.preventDefault();togglePersonaPanel();return false" title="Cycle theme (right-click → full picker)"><span class="ico">🎨</span>THEME</button>
    <button class="qchip qb-item" onclick="_qcAsk(&quot;what's my local weather and forecast for today and this week?&quot;)"><span class="ico">🌤</span>WEATHER</button>
    <button class="qchip qb-item" onclick="_qcAsk('Show me current system stats')"><span class="ico">📊</span>SYSTEM</button>
    <button class="qchip qb-item" onclick="_qcAsk('git status')"><span class="ico">⎇</span>GIT</button>
    <button class="qchip qb-item qc-learn" onclick="_qcAsk('what are you learning right now?')"><span class="ico">🧠</span>DAEMON</button>
    <button class="qchip qb-item" onclick="_qcAsk('summarize this conversation so far')"><span class="ico">📝</span>SUMMARIZE</button>
    <button class="qchip qb-item" onclick="_qcBriefing()" title="Adam's last-24h activity briefing"><span class="ico">◈</span>BRIEFING</button>
    <button class="qchip qb-item" onclick="toggleSessionsPanel()" title="Browse past chat sessions"><span class="ico">🗂</span>SESSIONS</button>
    <button class="qchip qb-item" onclick="_qcAsk('what can you do?')"><span class="ico">?</span>HELP</button>
    <span class="qchip-hint qb-item">drag a file · paste image · Ctrl+K to search</span>
  </div>
  <div id="composer">
    <button id="mic-shell" type="button" onclick="toggleMic()" title="Voice input">⏵</button>
    <div id="input-shell"><textarea id="input" placeholder="Speak or type… (/help for commands)" autofocus></textarea></div>
    <button id="jarvis-toggle" type="button" onclick="toggleJarvisMode()" title="Engage hands-free mode — convo + wake + gesture + voice all on">HANDS FREE</button>
    <button id="tools-toggle" type="button" onclick="toggleToolsDrawer()" title="Open tools drawer (voice, gesture, coach, export, etc)">TOOLS</button>
    <button id="send" onclick="send()">TRANSMIT</button>
  </div>
</div>
<div id="tools-drawer" class="td-hidden">
  <div class="td-head"><span>◆ TOOLS</span><span class="td-close" onclick="toggleToolsDrawer()">CLOSE</span></div>
  <div class="td-section">
    <div class="td-label">VOICE &amp; CONVO</div>
    <div class="td-grid">
      <button class="td-btn" id="voiceout-toggle" type="button" onclick="toggleVoiceOut()" title="Speak responses (TTS)">VOICE OUT</button>
      <button class="td-btn" id="convo-toggle" type="button" onclick="toggleConvo()" title="Continuous hands-free conversation (VAD)"><span class="convo-dot"></span>CONVO</button>
      <button class="td-btn" id="wake-toggle" type="button" onclick="toggleWake()" oncontextmenu="event.preventDefault();_wakeConfigPrompt();return false" title='Wake word gate. Right-click to add custom wake words.'>WAKE</button>
      <button class="td-btn" id="vad-toggle" type="button" onclick="toggleVadPanel()" title="Tune VAD thresholds for your microphone">VAD</button>
    </div>
  </div>
  <div class="td-section">
    <div class="td-label">VISION</div>
    <div class="td-grid">
      <button class="td-btn" id="gesture-toggle" type="button" onclick="toggleGesture()" title="Hand gesture control (webcam)">GESTURE</button>
      <button class="td-btn" id="pose-toggle" type="button" onclick="togglePoseCoach()" title="Camera form coach — counts reps + checks angles for push-ups, sit-ups, squats, curls">PT COACH</button>
    </div>
  </div>
  <div class="td-section">
    <div class="td-label">LEARNING</div>
    <div class="td-grid">
      <button class="td-btn" id="coach-toggle" type="button" onclick="toggleCoachPanel()" title="Ask-answer-ask coaching session">COACH</button>
      <button class="td-btn" id="mem-toggle" type="button" onclick="toggleMem()" title="Inspect what Adam knows">MEMORY</button>
      <button class="td-btn" id="se-toggle" type="button" onclick="toggleSePanel()" title="Software-engineer dashboard: code map + coding attempts + open runs">SW ENG</button>
    </div>
  </div>
  <div class="td-section">
    <div class="td-label">SESSION</div>
    <div class="td-grid">
      <button class="td-btn" id="export-btn" type="button" onclick="_exportChatMd()" title="Download this conversation as Markdown">EXPORT</button>
      <button class="td-btn" id="perms-btn" type="button" onclick="togglePermsPanel()" title="Location, microphone, camera + notification permissions">PERMISSIONS</button>
      <button class="td-btn" id="pclog-btn" type="button" onclick="togglePcLogPanel()" title="Audit log of every PC action Adam proposed/ran/refused">PC LOG</button>
    </div>
  </div>
</div>
<div id="task-tray">
  <div class="tray-head"><span class="dot"></span><span>RUNNING TASKS</span><span class="ct" id="tray-ct">0 active</span></div>
  <div id="tray-rows"></div>
</div>
<div id="mem-panel">
  <div class="mem-head"><span>◆ MEMORY INSPECTOR</span><span class="close" onclick="toggleMem()">CLOSE</span></div>
  <div class="mem-section" id="mem-stats"><h3>SUBSTRATE OVERVIEW <span class="count" id="mem-uptime">—</span></h3><div id="mem-stat-grid">loading...</div></div>
  <div class="mem-section" id="mem-profile-sec"><h3>WHAT I KNOW ABOUT YOU <span class="count" id="mem-profile-n">—</span></h3><div id="mem-pending"></div><div id="mem-profile">loading...</div></div>
  <div class="mem-section" id="mem-kg-sec"><h3>KNOWLEDGE GRAPH <span class="count" id="mem-kg-n">—</span></h3><div id="mem-kg">loading...</div></div>
  <div class="mem-section" id="mem-coach-sec"><h3>COACH MASTERY <span class="count" id="mem-coach-n">—</span></h3><div id="mem-coach">loading...</div></div>
  <div class="mem-section" id="mem-daemon-sec"><h3>LEARNING DAEMON <span class="count" id="mem-daemon-status">—</span></h3><div id="mem-daemon">loading...</div></div>
</div>
<div id="vad-panel">
  <div class="vp-head"><span>◆ VAD TUNING</span><span class="close" onclick="toggleVadPanel()">CLOSE</span></div>
  <div class="vp-body">
    <div class="vp-row"><div class="vp-lbl"><span>Speech threshold</span><span class="v" id="vp-vt-v">18</span></div><input type="range" id="vp-vt" min="5" max="50" step="1" oninput="_onVadSlider('vad_threshold',this.value)"></div>
    <div class="vp-row"><div class="vp-lbl"><span>Barge-in threshold</span><span class="v" id="vp-bt-v">26</span></div><input type="range" id="vp-bt" min="15" max="60" step="1" oninput="_onVadSlider('barge_threshold',this.value)"></div>
    <div class="vp-row"><div class="vp-lbl"><span>Silence to send (ms)</span><span class="v" id="vp-sl-v">600</span></div><input type="range" id="vp-sl" min="200" max="2000" step="50" oninput="_onVadSlider('silence_ms',this.value)"></div>
    <div class="vp-row"><div class="vp-lbl"><span>Min utterance (ms)</span><span class="v" id="vp-ms-v">150</span></div><input type="range" id="vp-ms" min="50" max="500" step="10" oninput="_onVadSlider('min_speech_ms',this.value)"></div>
    <div class="vp-meter">
      <div class="lbl">Live mic level (open CONVO to populate)</div>
      <div class="bar-track"><div class="bar-fill" id="vp-meter-fill"></div><div class="threshold-line" id="vp-vt-line"></div><div class="barge-line" id="vp-bt-line"></div></div>
      <div class="hint">Gold = speech threshold · Magenta = barge-in threshold. Talk normally and tune sliders so gold sits just below your speaking level and magenta above.</div>
    </div>
    <div class="vp-row act"><button onclick="_resetVad()">RESET</button><button onclick="toggleVadPanel()">DONE</button></div>
  </div>
</div>
<div id="convo-banner"><span id="convo-state-label">LISTENING</span><span class="level"><span class="bar" id="convo-level-bar"></span></span></div>
<canvas id="voice-wave" width="560" height="92"></canvas>
<div id="persona-panel">
  <div class="pp-head"><span>◆ PERSONA + VOICE</span><span class="close" onclick="togglePersonaPanel()">CLOSE</span></div>
  <div class="pp-section">
    <h3>PERSONA</h3>
    <div id="pp-list"><div class="pp-empty">loading…</div></div>
    <div class="pp-input-row" style="margin-top:8px"><input type="text" id="pp-new" placeholder="learn new (e.g. Sherlock, Tony Stark)"><button class="act" onclick="_personaLearnNew()">LEARN</button><button class="act" onclick="_personaImportFile()" title="Import a persona from .json file">IMPORT</button></div>
    <div style="font-size:9px;color:var(--mute);margin-top:4px;letter-spacing:.05em">Unknown personas trigger a web-learn flow.</div>
    <input type="file" id="pp-import-file" accept="application/json,.json" style="display:none" onchange="_personaImportPicked(event)">
  </div>
  <div class="pp-section">
    <h3>EDIT CURRENT</h3>
    <div id="pp-edit"><div class="pp-empty">pick a persona above first</div></div>
  </div>
  <div class="pp-section">
    <h3>THEME</h3>
    <div id="pp-themes" class="pp-themes"><div class="pp-empty">loading…</div></div>
    <div style="font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.04em">Amni-scient palettes. Clean themes drop the scanline + particle field for a minimal, AnythingLLM-style surface.</div>
  </div>
  <div class="pp-section">
    <h3>STREAM PACE</h3>
    <div class="pe-row"><label for="pp-pace">pace</label><input type="range" min="20" max="200" step="5" id="pp-pace" oninput="_paceSliderChange(this.value)"><span class="pe-val" id="pp-pace-v">—</span></div>
    <div style="display:flex;align-items:center;gap:8px;margin-top:4px"><span id="pp-pace-mode" style="font-size:9px;color:var(--mute);letter-spacing:.1em;flex:1">—</span><button class="pe-btn" onclick="_paceReset()" style="padding:3px 8px;font-size:8.5px">RESET TO PERSONA</button></div>
    <div style="font-size:9px;color:var(--mute);margin-top:4px;letter-spacing:.04em">Characters per second the chat bubble reveals. Persona default kicks in when override is reset.</div>
  </div>
  <div class="pp-section">
    <h3>TTS VOICE</h3>
    <div id="pp-voices"><div class="pp-empty">loading…</div></div>
    <div style="font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.05em">Piper voices ship as ~50MB .onnx — install more from <code style="color:var(--cyan)">github.com/rhasspy/piper</code></div>
  </div>
  <div class="pp-section">
    <h3>STATUS</h3>
    <div style="font-size:10px;color:var(--mute)">Current: <span id="pp-current" style="color:var(--cyan)">—</span></div>
    <div style="font-size:10px;color:var(--mute);margin-top:4px">TTS backend: <span id="pp-tts-backend" style="color:var(--cyan)">—</span></div>
  </div>
</div>
<div id="learn-panel">
  <div class="lp-head"><span>◆ LEARNING DAEMON</span><span class="close" onclick="toggleLearnPanel()">CLOSE</span></div>
  <div class="lp-section">
    <h3>NOW LEARNING</h3>
    <div id="lp-now" class="lp-now idle"><div class="topic">— idle —</div><div class="phase">waiting for curiosity tick</div></div>
  </div>
  <div class="lp-section">
    <h3>STATS</h3>
    <div class="lp-stat-grid">
      <div class="lp-stat"><span class="v" id="lp-facts">—</span><span class="k">facts learned</span></div>
      <div class="lp-stat"><span class="v" id="lp-rate">—</span><span class="k">facts / hour</span></div>
      <div class="lp-stat"><span class="v" id="lp-queue">—</span><span class="k">queue depth</span></div>
      <div class="lp-stat"><span class="v" id="lp-uptime">—</span><span class="k">uptime</span></div>
      <div class="lp-stat"><span class="v" id="lp-urls">—</span><span class="k">urls ingested</span></div>
      <div class="lp-stat"><span class="v" id="lp-cells">—</span><span class="k">atlas cells</span></div>
    </div>
    <div class="lp-btn-row"><button class="lp-act" id="lp-pause-btn" onclick="_daemonToggle()">PAUSE</button><button class="lp-act" onclick="_daemonTick()">CURIOSITY TICK</button></div>
    <div class="lp-queue-row"><input type="text" id="lp-queue-topic" placeholder="queue topic (e.g. mitochondrial dynamics)" onkeydown="if(event.key==='Enter')_daemonQueue()"><button class="lp-act" onclick="_daemonQueue()">QUEUE</button></div>
  </div>
  <div class="lp-section">
    <h3>RECENT TOPICS</h3>
    <div id="lp-recent"><div style="font-size:10px;color:var(--mute);text-align:center;padding:8px;font-style:italic">no completed topics yet</div></div>
  </div>
</div>
<div id="sessions-panel">
  <div class="sp-head"><span>◆ CHAT SESSIONS</span><span class="close" onclick="toggleSessionsPanel()">CLOSE</span></div>
  <div class="sp-toolbar"><button onclick="_pollSessionsList()">REFRESH</button><button onclick="_spNewSession()" title="Start a fresh session">NEW</button><span class="sp-count" id="sp-count">—</span></div>
  <div class="sp-list" id="sp-list"><div class="sp-empty">loading…</div></div>
</div>
<div id="coach-panel">
  <div class="cp-head"><span>◆ COACH · ASK-ANSWER-ASK</span><span id="cp-streak-badge" class="cp-streak-badge" title="Consecutive days you've practiced">—</span><span class="close" onclick="toggleCoachPanel()">CLOSE</span></div>
  <div class="cp-section" id="cp-start-section">
    <h3>NEW SESSION</h3>
    <div class="cp-topic-row"><input type="text" id="cp-topic" placeholder="topic (e.g. python decorators, krebs cycle)"><select id="cp-diff"><option value="1">1 — intro</option><option value="2" selected>2 — basic</option><option value="3">3 — intermediate</option><option value="4">4 — advanced</option><option value="5">5 — expert</option></select><button class="cp-act" onclick="_coachStart()">START</button></div>
    <div style="font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.05em">Adam will ask a question, grade your answer, then escalate or back off based on your streak.</div>
  </div>
  <div class="cp-section" id="cp-reviews-section" style="display:none">
    <h3 style="display:flex;justify-content:space-between;align-items:center">DUE FOR REVIEW <span id="cp-reviews-count" style="font-size:9px;color:var(--magenta)"></span></h3>
    <div id="cp-reviews-list"></div>
  </div>
  <div class="cp-section" id="cp-topics-section">
    <h3 style="display:flex;justify-content:space-between;align-items:center">PRACTICED TOPICS <button class="cp-act" onclick="_coachLoadTopics()" style="padding:3px 8px;font-size:8px">REFRESH</button></h3>
    <div id="cp-topics-list"><div style="font-size:10px;color:var(--mute);font-style:italic;text-align:center;padding:6px">loading…</div></div>
  </div>
  <div class="cp-section" id="cp-active-section" style="display:none">
    <h3 id="cp-topic-head">TOPIC</h3>
    <div id="cp-q" class="cp-q empty">— ask to begin —</div>
    <div class="cp-meta"><div>DIFFICULTY<span class="v" id="cp-diff-v">—</span></div><div>ASKED<span class="v" id="cp-asked-v">0</span></div><div><span class="cp-streak correct" id="cp-streak-correct">0 ✓</span> <span class="cp-streak wrong" id="cp-streak-wrong">0 ✗</span></div></div>
    <div style="font-size:9px;color:var(--mute);margin-top:8px;letter-spacing:.15em;text-transform:uppercase">MASTERY <span style="color:var(--magenta);font-size:11px;margin-left:4px" id="cp-mastery-pct">0%</span></div>
    <div class="cp-mastery-bar"><div class="cp-mastery-fill" id="cp-mastery-fill" style="width:0%"></div></div>
    <div id="cp-hint" class="cp-hint" style="display:none"></div>
    <div id="cp-grade-slot"></div>
    <textarea id="cp-answer" placeholder="type your answer (or use mic) — Ctrl+Enter to submit" rows="3" style="margin-top:8px;width:100%"></textarea>
    <div class="cp-btn-row"><button class="cp-act" onclick="_coachAnswer()">SUBMIT</button><button class="cp-act" onclick="_coachHint()">HINT</button><button class="cp-act" onclick="_coachSkip()">SKIP</button><button class="cp-act" onclick="_coachAsk()">NEXT</button><button class="cp-act" id="cp-voice-toggle" onclick="_coachToggleVoice()" title="Auto-speak questions, feedback, and hints">VOICE OFF</button><button class="cp-act" id="cp-replay-btn" onclick="_coachReplayQuestion()" title="Re-speak current question">↻ REPLAY</button><button class="cp-act danger" onclick="_coachEnd()" style="margin-left:auto">END SESSION</button></div>
  </div>
</div>
<div id="shell-panel">
  <div class="sh-head"><span>◆ SHELL AUDIT</span><span class="close" onclick="toggleShellPanel()">CLOSE</span></div>
  <div class="sh-toolbar"><button onclick="_pollShellHistory()">REFRESH</button><button id="sh-errors-btn" onclick="_shToggleErrors()">ERRORS ONLY</button><span class="sh-summary" id="sh-summary">—</span></div>
  <div class="sh-section" id="sh-list"><div class="sh-empty">loading…</div></div>
</div>
<div id="tests-panel">
  <div class="tp-head"><span>◆ PENDING TESTS</span><span class="close" onclick="toggleTestsPanel()">CLOSE</span></div>
  <div class="tp-toolbar">
    <button onclick="_pollTestsList()">REFRESH</button>
    <button onclick="_tpShowDone(!_tpIncludeDone)" id="tp-toggle-done">SHOW DONE</button>
    <span class="tp-summary" id="tp-summary">—</span>
  </div>
  <div class="tp-section">
    <div id="tp-list"><div class="tp-empty">loading…</div></div>
  </div>
</div>
<div id="gesture-tour">
  <h3>◆ HAND GESTURES READY</h3>
  <div class="tour-intro">Adam tracks your hand at 60 fps via MediaPipe. Hold a gesture in front of the camera and Adam will act. Six built-in gestures shown below — and you can teach Adam new ones too.</div>
  <div class="gesture-grid">
    <div class="g-card"><div class="g-emoji">🤏</div><div class="g-name">PINCH</div><div class="g-action">toggle voice output</div></div>
    <div class="g-card"><div class="g-emoji">✊</div><div class="g-name">FIST</div><div class="g-action">clear the chat</div></div>
    <div class="g-card"><div class="g-emoji">🖐️</div><div class="g-name">OPEN PALM</div><div class="g-action">show system stats</div></div>
    <div class="g-card"><div class="g-emoji">✌️</div><div class="g-name">PEACE</div><div class="g-action">cycle color theme</div></div>
    <div class="g-card"><div class="g-emoji">👆</div><div class="g-name">POINT</div><div class="g-action">"tell me more"</div></div>
    <div class="g-card"><div class="g-emoji">👍</div><div class="g-name">THUMB UP</div><div class="g-action">submit current input</div></div>
  </div>
  <div class="tour-actions">
    <button class="gt-act" onclick="_gtTrainFromTour()">+ TEACH ADAM A NEW ONE</button>
    <span class="tour-hint">Hold each pose for ~0.4s; brief cooldown between fires</span>
    <button class="gt-act primary" onclick="_gtClose()">GOT IT</button>
  </div>
</div>
<div id="kbd-overlay">
  <div class="kbd-head"><span>◆ KEYBOARD SHORTCUTS</span><span class="kbd-close" onclick="toggleKbdOverlay()">CLOSE</span></div>
  <div class="kbd-body"><div style="font-size:10px;color:var(--mute);text-align:center;padding:10px">loading…</div></div>
  <div class="kbd-foot">Press <kbd>?</kbd> anytime to toggle this overlay · <kbd>Esc</kbd> closes anything open</div>
</div>
<div id="train-modal">
  <h3 id="tm-title">◆ TEACH NEW GESTURE</h3>
  <div id="tm-step-1">
    <label>GESTURE NAME</label>
    <input type="text" id="tm-name" placeholder="e.g. spock, ok-sign, hang-loose">
    <label>ACTION WHEN RECOGNIZED</label>
    <select id="tm-action-type"><option value="prompt">Send chat message</option><option value="builtin">Trigger built-in action</option><option value="panel">Open panel</option></select>
    <div class="tm-prompt-row" id="tm-action-input-wrap"><input type="text" id="tm-action-value" placeholder="text or selector"></div>
    <div class="tm-actions"><button class="tm-act" onclick="_tmClose()">CANCEL</button><button class="tm-act primary" onclick="_tmStartRecord()">RECORD</button></div>
    <div class="tm-status" id="tm-status">Hold the gesture in view of the camera, then click RECORD</div>
  </div>
  <div id="tm-step-2" style="display:none">
    <div class="tm-countdown" id="tm-countdown">3</div>
    <div class="tm-status" id="tm-rec-status">Get your hand ready…</div>
  </div>
</div>
<div id="cam-panel">
  <div class="cam-head"><span><span class="dot"></span>HAND TRACK</span><span><button class="cam-train-btn" onclick="_gtOpen()" title="Show built-in gestures">?</button><button class="cam-train-btn" id="cam-train-btn" onclick="_tmOpen()" title="Teach a new gesture">+ TRAIN</button><button class="cam-train-btn" onclick="_exportCustomGestures()" title="Download trained gestures as JSON">⬇</button><button class="cam-train-btn" onclick="document.getElementById('gesture-import-input').click()" title="Import gestures from JSON">⬆</button><input type="file" id="gesture-import-input" accept="application/json,.json" style="display:none" onchange="_importCustomGestures(this)"><span id="cam-fps">— fps</span></span></div>
  <div id="cam-stage">
    <video id="cam-video" autoplay playsinline muted></video>
    <canvas id="cam-landmarks"></canvas>
  </div>
  <div class="gesture-readout" id="gesture-readout">—</div>
  <div class="custom-list" id="custom-list"></div>
</div>
<div id="pose-panel">
  <div class="pc-head"><span><span class="dot"></span>PT COACH</span><span><span class="pc-fps" id="pose-fps">— fps</span> <span class="pc-close" onclick="togglePoseCoach(false)">CLOSE</span></span></div>
  <div id="pose-stage">
    <video id="pose-video" autoplay playsinline muted></video>
    <canvas id="pose-landmarks"></canvas>
  </div>
  <div class="pc-controls">
    <select id="pose-exercise" title="Pick an exercise"><option value="pushup">Push-up</option><option value="situp">Sit-up / Crunch</option><option value="squat">Squat</option><option value="bicep_curl">Bicep curl</option></select>
    <button class="pc-go" id="pose-go" onclick="_poseToggleSession()">START</button>
  </div>
  <div class="pc-stats">
    <div class="pc-stat"><div class="v" id="pose-reps">0</div><div class="l">reps</div></div>
    <div class="pc-stat"><div class="v" id="pose-clean">0</div><div class="l">clean</div></div>
    <div class="pc-stat"><div class="v" id="pose-angle">—</div><div class="l">angle</div></div>
  </div>
  <div class="pc-fb" id="pose-fb">Pick an exercise, hit START, and step back so your whole body is in frame.</div>
</div>
<div id="perms-panel" class="td-hidden">
  <div class="pm-head"><span>◆ PERMISSIONS</span><span class="pm-close" onclick="togglePermsPanel(false)">CLOSE</span></div>
  <div class="pm-row" data-perm="geolocation"><div class="pm-info"><span class="pm-name">📍 Location</span><span class="pm-why">local weather + "near me" searches</span></div><span class="pm-state" id="pm-state-geolocation">—</span><button class="pm-grant" onclick="_permsRequest('geolocation')">GRANT</button></div>
  <div class="pm-row" data-perm="microphone"><div class="pm-info"><span class="pm-name">🎤 Microphone</span><span class="pm-why">voice input · wake word · convo mode</span></div><span class="pm-state" id="pm-state-microphone">—</span><button class="pm-grant" onclick="_permsRequest('microphone')">GRANT</button></div>
  <div class="pm-row" data-perm="camera"><div class="pm-info"><span class="pm-name">📷 Camera</span><span class="pm-why">gestures + PT form coaching</span></div><span class="pm-state" id="pm-state-camera">—</span><button class="pm-grant" onclick="_permsRequest('camera')">GRANT</button></div>
  <div class="pm-row" data-perm="notifications"><div class="pm-info"><span class="pm-name">🔔 Notifications</span><span class="pm-why">due reminders when tabbed away</span></div><span class="pm-state" id="pm-state-notifications">—</span><button class="pm-grant" onclick="_permsRequest('notifications')">GRANT</button></div>
  <div class="pm-foot">Granted locally in your browser. Adam never sends raw location, audio, or video off-box — only PII-scrubbed search text ever leaves.</div>
</div>
<div id="pclog-panel" class="td-hidden">
  <div class="pl-head"><span>◆ PC ACTION LOG</span><span class="pl-close" onclick="togglePcLogPanel(false)">CLOSE</span></div>
  <div class="pl-summary" id="pl-summary">loading…</div>
  <div id="pl-rows"></div>
</div>
<div id="se-panel" class="td-hidden">
  <div class="se-head"><span>◆ SOFTWARE ENGINEER</span><span class="se-close" onclick="toggleSePanel(false)">CLOSE</span></div>
  <div id="se-body"><div class="se-empty">loading…</div></div>
</div>
<div id="chat-search"><div class="cs-row"><input type="text" id="cs-input" placeholder="search chat… (case-insensitive substring)" autocomplete="off"><span class="cs-count" id="cs-count">0/0</span><button class="cs-btn" onclick="_csPrev()" title="Previous match (Shift+Enter)">↑</button><button class="cs-btn" onclick="_csNext()" title="Next match (Enter)">↓</button><button class="cs-btn" onclick="closeChatSearch()" title="Close (Esc)">✕</button></div><div class="cs-help">Ctrl+K to open · Enter / ↑↓ to navigate · Esc to close · empty query restores all bubbles</div></div>
<div id="cmd-menu" onclick="if(event.target===this)toggleCmdMenu(false)"><div class="cm-box"><div class="cm-head"><span>◆ MENU — modes &amp; commands</span><span class="cm-close" onclick="toggleCmdMenu(false)">CLOSE</span></div><div id="cmd-menu-list" class="cm-list"></div></div></div>
<canvas id="adam-core" width="120" height="120" title="Adam core — click to collapse"></canvas>
<div id="toast-stack"></div>
<div id="drop-overlay" class="drop-overlay"><div class="label">◆ DROP IMAGE OR TEXT FILE FOR ADAM</div></div>
<div id="gesture-flash" class="gesture-flash"></div>
<div class="sidehint">Adam • Amni-Ai • Local • GF(17)</div>
<span id="wd-pill" onclick="_wdToggle()" oncontextmenu="event.preventDefault();_wdCopy();return false" title="Click to browse · Right-click to copy path"><span class="wd-lbl">WORKDIR</span><span id="wd-path">—</span></span>
<div id="wd-panel">
  <div class="wp-head"><span>◆ WORKDIR TREE</span><span class="close" onclick="_wdToggle()">CLOSE</span></div>
  <div class="wp-base" id="wp-base">—</div>
  <div class="wp-list" id="wp-list"><div class="wp-empty">loading…</div></div>
</div>
<script>
(function(){
  try{var u=new URL(location.href);var qt=u.searchParams.get('token');if(qt){try{localStorage.setItem('amni_token',qt)}catch(_){}u.searchParams.delete('token');try{history.replaceState(null,'',u.pathname+(u.search||'')+u.hash)}catch(_){}}}catch(_){}
  var _F=window.fetch.bind(window);
  window.fetch=function(input,init){
    init=init||{};
    try{var url=(typeof input==='string')?input:((input&&input.url)||'');if(url.charAt(0)==='/'||url.indexOf(location.origin)===0){var t='';try{t=localStorage.getItem('amni_token')||''}catch(_){}if(t){var hh=new Headers((init&&init.headers)||(typeof input!=='string'&&input&&input.headers)||{});if(!hh.has('X-Amni-Token'))hh.set('X-Amni-Token',t);init.headers=hh}}}catch(_){}
    return _F(input,init).then(function(r){if(r&&r.status===401){try{r.clone().json().then(function(j){if(j&&j.auth_required&&!window._amniTokPrompting){window._amniTokPrompting=true;var nt=window.prompt('Adam access token (set on the server via AMNI_AUTH_TOKEN):','');window._amniTokPrompting=false;if(nt){try{localStorage.setItem('amni_token',nt.trim())}catch(_){}location.reload()}}}).catch(function(){})}catch(_){}}return r});
  };
  try{if('serviceWorker' in navigator)window.addEventListener('load',function(){navigator.serviceWorker.register('/sw.js').catch(function(){})})}catch(_){}
})();
const SKEY='amni_jarvis_session',VKEY='amni_jarvis_voiceout';
window._TC={c:'0,229,255',m:'255,43,214',g:'255,224,102',hexC:'#00e5ff',hexG:'#ffd770'};
const THEME_KEY='amni_jarvis_theme';
const THEMES={
  jarvis:{label:'Jarvis',min:false,vars:{'--bg':'#040711','--bg2':'#0a1224','--glass':'rgba(10,18,36,.55)','--panel-rgb':'10,18,36','--cyan':'#00e5ff','--cyan2':'#00b8d4','--magenta':'#ff2bd6','--gold':'#ffd770','--ok':'#00ff9d','--err':'#ff5577','--fg':'#dff6ff','--mute':'#5e7a99','--c-rgb':'0,229,255','--m-rgb':'255,43,214','--g-rgb':'255,224,102'}},
  scient:{label:'Scient',min:false,vars:{'--bg':'#05080f','--bg2':'#0c1526','--glass':'rgba(12,21,38,.6)','--panel-rgb':'12,21,38','--cyan':'#22d3ee','--cyan2':'#0ea5c4','--magenta':'#e879f9','--gold':'#fde047','--ok':'#34d399','--err':'#fb7185','--fg':'#e6f6ff','--mute':'#64748b','--c-rgb':'34,211,238','--m-rgb':'232,121,249','--g-rgb':'253,224,71'}},
  'clean-dark':{label:'Clean Dark',min:true,vars:{'--bg':'#0e1116','--bg2':'#161b22','--glass':'rgba(22,27,34,.88)','--panel-rgb':'22,27,34','--cyan':'#3b82f6','--cyan2':'#2563eb','--magenta':'#8b5cf6','--gold':'#f59e0b','--ok':'#3fb950','--err':'#f85149','--fg':'#e6edf3','--mute':'#7d8590','--c-rgb':'59,130,246','--m-rgb':'139,92,246','--g-rgb':'245,158,11'}},
  'clean-light':{label:'Clean Light',min:true,vars:{'--bg':'#ffffff','--bg2':'#f4f6f8','--glass':'rgba(255,255,255,.92)','--panel-rgb':'244,246,248','--cyan':'#2563eb','--cyan2':'#1d4ed8','--magenta':'#7c3aed','--gold':'#d97706','--ok':'#059669','--err':'#dc2626','--fg':'#1f2937','--mute':'#6b7280','--c-rgb':'37,99,235','--m-rgb':'124,58,237','--g-rgb':'217,119,6'}},
  nebula:{label:'Nebula',min:false,vars:{'--bg':'#0c0518','--bg2':'#1a0a2e','--glass':'rgba(26,10,46,.6)','--panel-rgb':'26,10,46','--cyan':'#d946ef','--cyan2':'#a21caf','--magenta':'#22d3ee','--gold':'#f0abfc','--ok':'#34d399','--err':'#fb7185','--fg':'#fbe8ff','--mute':'#8b6f9e','--c-rgb':'217,70,239','--m-rgb':'34,211,238','--g-rgb':'240,171,252'}},
  terminal:{label:'Terminal',min:false,vars:{'--bg':'#000800','--bg2':'#001a00','--glass':'rgba(0,26,0,.6)','--panel-rgb':'0,26,0','--cyan':'#00ff66','--cyan2':'#00cc52','--magenta':'#39ff14','--gold':'#aaff00','--ok':'#00ff66','--err':'#ff5555','--fg':'#c7ffd0','--mute':'#3a7a4a','--c-rgb':'0,255,102','--m-rgb':'57,255,20','--g-rgb':'170,255,0'}},
  solar:{label:'Solar',min:false,vars:{'--bg':'#1a1410','--bg2':'#241b12','--glass':'rgba(36,27,18,.6)','--panel-rgb':'36,27,18','--cyan':'#ffb347','--cyan2':'#e6962e','--magenta':'#ff7e5f','--gold':'#ffd166','--ok':'#9acd68','--err':'#ef6f6c','--fg':'#fdf0dd','--mute':'#9a836a','--c-rgb':'255,179,71','--m-rgb':'255,126,95','--g-rgb':'255,209,102'}}
};
function _syncThemeCanvas(){const cs=getComputedStyle(document.documentElement);const g=(k,d)=>(cs.getPropertyValue(k).trim()||d);window._TC={c:g('--c-rgb','0,229,255'),m:g('--m-rgb','255,43,214'),g:g('--g-rgb','255,224,102'),hexC:g('--cyan','#00e5ff'),hexG:g('--gold','#ffd770')}}
function applyTheme(name){const t=THEMES[name]||THEMES.jarvis;const r=document.documentElement;for(const k in t.vars)r.style.setProperty(k,t.vars[k]);if(document.body){document.body.setAttribute('data-theme',name);document.body.classList.toggle('theme-min',!!t.min)}localStorage.setItem(THEME_KEY,name);_syncThemeCanvas();try{_renderThemePicker()}catch(_){}}
function _renderThemePicker(){const el=document.getElementById('pp-themes');if(!el)return;const cur=localStorage.getItem(THEME_KEY)||'jarvis';el.innerHTML=Object.keys(THEMES).map(n=>{const v=THEMES[n].vars;return `<div class="th-sw${n===cur?' active':''}" onclick="applyTheme('${n}')" title="${THEMES[n].label}" style="background:${v['--bg']};border-color:rgba(${v['--c-rgb']},.5)"><span class="th-dot" style="background:${v['--cyan']}"></span><span class="th-dot" style="background:${v['--magenta']}"></span><span class="th-dot" style="background:${v['--gold']}"></span><span class="th-name" style="color:${v['--fg']}">${THEMES[n].label}</span></div>`}).join('')}
applyTheme(localStorage.getItem(THEME_KEY)||'jarvis');
let sid=localStorage.getItem(SKEY)||'';
let voiceOut=localStorage.getItem(VKEY)==='1';
let recog=null,recoOn=false;
const log=document.getElementById('log'),input=document.getElementById('input'),send_btn=document.getElementById('send'),lessonPill=document.getElementById('lesson-pill'),personaPill=document.getElementById('persona-pill');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function _netHint(e){const m=(e&&e.message)||String(e||'');return /failed to fetch|networkerror|load failed|fetch failed/i.test(m)?'connection dropped or Adam is still warming up — give it a moment and try again':m}
const _MATH_PLACEHOLDERS=[];
function _stashMath(latex,display){
  const idx=_MATH_PLACEHOLDERS.length;_MATH_PLACEHOLDERS.push({latex,display});
  return `MATH${idx}`;
}
function _restoreMath(html){
  return html.replace(/MATH(\d+)/g,(_,i)=>{
    const e=_MATH_PLACEHOLDERS[+i];if(!e)return '';
    if(window._katexReady&&window.katex){
      try{return window.katex.renderToString(e.latex,{displayMode:e.display,throwOnError:false,output:'html'})}catch(err){return `<span class="math-fail" title="KaTeX: ${esc(err.message||'render error')}">$${esc(e.latex)}$</span>`}
    }
    const cls=e.display?'math-pending math-display':'math-pending';
    return `<span class="${cls}" data-latex="${esc(e.latex)}" data-display="${e.display?'1':'0'}">${e.display?'$$':'$'}${esc(e.latex)}${e.display?'$$':'$'}</span>`;
  });
}
function _mdCells(l){return l.replace(/^\s*\|/,'').replace(/\|\s*$/,'').split('|').map(function(c){return c.trim()})}
function _mdIsSep(l){if(l.indexOf('-')<0)return false;var c=l.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(function(x){return x.trim()});return c.length>=2&&c.every(function(x){return /^:?-{1,}:?$/.test(x)})}
function _mdTables(s){var lines=s.split('\n'),out=[],i=0,inPre=false;while(i<lines.length){var ln=lines[i];if(inPre){out.push(ln);if(ln.indexOf('</pre>')>=0)inPre=false;i++;continue}if(ln.indexOf('<pre')>=0&&ln.indexOf('</pre>')<0){inPre=true;out.push(ln);i++;continue}if(i+1<lines.length&&ln.indexOf('|')>=0&&_mdIsSep(lines[i+1])){var head=_mdCells(ln),al=_mdCells(lines[i+1]).map(function(c){var L=c.charAt(0)===':',R=c.charAt(c.length-1)===':';return L&&R?'center':R?'right':L?'left':''});i+=2;var rows=[];while(i<lines.length&&lines[i].indexOf('|')>=0&&lines[i].trim()!==''&&!_mdIsSep(lines[i])){rows.push(_mdCells(lines[i]));i++}var t='<table class="md-table"><thead><tr>'+head.map(function(h,j){return '<th'+(al[j]?' style="text-align:'+al[j]+'"':'')+'>'+h+'</th>'}).join('')+'</tr></thead><tbody>'+rows.map(function(r){return '<tr>'+r.map(function(c,j){return '<td'+(al[j]?' style="text-align:'+al[j]+'"':'')+'>'+(c||'')+'</td>'}).join('')+'</tr>'}).join('')+'</tbody></table>';out.push(t)}else{out.push(ln);i++}}return out.join('\n')}
function md(src){
  src=String(src==null?'':src);
  src=src.replace(/\\\[([\s\S]+?)\\\]/g,(_,l)=>_stashMath(l.trim(),true));
  src=src.replace(/\\\(([\s\S]+?)\\\)/g,(_,l)=>_stashMath(l.trim(),false));
  src=src.replace(/\$\$([^$\n][^$]*?)\$\$/g,(_,l)=>_stashMath(l.trim(),true));
  src=src.replace(/\$([^$\n][^$\n]*?)\$/g,(_,l)=>{const t=l.trim();if(/^\s*$/.test(t)||!/[a-zA-Z\\^_{}\d]/.test(t))return '$'+l+'$';if(/\s/.test(t)&&!/[\\^_{}]/.test(t))return '$'+l+'$';return _stashMath(t,false)});
  src=esc(src);
  src=src.replace(/```([\w-]*)\n([\s\S]*?)```/g,(_,l,c)=>{
    if((l||'').toLowerCase()==='mermaid'){const id='mm_'+Math.random().toString(36).slice(2,10);return `<div class="mermaid-pending" id="${id}" data-src="${c.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}">${c.replace(/\n$/,'')}</div>`}
    const lang=(l||'').toLowerCase().trim();
    const cls=lang?` class="language-${lang}"`:'';
    return `<pre${cls}><code${cls}>${c.replace(/\n$/,'')}</code></pre>`;
  });
  src=src.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  src=src.replace(/\[([^\]]+)\]\(((?:https?:\/\/|\/)[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');
  src=src.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)(?=[\s.,!?)]|$)/g,'$1<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$2</a>');
  src=src.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  src=src.replace(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/gm,'<strong>$1</strong>').replace(/^\s{0,3}#{1,6}\s*$/gm,'');
  src=_mdTables(src);
  src=src.replace(/\n\n+/g,'<br><br>').replace(/\n/g,'<br>');
  src=src.replace(/(<\/table>)<br><br>/g,'$1').replace(/(<\/table>)<br>/g,'$1').replace(/<br><br>(<table)/g,'$1').replace(/<br>(<table)/g,'$1');
  src=_restoreMath(src);
  return src;
}
function _rerenderPendingMath(){
  if(!(window._katexReady&&window.katex))return;
  const nodes=document.querySelectorAll('.math-pending');
  nodes.forEach(el=>{
    const latex=el.getAttribute('data-latex')||'';const display=el.getAttribute('data-display')==='1';
    try{el.outerHTML=window.katex.renderToString(latex,{displayMode:display,throwOnError:false,output:'html'})}catch(err){el.outerHTML=`<span class="math-fail">${display?'$$':'$'}${esc(latex)}${display?'$$':'$'}</span>`}
  });
}
(function(){
  const wait=setInterval(()=>{if(window._katexReady&&window.katex){clearInterval(wait);_rerenderPendingMath()}},120);
  setTimeout(()=>clearInterval(wait),15000);
})();
async function _rerenderPendingMermaid(){
  if(!(window._mermaidReady&&window._mermaid))return;
  const nodes=document.querySelectorAll('.mermaid-pending:not(.mermaid-rendered)');
  for(const el of nodes){
    el.classList.add('mermaid-rendered');
    const src=(el.getAttribute('data-src')||'').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const id='svg-'+Math.random().toString(36).slice(2,10);
    try{const {svg}=await window._mermaid.render(id,src);el.innerHTML=svg;el.classList.add('mermaid-ok')}
    catch(err){el.innerHTML=`<div class="mermaid-fail">⚠ mermaid render error: ${esc(err.message||err)}</div><pre><code>${esc(src)}</code></pre>`}
  }
}
(function(){
  const wait=setInterval(()=>{if(window._mermaidReady&&window._mermaid){clearInterval(wait);_rerenderPendingMermaid()}},150);
  setTimeout(()=>clearInterval(wait),18000);
})();
function _rehighlightCode(root){
  if(!(window._prismReady&&window.Prism))return;
  try{const scope=root||document;const nodes=scope.querySelectorAll('pre code[class*="language-"]:not([data-prism-done])');nodes.forEach(el=>{try{window.Prism.highlightElement(el);el.setAttribute('data-prism-done','1')}catch(_){}})}
  catch(_){}
}
(function(){
  const wait=setInterval(()=>{if(window._prismReady&&window.Prism){clearInterval(wait);_rehighlightCode()}},150);
  setTimeout(()=>clearInterval(wait),18000);
})();
const _origBubble=bubble;
window.bubble=function(role,text,meta){const r=_origBubble(role,text,meta);if(role==='bot'){setTimeout(_rerenderPendingMermaid,50);setTimeout(_rehighlightCode,50)}return r};
function bubble(role,text,meta){
  const w=document.querySelector('.welcome');if(w)w.remove();
  const m=document.createElement('div');m.className='msg '+role;
  const b=document.createElement('div');b.className='bubble';
  if(role==='bot')b.innerHTML=md(text||'');else b.textContent=text||'';
  m.appendChild(b);
  if(meta){const mt=document.createElement('div');mt.className='meta';mt.innerHTML=meta;m.appendChild(mt)}
  if(role==='bot'){
    const retry=document.createElement('button');retry.className='msg-retry';retry.textContent='↻ retry';retry.title='Discard this reply and edit the prompt';retry.onclick=ev=>{ev.stopPropagation();_retryBubble(m)};
    m.appendChild(retry);
    const star=document.createElement('button');star.className='msg-star';star.textContent='☆';star.title='Bookmark this reply';star.onclick=ev=>{ev.stopPropagation();_bookmarkBubble(m,star)};
    m.appendChild(star);
  }
  log.appendChild(m);log.scrollTop=log.scrollHeight;
  return {msg:m,bubble:b};
}
async function _bookmarkBubble(msg,btn){
  if(!msg)return;
  const botText=(msg.querySelector('.bubble')||{}).textContent||'';
  let prev=msg.previousElementSibling;while(prev&&!prev.classList.contains('msg'))prev=prev.previousElementSibling;
  const userText=(prev&&prev.classList.contains('user'))?((prev.querySelector('.bubble')||{}).textContent||''):'';
  const tierBadge=(msg.querySelector('.meta .badge')||{}).textContent||'';
  const personaBadge=(msg.querySelector('.meta .badge.persona')||{}).textContent||'';
  const note=prompt('Optional note for this bookmark (Cancel to skip):','');
  if(note===null)return;
  try{
    const r=await fetch('/memory/bookmarks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid||'',user_msg:userText,bot_msg:botText,note:note,tier:tierBadge,persona:personaBadge})});
    if(!r.ok){bubble('bot','Bookmark failed: HTTP '+r.status,'<span class="badge err">bookmark</span>');return}
    if(btn){btn.textContent='★';btn.classList.add('starred');btn.title='Bookmarked'}
    bubble('bot','Bookmarked. List them anytime with `/bookmarks`.','<span class="badge">bookmark</span>');
  }catch(e){bubble('bot','Bookmark failed: '+esc(String(e)),'<span class="badge err">bookmark</span>')}
}
let _bookmarksPanelOpen=false;let _bookmarksData=[];
function toggleBookmarksPanel(force){
  const open=(typeof force==='boolean')?force:!_bookmarksPanelOpen;
  _bookmarksPanelOpen=open;
  const el=document.getElementById('bookmarks-panel');if(el)el.classList.toggle('show',open);
  if(open)_bookmarksLoad();
}
async function _bookmarksLoad(){
  try{
    const r=await fetch('/memory/bookmarks?limit=200');if(!r.ok){_bookmarksData=[];_bookmarksPanelRender();return}
    const j=await r.json();_bookmarksData=j.bookmarks||[];_bookmarksPanelRender();
  }catch{_bookmarksData=[];_bookmarksPanelRender()}
}
function _bookmarksPanelRender(){
  const list=document.getElementById('bm-list');if(!list)return;
  const q=((document.getElementById('bm-search')||{}).value||'').trim().toLowerCase();
  const filtered=q?_bookmarksData.filter(bm=>((bm.user_msg||'')+' '+(bm.bot_msg||'')+' '+(bm.note||'')).toLowerCase().includes(q)):_bookmarksData;
  if(filtered.length===0){list.innerHTML='<div style="font-size:10px;color:var(--mute);text-align:center;padding:14px;font-style:italic">'+(q?'no matches':'no bookmarks yet — click ☆ on any reply')+'</div>';return}
  list.innerHTML=filtered.map(bm=>{
    const id=esc(bm.id||'');const ts=esc(bm.iso||'?');const u=esc((bm.user_msg||'').slice(0,160));const a=esc((bm.bot_msg||'').slice(0,500));const n=bm.note?esc(bm.note):'';const p=bm.persona?esc(bm.persona):'';
    return `<div class="bm-row" data-id="${id}"><div class="bm-head"><span class="bm-ts">★ ${ts}</span>${p?'<span class="bm-pers">'+p+'</span>':''}<button class="bm-del" onclick="event.stopPropagation();_bookmarksDelete('${id}')" title="Delete bookmark">✕</button></div><div class="bm-q">"${u}"</div><div class="bm-a">${a}</div>${n?'<div class="bm-note">note: '+n+'</div>':''}</div>`;
  }).join('');
}
async function _bookmarksDelete(id){
  if(!id)return;
  if(!confirm('Delete this bookmark?'))return;
  try{
    const r=await fetch('/memory/bookmarks/'+encodeURIComponent(id),{method:'DELETE'});
    if(!r.ok){bubble('bot','Delete failed: HTTP '+r.status,'<span class="badge err">bookmark</span>');return}
    _bookmarksData=_bookmarksData.filter(b=>b.id!==id);
    _bookmarksPanelRender();
  }catch(e){bubble('bot','Delete failed: '+esc(String(e)),'<span class="badge err">bookmark</span>')}
}
function _bookmarksShow(){toggleBookmarksPanel(true)}
let _remindersPanelOpen=false;let _remindersData=[];let _remindersDueIds=new Set();
function toggleRemindersPanel(force){
  const open=(typeof force==='boolean')?force:!_remindersPanelOpen;
  _remindersPanelOpen=open;
  const el=document.getElementById('reminders-panel');if(el)el.classList.toggle('show',open);
  if(open)_remindersLoad();
}
async function _remindersLoad(){
  try{
    const r=await fetch('/memory/reminders?limit=200');if(!r.ok){_remindersData=[];_remindersPanelRender();return}
    const j=await r.json();_remindersData=j.reminders||[];_remindersDueIds=new Set((j.due||[]).map(x=>x.id));_remindersPanelRender();
  }catch{_remindersData=[];_remindersPanelRender()}
}
function _remindersPanelRender(){
  const list=document.getElementById('rm-list');if(!list)return;
  if(_remindersData.length===0){list.innerHTML='<div style="font-size:10px;color:var(--mute);text-align:center;padding:14px;font-style:italic">no active reminders — say "remind me to..." or use the input above</div>';return}
  list.innerHTML=_remindersData.map(r=>{
    const id=esc(r.id||'');const t=esc(r.text||'');const due=r.due_iso?esc(r.due_iso):'';const isDue=_remindersDueIds.has(r.id);
    return `<div class="rm-row${isDue?' rm-due':''}" data-id="${id}"><div class="rm-head"><span class="rm-icon">${isDue?'!':'·'}</span><span class="rm-due-iso">${due||'no due'}</span><button class="rm-del" onclick="event.stopPropagation();_remindersDismiss('${id}')" title="Dismiss reminder">✓</button></div><div class="rm-text">${t}</div></div>`;
  }).join('');
}
async function _remindersDismiss(id){
  if(!id)return;
  try{
    const r=await fetch('/memory/reminders/'+encodeURIComponent(id)+'/dismiss',{method:'POST'});
    if(!r.ok){bubble('bot','Dismiss failed: HTTP '+r.status,'<span class="badge err">reminder</span>');return}
    _remindersData=_remindersData.filter(x=>x.id!==id);_remindersDueIds.delete(id);
    _remindersPanelRender();_refreshRemindersBadge();
  }catch(e){bubble('bot','Dismiss failed: '+esc(String(e)),'<span class="badge err">reminder</span>')}
}
async function _remindersAddFromPanel(){
  const inp=document.getElementById('rmp-new');if(!inp)return;
  const text=(inp.value||'').trim();if(!text)return;
  try{
    const r=await fetch('/memory/reminders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text,session_id:sid||''})});
    const j=await r.json();
    if(!r.ok||j.error){bubble('bot','Add failed: '+esc(j.error||r.status),'<span class="badge err">reminder</span>');return}
    inp.value='';_remindersLoad();_refreshRemindersBadge();
  }catch(e){bubble('bot','Add failed: '+esc(String(e)),'<span class="badge err">reminder</span>')}
}
const _REM_NOTIFIED_KEY='amni_jarvis_rem_notified';
let _remNotifiedIds=(()=>{try{const a=JSON.parse(localStorage.getItem(_REM_NOTIFIED_KEY)||'[]');return new Set(Array.isArray(a)?a:[])}catch{return new Set()}})();
function _remPersistNotified(){try{localStorage.setItem(_REM_NOTIFIED_KEY,JSON.stringify(Array.from(_remNotifiedIds).slice(-200)))}catch{}}
async function _refreshRemindersBadge(){
  try{
    const r=await fetch('/memory/reminders?limit=50');if(!r.ok)return;
    const j=await r.json();const stats=j.stats||{};const active=stats.active||0;const dueNow=stats.due_now||0;const due=j.due||[];
    const txt=document.getElementById('rm-text');const led=document.getElementById('rm-led');
    if(txt)txt.textContent=active===0?'reminders ⏰':`reminders · ${active}${dueNow>0?' · '+dueNow+' due':''}`;
    if(led)led.className=dueNow>0?'ld-led error':(active>0?'ld-led active':'ld-led idle');
    for(const r of due){
      if(!r||!r.id||_remNotifiedIds.has(r.id))continue;
      _remNotifiedIds.add(r.id);_remPersistNotified();
      _remNotifyDue(r);
    }
  }catch{}
}
function _remNotifyDue(r){
  if(!r)return;
  const text=r.text||'(no text)';const id=r.id||'';
  bubble('bot',`⏰ **Reminder due:** _${esc(text)}_  \n[dismiss](#) — say or click ✓ in the reminders panel.`,'<span class="badge" style="background:rgba(255,91,91,.15);border-color:rgba(255,91,91,.4);color:#ff7b7b">⏰ DUE</span>');
  if(voiceOut&&typeof speak==='function')try{speak('Reminder: '+text)}catch{}
  if('Notification' in window){
    try{
      if(Notification.permission==='granted'){new Notification('Adam reminder',{body:text,tag:'amni-rem-'+id})}
      else if(Notification.permission==='default'){Notification.requestPermission().then(p=>{if(p==='granted')new Notification('Adam reminder',{body:text,tag:'amni-rem-'+id})})}
    }catch{}
  }
}
setInterval(_refreshRemindersBadge,30000);_refreshRemindersBadge();
function _retryBubble(botMsg){
  if(!botMsg)return;
  let prev=botMsg.previousElementSibling;
  while(prev&&!prev.classList.contains('msg'))prev=prev.previousElementSibling;
  if(prev&&prev.classList.contains('user')){
    const userText=(prev.querySelector('.bubble')||{}).textContent||'';
    if(userText){input.value=userText;input.focus();if(input.style)input.style.height=Math.min(160,input.scrollHeight)+'px'}
    prev.remove();
  }
  botMsg.remove();
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
  }else if(t==='news'){
    const items=(d.items||[]).map(it=>`<a class="news-item" href="${esc(it.url||'#')}" target="_blank" rel="noopener"><div class="title">${esc(it.title||'(no title)')}</div><div class="src">${esc(it.source||'')}</div></a>`).join('');
    body=`<div class="w-body">${items||'<div style="color:var(--mute);font-size:11px">no items</div>'}</div>`;
  }else if(t==='stock'){
    const cards=(d.quotes||[]).map(q=>{const chg=q.change||0;const pct=q.change_pct||0;const cls=chg>=0?'up':'down';const arrow=chg>=0?'▲':'▼';const cur=q.currency||'USD';return `<div class="quote"><div class="sym">${esc(q.symbol||'?')}</div><div class="name">${esc(q.name||'')}</div><div class="price">${(q.price!=null?Number(q.price).toFixed(2):'?')} <span class="meta" style="display:inline;padding:0;border:none;margin-left:4px">${esc(cur)}</span></div><div class="chg ${cls}">${arrow} ${(Math.abs(chg)).toFixed(2)} (${pct>=0?'+':''}${pct.toFixed(2)}%)</div><div class="meta">H ${q.day_high!=null?Number(q.day_high).toFixed(2):'?'} · L ${q.day_low!=null?Number(q.day_low).toFixed(2):'?'} · ${esc(q.market_state||'?')}</div></div>`}).join('');
    body=`<div class="quotes">${cards||'<div style="color:var(--mute);font-size:11px">no quotes</div>'}</div>`;
  }else if(t==='file'){
    body=`<div class="file-meta"><div>PATH<span class="v">${esc((d.path||'').split('/').pop().split('\\\\').pop())}</span></div><div>LINES<span class="v">${d.lines_shown||'?'}</span></div><div>SIZE<span class="v">${d.size_bytes!=null?Math.round(d.size_bytes/1024)+' kb':'?'}</span></div>${d.ext?`<div>EXT<span class="v">${esc(d.ext)}</span></div>`:''}</div><pre>${esc(d.preview||'')}</pre>`;
  }else if(t==='disk'){
    const parts=(d.partitions||[]).map(p=>`<div class="part"><div class="mount">${esc(p.mount||'?')}</div><div class="stat">USED<span class="v">${p.used_gb||'?'} / ${p.total_gb||'?'} GB · ${p.used_pct||0}%</span></div><div class="stat">FREE<span class="v">${p.free_gb||'?'} GB</span></div><div class="bar"><div class="bar-fill" style="width:${p.used_pct||0}%"></div></div></div>`).join('');
    body=`<div class="partitions">${parts||'<div style="color:var(--mute);font-size:11px">no partitions</div>'}</div>`;
  }else if(t==='watch'){
    if(d.events && d.events.length){
      const rows=(d.events||[]).slice().reverse().map(ev=>{const k=esc(ev.kind||'?');const sz=ev.size!=null?(ev.size<1024?ev.size+'b':Math.round(ev.size/1024)+'k'):'';return `<div class="event ${k}"><div class="kind">${k}</div><div class="path">${esc(ev.path||'')}</div><div class="size">${sz}</div></div>`}).join('');
      body=`<div class="watch-head"><span>EVENTS</span><span class="id">${esc(d.watch_id||'').slice(-8)}</span></div><div class="events">${rows}</div>`;
    }else if(d.n_watches!=null){
      body=`<div class="watch-head"><span>WATCHERS</span><span class="id">${d.n_watches} active</span></div><div style="font-size:10px;color:var(--mute);letter-spacing:.15em;padding:4px">${d.enabled||0} enabled · ${d.total_recent_events||0} recent events · ${d.auto_fires||0} on_change fires</div>`;
    }else{
      body=`<div style="font-size:10px;color:var(--mute);text-align:center;padding:8px">no events yet</div>`;
    }
  }else if(t==='git'){
    const branch=esc(d.branch||'?');
    const dcls=d.dirty_n>0?' dirty':'';
    const commits=(d.recent_commits||[]).map(c=>{const sha=esc((c||'').slice(0,7));const msg=esc((c||'').slice(8));return `<div class="row"><span class="sha">${sha}</span>${msg}</div>`}).join('');
    const dirtyHtml=d.dirty_n>0&&d.dirty_sample?`<div class="dirty-files"><div class="lbl">UNSTAGED · ${d.dirty_n}</div>${(d.dirty_sample||[]).map(s=>esc(s)).join('<br>')}</div>`:'';
    body=`<div class="git-branch">⎇ ${branch}</div><div class="git-stats"><div>DIRTY<span class="v${dcls}">${d.dirty_n||0}</span></div><div>AHEAD<span class="v">${d.ahead||0}</span></div><div>BEHIND<span class="v">${d.behind||0}</span></div></div><div class="commits">${commits||'<div style="color:var(--mute)">no commits</div>'}</div>${dirtyHtml}`;
  }else if(t==='code'){
    body=`<pre><code>${esc(d.code||'')}</code></pre>`;
  }else if(t==='file_change'){
    const op=esc(d.op||'edit');const path=esc(d.path||'?');const ext=esc(d.ext||'');const la=d.lines_added||0;const lr=d.lines_removed||0;const repl=d.replacements;
    const bn=path.split(/[\\/]/).pop();const folder=path.length>bn.length?path.slice(0,path.length-bn.length-1):'';
    const opCls='op-'+op;
    const vstat=esc(d.verification_status||'manual');const vIssues=d.verification_issues||[];const vChecks=d.verification_checks||[];const vRsn=esc(d.verification_reason||'');const suggested=d.suggested_tests||[];
    const vBadge=vstat==='pass'?`<span class="fc-verify pass" title="${vChecks.join(', ')} all passed">✓ VERIFIED</span>`:vstat==='fail'?`<span class="fc-verify fail" title="${esc(vIssues.join('; '))}">✗ FAILED</span>`:`<span class="fc-verify manual" title="${vRsn||'manual verification required'}">⚠ MANUAL</span>`;
    const issueList=vIssues.length?`<div class="fc-issues">${vIssues.map(i=>`<div class="fc-issue">${esc(i)}</div>`).join('')}</div>`:'';
    const tr=d.test_run||{};
    let testRunBlock='';
    if(tr.ran){
      const cls=tr.ok?'pass':(tr.timeout?'timeout':'fail');
      const head=tr.ok?`✓ TESTS PASSED · ${tr.passed||0}p / ${tr.skipped||0}s in ${tr.duration_s||'?'}s`:(tr.timeout?`⏱ TIMEOUT · ${tr.duration_s||'?'}s cap`:`✗ TESTS FAILED · ${tr.failed||0}f / ${tr.passed||0}p`);
      const fails=Array.isArray(tr.failures)?tr.failures:[];
      const failList=fails.length?`<div class="fc-test-fails">${fails.map(f=>`<div class="fc-test-fail"><span class="t">${esc(f.test||'?')}</span><span class="m">${esc(f.msg||'')}</span></div>`).join('')}</div>`:'';
      testRunBlock=`<div class="fc-test-run ${cls}"><div class="fc-test-head">${head}</div>${failList}</div>`;
    }
    const suggList=(suggested.length && !tr.ran)?`<div class="fc-suggested">recommended test: <code>${esc(suggested[0])}</code></div>`:'';
    const diffStr=d.diff_unified||'';const beforeStr=d.before_preview||'';const afterStr=d.preview||'';
    const hasDiff=!!diffStr.trim()&&op!=='create';const hasBefore=!!beforeStr.trim()&&op!=='create';
    const wid='fc_'+Math.random().toString(36).slice(2,10);
    const toggle=hasDiff?`<div class="fc-viewtoggle" data-fcid="${wid}"><button class="on" data-v="diff">DIFF</button>${hasBefore?'<button data-v="before">BEFORE</button>':''}<button data-v="after">AFTER</button></div>`:'';
    const diffHtml=hasDiff?`<div class="fc-diff" data-view="diff" data-fcid="${wid}">${_renderDiffLines(diffStr)}</div>`:'';
    const beforeHtml=hasBefore?`<pre class="fc-preview" data-view="before" data-fcid="${wid}" style="display:none">${esc(beforeStr)}</pre>`:'';
    const afterHtml=afterStr?`<pre class="fc-preview" data-view="after" data-fcid="${wid}" style="display:${hasDiff?'none':'block'}">${esc(afterStr)}</pre>`:'';
    body=`<div class="fc-head"><span class="fc-op ${opCls}">${op.toUpperCase()}</span><span class="fc-bn">${bn}</span>${ext?`<span class="fc-ext">.${ext}</span>`:''}${vBadge}</div>${folder?`<div class="fc-folder">${folder}</div>`:''}<div class="fc-stats"><span class="fc-add">+${la}</span><span class="fc-rem">-${lr}</span>${repl!=null?`<span class="fc-repl">${repl} replacement${repl===1?'':'s'}</span>`:''}<span class="fc-size">${d.lines_after||0} lines · ${d.bytes_after!=null?(d.bytes_after<1024?d.bytes_after+'b':Math.round(d.bytes_after/1024)+'kb'):'?'}</span></div>${issueList}${testRunBlock}${suggList}${toggle}${diffHtml}${beforeHtml}${afterHtml}<div class="fc-actions"><button class="fc-btn" onclick="_fcOpen('${esc(d.path||'').replace(/'/g,"\\\\'")}')">OPEN</button><button class="fc-btn" onclick="_fcCopyPath('${esc(d.path||'').replace(/'/g,"\\\\'")}')">COPY PATH</button>${vstat==='manual'?`<button class="fc-btn" onclick="_fcMarkTested('${esc(d.path||'').replace(/'/g,"\\\\'")}')">MARK TESTED</button>`:''}</div>`;
  }else if(t==='pc_confirm'){
    const tok=esc(d.token||'');const risk=esc(d.risk||'?');const act=esc(d.action||'?');const tgt=esc(d.target||'');
    body=`<div class="pcw-risk pcw-${risk}">${risk.toUpperCase()} RISK · ${act}</div><div class="pcw-target">${tgt}</div><div class="pcw-note">Nothing runs until you confirm. Adam never executes destructive commands.</div><div class="pcw-actions"><button class="pcw-btn confirm" data-tok="${tok}" onclick="_pcConfirm('${tok}')">✓ CONFIRM</button><button class="pcw-btn cancel" data-tok="${tok}" onclick="_pcCancel('${tok}')">✕ CANCEL</button></div>`;
  }else if(t==='skill_error'){
    const sk=esc(d.skill||'?');const err=esc(d.error||'unknown error');const msg=esc(d.message||'');
    const argsStr=esc(JSON.stringify(d.args||{}).slice(0,200));
    body=`<div class="se-head"><span class="se-skill">${sk}</span><span class="se-status">FAILED</span></div><div class="se-msg">"${msg}"</div><div class="se-err">${err}</div><div class="se-args">args: ${argsStr}</div><div class="se-actions"><button class="se-btn" onclick="_skillErrorRetry('${esc(d.message||'').replace(/'/g,"\\\\'")}')">↻ RETRY</button><button class="se-btn" onclick="_skillFailuresShow()">VIEW LOG</button></div>`;
  }else if(t==='error'||t==='info'){
    body=esc(d.message||'');
  }else{
    body='<pre>'+esc(JSON.stringify(d,null,2)).slice(0,800)+'</pre>';
  }
  el.innerHTML=head+'<div class="w-body">'+body+'</div>';
  return el;
}
async function _pcAction(token,act){
  const card=document.querySelector('.widget.pc_confirm .pcw-btn[data-tok="'+token+'"]');
  const widget=card?card.closest('.widget.pc_confirm'):null;
  try{
    const r=await fetch('/skills/pc_action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{action:act,token:token}})});
    const j=await r.json();const out=(j&&j.output)?j.output:j;
    if(widget)widget.classList.add('resolved');
    if(act==='confirm'){
      if(out&&out.executed){const res=JSON.stringify(out.result||{}).slice(0,500);bubble('bot','✅ Ran '+esc(out.action||'')+': `'+esc((out.target||'').slice(0,80))+'`\n```\n'+esc(res)+'\n```','<span class="badge">pc</span>')}
      else{bubble('bot','PC action did not run: '+esc((out&&out.error)||('HTTP '+r.status)),'<span class="badge err">pc</span>')}
    }else{bubble('bot','Cancelled the PC action.','<span class="badge">pc</span>')}
  }catch(e){bubble('bot','PC action request failed: '+esc(e.message),'<span class="badge err">pc</span>')}
}
function _pcConfirm(token){_pcAction(token,'confirm')}
function _pcCancel(token){_pcAction(token,'cancel')}
function appendWidgets(msgEl,widgets){
  if(!widgets||!widgets.length)return;
  const wrap=document.createElement('div');wrap.className='widgets';
  for(const w of widgets){wrap.appendChild(renderWidget(w))}
  msgEl.appendChild(wrap);log.scrollTop=log.scrollHeight;
}
function quick(t){input.value=t;send()}
function _qcAsk(t){input.value=t;send()}
function _briefingHumanCount(n,sing,plur){return (n||0)+' '+((n||0)===1?sing:(plur||sing+'s'))}
async function _qcBriefing(){
  try{
    const r=await fetch('/memory/digest?hours=24');if(!r.ok){bubble('bot','Briefing unavailable: '+r.status,'<span class="badge err">briefing</span>');return}
    const j=await r.json();
    const learn=j.learning||{};const shell=j.shell||{};const ver=j.verifier||{};const coach=j.coach||{};const skills=j.skills||{};
    const topics=(learn.topics_today||[]).slice(0,4).filter(Boolean);
    const topicsHtml=topics.length?topics.map(t=>`<span class="b-tag">${esc(t)}</span>`).join(''):'<span class="b-mute">none</span>';
    const topSkills=(skills.top||[]).slice(0,4);
    const skillsHtml=topSkills.length?topSkills.map(s=>`<span class="b-tag" title="${s.n} calls · ${(s.ok_rate*100).toFixed(0)}% ok">${esc(s.name)} <span style="color:var(--mute);font-size:8px">${s.n}× · ${s.avg_ms}ms</span></span>`).join(''):'<span class="b-mute">no skill calls yet</span>';
    const lines=[
      `<div class="b-section"><span class="b-lbl">LEARNING</span>${learn.facts_today||0} new facts · ${(learn.topics_today||[]).length} topic${(learn.topics_today||[]).length===1?'':'s'} <div class="b-tags">${topicsHtml}</div></div>`,
      `<div class="b-section"><span class="b-lbl">SHELL</span>${shell.runs_today||0} run${(shell.runs_today||0)===1?'':'s'} · ${shell.errors_today||0} error${(shell.errors_today||0)===1?'':'s'}</div>`,
      `<div class="b-section"><span class="b-lbl">EDITS</span>${ver.pass_today||0} verified · ${ver.fail_today||0} failed · ${ver.pending||0} pending review</div>`,
      `<div class="b-section"><span class="b-lbl">SKILLS</span>${skills.n_calls||0} call${(skills.n_calls||0)===1?'':'s'} · avg ${skills.avg_ms||0}ms <div class="b-tags">${skillsHtml}</div></div>`,
      `<div class="b-section"><span class="b-lbl">COACH</span>${(coach.streak_days||0)>=14?'⚡':(coach.streak_days||0)>=3?'🔥':'·'} ${coach.streak_days||0} day streak${coach.today_active?' · today ✓':''} · ${coach.topics||0} topic${(coach.topics||0)===1?'':'s'} practiced</div>`
    ];
    const head='<div class="b-head"><span class="b-icon">◈</span>24-HOUR BRIEFING</div>';
    const html='<div class="briefing">'+head+lines.join('')+'</div>';
    const m=document.createElement('div');m.className='msg bot';
    const b=document.createElement('div');b.className='bubble';b.innerHTML=html;
    m.appendChild(b);log.appendChild(m);log.scrollTop=log.scrollHeight;
  }catch(e){bubble('bot','Briefing error: '+esc(e.message),'<span class="badge err">briefing</span>')}
}
let _spPanelOpen=false,_spItems=[];
function _spHumanAge(ts){if(!ts)return '—';const s=Math.max(0,Date.now()/1000-ts);if(s<60)return Math.round(s)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';if(s<86400)return (s/3600).toFixed(1)+'h ago';if(s<86400*7)return (s/86400).toFixed(1)+'d ago';return Math.round(s/86400)+'d ago'}
async function _pollSessionsList(){
  const list=document.getElementById('sp-list');if(!list)return;
  try{const r=await fetch('/sessions?enrich=true&limit=30');if(!r.ok){list.innerHTML='<div class="sp-empty">sessions endpoint unavailable</div>';return}
    const j=await r.json();_spItems=j.sessions||[];
    const cnt=document.getElementById('sp-count');if(cnt)cnt.textContent=_spItems.length+' session'+(_spItems.length===1?'':'s');
    if(!_spItems.length){list.innerHTML='<div class="sp-empty">no past sessions yet — start chatting!</div>';return}
    list.innerHTML=_spItems.map(s=>{
      const isCur=(s.session_id===sid);const cls='sp-item'+(isCur?' current':'');
      const safeId=esc(s.session_id||'').replace(/'/g,"\\\\'");
      const turns=s.turns_n||0;const first=esc(s.first_msg||'(no messages)');
      return `<div class="${cls}" onclick="_spLoadSession('${safeId}')"><div class="sp-row1"><span class="sp-sid">${esc((s.session_id||'?').slice(-12))}</span><span class="sp-turns">${turns} turn${turns===1?'':'s'}</span><span class="sp-age">${_spHumanAge(s.updated_ts)}</span></div><div class="sp-first">${first}</div><button class="sp-del" onclick="event.stopPropagation();_spDeleteSession('${safeId}')" title="Delete this session">✕</button></div>`
    }).join('');
  }catch(e){list.innerHTML='<div class="sp-empty">'+esc(_netHint(e))+'</div>'}
}
function toggleSessionsPanel(){_spPanelOpen=!_spPanelOpen;const p=document.getElementById('sessions-panel');p.classList.toggle('show',_spPanelOpen);['persona-panel','learn-panel','tests-panel','shell-panel','coach-panel'].forEach(id=>{const el=document.getElementById(id);if(_spPanelOpen&&el&&el.classList.contains('show'))el.classList.remove('show')});if(_spPanelOpen){_personaPanelOpen=false;_ldPanelOpen=false;_tpPanelOpen=false;_shPanelOpen=false;_coachPanelOpen=false;document.getElementById('coach-toggle').classList.remove('on');_pollSessionsList()}}
async function _spLoadSession(targetSid){
  if(!targetSid)return;
  if(targetSid===sid){toggleSessionsPanel();return}
  sid=targetSid;localStorage.setItem(SKEY,sid);
  document.querySelectorAll('#log .msg').forEach(m=>m.remove());
  const banner=document.querySelector('.restore-banner');if(banner)banner.remove();
  await _restoreSession();
  bubble('bot','Switched to session **'+esc(sid.slice(-12))+'**.','<span class="badge">session</span>');
  toggleSessionsPanel();
}
async function _spDeleteSession(targetSid){
  if(!targetSid)return;
  if(!confirm('Delete session '+targetSid.slice(-12)+'? This cannot be undone.'))return;
  try{await fetch('/sessions/'+encodeURIComponent(targetSid),{method:'DELETE'});if(targetSid===sid){sid='';localStorage.removeItem(SKEY)}_pollSessionsList();bubble('bot','Deleted session '+esc(targetSid.slice(-12))+'.','<span class="badge">session</span>')}
  catch(e){bubble('bot','Delete failed: '+esc(e.message),'<span class="badge err">session</span>')}
}
function _spNewSession(){sid='';localStorage.removeItem(SKEY);document.querySelectorAll('#log .msg').forEach(m=>m.remove());const banner=document.querySelector('.restore-banner');if(banner)banner.remove();bubble('bot','Started a fresh session. Anything you say next will create a new one.','<span class="badge">session</span>');toggleSessionsPanel()}
async function _fcOpen(path){
  if(!path)return;
  try{const r=await fetch('/skills/file_read',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{path}})});const j=await r.json();const c=(j.output&&j.output.content)||j.content||(typeof j==='string'?j:'(no content)');bubble('bot','```\n'+c.slice(0,4000)+(c.length>4000?'\n... ('+(c.length-4000)+' more chars)':'')+'\n```','<span class="badge">file</span>')}
  catch(e){bubble('bot','Could not open file: '+esc(e.message),'<span class="badge err">err</span>')}
}
function _fcCopyPath(path){if(!path)return;try{navigator.clipboard.writeText(path);bubble('bot','Copied path to clipboard: `'+esc(path)+'`','<span class="badge">copy</span>')}catch{bubble('bot','Clipboard unavailable. Path: `'+esc(path)+'`','<span class="badge err">err</span>')}}
function _renderDiffLines(diff){
  if(!diff)return '';
  return diff.split('\n').map(line=>{
    if(line.startsWith('@@'))return `<span class="dl hunk">${esc(line)}</span>`;
    if(line.startsWith('+++')||line.startsWith('---'))return `<span class="dl ctx">${esc(line)}</span>`;
    if(line.startsWith('+'))return `<span class="dl add">${esc(line)}</span>`;
    if(line.startsWith('-'))return `<span class="dl rem">${esc(line)}</span>`;
    return `<span class="dl ctx">${esc(line)}</span>`;
  }).join('');
}
document.addEventListener('click',function(ev){
  const btn=ev.target.closest('.fc-viewtoggle button[data-v]');if(!btn)return;
  const wrap=btn.closest('.fc-viewtoggle');if(!wrap)return;
  const fcid=wrap.getAttribute('data-fcid');const v=btn.getAttribute('data-v');
  wrap.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b===btn));
  const widget=wrap.closest('.widget.file_change');if(!widget)return;
  widget.querySelectorAll(`[data-fcid="${fcid}"][data-view]`).forEach(el=>{
    el.style.display=el.getAttribute('data-view')===v?'block':'none';
  });
});
async function _fcMarkTested(path){
  if(!path)return;
  try{const r=await fetch('/memory/needs-testing/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path_substring:path})});const j=await r.json();bubble('bot','Marked '+j.marked_done+' testing item(s) as done for `'+esc(path)+'`','<span class="badge">tested</span>')}
  catch(e){bubble('bot','Could not mark tested: '+esc(e.message),'<span class="badge err">err</span>')}
}
let _streamAbort=null;
const TYPE_SPEED_KEY='amni_jarvis_type_speed';
const SCROLL_THRESHOLD=60;
function _isAtBottom(){return (log.scrollHeight-log.scrollTop-log.clientHeight)<SCROLL_THRESHOLD}
function _smartScroll(){if(_isAtBottom())log.scrollTop=log.scrollHeight;else _updateJumpPill()}
function _updateJumpPill(){
  let pill=document.getElementById('jump-to-bottom');
  if(!pill){pill=document.createElement('div');pill.id='jump-to-bottom';pill.className='jtb-pill';pill.innerHTML='▼ jump to bottom';pill.onclick=()=>{log.scrollTop=log.scrollHeight;pill.classList.remove('show')};document.body.appendChild(pill)}
  const atBottom=_isAtBottom();
  pill.classList.toggle('show',!atBottom);
}
log.addEventListener('scroll',_updateJumpPill,{passive:true});
function _addCopyButtonsTo(root){
  if(!root||typeof root.querySelectorAll!=='function')return;
  const blocks=root.querySelectorAll('pre:not([data-copy-wired])');
  blocks.forEach(pre=>{
    if(!pre.querySelector('code'))return;
    pre.setAttribute('data-copy-wired','1');
    pre.style.position=pre.style.position||'relative';
    const btn=document.createElement('button');
    btn.className='code-copy';btn.type='button';btn.textContent='copy';btn.title='Copy code to clipboard';
    btn.onclick=ev=>{
      ev.stopPropagation();
      const code=pre.querySelector('code');
      const text=code?(code.textContent||''):(pre.textContent||'');
      try{
        navigator.clipboard.writeText(text).then(()=>{btn.textContent='copied!';btn.classList.add('ok');setTimeout(()=>{btn.textContent='copy';btn.classList.remove('ok')},1400)})
        .catch(()=>{btn.textContent='clipboard?';btn.classList.add('err');setTimeout(()=>{btn.textContent='copy';btn.classList.remove('err')},1800)});
      }catch(_){btn.textContent='err';setTimeout(()=>{btn.textContent='copy'},1400)}
    };
    pre.appendChild(btn);
  });
}
const _copyObs=new MutationObserver(muts=>{
  for(const m of muts){
    if(m.type!=='childList')continue;
    m.addedNodes.forEach(n=>{
      if(n.nodeType!==1)return;
      if(n.tagName==='PRE')_addCopyButtonsTo(n.parentElement||n);
      else _addCopyButtonsTo(n);
    });
    if(m.target&&m.target.tagName==='CODE')_addCopyButtonsTo(m.target.parentElement);
  }
});
try{_copyObs.observe(log,{childList:true,subtree:true,characterData:false})}catch(_){}
_addCopyButtonsTo(log);
let _typePending='';let _typeShown=0;let _typeRAF=null;let _typeBot=null;let _typeOnDone=null;
function _personaTypeCps(){
  const cur=(_selectedPersona||personaName||'').toLowerCase();
  if(!cur||!_knownPersonas||_knownPersonas.length===0)return null;
  const p=_knownPersonas.find(x=>typeof x==='object'&&(x.name||'').toLowerCase()===cur);
  if(!p||typeof p!=='object')return null;
  if(cur==='haiku')return 42;
  if(cur==='yoda')return 60;
  const ex=Number(p.excitement||0),fo=Number(p.formality||0),le=Number(p.length||0);
  let cps=85;
  cps+=Math.round((ex-0.4)*60);
  cps-=Math.round(Math.max(0,fo-0.5)*40);
  cps-=Math.round(Math.max(0,le-0.5)*25);
  return Math.max(45,Math.min(140,cps));
}
function _typeSpeedCps(){
  const raw=localStorage.getItem(TYPE_SPEED_KEY);
  if(raw){const v=parseInt(raw,10);if(isFinite(v))return Math.max(10,Math.min(2000,v))}
  const pc=_personaTypeCps();
  if(pc!=null)return pc;
  return 85;
}
function _paceSliderChange(v){
  const cps=parseInt(v,10);if(!isFinite(cps))return;
  localStorage.setItem(TYPE_SPEED_KEY,String(cps));
  _renderPaceUI();
}
function _paceReset(){
  localStorage.removeItem(TYPE_SPEED_KEY);
  _renderPaceUI();
  bubble('bot','Pace reset — now using **'+esc(String(_typeSpeedCps()))+' cps** from persona default.','<span class="badge">pace</span>');
}
function _renderPaceUI(){
  const slider=document.getElementById('pp-pace');const val=document.getElementById('pp-pace-v');const mode=document.getElementById('pp-pace-mode');
  if(!slider)return;
  const override=localStorage.getItem(TYPE_SPEED_KEY);
  const cps=_typeSpeedCps();
  slider.value=String(cps);
  if(val)val.textContent=cps+' cps';
  if(mode)mode.textContent=override?'override · user-set':'auto · from persona';
}
function _typeStart(botRef,onDone){
  _typeBot=botRef;_typePending='';_typeShown=0;_typeOnDone=onDone||null;
  if(_typeRAF)cancelAnimationFrame(_typeRAF);
  _typeLastTs=0;_typeTick();
}
let _typeLastTs=0;
function _typeTick(ts){
  if(!_typeBot){_typeRAF=null;return}
  if(!ts){ts=performance.now()}
  if(!_typeLastTs)_typeLastTs=ts;
  const dt=ts-_typeLastTs;
  const remaining=_typePending.length-_typeShown;
  const base=_typeSpeedCps();
  const speed=remaining>800?Math.max(base*4,400):remaining>200?base*2:base;
  const want=Math.floor(dt*speed/1000);
  if(want>=1){
    _typeShown=Math.min(_typePending.length,_typeShown+want);
    _typeLastTs=ts;
    try{_typeBot.bubble.innerHTML=md(_typePending.slice(0,_typeShown));_smartScroll()}catch(_){}
  }
  if(_typeShown<_typePending.length||_typePending.length===0){
    _typeRAF=requestAnimationFrame(_typeTick);
  }else{
    _typeRAF=null;
    const _tb=_typeBot;const fn=_typeOnDone;_typeOnDone=null;_typeBot=null;
    if(_tb){try{_rerenderPendingMath()}catch(_){}try{_rehighlightCode(_tb.bubble)}catch(_){}try{_rerenderPendingMermaid()}catch(_){}}
    if(typeof fn==='function')try{fn()}catch(_){}
  }
}
function _typePush(chunk){_typePending+=chunk;if(!_typeRAF&&_typeBot)_typeRAF=requestAnimationFrame(_typeTick)}
function _typeFlushAll(){
  if(_typeRAF){cancelAnimationFrame(_typeRAF);_typeRAF=null}
  const _tb=_typeBot;
  if(_tb){try{_tb.bubble.innerHTML=md(_typePending);_smartScroll()}catch(_){}}
  _typeShown=_typePending.length;
  const fn=_typeOnDone;_typeOnDone=null;_typeBot=null;
  if(_tb){try{_rerenderPendingMath()}catch(_){}try{_rehighlightCode(_tb.bubble)}catch(_){}try{_rerenderPendingMermaid()}catch(_){}}
  if(typeof fn==='function')try{fn()}catch(_){}
}
function _setSendButtonState(streaming){
  if(!send_btn)return;
  if(streaming){send_btn.dataset.streaming='1';send_btn.textContent='STOP';send_btn.classList.add('stop-mode');send_btn.disabled=false}
  else{delete send_btn.dataset.streaming;send_btn.textContent='TRANSMIT';send_btn.classList.remove('stop-mode')}
}
function stopStream(){
  if(_streamAbort){try{_streamAbort.abort()}catch{}_streamAbort=null}
  _typeFlushAll();
  _setSendButtonState(false);
}
const _SLASH_COMMANDS=[
  {cmd:'new',hint:'clear chat + start a fresh session'},
  {cmd:'clear',hint:'clear chat (keep session)'},
  {cmd:'help',hint:'open keyboard shortcuts overlay'},
  {cmd:'persona',hint:'switch persona (e.g. /persona rikku)'},
  {cmd:'sessions',hint:'open sessions browser'},
  {cmd:'tools',hint:'toggle the TOOLS drawer'},
  {cmd:'status',hint:'toggle the STATUS panel'},
  {cmd:'handsfree',hint:'engage/disengage hands-free mode'},
  {cmd:'update',hint:'check for + apply git updates'},
  {cmd:'find',hint:'find <query> in workdir'},
  {cmd:'search',hint:'web search anytime (e.g. /search rust async)'},
  {cmd:'pace',hint:'set stream cps (10-2000)'},
  {cmd:'bookmarks',hint:'list recent starred replies'},
  {cmd:'reminders',hint:'open reminders panel'},
  {cmd:'perms',hint:'location/mic/camera/notification permissions'},
  {cmd:'pclog',hint:'PC action audit log (proposed/ran/refused)'},
  {cmd:'se',hint:'software-engineer dashboard (code map + attempts)'},
];
let _slashAcOpen=false;let _slashAcIdx=0;let _slashAcMatches=[];
function _slashAcRender(){
  let el=document.getElementById('slash-ac');
  if(!el){el=document.createElement('div');el.id='slash-ac';el.className='slash-ac';document.body.appendChild(el)}
  if(!_slashAcOpen||_slashAcMatches.length===0){el.classList.remove('show');return}
  const rect=input.getBoundingClientRect();
  el.style.left=rect.left+'px';el.style.bottom=(window.innerHeight-rect.top+6)+'px';el.style.width=Math.max(260,rect.width-180)+'px';
  el.innerHTML=_slashAcMatches.map((c,i)=>`<div class="sl-row${i===_slashAcIdx?' active':''}" data-cmd="${esc(c.cmd)}"><span class="sl-cmd">/${esc(c.cmd)}</span><span class="sl-hint">${esc(c.hint)}</span></div>`).join('');
  el.classList.add('show');
  el.querySelectorAll('.sl-row').forEach((row,i)=>{row.onclick=()=>{_slashAcAccept(_slashAcMatches[i].cmd)};row.onmouseenter=()=>{_slashAcIdx=i;_slashAcRender()}});
}
function _slashAcUpdate(){
  const t=input.value;
  if(!t.startsWith('/')||t.includes(' ')){_slashAcOpen=false;_slashAcRender();return}
  const q=t.slice(1).toLowerCase();
  _slashAcMatches=_SLASH_COMMANDS.filter(c=>c.cmd.startsWith(q));
  _slashAcOpen=_slashAcMatches.length>0;
  if(_slashAcIdx>=_slashAcMatches.length)_slashAcIdx=0;
  _slashAcRender();
}
function _slashAcAccept(cmd){
  const needsArg=(cmd==='persona'||cmd==='find'||cmd==='pace'||cmd==='search');
  input.value='/'+cmd+(needsArg?' ':'');
  _slashAcOpen=false;_slashAcRender();
  input.focus();try{input.setSelectionRange(input.value.length,input.value.length)}catch{}
  if(!needsArg){send()}
}
function _slashAcClose(){_slashAcOpen=false;_slashAcRender()}
input.addEventListener('input',_slashAcUpdate);
function cycleTheme(){const ks=Object.keys(THEMES);const cur=localStorage.getItem(THEME_KEY)||'jarvis';const nx=ks[(ks.indexOf(cur)+1)%ks.length];applyTheme(nx);try{_showToast({id:'theme_'+nx,level:'success',source:'THEME',title:THEMES[nx].label+' theme',age_s:0})}catch(_){}}
let _cmdMenuOpen=false;
function toggleCmdMenu(force){_cmdMenuOpen=(typeof force==='boolean')?force:!_cmdMenuOpen;const el=document.getElementById('cmd-menu');if(!el)return;if(_cmdMenuOpen)_renderCmdMenu();el.classList.toggle('show',_cmdMenuOpen)}
function _menuRun(cmd){toggleCmdMenu(false);_slashAcAccept(cmd)}
async function _openPeer(){try{const r=await fetch('/launch/peer',{method:'POST'});const j=await r.json();const url=(j&&j.url)||'http://localhost:3000';if(j&&j.ok){try{_showToast({id:'peer',level:'success',source:'LAUNCH',title:'Amni-Code launching…',body:'standalone window opening — '+((j&&j.launched)||url),age_s:0})}catch(_){}}else{try{_showToast({id:'peer',level:'warn',source:'LAUNCH',title:'Amni-Code standalone not found',body:((j&&j.error)||'opening in a browser tab instead'),age_s:0})}catch(_){}window.open(url,'_blank')}}catch(e){window.open('http://localhost:3000','_blank')}}
async function _checkUpdate(){
  try{_showToast({id:'upd',level:'info',source:'UPDATE',title:'Checking for updates…',body:'git fetch origin',age_s:0})}catch(_){}
  let j;try{j=await(await fetch('/update/check')).json()}catch(e){try{_showToast({id:'upd',level:'warn',source:'UPDATE',title:'Update check failed',body:String(e&&e.message||e),age_s:0})}catch(_){}return}
  if(!j||!j.ok){bubble('bot','Couldn\'t check for updates: '+esc((j&&j.error)||'unknown')+'. (Adam must be running from a git checkout with a remote.)','<span class="badge err">update</span>');return}
  if(!j.update_available){bubble('bot','✅ **Adam is up to date** — on `'+esc(j.branch)+'` at `'+esc(j.current)+'`'+(j.ahead?(' · '+j.ahead+' local commit'+(j.ahead===1?'':'s')+' ahead of origin'):'')+'.','<span class="badge">update</span>');try{_showToast({id:'upd',level:'success',source:'UPDATE',title:'Up to date',body:j.branch+' @ '+j.current,age_s:0})}catch(_){}return}
  const inc=(j.incoming||[]).slice(0,5).map(esc).join('\n');
  bubble('bot','**Update available** — '+j.behind+' new commit'+(j.behind===1?'':'s')+' on `'+esc(j.branch)+'`.'+(j.dirty?' ⚠ You have **uncommitted local changes** — commit or stash them before updating.':'')+(inc?'\n\n```\n'+inc+'\n```':''),'<span class="badge">update</span>');
  if(j.dirty)return;
  if(!confirm('Pull '+j.behind+' update'+(j.behind===1?'':'s')+' on "'+j.branch+'"?\n\nApplies now; Adam needs a restart afterward to load it.'))return;
  try{_showToast({id:'upd',level:'info',source:'UPDATE',title:'Updating…',body:'git pull --ff-only',age_s:0})}catch(_){}
  let a;try{a=await(await fetch('/update/apply',{method:'POST'})).json()}catch(e){bubble('bot','Update failed to run: '+esc(String(e&&e.message||e)),'<span class="badge err">update</span>');return}
  if(a&&a.ok){bubble('bot','✅ **Updated to** `'+esc(a.new)+'` on `'+esc(a.branch)+'`. '+esc(a.message||'Restart Adam to apply.')+'\n\n**Close this window and relaunch Adam** (or restart it from Mission Control) to load the new version.','<span class="badge">update</span>');try{_showToast({id:'upd',level:'success',source:'UPDATE',title:'Update applied — restart Adam',body:'now at '+a.new,age_s:0})}catch(_){}}
  else{bubble('bot','⚠ Update failed: '+esc((a&&a.error)||'unknown')+((a&&a.changes&&a.changes.length)?('\n\nLocal changes blocking update:\n```\n'+a.changes.map(esc).join('\n')+'\n```'):''),'<span class="badge err">update</span>')}
}
function _showCliInfo(){try{_showToast({id:'cli',level:'success',source:'CLI',title:'Adam terminal CLI',body:'Run  amni chat  in a terminal (or: python -m amni.cli). Inside chat, prefix ! to run a shell command.',age_s:0})}catch(_){alert('Adam CLI: run "amni chat" in a terminal (or "python -m amni.cli").')}}
function _renderCmdMenu(){const body=document.getElementById('cmd-menu-list');if(!body)return;const cur=localStorage.getItem(THEME_KEY)||'jarvis';const modes=[{label:'🎨 Theme',hint:'cycle palette — now: '+THEMES[cur].label,act:'toggleCmdMenu(false);cycleTheme()'},{label:'🔊 Voice Out',hint:'speak replies aloud (TTS)',act:'toggleCmdMenu(false);toggleVoiceOut()'},{label:'🛠 Tools drawer',hint:'voice · gesture · coach · export',act:'toggleCmdMenu(false);toggleToolsDrawer()'},{label:'⚙ Settings panel',hint:'personas · voices · themes · pace',act:'toggleCmdMenu(false);togglePersonaPanel()'},{label:'↗ Open Amni-Code',hint:'launch the granite-powered coding IDE',act:'toggleCmdMenu(false);_openPeer()'},{label:'⬆ Check for updates',hint:'git fetch + pull new versions from the UI',act:'toggleCmdMenu(false);_checkUpdate()'}];const mh='<div class="cm-group">MODES &amp; PANELS</div>'+modes.map(m=>`<div class="cm-row" onclick="${m.act}"><span class="cm-cmd">${m.label}</span><span class="cm-hint">${esc(m.hint)}</span></div>`).join('');const ch='<div class="cm-group">COMMANDS — click to run, no typing</div>'+_SLASH_COMMANDS.map(c=>`<div class="cm-row" onclick="_menuRun('${c.cmd}')"><span class="cm-cmd">/${esc(c.cmd)}</span><span class="cm-hint">${esc(c.hint)}</span></div>`).join('');body.innerHTML=mh+ch}
const _GEO_CACHE_KEY='amni_jarvis_geo_cache';let _geoCache=null;
(function(){try{const j=JSON.parse(localStorage.getItem(_GEO_CACHE_KEY)||'null');if(j&&typeof j.lat==='number'&&typeof j.lon==='number'&&Date.now()-(j.ts||0)<3600000)_geoCache=j}catch{}})();
function _resolveLocalLocation(){
  return new Promise((resolve,reject)=>{
    if(_geoCache){resolve(_geoCache);return}
    if(!('geolocation' in navigator)){reject(new Error('geolocation unavailable'));return}
    navigator.geolocation.getCurrentPosition(pos=>{
      const c={lat:pos.coords.latitude,lon:pos.coords.longitude,ts:Date.now()};
      _geoCache=c;try{localStorage.setItem(_GEO_CACHE_KEY,JSON.stringify(c))}catch{}
      resolve(c);
    },err=>reject(err),{timeout:8000,maximumAge:300000,enableHighAccuracy:false});
  });
}
async function _runWebSearch(query){
  const w=document.querySelector('.welcome');if(w)w.remove();
  const uMsg=document.createElement('div');uMsg.className='msg user';const uB=document.createElement('div');uB.className='bubble';uB.textContent='/search '+query;uMsg.appendChild(uB);log.appendChild(uMsg);
  const botMsg=document.createElement('div');botMsg.className='msg bot';const botB=document.createElement('div');botB.className='bubble thinking';botB.textContent='searching the web…';botMsg.appendChild(botB);log.appendChild(botMsg);log.scrollTop=log.scrollHeight;
  try{
    const r=await fetch('/skills/web',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{query:query}})});
    const j=await r.json();
    const out=(j&&j.output)?j.output:j;
    botB.classList.remove('thinking');
    if(!r.ok||!out||out.error){botB.innerHTML=md('Web search unavailable: '+esc((out&&out.error)||('HTTP '+r.status))+'\n\n_The crawler may be offline (optional `ddgs`+`trafilatura` deps). Search stays gated behind the PII egress scrubber when it runs._')}
    else{
      const ans=out.answer||'(no answer distilled)';const srcs=out.sources||[];
      let html=md(ans);
      if(srcs.length){html+='<div class="meta" style="margin-top:6px"><b>Sources:</b><br>'+srcs.slice(0,5).map((s,i)=>(i+1)+'. <a href="'+esc(s)+'" target="_blank" rel="noopener">'+esc(s)+'</a>').join('<br>')+'</div>'}
      if(out.pii_scrubbed)html+='<div class="meta" style="margin-top:4px"><span class="badge">PII scrubbed before send</span></div>';
      botB.innerHTML=html;
    }
    const metaEl=document.createElement('div');metaEl.className='meta';metaEl.innerHTML='<span class="badge">web</span>'+(out&&out.tokens?'<span class="badge">'+out.tokens+' tok</span>':'');botMsg.appendChild(metaEl);
  }catch(e){botB.classList.remove('thinking');botB.textContent='Search request failed: '+e.message}
  log.scrollTop=log.scrollHeight;
}
function _handleSlashCommand(text){
  const m=text.match(/^\/(\w[\w-]*)(?:\s+(.*))?$/);
  if(!m)return false;
  const cmd=m[1].toLowerCase();const arg=(m[2]||'').trim();
  if(cmd==='new'||cmd==='clear'){
    document.querySelectorAll('#log .msg').forEach(el=>el.remove());
    if(cmd==='new'){sid='';try{localStorage.removeItem(SKEY)}catch{}}
    bubble('bot',cmd==='new'?'Started a fresh session.':'Chat cleared (session kept).','<span class="badge">cmd</span>');
    return true;
  }
  if(cmd==='help'){const fn=window.toggleKbdOverlay||null;if(fn)try{fn()}catch{}else bubble('bot','Press `?` to open keyboard shortcuts.','<span class="badge">cmd</span>');return true}
  if(cmd==='persona'){
    if(!arg){bubble('bot','Usage: `/persona <name>` — e.g. `/persona rikku`. Use the panel for new persona learning.','<span class="badge">cmd</span>');return true}
    _pickPersona(arg);return true;
  }
  if(cmd==='sessions'){toggleSessionsPanel();return true}
  if(cmd==='tools'){toggleToolsDrawer();return true}
  if(cmd==='status'){toggleStatusPanel();return true}
  if(cmd==='jarvis'||cmd==='handsfree'){toggleJarvisMode();return true}
  if(cmd==='update'||cmd==='updates'){_checkUpdate();return true}
  if(cmd==='find'){if(!arg){bubble('bot','Usage: `/find <query>` — searches your workdir.','<span class="badge">cmd</span>');return true}input.value='find "'+arg.replace(/"/g,'\\"')+'"';return false}
  if(cmd==='search'){if(!arg){bubble('bot','Usage: `/search <query>` — web search anytime, no matter the context.','<span class="badge">cmd</span>');return true}_runWebSearch(arg);return true}
  if(cmd==='pace'){const n=parseInt(arg,10);if(isFinite(n)){_paceSliderChange(n);bubble('bot','Stream pace set to **'+n+' cps**.','<span class="badge">cmd</span>')}else{bubble('bot','Usage: `/pace <cps>` (10-2000). Current: **'+_typeSpeedCps()+' cps**.','<span class="badge">cmd</span>')}return true}
  if(cmd==='bookmarks'){_bookmarksShow();return true}
  if(cmd==='reminders'){toggleRemindersPanel(true);return true}
  if(cmd==='perms'){togglePermsPanel(true);return true}
  if(cmd==='pclog'){togglePcLogPanel(true);return true}
  if(cmd==='se'){toggleSePanel(true);return true}
  return false;
}
let _seOpen=false;
function toggleSePanel(force){
  const open=(typeof force==='boolean')?force:!_seOpen;_seOpen=open;
  const el=document.getElementById('se-panel');if(el)el.classList.toggle('show',open);
  if(open)_seRefresh();
}
async function _seRefresh(){
  const body=document.getElementById('se-body');if(!body)return;
  try{
    const j=await(await fetch('/memory/se-dashboard')).json();
    const ci=j.code_index||{};const c=j.coding||{};const runs=j.open_runs||[];
    const rate=c.success_rate_pct||0;
    let html='<div class="se-grid">';
    html+='<div class="se-stat"><div class="v">'+(ci.built?(ci.n_files||0):'—')+'</div><div class="l">files mapped</div></div>';
    html+='<div class="se-stat"><div class="v">'+(ci.built?(ci.n_symbols||0):'—')+'</div><div class="l">symbols</div></div>';
    html+='<div class="se-stat"><div class="v">'+(c.total||0)+'</div><div class="l">coding attempts</div></div>';
    html+='<div class="se-stat"><div class="v">'+rate+'%</div><div class="l">success rate</div></div>';
    html+='</div>';
    if(ci.built&&ci.languages){const ls=Object.entries(ci.languages).slice(0,6).map(([k,v])=>esc(k)+' '+v).join(' · ');html+='<div class="se-langs">'+ls+(ci.iso?' · indexed '+esc(ci.iso.replace("T"," ").slice(5,16)):'')+'</div>'}
    else{html+='<div class="se-empty">No code map yet — say <b>code_index build</b> or train Adam on a folder.</div>'}
    html+='<div class="se-sec">attempts</div>';
    html+='<div class="se-bar"><div class="fill" style="width:'+rate+'%"></div></div>';
    html+='<div class="se-langs">'+(c.succeeded||0)+' passed · '+(c.failed||0)+' failed · '+(c.retried_attempts||0)+' retries</div>';
    html+='<div class="se-sec">open runs</div>';
    if(runs.length){for(const r of runs){html+='<div class="se-run"><span class="id">'+esc(r.run_id||'')+'</span><span>'+esc((r.task||'').slice(0,40))+' · #'+(r.attempt||1)+'</span></div>'}}
    else{html+='<div class="se-empty">No open runs. Start one with <b>code this: &lt;task&gt;</b>.</div>'}
    body.innerHTML=html;
  }catch(e){body.innerHTML='<div class="se-empty">dashboard unavailable: '+esc(e.message)+'</div>'}
}
let _pcLogOpen=false;
function togglePcLogPanel(force){
  const open=(typeof force==='boolean')?force:!_pcLogOpen;_pcLogOpen=open;
  const el=document.getElementById('pclog-panel');if(el)el.classList.toggle('show',open);
  if(open)_pcLogRefresh();
}
async function _pcLogRefresh(){
  const rows=document.getElementById('pl-rows'),sum=document.getElementById('pl-summary');
  try{
    const j=await(await fetch('/memory/pc-actions?limit=40')).json();
    const by=j.by_status||{};
    if(sum)sum.textContent=(j.total||0)+' logged · '+(by.executed||0)+' ran · '+(by.refused||0)+' refused · '+(by.cancelled||0)+' cancelled';
    let html='';
    const pend=j.pending||[];
    if(pend.length){html+='<div class="pl-pending">awaiting confirm</div>';for(const p of pend){html+='<div class="pl-row"><span class="pl-st proposed">pending</span><span class="pl-act">['+esc(p.risk||'?')+'] '+esc(p.action||'?')+': '+esc((p.target||'').slice(0,60))+'</span></div>'}}
    const recent=j.recent||[];
    if(!recent.length&&!pend.length){rows.innerHTML='<div class="pl-empty">No PC actions yet. Adam asks before doing anything on your machine.</div>';return}
    for(const r of recent){
      const st=esc(r.status||'?');const when=esc((r.iso||'').replace('T',' ').slice(5,16));
      const tgt=esc((r.target||'').slice(0,60));const act=esc(r.action||'?');
      html+='<div class="pl-row"><span class="pl-when">'+when+'</span><span class="pl-st '+st+'">'+st+'</span><span class="pl-act">'+act+(tgt?': '+tgt:'')+'</span></div>';
    }
    rows.innerHTML=html;
  }catch(e){if(rows)rows.innerHTML='<div class="pl-empty">log unavailable: '+esc(e.message)+'</div>'}
}
let _permsOpen=false;
function togglePermsPanel(force){
  const open=(typeof force==='boolean')?force:!_permsOpen;_permsOpen=open;
  const el=document.getElementById('perms-panel');if(el)el.classList.toggle('show',open);
  if(open)_permsRefresh();
}
async function _permsQuery(name){
  try{if(navigator.permissions&&navigator.permissions.query){const s=await navigator.permissions.query({name:name});return s.state}}catch(e){}
  return null;
}
function _permsSetState(k,st){
  const el=document.getElementById('pm-state-'+k);if(!el)return;
  const s=st||'unknown';el.textContent=s;
  el.className='pm-state '+(s==='granted'?'granted':(s==='denied'?'denied':(s==='prompt'?'prompt':'')));
}
async function _permsRefresh(){
  _permsSetState('geolocation',await _permsQuery('geolocation'));
  _permsSetState('microphone',await _permsQuery('microphone'));
  _permsSetState('camera',await _permsQuery('camera'));
  let n='unknown';try{if(typeof Notification!=='undefined')n=(Notification.permission==='default'?'prompt':Notification.permission)}catch(e){}
  _permsSetState('notifications',n);
}
async function _permsRequest(kind){
  try{
    if(kind==='geolocation'){await _resolveLocalLocation().catch(()=>{})}
    else if(kind==='microphone'){const s=await navigator.mediaDevices.getUserMedia({audio:true});s.getTracks().forEach(t=>t.stop())}
    else if(kind==='camera'){const s=await navigator.mediaDevices.getUserMedia({video:true});s.getTracks().forEach(t=>t.stop())}
    else if(kind==='notifications'){if(typeof Notification!=='undefined')await Notification.requestPermission()}
  }catch(e){bubble('bot','Permission for '+esc(kind)+' was blocked or unavailable: '+esc(e.message||String(e))+' — you may need to allow it in your browser site settings.','<span class="badge err">perms</span>')}
  setTimeout(_permsRefresh,400);
}
async function send(){
  if(send_btn&&send_btn.dataset.streaming==='1'){stopStream();return}
  const text=input.value.trim();if(!text)return;
  if(text.startsWith('/')&&_handleSlashCommand(text)){pushInputHistory(text);input.value='';input.style.height='auto';_clearDraft();input.focus();return}
  pushInputHistory(text);_clearDraft();
  input.value='';input.style.height='auto';
  bubble('user',text);
  const bot=bubble('bot','...');bot.bubble.classList.add('thinking');
  let acc='';let tier='?';let wall='';let persona='';let category='';let widgets=[];let aborted=false;let reasonEl=null;let reasonText='';
  _streamAbort=new AbortController();
  _setSendButtonState(true);
  _typeStart(bot,null);
  let _geoCoords=null;
  if(/\b(?:my\s+(?:local\s+)?weather|local\s+weather|weather\s+(?:here|now|today|outside)|near\s+me|around\s+me)\b/i.test(text)){
    try{_geoCoords=await _resolveLocalLocation()}catch(_){_geoCoords=null}
  }
  try{
    const body=sid?{message:text,session_id:sid}:{message:text};
    if(_geoCoords){body.client_lat=_geoCoords.lat;body.client_lon=_geoCoords.lon}
    const _pp=((typeof _selectedPersona!=='undefined'&&_selectedPersona)||(typeof personaName!=='undefined'&&personaName)||'');
    if(_pp)body.persona=_pp;
    const resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:_streamAbort.signal});
    if(!resp.body){throw new Error('streaming not supported by browser')}
    const reader=resp.body.getReader();const decoder=new TextDecoder();let buf='';
    while(true){
      let chunk;
      try{chunk=await reader.read()}
      catch(e){if(e&&e.name==='AbortError'){aborted=true;break}throw e}
      const {done,value}=chunk;if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const events=buf.split('\n\n');buf=events.pop()||'';
      for(const evt of events){
        const lines=evt.split('\n');let etype='message',edata='';
        for(const ln of lines){if(ln.startsWith('event: '))etype=ln.slice(7);else if(ln.startsWith('data: '))edata+=ln.slice(6)}
        if(!edata)continue;
        try{
          if(etype==='token'){
            const chunk=JSON.parse(edata);
            if(bot._st){clearInterval(bot._st);bot._st=null}
            if(reasonEl&&reasonEl.classList.contains('open')){reasonEl.classList.remove('open');reasonEl.classList.add('done')}
            if(bot.bubble.classList.contains('thinking')){bot.bubble.classList.remove('thinking');bot.bubble.textContent=''}
            acc+=chunk;_typePush(chunk);
            if(typeof _corePulse==='function'&&performance.now()-_coreLastToken>180){_coreLastToken=performance.now();_corePulse()}
          }else if(etype==='reasoning'){
            const r=JSON.parse(edata);reasonText+=r;
            if(!reasonEl){bot.bubble.classList.remove('thinking');bot.bubble.textContent='';reasonEl=document.createElement('div');reasonEl.className='reasoning open';reasonEl.innerHTML='<div class="reasoning-head" onclick="this.parentElement.classList.toggle(\'open\')"><span class="reasoning-dot"></span>thinking<span class="reasoning-toggle">▾</span></div><div class="reasoning-body"></div>';bot.msg.insertBefore(reasonEl,bot.bubble)}
            const rb=reasonEl.querySelector('.reasoning-body');rb.textContent=reasonText;rb.scrollTop=rb.scrollHeight;_smartScroll();
          }else if(etype==='thinking'){
            if(!bot._st&&!reasonEl&&bot.bubble.classList.contains('thinking')){try{const tm=JSON.parse(edata);bot.bubble.textContent='thinking… '+(tm.elapsed||0)+'s'}catch(_){}}
          }else if(etype==='status'){
            try{var _stg=JSON.parse(edata).stage;
              window._SF||(window._SF={understanding:["Rao! lemme get my head around this…","figuring out what you need…","okay, processing your ask…","reading between the lines…"],recall:["rummaging through my memory…","checking what I remember…","digging through my notes…"],web:["hitting the web for the latest…","looking that up online…","scouring the net for you…","chasing down sources…"],reasoning:["thinking it through, step by step…","working out the logic…","reasoning it out… almost there","turning it over in my head…"],writing:["okay, here it comes!","putting it together…","writing it up…"]});
              if(bot.bubble.classList.contains('thinking')){var _F=window._SF[_stg]||window._SF.understanding,_si=0;var _show=function(){if(bot.bubble.classList.contains('thinking'))bot.bubble.innerHTML='<span style="color:var(--mute);font-style:italic">'+_F[_si%_F.length]+'</span>';_si++};if(bot._st)clearInterval(bot._st);_show();bot._st=setInterval(_show,2600)}
            }catch(_){}
          }else if(etype==='meta'){
            const m=JSON.parse(edata);
            if(m.session_id){sid=m.session_id;localStorage.setItem(SKEY,sid)}
            if(m.persona)persona=m.persona;
            if(m.skill)tier='tier0_skill_'+m.skill;
            if(m.category)category=m.category;
          }else if(etype==='widget'){
            try{widgets.push(JSON.parse(edata))}catch{}
          }else if(etype==='done'){
            const d=JSON.parse(edata);tier=d.tier||tier;wall=d.wall_s||'';
          }
        }catch(p){}
      }
    }
    _typeFlushAll();
    if(bot._st){clearInterval(bot._st);bot._st=null}
    bot.bubble.classList.remove('thinking');
    if(reasonEl){reasonEl.classList.remove('open');reasonEl.classList.add('done')}
    if(!acc)bot.bubble.textContent=aborted?'(stopped before any tokens arrived)':'(empty response)';
    if(widgets.length){appendWidgets(bot.msg,widgets)}
    const metaEl=document.createElement('div');metaEl.className='meta';
    const approxTok=acc?Math.max(1,Math.round(acc.length/4)):0;
    const tokLabel=approxTok>=1000?(approxTok/1000).toFixed(1)+'k':String(approxTok);
    metaEl.innerHTML=`<span class="badge">${esc(tier)}</span>${wall?`<span>${wall}s</span>`:''}${approxTok?`<span class="badge tok" title="~${approxTok} tokens (estimated 4 chars/token)">~${tokLabel} tok</span>`:''}${persona?`<span class="badge persona">${esc(persona)}</span>`:''}${widgets.length?`<span class="badge">${widgets.length} widget(s)</span>`:''}${aborted?'<span class="badge err">stopped</span>':''}`;
    bot.msg.appendChild(metaEl);
    if(voiceOut&&acc&&!aborted)speak(acc);
  }catch(err){
    if(reasonEl){reasonEl.classList.remove('open');reasonEl.classList.add('done')}
    if(err&&err.name==='AbortError'){bot.bubble.classList.remove('thinking');if(!acc)bot.bubble.textContent='(stopped)'}
    else{bot.bubble.classList.remove('thinking');bot.bubble.textContent=_netHint(err)}
  }
  _streamAbort=null;_setSendButtonState(false);
  input.focus();log.scrollTop=log.scrollHeight;
}
let _voiceBackends={tts:false,stt:false,tts_backend:'',stt_backend:''};
async function probeVoiceBackends(){
  try{const j=await(await fetch('/voice/status')).json();
    _voiceBackends.tts=!!(j.tts&&j.tts.available);
    _voiceBackends.stt=!!(j.stt&&j.stt.available);
    _voiceBackends.tts_backend=(j.tts&&j.tts.backend)||'';
    _voiceBackends.stt_backend=(j.stt&&j.stt.backend)||'';
  }catch{}
}
probeVoiceBackends();
const PERSONA_KEY='amni_jarvis_persona',VOICE_KEY='amni_jarvis_voice';
let _selectedPersona=localStorage.getItem(PERSONA_KEY)||'';
let _selectedVoice=localStorage.getItem(VOICE_KEY)||'';if(_selectedVoice==='[object Object]'||_selectedVoice==='undefined'||_selectedVoice==='null'){_selectedVoice='';try{localStorage.removeItem(VOICE_KEY)}catch(_){}}
let _personaPanelOpen=false,_knownPersonas=[],_availableVoices=[];
async function _loadPersonas(){
  try{const j=await(await fetch('/personas')).json();_knownPersonas=j.known||j.list||(Array.isArray(j)?j:[]);if(!Array.isArray(_knownPersonas))_knownPersonas=[]}
  catch{_knownPersonas=[]}
}
async function _loadVoices(){
  try{const j=await(await fetch('/voice/status')).json();_availableVoices=(j.tts&&j.tts.voices)||[];document.getElementById('pp-tts-backend').textContent=(j.tts&&j.tts.backend)||'none'}
  catch{_availableVoices=[]}
}
function _renderPersonaPanel(){
  const cur=_selectedPersona||personaName||'Rikku';
  document.getElementById('pp-current').textContent=cur+(_selectedVoice?' · '+_selectedVoice:'');
  const list=document.getElementById('pp-list');
  if(_knownPersonas.length===0){list.innerHTML='<div class="pp-empty">no personas loaded</div>'}
  else{
    list.innerHTML=_knownPersonas.map(p=>{
      const nm=(typeof p==='string'?p:p.name)||'?';
      const voice=(typeof p==='object'&&p.tts_voice)?p.tts_voice:'';
      const active=(nm.toLowerCase()===(cur||'').toLowerCase());
      return `<div class="pp-row${active?' active':''}" onclick="_pickPersona('${esc(nm).replace(/'/g,"\\\\'")}')"><span class="nm">${esc(nm)}</span><span class="voice">${esc(voice)}</span></div>`;
    }).join('');
  }
  _renderPersonaEdit();
  _renderPaceUI();
  _renderThemePicker();
  const voices=document.getElementById('pp-voices');
  if(_availableVoices.length===0){voices.innerHTML='<div class="pp-empty">no piper voices · install via <code style="color:var(--cyan)">pip install piper-tts</code> + download a voice</div>'}
  else{
    voices.innerHTML='<div class="pp-row'+(!_selectedVoice?' active':'')+'" onclick="_pickVoice(\'\')"><span class="nm">auto (persona default)</span></div>'+_availableVoices.map(v=>{
      const vid=(v&&typeof v==='object')?String(v.id||v.alias||v.name||''):String(v);
      const lbl=(v&&typeof v==='object')?String(v.label||v.name||v.id||'voice'):String(v);
      const sub=(v&&typeof v==='object'&&v.name&&v.name!==lbl)?String(v.name):'';
      const active=(vid===_selectedVoice);
      return `<div class="pp-row${active?' active':''}" onclick="_pickVoice('${esc(vid).replace(/'/g,"\\\\'")}')"><span class="nm">${esc(lbl)}</span>${sub?' <span style="color:var(--mute);font-size:9px;font-family:Consolas,monospace">'+esc(sub)+'</span>':''}</div>`;
    }).join('');
  }
}
function _renderPersonaEdit(){
  const slot=document.getElementById('pp-edit');if(!slot)return;
  const cur=_selectedPersona||personaName||'';
  const p=_knownPersonas.find(x=>(typeof x==='object'&&(x.name||'').toLowerCase()===cur.toLowerCase()));
  if(!p||typeof p!=='object'){slot.innerHTML='<div class="pp-empty">pick a persona above to edit</div>';return}
  const src=(p.source||'preset');
  const dims=[['warmth',p.warmth],['formality',p.formality],['excitement',p.excitement],['length',p.length]];
  const rows=dims.map(([k,v])=>{const val=(v==null?.5:Number(v));return `<div class="pe-row"><label for="pe-${k}">${k}</label><input type="range" min="0" max="1" step="0.05" id="pe-${k}" value="${val}" oninput="document.getElementById('pe-${k}-v').textContent=Number(this.value).toFixed(2)"><span class="pe-val" id="pe-${k}-v">${val.toFixed(2)}</span></div>`}).join('');
  const hints=(p.voice_hints||[]).join('\n');
  const canDelete=src!=='preset';
  slot.innerHTML=`<div class="pe-source">${esc(p.name)} · source: ${esc(src)}</div><div class="pe-desc" id="pe-desc" contenteditable="true" spellcheck="true">${esc(p.description||'')}</div>${rows}<div class="pe-row"><label for="pe-hints">hints</label></div><textarea class="pe-hints" id="pe-hints" rows="3" placeholder="one voice hint per line">${esc(hints)}</textarea><div class="pe-actions"><button class="pe-btn" onclick="_personaEditSave()">SAVE</button><button class="pe-btn" onclick="_personaExport()" title="Download as .json">EXPORT</button><button class="pe-btn" onclick="_personaEditReset()">RESET</button>${canDelete?'<button class="pe-btn danger" onclick="_personaDelete()" title="Remove this edited/imported override (presets cannot be deleted)">DELETE</button>':''}</div>`;
}
async function _personaExport(){
  const cur=_selectedPersona||personaName||'';if(!cur)return;
  try{
    const r=await fetch('/persona/'+encodeURIComponent(cur)+'/export');
    if(!r.ok){bubble('bot','Export failed: HTTP '+r.status,'<span class="badge err">persona</span>');return}
    const j=await r.json();const blob=new Blob([JSON.stringify(j,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=cur.toLowerCase().replace(/[^a-z0-9_-]/g,'_')+'.persona.json';document.body.appendChild(a);a.click();setTimeout(()=>{document.body.removeChild(a);URL.revokeObjectURL(url)},400);
    bubble('bot','Exported **'+esc(cur)+'** to '+esc(a.download),'<span class="badge persona">'+esc(cur)+'</span>');
  }catch(e){bubble('bot','Export failed: '+esc(String(e)),'<span class="badge err">persona</span>')}
}
async function _personaDelete(){
  const cur=_selectedPersona||personaName||'';if(!cur)return;
  if(!confirm('Delete persona "'+cur+'"?\n\nOnly learned/edited/imported overrides are removed — presets are restored.'))return;
  try{
    const r=await fetch('/persona/'+encodeURIComponent(cur),{method:'DELETE'});
    const j=await r.json().catch(()=>({}));
    if(!r.ok){bubble('bot','Delete failed: '+esc(j.detail||r.status),'<span class="badge err">persona</span>');return}
    bubble('bot','Removed **'+esc(cur)+'** override.','<span class="badge persona">'+esc(cur)+'</span>');
    _selectedPersona=null;await _loadPersonas();_renderPersonaPanel();
  }catch(e){bubble('bot','Delete failed: '+esc(String(e)),'<span class="badge err">persona</span>')}
}
function _personaImportFile(){const inp=document.getElementById('pp-import-file');if(inp){inp.value='';inp.click()}}
async function _personaImportPicked(ev){
  const f=(ev.target.files||[])[0];if(!f)return;
  try{
    const text=await f.text();const data=JSON.parse(text);
    const r=await fetch('/persona/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json();
    if(r.status===409){
      if(!confirm('Persona already exists. Overwrite?'))return;
      const data2={...data,overwrite:true};
      const r2=await fetch('/persona/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data2)});
      const j2=await r2.json();
      if(!r2.ok){bubble('bot','Import failed: '+esc(j2.detail||r2.status),'<span class="badge err">persona</span>');return}
      bubble('bot','Imported (replaced) **'+esc((j2.persona||{}).name||'?')+'**','<span class="badge persona">import</span>');
    }else if(!r.ok){bubble('bot','Import failed: '+esc(j.detail||r.status),'<span class="badge err">persona</span>');return}
    else{bubble('bot','Imported **'+esc((j.persona||{}).name||'?')+'**','<span class="badge persona">import</span>')}
    await _loadPersonas();_renderPersonaPanel();
  }catch(e){bubble('bot','Import failed: '+esc(String(e)),'<span class="badge err">persona</span>')}
}
async function _personaEditSave(){
  const cur=_selectedPersona||personaName||'';if(!cur)return;
  const desc=(document.getElementById('pe-desc')||{}).textContent||'';
  const hintsRaw=(document.getElementById('pe-hints')||{}).value||'';
  const hints=hintsRaw.split(/\n+/).map(s=>s.trim()).filter(Boolean);
  const get=(k)=>{const el=document.getElementById('pe-'+k);return el?Number(el.value):null};
  const body={description:desc,voice_hints:hints,warmth:get('warmth'),formality:get('formality'),excitement:get('excitement'),length:get('length')};
  try{
    const r=await fetch('/persona/'+encodeURIComponent(cur),{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok){bubble('bot','Persona edit failed: '+esc(j.detail||r.status),'<span class="badge err">persona</span>');return}
    await _loadPersonas();_renderPersonaPanel();
    bubble('bot','Updated **'+esc(cur)+'** — voice/style edits saved.','<span class="badge persona">'+esc(cur)+'</span>');
  }catch(e){bubble('bot','Persona PATCH failed: '+esc(String(e)),'<span class="badge err">persona</span>')}
}
function _personaEditReset(){_renderPersonaEdit()}
async function togglePersonaPanel(){
  _personaPanelOpen=!_personaPanelOpen;
  const p=document.getElementById('persona-panel');p.classList.toggle('show',_personaPanelOpen);
  if(_personaPanelOpen){await Promise.all([_loadPersonas(),_loadVoices()]);_renderPersonaPanel()}
}
async function _pickPersona(name){
  try{
    const r=await fetch('/persona',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,session_id:sid,learn_via_web:false})});
    const j=await r.json();
    if(r.ok&&j.persona){_selectedPersona=j.persona.name;localStorage.setItem(PERSONA_KEY,_selectedPersona);personaName=_selectedPersona;personaPill.textContent='persona '+_selectedPersona;_renderPersonaPanel();bubble('bot','Now: **'+esc(_selectedPersona)+'** — '+(esc(j.persona.description||'')),'<span class="badge persona">'+esc(_selectedPersona)+'</span>')}
  }catch(e){console.warn('persona switch failed',e)}
}
async function _personaLearnNew(){
  const inp=document.getElementById('pp-new');const name=(inp.value||'').trim();
  if(!name)return;inp.value='';
  bubble('bot','Web-learning persona "'+esc(name)+'" — this may take ~10s…','<span class="badge persona">learn</span>');
  try{
    const r=await fetch('/persona',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,session_id:sid,learn_via_web:true})});
    const j=await r.json();
    if(r.ok&&j.persona){_selectedPersona=j.persona.name;localStorage.setItem(PERSONA_KEY,_selectedPersona);personaName=_selectedPersona;try{personaPill.textContent='persona '+_selectedPersona;await _loadPersonas();_renderPersonaPanel()}catch(_){}bubble('bot','Learned **'+esc(_selectedPersona)+'** — '+esc(j.persona.description||''),'<span class="badge persona">'+esc(_selectedPersona)+'</span>')}
    else bubble('bot','Learn failed: '+esc(j.error||j.detail||JSON.stringify(j)),'<span class="badge err">err</span>')
  }catch(e){bubble('bot','Learn error: '+esc(e.message),'<span class="badge err">err</span>')}
}
function _pickVoice(v){_selectedVoice=v;localStorage.setItem(VOICE_KEY,v);_renderPersonaPanel();if(v)bubble('bot','TTS voice set to **'+esc(v)+'**','<span class="badge">voice</span>')}
const _origProbeVoiceBackends=probeVoiceBackends;
const _PERSONA_WELCOME={
  alfred:{h:'AT YOUR SERVICE',t:'Master Anthony — Alfred Pennyworth at your service. How may I assist you this morning?'},
  rikku:{h:'RAO! READY TO GO!',t:"Fryd's ib?! Rikku here — let's get scrappy and figure things out together!"},
  jarvis:{h:'STANDING BY, SIR',t:"Of course, sir. All systems green. What's first on the agenda?"},
  yoda:{h:'READY, I AM',t:'Help you, I shall. Begin where you wish, you may.'},
  mentor:{h:'READY TO HELP',t:"Tell me what you're working on — I'll meet you wherever you are."},
  pirate:{h:'AHOY, MATEY!',t:'Charts plotted, sails trimmed, captain. Where to next?'},
  scientist:{h:'INSTRUMENTS ONLINE',t:'Ready when you are — what hypothesis are we testing today?'},
  jobs:{h:'LET\'S MAKE SOMETHING GREAT',t:'What are we building? Strip it to the essence — start there.'},
  haiku:{h:'STILL WATER WAITS',t:'Three lines I weave / from your single quiet thought / let us begin friend'},
  sherlock:{h:'OBSERVATIONS PENDING',t:"You've a problem to share. Lay out the details — I'll handle the deduction."},
  neutral:{h:'NEURAL INTERFACE READY',t:'Ask anything. Live data renders inline as glowing cards.'}
};
async function _applyWelcomeForPersona(){
  const wh=document.getElementById('welcome-heading');const wt=document.getElementById('welcome-tagline');const ws=document.getElementById('welcome-persona-stamp');
  if(!wh||!wt)return;
  try{const r=await fetch('/personas');if(!r.ok)return;const j=await r.json();const name=(j.default||'').toLowerCase();
    const w=_PERSONA_WELCOME[name]||_PERSONA_WELCOME.neutral;
    wh.textContent=w.h;wt.textContent=w.t;if(ws)ws.textContent='◆ '+(name||'neutral').toUpperCase()+' MODE';
    const tint=(typeof _personaToastTint==='function')?_personaToastTint():null;
    if(tint){wh.style.color=tint.hex;wh.style.textShadow=`0 0 8px ${tint.hex}`;if(ws)ws.style.color=tint.hex}
    else{wh.style.color='';wh.style.textShadow='';if(ws)ws.style.color=''}
  }catch{}
}
setTimeout(_applyWelcomeForPersona,250);
const KICKOFF_KEY='amni_jarvis_kickoff_date';
function _kickoffDateKey(){const d=new Date();return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0')}
const _PERSONA_KICKOFF_PHRASES={
  alfred:{intro:'{greet}, sir. Allow me to summarize what awaits you:',tail_review:'\n\nWhen you\'re ready, sir, the coach panel ({key}) clears that queue handily.',tail_pending:'\n\nA glance at TESTS or SHELL will likely settle matters, sir.',tail_calm:'\n\nNothing pressing, sir. The day is yours to direct.'},
  rikku:{intro:'Rao! {greet}! Look at this stack waiting for us:',tail_review:'\n\nLet\'s knock those review cards out — coach panel is just {key}!',tail_pending:'\n\nTap the TESTS or SHELL pill and we\'ll sort it together!',tail_calm:'\n\nFryd\'s ib next? Your pick!'},
  jarvis:{intro:'{greet}, sir. Current standing:',tail_review:'\n\nCoach panel via {key} when you\'re ready, sir.',tail_pending:'\n\nTESTS or SHELL pill should clarify — at your convenience.',tail_calm:'\n\nAll systems nominal. Awaiting your direction.'},
  yoda:{intro:'{greet}. Waiting for you, these are:',tail_review:'\n\nClear the queue, you must. The coach panel ({key}), use it.',tail_pending:'\n\nLook at TESTS or SHELL, you should.',tail_calm:'\n\nQuiet, the path is. Choose freely, you may.'},
  mentor:{intro:'{greet}. Here\'s what\'s on the table today:',tail_review:'\n\nThe coach panel ({key}) is the fastest way to work through those.',tail_pending:'\n\nThe TESTS or SHELL pill will show you what needs attention.',tail_calm:'\n\nNothing urgent — what would you like to work on?'},
  pirate:{intro:'{greet}, captain! Charts show the following on the horizon:',tail_review:'\n\nCoach panel ({key}) is yer cleanest course to clear that queue.',tail_pending:'\n\nTap the TESTS or SHELL pill — we\'ll plot a fix together.',tail_calm:'\n\nCalm seas ahead. Set yer own heading.'},
  scientist:{intro:'{greet}. Observations from the last 24 hours:',tail_review:'\n\nThe coach panel ({key}) is the most efficient way to process the review queue.',tail_pending:'\n\nTESTS and SHELL pills surface the relevant evidence.',tail_calm:'\n\nNo immediate signals — open to your hypothesis.'},
  jobs:{intro:'{greet}. Here\'s what matters today:',tail_review:'\n\nStrip everything else. Open coach ({key}) and clear the queue. Insanely great work is showing up.',tail_pending:'\n\nFocus: TESTS or SHELL. Make it just work.',tail_calm:'\n\nNothing in the way. What are you going to build?'},
  haiku:{intro:'{greet}.\nQuiet morning unfolds — / waiting on your hand to write / the day\'s first new line.\n\nWhat waits:',tail_review:'\n\nReviews call your name — / coach panel by {key} key / opens still water.',tail_pending:'\n\nTESTS or SHELL pill / opens like a folded page / wanting fresh eyes.',tail_calm:'\n\nStillness. The page blank. / Yours to fill however you / chose this morning to.'},
  sherlock:{intro:'{greet}. The overnight evidence stack:',tail_review:'\n\nObserve: the coach panel ({key}) is the most economical path forward.',tail_pending:'\n\nThe TESTS or SHELL pill reveals the relevant clues.',tail_calm:'\n\nNothing of immediate interest. Lead the case where you will.'},
  neutral:{intro:'{greet}. Here\'s what\'s waiting for you today:',tail_review:'\n\nOpen the coach panel ({key}) to clear the review queue.',tail_pending:'\n\nTap the TESTS or SHELL pill to drill in.',tail_calm:'\n\nNothing on fire — your call where to start.'}
};
async function _dailyKickoff(){
  if(localStorage.getItem(KICKOFF_KEY)===_kickoffDateKey())return;
  try{
    const [digestR,reviewsR,personaR]=await Promise.all([fetch('/memory/digest?hours=24'),fetch('/memory/coach/reviews?limit=5').catch(()=>null),fetch('/personas').catch(()=>null)]);
    if(!digestR||!digestR.ok)return;
    const dg=await digestR.json();const reviews=(reviewsR&&reviewsR.ok)?(await reviewsR.json()).reviews||[]:[];
    const personaName=((personaR&&personaR.ok)?(await personaR.json()).default:'')||'neutral';
    const facts=(dg.learning||{}).facts_today||0;const errs=(dg.shell||{}).errors_today||0;const pending=(dg.verifier||{}).pending||0;const streak=(dg.coach||{}).streak_days||0;const today_active=(dg.coach||{}).today_active;
    const _kickPrevPending=parseInt(localStorage.getItem('amni_jarvis_kickoff_pending')||'0',10);try{localStorage.setItem('amni_jarvis_kickoff_pending',String(pending))}catch(_){}
    const _newPending=Math.max(0,pending-_kickPrevPending);
    const parts=[];
    if(reviews.length)parts.push(reviews.length+' coach card'+(reviews.length===1?'':'s')+' due for review');
    if(_newPending)parts.push(_newPending+' new edit'+(_newPending===1?'':'s')+' awaiting your review');
    if(errs)parts.push(errs+' shell error'+(errs===1?'':'s')+' overnight');
    if(facts)parts.push(facts+' new fact'+(facts===1?'':'s')+' learned via daemon');
    if(streak>0&&!today_active)parts.push('your '+streak+'-day coach streak is at stake — practice once today to keep it');
    if(!parts.length){localStorage.setItem(KICKOFF_KEY,_kickoffDateKey());return}
    const hour=new Date().getHours();const greet=hour<5?'Up late':(hour<12?'Good morning':(hour<17?'Good afternoon':'Good evening'));
    const phrases=_PERSONA_KICKOFF_PHRASES[personaName.toLowerCase()]||_PERSONA_KICKOFF_PHRASES.neutral;
    const key=navigator.platform.toLowerCase().includes('mac')?'⌘+G':'Ctrl+G';
    const head=phrases.intro.replace('{greet}',greet).replace('{key}',key);
    const body=parts.map((p,i)=>(i+1)+'. '+p).join('\n');
    const tailTemplate=reviews.length?phrases.tail_review:((pending||errs)?phrases.tail_pending:phrases.tail_calm);
    const tail=tailTemplate.replace('{key}',key);
    bubble('bot',head+'\n\n'+body+tail,'<span class="badge">kickoff · '+esc(personaName)+'</span>');
    localStorage.setItem(KICKOFF_KEY,_kickoffDateKey());
  }catch{}
}
setTimeout(_dailyKickoff,1400);
let _wdFull='';
function _wdTail(full,n=3){if(!full)return '';const parts=full.replace(/\\/g,'/').split('/').filter(Boolean);return (parts.length>n?'…/':'')+parts.slice(-n).join('/')}
async function _loadWorkdir(){
  try{const r=await fetch('/workdir');if(!r.ok)return;const j=await r.json();const wd=j.workdir||'';if(!wd)return;
    _wdFull=wd;const pill=document.getElementById('wd-pill');const path=document.getElementById('wd-path');if(!pill||!path)return;
    path.textContent=_wdTail(wd,3);pill.title='Workdir: '+wd+(j.unrestricted?' (UNRESTRICTED MODE)':'')+'\nClick to copy full path';
    pill.classList.add('show');if(j.unrestricted)pill.classList.add('unrestricted');
  }catch{}
}
function _wdCopy(){
  if(!_wdFull)return;
  try{navigator.clipboard.writeText(_wdFull);bubble('bot','Copied workdir to clipboard: `'+esc(_wdFull)+'`','<span class="badge">copy</span>')}
  catch{bubble('bot','Workdir: `'+esc(_wdFull)+'` (clipboard unavailable)','<span class="badge err">copy</span>')}
}
_loadWorkdir();
let _wdPanelOpen=false;
const _WD_EXT_ICON={py:'🐍',js:'📜',ts:'📜',tsx:'📜',jsx:'📜',rs:'🦀',go:'🐹',rb:'💎',md:'📝',txt:'📝',json:'⚙',yaml:'⚙',yml:'⚙',toml:'⚙',html:'🌐',htm:'🌐',css:'🎨',scss:'🎨',sql:'🗄',log:'📋',csv:'📊',tsv:'📊',png:'🖼',jpg:'🖼',jpeg:'🖼',gif:'🖼',svg:'🖼',pdf:'📄',sh:'🖥',bash:'🖥',ps1:'🖥',xml:'⚙',ini:'⚙',cfg:'⚙',env:'⚙',lock:'🔒',exe:'⚡',bat:'🖥',dll:'⚡'};
function _wdFmtSize(n){if(n==null)return '';if(n<1024)return n+'b';if(n<1024*1024)return Math.round(n/1024)+'kb';if(n<1024*1024*1024)return (n/1024/1024).toFixed(1)+'mb';return (n/1024/1024/1024).toFixed(1)+'gb'}
async function _wdToggle(){
  _wdPanelOpen=!_wdPanelOpen;const p=document.getElementById('wd-panel');p.classList.toggle('show',_wdPanelOpen);
  if(_wdPanelOpen)await _wdLoadTree('');
}
async function _wdLoadTree(subpath){
  const base=document.getElementById('wp-base');const list=document.getElementById('wp-list');
  list.innerHTML='<div class="wp-empty">loading…</div>';
  try{
    const url='/workdir/tree?max_depth=2&max_files=200'+(subpath?'&subpath='+encodeURIComponent(subpath):'');
    const r=await fetch(url);if(!r.ok){list.innerHTML='<div class="wp-empty">tree unavailable ('+r.status+')</div>';return}
    const j=await r.json();base.textContent=j.base||'';
    const entries=j.entries||[];
    if(!entries.length){list.innerHTML='<div class="wp-empty">empty directory</div>';return}
    list.innerHTML=entries.map(e=>{
      const icon=e.is_dir?'📁':(_WD_EXT_ICON[e.ext]||'📄');
      const cls='wp-row'+(e.is_dir?' dir':'');
      const indent=' '.repeat(Math.max(0,e.depth)*2);
      const safe=e.rel.replace(/'/g,"\\\\'");
      const size=e.is_dir?'':_wdFmtSize(e.size);
      return `<div class="${cls}" onclick="_wdRowClick('${safe}',${e.is_dir})" title="${esc(e.rel)}"><span class="wp-icon">${icon}</span><span class="wp-name">${esc(indent+e.name)}</span><span class="wp-size">${size}</span></div>`;
    }).join('');
    if(j.truncated)list.insertAdjacentHTML('afterend','<div class="wp-trunc">truncated at '+j.max_files+' files — try a deeper subpath</div>');
  }catch(e){list.innerHTML='<div class="wp-empty">error: '+esc(e.message)+'</div>'}
}
function _wdRowClick(rel,isDir){
  if(isDir){bubble('bot','Drilling into `'+esc(rel)+'`…','<span class="badge">workdir</span>');_wdLoadTree(rel);return}
  input.value='Read `'+rel+'` and tell me what it does';input.focus();
}
async function _initPersonaPill(){await _origProbeVoiceBackends();await _loadPersonas();if(_selectedPersona){personaName=_selectedPersona;personaPill.textContent='persona '+_selectedPersona;_applyWelcomeForPersona()}}
_initPersonaPill();
let _ldStats=null,_ldPanelOpen=false,_ldPollTimer=null,_ldLastTopic=null,_ldErrCount=0;
function _ldHumanDur(s){if(!s||s<60)return Math.round(s||0)+'s';if(s<3600)return Math.round(s/60)+'m';if(s<86400)return (s/3600).toFixed(1)+'h';return (s/86400).toFixed(1)+'d'}
function _ldHumanRate(r){return (r||0).toFixed(r>=10?0:1)}
async function _pollLearningDaemon(){
  try{
    const r=await fetch('/memory/daemon');
    if(!r.ok){_ldErrCount++;_ldUpdatePill('error',null);return}
    _ldErrCount=0;const j=await r.json();_ldStats=j;_ldUpdatePill(j.enabled?(j.current_topic?'active':'idle'):'paused',j);
    if(_ldPanelOpen)_renderLearnPanel();
    if(j.current_topic && j.current_topic!==_ldLastTopic){_ldLastTopic=j.current_topic}
  }catch(e){_ldErrCount++;_ldUpdatePill('error',null)}
}
function _ldUpdatePill(state,j){
  const led=document.getElementById('ld-led');const txt=document.getElementById('ld-text');if(!led||!txt)return;
  led.className='ld-led '+state;
  if(state==='error'){txt.textContent='learning offline';return}
  if(!j){txt.textContent='learning —';return}
  const n=(j.counters&&j.counters.qa_pairs_new)||0;
  if(state==='active'){const t=j.current_topic||'';txt.textContent='learning: '+(t.length>22?t.slice(0,22)+'…':t)}
  else if(state==='paused')txt.textContent='learning paused • '+n+' facts'
  else txt.textContent='idle • '+n+' facts • '+_ldHumanRate(j.facts_per_hour||0)+'/h'
}
function toggleLearnPanel(){_ldPanelOpen=!_ldPanelOpen;const p=document.getElementById('learn-panel');p.classList.toggle('show',_ldPanelOpen);const pp=document.getElementById('persona-panel');if(_ldPanelOpen&&pp&&pp.classList.contains('show')){pp.classList.remove('show');_personaPanelOpen=false}if(_ldPanelOpen)_pollLearningDaemon()}
function _renderLearnPanel(){
  const j=_ldStats;if(!j)return;
  const now=document.getElementById('lp-now');const active=!!j.current_topic;
  now.className='lp-now'+(active?'':' idle');
  if(active){now.innerHTML='<div class="topic">'+esc(j.current_topic)+'</div><div class="phase">'+esc(j.current_topic_phase||'working')+' • '+_ldHumanDur(j.current_topic_age_s)+'</div>'}
  else{const reason=j.enabled?(j.user_active_recently?'paused — user is active':'waiting for curiosity tick'):'daemon paused';now.innerHTML='<div class="topic">— idle —</div><div class="phase">'+esc(reason)+'</div>'}
  const c=j.counters||{};const a=j.atlas||{};
  document.getElementById('lp-facts').textContent=(c.qa_pairs_new||0);
  document.getElementById('lp-rate').textContent=_ldHumanRate(j.facts_per_hour||0);
  document.getElementById('lp-queue').textContent=(j.queue_depth||0);
  document.getElementById('lp-uptime').textContent=_ldHumanDur(j.uptime_s||0);
  document.getElementById('lp-urls').textContent=(c.urls_ingested||0);
  document.getElementById('lp-cells').textContent=(a.n_cells||a.cells||0);
  const btn=document.getElementById('lp-pause-btn');btn.textContent=j.enabled?'PAUSE':'RESUME';
  const recent=document.getElementById('lp-recent');const rt=j.recent_topics||[];
  if(rt.length===0){recent.innerHTML='<div style="font-size:10px;color:var(--mute);text-align:center;padding:8px;font-style:italic">no completed topics yet</div>'}
  else{recent.innerHTML=rt.slice(0,6).map(t=>'<div class="lp-recent"><div class="t">'+esc(t.topic||'')+'</div><div class="meta">+'+(t.new||0)+' new · '+(t.reinforced||0)+' reinforced · '+_ldHumanDur(t.duration_s||0)+'</div></div>').join('')}
}
async function _daemonToggle(){if(!_ldStats)return;const action=_ldStats.enabled?'pause':'resume';try{await fetch('/skills/learning_daemon',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{action}})});await _pollLearningDaemon()}catch{}}
async function _daemonTick(){try{await fetch('/skills/learning_daemon',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{action:'curiosity_tick'}})});setTimeout(_pollLearningDaemon,500)}catch{}}
async function _daemonQueue(){const inp=document.getElementById('lp-queue-topic');const t=(inp.value||'').trim();if(!t)return;try{const r=await fetch('/skills/learning_daemon',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{action:'queue_topic',topic:t}})});if(r.ok){inp.value='';bubble('bot','Queued **'+esc(t)+'** for autonomous learning','<span class="badge">learn</span>');setTimeout(_pollLearningDaemon,400)}}catch{}}
function _startLearnPolling(){if(_ldPollTimer)return;_pollLearningDaemon();_ldPollTimer=setInterval(_pollLearningDaemon,8000)}
_startLearnPolling();
let _tpItems=[],_tpPanelOpen=false,_tpPollTimer=null,_tpIncludeDone=false;
function _tpHumanAge(s){if(!s||s<60)return Math.round(s||0)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';if(s<86400)return (s/3600).toFixed(1)+'h ago';return (s/86400).toFixed(1)+'d ago'}
async function _pollTestsList(){
  try{
    const r=await fetch('/memory/needs-testing?include_done='+(_tpIncludeDone?'true':'false'));
    if(!r.ok){_tpUpdatePill('error',0);return}
    const j=await r.json();_tpItems=j.items||[];const pending=j.pending||0;const total=j.count||0;
    _tpUpdatePill(pending>0?'pending':'empty',pending);
    if(_tpPanelOpen)_renderTestsPanel();
    const sum=document.getElementById('tp-summary');if(sum)sum.textContent=pending+' pending · '+total+' shown'
  }catch{_tpUpdatePill('error',0)}
}
function _tpUpdatePill(state,n){
  const led=document.getElementById('tp-led');const txt=document.getElementById('tp-text');if(!led||!txt)return;
  led.className='tp-led '+state;
  if(state==='error')txt.textContent='tests offline';
  else if(state==='empty')txt.textContent='tests · all clear';
  else txt.textContent='tests · '+n+' pending';
}
function _tpShowDone(v){_tpIncludeDone=v;const b=document.getElementById('tp-toggle-done');if(b)b.textContent=v?'HIDE DONE':'SHOW DONE';_pollTestsList()}
function toggleTestsPanel(){_tpPanelOpen=!_tpPanelOpen;const p=document.getElementById('tests-panel');p.classList.toggle('show',_tpPanelOpen);const pp=document.getElementById('persona-panel');if(_tpPanelOpen&&pp&&pp.classList.contains('show')){pp.classList.remove('show');_personaPanelOpen=false}const lp=document.getElementById('learn-panel');if(_tpPanelOpen&&lp&&lp.classList.contains('show')){lp.classList.remove('show');_ldPanelOpen=false}if(_tpPanelOpen)_pollTestsList()}
function _renderTestsPanel(){
  const list=document.getElementById('tp-list');
  if(!_tpItems.length){list.innerHTML='<div class="tp-empty">No pending tests — Adam is keeping up.</div>';return}
  const now=Date.now()/1000;
  list.innerHTML=_tpItems.map(item=>{
    const age=now-(item.ts||now);
    const done=item.status==='done';
    const path=esc(item.path||'?');
    const reason=esc(item.reason||'');
    const op=esc((item.op||'edit').toUpperCase());
    const checks=Array.isArray(item.checks_already_done)?item.checks_already_done.join(', '):'';
    return `<div class="tp-item" style="${done?'opacity:.55;':''}"><div class="path">${path}</div><div class="reason">${reason}</div>${checks?`<div class="meta">already checked: ${esc(checks)}</div>`:''}<div class="row"><span class="op">${op}</span>${done?'<span class="op" style="background:rgba(0,255,156,.15);color:#00ff9c">DONE</span>':''}<span class="age">${_tpHumanAge(age)}</span>${done?'':`<button class="act" onclick="_tpMarkDone('${path.replace(/'/g,"\\\\'")}')">MARK TESTED</button>`}</div></div>`
  }).join('');
}
async function _tpMarkDone(path){
  try{const r=await fetch('/memory/needs-testing/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path_substring:path})});const j=await r.json();bubble('bot','Marked '+(j.marked_done||0)+' item(s) tested for `'+esc(path)+'`','<span class="badge">tested</span>');_pollTestsList()}
  catch(e){bubble('bot','Could not mark tested: '+esc(e.message),'<span class="badge err">err</span>')}
}
function _startTestsPolling(){if(_tpPollTimer)return;_pollTestsList();_tpPollTimer=setInterval(_pollTestsList,15000)}
_startTestsPolling();
let _shItems=[],_shStats={n_total:0,n_errors:0},_shPanelOpen=false,_shPollTimer=null,_shErrorsOnly=false;
function _shHumanAge(s){if(!s||s<60)return Math.round(s||0)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';if(s<86400)return (s/3600).toFixed(1)+'h ago';return (s/86400).toFixed(1)+'d ago'}
async function _pollShellHistory(){
  try{
    const r=await fetch('/memory/shell-history?limit=50&errors_only='+(_shErrorsOnly?'true':'false'));
    if(!r.ok){_shUpdatePill('error',0,0);return}
    const j=await r.json();_shItems=j.items||[];_shStats=j.stats||{n_total:0,n_errors:0};
    _shUpdatePill((_shStats.n_errors||0)>0?'dirty':'clean',_shStats.n_total||0,_shStats.n_errors||0);
    if(_shPanelOpen)_renderShellPanel();
    const sum=document.getElementById('sh-summary');if(sum)sum.textContent=(_shStats.n_total||0)+' total · '+(_shStats.n_errors||0)+' errors';
  }catch{_shUpdatePill('error',0,0)}
}
function _shUpdatePill(state,total,errors){
  const led=document.getElementById('sh-led');const txt=document.getElementById('sh-text');if(!led||!txt)return;
  led.className='sh-led '+state;
  if(state==='error')txt.textContent='shell offline';
  else if(total===0)txt.textContent='shell · idle';
  else if(errors>0)txt.textContent='shell · '+total+' runs · '+errors+' errors';
  else txt.textContent='shell · '+total+' runs';
}
function _shToggleErrors(){_shErrorsOnly=!_shErrorsOnly;const b=document.getElementById('sh-errors-btn');if(b)b.classList.toggle('on',_shErrorsOnly);_pollShellHistory()}
function toggleShellPanel(){_shPanelOpen=!_shPanelOpen;const p=document.getElementById('shell-panel');p.classList.toggle('show',_shPanelOpen);['persona-panel','learn-panel','tests-panel','coach-panel'].forEach(id=>{const el=document.getElementById(id);if(_shPanelOpen&&el&&el.classList.contains('show'))el.classList.remove('show')});if(_shPanelOpen){_personaPanelOpen=false;_ldPanelOpen=false;_tpPanelOpen=false;_coachPanelOpen=false;document.getElementById('coach-toggle').classList.remove('on');_pollShellHistory()}}
function _renderShellPanel(){
  const list=document.getElementById('sh-list');
  if(!_shItems.length){list.innerHTML='<div class="sh-empty">no shell commands run yet</div>';return}
  const now=Date.now()/1000;
  list.innerHTML=_shItems.map((it,i)=>{
    const rc=it.returncode||0;const ok=rc===0;const cls=ok?'':' fail';
    const cmd=esc(it.cmd||'?');const kind=esc(it.kind||'shell').toUpperCase();
    const age=now-(it.ts||now);const cwd=esc((it.cwd||'').slice(-50));const dur=it.duration_s!=null?it.duration_s.toFixed(2)+'s':'?';
    const out=esc((it.stdout_tail||'').slice(0,1000));const err=esc((it.stderr_tail||'').slice(0,500));
    const hasOut=out.length>0||err.length>0;
    return `<div class="sh-item${cls}"><div class="cmd">$ ${cmd}</div><div class="meta"><span class="kind">${kind}</span><span class="rc ${ok?'ok':'bad'}">rc ${rc}</span><span>${dur}</span>${cwd?`<span title="${cwd}">${cwd.length>40?'…'+cwd.slice(-40):cwd}</span>`:''}<span class="age">${_shHumanAge(age)}</span></div>${hasOut?`<span class="toggle" onclick="_shToggleOut(${i})">SHOW OUTPUT</span><pre id="sh-out-${i}">${out}${err?`\n--- STDERR ---\n${err}`:''}</pre>`:''}</div>`;
  }).join('');
}
function _shToggleOut(i){const el=document.getElementById('sh-out-'+i);if(!el)return;el.classList.toggle('show');const tog=el.previousElementSibling;if(tog&&tog.classList.contains('toggle'))tog.textContent=el.classList.contains('show')?'HIDE OUTPUT':'SHOW OUTPUT'}
function _startShellPolling(){if(_shPollTimer)return;_pollShellHistory();_shPollTimer=setInterval(_pollShellHistory,20000)}
_startShellPolling();
const NOTIF_VOICE_KEY='amni_jarvis_notif_voice';
let _notifShown=new Set(),_notifPollTimer=null,_notifVoiceOn=localStorage.getItem(NOTIF_VOICE_KEY)==='1';
function _notifHumanAge(s){if(!s||s<60)return Math.round(s||0)+'s';if(s<3600)return Math.round(s/60)+'m';return (s/3600).toFixed(1)+'h'}
async function _pollNotifications(){
  try{const r=await fetch('/notifications?limit=10');if(!r.ok)return;const j=await r.json();const items=j.items||[];
    for(const n of items){if(_notifShown.has(n.id))continue;_notifShown.add(n.id);_showToast(n)}
  }catch{}
}
function _personaToastTint(){
  const cur=(_selectedPersona||personaName||'').toLowerCase();
  if(!cur||!_knownPersonas||_knownPersonas.length===0)return null;
  const p=_knownPersonas.find(x=>typeof x==='object'&&(x.name||'').toLowerCase()===cur);
  if(!p||typeof p!=='object')return null;
  const warmth=Number(p.warmth||0),excitement=Number(p.excitement||0),formality=Number(p.formality||0);
  if(excitement>=0.55&&excitement>=warmth)return {hex:'#ff2bd6',rgb:'255,43,214',name:'spirited'};
  if(warmth>=0.7)return {hex:'#ffb547',rgb:'255,181,71',name:'warm'};
  if(formality>=0.7)return {hex:'#9fb8c8',rgb:'159,184,200',name:'formal'};
  return null;
}
function _showToast(n){
  const stack=document.getElementById('toast-stack');if(!stack)return;
  const el=document.createElement('div');el.className='toast '+(n.level||'info');el.dataset.id=n.id;
  el.innerHTML=`<div class="t-head"><span class="t-src">${esc(n.source||'')}</span><span class="t-age">${_notifHumanAge(n.age_s||0)} ago</span><span class="t-close" onclick="event.stopPropagation();_dismissToast('${n.id}')">✕</span></div><div class="t-title">${esc(n.title||'')}</div>${n.body?`<div class="t-body">${esc(n.body)}</div>`:''}`;
  const lvl=(n.level||'info');
  if(lvl==='info'||lvl==='success'){
    const tint=_personaToastTint();
    if(tint){
      el.classList.add('persona-tint');el.dataset.personaTint=tint.name;
      el.style.borderLeftColor=tint.hex;el.style.borderColor=`rgba(${tint.rgb},.35)`;el.style.boxShadow=`0 0 12px rgba(${tint.rgb},.22)`;
      const src=el.querySelector('.t-src');if(src)src.style.color=tint.hex;
    }
  }
  el.onclick=()=>{_dismissToast(n.id);if(n.body)bubble('bot','**'+esc(n.source||'')+'** · '+esc(n.title||'')+'\n\n'+esc(n.body),'<span class="badge">notif</span>')};
  stack.appendChild(el);
  requestAnimationFrame(()=>el.classList.add('show'));
  if(_notifVoiceOn&&_voiceBackends.tts&&n.level!=='warn'){const phrase=(n.title||'')+(n.body?'. '+n.body.slice(0,140):'');speak(phrase)}
  setTimeout(()=>{if(el.parentElement)_dismissToast(n.id)},(n.level==='error'?14000:n.level==='warn'?10000:7000));
}
function _dismissToast(id){
  const el=document.querySelector(`.toast[data-id="${id}"]`);if(!el)return;
  el.classList.remove('show');el.classList.add('dismiss');
  try{fetch(`/notifications/${id}/read`,{method:'POST'})}catch{}
  setTimeout(()=>el.remove(),400);
}
function toggleNotifVoice(){_notifVoiceOn=!_notifVoiceOn;localStorage.setItem(NOTIF_VOICE_KEY,_notifVoiceOn?'1':'0');bubble('bot','Proactive notification voice **'+(_notifVoiceOn?'on':'off')+'**.','<span class="badge">notif</span>')}
function _startNotifPolling(){if(_notifPollTimer)return;_pollNotifications();_notifPollTimer=setInterval(_pollNotifications,10000)}
_startNotifPolling();
const CORE_KEY='amni_jarvis_core_state';
let _coreCollapsed=localStorage.getItem(CORE_KEY)==='collapsed';
let _corePulses=[],_coreFrame=0,_coreLastToken=0;
function _coreInit(){
  const el=document.getElementById('adam-core');if(!el)return;
  if(_coreCollapsed)el.classList.add('collapsed');
  el.addEventListener('click',_coreToggle);
  requestAnimationFrame(_coreDraw);
}
function _coreToggle(){_coreCollapsed=!_coreCollapsed;const el=document.getElementById('adam-core');el.classList.toggle('collapsed',_coreCollapsed);localStorage.setItem(CORE_KEY,_coreCollapsed?'collapsed':'normal')}
function _coreColor(){
  if(convoState==='speaking')return ['#ffd770','rgba(255,215,112,'];
  if(convoState==='thinking'||convoState==='transcribing')return ['#ff4dc8','rgba(255,77,200,'];
  if(convoState==='recording')return [_TC.hexC,'rgba('+_TC.c+','];
  if(convoState==='error')return ['#ff5b5b','rgba(255,91,91,'];
  const tint=(typeof _personaToastTint==='function')?_personaToastTint():null;
  if(tint)return [tint.hex,`rgba(${tint.rgb},`];
  if(convoOn)return [_TC.hexC,'rgba('+_TC.c+','];
  return ['#7ad6ff','rgba(122,214,255,'];
}
function _corePulse(){_corePulses.push({t:performance.now(),life:900})}
function _coreDraw(){
  _coreFrame++;
  const el=document.getElementById('adam-core');if(!el){requestAnimationFrame(_coreDraw);return}
  const ctx=el.getContext('2d');const W=el.width,H=el.height,cx=W/2,cy=H/2;
  ctx.clearRect(0,0,W,H);
  const [stroke,rgba]=_coreColor();const now=performance.now();
  ctx.save();ctx.translate(cx,cy);ctx.rotate((_coreFrame*0.012)%(Math.PI*2));
  ctx.strokeStyle=rgba+'0.55)';ctx.lineWidth=1.5;
  for(let i=0;i<6;i++){
    const a0=(i/6)*Math.PI*2;const a1=a0+0.55;
    ctx.beginPath();ctx.arc(0,0,46,a0,a1);ctx.stroke();
  }
  ctx.restore();
  ctx.save();ctx.translate(cx,cy);ctx.rotate(-(_coreFrame*0.022)%(Math.PI*2));
  ctx.strokeStyle=rgba+'0.7)';ctx.lineWidth=1.2;
  for(let i=0;i<4;i++){
    const a0=(i/4)*Math.PI*2;const a1=a0+0.35;
    ctx.beginPath();ctx.arc(0,0,33,a0,a1);ctx.stroke();
  }
  ctx.restore();
  ctx.strokeStyle=rgba+'0.18)';ctx.lineWidth=1;
  ctx.beginPath();ctx.arc(cx,cy,52,0,Math.PI*2);ctx.stroke();
  ctx.beginPath();ctx.arc(cx,cy,40,0,Math.PI*2);ctx.stroke();
  ctx.beginPath();ctx.arc(cx,cy,26,0,Math.PI*2);ctx.stroke();
  _corePulses=_corePulses.filter(p=>now-p.t<p.life);
  for(const p of _corePulses){
    const k=(now-p.t)/p.life;const r=18+k*38;const a=Math.max(0,1-k);
    ctx.strokeStyle=rgba+(a*0.5).toFixed(3)+')';ctx.lineWidth=1.8;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();
  }
  const breath=0.5+0.5*Math.sin(_coreFrame*0.04);
  const dotR=8+breath*3;
  ctx.fillStyle=stroke;ctx.shadowColor=stroke;ctx.shadowBlur=12+breath*8;
  ctx.beginPath();ctx.arc(cx,cy,dotR,0,Math.PI*2);ctx.fill();
  ctx.shadowBlur=0;
  if(convoState==='recording'||convoState==='listening'){
    ctx.fillStyle=rgba+'0.85)';
    for(let i=0;i<3;i++){
      const a=(_coreFrame*0.06+i*2.094)%(Math.PI*2);
      const x=cx+Math.cos(a)*18,y=cy+Math.sin(a)*18;
      ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill();
    }
  }
  requestAnimationFrame(_coreDraw);
}
_coreInit();
const SESSION_RESTORE_LIMIT=24;
async function _restoreSession(){
  if(!sid)return;
  try{
    const r=await fetch('/sessions/'+encodeURIComponent(sid)+'?limit='+SESSION_RESTORE_LIMIT);
    if(!r.ok)return;
    const j=await r.json();const turns=j.turns||[];
    if(turns.length===0)return;
    const w=document.querySelector('.welcome');if(w)w.remove();
    const banner=document.createElement('div');banner.className='restore-banner';banner.innerHTML=`<span>↻ ${turns.length} TURN${turns.length===1?'':'S'} · ${esc(sid.slice(-8))}</span><span class="rb-close" onclick="this.parentElement.remove()">×</span>`;log.appendChild(banner);setTimeout(()=>{if(banner.parentElement)banner.classList.add('rb-fade')},8000);
    for(const t of turns){
      const role=t.role==='assistant'?'bot':(t.role==='user'?'user':null);if(!role)continue;
      const text=t.content||t.message||'';if(!text)continue;
      const meta=t.metadata||{};const tier=meta.tier||t.tier||'';const cat=meta.category||t.category||'';
      let metaHtml='';if(tier||cat){const parts=[];if(tier)parts.push(`<span class="badge">${esc(tier)}</span>`);if(cat&&cat!=='general')parts.push(`<span class="badge">${esc(cat)}</span>`);metaHtml=parts.join(' ')}
      const m=document.createElement('div');m.className='msg '+role+' restored';
      const b=document.createElement('div');b.className='bubble';
      if(role==='bot')b.innerHTML=md(text);else b.textContent=text;
      m.appendChild(b);
      if(metaHtml){const mt=document.createElement('div');mt.className='meta';mt.innerHTML=metaHtml;m.appendChild(mt)}
      log.appendChild(m);
    }
    log.scrollTop=log.scrollHeight;
  }catch(e){console.debug('session restore skipped:',e)}
}
setTimeout(_restoreSession,200);
let _csOpen=false,_csHits=[],_csIdx=-1;
function openChatSearch(){_csOpen=true;const el=document.getElementById('chat-search');el.classList.add('show');const inp=document.getElementById('cs-input');inp.value='';_csHits=[];_csIdx=-1;_csClearHighlights();_csUpdateCount(0,0);setTimeout(()=>inp.focus(),60)}
function closeChatSearch(){_csOpen=false;const el=document.getElementById('chat-search');el.classList.remove('show');_csClearHighlights();document.querySelectorAll('.msg.cs-hidden').forEach(m=>m.classList.remove('cs-hidden'))}
function _csClearHighlights(){document.querySelectorAll('mark.cs-hit').forEach(m=>{const p=m.parentNode;if(!p)return;p.replaceChild(document.createTextNode(m.textContent),m);p.normalize()})}
function _csEscapeRe(s){return (s||'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}
function _csHighlightNode(node,re){
  if(node.nodeType===3){
    const t=node.nodeValue;if(!t)return 0;
    const matches=[...t.matchAll(re)];if(!matches.length)return 0;
    const frag=document.createDocumentFragment();let last=0;
    for(const m of matches){if(m.index>last)frag.appendChild(document.createTextNode(t.slice(last,m.index)));const mk=document.createElement('mark');mk.className='cs-hit';mk.textContent=m[0];frag.appendChild(mk);last=m.index+m[0].length}
    if(last<t.length)frag.appendChild(document.createTextNode(t.slice(last)));
    node.parentNode.replaceChild(frag,node);return matches.length;
  }
  if(node.nodeType===1 && !['SCRIPT','STYLE','MARK','INPUT','TEXTAREA','BUTTON'].includes(node.nodeName)){
    let total=0;const kids=Array.from(node.childNodes);for(const c of kids)total+=_csHighlightNode(c,re);return total;
  }
  return 0;
}
function _csRunSearch(){
  if(!_csOpen)return;
  const q=(document.getElementById('cs-input').value||'').trim();
  _csClearHighlights();
  const msgs=document.querySelectorAll('#log .msg');
  if(!q){msgs.forEach(m=>m.classList.remove('cs-hidden'));_csHits=[];_csIdx=-1;_csUpdateCount(0,0);return}
  const re=new RegExp(_csEscapeRe(q),'gi');let totalHits=0;let msgHits=0;
  msgs.forEach(m=>{
    const txt=(m.textContent||'').toLowerCase();
    if(txt.indexOf(q.toLowerCase())===-1){m.classList.add('cs-hidden');return}
    m.classList.remove('cs-hidden');msgHits++;totalHits+=_csHighlightNode(m,re);
  });
  _csHits=Array.from(document.querySelectorAll('mark.cs-hit'));_csIdx=_csHits.length?0:-1;
  _csUpdateCount(_csIdx+1,_csHits.length);_csMarkCurrent();_csScrollToCurrent();
}
function _csUpdateCount(cur,total){const el=document.getElementById('cs-count');if(el)el.textContent=total?cur+'/'+total:'0/0'}
function _csMarkCurrent(){_csHits.forEach((m,i)=>m.classList.toggle('current',i===_csIdx))}
function _csScrollToCurrent(){if(_csIdx<0||!_csHits[_csIdx])return;_csHits[_csIdx].scrollIntoView({block:'center',behavior:'smooth'})}
function _csNext(){if(!_csHits.length)return;_csIdx=(_csIdx+1)%_csHits.length;_csUpdateCount(_csIdx+1,_csHits.length);_csMarkCurrent();_csScrollToCurrent()}
function _csPrev(){if(!_csHits.length)return;_csIdx=(_csIdx-1+_csHits.length)%_csHits.length;_csUpdateCount(_csIdx+1,_csHits.length);_csMarkCurrent();_csScrollToCurrent()}
document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='k'){e.preventDefault();_csOpen?closeChatSearch():openChatSearch();return}
  if(_csOpen){
    if(e.key==='Escape'){e.preventDefault();closeChatSearch();return}
    if(e.key==='Enter'){e.preventDefault();e.shiftKey?_csPrev():_csNext();return}
  }
  if(e.altKey&&(e.key==='p'||e.key==='P')&&!e.ctrlKey&&!e.metaKey){
    const tag=(document.activeElement||{}).tagName||'';
    if(tag==='INPUT'||tag==='TEXTAREA'||(document.activeElement||{}).isContentEditable)return;
    e.preventDefault();_personaCycle(e.shiftKey?-1:1);
  }
});
async function _personaCycle(step){
  step=step||1;
  if(!_knownPersonas||_knownPersonas.length<2){
    if(_knownPersonas.length===0){try{await _loadPersonas()}catch{}if(_knownPersonas.length===0)return}
  }
  if(typeof _corePulse==='function')_corePulse();
  const names=_knownPersonas.map(p=>(typeof p==='string'?p:(p.name||''))).filter(Boolean);
  if(names.length<2)return;
  const cur=(_selectedPersona||personaName||names[0]).toLowerCase();
  let idx=names.findIndex(n=>n.toLowerCase()===cur);
  if(idx<0)idx=0;
  const next=names[(idx+step+names.length)%names.length];
  _personaFlash('persona → '+next);
  try{
    const r=await fetch('/persona',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:next,session_id:sid,learn_via_web:false})});
    const j=await r.json();
    if(r.ok&&j.persona){_selectedPersona=j.persona.name;localStorage.setItem(PERSONA_KEY,_selectedPersona);personaName=_selectedPersona;personaPill.textContent='persona '+_selectedPersona;_renderPersonaPanel();if(typeof _applyWelcomeForPersona==='function')try{_applyWelcomeForPersona()}catch(_){}}
  }catch(_){}
}
let _personaFlashTimer=null;
function _personaFlash(text){
  let el=document.getElementById('persona-flash');
  if(!el){el=document.createElement('div');el.id='persona-flash';el.className='persona-flash';document.body.appendChild(el)}
  el.textContent=text;el.classList.remove('show');void el.offsetWidth;el.classList.add('show');
  const tint=_personaToastTint();
  if(tint){el.style.borderColor=tint.hex;el.style.color=tint.hex;el.style.textShadow=`0 0 8px rgba(${tint.rgb},.7)`;el.style.boxShadow=`0 0 24px rgba(${tint.rgb},.35),inset 0 0 14px rgba(${tint.rgb},.08)`}
  else{el.style.borderColor='';el.style.color='';el.style.textShadow='';el.style.boxShadow=''}
  clearTimeout(_personaFlashTimer);_personaFlashTimer=setTimeout(()=>el.classList.remove('show'),1400);
}
document.addEventListener('input',e=>{if(e.target&&e.target.id==='cs-input')_csRunSearch()});
async function _exportChatMd(){
  if(!sid){bubble('bot','No active session to export yet — start a conversation first.','<span class="badge err">export</span>');return}
  try{
    const r=await fetch('/sessions/'+encodeURIComponent(sid)+'/export.md');
    if(!r.ok){bubble('bot','Export failed: '+r.status+' '+r.statusText,'<span class="badge err">export</span>');return}
    const md=await r.text();const blob=new Blob([md],{type:'text/markdown;charset=utf-8'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');
    a.href=url;a.download='adam-chat-'+sid.slice(-8)+'.md';document.body.appendChild(a);a.click();document.body.removeChild(a);
    setTimeout(()=>URL.revokeObjectURL(url),2000);
    const lines=md.split('\n').length;bubble('bot','Exported **'+lines+' lines** to `adam-chat-'+esc(sid.slice(-8))+'.md` (download started).','<span class="badge">export</span>');
  }catch(e){bubble('bot','Export error: '+esc(e.message),'<span class="badge err">export</span>')}
}
const COACH_SID_KEY='amni_jarvis_coach_sid',COACH_VOICE_KEY='amni_jarvis_coach_voice';
let _coachSid=localStorage.getItem(COACH_SID_KEY)||'',_coachPanelOpen=false,_coachTopic='',_coachBusy=false,_coachVoiceOn=localStorage.getItem(COACH_VOICE_KEY)==='1',_coachLastQuestion='';
function _coachToggleVoice(){_coachVoiceOn=!_coachVoiceOn;localStorage.setItem(COACH_VOICE_KEY,_coachVoiceOn?'1':'0');_coachUpdateVoiceBtn();if(_coachVoiceOn){voiceOut=true;localStorage.setItem(VKEY,'1');const vb=document.getElementById('voiceout-toggle');if(vb)vb.classList.add('on');if(_coachLastQuestion)speak(_coachLastQuestion)}}
function _coachUpdateVoiceBtn(){const b=document.getElementById('cp-voice-toggle');if(!b)return;b.textContent=_coachVoiceOn?'VOICE ON':'VOICE OFF';b.classList.toggle('on',_coachVoiceOn)}
function _coachReplayQuestion(){if(_coachLastQuestion)speak(_coachLastQuestion);else bubble('bot','No active question to replay.','<span class="badge err">coach</span>')}
function _coachSpeakIfOn(text){if(_coachVoiceOn&&text&&text.trim()&&_voiceBackends.tts)speak(text)}
function toggleCoachPanel(){_coachPanelOpen=!_coachPanelOpen;const p=document.getElementById('coach-panel');p.classList.toggle('show',_coachPanelOpen);document.getElementById('coach-toggle').classList.toggle('on',_coachPanelOpen);['persona-panel','learn-panel','tests-panel'].forEach(id=>{const el=document.getElementById(id);if(_coachPanelOpen&&el&&el.classList.contains('show'))el.classList.remove('show')});if(_coachPanelOpen){_personaPanelOpen=false;_ldPanelOpen=false;_tpPanelOpen=false;_coachUpdateVoiceBtn();_coachLoadTopics();if(_coachSid)_coachSyncStatus()}}
async function _coachLoadReviews(){
  const sec=document.getElementById('cp-reviews-section');const list=document.getElementById('cp-reviews-list');const ct=document.getElementById('cp-reviews-count');if(!sec||!list)return;
  try{
    const r=await fetch('/memory/coach/reviews?limit=10');if(!r.ok){sec.style.display='none';return}
    const j=await r.json();const reviews=j.reviews||[];
    if(!reviews.length){sec.style.display='none';return}
    sec.style.display='block';if(ct)ct.textContent=reviews.length+' due';
    list.innerHTML=reviews.map((rv,i)=>{const urgent=rv.overdue_ratio>=2?' urgent':'';
      return `<div class="cp-review-card${urgent}" onclick="_coachStartReview(${i})" title="Click to re-ask this question now"><div class="rv-topic">${esc(rv.topic)}</div><div class="rv-q">${esc(rv.question)}</div><div class="rv-meta"><span>last: ${rv.last_score}/100 · ${rv.age_days}d ago</span><span class="rv-overdue">${rv.overdue_ratio}× overdue</span></div></div>`
    }).join('');
    window._coachReviewItems=reviews;
  }catch{sec.style.display='none'}
}
async function _coachLoadTopics(){
  const list=document.getElementById('cp-topics-list');if(!list)return;
  _coachLoadReviews();
  try{
    const r=await fetch('/memory/coach');if(!r.ok){list.innerHTML='<div style="font-size:10px;color:var(--mute);font-style:italic;text-align:center;padding:6px">coach memory unavailable</div>';return}
    const j=await r.json();const topics=j.topics||[];_coachUpdateStreakBadge(j.streak||{});
    if(!topics.length){list.innerHTML='<div style="font-size:10px;color:var(--mute);font-style:italic;text-align:center;padding:6px">no topics practiced yet · start a session above</div>';return}
    list.innerHTML=topics.slice(0,20).map(t=>{
      const pct=Math.round(t.mastery_pct||0);
      const lvl=pct>=85?'master':(pct>=65?'good':(pct>=40?'fair':'novice'));
      const name=esc(t.topic||'?');const safe=name.replace(/'/g,"\\\\'");
      return `<div class="cp-topic-card lvl-${lvl}" onclick="_coachResumeTopic('${safe}')" title="Click to start a new session on this topic"><span class="name">${name}</span><span class="mini-bar"><span class="mini-bar-fill" style="width:${pct}%"></span></span><span class="pct">${pct}%</span><span class="n">${t.n_questions||0}q</span><button class="tc-export" title="Export deck (Markdown)" onclick="event.stopPropagation();_coachExportTopic('${safe}')">⬇</button></div>`
    }).join('');
  }catch(e){list.innerHTML='<div style="font-size:10px;color:var(--err);font-style:italic;text-align:center;padding:6px">load error</div>'}
}
function _coachUpdateStreakBadge(s){
  const b=document.getElementById('cp-streak-badge');if(!b)return;
  const cur=s.current_streak||0;const best=s.best_streak||0;const total=s.total_days_active||0;
  if(cur===0){b.classList.remove('active','fire','elite');b.textContent='—';return}
  b.classList.add('active');b.classList.remove('fire','elite');
  if(cur>=14)b.classList.add('elite');else if(cur>=3)b.classList.add('fire');
  const flame=cur>=14?'⚡':(cur>=3?'🔥':'·');
  b.textContent=flame+' '+cur+' day'+(cur===1?'':'s');
  b.title=`current streak ${cur} day${cur===1?'':'s'} · best ${best} · ${total} active days total${s.today_active?' · today ✓':''}`;
}
async function _coachExportTopic(topic){
  if(!topic)return;
  try{
    const url='/memory/coach/topic/'+encodeURIComponent(topic)+'/export.md';
    const r=await fetch(url);if(!r.ok){bubble('bot','Export failed: '+r.status+' '+r.statusText,'<span class="badge err">coach</span>');return}
    const md=await r.text();const blob=new Blob([md],{type:'text/markdown;charset=utf-8'});
    const u=URL.createObjectURL(blob);const a=document.createElement('a');
    a.href=u;a.download='coach-'+topic.replace(/[^a-z0-9_\-]+/gi,'-').toLowerCase()+'.md';
    document.body.appendChild(a);a.click();document.body.removeChild(a);
    setTimeout(()=>URL.revokeObjectURL(u),2000);
    const cards=md.match(/## Card \d+/g);const n=cards?cards.length:0;
    bubble('bot','Exported **'+n+' card'+(n===1?'':'s')+'** from **'+esc(topic)+'** to your downloads.','<span class="badge">coach</span>');
  }catch(e){bubble('bot','Export error: '+esc(e.message),'<span class="badge err">coach</span>')}
}
async function _coachStartReview(idx){
  const items=window._coachReviewItems||[];const rv=items[idx];if(!rv){bubble('bot','Review card no longer available — refresh the coach panel.','<span class="badge err">coach</span>');return}
  bubble('bot','Re-asking due card from **'+esc(rv.topic)+'** ('+rv.age_days+'d since last attempt, '+rv.overdue_ratio+'× overdue)…','<span class="badge">review</span>');
  const res=await _coachCall({action:'start',topic:rv.topic,difficulty:rv.last_difficulty||2,seed_question:rv.question});
  if(!res||res.error){bubble('bot','Could not start review: '+esc(res&&res.error||'unknown'),'<span class="badge err">coach</span>');return}
  _coachSid=res.session_id;_coachTopic=res.topic;localStorage.setItem(COACH_SID_KEY,_coachSid);
  _coachShowActive(true);document.getElementById('cp-grade-slot').innerHTML='';document.getElementById('cp-answer').value='';
  _coachRender(res);_coachLastQuestion=res.question||'';_coachSpeakIfOn(res.question);
  setTimeout(_coachLoadTopics,400);
}
function _coachResumeTopic(topic){
  const t=document.getElementById('cp-topic');if(t){t.value=topic;t.focus()}
  bubble('bot','Topic loaded: **'+esc(topic)+'**. Click START to begin a new session on it.','<span class="badge">coach</span>');
}
function _coachShowActive(show){document.getElementById('cp-start-section').style.display=show?'none':'block';document.getElementById('cp-active-section').style.display=show?'block':'none'}
function _coachRender(res){
  if(!res)return;
  const topic=res.topic||_coachTopic||'?';
  document.getElementById('cp-topic-head').textContent='TOPIC · '+topic.toUpperCase();
  const q=res.question||res.next_question||res.pending_question||'(no question)';
  const qEl=document.getElementById('cp-q');qEl.textContent=q;qEl.className='cp-q'+(q.startsWith('(')?' empty':'');
  document.getElementById('cp-diff-v').textContent=res.difficulty||'?';
  const s=res._session||{};const m=res.mastery||{};
  document.getElementById('cp-asked-v').textContent=(s.n_answered!=null?s.n_answered:m.n_answered||0);
  document.getElementById('cp-streak-correct').textContent=(s.streak_correct||0)+' ✓';
  document.getElementById('cp-streak-wrong').textContent=(s.streak_wrong||0)+' ✗';
  const pct=Math.round((m.pct||0)*100)/100||0;
  document.getElementById('cp-mastery-pct').textContent=pct.toFixed(0)+'%';
  document.getElementById('cp-mastery-fill').style.width=pct+'%';
  if(res.score!=null){
    const cls=res.score>=70?'good':(res.score>=50?'ok':'bad');
    const cf=Array.isArray(res.correct_facts)?res.correct_facts:[];const mf=Array.isArray(res.missing_facts)?res.missing_facts:[];
    document.getElementById('cp-grade-slot').innerHTML=`<div class="cp-grade ${cls}"><span class="score">${res.score}/100</span>${esc(res.feedback||'')}${cf.length?'<div style="margin-top:6px;font-size:10px;color:#a0ffd0">✓ correct: '+cf.map(esc).join(', ')+'</div>':''}${mf.length?'<div style="font-size:10px;color:#ffb0b0">✗ missing: '+mf.map(esc).join(', ')+'</div>':''}</div>`;
  }
  const hintEl=document.getElementById('cp-hint');if(res.hint){hintEl.textContent='💡 '+res.hint;hintEl.style.display='block'}else hintEl.style.display='none';
}
async function _coachCall(args){
  if(_coachBusy)return null;_coachBusy=true;
  try{const r=await fetch('/skills/coach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args})});const j=await r.json();const out=j.output||j;return out}
  catch(e){bubble('bot','Coach call failed: '+esc(e.message),'<span class="badge err">coach</span>');return null}
  finally{_coachBusy=false}
}
async function _coachStart(){
  const t=(document.getElementById('cp-topic').value||'').trim();if(!t){bubble('bot','Coach needs a topic. Try "python decorators" or "krebs cycle".','<span class="badge err">coach</span>');return}
  const d=parseInt(document.getElementById('cp-diff').value||'2',10);
  bubble('bot','Starting coaching session on **'+esc(t)+'** at difficulty '+d+'. Adam is composing your first question…','<span class="badge">coach</span>');
  const res=await _coachCall({action:'start',topic:t,difficulty:d});
  if(!res||res.error){bubble('bot','Could not start: '+esc(res&&res.error||'unknown'),'<span class="badge err">coach</span>');return}
  _coachSid=res.session_id;_coachTopic=res.topic;localStorage.setItem(COACH_SID_KEY,_coachSid);
  _coachShowActive(true);document.getElementById('cp-grade-slot').innerHTML='';document.getElementById('cp-answer').value='';
  _coachRender(res);_coachLastQuestion=res.question||'';_coachSpeakIfOn(res.question);
  setTimeout(_coachLoadTopics,300);
}
async function _coachAnswer(){
  if(!_coachSid){bubble('bot','No active coach session. Start one first.','<span class="badge err">coach</span>');return}
  const a=(document.getElementById('cp-answer').value||'').trim();if(!a){bubble('bot','Type an answer first.','<span class="badge err">coach</span>');return}
  const res=await _coachCall({action:'answer',session_id:_coachSid,answer:a});
  if(!res||res.error){bubble('bot','Grade failed: '+esc(res&&res.error||'unknown'),'<span class="badge err">coach</span>');return}
  _coachRender(res);document.getElementById('cp-answer').value='';
  if(_coachVoiceOn){const phrase=(res.feedback||'')+(res.next_question?(' Next question: '+res.next_question):'');_coachSpeakIfOn(phrase)}
  else if(voiceOut&&res.feedback)speak(res.feedback);
  if(res.next_question)_coachLastQuestion=res.next_question;
}
async function _coachHint(){if(!_coachSid)return;const res=await _coachCall({action:'hint',session_id:_coachSid});if(res&&res.hint){const el=document.getElementById('cp-hint');el.textContent='💡 '+res.hint;el.style.display='block';_coachSpeakIfOn('Hint: '+res.hint)}}
async function _coachSkip(){if(!_coachSid)return;const res=await _coachCall({action:'skip',session_id:_coachSid});_coachRender(res||{});document.getElementById('cp-grade-slot').innerHTML='';document.getElementById('cp-answer').value='';if(res&&res.next_question){_coachLastQuestion=res.next_question;_coachSpeakIfOn('Skipped. Next question: '+res.next_question)}}
async function _coachAsk(){if(!_coachSid)return;const res=await _coachCall({action:'ask',session_id:_coachSid});_coachRender(res||{});document.getElementById('cp-grade-slot').innerHTML='';document.getElementById('cp-answer').value='';if(res&&res.question){_coachLastQuestion=res.question;_coachSpeakIfOn(res.question)}}
async function _coachEnd(){
  if(!_coachSid)return;const res=await _coachCall({action:'summary',session_id:_coachSid});
  const top=res&&res.topic||_coachTopic||'?';const pct=res&&res.mastery&&res.mastery.pct!=null?Math.round(res.mastery.pct):0;
  bubble('bot','**Coach session complete:** '+esc(top)+' · final mastery '+pct+'% · '+(res&&res.n_answered||0)+' answered.','<span class="badge">coach</span>');
  localStorage.removeItem(COACH_SID_KEY);_coachSid='';_coachTopic='';_coachShowActive(false);
  _coachLoadTopics();
}
async function _coachSyncStatus(){if(!_coachSid)return;const res=await _coachCall({action:'status',session_id:_coachSid});if(res&&!res.error){_coachShowActive(true);_coachRender(res)}else{localStorage.removeItem(COACH_SID_KEY);_coachSid=''}}
document.addEventListener('keydown',e=>{if(_coachPanelOpen&&e.ctrlKey&&e.key==='Enter'&&document.activeElement&&document.activeElement.id==='cp-answer'){e.preventDefault();_coachAnswer()}});
function toggleVoiceOut(){voiceOut=!voiceOut;localStorage.setItem(VKEY,voiceOut?'1':'0');const el=document.getElementById('voiceout-toggle');el.classList.toggle('on',voiceOut)}
let _toolsDrawerOpen=false;
function toggleToolsDrawer(force){
  const open=(typeof force==='boolean')?force:!_toolsDrawerOpen;
  _toolsDrawerOpen=open;
  const el=document.getElementById('tools-drawer');if(el)el.classList.toggle('show',open);
  const btn=document.getElementById('tools-toggle');if(btn)btn.classList.toggle('on',open);
}
const QB_KEY='amni_jarvis_qb_expanded';
let _qbExpanded=localStorage.getItem(QB_KEY)==='1';
function toggleQuickBar(force){
  const open=(typeof force==='boolean')?force:!_qbExpanded;
  _qbExpanded=open;localStorage.setItem(QB_KEY,open?'1':'0');
  const el=document.getElementById('quick-bar');if(el)el.classList.toggle('qb-collapsed',!open);
  const lbl=document.getElementById('qb-toggle-label');if(lbl)lbl.textContent=open?'HIDE EXAMPLES':'EXAMPLES';
}
(function(){const el=document.getElementById('quick-bar');if(el&&_qbExpanded){el.classList.remove('qb-collapsed');const lbl=document.getElementById('qb-toggle-label');if(lbl)lbl.textContent='HIDE EXAMPLES'}})();
let _statusPanelOpen=false;
function toggleStatusPanel(force){
  const open=(typeof force==='boolean')?force:!_statusPanelOpen;
  _statusPanelOpen=open;
  const el=document.getElementById('status-panel');if(el)el.classList.toggle('show',open);
  if(open)_refreshStatusRollupLessons();
}
document.addEventListener('click',e=>{
  if(!_statusPanelOpen)return;
  const panel=document.getElementById('status-panel'),trigger=document.getElementById('status-pill');
  if(!panel||!trigger)return;
  if(panel.contains(e.target)||trigger.contains(e.target))return;
  toggleStatusPanel(false);
});
async function _refreshStatusRollupLessons(){
  try{const r=await fetch('/stats');const j=await r.json();const v=document.getElementById('sp-lessons');if(v)v.textContent=(j.lessons_n||0)+' indexed'}catch(_){}
}
function _statusRollupSeverity(){
  const leds=['ld-led','tp-led','sh-led','sfl-led'].map(id=>{const el=document.getElementById(id);return el?(el.className||''):''});
  if(leds.some(c=>c.includes('error')||c.includes('failed')))return 'error';
  if(leds.some(c=>c.includes('pending')||c.includes('paused')))return 'attention';
  if(leds.some(c=>c.includes('active')||c.includes('empty')))return 'ok';
  return '';
}
function _refreshStatusPillBadge(){
  const pill=document.getElementById('status-pill');if(!pill)return;
  const sev=_statusRollupSeverity();
  pill.classList.remove('attention','error','ok');
  if(sev)pill.classList.add(sev);
  const txt=document.getElementById('sp-text');
  if(txt){
    const ldOn=(document.getElementById('ld-led')||{className:''}).className.includes('active');
    const tp=(document.getElementById('tp-text')||{textContent:''}).textContent;
    const m=/tests\s+(\d+)/i.exec(tp||'');const pending=m?m[1]:'0';
    txt.textContent=`status${ldOn?' · live':''}${pending!=='0'?' · '+pending+' pending':''}`;
  }
}
setInterval(_refreshStatusPillBadge,2500);_refreshStatusPillBadge();
let _skillFailuresLastUnacked=0;
async function _refreshSkillFailures(){
  try{
    const r=await fetch('/memory/skill-failures?limit=5');if(!r.ok)return;
    const j=await r.json();const recent=j.failures||[];const stats=j.stats||{};
    const total=stats.total||0;const unacked=stats.unacked||0;
    const txt=document.getElementById('sfl-text');const led=document.getElementById('sfl-led');
    if(led){
      let cls='ld-led idle';
      if(unacked>0){const newCount=unacked-_skillFailuresLastUnacked;cls=newCount>0?'ld-led error':'ld-led paused'}
      led.className=cls;
    }
    if(txt){
      if(total===0){txt.textContent='skill failures — none'}
      else if(unacked===0){txt.textContent=`skill failures · ${total} total · all acked ✓`}
      else{const last=recent[recent.length-1]||{};const lastSkill=(last.skill||'?');txt.textContent=`skill failures · ${unacked} new / ${total} total · last: ${lastSkill}`}
    }
    _skillFailuresLastUnacked=unacked;
  }catch(_){}
}
async function _skillFailuresShow(){
  try{
    const r=await fetch('/memory/skill-failures?limit=10');if(!r.ok){bubble('bot','Skill failures endpoint unreachable.','<span class="badge err">diag</span>');return}
    const j=await r.json();const recent=(j.failures||[]).slice().reverse();const stats=j.stats||{};
    if(recent.length===0){bubble('bot','No skill failures recorded. ✓','<span class="badge">diag</span>');return}
    const rows=recent.slice(0,8).map(f=>{const t=f.iso||'?';const sk=esc(f.skill||'?');const err=esc((f.error||'').slice(0,140));const msg=esc((f.message||'').slice(0,60));return `<div style="margin:6px 0;padding:6px 8px;background:rgba(255,91,91,.04);border-left:2px solid rgba(255,91,91,.4);border-radius:0 3px 3px 0;font-size:10.5px;line-height:1.5;font-family:JetBrains Mono,monospace"><div><span style="color:#ff7b7b;font-weight:600">${sk}</span> <span style="color:var(--mute)">· ${esc(t)}</span></div><div style="color:var(--mute);margin-top:2px">"${msg}"</div><div style="color:#ffb7b7;margin-top:3px">${err}</div></div>`}).join('');
    const unacked=stats.unacked||0;
    const ackBtn=unacked>0?`<button class="se-btn" onclick="_skillFailuresAck()" style="margin-top:8px">✓ MARK ALL ACKED (${unacked} new)</button>`:'';
    const head=`<div style="font-size:9.5px;letter-spacing:.22em;color:var(--mute);text-transform:uppercase;margin-bottom:6px">◆ RECENT SKILL FAILURES · ${stats.total||0} TOTAL · ${unacked} UNACKED</div>`;
    bubble('bot',head+rows+ackBtn,'<span class="badge err">diag</span>');
  }catch(e){bubble('bot','Skill failures: '+esc(_netHint(e)),'<span class="badge err">diag</span>')}
}
async function _skillFailuresAck(){
  try{
    const r=await fetch('/memory/skill-failures/ack',{method:'POST'});
    if(!r.ok){bubble('bot','Ack failed: HTTP '+r.status,'<span class="badge err">diag</span>');return}
    const j=await r.json();
    bubble('bot',`Acknowledged ${j.acked||0} skill failure(s). STATUS pill will clear shortly.`,'<span class="badge">diag</span>');
    _refreshSkillFailures();
  }catch(e){bubble('bot','Ack failed: '+esc(String(e)),'<span class="badge err">diag</span>')}
}
setInterval(_refreshSkillFailures,5000);_refreshSkillFailures();
function _skillErrorRetry(originalMsg){
  if(!originalMsg)return;
  input.value=originalMsg;input.focus();
  bubble('bot','Restored your original message to the input — edit if needed, then press TRANSMIT to retry.','<span class="badge">retry</span>');
}
document.addEventListener('keydown',e=>{
  if(e.key!=='Escape')return;
  if(_streamAbort){e.preventDefault();stopStream();return}
  const gt=document.getElementById('gesture-tour');
  if(gt&&gt.classList.contains('show')){e.preventDefault();_gtClose();return}
  if(_bookmarksPanelOpen){e.preventDefault();toggleBookmarksPanel(false);return}
  if(_remindersPanelOpen){e.preventDefault();toggleRemindersPanel(false);return}
  if(_toolsDrawerOpen){e.preventDefault();toggleToolsDrawer(false);return}
  if(_statusPanelOpen){e.preventDefault();toggleStatusPanel(false);return}
});
document.addEventListener('click',e=>{
  if(!_toolsDrawerOpen)return;
  const drawer=document.getElementById('tools-drawer'),trigger=document.getElementById('tools-toggle');
  if(!drawer||!trigger)return;
  if(drawer.contains(e.target)||trigger.contains(e.target))return;
  toggleToolsDrawer(false);
});
function _jarvisModeActive(){return convoOn&&voiceOut&&wakeOn&&gestureOn}
function toggleJarvisMode(){
  const next=!_jarvisModeActive();
  const setBtn=()=>{const b=document.getElementById('jarvis-toggle');if(b)b.classList.toggle('on',_jarvisModeActive())};
  if(next){
    if(!voiceOut)toggleVoiceOut();
    if(!convoOn)try{toggleConvo()}catch(_){}
    if(!gestureOn)try{toggleGesture()}catch(_){}
    setTimeout(setBtn,200);
    bubble('bot','**Hands-free mode engaged.** Convo + gesture + voice online — just talk, or wave to me. (Tap WAKE in TOOLS if you want it to require "Adam, …" first.)','<span class="badge">hands-free</span>');
  }else{
    if(convoOn)try{toggleConvo()}catch(_){}
    if(gestureOn)try{toggleGesture()}catch(_){}
    if(wakeOn)try{toggleWake()}catch(_){}
    if(voiceOut)try{toggleVoiceOut()}catch(_){}
    setTimeout(setBtn,200);
    bubble('bot','**Hands-free mode disengaged.** Back to manual.','<span class="badge">hands-free</span>');
  }
  setBtn();
}
function _refreshJarvisButton(){const b=document.getElementById('jarvis-toggle');if(b)b.classList.toggle('on',_jarvisModeActive())}
setInterval(_refreshJarvisButton,1500);
let _audioEl=null;let _ttsGen=0;
function _stopAllTTS(){_ttsGen++;if(_audioEl){try{_audioEl.pause();_audioEl.currentTime=0}catch{}}try{if(window.speechSynthesis)speechSynthesis.cancel()}catch{}}
async function speak(text){
  if(!voiceOut)return;
  const clean=text.replace(/```[\s\S]*?```/g,'(code)').replace(/[*_`#<>]/g,'').slice(0,800);
  if(_voiceBackends.tts){
    try{
      const body={text:clean};if(_selectedVoice)body.voice=_selectedVoice;
      const r=await fetch('/voice/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if(r.ok){
        const j=await r.json();
        if(j.audio_base64){
          if(_audioEl){try{_audioEl.pause()}catch{}}
          _audioEl=new Audio('data:'+(j.content_type||'audio/wav')+';base64,'+j.audio_base64);
          _audioEl.play().catch(()=>_speakBrowser(clean));
          return;
        }
      }
    }catch{}
  }
  _speakBrowser(clean);
}
function _personaVoiceTuning(){
  const cur=(_selectedPersona||personaName||'').toLowerCase();
  if(!cur||!_knownPersonas||_knownPersonas.length===0)return null;
  const p=_knownPersonas.find(x=>typeof x==='object'&&(x.name||'').toLowerCase()===cur);
  if(!p||typeof p!=='object')return null;
  if(cur==='yoda')return {rate:0.78,pitch:0.85,prefer:['male','low']};
  if(cur==='haiku')return {rate:0.75,pitch:1.05,prefer:['female','soft']};
  if(cur==='alfred'||cur==='jarvis')return {rate:0.95,pitch:0.9,prefer:['male','british','daniel']};
  if(cur==='rikku')return {rate:1.18,pitch:1.18,prefer:['female','young']};
  if(cur==='pirate')return {rate:1.0,pitch:0.85,prefer:['male','rough']};
  if(cur==='sherlock')return {rate:0.95,pitch:0.95,prefer:['male','british']};
  const ex=Number(p.excitement||0),fo=Number(p.formality||0);
  return {rate:Math.max(0.7,Math.min(1.3,1+ex*0.3-fo*0.15)),pitch:Math.max(0.7,Math.min(1.3,1+(ex-0.5)*0.4)),prefer:[]};
}
let _browserVoices=[];
function _refreshBrowserVoices(){try{_browserVoices=window.speechSynthesis.getVoices()||[]}catch{_browserVoices=[]}}
if('speechSynthesis' in window){_refreshBrowserVoices();try{window.speechSynthesis.onvoiceschanged=_refreshBrowserVoices}catch{}}
function _pickBrowserVoice(prefer){
  if(!_browserVoices||_browserVoices.length===0)return null;
  const lang=(navigator.language||'en').toLowerCase();
  const localized=_browserVoices.filter(v=>(v.lang||'').toLowerCase().startsWith(lang.split('-')[0]));
  const pool=localized.length?localized:_browserVoices;
  for(const tag of (prefer||[])){
    const t=tag.toLowerCase();
    const hit=pool.find(v=>(v.name||'').toLowerCase().includes(t)||((v.voiceURI||'').toLowerCase().includes(t)));
    if(hit)return hit;
  }
  return pool[0]||null;
}
function _speakBrowser(clean){
  if(!('speechSynthesis' in window))return;
  try{
    const u=new SpeechSynthesisUtterance(clean);
    const tune=_personaVoiceTuning();
    if(tune){u.rate=tune.rate;u.pitch=tune.pitch;const v=_pickBrowserVoice(tune.prefer);if(v)u.voice=v}
    else{u.rate=1;u.pitch=1}
    speechSynthesis.cancel();speechSynthesis.speak(u);
  }catch{}
}
let _mediaRec=null,_mediaChunks=[],_recAbort=null;
async function _startServerSTT(){
  const m=document.getElementById('mic-shell');
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    _mediaChunks=[];
    const mime=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':(MediaRecorder.isTypeSupported('audio/webm')?'audio/webm':'');
    _mediaRec=mime?new MediaRecorder(stream,{mimeType:mime}):new MediaRecorder(stream);
    _mediaRec.ondataavailable=e=>{if(e.data.size>0)_mediaChunks.push(e.data)};
    _mediaRec.onstop=async()=>{
      stream.getTracks().forEach(t=>t.stop());
      m.classList.remove('listening');recoOn=false;
      if(_recAbort){_recAbort=null;return}
      if(_mediaChunks.length===0)return;
      const blob=new Blob(_mediaChunks,{type:_mediaRec.mimeType||'audio/webm'});
      const dataUrl=await new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(blob)});
      const b64=dataUrl.split(',',2)[1];
      input.placeholder='transcribing…';
      try{
        const r=await fetch('/voice/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_base64:b64})});
        const j=await r.json();
        if(r.ok&&j.text){input.value=(input.value+' '+j.text).trim();send()}
        else{console.warn('stt fail',j);_startBrowserSTT()}
      }catch(e){console.warn('stt error',e);_startBrowserSTT()}
      finally{input.placeholder='Speak or type...'}
    };
    _mediaRec.start();recoOn=true;m.classList.add('listening');
  }catch(e){console.warn('getUserMedia for STT failed',e);_startBrowserSTT()}
}
function _startBrowserSTT(){
  if(!('webkitSpeechRecognition' in window||'SpeechRecognition' in window)){alert('Voice input needs Chrome/Edge or a server STT backend.');return}
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const m=document.getElementById('mic-shell');
  recog=new SR();recog.lang='en-US';recog.interimResults=false;recog.continuous=false;
  recoOn=true;m.classList.add('listening');
  recog.onresult=e=>{input.value=e.results[0][0].transcript;send()};
  recog.onerror=()=>{recoOn=false;m.classList.remove('listening')};
  recog.onend=()=>{recoOn=false;m.classList.remove('listening')};
  recog.start();
}
function toggleMic(){
  if(recoOn){
    if(_mediaRec&&_mediaRec.state==='recording'){_mediaRec.stop();return}
    if(recog){_recAbort=true;try{recog.stop()}catch{};recoOn=false;document.getElementById('mic-shell').classList.remove('listening');return}
  }
  if(_voiceBackends.stt && typeof MediaRecorder!=='undefined' && navigator.mediaDevices)_startServerSTT();
  else _startBrowserSTT();
}
const _INPUT_DRAFT_KEY='amni_jarvis_input_draft';
(function(){try{const d=localStorage.getItem(_INPUT_DRAFT_KEY);if(d&&!input.value){input.value=d;requestAnimationFrame(()=>{try{input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px'}catch(_){}})}}catch(_){}})();
let _draftSaveTimer=null;
input.addEventListener('input',()=>{
  clearTimeout(_draftSaveTimer);
  _draftSaveTimer=setTimeout(()=>{try{const v=input.value||'';if(v.trim())localStorage.setItem(_INPUT_DRAFT_KEY,v);else localStorage.removeItem(_INPUT_DRAFT_KEY)}catch(_){}},250);
});
function _clearDraft(){try{localStorage.removeItem(_INPUT_DRAFT_KEY)}catch(_){}}
const _INPUT_HISTORY_KEY='amni_jarvis_input_history';const _INPUT_HISTORY_MAX=20;
let _inputHistory=(()=>{try{const j=JSON.parse(localStorage.getItem(_INPUT_HISTORY_KEY)||'[]');return Array.isArray(j)?j.slice(-_INPUT_HISTORY_MAX):[]}catch{return[]}})();
let _historyIdx=-1;let _draftBeforeRecall='';
function pushInputHistory(text){
  if(!text||typeof text!=='string')return;
  const t=text.trim();if(!t)return;
  if(_inputHistory.length&&_inputHistory[_inputHistory.length-1]===t)return;
  _inputHistory.push(t);if(_inputHistory.length>_INPUT_HISTORY_MAX)_inputHistory.shift();
  try{localStorage.setItem(_INPUT_HISTORY_KEY,JSON.stringify(_inputHistory))}catch{}
  _historyIdx=-1;_draftBeforeRecall='';
}
function _inputCursorAtStart(){try{return input.selectionStart===0&&input.selectionEnd===0}catch{return true}}
function _inputCursorAtEnd(){try{return input.selectionStart===(input.value||'').length}catch{return true}}
input.addEventListener('keydown',e=>{
  if(_slashAcOpen){
    if(e.key==='ArrowDown'){e.preventDefault();_slashAcIdx=(_slashAcIdx+1)%_slashAcMatches.length;_slashAcRender();return}
    if(e.key==='ArrowUp'){e.preventDefault();_slashAcIdx=(_slashAcIdx-1+_slashAcMatches.length)%_slashAcMatches.length;_slashAcRender();return}
    if(e.key==='Tab'){e.preventDefault();_slashAcAccept(_slashAcMatches[_slashAcIdx].cmd);return}
    if(e.key==='Enter'){e.preventDefault();_slashAcAccept(_slashAcMatches[_slashAcIdx].cmd);return}
    if(e.key==='Escape'){e.preventDefault();_slashAcClose();return}
  }
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();return}
  if(e.key==='ArrowUp'&&_inputHistory.length>0&&_inputCursorAtStart()){
    if(_historyIdx===-1)_draftBeforeRecall=input.value;
    if(_historyIdx===-1)_historyIdx=_inputHistory.length-1;
    else if(_historyIdx>0)_historyIdx--;
    else return;
    e.preventDefault();input.value=_inputHistory[_historyIdx];
    input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px';
    try{input.setSelectionRange(input.value.length,input.value.length)}catch{}
    return;
  }
  if(e.key==='ArrowDown'&&_historyIdx!==-1&&_inputCursorAtEnd()){
    e.preventDefault();
    if(_historyIdx<_inputHistory.length-1){_historyIdx++;input.value=_inputHistory[_historyIdx]}
    else{_historyIdx=-1;input.value=_draftBeforeRecall;_draftBeforeRecall=''}
    input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px';
    try{input.setSelectionRange(input.value.length,input.value.length)}catch{}
    return;
  }
  if(_historyIdx!==-1&&e.key!=='ArrowUp'&&e.key!=='ArrowDown'&&e.key!=='Shift'&&e.key!=='Meta'&&e.key!=='Control'&&e.key!=='Alt'){_historyIdx=-1;_draftBeforeRecall=''}
});
const _KBD_SHORTCUTS=[
  {k:'Ctrl+/',d:'Focus input'},
  {k:'Ctrl+K',d:'Search chat'},
  {k:'Ctrl+E',d:'Export chat to Markdown'},
  {k:'Ctrl+B',d:'Show 24-hour briefing'},
  {k:'Ctrl+G',d:'Open coach panel'},
  {k:'Ctrl+L',d:'Open learning daemon panel'},
  {k:'Ctrl+M',d:'Open memory inspector'},
  {k:'Ctrl+Shift+S',d:'Open sessions browser'},
  {k:'Ctrl+Shift+P',d:'Open persona / voice picker'},
  {k:'Alt+P',d:'Cycle to next persona (Alt+Shift+P for previous)'},
  {k:'Ctrl+Shift+E',d:'Open shell audit log'},
  {k:'?',d:'Show / hide this shortcuts overlay'},
  {k:'Esc',d:'Close any open panel / overlay'}
];
function _kbdOverlayHTML(){return '<div class="kbd-grid">'+_KBD_SHORTCUTS.map(s=>`<div class="kbd-row"><kbd>${esc(s.k)}</kbd><span>${esc(s.d)}</span></div>`).join('')+'</div>'}
function toggleKbdOverlay(){const ov=document.getElementById('kbd-overlay');if(!ov)return;ov.classList.toggle('show');if(ov.classList.contains('show')&&!ov.dataset.populated){ov.querySelector('.kbd-body').innerHTML=_kbdOverlayHTML();ov.dataset.populated='1'}}
function _closeAllOverlays(){
  ['gesture-tour','train-modal','kbd-overlay','chat-search'].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.remove('show')});
  ['persona-panel','learn-panel','tests-panel','shell-panel','sessions-panel','coach-panel'].forEach(id=>{const el=document.getElementById(id);if(el&&el.classList.contains('show'))el.classList.remove('show')});
  _personaPanelOpen=false;_ldPanelOpen=false;_tpPanelOpen=false;_shPanelOpen=false;_spPanelOpen=false;_coachPanelOpen=false;
  document.getElementById('coach-toggle').classList.remove('on');
  if(_csOpen)closeChatSearch();
}
document.addEventListener('keydown',e=>{
  const inField=e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT'||e.target.isContentEditable);
  if(e.key==='Escape'){_closeAllOverlays();return}
  if(e.key==='?'&&!inField&&!e.ctrlKey&&!e.metaKey){e.preventDefault();toggleKbdOverlay();return}
  if(!(e.ctrlKey||e.metaKey))return;
  const key=e.key.toLowerCase();const sh=e.shiftKey;
  if(key==='/'&&!sh){e.preventDefault();input.focus();return}
  if(key==='e'&&!sh){e.preventDefault();_exportChatMd();return}
  if(key==='b'&&!sh){e.preventDefault();_qcBriefing();return}
  if(key==='g'&&!sh){e.preventDefault();toggleCoachPanel();return}
  if(key==='l'&&!sh){e.preventDefault();toggleLearnPanel();return}
  if(key==='m'&&!sh){e.preventDefault();toggleMem();return}
  if(key==='s'&&sh){e.preventDefault();toggleSessionsPanel();return}
  if(key==='p'&&sh){e.preventDefault();togglePersonaPanel();return}
  if(key==='e'&&sh){e.preventDefault();toggleShellPanel();return}
});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px'});
async function refreshStats(){try{const r=await fetch('/stats');const j=await r.json();lessonPill.textContent='lessons '+(j.lessons_n||0);const lr=document.getElementById('sp-lessons');if(lr)lr.textContent=(j.lessons_n||0)+' indexed'}catch{}}
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
      if(d<160){const alpha=(1-d/160)*.35;ctx.strokeStyle=`rgba(${_TC.c},${alpha})`;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    }
  }
  for(const n of nodes){
    ctx.fillStyle='rgba('+_TC.c+',.85)';
    ctx.shadowBlur=8;ctx.shadowColor='rgba('+_TC.c+',.7)';
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
const CUSTOM_GESTURE_KEY='amni_jarvis_custom_gestures';
const _CUSTOM_MATCH_THRESHOLD=0.18;
function _featurize(lm){
  if(!lm||lm.length<21)return null;
  const t=lm[4],i=lm[8],m=lm[12],r=lm[16],p=lm[20];
  const ext={t:_dist(t,lm[0])>_dist(lm[2],lm[0])?1:0,i:_fingerExtended(lm,8,6,5)?1:0,m:_fingerExtended(lm,12,10,9)?1:0,r:_fingerExtended(lm,16,14,13)?1:0,p:_fingerExtended(lm,20,18,17)?1:0};
  const pinch=_dist(t,i),tm=_dist(t,m),im=_dist(i,m),mr=_dist(m,r),rp=_dist(r,p);
  return [ext.t,ext.i,ext.m,ext.r,ext.p,Math.min(1,pinch*4),Math.min(1,tm*3),Math.min(1,im*5),Math.min(1,mr*5),Math.min(1,rp*5)];
}
function _featDist(a,b){let s=0;const n=Math.min(a.length,b.length);for(let i=0;i<n;i++){const d=a[i]-b[i];s+=d*d;}return Math.sqrt(s/n)}
function _loadCustomGestures(){try{return JSON.parse(localStorage.getItem(CUSTOM_GESTURE_KEY)||'[]')||[]}catch{return []}}
function _saveCustomGestures(arr){try{localStorage.setItem(CUSTOM_GESTURE_KEY,JSON.stringify(arr.slice(0,20)))}catch{}}
let _customGestures=_loadCustomGestures();
function _matchCustom(features){
  if(!features||!_customGestures.length)return null;
  let best=null,bestD=999;
  for(const g of _customGestures){
    const d=_featDist(features,g.template);
    if(d<bestD){bestD=d;best=g}
  }
  return (best&&bestD<_CUSTOM_MATCH_THRESHOLD)?{gesture:best,distance:bestD}:null;
}
function _renderCustomList(){
  const el=document.getElementById('custom-list');if(!el)return;
  if(!_customGestures.length){el.innerHTML='';return}
  el.innerHTML=_customGestures.map((g,i)=>`<div class="custom-row"><span class="cg-name">${esc(g.name)}</span><span class="cg-act">${esc(g.action_type||'prompt')}</span><span class="cg-del" onclick="_deleteCustomGesture(${i})" title="Delete">✕</span></div>`).join('');
}
function _deleteCustomGesture(idx){if(idx<0||idx>=_customGestures.length)return;const name=_customGestures[idx].name;_customGestures.splice(idx,1);_saveCustomGestures(_customGestures);_renderCustomList();bubble('bot','Deleted custom gesture **'+esc(name)+'**','<span class="badge">gesture</span>')}
let _tmRecording=false,_tmFrames=[],_tmRecordTimer=null;
function _tmOpen(){const m=document.getElementById('train-modal');m.classList.add('show');document.getElementById('tm-step-1').style.display='block';document.getElementById('tm-step-2').style.display='none';document.getElementById('tm-name').value='';document.getElementById('tm-action-value').value='';document.getElementById('tm-name').focus()}
function _tmClose(){const m=document.getElementById('train-modal');m.classList.remove('show');_tmRecording=false;if(_tmRecordTimer){clearInterval(_tmRecordTimer);_tmRecordTimer=null}}
function _tmStartRecord(){
  const name=(document.getElementById('tm-name').value||'').trim().slice(0,30);
  if(!name){document.getElementById('tm-status').textContent='Name required.';return}
  if(!convoAnalyser&&!_hands){}
  const actType=document.getElementById('tm-action-type').value;const actVal=(document.getElementById('tm-action-value').value||'').trim();
  if(actType==='prompt'&&!actVal){document.getElementById('tm-status').textContent='Action message required.';return}
  document.getElementById('tm-step-1').style.display='none';document.getElementById('tm-step-2').style.display='block';
  let cd=3;const cdEl=document.getElementById('tm-countdown');const stEl=document.getElementById('tm-rec-status');cdEl.textContent=cd;stEl.textContent='Hold your gesture steady…';
  const cdTimer=setInterval(()=>{cd--;if(cd>0){cdEl.textContent=cd}else{clearInterval(cdTimer);_tmRecording=true;_tmFrames=[];cdEl.textContent='REC';stEl.textContent='Recording 1.5s of landmarks…';
    setTimeout(()=>{_tmRecording=false;if(_tmFrames.length<5){stEl.textContent='Not enough samples — enable GESTURE first, then retry.';setTimeout(_tmClose,1500);return}
      const mean=new Array(_tmFrames[0].length).fill(0);
      for(const f of _tmFrames)for(let k=0;k<f.length;k++)mean[k]+=f[k];
      for(let k=0;k<mean.length;k++)mean[k]/=_tmFrames.length;
      _customGestures.push({name,action_type:actType,action_value:actVal,template:mean,samples:_tmFrames.length,created_at:Date.now()});
      _saveCustomGestures(_customGestures);_renderCustomList();
      stEl.textContent='Saved **'+name+'** ('+_tmFrames.length+' samples).';bubble('bot','Trained new gesture **'+esc(name)+'** ('+_tmFrames.length+' samples). Make it again on camera to fire its action.','<span class="badge">gesture</span>');
      setTimeout(_tmClose,1200);},1500);
  }},1000);
}
const _GESTURE_PACK_VERSION=1;
function _exportCustomGestures(){
  if(!_customGestures.length){bubble('bot','No custom gestures to export. Click **+ TRAIN** to teach one first.','<span class="badge err">gesture</span>');return}
  const pack={schema:'amni-ai-gesture-pack',version:_GESTURE_PACK_VERSION,exported_at:new Date().toISOString(),count:_customGestures.length,gestures:_customGestures.map(g=>({name:g.name,action_type:g.action_type,action_value:g.action_value,template:g.template,samples:g.samples,created_at:g.created_at}))};
  try{
    const blob=new Blob([JSON.stringify(pack,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');
    a.href=url;a.download='adam-gestures-'+new Date().toISOString().slice(0,10)+'.json';
    document.body.appendChild(a);a.click();document.body.removeChild(a);
    setTimeout(()=>URL.revokeObjectURL(url),2000);
    bubble('bot','Exported **'+_customGestures.length+'** custom gesture'+(_customGestures.length===1?'':'s')+' to your downloads.','<span class="badge">gesture</span>');
  }catch(e){bubble('bot','Export failed: '+esc(e.message),'<span class="badge err">gesture</span>')}
}
const GESTURE_TOUR_KEY='amni_jarvis_gesture_tour_seen';
function _gtOpen(){const el=document.getElementById('gesture-tour');if(el){el.classList.add('show');setTimeout(()=>document.addEventListener('click',_gtOutsideClick,true),100)}}
function _gtClose(){const el=document.getElementById('gesture-tour');if(el)el.classList.remove('show');localStorage.setItem(GESTURE_TOUR_KEY,'1');document.removeEventListener('click',_gtOutsideClick,true)}
function _gtOutsideClick(ev){const el=document.getElementById('gesture-tour');if(!el||!el.classList.contains('show'))return;const inner=el.querySelector('.gt-card')||el.firstElementChild;if(inner&&inner.contains(ev.target))return;_gtClose()}
function _gtTrainFromTour(){_gtClose();setTimeout(()=>_tmOpen(),250)}
function _maybeShowGestureTour(){if(!localStorage.getItem(GESTURE_TOUR_KEY))setTimeout(_gtOpen,400)}
async function _importCustomGestures(inputEl){
  const f=inputEl.files&&inputEl.files[0];if(!f){return}
  inputEl.value='';
  try{
    const text=await f.text();const pack=JSON.parse(text);
    if(!pack||pack.schema!=='amni-ai-gesture-pack'){bubble('bot','Not a valid Adam gesture pack (missing schema marker).','<span class="badge err">gesture</span>');return}
    if(!Array.isArray(pack.gestures)){bubble('bot','Gesture pack has no `gestures` array.','<span class="badge err">gesture</span>');return}
    const existing=new Map(_customGestures.map(g=>[g.name.toLowerCase(),g]));let added=0,replaced=0,skipped=0;
    for(const raw of pack.gestures){
      if(!raw||!raw.name||!Array.isArray(raw.template)||raw.template.length!==10){skipped++;continue}
      const nm=String(raw.name).slice(0,30);
      const entry={name:nm,action_type:raw.action_type||'prompt',action_value:String(raw.action_value||'').slice(0,200),template:raw.template.map(Number),samples:raw.samples||0,created_at:raw.created_at||Date.now(),imported_at:Date.now()};
      if(existing.has(nm.toLowerCase())){const idx=_customGestures.findIndex(g=>g.name.toLowerCase()===nm.toLowerCase());_customGestures[idx]=entry;replaced++}
      else{_customGestures.push(entry);added++}
    }
    _customGestures=_customGestures.slice(0,20);_saveCustomGestures(_customGestures);_renderCustomList();
    bubble('bot','Imported gesture pack: **'+added+'** new · '+replaced+' replaced · '+skipped+' skipped ('+(pack.exported_at?'exported '+pack.exported_at.slice(0,10):'unknown date')+').','<span class="badge">gesture</span>');
  }catch(e){bubble('bot','Import failed: '+esc(e.message),'<span class="badge err">gesture</span>')}
}
function applyCustomAction(g){
  if(g.action_type==='prompt'&&g.action_value){quick(g.action_value);return}
  if(g.action_type==='builtin'){
    const map={'voice':toggleVoiceOut,'clear':()=>{log.innerHTML='';bubble('bot','(chat cleared)')},'theme':_cycleTheme,'submit':()=>{const t=input.value.trim();if(t)send()},'system':()=>quick('Show me current system stats')};
    const fn=map[(g.action_value||'').toLowerCase()];if(fn)fn();return;
  }
  if(g.action_type==='panel'){
    const panels={'coach':toggleCoachPanel,'learn':toggleLearnPanel,'tests':toggleTestsPanel,'shell':toggleShellPanel,'sessions':toggleSessionsPanel,'persona':togglePersonaPanel};
    const fn=panels[(g.action_value||'').toLowerCase()];if(fn)fn();return;
  }
}
_renderCustomList();
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
let _gestureLastClearedHTML=null;
let _gestureLastFiredTs=0;
const _GESTURE_COOLDOWN_MS=1500;
function applyGestureAction(g){
  const now=Date.now();
  if(now-_gestureLastFiredTs<_GESTURE_COOLDOWN_MS)return;
  _gestureLastFiredTs=now;
  if(g==='pinch')toggleVoiceOut();
  else if(g==='fist'){_gestureLastClearedHTML=log.innerHTML;log.innerHTML='';bubble('bot','Chat cleared by gesture (within 15s).','<span class="badge">gesture</span> <a href="#" onclick="_gestureUndoClear();return false" style="color:var(--cyan);text-decoration:underline;margin-left:8px">↶ undo</a>');setTimeout(()=>{_gestureLastClearedHTML=null},15000)}
  else if(g==='open_palm')quick('Show me current system stats');
  else if(g==='peace')_cycleTheme();
  else if(g==='point'){const last=log.querySelectorAll('.msg.bot .meta .badge');if(last.length)quick('Tell me more about that')}
  else if(g==='thumb_up'){const t=input.value.trim();if(t)send()}
}
function _gestureUndoClear(){
  if(!_gestureLastClearedHTML){bubble('bot','Undo window has expired — sorry, chat is gone.','<span class="badge err">gesture</span>');return}
  log.innerHTML=_gestureLastClearedHTML;_gestureLastClearedHTML=null;
  bubble('bot','Chat restored.','<span class="badge">gesture</span>');
}
function _onHandsResults(res){
  const ctx=_camLm.getContext('2d');_camLm.width=_camLm.clientWidth*window.devicePixelRatio;_camLm.height=_camLm.clientHeight*window.devicePixelRatio;
  ctx.clearRect(0,0,_camLm.width,_camLm.height);
  const lms=(res.multiHandLandmarks||[])[0];
  if(!lms){_readout.textContent='—';_camPanel.classList.add('idle');return}
  _camPanel.classList.remove('idle');
  ctx.fillStyle='rgba('+_TC.c+',.95)';ctx.shadowBlur=6;ctx.shadowColor='rgba('+_TC.c+',.7)';
  for(const lm of lms){const x=(1-lm.x)*_camLm.width,y=lm.y*_camLm.height;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}
  ctx.shadowBlur=0;ctx.strokeStyle='rgba('+_TC.c+',.45)';ctx.lineWidth=1.4;
  const conns=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  for(const [a,b] of conns){ctx.beginPath();ctx.moveTo((1-lms[a].x)*_camLm.width,lms[a].y*_camLm.height);ctx.lineTo((1-lms[b].x)*_camLm.width,lms[b].y*_camLm.height);ctx.stroke()}
  const features=_featurize(lms);
  if(_tmRecording&&features)_tmFrames.push(features);
  let g=classifyGesture(lms);let custom=null;
  if(g==='unknown'&&features){const m=_matchCustom(features);if(m){custom=m;g='custom:'+m.gesture.name}}
  _readout.textContent=g==='unknown'?'—':(custom?('★ '+custom.gesture.name.toUpperCase()):g.replace('_',' ').toUpperCase());
  const now=performance.now();frameTimes.push(now);if(frameTimes.length>30)frameTimes.shift();
  if(frameTimes.length>=2){const fps=Math.round(1000*(frameTimes.length-1)/(frameTimes[frameTimes.length-1]-frameTimes[0]));document.getElementById('cam-fps').textContent=fps+' fps'}
  if(g!=='unknown'&&g!==lastGesture&&(now-lastGestureAt)>GESTURE_COOLDOWN_MS){
    lastGesture=g;lastGestureAt=now;_flashGesture(custom?custom.gesture.name:g);if(custom)applyCustomAction(custom.gesture);else applyGestureAction(g);
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
  if(gestureOn){startGesture();_maybeShowGestureTour()}else stopGesture();
}
if(localStorage.getItem(GKEY)==='1'){setTimeout(()=>{gestureOn=true;_gToggle.classList.add('on');startGesture()},800)}
let poseOn=false,poseStream=null,poseRAF=null,pose=null,poseSessionId=null,poseSessionActive=false,_poseLastSendAt=0,_poseSending=false,poseFrameTimes=[];
const _POSE_CONNS=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[15,17],[16,18],[27,29],[28,30],[29,31],[30,32]];
const _poseVideo=()=>document.getElementById('pose-video'),_poseCanvas=()=>document.getElementById('pose-landmarks'),_posePanel=()=>document.getElementById('pose-panel');
async function _loadPoseLib(){
  if(window.Pose)return true;
  await new Promise((res,rej)=>{const s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/pose.js';s.crossOrigin='anonymous';s.onload=res;s.onerror=rej;document.head.appendChild(s)}).catch(e=>console.warn('pose load fail',e));
  return !!window.Pose;
}
function _onPoseResults(res){
  const cv=_poseCanvas();if(!cv)return;const ctx=cv.getContext('2d');
  cv.width=cv.clientWidth*window.devicePixelRatio;cv.height=cv.clientHeight*window.devicePixelRatio;
  ctx.clearRect(0,0,cv.width,cv.height);
  const lms=res.poseLandmarks;
  const panel=_posePanel();
  if(!lms){if(panel)panel.classList.remove('live');return}
  if(panel)panel.classList.add('live');
  ctx.strokeStyle='rgba('+_TC.g+',.55)';ctx.lineWidth=2;
  for(const [a,b] of _POSE_CONNS){if(lms[a]&&lms[b]){ctx.beginPath();ctx.moveTo((1-lms[a].x)*cv.width,lms[a].y*cv.height);ctx.lineTo((1-lms[b].x)*cv.width,lms[b].y*cv.height);ctx.stroke()}}
  ctx.fillStyle='rgba('+_TC.g+',.95)';ctx.shadowBlur=5;ctx.shadowColor='rgba('+_TC.g+',.7)';
  for(const lm of lms){if((lm.visibility||1)<0.3)continue;ctx.beginPath();ctx.arc((1-lm.x)*cv.width,lm.y*cv.height,3,0,Math.PI*2);ctx.fill()}
  ctx.shadowBlur=0;
  const now=performance.now();poseFrameTimes.push(now);if(poseFrameTimes.length>30)poseFrameTimes.shift();
  if(poseFrameTimes.length>=2){const fps=Math.round(1000*(poseFrameTimes.length-1)/(poseFrameTimes[poseFrameTimes.length-1]-poseFrameTimes[0]));const fe=document.getElementById('pose-fps');if(fe)fe.textContent=fps+' fps'}
  if(poseSessionActive&&poseSessionId&&!_poseSending&&(now-_poseLastSendAt)>120){
    _poseLastSendAt=now;_poseSending=true;
    const pl=lms.map(l=>({x:l.x,y:l.y,visibility:l.visibility}));
    fetch('/vision/pose/frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:poseSessionId,landmarks:pl,exercise:document.getElementById('pose-exercise').value})})
      .then(r=>r.json()).then(_updatePoseUI).catch(()=>{}).finally(()=>{_poseSending=false});
  }
}
function _updatePoseUI(j){
  if(!j||j.error)return;
  const re=document.getElementById('pose-reps'),ce=document.getElementById('pose-clean'),ae=document.getElementById('pose-angle'),fe=document.getElementById('pose-fb');
  if(typeof j.reps==='number'&&re)re.textContent=j.reps;
  if(typeof j.good_reps==='number'&&ce)ce.textContent=j.good_reps;
  if(ae)ae.textContent=(j.angle==null?'—':Math.round(j.angle)+'°');
  if(fe&&j.feedback){fe.textContent=j.feedback;fe.classList.remove('warn','good');
    if(j.rep_completed&&j.rep&&j.rep.clean)fe.classList.add('good');
    else if((j.form_issues&&j.form_issues.length)||(j.rep_completed&&j.rep&&!j.rep.clean))fe.classList.add('warn');}
}
async function startPoseCam(){
  const ok=await _loadPoseLib();
  if(!ok){bubble('bot','Could not load MediaPipe Pose from CDN — check your network or content blocker.','<span class="badge err">coach</span>');return false}
  try{poseStream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240,facingMode:'user'},audio:false})}
  catch(e){bubble('bot','Camera permission denied or unavailable: '+esc(e.message)+' — PT Coach needs your camera to see your form.','<span class="badge err">coach</span>');return false}
  const v=_poseVideo();v.srcObject=poseStream;await v.play().catch(()=>{});
  pose=new window.Pose({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${f}`});
  pose.setOptions({modelComplexity:1,smoothLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
  pose.onResults(_onPoseResults);
  const loop=async()=>{if(!poseOn)return;if(v.readyState>=2){try{await pose.send({image:v})}catch{}}poseRAF=requestAnimationFrame(loop)};
  poseRAF=requestAnimationFrame(loop);
  return true;
}
function stopPoseCam(){
  if(poseRAF){cancelAnimationFrame(poseRAF);poseRAF=null}
  if(poseStream){poseStream.getTracks().forEach(t=>t.stop());poseStream=null}
  if(pose){try{pose.close()}catch{}pose=null}
  poseFrameTimes=[];const fe=document.getElementById('pose-fps');if(fe)fe.textContent='— fps';
}
async function togglePoseCoach(force){
  const want=(typeof force==='boolean')?force:!poseOn;
  const tog=document.getElementById('pose-toggle');
  if(want){
    poseOn=true;if(tog)tog.classList.add('on');_posePanel().classList.add('show');
    const started=await startPoseCam();
    if(!started){poseOn=false;if(tog)tog.classList.remove('on');_posePanel().classList.remove('show')}
  }else{
    if(poseSessionActive)await _poseStopSession(true);
    poseOn=false;if(tog)tog.classList.remove('on');_posePanel().classList.remove('show');stopPoseCam();
  }
}
async function _poseStartSession(){
  const ex=document.getElementById('pose-exercise').value;
  try{
    const j=await(await fetch('/vision/pose/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({exercise:ex,session_id:''})})).json();
    if(j.error){bubble('bot','Could not start coach session: '+esc(j.error),'<span class="badge err">coach</span>');return}
    poseSessionId=j.session_id;poseSessionActive=true;
    document.getElementById('pose-reps').textContent='0';document.getElementById('pose-clean').textContent='0';
    const fb=document.getElementById('pose-fb');fb.classList.remove('warn','good');fb.textContent=j.cue?('Go! '+j.cue):'Go! I\'m watching your form.';
    const go=document.getElementById('pose-go');go.textContent='STOP';go.classList.add('live');
  }catch(e){bubble('bot','Coach start failed: '+esc(e.message),'<span class="badge err">coach</span>')}
}
async function _poseStopSession(silent){
  if(!poseSessionActive||!poseSessionId)return;
  poseSessionActive=false;const sid=poseSessionId;poseSessionId=null;
  const go=document.getElementById('pose-go');if(go){go.textContent='START';go.classList.remove('live')}
  try{
    const s=await(await fetch('/vision/pose/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})})).json();
    if(!silent&&s&&!s.error){
      const issues=(s.common_issues||[]).map(i=>i.msg).join('; ');
      bubble('bot','**'+(s.label||'Workout')+' done!** '+s.reps+' reps · '+s.good_reps+' clean ('+s.clean_rate_pct+'%) · peak depth '+(s.peak_depth==null?'—':Math.round(s.peak_depth)+'°')+' · '+s.duration_s+'s.'+(issues?'\n\nWatch next time: '+issues:'\n\nClean form — nice work!'),'<span class="badge">coach</span>');
    }
  }catch(e){if(!silent)bubble('bot','Coach summary failed: '+esc(e.message),'<span class="badge err">coach</span>')}
}
function _poseToggleSession(){if(!poseOn){return}poseSessionActive?_poseStopSession(false):_poseStartSession()}
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
const _trayEl=document.getElementById('task-tray'),_trayRows=document.getElementById('tray-rows'),_trayCt=document.getElementById('tray-ct');
let _taskPollOn=false,_taskPollIdleAt=0;
async function pollTasks(){
  try{
    const j=await(await fetch('/tasks')).json();
    const active=j.active||[];
    _trayCt.textContent=`${active.length} active`;
    if(active.length===0){
      if(_taskPollIdleAt===0)_taskPollIdleAt=Date.now();
      if(Date.now()-_taskPollIdleAt>3500){_trayEl.classList.remove('show');_trayRows.innerHTML=''}
    }else{
      _taskPollIdleAt=0;_trayEl.classList.add('show');
      _trayRows.innerHTML=active.map(t=>`<div class="task-row${t.cancel_requested?' cancelling':''}" data-tid="${esc(t.id)}"><div><div class="label"><span class="kind">${esc(t.kind)}</span>${esc(t.label||'')}${t.progress_msg?'<div class="msg">'+esc(t.progress_msg)+'</div>':''}</div><div class="pbar"><div class="fill" style="width:${t.progress_pct||0}%"></div></div></div><button class="cancel" onclick="cancelTask('${esc(t.id)}')">${t.cancel_requested?'…':'×'}</button></div>`).join('');
    }
  }catch(e){}
}
async function cancelTask(tid){try{await fetch('/tasks/'+encodeURIComponent(tid)+'/cancel',{method:'POST'})}catch{}}
function startTaskPolling(){if(_taskPollOn)return;_taskPollOn=true;pollTasks();setInterval(pollTasks,2000)}
startTaskPolling();
let _lastImage=null;
function _readAsDataURL(blob){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(blob)})}
async function handleImageBlob(blob){
  const w=document.querySelector('.welcome');if(w)w.remove();
  const dataUrl=await _readAsDataURL(blob).catch(()=>null);
  if(!dataUrl)return;
  _lastImage={dataUrl,blob};
  const uMsg=document.createElement('div');uMsg.className='msg user';
  const uB=document.createElement('div');uB.className='bubble';uB.innerHTML='[image]<br><img class="img-attach" src="'+esc(dataUrl)+'" alt="attached"/>';
  uMsg.appendChild(uB);log.appendChild(uMsg);log.scrollTop=log.scrollHeight;
  const botMsg=document.createElement('div');botMsg.className='msg bot';
  const botB=document.createElement('div');botB.className='bubble thinking';botB.textContent='describing image…';botMsg.appendChild(botB);
  log.appendChild(botMsg);log.scrollTop=log.scrollHeight;
  try{
    const b64=dataUrl.split(',',2)[1];
    const r=await fetch('/vision/describe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:b64})});
    const j=await r.json();
    botB.classList.remove('thinking');
    if(j.error){botB.innerHTML='Vision unavailable: '+esc(j.error)}
    else{const cap=j.caption||'(no caption)';botB.innerHTML=md('**Image:** '+cap+'\n\n_'+j.width+'×'+j.height+' · '+(j.wall_s||'?')+'s_\n\nAsk me about it — I\'ll route follow-ups through /vision/ask.')}
    const metaEl=document.createElement('div');metaEl.className='meta';metaEl.innerHTML='<span class="badge">vision</span>'+(j.caption?'<span class="badge">'+esc(j.caption.slice(0,40))+'</span>':'');botMsg.appendChild(metaEl);
  }catch(e){botB.classList.remove('thinking');botB.textContent='Vision request failed: '+e.message}
  log.scrollTop=log.scrollHeight;
}
async function askAboutLastImage(question){
  if(!_lastImage)return false;
  const botMsg=document.createElement('div');botMsg.className='msg bot';
  const botB=document.createElement('div');botB.className='bubble thinking';botB.textContent='answering about the image…';botMsg.appendChild(botB);
  log.appendChild(botMsg);log.scrollTop=log.scrollHeight;
  try{
    const b64=_lastImage.dataUrl.split(',',2)[1];
    const r=await fetch('/vision/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:b64,question:question})});
    const j=await r.json();
    botB.classList.remove('thinking');
    botB.innerHTML=j.error?'Error: '+esc(j.error):md('**Answer:** '+(j.answer||'(no answer)'));
    const metaEl=document.createElement('div');metaEl.className='meta';metaEl.innerHTML='<span class="badge">vqa</span>';botMsg.appendChild(metaEl);
  }catch(e){botB.classList.remove('thinking');botB.textContent='vqa failed: '+e.message}
  log.scrollTop=log.scrollHeight;return true;
}
const _origSend=send;
window.send=async function(){
  const t=input.value.trim();
  if(!t)return;
  if(_lastImage && /\b(this|that|the image|the picture|in it|show |describe |what is|what's|color|shape)/i.test(t)){
    input.value='';input.style.height='auto';
    const uMsg=document.createElement('div');uMsg.className='msg user';
    const uB=document.createElement('div');uB.className='bubble';uB.textContent=t;uMsg.appendChild(uB);
    log.appendChild(uMsg);log.scrollTop=log.scrollHeight;
    await askAboutLastImage(t);return;
  }
  if(!convoOn){
    input.value='';input.style.height='auto';send_btn.disabled=true;
    try{await _streamReplyWithTTS(t,{createBubbles:true,tts:voiceOut && _voiceBackends.tts})}
    finally{send_btn.disabled=false;input.focus()}
    return;
  }
  return _origSend.apply(this,arguments);
};
const CONVO_KEY='amni_jarvis_convo';
const VAD_KEY='amni_jarvis_vad';
const CONVO_MAX_UTTERANCE_MS=10000;
const CONVO_MAX_FAILS=3;
const _vadDefaults={silence_ms:600,min_speech_ms:300,vad_threshold:26,barge_threshold:38};
const _WHISPER_HALLUCINATIONS=new Set(['you','thank you','thanks','thanks for watching','thanks for watching!','bye','bye.','okay','ok','.','...','...thank you','sub','please subscribe','subscribe','♪','[music]','[silence]','[blank_audio]','um','uh','huh','hmm','yeah','yep','um.','uh.','mm-hmm']);
function _jaccardWords(a,b){const sa=new Set(a.split(/\s+/).filter(Boolean));const sb=new Set(b.split(/\s+/).filter(Boolean));if(sa.size===0||sb.size===0)return 0;let inter=0;for(const x of sa)if(sb.has(x))inter++;return inter/(sa.size+sb.size-inter)}
function _isWhisperHallucination(text){
  const t=(text||'').trim().toLowerCase();
  if(t.length===0||t.length<3)return true;
  if(_WHISPER_HALLUCINATIONS.has(t))return true;
  if(/^[.\s!?,]+$/.test(t))return true;
  const parts=t.split(/[.!?]+/).map(s=>s.replace(/[,\s]+/g,' ').trim()).filter(s=>s.length>=2);
  if(parts.length>=2){
    let dupePairs=0;
    for(let i=0;i<parts.length;i++)for(let j=i+1;j<parts.length;j++){
      if(parts[i]===parts[j] || _jaccardWords(parts[i],parts[j])>=0.65){dupePairs++;break}
    }
    if(dupePairs>=Math.max(1,Math.ceil(parts.length*0.4)))return true;
  }
  const words=t.split(/\s+/).filter(Boolean);
  if(words.length>=3){
    for(let n=1;n<=Math.min(4,Math.floor(words.length/3));n++){
      for(let i=0;i+n*3<=words.length;i++){
        const a=words.slice(i,i+n).join(' ');
        const b=words.slice(i+n,i+n*2).join(' ');
        const c=words.slice(i+n*2,i+n*3).join(' ');
        if(a===b && b===c)return true;
      }
    }
  }
  return false;
}
const WAKE_KEY='amni_jarvis_wake';
const WAKE_EXTRA_KEY='amni_jarvis_wake_extra';
const WAKE_CHIRP_KEY='amni_jarvis_wake_chirp';
let wakeOn=localStorage.getItem(WAKE_KEY)==='1';
let wakeChirp=localStorage.getItem(WAKE_CHIRP_KEY)!=='0';
const _DEFAULT_WAKE_WORDS=['adam','atom','adams','adan'];
function _loadWakeExtras(){
  try{const raw=localStorage.getItem(WAKE_EXTRA_KEY)||'';return raw.split(',').map(s=>s.trim().toLowerCase()).filter(s=>/^[a-z]{2,16}$/.test(s))}
  catch{return []}
}
function _saveWakeExtras(arr){try{localStorage.setItem(WAKE_EXTRA_KEY,(arr||[]).join(','))}catch{}}
let _wakePatternPunct,_wakePatternSpace;
function _buildWakePatterns(){
  const extras=_loadWakeExtras();
  const all=[..._DEFAULT_WAKE_WORDS,...extras].map(w=>w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'));
  const wordGroup='(?:'+all.join('|')+')';
  const prefix='(?:hey[,\\s]+|ok(?:ay)?[,\\s]+|yo[,\\s]+|hi[,\\s]+|hello[,\\s]+)?'+wordGroup;
  _wakePatternPunct=new RegExp('^\\s*'+prefix+'[,:;!?]+\\s*(.+)','i');
  _wakePatternSpace=new RegExp('^\\s*'+prefix+'\\s+(.+)','i');
}
_buildWakePatterns();
const _WAKE_DECLARATIVE=/^(?:is|isn't|was|wasn't|were|weren't|has|hasn't|have|haven't|had|hadn't|seems|seemed|looks|looked|appears|appeared|sounds|sounded|seems\s+to|got|gets)\b/i;
function _wakeWordGate(text){
  if(!wakeOn)return text;
  const t=text||'';
  const mp=t.match(_wakePatternPunct);
  if(mp){const s=(mp[1]||'').trim();if(s.length>=2){_wakeFired();return s}return null}
  const ms=t.match(_wakePatternSpace);
  if(!ms)return null;
  const s=(ms[1]||'').trim();
  if(_WAKE_DECLARATIVE.test(s))return null;
  if(s.length>=2){_wakeFired();return s}
  return null;
}
function _wakeFired(){
  const b=document.getElementById('wake-toggle');
  if(b){b.classList.remove('fired');void b.offsetWidth;b.classList.add('fired');setTimeout(()=>b.classList.remove('fired'),1100)}
  if(wakeChirp)_playWakeChirp();
}
let _wakeAudioCtx=null;
function _playWakeChirp(){
  try{
    if(!_wakeAudioCtx)_wakeAudioCtx=new (window.AudioContext||window.webkitAudioContext)();
    const ctx=_wakeAudioCtx;const now=ctx.currentTime;
    const o=ctx.createOscillator();const g=ctx.createGain();
    o.type='sine';o.frequency.setValueAtTime(880,now);o.frequency.exponentialRampToValueAtTime(1320,now+.09);
    g.gain.setValueAtTime(0,now);g.gain.linearRampToValueAtTime(.12,now+.015);g.gain.exponentialRampToValueAtTime(.0001,now+.18);
    o.connect(g);g.connect(ctx.destination);o.start(now);o.stop(now+.2);
  }catch(_){}
}
function _wakeConfigPrompt(){
  const cur=_loadWakeExtras().join(', ');
  const v=prompt('Extra wake words (comma-separated, letters only, 2-16 chars each).\n\nDefaults always active: '+_DEFAULT_WAKE_WORDS.join(', ')+'\n\nExamples: jarvis, computer, luna',cur);
  if(v===null)return;
  const arr=v.split(',').map(s=>s.trim().toLowerCase()).filter(s=>/^[a-z]{2,16}$/.test(s));
  _saveWakeExtras(arr);_buildWakePatterns();
  const final=[..._DEFAULT_WAKE_WORDS,...arr];
  bubble('bot','Wake words updated: **'+final.join(', ')+'** (right-click WAKE again to edit).','<span class="badge">wake</span>');
}
function _wakeChirpToggle(){
  wakeChirp=!wakeChirp;try{localStorage.setItem(WAKE_CHIRP_KEY,wakeChirp?'1':'0')}catch{}
  bubble('bot','Wake chirp **'+(wakeChirp?'on':'off')+'**.','<span class="badge">wake</span>');
}
let _ambientFlashTimer=null;
function _flashAmbient(text){
  const led=document.getElementById('ld-led');
  let amb=document.getElementById('wake-ambient');
  if(!amb){amb=document.createElement('div');amb.id='wake-ambient';amb.className='wake-ambient';document.body.appendChild(amb)}
  amb.textContent='heard: '+(text||'').slice(0,60)+((text||'').length>60?'…':'');
  amb.classList.add('show');
  clearTimeout(_ambientFlashTimer);
  _ambientFlashTimer=setTimeout(()=>amb.classList.remove('show'),2200);
}
function toggleWake(){wakeOn=!wakeOn;localStorage.setItem(WAKE_KEY,wakeOn?'1':'0');document.getElementById('wake-toggle').classList.toggle('on',wakeOn);if(wakeOn)bubble('bot','Wake word **on** — only respond when you say "Adam, ..." in convo mode.','<span class="badge">wake</span>');else bubble('bot','Wake word **off** — respond to every convo utterance.','<span class="badge">wake</span>')}
if(wakeOn){setTimeout(()=>{const b=document.getElementById('wake-toggle');if(b)b.classList.add('on')},50)}
let _vadConfig={..._vadDefaults};
try{const saved=JSON.parse(localStorage.getItem(VAD_KEY)||'null');if(saved&&typeof saved==='object')_vadConfig={..._vadDefaults,...saved}}catch{}
function _saveVadConfig(){try{localStorage.setItem(VAD_KEY,JSON.stringify(_vadConfig))}catch{}}
function _refreshVadPanelUI(){
  for(const [key,el] of [['vad_threshold','vp-vt'],['barge_threshold','vp-bt'],['silence_ms','vp-sl'],['min_speech_ms','vp-ms']]){
    const slider=document.getElementById(el);const valEl=document.getElementById(el+'-v');
    if(slider){slider.value=_vadConfig[key];if(valEl)valEl.textContent=_vadConfig[key]}
  }
  const vtL=document.getElementById('vp-vt-line');if(vtL)vtL.style.left=Math.min(100,(_vadConfig.vad_threshold/60)*100)+'%';
  const btL=document.getElementById('vp-bt-line');if(btL)btL.style.left=Math.min(100,(_vadConfig.barge_threshold/60)*100)+'%';
}
function _onVadSlider(key,val){_vadConfig[key]=Number(val);_saveVadConfig();_refreshVadPanelUI()}
function _resetVad(){_vadConfig={..._vadDefaults};_saveVadConfig();_refreshVadPanelUI()}
let _vadPanelOpen=false,_vadMicStream=null,_vadAnalyser=null,_vadCtx=null,_vadMeterRAF=null;
async function toggleVadPanel(){
  _vadPanelOpen=!_vadPanelOpen;
  const p=document.getElementById('vad-panel');p.classList.toggle('show',_vadPanelOpen);
  document.getElementById('vad-toggle').classList.toggle('on',_vadPanelOpen);
  if(_vadPanelOpen){
    _refreshVadPanelUI();
    if(convoOn&&convoAnalyser){
      const meterFill=document.getElementById('vp-meter-fill');
      const loop=()=>{if(!_vadPanelOpen)return;const buf=new Uint8Array(convoAnalyser.frequencyBinCount);convoAnalyser.getByteFrequencyData(buf);let s=0;for(let i=0;i<buf.length;i++)s+=buf[i];const avg=s/buf.length;meterFill.style.width=Math.min(100,(avg/60)*100)+'%';_vadMeterRAF=requestAnimationFrame(loop)};
      loop();
    }else if(navigator.mediaDevices){
      try{
        _vadMicStream=await navigator.mediaDevices.getUserMedia({audio:true});
        _vadCtx=new (window.AudioContext||window.webkitAudioContext)();
        const src=_vadCtx.createMediaStreamSource(_vadMicStream);
        _vadAnalyser=_vadCtx.createAnalyser();_vadAnalyser.fftSize=256;_vadAnalyser.smoothingTimeConstant=0.5;
        src.connect(_vadAnalyser);
        const meterFill=document.getElementById('vp-meter-fill');
        const loop=()=>{if(!_vadPanelOpen)return;const buf=new Uint8Array(_vadAnalyser.frequencyBinCount);_vadAnalyser.getByteFrequencyData(buf);let s=0;for(let i=0;i<buf.length;i++)s+=buf[i];const avg=s/buf.length;meterFill.style.width=Math.min(100,(avg/60)*100)+'%';_vadMeterRAF=requestAnimationFrame(loop)};
        loop();
      }catch(e){console.warn('vad mic init failed',e)}
    }
  }else{
    if(_vadMeterRAF){cancelAnimationFrame(_vadMeterRAF);_vadMeterRAF=null}
    if(_vadMicStream){_vadMicStream.getTracks().forEach(t=>t.stop());_vadMicStream=null}
    if(_vadCtx){try{_vadCtx.close()}catch{};_vadCtx=null}
    _vadAnalyser=null;
  }
}
let convoOn=false,convoStream=null,convoCtx=null,convoAnalyser=null,convoRAF=null,convoState='idle',convoRecorder=null,convoChunks=[],convoFailCount=0,convoSilenceStart=0,convoSpeechStart=0,convoUtteranceStart=0,convoActive=false;
const _convoToggle=document.getElementById('convo-toggle'),_convoBanner=document.getElementById('convo-banner'),_convoStateLabel=document.getElementById('convo-state-label'),_convoLevelBar=document.getElementById('convo-level-bar');
function _setConvoState(s){
  convoState=s;
  for(const c of ['state-recording','state-thinking','state-speaking','state-error'])_convoToggle.classList.remove(c);
  if(s==='recording')_convoToggle.classList.add('state-recording');
  if(s==='transcribing'||s==='thinking')_convoToggle.classList.add('state-thinking');
  if(s==='speaking')_convoToggle.classList.add('state-speaking');
  if(s==='error')_convoToggle.classList.add('state-error');
  _convoStateLabel.textContent=({listening:'LISTENING',recording:'RECORDING',transcribing:'TRANSCRIBING',thinking:'THINKING',speaking:'SPEAKING',error:'PAUSED',idle:'IDLE'})[s]||s.toUpperCase();
  const wv=document.getElementById('voice-wave');
  if(wv){
    for(const c of ['state-listening','state-recording','state-thinking','state-speaking'])wv.classList.remove(c);
    if(s==='listening'||s==='recording'||s==='thinking'||s==='transcribing'||s==='speaking'){
      wv.classList.add('show');wv.classList.add(s==='transcribing'?'state-thinking':'state-'+s);
    }else wv.classList.remove('show');
  }
}
function _vwColorForState(){
  if(convoState==='speaking')return ['#ffd770','rgba(255,215,112,'];
  if(convoState==='thinking'||convoState==='transcribing')return ['#ff4dc8','rgba(255,77,200,'];
  if(convoState==='recording')return [_TC.hexC,'rgba('+_TC.c+','];
  return ['#7ad6ff','rgba(122,214,255,'];
}
function _drawVoiceWave(buf){
  const cv=document.getElementById('voice-wave');if(!cv||!cv.classList.contains('show'))return;
  const ctx=cv.getContext('2d');const W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  const BINS=32;const step=Math.max(1,Math.floor(buf.length/BINS));
  const [stroke,rgba]=_vwColorForState();const barW=W/(BINS*1.4);
  ctx.fillStyle=rgba+'0.06)';ctx.fillRect(0,0,W,H);
  ctx.fillStyle=stroke;
  for(let i=0;i<BINS;i++){
    let s=0,n=0;for(let j=0;j<step;j++){const idx=i*step+j;if(idx<buf.length){s+=buf[idx];n++}}
    const v=n?s/n:0;const h=Math.max(2,(v/255)*H*0.92);
    const x=(i+0.2)*(W/BINS);ctx.fillRect(x,H-h-2,barW,h);
  }
  ctx.fillStyle=rgba+'0.5)';ctx.fillRect(0,H-1,W,1);
}
function _vadLoop(){
  if(!convoOn||!convoAnalyser){convoRAF=null;return}
  const buf=new Uint8Array(convoAnalyser.frequencyBinCount);
  convoAnalyser.getByteFrequencyData(buf);
  let sum=0;for(let i=0;i<buf.length;i++)sum+=buf[i];
  const avg=sum/buf.length;
  _convoLevelBar.style.width=Math.min(100,(avg/40)*100)+'%';
  _drawVoiceWave(buf);
  const now=performance.now();
  if((convoState==='speaking'||convoState==='thinking') && avg>_vadConfig.barge_threshold){
    if(_audioEl){try{_audioEl.pause();_audioEl.currentTime=0}catch{}}
    try{window.speechSynthesis&&speechSynthesis.cancel()}catch{}
    _setConvoState('listening');
  }
  if(convoState==='listening'){
    if(avg>_vadConfig.vad_threshold){
      convoSpeechStart=now;_setConvoState('recording');convoUtteranceStart=now;convoChunks=[];convoSilenceStart=0;
      try{
        const mime=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':(MediaRecorder.isTypeSupported('audio/webm')?'audio/webm':'');
        convoRecorder=mime?new MediaRecorder(convoStream,{mimeType:mime}):new MediaRecorder(convoStream);
        convoRecorder.ondataavailable=e=>{if(e.data.size>0)convoChunks.push(e.data)};
        convoRecorder.onstop=_convoFinishRecording;
        convoRecorder.start();
      }catch(e){console.warn('convo recorder failed',e);_convoFail('recorder init')}
    }
  }else if(convoState==='recording'){
    if(avg<_vadConfig.vad_threshold){
      if(convoSilenceStart===0)convoSilenceStart=now;
      else if(now-convoSilenceStart>_vadConfig.silence_ms && now-convoSpeechStart>_vadConfig.min_speech_ms){
        convoSilenceStart=0;try{convoRecorder&&convoRecorder.stop()}catch{}
      }
    }else convoSilenceStart=0;
    if(now-convoUtteranceStart>CONVO_MAX_UTTERANCE_MS){
      try{convoRecorder&&convoRecorder.stop()}catch{}
    }
  }
  if(convoOn)convoRAF=requestAnimationFrame(_vadLoop);
}
async function _prewarmWhisper(){
  try{
    const buf=new ArrayBuffer(2048);const blob=new Blob([buf],{type:'audio/webm'});
    const dataUrl=await new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(blob)});
    const b64=dataUrl.split(',',2)[1];
    fetch('/voice/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_base64:b64,model_size:'tiny'})}).catch(()=>{});
  }catch{}
}
async function _convoFinishRecording(){
  if(!convoOn){return}
  _setConvoState('transcribing');
  if(convoChunks.length===0){_setConvoState('listening');return}
  const blob=new Blob(convoChunks,{type:(convoRecorder&&convoRecorder.mimeType)||'audio/webm'});
  if(blob.size<800){_setConvoState('listening');return}
  try{
    const dataUrl=await new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(blob)});
    const b64=dataUrl.split(',',2)[1];
    const r=await fetch('/voice/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio_base64:b64})});
    const j=await r.json();
    if(!r.ok||!(j.text||'').trim()){convoFailCount++;if(convoFailCount>=CONVO_MAX_FAILS){_convoFail('too many empty transcriptions');return};_setConvoState('listening');return}
    const text=j.text.trim();
    if(_isWhisperHallucination(text)){console.debug('convo: dropped Whisper hallucination:',JSON.stringify(text));_setConvoState('listening');return}
    const gated=_wakeWordGate(text);
    if(gated===null){console.debug('convo: no wake word — ignoring',JSON.stringify(text));_flashAmbient(text);_setConvoState('listening');return}
    convoFailCount=0;
    _setConvoState('thinking');
    input.value='';
    await _convoStreamSend(gated);
  }catch(e){console.warn('convo transcribe failed',e);convoFailCount++;if(convoFailCount>=CONVO_MAX_FAILS){_convoFail('transcribe error')}else _setConvoState('listening')}
}
function _convoFail(reason){
  console.warn('convo disabling:',reason);
  convoOn=false;localStorage.setItem(CONVO_KEY,'0');_convoToggle.classList.remove('on');_setConvoState('error');_convoBanner.classList.remove('show');
  if(convoRecorder){try{convoRecorder.stop()}catch{}}
  if(convoStream){convoStream.getTracks().forEach(t=>t.stop());convoStream=null}
  if(convoCtx){try{convoCtx.close()}catch{};convoCtx=null}
}
async function _startConvo(){
  try{convoStream=await navigator.mediaDevices.getUserMedia({audio:true})}
  catch(e){bubble('bot','Webcam permission denied: '+e.message,'<span class="badge err">convo</span>');_convoFail('mic denied');return}
  try{
    convoCtx=new (window.AudioContext||window.webkitAudioContext)();
    const src=convoCtx.createMediaStreamSource(convoStream);
    convoAnalyser=convoCtx.createAnalyser();convoAnalyser.fftSize=256;convoAnalyser.smoothingTimeConstant=0.5;
    src.connect(convoAnalyser);
  }catch(e){_convoFail('audio context: '+e.message);return}
  convoFailCount=0;_setConvoState('listening');_convoBanner.classList.add('show');convoActive=false;
  _prewarmWhisper();
  _vadLoop();
}
async function _streamReplyWithTTS(text,opts){
  opts=opts||{};
  const createBubbles=opts.createBubbles!==false;
  const onDoneState=opts.onDone||null;
  const useTTS=opts.tts!==false && voiceOut;
  if(createBubbles)bubble('user',text);
  const bot=opts.bot||bubble('bot','...');
  if(!opts.bot)bot.bubble.classList.add('thinking');
  const tokMeter=document.createElement('div');tokMeter.className='tok-meter';tokMeter.textContent='— tok/s · 0 tok · 0.0s';bot.msg.appendChild(tokMeter);
  const t0=performance.now();let tokCount=0,lastUpdate=0;
  function _updateTokMeter(force){
    const now=performance.now();if(!force&&now-lastUpdate<180)return;lastUpdate=now;
    const sec=Math.max(0.01,(now-t0)/1000);const rate=tokCount/sec;
    tokMeter.textContent=`${rate>=10?rate.toFixed(0):rate.toFixed(1)} tok/s · ${tokCount} tok · ${sec.toFixed(1)}s`;
  }
  _stopAllTTS();const _myGen=_ttsGen;
  let acc='',spoken='',ttsQueue=[],ttsPlaying=false;
  const _SENT_RE=/([.!?…][\s"')\]\}]*)/;
  async function _flushTTS(chunk){
    if(!chunk||!chunk.trim())return;
    if(_myGen!==_ttsGen)return;
    if(!useTTS)return;
    if(!_voiceBackends.tts){_speakBrowser(chunk);return}
    try{
      const body={text:chunk};if(_selectedVoice)body.voice=_selectedVoice;
      const r=await fetch('/voice/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if(!r.ok){_speakBrowser(chunk);return}
      const j=await r.json();if(!j.audio_base64){_speakBrowser(chunk);return}
      ttsQueue.push('data:'+(j.content_type||'audio/wav')+';base64,'+j.audio_base64);
      _drainTTS();
    }catch{_speakBrowser(chunk)}
  }
  function _drainTTS(){
    if(_myGen!==_ttsGen){ttsQueue.length=0;return}
    if(ttsPlaying||ttsQueue.length===0)return;
    ttsPlaying=true;const url=ttsQueue.shift();
    if(_audioEl){try{_audioEl.pause()}catch{}}
    _audioEl=new Audio(url);
    _audioEl.onended=()=>{ttsPlaying=false;_drainTTS();if(ttsQueue.length===0){if(convoOn)_setConvoState('listening');if(onDoneState)onDoneState()}};
    _audioEl.onerror=()=>{ttsPlaying=false;_drainTTS()};
    _audioEl.play().catch(()=>{ttsPlaying=false;_drainTTS()});
  }
  function _consumeSentence(){
    const m=acc.slice(spoken.length).match(_SENT_RE);
    if(m){const idx=m.index+m[0].length;const piece=acc.slice(spoken.length,spoken.length+idx);spoken+=piece;_flushTTS(piece)}
  }
  try{
    const resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sid?{message:text,session_id:sid}:{message:text})});
    const reader=resp.body.getReader();const decoder=new TextDecoder();let buf='';
    while(true){
      const {done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const events=buf.split('\n\n');buf=events.pop()||'';
      for(const evt of events){
        const lines=evt.split('\n');let etype='message',edata='';
        for(const ln of lines){if(ln.startsWith('event: '))etype=ln.slice(7);else if(ln.startsWith('data: '))edata+=ln.slice(6)}
        if(!edata)continue;
        try{
          if(etype==='token'){const chunk=JSON.parse(edata);if(bot.bubble.classList.contains('thinking')){bot.bubble.classList.remove('thinking');bot.bubble.textContent=''}acc+=chunk;tokCount+=Math.max(1,Math.ceil(chunk.length/4));_updateTokMeter();bot.bubble.innerHTML=md(acc);_consumeSentence();log.scrollTop=log.scrollHeight;if(typeof _corePulse==='function'&&performance.now()-_coreLastToken>180){_coreLastToken=performance.now();_corePulse()}}
          else if(etype==='meta'){const m=JSON.parse(edata);if(m.session_id){sid=m.session_id;localStorage.setItem(SKEY,sid)}}
          else if(etype==='done'){const tail=acc.slice(spoken.length);if(tail.trim()){spoken=acc;_flushTTS(tail)};_updateTokMeter(true);tokMeter.classList.add('done');setTimeout(()=>{tokMeter.classList.add('fade')},2500);setTimeout(()=>{try{tokMeter.remove()}catch{}},3800)}
        }catch(p){}
      }
    }
  }catch(e){bot.bubble.classList.remove('thinking');bot.bubble.textContent='stream error: '+e.message;try{tokMeter.remove()}catch{};if(convoOn)_setConvoState('listening')}
}
async function _convoStreamSend(text){return _streamReplyWithTTS(text,{createBubbles:true})}
function _stopConvo(){
  if(convoRAF){cancelAnimationFrame(convoRAF);convoRAF=null}
  if(convoRecorder&&convoRecorder.state==='recording'){try{convoRecorder.stop()}catch{}}
  if(convoStream){convoStream.getTracks().forEach(t=>t.stop());convoStream=null}
  if(convoCtx){try{convoCtx.close()}catch{};convoCtx=null}
  convoAnalyser=null;_setConvoState('idle');_convoBanner.classList.remove('show');convoActive=false;
}
function toggleConvo(){
  convoOn=!convoOn;localStorage.setItem(CONVO_KEY,convoOn?'1':'0');_convoToggle.classList.toggle('on',convoOn);
  if(convoOn){
    if(!_voiceBackends.stt){bubble('bot','Convo mode needs local STT (faster-whisper). Install: pip install faster-whisper','<span class="badge err">convo</span>');convoOn=false;_convoToggle.classList.remove('on');return}
    if(typeof MediaRecorder==='undefined'||!navigator.mediaDevices){bubble('bot','Convo mode needs MediaRecorder + getUserMedia. Use a recent browser.','<span class="badge err">convo</span>');convoOn=false;_convoToggle.classList.remove('on');return}
    voiceOut=true;localStorage.setItem(VKEY,'1');document.getElementById('voiceout-toggle').classList.add('on');
    _startConvo();
  }else _stopConvo();
}
const _origSpeak=speak;
window.speak=async function(text){
  if(convoOn)_setConvoState('speaking');
  try{await _origSpeak.call(this,text)}catch{}
  if(convoOn){
    if(_audioEl){
      const tick=()=>{if(_audioEl&&!_audioEl.paused&&!_audioEl.ended){requestAnimationFrame(tick);return}_setConvoState('listening')};
      requestAnimationFrame(tick);
    }else _setConvoState('listening');
  }
};
if(localStorage.getItem(CONVO_KEY)==='1'){setTimeout(()=>{convoOn=true;_convoToggle.classList.add('on');voiceOut=true;localStorage.setItem(VKEY,'1');document.getElementById('voiceout-toggle').classList.add('on');_startConvo()},1200)}
document.addEventListener('paste',async e=>{
  if(!e.clipboardData)return;
  for(const item of e.clipboardData.items){
    if(item.type&&item.type.startsWith('image/')){
      const blob=item.getAsFile();if(blob){e.preventDefault();await handleImageBlob(blob);return}
    }
  }
});
const _dropOverlay=document.getElementById('drop-overlay');
let _dragDepth=0;
document.addEventListener('dragenter',e=>{if(e.dataTransfer&&Array.from(e.dataTransfer.types||[]).includes('Files')){e.preventDefault();_dragDepth++;_dropOverlay.classList.add('show')}});
document.addEventListener('dragover',e=>{if(_dropOverlay.classList.contains('show'))e.preventDefault()});
document.addEventListener('dragleave',()=>{_dragDepth=Math.max(0,_dragDepth-1);if(_dragDepth===0)_dropOverlay.classList.remove('show')});
const _TEXT_DROP_EXT=new Set(['md','txt','py','js','mjs','ts','tsx','jsx','rs','go','c','cpp','h','hpp','cs','java','kt','swift','rb','php','sh','bash','zsh','ps1','yaml','yml','toml','json','xml','html','htm','css','scss','sql','log','csv','tsv','ini','cfg','conf','env','lock','gitignore','dockerfile','makefile','rst','tex','vue','svelte','astro','lua','pl','r','jl','dart','m','f90','f95','asm']);
const _TEXT_DROP_MAX=200_000;
async function handleTextFileBlob(file){
  try{
    const text=await file.text();
    const ext=(file.name.split('.').pop()||'').toLowerCase();
    const truncated=text.length>_TEXT_DROP_MAX;
    const body=truncated?text.slice(0,_TEXT_DROP_MAX):text;
    const head=`Dropped file **${esc(file.name)}** (${file.size} bytes${truncated?', truncated to '+_TEXT_DROP_MAX:''}). Asking Adam to analyze it…`;
    bubble('user',head,'<span class="badge">drop</span>');
    const fence=ext||'text';
    const prompt=`I just dropped a file into the chat. Please analyze it. File: \`${file.name}\` · size: ${file.size} bytes${truncated?' (truncated)':''}\n\n\`\`\`${fence}\n${body}\n\`\`\``;
    input.value=prompt;await send();
  }catch(e){bubble('bot','Could not read dropped file: '+esc(e.message),'<span class="badge err">drop</span>')}
}
function _isTextDrop(f){
  if(!f||!f.name)return false;
  const ext=(f.name.split('.').pop()||'').toLowerCase();
  if(_TEXT_DROP_EXT.has(ext))return true;
  if(f.type&&(f.type.startsWith('text/')||f.type==='application/json'||f.type==='application/x-yaml'||f.type==='application/xml'))return true;
  return false;
}
document.addEventListener('drop',async e=>{
  _dragDepth=0;_dropOverlay.classList.remove('show');
  if(!e.dataTransfer)return;
  for(const f of e.dataTransfer.files||[]){
    if(f.type&&f.type.startsWith('image/')){e.preventDefault();await handleImageBlob(f);return}
    if(_isTextDrop(f)){e.preventDefault();await handleTextFileBlob(f);return}
  }
});
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
</script><script>(function(){var n=0;function poll(){if(++n>40)return;fetch('/healthz').then(function(r){return r.json()}).then(function(j){var w=(j&&j.warmup)||{},p=document.getElementById('_codewarm');if(w.coding){if(p)p.remove();return}if(!p){p=document.createElement('div');p.id='_codewarm';p.style.cssText='position:fixed;bottom:14px;right:14px;z-index:9999;font:12px system-ui,sans-serif;color:var(--text2,#8aa);background:var(--panel,#0b1322);border:1px solid var(--border,#1b2b44);border-radius:14px;padding:6px 12px;opacity:.85;box-shadow:0 2px 10px rgba(0,0,0,.3);display:flex;align-items:center;gap:7px';p.innerHTML='<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--accent,#00e5ff);animation:_cwp 1.1s ease-in-out infinite"></span>warming up coding tools…';var st=document.createElement('style');st.textContent='@keyframes _cwp{0%,100%{opacity:.35}50%{opacity:1}}';document.head.appendChild(st);document.body.appendChild(p)}setTimeout(poll,2500)}).catch(function(){setTimeout(poll,4000)})}if(document.readyState!=='loading')poll();else document.addEventListener('DOMContentLoaded',poll)})();</script></body></html>"""
def mount(app):
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _AP
    _assets=_AP(__file__).resolve().parent/'assets'
    if _assets.exists():app.mount('/assets',StaticFiles(directory=str(_assets)),name='amni_assets')
    @app.get('/jarvis',response_class=HTMLResponse)
    def jarvis():return HTMLResponse(content=_HTML)
    @app.post('/launch/peer')
    def _launch_peer(app_name:str='amni-code'):
        import subprocess,sys,os
        root=_AP(__file__).resolve().parents[2]
        peer=_AP(os.environ.get('AMNI_CODE_DIR') or (root.parent/'Amni-Code'))
        url='http://localhost:3000'
        if not peer.exists():return {'ok':False,'error':f'Amni-Code not found at {peer} (set AMNI_CODE_DIR)','url':url}
        cands=[peer/'target'/'release'/'amni.exe',peer/'target'/'release'/'amni',peer/'run.bat',peer/'amni-launcher.cmd']
        exe=next((c for c in cands if c.exists()),None)
        if exe is None:return {'ok':False,'error':'no built binary or launcher found — run `cargo build --release` in Amni-Code','url':url}
        try:
            _win=sys.platform.startswith('win')
            _flags=(getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)) if _win else 0
            _cmd=['cmd','/c',str(exe)] if (_win and exe.suffix.lower() in ('.bat','.cmd')) else [str(exe)]
            subprocess.Popen(_cmd,cwd=str(peer),creationflags=_flags,start_new_session=(not _win),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            return {'ok':True,'launched':str(exe),'url':url}
        except Exception as _le:return {'ok':False,'error':str(_le),'url':url}
    @app.get('/notifications')
    def _notifs(limit:int=20,include_read:bool=False):
        from amni.serve.notifications import list_active,stats
        return {'items':list_active(limit=limit,include_read=include_read),'stats':stats()}
    @app.post('/notifications/{nid}/read')
    def _notifs_read(nid:str):
        from amni.serve.notifications import mark_read
        return {'marked':mark_read(nid),'id':nid}
    @app.post('/notifications/read-all')
    def _notifs_read_all():
        from amni.serve.notifications import mark_all_read
        return {'marked':mark_all_read()}
    @app.get('/memory/skill-stats')
    def _skill_stats(hours:float=24,limit_per_skill:int=2000):
        from amni.serve.skill_stats import aggregate
        return aggregate(hours=hours if hours>0 else None,limit_per_skill=limit_per_skill)
    @app.get('/memory/digest')
    def _digest(hours:int=24):
        import time as _t,json as _j
        from pathlib import Path as _P
        cutoff=_t.time()-max(1,hours)*3600;out={'hours':hours,'cutoff_ts':cutoff,'generated_at':_t.time()}
        data_dir=_P(__file__).resolve().parents[2]/'data'
        out['learning']={'facts_today':0,'topics_today':[],'current_topic':None,'enabled':False}
        try:
            from amni.serve import notifications as _nf
            ld_notifs=[n for n in _nf.list_active(limit=200,include_read=True) if n.get('source')=='learning_daemon' and n.get('ts',0)>=cutoff]
            out['learning']['facts_today']=sum(((n.get('extras') or {}).get('new',0) or 0) for n in ld_notifs)
            out['learning']['topics_today']=[((n.get('extras') or {}).get('topic') or n.get('title','').replace('Learned about ','')) for n in ld_notifs[:8]]
        except Exception:pass
        out['shell']={'runs_today':0,'errors_today':0,'kinds':{}}
        sh_log=data_dir/'shell_history.jsonl'
        if sh_log.exists():
            try:
                for ln in sh_log.read_text(encoding='utf-8').splitlines():
                    if not ln.strip():continue
                    try:r=_j.loads(ln)
                    except Exception:continue
                    if float(r.get('ts') or 0)<cutoff:continue
                    out['shell']['runs_today']+=1
                    if (r.get('returncode') or 0)!=0:out['shell']['errors_today']+=1
                    k=r.get('kind','?');out['shell']['kinds'][k]=out['shell']['kinds'].get(k,0)+1
            except Exception:pass
        out['verifier']={'pass_today':0,'fail_today':0,'pending':0}
        v_log=data_dir/'verification_log.jsonl'
        if v_log.exists():
            try:
                for ln in v_log.read_text(encoding='utf-8').splitlines():
                    if not ln.strip():continue
                    try:r=_j.loads(ln)
                    except Exception:continue
                    if float(r.get('ts') or 0)<cutoff:continue
                    if r.get('verified') is True:out['verifier']['pass_today']+=1
                    elif r.get('verified') is False:out['verifier']['fail_today']+=1
            except Exception:pass
        nt_log=data_dir/'needs_testing.jsonl'
        if nt_log.exists():
            try:
                for ln in nt_log.read_text(encoding='utf-8').splitlines():
                    if not ln.strip():continue
                    try:r=_j.loads(ln)
                    except Exception:continue
                    if r.get('status')=='pending':out['verifier']['pending']+=1
            except Exception:pass
        out['coach']={'streak_days':0,'today_active':False,'total_days_active':0,'topics':0}
        try:
            from pathlib import Path as _P2
            from amni.storage.coach_atlas import CoachAtlas
            ca=CoachAtlas(root='experiences/coach_atlas');s=ca.streak_stats() if hasattr(ca,'streak_stats') else {}
            out['coach'].update({'streak_days':s.get('current_streak',0),'today_active':bool(s.get('today_active')),'total_days_active':s.get('total_days_active',0),'topics':len(ca.list_topics())})
        except Exception:pass
        out['skills']={'n_calls':0,'avg_ms':0,'top':[]}
        try:
            from amni.serve.skill_stats import aggregate as _agg
            ss=_agg(hours=hours);tot=ss.get('totals') or {};sk=ss.get('skills') or {}
            top=[{'name':n,'n':d['n_calls'],'avg_ms':d['avg_ms'],'ok_rate':d['ok_rate']} for n,d in list(sk.items())[:5]]
            out['skills']={'n_calls':tot.get('n_calls',0),'avg_ms':tot.get('avg_ms',0),'overall_ok_rate':tot.get('overall_ok_rate',0.0),'top':top}
        except Exception:pass
        return out
