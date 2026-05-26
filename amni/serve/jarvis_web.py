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
#frame::before,#frame::after{content:'';position:absolute;width:14px;height:14px;border:1.5px solid var(--cyan);box-shadow:0 0 8px rgba(0,229,255,.5)}
#frame::before{top:-2px;left:-2px;border-right:none;border-bottom:none}
#frame::after{bottom:-2px;right:-2px;border-left:none;border-top:none}
#corner-tr,#corner-bl{position:fixed;width:14px;height:14px;border:1.5px solid var(--cyan);box-shadow:0 0 8px rgba(0,229,255,.5);pointer-events:none;z-index:5}
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
.widget.file_change .fc-head{display:flex;gap:8px;align-items:baseline;font-size:13px;flex-wrap:wrap}
.widget.file_change .fc-op{font-size:9px;letter-spacing:.18em;padding:2px 6px;border-radius:2px;text-transform:uppercase;font-weight:bold}
.widget.file_change .fc-op.op-create{background:rgba(0,255,156,.15);color:#00ff9c;border:1px solid rgba(0,255,156,.3)}
.widget.file_change .fc-op.op-edit{background:rgba(0,229,255,.12);color:var(--cyan);border:1px solid rgba(0,229,255,.3)}
.widget.file_change .fc-op.op-overwrite{background:rgba(255,181,71,.12);color:#ffb547;border:1px solid rgba(255,181,71,.3)}
.widget.file_change .fc-bn{color:var(--fg);font-weight:600;font-family:JetBrains Mono,monospace}
.widget.file_change .fc-ext{color:var(--mute);font-size:11px;font-family:JetBrains Mono,monospace}
.widget.file_change .fc-folder{font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:4px;font-family:JetBrains Mono,monospace;word-break:break-all}
.widget.file_change .fc-stats{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;letter-spacing:.1em;color:var(--mute);margin:8px 0;border-top:1px solid rgba(0,229,255,.08);padding-top:6px}
.widget.file_change .fc-add{color:#00ff9c;font-weight:bold}
.widget.file_change .fc-rem{color:#ff7b7b;font-weight:bold}
.widget.file_change .fc-repl{color:var(--cyan)}
.widget.file_change .fc-size{margin-left:auto;color:var(--mute)}
.widget.file_change .fc-preview{font-size:10px;font-family:JetBrains Mono,monospace;background:rgba(0,0,0,.35);border:1px solid rgba(0,229,255,.1);border-radius:3px;padding:6px 8px;max-height:160px;overflow:auto;color:var(--fg);white-space:pre;line-height:1.4;margin:4px 0}
.widget.file_change .fc-actions{display:flex;gap:6px;margin-top:6px}
.widget.file_change .fc-btn{flex:0 0 auto;padding:4px 10px;background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.25);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
.widget.file_change .fc-btn:hover{background:rgba(0,229,255,.14);border-color:var(--cyan)}
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
.widget.news .news-item{padding:6px 8px;border:1px solid rgba(0,229,255,.08);border-radius:3px;background:rgba(0,229,255,.02);text-decoration:none;color:var(--fg);display:block;transition:all .15s}
.widget.news .news-item:hover{border-color:rgba(0,229,255,.4);background:rgba(0,229,255,.05);box-shadow:0 0 8px rgba(0,229,255,.15)}
.widget.news .news-item .title{font-size:12px;line-height:1.3}
.widget.news .news-item .src{font-size:9px;color:var(--cyan);letter-spacing:.15em;text-transform:uppercase;margin-top:3px}
.widget.stock .quotes{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.widget.stock .quote{padding:10px;border:1px solid rgba(0,229,255,.15);border-radius:3px;background:rgba(0,229,255,.02)}
.widget.stock .quote .sym{font-size:14px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);letter-spacing:.1em;font-weight:600}
.widget.stock .quote .name{font-size:9px;color:var(--mute);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
.widget.stock .quote .price{font-size:22px;color:var(--fg);text-shadow:0 0 4px rgba(0,229,255,.4)}
.widget.stock .quote .chg{font-size:11px;margin-top:3px}
.widget.stock .quote .chg.up{color:var(--ok);text-shadow:0 0 4px var(--ok)}
.widget.stock .quote .chg.down{color:var(--err);text-shadow:0 0 4px var(--err)}
.widget.stock .quote .meta{font-size:9px;color:var(--mute);margin-top:6px;border-top:1px solid rgba(0,229,255,.08);padding-top:4px}
.widget.file .file-meta{display:flex;gap:12px;font-size:9px;color:var(--mute);letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px}
.widget.file .file-meta .v{color:var(--cyan);margin-left:4px}
.widget.file pre{background:rgba(0,0,0,.6);padding:10px;border-radius:2px;font-size:10px;color:var(--cyan);max-height:300px;overflow:auto;line-height:1.4}
.widget.disk .partitions{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}
.widget.disk .part{padding:8px 10px;border:1px solid rgba(0,229,255,.12);border-radius:3px;background:rgba(0,229,255,.02)}
.widget.disk .part .mount{font-size:11px;color:var(--cyan);text-shadow:0 0 4px var(--cyan);letter-spacing:.1em;font-family:Consolas,monospace}
.widget.disk .part .stat{font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:var(--mute);margin-top:4px}
.widget.disk .part .stat .v{color:var(--fg);margin-left:4px}
.widget.disk .part .bar{height:3px;background:rgba(0,229,255,.1);border-radius:2px;margin-top:6px;overflow:hidden}
.widget.disk .part .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 4px var(--cyan)}
.widget.git .git-branch{font-size:14px;color:var(--cyan);text-shadow:0 0 6px var(--cyan);letter-spacing:.1em}
.widget.git .git-stats{display:flex;gap:14px;font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--mute);margin:6px 0 10px;padding-bottom:6px;border-bottom:1px solid rgba(0,229,255,.08)}
.widget.git .git-stats .v{color:var(--cyan);margin-left:4px}
.widget.git .git-stats .v.dirty{color:var(--gold)}
.widget.git .commits{font-size:10px;color:var(--fg);font-family:Consolas,monospace;line-height:1.5}
.widget.git .commits .row{padding:2px 0;border-bottom:1px solid rgba(0,229,255,.05)}
.widget.git .commits .row:last-child{border-bottom:none}
.widget.git .commits .row .sha{color:var(--cyan);margin-right:8px}
.widget.git .dirty-files{margin-top:8px;font-size:10px;color:var(--gold);font-family:Consolas,monospace}
.widget.git .dirty-files .lbl{color:var(--mute);font-size:9px;letter-spacing:.15em;text-transform:uppercase;margin-bottom:3px}
.widget.watch .watch-head{font-size:11px;color:var(--cyan);letter-spacing:.1em;font-family:Consolas,monospace;margin-bottom:8px;border-bottom:1px solid rgba(0,229,255,.1);padding-bottom:6px;display:flex;justify-content:space-between}
.widget.watch .watch-head .id{color:var(--mute);font-size:9px;letter-spacing:.2em}
.widget.watch .events{display:flex;flex-direction:column;gap:3px;font-family:Consolas,monospace;font-size:10px;max-height:260px;overflow-y:auto}
.widget.watch .event{padding:3px 6px;border-radius:2px;background:rgba(0,229,255,.02);display:grid;grid-template-columns:60px 1fr auto;gap:8px;align-items:center}
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
.sidehint{position:fixed;bottom:14px;left:36px;font-size:9px;color:var(--mute);letter-spacing:.2em;z-index:3;pointer-events:none;opacity:.6}
.status .pill.clickable{cursor:pointer}
.status .pill.clickable:hover{border-color:var(--cyan);background:rgba(0,229,255,.12)}
#persona-panel{position:fixed;top:60px;right:24px;width:300px;z-index:11;border:1px solid rgba(0,229,255,.4);border-radius:4px;background:rgba(8,14,28,.96);box-shadow:0 0 28px rgba(0,229,255,.22);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#persona-panel.show{display:block}
#persona-panel .pp-head{padding:10px 14px;border-bottom:1px solid rgba(0,229,255,.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(8,14,28,.98)}
#persona-panel .pp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(0,229,255,.2);border-radius:3px;font-size:10px}
#persona-panel .pp-head .close:hover{color:var(--err);border-color:var(--err)}
#persona-panel .pp-section{padding:12px 14px;border-bottom:1px solid rgba(0,229,255,.08)}
#persona-panel .pp-section h3{font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
#persona-panel .pp-row{padding:6px 8px;border:1px solid rgba(0,229,255,.12);border-radius:3px;margin-bottom:4px;cursor:pointer;font-size:11px;display:flex;justify-content:space-between;align-items:center;transition:all .15s;background:rgba(0,229,255,.02)}
#persona-panel .pp-row:hover{border-color:rgba(0,229,255,.4);background:rgba(0,229,255,.08)}
#persona-panel .pp-row.active{border-color:var(--cyan);background:rgba(0,229,255,.14);box-shadow:inset 0 0 6px rgba(0,229,255,.2)}
#persona-panel .pp-row .nm{color:var(--fg);text-transform:capitalize}
#persona-panel .pp-row.active .nm{color:var(--cyan);text-shadow:0 0 4px var(--cyan)}
#persona-panel .pp-row .voice{font-size:9px;color:var(--mute);letter-spacing:.1em}
#persona-panel .pp-empty{font-size:10px;color:var(--mute);text-align:center;padding:10px;font-style:italic}
#persona-panel .pp-input-row{display:flex;gap:6px}
#persona-panel input[type=text]{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(0,229,255,.2);color:var(--fg);padding:5px 8px;border-radius:3px;font-family:inherit;font-size:11px}
#persona-panel input[type=text]:focus{outline:none;border-color:var(--cyan)}
#persona-panel button.act{padding:5px 10px;border:1px solid rgba(0,229,255,.4);background:rgba(0,229,255,.04);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#persona-panel button.act:hover{background:rgba(0,229,255,.14)}
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
#tests-panel{position:fixed;top:60px;right:24px;width:380px;z-index:11;border:1px solid rgba(255,181,71,.4);border-radius:4px;background:rgba(8,14,28,.96);box-shadow:0 0 28px rgba(255,181,71,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#tests-panel.show{display:block}
#tests-panel .tp-head{padding:10px 14px;border-bottom:1px solid rgba(255,181,71,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:#ffb547;text-shadow:0 0 4px #ffb547;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(8,14,28,.98)}
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
.sh-led{display:inline-block;width:6px;height:6px;border-radius:50%;background:#4a5568;box-shadow:0 0 4px #4a5568;margin-right:6px;vertical-align:middle;transition:background .25s}
.sh-led.clean{background:#00e5ff;box-shadow:0 0 6px #00e5ff}
.sh-led.dirty{background:#ff7b7b;box-shadow:0 0 8px #ff7b7b, 0 0 14px rgba(255,123,123,.4);animation:shPulse 1.4s ease-in-out infinite}
.sh-led.error{background:#4a5568}
@keyframes shPulse{0%,100%{opacity:1}50%{opacity:.55}}
#shell-panel{position:fixed;top:60px;right:24px;width:480px;z-index:11;border:1px solid rgba(0,229,255,.4);border-radius:4px;background:rgba(8,14,28,.96);box-shadow:0 0 28px rgba(0,229,255,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#shell-panel.show{display:block}
#shell-panel .sh-head{padding:10px 14px;border-bottom:1px solid rgba(0,229,255,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(8,14,28,.98)}
#shell-panel .sh-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(0,229,255,.22);border-radius:3px;font-size:10px}
#shell-panel .sh-head .close:hover{color:var(--err);border-color:var(--err)}
#shell-panel .sh-toolbar{display:flex;gap:6px;padding:8px 14px;border-bottom:1px solid rgba(0,229,255,.08);font-size:10px}
#shell-panel .sh-toolbar button{padding:4px 8px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.04);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.18em;cursor:pointer;border-radius:3px}
#shell-panel .sh-toolbar button:hover{background:rgba(0,229,255,.12)}
#shell-panel .sh-toolbar button.on{background:rgba(255,123,123,.12);border-color:#ff7b7b;color:#ff7b7b}
#shell-panel .sh-summary{font-size:10px;color:var(--mute);letter-spacing:.05em;margin-left:auto;align-self:center}
#shell-panel .sh-section{padding:8px 14px}
#shell-panel .sh-item{padding:8px 10px;border-left:2px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);margin-bottom:6px;font-family:JetBrains Mono,monospace}
#shell-panel .sh-item.fail{border-left-color:#ff7b7b;background:rgba(255,123,123,.05)}
#shell-panel .sh-item .cmd{font-size:11px;color:var(--fg);word-break:break-all;line-height:1.4}
#shell-panel .sh-item .meta{display:flex;gap:8px;font-size:9px;color:var(--mute);letter-spacing:.1em;margin-top:4px;text-transform:uppercase;flex-wrap:wrap}
#shell-panel .sh-item .kind{display:inline-block;padding:1px 5px;border-radius:2px;background:rgba(0,229,255,.12);color:var(--cyan);font-size:8px;font-weight:bold}
#shell-panel .sh-item.fail .kind{background:rgba(255,123,123,.15);color:#ff7b7b}
#shell-panel .sh-item .rc{padding:1px 5px;border-radius:2px;font-size:8px;font-weight:bold}
#shell-panel .sh-item .rc.ok{background:rgba(0,255,156,.12);color:#00ff9c}
#shell-panel .sh-item .rc.bad{background:rgba(255,91,91,.12);color:#ff7b7b}
#shell-panel .sh-item .age{margin-left:auto;color:var(--mute);font-size:9px}
#shell-panel .sh-item .toggle{cursor:pointer;color:var(--cyan);font-size:9px;letter-spacing:.15em;margin-top:6px;display:inline-block;text-transform:uppercase}
#shell-panel .sh-item pre{margin:6px 0 0 0;font-size:9.5px;background:rgba(0,0,0,.4);padding:6px 8px;border-radius:2px;max-height:160px;overflow:auto;color:var(--fg);line-height:1.35;display:none;white-space:pre-wrap;word-break:break-all}
#shell-panel .sh-item pre.show{display:block}
#shell-panel .sh-empty{padding:24px;text-align:center;color:var(--mute);font-size:10px;font-style:italic}
#chat-search{position:fixed;top:60px;left:50%;transform:translateX(-50%) translateY(-20px);width:min(560px,90vw);z-index:13;background:rgba(8,14,28,.97);border:1px solid rgba(0,229,255,.5);border-radius:4px;padding:10px 14px;box-shadow:0 0 28px rgba(0,229,255,.28);backdrop-filter:blur(8px);display:none;opacity:0;transition:opacity .2s, transform .2s}
#chat-search.show{display:block;opacity:1;transform:translateX(-50%) translateY(0)}
#chat-search .cs-row{display:flex;gap:8px;align-items:center}
#chat-search input{flex:1;background:rgba(0,0,0,.4);border:1px solid rgba(0,229,255,.25);color:var(--fg);padding:7px 11px;border-radius:3px;font-family:inherit;font-size:13px;letter-spacing:.02em}
#chat-search input:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 8px rgba(0,229,255,.4)}
#chat-search .cs-count{font-size:10px;color:var(--cyan);letter-spacing:.15em;min-width:60px;text-align:right}
#chat-search .cs-btn{padding:4px 8px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.06);color:var(--cyan);font-family:inherit;font-size:11px;cursor:pointer;border-radius:3px}
#chat-search .cs-btn:hover{background:rgba(0,229,255,.16)}
#chat-search .cs-help{font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace}
.msg.cs-hidden{display:none}
mark.cs-hit{background:rgba(255,224,102,.32);color:var(--fg);padding:0 2px;border-radius:2px;box-shadow:0 0 6px rgba(255,224,102,.4)}
mark.cs-hit.current{background:rgba(0,255,156,.4);box-shadow:0 0 8px rgba(0,255,156,.6);color:#fff}
.restore-banner{display:flex;align-items:center;gap:10px;padding:6px 12px;border:1px dashed rgba(0,229,255,.25);background:rgba(0,229,255,.04);border-radius:3px;font-size:9px;color:var(--mute);letter-spacing:.18em;text-transform:uppercase;margin:4px 0 12px;font-family:JetBrains Mono,monospace}
.restore-banner .rb-close{margin-left:auto;cursor:pointer;color:var(--cyan);padding:1px 6px;border:1px solid rgba(0,229,255,.3);border-radius:2px;font-size:8px}
.restore-banner .rb-close:hover{background:rgba(0,229,255,.12)}
.msg.restored{opacity:.78}
.msg.restored .bubble{border-left:2px solid rgba(255,255,255,.12)}
.msg.restored .bubble::before{content:'';display:none}
#toast-stack{position:fixed;bottom:140px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:14;max-width:340px;pointer-events:none}
.toast{pointer-events:auto;padding:10px 12px;border:1px solid rgba(0,229,255,.35);background:rgba(8,14,28,.94);border-left:3px solid var(--cyan);border-radius:3px;font-size:11px;color:var(--fg);box-shadow:0 0 14px rgba(0,229,255,.18);backdrop-filter:blur(6px);transform:translateX(60px);opacity:0;transition:transform .25s ease-out, opacity .25s ease-out;cursor:pointer;font-family:inherit}
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
#coach-panel{position:fixed;top:60px;right:24px;width:440px;z-index:11;border:1px solid rgba(255,77,200,.4);border-radius:4px;background:rgba(8,14,28,.96);box-shadow:0 0 28px rgba(255,77,200,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#coach-panel.show{display:block}
#coach-panel .cp-head{padding:10px 14px;border-bottom:1px solid rgba(255,77,200,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--magenta);text-shadow:0 0 4px var(--magenta);display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(8,14,28,.98)}
#coach-panel .cp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(255,77,200,.22);border-radius:3px;font-size:10px}
#coach-panel .cp-head .close:hover{color:var(--err);border-color:var(--err)}
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
#coach-panel .cp-grade.ok{background:rgba(255,224,102,.06);border:1px solid rgba(255,224,102,.3);color:#ffe6a0}
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
#coach-panel .cp-hint{font-size:11px;color:#ffe066;background:rgba(255,224,102,.05);border-left:2px solid #ffe066;padding:6px 10px;border-radius:0 3px 3px 0;margin-top:8px;font-style:italic}
@keyframes ldPulse{0%,100%{opacity:1}50%{opacity:.45}}
#learn-panel{position:fixed;top:60px;right:24px;width:340px;z-index:11;border:1px solid rgba(0,255,156,.4);border-radius:4px;background:rgba(8,14,28,.96);box-shadow:0 0 28px rgba(0,255,156,.18);backdrop-filter:blur(8px);display:none;max-height:calc(100vh - 120px);overflow-y:auto}
#learn-panel.show{display:block}
#learn-panel .lp-head{padding:10px 14px;border-bottom:1px solid rgba(0,255,156,.22);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:#00ff9c;text-shadow:0 0 4px #00ff9c;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;background:rgba(8,14,28,.98)}
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
#task-tray{position:fixed;left:50%;bottom:88px;transform:translateX(-50%) translateY(140%);width:min(560px,92vw);z-index:8;border:1px solid rgba(0,229,255,.3);border-radius:4px;background:rgba(8,14,28,.95);box-shadow:0 0 24px rgba(0,229,255,.18);transition:transform .25s ease-out;backdrop-filter:blur(6px);max-height:30vh;overflow-y:auto}
#task-tray.show{transform:translateX(-50%) translateY(0)}
#task-tray .tray-head{padding:6px 12px;font-size:9px;letter-spacing:.25em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);border-bottom:1px solid rgba(0,229,255,.15);display:flex;align-items:center;gap:8px;position:sticky;top:0;background:rgba(8,14,28,.95)}
#task-tray .tray-head .dot{width:5px;height:5px;border-radius:50%;background:var(--ok);box-shadow:0 0 5px var(--ok);animation:pulse 1.6s ease-in-out infinite}
#task-tray .tray-head .ct{margin-left:auto;color:var(--mute);font-size:9px}
.task-row{padding:6px 12px;border-bottom:1px solid rgba(0,229,255,.06);display:grid;grid-template-columns:1fr auto;gap:8px;font-size:10px;align-items:center}
.task-row:last-child{border-bottom:none}
.task-row .label{color:var(--fg);letter-spacing:.05em;line-height:1.3}
.task-row .label .kind{color:var(--mute);font-size:8px;letter-spacing:.2em;text-transform:uppercase;margin-right:6px}
.task-row .label .msg{color:var(--mute);font-size:9px;margin-top:1px}
.task-row .pbar{height:3px;background:rgba(0,229,255,.1);border-radius:2px;overflow:hidden;margin-top:4px}
.task-row .pbar .fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));box-shadow:0 0 4px var(--cyan);transition:width .3s}
.task-row .cancel{background:transparent;border:1px solid rgba(255,85,119,.4);color:var(--err);font-family:inherit;font-size:9px;padding:3px 8px;border-radius:2px;cursor:pointer;letter-spacing:.15em}
.task-row .cancel:hover{background:rgba(255,85,119,.1)}
.task-row.cancelling .cancel{opacity:.4;cursor:wait}
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
#convo-toggle{padding:0 12px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px;position:relative}
#convo-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(0,229,255,.08);box-shadow:0 0 12px rgba(0,229,255,.3)}
#wake-toggle.on{color:#ffe066;border-color:#ffe066;background:rgba(255,224,102,.08);box-shadow:0 0 10px rgba(255,224,102,.28)}
.wake-ambient{position:fixed;bottom:140px;left:50%;transform:translateX(-50%);background:rgba(8,14,28,.85);border:1px solid rgba(255,224,102,.35);color:#ffe066;font-size:10px;letter-spacing:.18em;padding:6px 14px;border-radius:3px;backdrop-filter:blur(6px);z-index:6;opacity:0;pointer-events:none;transition:opacity .25s, transform .25s;text-transform:uppercase;font-family:inherit;text-shadow:0 0 4px rgba(255,224,102,.6)}
.wake-ambient.show{opacity:.85}
#convo-toggle .convo-dot{position:absolute;top:6px;right:6px;width:6px;height:6px;border-radius:50%;background:var(--mute);transition:all .2s}
#convo-toggle.on .convo-dot{background:var(--cyan);box-shadow:0 0 6px var(--cyan);animation:convoPulse 1.4s ease-in-out infinite}
#convo-toggle.state-recording .convo-dot{background:var(--cyan);box-shadow:0 0 10px var(--cyan);animation:convoPulse .8s ease-in-out infinite}
#convo-toggle.state-thinking .convo-dot{background:var(--gold);box-shadow:0 0 10px var(--gold);animation:convoSpin 1s linear infinite}
#convo-toggle.state-speaking .convo-dot{background:var(--magenta);box-shadow:0 0 10px var(--magenta);animation:convoPulse 1s ease-in-out infinite}
#convo-toggle.state-error .convo-dot{background:var(--err);box-shadow:0 0 8px var(--err)}
@keyframes convoPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.4)}}
@keyframes convoSpin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
#vad-panel{position:fixed;top:60px;left:24px;width:300px;z-index:10;border:1px solid rgba(0,229,255,.35);border-radius:4px;background:rgba(8,14,28,.94);box-shadow:0 0 24px rgba(0,229,255,.22);overflow:hidden;backdrop-filter:blur(8px);display:none}
#vad-panel.show{display:block}
#vad-panel .vp-head{padding:8px 12px;border-bottom:1px solid rgba(0,229,255,.2);font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--cyan);text-shadow:0 0 4px var(--cyan);display:flex;align-items:center;justify-content:space-between;background:rgba(8,14,28,.98)}
#vad-panel .vp-head .close{cursor:pointer;color:var(--mute);padding:1px 7px;border:1px solid rgba(0,229,255,.2);border-radius:3px;font-size:10px}
#vad-panel .vp-head .close:hover{color:var(--err);border-color:var(--err)}
#vad-panel .vp-body{padding:10px 12px}
#vad-panel .vp-row{margin-bottom:10px}
#vad-panel .vp-row .vp-lbl{font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:var(--mute);display:flex;justify-content:space-between;align-items:baseline}
#vad-panel .vp-row .vp-lbl .v{color:var(--cyan);font-size:11px}
#vad-panel input[type=range]{width:100%;-webkit-appearance:none;appearance:none;background:transparent;margin-top:5px;cursor:pointer}
#vad-panel input[type=range]::-webkit-slider-runnable-track{height:3px;background:rgba(0,229,255,.15);border-radius:2px}
#vad-panel input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);margin-top:-6px;cursor:grab}
#vad-panel input[type=range]::-moz-range-track{height:3px;background:rgba(0,229,255,.15);border-radius:2px}
#vad-panel input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);border:0;cursor:grab}
#vad-panel .vp-meter{padding:8px 0;border-top:1px solid rgba(0,229,255,.08);margin-top:8px}
#vad-panel .vp-meter .lbl{font-size:9px;letter-spacing:.2em;color:var(--mute);text-transform:uppercase;margin-bottom:4px}
#vad-panel .vp-meter .bar-track{height:8px;background:rgba(0,229,255,.05);border-radius:2px;position:relative;border:1px solid rgba(0,229,255,.15)}
#vad-panel .vp-meter .bar-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));transition:width .06s;border-radius:2px}
#vad-panel .vp-meter .threshold-line{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--gold);box-shadow:0 0 4px var(--gold)}
#vad-panel .vp-meter .barge-line{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--magenta);box-shadow:0 0 4px var(--magenta)}
#vad-panel .vp-row.act{display:flex;gap:6px;margin-top:12px;border-top:1px solid rgba(0,229,255,.08);padding-top:10px}
#vad-panel .vp-row.act button{flex:1;padding:5px 8px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--cyan);font-family:inherit;font-size:9px;letter-spacing:.2em;cursor:pointer;border-radius:3px}
#vad-panel .vp-row.act button:hover{background:rgba(0,229,255,.1);border-color:var(--cyan)}
#vad-panel .hint{font-size:9px;color:var(--mute);letter-spacing:.05em;margin-top:6px;font-style:italic}
#vad-toggle{padding:0 10px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#vad-toggle.on{color:var(--cyan);border-color:var(--cyan);background:rgba(0,229,255,.08)}
#convo-banner{position:fixed;left:50%;top:64px;transform:translateX(-50%);z-index:7;padding:6px 18px;border:1px solid var(--cyan);background:rgba(0,229,255,.08);border-radius:99px;color:var(--cyan);font-size:10px;letter-spacing:.3em;text-transform:uppercase;text-shadow:0 0 6px var(--cyan);box-shadow:0 0 14px rgba(0,229,255,.25);display:none;align-items:center;gap:8px}
#convo-banner.show{display:flex}
#convo-banner .level{height:14px;width:60px;background:rgba(0,229,255,.1);border-radius:2px;overflow:hidden;border:1px solid rgba(0,229,255,.2)}
#convo-banner .level .bar{height:100%;background:linear-gradient(90deg,var(--cyan),var(--magenta));transition:width .08s}
#gesture-toggle{padding:0 12px;height:46px;border:1px solid rgba(0,229,255,.3);background:rgba(0,229,255,.03);color:var(--mute);font-family:inherit;font-size:10px;letter-spacing:.2em;cursor:pointer;border-radius:4px}
#gesture-toggle.on{color:var(--magenta);border-color:var(--magenta);background:rgba(255,43,214,.08);box-shadow:0 0 12px rgba(255,43,214,.3)}
.msg .img-attach{max-width:280px;max-height:200px;border:1px solid rgba(0,229,255,.4);border-radius:3px;margin-top:6px;box-shadow:0 0 10px rgba(0,229,255,.18)}
.drop-overlay{position:fixed;inset:0;background:rgba(0,229,255,.08);border:2px dashed var(--cyan);box-shadow:inset 0 0 60px rgba(0,229,255,.2);z-index:50;display:none;align-items:center;justify-content:center;pointer-events:none}
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
    <div class="title">A D A M ▸ JARVIS</div>
    <div class="status">
      <span class="pill"><span class="dot"></span>GF(17) online</span>
      <span class="pill" id="lesson-pill">lessons —</span>
      <span class="pill clickable" id="persona-pill" onclick="togglePersonaPanel()" title="Click to change persona + voice">persona —</span>
      <span class="pill clickable" id="learn-pill" onclick="toggleLearnPanel()" title="Click to inspect 24/7 learning daemon"><span class="ld-led" id="ld-led"></span><span id="ld-text">learning —</span></span>
      <span class="pill clickable" id="tests-pill" onclick="toggleTestsPanel()" title="Pending verification items Adam couldn't auto-check"><span class="tp-led" id="tp-led"></span><span id="tp-text">tests —</span></span>
      <span class="pill clickable" id="shell-pill" onclick="toggleShellPanel()" title="Audit log of every shell command Adam has run"><span class="sh-led" id="sh-led"></span><span id="sh-text">shell —</span></span>
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
    <button id="coach-toggle" type="button" onclick="toggleCoachPanel()" title="Ask-answer-ask coaching session — Adam tutors you on any topic">COACH</button>
    <button id="convo-toggle" type="button" onclick="toggleConvo()" title="Continuous hands-free conversation (VAD)"><span class="convo-dot"></span>CONVO</button>
    <button id="wake-toggle" type="button" onclick="toggleWake()" title='When on, only respond in convo mode if you say "Adam, ..." or "Hey Adam, ..." (Jarvis-style wake word)'>WAKE</button>
    <button id="vad-toggle" type="button" onclick="toggleVadPanel()" title="Tune VAD thresholds for your microphone">VAD</button>
    <button id="send" onclick="send()">TRANSMIT</button>
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
<div id="persona-panel">
  <div class="pp-head"><span>◆ PERSONA + VOICE</span><span class="close" onclick="togglePersonaPanel()">CLOSE</span></div>
  <div class="pp-section">
    <h3>PERSONA</h3>
    <div id="pp-list"><div class="pp-empty">loading…</div></div>
    <div class="pp-input-row" style="margin-top:8px"><input type="text" id="pp-new" placeholder="learn new (e.g. Sherlock, Tony Stark)"><button class="act" onclick="_personaLearnNew()">LEARN</button></div>
    <div style="font-size:9px;color:var(--mute);margin-top:4px;letter-spacing:.05em">Unknown personas trigger a web-learn flow.</div>
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
<div id="coach-panel">
  <div class="cp-head"><span>◆ COACH · ASK-ANSWER-ASK</span><span class="close" onclick="toggleCoachPanel()">CLOSE</span></div>
  <div class="cp-section" id="cp-start-section">
    <h3>NEW SESSION</h3>
    <div class="cp-topic-row"><input type="text" id="cp-topic" placeholder="topic (e.g. python decorators, krebs cycle)"><select id="cp-diff"><option value="1">1 — intro</option><option value="2" selected>2 — basic</option><option value="3">3 — intermediate</option><option value="4">4 — advanced</option><option value="5">5 — expert</option></select><button class="cp-act" onclick="_coachStart()">START</button></div>
    <div style="font-size:9px;color:var(--mute);margin-top:6px;letter-spacing:.05em">Adam will ask a question, grade your answer, then escalate or back off based on your streak.</div>
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
<div id="cam-panel">
  <div class="cam-head"><span><span class="dot"></span>HAND TRACK</span><span id="cam-fps">— fps</span></div>
  <div id="cam-stage">
    <video id="cam-video" autoplay playsinline muted></video>
    <canvas id="cam-landmarks"></canvas>
  </div>
  <div class="gesture-readout" id="gesture-readout">—</div>
</div>
<div id="chat-search"><div class="cs-row"><input type="text" id="cs-input" placeholder="search chat… (case-insensitive substring)" autocomplete="off"><span class="cs-count" id="cs-count">0/0</span><button class="cs-btn" onclick="_csPrev()" title="Previous match (Shift+Enter)">↑</button><button class="cs-btn" onclick="_csNext()" title="Next match (Enter)">↓</button><button class="cs-btn" onclick="closeChatSearch()" title="Close (Esc)">✕</button></div><div class="cs-help">Ctrl+K to open · Enter / ↑↓ to navigate · Esc to close · empty query restores all bubbles</div></div>
<div id="toast-stack"></div>
<div id="drop-overlay" class="drop-overlay"><div class="label">◆ DROP IMAGE FOR ADAM</div></div>
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
    body=`<div class="fc-head"><span class="fc-op ${opCls}">${op.toUpperCase()}</span><span class="fc-bn">${bn}</span>${ext?`<span class="fc-ext">.${ext}</span>`:''}${vBadge}</div>${folder?`<div class="fc-folder">${folder}</div>`:''}<div class="fc-stats"><span class="fc-add">+${la}</span><span class="fc-rem">-${lr}</span>${repl!=null?`<span class="fc-repl">${repl} replacement${repl===1?'':'s'}</span>`:''}<span class="fc-size">${d.lines_after||0} lines · ${d.bytes_after!=null?(d.bytes_after<1024?d.bytes_after+'b':Math.round(d.bytes_after/1024)+'kb'):'?'}</span></div>${issueList}${testRunBlock}${suggList}${d.preview?`<pre class="fc-preview">${esc(d.preview)}</pre>`:''}<div class="fc-actions"><button class="fc-btn" onclick="_fcOpen('${esc(d.path||'').replace(/'/g,"\\\\'")}')">OPEN</button><button class="fc-btn" onclick="_fcCopyPath('${esc(d.path||'').replace(/'/g,"\\\\'")}')">COPY PATH</button>${vstat==='manual'?`<button class="fc-btn" onclick="_fcMarkTested('${esc(d.path||'').replace(/'/g,"\\\\'")}')">MARK TESTED</button>`:''}</div>`;
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
async function _fcOpen(path){
  if(!path)return;
  try{const r=await fetch('/skills/file_read',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{path}})});const j=await r.json();const c=(j.output&&j.output.content)||j.content||(typeof j==='string'?j:'(no content)');bubble('bot','```\n'+c.slice(0,4000)+(c.length>4000?'\n... ('+(c.length-4000)+' more chars)':'')+'\n```','<span class="badge">file</span>')}
  catch(e){bubble('bot','Could not open file: '+esc(e.message),'<span class="badge err">err</span>')}
}
function _fcCopyPath(path){if(!path)return;try{navigator.clipboard.writeText(path);bubble('bot','Copied path to clipboard: `'+esc(path)+'`','<span class="badge">copy</span>')}catch{bubble('bot','Clipboard unavailable. Path: `'+esc(path)+'`','<span class="badge err">err</span>')}}
async function _fcMarkTested(path){
  if(!path)return;
  try{const r=await fetch('/memory/needs-testing/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path_substring:path})});const j=await r.json();bubble('bot','Marked '+j.marked_done+' testing item(s) as done for `'+esc(path)+'`','<span class="badge">tested</span>')}
  catch(e){bubble('bot','Could not mark tested: '+esc(e.message),'<span class="badge err">err</span>')}
}
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
let _selectedVoice=localStorage.getItem(VOICE_KEY)||'';
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
  const voices=document.getElementById('pp-voices');
  if(_availableVoices.length===0){voices.innerHTML='<div class="pp-empty">no piper voices · install via <code style="color:var(--cyan)">pip install piper-tts</code> + download a voice</div>'}
  else{
    voices.innerHTML='<div class="pp-row'+(!_selectedVoice?' active':'')+'" onclick="_pickVoice(\'\')"><span class="nm">auto (persona default)</span></div>'+_availableVoices.map(v=>{
      const active=(v===_selectedVoice);
      return `<div class="pp-row${active?' active':''}" onclick="_pickVoice('${esc(v).replace(/'/g,"\\\\'")}')"><span class="nm" style="font-family:Consolas,monospace;font-size:10px">${esc(v)}</span></div>`;
    }).join('');
  }
}
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
    if(r.ok&&j.persona){_selectedPersona=j.persona.name;localStorage.setItem(PERSONA_KEY,_selectedPersona);personaName=_selectedPersona;personaPill.textContent='persona '+_selectedPersona;await _loadPersonas();_renderPersonaPanel();bubble('bot','Learned **'+esc(_selectedPersona)+'** — '+esc(j.persona.description||''),'<span class="badge persona">'+esc(_selectedPersona)+'</span>')}
    else bubble('bot','Learn failed: '+(esc(j.error||JSON.stringify(j))),'<span class="badge err">err</span>')
  }catch(e){bubble('bot','Learn error: '+esc(e.message),'<span class="badge err">err</span>')}
}
function _pickVoice(v){_selectedVoice=v;localStorage.setItem(VOICE_KEY,v);_renderPersonaPanel();if(v)bubble('bot','TTS voice set to **'+esc(v)+'**','<span class="badge">voice</span>')}
const _origProbeVoiceBackends=probeVoiceBackends;
async function _initPersonaPill(){await _origProbeVoiceBackends();await _loadPersonas();if(_selectedPersona){personaName=_selectedPersona;personaPill.textContent='persona '+_selectedPersona}}
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
function _showToast(n){
  const stack=document.getElementById('toast-stack');if(!stack)return;
  const el=document.createElement('div');el.className='toast '+(n.level||'info');el.dataset.id=n.id;
  el.innerHTML=`<div class="t-head"><span class="t-src">${esc(n.source||'')}</span><span class="t-age">${_notifHumanAge(n.age_s||0)} ago</span><span class="t-close" onclick="event.stopPropagation();_dismissToast('${n.id}')">✕</span></div><div class="t-title">${esc(n.title||'')}</div>${n.body?`<div class="t-body">${esc(n.body)}</div>`:''}`;
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
const SESSION_RESTORE_LIMIT=24;
async function _restoreSession(){
  if(!sid)return;
  try{
    const r=await fetch('/sessions/'+encodeURIComponent(sid)+'?limit='+SESSION_RESTORE_LIMIT);
    if(!r.ok)return;
    const j=await r.json();const turns=j.turns||[];
    if(turns.length===0)return;
    const w=document.querySelector('.welcome');if(w)w.remove();
    const banner=document.createElement('div');banner.className='restore-banner';banner.innerHTML=`<span>↻ RESTORED ${turns.length} TURN${turns.length===1?'':'S'} FROM SESSION ${esc(sid.slice(-8))}</span><span class="rb-close" onclick="this.parentElement.remove()">CLEAR</span>`;log.appendChild(banner);
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
});
document.addEventListener('input',e=>{if(e.target&&e.target.id==='cs-input')_csRunSearch()});
const COACH_SID_KEY='amni_jarvis_coach_sid',COACH_VOICE_KEY='amni_jarvis_coach_voice';
let _coachSid=localStorage.getItem(COACH_SID_KEY)||'',_coachPanelOpen=false,_coachTopic='',_coachBusy=false,_coachVoiceOn=localStorage.getItem(COACH_VOICE_KEY)==='1',_coachLastQuestion='';
function _coachToggleVoice(){_coachVoiceOn=!_coachVoiceOn;localStorage.setItem(COACH_VOICE_KEY,_coachVoiceOn?'1':'0');_coachUpdateVoiceBtn();if(_coachVoiceOn){voiceOut=true;localStorage.setItem(VKEY,'1');const vb=document.getElementById('voiceout-toggle');if(vb)vb.classList.add('on');if(_coachLastQuestion)speak(_coachLastQuestion)}}
function _coachUpdateVoiceBtn(){const b=document.getElementById('cp-voice-toggle');if(!b)return;b.textContent=_coachVoiceOn?'VOICE ON':'VOICE OFF';b.classList.toggle('on',_coachVoiceOn)}
function _coachReplayQuestion(){if(_coachLastQuestion)speak(_coachLastQuestion);else bubble('bot','No active question to replay.','<span class="badge err">coach</span>')}
function _coachSpeakIfOn(text){if(_coachVoiceOn&&text&&text.trim()&&_voiceBackends.tts)speak(text)}
function toggleCoachPanel(){_coachPanelOpen=!_coachPanelOpen;const p=document.getElementById('coach-panel');p.classList.toggle('show',_coachPanelOpen);document.getElementById('coach-toggle').classList.toggle('on',_coachPanelOpen);['persona-panel','learn-panel','tests-panel'].forEach(id=>{const el=document.getElementById(id);if(_coachPanelOpen&&el&&el.classList.contains('show'))el.classList.remove('show')});if(_coachPanelOpen){_personaPanelOpen=false;_ldPanelOpen=false;_tpPanelOpen=false;_coachUpdateVoiceBtn();_coachLoadTopics();if(_coachSid)_coachSyncStatus()}}
async function _coachLoadTopics(){
  const list=document.getElementById('cp-topics-list');if(!list)return;
  try{
    const r=await fetch('/memory/coach');if(!r.ok){list.innerHTML='<div style="font-size:10px;color:var(--mute);font-style:italic;text-align:center;padding:6px">coach memory unavailable</div>';return}
    const j=await r.json();const topics=j.topics||[];
    if(!topics.length){list.innerHTML='<div style="font-size:10px;color:var(--mute);font-style:italic;text-align:center;padding:6px">no topics practiced yet · start a session above</div>';return}
    list.innerHTML=topics.slice(0,20).map(t=>{
      const pct=Math.round(t.mastery_pct||0);
      const lvl=pct>=85?'master':(pct>=65?'good':(pct>=40?'fair':'novice'));
      const name=esc(t.topic||'?');const safe=name.replace(/'/g,"\\\\'");
      return `<div class="cp-topic-card lvl-${lvl}" onclick="_coachResumeTopic('${safe}')" title="Click to start a new session on this topic"><span class="name">${name}</span><span class="mini-bar"><span class="mini-bar-fill" style="width:${pct}%"></span></span><span class="pct">${pct}%</span><span class="n">${t.n_questions||0}q</span></div>`
    }).join('');
  }catch(e){list.innerHTML='<div style="font-size:10px;color:var(--err);font-style:italic;text-align:center;padding:6px">load error</div>'}
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
let _audioEl=null;
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
function _speakBrowser(clean){if(!('speechSynthesis' in window))return;try{const u=new SpeechSynthesisUtterance(clean);u.rate=1;u.pitch=1;speechSynthesis.cancel();speechSynthesis.speak(u)}catch{}}
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
let wakeOn=localStorage.getItem(WAKE_KEY)==='1';
const _WAKE_PREFIX='(?:hey[,\\s]+|ok(?:ay)?[,\\s]+|yo[,\\s]+|hi[,\\s]+|hello[,\\s]+)?(?:adam|atom|adams|adan)';
const _WAKE_PATTERN_PUNCT=new RegExp('^\\s*'+_WAKE_PREFIX+'[,:;!?]+\\s*(.+)','i');
const _WAKE_PATTERN_SPACE=new RegExp('^\\s*'+_WAKE_PREFIX+'\\s+(.+)','i');
const _WAKE_PATTERN=_WAKE_PATTERN_SPACE;
const _WAKE_DECLARATIVE=/^(?:is|isn't|was|wasn't|were|weren't|has|hasn't|have|haven't|had|hadn't|seems|seemed|looks|looked|appears|appeared|sounds|sounded|seems\s+to|got|gets)\b/i;
function _wakeWordGate(text){
  if(!wakeOn)return text;
  const t=text||'';
  const mp=t.match(_WAKE_PATTERN_PUNCT);
  if(mp){const s=(mp[1]||'').trim();return s.length>=2?s:null}
  const ms=t.match(_WAKE_PATTERN_SPACE);
  if(!ms)return null;
  const s=(ms[1]||'').trim();
  if(_WAKE_DECLARATIVE.test(s))return null;
  return s.length>=2?s:null;
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
}
function _vadLoop(){
  if(!convoOn||!convoAnalyser){convoRAF=null;return}
  const buf=new Uint8Array(convoAnalyser.frequencyBinCount);
  convoAnalyser.getByteFrequencyData(buf);
  let sum=0;for(let i=0;i<buf.length;i++)sum+=buf[i];
  const avg=sum/buf.length;
  _convoLevelBar.style.width=Math.min(100,(avg/40)*100)+'%';
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
  let acc='',spoken='',ttsQueue=[],ttsPlaying=false;
  const _SENT_RE=/([.!?…][\s"')\]\}]*)/;
  async function _flushTTS(chunk){
    if(!chunk||!chunk.trim())return;
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
          if(etype==='token'){const chunk=JSON.parse(edata);if(bot.bubble.classList.contains('thinking')){bot.bubble.classList.remove('thinking');bot.bubble.textContent=''}acc+=chunk;bot.bubble.innerHTML=md(acc);_consumeSentence();log.scrollTop=log.scrollHeight}
          else if(etype==='meta'){const m=JSON.parse(edata);if(m.session_id){sid=m.session_id;localStorage.setItem(SKEY,sid)}}
          else if(etype==='done'){const tail=acc.slice(spoken.length);if(tail.trim()){spoken=acc;_flushTTS(tail)}}
        }catch(p){}
      }
    }
  }catch(e){bot.bubble.classList.remove('thinking');bot.bubble.textContent='stream error: '+e.message;if(convoOn)_setConvoState('listening')}
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
document.addEventListener('drop',async e=>{
  _dragDepth=0;_dropOverlay.classList.remove('show');
  if(!e.dataTransfer)return;
  for(const f of e.dataTransfer.files||[]){
    if(f.type&&f.type.startsWith('image/')){e.preventDefault();await handleImageBlob(f);return}
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
</script></body></html>"""
def mount(app):
    from fastapi.responses import HTMLResponse
    @app.get('/jarvis',response_class=HTMLResponse)
    def jarvis():return HTMLResponse(content=_HTML)
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
