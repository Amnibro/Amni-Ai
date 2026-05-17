"""Embedded chat UI v6.5 — first-run wizard, sidebar with quick actions, file-tree (code mode), markdown render, voice in/out, persona selector."""
_HTML=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adam — Amni-Ai</title>
<style>
:root{--bg:#0b0d10;--bg2:#13171c;--fg:#e6e8eb;--mute:#7a8088;--user:#1c5fd1;--bot:#1a2027;--accent:#7fd6c5;--accent2:#ffd770;--err:#d65a5a;--ok:#7fd6c5;--badge:#2a3038;--border:#1f242a;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--fg);font-size:14px;overflow:hidden}
#app{display:grid;grid-template-columns:240px 1fr;height:100vh}
#sidebar{background:#0e1115;border-right:1px solid var(--border);overflow-y:auto;display:flex;flex-direction:column}
#sidebar .brand{padding:14px 16px;border-bottom:1px solid var(--border)}
#sidebar .brand h1{margin:0;font-size:16px;font-weight:600}
#sidebar .brand .sub{font-size:11px;color:var(--mute);margin-top:2px}
.section{padding:10px 12px;border-bottom:1px solid var(--border)}
.section h3{margin:0 0 8px 0;font-size:10px;color:var(--mute);text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.action{display:flex;align-items:center;gap:8px;width:100%;background:transparent;border:0;color:var(--fg);padding:7px 8px;border-radius:6px;text-align:left;cursor:pointer;font-size:13px;font-family:inherit}
.action:hover{background:#1a1f25;color:var(--accent)}
.action .icon{font-size:14px;width:18px;display:inline-block;text-align:center}
#main{display:flex;flex-direction:column;min-width:0;min-height:0;height:100vh;overflow:hidden}
header{padding:11px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-shrink:0;background:var(--bg)}
header .title{font-size:13px;font-weight:600}
header .chip{background:var(--bg2);border:1px solid var(--border);padding:2px 8px;border-radius:10px;font-size:11px;color:var(--accent)}
header .stats{margin-left:auto;font-size:11px;color:var(--mute)}
#log{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;padding:18px 24px;display:flex;flex-direction:column;gap:14px}
#empty{text-align:center;padding:40px 20px;color:var(--mute)}
#empty h2{color:var(--fg);margin:0 0 6px 0;font-size:18px}
#empty .examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin-top:20px;max-width:680px;margin-left:auto;margin-right:auto}
#empty .ex{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;cursor:pointer;text-align:left;font-size:12px;color:var(--fg);font-family:inherit}
#empty .ex:hover{border-color:var(--accent);color:var(--accent)}
#empty .ex .lbl{color:var(--accent);font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;display:block}
.msg{display:flex;flex-direction:column;gap:4px;max-width:88%}
.msg.user{align-self:flex-end}
.msg.bot{align-self:flex-start}
.bubble{padding:10px 14px;border-radius:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word;max-height:70vh;overflow-y:auto;overflow-x:hidden}
.bubble::-webkit-scrollbar{width:8px}
.bubble::-webkit-scrollbar-track{background:transparent}
.bubble::-webkit-scrollbar-thumb{background:#2a3038;border-radius:4px}
.bubble::-webkit-scrollbar-thumb:hover{background:var(--accent)}
#log::-webkit-scrollbar{width:10px}
#log::-webkit-scrollbar-track{background:transparent}
#log::-webkit-scrollbar-thumb{background:#2a3038;border-radius:5px}
#log::-webkit-scrollbar-thumb:hover{background:var(--accent)}
.msg.user .bubble{background:var(--user);color:#fff;border-bottom-right-radius:4px}
.msg.bot .bubble{background:var(--bot);color:var(--fg);border-bottom-left-radius:4px}
.bubble code{background:#000;padding:1px 5px;border-radius:3px;font-size:12px;font-family:Consolas,Menlo,monospace}
.bubble pre{background:#000;padding:10px;border-radius:6px;overflow-x:auto;font-size:12px;margin:6px 0;font-family:Consolas,Menlo,monospace}
.bubble pre code{background:none;padding:0}
.bubble pre{position:relative}
.copy-btn{position:absolute;top:4px;right:4px;background:#2a3038;color:var(--mute);border:1px solid var(--border);border-radius:4px;padding:1px 8px;font-size:10px;cursor:pointer;font-family:inherit;opacity:.55}
.copy-btn:hover{opacity:1;color:var(--accent);border-color:var(--accent)}
.bubble p{margin:0 0 8px 0}
.bubble p:last-child{margin-bottom:0}
.bubble h1,.bubble h2,.bubble h3{margin:8px 0 4px 0;color:var(--accent);font-weight:600}
.bubble h1{font-size:16px}.bubble h2{font-size:14px}.bubble h3{font-size:13px}
.bubble ul,.bubble ol{margin:4px 0 8px 0;padding-left:20px}
.bubble a{color:var(--accent)}
.bubble strong{color:#fff}
.meta{display:flex;gap:6px;font-size:10px;color:var(--mute);padding:0 4px;align-items:center;flex-wrap:wrap}
.badge{background:var(--badge);padding:2px 6px;border-radius:8px;font-size:10px;color:var(--accent)}
.badge.skill{color:var(--accent2)}
.badge.err{color:var(--err)}
.badge.persona{color:#c5a3ff}
form{display:flex;gap:8px;padding:14px 16px;border-top:1px solid var(--border);flex-shrink:0;background:var(--bg)}
#input{flex:1;background:var(--bg2);color:var(--fg);border:1px solid var(--border);border-radius:10px;padding:11px 14px;font-size:14px;resize:none;font-family:inherit;line-height:1.4;min-height:44px;max-height:160px}
#input:focus{outline:none;border-color:var(--accent)}
button.primary{background:var(--accent);color:#0b0d10;border:0;border-radius:10px;padding:0 18px;font-weight:600;cursor:pointer;font-size:13px;font-family:inherit;height:44px}
button.icon-btn{background:var(--bg2);color:var(--mute);border:1px solid var(--border);border-radius:10px;padding:0 12px;cursor:pointer;font-size:12px;font-family:inherit;height:44px;min-width:44px}
button.icon-btn:hover{border-color:var(--accent);color:var(--accent)}
button.icon-btn.active{color:var(--accent2);border-color:var(--accent2)}
button:disabled{opacity:.4;cursor:wait}
.thinking{font-style:italic;color:var(--mute);font-size:13px}
#filetree{font-size:12px;padding:0 6px}
.tree-item{display:flex;align-items:center;padding:3px 6px;border-radius:4px;cursor:pointer;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:Consolas,Menlo,monospace}
.tree-item:hover{background:#1a1f25;color:var(--accent)}
.tree-item.dir{color:var(--accent)}
.tree-item .ind{color:var(--mute);margin-right:4px}
#wizard{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:1000;padding:24px}
#wizard.show{display:flex}
.wizard-card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;max-width:600px;width:100%;padding:32px;box-shadow:0 20px 60px rgba(0,0,0,.6)}
.wizard-card h2{margin:0 0 8px 0;color:var(--accent);font-size:22px}
.wizard-card p{color:var(--fg);line-height:1.55;margin:8px 0}
.wizard-card .step-dots{display:flex;gap:6px;margin:16px 0}
.wizard-card .dot{width:8px;height:8px;border-radius:50%;background:var(--border)}
.wizard-card .dot.active{background:var(--accent)}
.wizard-card .actions{display:flex;gap:8px;margin-top:20px;justify-content:flex-end}
.wizard-card .feature{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin:8px 0;display:flex;gap:10px;align-items:flex-start}
.wizard-card .feature .ico{font-size:20px}
.wizard-card .feature .txt{flex:1}
.wizard-card .feature .txt strong{color:var(--accent);font-size:13px}
.wizard-card .feature .txt span{display:block;color:var(--mute);font-size:12px;margin-top:2px}
@media(max-width:768px){#app{grid-template-columns:1fr}#sidebar{display:none}}
</style></head>
<body>
<div id="app">
  <aside id="sidebar">
    <div class="brand">
      <h1>Adam</h1>
      <div class="sub" id="brandSub">GF(17) texture-native</div>
    </div>
    <div class="section">
      <h3>Quick actions</h3>
      <button class="action" onclick="qaScanFolder()"><span class="icon">[+]</span>Scan folder into memory</button>
      <button class="action" onclick="qaLearnPersona()"><span class="icon">[P]</span>Learn a new persona</button>
      <button class="action" onclick="qaBrowseLessons()"><span class="icon">[B]</span>Browse what I know</button>
      <button class="action" onclick="qaReflect()"><span class="icon">[R]</span>Self-reflect once</button>
      <button class="action" onclick="qaSearchMem()"><span class="icon">[S]</span>Search my memory</button>
    </div>
    <div class="section">
      <h3>Persona</h3>
      <button class="action" onclick="setPersonaUI()"><span class="icon">[#]</span><span id="personaLabel">switch persona...</span></button>
    </div>
    <div class="section">
      <h3>Voice</h3>
      <button class="action" onclick="toggleVoiceOut()"><span class="icon">[V]</span><span id="vOutLbl">Speak responses: off</span></button>
    </div>
    <div class="section" id="codeSection" style="display:none">
      <h3>Project</h3>
      <div id="projectMeta" style="font-size:11px;color:var(--mute);padding:4px 6px"></div>
      <button class="action" onclick="loadFileTree()"><span class="icon">[T]</span>Refresh file tree</button>
      <div id="filetree"></div>
    </div>
    <div class="section">
      <h3>Session</h3>
      <button class="action" onclick="newSession()"><span class="icon">[N]</span>New session</button>
      <button class="action" onclick="clearChat()"><span class="icon">[X]</span>Clear screen</button>
      <button class="action" onclick="showWizard()"><span class="icon">[?]</span>Tutorial</button>
    </div>
  </aside>
  <main id="main">
    <header>
      <span class="title" id="headerTitle">Chat</span>
      <span class="chip" id="personaChip" style="display:none"></span>
      <span class="chip" id="modeChip" style="display:none"></span>
      <div class="stats" id="stats"></div>
    </header>
    <div id="log">
      <div id="empty">
        <h2>Hey! What can I help with?</h2>
        <p style="color:var(--mute);font-size:13px">Try one of these or ask anything:</p>
        <div class="examples">
          <button class="ex" onclick="quick('What is 17 * 23?')"><span class="lbl">math</span>What is 17 * 23?</button>
          <button class="ex" onclick="quick('Write a Python function to reverse a list')"><span class="lbl">code</span>Write a Python function to reverse a list</button>
          <button class="ex" onclick="quick('Tell me a haiku about AI')"><span class="lbl">creative</span>Tell me a haiku about AI</button>
          <button class="ex" onclick="quick('What can you do?')"><span class="lbl">intro</span>What can you do?</button>
          <button class="ex" onclick="quick('What time is it?')"><span class="lbl">tool</span>What time is it?</button>
          <button class="ex" onclick="qaScanFolder()"><span class="lbl">action</span>Teach me from a folder</button>
        </div>
      </div>
    </div>
    <form id="form" onsubmit="return send(event)">
      <textarea id="input" placeholder="Ask Adam... (Shift+Enter for newline, mic for voice)" autofocus></textarea>
      <button class="icon-btn" id="micBtn" type="button" onclick="toggleVoiceIn()" title="voice input">mic</button>
      <button class="primary" id="btn" type="submit">Send</button>
    </form>
  </main>
</div>
<div id="wizard"><div class="wizard-card">
  <h2 id="wzTitle">Welcome to Adam</h2>
  <div class="step-dots" id="wzDots"></div>
  <div id="wzBody"></div>
  <div class="actions">
    <button class="icon-btn" onclick="closeWizard()">Skip</button>
    <button class="icon-btn" onclick="wzPrev()" id="wzPrev" style="display:none">Back</button>
    <button class="primary" onclick="wzNext()" id="wzNext">Next</button>
  </div>
</div></div>
<script>
const KEY='amni_session',PKEY='amni_persona',VKEY='amni_voice_out',WKEY='amni_wizard_done';
let sid=localStorage.getItem(KEY);
let personaName=localStorage.getItem(PKEY)||'';
let voiceOut=localStorage.getItem(VKEY)==='1';
let recog=null,recoOn=false;
let projectInfo=null,wzStep=0;
const log=document.getElementById('log'),input=document.getElementById('input'),btn=document.getElementById('btn'),stats=document.getElementById('stats'),emptyState=document.getElementById('empty');
function escape(s){return (s||'').replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function hookCopyButtons(container){
  container.querySelectorAll('pre').forEach(pre=>{
    if(pre.querySelector('.copy-btn'))return;
    const btn=document.createElement('button');btn.className='copy-btn';btn.textContent='copy';
    btn.onclick=()=>{const code=pre.querySelector('code')||pre;navigator.clipboard.writeText(code.textContent).then(()=>{btn.textContent='copied!';setTimeout(()=>btn.textContent='copy',1200)})};
    pre.style.position='relative';pre.appendChild(btn);
  });
}
function md(src){
  src=escape(src);
  src=src.replace(/```([\w-]*)\n([\s\S]*?)```/g,(_,l,c)=>`<pre><code>${c.replace(/\n$/,'')}</code></pre>`);
  src=src.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  src=src.replace(/^###\s+(.+)$/gm,'<h3>$1</h3>').replace(/^##\s+(.+)$/gm,'<h2>$1</h2>').replace(/^#\s+(.+)$/gm,'<h1>$1</h1>');
  src=src.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>').replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g,'<em>$1</em>');
  src=src.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  src=src.replace(/^(?:[-*]\s+.+\n?)+/gm,m=>'<ul>'+m.trim().split(/\n/).map(li=>`<li>${li.replace(/^[-*]\s+/,'')}</li>`).join('')+'</ul>');
  src=src.replace(/^(?:\d+\.\s+.+\n?)+/gm,m=>'<ol>'+m.trim().split(/\n/).map(li=>`<li>${li.replace(/^\d+\.\s+/,'')}</li>`).join('')+'</ol>');
  src=src.replace(/\n\n+/g,'</p><p>');
  return '<p>'+src+'</p>';
}
function bubble(role,text,meta){
  if(emptyState){emptyState.remove()}
  const m=document.createElement('div');m.className='msg '+role;
  const b=document.createElement('div');b.className='bubble';
  if(role==='bot')b.innerHTML=md(text);else b.textContent=text;
  m.appendChild(b);
  if(meta){const mt=document.createElement('div');mt.className='meta';mt.innerHTML=meta;m.appendChild(mt)}
  log.appendChild(m);log.scrollTop=log.scrollHeight;return m;
}
function quick(t){input.value=t;document.getElementById('form').requestSubmit()}
async function send(e){
  e.preventDefault();
  const text=input.value.trim();if(!text)return false;
  input.value='';input.style.height='auto';btn.disabled=true;
  bubble('user',text);
  if(emptyState){emptyState.remove()}
  const botMsg=document.createElement('div');botMsg.className='msg bot';
  const botB=document.createElement('div');botB.className='bubble thinking';botB.textContent='...';botMsg.appendChild(botB);
  const botMeta=document.createElement('div');botMeta.className='meta';botMsg.appendChild(botMeta);
  log.appendChild(botMsg);log.scrollTop=log.scrollHeight;
  let acc='',gotFirst=false,meta={};
  try{
    const body=sid?{message:text,session_id:sid}:{message:text};
    const resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
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
          if(etype==='token'){const chunk=JSON.parse(edata);if(!gotFirst){botB.classList.remove('thinking');gotFirst=true;botB.textContent=''}acc+=chunk;botB.innerHTML=md(acc);log.scrollTop=log.scrollHeight}
          else if(etype==='meta'){const m=JSON.parse(edata);Object.assign(meta,m);if(m.session_id){sid=m.session_id;localStorage.setItem(KEY,sid)}}
          else if(etype==='done'){const m=JSON.parse(edata);Object.assign(meta,m);const tier=meta.tier||'?';const persona=meta.persona||'';const cat=meta.category||'';const tierBadge=`<span class="badge">${escape(tier)}</span>`;const personaBadge=persona?`<span class="badge persona">${escape(persona)}</span>`:'';const catBadge=cat?`<span class="badge">${escape(cat)}</span>`:'';const wallBadge=meta.wall_s?`<span>${meta.wall_s}s</span>`:'';botMeta.innerHTML=`${tierBadge}${personaBadge}${catBadge}${wallBadge}`;if(acc)speak(acc);refreshStats()}
          else if(etype==='validate'){const d=JSON.parse(edata);botMeta.innerHTML+=`<span class="badge err">${d.syntax_errors} syntax err</span>`}
          else if(etype==='exec'){const d=JSON.parse(edata);if(d.error){botMeta.innerHTML+=`<span class="badge err">exec: ${escape(d.error.slice(0,40))}</span>`}else{const ok=d.returncode===0;botMeta.innerHTML+=`<span class="badge ${ok?"":"err"}">sandbox exit ${d.returncode}${d.timed_out?" (timeout)":""}</span>`;if(d.stdout||d.stderr){const execMd=`\n\n**[Sandbox execution — exit ${d.returncode}${d.timed_out?"  (timed out)":""}]**\n${d.stdout?'```\n'+d.stdout+'\n```\n':''}${d.stderr?'_stderr:_\n```\n'+d.stderr+'\n```':''}`;acc+=execMd;botB.innerHTML=md(acc);hookCopyButtons(botB);log.scrollTop=log.scrollHeight}}}
          else if(etype==='test_run'){const d=JSON.parse(edata);const div=d.diversity!==undefined?` div=${d.diversity}`:'';botMeta.innerHTML+=`<span class="badge ${d.passed?'':'err'}">tests ${d.passed?'✓':'✗'} ${d.asserts_n||0}${div}</span>`}
          else if(etype==='promoted'){const d=JSON.parse(edata);if(d.error){botMeta.innerHTML+=`<span class="badge err">promote: ${escape(d.error.slice(0,40))}</span>`}else{botMeta.innerHTML+=`<span class="badge">learned ✓ #${d.lessons_n}</span>`}}
          else if(etype==='perturb'){const d=JSON.parse(edata);if(d.final){const ok=d.success;botMeta.innerHTML+=`<span class="badge ${ok?'':'err'}">perturb ${ok?'✓ '+escape(d.magnitude||''):'✗ x'+(d.steps||0)}</span>`}else{const mag=d.magnitude||'?';const rcOk=d.rc===0&&!d.stderr;botMeta.innerHTML+=`<span class="badge ${rcOk?'':'err'}">${escape(mag)} ${d.status||('rc='+d.rc)}</span>`}}
          else if(etype==='error'){botB.textContent='Error: '+edata;botB.classList.remove('thinking')}
        }catch(parseErr){console.warn('SSE parse',parseErr,edata)}
      }
    }
    hookCopyButtons(botB);
  }catch(err){botB.textContent='Error: '+err.message;botB.classList.remove('thinking');botMeta.innerHTML='<span class="badge err">error</span>'}
  btn.disabled=false;input.focus();return false;
}
function clearChat(){log.innerHTML=''}
function newSession(){sid=null;localStorage.removeItem(KEY);clearChat();location.reload()}
async function refreshStats(){
  try{
    const r=await fetch('/stats');const j=await r.json();
    stats.textContent=`lessons=${j.lessons_n||0} · skills=${(j.skills||[]).length}`;
    document.getElementById('brandSub').textContent=personaName?`persona: ${personaName}`:'GF(17) texture-native';
    document.getElementById('personaLabel').textContent=personaName||'switch persona...';
    if(personaName){const c=document.getElementById('personaChip');c.textContent=personaName;c.style.display='inline-block'}
  }catch{}
}
async function loadProjectInfo(){
  try{
    const r=await fetch('/project');projectInfo=await r.json();
    if(projectInfo.code_mode){
      document.getElementById('codeSection').style.display='block';
      document.getElementById('projectMeta').innerHTML=`<div>${escape(projectInfo.root)}</div><div style="color:var(--accent)">${escape(projectInfo.type)}</div>`;
      document.getElementById('headerTitle').textContent='Code Mode';
      const c=document.getElementById('modeChip');c.textContent='code · '+projectInfo.type;c.style.display='inline-block';
      loadFileTree();
    }
  }catch{}
}
async function loadFileTree(){
  try{
    const r=await fetch('/project/tree?depth=2&limit=100');const j=await r.json();
    const ft=document.getElementById('filetree');ft.innerHTML='';
    for(const it of j.items){
      const el=document.createElement('div');
      el.className='tree-item'+(it.is_dir?' dir':'');
      el.innerHTML=`<span class="ind">${'  '.repeat(it.depth)}</span>${escape(it.is_dir?'+ ':'  ')}${escape(it.name)}`;
      el.onclick=()=>{if(it.is_dir){quick('scan the directory '+projectInfo.root+'/'+it.path)}else{quick('read '+projectInfo.root+'/'+it.path)}};
      ft.appendChild(el);
    }
  }catch{}
}
async function setPersonaUI(){
  const known=await fetch('/personas').then(r=>r.json()).catch(()=>({known:[]}));
  const names=known.known.map(p=>p.name).join(', ');
  const choice=prompt(`Pick a persona (or type a NEW name and Adam will web-learn it).\nKnown: ${names}\n\nCurrent: ${personaName||'(default)'}`,personaName||'');
  if(choice===null)return;
  if(!choice.trim()){personaName='';localStorage.removeItem(PKEY);refreshStats();return}
  bubble('bot',`(switching persona to "${choice}"...)`,'<span class="badge">persona</span>');
  try{
    const r=await fetch('/persona',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:choice,session_id:sid,learn_via_web:true})});
    const j=await r.json();
    if(r.ok&&j.persona){personaName=j.persona.name;localStorage.setItem(PKEY,personaName);bubble('bot',`Persona set to **${j.persona.name}** (source: ${j.persona.source}).\n\n${j.persona.description}`,'<span class="badge">persona</span>');refreshStats()}
    else bubble('bot',`(persona switch failed: ${JSON.stringify(j)})`,'<span class="badge err">err</span>')
  }catch(e){bubble('bot','Error: '+e.message,'<span class="badge err">err</span>')}
}
async function qaScanFolder(){
  const path=prompt('Folder path to scan (Adam will read text files and learn from them):',projectInfo&&projectInfo.root||'');
  if(!path)return;
  quick(`scan the directory ${path}`);
}
async function qaLearnPersona(){await setPersonaUI()}
async function qaSearchMem(){
  const q=prompt('Search Adam memory for:');if(!q)return;
  bubble('user',`search memory for ${q}`);
  try{
    const r=await fetch('/skills/mem',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({args:{query:q,k:5}})});
    const j=await r.json();
    let out='';if(j.output&&j.output.hits){
      out=`Found ${j.output.hits.length} match(es) in ${j.output.lessons_n}-lesson bank:\n\n`;
      for(const h of j.output.hits.slice(0,5)){if(h.error){out+=`(error: ${h.error})\n`;continue}out+=`**${h.score?h.score.toFixed(2):'?'}** ${h.q?h.q.slice(0,100):''}\n  -> ${h.a?h.a.slice(0,200):''}\n\n`}
    }
    bubble('bot',out||'(no matches)','<span class="badge">mem</span>');
  }catch(e){bubble('bot','Error: '+e.message,'<span class="badge err">err</span>')}
}
async function qaBrowseLessons(){
  const q=prompt('Filter lessons (leave blank to browse all, ESC to cancel):','');
  if(q===null)return;
  try{
    const qs=q?`?q=${encodeURIComponent(q)}&limit=15`:'?limit=15';
    const r=await fetch('/lessons'+qs);const j=await r.json();
    let out=`**Lesson bank** — showing ${j.items.length} of ${j.total}${q?` matching ${JSON.stringify(q)}`:''} (total in bank: ${j.lessons_n})\n\n`;
    for(const it of j.items){out+=`**[#${it.idx}]** ${it.q.slice(0,100)}\n  -> ${it.a.slice(0,140)}\n\n`}
    if(j.total>j.items.length)out+=`_...${j.total - j.items.length} more match — refine filter to narrow._`;
    bubble('bot',out,'<span class="badge">lessons</span>');
  }catch(e){bubble('bot','Error: '+e.message,'<span class="badge err">err</span>')}
}
async function qaReflect(){
  bubble('bot','Reflecting (this can take a while — Adam will pick uncertain lessons and re-research them)...','<span class="badge">reflect</span>');
  try{
    const r=await fetch('/reflect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({max_n:3,min_age_sec:0})});
    const j=await r.json();
    bubble('bot',`Reflection cycle: ${j.targets} targets, ${j.updated} updated, ${j.kept} kept, ${j.failed} failed in ${j.cycle_wall_s}s.`,'<span class="badge">reflect</span>');
  }catch(e){bubble('bot','Error: '+e.message,'<span class="badge err">err</span>')}
}
function toggleVoiceOut(){voiceOut=!voiceOut;localStorage.setItem(VKEY,voiceOut?'1':'0');document.getElementById('vOutLbl').textContent='Speak responses: '+(voiceOut?'on':'off')}
function speak(text){if(!voiceOut||!('speechSynthesis' in window))return;try{const u=new SpeechSynthesisUtterance(text.replace(/```[\s\S]*?```/g,'(code block)').replace(/[*_`#]/g,'').slice(0,800));speechSynthesis.cancel();speechSynthesis.speak(u)}catch{}}
function toggleVoiceIn(){
  if(!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)){alert('Voice input not supported in this browser. Use Chrome or Edge.');return}
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(recoOn&&recog){recog.stop();return}
  recog=new SR();recog.lang='en-US';recog.interimResults=false;recog.continuous=false;
  recoOn=true;document.getElementById('micBtn').classList.add('active');
  recog.onresult=e=>{const t=e.results[0][0].transcript;input.value=t;document.getElementById('form').requestSubmit()};
  recog.onerror=e=>{console.error(e);recoOn=false;document.getElementById('micBtn').classList.remove('active')};
  recog.onend=()=>{recoOn=false;document.getElementById('micBtn').classList.remove('active')};
  recog.start();
}
const WIZARD=[
  {title:'Welcome to Adam',body:'<p>I am <strong>Adam</strong> — a self-hosted AI built on novel GF(17) texture-native architecture by Amnibro. Everything runs locally on your machine. Nothing leaves unless you choose to share.</p><p>This quick tour shows what makes me different.</p>'},
  {title:'I learn and remember',body:'<div class="feature"><div class="ico">[B]</div><div class="txt"><strong>Persistent lesson bank</strong><span>Anything you teach me, I remember across sessions. I can ingest entire folders.</span></div></div><div class="feature"><div class="ico">[S]</div><div class="txt"><strong>Self-reflection</strong><span>I review uncertain answers, re-research, and improve myself.</span></div></div><div class="feature"><div class="ico">[+]</div><div class="txt"><strong>Knowledge from any source</strong><span>Click "Scan folder" and point me at notes, docs, or code. I learn the contents.</span></div></div>'},
  {title:'I have many voices',body:'<div class="feature"><div class="ico">[#]</div><div class="txt"><strong>Personas</strong><span>I can be Rikku (cheerful Al Bhed), Yoda (inverted syntax), Scientist (precise), or any character you want — I will web-learn the voice if I do not know them.</span></div></div><div class="feature"><div class="ico">[T]</div><div class="txt"><strong>Tools at hand</strong><span>I can do math, read/write files, run shell commands, search the web, and chain steps to achieve goals.</span></div></div>'},
  {title:'Try me',body:'<p>Start with one of the example prompts on the home screen, or just type anything.</p><p style="color:var(--mute);font-size:12px">Tip: try <code>scan C:/path/to/notes</code> to teach me from a folder, or <code>amni code</code> from a project directory to enter project-aware mode.</p>'}
];
function showWizard(){wzStep=0;document.getElementById('wizard').classList.add('show');renderWizard()}
function closeWizard(){document.getElementById('wizard').classList.remove('show');localStorage.setItem(WKEY,'1')}
function renderWizard(){
  const w=WIZARD[wzStep];
  document.getElementById('wzTitle').textContent=w.title;
  document.getElementById('wzBody').innerHTML=w.body;
  const dots=WIZARD.map((_,i)=>`<div class="dot ${i===wzStep?'active':''}"></div>`).join('');
  document.getElementById('wzDots').innerHTML=dots;
  document.getElementById('wzPrev').style.display=wzStep>0?'inline-block':'none';
  document.getElementById('wzNext').textContent=wzStep===WIZARD.length-1?'Got it!':'Next';
}
function wzNext(){if(wzStep===WIZARD.length-1){closeWizard();return}wzStep++;renderWizard()}
function wzPrev(){if(wzStep>0){wzStep--;renderWizard()}}
input.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('form').requestSubmit()}
});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(160,input.scrollHeight)+'px'});
document.getElementById('vOutLbl').textContent='Speak responses: '+(voiceOut?'on':'off');
loadProjectInfo();refreshStats();
if(!localStorage.getItem(WKEY))setTimeout(showWizard,800);
</script></body></html>"""
def mount(app):
    from fastapi.responses import HTMLResponse,Response
    @app.get('/',response_class=HTMLResponse)
    def index():return HTMLResponse(content=_HTML)
    @app.get('/favicon.ico')
    def favicon():return Response(status_code=204)
