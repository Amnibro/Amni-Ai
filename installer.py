import os,sys,subprocess,threading,json,platform,webbrowser,shutil,urllib.parse
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
PY=sys.executable
SUPPORT='the maintainer (via GitHub)'
def ensure_pywebview():
    try:import webview;return True
    except ImportError:pass
    print('[installer] installing pywebview (one-time, ~5 MB)',flush=True)
    r=subprocess.run([PY,'-m','pip','install','--upgrade','pywebview>=5'],capture_output=True,text=True)
    if r.returncode!=0:
        print('[installer] FAILED to install pywebview:',flush=True);print(r.stderr,flush=True)
        print('[installer] Fallback: run `python install.py` for the headless CLI installer.',flush=True);return False
    return True
def list_drives():
    out=[]
    if platform.system()=='Windows':
        import ctypes
        bitmask=ctypes.windll.kernel32.GetLogicalDrives()
        for i in range(26):
            if bitmask & (1<<i):
                letter=f'{chr(65+i)}:\\'
                try:t,u,f=shutil.disk_usage(letter);out.append({'path':letter,'free_gb':round(f/1e9,1),'total_gb':round(t/1e9,1)})
                except Exception:continue
    else:
        seen=set()
        for p in (Path.home(),Path('/'),Path('/mnt'),Path('/media')):
            if not p.exists():continue
            try:
                t,u,f=shutil.disk_usage(str(p))
                if (t,) in seen:continue
                seen.add((t,));out.append({'path':str(p),'free_gb':round(f/1e9,1),'total_gb':round(t/1e9,1)})
            except Exception:continue
    return out
def detect_gpu():
    nv=shutil.which('nvidia-smi')
    if nv:
        try:
            r=subprocess.run([nv,'--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            if r.returncode==0 and r.stdout.strip():return 'nvidia'
        except Exception:pass
    if platform.system()=='Linux' and shutil.which('rocminfo'):return 'amd'
    if platform.system()=='Windows':
        ps=shutil.which('powershell') or shutil.which('pwsh')
        if ps:
            try:
                r=subprocess.run([ps,'-NoProfile','-Command','(Get-CimInstance Win32_VideoController).Name'],capture_output=True,text=True,timeout=10)
                out=(r.stdout or '').lower()
                if any(k in out for k in ('nvidia','geforce','rtx','gtx','quadro','tesla')):return 'nvidia'
                if any(k in out for k in ('radeon','amd ','firepro')):return 'amd'
            except Exception:pass
    return 'cpu'
HTML=r'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Adam Installer</title><style>
:root{--bg:#0a0c10;--panel:#11141a;--ink:#e8edf2;--dim:#7a8492;--acc:#7cdcfe;--ok:#76e08c;--err:#ff7676;--mono:'Consolas','Courier New',monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,'Segoe UI',Inter,sans-serif;font-size:14px;line-height:1.5}
.h{padding:1rem 1.5rem;border-bottom:1px solid #1f2530}.h h1{margin:0;font-size:1.15rem;letter-spacing:.04em;color:var(--acc)}.h .sub{color:var(--dim);font-size:.85rem;margin-top:.25rem}
.body{padding:1.25rem 1.5rem;display:grid;gap:1rem;max-width:860px;margin:0 auto}
.row{display:grid;grid-template-columns:140px 1fr;gap:.75rem;align-items:center}
label.lbl{color:var(--dim);font-size:.85rem;text-align:right;padding-right:.5rem}
input[type=text],select,button{background:var(--panel);color:var(--ink);border:1px solid #232a35;padding:.5rem .7rem;border-radius:4px;font-family:inherit;font-size:.9rem}
input:focus,select:focus{outline:none;border-color:var(--acc)}
button{cursor:pointer;background:#1a2030}button:hover:not(:disabled){border-color:var(--acc)}button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:var(--acc);color:#0a0c10;border-color:var(--acc);font-weight:600}button.primary:hover:not(:disabled){filter:brightness(1.1)}
button.danger{border-color:var(--err);color:var(--err)}
.console{background:#06080b;border:1px solid #1f2530;border-radius:4px;padding:.75rem;height:300px;overflow:auto;font-family:var(--mono);font-size:.78rem;line-height:1.4;color:#cdd6e0;white-space:pre-wrap;word-break:break-all}
.status{padding:.6rem .8rem;border-radius:4px;font-size:.85rem;display:none}.status.ok{background:rgba(118,224,140,.08);border:1px solid var(--ok);color:var(--ok);display:block}.status.err{background:rgba(255,118,118,.08);border:1px solid var(--err);color:var(--err);display:block}.status.info{background:rgba(124,220,254,.06);border:1px solid var(--acc);color:var(--acc);display:block}
.foot{display:flex;gap:.5rem;justify-content:flex-end;padding-top:.5rem;flex-wrap:wrap}
.drive-pick{display:flex;gap:.5rem;align-items:center}.drive-pick select{flex:1}
.hint{color:var(--dim);font-size:.78rem;font-style:italic;text-align:right;padding-right:.25rem}.hint .ok{color:var(--ok);font-style:normal}.hint .err{color:var(--err);font-style:normal}
.opt-row{display:flex;gap:1.2rem;flex-wrap:wrap;align-items:center}.opt-row label{display:inline-flex;align-items:center;gap:.35rem;color:var(--ink);font-size:.85rem}.opt-row input[type=checkbox]{margin:0}
</style></head><body>
<div class="h"><h1>Adam — Amni-Ai Installer</h1><div class="sub">GF(17) texture-native AI assistant. ~20 GB self-contained bake from HuggingFace, one-time.</div></div>
<div class="body">
<div id="form">
<div class="row"><label class="lbl">Install drive</label><div class="drive-pick"><select id="drive"></select><button onclick="browseFolder()">Browse…</button></div></div>
<div class="row"><label class="lbl">Subfolder</label><input type="text" id="subfolder" value=".amni-ai" placeholder=".amni-ai"></div>
<div class="row"><label class="lbl"></label><div class="hint" id="space_hint">&nbsp;</div></div>
<div class="row"><label class="lbl">GPU vendor</label><select id="gpu"><option value="auto">Auto-detect</option><option value="nvidia">NVIDIA CUDA</option><option value="amd">AMD ROCm (Linux only)</option><option value="cpu">CPU only (~1 tok/s)</option></select></div>
<div class="row"><label class="lbl">Persona</label><select id="persona"><option>rikku</option><option>adam</option><option>yoda</option><option>mentor</option><option>scientist</option><option>haiku</option></select></div>
<div class="row"><label class="lbl">Options</label><div class="opt-row"><label><input type="checkbox" id="no_launch">Don't auto-launch server</label><label><input type="checkbox" id="skip_kernels">Skip amni_kernels build</label></div></div>
<div class="foot"><button class="primary" id="install_btn" onclick="startInstall()">Start install</button></div>
</div>
<div id="status" class="status"></div>
<div class="console" id="console" style="display:none"></div>
<div class="foot" id="post_actions" style="display:none">
<button onclick="emailSupport()">Email log to support</button>
<button onclick="copyLogPath()">Copy log path</button>
<button class="primary" id="open_chat_btn" onclick="openChat()" style="display:none">Open chat</button>
</div>
</div>
<script>
let drives=[],installing=false,logPath=null,homePath=null;
async function init(){
  const info=await pywebview.api.init();
  drives=info.drives;
  const sel=document.getElementById('drive');
  drives.forEach((d,i)=>{const o=document.createElement('option');o.value=d.path;o.textContent=`${d.path}  (${d.free_gb} GB free / ${d.total_gb} GB total)`;sel.appendChild(o);});
  let def=drives.findIndex(d=>d.free_gb>25 && d.path[0].toUpperCase()!=='C');
  if(def<0)def=drives.findIndex(d=>d.free_gb>25);
  if(def<0)def=0;
  sel.selectedIndex=def;
  document.getElementById('gpu').value=info.gpu;
  document.getElementById('persona').value=info.persona;
  updateHint();
}
function updateHint(){
  const sel=document.getElementById('drive'),d=drives.find(x=>x.path===sel.value);
  if(!d)return;
  const h=document.getElementById('space_hint');
  h.innerHTML=d.free_gb>25?`<span class="ok">${d.free_gb} GB free — plenty for the ~20 GB bake</span>`:`<span class="err">${d.free_gb} GB free — bake needs ~20 GB. Pick another drive or free up space.</span>`;
}
async function browseFolder(){
  const sel=document.getElementById('drive');
  const r=await pywebview.api.open_browse(sel.value);
  if(r){
    const letter=r.substring(0,3);
    const i=drives.findIndex(d=>d.path.toLowerCase()===letter.toLowerCase());
    if(i>=0){sel.selectedIndex=i;document.getElementById('subfolder').value=r.substring(3).replace(/^[\\\/]+/,'')||'.amni-ai';updateHint();}
  }
}
async function startInstall(){
  installing=true;
  const drive=document.getElementById('drive').value;
  const sub=document.getElementById('subfolder').value||'.amni-ai';
  homePath=drive+sub;
  const opts={home:homePath,gpu:document.getElementById('gpu').value,persona:document.getElementById('persona').value,no_launch:document.getElementById('no_launch').checked,skip_kernels:document.getElementById('skip_kernels').checked};
  document.getElementById('form').style.display='none';
  document.getElementById('console').style.display='block';
  document.getElementById('install_btn').disabled=true;
  setStatus(`Installing to ${homePath} — first run can take 10-30 min on a fast connection (PyTorch ~2.5 GB + bake ~20 GB).`,'info');
  pywebview.api.start_install(opts);
}
function appendLog(line){
  const c=document.getElementById('console');
  const atBottom=c.scrollTop+c.clientHeight>=c.scrollHeight-20;
  c.textContent+=line;
  if(atBottom)c.scrollTop=c.scrollHeight;
}
function installDone(rc,path){
  installing=false;logPath=path;
  document.getElementById('post_actions').style.display='flex';
  if(rc===0){setStatus(`Install complete. Log saved to ${path}`,'ok');document.getElementById('open_chat_btn').style.display='inline-block';}
  else{setStatus(`Install failed (exit code ${rc}). Log saved to ${path}. Click "Email log to support" to send it to the maintainer (via GitHub).`,'err');}
}
function setStatus(msg,cls){const s=document.getElementById('status');s.className='status '+cls;s.textContent=msg;}
async function emailSupport(){
  const path=await pywebview.api.email_support();
  setStatus(`Mail app opened. PLEASE ATTACH the log file: ${path}`,'info');
}
async function copyLogPath(){
  if(!logPath)return;
  try{await navigator.clipboard.writeText(logPath);setStatus(`Log path copied to clipboard: ${logPath}`,'info');}catch(e){setStatus(`Log path: ${logPath}`,'info');}
}
async function openChat(){await pywebview.api.open_chat();}
document.getElementById('drive')?.addEventListener('change',updateHint);
if(window.pywebview){init();}else{window.addEventListener('pywebviewready',init);}
</script></body></html>'''
class API:
    def __init__(self):self.window=None;self.log_path=None;self.log_lines=[];self.proc=None;self.port=8002
    def init(self):return {'drives':list_drives(),'gpu':detect_gpu(),'persona':'rikku'}
    def open_browse(self,start):
        if not self.window:return None
        import webview
        try:r=self.window.create_file_dialog(webview.FOLDER_DIALOG,directory=start)
        except Exception:r=None
        return r[0] if r else None
    def start_install(self,opts):threading.Thread(target=self._run,args=(opts,),daemon=True).start()
    def _run(self,opts):
        home=Path(opts['home']).expanduser()
        try:home.mkdir(parents=True,exist_ok=True)
        except Exception as e:
            self._emit(f'[installer] cannot create install dir {home}: {e}\n')
            if self.window:self.window.evaluate_js(f'installDone(2,{json.dumps(str(home))})')
            return
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path=home/f'install_log_{ts}.txt'
        cmd=[PY,str(ROOT/'install.py'),'--home',str(home),'--gpu',opts['gpu'],'--persona',opts['persona']]
        if opts.get('no_launch'):cmd.append('--no-launch')
        if opts.get('skip_kernels'):cmd.append('--skip-kernels')
        header=f'# Amni-Ai install log {ts}\n# Command: {" ".join(cmd)}\n# Platform: {platform.platform()} | Python {sys.version.split()[0]}\n# Install dir: {home}\n\n'
        with open(self.log_path,'w',encoding='utf-8',errors='replace') as lf:
            lf.write(header);lf.flush()
            self._emit(header)
            try:self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding='utf-8',errors='replace',cwd=str(ROOT))
            except Exception as e:
                msg=f'[installer] failed to spawn install.py: {e}\n'
                lf.write(msg);self._emit(msg)
                if self.window:self.window.evaluate_js(f'installDone(2,{json.dumps(str(self.log_path))})')
                return
            for line in self.proc.stdout:
                lf.write(line);lf.flush();self._emit(line)
            rc=self.proc.wait();lf.write(f'\n# exit code: {rc}\n')
        if self.window:self.window.evaluate_js(f'installDone({rc},{json.dumps(str(self.log_path))})')
    def _emit(self,line):
        self.log_lines.append(line)
        if len(self.log_lines)>4000:self.log_lines=self.log_lines[-2000:]
        if self.window:
            try:self.window.evaluate_js(f'appendLog({json.dumps(line)})')
            except Exception:pass
    def email_support(self):
        tail=''.join(self.log_lines[-30:]) if self.log_lines else '(no log captured)'
        subj=f'Amni-Ai install log [{platform.node()}]'
        body=(f'Adam install ran into a problem.\n\nPLEASE ATTACH the full log file from this path:\n  {self.log_path}\n\n'
              f'Platform: {platform.platform()}\nPython: {sys.version.split()[0]}\n\nLast 30 lines of output:\n----\n{tail}\n----\n')
        url=f'mailto:{SUPPORT}?subject={urllib.parse.quote(subj)}&body={urllib.parse.quote(body)}'
        try:webbrowser.open(url)
        except Exception:pass
        return str(self.log_path) if self.log_path else '(no log path)'
    def open_chat(self):
        try:webbrowser.open(f'http://127.0.0.1:{self.port}/')
        except Exception:pass
def main():
    if not ensure_pywebview():sys.exit(1)
    import webview
    api=API()
    api.window=webview.create_window('Adam — Amni-Ai Installer',html=HTML,js_api=api,width=920,height=760,resizable=True,background_color='#0a0c10')
    webview.start(debug=False)
if __name__=='__main__':main()
