"""User model-installer: search Hugging Face, warn Radeon users about unsupported archs, download + bake to GF17(rgba4)/rgba16. v6.11.40."""
import os,json,subprocess,sys,threading,time
from pathlib import Path
from typing import Dict,List,Any,Optional
_ROOT=Path(__file__).resolve().parents[2]
_LINEAR_ATTN_HINTS=('mamba','ssm','state_space','state-space','linear_attention','linear-attention','gated_delta','gateddelta','gdn','rwkv','retnet','lightning_attn','recurrent_gemma')
_MOE_HINTS=('"moe"','mixture_of_experts','num_experts','num_local_experts','n_routed_experts','moe_intermediate','expert_used','router_aux')
_SCHEMES={'rgba4':'scripts/v5_0_3_bake.py','gf17':'scripts/v5_0_3_bake.py','rgba16':'scripts/bake_fp16_to_rgba16.py'}
_JOBS:Dict[str,Dict[str,Any]]={}
def search_hf(query:str,limit:int=20)->Dict[str,Any]:
    try:from huggingface_hub import HfApi
    except Exception as e:return {'error':f'huggingface_hub unavailable: {e}','models':[]}
    try:
        api=HfApi()
        try:res=api.list_models(search=query or '',limit=int(limit),sort='downloads',direction=-1,filter='text-generation')
        except TypeError:res=api.list_models(search=query or '',limit=int(limit),sort='downloads',filter='text-generation')
        return {'query':query,'models':[{'id':m.id,'downloads':getattr(m,'downloads',0) or 0,'likes':getattr(m,'likes',0) or 0,'tags':(getattr(m,'tags',[]) or [])[:12]} for m in res]}
    except Exception as e:return {'error':str(e),'models':[]}
def detect_warnings(repo:str)->Dict[str,Any]:
    try:from huggingface_hub import hf_hub_download
    except Exception as e:return {'error':f'huggingface_hub unavailable: {e}'}
    try:cfg=json.loads(Path(hf_hub_download(repo,'config.json')).read_text(encoding='utf-8'))
    except Exception as e:return {'repo':repo,'error':f'could not read config.json: {e}'}
    txt=json.dumps(cfg).lower();arch=(cfg.get('architectures') or [''])[0]
    is_linear=any(h in txt for h in _LINEAR_ATTN_HINTS) or 'linear' in (cfg.get('layer_types') and json.dumps(cfg.get('layer_types')).lower() or '')
    is_moe=any(h in txt for h in _MOE_HINTS)
    warnings=[]
    if is_linear and is_moe:warnings.append('LINEAR-ATTENTION MoE detected — does NOT work correctly on AMD Radeon/ROCm in this build. The state-space / gated-delta path produces wrong results on gfx11xx. Strongly discouraged on this GPU.')
    elif is_linear:warnings.append('Linear-attention / state-space (Mamba/SSM/GDN/RWKV) layers detected — these are unreliable on AMD Radeon/ROCm in this build.')
    elif is_moe:warnings.append('Mixture-of-Experts model — works but uses more VRAM; ensure it fits your card.')
    nl=cfg.get('num_hidden_layers') or cfg.get('n_layer');hs=cfg.get('hidden_size') or cfg.get('n_embd')
    return {'repo':repo,'arch':arch,'is_linear_attention':bool(is_linear),'is_moe':bool(is_moe),'num_hidden_layers':nl,'hidden_size':hs,'radeon_warnings':warnings,'recommended_scheme':'rgba4','schemes_available':list(_SCHEMES.keys())}
def _do_install(job_id,repo,scheme,dest,out,model_name):
    j=_JOBS[job_id]
    try:
        from huggingface_hub import snapshot_download
        j['phase']='downloading'
        snapshot_download(repo_id=repo,local_dir=dest,allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','*.model','*.jinja'])
        j['phase']='baking'
        py=str(_ROOT/'.venv/Scripts/python.exe');py=py if Path(py).exists() else sys.executable
        env=dict(os.environ);env['AMNI_NO_GPU_DETECT']='1';env['HIP_VISIBLE_DEVICES']='';env['CUDA_VISIBLE_DEVICES']=''
        r=subprocess.run([py,str(_ROOT/_SCHEMES[scheme]),'--src',dest,'--out',out,'--model-name',model_name],capture_output=True,text=True,env=env)
        for f in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','processor_config.json','special_tokens_map.json'):
            s=Path(dest)/f
            if s.exists():
                try:(Path(out)/f).write_bytes(s.read_bytes())
                except Exception:pass
        j['phase']='done' if r.returncode==0 else 'failed';j['bake_rc']=r.returncode;j['bake_tail']=(r.stdout or '')[-600:]+(r.stderr or '')[-400:];j['out']=out
    except Exception as e:j['phase']='failed';j['error']=str(e)
def install(repo:str,scheme:str='rgba4',model_name:Optional[str]=None)->Dict[str,Any]:
    if scheme not in _SCHEMES:return {'error':f'unknown scheme {scheme!r}; valid: {list(_SCHEMES.keys())}'}
    safe=repo.replace('/','_').replace('.','_').lower();model_name=model_name or f'{safe}_{scheme}'
    dest=str(_ROOT/'downloaded_models'/safe);out=str(_ROOT/'bakes'/f'{safe}_{scheme}')
    job_id=f'{safe}_{scheme}_{int(time.time())}'
    _JOBS[job_id]={'job_id':job_id,'repo':repo,'scheme':scheme,'phase':'queued','out':out}
    threading.Thread(target=_do_install,args=(job_id,repo,scheme,dest,out,model_name),daemon=True).start()
    return {'job_id':job_id,'repo':repo,'scheme':scheme,'phase':'queued','poll':f'/install/status/{job_id}'}
def status(job_id:str)->Dict[str,Any]:return _JOBS.get(job_id) or {'error':'unknown job_id'}
def detect_hardware()->Dict[str,Any]:
    hw={'vram_gb':0.0,'ram_gb':0.0,'gpu':'CPU only','backend':'cpu'}
    try:import psutil;hw['ram_gb']=round(psutil.virtual_memory().total/1e9,1)
    except Exception:
        try:hw['ram_gb']=round(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9,1)
        except Exception:pass
    try:
        import torch
        if torch.cuda.is_available():
            p=torch.cuda.get_device_properties(0);hw['vram_gb']=round(p.total_memory/1e9,1);hw['gpu']=p.name
            hw['backend']='rocm' if getattr(torch.version,'hip',None) else 'cuda'
    except Exception:pass
    return hw
_CATALOG=[
    {'tier':'flagship','name':'Adam · Gemma-4-12B (NVFP4 lossless)','params':'12B','min_vram':12.5,'min_ram':16,'quality':'Best — MMLU-Pro 71%, coding 90→100% (self-correct); matches stock Gemma-4-12B','source':'AxionML/Gemma-4-12B-NVFP4','scheme':'nvfp4','bake':'bakes/gemma4_12b_nvfp4_atex'},
    {'tier':'balanced','name':'Adam · Qwen3-4B','params':'4B','min_vram':5.5,'min_ram':10,'quality':'Strong reasoning for its size — great on 8GB cards','source':'Qwen/Qwen3-4B-Instruct-2507','scheme':'rgba4','bake':None},
    {'tier':'light','name':'Adam · Granite-3B (GF17)','params':'3B','min_vram':3.0,'min_ram':8,'quality':'Solid + very light; fits small GPUs','source':'ibm-granite/granite-3.1-3b-a800m-instruct','scheme':'rgba4','bake':'bakes/granite41_3b_tilepack'},
    {'tier':'cpu','name':'Adam · Granite-350M','params':'350M','min_vram':0.0,'min_ram':4,'quality':'Runs anywhere (CPU/iGPU); basic but always works','source':'ibm-granite/granite-3.1-1b-a400m-instruct','scheme':'rgba4','bake':None},
]
def advise(hw:Optional[Dict[str,Any]]=None,headroom_gb:float=1.8)->Dict[str,Any]:
    hw=hw or detect_hardware()
    usable=max(0.0,hw['vram_gb']-headroom_gb) if hw['vram_gb']>0 else 0.0
    pick=None
    for m in _CATALOG:
        if m['tier']=='cpu':continue
        if usable>=m['min_vram'] and hw['ram_gb']>=m['min_ram']*0.75:pick=m;break
    if pick is None:pick=_CATALOG[-1]
    nxt=next((m for m in reversed(_CATALOG) if m['min_vram']>pick['min_vram']),None)
    if pick['tier']=='cpu':
        why=f"No usable GPU detected (VRAM {hw['vram_gb']}GB) — running the CPU-friendly {pick['params']} model on your {hw['ram_gb']}GB RAM."
    else:
        why=f"Your {hw['gpu']} has {hw['vram_gb']}GB VRAM (~{usable:.1f}GB usable after a {headroom_gb}GB headroom for activations/KV). That fits the {pick['params']} {pick['tier']} model (needs ~{pick['min_vram']}GB)."
        if nxt and usable<nxt['min_vram']:why+=f" The bigger {nxt['params']} tier needs ~{nxt['min_vram']}GB — more than your card has."
        elif not nxt:why+=" That's the top tier — you can run the best Adam."
    installed=bool(pick.get('bake')) and (_ROOT/pick['bake']/'bake_manifest.json').exists()
    return {'hardware':hw,'recommended':pick,'why':why,'already_installed':installed,'next_tier_needs':(nxt['min_vram'] if nxt else None),'catalog':[{k:v for k,v in m.items() if k!='bake'} for m in _CATALOG]}
def advise_install()->Dict[str,Any]:
    a=advise();m=a['recommended']
    if a['already_installed']:return {**a,'action':'ready','message':f"{m['name']} is already installed — just (re)start the server."}
    if m['scheme']=='nvfp4':return {**a,'action':'manual','message':f"Fetch the flagship: download {m['source']} then bake with scripts/bake_nvfp4_atex.py (needs ~12GB free)."}
    job=install(m['source'],scheme=m['scheme'],model_name=f"adam_{m['tier']}")
    return {**a,'action':'installing','job':job}
_BAKE_CATALOG=[
    {'key':'gemma-e2b','repo':'amnibro/gemma-4-E2B-it-gf17','dir':'gemma4_e2b_it_gf17','label':'Gemma-4 E2B (tiny / CPU-friendly)','download_gb':3.0,'resident_gb':3.0,'min_vram':0.0,'min_ram':6,'speed':'Runs anywhere — CPU ok (~1 tok/s on CPU)','quality':'Basic but always works','ready':True},
    {'key':'granite-3b','repo':'amnibro/granite41-3b-palette','dir':'granite41_3b_palette','label':'Granite-3B Palette (default, lossless)','download_gb':8.0,'resident_gb':4.0,'min_vram':4.0,'min_ram':8,'speed':'Fast on any 4GB+ GPU','quality':'Solid all-rounder — the default','ready':True},
    {'key':'gemma-12b','repo':'amnibro/gemma-4-12b-nvfp4','dir':'gemma4_12b_nvfp4_atex','label':'Gemma-4-12B NVFP4 (flagship)','download_gb':13.0,'resident_gb':13.0,'min_vram':14.0,'min_ram':16,'speed':'Fast on 16GB+ GPUs','quality':'Best — MMLU-Pro 71%, coding 100% (self-correct)','ready':False},
]
def _bake_has_weights(dest):
    p=Path(dest)
    return (p/'bake_manifest.json').exists() or (p/'manifest.json').exists() or (p/'tensors').exists() or any(p.glob('*.palette')) or any(p.glob('*.ptex')) or any(p.glob('*.safetensors'))
def _do_bake_download(job_id,repo,dest):
    j=_JOBS[job_id]
    try:
        from huggingface_hub import snapshot_download
        j['phase']='downloading';snapshot_download(repo_id=repo,local_dir=dest)
        if not _bake_has_weights(dest):j['phase']='failed';j['error']='downloaded but no bake manifest/weights found';return
        try:
            from amni.bootstrap import load_config,save_config
            cfg=load_config();cfg['bake']=dest;cfg['model']=dest;cfg['hf_bake_repo']=repo;save_config(cfg);j['config_saved']=True
        except Exception as ce:j['config_saved']=False;j['config_error']=str(ce)[:100]
        j['phase']='done';j['bake']=dest
    except Exception as e:j['phase']='failed';j['error']=str(e)
def download_bake_tier(key:str)->Dict[str,Any]:
    cat=next((c for c in _BAKE_CATALOG if c['key']==key),None)
    if cat is None:return {'error':f'unknown bake {key!r}; valid: {[c["key"] for c in _BAKE_CATALOG]}'}
    if not cat.get('ready'):return {'error':f"{cat['label']} is not on HuggingFace yet — coming soon"}
    dest=str(_ROOT/'bakes'/cat['dir']);job_id=f"bake_{cat['dir']}_{int(time.time())}"
    _JOBS[job_id]={'job_id':job_id,'repo':cat['repo'],'dir':cat['dir'],'phase':'queued','bake':dest}
    threading.Thread(target=_do_bake_download,args=(job_id,cat['repo'],dest),daemon=True).start()
    return {'job_id':job_id,'repo':cat['repo'],'phase':'queued','poll':f'/install/status/{job_id}'}
def bake_catalog_view()->Dict[str,Any]:
    a=advise();hw=a['hardware'];out=[]
    for c in _BAKE_CATALOG:
        installed=(_ROOT/'bakes'/c['dir']/'bake_manifest.json').exists() or (_ROOT/'bakes'/c['dir']/'manifest.json').exists()
        fit='green' if (c['min_vram']>0 and hw['vram_gb']>=c['min_vram']) else ('amber' if hw['ram_gb']>=c['min_ram'] else 'red')
        out.append({**c,'fit':fit,'installed':bool(installed),'can_download':bool(c.get('ready')) and fit!='red'})
    return {'hardware':hw,'recommended':(a.get('recommended') or {}).get('name'),'bakes':out}
def mount(app):
    from fastapi import Query
    from fastapi.responses import HTMLResponse
    @app.get('/advise')
    def _advise():return advise()
    @app.post('/advise/install')
    def _advise_install():return advise_install()
    @app.get('/install/catalog')
    def _catalog():return bake_catalog_view()
    @app.post('/install/bake')
    def _bake(key:str=Query(...)):return download_bake_tier(key)
    @app.get('/picker',response_class=HTMLResponse)
    def _picker():return _PICKER_HTML
    @app.get('/first-run',response_class=HTMLResponse)
    def _firstrun():return _PICKER_HTML
    @app.get('/install/search')
    def _search(q:str=Query(''),limit:int=20):return search_hf(q,limit)
    @app.get('/install/inspect')
    def _inspect(repo:str=Query(...)):return detect_warnings(repo)
    @app.post('/install/start')
    def _start(repo:str=Query(...),scheme:str=Query('rgba4'),model_name:str=Query(None)):return install(repo,scheme,model_name)
    @app.get('/install/status/{job_id}')
    def _status(job_id:str):return status(job_id)
    @app.get('/install',response_class=HTMLResponse)
    def _ui():return _INSTALLER_HTML
    @app.get('/models',response_class=HTMLResponse)
    def _ui2():return _INSTALLER_HTML
    return app
_INSTALLER_HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adam · Model Installer</title>
<style>
:root{--bg:#070b14;--panel:#0e1626;--panel2:#131f33;--line:#1f2e47;--ink:#e8eef9;--dim:#8aa0c0;--accent:#ff8a3c;--accent2:#ffb070;--ok:#34d399;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#10203a 0,var(--bg) 60%);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 2px;letter-spacing:.5px}h1 b{color:var(--accent)}
.sub{color:var(--dim);margin:0 0 22px;font-size:13px}
.bar{display:flex;gap:10px;margin-bottom:18px}
input,select,button{font:inherit;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:11px 14px;outline:none}
input{flex:1}input:focus{border-color:var(--accent)}
button{cursor:pointer;background:linear-gradient(180deg,var(--accent2),var(--accent));color:#1a0f06;border:none;font-weight:700;box-shadow:0 6px 18px -8px var(--accent)}
button.ghost{background:var(--panel2);color:var(--ink);font-weight:600;box-shadow:none;border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}
.card{background:linear-gradient(180deg,var(--panel),#0b1322);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:12px}
.row{display:flex;align-items:center;gap:12px;justify-content:space-between}
.mid{font-weight:700}.id{font-family:ui-monospace,Consolas,monospace;font-size:13.5px;color:var(--accent2);word-break:break-all}
.meta{color:var(--dim);font-size:12.5px;display:flex;gap:14px;flex-wrap:wrap;margin-top:3px}
.tags{margin-top:7px;display:flex;gap:6px;flex-wrap:wrap}.tag{font-size:11px;color:var(--dim);background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:2px 9px}
.detail{margin-top:12px;border-top:1px solid var(--line);padding-top:12px;display:none}
.warn{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.45);color:#ffd7d7;border-radius:10px;padding:10px 12px;margin:8px 0;font-size:13px}
.warn.amber{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.45);color:#ffe9bd}
.okline{color:var(--ok);font-size:13px;margin:6px 0}
.kv{color:var(--dim);font-size:13px;margin:2px 0}.kv b{color:var(--ink);font-weight:600}
.controls{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
.job{margin-top:10px;font-size:13px;color:var(--dim)}.phase{font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:.5px}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px;margin-right:6px}@keyframes s{to{transform:rotate(360deg)}}
.empty{color:var(--dim);text-align:center;padding:40px 0}
.foot{color:var(--dim);font-size:12px;margin-top:26px;text-align:center}
</style></head><body><div class="wrap">
<h1><b>⬡ Adam</b> · Model Installer</h1>
<p class="sub">Search Hugging Face → bake to GF17 / rgba16 (lossless) → run on Adam. AMD Radeon users are warned about unsupported architectures.</p>
<div class="bar"><input id="q" placeholder="Search models (e.g. granite, llama, qwen, phi)…" autofocus><button id="go">Search</button></div>
<div id="results"><div class="empty">Type a query and hit Search.</div></div>
<div class="foot">Backend: <code>/install/search · /install/inspect · /install/start · /install/status</code> — bakes are CPU-only and verified bit-exact.</div>
</div>
<script>
const $=s=>document.querySelector(s),res=$('#results');
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function jget(u){const r=await fetch(u);return r.json()}
async function search(){const q=$('#q').value.trim();res.innerHTML='<div class="empty"><span class="spin"></span>searching…</div>';
 let d;try{d=await jget('/install/search?q='+encodeURIComponent(q)+'&limit=24')}catch(e){res.innerHTML='<div class="empty">search failed: '+esc(e)+'</div>';return}
 if(d.error){res.innerHTML='<div class="empty">'+esc(d.error)+'</div>';return}
 const ms=d.models||[];if(!ms.length){res.innerHTML='<div class="empty">no models found.</div>';return}
 res.innerHTML=ms.map((m,i)=>`<div class="card" data-repo="${esc(m.id)}"><div class="row"><div><div class="id">${esc(m.id)}</div><div class="meta"><span>⬇ ${Number(m.downloads).toLocaleString()}</span><span>♥ ${esc(m.likes)}</span></div></div><button class="ghost insp" data-i="${i}">Inspect →</button></div><div class="tags">${(m.tags||[]).slice(0,8).map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</div><div class="detail" id="d${i}"></div></div>`).join('');
}
async function inspect(i,repo){const box=$('#d'+i);box.style.display='block';box.innerHTML='<span class="spin"></span>inspecting architecture…';
 let d;try{d=await jget('/install/inspect?repo='+encodeURIComponent(repo))}catch(e){box.innerHTML='inspect failed';return}
 if(d.error){box.innerHTML='<div class="warn">'+esc(d.error)+'</div>';return}
 const warns=(d.radeon_warnings||[]);
 let h=`<div class="kv">arch <b>${esc(d.arch||'?')}</b> · layers <b>${esc(d.num_hidden_layers)}</b> · hidden <b>${esc(d.hidden_size)}</b> · linear-attn <b>${d.is_linear_attention}</b> · MoE <b>${d.is_moe}</b></div>`;
 if(warns.length){const amber=!d.is_linear_attention; h+=warns.map(w=>`<div class="warn ${amber?'amber':''}">⚠️ ${esc(w)}</div>`).join('');}
 else h+=`<div class="okline">✓ No Radeon compatibility issues detected.</div>`;
 h+=`<div class="controls"><label class="kv">Bake scheme</label><select id="sch${i}"><option value="rgba4">rgba4 (GF17, ~4B/wt)</option><option value="rgba16">rgba16 (~2B/wt, RAM-light · experimental)</option></select><button class="dl" data-i="${i}" data-repo="${esc(repo)}" ${d.is_linear_attention?'data-warn="1"':''}>⬇ Download & Bake</button></div><div class="job" id="j${i}"></div>`;
 box.innerHTML=h;
}
async function startInstall(i,repo,warned){if(warned&&!confirm('This architecture is flagged as unsupported on AMD Radeon and will likely produce wrong results. Install anyway?'))return;
 const scheme=$('#sch'+i).value,jb=$('#j'+i);jb.innerHTML='<span class="spin"></span>starting…';
 let d;try{d=await(await fetch('/install/start?repo='+encodeURIComponent(repo)+'&scheme='+scheme,{method:'POST'})).json()}catch(e){jb.innerHTML='start failed';return}
 if(d.error){jb.innerHTML='<div class="warn">'+esc(d.error)+'</div>';return}
 poll(d.job_id,jb);
}
async function poll(id,jb){let d;try{d=await jget('/install/status/'+id)}catch(e){jb.innerHTML='status lost';return}
 const ph=d.phase||'?';const done=ph==='done',fail=ph==='failed';
 jb.innerHTML=`<span class="phase">${done?'✓ ':''}${fail?'✗ ':'<span class=spin></span>'}${esc(ph)}</span>${d.out?' → <code>'+esc(d.out)+'</code>':''}${fail&&d.bake_tail?'<div class="warn">'+esc((d.error||'')+' '+(d.bake_tail||''))+'</div>':''}`;
 if(!done&&!fail)setTimeout(()=>poll(id,jb),1500);
}
$('#go').onclick=search;$('#q').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
if(location.search.includes('demo')){(async()=>{$('#q').value='mamba';await search();const f=document.querySelector('.insp');if(f){const c=f.closest('.card');await inspect(f.dataset.i,c.dataset.repo)}})()}
res.addEventListener('click',e=>{const ins=e.target.closest('.insp');if(ins){const c=ins.closest('.card');inspect(ins.dataset.i,c.dataset.repo);return}
 const dl=e.target.closest('.dl');if(dl){startInstall(dl.dataset.i,dl.dataset.repo,dl.dataset.warn==='1')}});
</script></body></html>'''
_PICKER_HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adam - Choose your model</title>
<style>
:root{--bg:#070b14;--panel:#0e1626;--panel2:#131f33;--line:#1f2e47;--ink:#e8eef9;--dim:#8aa0c0;--accent:#ff8a3c;--accent2:#ffb070;--green:#34d399;--amber:#fbbf24;--red:#f87171}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#10203a 0,var(--bg) 60%);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
.wrap{max-width:920px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 2px}h1 b{color:var(--accent)}
.sub{color:var(--dim);margin:0 0 18px;font-size:13px}
.hw{background:linear-gradient(180deg,var(--panel),#0b1322);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:18px;font-size:14px}
.hw b{color:var(--accent2)}
.card{background:linear-gradient(180deg,var(--panel),#0b1322);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:12px;position:relative}
.card.rec{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.rectag{position:absolute;top:-9px;right:14px;background:var(--accent);color:#1a0f06;font-size:11px;font-weight:700;border-radius:999px;padding:2px 10px}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:14px}
.title{font-weight:700;font-size:16px}.q{color:var(--dim);font-size:13px;margin:2px 0 8px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.b{font-size:12px;border-radius:999px;padding:2px 10px;border:1px solid var(--line);color:var(--dim);background:var(--panel2)}
.fit{font-weight:700}.fit.green{color:var(--green);border-color:var(--green)}.fit.amber{color:var(--amber);border-color:var(--amber)}.fit.red{color:var(--red);border-color:var(--red)}
.speed{font-size:13px;color:var(--dim);margin-top:4px}
button{font:inherit;cursor:pointer;border:none;border-radius:10px;padding:10px 18px;font-weight:700;background:linear-gradient(180deg,var(--accent2),var(--accent));color:#1a0f06;white-space:nowrap}
button:disabled{opacity:.45;cursor:default;background:var(--panel2);color:var(--dim)}
.job{margin-top:10px;font-size:13px;color:var(--accent2)}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px;margin-right:6px}@keyframes s{to{transform:rotate(360deg)}}
.empty{color:var(--dim);text-align:center;padding:40px}
</style></head><body><div class="wrap">
<h1><b>&#9889; Adam</b> - Choose your model</h1>
<p class="sub">Pick the bake that fits your hardware. We detected your machine and highlight the best fit. You can change it later.</p>
<div id="hw" class="hw"><span class="spin"></span>detecting hardware...</div>
<div id="cards"><div class="empty"><span class="spin"></span>loading catalog...</div></div>
</div>
<script>
const $=s=>document.querySelector(s),esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const FITLABEL={green:'✓ Fits your GPU',amber:'⚠ CPU / partial - slower',red:'✕ Needs more VRAM/RAM'};
async function load(){let d;try{d=await(await fetch('/install/catalog')).json()}catch(e){$('#cards').innerHTML='<div class=empty>catalog failed</div>';return}
 const h=d.hardware||{};$('#hw').innerHTML='<b>'+esc(h.gpu||'CPU')+'</b> &middot; VRAM <b>'+(h.vram_gb||0)+' GB</b> &middot; RAM <b>'+(h.ram_gb||0)+' GB</b>'+(d.recommended?' &nbsp;|&nbsp; Recommended: <b>'+esc(d.recommended)+'</b>':'');
 const recName=d.recommended||'';
 $('#cards').innerHTML=(d.bakes||[]).map(function(b){
  const rec=recName&&b.label&&recName.indexOf(b.label.split(' (')[0])>=0;
  const btn=b.installed?'<button disabled>Installed ✓</button>':(!b.ready?'<button disabled>Coming soon</button>':(b.fit==='red'?'<button disabled>Needs more VRAM</button>':'<button onclick="dl(\''+b.key+'\',this)">⬇ Download ('+b.download_gb+' GB)</button>'));
  return '<div class="card'+(rec?' rec':'')+'">'+(rec?'<span class=rectag>RECOMMENDED</span>':'')+
   '<div class=row><div><div class=title>'+esc(b.label)+'</div><div class=q>'+esc(b.quality)+'</div>'+
   '<div class=badges><span class="b fit '+b.fit+'">'+FITLABEL[b.fit]+'</span><span class=b>⬇ '+b.download_gb+' GB download</span><span class=b>'+b.resident_gb+' GB resident</span></div>'+
   '<div class=speed>'+esc(b.speed)+'</div></div><div>'+btn+'</div></div><div class=job id="j_'+b.key+'"></div></div>'}).join('');
}
async function dl(key,btn){btn.disabled=true;const jb=$('#j_'+key);jb.innerHTML='<span class=spin></span>starting...';
 let d;try{d=await(await fetch('/install/bake?key='+encodeURIComponent(key),{method:'POST'})).json()}catch(e){jb.innerHTML='start failed';btn.disabled=false;return}
 if(d.error){jb.innerHTML=esc(d.error);btn.disabled=false;return}poll(d.job_id,jb,btn)}
async function poll(id,jb,btn){let d;try{d=await(await fetch('/install/status/'+id)).json()}catch(e){jb.innerHTML='status lost';return}
 const p=d.phase||'?';if(p==='done'){jb.innerHTML='✓ Downloaded & set as your model - restart Adam to use it.';btn.textContent='Installed ✓';return}
 if(p==='failed'){jb.innerHTML='✕ '+esc(d.error||'failed');btn.disabled=false;return}
 jb.innerHTML='<span class=spin></span>'+esc(p)+'... (large download, please wait)';setTimeout(function(){poll(id,jb,btn)},1500)}
load();
</script></body></html>'''
