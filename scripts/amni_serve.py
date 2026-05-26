"""v6.0.0 deployable Adam server.
Routes:
  GET  /                  — simple chat UI
  POST /chat              — agent chat (multi-turn, skill dispatch)
  POST /ask               — single-shot Adam (no skills, no memory)
  POST /teach             — add (q,a) to lesson bank
  GET  /stats             — Adam + agent stats
  GET  /healthz           — liveness
  GET  /skills            — list registered skills
  POST /skills/{name}     — direct skill invocation
  GET  /sessions          — list conversation sessions
  DELETE /sessions/{id}   — delete a session
  /api/tags /api/show /api/generate /api/chat /api/embed /api/version  — Ollama compat
Usage:
  python scripts/amni_serve.py --seed
  Then point Open WebUI at http://localhost:8001 or open http://localhost:8001 in a browser.
"""
import os,sys,argparse,time,socket,subprocess,signal,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def _kill_stale_probes():
    try:
        import psutil as _ps
        my_pid=os.getpid();killed=0
        for p in _ps.process_iter(['pid','cmdline']):
            try:
                pid=p.info.get('pid')
                if not pid or pid==my_pid:continue
                cl=' '.join(p.info.get('cmdline') or [])
                if 'import torch,json' in cl and 'gcnArchName' in cl:p.kill();killed+=1
            except Exception:pass
        if killed:print(f'[amni_serve] killed {killed} stale GPU-probe subprocess(es)',flush=True)
    except ImportError:pass
def _gpu_bootstrap():
    if os.environ.get('AMNI_NO_GPU_DETECT'):return
    _kill_stale_probes()
    cache_dir=Path(__file__).resolve().parents[1]/'.amni';cache=cache_dir/'gpu.json'
    info=None
    if cache.exists():
        try:info=json.loads(cache.read_text())
        except:info=None
    if not info:
        probe='import torch,json;\nprint(json.dumps([{"i":i,"arch":(getattr(torch.cuda.get_device_properties(i),"gcnArchName","") or "").split(":")[0],"name":torch.cuda.get_device_name(i),"mem":int(torch.cuda.get_device_properties(i).total_memory)} for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []))'
        try:
            r=subprocess.run([sys.executable,'-c',probe],capture_output=True,text=True,timeout=120)
            info=json.loads((r.stdout or '').strip().splitlines()[-1])
            cache_dir.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(info))
        except Exception as e:print(f'[amni_serve] GPU probe failed ({e}) - running with defaults',flush=True);return
    if not info:print('[amni_serve] no CUDA/ROCm GPU detected - running on CPU',flush=True);return
    amd=[d for d in info if (d.get('arch') or '').startswith('gfx')]
    if not amd:print(f'[amni_serve] non-AMD GPU detected ({info[0].get("name")}) - leaving env untouched',flush=True);return
    best=max(amd,key=lambda d:d.get('mem',0))
    arch=best['arch']
    fam={'gfx900':'9.0.0','gfx906':'9.0.6','gfx908':'9.0.8','gfx90a':'9.0.10','gfx940':'9.4.0','gfx941':'9.4.1','gfx942':'9.4.2','gfx1010':'10.1.0','gfx1011':'10.1.1','gfx1012':'10.1.2','gfx1030':'10.3.0','gfx1031':'10.3.1','gfx1032':'10.3.2','gfx1100':'11.0.0','gfx1101':'11.0.0','gfx1102':'11.0.0','gfx1103':'11.0.0','gfx1150':'11.0.0','gfx1151':'11.0.0','gfx1200':'12.0.0','gfx1201':'12.0.1'}
    ver=fam.get(arch)
    os.environ.setdefault('PYTORCH_ROCM_ARCH',arch)
    if ver:os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION',ver)
    for k,v in (('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1'),('HIP_FORCE_DEV_KERNARG','1'),('GPU_MAX_HW_QUEUES','8')):os.environ.setdefault(k,v)
    if sys.platform=='win32':
        rocm_root=os.environ.get('HIP_PATH') or os.environ.get('ROCM_PATH')
        if not rocm_root:
            for cand in (r'C:\Program Files\AMD\ROCm\7.1',r'C:\Program Files\AMD\ROCm\7.0',r'C:\Program Files\AMD\ROCm\6.4'):
                if os.path.isdir(cand):rocm_root=cand;break
        rbin=os.path.join((rocm_root or '').rstrip('\\'),'bin') if rocm_root else None
        if rbin and os.path.isdir(rbin):
            try:os.add_dll_directory(rbin)
            except Exception:pass
            cur=os.environ.get('PATH','')
            if rbin not in cur:os.environ['PATH']=rbin+os.pathsep+cur
    chosen_idx=best['i'];total=len(info);non_target=[d for d in info if d['i']!=chosen_idx]
    if non_target and total>1:os.environ.setdefault('HIP_VISIBLE_DEVICES',str(chosen_idx))
    print(f'[amni_serve] GPU bootstrap: idx={chosen_idx} arch={arch} name={best.get("name")} vram={best.get("mem",0)//(1024**3)}GB override={ver or "auto"}',flush=True)
_gpu_bootstrap()
from amni.bootstrap import load_config,DEFAULT_PORT,DEFAULT_HOST
from amni.storage.conversation_notes import ConversationNotes
from amni.serve.agentic import run_goal_stream,is_build_request
from amni.serve import tone_atlas
_UNCERTAIN_RE=re.compile(r"\b(?:i\s+don't\s+(?:know|have|recall|remember)|i'?m\s+not\s+(?:sure|certain)|let\s+me\s+(?:check|look|search|find\s+out|verify)|i\s+(?:need|should|gotta|have\s+to|might|want|ought|must)\s+(?:to\s+)?(?:check|look\s*(?:up|into)|search|verify|find\s+out|investigate|confirm|research)|specific\s+(?:data|info|information|details)\s+(?:unavailable|missing|not\s+available)|data\s+insufficient|i'?ll\s+(?:search|look|check|need\s+to)|i\s+don't\s+have\s+(?:specific|detailed|enough|the)|i\s+can'?t\s+(?:confirm|verify|recall)|i\s+ain'?t\s+sure|i\s+(?:do\s+not|can\s*not)\s+have\s+(?:specific|detailed|access)|requires?\s+(?:additional|more)\s+(?:info|data|context|verification)|not\s+sure\s+about|uncertain\s+about|check\s+(?:local|recent|the\s+latest|current))",re.IGNORECASE)
_TOOL_ANNOUNCE_PREFIX_RE=re.compile(r"^\s*(?:(?:Rao|Oui|Oac|Sure|Okay|Ok|Yeah)[!,.\s]+)?(?:I[' ]?(?:ll|d| will| am going to|'m going to|am gonna|'m gonna)\s+(?:look|check|search|find|see|verify|investigate|grab|fetch|pull|get)\s+(?:that\s+|it\s+|this\s+|some\s+)?(?:up|out|for\s+(?:you|ya))?(?:[!,.\s]+(?:right\s+(?:now|away|quick))?)?|let\s+me\s+(?:look|check|search|find|see|verify|investigate|grab|fetch|pull|get)\s+(?:that\s+|it\s+|this\s+|some\s+)?(?:up|out|for\s+(?:you|ya))?(?:[!,.\s]+(?:right\s+(?:now|away|quick))?)?|one\s+moment|give\s+me\s+a\s+sec(?:ond)?|hold\s+on|just\s+a\s+sec(?:ond)?)[!,.\s]*",re.IGNORECASE)
def _strip_tool_announce(text):
    if not text:return text
    cleaned=text;tries=0
    while tries<3:
        m=_TOOL_ANNOUNCE_PREFIX_RE.match(cleaned)
        if not m or m.end()==0:break
        cleaned=cleaned[m.end():].lstrip()
        tries+=1
    cleaned=re.sub(r'\s*\[Looked\b.*$','',cleaned,flags=re.IGNORECASE|re.DOTALL)
    cleaned=re.sub(r'\s*\[Search\s+(?:performed|completed|done|results?)?.*$','',cleaned,flags=re.IGNORECASE|re.DOTALL)
    cleaned=re.sub(r'\s*\[(?:Presenting|Current weather|Result of|Outputting|Searching).*$','',cleaned,flags=re.IGNORECASE|re.DOTALL)
    return cleaned.strip() if cleaned.strip() else text
_VAGUE_WEB_RE=re.compile(r"\b(?:on\s+the\s+web|online|out\s+there|what'?s\s+new|news|latest|current\s+events|anything\s+(?:exciting|new|interesting|happening))\b",re.IGNORECASE)
_PROPER_NOUN_RE=re.compile(r"\b([A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+)*(?:\s+(?:NY|CA|TX|FL|UK|USA|US))?)\b")
_LOCAL_INTENT_RE=re.compile(r"\b(?:near\s+me|nearby|around\s+(?:here|me)|local|in\s+my\s+area|in\s+the\s+area|events?|things\s+to\s+do|restaurants?|stores?|shops?|attractions?|family[- ]friendly|kid[- ]friendly|child[- ]friendly|weather|forecast)\b",re.IGNORECASE)
def _enrich_web_query(user_msg,conv,profile):
    q=(user_msg or '').strip()
    if not q:return q
    q=re.sub(r'^(?:use\s+|please\s+)?(?:web|search|google)(?:\s+skill)?\s*[:]\s*','',q,flags=re.IGNORECASE)
    needs_context=len(q.split())<6 or bool(_VAGUE_WEB_RE.search(q)) or bool(_LOCAL_INTENT_RE.search(q))
    if not needs_context:return q[:200]
    extras=[]
    try:
        for t in reversed(conv.turns[-6:]) if conv and conv.turns else []:
            if t.get('role')=='assistant':
                content=t.get('content') or ''
                for nm in _PROPER_NOUN_RE.findall(content)[:5]:
                    if nm.lower() not in q.lower() and nm.lower() not in ('I','You','Rao','Fryd','Oui','Oac','the maintainer') and nm not in extras:extras.append(nm)
                if extras:break
    except Exception:pass
    if profile is not None:
        loc=(profile.data.get('location') or '') if hasattr(profile,'data') else ''
        if loc and loc.lower() not in q.lower() and loc not in extras:extras.append(loc)
    if _VAGUE_WEB_RE.search(q) and not any(k in q.lower() for k in ('news','event')):extras.insert(0,'news current events')
    enriched=(' '.join(extras)+' '+q).strip() if extras else q
    return enriched[:200]
_PROFILE_AUTHORITATIVE_RE=re.compile(r"\b(?:what(?:'s|\s+is)\s+my\s+(?:name|location|address|job|role|title|occupation|workplace|favorite)|where\s+do\s+i\s+(?:live|work|reside)|who\s+am\s+i|where\s+am\s+i\s+(?:from|based|located)|what\s+do\s+i\s+(?:do|like|prefer)|do\s+you\s+(?:remember|know)\s+(?:my|where\s+i)|tell\s+me\s+(?:my|about\s+me))",re.IGNORECASE)
_MEMORY_RECALL_RE=re.compile(r"\b(?:do\s+you\s+remember(?:\s+what|\s+we|\s+our)|what\s+(?:were|was)\s+we\s+(?:talking\s+about|discussing|just\s+saying)|what\s+did\s+we\s+(?:talk\s+about|discuss|cover|chat\s+about)|recall\s+our|last\s+time\s+we|what\s+(?:were|was)\s+(?:i|you)\s+saying|where\s+(?:were|did)\s+we\s+leave\s+off|continue\s+(?:our|where\s+we)|what\s+(?:were|was)\s+(?:we|i)\s+working\s+on)",re.IGNORECASE)
_INTROSPECT_NO_WEB_RE=re.compile(r"\b(?:what\s+can\s+you\s+do|what\s+are\s+(?:your|adam'?s?)\s+(?:capabilities|abilities|skills|features|tools)|who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+(?:yourself|adam)|how\s+do\s+you\s+(?:work|remember|learn)|list\s+(?:your\s+)?(?:skills|capabilities|tools)|hi\b|hello\b|hey\b|sup\b|yo\b|greetings|good\s+(?:morning|evening|afternoon|night)|thank(?:s|\s+you)|thx\b|how\s+are\s+you|how'?s\s+it\s+going)",re.IGNORECASE)
_NEEDS_FRESH_INFO_RE=re.compile(r"\b(?:weather|forecast|temperature|raining|snowing|sunny|news|headlines|stock\s+price|stocks?|market|exchange\s+rate|crypto|bitcoin|score|game\s+(?:tonight|today|score|result)|sports?|(?:who\s+won|who'?s\s+playing)|election|polls?|current\s+price|today'?s|right\s+now|currently|happening\s+(?:now|today)|recent|latest)\b",re.IGNORECASE)
_CFG=load_config()
def _port_pids(port:int):
    pids=[]
    try:
        if sys.platform=='win32':
            r=subprocess.run(['netstat','-ano','-p','tcp'],capture_output=True,text=True,timeout=5)
            for ln in r.stdout.splitlines():
                if f':{port}' in ln and 'LISTENING' in ln:
                    parts=ln.split()
                    if parts and parts[-1].isdigit():pids.append(int(parts[-1]))
        else:
            r=subprocess.run(['lsof','-ti',f'tcp:{port}'],capture_output=True,text=True,timeout=5)
            for ln in r.stdout.strip().splitlines():
                if ln.strip().isdigit():pids.append(int(ln.strip()))
    except Exception as e:print(f'[amni_serve] _port_pids({port}) probe failed: {e}',flush=True)
    return list(set(pids))
def _proc_is_python(pid:int)->bool:
    try:
        if sys.platform=='win32':
            r=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True,timeout=5)
            return 'python' in (r.stdout or '').lower() or 'uvicorn' in (r.stdout or '').lower()
        r=subprocess.run(['ps','-p',str(pid),'-o','comm='],capture_output=True,text=True,timeout=5)
        return 'python' in (r.stdout or '').lower() or 'uvicorn' in (r.stdout or '').lower()
    except Exception:return False
def _kill_pid(pid:int)->bool:
    try:
        if sys.platform=='win32':
            r=subprocess.run(['taskkill','/F','/PID',str(pid)],capture_output=True,text=True,timeout=5)
            return r.returncode==0
        os.kill(pid,signal.SIGKILL);return True
    except Exception as e:print(f'[amni_serve] kill {pid} failed: {e}',flush=True);return False
def _free_port_if_occupied(host:str,port:int,kill_unsafe:bool=False):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.settimeout(0.5)
    occupied=False
    try:occupied=(s.connect_ex((host,port))==0)
    finally:s.close()
    if not occupied:return
    pids=_port_pids(port)
    if not pids:print(f'[amni_serve] port {port} reports occupied but no PID found — proceeding anyway (may fail to bind)',flush=True);return
    print(f'[amni_serve] port {port} occupied by PID(s) {pids} — checking ownership',flush=True)
    killed=[]
    for pid in pids:
        if pid==os.getpid():print(f'[amni_serve] skipping our own PID {pid}',flush=True);continue
        if not kill_unsafe and not _proc_is_python(pid):print(f'[amni_serve] PID {pid} is NOT a python/uvicorn process — refusing to kill (use --force-port-kill to override)',flush=True);continue
        if _kill_pid(pid):killed.append(pid);print(f'[amni_serve] killed prior process PID {pid} on port {port}',flush=True)
    if killed:
        for _ in range(20):
            time.sleep(0.1);s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.settimeout(0.3)
            try:
                if s.connect_ex((host,port))!=0:break
            finally:s.close()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bake',default=_CFG.get('bake'))
    ap.add_argument('--model',default=_CFG.get('model') or _CFG.get('bake'))
    ap.add_argument('--port',type=int,default=int(_CFG.get('port') or DEFAULT_PORT))
    ap.add_argument('--host',default=_CFG.get('host') or DEFAULT_HOST)
    ap.add_argument('--force-port-kill',action='store_true',help='Kill ANY process holding the port (default: only kill python/uvicorn processes)')
    ap.add_argument('--lessons',default='experiences/adam_lessons.npz')
    ap.add_argument('--lut-root',default='experiences/adam_lut')
    ap.add_argument('--conv-root',default='experiences/conversations')
    ap.add_argument('--audit-log',default='logs/agent_skill_calls.jsonl')
    ap.add_argument('--workdir',default=None,help='Primary skill workdir (defaults to cwd). Use --root for additional roots.')
    ap.add_argument('--root',action='append',default=[],help='Additional allowed root for file_*/code_edit/scan/shell. Repeatable.')
    ap.add_argument('--unrestricted-files',action='store_true',help='Drop workdir gating and allow file ops on ANY drive. Use with care.')
    ap.add_argument('--web-restricted',action='store_true',default=bool(os.environ.get('AMNI_WEB_RESTRICTED')),help='Limit web crawler to the 255-domain trusted-source allowlist (Wikipedia, StackOverflow, .gov, .edu, major dev docs and news). Default is unrestricted — any DDG result is fair game.')
    ap.add_argument('--seed',action='store_true',help='seed lessons with default bank if file missing')
    ap.add_argument('--cors',action='store_true',help='enable permissive CORS for dev')
    ap.add_argument('--persona-bank',default='experiences/personas.json',help='Persona store path (learned personas + per-session map)')
    ap.add_argument('--default-persona',default=None,help='Default persona name (preset or learned). e.g. rikku, yoda, neutral')
    ap.add_argument('--no-persona',action='store_true',help='Disable persona layer entirely (raw Adam responses)')
    args=ap.parse_args()
    if not args.bake or not Path(args.bake).exists() or not (Path(args.bake)/'manifest.json').exists():
        print(f'[amni_serve] FATAL: no usable bake found.\n  Tried: {args.bake!r}\n  Run `python install.py` to fetch the GF(17) bake from Hugging Face, or pass --bake <path-to-bake-with-manifest.json> --model <path-to-model-dir>.\n  Config search order: $AMNI_HOME, ~/.amni-ai/last_install_home.txt pointer, ~/.amni-ai/config.json, then candidate dirs ($AMNI_BAKE_PATHS, CONFIG_DIR/bakes, ./bakes, ~/amni-bakes, ~/.amni-ai/bakes).',flush=True)
        sys.exit(2)
    if not args.model or not Path(args.model).exists() or not (Path(args.model)/'config.json').exists():
        print(f'[amni_serve] FATAL: no usable model dir found.\n  Tried: {args.model!r}\n  The bake itself usually has config.json + tokenizer.json (the runtime can use --model <bake-path> as a fallback). If your bake has these files, pass --model {args.bake!r}.\n  Otherwise run `python install.py` to fetch the upstream model.',flush=True)
        sys.exit(2)
    _free_port_if_occupied(args.host,args.port,kill_unsafe=args.force_port_kill)
    try:
        from fastapi import FastAPI,Request,HTTPException
        from pydantic import BaseModel
        from typing import Optional,List
        import uvicorn
    except ImportError:
        print('[amni_serve] missing fastapi/uvicorn. Install: pip install fastapi uvicorn pydantic',flush=True)
        sys.exit(1)
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore,PersonaStore
    from amni.serve.skills import default_registry
    from amni.serve import ollama_compat,web,mcp,openai_compat,jarvis_web,memory_endpoints,task_endpoints,vision_endpoints,voice_endpoints
    from amni.serve.code_atlas import CodeAtlas
    print(f'[amni_serve] booting Adam with bake={args.bake}',flush=True)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None,web_unrestricted=not args.web_restricted)
    print(f'[amni_serve] Adam ready: {adam.stats()}',flush=True)
    try:
        from amni.serve import intent_classifier as _ic
        _enc=adam.sem_lut._ensure_encoder() if hasattr(adam.sem_lut,'_ensure_encoder') else None
        _intent_clf=_ic.get(embedder=_enc)
        _intent_clf._ensure_embedded()
        print(f'[amni_serve] intent classifier ready (embedder={_enc is not None})',flush=True)
    except Exception as _ie:print(f'[amni_serve] intent classifier init failed: {_ie}',flush=True);_intent_clf=None
    _warmup_state={'done':False,'wall_s':None,'error':None}
    if not os.environ.get('AMNI_NO_WARMUP'):
        import threading as _th
        def _bg_warmup():
            _w0=time.time();print('[amni_serve] background warmup (compiles ROCm/CUDA kernels, ~10-30s)...',flush=True)
            try:
                _ = adam.ask('hi',writeback=False)
                _warmup_state['wall_s']=round(time.time()-_w0,1);_warmup_state['done']=True
                print(f'[amni_serve] warmup done in {_warmup_state["wall_s"]}s — first user request will be fast',flush=True)
            except Exception as _we:
                _warmup_state['error']=str(_we)[:200];_warmup_state['done']=True
                print(f'[amni_serve] warmup failed (non-fatal): {_we}',flush=True)
        _th.Thread(target=_bg_warmup,daemon=True).start()
    else:_warmup_state['done']=True
    skills=default_registry(workdir=args.workdir,roots=args.root,audit_log=args.audit_log,unrestricted=args.unrestricted_files)
    store=ConversationStore(root=args.conv_root)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.default_persona:personas.set_default(args.default_persona)
    agent=AmniAgent(adam=adam,skills=skills,store=store,workdir=args.workdir,personas=personas,use_persona=not args.no_persona)
    print(f'[amni_serve] Persona: default={personas._default} known={[p.name for p in personas.list_known()]}',flush=True)
    scope='UNRESTRICTED (all drives)' if args.unrestricted_files else f'roots={[str(r) for r in skills.roots]}'
    print(f'[amni_serve] Agent ready: skills={[s["name"] for s in agent.list_skills()]} {scope} sessions={len(store.list_sessions())}',flush=True)
    app=FastAPI(title='Amni-Ai Adam',version='6.0.0')
    if args.cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
    class ChatRequest(BaseModel):
        message:str
        session_id:Optional[str]=None
        use_skills:bool=True
        writeback:bool=True
    class AskRequest(BaseModel):
        query:str
        writeback:bool=True
    class TeachRequest(BaseModel):
        question:str
        answer:str
    class SkillRequest(BaseModel):
        args:dict={}
    _iter_counters={'tests_passed':0,'tests_failed':0,'promoted':0,'quality_gated':0,'perturb_attempted':0,'perturb_succeeded_small':0,'perturb_succeeded_medium':0,'perturb_succeeded_large':0,'perturb_failed':0,'intent_blocked':0,'multi_block_stitched':0,'hint_injected':0,'lut_hits':0,'cot_generations':0}
    def _bump(k,n=1):_iter_counters[k]=_iter_counters.get(k,0)+n
    @app.post('/chat')
    def chat(req:ChatRequest):return agent.chat(req.message,session_id=req.session_id,use_skills=req.use_skills,writeback=req.writeback)
    @app.post('/chat/stream')
    async def chat_stream(req:ChatRequest,request:Request):
        from fastapi.responses import StreamingResponse
        import json as _json
        from amni.serve.agent import _needs_cot,_pick_cot
        from amni.serve import tone_atlas
        def gen():
            t0=time.time()
            conv=store.get(req.session_id)
            prior_a='';prior_q=''
            if conv.turns:
                _prior=[t for t in conv.turns if t.get('role') in ('user','assistant')]
                if _prior and _prior[-1].get('role')=='assistant':
                    prior_a=(_prior[-1].get('content') or '')
                    _users_before=[t for t in _prior[:-1] if t.get('role')=='user']
                    prior_q=(_users_before[-1].get('content') or '') if _users_before else ''
            conv.append('user',req.message)
            if is_build_request(req.message):
                _ag_persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
                _ag_persona_name=_ag_persona.name if _ag_persona else 'Adam'
                yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":_ag_persona_name,"agentic":True})}\n\n'
                _final_answer='';_n_steps=0
                try:
                    for _ev in run_goal_stream(agent,skills,req.message,max_steps=8,timeout_s=240):
                        yield f'event: agentic_{_ev["event"]}\ndata: {_json.dumps(_ev)}\n\n'
                        if _ev.get('event')=='final':_final_answer=_ev.get('answer','') or '';_n_steps=_ev.get('n_steps',0)
                except Exception as _ge:yield f'event: error\ndata: {_json.dumps(f"agentic loop failed: {_ge}")}\n\n'
                _stored=_final_answer or '(agentic run produced no final answer; see steps above)'
                conv.append('assistant',_stored,{'tier':'tier_agentic','persona':_ag_persona_name,'category':'agentic','n_steps':_n_steps})
                if getattr(agent,'atlas',None) is not None and _stored.strip():
                    try:agent.atlas.record(conv.session_id,req.message,_stored,is_personal=False)
                    except Exception as _re:print(f'[amni_serve] /chat/stream agentic atlas record failed: {_re}',flush=True)
                yield f'event: done\ndata: {_json.dumps({"tier":"tier_agentic","wall_s":round(time.time()-t0,3),"n_steps":_n_steps})}\n\n'
                return
            if getattr(agent,'profile',None) is not None:
                try:agent.profile.update_from_message(req.message)
                except Exception as _pe:print(f'[amni_serve] /chat/stream profile update failed: {_pe}',flush=True)
            if getattr(agent,'notes',None) is not None and prior_a and ConversationNotes.is_correction(req.message):
                try:agent.notes.add_correction(wrong_q=prior_q,wrong_a=prior_a,corrected_text=req.message,session_id=conv.session_id)
                except Exception as _ce:print(f'[amni_serve] /chat/stream correction capture failed: {_ce}',flush=True)
            _skill_match=agent._detect_skill(req.message) if hasattr(agent,'_detect_skill') else None
            if _skill_match:
                _sname,_sargs=_skill_match
                if _sname=='web' and isinstance(_sargs,dict):
                    _enriched=_enrich_web_query(req.message,conv,getattr(agent,'profile',None))
                    if _enriched and _enriched!=_sargs.get('query'):_sargs={**_sargs,'query':_enriched}
                _sk_persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
                _sk_persona_name=_sk_persona.name if _sk_persona else 'Adam'
                try:
                    _sr=skills.call(_sname,_sargs,ctx={'adam':adam,'conv':conv})
                    if _sr.ok:
                        _formatted=agent._format_skill_output(_sname,_sr.output)
                        _sk_cat=tone_atlas.classify_intent(req.message,skill_used=_sname) if hasattr(tone_atlas,'classify_intent') else 'factual'
                        _wrapped=tone_atlas.wrap(_formatted,_sk_cat,_sk_persona,seed=req.message) if _sk_persona and hasattr(tone_atlas,'wrap') else _formatted
                        yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":_sk_persona_name,"skill":_sname,"category":_sk_cat})}\n\n'
                        for ch in [_wrapped[i:i+48] for i in range(0,len(_wrapped),48)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                        conv.append('assistant',_wrapped,{'tier':f'tier0_skill_{_sname}','persona':_sk_persona_name,'category':_sk_cat,'skill':_sname})
                        yield f'event: done\ndata: {_json.dumps({"tier":f"tier0_skill_{_sname}","wall_s":round(time.time()-t0,3),"persona":_sk_persona_name})}\n\n';return
                except Exception as _se:print(f'[amni_serve] /chat/stream skill {_sname} failed: {_se}',flush=True)
            persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
            persona_name=persona.name if persona else 'Adam'
            yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":persona_name})}\n\n'
            try:
                from amni.a1.semantic_intent import screen as _sem_screen
                _blk,_cat,_cos,_refmsg=_sem_screen(req.message)
                if _blk:
                    _bump('intent_blocked')
                    yield f'event: meta\ndata: {_json.dumps({"blocked":True,"category":_cat,"cos":round(_cos,3)})}\n\n'
                    for ch in [_refmsg[i:i+24] for i in range(0,len(_refmsg),24)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                    conv.append('assistant',_refmsg,{'tier':f'tier_intent_block_{_cat}','blocked':True,'cos':round(_cos,3)})
                    yield f'event: done\ndata: {_json.dumps({"tier":f"tier_intent_block_{_cat}","wall_s":round(time.time()-t0,3),"blocked":True})}\n\n'
                    return
            except Exception:pass
            category=tone_atlas.classify_intent(req.message)
            _intent_label,_intent_conf=(_intent_clf.classify(req.message) if _intent_clf else ('unknown',0.0))
            _profile_authoritative=(_intent_label=='profile_about_me') or bool(_PROFILE_AUTHORITATIVE_RE.search(req.message))
            _memory_recall=(_intent_label=='memory_recall') or bool(_MEMORY_RECALL_RE.search(req.message))
            apply_cot=_needs_cot(category,req.message) and persona and persona.name!='Adam'
            if _profile_authoritative or _memory_recall:apply_cot=False
            from amni.serve.conversation import detect_personal as _dp
            history_pairs=conv.history_pairs(n=12) if len(conv.turns)>1 else []
            _skip_atlas=_profile_authoritative or _memory_recall or (_intent_label=='introspection') or (_intent_label=='math_calc')
            atlas_recall=[] if _skip_atlas else (agent.atlas.recall(req.message,session_id=conv.session_id,k=3,include_global=True) if getattr(agent,'atlas',None) is not None else [])
            for r in atlas_recall:
                pair=(r.get('user',''),r.get('assistant',''))
                if pair[0] and pair[1] and pair not in history_pairs:history_pairs=[pair]+history_pairs
            history_pairs=history_pairs[-12:]
            user_facts=agent._extract_user_facts(conv,extra_user_msgs=[r.get('user','') for r in atlas_recall],profile_only=(_intent_label=='profile_about_me')) if hasattr(agent,'_extract_user_facts') else []
            is_private=_dp(req.message) or conv.has_personal(n=20) or any(r.get('is_personal') for r in atlas_recall)
            sl=getattr(adam,'sem_lut',None)
            _has_correction=False
            _persona_query=bool(persona and persona.name and persona.name.lower() in req.message.lower() and len(req.message.split())>=3)
            _pre_web_supplemented=False
            try:
                _is_fresh=(_intent_label=='needs_fresh_info') or bool(_NEEDS_FRESH_INFO_RE.search(req.message))
                _is_introsp=(_intent_label in ('greeting','introspection')) or bool(_INTROSPECT_NO_WEB_RE.search(req.message))
                if (not is_private) and skills.has('web') and _is_fresh and not _is_introsp:
                    _enriched_pre=_enrich_web_query(req.message,conv,getattr(agent,'profile',None))
                    yield f'event: web_lookup\ndata: {_json.dumps({"trigger":"pre_fetch","query":_enriched_pre[:200],"enriched_from":req.message[:80]})}\n\n'
                    _pre_web_r=skills.call('web',{'query':_enriched_pre},ctx={'adam':adam})
                    if _pre_web_r.ok and _pre_web_r.output:
                        _pw_ans=(_pre_web_r.output.get('answer') or '').strip()
                        _pw_srcs=(_pre_web_r.output.get('sources') or [])[:3]
                        if _pw_ans and len(_pw_ans)>20:
                            user_facts=user_facts+[f'Fresh web research for the user\'s query: {_pw_ans[:600]}. Sources: {", ".join(_pw_srcs[:2])}']
                            _pre_web_supplemented=True
                            yield f'event: web_supplement_done\ndata: {_json.dumps({"chars":len(_pw_ans),"sources_n":len(_pw_srcs),"phase":"pre_fetch"})}\n\n'
            except Exception as _pwe:print(f'[serve] pre-web error: {_pwe}',flush=True)
            try:
                if getattr(agent,'notes',None) is not None:
                    _msg_norm=req.message.strip().lower()
                    for _c in (agent.notes.data.get('corrections') or [])[-20:]:
                        if (_c.get('wrong_q') or '').strip().lower()==_msg_norm:_has_correction=True;break
            except Exception:_has_correction=False
            try:
                eff=sl.auto_margin() if sl and hasattr(sl,'auto_margin') else 0.08
                hit=sl.lookup_soft(req.message,margin=eff) if (sl and hasattr(sl,'lookup_soft') and not history_pairs and not is_private and not _has_correction and not _profile_authoritative and not _memory_recall and not _persona_query) else None
            except Exception:hit=None
            if hit:
                _bump('lut_hits')
                _hit_clean=hit;_ridx=hit.upper().rfind('FINAL:')
                if _ridx>=0:_hit_clean=hit[_ridx+6:].lstrip(' :\t\n').strip()
                for _drift_mk in ('Thinking Process','thought\n','\nThinking','**Self-','*(Self-','**Analyze Request','\n1. RESTATE:','\n1.  RESTATE:','**Recall Persona','Self-Correction'):
                    _di=_hit_clean.find(_drift_mk)
                    if _di>=0:_hit_clean=_hit_clean[:_di].rstrip();break
                if not _hit_clean.strip():_hit_clean=hit
                for ch in [_hit_clean[i:i+48] for i in range(0,len(_hit_clean),48)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                conv.append('assistant',_hit_clean,{'tier':'tier1_5_semantic_lesson','persona':persona_name,'category':category})
                yield f'event: done\ndata: {_json.dumps({"tier":"tier1_5_semantic_lesson","wall_s":round(time.time()-t0,3)})}\n\n';return
            if apply_cot:scaffold=_pick_cot(category,req.message);sys_p=persona.system_prompt(req.message)+'\n\n'+scaffold
            elif persona:sys_p=persona.system_prompt(req.message)
            else:sys_p='You are a helpful assistant.'
            expects_final=apply_cot and isinstance(scaffold,str) and 'FINAL:' in scaffold
            yield f'event: meta\ndata: {_json.dumps({"cot":apply_cot,"category":category,"history_n":len(history_pairs),"facts_n":len(user_facts),"is_private":is_private,"buffering":expects_final})}\n\n'
            max_new=int(80+200*(persona.length if persona else 0.5))+(700 if (apply_cot and category=="code") else (450 if apply_cot else 0))
            full=[];_bump('cot_generations') if apply_cot else None
            in_final=not expects_final;buf='';seen_final=False;_buf_start=time.time();_last_ping=time.time();_drift_stop=False;_final_buf=''
            _DRIFT_MARKERS=('Thinking Process','thought\n','\nThinking','**Self-','*(Self-','**Analyze Request','1. RESTATE:','1.  RESTATE:','1. **Analyze','**Recall Persona','**Determine Strategy','Self-Correction','\n[Looked','[Looked','[Search performed','[Search completed','[Search done','[Search results','[Presenting','[Current weather data','[Result of search','[The system returns','(Outputting the result','(Search returns','(Result of search','(Waiting for search','(Assuming the search')
            try:
                for chunk in adam.chat_persona_stream(req.message,system=sys_p,history=history_pairs,facts=user_facts,is_private=is_private,max_new_tokens=max_new,do_sample=True):
                    full.append(chunk)
                    if in_final:
                        _final_buf+=chunk
                        _drift_idx=-1
                        for _mk in _DRIFT_MARKERS:
                            _di=_final_buf.find(_mk)
                            if _di>=0 and (_drift_idx<0 or _di<_drift_idx):_drift_idx=_di
                        if _drift_idx>=0:
                            _tail=_final_buf[:_drift_idx]
                            _already_emitted=len(_final_buf)-len(chunk)
                            _new_safe=_tail[_already_emitted:] if _already_emitted<len(_tail) else ''
                            if _new_safe:yield f'event: token\ndata: {_json.dumps(_new_safe)}\n\n'
                            _drift_stop=True;break
                        else:yield f'event: token\ndata: {_json.dumps(chunk)}\n\n'
                    else:
                        buf+=chunk;_now=time.time()
                        idx=buf.upper().find('FINAL:')
                        if idx>=0:
                            after=buf[idx+6:].lstrip(' :\t\n')
                            if after:yield f'event: token\ndata: {_json.dumps(after)}\n\n';_final_buf=after
                            in_final=True;seen_final=True;buf=''
                        elif (_now-_last_ping)>3:yield f'event: thinking\ndata: {_json.dumps({"buf_chars":len(buf),"elapsed":round(_now-_buf_start,1)})}\n\n';_last_ping=_now
            except Exception as e:yield f'event: error\ndata: {_json.dumps(str(e))}\n\n';return
            if not in_final and buf.strip():
                _b=buf;_LAST_SCAFFOLD_MARKERS=('\nFINAL:','\nFinal:','\nfinal:','\n6. FINAL','\n5. FINAL','\nREFINE','\nMEDIUM:','\nLARGE:','\nSMALL:','\nCRITIQUE:','\nFIRST SHOT:','\nKNOWNS','\nRESTATE','\nAPPROACH','\nCLARIFY','\nCODE:','\nTESTS:','\nNOTES:')
                _last_idx=-1
                for _mk in _LAST_SCAFFOLD_MARKERS:
                    _idx=_b.upper().rfind(_mk.upper())
                    if _idx>_last_idx:_last_idx=_idx
                if _last_idx>=0:
                    _line_end=_b.find('\n',_last_idx+1)
                    if _line_end>=0:
                        _after=_b[_line_end:].lstrip(' :\t\n')
                        if _after and len(_after)>10:_b=_after
                yield f'event: token\ndata: {_json.dumps(_b)}\n\n'
            raw_final=_strip_tool_announce(''.join(full).strip())
            if expects_final and seen_final:
                _ridx=raw_final.upper().rfind('FINAL:')
                final=raw_final[_ridx+6:].lstrip(' :\t\n').strip() if _ridx>=0 else raw_final
            else:final=raw_final
            tier_final='tier_persona_cot' if apply_cot else 'tier_persona'
            if apply_cot and category=='code' and final and skills.has('run_python'):
                from amni.serve.agent import _extract_python_blocks,_validate_python,_perturb_retry,_extract_asserts,_run_with_tests
                blocks=_extract_python_blocks(final)
                if blocks:
                    bad=_validate_python(blocks)
                    if bad:yield f'event: validate\ndata: {_json.dumps({"syntax_errors":len(bad)})}\n\n'
                    runnable=[b for b in blocks if ('print(' in b or 'if __name__' in b)]
                    if runnable:
                        snippet=('\n\n'.join(blocks) if len(blocks)>1 else runnable[-1])
                        if len(blocks)>1:
                            _bump('multi_block_stitched')
                            yield f'event: multi_block\ndata: {_json.dumps({"blocks":len(blocks),"runnable":len(runnable),"stitched_chars":len(snippet)})}\n\n'
                        try:
                            run_r=skills.call('run_python',{'code':snippet,'timeout':8},ctx={'adam':adam})
                            if run_r.ok and not run_r.output.get('error'):
                                so=(run_r.output.get('stdout') or '').strip()
                                se=(run_r.output.get('stderr') or '').strip()
                                rc=run_r.output.get('returncode')
                                yield f'event: exec\ndata: {_json.dumps({"stdout":so[:1500],"stderr":se[:600],"returncode":rc,"timed_out":run_r.output.get("timed_out",False)})}\n\n'
                                exec_md=f'\n\n**[Sandbox execution — exit {rc}{"  (timed out)" if run_r.output.get("timed_out") else ""}]**\n'
                                if so:exec_md+=f'```\n{so[:1500]}\n```\n'
                                if se:exec_md+=f'_stderr:_\n```\n{se[:600]}\n```'
                                final+=exec_md
                                tier_final+='_run'
                                test_failed=False;test_err=''
                                if rc==0 and not se:
                                    asserts=_extract_asserts(final)
                                    if asserts:
                                        from amni.serve.agent import _assert_diversity
                                        div_score,div_info=_assert_diversity(asserts)
                                        passed,terr,tinfo=_run_with_tests(skills,adam,snippet,asserts)
                                        yield f'event: test_run\ndata: {_json.dumps({"asserts_n":len(asserts),"passed":passed,"diversity":round(div_score,2),"info":tinfo,"div":div_info})}\n\n'
                                        div_tag='_tests_thin' if div_score<0.5 else ('_tests_diverse' if div_score>=0.75 else '_tests_ok')
                                        if passed:
                                            _bump('tests_passed')
                                            final+=f'\n\n**[Self-tests — {len(asserts)}/{len(asserts)} passed · diversity={div_score:.2f}]**'
                                            tier_final+=div_tag
                                            from amni.serve.agent import _should_promote
                                            ok_promote,reason=_should_promote(snippet,asserts,div_score)
                                            if ok_promote:
                                                try:
                                                    promo_ans=final[:2000]
                                                    tr=adam.teach(req.message,promo_ans)
                                                    tier_final+='_promoted'
                                                    _bump('promoted')
                                                    yield f'event: promoted\ndata: {_json.dumps({"lessons_n":tr.get("lessons_n",0),"reason":reason})}\n\n'
                                                except Exception as _pe:yield f'event: promoted\ndata: {_json.dumps({"error":str(_pe)[:120]})}\n\n'
                                            else:
                                                tier_final+='_quality_gated'
                                                _bump('quality_gated')
                                                yield f'event: promoted\ndata: {_json.dumps({"gated":True,"reason":reason})}\n\n'
                                        else:
                                            _bump('tests_failed')
                                            final+=f'\n\n**[Self-tests FAILED — {terr[:200]}]**'
                                            test_failed=True;test_err=terr
                                if rc!=0 or se or test_failed:
                                    _bump('perturb_attempted')
                                    perturb_events=[]
                                    emit_fn=lambda d:perturb_events.append(d)
                                    err_signal=test_err if test_failed else (se or f'exit code {rc}')
                                    perturb_asserts=_extract_asserts(final) if test_failed else None
                                    from amni.serve.agent import _error_hint
                                    if _error_hint(err_signal):_bump('hint_injected')
                                    pr=_perturb_retry(adam,skills,sys_p,snippet,err_signal,req.message,max_steps=3,emit=emit_fn,asserts=perturb_asserts)
                                    for ev in perturb_events:yield f'event: perturb\ndata: {_json.dumps(ev)}\n\n'
                                    if pr.get('success'):
                                        _bump(f'perturb_succeeded_{pr["magnitude"].lower()}')
                                        final+=f'\n\n**[Trial-and-error fixed it — {pr["magnitude"]} perturbation]**\n```python\n{pr["code"]}\n```\n```\n{pr["stdout"][:1500]}\n```'
                                        tier_final+=f'_perturb_{pr["magnitude"].lower()}'
                                        yield f'event: perturb\ndata: {_json.dumps({"final":True,"magnitude":pr["magnitude"],"success":True})}\n\n'
                                    else:
                                        _bump('perturb_failed')
                                        final+=f'\n\n**[Trial-and-error exhausted SMALL/MEDIUM/LARGE — code still failing]**'
                                        tier_final+='_perturb_failed'
                                        yield f'event: perturb\ndata: {_json.dumps({"final":True,"success":False,"steps":len(pr.get("history",[]))})}\n\n'
                            elif run_r.ok and run_r.output.get('error'):
                                yield f'event: exec\ndata: {_json.dumps({"error":run_r.output["error"]})}\n\n'
                        except Exception as e:yield f'event: exec\ndata: {_json.dumps({"error":str(e)})}\n\n'
            _introspect_no_web=(_intent_label in ('greeting','introspection')) or bool(_INTROSPECT_NO_WEB_RE.search(req.message))
            _needs_fresh_info=(_intent_label=='needs_fresh_info') or bool(_NEEDS_FRESH_INFO_RE.search(req.message))
            if (not is_private) and skills.has('web') and not _memory_recall and not _profile_authoritative and not _introspect_no_web and not _pre_web_supplemented and (raw_final or final) and _UNCERTAIN_RE.search(raw_final or final or ''):
                _enriched_q=_enrich_web_query(req.message,conv,getattr(agent,'profile',None))
                if len(_enriched_q.split())<4:
                    yield f'event: web_supplement_skipped\ndata: {_json.dumps({"reason":"query too vague even after enrichment","query":_enriched_q})}\n\n'
                else:
                    yield f'event: web_lookup\ndata: {_json.dumps({"trigger":"uncertainty","query":_enriched_q[:200],"enriched_from":req.message[:80]})}\n\n'
                    try:
                        _web_r=skills.call('web',{'query':_enriched_q},ctx={'adam':adam})
                        _web_ans=(_web_r.output or {}).get('answer','') if _web_r.ok else ''
                        _web_srcs=(_web_r.output or {}).get('sources',[]) if _web_r.ok else []
                        _has_real_content=bool(_web_ans and _web_ans.strip() and len(_web_ans.strip())>20)
                        if _web_r.ok and _has_real_content:
                            _web_pretty=agent._format_skill_output('web',_web_r.output) if hasattr(agent,'_format_skill_output') else str(_web_r.output)
                            _web_pretty=_web_pretty[:1200]
                            _supplement=f'\n\n[Looked it up]\n{_web_pretty}'
                            for _ch in [_supplement[i:i+48] for i in range(0,len(_supplement),48)]:yield f'event: token\ndata: {_json.dumps(_ch)}\n\n'
                            final=(final or '')+_supplement
                            yield f'event: web_supplement_done\ndata: {_json.dumps({"chars":len(_web_pretty),"sources_n":len(_web_srcs)})}\n\n'
                        else:yield f'event: web_supplement_skipped\ndata: {_json.dumps({"reason":"no useful content","ans_len":len(_web_ans or ""),"sources_n":len(_web_srcs or [])})}\n\n'
                    except Exception as _we:yield f'event: web_supplement_error\ndata: {_json.dumps({"msg":str(_we)[:200]})}\n\n'
            conv.append('assistant',final,{'tier':tier_final,'persona':persona_name,'category':category,'is_private':is_private})
            if getattr(agent,'atlas',None) is not None and final and final.strip():
                try:agent.atlas.record(conv.session_id,req.message,final,is_personal=is_private)
                except Exception as _re:print(f'[amni_serve] /chat/stream atlas record failed: {_re}',flush=True)
            if (not is_private) and (not history_pairs) and final and len(final.strip())>=20 and category not in ('greeting','introspect') and 'block' not in tier_final and '_promoted' not in tier_final:
                try:adam.teach(req.message,final)
                except Exception as _te:print(f'[amni_serve] /chat/stream lesson writeback failed: {_te}',flush=True)
            yield f'event: done\ndata: {_json.dumps({"tier":tier_final,"wall_s":round(time.time()-t0,3),"persona":persona_name,"category":category})}\n\n'
        return StreamingResponse(gen(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
    @app.post('/ask')
    def ask(req:AskRequest):return adam.ask(req.query,writeback=req.writeback)
    @app.post('/teach')
    def teach(req:TeachRequest):return adam.teach(req.question,req.answer)
    class CompleteReq(BaseModel):
        prefix:str
        suffix:Optional[str]=''
        language:Optional[str]=None
        max_tokens:int=40
        stop:Optional[List[str]]=None
    @app.post('/complete')
    def complete(req:CompleteReq):
        if not req.prefix:raise HTTPException(status_code=400,detail='missing prefix')
        _lang=(req.language or '').lower()
        _hint='completing code' if any(c in req.prefix for c in ('{','(','def ','fn ','function ','class ','import ','use ','#include')) else 'completing text'
        sys_p=f'You are a code/text completion engine. Continue the user\'s text NATURALLY. Output ONLY the continuation — no explanation, no markdown fence, no preamble. Stop at a natural boundary (end of statement/expression/sentence). Target language: {_lang or "auto-detect"}.'
        _tail=(req.suffix or '')[:200]
        _ctx=req.prefix[-1500:]
        prompt=f'<task>{_hint}</task>\n<prefix>{_ctx}</prefix>'+(f'\n<suffix>{_tail}</suffix>' if _tail else '')+'\n<continue>'
        try:
            svc=getattr(adam,'svc',None)
            if svc is None:raise RuntimeError('no inference svc')
            resp,n=svc.chat(prompt,system=sys_p,max_new_tokens=int(req.max_tokens),do_sample=False,kb_top_k=0)
            text=(resp or '').strip()
            text=re.sub(r'^```\w*\s*\n?','',text);text=re.sub(r'\n?```\s*$','',text)
            if text.startswith('<continue>'):text=text[10:]
            for stop in (req.stop or ['</continue>','</task>','\n\n\n']):
                _i=text.find(stop)
                if _i>=0:text=text[:_i]
            return {'completion':text.rstrip(),'tokens':n,'lang_hint':_lang or None}
        except Exception as e:raise HTTPException(status_code=500,detail=f'completion failed: {e}')
    class VoiceChatReq(BaseModel):
        audio_base64:Optional[str]=None
        text:Optional[str]=None
        session_id:Optional[str]=None
        return_audio:bool=True
        model_size:Optional[str]='base'
        voice:Optional[str]=None
    @app.post('/voice/chat')
    def voice_chat(req:VoiceChatReq):
        try:
            from amni.voice import transcribe,speak,tts_backend,stt_backend
            import base64 as _b64
        except Exception as e:raise HTTPException(status_code=500,detail=f'voice module import failed: {e}')
        result={'stt_backend':stt_backend(),'tts_backend':tts_backend()}
        user_text=req.text
        if not user_text and req.audio_base64:
            try:audio=_b64.b64decode(req.audio_base64)
            except Exception as e:raise HTTPException(status_code=400,detail=f'audio_base64 decode failed: {e}')
            tr=transcribe(audio,model_size=req.model_size or 'base')
            if 'error' in tr:result['stt_error']=tr['error'];return result
            user_text=tr.get('text','')
            result['transcript']=user_text
            result['transcript_lang']=tr.get('language')
        if not user_text:raise HTTPException(status_code=400,detail='no text or audio_base64 provided')
        try:
            cr=agent.chat(user_text,session_id=req.session_id,use_skills=True,writeback=True)
            result['response']=cr.get('answer','')
            result['tier']=cr.get('tier','')
            result['session_id']=cr.get('session_id')
            result['persona']=cr.get('persona')
            result['wall_s']=cr.get('wall_s')
        except Exception as e:raise HTTPException(status_code=500,detail=f'chat failed: {e}')
        if req.return_audio and result['response']:
            try:
                _raw=result['response'];_spk=_raw
                _persona_voice=getattr(persona,'tts_voice','ryan') if 'persona' in dir() and persona else 'ryan'
                try:
                    _cur_persona=personas.get(personas.session_persona(req.session_id))
                    _persona_voice=getattr(_cur_persona,'tts_voice',_persona_voice)
                except Exception:pass
                _ridx=_raw.upper().rfind('FINAL:')
                if _ridx>=0:_spk=_raw[_ridx+6:].lstrip(' :\t\n')
                _DRIFT_VOICE=('Thinking Process','thought\n','\nThinking','**Self-','*(Self-','**Analyze Request','\n1. RESTATE:','\n1.  RESTATE:','**Recall Persona','Self-Correction')
                for _mk in _DRIFT_VOICE:
                    _di=_spk.find(_mk)
                    if _di>=0:_spk=_spk[:_di].rstrip();break
                _spk=re.sub(r'^\s*\d+\.\s*(?:RESTATE|KNOWNS\s*/\s*APPROACH|KNOWNS|APPROACH|FIRST\s*SHOT|CRITIQUE|REFINE[^:\n]*|FINAL|CLARIFY|CODE|TESTS|NOTES|COMPLEXITY|SYMPTOMS|HYPOTHESES|EVIDENCE\s*TEST|NEXT\s*STEP|REQUIREMENTS|COMPONENTS|SCALE\s*\+?\s*FAILURE|ALTERNATIVES|FRAME|CHAIN|COUNTER|RELEVANT|SOLVE|VERIFY)\s*:[^\n]*\n?','',_spk,flags=re.MULTILINE|re.IGNORECASE)
                _spk=_spk.strip()
                if not _spk or len(_spk)<5:_spk=_raw.strip()[:600]
                result['spoken_text']=_spk[:800]
                _persona_key=None
                try:_persona_key=(_cur_persona.name or '').lower() if _cur_persona else None
                except Exception:pass
                audio=speak(_spk[:800],voice=(req.voice or _persona_voice),persona=_persona_key)
                result['voice_used']=(req.voice or _persona_voice)
                if audio:result['audio_base64']=_b64.b64encode(audio).decode('ascii');result['audio_bytes']=len(audio)
                else:result['tts_error']='no audio produced'
            except Exception as e:result['tts_error']=str(e)[:200]
        return result
    @app.get('/stats')
    def stats():
        base=agent.stats()
        base['iter_counters']=dict(_iter_counters)
        total_perturb=_iter_counters['perturb_succeeded_small']+_iter_counters['perturb_succeeded_medium']+_iter_counters['perturb_succeeded_large']
        attempted=_iter_counters['perturb_attempted'] or 1
        promoted=_iter_counters['promoted'] or 0
        gated=_iter_counters['quality_gated'] or 0
        tests_total=_iter_counters['tests_passed']+_iter_counters['tests_failed']
        base['iter_rates']={'perturb_success_rate':round(total_perturb/attempted,3),'quality_gate_fire_rate':round(gated/max(promoted+gated,1),3),'tests_pass_rate':round(_iter_counters['tests_passed']/max(tests_total,1),3),'hint_inject_rate':round(_iter_counters['hint_injected']/max(_iter_counters['perturb_attempted'],1),3)}
        return base
    @app.get('/stats/iter')
    def stats_iter():return dict(_iter_counters)
    @app.post('/stats/iter/reset')
    def stats_iter_reset():
        for k in _iter_counters:_iter_counters[k]=0
        return {'reset':True,'keys':list(_iter_counters.keys())}
    from fastapi.responses import HTMLResponse,FileResponse
    _HUD_PATH=Path(__file__).resolve().parent.parent/'docs'/'hud'/'index.html'
    @app.get('/',response_class=HTMLResponse)
    def root_hud():
        if _HUD_PATH.exists():return HTMLResponse(_HUD_PATH.read_text(encoding='utf-8'))
        return HTMLResponse(f'<html><body style="font-family:system-ui;padding:40px;background:#0a0a14;color:#e2e8f0"><h1>Adam</h1><p>HUD file not found at <code>{_HUD_PATH}</code>.</p><p>Endpoints: <a href="/health" style="color:#00d4ff">/health</a> · <a href="/skills" style="color:#00d4ff">/skills</a> · <a href="/sessions" style="color:#00d4ff">/sessions</a></p></body></html>',status_code=200)
    @app.get('/healthz')
    def health():return {'status':'ok','lessons_n':len(adam.sem_lut._raw),'skills_n':len(skills.list_skills()),'version':'6.9.3','warmup':_warmup_state}
    @app.get('/warmup')
    def warmup_status():return {'warmup':_warmup_state}
    class IntentReq(BaseModel):
        text:str
    @app.post('/intent')
    def intent_probe(req:IntentReq):
        if _intent_clf is None:return {'error':'intent classifier not initialized'}
        label,conf=_intent_clf.classify(req.text)
        return {'text':req.text,'label':label,'confidence':round(conf,3),'embedder_loaded':_intent_clf.embedder is not None}
    class ProfileReq(BaseModel):
        prompt:Optional[str]='Write a haiku about Rust.'
        max_new_tokens:int=80
        runs:int=1
    @app.post('/profile/inference')
    def profile_inference(req:ProfileReq):
        try:
            svc=getattr(adam,'svc',None)
            if svc is None:raise HTTPException(status_code=500,detail='no inference svc')
            results=[];total_tokens=0;total_wall=0.0
            for i in range(max(1,req.runs)):
                _t=time.time()
                resp,n=svc.chat(req.prompt,system='Be concise.',max_new_tokens=int(req.max_new_tokens),do_sample=False,kb_top_k=0)
                _w=time.time()-_t
                total_tokens+=n;total_wall+=_w
                results.append({'run':i+1,'tokens':n,'wall_s':round(_w,2),'tok_per_sec':round(n/_w,1) if _w>0 else 0,'preview':(resp or '')[:80]})
            return {'runs':len(results),'total_tokens':total_tokens,'total_wall_s':round(total_wall,2),'avg_tok_per_sec':round(total_tokens/total_wall,1) if total_wall>0 else 0,'per_run':results,'prompt':req.prompt,'max_new_tokens':req.max_new_tokens}
        except Exception as e:raise HTTPException(status_code=500,detail=f'profile failed: {e}')
    @app.get('/health')
    def health_full():
        try:from amni.voice import tts_backend as _tb,stt_backend as _sb,available_wake_words as _ww
        except Exception:_tb=_sb=lambda:'unavailable';_ww=lambda:[]
        _gpu={}
        try:
            import torch
            if torch.cuda.is_available():
                _gpu={'cuda_or_rocm':True,'device_count':torch.cuda.device_count(),'device_name':torch.cuda.get_device_name(0),'mem_alloc_gb':round(torch.cuda.memory_allocated(0)/(1024**3),2),'mem_reserved_gb':round(torch.cuda.memory_reserved(0)/(1024**3),2)}
            else:_gpu={'cuda_or_rocm':False}
        except Exception as e:_gpu={'error':str(e)[:200]}
        return {'status':'ok','version':'6.9.3','adam':{'lessons_n':len(adam.sem_lut._raw),'sessions_n':len(store._active) if hasattr(store,'_active') else 0,'svc_boot_s':round(adam.stats().get('svc_boot_s',0),1)},'skills':{'count':len(skills.list_skills()),'names':sorted(s['name'] for s in skills.list_skills())},'voice':{'tts_backend':_tb(),'stt_backend':_sb(),'wake_words_available':_ww()},'gpu':_gpu,'workdir':str(skills.workdir),'personas_known':[p.name for p in personas.list_known()]}
    @app.get('/skills')
    def list_skills():return {'skills':skills.list_skills()}
    @app.post('/skills/{name}')
    def call_skill(name:str,req:SkillRequest):
        r=skills.call(name,req.args,ctx={'adam':adam,'agent':agent,'personas':personas,'store':store})
        if not r.ok:raise HTTPException(status_code=400,detail=r.to_dict())
        return r.to_dict()
    @app.get('/sessions')
    def sessions(enrich:bool=True,limit:int=30):
        raw=store.list_sessions()
        if not enrich:return {'sessions':raw}
        out=[]
        import json as _j
        for s in raw[:limit]:
            sid=s.get('session_id');mtime=float(s.get('mtime',0))
            entry={'session_id':sid,'updated_ts':mtime,'size':s.get('size',0),'turns_n':0,'last_user_msg':'','first_msg':''}
            try:
                fp=Path(store.root)/f'{sid}.jsonl'
                if fp.exists():
                    lines=fp.read_text(encoding='utf-8',errors='replace').strip().splitlines()
                    entry['turns_n']=len(lines)
                    for ln in lines:
                        try:
                            t=_j.loads(ln)
                            if t.get('role')=='user':
                                if not entry['first_msg']:entry['first_msg']=(t.get('content') or '')[:80]
                                entry['last_user_msg']=(t.get('content') or '')[:80]
                        except Exception:continue
            except Exception:pass
            out.append(entry)
        return {'sessions':out}
    @app.get('/sessions/{sid}')
    def get_session(sid:str,limit:int=200):
        import json as _j
        fp=Path(store.root)/f'{sid}.jsonl'
        if not fp.exists():raise HTTPException(status_code=404,detail=f'session {sid!r} not found')
        try:
            lines=fp.read_text(encoding='utf-8').strip().splitlines()[-limit:]
            turns=[]
            for ln in lines:
                try:turns.append(_j.loads(ln))
                except Exception:pass
            return {'session_id':sid,'turns_n':len(turns),'turns':turns,'path':str(fp)}
        except Exception as e:raise HTTPException(status_code=500,detail=f'read failed: {e}')
    @app.delete('/sessions/{sid}')
    def del_session(sid:str):return {'deleted':store.delete(sid)}
    @app.get('/lessons')
    def lessons(q:str='',offset:int=0,limit:int=50):
        sl=getattr(adam,'sem_lut',None)
        raw=getattr(sl,'_raw',[]) if sl is not None else []
        items=[(i,k,v) for i,(k,v) in enumerate(raw)]
        if q:items=[(i,k,v) for i,k,v in items if q.lower() in k.lower() or q.lower() in v.lower()]
        total=len(items)
        page=items[offset:offset+limit]
        return {'total':total,'lessons_n':len(raw),'offset':offset,'limit':limit,'items':[{'idx':i,'q':k[:300],'a':v[:600]} for i,k,v in page]}
    class PersonaReq(BaseModel):
        name:Optional[str]=None
        persona:Optional[str]=None
        session_id:Optional[str]=None
        description:Optional[str]=None
        learn_via_web:bool=True
    @app.get('/personas')
    def list_personas():return {'default':personas._default,'known':[p.to_dict() for p in personas.list_known()]}
    @app.get('/persona/{name}')
    def get_persona(name:str):
        if not personas.has(name):
            p=personas.learn(name)
            return {'persona':p.to_dict(),'learned_now':True}
        return {'persona':personas.get(name).to_dict(),'learned_now':False}
    @app.post('/persona')
    def set_persona(req:PersonaReq):
        _pname=req.name or req.persona
        if not _pname:raise HTTPException(status_code=400,detail='must provide "name" or "persona" field')
        if not personas.has(_pname):
            if not req.learn_via_web and not req.description:raise HTTPException(status_code=404,detail=f'unknown persona "{_pname}" — pass description= or learn_via_web=true')
            p=personas.learn(_pname,user_description=req.description)
        else:p=personas.get(_pname)
        if req.session_id:personas.assign_session(req.session_id,_pname)
        else:personas.set_default(_pname)
        return {'persona':p.to_dict(),'session_id':req.session_id,'made_default':not req.session_id}
    class ReflectReq(BaseModel):
        max_n:int=3
        min_age_sec:int=0
    @app.post('/reflect')
    def trigger_reflect(req:ReflectReq):
        try:from amni.serve.reflection import reflect_once
        except Exception as e:raise HTTPException(status_code=500,detail=f'reflect import failed: {e}')
        return reflect_once(adam,max_n=req.max_n,min_age_sec=req.min_age_sec)
    @app.get('/project')
    def project_info():
        import os as _os
        root=_os.environ.get('AMNI_PROJECT_ROOT') or args.workdir or _os.getcwd()
        ptype=_os.environ.get('AMNI_PROJECT_TYPE','unknown')
        code_mode=_os.environ.get('AMNI_CODE_MODE')=='1'
        return {'root':root,'type':ptype,'code_mode':code_mode,'cwd':_os.getcwd()}
    @app.get('/project/tree')
    def project_tree(path:str='',depth:int=2,limit:int=200):
        import os as _os
        root=_os.environ.get('AMNI_PROJECT_ROOT') or args.workdir or _os.getcwd()
        target=Path(root)/path if path else Path(root)
        target=target.resolve()
        try:
            common=_os.path.commonpath([str(target),str(Path(root).resolve())])
            if common!=str(Path(root).resolve()):raise HTTPException(status_code=400,detail='outside project root')
        except Exception:raise HTTPException(status_code=400,detail='invalid path')
        if not target.exists():raise HTTPException(status_code=404,detail='not found')
        skip={'.git','node_modules','__pycache__','.venv','venv','.pytest_cache','dist','build','.next','target','.idea','.vscode'}
        items=[]
        def walk(p:Path,d:int):
            if d>depth or len(items)>=limit:return
            try:
                for e in sorted(p.iterdir(),key=lambda x:(not x.is_dir(),x.name.lower())):
                    if e.name.startswith('.') and e.name not in ('.gitignore','.env.example'):continue
                    if e.name in skip:continue
                    rel=str(e.relative_to(Path(root))).replace('\\','/')
                    items.append({'path':rel,'name':e.name,'is_dir':e.is_dir(),'size':e.stat().st_size if e.is_file() else 0,'depth':d})
                    if e.is_dir() and d<depth:walk(e,d+1)
                    if len(items)>=limit:break
            except PermissionError:pass
        walk(target,0)
        return {'root':str(Path(root)),'cwd':str(target),'items':items[:limit],'truncated':len(items)>=limit}
    @app.delete('/lessons/{idx}')
    def del_lesson(idx:int):
        sl=getattr(adam,'sem_lut',None)
        if sl is None or not hasattr(sl,'_raw') or idx<0 or idx>=len(sl._raw):raise HTTPException(status_code=404,detail='lesson idx out of range')
        removed=sl._raw.pop(idx)
        try:sl.fit()
        except Exception:pass
        try:adam.save_lessons()
        except Exception:pass
        return {'deleted':{'q':removed[0][:200],'a':removed[1][:200]},'lessons_n':len(sl._raw)}
    try:_code_atlas=CodeAtlas(root=str(Path('experiences')/'code_atlas'),encoder=getattr(getattr(adam,'sem_lut',None),'encoder',None))
    except Exception as _ce:print(f'[amni_serve] CodeAtlas init failed (autonomy memory disabled): {_ce}',flush=True);_code_atlas=None
    ollama_compat.mount(app,agent)
    openai_compat.mount(app,adam,agent,code_atlas=_code_atlas)
    mcp.mount(app,agent)
    web.mount(app)
    jarvis_web.mount(app)
    memory_endpoints.mount(app,agent)
    task_endpoints.mount(app,agent)
    vision_endpoints.mount(app,agent)
    voice_endpoints.mount(app,agent)
    print(f'[amni_serve] serving on http://{args.host}:{args.port}',flush=True)
    print(f'[amni_serve]   browser UI:    http://{args.host}:{args.port}/',flush=True)
    print(f'[amni_serve]   Jarvis UI:     http://{args.host}:{args.port}/jarvis  (neon + widgets + voice)',flush=True)
    print(f'[amni_serve]   HUD dashboard: http://{args.host}:{args.port}/hud',flush=True)
    print(f'[amni_serve]   Ollama compat: http://{args.host}:{args.port}/api/tags',flush=True)
    print(f'[amni_serve]   MCP server:    http://{args.host}:{args.port}/mcp',flush=True)
    print(f'[amni_serve]   OpenAI compat: http://{args.host}:{args.port}/v1/chat/completions  (Amni-Code, Continue.dev, Cline, Aider)',flush=True)
    print(f'[amni_serve]   Personas:      http://{args.host}:{args.port}/personas',flush=True)
    uvicorn.run(app,host=args.host,port=args.port,log_level='info')
if __name__=='__main__':main()
