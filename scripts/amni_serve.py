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
_cpu_cap=os.environ.get('AMNI_CPU_THREADS') or str(max(4,(os.cpu_count() or 8)//2))
os.environ['AMNI_CPU_THREADS']=_cpu_cap
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(_v,_cpu_cap)
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
    for k,v in (('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL','1'),('HIP_FORCE_DEV_KERNARG','1'),('GPU_MAX_HW_QUEUES','8'),('MIOPEN_FIND_MODE','2'),('MIOPEN_FIND_ENFORCE','NONE')):os.environ.setdefault(k,v)
    _mio=os.path.join(os.path.expanduser('~'),'.miopen');_tri=os.path.join(os.path.expanduser('~'),'.triton')
    for _p in (_mio,_tri):
        try:os.makedirs(_p,exist_ok=True)
        except Exception:pass
    for k,v in (('MIOPEN_USER_DB_PATH',_mio),('MIOPEN_CUSTOM_CACHE_DIR',_mio),('TRITON_CACHE_DIR',_tri)):os.environ.setdefault(k,v)
    print(f'[amni_serve] kernel caches: MIOPEN_USER_DB_PATH={_mio} TRITON_CACHE_DIR={_tri} (persistent — compiles amortize across runs)',flush=True)
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
from amni import APP_VERSION
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
_RESEARCH_WEB_RE=re.compile(r"\b(?:find|get|look\s*up|search\s+for|pull\s*up|show\s+me|are\s+there\s+any|any)\b[^.?!]{0,40}\b(?:sources?|citations?|references?|proof|evidence|articles?|studies|papers?)\b|\b(?:confirm|verify|fact[-\s]?check|double[-\s]?check)\s+(?:this|that|it|the\b)|\bsources?\s+(?:for|confirming|that\s+confirm|to\s+(?:confirm|back))\b",re.IGNORECASE)
_REF_WEB_RE=re.compile(r"\b(?:this|that|it|the\s+above|the\s+same|same\s+thing|earlier|confirm(?:ing|s)?|verify(?:ing)?)\b",re.IGNORECASE)
def _enrich_web_query(user_msg,conv,profile):
    q=(user_msg or '').strip()
    if not q:return q
    q=re.sub(r'^(?:use\s+|please\s+)?(?:web|search|google)(?:\s+skill)?\s*[:]\s*','',q,flags=re.IGNORECASE)
    needs_context=len(q.split())<6 or bool(_VAGUE_WEB_RE.search(q)) or bool(_LOCAL_INTENT_RE.search(q)) or bool(_REF_WEB_RE.search(q))
    if not needs_context:return q[:200]
    extras=[]
    try:
        for t in reversed(conv.turns[-6:]) if conv and conv.turns else []:
            if t.get('role')=='assistant':
                content=t.get('content') or ''
                for nm in _PROPER_NOUN_RE.findall(content)[:5]:
                    if nm.lower() not in q.lower() and nm.lower() not in ('I','You','Rao','Fryd','Oui','Oac','Anthony') and nm not in extras:extras.append(nm)
                if extras:break
    except Exception:pass
    if profile is not None:
        loc=(profile.data.get('location') or '') if hasattr(profile,'data') else ''
        if loc and loc.lower() not in q.lower() and loc not in extras:extras.append(loc)
    if _VAGUE_WEB_RE.search(q) and not any(k in q.lower() for k in ('news','event')):extras.insert(0,'news current events')
    enriched=(' '.join(extras)+' '+q).strip() if extras else q
    return enriched[:200]
_MISSION_START_RE=re.compile(r"^\s*(?:hey\s+adam[,\s]*)?(?:i\s+want\s+you\s+to|i'?d\s+like\s+you\s+to|i\s+need\s+you\s+to|go|please|can\s+you|could\s+you|would\s+you)\s+(?:go\s+)?(?:and\s+)?(?:learn|become|master|get\s+(?:much\s+)?better\s+at|study\s+up\s+on|train(?:\s+yourself)?)\b|^\s*(?:become|master|get\s+better\s+at|study\s+up\s+on|train\s+yourself)\b",re.IGNORECASE)
_MISSION_BARE_RE=re.compile(r"^\s*(?:hey\s+adam[,\s]*)?(?:can\s+you\s+|please\s+|go\s+|i\s+want\s+you\s+to\s+)?(?:start\s+|go\s+)?learn(?:ing)?(?:\s+something(?:\s+new)?)?\s*[?.!]*\s*$",re.IGNORECASE)
_MISSION_STOP_RE=re.compile(r"\bstop\s+learning\b|\b(?:stop|end|cancel|halt|quit|abort|finish)\s+(?:the\s+|your\s+)?(?:learning|mission|learning\s+mission|studying)\b",re.IGNORECASE)
_MISSION_STATUS_RE=re.compile(r"\b(?:how(?:'?s|\s+is|\s+are|\s+goes)\s+(?:your\s+|the\s+|that\s+)?(?:learning|mission|studying|study)|learning\s+(?:status|progress|going)|mission\s+(?:status|progress)|what\s+are\s+you\s+learning|how\s+(?:much|far)\s+have\s+you\s+learned)\b",re.IGNORECASE)
_MISSION_RESUME_RE=re.compile(r"\b(?:keep|continue|resume)\s+(?:learning|going|the\s+mission|studying)\b|\bgo\s+deeper\b|\blearn\s+(?:even\s+)?more\b",re.IGNORECASE)
_MISSION_SPLIT_RE=re.compile(r"[,;]?\s*\b(?:stop(?:ping)?(?:\s+(?:once|when|after))?|until|till|once|when|after\s+you|and\s+(?:then\s+)?stop)\b\s+",re.IGNORECASE)
def _parse_mission(text):
    t=(text or '').strip().rstrip('?.! ')
    parts=_MISSION_SPLIT_RE.split(t,maxsplit=1)
    mission=parts[0].strip();stop=(parts[1].strip() if len(parts)>1 else '')
    mission=re.sub(r"^(?:hey\s+adam[,\s]*|i\s+want\s+you\s+to\s+|i'?d\s+like\s+you\s+to\s+|i\s+need\s+you\s+to\s+|please\s+|go\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)+","",mission,flags=re.IGNORECASE).strip()
    mission=re.sub(r"^(?:go\s+)?(?:and\s+)?(?:start\s+)?(?:learn(?:ing)?(?:\s+(?:to|about|how\s+to|to\s+be))?|become(?:\s+the)?(?:\s+(?:best|greatest|world'?s\s+best))?|master|get\s+(?:much\s+)?better\s+at|study\s+up\s+on|train(?:\s+yourself)?(?:\s+(?:on|to|to\s+be))?)\s+","",mission,flags=re.IGNORECASE).strip()
    mission=re.sub(r"\s+(?:ever|please|for\s+me|on\s+your\s+own)$","",mission,flags=re.IGNORECASE).strip().rstrip(',. ')
    return (mission or t),stop
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
        from amni.serve.rate_limit import from_env as _rl_from_env,client_key as _rl_key
        from amni.serve.code_safety import scrub_egress as _egress,scrub_secrets as _secrets
        _RL_TEACH=_rl_from_env('teach',30);_RL_CHAT=_rl_from_env('chat',60)
        _MAX_INPUT_CHARS=int(os.environ.get('AMNI_MAX_INPUT_CHARS','100000'))
        import uvicorn
    except ImportError:
        print('[amni_serve] missing fastapi/uvicorn. Install: pip install fastapi uvicorn pydantic',flush=True)
        sys.exit(1)
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore,PersonaStore
    from amni.serve.skills import default_registry
    from amni.serve import ollama_compat,web,mcp,openai_compat,jarvis_web,memory_endpoints,task_endpoints,vision_endpoints,voice_endpoints,amni_chat_bridge,unified_web,model_installer,mode_endpoints
    try:from amni.serve import trace_endpoints
    except Exception:trace_endpoints=None
    from amni.serve.code_atlas import CodeAtlas
    if not os.environ.get('AMNI_NO_NVFP4'):
        try:
            from amni.adam import select_model_bake as _selbake
            _nvb,_why=_selbake()
            if _nvb:args.bake=_nvb;args.model=_nvb;print(f'[amni_serve] VRAM gate: serving NVFP4 12B as the model ({_why}) -> {_nvb}',flush=True)
            else:print(f'[amni_serve] VRAM gate: keeping light bake ({_why}); NVFP4 12B not selected (set AMNI_NO_NVFP4=1 to force-skip)',flush=True)
        except Exception as _se:print(f'[amni_serve] NVFP4 auto-select skipped: {_se}',flush=True)
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
    _warmup_state={'done':False,'wall_s':None,'error':None,'coding':False}
    if not os.environ.get('AMNI_NO_WARMUP'):
        import threading as _th
        def _bg_warmup():
            _w0=time.time();print('[amni_serve] background warmup (compiles ROCm/CUDA kernels + coding tools, ~10-30s)...',flush=True)
            try:
                _ = adam.ask('hi',writeback=False)
                _warmup_state['wall_s']=round(time.time()-_w0,1);_warmup_state['done']=True
                print(f'[amni_serve] warmup done in {_warmup_state["wall_s"]}s — first user request will be fast',flush=True)
            except Exception as _we:
                _warmup_state['error']=str(_we)[:200];_warmup_state['done']=True
                print(f'[amni_serve] warmup failed (non-fatal): {_we}',flush=True)
            try:
                _se=getattr(adam,'sem_lut',None)
                if _se is not None:
                    if hasattr(_se,'_ensure_encoder'):_se._ensure_encoder()
                    if hasattr(_se,'fit'):
                        try:_se.fit()
                        except Exception:pass
                print('[amni_serve] coding warmup done (embedder + lesson-embedding cache primed — first scan/distill is now incremental-fast)',flush=True)
            except Exception as _ce:print(f'[amni_serve] coding warmup failed (non-fatal): {_ce}',flush=True)
            _warmup_state['coding']=True
        _th.Thread(target=_bg_warmup,daemon=True).start()
    else:_warmup_state['done']=True;_warmup_state['coding']=True
    skills=default_registry(workdir=args.workdir,roots=args.root,audit_log=args.audit_log,unrestricted=args.unrestricted_files)
    store=ConversationStore(root=args.conv_root)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.default_persona:personas.set_default(args.default_persona)
    agent=AmniAgent(adam=adam,skills=skills,store=store,workdir=args.workdir,personas=personas,use_persona=not args.no_persona)
    print(f'[amni_serve] Persona: default={personas._default} known={[p.name for p in personas.list_known()]}',flush=True)
    scope='UNRESTRICTED (all drives)' if args.unrestricted_files else f'roots={[str(r) for r in skills.roots]}'
    print(f'[amni_serve] Agent ready: skills={[s["name"] for s in agent.list_skills()]} {scope} sessions={len(store.list_sessions())}',flush=True)
    try:
        from amni.serve.integrity_gate import gate as _igate
        _ig=_igate(adam)
        if _ig.get('enabled') is not False:print(f'[amni_serve] integrity gate: core_ok={_ig.get("ok")} ptex={_ig.get("ptex")}',flush=True)
    except Exception as _ige:print(f'[amni_serve] integrity gate error (non-fatal): {_ige}',flush=True)
    app=FastAPI(title='Amni-Ai Adam',version='6.0.0')
    if os.environ.get('AMNI_SCRUB_EGRESS','1').lower() not in ('0','false','no'):
        from fastapi.responses import JSONResponse as _JR
        @app.exception_handler(Exception)
        async def _scrub_unhandled(request,exc):
            try:
                from amni.serve.code_safety import scrub_egress as _se
                msg=_se(f'{type(exc).__name__}: {exc}')[:400]
            except Exception:msg='internal error'
            return _JR(status_code=500,content={'error':msg,'note':'host paths/secrets scrubbed from error responses'})
        @app.exception_handler(HTTPException)
        async def _scrub_http_exc(request,exc):
            d=exc.detail
            try:d=_egress(d) if isinstance(d,str) else d
            except Exception:pass
            return _JR(status_code=exc.status_code,content={'detail':d},headers=getattr(exc,'headers',None))
    if args.cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
    _AUTH_TOKEN=(os.environ.get('AMNI_AUTH_TOKEN') or '').strip()
    if not _AUTH_TOKEN:
        _atf=(os.environ.get('AMNI_AUTH_TOKEN_FILE') or '').strip()
        if _atf and Path(_atf).exists():
            try:_AUTH_TOKEN=Path(_atf).read_text(encoding='utf-8').strip()
            except Exception:_AUTH_TOKEN=''
    if _AUTH_TOKEN:
        import hmac as _hmac
        from fastapi.responses import JSONResponse as _AuthJR
        _AUTH_OPEN={'/','/unified','/jarvis','/hud','/healthz','/manifest.webmanifest','/sw.js','/favicon.ico'}
        _AUTH_OPEN_PFX=('/assets/','/icons/')
        @app.middleware('http')
        async def _auth_gate(request,call_next):
            p=request.url.path
            if request.method=='OPTIONS' or p in _AUTH_OPEN or any(p.startswith(x) for x in _AUTH_OPEN_PFX):return await call_next(request)
            t=request.headers.get('x-amni-token','')
            if not t:
                _h=request.headers.get('authorization','')
                if _h[:7].lower()=='bearer ':t=_h[7:].strip()
            if not t:t=request.query_params.get('token','') or request.cookies.get('amni_token','')
            if t and _hmac.compare_digest(t,_AUTH_TOKEN):return await call_next(request)
            return _AuthJR(status_code=401,content={'error':'unauthorized','auth_required':True,'hint':'Adam access token required (X-Amni-Token header, ?token=, or amni_token cookie).'})
        print('[amni_serve] AUTH GATE ON — token required for everything but the app shell + /healthz',flush=True)
    else:print('[amni_serve] auth gate OFF — set AMNI_AUTH_TOKEN before exposing Adam beyond localhost (it can run shell/file/PC ops)',flush=True)
    class ChatRequest(BaseModel):
        message:str
        session_id:Optional[str]=None
        use_skills:bool=True
        writeback:bool=True
        client_lat:Optional[float]=None
        client_lon:Optional[float]=None
        persona:Optional[str]=None
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
    def chat(req:ChatRequest):
        if len(req.message or '')>_MAX_INPUT_CHARS:raise HTTPException(status_code=413,detail=f'message too large ({len(req.message)} chars > {_MAX_INPUT_CHARS}); split it up')
        if req.client_lat is not None:agent._client_lat=req.client_lat
        if req.client_lon is not None:agent._client_lon=req.client_lon
        try:return agent.chat(req.message,session_id=req.session_id,use_skills=req.use_skills,writeback=req.writeback)
        finally:agent._client_lat=None;agent._client_lon=None
    @app.post('/chat/stream')
    async def chat_stream(req:ChatRequest,request:Request):
        _ok,_rl=_RL_CHAT.allow(_rl_key(request))
        if not _ok:raise HTTPException(status_code=429,detail=f"rate limit: max {_rl['limit']}/{int(_rl['window_s'])}s — retry in {_rl['retry_after_s']}s")
        if len(req.message or '')>_MAX_INPUT_CHARS:raise HTTPException(status_code=413,detail=f'message too large ({len(req.message)} chars > {_MAX_INPUT_CHARS}); split it up')
        from fastapi.responses import StreamingResponse
        import json as _json
        from amni.serve.agent import _needs_cot,_pick_cot
        from amni.serve import tone_atlas
        if req.client_lat is not None:agent._client_lat=req.client_lat
        if req.client_lon is not None:agent._client_lon=req.client_lon
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
            _route=None
            try:_route=_intent_clf.route(req.message) if _intent_clf else None
            except Exception:_route=None
            if req.persona and agent.use_persona:
                try:
                    _reqp=req.persona.strip().lower()
                    if _reqp and agent.personas.has(_reqp) and agent.personas._session_persona.get(conv.session_id)!=_reqp:agent.personas.assign_session(conv.session_id,_reqp)
                except Exception as _pae:print(f'[amni_serve] persona assign failed: {_pae}',flush=True)
            if (is_build_request(req.message) or (_route and _route.get('is_skill') and _route.get('intent')=='build_request')) and not _MISSION_START_RE.search(req.message.strip()) and not _MISSION_BARE_RE.match(req.message.strip()):
                _ag_persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
                _ag_persona_name=_ag_persona.name if _ag_persona else 'Adam'
                yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":_ag_persona_name,"agentic":True})}\n\n'
                _final_answer='';_n_steps=0
                try:
                    for _ev in run_goal_stream(agent,skills,req.message,max_steps=int(os.environ.get('AMNI_AGENTIC_MAX_STEPS','8')),timeout_s=float(os.environ.get('AMNI_AGENTIC_TIMEOUT_S','240'))):
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
            _daemon=getattr(agent,'learning_daemon',None)
            if _daemon is not None:
                _mp=getattr(agent,'_mission_pending',None)
                if _mp is None:_mp=set();agent._mission_pending=_mp
                _persona_x=agent.personas.for_session(conv.session_id) if agent.use_persona else None
                _pname_x=_persona_x.name if _persona_x else 'Adam'
                _ml=req.message.strip();_mll=_ml.lower();_mtxt=None
                if conv.session_id in _mp and _mll not in ('cancel','never mind','nevermind','stop','no','forget it'):
                    _mp.discard(conv.session_id);_mt,_sc=_parse_mission(_ml);_r=_daemon.set_mission(_mt,stop_condition=_sc)
                    _mtxt=((f"On it! \U0001F680 New mission: **master {_r['mission']}**"+(f" — I'll keep at it until {_r['stop_condition']}." if _r.get('stop_condition') else " — I'll keep going until you tell me to stop.")+f"\n\nI broke it into {len(_r.get('subtopics') or [])} starting subtopics ("+', '.join((_r.get('subtopics') or [])[:5])+"…) and I'm crawling the web on them right now, storing what I learn. Ask me **“how's your learning?”** anytime, or say **“stop learning”** to finish.") if _r.get('started') else f"Hmm, couldn't start that mission: {_r.get('error') or _r}")
                elif conv.session_id in _mp:
                    _mp.discard(conv.session_id);_mtxt="No worries — scrapped that. Say “learn” whenever you want to give me a mission."
                elif _MISSION_STOP_RE.search(_mll):
                    _r=_daemon.stop_mission();_mtxt=((f"Done! \U0001F3C1 Wrapped the **{_r['mission']}** mission: **{_r['facts_new']} new facts** across {_r['subtopics_covered']} subtopics in {round(_r['duration_s']/60,1)} min (self-rated confidence {_r['confidence']}%). It's all baked into my memory now — try me on it!") if _r.get('stopped') else f"{_r.get('error','No mission is running right now.')}")
                elif _MISSION_STATUS_RE.search(_mll):
                    _r=_daemon.mission_status();_mtxt=((f"\U0001F4DA Mission: **{_r['mission']}** — {_r['subtopics_covered']} subtopics studied (round {_r['rounds']}), **{_r['facts_new']} new facts** stored, currently on _{_r.get('current_topic') or 'queuing the next one'}_. Self-confidence {_r['confidence']}%."+(f" Still chasing: {', '.join(_r['gaps'][:3])}." if _r.get('gaps') else "")) if _r.get('active') else "I don't have a learning mission going right now — give me one! Like “learn to be a great Rust programmer, stop when you're confident you can one-shot code.”")
                elif _MISSION_RESUME_RE.search(_mll) and _daemon.mission is not None:
                    _r=_daemon.resume_mission();_mtxt=((f"Back at it! \U0001F4AA Pushing deeper on **{_r['mission']}**.") if _r.get('resumed') else f"{_r.get('error')}")
                elif _MISSION_BARE_RE.match(_ml):
                    _mp.add(conv.session_id);_mtxt="Oac! What would you like me to get really good at? Give me a **mission** and a **stopping point** — like _“become the best Python programmer, stop once you've mastered every function and can one-shot code.”_ I'll web-crawl and study on my own until I get there (or you say stop)."
                elif _MISSION_START_RE.search(_ml) and len(_ml.split())>=4 and not _NEEDS_FRESH_INFO_RE.search(_mll):
                    _mt,_sc=_parse_mission(_ml)
                    if _mt and len(_mt)>=3 and (_daemon.mission is None or _daemon.mission.get('status')!='active'):
                        _r=_daemon.set_mission(_mt,stop_condition=_sc)
                        _mtxt=((f"On it! \U0001F680 New mission: **master {_r['mission']}**"+(f" — until {_r['stop_condition']}." if _r.get('stop_condition') else " — until you say stop.")+f"\n\nDecomposed into {len(_r.get('subtopics') or [])} subtopics ("+', '.join((_r.get('subtopics') or [])[:5])+"…); crawling now and storing facts. Say **“how's your learning?”** or **“stop learning”** anytime.") if _r.get('started') else None)
                    elif _daemon.mission is not None and _daemon.mission.get('status')=='active':
                        _mtxt=f"I'm already on a mission (**{_daemon.mission.get('text')}**). Say “stop learning” first if you want to switch."
                if _mtxt:
                    yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":_pname_x,"skill":"learning_daemon"})}\n\n'
                    for _ch in [_mtxt[i:i+48] for i in range(0,len(_mtxt),48)]:yield f'event: token\ndata: {_json.dumps(_ch)}\n\n'
                    conv.append('assistant',_mtxt,{'tier':'tier0_skill_learning_mission','persona':_pname_x,'category':'skill'})
                    yield f'event: done\ndata: {_json.dumps({"tier":"tier0_skill_learning_mission","wall_s":round(time.time()-t0,3),"persona":_pname_x})}\n\n';return
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
                        _wenv=_sr.output.get('widget') if isinstance(_sr.output,dict) else None
                        if _wenv:yield f'event: widget\ndata: {_json.dumps(_wenv)}\n\n'
                        for ch in [_wrapped[i:i+48] for i in range(0,len(_wrapped),48)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                        conv.append('assistant',_wrapped,{'tier':f'tier0_skill_{_sname}','persona':_sk_persona_name,'category':_sk_cat,'skill':_sname})
                        yield f'event: done\ndata: {_json.dumps({"tier":f"tier0_skill_{_sname}","wall_s":round(time.time()-t0,3),"persona":_sk_persona_name})}\n\n';return
                    else:
                        _err=str(getattr(_sr,'error','') or 'no data returned')
                        _emsg=f"I tried the **{_sname}** skill but it couldn't return live data ({_err}). I won't make up values — please retry, or check the STATUS panel's skill-failures."
                        yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":_sk_persona_name,"skill":_sname,"skill_error":True})}\n\n'
                        for ch in [_emsg[i:i+48] for i in range(0,len(_emsg),48)]:yield f'event: token\ndata: {_json.dumps(ch)}\n\n'
                        conv.append('assistant',_emsg,{'tier':f'tier0_skill_{_sname}_failed','persona':_sk_persona_name,'category':'factual','skill':_sname})
                        yield f'event: done\ndata: {_json.dumps({"tier":f"tier0_skill_{_sname}_failed","wall_s":round(time.time()-t0,3)})}\n\n';return
                except Exception as _se:print(f'[amni_serve] /chat/stream skill {_sname} failed: {_se}',flush=True)
            persona=agent.personas.for_session(conv.session_id) if agent.use_persona else None
            persona_name=persona.name if persona else 'Adam'
            yield f'event: meta\ndata: {_json.dumps({"session_id":conv.session_id,"persona":persona_name})}\n\n'
            yield f'event: status\ndata: {_json.dumps({"stage":"understanding"})}\n\n'
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
            _intent_label,_intent_conf=((_route['intent'],_route.get('confidence',0.0)) if _route else (_intent_clf.classify(req.message) if _intent_clf else ('unknown',0.0)))
            if _route and _route.get('ambiguous') and not _route.get('is_skill') and _route.get('alt_intent')=='needs_fresh_info' and not _NEEDS_FRESH_INFO_RE.search(req.message):
                yield f'event: disambiguate\ndata: {_json.dumps({"reason":"I answered from what I know — want me to check the web for the latest instead?","options":[{"label":"\U0001F310 Search the web","send":"search the web for: "+req.message[:200]}]})}\n\n'
            _profile_authoritative=(_intent_label=='profile_about_me') or bool(_PROFILE_AUTHORITATIVE_RE.search(req.message))
            _memory_recall=(_intent_label=='memory_recall') or bool(_MEMORY_RECALL_RE.search(req.message))
            apply_cot=_needs_cot(category,req.message) and persona and persona.name!='Adam'
            if _profile_authoritative or _memory_recall:apply_cot=False
            from amni.serve.conversation import detect_personal as _dp
            _hist_n=int(os.environ.get('AMNI_HISTORY_TURNS','12'))
            history_pairs=conv.history_pairs(n=_hist_n) if len(conv.turns)>1 else []
            _skip_atlas=_profile_authoritative or _memory_recall or (_intent_label=='introspection') or (_intent_label=='math_calc')
            atlas_recall=[] if _skip_atlas else (agent.atlas.recall(req.message,session_id=conv.session_id,k=3,include_global=True) if getattr(agent,'atlas',None) is not None else [])
            for r in atlas_recall:
                pair=(r.get('user',''),r.get('assistant',''))
                if pair[0] and pair[1] and pair not in history_pairs:history_pairs=[pair]+history_pairs
            history_pairs=history_pairs[-_hist_n:]
            user_facts=agent._extract_user_facts(conv,extra_user_msgs=[r.get('user','') for r in atlas_recall],profile_only=(_intent_label=='profile_about_me')) if hasattr(agent,'_extract_user_facts') else []
            user_facts=['The current local date and time is '+time.strftime('%A, %B %d, %Y at %I:%M %p',time.localtime())+'. Use this exact value for any date, time, "today", "now", or current-year reasoning — never guess or invent a date. For live system stats (CPU/memory/disk) or weather, rely on the tool widgets, never fabricate numbers.']+user_facts
            is_private=_dp(req.message) or conv.has_personal(n=20) or any(r.get('is_personal') for r in atlas_recall)
            sl=getattr(adam,'sem_lut',None)
            _has_correction=False
            _persona_query=bool(persona and persona.name and persona.name.lower() in req.message.lower() and len(req.message.split())>=3)
            _pre_web_supplemented=False
            try:
                _is_fresh=(_intent_label=='needs_fresh_info') or bool(_NEEDS_FRESH_INFO_RE.search(req.message)) or bool(_RESEARCH_WEB_RE.search(req.message))
                _is_introsp=(_intent_label in ('greeting','introspection')) or bool(_INTROSPECT_NO_WEB_RE.search(req.message))
                if (not is_private) and skills.has('web') and _is_fresh and not _is_introsp:
                    _enriched_pre=_enrich_web_query(req.message,conv,getattr(agent,'profile',None))
                    yield f'event: status\ndata: {_json.dumps({"stage":"web"})}\n\n'
                    yield f'event: web_lookup\ndata: {_json.dumps({"trigger":"pre_fetch","query":_enriched_pre[:200],"enriched_from":req.message[:80]})}\n\n'
                    import concurrent.futures as _cf
                    _pre_web_r=None;_wex=_cf.ThreadPoolExecutor(max_workers=1)
                    try:_pre_web_r=_wex.submit(skills.call,'web',{'query':_enriched_pre},ctx={'adam':adam}).result(timeout=float(os.environ.get('AMNI_WEB_PREFETCH_TIMEOUT','22')))
                    except Exception:yield f'event: web_supplement_skipped\ndata: {_json.dumps({"reason":"web search slow/unavailable","phase":"pre_fetch"})}\n\n'
                    finally:_wex.shutdown(wait=False)
                    if _pre_web_r is not None and _pre_web_r.ok and _pre_web_r.output:
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
            if hit is None and sl is not None and hasattr(sl,'lookup_soft') and os.environ.get('AMNI_FEDERATION_ONDEMAND','1')!='0' and not is_private and not _memory_recall and not _profile_authoritative and not _persona_query:
                try:
                    import amni.serve.federation_store as _fstore
                    _fr=_fstore.fetch_for_query(adam,req.message)
                    if _fr.get('fetched') and _fr.get('merged_new',0)>0:
                        yield f'event: status\ndata: {_json.dumps({"stage":"federation"})}\n\n'
                        yield f'event: federation_fetch\ndata: {_json.dumps({"pack":_fr.get("id"),"merged":_fr.get("merged_new"),"lang":_fr.get("matched_lang")})}\n\n'
                        hit=sl.lookup_soft(req.message,margin=eff)
                except Exception as _fe:print(f'[amni_serve] federation on-demand skipped: {_fe}',flush=True)
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
            _msg_l=req.message.lower();_is_code_q=bool(re.search(r"\b(?:rust|wasm|cargo|javascript|node\.?js|typescript|python|go(?:lang)?|c\+\+|cpp|c#|csharp|java|kotlin|swift|ruby|php|bash|sql|haskell|elixir|zig|ktor|websocket|server|client|implement|function|class|interface|struct|trait|module|library|framework|api|endpoint|sdk|build|setup|configure|deploy)\b",_msg_l))
            max_new=int(os.environ.get('AMNI_MAX_NEW_TOKENS','4096'))
            full=[];_bump('cot_generations') if apply_cot else None
            in_final=not expects_final;buf='';seen_final=False;_buf_start=time.time();_last_ping=time.time();_drift_stop=False;_final_buf=''
            _max_out=int(os.environ.get('AMNI_MAX_OUTPUT_BYTES',str(512*1024)));_out_bytes=0
            _DRIFT_MARKERS=('Thinking Process','thought\n','\nThinking','**Self-','*(Self-','**Analyze Request','1. RESTATE:','1.  RESTATE:','1. **Analyze','**Recall Persona','**Determine Strategy','Self-Correction','\n[Looked','[Looked','[Search performed','[Search completed','[Search done','[Search results','[Presenting','[Current weather data','[Result of search','[The system returns','(Outputting the result','(Search returns','(Result of search','(Waiting for search','(Assuming the search')
            try:
                import json as _ppj
                _ppf=Path(_REPO_ROOT)/'experiences'/'perf'/'pipeline_telemetry.jsonl';_ppf.parent.mkdir(parents=True,exist_ok=True)
                with open(_ppf,'a',encoding='utf-8') as _ppfh:_ppfh.write(_ppj.dumps({'ts':time.time(),'to_gen_ms':round((time.time()-t0)*1000,1),'facts_n':len(user_facts),'hist_n':len(history_pairs),'pre_web':bool(_pre_web_supplemented),'cot':bool(apply_cot),'cat':category})+'\n')
            except Exception:pass
            yield f'event: status\ndata: {_json.dumps({"stage":"reasoning" if apply_cot else "writing"})}\n\n'
            _genstream=adam.chat_persona_stream(req.message,system=sys_p,history=history_pairs,facts=user_facts,is_private=is_private,max_new_tokens=max_new,do_sample=True)
            try:
                for chunk in _genstream:
                    full.append(chunk);_out_bytes+=len(chunk.encode('utf-8') if isinstance(chunk,str) else chunk)
                    if _out_bytes>_max_out:
                        yield f'event: truncated\ndata: {_json.dumps({"reason":"output byte cap reached","limit_bytes":_max_out})}\n\n';_drift_stop=True;break
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
                        _now=time.time();_pre_len=len(buf);buf+=chunk
                        idx=buf.upper().find('FINAL:')
                        if idx>=0:
                            _r=buf[_pre_len:idx]
                            if _r:yield f'event: reasoning\ndata: {_json.dumps(_secrets(_r))}\n\n'
                            after=buf[idx+6:].lstrip(' :\t\n')
                            if after:yield f'event: token\ndata: {_json.dumps(after)}\n\n';_final_buf=after
                            in_final=True;seen_final=True;buf=''
                        else:
                            yield f'event: reasoning\ndata: {_json.dumps(_secrets(chunk))}\n\n'
                            if (_now-_last_ping)>3:yield f'event: thinking\ndata: {_json.dumps({"buf_chars":len(buf),"elapsed":round(_now-_buf_start,1)})}\n\n';_last_ping=_now
            except Exception as e:yield f'event: error\ndata: {_json.dumps(_egress(str(e)))}\n\n';return
            finally:
                try:_genstream.close()
                except Exception:pass
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
                    try:
                        from amni.serve.self_debug import review as _review,debug_report as _dbgrep
                        _allcode='\n\n'.join(blocks)
                        _rev=_review(_allcode)
                        if not _rev['clean']:
                            yield f'event: debug\ndata: {_json.dumps({"lint":len(_rev["lint"]),"runtime":len(_rev["runtime"]),"func":_rev.get("func")})}\n\n'
                            _pr=_perturb_retry(adam,skills,sys_p,_allcode,_dbgrep(_allcode),req.message,max_steps=2,emit=lambda d:None)
                            if _pr.get('success') and _pr.get('code') and _review(_pr['code'])['clean']:
                                final=(final.replace(_allcode,_pr['code']) if _allcode in final else final+f"\n\n```python\n{_pr['code']}\n```")
                                blocks=_extract_python_blocks(final)
                                final+='\n\n**[Self-debug fixed it before posting — now passes lint + edge-case probes]**'
                                yield f'event: debug\ndata: {_json.dumps({"fixed":True})}\n\n'
                            else:
                                final+=f"\n\n**[Self-debug flags — review before use]**\n```\n{_dbgrep(_allcode)[:600]}\n```"
                                yield f'event: debug\ndata: {_json.dumps({"fixed":False})}\n\n'
                    except Exception as _de:print(f'[serve] self-debug step error: {_de}',flush=True)
                    runnable=[b for b in blocks if ('print(' in b or 'if __name__' in b)]
                    if runnable:
                        snippet=('\n\n'.join(blocks) if len(blocks)>1 else runnable[-1])
                        if len(blocks)>1:
                            _bump('multi_block_stitched')
                            yield f'event: multi_block\ndata: {_json.dumps({"blocks":len(blocks),"runnable":len(runnable),"stitched_chars":len(snippet)})}\n\n'
                        try:
                            run_r=skills.call('run_python',{'code':snippet,'timeout':int(os.environ.get('AMNI_SANDBOX_TIMEOUT_S','8'))},ctx={'adam':adam})
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
                        except Exception as e:yield f'event: exec\ndata: {_json.dumps({"error":_egress(str(e))})}\n\n'
            _introspect_no_web=(_intent_label in ('greeting','introspection')) or bool(_INTROSPECT_NO_WEB_RE.search(req.message))
            _needs_fresh_info=(_intent_label=='needs_fresh_info') or bool(_NEEDS_FRESH_INFO_RE.search(req.message)) or bool(_RESEARCH_WEB_RE.search(req.message))
            if (not is_private) and skills.has('web') and not _memory_recall and not _profile_authoritative and not _introspect_no_web and not _pre_web_supplemented and (raw_final or final) and _UNCERTAIN_RE.search(raw_final or final or ''):
                _enriched_q=_enrich_web_query(req.message,conv,getattr(agent,'profile',None))
                if len(_enriched_q.split())<4:
                    yield f'event: web_supplement_skipped\ndata: {_json.dumps({"reason":"query too vague even after enrichment","query":_enriched_q})}\n\n'
                else:
                    yield f'event: web_lookup\ndata: {_json.dumps({"trigger":"uncertainty","query":_enriched_q[:200],"enriched_from":req.message[:80]})}\n\n'
                    try:
                        import concurrent.futures as _cf2
                        _wex2=_cf2.ThreadPoolExecutor(max_workers=1)
                        try:_web_r=_wex2.submit(skills.call,'web',{'query':_enriched_q},ctx={'adam':adam}).result(timeout=float(os.environ.get('AMNI_WEB_SUPPLEMENT_TIMEOUT','22')))
                        finally:_wex2.shutdown(wait=False)
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
                    except Exception as _we:yield f'event: web_supplement_error\ndata: {_json.dumps({"msg":_egress(str(_we))[:200]})}\n\n'
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
    def teach(req:TeachRequest,request:Request):
        _ok,_rl=_RL_TEACH.allow(_rl_key(request))
        if not _ok:raise HTTPException(status_code=429,detail=f"rate limit: max {_rl['limit']}/{int(_rl['window_s'])}s — retry in {_rl['retry_after_s']}s")
        if len(req.question or '')+len(req.answer or '')>_MAX_INPUT_CHARS:raise HTTPException(status_code=413,detail=f'teach payload too large (>{_MAX_INPUT_CHARS} chars)')
        try:
            from amni.serve.conversation import detect_personal as _dp
            if _dp(req.question) or _dp(req.answer):
                return {'taught':False,'rejected':'personal/PII detected — lessons are PTEX-stored and federation-eligible, so personal data must not enter. Keep it in a local session instead.'}
        except Exception:pass
        _bus=getattr(agent,'memory_bus',None)
        if _bus is not None:
            try:
                _r=_bus.record_learning(req.question,req.answer,kind='fact',provenance='user:teach',exactness='exact')
                _base=adam.teach(req.question,req.answer)
                if isinstance(_base,dict):_base['memory_bus']=_r
                return _base
            except Exception as _te:print(f'[teach] bus.record_learning failed, fell back to teach: {_te}',flush=True)
        return adam.teach(req.question,req.answer)
    class CompleteReq(BaseModel):
        prefix:str
        suffix:Optional[str]=''
        language:Optional[str]=None
        max_tokens:int=40
        stop:Optional[List[str]]=None
    @app.post('/complete')
    def complete(req:CompleteReq):
        if not req.prefix:raise HTTPException(status_code=400,detail='missing prefix')
        if len(req.prefix or '')+len(req.suffix or '')>_MAX_INPUT_CHARS:raise HTTPException(status_code=413,detail=f'completion payload too large (>{_MAX_INPUT_CHARS} chars)')
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
            except Exception as e:result['tts_error']=_egress(str(e))[:200]
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
    def root_unified():
        try:return HTMLResponse(unified_web.page())
        except Exception as _ue:
            print(f'[amni_serve] unified UI failed ({_ue}), falling back to HUD',flush=True)
            if _HUD_PATH.exists():return HTMLResponse(_HUD_PATH.read_text(encoding='utf-8'))
            return HTMLResponse(f'<html><body style="font-family:system-ui;padding:40px;background:#0a0a14;color:#e2e8f0"><h1>Adam</h1><p>UI unavailable: {_ue}</p></body></html>',status_code=200)
    @app.get('/hud',response_class=HTMLResponse)
    def root_hud_explicit():
        if _HUD_PATH.exists():return HTMLResponse(_HUD_PATH.read_text(encoding='utf-8'))
        return HTMLResponse(f'<html><body style="font-family:system-ui;padding:40px;background:#0a0a14;color:#e2e8f0"><h1>Adam</h1><p>HUD file not found at <code>{_HUD_PATH}</code>.</p></body></html>',status_code=200)
    @app.get('/healthz')
    def health():return {'status':'ok','lessons_n':len(adam.sem_lut._raw),'skills_n':len(skills.list_skills()),'version':APP_VERSION,'warmup':_warmup_state,'auth_required':bool(_AUTH_TOKEN)}
    @app.get('/manifest.webmanifest')
    def _pwa_manifest():
        from fastapi.responses import JSONResponse as _MJR
        return _MJR(content={'name':'Adam — Amni-Ai','short_name':'Adam','description':'GF(17) texture-native AI — persistent learning, agentic skills.','start_url':'/unified','scope':'/','display':'standalone','background_color':'#040711','theme_color':'#040711','orientation':'any','icons':[{'src':'/assets/icons/adam-192.png','sizes':'192x192','type':'image/png','purpose':'any maskable'},{'src':'/assets/icons/adam-512.png','sizes':'512x512','type':'image/png','purpose':'any maskable'}]},media_type='application/manifest+json')
    @app.get('/sw.js')
    def _pwa_sw():
        from fastapi.responses import Response as _SWResp
        js="const C='adam-shell-v1';self.addEventListener('install',e=>self.skipWaiting());self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==location.origin)return;if(/\\.(png|svg|ico|css|js|woff2?|ttf)$/i.test(u.pathname)){e.respondWith(caches.open(C).then(c=>c.match(e.request).then(r=>r||fetch(e.request).then(resp=>{try{if(resp&&resp.ok)c.put(e.request,resp.clone())}catch(_){}return resp}))))}});"
        return _SWResp(content=js,media_type='application/javascript')
    _REPO_ROOT=str(Path(__file__).resolve().parents[1])
    def _git(*a,timeout=30):
        import subprocess as _sp
        try:return _sp.run(['git','-C',_REPO_ROOT,*a],capture_output=True,text=True,timeout=timeout)
        except Exception:return None
    @app.get('/update/check')
    def _update_check():
        head=_git('rev-parse','--abbrev-ref','HEAD')
        if head is None or head.returncode!=0:return {'ok':False,'error':'not a git repo or git unavailable','version':APP_VERSION}
        branch=(head.stdout or '').strip() or 'HEAD'
        fetch=_git('fetch','--quiet','origin',branch,timeout=45)
        cur=_git('rev-parse','HEAD');rem=_git('rev-parse',f'origin/{branch}')
        cur_sha=(cur.stdout or '').strip()[:9] if cur else '';rem_sha=(rem.stdout or '').strip()[:9] if rem else ''
        behind=_git('rev-list','--count',f'HEAD..origin/{branch}');ahead=_git('rev-list','--count',f'origin/{branch}..HEAD')
        nb=int((behind.stdout or '0').strip() or 0) if (behind and behind.returncode==0) else 0
        na=int((ahead.stdout or '0').strip() or 0) if (ahead and ahead.returncode==0) else 0
        log=_git('log','--oneline','-5',f'HEAD..origin/{branch}');dirty=_git('status','--porcelain')
        return {'ok':True,'branch':branch,'behind':nb,'ahead':na,'current':cur_sha,'remote':rem_sha,'update_available':nb>0,'dirty':bool((dirty.stdout or '').strip()) if dirty else False,'incoming':[l for l in (log.stdout or '').splitlines() if l][:5],'fetch_ok':bool(fetch and fetch.returncode==0),'version':APP_VERSION}
    @app.post('/update/apply')
    def _update_apply():
        head=_git('rev-parse','--abbrev-ref','HEAD')
        if head is None or head.returncode!=0:return {'ok':False,'error':'not a git repo or git unavailable'}
        branch=(head.stdout or '').strip() or 'main'
        dirty=_git('status','--porcelain')
        if dirty and (dirty.stdout or '').strip():return {'ok':False,'error':'working tree has uncommitted local changes — commit or stash them first','dirty':True,'changes':[l for l in (dirty.stdout or '').splitlines()][:20]}
        pull=_git('pull','--ff-only','origin',branch,timeout=120)
        if pull is None:return {'ok':False,'error':'git pull failed to run'}
        out=((pull.stdout or '')+(pull.stderr or '')).strip()
        if pull.returncode!=0:return {'ok':False,'error':out[:500] or 'git pull failed'}
        new=_git('rev-parse','HEAD');new_sha=(new.stdout or '').strip()[:9] if new else ''
        return {'ok':True,'pulled':True,'branch':branch,'new':new_sha,'output':out[:500],'restart_required':True,'message':'Update applied. Restart Adam to load the new version.'}
    @app.get('/perf/prefill')
    def _perf_prefill(n:int=3000):
        import json as _pj
        from pathlib import Path as _PP
        f=_PP(_REPO_ROOT)/'experiences'/'perf'/'prefill_telemetry.jsonl'
        recs=[]
        try:
            if f.exists():
                for ln in f.read_text(encoding='utf-8').splitlines()[-max(1,n):]:
                    try:recs.append(_pj.loads(ln))
                    except Exception:pass
        except Exception:pass
        def _agg(key):
            vals=sorted(r[key] for r in recs if isinstance(r.get(key),(int,float)))
            if not vals:return None
            return {'n':len(vals),'avg':round(sum(vals)/len(vals),1),'p50':vals[len(vals)//2],'p95':vals[min(len(vals)-1,int(len(vals)*0.95))],'max':vals[-1]}
        lh=_iter_counters.get('lut_hits',0);cg=_iter_counters.get('cot_generations',0)
        return {'samples':len(recs),'prompt_tokens':_agg('prompt_tokens'),'ttft_ms':_agg('ttft_ms'),'gen_ms':_agg('gen_ms'),'new_tokens':_agg('new_tokens'),'counters':{'lut_hits':lh,'cot_generations':cg,'lut_hit_rate_pct':round(100*lh/max(1,lh+cg),1)},'note':'prompt_tokens = prefill size re-paid every cot turn; ttft_ms ~= prefill latency'}
    @app.get('/perf/pipeline')
    def _perf_pipeline_agg(n:int=2000):
        import json as _pj
        from pathlib import Path as _PP
        f=_PP(_REPO_ROOT)/'experiences'/'perf'/'pipeline_telemetry.jsonl';recs=[]
        try:
            if f.exists():
                for ln in f.read_text(encoding='utf-8').splitlines()[-max(1,n):]:
                    try:recs.append(_pj.loads(ln))
                    except Exception:pass
        except Exception:pass
        def _agg(key):
            vals=sorted(r[key] for r in recs if isinstance(r.get(key),(int,float)))
            if not vals:return None
            return {'n':len(vals),'avg':round(sum(vals)/len(vals),1),'p50':vals[len(vals)//2],'p95':vals[min(len(vals)-1,int(len(vals)*0.95))],'max':vals[-1]}
        return {'samples':len(recs),'to_gen_ms':_agg('to_gen_ms'),'with_pre_web':sum(1 for r in recs if r.get('pre_web')),'note':'to_gen_ms = full pre-generation pipeline (recall/facts/skill/intent/LUT) before the model is even called; compare to ttft_ms in /perf/prefill'}
    @app.get('/perf/kv_verify')
    def _kv_verify():
        if getattr(adam,'svc',None) is None:return {'error':'runtime not loaded'}
        try:return adam.svc.kv_selftest()
        except Exception as _ke:return {'error':f'{type(_ke).__name__}: {_ke}'[:300]}
    @app.get('/perf/kv_prefix')
    def _kv_prefix():
        if getattr(adam,'svc',None) is None:return {'error':'runtime not loaded'}
        try:
            import time as _kt
            _p=agent.personas.get('rikku') if agent.use_persona else None
            _sysp=_p.system_prompt('') if _p else 'You are a helpful assistant.'
            _f1=['The current local date and time is '+_kt.strftime('%A, %B %d, %Y at %I:%M %p',_kt.localtime())+'.']
            _f2=['The current local date and time is '+_kt.strftime('%A, %B %d, %Y at %I:%M:%S %p',_kt.localtime())+'.']
            return adam.svc.kv_prefix_compare(_sysp,'What is the capital of France?','And roughly what is its population?',facts1=_f1,facts2=_f2)
        except Exception as _ke:return {'error':f'{type(_ke).__name__}: {_ke}'[:300]}
    @app.get('/workdir')
    def workdir():
        wd=str(skills.workdir) if hasattr(skills,'workdir') else ''
        roots=[str(p) for p in (getattr(skills,'roots',[]) or [])]
        return {'workdir':wd,'roots':roots,'unrestricted':bool(getattr(skills,'unrestricted',False))}
    @app.get('/workdir/tree')
    def workdir_tree(max_depth:int=2,max_files:int=200,subpath:str=''):
        import os as _os
        wd=Path(skills.workdir) if hasattr(skills,'workdir') else Path('.')
        base=(wd/subpath).resolve() if subpath else wd.resolve()
        try:base.relative_to(wd.resolve())
        except Exception:
            if not getattr(skills,'unrestricted',False):raise HTTPException(403,'path escapes workdir scope')
        if not base.exists():raise HTTPException(404,f'path not found: {base}')
        _IGNORE={'.git','.venv','venv','__pycache__','node_modules','.pytest_cache','.mypy_cache','.idea','.vscode','dist','build','.next','.nuxt','.adam-venvs'}
        entries=[];count=[0]
        def _walk(p,depth):
            if count[0]>=max_files or depth>max_depth:return
            try:children=sorted(p.iterdir(),key=lambda x:(not x.is_dir(),x.name.lower()))
            except (PermissionError,OSError):return
            for c in children:
                if count[0]>=max_files:break
                if c.name.startswith('.') and c.name not in {'.env','.gitignore'}:continue
                if c.name in _IGNORE:continue
                try:
                    is_dir=c.is_dir();size=0 if is_dir else c.stat().st_size;mtime=c.stat().st_mtime
                    rel=str(c.relative_to(base)).replace('\\','/')
                except Exception:continue
                entries.append({'rel':rel,'name':c.name,'is_dir':is_dir,'size':size,'mtime':mtime,'depth':depth,'ext':c.suffix.lstrip('.').lower() if not is_dir else ''})
                count[0]+=1
                if is_dir and depth<max_depth:_walk(c,depth+1)
        _walk(base,0)
        return {'base':str(base),'workdir':str(wd),'subpath':subpath,'max_depth':max_depth,'max_files':max_files,'truncated':count[0]>=max_files,'entries':entries}
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
        except Exception as e:_gpu={'error':_egress(str(e))[:200]}
        return {'status':'ok','version':'6.9.3','adam':{'lessons_n':len(adam.sem_lut._raw),'sessions_n':len(store._active) if hasattr(store,'_active') else 0,'svc_boot_s':round(adam.stats().get('svc_boot_s',0),1)},'skills':{'count':len(skills.list_skills()),'names':sorted(s['name'] for s in skills.list_skills())},'voice':{'tts_backend':_tb(),'stt_backend':_sb(),'wake_words_available':_ww()},'gpu':_gpu,'workdir':_egress(str(skills.workdir)),'personas_known':[p.name for p in personas.list_known()]}
    @app.get('/skills')
    def list_skills():return {'skills':skills.list_skills()}
    @app.post('/skills/{name}')
    def call_skill(name:str,req:SkillRequest):
        r=skills.call(name,req.args,ctx={'adam':adam,'agent':agent,'personas':personas,'store':store,'conv':None,'coach_atlas':getattr(agent,'coach_atlas',None),'personal_atlas':getattr(agent,'personal_atlas',None),'scheduler':getattr(agent,'scheduler',None),'learning_daemon':getattr(agent,'learning_daemon',None),'knowledge_graph':getattr(agent,'knowledge_graph',None),'task_registry':getattr(agent,'task_registry',None),'vision':getattr(agent,'vision',None),'file_watcher':getattr(agent,'file_watcher',None)})
        if not r.ok:raise HTTPException(status_code=400,detail=r.to_dict())
        return r.to_dict()
    import re as _re_sid
    _SID_RE=_re_sid.compile(r'^[A-Za-z0-9_-]{1,128}$')
    def _session_fp(sid):
        if not _SID_RE.match(sid or ''):raise HTTPException(status_code=400,detail='invalid session id')
        root=Path(store.root).resolve();fp=(root/f'{sid}.jsonl').resolve()
        if not (root in fp.parents):raise HTTPException(status_code=400,detail='invalid session id')
        return fp
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
        fp=_session_fp(sid)
        if not fp.exists():raise HTTPException(status_code=404,detail=f'session {sid!r} not found')
        try:
            lines=fp.read_text(encoding='utf-8').strip().splitlines()[-limit:]
            turns=[]
            for ln in lines:
                try:turns.append(_j.loads(ln))
                except Exception:pass
            return {'session_id':sid,'turns_n':len(turns),'turns':turns,'path':_egress(str(fp))}
        except Exception as e:raise HTTPException(status_code=500,detail=f'read failed: {e}')
    @app.delete('/sessions/{sid}')
    def del_session(sid:str):
        _session_fp(sid)
        return {'deleted':store.delete(sid)}
    @app.get('/sessions/{sid}/export.md')
    def export_session_md(sid:str,limit:int=500):
        from fastapi.responses import PlainTextResponse
        import json as _j,datetime as _dt
        fp=_session_fp(sid)
        if not fp.exists():raise HTTPException(status_code=404,detail=f'session {sid!r} not found')
        try:
            lines=fp.read_text(encoding='utf-8').strip().splitlines()[-limit:]
            turns=[]
            for ln in lines:
                try:turns.append(_j.loads(ln))
                except Exception:pass
            md=[f'# Conversation `{sid}`',f'_{len(turns)} turn(s) · exported {_dt.datetime.now().isoformat(timespec="seconds")}_','']
            for t in turns:
                role=t.get('role','?');content=(t.get('content') or t.get('message') or '').rstrip()
                if not content:continue
                ts=t.get('ts');hdr_ts=''
                if ts:
                    try:hdr_ts=' · '+_dt.datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:pass
                meta=t.get('metadata') or {};persona=meta.get('persona') or t.get('persona');tier=meta.get('tier') or t.get('tier');cat=meta.get('category') or t.get('category')
                tags=[];_=tags.append(f'persona={persona}') if persona else None;_=tags.append(f'tier={tier}') if tier else None;_=tags.append(f'category={cat}') if cat and cat!='general' else None
                hdr_tags=(' · '+' · '.join(tags)) if tags else ''
                speaker={'user':'User','assistant':'Adam','system':'System'}.get(role,role.title())
                md.append(f'## {speaker}{hdr_ts}{hdr_tags}');md.append('');md.append(content);md.append('')
            body='\n'.join(md)
            return PlainTextResponse(content=body,media_type='text/markdown; charset=utf-8',headers={'Content-Disposition':f'attachment; filename="adam-chat-{sid[-8:]}.md"'})
        except HTTPException:raise
        except Exception as e:raise HTTPException(status_code=500,detail=f'export failed: {e}')
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
    @app.get('/persona/panel',response_class=HTMLResponse)
    def persona_panel():
        from amni.serve.persona_panel import PERSONA_PANEL_HTML
        return HTMLResponse(PERSONA_PANEL_HTML)
    @app.get('/persona/observe')
    def observe_persona(session_id:str=''):
        """Full render-state of the active persona. Lets external clients (Amni-Code side-panel, status bars) mirror /jarvis without polling N endpoints."""
        from amni.serve.persona import persona_tint,sample_sentences,PRESETS
        active=personas.for_session(session_id) if session_id else personas.get(personas._default)
        tint=persona_tint(active)
        try:sys_prompt=active.system_prompt('')[:420]
        except Exception:sys_prompt=''
        return {'active':active.to_dict(),'tint':tint,'default':personas._default,'known_count':len(personas.list_known()),'session':{'session_id':session_id or None,'session_persona':personas._session_persona.get(session_id) if session_id else None},'samples':sample_sentences(active),'system_prompt_preview':sys_prompt,'presets_count':len(PRESETS),'learned_count':len(personas._learned)}
    @app.get('/persona/{name}/export')
    def export_persona_route(name:str):
        if not personas.has(name):raise HTTPException(status_code=404,detail=f'unknown persona {name!r}')
        import time as _t
        return {'_amni_persona_format':'v1','exported_at':_t.time(),'persona':personas.get(name).to_dict()}
    @app.post('/persona/import')
    async def import_persona_route(req:Request):
        try:body=await req.json()
        except Exception:body={}
        if not isinstance(body,dict):raise HTTPException(status_code=400,detail='body must be JSON object')
        data=body.get('persona') if isinstance(body.get('persona'),dict) else body
        if not isinstance(data,dict):raise HTTPException(status_code=400,detail='no persona object found')
        p=personas.import_persona(data,new_name=body.get('rename'),overwrite=bool(body.get('overwrite',False)))
        if p is None:
            existing=body.get('rename') or data.get('name','')
            if not body.get('overwrite') and existing and personas.has(existing):raise HTTPException(status_code=409,detail=f'persona {existing!r} already exists — pass overwrite=true to replace')
            raise HTTPException(status_code=400,detail='invalid persona data (missing description, bad name, or unparseable dims)')
        return {'persona':p.to_dict(),'imported':True}
    @app.delete('/persona/{name}')
    def delete_persona_route(name:str):
        if not personas.has(name):raise HTTPException(status_code=404,detail=f'unknown persona {name!r}')
        ok=personas.delete_persona(name)
        if not ok:raise HTTPException(status_code=400,detail=f'{name!r} is a preset — only learned/edited/imported overrides can be deleted')
        return {'deleted':name,'default':personas._default}
    @app.patch('/persona/{name}')
    async def edit_persona(name:str,req:Request):
        try:body=await req.json()
        except Exception:body={}
        if not personas.has(name):raise HTTPException(status_code=404,detail=f'unknown persona {name!r}')
        p=personas.update_persona(name,body or {})
        if p is None:raise HTTPException(status_code=400,detail='no editable fields supplied or update failed')
        return {'persona':p.to_dict(),'updated':True}
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
    unified_web.mount(app)
    memory_endpoints.mount(app,agent)
    mode_endpoints.mount(app,adam)
    task_endpoints.mount(app,agent)
    vision_endpoints.mount(app,agent)
    voice_endpoints.mount(app,agent)
    amni_chat_bridge.mount(app,agent)
    model_installer.mount(app)
    trace_endpoints.mount(app,agent) if trace_endpoints is not None else None
    try:
        from amni.serve import guardian_service;guardian_service.mount(app,agent)
        print(f'[amni_serve]   Guardian app:  http://{args.host}:{args.port}/guardian  (self-improve + dispatch & discussion, phone-ready)',flush=True)
    except Exception as _ge:print(f'[amni_serve] guardian_service mount failed (non-fatal): {_ge}',flush=True)
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
