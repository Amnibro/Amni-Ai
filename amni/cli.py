"""amni — single CLI entry point. Subcommands: init, serve, chat, ask, scan, code, persona, publish, pull, reflect, stats, personas.
Mirrors Amni-Prism's CLI shape so installing both packages gives a coherent toolset."""
import argparse,sys,os,json,time,webbrowser,threading
from pathlib import Path
from amni.bootstrap import load_config,save_config,ensure_dirs,download_bake,download_base_model,detect_bake,detect_model,bake_has_runtime_metadata,CONFIG_DIR,CONFIG_FILE,is_first_run,mark_first_run_done,DEFAULT_PORT,DEFAULT_HOST
def _add_common_adam(p):
    cfg=load_config()
    default_bake=cfg.get('bake') or str(CONFIG_DIR/'bakes'/'granite41_3b_gf17')
    default_model=cfg.get('model') or cfg.get('bake') or default_bake
    p.add_argument('--bake',default=os.environ.get('AMNI_BAKE',default_bake))
    p.add_argument('--model',default=os.environ.get('AMNI_MODEL',default_model))
    p.add_argument('--lessons',default=cfg.get('lessons') or 'experiences/adam_lessons.npz')
    p.add_argument('--lut-root',default=cfg.get('lut_root') or 'experiences/adam_lut')
def cmd_init(args):
    cfg=load_config()
    print(f'\nAdam first-run initialization',flush=True)
    print(f'Config: {CONFIG_FILE}',flush=True)
    print(f'Detected bake: {cfg.get("bake") or "(none)"}',flush=True)
    print(f'Detected model: {cfg.get("model") or "(none)"}',flush=True)
    if not args.skip_model and not cfg.get('bake'):
        if args.non_interactive or _ask('Download Granite-4.1-3B GF(17) bake from HF (~13 GB, one-time)?'):
            b=download_bake(cfg)
            if b:cfg['bake']=str(b)
    if cfg.get('bake') and bake_has_runtime_metadata(cfg.get('bake')):
        cfg['model']=str(cfg['bake'])
        print(f'[init] runtime model dir = bake dir ({cfg["bake"]}) — Adam ships as a self-contained GF(17) artifact; no upstream download needed.',flush=True)
    ensure_dirs(cfg)
    cfg['first_run_done']=True
    save_config(cfg)
    print(f'\n[init] saved config to {CONFIG_FILE}',flush=True)
    print(f'[init] next: `amni serve` (no flags needed) OR `amni chat`',flush=True)
def _ask(prompt:str)->bool:
    try:r=input(f'  {prompt} [Y/n] ').strip().lower()
    except EOFError:return False
    return r in ('','y','yes')
def _wait_ready(url:str,timeout:float=300.0)->bool:
    import urllib.request
    hz=url.rstrip('/')+'/healthz';t0=time.time()
    while time.time()-t0<timeout:
        try:
            with urllib.request.urlopen(hz,timeout=4) as r:
                if r.status==200:return True
        except Exception:pass
        time.sleep(1.5)
    return False
def _find_chromium():
    import shutil
    cands=[os.environ.get('AMNI_BROWSER',''),
           r'C:\Program Files\Google\Chrome\Application\chrome.exe',
           r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
           r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
           r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
           shutil.which('chrome'),shutil.which('chromium'),shutil.which('msedge'),
           '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
           shutil.which('google-chrome'),shutil.which('chromium-browser')]
    for c in cands:
        if c and Path(c).exists():return c
    return None
def _open_browser_when_ready(url:str,app_window:bool=False):
    import subprocess,tempfile
    def _go():
        ok=_wait_ready(url)
        if not ok:print(f'[serve] browser auto-open skipped — server not ready in time. Open {url} manually.',flush=True);return
        if app_window:
            exe=_find_chromium()
            if exe:
                prof=str(Path(tempfile.gettempdir())/'amni_adam_app')
                try:
                    subprocess.Popen([exe,f'--app={url}',f'--user-data-dir={prof}','--window-size=1280,860','--no-first-run','--no-default-browser-check'])
                    print(f'[serve] opened Adam in a standalone app window ({Path(exe).name}).',flush=True);return
                except Exception as e:print(f'[serve] app-window launch failed ({e}); falling back to browser tab.',flush=True)
            else:print('[serve] no Chrome/Edge found for --app-window; opening a normal browser tab.',flush=True)
        try:webbrowser.open(url)
        except Exception:pass
    threading.Thread(target=_go,daemon=True).start()
def _open_browser_delayed(url:str,delay:float=2.5):
    _open_browser_when_ready(url,app_window=False)
def _print_serve_banner(host:str,port:int,workdir:str=''):
    base=f'http://{host}:{port}' if host not in ('0.0.0.0','::') else f'http://localhost:{port}'
    enc=(getattr(sys.stdout,'encoding','') or '').lower()
    uni=any(s in enc for s in ('utf','u8','u-8'))
    tl,tr,bl,br,h,v,lm,rm='┌','┐','└','┘','─','│','├','┤' if uni else (None,)*8
    if not uni:tl,tr,bl,br,h,v,lm,rm='+','+','+','+','-','|','+','+'
    dash=h*59;dash_lm=h*59
    lines=[
        '',
        f'  {tl}{dash}{tr}',
        f'  {v}  Adam -- local AI server                                 {v}',
        f'  {lm}{dash_lm}{rm}',
        f'  {v}  chat ui      {base:<42} {v}',
        f'  {v}  jarvis mode  {base+"/jarvis":<42} {v}',
        f'  {v}  memory       {base+"/memory":<42} {v}',
        f'  {v}  api docs     {base+"/docs":<42} {v}',
        f'  {v}  health       {base+"/health":<42} {v}',
        f'  {lm}{dash_lm}{rm}',
        f'  {v}  Ollama drop-in: OLLAMA_HOST={base:<28} {v}',
        f'  {v}  OpenAI drop-in: OPENAI_BASE_URL={base+"/v1":<24} {v}',
    ]
    if workdir:lines.append(f'  {v}  workdir      {workdir[:42]:<42} {v}')
    lines.append(f'  {bl}{dash}{br}')
    lines.append('')
    try:print('\n'.join(lines),flush=True)
    except UnicodeEncodeError:
        ascii_lines=['' if not l.strip() else l.encode('ascii','replace').decode('ascii') for l in lines]
        print('\n'.join(ascii_lines),flush=True)
def cmd_serve(args):
    cfg=load_config()
    if is_first_run() and not (Path(args.bake).exists() or Path(cfg.get('bake') or '').exists()):
        print('[serve] First run detected — running `amni init` first.',flush=True)
        class _A:non_interactive=True;skip_model=False
        cmd_init(_A())
        cfg=load_config()
        if cfg.get('bake'):args.bake=cfg['bake']
        if cfg.get('model'):args.model=cfg['model']
    sys.argv=['amni_serve.py','--port',str(args.port),'--host',args.host,'--bake',args.bake,'--model',args.model,'--lessons',args.lessons,'--lut-root',args.lut_root,'--conv-root',args.conv_root,'--audit-log',args.audit_log,'--persona-bank',args.persona_bank]
    if args.workdir:sys.argv+=['--workdir',args.workdir]
    for r in (args.root or []):sys.argv+=['--root',r]
    if args.unrestricted_files:sys.argv.append('--unrestricted-files')
    if args.seed or not Path(args.lessons).exists():sys.argv.append('--seed')
    if args.cors:sys.argv.append('--cors')
    if args.default_persona:sys.argv+=['--default-persona',args.default_persona]
    if args.no_persona:sys.argv.append('--no-persona')
    _aw=getattr(args,'app_window',False)
    if getattr(args,'open_browser',False) or _aw:_open_browser_when_ready(f'http://{args.host}:{args.port}/',app_window=_aw)
    _print_serve_banner(args.host,args.port,workdir=getattr(args,'workdir','') or '')
    from scripts import amni_serve
    amni_serve.main()
def cmd_code(args):
    cwd=Path(args.path or os.getcwd()).resolve()
    if not cwd.exists():print(f'[code] path not found: {cwd}',flush=True);sys.exit(1)
    project_type=_detect_project(cwd)
    print(f'[code] project root: {cwd}',flush=True)
    print(f'[code] detected type: {project_type}',flush=True)
    cfg=load_config()
    args.workdir=str(cwd)
    args.root=(args.root or [])+[str(cwd)]
    args.unrestricted_files=False
    args.cors=True
    args.open_browser=True
    args.default_persona=args.persona or 'mentor'
    args.no_persona=False
    if not args.bake or not Path(args.bake).exists():args.bake=cfg.get('bake') or args.bake
    if not args.model or not Path(args.model).exists():args.model=cfg.get('model') or args.model
    os.environ['AMNI_CODE_MODE']='1'
    os.environ['AMNI_PROJECT_ROOT']=str(cwd)
    os.environ['AMNI_PROJECT_TYPE']=project_type
    print(f'[code] launching server with workdir={cwd}, persona={args.default_persona}, code_mode=1',flush=True)
    cmd_serve(args)
def _detect_project(p:Path)->str:
    if (p/'package.json').exists():return 'node-js'
    if (p/'Cargo.toml').exists():return 'rust'
    if (p/'go.mod').exists():return 'go'
    if (p/'pyproject.toml').exists() or (p/'setup.py').exists() or (p/'requirements.txt').exists():return 'python'
    if (p/'pom.xml').exists() or (p/'build.gradle').exists():return 'java'
    if (p/'CMakeLists.txt').exists() or (p/'Makefile').exists():return 'c-cpp'
    if (p/'index.html').exists():return 'web-static'
    if (p/'.git').exists():return 'git-repo'
    return 'unknown'
def cmd_chat(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore,PersonaStore
    from amni.serve.skills import default_registry
    if getattr(args,'list_sessions',False):
        store=ConversationStore(root=args.conv_root);sessions=store.list_sessions()
        if not sessions:print('(no prior sessions)',flush=True);return
        print(f'Recent sessions ({len(sessions)}):',flush=True)
        for s in sessions[:20]:
            ts=time.strftime('%Y-%m-%d %H:%M',time.localtime(s['mtime']));kb=round(s['size']/1024,1)
            print(f'  {s["session_id"]}  {ts}  {kb}kb',flush=True)
        return
    print('[amni] booting Adam... ',end='',flush=True)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    skills=default_registry(workdir=args.workdir,unrestricted=args.unrestricted_files,audit_log='logs/agent_skill_calls.jsonl')
    store=ConversationStore(root=args.conv_root)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.persona:personas.set_default(args.persona)
    agent=AmniAgent(adam=adam,skills=skills,store=store,personas=personas,workdir=args.workdir)
    print(f'ready. lessons={adam.stats().get("lessons_n",0)} skills={len(skills.list_skills())} persona={personas._default}',flush=True)
    sid=None
    if getattr(args,'session',None):sid=args.session
    elif getattr(args,'resume',False):
        sess=store.list_sessions()
        if sess:sid=sess[0]['session_id'];print(f'(resuming session {sid} from {time.strftime("%Y-%m-%d %H:%M",time.localtime(sess[0]["mtime"]))})',flush=True)
        else:print('(no prior session to resume — starting new)',flush=True)
    if sid:
        conv=store.get(sid);recent=conv.recent(4) if hasattr(conv,'recent') else []
        if recent:
            print('(last turns:)',flush=True)
            for t in recent:
                role=t.get('role','');content=(t.get('content') or '')[:120]
                if role=='user' and content:print(f'  > {content}',flush=True)
                elif role=='assistant' and content:print(f'    {content}',flush=True)
    print('Type /quit to exit, /persona <name> to switch, /skills to list, /stats for stats, /sessions to list past sessions.\n',flush=True)
    while True:
        try:msg=input('> ').strip()
        except (EOFError,KeyboardInterrupt):print('\nbye!',flush=True);break
        if not msg:continue
        if msg in ('/quit','/exit'):print('bye!',flush=True);break
        if msg=='/skills':
            for s in agent.list_skills():print(f'  {s["name"]:<12} {s["desc"]}',flush=True)
            continue
        if msg=='/stats':print(json.dumps(agent.stats(),indent=2,default=str),flush=True);continue
        if msg.startswith('/persona '):
            name=msg.split(None,1)[1].strip()
            if not personas.has(name):p=personas.learn(name)
            else:p=personas.get(name)
            if sid:personas.assign_session(sid,name)
            else:personas.set_default(name)
            print(f'(persona -> {p.name}, source={p.source})\n  {p.description[:200]}\n',flush=True);continue
        if msg=='/new':sid=None;print('(new session)',flush=True);continue
        if msg=='/sessions':
            sess=store.list_sessions()[:10]
            if not sess:print('(no sessions yet)',flush=True)
            else:
                for s in sess:
                    ts=time.strftime('%Y-%m-%d %H:%M',time.localtime(s['mtime']));mark=' *' if s['session_id']==sid else ''
                    print(f'  {s["session_id"]}  {ts}  {round(s["size"]/1024,1)}kb{mark}',flush=True)
            continue
        if msg.startswith('/resume '):
            target=msg.split(None,1)[1].strip()
            sess=store.list_sessions();match=[s for s in sess if s['session_id'].startswith(target)]
            if not match:print(f'(no session matches {target!r})',flush=True);continue
            sid=match[0]['session_id'];print(f'(resumed {sid})',flush=True);continue
        r=agent.chat(msg,session_id=sid)
        sid=r.get('session_id') or sid
        print(f'\n{r.get("answer")}\n',flush=True)
        print(f'  [tier={r.get("tier")} persona={r.get("persona")} category={r.get("category")} tokens={r.get("tokens")} wall={r.get("wall_s")}s]\n',flush=True)
def cmd_export_session(args):
    from amni.serve.conversation import ConversationStore
    store=ConversationStore(root=args.conv_root)
    target=(args.session or '').strip()
    sess=store.list_sessions();matches=[s for s in sess if s['session_id']==target or s['session_id'].startswith(target)] if target else (sess[:1] if sess else [])
    if not matches:print(f'[export] no session matches {target!r}',flush=True);sys.exit(1)
    sid=matches[0]['session_id'];conv=store.get(sid);turns=conv.turns
    if args.strip_personal:turns=[t for t in turns if not t.get('is_personal')]
    fmt=(args.format or 'md').lower()
    if fmt=='md':content=_session_to_markdown(sid,turns)
    elif fmt=='json':content=json.dumps({'session_id':sid,'exported_at':time.time(),'n_turns':len(turns),'turns':turns},indent=2,default=str)
    elif fmt=='jsonl':content='\n'.join(json.dumps(t,default=str) for t in turns)
    else:print(f'[export] unknown format {fmt!r}; use md|json|jsonl',flush=True);sys.exit(1)
    out=Path(args.output) if args.output else Path(f'{sid}.{fmt}')
    out.write_text(content,encoding='utf-8')
    n_personal=sum(1 for t in conv.turns if t.get('is_personal'))
    print(f'[export] wrote {len(turns)} turn(s) ({len(content)} bytes) to {out}',flush=True)
    if args.strip_personal and n_personal:print(f'[export] stripped {n_personal} turn(s) flagged personal',flush=True)
def _session_to_markdown(sid:str,turns:list)->str:
    head=f'# Amni-Ai session {sid}\n\nExported: {time.strftime("%Y-%m-%d %H:%M:%S")}  \nTurns: {len(turns)}\n\n---\n'
    lines=[head]
    for t in turns:
        role=t.get('role','?').upper();content=t.get('content','');ts=t.get('ts')
        ts_s=time.strftime('%H:%M:%S',time.localtime(ts)) if ts else ''
        marks=' (personal)' if t.get('is_personal') else ''
        lines.append(f'\n## {role}{(" · "+ts_s) if ts_s else ""}{marks}\n\n{content}\n')
    return '\n'.join(lines)
def cmd_import_session(args):
    from amni.serve.conversation import ConversationStore
    src=Path(args.input)
    if not src.exists():print(f'[import] file not found: {src}',flush=True);sys.exit(1)
    raw=src.read_text(encoding='utf-8')
    turns=_parse_session_file(raw,src.suffix.lstrip('.').lower())
    if not turns:print('[import] no turns found in input file',flush=True);sys.exit(1)
    store=ConversationStore(root=args.conv_root);conv=store.get(args.session_id or None)
    for t in turns:
        role=t.get('role','user') if isinstance(t,dict) else 'user'
        content=t.get('content','') if isinstance(t,dict) else str(t)
        meta={k:v for k,v in t.items() if k not in ('role','content','ts','is_personal')} if isinstance(t,dict) else None
        conv.append(role,content,meta=meta or None)
    print(f'[import] imported {len(turns)} turn(s) into session {conv.session_id}',flush=True)
    print(f'[import] resume with: amni chat --session {conv.session_id}',flush=True)
def _parse_session_file(raw:str,ext:str)->list:
    raw=raw.strip()
    if not raw:return []
    if ext=='jsonl':
        out=[]
        for line in raw.splitlines():
            line=line.strip()
            if not line:continue
            try:out.append(json.loads(line))
            except Exception:pass
        return out
    if ext=='json':
        try:obj=json.loads(raw)
        except Exception:return []
        if isinstance(obj,dict) and 'turns' in obj:return list(obj['turns'])
        if isinstance(obj,list):return list(obj)
        return []
    if ext in ('md','markdown'):return _markdown_to_turns(raw)
    try:obj=json.loads(raw)
    except Exception:return _markdown_to_turns(raw)
    if isinstance(obj,dict) and 'turns' in obj:return list(obj['turns'])
    if isinstance(obj,list):return list(obj)
    return []
def _markdown_to_turns(md:str)->list:
    import re
    turns=[];parts=re.split(r'\n## +(USER|ASSISTANT|SYSTEM|BOT)\b[^\n]*\n',md)
    if len(parts)<3:return []
    for i in range(1,len(parts),2):
        role=parts[i].lower();body=parts[i+1].strip() if i+1<len(parts) else ''
        if role=='bot':role='assistant'
        if body:turns.append({'role':role,'content':body})
    return turns
def cmd_ask(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore,PersonaStore
    from amni.serve.skills import default_registry
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    skills=default_registry(workdir=args.workdir,unrestricted=args.unrestricted_files,audit_log=None)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.persona:personas.set_default(args.persona)
    agent=AmniAgent(adam=adam,skills=skills,store=ConversationStore(),personas=personas,workdir=args.workdir)
    r=agent.chat(' '.join(args.query))
    print(r.get('answer'),flush=True)
    if args.json:print(json.dumps({k:v for k,v in r.items() if k!='answer'},indent=2,default=str),flush=True)
def cmd_scan(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve.skills import default_registry
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    skills=default_registry(workdir=args.workdir,unrestricted=True,audit_log=None)
    r=skills.call('scan',{'path':args.path,'glob':args.glob,'max_files':args.max_files,'distill':args.distill,'max_chars_per_file':args.max_chars_per_file},ctx={'adam':adam})
    print(json.dumps(r.to_dict(),indent=2,default=str),flush=True)
def cmd_publish(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve.federated import publish_lessons
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    r=publish_lessons(adam,codex_dir=args.codex,contributor_id=args.contributor,min_confidence=args.min_confidence,domain=args.domain,limit=args.limit,dry_run=args.dry_run)
    print(json.dumps(r,indent=2,default=str),flush=True)
def cmd_pull(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve.federated import pull_lessons
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    r=pull_lessons(adam,codex_dir=args.codex,domain=args.domain,limit=args.limit,dry_run=args.dry_run)
    print(json.dumps(r,indent=2,default=str),flush=True)
def cmd_reflect(args):
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve.reflection import reflect_loop
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    reflect_loop(adam,interval_sec=args.interval,max_per_cycle=args.max_per_cycle,min_age_sec=args.min_age,one_shot=args.once)
def cmd_teach_cot(args):
    from amni.adam import Adam
    from amni.seeds import COT_LESSONS,CODING_LESSONS,JS_LESSONS,SQL_LESSONS,DEVOPS_LESSONS,CREATIVE_LESSONS,FACTS_LESSONS,ADVANCED_LESSONS,RUST_LESSONS,CONCURRENCY_LESSONS,ALGO_ADV_LESSONS,PYTHON_ADV_LESSONS,GO_LESSONS,FRONTEND_LESSONS,MOBILE_LESSONS,DATA_ENG_LESSONS,LEETCODE_LESSONS,ML_LESSONS,SECURITY_DEEP_LESSONS,DISTRIBUTED_LESSONS,PERFORMANCE_LESSONS,ARCHITECTURE_LESSONS,NETWORKING_LESSONS,GAMEDEV_LESSONS,EMBEDDED_LESSONS,MATH_ADV_LESSONS,FACTS_EXT_LESSONS,PYTHON_LIBS_LESSONS,AI_RAG_LESSONS,LEETCODE_HARD_LESSONS,PARAPHRASES_LESSONS,DEBUG_ADV_LESSONS,ALL_LESSONS
    bank={'cot':COT_LESSONS,'coding':CODING_LESSONS,'js':JS_LESSONS,'sql':SQL_LESSONS,'devops':DEVOPS_LESSONS,'creative':CREATIVE_LESSONS,'facts':FACTS_LESSONS,'advanced':ADVANCED_LESSONS,'rust':RUST_LESSONS,'concurrency':CONCURRENCY_LESSONS,'algo_adv':ALGO_ADV_LESSONS,'python_adv':PYTHON_ADV_LESSONS,'go':GO_LESSONS,'frontend':FRONTEND_LESSONS,'mobile':MOBILE_LESSONS,'data_eng':DATA_ENG_LESSONS,'leetcode':LEETCODE_LESSONS,'ml':ML_LESSONS,'security_deep':SECURITY_DEEP_LESSONS,'distributed':DISTRIBUTED_LESSONS,'performance':PERFORMANCE_LESSONS,'architecture':ARCHITECTURE_LESSONS,'networking':NETWORKING_LESSONS,'gamedev':GAMEDEV_LESSONS,'embedded':EMBEDDED_LESSONS,'math_adv':MATH_ADV_LESSONS,'facts_ext':FACTS_EXT_LESSONS,'python_libs':PYTHON_LIBS_LESSONS,'ai_rag':AI_RAG_LESSONS,'leetcode_hard':LEETCODE_HARD_LESSONS,'paraphrases':PARAPHRASES_LESSONS,'debug_adv':DEBUG_ADV_LESSONS,'all':ALL_LESSONS}.get(args.bank,ALL_LESSONS)
    print(f'[teach-cot] loading bank={args.bank!r} ({len(bank)} curated lessons)',flush=True)
    if args.dry_run:
        print('[teach-cot] DRY RUN — sample first 3:',flush=True)
        for q,a in bank[:3]:print(f'  Q: {q!r}\n  A: {a[:160]!r}\n',flush=True)
        print(f'[teach-cot] would teach {len(bank)} lessons. Re-run without --dry-run.',flush=True);return
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    sl=adam.sem_lut;before=len(sl._raw)
    seen_qs=set(q for q,_ in sl._raw)
    added=0;skipped=0
    for q,a in bank:
        if q in seen_qs:skipped+=1;continue
        sl.add(q,a);added+=1;seen_qs.add(q)
    if added>0:
        print(f'[teach-cot] added {added} new (skipped {skipped} duplicates). Fitting + saving...',flush=True)
        sl.fit();adam.save_lessons()
    after=len(sl._raw)
    print(f'[teach-cot] DONE. lessons_n: {before} -> {after} (+{after-before})',flush=True)
def cmd_stats(args):
    if getattr(args,'watch',False):return _stats_watch_loop(args)
    if getattr(args,'remote',False) or getattr(args,'url',None):return _stats_remote_snapshot(args)
    from amni.adam import Adam
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    print(json.dumps(adam.stats(),indent=2,default=str),flush=True)
def _stats_remote_url(args)->str:
    base=getattr(args,'url',None) or 'http://127.0.0.1:7700'
    if not base.startswith('http'):base='http://'+base
    return base.rstrip('/')
def _fetch_json(url:str,timeout:float=1.8):
    import urllib.request,urllib.error
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
    except Exception:return None
def _stats_remote_snapshot(args):
    base=_stats_remote_url(args)
    print(_render_dashboard(base,_collect_remote(base)),flush=True)
def _collect_remote(base:str)->dict:
    return {'learning':_fetch_json(base+'/learning/stats'),'skill_stats':_fetch_json(base+'/memory/skill-stats'),'coach':_fetch_json(base+'/memory/coach'),'metrics':_fetch_json(base+'/memory/metrics'),'reflection':_fetch_json(base+'/memory/self-reflection'),'proposals':_fetch_json(base+'/memory/self-improvement?limit=5')}
def _stats_watch_loop(args):
    base=_stats_remote_url(args);interval=max(1,int(getattr(args,'interval',3) or 3))
    try:
        while True:
            data=_collect_remote(base)
            sys.stdout.write('\x1b[2J\x1b[H');sys.stdout.write(_render_dashboard(base,data));sys.stdout.write(f'\n  (refresh every {interval}s — Ctrl+C to quit)\n');sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:print('\n(stopped)',flush=True)
def _fmt_num(v,unit:str='')->str:
    if v is None:return '?'
    try:n=float(v)
    except Exception:return str(v)
    return f'{int(n)}{unit}' if n==int(n) else f'{n:.2f}{unit}'
def _bar(pct:float,width:int=20)->str:
    pct=max(0,min(1,pct or 0));filled=int(round(pct*width));return '█'*filled+'░'*(width-filled)
def _render_dashboard(base:str,data:dict)->str:
    lines=[];now=time.strftime('%H:%M:%S')
    lines.append(f'  ╔══════════════════════════════════════════════════════════════════╗')
    lines.append(f'  ║  Adam — live stats · {base:<32}{now:>11}  ║')
    lines.append(f'  ╠══════════════════════════════════════════════════════════════════╣')
    ld=data.get('learning') or {}
    if ld:
        new=ld.get('qa_pairs_new',0);rein=ld.get('qa_pairs_reinforced',0);urls=ld.get('urls_ingested',0);sleeps=ld.get('sleep_passes',0);active=ld.get('active',True);topic=(ld.get('current_topic') or '—')[:30]
        status='active' if active else 'paused'
        lines.append(f'  ║  LEARNING DAEMON ({status:<8})                                       ║')
        lines.append(f'  ║    facts new+reinforced  {_fmt_num(new):>6} new / {_fmt_num(rein):>6} reinforced            ║')
        lines.append(f'  ║    urls ingested         {_fmt_num(urls):>6} · sleep passes {_fmt_num(sleeps):>4}             ║')
        lines.append(f'  ║    current topic         {topic:<38}║')
    else:
        lines.append(f'  ║  LEARNING DAEMON              (endpoint unreachable)             ║')
    lines.append(f'  ╟──────────────────────────────────────────────────────────────────╢')
    sk=data.get('skill_stats') or {}
    if sk and sk.get('log_exists') is not False:
        totals=sk.get('totals') or {}
        n_calls=totals.get('n_calls',0);ok_rate=totals.get('overall_ok_rate');avg=totals.get('avg_ms',0)
        ok_pct=f'{(ok_rate or 0)*100:.0f}%' if ok_rate is not None else '?'
        lines.append(f'  ║  SKILLS                                                          ║')
        lines.append(f'  ║    total calls {_fmt_num(n_calls):>7}  ok-rate {ok_pct:>5}  avg {_fmt_num(avg,"ms"):>7}              ║')
        skills_map=sk.get('skills') or {}
        if isinstance(skills_map,dict):
            top_items=sorted(skills_map.items(),key=lambda kv:-int(kv[1].get('n_calls') or 0))[:3]
            for name,r in top_items:
                n=r.get('n_calls',0);ok=r.get('ok',0);rate=(ok/n) if n else 0
                rate_s=f'{rate*100:.0f}%';p90=r.get('p90',0)
                lines.append(f'  ║      {name[:14]:<14} {_fmt_num(n):>5}× · ok {rate_s:>4} · p90 {_fmt_num(p90,"ms"):>7}      ║')
    else:
        lines.append(f'  ║  SKILLS                       (endpoint unreachable or empty)    ║')
    lines.append(f'  ╟──────────────────────────────────────────────────────────────────╢')
    co=data.get('coach') or {};streak=co.get('streak') or {}
    cur=streak.get('current',0);longest=streak.get('longest',0);today=streak.get('today_count',0)
    n_topics=len(co.get('topics') or []) if isinstance(co.get('topics'),list) else 0
    if cur or longest or n_topics:
        flame='⚡' if cur>=14 else ('🔥' if cur>=3 else '·')
        lines.append(f'  ║  COACH                                                           ║')
        lines.append(f'  ║    {flame} streak {cur}d · longest {longest}d · today {today} · {n_topics} topic(s){"":<14}║')
    re=data.get('reflection') or {}
    if re:
        cyc=re.get('cycle_count',0);last=re.get('last_subsystem') or '—';nxt=re.get('next_subsystem') or '—'
        eta=int(re.get('seconds_until_eligible') or 0)
        lines.append(f'  ║  SELF-REFLECTION                                                 ║')
        lines.append(f'  ║    cycles {cyc} · last {last[:16]:<16} · next {nxt[:16]:<16}  ║')
        if eta>0:lines.append(f'  ║    next cycle eligible in {eta//3600}h{(eta%3600)//60}m{"":<38}║')
    mt=data.get('metrics') or {}
    if mt and (mt.get('snapshot_count') or 0)>0:
        lines.append(f'  ║  METRIC SNAPSHOTS                                                ║')
        lines.append(f'  ║    {mt.get("snapshot_count",0)} snapshot(s) · last {(mt.get("last_run_iso") or "—"):<28}    ║')
    pr=data.get('proposals') or {};pst=pr.get('stats') if isinstance(pr,dict) else None
    if pst:
        bs=pst.get('by_status') or {}
        proposed=bs.get('proposed',0);attempted=bs.get('attempted',0);validated=bs.get('validated',0);deployed=bs.get('deployed',0)
        lines.append(f'  ║  PROPOSALS                                                       ║')
        lines.append(f'  ║    proposed {proposed:>3} · attempted {attempted:>3} · validated {validated:>3} · deployed {deployed:>3}      ║')
    lines.append(f'  ╚══════════════════════════════════════════════════════════════════╝')
    return '\n'.join(lines)
def cmd_reminders(args):
    action=(getattr(args,'action',None) or 'list').strip().lower()
    if action=='dismiss' and not getattr(args,'id',None) and getattr(args,'text',None):
        args.id=args.text;args.text=None
    url=getattr(args,'url',None)
    if url and not url.startswith('http'):url='http://'+url
    if url:url=url.rstrip('/')
    if action=='list':return _reminders_list(args,url)
    if action=='due':return _reminders_due(args,url)
    if action=='stats':return _reminders_stats(args,url)
    if action=='add':return _reminders_add(args,url)
    if action=='dismiss':return _reminders_dismiss(args,url)
    print(f'[reminders] unknown action {action!r}; use list|due|stats|add|dismiss',flush=True);sys.exit(1)
def _reminders_fetch(args,url):
    if url:
        r=_fetch_json(url+f'/memory/reminders?limit={int(args.limit or 50)}'+(f'&session_id={args.session}' if args.session else ''))
        if r is None:print(f'[reminders] unreachable: {url}',flush=True);sys.exit(1)
        return r.get('reminders') or [],r.get('due') or [],r.get('stats') or {}
    from amni.serve.reminders import list_active,list_due,stats
    return list_active(session_id=args.session or None,limit=int(args.limit or 50)),list_due(),stats()
def _reminders_list(args,url):
    items,due,stats=_reminders_fetch(args,url)
    if args.json:print(json.dumps({'reminders':items,'due':due,'stats':stats},indent=2,default=str),flush=True);return
    if not items:print('\n  No active reminders. Add one with `amni reminders add "text"`.\n',flush=True);return
    due_ids={r.get('id') for r in due}
    print(f'\n  {len(items)} active reminder(s) · {stats.get("due_now",0)} due now\n',flush=True)
    for r in items:
        rid=r.get('id','?');text=r.get('text','');due_iso=r.get('due_iso') or '(no due)';marker='!' if rid in due_ids else ' '
        print(f'  {marker} {rid}  {due_iso}  {text}',flush=True)
    print('',flush=True)
def _reminders_due(args,url):
    _,due,_=_reminders_fetch(args,url)
    if args.json:print(json.dumps({'due':due},indent=2,default=str),flush=True);return
    if not due:print('\n  Nothing due right now.\n',flush=True);return
    print(f'\n  {len(due)} reminder(s) due:\n',flush=True)
    for r in due:print(f'  ! {r.get("id","?")}  {r.get("due_iso","?")}  {r.get("text","")}',flush=True)
    print('',flush=True)
def _reminders_stats(args,url):
    _,_,stats=_reminders_fetch(args,url)
    if args.json:print(json.dumps(stats,indent=2,default=str),flush=True);return
    print(f'\n  active   {stats.get("active",0)}',flush=True)
    print(f'  due now  {stats.get("due_now",0)}',flush=True)
    print(f'  with due {stats.get("with_due",0)}\n',flush=True)
def _reminders_add(args,url):
    text=(args.text or '').strip()
    if not text:print('[reminders add] text required (positional arg or --text)',flush=True);sys.exit(1)
    if url:
        import urllib.request
        body=json.dumps({'text':text,'session_id':args.session or ''}).encode('utf-8')
        try:
            req=urllib.request.Request(url+'/memory/reminders',data=body,headers={'content-type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[reminders] add failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.reminders import add
        r=add(text=text,session_id=args.session or '')
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[reminders] {r["error"]}',flush=True);sys.exit(1)
    tail=f' · due {r.get("due_iso")}' if r.get('due_iso') else ' · no due parsed'
    print(f'  Added {r.get("id","?")}{tail}',flush=True)
def _reminders_dismiss(args,url):
    rid=(args.id or '').strip()
    if not rid:print('[reminders dismiss] need an id',flush=True);sys.exit(1)
    if url:
        import urllib.request
        try:
            req=urllib.request.Request(url+'/memory/reminders/'+rid+'/dismiss',data=b'',method='POST')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[reminders] dismiss failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.reminders import dismiss
        r=dismiss(rid)
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[reminders] {r["error"]}',flush=True);sys.exit(1)
    print(f'  Dismissed {rid}',flush=True)
def cmd_bookmarks(args):
    action=(getattr(args,'action',None) or 'list').strip().lower()
    url=getattr(args,'url',None)
    if url and not url.startswith('http'):url='http://'+url
    if url:url=url.rstrip('/')
    if action=='list':return _bookmarks_list(args,url)
    if action=='stats':return _bookmarks_stats(args,url)
    if action=='delete':return _bookmarks_delete(args,url)
    if action=='export':return _bookmarks_export(args,url)
    if action=='add':return _bookmarks_add(args,url)
    print(f'[bookmarks] unknown action {action!r}; use list|stats|delete|export|add',flush=True);sys.exit(1)
def _bookmarks_fetch(args,url):
    if url:
        qs=f'?limit={int(args.limit or 50)}'+(f'&session_id={args.session}' if args.session else '')+(f'&search={args.search}' if args.search else '')
        r=_fetch_json(url+'/memory/bookmarks'+qs)
        if r is None:print(f'[bookmarks] unreachable: {url}',flush=True);sys.exit(1)
        return r.get('bookmarks') or [],r.get('stats') or {}
    from amni.serve.bookmarks import list_recent,stats
    return list_recent(limit=int(args.limit or 50),session_id=args.session or None,search=args.search or ''),stats()
def _bookmarks_list(args,url):
    items,stats=_bookmarks_fetch(args,url)
    if args.json:print(json.dumps({'bookmarks':items,'stats':stats},indent=2,default=str),flush=True);return
    if not items:print('\n  No bookmarks recorded.\n',flush=True);return
    print(f'\n  {len(items)} bookmark(s) · total {stats.get("total",0)}\n',flush=True)
    for bm in items:
        ts=bm.get('iso','?');bid=bm.get('id','?');u=(bm.get('user_msg') or '')[:80];a=(bm.get('bot_msg') or '')[:200];n=(bm.get('note') or '').strip()
        print(f'  {ts}  [{bid}]',flush=True);print(f'    Q: {u}',flush=True);print(f'    A: {a}',flush=True)
        if n:print(f'    note: {n}',flush=True)
        print('',flush=True)
def _bookmarks_stats(args,url):
    _,stats=_bookmarks_fetch(args,url)
    if args.json:print(json.dumps(stats,indent=2,default=str),flush=True);return
    print(f'\n  total   {stats.get("total",0)}',flush=True)
    print(f'  last    {stats.get("last_iso") or "—"}\n',flush=True)
    by=stats.get('by_persona') or {}
    if by:print('  by persona:',flush=True)
    for p,n in sorted(by.items(),key=lambda kv:-kv[1]):print(f'    {p:<20} {n}',flush=True)
    print('',flush=True)
def _bookmarks_delete(args,url):
    bid=(args.id or '').strip()
    if not bid:print('[bookmarks delete] need an id (use `amni bookmarks list` to find one)',flush=True);sys.exit(1)
    if url:
        import urllib.request
        try:
            req=urllib.request.Request(url+'/memory/bookmarks/'+bid,method='DELETE')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[bookmarks] delete failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.bookmarks import delete
        r=delete(bid)
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[bookmarks] {r["error"]}',flush=True);sys.exit(1)
    print(f'  Deleted {bid} · {r.get("remaining",0)} remaining.',flush=True)
def _bookmarks_export(args,url):
    items,stats=_bookmarks_fetch(args,url)
    if not items:print('[bookmarks export] no bookmarks to export',flush=True);return
    lines=[f'# Amni-Ai bookmarks ({len(items)} entries)',f'',f'Exported: {time.strftime("%Y-%m-%d %H:%M:%S")}','','---','']
    for bm in items:
        ts=bm.get('iso','?');u=bm.get('user_msg','');a=bm.get('bot_msg','');n=(bm.get('note') or '').strip();p=bm.get('persona','');t=bm.get('tier','')
        lines.append(f'## {ts}')
        if p or t:lines.append(f'_{p}{" · "+t if t else ""}_')
        lines.append('')
        lines.append('**Q:** '+u);lines.append('')
        lines.append('**A:** '+a);lines.append('')
        if n:lines.append('> '+n);lines.append('')
        lines.append('---');lines.append('')
    content='\n'.join(lines)
    out=getattr(args,'output',None)
    if out:Path(out).write_text(content,encoding='utf-8');print(f'  Wrote {len(items)} bookmark(s) to {out}',flush=True)
    else:print(content,flush=True)
def _bookmarks_add(args,url):
    user_msg=(args.user or '').strip();bot_msg=(args.bot or '').strip();note=(args.note or '').strip();persona=(args.persona or '').strip();tier=(args.tier or '').strip();session=(args.session or '').strip()
    if not bot_msg:print('[bookmarks add] --bot is required',flush=True);sys.exit(1)
    if url:
        import urllib.request
        body=json.dumps({'session_id':session,'user_msg':user_msg,'bot_msg':bot_msg,'note':note,'tier':tier,'persona':persona}).encode('utf-8')
        try:
            req=urllib.request.Request(url+'/memory/bookmarks',data=body,headers={'content-type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[bookmarks] add failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.bookmarks import add
        r=add(session_id=session,user_msg=user_msg,bot_msg=bot_msg,note=note,tier=tier,persona=persona)
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[bookmarks] {r["error"]}',flush=True);sys.exit(1)
    print(f'  Added {r.get("id","?")} at {r.get("iso","?")}.',flush=True)
def cmd_notes(args):
    action=(getattr(args,'action',None) or 'list').strip().lower()
    if action=='delete' and not getattr(args,'id',None) and getattr(args,'text',None):
        args.id=args.text;args.text=None
    url=getattr(args,'url',None)
    if url and not url.startswith('http'):url='http://'+url
    if url:url=url.rstrip('/')
    if action=='list':return _notes_list(args,url)
    if action=='add':return _notes_add(args,url)
    if action=='delete':return _notes_delete(args,url)
    if action=='tags':return _notes_tags(args,url)
    if action=='stats':return _notes_stats(args,url)
    print(f'[notes] unknown action {action!r}; use list|add|delete|tags|stats',flush=True);sys.exit(1)
def _notes_fetch(args,url):
    if url:
        qs=f'?limit={int(args.limit or 50)}'+(f'&tag={args.tag}' if getattr(args,"tag",None) else '')+(f'&search={args.search}' if getattr(args,"search",None) else '')
        r=_fetch_json(url+'/memory/notes'+qs)
        if r is None:print(f'[notes] unreachable: {url}',flush=True);sys.exit(1)
        return r.get('notes') or [],r.get('tags') or [],r.get('stats') or {}
    from amni.serve.notes import list_recent,stats,all_tags
    return list_recent(limit=int(args.limit or 50),tag=getattr(args,'tag',None),search=getattr(args,'search','') or ''),all_tags(),stats()
def _notes_list(args,url):
    items,tags,stats=_notes_fetch(args,url)
    if args.json:print(json.dumps({'notes':items,'tags':tags,'stats':stats},indent=2,default=str),flush=True);return
    if not items:print('\n  No notes yet. Add one with `amni notes add "text #tag"`.\n',flush=True);return
    print(f'\n  {len(items)} note(s) · total {stats.get("total",0)}\n',flush=True)
    for n in items:
        ts=(n.get('iso','') or '').replace('T',' ')[:16];nid=n.get('id','?');tg=' '.join(f'#{t}' for t in (n.get('tags') or []));txt=n.get('text','')
        print(f'  {ts}  [{nid}]'+(f'  {tg}' if tg else ''),flush=True);print(f'    {txt}',flush=True);print('',flush=True)
def _notes_add(args,url):
    text=(args.text or '').strip()
    if not text:print('[notes add] text required (positional arg or --text)',flush=True);sys.exit(1)
    tags=[t.strip() for t in (getattr(args,'tag',None) or '').split(',') if t.strip()] or None
    if url:
        import urllib.request
        body=json.dumps({'text':text,'tags':tags,'session_id':getattr(args,'session','') or ''}).encode('utf-8')
        try:
            req=urllib.request.Request(url+'/memory/notes',data=body,headers={'content-type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[notes] add failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.notes import add
        r=add(text=text,tags=tags,session_id=getattr(args,'session','') or '')
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[notes] {r["error"]}',flush=True);sys.exit(1)
    tg=' '.join(f'#{t}' for t in (r.get('tags') or []))
    print(f'  Added {r.get("id","?")}'+(f' · {tg}' if tg else ''),flush=True)
def _notes_delete(args,url):
    nid=(args.id or '').strip()
    if not nid:print('[notes delete] need an id (use `amni notes list` to find one)',flush=True);sys.exit(1)
    if url:
        import urllib.request
        try:
            req=urllib.request.Request(url+'/memory/notes/'+nid,method='DELETE')
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[notes] delete failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.notes import delete
        r=delete(nid)
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    if 'error' in r:print(f'[notes] {r["error"]}',flush=True);sys.exit(1)
    print(f'  Deleted {nid} · {r.get("remaining",0)} remaining.',flush=True)
def _notes_tags(args,url):
    _,tags,_=_notes_fetch(args,url)
    if args.json:print(json.dumps({'tags':tags},indent=2,default=str),flush=True);return
    if not tags:print('\n  No tags in use.\n',flush=True);return
    print('\n  '+' '.join(f'#{t}' for t in tags)+'\n',flush=True)
def _notes_stats(args,url):
    _,_,stats=_notes_fetch(args,url)
    if args.json:print(json.dumps(stats,indent=2,default=str),flush=True);return
    print(f'\n  total   {stats.get("total",0)}',flush=True);print(f'  last    {stats.get("last_iso") or "—"}\n',flush=True)
    tc=stats.get('tag_counts') or {}
    if tc:
        print('  by tag:',flush=True)
        for t,n in tc.items():print(f'    #{t:<18} {n}',flush=True)
    print('',flush=True)
def cmd_search(args):
    import urllib.request
    q=(getattr(args,'query',None) or '').strip()
    if not q:print('[search] need a query: amni search <words>',flush=True);sys.exit(1)
    url=getattr(args,'url',None) or 'http://127.0.0.1:7700'
    if not url.startswith('http'):url='http://'+url
    url=url.rstrip('/')
    body=json.dumps({'args':{'query':q}}).encode('utf-8')
    try:
        req=urllib.request.Request(url+'/skills/web',data=body,headers={'content-type':'application/json'},method='POST')
        with urllib.request.urlopen(req,timeout=60) as resp:j=json.loads(resp.read().decode('utf-8','ignore'))
    except Exception as e:print(f'[search] request failed ({url}): {e}\n  Is the server running?  amni serve',flush=True);sys.exit(1)
    out=j.get('output') if isinstance(j,dict) and 'output' in j else j
    if args.json:print(json.dumps(out,indent=2,default=str),flush=True);return
    if not out or out.get('error'):print(f'[search] {out.get("error","no result") if out else "no result"}',flush=True);sys.exit(1)
    print('\n'+(out.get('answer') or '(no answer)')+'\n',flush=True)
    for i,s in enumerate(out.get('sources',[])[:5]):print(f'  {i+1}. {s}',flush=True)
    if out.get('pii_scrubbed'):print('\n  [PII scrubbed before the query left the box]',flush=True)
    print('',flush=True)
def cmd_failures(args):
    action=(getattr(args,'action',None) or 'list').strip().lower()
    url=getattr(args,'url',None)
    if url and not url.startswith('http'):url='http://'+url
    if url:url=url.rstrip('/')
    if action=='list':return _failures_list(args,url)
    if action=='stats':return _failures_stats(args,url)
    if action=='ack':return _failures_ack(args,url)
    print(f'[failures] unknown action {action!r}; use list|stats|ack',flush=True);sys.exit(1)
def _failures_list(args,url):
    if url:
        r=_fetch_json(url+f'/memory/skill-failures?limit={int(args.limit or 20)}'+(f'&skill={args.skill}' if args.skill else ''))
        if r is None:print(f'[failures] unreachable: {url}',flush=True);sys.exit(1)
        failures=r.get('failures') or [];stats=r.get('stats') or {}
    else:
        from amni.serve.skill_failures import recent,stats as _stats
        failures=recent(limit=int(args.limit or 20),skill_filter=args.skill or None);stats=_stats()
    if args.json:print(json.dumps({'failures':failures,'stats':stats},indent=2,default=str),flush=True);return
    if not failures:print('\n  No skill failures recorded.  [OK]\n',flush=True);return
    total=stats.get('total',0);unacked=stats.get('unacked',0)
    print(f'\n  Skill failures · {total} total · {unacked} unacked\n',flush=True)
    for f in failures[-int(args.limit or 20):]:
        ts=f.get('iso','?');sk=f.get('skill','?');err=(f.get('error') or '')[:120];msg=(f.get('message') or '')[:80]
        print(f'  {ts}  [{sk}]',flush=True);print(f'    msg: {msg}',flush=True);print(f'    err: {err}',flush=True);print('',flush=True)
def _failures_stats(args,url):
    if url:
        r=_fetch_json(url+'/memory/skill-failures?limit=1')
        if r is None:print(f'[failures] unreachable: {url}',flush=True);sys.exit(1)
        s=r.get('stats') or {}
    else:
        from amni.serve.skill_failures import stats
        s=stats()
    if args.json:print(json.dumps(s,indent=2,default=str),flush=True);return
    print(f'\n  total      {s.get("total",0)}',flush=True)
    print(f'  unacked    {s.get("unacked",0)}',flush=True)
    print(f'  ack count  {s.get("ack_count",0)}',flush=True)
    print(f'  last       {s.get("last_iso") or "—"}\n',flush=True)
    by=s.get('by_skill') or {}
    if by:
        print('  by skill:',flush=True)
        for sk,n in by.items():print(f'    {sk:<24} {n}',flush=True)
        print('',flush=True)
def _failures_ack(args,url):
    if url:
        import urllib.request
        try:
            req=urllib.request.Request(url+'/memory/skill-failures/ack',data=b'',method='POST',headers={'content-type':'application/json'})
            with urllib.request.urlopen(req,timeout=5) as resp:r=json.loads(resp.read().decode('utf-8','ignore'))
        except Exception as e:print(f'[failures] ack failed: {e}',flush=True);sys.exit(1)
    else:
        from amni.serve.skill_failures import ack_all
        r=ack_all()
    if args.json:print(json.dumps(r,indent=2,default=str),flush=True);return
    print(f'  Acknowledged {r.get("acked",0)} skill failure(s) as of {r.get("ack_iso","?")}.',flush=True)
def cmd_mcp(args):
    action=(getattr(args,'action',None) or 'list').strip().lower()
    if action=='list':return _mcp_list(args)
    if action=='test':return _mcp_test(args)
    if action=='ping':return _mcp_ping(args)
    print(f'[mcp] unknown action {action!r}; use list|test|ping',flush=True);sys.exit(1)
def _mcp_request(base:str,method:str,params=None,timeout:float=10.0):
    import urllib.request,urllib.error
    body=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params or {}}).encode('utf-8')
    req=urllib.request.Request(base.rstrip('/')+'/mcp',data=body,headers={'content-type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
    except urllib.error.HTTPError as e:return {'_http_error':e.code,'_body':e.read().decode('utf-8','ignore')[:400]}
    except Exception as e:return {'_error':f'{type(e).__name__}: {e}'}
def _mcp_url(args)->str:
    base=getattr(args,'url',None) or 'http://127.0.0.1:7700'
    if not base.startswith('http'):base='http://'+base
    return base.rstrip('/')
def _mcp_list(args):
    base=_mcp_url(args);r=_mcp_request(base,'tools/list')
    if '_error' in r or '_http_error' in r:print(f'[mcp] cannot reach {base}/mcp: {r.get("_error") or r.get("_http_error")}',flush=True);sys.exit(1)
    tools=(r.get('result') or {}).get('tools') or []
    if args.json:print(json.dumps({'tools':tools,'count':len(tools)},indent=2,default=str),flush=True);return
    print(f'\n  Adam MCP server · {base}/mcp · {len(tools)} tool(s)\n',flush=True)
    grep=(args.grep or '').strip().lower()
    for t in tools:
        name=t.get('name','?');desc=(t.get('description') or '')[:140]
        if grep and grep not in name.lower() and grep not in desc.lower():continue
        props=((t.get('inputSchema') or {}).get('properties') or {});params_s=', '.join(props.keys()) if props else '(no params)'
        print(f'  {name:<26} {desc}',flush=True)
        if args.verbose:print(f'    params: {params_s}',flush=True)
    print('',flush=True)
def _mcp_test(args):
    name=(getattr(args,'tool',None) or '').strip()
    if not name:print('[mcp] need a tool name; try `amni mcp list`',flush=True);sys.exit(1)
    base=_mcp_url(args)
    try:tool_args=json.loads(args.args) if args.args else {}
    except Exception as e:print(f'[mcp] --args must be valid JSON: {e}',flush=True);sys.exit(1)
    r=_mcp_request(base,'tools/call',{'name':name,'arguments':tool_args},timeout=float(args.timeout or 30))
    if '_error' in r:print(f'[mcp] {r["_error"]}',flush=True);sys.exit(1)
    res=r.get('result') or {};is_err=res.get('isError');raw=res.get('_raw')
    if args.json:print(json.dumps({'tool':name,'arguments':tool_args,'is_error':bool(is_err),'result':raw if raw is not None else res},indent=2,default=str),flush=True);return
    print(f'\n  tool      {name}',flush=True);print(f'  args      {json.dumps(tool_args,default=str)}',flush=True)
    print(f'  status    {"ERROR" if is_err else "OK"}',flush=True)
    if isinstance(raw,dict):
        if 'answer' in raw:print(f'  answer    {(raw.get("answer") or "")[:400]}',flush=True);print(f'  tier      {raw.get("tier")}  persona={raw.get("persona")}  tokens={raw.get("tokens")}',flush=True)
        else:print(f'  result    {json.dumps(raw,default=str)[:600]}',flush=True)
    else:print(f'  result    {str(res)[:600]}',flush=True)
    print('',flush=True)
    if is_err:sys.exit(2)
def _mcp_ping(args):
    base=_mcp_url(args);t0=time.time();r=_mcp_request(base,'ping')
    dur=round((time.time()-t0)*1000,1)
    if '_error' in r:print(f'[mcp] {base}/mcp unreachable: {r["_error"]}',flush=True);sys.exit(1)
    if args.json:print(json.dumps({'url':base+'/mcp','latency_ms':dur,'ok':True,'response':r},default=str),flush=True);return
    print(f'  {base}/mcp — pong ({dur}ms)',flush=True)
def cmd_doctor(args):
    checks=[_doctor_python(),_doctor_core_deps(),_doctor_optional_deps(),_doctor_config_dir(),_doctor_bake(args),_doctor_model(args),_doctor_lessons(args),_doctor_gpu(),_doctor_disk(args),_doctor_port(args),_doctor_workdir(args)]
    if getattr(args,'network',False):checks.append(_doctor_network())
    if getattr(args,'json',False):print(json.dumps({'checks':checks,'overall':_doctor_overall(checks)},indent=2,default=str),flush=True);return
    print(_render_doctor_report(checks),flush=True)
    overall=_doctor_overall(checks)
    if overall=='fail':sys.exit(2)
    if overall=='warn':sys.exit(1)
def _doctor_overall(checks):
    if any(c.get('status')=='fail' for c in checks):return 'fail'
    if any(c.get('status')=='warn' for c in checks):return 'warn'
    return 'ok'
def _doctor_python():
    v=sys.version_info;s=f'{v.major}.{v.minor}.{v.micro}'
    ok=(v.major,v.minor)>=(3,10)
    return {'name':'Python version','status':'ok' if ok else 'fail','detail':s+(' (need >=3.10)' if not ok else '')}
def _doctor_core_deps():
    missing=[]
    for mod in ('torch','fastapi','uvicorn','numpy','transformers','huggingface_hub','safetensors'):
        try:__import__(mod)
        except Exception:missing.append(mod)
    return {'name':'Core dependencies','status':'fail' if missing else 'ok','detail':'missing: '+', '.join(missing) if missing else 'all 7 importable'}
def _doctor_optional_deps():
    opt={'faster_whisper':'STT','piper':'TTS (offline)','mediapipe':'gesture control','sentence_transformers':'embedding sidecar','trafilatura':'web ingestion','ddgs':'duckduckgo search','amni_prism':'federated learning'}
    have=[];missing=[]
    for mod,what in opt.items():
        try:__import__(mod);have.append(mod)
        except Exception:missing.append(f'{mod} ({what})')
    return {'name':'Optional dependencies','status':'info','detail':f'{len(have)}/{len(opt)} present'+(' · missing: '+', '.join(missing) if missing else '')}
def _doctor_config_dir():
    from amni.bootstrap import CONFIG_DIR,CONFIG_FILE
    cd=Path(CONFIG_DIR);cf=Path(CONFIG_FILE)
    if not cd.exists():return {'name':'Config directory','status':'warn','detail':f'{cd} does not exist (run `amni init`)'}
    if not cf.exists():return {'name':'Config directory','status':'warn','detail':f'config dir present but {cf.name} missing'}
    return {'name':'Config directory','status':'ok','detail':str(cd)}
def _doctor_bake(args):
    bake=getattr(args,'bake',None) or load_config().get('bake')
    if not bake:return {'name':'Bake (GF17 model)','status':'fail','detail':'no bake configured — run `amni init`'}
    p=Path(bake)
    if not p.exists():return {'name':'Bake (GF17 model)','status':'fail','detail':f'path does not exist: {p}'}
    if not p.is_dir():return {'name':'Bake (GF17 model)','status':'fail','detail':f'not a directory: {p}'}
    n_files=sum(1 for _ in p.iterdir());size_mb=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6
    return {'name':'Bake (GF17 model)','status':'ok','detail':f'{p} · {n_files} top-level items · {size_mb:.0f} MB'}
def _doctor_model(args):
    model=getattr(args,'model',None) or load_config().get('model') or load_config().get('bake')
    if not model:return {'name':'Runtime model','status':'fail','detail':'no model path configured'}
    p=Path(model)
    if not p.exists():return {'name':'Runtime model','status':'fail','detail':f'path does not exist: {p}'}
    return {'name':'Runtime model','status':'ok','detail':str(p)}
def _doctor_lessons(args):
    lessons=getattr(args,'lessons',None) or 'experiences/adam_lessons.npz'
    p=Path(lessons)
    if not p.exists():return {'name':'Lessons bank','status':'warn','detail':f'{p} not found (will seed on first run if --seed)'}
    try:size_kb=p.stat().st_size/1024
    except Exception:size_kb=0
    return {'name':'Lessons bank','status':'ok','detail':f'{p} · {size_kb:.1f} kb'}
def _doctor_gpu():
    try:import torch
    except Exception:return {'name':'GPU acceleration','status':'warn','detail':'torch not importable'}
    info=[];status='warn';detail='no GPU detected (CPU-only inference will be slow)'
    if hasattr(torch,'cuda') and torch.cuda.is_available():
        n=torch.cuda.device_count();names=[torch.cuda.get_device_name(i) for i in range(n)]
        status='ok';detail=f'CUDA · {n} device(s): '+', '.join(names)
    elif hasattr(torch.version,'hip') and torch.version.hip:
        status='ok';detail=f'ROCm/HIP · torch.version.hip={torch.version.hip}'
    elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        status='ok';detail='Apple MPS (Metal) available'
    return {'name':'GPU acceleration','status':status,'detail':detail}
def _doctor_disk(args):
    try:import shutil as _sh
    except Exception:return {'name':'Disk space','status':'warn','detail':'shutil unavailable'}
    target=Path(getattr(args,'workdir',None) or '.')
    try:u=_sh.disk_usage(str(target.resolve()))
    except Exception as e:return {'name':'Disk space','status':'warn','detail':f'disk_usage failed: {e}'}
    free_gb=u.free/1e9;pct=u.used/u.total
    status='ok' if free_gb>=5 else ('warn' if free_gb>=1 else 'fail')
    return {'name':'Disk space','status':status,'detail':f'{free_gb:.1f} GB free at {target.resolve()} ({pct*100:.0f}% used)'}
def _doctor_port(args):
    import socket
    port=int(getattr(args,'port',7700) or 7700)
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.settimeout(0.5)
    try:s.bind(('127.0.0.1',port));free=True
    except OSError:free=False
    finally:
        try:s.close()
        except Exception:pass
    if free:return {'name':f'Port {port}','status':'ok','detail':'bindable on 127.0.0.1'}
    return {'name':f'Port {port}','status':'warn','detail':'already in use — server may already be running, or pick another --port'}
def _doctor_workdir(args):
    wd=Path(getattr(args,'workdir',None) or '.').resolve()
    if not wd.exists():return {'name':'Workdir','status':'fail','detail':f'{wd} does not exist'}
    test=wd/'.amni_doctor_writetest'
    try:test.write_text('x',encoding='utf-8');test.unlink()
    except Exception as e:return {'name':'Workdir','status':'fail','detail':f'not writable: {e}'}
    return {'name':'Workdir','status':'ok','detail':f'{wd} writable'}
def _doctor_network():
    import urllib.request
    try:
        with urllib.request.urlopen('https://huggingface.co/',timeout=3) as r:code=r.status
        return {'name':'Network (HuggingFace)','status':'ok','detail':f'reachable · HTTP {code}'}
    except Exception as e:return {'name':'Network (HuggingFace)','status':'warn','detail':f'unreachable: {type(e).__name__}'}
def _render_doctor_report(checks)->str:
    enc=(getattr(sys.stdout,'encoding','') or '').lower()
    unicode_ok=any(s in enc for s in ('utf','u8','u-8'))
    if unicode_ok:marks={'ok':'\x1b[32m✓\x1b[0m','warn':'\x1b[33m⚠\x1b[0m','fail':'\x1b[31m✗\x1b[0m','info':'\x1b[36m·\x1b[0m'};dot='✗'
    else:marks={'ok':'\x1b[32m[OK]\x1b[0m','warn':'\x1b[33m[!]\x1b[0m','fail':'\x1b[31m[X]\x1b[0m','info':'\x1b[36m[.]\x1b[0m'};dot='[X]'
    lines=['','  Adam -- install diagnostics','']
    for c in checks:
        mark=marks.get(c.get('status'),'[.]');name=c.get('name','?');detail=c.get('detail','')
        lines.append(f'  {mark}  {name:<28}  {detail}')
    overall=_doctor_overall(checks)
    summary={'ok':'\x1b[32mAll systems nominal.\x1b[0m','warn':'\x1b[33mSome warnings -- Adam will run but check the items above.\x1b[0m','fail':f'\x1b[31mFailing checks present -- Adam may not start. Fix the {dot} items first.\x1b[0m'}.get(overall,'')
    lines.extend(['',f'  {summary}',''])
    return '\n'.join(lines)
def cmd_persona(args):
    from amni.serve.persona import PersonaStore
    action=(getattr(args,'action',None) or 'list').strip().lower()
    if action=='import' and getattr(args,'input',None) is None and getattr(args,'name',None):
        args.input=args.name;args.name=None
    ps=PersonaStore(adam=None,bank_path=args.persona_bank)
    if action=='list':return _persona_list(ps,args)
    if action=='show':return _persona_show(ps,args)
    if action=='export':return _persona_export(ps,args)
    if action=='import':return _persona_import(ps,args)
    if action=='delete':return _persona_delete(ps,args)
    print(f'[persona] unknown action {action!r}; use list|show|export|import|delete',flush=True);sys.exit(1)
def _persona_list(ps,args):
    known=ps.list_known()
    if getattr(args,'json',False):print(json.dumps([p.to_dict() for p in known],indent=2,default=str),flush=True);return
    print(f'\n  {len(known)} persona(s) · default={ps._default}\n',flush=True)
    for p in known:
        src=p.source if p.source!='preset' else ''
        mark='*' if p.name.lower()==ps._default.lower() else ' '
        suffix=f' [{src}]' if src else ''
        print(f'  {mark} {p.name:<22} warmth={p.warmth:.2f} formality={p.formality:.2f} length={p.length:.2f}{suffix}',flush=True)
    print('',flush=True)
def _persona_show(ps,args):
    name=(args.name or '').strip()
    if not name:print('[persona show] need a name',flush=True);sys.exit(1)
    if not ps.has(name):print(f'[persona show] unknown persona {name!r}',flush=True);sys.exit(1)
    p=ps.get(name)
    if getattr(args,'json',False):print(json.dumps(p.to_dict(),indent=2,default=str),flush=True);return
    print(f'\n  {p.name}',flush=True)
    print(f'  source: {p.source}',flush=True)
    print(f'\n  {p.description}\n',flush=True)
    if p.voice_hints:
        print('  voice hints:',flush=True)
        for h in p.voice_hints:print(f'    · {h}',flush=True)
    print(f'\n  dims:  warmth={p.warmth:.2f}  formality={p.formality:.2f}  excitement={p.excitement:.2f}  length={p.length:.2f}',flush=True)
    print(f'  tts:   {p.tts_voice}\n',flush=True)
def _persona_export(ps,args):
    name=(args.name or '').strip()
    if not name:print('[persona export] need a name',flush=True);sys.exit(1)
    if not ps.has(name):print(f'[persona export] unknown persona {name!r}',flush=True);sys.exit(1)
    p=ps.get(name);data=p.to_dict()
    payload={'_amni_persona_format':'v1','exported_at':time.time(),'persona':data}
    body=json.dumps(payload,indent=2,default=str)
    out=getattr(args,'output',None)
    if out:Path(out).write_text(body,encoding='utf-8');print(f'[export] wrote {p.name} to {out}',flush=True)
    else:print(body,flush=True)
def _persona_import(ps,args):
    src=Path(args.input)
    if not src.exists():print(f'[persona import] file not found: {src}',flush=True);sys.exit(1)
    try:raw=json.loads(src.read_text(encoding='utf-8'))
    except Exception as e:print(f'[persona import] bad JSON: {e}',flush=True);sys.exit(1)
    if isinstance(raw,dict) and isinstance(raw.get('persona'),dict):data=raw['persona']
    else:data=raw
    if not isinstance(data,dict):print('[persona import] no persona object found in file',flush=True);sys.exit(1)
    p=ps.import_persona(data,new_name=getattr(args,'rename',None),overwrite=bool(getattr(args,'overwrite',False)))
    if p is None:
        if not getattr(args,'overwrite',False) and ps.has(getattr(args,'rename',None) or data.get('name','')):print('[persona import] already exists — pass --overwrite to replace',flush=True);sys.exit(1)
        print('[persona import] invalid persona data (missing description, bad name, or unparseable dims)',flush=True);sys.exit(1)
    print(f'[persona import] added {p.name} (source={p.source})',flush=True)
def _persona_delete(ps,args):
    name=(args.name or '').strip()
    if not name:print('[persona delete] need a name',flush=True);sys.exit(1)
    if not ps.has(name):print(f'[persona delete] unknown persona {name!r}',flush=True);sys.exit(1)
    ok=ps.delete_persona(name)
    if not ok:print(f'[persona delete] {name!r} is a preset — cannot delete (only learned/edited/imported overrides can be removed)',flush=True);sys.exit(1)
    print(f'[persona delete] removed {name}',flush=True)
def cmd_personas(args):
    from amni.adam import Adam
    from amni.serve import PersonaStore
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None) if args.with_adam else None
    ps=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.learn:
        p=ps.learn(args.learn)
        print(json.dumps(p.to_dict(),indent=2,default=str),flush=True)
        return
    print(json.dumps({'default':ps._default,'known':[p.to_dict() for p in ps.list_known()]},indent=2,default=str),flush=True)
def main():
    p=argparse.ArgumentParser(prog='amni',description='Adam — GF(17) texture-native AI. One-line install + zero-friction launch.')
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init',help='First-run setup: download model, seed lessons, write config')
    init.add_argument('--non-interactive',action='store_true');init.add_argument('--skip-model',action='store_true')
    init.set_defaults(func=cmd_init)
    s=sub.add_parser('serve',help='Run HTTP server (browser UI, /chat, MCP, Ollama compat)')
    _add_common_adam(s)
    s.add_argument('--port',type=int,default=DEFAULT_PORT);s.add_argument('--host',default=DEFAULT_HOST)
    s.add_argument('--conv-root',default='experiences/conversations');s.add_argument('--audit-log',default='logs/agent_skill_calls.jsonl')
    s.add_argument('--persona-bank',default='experiences/personas.json');s.add_argument('--workdir',default=None)
    s.add_argument('--root',action='append',default=[]);s.add_argument('--unrestricted-files',action='store_true')
    s.add_argument('--seed',action='store_true');s.add_argument('--cors',action='store_true')
    s.add_argument('--default-persona',default=None);s.add_argument('--no-persona',action='store_true')
    s.add_argument('--open-browser',action='store_true',help='Auto-open a browser tab once the server is ready (waits for warmup)')
    s.add_argument('--app-window',action='store_true',help='Open Adam in a standalone chromeless desktop window (Chrome/Edge --app) once ready')
    s.set_defaults(func=cmd_serve)
    co=sub.add_parser('code',help='Project-aware coding mode: workdir=cwd, mentor persona, file-tree UI, browser auto-opens')
    _add_common_adam(co)
    co.add_argument('path',nargs='?',default=None,help='Project root (defaults to current dir)')
    co.add_argument('--port',type=int,default=DEFAULT_PORT);co.add_argument('--host',default=DEFAULT_HOST)
    co.add_argument('--conv-root',default='experiences/conversations');co.add_argument('--audit-log',default='logs/agent_skill_calls.jsonl')
    co.add_argument('--persona-bank',default='experiences/personas.json')
    co.add_argument('--root',action='append',default=[]);co.add_argument('--persona',default=None)
    co.add_argument('--seed',action='store_true')
    co.set_defaults(func=cmd_code)
    c=sub.add_parser('chat',help='Interactive REPL with full agent (no HTTP)')
    _add_common_adam(c)
    c.add_argument('--persona',default=None);c.add_argument('--workdir',default=None)
    c.add_argument('--unrestricted-files',action='store_true');c.add_argument('--seed',action='store_true')
    c.add_argument('--conv-root',default='experiences/conversations');c.add_argument('--persona-bank',default='experiences/personas.json')
    c.add_argument('--resume',action='store_true',help='Resume the most recent session')
    c.add_argument('--session',default=None,help='Resume a specific session id (or prefix)')
    c.add_argument('--list-sessions',action='store_true',help='List recent sessions and exit')
    c.set_defaults(func=cmd_chat)
    es=sub.add_parser('export-session',help='Export a conversation session to Markdown/JSON/JSONL')
    es.add_argument('session',nargs='?',default=None,help='Session id (or prefix); defaults to most recent')
    es.add_argument('--conv-root',default='experiences/conversations')
    es.add_argument('--format',choices=['md','json','jsonl'],default='md',help='Output format (default: md)')
    es.add_argument('--output','-o',default=None,help='Output path (default: <sid>.<ext>)')
    es.add_argument('--strip-personal',action='store_true',help='Drop turns flagged is_personal before export')
    es.set_defaults(func=cmd_export_session)
    iss=sub.add_parser('import-session',help='Import a session from Markdown/JSON/JSONL')
    iss.add_argument('input',help='Path to .md / .json / .jsonl session file')
    iss.add_argument('--conv-root',default='experiences/conversations')
    iss.add_argument('--session-id',default=None,help='Target session id; defaults to a fresh one')
    iss.set_defaults(func=cmd_import_session)
    a=sub.add_parser('ask',help='Single-shot question (no REPL)')
    _add_common_adam(a)
    a.add_argument('query',nargs='+');a.add_argument('--persona',default=None);a.add_argument('--workdir',default=None)
    a.add_argument('--unrestricted-files',action='store_true');a.add_argument('--seed',action='store_true')
    a.add_argument('--persona-bank',default='experiences/personas.json');a.add_argument('--json',action='store_true')
    a.set_defaults(func=cmd_ask)
    sc=sub.add_parser('scan',help='Bulk-ingest a directory of text files into lesson bank')
    _add_common_adam(sc)
    sc.add_argument('path');sc.add_argument('--glob',default='**/*');sc.add_argument('--max-files',type=int,default=50)
    sc.add_argument('--max-chars-per-file',type=int,default=8000);sc.add_argument('--distill',action='store_true')
    sc.add_argument('--workdir',default=None);sc.add_argument('--seed',action='store_true')
    sc.set_defaults(func=cmd_scan)
    pu=sub.add_parser('publish',help='Push PII-stripped lessons to Amni-Prism / HF')
    _add_common_adam(pu)
    pu.add_argument('--codex',default='./codex',help='Local Prism codex dir');pu.add_argument('--contributor',default='amni-ai-anonymous')
    pu.add_argument('--min-confidence',type=float,default=0.8);pu.add_argument('--domain',default=None)
    pu.add_argument('--limit',type=int,default=100);pu.add_argument('--dry-run',action='store_true')
    pu.set_defaults(func=cmd_publish)
    pl=sub.add_parser('pull',help='Fetch community lessons from Prism into local bank')
    _add_common_adam(pl)
    pl.add_argument('--codex',default='./codex');pl.add_argument('--domain',default=None)
    pl.add_argument('--limit',type=int,default=200);pl.add_argument('--dry-run',action='store_true')
    pl.set_defaults(func=cmd_pull)
    rf=sub.add_parser('reflect',help='Run self-reflection daemon: re-research low-confidence lessons')
    _add_common_adam(rf)
    rf.add_argument('--interval',type=int,default=300,help='Seconds between cycles');rf.add_argument('--max-per-cycle',type=int,default=5)
    rf.add_argument('--min-age',type=int,default=86400,help='Only reflect on lessons older than N seconds');rf.add_argument('--once',action='store_true')
    rf.set_defaults(func=cmd_reflect)
    tc=sub.add_parser('teach-cot',help='Bulk-teach Adam from curated CoT + coding corpus (~250 lessons)')
    _add_common_adam(tc)
    tc.add_argument('--bank',choices=['cot','coding','js','sql','devops','creative','facts','advanced','rust','concurrency','algo_adv','python_adv','go','frontend','mobile','data_eng','leetcode','ml','security_deep','distributed','performance','architecture','networking','gamedev','embedded','math_adv','facts_ext','python_libs','ai_rag','leetcode_hard','paraphrases','debug_adv','all'],default='all',help='Which seed bank to load')
    tc.add_argument('--dry-run',action='store_true')
    tc.set_defaults(func=cmd_teach_cot)
    pers=sub.add_parser('persona',help='Persona management: list, show, export, import, delete')
    pers.add_argument('action',nargs='?',default='list',choices=['list','show','export','import','delete'])
    pers.add_argument('name',nargs='?',default=None,help='Persona name (for show/export/delete)')
    pers.add_argument('input',nargs='?',default=None,help='Input file path (for import)')
    pers.add_argument('--persona-bank',default='experiences/personas.json')
    pers.add_argument('--output','-o',default=None,help='Output file (for export); stdout if omitted')
    pers.add_argument('--rename',default=None,help='Rename on import (overrides the persona\'s own name)')
    pers.add_argument('--overwrite',action='store_true',help='Allow import to replace an existing same-named persona')
    pers.add_argument('--json',action='store_true',help='JSON output for list/show')
    pers.set_defaults(func=cmd_persona)
    rm=sub.add_parser('reminders',help='Manage reminders: list / due / stats / add / dismiss')
    rm.add_argument('action',nargs='?',default='list',choices=['list','due','stats','add','dismiss'])
    rm.add_argument('text',nargs='?',default=None,help='Reminder text (for add)')
    rm.add_argument('--id',default=None,help='Reminder id (for dismiss)')
    rm.add_argument('--limit',type=int,default=50)
    rm.add_argument('--session',default=None,help='Filter to a single session id')
    rm.add_argument('--url',default=None,help='Hit a running server instead of reading local files')
    rm.add_argument('--json',action='store_true')
    rm.set_defaults(func=cmd_reminders)
    bm=sub.add_parser('bookmarks',help='Manage starred bot replies: list / stats / add / delete / export')
    bm.add_argument('action',nargs='?',default='list',choices=['list','stats','add','delete','export'])
    bm.add_argument('id',nargs='?',default=None,help='Bookmark id (for delete)')
    bm.add_argument('--limit',type=int,default=50)
    bm.add_argument('--session',default=None,help='Filter to a single session id')
    bm.add_argument('--search',default=None,help='Case-insensitive substring search')
    bm.add_argument('--url',default=None,help='Hit a running server instead of reading local files')
    bm.add_argument('--json',action='store_true')
    bm.add_argument('--output','-o',default=None,help='Output path (for export); stdout if omitted')
    bm.add_argument('--user',default=None,help='User message (for add)')
    bm.add_argument('--bot',default=None,help='Bot response (for add, required)')
    bm.add_argument('--note',default=None,help='Optional note (for add)')
    bm.add_argument('--persona',default=None,help='Persona tag (for add)')
    bm.add_argument('--tier',default=None,help='Tier tag (for add)')
    bm.set_defaults(func=cmd_bookmarks)
    nt=sub.add_parser('notes',help='Quick text capture: list / add / delete / tags / stats')
    nt.add_argument('action',nargs='?',default='list',choices=['list','add','delete','tags','stats'])
    nt.add_argument('text',nargs='?',default=None,help='Note text (for add) or id (for delete)')
    nt.add_argument('--id',default=None,help='Note id (for delete)')
    nt.add_argument('--tag',default=None,help='Filter by tag (list) or comma-separated tags (add)')
    nt.add_argument('--search',default=None,help='Case-insensitive substring search (list)')
    nt.add_argument('--limit',type=int,default=50)
    nt.add_argument('--session',default=None,help='Session id to attach (add)')
    nt.add_argument('--url',default=None,help='Hit a running server instead of reading local files')
    nt.add_argument('--json',action='store_true')
    nt.set_defaults(func=cmd_notes)
    se=sub.add_parser('search',help='Web search anytime via a running server (PII-scrubbed before send)')
    se.add_argument('query',nargs='+',help='Search query words')
    se.add_argument('--url',default=None,help='Server URL (default http://127.0.0.1:7700)')
    se.add_argument('--json',action='store_true')
    se.set_defaults(func=lambda a:(setattr(a,'query',' '.join(a.query)),cmd_search(a))[1])
    fai=sub.add_parser('failures',help='Inspect skill-failure log: list / stats / ack')
    fai.add_argument('action',nargs='?',default='list',choices=['list','stats','ack'])
    fai.add_argument('--limit',type=int,default=20)
    fai.add_argument('--skill',default=None,help='Filter to a single skill name')
    fai.add_argument('--url',default=None,help='Hit a running server instead of reading local files')
    fai.add_argument('--json',action='store_true')
    fai.set_defaults(func=cmd_failures)
    mcp=sub.add_parser('mcp',help='Inspect or test the MCP tool surface of a running Adam server')
    mcp.add_argument('action',nargs='?',default='list',choices=['list','test','ping'],help='list (default) | test | ping')
    mcp.add_argument('tool',nargs='?',default=None,help='Tool name (for action=test); e.g. ask_adam, skill_calc')
    mcp.add_argument('--url',default=None,help='Server URL (default http://127.0.0.1:7700)')
    mcp.add_argument('--args',default=None,help='JSON-encoded arguments for action=test (e.g. \'{"question":"hi"}\')')
    mcp.add_argument('--timeout',type=float,default=30.0,help='Request timeout in seconds (default 30)')
    mcp.add_argument('--grep',default=None,help='Filter tools by name/description substring (for action=list)')
    mcp.add_argument('--verbose','-v',action='store_true',help='Show parameter names for each tool (action=list)')
    mcp.add_argument('--json',action='store_true',help='Emit JSON instead of pretty output')
    mcp.set_defaults(func=cmd_mcp)
    doc=sub.add_parser('doctor',help='Diagnose install: deps, bake, model, GPU, disk, port, workdir')
    _add_common_adam(doc)
    doc.add_argument('--workdir',default=None)
    doc.add_argument('--port',type=int,default=DEFAULT_PORT)
    doc.add_argument('--network',action='store_true',help='Also check HuggingFace reachability')
    doc.add_argument('--json',action='store_true',help='Emit JSON instead of a pretty report')
    doc.set_defaults(func=cmd_doctor)
    st=sub.add_parser('stats',help='Show Adam stats (local snapshot or live dashboard from a running server)')
    _add_common_adam(st)
    st.add_argument('--watch',action='store_true',help='Live-refresh terminal dashboard polling the running server')
    st.add_argument('--remote',action='store_true',help='Hit the HTTP server for one snapshot instead of booting Adam')
    st.add_argument('--url',default=None,help='Server URL (default http://127.0.0.1:7700)')
    st.add_argument('--interval',type=int,default=3,help='Refresh interval for --watch (seconds, default 3)')
    st.set_defaults(func=cmd_stats)
    pe=sub.add_parser('personas',help='List or learn personas')
    _add_common_adam(pe)
    pe.add_argument('--learn',default=None,help='Learn a new persona by name (web-search if unknown)')
    pe.add_argument('--persona-bank',default='experiences/personas.json');pe.add_argument('--with-adam',action='store_true',help='Boot Adam for web-learn')
    pe.set_defaults(func=cmd_personas)
    args=p.parse_args()
    args.func(args)
if __name__=='__main__':main()
