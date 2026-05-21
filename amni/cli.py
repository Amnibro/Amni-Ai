"""amni — single CLI entry point. Subcommands: init, serve, chat, ask, scan, code, persona, publish, pull, reflect, stats, personas.
Mirrors Amni-Prism's CLI shape so installing both packages gives a coherent toolset."""
import argparse,sys,os,json,time,webbrowser,threading
from pathlib import Path
from amni.bootstrap import load_config,save_config,ensure_dirs,download_bake,download_base_model,detect_bake,detect_model,bake_has_runtime_metadata,CONFIG_DIR,CONFIG_FILE,is_first_run,mark_first_run_done,DEFAULT_PORT,DEFAULT_HOST
def _add_common_adam(p):
    cfg=load_config()
    default_bake=cfg.get('bake') or str(CONFIG_DIR/'bakes'/'gemma4_e2b_it_gf17')
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
        if args.non_interactive or _ask('Download Gemma-4 E2B GF(17) bake from HF (~20 GB, one-time)?'):
            b=download_bake(cfg)
            if b:cfg['bake']=str(b)
    if cfg.get('bake') and bake_has_runtime_metadata(cfg.get('bake')):
        cfg['model']=str(cfg['bake'])
        print(f'[init] runtime model dir = bake dir ({cfg["bake"]}) — Adam ships as a self-contained GF(17) artifact; no upstream Gemma 4 download needed.',flush=True)
    ensure_dirs(cfg)
    cfg['first_run_done']=True
    save_config(cfg)
    print(f'\n[init] saved config to {CONFIG_FILE}',flush=True)
    print(f'[init] next: `amni serve` (no flags needed) OR `amni chat`',flush=True)
def _ask(prompt:str)->bool:
    try:r=input(f'  {prompt} [Y/n] ').strip().lower()
    except EOFError:return False
    return r in ('','y','yes')
def _open_browser_delayed(url:str,delay:float=2.5):
    def _go():
        time.sleep(delay)
        try:webbrowser.open(url)
        except Exception:pass
    threading.Thread(target=_go,daemon=True).start()
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
    if getattr(args,'open_browser',False):_open_browser_delayed(f'http://{args.host}:{args.port}/')
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
    print('[amni] booting Adam... ',end='',flush=True)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    skills=default_registry(workdir=args.workdir,unrestricted=args.unrestricted_files,audit_log='logs/agent_skill_calls.jsonl')
    store=ConversationStore(root=args.conv_root)
    personas=PersonaStore(adam=adam,bank_path=args.persona_bank)
    if args.persona:personas.set_default(args.persona)
    agent=AmniAgent(adam=adam,skills=skills,store=store,personas=personas,workdir=args.workdir)
    print(f'ready. lessons={adam.stats().get("lessons_n",0)} skills={len(skills.list_skills())} persona={personas._default}',flush=True)
    print('Type /quit to exit, /persona <name> to switch, /skills to list, /stats for stats.\n',flush=True)
    sid=None
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
        r=agent.chat(msg,session_id=sid)
        sid=r.get('session_id') or sid
        print(f'\n{r.get("answer")}\n',flush=True)
        print(f'  [tier={r.get("tier")} persona={r.get("persona")} category={r.get("category")} tokens={r.get("tokens")} wall={r.get("wall_s")}s]\n',flush=True)
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
    from amni.adam import Adam
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=None)
    print(json.dumps(adam.stats(),indent=2,default=str),flush=True)
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
    s.add_argument('--open-browser',action='store_true',help='Auto-open browser tab on launch')
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
    c.set_defaults(func=cmd_chat)
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
    st=sub.add_parser('stats',help='Show Adam stats')
    _add_common_adam(st);st.set_defaults(func=cmd_stats)
    pe=sub.add_parser('personas',help='List or learn personas')
    _add_common_adam(pe)
    pe.add_argument('--learn',default=None,help='Learn a new persona by name (web-search if unknown)')
    pe.add_argument('--persona-bank',default='experiences/personas.json');pe.add_argument('--with-adam',action='store_true',help='Boot Adam for web-learn')
    pe.set_defaults(func=cmd_personas)
    args=p.parse_args()
    args.func(args)
if __name__=='__main__':main()
