"""Interactive REPL chat with Adam — multi-turn + skills, no HTTP server needed.
Commands:
  /skills          list registered skills
  /stats           show Adam + agent stats
  /new             start a new session
  /clear           clear the screen
  /quit  /exit     leave
Anything else is sent to the agent."""
import os,sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.bootstrap import load_config
_CFG=load_config()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bake',default=_CFG.get('bake'))
    ap.add_argument('--model',default=_CFG.get('model') or _CFG.get('bake'))
    ap.add_argument('--lessons',default=_CFG.get('lessons'))
    ap.add_argument('--lut-root',default=_CFG.get('lut_root'))
    ap.add_argument('--conv-root',default=_CFG.get('conv_root'))
    ap.add_argument('--audit-log',default=_CFG.get('audit_log'))
    ap.add_argument('--workdir',default=_CFG.get('workdir'))
    ap.add_argument('--seed',action='store_true')
    ap.add_argument('--session',default=None)
    args=ap.parse_args()
    if not args.bake or not Path(args.bake).exists() or not (Path(args.bake)/'manifest.json').exists():
        print(f'[amni_chat] FATAL: no usable bake found ({args.bake!r}). Run `python install.py` or pass --bake.',flush=True);sys.exit(2)
    if not args.model or not Path(args.model).exists() or not (Path(args.model)/'config.json').exists():
        print(f'[amni_chat] FATAL: no usable model dir ({args.model!r}). Run `python install.py` or pass --model.',flush=True);sys.exit(2)
    from amni.adam import Adam,SEED_LESSONS
    from amni.serve import AmniAgent,ConversationStore
    from amni.serve.skills import default_registry
    print(f'[amni_chat] booting Adam... ',end='',flush=True)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    skills=default_registry(workdir=args.workdir,audit_log=args.audit_log)
    store=ConversationStore(root=args.conv_root)
    agent=AmniAgent(adam=adam,skills=skills,store=store,workdir=args.workdir)
    sid=args.session
    print(f'ready. lessons={adam.stats().get("lessons_n",0)} skills={len(skills.list_skills())}',flush=True)
    print('Type /quit to exit, /skills for skill list. Anything else is a question.\n',flush=True)
    while True:
        try:msg=input('> ').strip()
        except (EOFError,KeyboardInterrupt):print('\nbye!',flush=True);break
        if not msg:continue
        if msg in ('/quit','/exit'):print('bye!',flush=True);break
        if msg=='/skills':
            for s in agent.list_skills():print(f'  {s["name"]:<12} {s["desc"]}',flush=True)
            continue
        if msg=='/stats':
            import json;print(json.dumps(agent.stats(),indent=2,default=str),flush=True);continue
        if msg=='/new':sid=None;print('(new session)',flush=True);continue
        if msg=='/clear':os.system('cls' if os.name=='nt' else 'clear');continue
        r=agent.chat(msg,session_id=sid,use_skills=True,writeback=True)
        sid=r.get('session_id') or sid
        tier=r.get('tier');toks=r.get('tokens');wall=r.get('wall_s');sk=r.get('skill_calls') or []
        sk_str=f' skills={[c["skill"] for c in sk]}' if sk else ''
        print(f'\n{r.get("answer")}\n',flush=True)
        print(f'  [tier={tier} tokens={toks} wall={wall}s session={sid}{sk_str}]\n',flush=True)
if __name__=='__main__':main()
