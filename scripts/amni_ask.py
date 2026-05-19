"""CLI: ask Adam a single question.
Usage:
  python scripts/amni_ask.py "What is 2+2?"
  python scripts/amni_ask.py --no-writeback "Some throwaway question"
"""
import os,sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.bootstrap import load_config
_CFG=load_config()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('query',nargs='+')
    ap.add_argument('--bake',default=_CFG.get('bake'))
    ap.add_argument('--model',default=_CFG.get('model') or _CFG.get('bake'))
    ap.add_argument('--lessons',default=_CFG.get('lessons'))
    ap.add_argument('--lut-root',default=_CFG.get('lut_root'))
    ap.add_argument('--no-writeback',action='store_true')
    ap.add_argument('--seed',action='store_true')
    ap.add_argument('--json',action='store_true')
    args=ap.parse_args()
    if not args.bake or not Path(args.bake).exists() or not (Path(args.bake)/'manifest.json').exists():
        print(f'[amni_ask] FATAL: no usable bake found ({args.bake!r}). Run `python install.py` or pass --bake.',flush=True);sys.exit(2)
    if not args.model or not Path(args.model).exists() or not (Path(args.model)/'config.json').exists():
        print(f'[amni_ask] FATAL: no usable model dir ({args.model!r}). Run `python install.py` or pass --model.',flush=True);sys.exit(2)
    from amni.adam import Adam,SEED_LESSONS
    query=' '.join(args.query)
    adam=Adam(bake=args.bake,model=args.model,lessons_path=args.lessons,lut_root=args.lut_root,seed_lessons=SEED_LESSONS if args.seed else None)
    result=adam.ask(query,writeback=not args.no_writeback)
    if args.json:print(json.dumps(result,indent=2))
    else:
        print(f'Q: {query}',flush=True)
        print(f'A: {result.get("answer")}',flush=True)
        print(f'   [tier={result.get("tier")} tokens={result.get("tokens")} wall={result.get("wall_s")}s lessons_n={result.get("lessons_n")}]',flush=True)
if __name__=='__main__':main()
