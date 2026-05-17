"""CLI: ask Adam a single question.
Usage:
  python scripts/amni_ask.py "What is 2+2?"
  python scripts/amni_ask.py --no-writeback "Some throwaway question"
"""
import os,sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('query',nargs='+')
    ap.add_argument('--bake',default='E:/Amni-Ai-Bakes/gemma4_e2b_it_gf17')
    ap.add_argument('--model',default='E:/Amni-Ai-Models/gemma-4-E2B-it')
    ap.add_argument('--lessons',default='experiences/adam_lessons.npz')
    ap.add_argument('--lut-root',default='experiences/adam_lut')
    ap.add_argument('--no-writeback',action='store_true')
    ap.add_argument('--seed',action='store_true')
    ap.add_argument('--json',action='store_true')
    args=ap.parse_args()
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
