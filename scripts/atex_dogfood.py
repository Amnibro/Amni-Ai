"""atex_dogfood: drive the ATEX KB during the public-release loop and log every call to JSONL for paper metrics."""
import argparse,json,sys,time
from pathlib import Path
try:sys.stdout.reconfigure(encoding='utf-8')
except Exception:pass
try:sys.stderr.reconfigure(encoding='utf-8')
except Exception:pass
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))
from amni.learning.knowledge_base import KnowledgeBase
from amni.inference.kb_retriever import KBRetriever
def _log(metrics_path:Path,event:dict):
    event['ts']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    with open(metrics_path,'a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
def _read_text(path:Path,max_bytes:int=200000)->str:
    return path.read_text(encoding='utf-8',errors='replace')[:max_bytes]
def cmd_init(args):
    atex=Path(args.atex_dir);atex.mkdir(parents=True,exist_ok=True)
    metrics=Path(args.metrics)
    kb=KnowledgeBase(atex);t0=time.time()
    _log(metrics,{'op':'init','atex_dir':str(atex),'wall_s':0.0,'iteration':args.iteration})
    print(f'[dogfood] KB initialized at {atex} ({len(kb)} entries)')
    return 0
def cmd_ingest(args):
    atex=Path(args.atex_dir);metrics=Path(args.metrics)
    kb=KnowledgeBase(atex);n=0;total_bytes=0;t0=time.time()
    for spec in args.files:
        key,sep,path=spec.partition('=')
        if not sep:print(f'[dogfood] skip malformed spec: {spec}',file=sys.stderr);continue
        p=Path(path)
        if not p.exists():print(f'[dogfood] missing file: {p}',file=sys.stderr);continue
        text=_read_text(p)
        kb.add(key,text,meta={'source_path':str(p),'kind':'dogfood_seed'},allow_overwrite=True)
        n+=1;total_bytes+=len(text)
    kb.flush()
    elapsed=time.time()-t0
    _log(metrics,{'op':'ingest','n_added':n,'bytes':total_bytes,'wall_s':round(elapsed,3),'iteration':args.iteration})
    print(f'[dogfood] ingested {n} entries ({total_bytes/1024:.1f} KB) in {elapsed:.2f}s')
    s=kb.stats()
    print(f'  KB: {s["n_entries"]} entries, {s["n_pages"]} pages, {s["used_bytes"]/1024:.1f} KB used')
    return 0
def cmd_search(args):
    atex=Path(args.atex_dir);metrics=Path(args.metrics)
    retr=KBRetriever(str(atex));t0=time.time()
    results=retr.retrieve(args.query,k=args.k,max_chars_per=args.max_chars)
    elapsed=time.time()-t0
    hit=len(results)>0
    _log(metrics,{'op':'search','query':args.query,'k':args.k,'n_results':len(results),'hit':hit,'wall_ms':round(elapsed*1000,2),'iteration':args.iteration,'note':args.note or ''})
    if not results:print('(no results)');return 0
    print(retr.format_as_context(results))
    return 0
def cmd_recall(args):
    atex=Path(args.atex_dir);metrics=Path(args.metrics)
    kb=KnowledgeBase(atex);t0=time.time()
    v=kb.lookup(args.key)
    elapsed=time.time()-t0
    hit=v is not None
    _log(metrics,{'op':'recall','key':args.key,'hit':hit,'bytes_returned':len(v) if v else 0,'wall_ms':round(elapsed*1000,2),'iteration':args.iteration,'note':args.note or ''})
    print(v if v is not None else '(not found)')
    return 0
def cmd_remember(args):
    atex=Path(args.atex_dir);metrics=Path(args.metrics)
    kb=KnowledgeBase(atex);t0=time.time()
    text=args.text if args.text else sys.stdin.read()
    key=f'manual::{args.key}' if not args.key.startswith('manual::') else args.key
    kb.add(key,text,meta={'kind':'dogfood_remember','iteration':args.iteration},allow_overwrite=True)
    kb.flush()
    elapsed=time.time()-t0
    _log(metrics,{'op':'remember','key':key,'bytes':len(text),'wall_ms':round(elapsed*1000,2),'iteration':args.iteration,'note':args.note or ''})
    print(f'[dogfood] remembered {key} ({len(text)} bytes)')
    return 0
def cmd_stats(args):
    atex=Path(args.atex_dir);metrics=Path(args.metrics)
    kb=KnowledgeBase(atex);s=kb.stats()
    n_calls={'init':0,'ingest':0,'search':0,'recall':0,'remember':0,'stats':0}
    n_hits={'search':0,'recall':0};total_search_ms=0.0;total_recall_ms=0.0
    by_iter={}
    if metrics.exists():
        for line in metrics.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            try:e=json.loads(line)
            except json.JSONDecodeError:continue
            op=e.get('op','?');n_calls[op]=n_calls.get(op,0)+1
            if op=='search' and e.get('hit'):n_hits['search']+=1;total_search_ms+=e.get('wall_ms',0)
            elif op=='recall' and e.get('hit'):n_hits['recall']+=1;total_recall_ms+=e.get('wall_ms',0)
            it=e.get('iteration','?')
            by_iter[it]=by_iter.get(it,0)+1
    out={'kb':s,'calls':n_calls,'hits':n_hits,'avg_search_ms':round(total_search_ms/max(n_hits['search'],1),2),'avg_recall_ms':round(total_recall_ms/max(n_hits['recall'],1),2),'calls_by_iteration':by_iter}
    _log(metrics,{'op':'stats','iteration':args.iteration})
    print(json.dumps(out,indent=2))
    return 0
def main():
    ap=argparse.ArgumentParser(description='atex_dogfood: instrumented KB driver for the public-release loop')
    ap.add_argument('--atex-dir',default='C:/Users/antho/Documents/ai/atex/.atex',help='where the dogfood KB lives')
    ap.add_argument('--metrics',default='C:/Users/antho/Documents/ai/atex/.atex_metrics.jsonl',help='JSONL metrics log')
    ap.add_argument('--iteration',type=int,default=2,help='loop iteration number')
    sub=ap.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init');sp.set_defaults(func=cmd_init)
    sp=sub.add_parser('ingest');sp.add_argument('files',nargs='+',help='key=path pairs');sp.set_defaults(func=cmd_ingest)
    sp=sub.add_parser('search');sp.add_argument('query');sp.add_argument('--k',type=int,default=3);sp.add_argument('--max-chars',type=int,default=600);sp.add_argument('--note',default='');sp.set_defaults(func=cmd_search)
    sp=sub.add_parser('recall');sp.add_argument('key');sp.add_argument('--note',default='');sp.set_defaults(func=cmd_recall)
    sp=sub.add_parser('remember');sp.add_argument('key');sp.add_argument('--text',default=None);sp.add_argument('--note',default='');sp.set_defaults(func=cmd_remember)
    sp=sub.add_parser('stats');sp.set_defaults(func=cmd_stats)
    args=ap.parse_args()
    return args.func(args)
if __name__=='__main__':sys.exit(main())
