"""bench_vs_codex — run Adam's 2B/4B model on a standard code benchmark TWICE and compare to Codex.
The point: cold run 1, then run 2 where every failure from run 1 is fed back as a lesson (the persistent-memory
edge). run 2 pass@1 should beat run 1. Drives a RUNNING Adam server (start it with: python scripts/amni_serve.py).

Usage:
  python scripts/bench_vs_codex.py --benchmark humaneval --limit 50 --url http://127.0.0.1:7700
  python scripts/bench_vs_codex.py --benchmark sample            # offline smoke (bundled problems)
"""
import sys,json,argparse,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import code_bench as cb
def _post(url,payload,timeout=120):
    data=json.dumps(payload).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
def _make_generate_fn(url,max_tokens,timeout):
    url=url.rstrip('/')
    def gen(prompt,prior_lesson=None):
        instr='Complete this Python function. Output ONLY the function definition (you may include needed imports), no prose, no markdown.'
        if prior_lesson:instr='Your previous attempt FAILED: '+str(prior_lesson)[:300]+'\nFix that specific problem. '+instr
        msg=instr+'\n\n'+prompt
        for path,key,outk in (('/complete','prompt',('completion','text','output')),('/chat','message',('answer','text','response'))):
            try:
                j=_post(url+path,{key:msg,'max_tokens':max_tokens,'max_new_tokens':max_tokens},timeout=timeout)
                if isinstance(j,dict):
                    for k in outk:
                        if j.get(k):return j[k]
                    if j.get('output') and isinstance(j['output'],dict):
                        for k in outk:
                            if j['output'].get(k):return j['output'][k]
            except Exception:continue
        return ''
    return gen
def main():
    ap=argparse.ArgumentParser(description='Run Adam on a code benchmark twice and compare to Codex.')
    ap.add_argument('--benchmark',default='humaneval',choices=['humaneval','mbpp','sample'])
    ap.add_argument('--limit',type=int,default=None,help='cap number of problems (None = all)')
    ap.add_argument('--url',default='http://127.0.0.1:7700',help='running Adam server URL')
    ap.add_argument('--max-tokens',type=int,default=512)
    ap.add_argument('--timeout',type=int,default=10,help='per-problem test execution timeout (s)')
    ap.add_argument('--req-timeout',type=int,default=120,help='per-generation request timeout (s)')
    ap.add_argument('--persist',action='store_true',help='also record attempts to coding_ledger (cross-run memory)')
    ap.add_argument('--baseline-url',default=None,help='raw base-model server (e.g. Gemma-4) for single-shot baseline; enables Adam-vs-baseline comparison')
    ap.add_argument('--max-attempts',dest='max_attempts2',type=int,default=3,help='Adam loop attempts in baseline comparison mode')
    ap.add_argument('--json',action='store_true')
    a=ap.parse_args()
    probs={'humaneval':cb.load_humaneval,'mbpp':cb.load_mbpp,'sample':cb.load_sample}[a.benchmark](limit=a.limit)
    if not probs:print(f'[bench] no problems loaded for {a.benchmark}',flush=True);sys.exit(1)
    print(f'[bench] {a.benchmark}: {len(probs)} problems · server {a.url}',flush=True)
    gen=_make_generate_fn(a.url,a.max_tokens,a.req_timeout)
    lesson_fn=None
    if a.persist:
        try:
            from amni.serve import coding_ledger as cl
            def lesson_fn(p,r):
                cl.record(task=p['task_id']+' :: '+p['entry_point'],success=False,errors=[r.get('error','')[:200]],lesson=f"failed: {r.get('error','')[:160]}",approach='benchmark')
                return f"your previous attempt FAILED with: {r.get('error','')[:300]} — fix that exact error"
        except Exception:pass
    if a.baseline_url:
        baseline_gen=_make_generate_fn(a.baseline_url,a.max_tokens,a.req_timeout)
        record_fn=synth_fn=None
        if a.persist:
            try:
                from amni.serve import coding_ledger as cl
                def record_fn(prob,res,attempt):cl.record(task=prob['task_id'],success=res['passed'],errors=[res['error'][:160]] if not res['passed'] else None,approach=f'attempt{attempt}')
                synth_fn=cl.synthesize
            except Exception:pass
        comp=cb.run_comparison(baseline_gen,gen,probs,max_attempts=a.max_attempts2,timeout=a.timeout,record_fn=record_fn,synth_fn=synth_fn)
        if a.json:print(json.dumps({k:v for k,v in comp.items()},indent=2));return
        print('\n'+cb.compare_baseline_table(comp,baseline_label='Gemma-4 (raw, single-shot)',benchmark=a.benchmark)+'\n',flush=True);return
    out=cb.run_twice(gen,probs,timeout=a.timeout,lesson_fn=lesson_fn)
    if a.json:print(json.dumps({k:v for k,v in out.items() if k not in ('run1','run2')},indent=2));return
    print('\n'+cb.compare_to_codex(out,a.benchmark)+'\n',flush=True)
if __name__=='__main__':main()
