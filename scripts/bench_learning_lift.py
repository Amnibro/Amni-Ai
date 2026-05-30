"""bench_learning_lift — demonstrate Adam's learning mechanism HONESTLY (checklist B3.5). Splits a bench into
disjoint PRACTICE and TEST. iter-1 = TEST cold (zero-shot). Lessons = practice items Adam SOLVED CORRECTLY, kept as
worked exemplars (a faithful proxy for Adam's PTEX nonce-retrieval, which conditions on relevant solved cases).
iter-2 = TEST with those practice exemplars injected as context. lift = iter2 - iter1.
GOLD-LEAKAGE GUARDRAIL: TEST gold is NEVER injected; exemplars are PRACTICE-only and audited to contain no TEST item.
Drives a running Adam server. Numeric benches only (clean scoring): gsm8k, math500.
Usage:
  python scripts/bench_learning_lift.py --bench math500 --practice 30 --test 30 --k-exemplars 4 --out eval_reports/lift_math500.json"""
import sys,json,argparse,urllib.request,re,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import modern_bench as mb
from amni.eval import suite_bench as sb
def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
def _gen(url,timeout,max_tokens):
    def g(prompt):
        for path,payload,outk in ((url+'/complete',{'prefix':prompt,'max_tokens':max_tokens,'stop':['Problem:','Question:','Exemplar']},('completion','text','output')),(url+'/chat',{'message':prompt,'max_new_tokens':max_tokens},('answer','text','response'))):
            try:
                j=_post(path,payload,timeout)
                if isinstance(j,dict):
                    for k in outk:
                        if j.get(k):return j[k]
            except Exception:continue
        return ''
    return g
def _num(out):
    m=re.search(r'####\s*([\-\d,\.]+)',out or '')
    if m:return m.group(1).replace(',','').strip().rstrip('.')
    nums=re.findall(r'-?\d[\d,]*\.?\d*',(out or '').replace(',',''))
    return nums[-1].rstrip('.') if nums else ''
def _boxed(out):
    m=re.search(r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',out or '')
    if m:return m.group(1).strip()
    return _num(out)
def _eq(a,b):
    try:return abs(float(a)-float(b))<1e-6
    except Exception:return re.sub(r'\s+','',str(a))==re.sub(r'\s+','',str(b))
def _load(bench,n):
    if bench=='gsm8k':
        items=sb.load_gsm8k(limit=n)
        return [{'q':it['question'],'a':it['answer'],'kind':'numeric'} for it in items]
    items=mb.load_math500(limit=n)
    return [{'q':it['question'],'a':it['answer'],'kind':'boxed'} for it in items]
def _prompt(it,exemplars=None):
    pre=''
    if exemplars:
        pre='Here are worked examples of similar problems:\n\n'+'\n\n'.join(f'Exemplar {i+1}:\n{e}' for i,e in enumerate(exemplars))+'\n\n---\nNow solve this new problem the same careful way.\n\n'
    if it['kind']=='boxed':
        return pre+f"Problem: {it['q']}\nSolve step by step. Put your final answer inside \\boxed{{ }} on its own line.\nSolution:"
    return pre+f"Problem: {it['q']}\nSolution (work step by step, then write the final answer on its own line prefixed by '####'):\n"
def _extract(it,out):return _boxed(out) if it['kind']=='boxed' else _num(out)
def _audit_no_test_leak(exemplars,test_items):
    test_qs=set(t['q'][:80] for t in test_items);test_as=set(str(t['a']) for t in test_items)
    for e in exemplars:
        for tq in test_qs:
            if tq and tq in e:return False,'exemplar contains a TEST question'
    return True,'clean'
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bench',default='math500',choices=['math500','gsm8k'])
    ap.add_argument('--practice',type=int,default=30);ap.add_argument('--test',type=int,default=30)
    ap.add_argument('--k-exemplars',type=int,default=4)
    ap.add_argument('--url',default='http://127.0.0.1:7700');ap.add_argument('--req-timeout',type=int,default=180)
    ap.add_argument('--max-tokens',type=int,default=768);ap.add_argument('--out',default=None)
    a=ap.parse_args();url=a.url.rstrip('/')
    allitems=_load(a.bench,a.practice+a.test)
    if len(allitems)<a.practice+a.test:print(f'[lift] only {len(allitems)} items available');return
    practice=allitems[:a.practice];test=allitems[a.practice:a.practice+a.test]
    gen=_gen(url,a.req_timeout,a.max_tokens)
    print(f'[lift] {a.bench}: practice={len(practice)} test={len(test)} k={a.k_exemplars}',flush=True)
    print('[lift] phase 1: building exemplars from PRACTICE (keep only correct, gold-clean) ...',flush=True)
    exemplars=[];t0=time.time()
    for it in practice:
        if len(exemplars)>=a.k_exemplars:break
        out=gen(_prompt(it))
        got=_extract(it,out)
        if _eq(got,it['a']):
            sol=out.strip()
            if it['q'][:80] not in sol:exemplars.append(f"Problem: {it['q']}\n{sol}")
    print(f'[lift] built {len(exemplars)} correct exemplars in {time.time()-t0:.0f}s',flush=True)
    ok,msg=_audit_no_test_leak(exemplars,test)
    print(f'[lift] GOLD-LEAK AUDIT: {msg}',flush=True)
    if not ok:print('[lift] ABORT — leak detected');return
    print('[lift] phase 2: iter-1 (TEST cold, zero-shot) ...',flush=True)
    c1=0;t0=time.time()
    for it in test:
        if _eq(_extract(it,gen(_prompt(it))),it['a']):c1+=1
    iter1=round(100*c1/len(test),1);print(f'[lift] iter-1 = {iter1}% ({c1}/{len(test)}) in {time.time()-t0:.0f}s',flush=True)
    print('[lift] phase 3: iter-2 (TEST with practice exemplars) ...',flush=True)
    c2=0;t0=time.time()
    for it in test:
        if _eq(_extract(it,gen(_prompt(it,exemplars))),it['a']):c2+=1
    iter2=round(100*c2/len(test),1);print(f'[lift] iter-2 = {iter2}% ({c2}/{len(test)}) in {time.time()-t0:.0f}s',flush=True)
    lift=round(iter2-iter1,1)
    print(f'\n[lift] === {a.bench} LEARNING LIFT: iter1={iter1}% -> iter2={iter2}%  (lift {lift:+}pp) ===',flush=True)
    print('[lift] (exemplars are PRACTICE-only worked solutions; TEST gold never injected — audited clean)',flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps({'bench':a.bench,'iter1':iter1,'iter2':iter2,'lift':lift,'n_test':len(test),'n_exemplars':len(exemplars),'gold_audit':msg},indent=2),encoding='utf-8')
        print(f'[lift] wrote {a.out}',flush=True)
if __name__=='__main__':main()
