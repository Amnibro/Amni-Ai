"""bench_modern — run Adam on 2026-relevant benchmarks (MMLU-Pro, MATH-500, HumanEval+, MBPP+) through the FROZEN
harness contract (amni/eval/harness_config). Greedy canonical decoding. Drives a running Adam server. Code benches
execute completions in a subprocess for pass@1. Writes a result JSON consumable by adam_iteration_ledger + regression_gate.
Usage:
  python scripts/bench_modern.py --suites mmlu_pro math500 --limit 25 --out eval_reports/leaderboard_v5.0.3_2026-05-28.json
  python scripts/bench_modern.py --suites humanevalplus --limit 25"""
import sys,json,re,argparse,urllib.request,subprocess,tempfile,os,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import modern_bench as mb
from amni.eval import harness_config as hc
def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
def _gen(url,timeout,max_tokens,numeric,code=False):
    stops=['Problem:','Question:'] if numeric else ['\n\n\n','</task>','Question:','Problem:']
    def g(prompt):
        if code:
            msg='Complete this Python function. Output the COMPLETE function definition including the signature line and any needed imports, correctly indented. No prose, no markdown fences.\n\n'+prompt
            for path,payload,outk in ((url+'/chat',{'message':msg,'max_new_tokens':max_tokens},('answer','text','response')),(url+'/complete',{'prefix':prompt,'max_tokens':max_tokens,'stop':stops},('completion','text','output'))):
                try:
                    j=_post(path,payload,timeout)
                    if isinstance(j,dict):
                        for k in outk:
                            if j.get(k):return j[k]
                except Exception:continue
            return ''
        for path,payload,outk in ((url+'/complete',{'prefix':prompt,'max_tokens':max_tokens,'stop':stops},('completion','text','output')),(url+'/chat',{'message':prompt,'max_new_tokens':max_tokens},('answer','text','response'))):
            try:
                j=_post(path,payload,timeout)
                if isinstance(j,dict):
                    for k in outk:
                        if j.get(k):return j[k]
            except Exception:continue
        return ''
    return g
def _strip_code(t):
    t=re.sub(r'^```\w*\s*\n?','',t.strip());t=re.sub(r'\n?```\s*$','',t);return t
def _reindent_body(body,indent='    '):
    lines=body.split('\n')
    out=[]
    for ln in lines:
        if ln.strip()=='':out.append('')
        elif ln.startswith((' ','\t')):out.append(ln)
        else:out.append(indent+ln)
    return '\n'.join(out)
def _extract_function(body,ep):
    imports=[l for l in body.split('\n') if l.startswith(('import ','from '))]
    idx=body.find(f'def {ep}')
    if idx<0:return None
    after=body[idx:]
    lines=after.split('\n')
    block=[lines[0]]
    for l in lines[1:]:
        if l.strip()=='':block.append('');continue
        ind=len(l)-len(l.lstrip())
        if ind>0:block.append(l)
        else:break
    fn='\n'.join(block).rstrip()
    return ('\n'.join(imports)+'\n' if imports else '')+fn
def _assemble_code(item,completion):
    body=_strip_code(completion)
    prompt=item['prompt'];ep=item.get('entry_point','')
    fn=_extract_function(body,ep)
    if fn is not None:
        pimp='\n'.join(l for l in prompt.split('\n') if l.startswith(('import ','from ')))
        return (pimp+'\n' if pimp else '')+fn
    return prompt+('' if prompt.endswith('\n') else '\n')+_reindent_body(body)
def _exec_code(item,completion,timeout=10):
    test=item.get('test','');ep=item.get('entry_point','')
    mod=_assemble_code(item,completion)
    src=mod+'\n'+test+(f'\ncheck({ep})\n' if 'def check' in test else '')
    p=None
    try:
        with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8') as f:f.write(src);p=f.name
        r=subprocess.run([sys.executable,'-B',p],capture_output=True,timeout=timeout,text=True)
        return r.returncode==0
    except Exception:return False
    finally:
        if p:
            try:os.unlink(p)
            except Exception:pass
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suites',nargs='+',default=['mmlu_pro','math500','humanevalplus'])
    ap.add_argument('--limit',type=int,default=None)
    ap.add_argument('--url',default='http://127.0.0.1:7700')
    ap.add_argument('--req-timeout',type=int,default=180)
    ap.add_argument('--out',default=None)
    a=ap.parse_args();url=a.url.rstrip('/')
    print('[modern] '+hc.describe().split(chr(10))[0],flush=True)
    scores={};full={}
    for s in a.suites:
        cfg=hc.PER_BENCH.get(s,{'max_tokens':512,'kind':'mcq'})
        lim=a.limit if a.limit is not None else cfg.get('limit')
        numeric=cfg['kind'] in ('boxed','numeric')
        is_code=cfg['kind']=='code'
        gen=_gen(url,a.req_timeout,cfg['max_tokens'],numeric,code=is_code)
        exec_fn=_exec_code if is_code else None
        print(f'[modern] running {s} (limit={lim}, kind={cfg["kind"]}, max_tokens={cfg["max_tokens"]}) ...',flush=True)
        t0=time.time()
        r=mb.run_suite(s,gen,limit=lim,exec_fn=exec_fn)
        full[s]=r
        if r['n']>0:scores[s]=r['accuracy'];print(f'[modern] {s}: {r["accuracy"]}% ({r["correct"]}/{r["n"]}) in {time.time()-t0:.0f}s',flush=True)
        else:print(f'[modern] {s}: skipped ({r.get("reason")})',flush=True)
    print();print(mb.leaderboard_modern(scores,adam_label='Adam (this run)'));print()
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps({'scores':scores,'full':full,'harness':hc.stamp()},indent=2,default=str),encoding='utf-8')
        print(f'[modern] wrote {a.out}',flush=True)
if __name__=='__main__':main()
