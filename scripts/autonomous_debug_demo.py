"""autonomous_debug_demo — prove Adam can debug REAL code autonomously. Gives Adam a real module with a real bug
plus its failing test output, asks for a fix, APPLIES the fix to a sandbox copy, RE-RUNS the real test, and loops
up to max-attempts feeding the new error each round (self-correction on real files). Records each attempt to the
coding_ledger (Adam's persistent lessons). Reports: fixed? in how many attempts? Optional --baseline-url runs the
IDENTICAL loop on native Gemma-4-E2B for comparison.
Usage:
  python scripts/autonomous_debug_demo.py --target tests/selfdebug_demo/amni_demo_stats.py --test tests/selfdebug_demo/test_amni_demo_stats.py --max-attempts 3"""
import sys,json,argparse,re,subprocess,shutil,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
def _ask(url,timeout,msg,max_tokens=900):
    for path,payload,outk in ((url+'/chat',{'message':msg,'max_new_tokens':max_tokens},('answer','text','response')),(url+'/complete',{'prefix':msg,'max_tokens':max_tokens,'stop':['\n\n\n\n']},('completion','text','output'))):
        try:
            j=_post(path,payload,timeout)
            if isinstance(j,dict):
                for k in outk:
                    if j.get(k):return j[k]
        except Exception:continue
    return ''
def _run_test(test_path,timeout=20):
    r=subprocess.run([sys.executable,'-B',test_path],capture_output=True,timeout=timeout,text=True)
    return r.returncode==0,(r.stdout+r.stderr).strip()
def _extract_code(t):
    m=re.search(r'```(?:python)?\s*\n(.*?)```',t,re.DOTALL)
    if m:return m.group(1).strip()
    return t.strip()
def _ledger(task,success,errors,lesson,approach):
    try:
        from amni.serve import coding_ledger as cl
        cl.record(task=task,success=success,errors=errors,lesson=lesson,approach=approach)
        return True
    except Exception:return False
def run(url,target,test,max_attempts,timeout,label):
    target=Path(target);test=Path(test)
    sandbox=target.with_suffix('.sandbox.py')
    shutil.copy(target,sandbox)
    orig=target.read_text(encoding='utf-8')
    work=target
    ok,out=_run_test(str(test))
    print(f'[{label}] initial test: {"PASS" if ok else "FAIL"}',flush=True)
    if ok:
        shutil.os.remove(sandbox);return {'label':label,'fixed':True,'attempts':0,'note':'already passing'}
    history=[];fixed=False;used=0
    cur=orig
    for attempt in range(1,max_attempts+1):
        used=attempt
        prior=('\n'.join(f'- attempt {h["n"]} failed: {h["err"][:160]}' for h in history)) or '(none yet)'
        msg=(f"You are debugging a Python module. Find and fix the bug so ALL tests pass. Output ONLY the complete corrected file content in a ```python code block.\n\n"
             f"=== FILE: {target.name} ===\n{cur}\n\n=== FAILING TEST OUTPUT ===\n{out}\n\n=== PRIOR ATTEMPTS ===\n{prior}\n\nOutput the full corrected {target.name}:")
        resp=_ask(url,timeout,msg)
        fix=_extract_code(resp)
        if 'def ' not in fix:
            history.append({'n':attempt,'err':'model produced no code'});print(f'[{label}] attempt {attempt}: no code produced',flush=True);continue
        target.write_text(fix,encoding='utf-8')
        ok,out=_run_test(str(test))
        history.append({'n':attempt,'err':'' if ok else out})
        print(f'[{label}] attempt {attempt}: {"PASS" if ok else "FAIL"}',flush=True)
        _ledger(task=f'debug {target.name}',success=ok,errors=None if ok else [out[:200]],lesson=(f'fixed {target.name} in {attempt} attempt(s)' if ok else f'attempt {attempt} still failing'),approach='autonomous_debug_demo')
        cur=fix
        if ok:fixed=True;break
    shutil.copy(sandbox,target);shutil.os.remove(sandbox)
    return {'label':label,'fixed':fixed,'attempts':used,'history':[h['err'][:120] for h in history]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',required=True);ap.add_argument('--test',required=True)
    ap.add_argument('--url',default='http://127.0.0.1:7700');ap.add_argument('--baseline-url',default=None)
    ap.add_argument('--max-attempts',type=int,default=3);ap.add_argument('--req-timeout',type=int,default=180)
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    print(f'[debug-demo] target={a.target} test={a.test} max_attempts={a.max_attempts}',flush=True)
    res={'adam':run(a.url.rstrip('/'),a.target,a.test,a.max_attempts,a.req_timeout,'adam')}
    if a.baseline_url:res['parent']=run(a.baseline_url.rstrip('/'),a.target,a.test,a.max_attempts,a.req_timeout,'parent')
    print('\n[debug-demo] RESULT:',flush=True)
    for k,v in res.items():print(f'  {k}: fixed={v["fixed"]} attempts={v["attempts"]}',flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(res,indent=2),encoding='utf-8')
        print(f'[debug-demo] wrote {a.out}',flush=True)
if __name__=='__main__':main()
