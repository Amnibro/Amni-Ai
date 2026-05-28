"""code_bench — run Adam on industry-standard code benchmarks (HumanEval / MBPP), TWICE, to prove the
persistent-memory thesis: fail once, do better every time after. Between run 1 and run 2 each failure becomes a
lesson (coding_ledger); run 2's prompt carries that lesson, so pass@1 should climb. Model-agnostic: you pass a
generate_fn(prompt, prior_lesson=None)->code, so the harness is testable without booting the model and works with
any backend. Success is decided OBJECTIVELY by the benchmark's own unit tests run in a sandboxed subprocess."""
import json,re,subprocess,sys,tempfile,time,os
from pathlib import Path
from typing import Dict,Any,List,Optional,Callable
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _sample_path()->Path:return Path(__file__).resolve().parent/'humaneval_sample.jsonl'
def load_humaneval(limit:Optional[int]=None)->List[Dict[str,Any]]:
    probs=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('openai_humaneval',split='test')
        for r in ds:
            probs.append({'task_id':r['task_id'],'prompt':r['prompt'],'entry_point':r['entry_point'],'test':r['test']})
            if limit and len(probs)>=limit:break
        if probs:return probs
    except Exception as e:print(f'[code_bench] HumanEval via datasets unavailable ({e}); using bundled sample',flush=True)
    return load_sample(limit=limit)
def load_mbpp(limit:Optional[int]=None)->List[Dict[str,Any]]:
    probs=[]
    try:
        from datasets import load_dataset
        ds=load_dataset('mbpp',split='test')
        for r in ds:
            tests='\n'.join(r.get('test_list') or [])
            ep=_guess_entry_point(r.get('code',''))
            probs.append({'task_id':f"mbpp/{r['task_id']}",'prompt':(r.get('text','')+'\n').replace('"""','')[:600],'entry_point':ep,'test':tests,'mbpp':True,'code':r.get('code','')})
            if limit and len(probs)>=limit:break
    except Exception as e:print(f'[code_bench] MBPP unavailable ({e})',flush=True)
    return probs
def load_sample(limit:Optional[int]=None)->List[Dict[str,Any]]:
    p=_sample_path()
    if not p.exists():return []
    out=[]
    for ln in p.read_text(encoding='utf-8').splitlines():
        ln=ln.strip()
        if not ln:continue
        try:out.append(json.loads(ln))
        except Exception:continue
        if limit and len(out)>=limit:break
    return out
def _guess_entry_point(code:str)->str:
    m=re.search(r'def\s+(\w+)\s*\(',code or '')
    return m.group(1) if m else 'solution'
def _clean(completion:str)->str:
    c=completion or ''
    m=re.search(r'```(?:python|py)?\s*\n(.*?)```',c,re.DOTALL)
    if m:c=m.group(1)
    return c.rstrip()
def _assemble(problem:Dict[str,Any],completion:str)->str:
    c=_clean(completion);ep=problem.get('entry_point','solution')
    if re.search(r'(?m)^\s*def\s+'+re.escape(ep)+r'\b',c) or problem.get('mbpp'):
        body=c
    else:
        body=problem['prompt']+c
    test=problem.get('test','')
    tail=f'\n{test}\n'+(f'check({ep})\n' if 'def check' in test else '')
    return body+tail
def run_problem(problem:Dict[str,Any],generate_fn:Callable,prior_lesson:Optional[str]=None,timeout:int=10)->Dict[str,Any]:
    t0=time.time()
    try:completion=generate_fn(problem['prompt'],prior_lesson) if prior_lesson is not None else generate_fn(problem['prompt'])
    except TypeError:completion=generate_fn(problem['prompt'])
    except Exception as e:return {'task_id':problem['task_id'],'passed':False,'error':f'generate failed: {e}','completion':'','wall_s':round(time.time()-t0,2)}
    program=_assemble(problem,completion or '')
    with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8') as f:f.write(program);path=f.name
    try:
        r=subprocess.run([sys.executable,'-B',path],capture_output=True,text=True,timeout=timeout)
        passed=r.returncode==0
        err='' if passed else (r.stderr or r.stdout or 'failed')[-600:]
    except subprocess.TimeoutExpired:passed=False;err=f'timeout >{timeout}s'
    except Exception as e:passed=False;err=str(e)[:300]
    finally:
        try:os.unlink(path)
        except Exception:pass
    return {'task_id':problem['task_id'],'passed':passed,'error':err,'completion':_clean(completion or '')[:1200],'wall_s':round(time.time()-t0,2)}
def run_benchmark(generate_fn:Callable,problems:List[Dict[str,Any]],timeout:int=10,prior:Optional[Dict[str,str]]=None)->Dict[str,Any]:
    results=[];passed=0
    for p in problems:
        lesson=(prior or {}).get(p['task_id']) if prior else None
        res=run_problem(p,generate_fn,prior_lesson=lesson,timeout=timeout)
        results.append(res);passed+=1 if res['passed'] else 0
    n=len(problems) or 1
    return {'n':len(problems),'passed':passed,'pass_at_1':round(100.0*passed/n,1),'results':results}
def run_twice(generate_fn:Callable,problems:List[Dict[str,Any]],timeout:int=10,lesson_fn:Optional[Callable]=None)->Dict[str,Any]:
    """The proof: run the whole benchmark, turn each failure into a lesson, run AGAIN with those lessons injected.
    lesson_fn(problem,fail_result)->str builds the lesson (defaults to the failure's error). run2 pass@1 should exceed run1."""
    run1=run_benchmark(generate_fn,problems,timeout=timeout)
    prior={}
    for r in run1['results']:
        if not r['passed']:
            lp=next((p for p in problems if p['task_id']==r['task_id']),None)
            prior[r['task_id']]=(lesson_fn(lp,r) if lesson_fn else f"your previous attempt FAILED with: {r['error'][:300]} — fix that specific error this time")
    run2=run_benchmark(generate_fn,problems,timeout=timeout,prior=prior)
    r1=set(r['task_id'] for r in run1['results'] if r['passed']);r2=set(r['task_id'] for r in run2['results'] if r['passed'])
    return {'run1_pass_at_1':run1['pass_at_1'],'run2_pass_at_1':run2['pass_at_1'],'delta':round(run2['pass_at_1']-run1['pass_at_1'],1),'newly_fixed':sorted(r2-r1),'regressed':sorted(r1-r2),'n':run1['n'],'run1':run1,'run2':run2}
_CODEX_REF={'humaneval':{'codex_frontier_pass_at_1':92.0,'note':'GPT-5/o-series-class published HumanEval pass@1 (~90%+); code-davinci-002 was ~47%'},
            'mbpp':{'codex_frontier_pass_at_1':80.0,'note':'frontier MBPP pass@1 ~80%+'},
            'swe-bench':{'codex_frontier_pct':70.0,'note':'SWE-bench Verified frontier ~70%+; out of scope for a 2-4B single-shot model'}}
def compare_to_codex(twice_result:Dict[str,Any],benchmark:str='humaneval')->str:
    ref=_CODEX_REF.get(benchmark.lower(),{})
    codex=ref.get('codex_frontier_pass_at_1','?')
    lines=[f'=== Adam vs Codex — {benchmark} (pass@1, n={twice_result.get("n")}) ===',
           f'  Adam run 1 (cold):      {twice_result.get("run1_pass_at_1")}%',
           f'  Adam run 2 (w/ memory): {twice_result.get("run2_pass_at_1")}%   (delta {twice_result.get("delta"):+}%)',
           f'  Codex (frontier ref):   {codex}%   [{ref.get("note","")}]',
           f'  Newly fixed on run 2:   {len(twice_result.get("newly_fixed") or [])}  {twice_result.get("newly_fixed") or []}']
    if twice_result.get('regressed'):lines.append(f'  ⚠ regressed on run 2:    {twice_result["regressed"]}')
    lines.append('  Thesis: the memory loop should make run 2 > run 1. That delta is Adam\'s structural edge over a stateless agent.')
    return '\n'.join(lines)
