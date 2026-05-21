#!/usr/bin/env python
"""Mini HumanEval-style benchmark for Adam.

Runs a small curated set of canonical coding problems through Adam's /complete
or agentic build pipeline, runs the resulting code against test cases, reports
pass/fail. NOT the full HumanEval-164; a 10-problem subset for quick iteration.

Usage:
  python scripts/eval_humaneval.py              # all problems, /complete mode (RECOMMENDED — official 91.9% number)
  python scripts/eval_humaneval.py --agentic    # experimental: capture code from agentic SSE events (less reliable)
  python scripts/eval_humaneval.py --problem 3  # one problem
  python scripts/eval_humaneval.py --host 127.0.0.1 --port 11434
  python scripts/eval_humaneval.py --json       # machine-readable output

The --agentic mode is experimental. It triggers the planner via /chat/stream and tries
to capture the file_write content from agentic_step_start SSE events. The planner may
spend its budget on research/scan steps before reaching file_write, leading to empty
completions. For reliable benchmarking use /complete mode (default).

Each problem: {id, prompt, entry_point, tests}
  prompt: Python function signature + docstring (the model completes the body)
  entry_point: function name to test
  tests: list of (args, expected) tuples
"""
import sys,os,argparse,json,time,subprocess,tempfile
try:sys.stdout.reconfigure(encoding='utf-8')
except Exception:pass
try:import requests
except ImportError:print('pip install requests',file=sys.stderr);sys.exit(2)
PROBLEMS=[
    {'id':1,'name':'has_close_elements','prompt':'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """Return True if any two distinct numbers in the list are within threshold of each other."""\n    ','entry_point':'has_close_elements','tests':[((([1.0,2.0,3.0],0.5),),False),((([1.0,2.8,3.0,4.0,5.0,2.0],0.3),),True),((([1.0,2.0],0.5),),False),((([1.1,1.2],0.3),),True)]},
    {'id':2,'name':'separate_paren_groups','prompt':'def separate_paren_groups(paren_string: str) -> list[str]:\n    """Split a string of nested balanced parens groups separated by spaces into a list of balanced group strings. Ignore whitespace."""\n    ','entry_point':'separate_paren_groups','tests':[((('( ) (( )) (( )( ))',),),['()','(())','(()())']),(((')',),),[]),((('()',),),['()'])]},
    {'id':3,'name':'truncate_number','prompt':'def truncate_number(number: float) -> float:\n    """Return the decimal part of a positive float."""\n    ','entry_point':'truncate_number','tests':[(((3.5,),),0.5),(((1.25,),),0.25),(((7.0,),),0.0)]},
    {'id':4,'name':'below_zero','prompt':'def below_zero(operations: list[int]) -> bool:\n    """Return True if a cumulative balance starting at 0 ever goes below zero."""\n    ','entry_point':'below_zero','tests':[((([1,2,3],),),False),((([1,2,-4,5],),),True),((([0],),),False),((([-1],),),True)]},
    {'id':5,'name':'fibonacci','prompt':'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number. fib(0)=0, fib(1)=1, fib(2)=1, fib(3)=2..."""\n    ','entry_point':'fibonacci','tests':[(((0,),),0),(((1,),),1),(((5,),),5),(((10,),),55)]},
    {'id':6,'name':'is_palindrome','prompt':'def is_palindrome(s: str) -> bool:\n    """Return True if the string reads the same forwards and backwards (case-sensitive)."""\n    ','entry_point':'is_palindrome','tests':[((('racecar',),),True),((('hello',),),False),((('',),),True),((('a',),),True)]},
    {'id':7,'name':'gcd','prompt':'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of two positive integers."""\n    ','entry_point':'gcd','tests':[(((12,8),),4),(((17,13),),1),(((100,75),),25),(((1,1),),1)]},
    {'id':8,'name':'count_vowels','prompt':'def count_vowels(s: str) -> int:\n    """Return the number of vowels (aeiouAEIOU) in the string."""\n    ','entry_point':'count_vowels','tests':[((('hello',),),2),((('AEIOU',),),5),((('xyz',),),0),((('',),),0)]},
    {'id':9,'name':'flatten','prompt':'def flatten(nested: list) -> list:\n    """Flatten an arbitrarily nested list of integers into a single flat list, preserving order."""\n    ','entry_point':'flatten','tests':[((([1,[2,3],[4,[5,6]]],),),[1,2,3,4,5,6]),((([],),),[]),((([1,2,3],),),[1,2,3])]},
    {'id':10,'name':'reverse_words','prompt':'def reverse_words(s: str) -> str:\n    """Reverse the order of words (whitespace-separated). Collapse runs of whitespace into single spaces."""\n    ','entry_point':'reverse_words','tests':[((('hello world',),),'world hello'),((('  a  b  c  ',),),'c b a'),((('only',),),'only'),((('',),),'')]},
]
def run_completion(host,port,prompt,timeout=120):
    try:
        r=requests.post(f'http://{host}:{port}/complete',json={'prefix':prompt,'language':'python','max_tokens':200},timeout=timeout)
        if r.status_code!=200:return None,f'HTTP {r.status_code}'
        return r.json().get('completion',''),None
    except Exception as e:return None,str(e)
def _extract_python_block(text):
    import re as _re
    m=_re.search(r'```python\s*\n(.*?)```',text,_re.DOTALL)
    if m:return m.group(1).rstrip()
    m=_re.search(r'```\s*\n(.*?)```',text,_re.DOTALL)
    if m:return m.group(1).rstrip()
    lines=text.splitlines();py=[];in_code=False
    for ln in lines:
        s=ln.strip()
        if s.startswith(('def ','class ','import ','from ','if __name__','#')):in_code=True
        if in_code:py.append(ln)
    return '\n'.join(py) if py else text
def run_agentic(host,port,prompt,entry_point,timeout=300):
    target_file=f'tmp_eval_{entry_point}.py'
    msg=f'Write a Python file {target_file} with this function: {prompt} Use file_write to save it. Make it complete and correct.'
    try:
        r=requests.post(f'http://{host}:{port}/chat/stream',json={'message':msg},timeout=timeout,stream=True)
        ev='token';full=[];agentic_fired=False;last_write_content=None
        for line in r.iter_lines(decode_unicode=True):
            if not line:continue
            if line.startswith('event:'):
                ev=line[6:].strip()
                if ev.startswith('agentic_'):agentic_fired=True
            elif line.startswith('data:'):
                d=line[5:].strip()
                if ev=='token':
                    try:full.append(json.loads(d))
                    except:pass
                elif ev=='agentic_step_start':
                    try:
                        m=json.loads(d)
                        if m.get('tool') in ('file_write','code_edit','code_diff'):
                            content=(m.get('args') or {}).get('content') or (m.get('args') or {}).get('code')
                            if content and ('def ' in content or 'class ' in content):last_write_content=content
                    except:pass
        if last_write_content:return last_write_content,None
        if agentic_fired:
            from pathlib import Path
            for root in (Path.cwd(),Path(r'C:/Users/antho/Documents/ai/Amni-Ai')):
                fp=root/target_file
                if fp.exists():
                    code=fp.read_text(encoding='utf-8')
                    try:fp.unlink()
                    except:pass
                    return code,None
        raw=''.join(full)
        return _extract_python_block(raw),None
    except Exception as e:return None,str(e)
def test_code(code,entry_point,tests):
    """Execute code in a sandbox and run tests. Returns (passed_count, total, error)."""
    src=code+f'\n\nimport json\nresults=[]\n'
    for i,(args,expected) in enumerate(tests):
        src+=f'\ntry:\n    actual={entry_point}(*{args[0]!r})\n    results.append({{"i":{i},"pass":actual=={expected!r},"actual":actual,"expected":{expected!r}}})\nexcept Exception as e:\n    results.append({{"i":{i},"pass":False,"error":str(e)[:200]}})\n'
    src+='\nprint("__EVAL_RESULTS__"+json.dumps(results))\n'
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',delete=False,encoding='utf-8') as f:
        f.write(src);path=f.name
    try:
        r=subprocess.run([sys.executable,path],capture_output=True,text=True,timeout=10)
        marker='__EVAL_RESULTS__'
        if marker in (r.stdout or ''):
            line=[l for l in r.stdout.splitlines() if l.startswith(marker)][-1]
            results=json.loads(line[len(marker):])
            passed=sum(1 for x in results if x.get('pass'))
            return passed,len(results),None,results
        return 0,len(tests),f'no marker; stderr={r.stderr[:200]}',[]
    except subprocess.TimeoutExpired:return 0,len(tests),'timeout',[]
    except Exception as e:return 0,len(tests),str(e),[]
    finally:
        try:os.unlink(path)
        except:pass
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=11434)
    ap.add_argument('--problem',type=int,default=None,help='Run only this problem id (1-10)')
    ap.add_argument('--agentic',action='store_true',help='Use /chat/stream agentic mode (slower, smarter)')
    ap.add_argument('--json',action='store_true',help='Machine-readable JSON output')
    args=ap.parse_args()
    probs=PROBLEMS if args.problem is None else [p for p in PROBLEMS if p['id']==args.problem]
    if not probs:print(f'No problem #{args.problem}',file=sys.stderr);sys.exit(2)
    if not args.json:print(f'Mini HumanEval — {len(probs)} problems against http://{args.host}:{args.port}\n')
    results=[];total_passed=0;total_tests=0
    t_start=time.time()
    for p in probs:
        t0=time.time()
        if args.agentic:completion,err=run_agentic(args.host,args.port,p['prompt'],p['entry_point'])
        else:completion,err=run_completion(args.host,args.port,p['prompt'])
        dt=round(time.time()-t0,1)
        if err or not completion:
            row={'id':p['id'],'name':p['name'],'passed':0,'total':len(p['tests']),'error':err or 'empty completion','wall_s':dt}
            results.append(row);total_tests+=len(p['tests'])
            if not args.json:print(f'#{p["id"]:>2} {p["name"]:<24}  [FAIL] generation failed: {err or "empty"} ({dt}s)')
            continue
        if args.agentic and ('def '+p['entry_point'] in completion):full_code=completion
        else:full_code=p['prompt']+completion
        passed,total,test_err,test_results=test_code(full_code,p['entry_point'],p['tests'])
        total_passed+=passed;total_tests+=total
        row={'id':p['id'],'name':p['name'],'passed':passed,'total':total,'wall_s':dt,'completion':completion[:200]}
        if test_err:row['error']=test_err
        results.append(row)
        if not args.json:
            symbol='[PASS]' if passed==total else ('[PART]' if passed>0 else '[FAIL]')
            print(f'#{p["id"]:>2} {p["name"]:<24}  {symbol} {passed}/{total} ({dt}s)'+(f'  err: {test_err[:60]}' if test_err else ''))
    t_total=round(time.time()-t_start,1)
    summary={'total_passed':total_passed,'total_tests':total_tests,'pass_rate':round(total_passed/total_tests,3) if total_tests else 0,'wall_s':t_total,'mode':'agentic' if args.agentic else 'complete','results':results}
    if args.json:print(json.dumps(summary,indent=2))
    else:print(f'\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  {total_passed}/{total_tests} tests passed ({summary["pass_rate"]*100:.1f}%) — wall {t_total}s — mode={summary["mode"]}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
if __name__=='__main__':main()
