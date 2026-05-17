"""Unit test for _extract_asserts + _run_with_tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _extract_asserts,_run_with_tests
SAMPLE='''**CLARIFY:** Input is N, output is fibonacci(N).
**APPROACH:** Memoization with dict.
```python
def fib(n,memo={}):
    if n in memo:return memo[n]
    if n<=1:return n
    memo[n]=fib(n-1,memo)+fib(n-2,memo)
    return memo[n]
print(f"Fib(10): {fib(10)}")
```
**TESTS:**
`assert fib(0) == 0`
`assert fib(6) == 8`
assert fib(10) == 55
**COMPLEXITY:** Time O(N), Space O(N).'''
asserts=_extract_asserts(SAMPLE)
print(f'extracted asserts: {asserts}')
assert len(asserts)==3,f'expected 3 asserts, got {len(asserts)}: {asserts}'
assert 'assert fib(0) == 0' in asserts
assert 'assert fib(6) == 8' in asserts
assert 'assert fib(10) == 55' in asserts
print('  test_extract PASS')
SAMPLE_BAD='```python\nprint("hi")\n```\nNo asserts here.'
assert _extract_asserts(SAMPLE_BAD)==[],'should find no asserts'
print('  test_no_asserts PASS')
SAMPLE_INLINE='Some text `assert x == 1` and more `assert y > 0` end'
res=_extract_asserts(SAMPLE_INLINE)
print(f'  inline test (text on single line, no newlines): {res}')
class FakeResult:
    def __init__(self,ok,output):self.ok=ok;self.output=output
class FakeSkills:
    def __init__(self,r):self.r=r
    def call(self,name,args,ctx):
        code=args['code']
        if 'assert fib(0)' in code and 'def fib' in code:return FakeResult(True,{'returncode':0,'stdout':'Fib(10): 55\nALL_TESTS_PASS\n','stderr':'','timed_out':False})
        if 'assert wrong' in code:return FakeResult(True,{'returncode':1,'stdout':'','stderr':'AssertionError\n','timed_out':False})
        return FakeResult(True,{'returncode':0,'stdout':'ok','stderr':'','timed_out':False})
fib_code='def fib(n,memo={}):\n    if n in memo:return memo[n]\n    if n<=1:return n\n    memo[n]=fib(n-1,memo)+fib(n-2,memo)\n    return memo[n]\nprint(f"Fib(10): {fib(10)}")'
passed,err,info=_run_with_tests(FakeSkills(None),None,fib_code,['assert fib(0) == 0','assert fib(6) == 8','assert fib(10) == 55'])
assert passed,f'should pass: err={err} info={info}'
print(f'  test_passing_asserts PASS  info={info}')
passed,err,info=_run_with_tests(FakeSkills(None),None,'def f():pass','assert wrong'.split('\n'))
assert not passed and 'AssertionError' in err,f'should fail: {err}'
print(f'  test_failing_asserts PASS  err={err[:60]}')
passed,err,info=_run_with_tests(FakeSkills(None),None,'def f():pass',[])
assert passed and err==''
print('  test_no_asserts_passthrough PASS')
print('ALL PASS')
