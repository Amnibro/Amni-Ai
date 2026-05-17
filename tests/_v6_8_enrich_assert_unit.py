"""Iter23 unit: _enrich_assert AST rewrites + end-to-end test that rich failures flow into stderr."""
import sys,subprocess,tempfile,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _enrich_assert
def t_eq_assert():
    e=_enrich_assert('assert fib(10) == 55')
    print(f'  eq: {e}')
    assert '_lhs' in e and '_rhs' in e and 'FAILED' in e
def t_neq_assert():
    e=_enrich_assert('assert x != y')
    print(f'  neq: {e}')
    assert '!=' in e and 'FAILED' in e
def t_in_assert():
    e=_enrich_assert('assert 3 in [1,2,3]')
    print(f'  in: {e}')
    assert '_lhs' in e and ' in ' in e
def t_truthy_assert():
    e=_enrich_assert('assert is_prime(7)')
    print(f'  truthy: {e}')
    assert '_v=' in e and 'FAILED' in e
def t_already_has_msg():
    s='assert fib(10) == 55, "bad fib"'
    e=_enrich_assert(s)
    assert e==s,f'should preserve existing msg: {e}'
    print('  preserves existing msg PASS')
def t_malformed():
    s='not an assert'
    e=_enrich_assert(s)
    assert e==s,f'should passthrough non-asserts: {e}'
    print('  passthrough PASS')
print('=== _enrich_assert unit ===')
t_eq_assert();t_neq_assert();t_in_assert();t_truthy_assert();t_already_has_msg();t_malformed()
print('=== end-to-end: rich failure message flows through subprocess ===')
buggy_snippet='def fib(n):\n    return n+1  # WRONG'
asserts=['assert fib(10) == 55','assert fib(0) == 0']
enriched='\n'.join(_enrich_assert(a) for a in asserts)
test_code=buggy_snippet+'\n'+enriched
with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False) as f:
    f.write(test_code);p=f.name
try:
    r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=5)
    print(f'  rc={r.returncode}\n  stderr:\n{r.stderr}')
    assert r.returncode!=0
    assert 'FAILED' in r.stderr
    assert 'lhs=11' in r.stderr or 'lhs=11,' in r.stderr,'expected rich lhs in stderr'
    assert '55' in r.stderr,'expected rhs (55) in stderr'
    print('  rich assertion E2E PASS — perturb loop will now see actual values')
finally:os.unlink(p)
print('ALL PASS')
