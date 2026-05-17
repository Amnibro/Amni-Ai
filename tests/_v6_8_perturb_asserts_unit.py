"""Iter20 unit: perturb retry must re-validate asserts; success only when asserts pass too."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _perturb_retry
class FakeAdam:
    def __init__(self,scripted):self.scripted=list(scripted);self.calls=0
    def chat_persona(self,message,system,max_new_tokens,do_sample):
        ans=self.scripted.pop(0) if self.scripted else ''
        self.calls+=1
        return {'answer':ans,'tier':'fake','tokens':10}
class FakeResult:
    def __init__(self,ok,output):self.ok=ok;self.output=output
class FakeSkills:
    def __init__(self,scripted):self.scripted=list(scripted)
    def call(self,name,args,ctx):return self.scripted.pop(0) if self.scripted else FakeResult(False,{'error':'no script'})
def test_perturb_runs_but_asserts_fail_then_succeed():
    adam=FakeAdam([
        '```python\ndef add_one(n):return n+2\n```',
        '```python\ndef add_one(n):return n+1\n```'])
    skills=FakeSkills([
        FakeResult(True,{'returncode':0,'stdout':'','stderr':'','timed_out':False}),
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'AssertionError\n','timed_out':False}),
        FakeResult(True,{'returncode':0,'stdout':'','stderr':'','timed_out':False}),
        FakeResult(True,{'returncode':0,'stdout':'ALL_TESTS_PASS\n','stderr':'','timed_out':False})])
    emits=[]
    r=_perturb_retry(adam,skills,'sys','def add_one(n):return n','init err','test',asserts=['assert add_one(5)==6','assert add_one(0)==1'],emit=emits.append)
    print(f'  result: {r["success"]}  mag={r.get("magnitude")}  history_len={len(r.get("history",[]))}')
    print(f'  emits: {emits}')
    assert r['success'] and r['magnitude']=='MEDIUM',f'expected MEDIUM success: {r}'
    assert r.get('tests_passed') is True
    print('  test_perturb_runs_but_asserts_fail_then_succeed PASS')
def test_perturb_skips_when_no_asserts():
    adam=FakeAdam(['```python\nprint("hi")\n```'])
    skills=FakeSkills([FakeResult(True,{'returncode':0,'stdout':'hi','stderr':'','timed_out':False})])
    r=_perturb_retry(adam,skills,'sys','bad','err','test',asserts=None)
    assert r['success'] and r.get('tests_passed') is False,f'no asserts -> success but tests_passed=False: {r}'
    print('  test_perturb_skips_when_no_asserts PASS')
def test_all_three_fail_asserts():
    adam=FakeAdam(['```python\ndef f(n):return 1\n```']*3)
    skills=FakeSkills([
        FakeResult(True,{'returncode':0,'stdout':'','stderr':'','timed_out':False}),
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'AssertionError','timed_out':False}),
        FakeResult(True,{'returncode':0,'stdout':'','stderr':'','timed_out':False}),
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'AssertionError','timed_out':False}),
        FakeResult(True,{'returncode':0,'stdout':'','stderr':'','timed_out':False}),
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'AssertionError','timed_out':False})])
    r=_perturb_retry(adam,skills,'sys','bad','err','test',asserts=['assert f(5)==6'])
    assert not r['success']
    print(f'  test_all_three_fail_asserts PASS  history_len={len(r.get("history",[]))}')
print('=== _perturb_retry with asserts unit ===')
test_perturb_runs_but_asserts_fail_then_succeed()
test_perturb_skips_when_no_asserts()
test_all_three_fail_asserts()
print('ALL PASS')
