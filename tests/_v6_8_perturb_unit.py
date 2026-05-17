"""Unit smoke for _perturb_retry — no Adam boot. Verifies SMALL->MEDIUM->LARGE escalation."""
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
def test_success_on_small():
    adam=FakeAdam(['```python\nprint(42)\n```'])
    skills=FakeSkills([FakeResult(True,{'returncode':0,'stdout':'42','stderr':'','timed_out':False})])
    emits=[]
    r=_perturb_retry(adam,skills,'sys','print(1/0)','ZeroDivisionError','test',emit=emits.append)
    assert r['success'],f'should succeed: {r}'
    assert r['magnitude']=='SMALL',f'should be SMALL: {r}'
    assert r['stdout']=='42'
    print(f'  test_success_on_small PASS  emits={len(emits)} adam_calls={adam.calls}')
def test_escalate_to_large():
    adam=FakeAdam(['```python\nx=1\n```','```python\ny=2\n```','```python\nprint("OK")\n```'])
    skills=FakeSkills([
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'err1','timed_out':False}),
        FakeResult(True,{'returncode':1,'stdout':'','stderr':'err2','timed_out':False}),
        FakeResult(True,{'returncode':0,'stdout':'OK','stderr':'','timed_out':False})])
    emits=[]
    r=_perturb_retry(adam,skills,'sys','bad','init err','test',emit=emits.append)
    assert r['success'] and r['magnitude']=='LARGE',f'should escalate to LARGE: {r}'
    assert len(emits)==3,f'should emit 3: {emits}'
    print(f'  test_escalate_to_large PASS  steps={len(r["history"])}')
def test_fail_all_three():
    adam=FakeAdam(['```python\nx=1\n```','```python\ny=2\n```','```python\nz=3\n```'])
    skills=FakeSkills([FakeResult(True,{'returncode':1,'stdout':'','stderr':f'err{i}','timed_out':False}) for i in range(3)])
    r=_perturb_retry(adam,skills,'sys','bad','init err','test')
    assert not r['success'],f'should fail: {r}'
    assert len(r['history'])==3
    print(f'  test_fail_all_three PASS')
def test_no_code_block():
    adam=FakeAdam(['I cannot fix this'])
    skills=FakeSkills([])
    emits=[]
    r=_perturb_retry(adam,skills,'sys','bad','err','test',max_steps=1,emit=emits.append)
    assert not r['success']
    assert emits[0]['status']=='no_code_block',emits
    print(f'  test_no_code_block PASS')
def test_syntax_error_in_perturbation():
    adam=FakeAdam(['```python\ndef bad(\n```','```python\nprint("ok")\n```'])
    skills=FakeSkills([FakeResult(True,{'returncode':0,'stdout':'ok','stderr':'','timed_out':False})])
    emits=[]
    r=_perturb_retry(adam,skills,'sys','bad','err','test',max_steps=2,emit=emits.append)
    assert r['success'] and r['magnitude']=='MEDIUM',f'{r}'
    assert emits[0]['status']=='syntax_error',emits
    print(f'  test_syntax_error_in_perturbation PASS')
print('=== _perturb_retry unit smoke ===')
test_success_on_small()
test_escalate_to_large()
test_fail_all_three()
test_no_code_block()
test_syntax_error_in_perturbation()
print('ALL PASS')
