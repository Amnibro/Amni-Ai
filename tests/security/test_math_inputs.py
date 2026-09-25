import sys,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from amni.math import answer,solve_math,exec_calc,run_sympy,solve_eq
from amni.math.sympy_tool import job
from amni.math.answer import evaluate
jail=pytest.mark.skipif(not (sys.platform.startswith('linux') and Path('/usr/bin/bwrap').exists()),reason='sympy runs only inside the bwrap jail')
BOMBS=['9**9**9','(10**100)**100','2**99999999','10**10**10','9^9^9','99999!','factorial(100000)','2**-99999999','(1/3)**99999999','1'+'*999999999'*6]
def test_power_and_factorial_bombs_bounded():
 t=time.perf_counter()
 for e in BOMBS:
  r=solve_math(e)
  assert r is None or r['value'] is None or abs(r['value'])<10**2500,(e,r)
  x=exec_calc(e);assert x is None or (isinstance(x,str) and len(x)<=300),(e,x)
 assert time.perf_counter()-t<5
 assert 'size guard' in answer('9**9**9')[0]
def test_evaluate_refuses_non_arith():
 for bad in ('__import__("os")','().__class__','[1]*9','lambda:1','x.y','open("f")','globals()','"a"*9'):
  with pytest.raises((ValueError,OverflowError,SyntaxError,TypeError)):evaluate(bad)
def test_no_host_code_execution(tmp_path,monkeypatch):
 mk=tmp_path/'pwned';monkeypatch.chdir(tmp_path)
 import amni.serve.self_debug as sd
 monkeypatch.setattr(sd,'run_in_sandbox',lambda *a,**k:{'ran':False,'ok':False,'returncode':1,'stdout':'','stderr':'blocked in test'})
 P=[f"__import__('os').system('touch {mk}')",f"exec(\"open('{mk}','w')\")",f"eval(\"open('{mk}','w')\")","().__class__.__bases__[0].__subclasses__()",f"what is __import__('os').system('touch {mk}')",f"solve __import__('os').system('touch {mk}')=0",f"Tom has 5 apples. __import__('os').system('touch {mk}'). How many apples does Tom have?"]
 for p in P:
  for f in (answer,solve_math,exec_calc,run_sympy):
   try:f(p)
   except Exception as e:pytest.fail(f"{f.__name__} raised {e!r} on {p!r}")
 assert not mk.exists()
def test_sympy_job_rejects_dunder_without_calling_jail(monkeypatch):
 import amni.serve.skills as sk
 monkeypatch.setattr(sk,'_sympy_jail',lambda *a,**k:pytest.fail('jail must not be reached for rejected input'))
 import amni.math.sympy_tool as st
 monkeypatch.setattr(st,'_sympy_jail',lambda *a,**k:pytest.fail('jail must not be reached for rejected input'))
 assert job('eval',[],body="__import__('os')") is None and job('eval',[],body="a"*501) is None and job('solve',['xy'],eqs=[['x','1']]) is None and job('eval',[],body="x;y") is None
@jail
def test_every_sympy_call_goes_through_the_jail(monkeypatch):
 import amni.serve.self_debug as sd
 calls=[];orig=sd.run_in_sandbox
 monkeypatch.setattr(sd,'run_in_sandbox',lambda *a,**k:(calls.append(1),orig(*a,**k))[1])
 assert answer("solve x^2-5x+6=0")[0].startswith("x = 2 or x = 3") and solve_eq("y^2=9","y")=="3, -3" and run_sympy("x^2+2x+1")=="x**2 + 2*x + 1"
 assert len(calls)==3
@jail
def test_sympy_pathological_input_times_out():
 t=time.perf_counter();r=run_sympy("factorial(10^7)");answer("integrate exp(x^7)*sin(x^5)*log(x^3+1)")
 assert time.perf_counter()-t<30 and (r is None or len(r)<500)
def test_long_input_rejected_fast():
 t=time.perf_counter();assert answer('1+'*5000+'1') is None and solve_math('x'*100000) is None;assert time.perf_counter()-t<2
