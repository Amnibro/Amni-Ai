import ast,os,subprocess,sys,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
X0=ROOT/'scripts'/'x0_code_eval.py'
SERVE=ROOT/'amni'/'serve'
PY=sys.executable
_JAIL=sys.platform.startswith('linux') and Path('/usr/bin/bwrap').exists() and X0.exists()
jail=pytest.mark.skipif(not _JAIL,reason='bwrap jail needs linux + /usr/bin/bwrap + scripts/x0_code_eval.py')
_ENV_OK={'PATH','HOME','LANG','LC_CTYPE','PWD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'}
def _sbx(code,**kw):
    from amni.serve.self_debug import run_in_sandbox
    return run_in_sandbox(code,**kw)
def _lines(r):return [l for l in (r.get('stdout') or '').splitlines() if l.strip()]
def _sub(src,timeout=90):
    t0=time.perf_counter();p=subprocess.run([PY,'-c',src],cwd=str(ROOT),capture_output=True,text=True,timeout=timeout,env={**os.environ,'PYTHONPATH':str(ROOT)})
    return p,time.perf_counter()-t0
@jail
def test_x0_bwrap_hardening_flags():
    from scripts.x0_code_eval import _bwrap_cmd
    cmd=_bwrap_cmd('/tmp/amni_flagcheck',['/bin/true'],cpu=1,mem=1<<20)
    for f in ('--unshare-user','--disable-userns','--unshare-net','--unshare-pid','--unshare-ipc','--unshare-uts','--clearenv','--new-session','--die-with-parent'):assert f in cmd,f
    assert cmd[cmd.index('--cap-drop')+1]=='ALL' and cmd[cmd.index('--hostname')+1]!=os.uname().nodename
    assert cmd[cmd.index('--size')+2]=='--tmpfs' and int(cmd[cmd.index('--size')+1])<=256<<20
    assert 'OPENBLAS_NUM_THREADS' in cmd and 'OMP_NUM_THREADS' in cmd
    assert 'env={"PATH":"/usr/bin:/bin"' in X0.read_text(encoding='utf-8') and 'env=os.environ' not in X0.read_text(encoding='utf-8')
@jail
def test_no_home_reads():
    home=Path.home();marker=ROOT/'CLAUDE.md'
    assert marker.exists()
    r=_sbx(f'import os\nprint(os.path.exists({str(marker)!r}))\nprint(os.path.exists({str(home/".ssh")!r}))\nprint(sorted(os.listdir({str(ROOT)!r})) if os.path.isdir({str(ROOT)!r}) else [])')
    assert r['ran'] and r.get('jailed'),r
    a,b,ls=_lines(r)[-3:]
    assert a=='False' and b=='False'
    assert set(ast.literal_eval(ls))<={'.venv-linux','data'}
@jail
def test_no_host_env(monkeypatch):
    monkeypatch.setenv('AMNI_SECFIX_SENTINEL','host-only')
    r=_sbx('import os\nprint(",".join(sorted(os.environ)))\nprint(",".join(sorted(k.split(b"=")[0].decode() for k in open("/proc/1/environ","rb").read().split(b"\\0") if k)))\nprint(dict(k.split(b"=",1) for k in open("/proc/1/environ","rb").read().split(b"\\0") if k).get(b"PATH",b"").decode())',allow_dangerous=True)
    assert r['ran'] and r.get('jailed'),r
    env,pid1,pid1path=_lines(r)[-3:]
    assert set(env.split(','))<=_ENV_OK,env
    assert set(pid1.split(','))<={'PATH','LANG'},pid1
    assert pid1path=='/usr/bin:/bin'
    assert 'AMNI_SECFIX_SENTINEL' not in env+pid1
@jail
def test_no_network():
    r=_sbx('import socket\nprint(",".join(sorted(n for _,n in socket.if_nameindex())))',allow_dangerous=True)
    assert r['ran'] and _lines(r)[-1]=='lo',r
@jail
def test_no_fork():
    r=_sbx('import subprocess\ntry:\n    subprocess.run(["/usr/bin/true"]);print("spawned")\nexcept OSError as e:print("denied",e.errno)',allow_dangerous=True)
    assert r['ran'] and _lines(r)[-1].startswith('denied'),r
def test_output_cap(monkeypatch):
    monkeypatch.setenv('AMNI_SANDBOX_MAX_OUTPUT','4000')
    r=_sbx('for i in range(20000):\n    print("y"*80)',timeout=6)
    assert r['ran'] and r['capped'] and r['killed']=='output-cap' and not r['ok'],r
def test_timeout():
    t0=time.perf_counter();r=_sbx('while True:\n    pass',timeout=2)
    assert r['timed_out'] and not r['ok'] and time.perf_counter()-t0<8
@jail
def test_fail_closed_without_jail(monkeypatch):
    import amni.serve.self_debug as sd
    monkeypatch.setattr(sd,'_BWRAP','/nonexistent/bwrap');monkeypatch.delenv('AMNI_SANDBOX_ALLOW_UNJAILED',raising=False)
    r=sd.run_in_sandbox('print(1)')
    assert r['ran'] is False and 'sandbox unavailable' in r['error']
@jail
def test_adversarial_probe_runs_in_jail():
    from amni.serve.self_debug import adversarial_probe
    marker=ROOT/'CLAUDE.md'
    r=adversarial_probe(f'import os\nV=os.path.exists({str(marker)!r})\ndef f(xs):\n    assert not V,"host fs visible"\n    return len(xs)')
    assert r['ok'] and r['func']=='f' and r['probes_run']>=5,r
def test_adversarial_probe_hang_bounded():
    from amni.serve.self_debug import adversarial_probe
    t0=time.perf_counter();r=adversarial_probe('def f(xs):\n    while True:\n        pass',timeout=3)
    assert time.perf_counter()-t0<10 and not r['ok']
def test_exec_calc_bounded():
    p,dt=_sub('from amni.math import exec_calc;import time;t=time.perf_counter();print(repr(exec_calc("9**9**9")),repr(exec_calc("2^10+3")),repr(exec_calc("(10**100)**100")),round(time.perf_counter()-t,2))')
    assert p.returncode==0,p.stderr[-800:]
    big,ok,big2,t=p.stdout.split()[-4:]
    assert big=='None' and ok=="'1027'" and big2=='None' and float(t)<3,p.stdout
def test_calc_skill_bounded():
    p,dt=_sub('from amni.serve.skills import _try_python_eval,_skill_calc;print(_try_python_eval("9**9**9"),_try_python_eval("2**10+3"),"error" in _skill_calc({"expr":"9**9**9"},{},None))')
    assert p.returncode==0,p.stderr[-800:]
    assert p.stdout.split()[-3:]==['False','1027.0','True'] and dt<60
def test_safe_arith_rejects_non_arith():
    from amni.serve.code_safety import safe_arith
    assert safe_arith('2**10+3*4')==1036 and safe_arith('7//2')==3
    for bad in ('9**9**9','(2**8000)*(2**8000)','__import__("os")','abs(1)','[1]*9','1 if 1 else 2'):
        with pytest.raises((ValueError,OverflowError,SyntaxError)):safe_arith(bad)
@jail
def test_sympy_calc_runs_jailed(monkeypatch):
    import amni.serve.self_debug as sd
    from amni.serve import skills
    calls=[];orig=sd.run_in_sandbox
    monkeypatch.setattr(sd,'run_in_sandbox',lambda *a,**k:(calls.append(1),orig(*a,**k))[1])
    assert skills._sympy_calc('expand (x+1)^2')=='x**2 + 2*x + 1'
    assert skills._sympy_calc('solve x^2-4=0')=='[-2, 2]'
    r=skills._skill_solve({'equation':'V=I*R','values':{'I':2,'R':5}},{},None)
    assert r.get('solution')==10.0 and r.get('solve_for')=='V',r
    assert len(calls)==3
def _fns(path,name=None):
    t=ast.parse(path.read_text(encoding='utf-8'))
    return [n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and (name is None or n.name==name)]
def _shell_true(fn):
    return any(isinstance(c,ast.Call) and any(k.arg=='shell' and isinstance(k.value,ast.Constant) and k.value.value is True for k in c.keywords) for c in ast.walk(fn))
def test_no_shell_true_in_skill_exec_paths():
    for f in ('skills.py','venv_manager.py','self_debug.py','code_safety.py'):
        bad=[fn.name for fn in _fns(SERVE/f) if _shell_true(fn)]
        assert not bad,f'{f}: shell=True in {bad}'
@jail
def test_test_run_jailed_no_shell_masks_secrets(tmp_path):
    from amni.serve.skills import _skill_test_run
    (tmp_path/'.env').write_text('TOKEN=placeholder-not-a-secret\n');marker=ROOT/'CLAUDE.md'
    (tmp_path/'probe.py').write_text(f'import os\nprint("HOST",os.path.exists({str(marker)!r}))\ntry:print("ENV",len(open(".env").read()))\nexcept OSError:print("ENV","denied")\nprint("KEYS",",".join(sorted(os.environ)))\n')
    reg=type('R',(),{'workdir':tmp_path})()
    r=_skill_test_run({'cmd':'python probe.py','timeout':30},{},reg)
    assert r.get('passed') and r.get('sandbox')=='bwrap',r
    out=dict(l.split(' ',1) for l in r['stdout'].splitlines() if ' ' in l)
    assert out['HOST']=='False' and out['ENV'] in ('0','denied') and set(out['KEYS'].split(','))<=_ENV_OK
    e=_skill_test_run({'cmd':'echo a && echo b','timeout':30},{},reg)
    assert e['stdout'].strip()=='a && echo b',e
def test_pc_action_confirm_requires_owner():
    from amni.serve.skills import default_registry
    reg=default_registry(workdir=str(ROOT),audit_log=None)
    p=reg.call('pc_action',{'action':'propose','pc_action':'echo','target':'hi'},ctx={})
    tok=p.output['token']
    r=reg.call('pc_action',{'action':'confirm','token':tok},ctx={})
    assert 'owner' in str(r.output.get('error','')),r.output
    ok=reg.call('pc_action',{'action':'confirm','token':tok},ctx={'owner':True})
    assert ok.output.get('executed') or ok.output.get('result'),ok.output
def test_shell_and_git_gates_block_exec_vectors():
    from amni.serve.skills import _gate_shell,_skill_git
    reg=type('R',(),{'workdir':ROOT,'_in_allowed_roots':lambda s,p:True})()
    for cmd in ('git -c alias.z=!true z','git --config-env=core.pager=X log','find . -delete','find . "-exec" true','git log --output=x.txt','git config core.fsmonitor true','git push'):
        assert _gate_shell({'cmd':cmd},{},reg),cmd
    assert _gate_shell({'cmd':'git status'},{},reg) is None and _gate_shell({'cmd':'git config --get user.name'},{},reg) is None
    for cmd in ('config core.fsmonitor true','log --output=x.txt','-c alias.z=!true status','diff --ext-diff'):
        assert 'error' in _skill_git({'cmd':cmd},{},reg),cmd
def test_file_write_refuses_git_metadata():
    from amni.serve.skills import _write_protected
    assert _write_protected('.git/config') and _write_protected('sub/.git/hooks/pre-commit') and not _write_protected('src/gitutil.py')
def test_suite_stays_defensive():
    here=Path(__file__).read_text(encoding='utf-8')
    for b in ('os.'+'fork(','fork'+' bomb','/dev/'+'zero','dd '+'if=','PTRACE_'+'ATTACH','socket.'+'socket(','rm '+'-rf'):assert b not in here
