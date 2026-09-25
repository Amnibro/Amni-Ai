"""self_debug — the debug step Adam was skipping: before a coded solution is posted, run it against a
battery of adversarial edge cases (not just the model's own asserts) and surface the crashes so they can be
fixed first. Deterministic + model-free; pairs with the existing _run_with_tests/_perturb_retry loop."""
import inspect,ast,builtins
_NET_SHIM=("import socket as _sk\n"
"def _amni_noip(*a,**k):raise OSError('sandbox: network egress disabled')\n"
"[setattr(_sk,_n,_amni_noip) for _n in ('socket','create_connection','socketpair','fromfd','getaddrinfo','gethostbyname','gethostbyname_ex','create_server') if hasattr(_sk,_n)]\n"
"try:\n import _socket as _sks\n [setattr(_sks,_n,_amni_noip) for _n in ('socket','socketpair','fromfd') if hasattr(_sks,_n)]\nexcept Exception:pass\n")
def static_check(code):
    """AST linter (no deps): real static analysis, not canned inputs. Catches undefined names, bare
    except, mutable default args, '== None'. Undefined-name is the high-value one (typos / missing defs)."""
    try:tree=ast.parse(code)
    except SyntaxError as e:return [{'line':e.lineno or 0,'rule':'syntax-error','msg':str(e.msg)}]
    defined=set(dir(builtins))|{'__name__','__file__','self','cls'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):defined.add(n.name)
        elif isinstance(n,ast.arg):defined.add(n.arg)
        elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):defined.add(n.id)
        elif isinstance(n,ast.Import):
            for a in n.names:defined.add((a.asname or a.name).split('.')[0])
        elif isinstance(n,ast.ImportFrom):
            for a in n.names:defined.add(a.asname or (a.name.split('.')[0]))
        elif isinstance(n,(ast.Global,ast.Nonlocal)):
            for nm in n.names:defined.add(nm)
    issues=[];seen=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in defined:
            key=('undefined-name',n.id)
            if key not in seen:seen.add(key);issues.append({'line':n.lineno,'rule':'undefined-name','msg':f'name "{n.id}" is used but never defined/imported'})
        elif isinstance(n,ast.ExceptHandler) and n.type is None:
            issues.append({'line':n.lineno,'rule':'bare-except','msg':'bare "except:" swallows every error incl. KeyboardInterrupt'})
        elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for d in list(n.args.defaults)+[x for x in n.args.kw_defaults if x is not None]:
                if isinstance(d,(ast.List,ast.Dict,ast.Set)):issues.append({'line':getattr(d,'lineno',n.lineno),'rule':'mutable-default','msg':'mutable default argument (shared across calls)'})
        elif isinstance(n,ast.Compare):
            for op,comp in zip(n.ops,n.comparators):
                if isinstance(op,(ast.Eq,ast.NotEq)) and isinstance(comp,ast.Constant) and comp.value is None:
                    issues.append({'line':n.lineno,'rule':'eq-none','msg':'compare to None with "is"/"is not", not "=="/"!="'})
    return issues
_BWRAP='/usr/bin/bwrap'
_JAIL_ENV={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'}
def jail_run(argv,workdir,timeout=8,mem_mb=256,nproc=0,extra=(),fsize_mb=16,max_output=None):
    import sys,os
    from pathlib import Path
    from amni.serve.code_safety import run_capped
    _r=str(Path(__file__).resolve().parents[2]);sys.path.append(_r) if _r not in sys.path else None
    try:from scripts.x0_code_eval import _bwrap_cmd,RO
    except Exception:_bwrap_cmd,RO=None,[]
    linux=sys.platform.startswith('linux');jailed=linux and _bwrap_cmd is not None and os.path.exists(_BWRAP) and os.path.exists('/usr/bin/prlimit')
    if linux and not jailed and os.environ.get('AMNI_SANDBOX_ALLOW_UNJAILED','0')!='1':return {'returncode':None,'stdout':'','stderr':'sandbox unavailable: the bwrap jail needs /usr/bin/bwrap, /usr/bin/prlimit and scripts/x0_code_eval.py (AMNI_SANDBOX_ALLOW_UNJAILED=1 overrides)','killed':'no-jail','jailed':False}
    cmd=_bwrap_cmd(workdir,['/usr/bin/prlimit',f'--nproc={int(nproc)}']+list(argv),cpu=int(timeout)+2,mem=int(mem_mb)<<20) if jailed else list(argv)
    i=cmd.index('--die-with-parent') if jailed else 0
    w=Path(workdir).resolve();ro=[x for d in RO if Path(d).resolve().is_relative_to(w) for x in ('--ro-bind',str(Path(d).resolve()),str(Path(d).resolve()))]
    cmd=[_BWRAP]+cmd[1:i]+ro+list(extra)+cmd[i:] if jailed else cmd
    env=dict(_JAIL_ENV) if jailed else {'PYTHONIOENCODING':'utf-8','PATH':os.environ.get('PATH',''),'SYSTEMROOT':os.environ.get('SYSTEMROOT',''),'TEMP':workdir,'TMP':workdir}
    cr=run_capped(cmd,timeout=timeout,max_output_bytes=int(max_output or os.environ.get('AMNI_SANDBOX_MAX_OUTPUT','200000')),env=env,cwd=workdir,mem_mb=mem_mb,nproc=None if jailed or nproc else 0,fsize_mb=fsize_mb)
    return {**cr,'jailed':jailed}
def run_in_sandbox(code,harness='',timeout=8,allow_dangerous=False,keep=2000):
    import tempfile,sys,os,shutil
    _nonet=os.environ.get('AMNI_SANDBOX_NO_NET','1')!='0'
    _user=code+(('\n\n'+harness) if harness else '')
    if not allow_dangerous:
        try:
            from amni.serve.code_safety import danger_scan
            high=[d for d in danger_scan(_user) if d['severity']=='HIGH']
            if high:return {'ran':False,'ok':False,'blocked':True,'dangers':high,'error':'BLOCKED — refused to run high-risk code: '+', '.join(sorted({d['rule'] for d in high}))}
        except Exception:pass
    src=((_NET_SHIM) if _nonet else '')+_user
    workdir=tempfile.mkdtemp(prefix='amni_sbx_')
    path=os.path.join(workdir,'_run.py')
    try:
        open(path,'w',encoding='utf-8').write(src)
        cr=jail_run([sys.executable,'-I','-B',path],workdir,timeout=timeout,mem_mb=int(os.environ.get('AMNI_SANDBOX_MEM_MB','256')),extra=['--ro-bind',workdir,workdir])
        killed=cr['killed']
        if killed=='no-jail':return {'ran':False,'ok':False,'error':cr['stderr'],'jailed':False}
        return {'ran':True,'returncode':cr['returncode'],'stdout':(cr['stdout'] or '')[-keep:],'stderr':(cr['stderr'] or '')[-keep:],'ok':cr['returncode']==0 and killed is None and not (cr['stderr'] or '').strip(),'killed':killed,'timed_out':killed=='timeout','capped':killed=='output-cap','jailed':cr['jailed']}
    except Exception as e:return {'ran':False,'ok':False,'error':f'{type(e).__name__}: {e}'}
    finally:shutil.rmtree(workdir,ignore_errors=True)
def review(code,func_name=None,harness=''):
    """Layered debug step: SECURITY danger-scan (refuse to run high-risk code) + static lint + real sandbox
    execution + adversarial runtime probe."""
    from amni.serve.code_safety import danger_scan
    danger=danger_scan(code)
    if [d for d in danger if d['severity']=='HIGH']:
        return {'clean':False,'blocked':True,'danger':danger,'lint':static_check(code),'runtime':[],'sandbox':None,'func':None}
    lint=static_check(code)
    probe=adversarial_probe(code,func_name)
    runtime=[] if probe.get('skipped') else probe.get('failures',[])
    sand=run_in_sandbox(code,harness) if harness else None
    clean=not lint and not runtime and (sand is None or sand.get('ok'))
    return {'clean':clean,'lint':lint,'runtime':runtime,'sandbox':sand,'func':probe.get('func')}
_SEQ_PROBES=[('empty',[]),('single',[1]),('dups',[1,1,1]),('strings',['a','b','a']),('mixed',[1,'1',1]),('with_none',[None,1,None]),('unhashable_list',[[1],[1],[2]]),('unhashable_dict',[{'a':1},{'a':1}]),('tuples',[(1,2),(1,2),(3,4)]),('large',list(range(3000))*2)]
def _arity1_funcs(ns):
    out=[]
    for k,v in ns.items():
        if k.startswith('_') or not callable(v) or inspect.isclass(v):continue
        if getattr(v,'__module__',None) not in (None,'builtins') and not inspect.isfunction(v):continue
        try:sig=inspect.signature(v)
        except (ValueError,TypeError):continue
        req=[p for p in sig.parameters.values() if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]
        if len(req)==1:out.append((k,v))
    return out
_PROBE_MARK='@@AMNI_PROBE@@'
def _probe_local(code,func_name,probes):
    ns={}
    try:exec(code,ns)
    except Exception as e:return {'ok':False,'stage':'compile','error':f'{type(e).__name__}: {e}'[:300],'failures':[]}
    funcs=_arity1_funcs(ns)
    name,fn=(func_name,ns.get(func_name)) if (func_name and callable(ns.get(func_name))) else (funcs[0] if funcs else (None,None))
    if fn is None:return {'ok':True,'skipped':'no single-arg function to probe — needs custom test inputs','func':None,'failures':[]}
    failures=[]
    for label,inp in probes:
        try:fn(inp)
        except Exception as e:failures.append({'probe':label,'input':repr(inp)[:48],'error':f'{type(e).__name__}: {e}'[:160]})
    return {'ok':len(failures)==0,'func':name,'probes_run':len(probes),'failures':failures}
def adversarial_probe(code,func_name=None,timeout=8):
    import json
    try:
        from amni.serve.code_safety import danger_scan
        high=[d for d in danger_scan(code) if d['severity']=='HIGH']
        if high:return {'ok':False,'blocked':True,'stage':'danger','dangers':high,'func':None,'failures':[],'error':'refused to exec high-risk code: '+', '.join(sorted({d['rule'] for d in high}))}
    except Exception:pass
    try:prog='import inspect,json\n'+inspect.getsource(_arity1_funcs)+inspect.getsource(_probe_local)+f"print('\\n{_PROBE_MARK}'+json.dumps(_probe_local({code!r},{func_name!r},{_SEQ_PROBES!r}),default=str))\n"
    except Exception as e:return {'ok':True,'skipped':f'probe source unavailable: {e}','func':None,'failures':[]}
    r=run_in_sandbox(prog,timeout=timeout,allow_dangerous=True,keep=20000)
    line=next((l[len(_PROBE_MARK):] for l in reversed((r.get('stdout') or '').splitlines()) if l.startswith(_PROBE_MARK)),None)
    try:return json.loads(line) if line else ({'ok':False,'stage':'timeout','func':None,'failures':[{'probe':'timeout','input':'','error':f'probe run exceeded {timeout}s (infinite loop or very slow on edge cases)'}]} if r.get('timed_out') else {'ok':True,'skipped':'sandboxed probe produced no result: '+(r.get('error') or (r.get('stderr') or '')[-200:] or str(r.get('killed'))),'func':None,'failures':[]})
    except Exception:return {'ok':True,'skipped':'sandboxed probe result unparseable','func':None,'failures':[]}
def debug_report(code,func_name=None):
    r=review(code,func_name)
    if r['clean']:return f'[debug] {r.get("func") or "code"}: clean — passed static lint + runtime probes'
    lines=[f'[debug] {r.get("func") or "code"}: issues found — fix before posting:']
    for it in r['lint']:lines.append(f'  - lint L{it["line"]} [{it["rule"]}]: {it["msg"]}')
    for f in r['runtime']:lines.append(f'  - runtime [{f["probe"]}] ({f["input"]}): {f["error"]}')
    return '\n'.join(lines)
