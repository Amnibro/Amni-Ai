#!/usr/bin/env python
"""End-to-end smoke test for the Adam HTTP server. Hits every endpoint,
reports PASS/FAIL with timings. Run after each restart or before commits.

Usage:
  python scripts/smoke_test.py                     # all checks
  python scripts/smoke_test.py --host 127.0.0.1    # custom host
  python scripts/smoke_test.py --skip slow         # skip slow inference checks
  python scripts/smoke_test.py --json              # machine-readable

Exit code: 0 if all critical checks pass, 1 otherwise.
"""
import sys,os,argparse,json,time,pathlib
try:sys.stdout.reconfigure(encoding='utf-8')
except Exception:pass
try:import requests
except ImportError:print('pip install requests',file=sys.stderr);sys.exit(2)
BASE=None
def http_get(path,timeout=5):
    r=requests.get(f'{BASE}{path}',timeout=timeout)
    return r.status_code==200,(r.json() if r.headers.get('content-type','').startswith('application/json') else r.text[:200])
def http_post(path,body,timeout=15):
    r=requests.post(f'{BASE}{path}',json=body,timeout=timeout)
    return r.status_code==200,(r.json() if r.headers.get('content-type','').startswith('application/json') else r.text[:200])
def check_auto_import():
    pathlib.Path('smoke_tmp.py').write_text('def f(x):\n    return Counter(x)\n')
    return http_post('/skills/auto_import',{'args':{'path':'smoke_tmp.py'}})
def check_code_diff_dry():
    pathlib.Path('smoke_tmp.py').write_text('def f(x):\n    return x\n')
    return http_post('/skills/code_diff',{'args':{'path':'smoke_tmp.py','diff':'@@ -1,1 +1,1 @@\n-def f(x):\n+def g(x):\n','dry_run':True}})
def wait_for_warmup(max_wait_s=180):
    t0=time.time()
    while time.time()-t0<max_wait_s:
        try:
            r=requests.get(f'{BASE}/warmup',timeout=3).json()
            if r.get('warmup',{}).get('done'):return True
        except Exception:pass
        time.sleep(2)
    return False
def main():
    global BASE
    ap=argparse.ArgumentParser()
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=11434)
    ap.add_argument('--skip',default='',help='comma-separated tags to skip (e.g. slow,voice)')
    ap.add_argument('--json',action='store_true')
    ap.add_argument('--no-wait-warmup',action='store_true',help='skip waiting for /warmup done before inference checks')
    args=ap.parse_args()
    BASE=f'http://{args.host}:{args.port}'
    skip=set(s.strip() for s in args.skip.split(',') if s)
    if 'slow' not in skip and not args.no_wait_warmup:
        if not args.json:print('Waiting for /warmup done before inference checks (up to 180s)...',end='',flush=True)
        ok=wait_for_warmup()
        if not args.json:print(' READY' if ok else ' TIMEOUT (slow checks may fail)')
    checks=[
        ('healthz',lambda:http_get('/healthz'),True,False),
        ('health (full)',lambda:http_get('/health'),True,False),
        ('warmup status',lambda:http_get('/warmup'),False,False),
        ('skills list',lambda:http_get('/skills'),True,False),
        ('sessions list',lambda:http_get('/sessions'),True,False),
        ('personas list',lambda:http_get('/personas'),False,False),
        ('skill: calc',lambda:http_post('/skills/calc',{'args':{'expr':'17*23'}}),True,False),
        ('skill: time',lambda:http_post('/skills/time',{'args':{}}),True,False),
        ('skill: project_info',lambda:http_post('/skills/project_info',{'args':{}}),True,False),
        ('skill: git status',lambda:http_post('/skills/git',{'args':{'cmd':'rev-parse --abbrev-ref HEAD'}}),False,False),
        ('skill: symbols (agent.py)',lambda:http_post('/skills/symbols',{'args':{'path':'amni/serve/agent.py'}}),True,False),
        ('skill: parse_error',lambda:http_post('/skills/parse_error',{'args':{'text':'IndexError: list index out of range'}}),True,False),
        ('skill: auto_import',check_auto_import,True,False),
        ('skill: code_diff dry-run',check_code_diff_dry,True,False),
        ('skill: tts list_voices',lambda:http_post('/skills/tts',{'args':{'text':'x','list_voices':True}}),False,False),
        ('inference: /complete short',lambda:http_post('/complete',{'prefix':'def add(a,b):\n    return ','max_tokens':5},timeout=60),True,True),
        ('inference: /profile (1 run)',lambda:http_post('/profile/inference',{'prompt':'hi','max_new_tokens':5,'runs':1},timeout=60),True,True),
    ]
    if not args.json:print(f'Adam smoke test — {BASE}\n')
    results=[]
    for name,fn,crit,slow in checks:
        if slow and 'slow' in skip:
            results.append({'name':name,'ok':None,'detail':'skipped','critical':crit,'wall_s':0});continue
        if 'voice' in skip and 'tts' in name:
            results.append({'name':name,'ok':None,'detail':'skipped','critical':crit,'wall_s':0});continue
        t0=time.time()
        try:ok,detail=fn();err=None
        except Exception as e:ok,detail,err=False,f'{type(e).__name__}: {e}',str(e)
        dt=round(time.time()-t0,2)
        results.append({'name':name,'ok':bool(ok),'detail':detail,'wall_s':dt,'critical':crit})
        if not args.json:
            sym='[PASS]' if ok else '[FAIL]' if crit else '[WARN]'
            line=f'{sym:<7} {name:<32} ({dt}s)'
            if not ok:line+=f'  — {str(detail)[:80]}'
            print(line)
    try:pathlib.Path('smoke_tmp.py').unlink(missing_ok=True)
    except Exception:pass
    crit_fail=[r for r in results if r['critical'] and r['ok'] is False]
    p=sum(1 for r in results if r['ok'] is True);f=sum(1 for r in results if r['ok'] is False);s=sum(1 for r in results if r['ok'] is None)
    summary={'pass':p,'fail':f,'skipped':s,'critical_failures':len(crit_fail),'results':results}
    if args.json:print(json.dumps(summary,indent=2))
    else:print(f'\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n  {p} passed, {f} failed, {s} skipped — {len(crit_fail)} critical fails\n━━━━━━━━━━━━━━━━━━━━━━━━━━')
    sys.exit(0 if len(crit_fail)==0 else 1)
if __name__=='__main__':main()
