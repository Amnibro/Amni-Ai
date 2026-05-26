"""test_runner — auto-spawn pytest on the suggested test file when Adam edits a .py module.
Timeout-bounded, captures pass/fail counts + failure messages. Synchronous run with a hard cap so
the edit response stays responsive. Returns a dict the widget can render directly."""
import subprocess,sys,re,os,time
from pathlib import Path
from typing import Dict,Any,List,Optional
_PYTEST_TIMEOUT_S=20
_COUNT_RE=re.compile(r'(\d+)\s+(passed|failed|skipped|error|errors|warning|warnings|deselected)\b',re.IGNORECASE)
_DURATION_RE=re.compile(r'\bin\s+([\d.]+)\s*s\b',re.IGNORECASE)
_FAIL_RE=re.compile(r'^(FAILED|ERROR)\s+(\S+::\S+)(?:\s+-\s+(.*))?$',re.MULTILINE)
def _parse_summary(text:str):
    counts={'passed':0,'failed':0,'skipped':0,'errors':0}
    last=None
    for m in _COUNT_RE.finditer(text):
        n=int(m.group(1));tag=m.group(2).lower().rstrip('s')
        if tag in ('error','errors'):tag='errors'
        elif tag=='warning':continue
        elif tag=='deselected':continue
        else:tag=tag+'s' if tag in ('error',) else tag
        if tag=='errors':counts['errors']=n
        elif tag in ('passed','failed','skipped'):counts[tag]=n
        last=m
    dm=_DURATION_RE.search(text);duration=float(dm.group(1)) if dm else None
    summary_line=text.strip().splitlines()[-1] if text.strip() else ''
    return counts,duration,summary_line
def _python_exe()->str:
    venv=Path(__file__).resolve().parents[2]/'.venv'/'Scripts'/'python.exe'
    return str(venv) if venv.exists() else sys.executable
def run_pytest(test_path:str,timeout_s:int=_PYTEST_TIMEOUT_S)->Dict[str,Any]:
    p=Path(test_path)
    if not p.exists():return {'ran':False,'reason':'test file not found','path':test_path}
    py=_python_exe();started=time.time()
    try:
        r=subprocess.run([py,'-m','pytest',str(p),'-x','--tb=short','-q','--no-header','--color=no'],capture_output=True,text=True,timeout=timeout_s,cwd=str(p.parent.parent) if p.parent.name=='tests' else str(p.parent),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        elapsed=round(time.time()-started,2)
        out=(r.stdout or '')+'\n'+(r.stderr or '')
        counts,_pdur,summary_line=_parse_summary(out)
        passed=counts['passed'];failed=counts['failed'];skipped=counts['skipped'];errors=counts['errors']
        failures=[]
        for fm in _FAIL_RE.finditer(out):
            failures.append({'kind':fm.group(1),'test':fm.group(2),'msg':(fm.group(3) or '').strip()[:200]})
        ok=(r.returncode==0) and failed==0 and errors==0
        return {'ran':True,'ok':ok,'returncode':r.returncode,'passed':passed,'failed':failed,'skipped':skipped,'errors':errors,'failures':failures[:5],'duration_s':elapsed,'path':test_path,'summary_line':summary_line,'stdout_tail':out[-1200:] if not ok else ''}
    except subprocess.TimeoutExpired:return {'ran':True,'ok':False,'timeout':True,'duration_s':round(time.time()-started,2),'path':test_path,'reason':f'pytest exceeded {timeout_s}s cap'}
    except FileNotFoundError as e:return {'ran':False,'reason':f'pytest unavailable: {e}','path':test_path}
    except Exception as e:return {'ran':False,'reason':f'pytest spawn failed: {type(e).__name__}: {e}','path':test_path}
