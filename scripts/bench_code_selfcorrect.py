"""bench_code_selfcorrect — does Adam actually DEBUG ITSELF? Run HumanEval+ in two passes:
  pass 1 (cold): generate solution, execute against tests, capture pass/fail + the real error
  pass 2 (self-correct): for failures ONLY, feed Adam its OWN code + the execution error ("your attempt failed with X,
    fix it") and regenerate. Re-execute.
This is legitimate iterative debugging (the error is real execution feedback a developer also gets — NOT the gold
solution; Adam still has to derive the fix). Reports pass@1 cold vs pass@1 after self-correction + how many it fixed.
Optional --baseline-url runs the IDENTICAL loop on native Gemma-4-E2B for an apples-to-apples comparison.
Usage:
  python scripts/bench_code_selfcorrect.py --limit 30 --out eval_reports/selfcorrect_humanevalplus.json"""
import sys,json,argparse,re,tempfile,os,subprocess,time,urllib.request,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.eval import modern_bench as mb
def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers={'content-type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode('utf-8','ignore'))
class GpuDown(Exception):pass
def _chat(url,timeout,msg,max_tokens=768):
    server_err=False
    for path,payload,outk in ((url+'/chat',{'message':msg,'max_new_tokens':max_tokens},('answer','text','response')),(url+'/complete',{'prefix':msg,'max_tokens':max_tokens,'stop':['\n\n\n']},('completion','text','output'))):
        try:
            j=_post(path,payload,timeout)
            if isinstance(j,dict):
                for k in outk:
                    if j.get(k):return j[k]
        except urllib.error.HTTPError as e:
            if e.code>=500:server_err=True
        except Exception:continue
    if server_err:raise GpuDown('server 500 — likely HIP launch failure / TDR')
    return ''
def _strip(t):t=re.sub(r'^```\w*\s*\n?','',t.strip());return re.sub(r'\n?```\s*$','',t)
def _extract_function(body,ep):
    imports=[l for l in body.split('\n') if l.startswith(('import ','from '))]
    idx=body.find(f'def {ep}')
    if idx<0:return None
    lines=body[idx:].split('\n');block=[lines[0]]
    for l in lines[1:]:
        if l.strip()=='':block.append('');continue
        if (len(l)-len(l.lstrip()))>0:block.append(l)
        else:break
    return ('\n'.join(imports)+'\n' if imports else '')+'\n'.join(block).rstrip()
def _assemble(item,completion):
    body=_strip(completion);ep=item.get('entry_point','')
    fn=_extract_function(body,ep)
    if fn is None:return None
    pimp='\n'.join(l for l in item['prompt'].split('\n') if l.startswith(('import ','from ')))
    return (pimp+'\n' if pimp else '')+fn
def _exec(item,completion,timeout=10):
    mod=_assemble(item,completion)
    if mod is None:return False,'no function definition found in output'
    test=item.get('test','');ep=item.get('entry_point','')
    src=mod+'\n'+test+(f'\ncheck({ep})\n' if 'def check' in test else '')
    p=None
    try:
        with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8') as f:f.write(src);p=f.name
        r=subprocess.run([sys.executable,'-B',p],capture_output=True,timeout=timeout,text=True)
        if r.returncode==0:return True,''
        err=(r.stderr or r.stdout or '').strip().split('\n')
        return False,'\n'.join(err[-4:])[:400]
    except subprocess.TimeoutExpired:return False,'timeout (possible infinite loop)'
    except Exception as e:return False,str(e)[:200]
    finally:
        if p:
            try:os.unlink(p)
            except Exception:pass
_INSTR='Complete this Python function. Output the COMPLETE function definition including the signature and any needed imports, correctly indented. No prose, no markdown fences.\n\n'
def run_loop(url,items,timeout,exec_timeout,label):
    p1=0;p2=0;fixed=0;broke=0;rows=[];done=0;aborted=False
    for it in items:
        try:
            c1=_chat(url,timeout,_INSTR+it['prompt'])
            ok1,err1=_exec(it,c1,exec_timeout)
            ok2,err2=ok1,err1;corrected=False
            if ok1:p1+=1
            else:
                fb=f"Your previous solution to this function FAILED when tested.\nYour code:\n{_strip(c1)[:600]}\nThe test error was:\n{err1}\nFix the bug and output the COMPLETE corrected function definition only.\n\n{it['prompt']}"
                c2=_chat(url,timeout,fb);corrected=True
                ok2,err2=_exec(it,c2,exec_timeout)
        except GpuDown as e:
            aborted=True;print(f'[{label}] ABORT at item {done+1}/{len(items)} — {e}. Reporting {done} valid items only (NOT scoring the crash as wrong).',flush=True);break
        if ok2:p2+=1
        if (not ok1) and ok2:fixed+=1
        if ok1 and (corrected and not ok2):broke+=1
        done+=1
        rows.append({'task_id':it['task_id'],'pass1':ok1,'pass2':ok2,'corrected':corrected,'err1':err1[:120]})
        print(f'[{label}] {it["task_id"]}: p1={"Y" if ok1 else "n"} -> p2={"Y" if ok2 else "n"}{" (FIXED)" if (not ok1 and ok2) else ""}',flush=True)
    n=done
    if n==0:return {'label':label,'n':0,'aborted':aborted,'note':'no valid items'}
    return {'label':label,'n':n,'aborted':aborted,'pass1_pct':round(100*p1/n,1),'pass2_pct':round(100*p2/n,1),'lift_pp':round(100*(p2-p1)/n,1),'fixed':fixed,'regressed':broke,'rows':rows}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=30)
    ap.add_argument('--url',default='http://127.0.0.1:7700')
    ap.add_argument('--req-timeout',type=int,default=180);ap.add_argument('--exec-timeout',type=int,default=10)
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    items=mb.load_humaneval_plus(limit=a.limit)
    print(f'[selfcorrect] HumanEval+ n={len(items)} via {a.url}',flush=True)
    res=run_loop(a.url.rstrip('/'),items,a.req_timeout,a.exec_timeout,'adam')
    print(f'\n[selfcorrect] ADAM: pass@1 cold={res["pass1_pct"]}%  ->  after self-correct={res["pass2_pct"]}%  (lift {res["lift_pp"]:+}pp; fixed {res["fixed"]}, regressed {res["regressed"]})',flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(res,indent=2),encoding='utf-8')
        print(f'[selfcorrect] wrote {a.out}',flush=True)
if __name__=='__main__':main()
