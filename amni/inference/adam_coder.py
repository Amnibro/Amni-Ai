"""AdamCoder — make ADAM do the coding, macro-accelerated. Adam (the NVFP4 svc) generates code from a task (optionally scaffolded with verified macro blocks), then it's sanitized (strip the invisible-unicode class), VERIFIED (ast.parse + optional functional test), repaired on failure (error fed back into a retry), and BANKED as new macros (learning ratchet -> known-good + faster next time). Strict-additive: verification gates everything; only verified code is returned and banked."""
import ast,re
_INVIS={c:None for c in (0x200b,0x200c,0x200d,0xfeff,0x2060,0x180e)}
class AdamCoder:
    def __init__(s,svc,engine=None):
        s.svc=svc;s.engine=engine
    def _extract(s,text):
        m=re.search(r'```(?:python|py)?\s*\n?(.*?)```',text or '',re.S)
        return (m.group(1) if m else (text or '')).strip()
    def _sanitize(s,code):
        return code.translate(_INVIS).replace(' ',' ').replace('\t','    ')
    def _verify(s,code,test_fn):
        if not code:return False,'empty output'
        try:ast.parse(code)
        except SyntaxError as e:return False,f'SyntaxError line {e.lineno}: {e.msg}'
        if test_fn is None:return True,None
        try:
            ns={};exec(code,ns);test_fn(ns);return True,None
        except AssertionError as e:return False,f'functional test failed: {e}'
        except Exception as e:return False,f'{type(e).__name__}: {e}'
    def _verify_sandboxed(s,code,timeout=8):
        import subprocess,tempfile,sys,os
        if not code:return False,'empty output'
        try:ast.parse(code)
        except SyntaxError as e:return False,f'SyntaxError line {e.lineno}: {e.msg}'
        f=tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8');f.write(code+'\n_selftest()\nprint("__VERIFY_OK__")\n');f.close()
        try:
            r=subprocess.run([sys.executable,f.name],capture_output=True,text=True,timeout=timeout)
            if '__VERIFY_OK__' in (r.stdout or '') and r.returncode==0:return True,None
            tail=(r.stderr or '').strip().splitlines();return False,(tail[-1][:160] if tail else 'selftest did not pass')
        except subprocess.TimeoutExpired:return False,f'timed out >{timeout}s (possible infinite loop)'
        except Exception as e:return False,f'{type(e).__name__}: {e}'
        finally:
            try:os.remove(f.name)
            except OSError:pass
    def write(s,task,test_fn=None,scaffold='',max_tries=3,max_tokens=700,selftest=False):
        sysp='You are an expert Python engineer. Output ONLY one ```python code block that fully solves the task. No prose, no markdown outside the block, no inline comments.'
        prompt=(f'Verified building blocks you may reuse/adapt:\n```python\n{scaffold}\n```\n\n' if scaffold else '')+f'Write {task}'
        hist=None;last='';err='unknown'
        for t in range(max_tries):
            raw,_=s.svc.chat(prompt,system=sysp,max_new_tokens=max_tokens,history=hist)
            code=s._sanitize(s._extract(raw));last=code
            ok,err=(s._verify_sandboxed(code) if selftest else s._verify(code,test_fn))
            if ok:
                interned=s.engine.learn(code) if s.engine is not None else 0
                return {'verified':True,'attempts':t+1,'code':code,'banked_macros':interned}
            hist=[(prompt,raw)];prompt=f'That code FAILED: {err}\nReturn ONLY the corrected ```python code block.'
        return {'verified':False,'attempts':max_tries,'code':last,'error':err}
