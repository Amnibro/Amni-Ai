"""Code review helper — a fast static dangerous-pattern scan (pure CPU, no model) plus a multi-dimension LLM review via an injected chat_fn (Adam). static_scan flags injection/secret/unsafe-deser/path-traversal patterns by line; review_code adds per-dimension findings. Reused by the agent + the lowest-level security checks; mounts /review/code and /review/scan."""
import re
from typing import Callable,Sequence,List,Dict,Any
_DANGER=[(r'\beval\s*\(','eval() — arbitrary code execution'),(r'\bexec\s*\(','exec() — arbitrary code execution'),(r'subprocess.*shell\s*=\s*True','shell=True — command injection risk'),(r'os\.system\s*\(','os.system — command injection risk'),(r'pickle\.loads?\s*\(','pickle load — untrusted deserialization'),(r'yaml\.load\s*\((?!.*(?:Safe|Loader))','yaml.load without SafeLoader'),(r'verify\s*=\s*False','TLS verification disabled'),(r'(?i)(?:password|passwd|secret)\s*=\s*[\'"][^\'"]{3,}[\'"]','hardcoded secret'),(r'(?i)api[_-]?key\s*=\s*[\'"][^\'"]{6,}[\'"]','hardcoded API key'),(r'\.\./\.\.','path traversal'),(r'rm\s+-rf\s','rm -rf'),(r'(?i)\bmd5\b|\bsha1\b','weak hash (md5/sha1)'),(r'requests\.\w+\([^)]*timeout\s*=\s*None','no request timeout')]
def static_scan(code:str)->List[Dict[str,Any]]:
    out=[]
    for i,ln in enumerate((code or '').splitlines(),1):
        for pat,msg in _DANGER:
            if re.search(pat,ln):out.append({'line':i,'severity':'high','issue':msg,'code':ln.strip()[:120]})
    return out
def review_code(chat_fn:Callable,code:str,dims:Sequence[str]=('correctness','edge cases','security','performance'))->Dict[str,Any]:
    flags=static_scan(code)
    findings=[{'dim':d,'review':chat_fn(f'Review the code ONLY for {d}. Each finding as "- <line/where>: <issue> -> <fix>". If clean, reply exactly "none".\n\n```\n{(code or "")[:6000]}\n```',system='You are a precise senior code reviewer. Terse, specific, actionable; no praise, only findings.')} for d in dims]
    return {'static_flags':flags,'static_high':len(flags),'dimensions':list(dims),'findings':findings}
def mount(app,agent=None):
    from fastapi import Request
    def _cf():
        from amni.serve.guardian_service import _model
        return (lambda p,system=None:_model(agent,p,system=system)) if agent is not None else (lambda p,system=None:'(no agent connected)')
    @app.post('/review/code')
    async def _rev(req:Request):
        b=await req.json();return review_code(_cf(),b.get('code',''),tuple(b.get('dims') or ('correctness','edge cases','security','performance')))
    @app.post('/review/scan')
    async def _scan(req:Request):
        b=await req.json();return {'flags':static_scan(b.get('code',''))}
    return app
