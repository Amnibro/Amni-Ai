import os,re,hmac,logging
from urllib.parse import unquote,urlsplit
from amni.serve.rate_limit import bucket_from_env,client_key,is_local
LOCAL_ORIGIN_RE=r'(?i)^(?:https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?|tauri://localhost|https?://tauri\.localhost)$'
_LOCAL_ORIGIN=re.compile(LOCAL_ORIGIN_RE)
_SECRET_RE=re.compile(r'(?i)((?:[?&;]|\b)(?:token|access_token|amni_token|api_key|apikey|key|auth)=)[^&\s"\';]+|(\bbearer\s+)[A-Za-z0-9._~+/=-]+')
AUTH_OPEN={'/','/unified','/jarvis','/hud','/healthz','/manifest.webmanifest','/sw.js','/favicon.ico'}
AUTH_OPEN_PFX=('/assets/','/icons/')
RL_GROUPS=(('chat',('/chat','/ask','/api/chat','/api/generate','/v1/chat/completions','/v1/completions','/complete','/voice/chat','/vision/ask','/rawgen','/reflect','/agents/','/delve','/api/delve/send','/guardian/ask','/teach'),30,10),('tools',('/skills','/mcp','/update/','/install','/models/activate','/tasks','/review/','/guardian/dispatch','/memory/coding-run/','/memory/confirm','/memory/forget','/bridge/','/launch/','/admin/','/persona/import','/project'),60,20))
_on=lambda k,d='0':os.environ.get(k,d).strip().lower() in ('1','true','yes','on')
def scrub(text):
    return _SECRET_RE.sub(lambda m:(m.group(1) or m.group(2))+'[REDACTED]',str(text))
class _Scrub(logging.Filter):
    def filter(self,record):
        try:
            record.msg=scrub(record.msg) if isinstance(record.msg,str) else record.msg
            record.args=tuple(scrub(a) if isinstance(a,str) else a for a in record.args) if isinstance(record.args,tuple) else record.args
        except Exception:pass
        return True
def install_log_scrub(names=('uvicorn.access','uvicorn.error','uvicorn','fastapi','amni')):
    f=_Scrub();[logging.getLogger(n).addFilter(f) for n in names if not any(isinstance(x,_Scrub) for x in logging.getLogger(n).filters)];return f
def cors_origins():
    return [o.strip().rstrip('/') for o in (os.environ.get('AMNI_CORS_ORIGINS') or '').split(',') if o.strip()]
def origin_allowed(origin,host=''):
    o=(origin or '').strip().rstrip('/');ex=cors_origins()
    return bool(o) and ('*' in ex or bool(_LOCAL_ORIGIN.match(o)) or o in ex or (bool(host) and urlsplit(o).netloc.lower()==host.strip().lower()))
def extract_token(request):
    h=request.headers;t=h.get('x-amni-token','').strip();a=h.get('authorization','')
    t=t or (a[7:].strip() if a[:7].lower()=='bearer ' else '')
    t=t or unquote(request.cookies.get('amni_token','') or '').strip()
    q=request.query_params.get('token','')
    return (t or (q if _on('AMNI_ALLOW_QUERY_TOKEN') else ''),bool(q) and not t and not _on('AMNI_ALLOW_QUERY_TOKEN'))
def route_group(path,method):
    m=method.upper();hit=next((g for g in RL_GROUPS if any(path==p.rstrip('/') or path.startswith(p if p.endswith('/') else p+'/') for p in g[1])),None)
    return hit if hit and (m in ('POST','PUT','PATCH','DELETE') or (m=='GET' and path.endswith('/stream'))) else None
def install(app,cors=False,token='',open_paths=None,open_prefixes=None):
    from fastapi.responses import JSONResponse
    op=set(open_paths or AUTH_OPEN);opx=tuple(open_prefixes or AUTH_OPEN_PFX)
    buckets={g[0]:bucket_from_env(g[0],g[2],g[3]) for g in RL_GROUPS};fail=bucket_from_env('authfail',30,10)
    rl_on=os.environ.get('AMNI_RATE_LIMIT','1').strip().lower() not in ('0','false','no','off')
    exempt=lambda r:is_local(r) and not _on('AMNI_RATE_LIMIT_LOCAL')
    lim=lambda info:JSONResponse(status_code=429,content={'error':'rate_limited',**info},headers={'Retry-After':str(max(1,int(info.get('retry_after_s') or 1)))})
    if token:
        @app.middleware('http')
        async def _auth_gate(request,call_next):
            p=request.url.path
            if request.method=='OPTIONS' or p in op or any(p.startswith(x) for x in opx):return await call_next(request)
            t,qtried=extract_token(request)
            if t and hmac.compare_digest(t.encode(),token.encode()):return await call_next(request)
            ok,info=(True,{}) if exempt(request) else fail.allow(client_key(request))
            if not ok:return lim(info)
            return JSONResponse(status_code=401,content={'error':'unauthorized','auth_required':True,'hint':'query-string tokens are disabled (AMNI_ALLOW_QUERY_TOKEN=0); send Authorization: Bearer <token>' if qtried else 'Adam access token required (Authorization: Bearer header, X-Amni-Token header, or amni_token cookie).'})
    if rl_on:
        @app.middleware('http')
        async def _rate_gate(request,call_next):
            g=route_group(request.url.path,request.method)
            if g is None or exempt(request):return await call_next(request)
            ok,info=buckets[g[0]].allow(client_key(request))
            return await call_next(request) if ok else lim(info)
    if os.environ.get('AMNI_ORIGIN_GUARD','1').strip().lower() not in ('0','false','no','off'):
        @app.middleware('http')
        async def _origin_gate(request,call_next):
            o=request.headers.get('origin')
            bad=request.method.upper() not in ('GET','HEAD','OPTIONS') and o is not None and not origin_allowed(o,request.headers.get('host',''))
            return JSONResponse(status_code=403,content={'error':'cross_origin_blocked','origin':o,'hint':'add the origin to AMNI_CORS_ORIGINS'}) if bad else await call_next(request)
    ex=cors_origins();star='*' in ex
    if cors or ex:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=False,allow_methods=['*'],allow_headers=['*']) if star else app.add_middleware(CORSMiddleware,allow_origins=ex,allow_origin_regex=LOCAL_ORIGIN_RE,allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
    install_log_scrub()
    return {'auth':bool(token),'cors':('*' if star else 'localhost+'+str(len(ex))) if (cors or ex) else 'off','rate_limit':rl_on,'rate_limit_local':_on('AMNI_RATE_LIMIT_LOCAL'),'query_token':_on('AMNI_ALLOW_QUERY_TOKEN'),'origin_guard':os.environ.get('AMNI_ORIGIN_GUARD','1')}
def load_token():
    t=(os.environ.get('AMNI_AUTH_TOKEN') or '').strip();f=(os.environ.get('AMNI_AUTH_TOKEN_FILE') or '').strip()
    try:return t or (open(f,encoding='utf-8').read().strip() if f and os.path.exists(f) else '')
    except Exception:return ''
def resolve_roots(roots,unrestricted,token,workdir=None):
    extra=[r.strip() for r in (os.environ.get('AMNI_ROOTS') or '').split(os.pathsep) if r.strip()]
    ok=bool(unrestricted) and _on('AMNI_UNRESTRICTED_FILES') and bool(token)
    why='' if ok or not unrestricted else 'refused --unrestricted-files: needs AMNI_UNRESTRICTED_FILES=1 and AMNI_AUTH_TOKEN; staying on project dir + AMNI_ROOTS'
    base=list(roots or []);lead=[str(workdir or os.getcwd())] if extra and not base else []
    return (list(dict.fromkeys(lead+base+extra)),ok,why)
