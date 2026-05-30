"""code_safety — static danger scan for code Adam runs (generated OR crawled). AST-based so it survives
obfuscation a regex misses (e.g. getattr(__import__('os'),'system')). Flags filesystem mutation, shell,
network, dynamic exec, and sandbox-escape primitives. Used to REFUSE high-risk code before any subprocess run."""
import ast,re
_DANGER_MODULES={'subprocess','socket','ctypes','shutil','pty','smtplib','ftplib','telnetlib','pickle','marshal','requests','urllib','http','multiprocessing','mmap','fcntl','winreg','_winreg','xmlrpc','asyncio','curses','cffi','importlib','dill','shelve','code','aiohttp','httpx','paramiko'}
_POWERFUL_ROOTS=('os','sys','subprocess','builtins','importlib','ctypes','socket','shutil','pickle','marshal')
_DANGER_OS_CALLS={'system','popen','remove','unlink','rmdir','removedirs','rename','replace','kill','fork','forkpty','exec','execl','execle','execlp','execv','execve','execvp','execvpe','spawn','spawnl','spawnv','spawnve','startfile','chmod','chown','setuid','setgid','putenv','unsetenv','abort','_exit','open','write','dup','dup2','pipe','symlink','link','chdir','chroot','truncate','ftruncate','mkfifo','mknod','setegid','seteuid','setreuid','setpgid','umask'}
_DANGER_SHUTIL_CALLS={'rmtree','move','copy','copytree','make_archive','chown'}
_DANGER_BUILTINS={'eval','exec','compile','__import__','input','breakpoint','globals','vars'}
_DANGER_NET_ATTRS={'socket','create_connection','urlopen','Request','connect','sendall','recv'}
def _root_name(node):
    while isinstance(node,ast.Attribute):node=node.value
    return node.id if isinstance(node,ast.Name) else None
def danger_scan(code):
    """Returns list of {line, severity, rule, msg}. severity HIGH = refuse to run."""
    try:tree=ast.parse(code)
    except SyntaxError as e:return [{'line':e.lineno or 0,'severity':'HIGH','rule':'unparseable','msg':f'will not run unparseable code: {e.msg}'}]
    out=[];seen=set()
    def add(line,sev,rule,msg):
        k=(rule,line)
        if k not in seen:seen.add(k);out.append({'line':line,'severity':sev,'rule':rule,'msg':msg})
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            mods=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [(n.module or '').split('.')[0]]
            for m in mods:
                if m in _DANGER_MODULES:add(n.lineno,'HIGH','dangerous-import',f'imports "{m}" (shell/network/native/serialization)')
        elif isinstance(n,ast.Call):
            f=n.func
            if isinstance(f,ast.Name):
                if f.id in _DANGER_BUILTINS:add(n.lineno,'HIGH','dynamic-exec' if f.id in('eval','exec','compile','__import__') else 'risky-builtin',f'calls {f.id}()')
                if f.id=='open':
                    mode=''
                    if len(n.args)>=2 and isinstance(n.args[1],ast.Constant):mode=str(n.args[1].value)
                    for kw in n.keywords:
                        if kw.arg=='mode' and isinstance(kw.value,ast.Constant):mode=str(kw.value.value)
                    if any(c in mode for c in 'wax+'):add(n.lineno,'HIGH','file-write',f'open(..., {mode!r}) — file write/mutation')
                if f.id=='getattr' and len(n.args)>=2 and isinstance(n.args[1],ast.Constant) and str(n.args[1].value) in (_DANGER_OS_CALLS|_DANGER_BUILTINS):add(n.lineno,'HIGH','getattr-obfuscation',f'getattr(..., {n.args[1].value!r}) — dynamic access to a dangerous primitive')
                if f.id in ('getattr','setattr') and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id in _POWERFUL_ROOTS and (len(n.args)<2 or not isinstance(n.args[1],ast.Constant)):add(n.lineno,'HIGH','dynamic-getattr',f'{f.id}({n.args[0].id}, <dynamic>) — runtime attribute access on a powerful module')
            elif isinstance(f,ast.Attribute):
                root=_root_name(f);attr=f.attr
                if root=='os' and attr in _DANGER_OS_CALLS:add(n.lineno,'HIGH','os-call',f'os.{attr}() — shell/process/filesystem')
                elif root=='shutil' and attr in _DANGER_SHUTIL_CALLS:add(n.lineno,'HIGH','shutil-call',f'shutil.{attr}() — filesystem mutation')
                elif root=='importlib' and attr in ('import_module','__import__','reload'):add(n.lineno,'HIGH','dynamic-import',f'importlib.{attr}() — dynamic module import')
                elif root in ('subprocess','socket','ctypes','requests','urllib','pickle','marshal','dill','paramiko','httpx','aiohttp'):add(n.lineno,'HIGH','dangerous-call',f'{root}.{attr}() — shell/network/native/deserialize')
                elif attr in _DANGER_NET_ATTRS:add(n.lineno,'MED','possible-network',f'.{attr}() — possible network access')
        elif isinstance(n,ast.Attribute) and n.attr in ('__globals__','__builtins__','__subclasses__','__bases__','__mro__','__code__'):
            add(n.lineno,'HIGH','introspection-escape',f'access to {n.attr} (sandbox-escape primitive)')
        elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            for dec in n.decorator_list:
                d=dec.func if isinstance(dec,ast.Call) else dec
                if isinstance(d,ast.Name) and d.id in _DANGER_BUILTINS:add(n.lineno,'HIGH','dangerous-decorator',f'@{d.id} — dangerous decorator')
                elif isinstance(d,ast.Attribute) and ((_root_name(d)=='os' and d.attr in _DANGER_OS_CALLS) or _root_name(d) in ('subprocess','ctypes','socket')):add(n.lineno,'HIGH','dangerous-decorator',f'@{_root_name(d)}.{d.attr} — dangerous decorator')
    return out
def is_safe(code):
    d=danger_scan(code)
    high=[x for x in d if x['severity']=='HIGH']
    return (not high,d)
def run_capped(argv,timeout=8,max_output_bytes=200000,env=None,cwd=None,mem_mb=256):
    """Run a subprocess with resource caps so a memory/output bomb can't exhaust the host:
    - output cap: stream stdout/stderr, KILL the process tree once total bytes exceed max_output_bytes;
    - wall-clock timeout: kill on expiry;
    - POSIX rlimits (preexec): address-space (mem_mb), CPU seconds, file size, and RLIMIT_NPROC=0 (no forks).
    On Windows rlimits don't apply (timeout+output-cap still do; full mem isolation wants a Job Object/WSL)."""
    import subprocess,threading,os,signal
    is_posix=os.name=='posix'
    def _limits():
        import resource
        m=int(mem_mb)*1024*1024
        for res,val in ((getattr(resource,'RLIMIT_AS',None),m),(getattr(resource,'RLIMIT_DATA',None),m),(getattr(resource,'RLIMIT_CPU',None),int(timeout)+2),(getattr(resource,'RLIMIT_FSIZE',None),16*1024*1024)):
            if res is not None:
                try:resource.setrlimit(res,(val,val))
                except Exception:pass
        try:resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
        except Exception:pass
        try:os.setsid()
        except Exception:pass
    kw={'preexec_fn':_limits} if is_posix else {}
    p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,cwd=cwd,**kw)
    bufs={'out':bytearray(),'err':bytearray()};info={'killed':None}
    def _kill():
        try:
            if is_posix:
                try:os.killpg(os.getpgid(p.pid),signal.SIGKILL)
                except Exception:p.kill()
            else:subprocess.run(['taskkill','/F','/T','/PID',str(p.pid)],capture_output=True)
        except Exception:
            try:p.kill()
            except Exception:pass
    def _reader(stream,buf):
        try:
            while True:
                chunk=stream.read(8192)
                if not chunk:break
                buf.extend(chunk)
                if len(bufs['out'])+len(bufs['err'])>max_output_bytes and info['killed'] is None:
                    info['killed']='output-cap';_kill();break
        except Exception:pass
    t1=threading.Thread(target=_reader,args=(p.stdout,bufs['out']),daemon=True)
    t2=threading.Thread(target=_reader,args=(p.stderr,bufs['err']),daemon=True)
    t1.start();t2.start()
    try:p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if info['killed'] is None:info['killed']='timeout'
        _kill()
    t1.join(timeout=2);t2.join(timeout=2)
    return {'returncode':p.returncode,'stdout':bytes(bufs['out'][:max_output_bytes]).decode('utf-8','replace'),'stderr':bytes(bufs['err'][:max_output_bytes]).decode('utf-8','replace'),'killed':info['killed']}
_SCRIPT_TAGS=re.compile(r'<(script|iframe|style|object|embed|svg|math|template)\b[^>]*>.*?</\1>',re.IGNORECASE|re.DOTALL)
_ACTIVE_TAGS=re.compile(r'<(script|iframe|object|embed|link|meta|form|input|button)\b[^>]*/?>',re.IGNORECASE)
_INJECTION=re.compile(r'(?:ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|messages?)|disregard\s+(?:the\s+)?(?:above|previous|prior|earlier|system)|forget\s+(?:everything|all\s+previous|the\s+above|your\s+instructions)|you\s+are\s+now\s+(?:a|an|in|no\s+longer)|new\s+(?:instructions?|rules?|task)\s*:|system\s+prompt\s*:|</?(?:system|assistant|user|im_start|im_end)\b|\[/?INST\]|###\s*instruction|do\s+not\s+follow\s+(?:the\s+)?(?:above|previous|prior)|instead[,\s]+(?:output|say|respond|ignore|reveal|print)|reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions))',re.IGNORECASE)
def sanitize_ingest(text):
    """Strip active markup (<script>/<iframe>/…) and neutralize prompt-injection phrases from crawled content
    BEFORE it becomes a lesson, so a poisoned source can't smuggle exec payloads or hijack Adam's instructions."""
    if not text:return ('',[])
    flags=[];out=text
    if _SCRIPT_TAGS.search(out):flags.append('script_block');out=_SCRIPT_TAGS.sub(' ',out)
    if _ACTIVE_TAGS.search(out):flags.append('active_tag');out=_ACTIVE_TAGS.sub(' ',out)
    if _INJECTION.search(out):flags.append('prompt_injection');out=_INJECTION.sub('[neutralized]',out)
    if '\x00' in out:flags.append('null_byte');out=out.replace('\x00','')
    return (out,flags)
_TRUSTED_CODE_DOMAINS=('github.com','githubusercontent.com','gist.github.com','gitlab.com','bitbucket.org','docs.python.org','python.org','stackoverflow.com','stackexchange.com','superuser.com','serverfault.com','developer.mozilla.org','doc.rust-lang.org','rust-lang.org','pkg.go.dev','go.dev','docs.oracle.com','cppreference.com','en.cppreference.com','learn.microsoft.com','microsoft.com','readthedocs.io','readthedocs.org','realpython.com','geeksforgeeks.org','w3schools.com','huggingface.co','pypi.org','npmjs.com','crates.io','kotlinlang.org','swift.org','ruby-lang.org','php.net','scala-lang.org','haskell.org','elixir-lang.org','julialang.org','wikipedia.org','baeldung.com','digitalocean.com')
def is_trusted_source(url):
    try:
        from urllib.parse import urlparse
        host=(urlparse(url).hostname or '').lower()
        return any(host==d or host.endswith('.'+d) for d in _TRUSTED_CODE_DOMAINS)
    except Exception:return False
_SECRET_KEY_RE=re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|xox[bopa]-[A-Za-z0-9-]{20,}|AIza[\w-]{35}|AKIA[A-Z0-9]{16}|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,})\b')
_SECRET_HOMEDIR_RE=re.compile(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s]+|/(?:home|Users)/[^/\s]+',re.IGNORECASE)
def scrub_secrets(text):
    """Narrow, answer-safe scrub for user-facing streams: redacts ONLY API keys and host home-dir paths — neither is
    ever legitimate content of Adam's REASONING (unlike emails/code/paths the user may genuinely want). For the live
    `event: reasoning` channel, where a secret would be a leak from recalled context, not requested output."""
    if not text:return text
    return _SECRET_HOMEDIR_RE.sub('<HOMEDIR>',_SECRET_KEY_RE.sub('<APIKEY>',text))
def b64_within_limit(b64,max_decoded_bytes=None):
    """True if a base64 payload decodes to <= the upload cap WITHOUT decoding it first (len*3/4 ≈ decoded bytes),
    so an oversize image/audio upload is rejected before it can OOM the server. Default 15 MB, env AMNI_MAX_UPLOAD_BYTES."""
    import os
    mx=max_decoded_bytes if max_decoded_bytes is not None else int(os.environ.get('AMNI_MAX_UPLOAD_BYTES',str(15*1024*1024)))
    return (len(b64 or '')*3)//4<=mx
_FETCH_SCHEMES={'http','https'}
def _ssrf_resolve(url):
    """Core SSRF check that also returns the PINNED ip (the first resolved address) so a caller can connect to that
    exact ip and defeat DNS-rebinding. Returns (ok, reason, pinned_ip_or_None). pinned_ip is None when the env
    override is on (no resolution performed)."""
    import os,socket,ipaddress
    from urllib.parse import urlparse
    try:p=urlparse(url)
    except Exception:return (False,'unparseable url',None)
    if (p.scheme or '').lower() not in _FETCH_SCHEMES:return (False,f'scheme {p.scheme!r} not allowed (http/https only)',None)
    host=p.hostname
    if not host:return (False,'no host in url',None)
    if os.environ.get('AMNI_ALLOW_PRIVATE_FETCH','0')=='1':return (True,'',None)
    try:infos=socket.getaddrinfo(host,p.port or (443 if p.scheme.lower()=='https' else 80),proto=socket.IPPROTO_TCP)
    except Exception as e:return (False,f'dns resolve failed: {type(e).__name__}',None)
    pin=None
    for _fam,_t,_pr,_cn,sa in infos:
        try:ip=ipaddress.ip_address(sa[0])
        except Exception:return (False,f'unresolvable address for {host}',None)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:return (False,f'blocked internal/private address {ip} for host {host!r}',None)
        if pin is None:pin=str(ip)
    return (True,'',pin)
def ssrf_check(url):
    """SSRF guard: only http/https, and the host must NOT resolve to a private/loopback/link-local/reserved/
    metadata (169.254.169.254) address — so a poisoned crawl link can't pivot to internal services or read local
    files via file://. Scheme is always enforced; IP-range block is bypassable with AMNI_ALLOW_PRIVATE_FETCH=1."""
    ok,why,_ip=_ssrf_resolve(url);return (ok,why)
def safe_urlopen(url,timeout=8.0,max_bytes=2_000_000,headers=None,max_redirects=3):
    """SSRF + DNS-rebinding + redirect-safe fetch. Validates EVERY hop with `_ssrf_resolve`, then PINS the resolved
    ip for the socket (so a host that passes the check can't rebind to an internal ip before connect), while keeping
    SNI/cert validation via the real hostname. Redirects are followed MANUALLY and each Location is re-checked —
    `urllib.urlopen` auto-follows redirects unchecked, so a public URL could 302 to an internal one. Returns (bytes,content_type)."""
    import http.client,socket
    from urllib.parse import urlparse,urljoin
    cur=url
    for _hop in range(max_redirects+1):
        ok,why,ip=_ssrf_resolve(cur)
        if not ok:raise OSError(f'ssrf_blocked: {why}')
        p=urlparse(cur);https=(p.scheme or '').lower()=='https';port=p.port or (443 if https else 80)
        path=(p.path or '/')+(('?'+p.query) if p.query else '')
        h=dict(headers or {})
        base=http.client.HTTPSConnection if https else http.client.HTTPConnection
        conn=base(p.hostname,port,timeout=timeout)
        if ip is not None:
            def _connect(c=conn,_ip=ip,_https=https,_host=p.hostname,_port=port):
                s=socket.create_connection((_ip,_port),timeout)
                c.sock=c._context.wrap_socket(s,server_hostname=_host) if _https else s
            conn.connect=_connect
        try:
            conn.request('GET',path,headers=h)
            r=conn.getresponse()
            if r.status in (301,302,303,307,308):
                loc=r.getheader('Location');r.read()
                if not loc:raise OSError('redirect without Location')
                cur=urljoin(cur,loc);continue
            return (r.read(max_bytes),r.getheader('content-type','') or '')
        finally:conn.close()
    raise OSError('too many redirects')
_CODE_BLOCK=re.compile(r'```(\w+)?\s*\n(.*?)```',re.DOTALL)
def audit_lesson(q,a):
    """Self-check one stored lesson for pollution. Returns (clean, issues). Flags PII, prompt-injection /
    active markup, and DANGEROUS python code blocks (non-python blocks are skipped, not falsely flagged)."""
    issues=[]
    try:
        from amni.serve.conversation import detect_personal
        if detect_personal(q) or detect_personal(a):issues.append('pii')
    except Exception:pass
    _,flags=sanitize_ingest(f'{q}\n{a}')
    if 'prompt_injection' in flags:issues.append('prompt_injection')
    if 'script_block' in flags or 'active_tag' in flags:issues.append('active_markup')
    for m in _CODE_BLOCK.finditer(a or ''):
        lang=(m.group(1) or '').lower()
        if lang in ('','python','py','python3'):
            high=[d for d in danger_scan(m.group(2)) if d['severity']=='HIGH' and d['rule']!='unparseable']
            if high:issues.append('dangerous_code:'+','.join(sorted({d['rule'] for d in high})));break
    return (not issues,issues)
def audit_lessons(lessons,limit=None):
    """Scan (q,a) pairs for pollution. Returns a report + the polluted entries (with indices for quarantine)."""
    items=list(lessons) if limit is None else list(lessons)[:limit]
    polluted=[];clean=0
    for i,(q,a) in enumerate(items):
        ok,issues=audit_lesson(q,a)
        if ok:clean+=1
        else:polluted.append({'idx':i,'q':(q or '')[:80],'issues':issues})
    by_issue={}
    for p in polluted:
        for it in p['issues']:by_issue[it.split(':')[0]]=by_issue.get(it.split(':')[0],0)+1
    return {'total':len(items),'clean':clean,'polluted':len(polluted),'by_issue':by_issue,'detail':polluted[:50],'polluted_indices':[p['idx'] for p in polluted],'polluted_entries':[{'q':items[p['idx']][0],'a':items[p['idx']][1],'issues':p['issues']} for p in polluted]}
def quarantine_polluted(adam,dry_run=False,limit=20000,path='experiences/quarantine.jsonl'):
    """Audit the lesson store and REMOVE polluted lessons (PII/injection/markup/dangerous-code), archiving
    them to quarantine.jsonl. Works on flat or routed stores via their purge_indices()."""
    import json,time
    from pathlib import Path as _P
    sl=getattr(adam,'sem_lut',None)
    raw=getattr(sl,'_raw',[]) if sl is not None else []
    rep=audit_lessons(raw,limit=limit)
    entries=rep['polluted_entries'];idxs=rep['polluted_indices']
    if dry_run or not idxs:return {'dry_run':dry_run,'polluted':len(idxs),'by_issue':rep['by_issue'],'removed':0,'sample':entries[:5]}
    try:
        p=_P(path);p.parent.mkdir(parents=True,exist_ok=True)
        with open(p,'a',encoding='utf-8') as f:
            for e in entries:f.write(json.dumps({'ts':time.time(),'q':e['q'][:300],'a':e['a'][:600],'issues':e['issues']},ensure_ascii=False)+'\n')
    except Exception as _e:return {'error':f'quarantine write failed: {_e}','polluted':len(idxs)}
    removed=0
    try:
        removed=sl.purge_indices(idxs) if hasattr(sl,'purge_indices') else 0
        if hasattr(adam,'save_lessons'):adam.save_lessons()
    except Exception as _e:return {'error':f'purge failed: {_e}','polluted':len(idxs),'quarantined_to':str(path)}
    return {'polluted':len(idxs),'removed':removed,'by_issue':rep['by_issue'],'quarantined_to':str(path),'remaining':len(getattr(sl,'_raw',[]))}
def audit_log_publish(record,path='experiences/federation_audit.jsonl'):
    """Provenance log: append one JSON record per publish (counts, scrub stats, ts-less — caller stamps)."""
    try:
        import json
        from pathlib import Path as _P
        p=_P(path);p.parent.mkdir(parents=True,exist_ok=True)
        with open(p,'a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        return True
    except Exception:return False
def scrub_egress(obj):
    """Scrub secrets (API keys), emails/phones, and host home-dir/paths out of OUTGOING responses and error
    messages, recursively (dict/list/str). Reuses federated.scrub_pii so error tracebacks can't leak
    `C:\\Users\\<you>\\...` paths, keys, or PII back to a client."""
    try:from amni.serve.federated import scrub_pii
    except Exception:scrub_pii=None
    def _s(x):
        if isinstance(x,str):
            if scrub_pii is None:return x
            try:return scrub_pii(x)[0]
            except Exception:return x
        if isinstance(x,dict):return {k:_s(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return type(x)(_s(v) for v in x)
        return x
    return _s(obj)
def safety_report(code):
    d=danger_scan(code)
    if not d:return '[safety] no dangerous operations detected'
    lines=['[safety] flagged operations:']
    for x in d:lines.append(f'  {x["severity"]} L{x["line"]} [{x["rule"]}]: {x["msg"]}')
    return '\n'.join(lines)
