"""SkillRegistry — extensible tool layer for AmniAgent. Each skill: (name, fn, gate, desc, schema).
Built-ins: time, calc, mem, web, file_read, file_write, code_edit, shell, scan.
Asimov gating: every call passes through `_gate(args, ctx)` returning rejection reason or None.
calc / web / mem are thin aliases over Adam's existing tiers — file_*/shell/code_edit/scan are I/O primitives.
v6.1.0: file gates use a roots LIST instead of single workdir. `unrestricted=True` adds drive roots so Adam reaches any file."""
import os,re,ast,json,time,string,subprocess,shlex
from pathlib import Path
from dataclasses import dataclass,asdict,field
from typing import Callable,Dict,Any,Optional,List
_SHELL_ALLOW={'ls','dir','cat','type','pwd','cd','git','python','pip','where','which','echo','head','tail','wc','find'}
_SHELL_BLOCK_ARGS={'rm','del','rmdir','rd','format','mkfs','dd','shutdown','reboot','kill','taskkill','--force','-rf','-f'}
_TEXT_EXT={'.txt','.md','.markdown','.rst','.py','.js','.ts','.tsx','.jsx','.html','.htm','.css','.json','.yaml','.yml','.toml','.ini','.cfg','.csv','.tsv','.log','.sh','.ps1','.bat','.c','.h','.cpp','.hpp','.cs','.go','.rs','.java','.kt','.swift','.rb','.php','.lua','.r','.sql','.tex','.bib','.xml','.svg','.gitignore','.dockerignore','.env','.example'}
def _enumerate_drives():
    return [Path(f'{d}:/') for d in string.ascii_uppercase if Path(f'{d}:/').exists()] if os.name=='nt' else [Path('/')]
@dataclass
class SkillResult:
    ok:bool
    output:Any=None
    error:Optional[str]=None
    skill:str=''
    elapsed_ms:int=0
    def to_dict(self)->Dict[str,Any]:return asdict(self)
@dataclass
class _Skill:
    name:str
    fn:Callable
    gate:Optional[Callable]
    desc:str
    schema:Dict[str,Any]=field(default_factory=dict)
class SkillRegistry:
    def __init__(self,workdir:Optional[str]=None,roots:Optional[List[str]]=None,audit_log:Optional[str]=None,unrestricted:bool=False):
        self._skills:Dict[str,_Skill]={}
        rs:List[Path]=[]
        if roots:rs.extend(Path(r).resolve() for r in roots)
        if workdir:rs.append(Path(workdir).resolve())
        if not rs and not unrestricted:rs.append(Path(os.getcwd()).resolve())
        if unrestricted:rs.extend(_enumerate_drives())
        seen=set();self.roots=[];_=[(self.roots.append(p),seen.add(str(p))) for p in rs if str(p) not in seen]
        self.workdir=self.roots[0] if self.roots else Path(os.getcwd()).resolve()
        self.unrestricted=unrestricted
        self.audit_log=Path(audit_log) if audit_log else None
        if self.audit_log:self.audit_log.parent.mkdir(parents=True,exist_ok=True)
    def register(self,name:str,fn:Callable,gate:Optional[Callable]=None,desc:str='',schema:Optional[Dict]=None):
        self._skills[name]=_Skill(name=name,fn=fn,gate=gate,desc=desc,schema=schema or {})
    def list_skills(self)->List[Dict[str,Any]]:
        return [{'name':s.name,'desc':s.desc,'schema':s.schema} for s in self._skills.values()]
    def has(self,name:str)->bool:return name in self._skills
    def call(self,name:str,args:Dict[str,Any],ctx:Optional[Dict]=None)->SkillResult:
        t0=time.time()
        ctx=ctx or {}
        if name not in self._skills:
            r=SkillResult(ok=False,error=f'unknown skill: {name}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
            self._audit(name,args,r);return r
        s=self._skills[name]
        if s.gate is not None:
            reason=s.gate(args,ctx,self)
            if reason:
                r=SkillResult(ok=False,error=f'gated: {reason}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
                self._audit(name,args,r);return r
        try:out=s.fn(args,ctx,self);r=SkillResult(ok=True,output=out,skill=name,elapsed_ms=int((time.time()-t0)*1000))
        except Exception as e:r=SkillResult(ok=False,error=f'{type(e).__name__}: {e}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
        self._audit(name,args,r);return r
    def _audit(self,name:str,args:Dict,r:SkillResult):
        if not self.audit_log:return
        rec={'ts':time.time(),'skill':name,'args':{k:str(v)[:500] for k,v in args.items()},'ok':r.ok,'error':r.error,'elapsed_ms':r.elapsed_ms}
        try:
            with open(self.audit_log,'a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        except Exception:pass
    def _in_workdir(self,p:str)->bool:return self._in_allowed_roots(p)
    def _in_allowed_roots(self,p:str)->bool:
        try:rp=Path(p).resolve();return any(str(rp).startswith(str(r)) for r in self.roots)
        except Exception:return False
def _gate_path(args,ctx,reg:'SkillRegistry')->Optional[str]:
    p=args.get('path')
    if not p:return 'missing path arg'
    if not reg._in_allowed_roots(p):return f'path outside allowed roots ({len(reg.roots)} configured): {p}'
    return None
def _gate_shell(args,ctx,reg:'SkillRegistry')->Optional[str]:
    cmd=args.get('cmd','')
    if not cmd:return 'missing cmd arg'
    try:parts=shlex.split(cmd,posix=False)
    except Exception:return 'unparseable cmd'
    if not parts:return 'empty cmd'
    base=parts[0].lower().split('\\')[-1].split('/')[-1].removesuffix('.exe')
    if base not in _SHELL_ALLOW:return f'command not in allowlist: {base}'
    for p in parts[1:]:
        if p.lower() in _SHELL_BLOCK_ARGS:return f'blocked arg: {p}'
        if '..' in p or p.startswith('/') or (len(p)>1 and p[1]==':'):
            if not reg._in_allowed_roots(p):return f'arg path outside allowed roots: {p}'
    return None
def _gate_code_edit(args,ctx,reg:'SkillRegistry')->Optional[str]:
    g=_gate_path(args,ctx,reg)
    if g:return g
    if not args.get('find') or args.get('replace') is None:return 'missing find/replace'
    return None
def _skill_time(args,ctx,reg):return {'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'epoch':int(time.time())}
_WORD_OPS={'plus':'+','minus':'-','times':'*','x':'*','×':'*','over':'/','divided by':'/','to the power of':'**','to the power':'**'}
def _try_python_eval(expr:str)->Optional[float]:
    e=expr.strip().lower()
    for w,sym in _WORD_OPS.items():e=e.replace(w,sym)
    e=re.sub(r'[^0-9+\-*/().\s%]','',e)
    if not e.strip() or not re.search(r'[\d]',e):return None
    if not re.search(r'[+\-*/%]',e):return None
    try:v=eval(e,{'__builtins__':{}},{});return float(v) if isinstance(v,(int,float)) else None
    except Exception:return None
def _skill_calc(args,ctx,reg):
    adam=ctx.get('adam')
    expr=args.get('expr') or args.get('query','')
    if not expr:return {'error':'missing expr'}
    fast=_try_python_eval(expr)
    if fast is not None:
        out=int(fast) if fast==int(fast) else round(fast,8)
        return {'value':str(out),'tier':'fast_eval','tokens':0}
    if adam is None:return {'error':'symbolic expr requires Adam: '+expr}
    r=adam.ask(f'Compute: {expr}',writeback=False)
    return {'value':r.get('answer'),'tier':r.get('tier'),'tokens':r.get('tokens')}
def _skill_mem(args,ctx,reg):
    adam=ctx.get('adam')
    q=args.get('query','')
    k=int(args.get('k',3))
    if adam is None or not q:return {'error':'missing adam or query'}
    sl=getattr(adam,'sem_lut',None)
    n=len(sl._raw) if sl is not None else 0
    if n==0:return {'hits':[],'lessons_n':0}
    hits=[]
    try:
        eff_margin=sl.auto_margin()
        soft=sl.lookup_soft(q,margin=eff_margin)
        if soft:hits.append({'q':'(soft-lookup)','a':soft,'score':None,'method':'soft_margin'})
    except Exception as e:hits.append({'error':f'soft-lookup: {e}'})
    try:
        if hasattr(sl,'_ensure_encoder') and hasattr(sl,'_raw') and sl._raw:
            import numpy as np
            enc=sl._ensure_encoder()
            qv=enc([q])[0].astype('float32')
            kv=getattr(sl,'_stored_embs',None)
            if kv is None or len(kv)!=len(sl._raw):kv=enc([r[0] for r in sl._raw]).astype('float32')
            scores=kv@qv
            order=np.argsort(-scores)[:k]
            for i in order:
                hits.append({'q':sl._raw[int(i)][0][:200],'a':sl._raw[int(i)][1][:400],'score':float(scores[int(i)]),'method':'flat_cosine'})
    except Exception as e:hits.append({'error':f'flat-cosine: {e}'})
    return {'hits':hits,'lessons_n':n}
def _skill_web(args,ctx,reg):
    adam=ctx.get('adam')
    q=args.get('query','')
    if not q:return {'error':'missing query'}
    if adam is None or not hasattr(adam,'adam') or adam.adam.crawler is None:return {'error':'web crawler not available'}
    try:
        ans,sources,n=adam.adam.crawler.crawl_and_distill(q,subject=None,letter_only=False)
        return {'answer':ans,'sources':sources[:5],'tokens':n}
    except Exception as e:return {'error':str(e)}
def _skill_file_read(args,ctx,reg):
    p=args['path'];max_bytes=int(args.get('max_bytes',65536))
    data=Path(p).read_text(encoding='utf-8',errors='replace')[:max_bytes]
    return {'path':p,'content':data,'bytes':len(data)}
def _skill_file_write(args,ctx,reg):
    p=Path(args['path']);content=args.get('content','')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
    return {'path':str(p),'bytes_written':len(content)}
def _skill_code_edit(args,ctx,reg):
    p=Path(args['path']);find=args['find'];replace=args['replace'];count=int(args.get('count',1))
    src=p.read_text(encoding='utf-8')
    if find not in src:return {'error':'find string not present','path':str(p)}
    new=src.replace(find,replace,count) if count>0 else src.replace(find,replace)
    if p.suffix=='.py':
        try:ast.parse(new)
        except SyntaxError as e:return {'error':f'syntax error after edit: {e}','path':str(p)}
    p.write_text(new,encoding='utf-8')
    return {'path':str(p),'replacements':src.count(find) if count==0 else min(count,src.count(find))}
def _skill_shell(args,ctx,reg):
    cmd=args['cmd'];timeout=int(args.get('timeout',15))
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
    return {'cmd':cmd,'returncode':r.returncode,'stdout':r.stdout[:8000],'stderr':r.stderr[:4000]}
_DANGEROUS_PYTHON=re.compile(r'\b(?:os\.system|subprocess\.|os\.remove|os\.unlink|os\.rmdir|shutil\.rmtree|__import__\([\'"]os|exec\s*\(|eval\s*\(|open\s*\([^)]*[\'"]w|requests\.|urllib\.|socket\.|os\.environ\[|os\.setuid|os\.fork)\b')
def _skill_run_python(args,ctx,reg):
    code=args.get('code') or args.get('expr','')
    timeout=int(args.get('timeout',8))
    if not code:return {'error':'no code provided'}
    if _DANGEROUS_PYTHON.search(code):return {'error':'rejected: code contains potentially dangerous operations (filesystem mutation, network, exec/eval, subprocess)'}
    import tempfile,sys as _sys
    f=tempfile.NamedTemporaryFile(mode='w',suffix='.py',delete=False,encoding='utf-8')
    try:
        f.write(code);f.close()
        r=subprocess.run([_sys.executable,f.name],capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
        return {'stdout':r.stdout[:6000],'stderr':r.stderr[:2000],'returncode':r.returncode,'timed_out':False}
    except subprocess.TimeoutExpired as e:return {'stdout':(e.stdout or b'')[:6000].decode('utf-8','replace'),'stderr':f'(killed after {timeout}s timeout)','returncode':None,'timed_out':True}
    except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
    finally:
        try:os.unlink(f.name)
        except Exception:pass
def _extract_synthetic_q(filename:str,chunk:str)->str:
    h=re.search(r'^#+\s+(.+?)$',chunk,re.MULTILINE)
    if h:return f'What does "{h.group(1).strip()}" describe in {filename}?'
    cls_or_def=re.search(r'^(?:class|def)\s+([A-Za-z_]\w*)',chunk,re.MULTILINE)
    if cls_or_def:return f'What is {cls_or_def.group(1)} in {filename}?'
    first_sent=re.match(r'^[^.\n]{20,180}[.!?]',chunk.strip())
    if first_sent:return f'In {filename}, what does this say: "{first_sent.group(0).strip()[:120]}..."?'
    first_words=' '.join(chunk.split()[:8])
    return f'What does {filename} say about "{first_words}"?'
def _chunk_text(text:str,max_chars:int=1500)->List[str]:
    paras=[p.strip() for p in re.split(r'\n\s*\n',text) if p.strip()]
    out=[];buf=''
    for p in paras:
        if len(buf)+len(p)+2<=max_chars:buf=f'{buf}\n\n{p}' if buf else p
        else:
            if buf:out.append(buf)
            if len(p)<=max_chars:buf=p
            else:
                for i in range(0,len(p),max_chars):out.append(p[i:i+max_chars])
                buf=''
    if buf:out.append(buf)
    return out
def _iter_files(root:Path,glob:str,max_files:int,exts:Optional[set])->List[Path]:
    if root.is_file():return [root][:max_files]
    out=[]
    for p in root.rglob(glob):
        if not p.is_file():continue
        if exts is not None and p.suffix.lower() not in exts:continue
        if p.stat().st_size>2_000_000:continue
        out.append(p)
        if len(out)>=max_files:break
    return out
def _skill_scan(args,ctx,reg):
    adam=ctx.get('adam')
    if adam is None:return {'error':'scan requires Adam (no ctx adam)'}
    p=args.get('path')
    if not p:return {'error':'missing path arg'}
    root=Path(p).resolve()
    if not root.exists():return {'error':f'path not found: {p}'}
    glob=args.get('glob','**/*');max_files=int(args.get('max_files',50));max_chars_per_file=int(args.get('max_chars_per_file',8000));distill=bool(args.get('distill',False));only_text=bool(args.get('only_text',True))
    exts=_TEXT_EXT if only_text else None
    files=_iter_files(root,glob,max_files,exts)
    if not files:return {'files_scanned':0,'lessons_added':0,'note':'no matching files'}
    n_lessons_before=len(adam.sem_lut._raw) if adam.sem_lut is not None else 0
    errors=[];scanned=[];pending=[]
    for f in files:
        try:txt=f.read_text(encoding='utf-8',errors='replace')[:max_chars_per_file]
        except Exception as e:errors.append({'file':str(f),'error':str(e)});continue
        chunks=_chunk_text(txt,max_chars=1500)
        for i,ch in enumerate(chunks):
            q=None
            if distill:
                try:
                    sys_p='Generate ONE concise question whose answer is found in the text below. Format: "Q: ..."'
                    qresp=adam.adam.svc.chat(f'Text:\n{ch[:1200]}',system=sys_p,max_new_tokens=40,do_sample=False,kb_top_k=0)
                    qline=(qresp[0] if isinstance(qresp,tuple) else qresp).strip()
                    q=re.sub(r'^Q:\s*','',qline,flags=re.IGNORECASE).strip() or None
                except Exception as e:errors.append({'file':str(f),'chunk':i,'distill_error':str(e)})
            if not q:q=_extract_synthetic_q(f.name,ch)
            pending.append((q,ch))
        scanned.append({'file':str(f),'chunks':len(chunks),'bytes':len(txt)})
    if adam.sem_lut is not None and pending:
        for q,a in pending:adam.sem_lut.add(q,a)
        try:adam.sem_lut.fit()
        except Exception as e:errors.append({'fit_error':str(e)})
        try:adam.save_lessons()
        except Exception as e:errors.append({'save_error':str(e)})
    n_lessons_after=len(adam.sem_lut._raw) if adam.sem_lut is not None else 0
    return {'files_scanned':len(scanned),'lessons_added':n_lessons_after-n_lessons_before,'lessons_total':n_lessons_after,'distilled':distill,'errors':errors[:10],'files':scanned[:20],'bulk_fit':True}
def default_registry(workdir:Optional[str]=None,roots:Optional[List[str]]=None,audit_log:Optional[str]='logs/agent_skill_calls.jsonl',unrestricted:bool=False,with_agentic:bool=True)->SkillRegistry:
    reg=SkillRegistry(workdir=workdir,roots=roots,audit_log=audit_log,unrestricted=unrestricted)
    scope='UNRESTRICTED (all drives)' if unrestricted else f'{len(reg.roots)} root(s)'
    reg.register('time',_skill_time,desc='Get current local time. Args: {}',schema={})
    reg.register('calc',_skill_calc,desc='Compute a math expression via Adam tier-3.7. Args: {expr}',schema={'expr':'str'})
    reg.register('mem',_skill_mem,desc="Query Adam's lesson bank. Args: {query}",schema={'query':'str'})
    reg.register('web',_skill_web,desc="DDG search + distill via Adam's crawler. Args: {query}",schema={'query':'str'})
    reg.register('file_read',_skill_file_read,gate=_gate_path,desc=f'Read a UTF-8 text file within {scope}. Args: {{path, max_bytes?}}',schema={'path':'str','max_bytes':'int?'})
    reg.register('file_write',_skill_file_write,gate=_gate_path,desc=f'Write/overwrite a UTF-8 text file within {scope}. Args: {{path, content}}',schema={'path':'str','content':'str'})
    reg.register('code_edit',_skill_code_edit,gate=_gate_code_edit,desc=f'Find-and-replace edit in a file within {scope}; .py edits ast-validated. Args: {{path, find, replace, count?}}',schema={'path':'str','find':'str','replace':'str','count':'int?'})
    reg.register('shell',_skill_shell,gate=_gate_shell,desc=f'Run a read-only allowlisted shell command from primary root. Scope: {scope}. Args: {{cmd, timeout?}}',schema={'cmd':'str','timeout':'int?'})
    reg.register('run_python',_skill_run_python,desc='Execute a Python snippet in a sandboxed subprocess (workdir-confined, timeout-bounded). Rejects dangerous ops (network, fs-mutation, subprocess, exec/eval). Returns stdout/stderr/returncode. Args: {code, timeout?}',schema={'code':'str','timeout':'int?'})
    reg.register('scan',_skill_scan,gate=_gate_path,desc=f'Walk path (file or dir + glob), chunk text, teach each chunk to Adam. Args: {{path, glob?, max_files?, max_chars_per_file?, distill?, only_text?}}',schema={'path':'str','glob':'str?','max_files':'int?','max_chars_per_file':'int?','distill':'bool?','only_text':'bool?'})
    if with_agentic:
        try:from amni.serve.agentic import register as _reg_agentic;_reg_agentic(reg)
        except Exception as e:print(f'[skills] agentic register failed: {e}',flush=True)
    return reg
