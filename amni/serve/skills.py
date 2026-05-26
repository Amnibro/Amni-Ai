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
def _scrub_pii_from_query(q:str,agent=None)->str:
    """Strip personal-info tokens (name parts, location, email, phone) from web queries before any external HTTP.
    Pulls known PII from the agent's PersonalAtlas if available."""
    if not q:return q
    import re as _re
    cleaned=q
    cleaned=_re.sub(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b','',cleaned)
    cleaned=_re.sub(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b','',cleaned)
    cleaned=_re.sub(r'\+\d{1,3}[ -.]?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b','',cleaned)
    cleaned=_re.sub(r'\b\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b','',cleaned)
    try:
        pa=getattr(agent,'personal_atlas',None) if agent is not None else None
        if pa is not None and hasattr(pa,'list_facts'):
            facts=pa.list_facts(include_confidential=True,limit=200) or []
            tokens=set()
            for f in facts:
                v=str(f.get('value') or '').strip()
                if not v:continue
                for tok in _re.findall(r"[A-Za-z][A-Za-z'\-]{2,}",v):
                    tl=tok.lower()
                    if tl in ('the','and','for','with','this','that','from','your','have'):continue
                    if len(tl)>=3:tokens.add(tok)
            for tok in sorted(tokens,key=len,reverse=True):
                cleaned=_re.sub(r'(?i)\b'+_re.escape(tok)+r'\b','',cleaned)
    except Exception:pass
    cleaned=_re.sub(r'\s+',' ',cleaned).strip(' ,.;:!?')
    return cleaned or q
def _skill_web(args,ctx,reg):
    adam=ctx.get('adam')
    agent=ctx.get('agent')
    q_raw=args.get('query','')
    if not q_raw:return {'error':'missing query'}
    if adam is None or not hasattr(adam,'adam') or adam.adam.crawler is None:return {'error':'web crawler not available'}
    q=_scrub_pii_from_query(q_raw,agent=agent)
    try:
        ans,sources,n=adam.adam.crawler.crawl_and_distill(q,subject=None,letter_only=False)
        return {'answer':ans,'sources':sources[:5],'tokens':n,'query_used':q,'pii_scrubbed':q!=q_raw}
    except Exception as e:return {'error':str(e),'query_used':q}
def _skill_find(args,ctx,reg):
    """Fast substring/regex search across the workdir. Returns top N hits as file:line + snippet."""
    import re as _re,fnmatch
    query=str(args.get('query') or '').strip()
    if not query:return {'error':'missing query'}
    use_regex=bool(args.get('regex',False))
    case_sensitive=bool(args.get('case_sensitive',False))
    glob_pat=str(args.get('glob') or '').strip()
    max_hits=int(args.get('max_hits',30) or 30)
    max_chars=int(args.get('max_chars',180) or 180)
    workdir=getattr(reg,'workdir',None) or '.'
    root=Path(workdir).resolve()
    if not root.exists():return {'error':f'workdir does not exist: {root}'}
    try:
        if use_regex:pat=_re.compile(query,0 if case_sensitive else _re.IGNORECASE)
        else:pat=_re.compile(_re.escape(query),0 if case_sensitive else _re.IGNORECASE)
    except _re.error as e:return {'error':f'bad regex: {e}'}
    _IGNORE_DIRS={'.git','.venv','venv','__pycache__','node_modules','.pytest_cache','.mypy_cache','.idea','.vscode','dist','build','.next','.nuxt','.adam-venvs','bakes','models','downloaded_models','hf_cache','ptex_hf','textures','checkpoints','eval_reports','full_lexicon_atlas','archive','environment_files'}
    _BINARY_EXTS={'.png','.jpg','.jpeg','.gif','.webp','.ico','.pdf','.zip','.tar','.gz','.7z','.exe','.dll','.so','.dylib','.bin','.npz','.safetensors','.onnx','.pt','.pth','.pyc','.pyo','.mp4','.mp3','.wav','.ogg','.woff','.woff2','.ttf','.eot'}
    hits=[];files_scanned=0;total_matches=0
    for p in root.rglob('*'):
        if not p.is_file():continue
        if any(part in _IGNORE_DIRS for part in p.relative_to(root).parts[:-1]):continue
        if p.suffix.lower() in _BINARY_EXTS:continue
        try:rel=str(p.relative_to(root)).replace('\\','/')
        except Exception:continue
        if glob_pat and not fnmatch.fnmatch(rel,glob_pat):continue
        try:size=p.stat().st_size
        except Exception:continue
        if size>500_000:continue
        try:text=p.read_text(encoding='utf-8',errors='ignore')
        except Exception:continue
        files_scanned+=1
        for ln,line in enumerate(text.splitlines(),1):
            if pat.search(line):
                snippet=line.strip()
                if len(snippet)>max_chars:
                    m=pat.search(snippet);mid=m.start() if m else 0
                    start=max(0,mid-max_chars//2);end=min(len(snippet),start+max_chars)
                    snippet=('…' if start>0 else '')+snippet[start:end]+('…' if end<len(snippet) else '')
                hits.append({'path':rel,'line':ln,'snippet':snippet})
                total_matches+=1
                if len(hits)>=max_hits:break
        if len(hits)>=max_hits:break
    return {'query':query,'regex':use_regex,'case_sensitive':case_sensitive,'glob':glob_pat or None,'hits':hits,'n_hits':len(hits),'files_scanned':files_scanned,'truncated':len(hits)>=max_hits and total_matches>=max_hits}
def _skill_file_read(args,ctx,reg):
    p=args['path'];max_bytes=int(args.get('max_bytes',65536))
    _offset=int(args.get('offset',0));_limit=int(args.get('limit',0));_line_offset=int(args.get('line_offset',0));_line_limit=int(args.get('line_limit',0))
    raw=Path(p).read_text(encoding='utf-8',errors='replace')
    if _line_offset or _line_limit:
        lines=raw.splitlines(keepends=True)
        sliced=lines[_line_offset:_line_offset+_line_limit] if _line_limit else lines[_line_offset:]
        data=''.join(sliced)[:max_bytes]
        return {'path':str(p),'content':data,'bytes':len(data),'total_lines':len(lines),'returned_lines':len(sliced),'line_offset':_line_offset}
    if _offset or _limit:
        end=_offset+_limit if _limit else _offset+max_bytes
        data=raw[_offset:end][:max_bytes]
        return {'path':str(p),'content':data,'bytes':len(data),'total_bytes':len(raw),'offset':_offset}
    data=raw[:max_bytes]
    return {'path':p,'content':data,'bytes':len(data)}
def _file_change_stats(before:str,after:str,max_preview_lines:int=10):
    bl=before.splitlines() if before else [];al=after.splitlines()
    return {'lines_before':len(bl),'lines_after':len(al),'lines_added':max(0,len(al)-len(bl)),'lines_removed':max(0,len(bl)-len(al)),'bytes_before':len(before),'bytes_after':len(after),'preview':'\n'.join(al[:max_preview_lines])+(f'\n... ({len(al)-max_preview_lines} more)' if len(al)>max_preview_lines else ''),'before_preview':'\n'.join(bl[:max_preview_lines])+(f'\n... ({len(bl)-max_preview_lines} more)' if len(bl)>max_preview_lines else ''),'diff_unified':_unified_diff(bl,al,max_lines=max_preview_lines*4)}
def _unified_diff(before_lines,after_lines,max_lines:int=40):
    import difflib
    if not before_lines and not after_lines:return ''
    if before_lines==after_lines:return ''
    raw=list(difflib.unified_diff(before_lines or [],after_lines or [],lineterm='',n=2))
    if len(raw)>max_lines:raw=raw[:max_lines]+[f'... ({len(raw)-max_lines} more diff lines)']
    return '\n'.join(raw)
def _skill_file_write(args,ctx,reg):
    from amni.serve.edit_verifier import verify_edit
    p=Path(args['path']);content=args.get('content','')
    existed=p.exists();before=p.read_text(encoding='utf-8',errors='ignore') if existed else ''
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
    op='create' if not existed else 'overwrite'
    return {'path':str(p),'bytes_written':len(content),'ext':p.suffix.lstrip('.') or 'txt','created':not existed,'change':_file_change_stats(before,content),'verification':verify_edit(str(p),content,op=op)}
def _skill_code_edit(args,ctx,reg):
    from amni.serve.edit_verifier import verify_edit
    p=Path(args['path']);find=args['find'];replace=args['replace'];count=int(args.get('count',1))
    src=p.read_text(encoding='utf-8')
    if find not in src:return {'error':'find string not present','path':str(p)}
    new=src.replace(find,replace,count) if count>0 else src.replace(find,replace)
    if p.suffix=='.py':
        try:ast.parse(new)
        except SyntaxError as e:return {'error':f'syntax error after edit: {e}','path':str(p)}
    p.write_text(new,encoding='utf-8')
    return {'path':str(p),'replacements':src.count(find) if count==0 else min(count,src.count(find)),'ext':p.suffix.lstrip('.') or 'txt','change':_file_change_stats(src,new),'verification':verify_edit(str(p),new,op='edit')}
def _skill_shell(args,ctx,reg):
    from amni.serve.shell_audit import log_shell_run
    cmd=args['cmd'];timeout=int(args.get('timeout',15));t0=time.time()
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
    dur=round(time.time()-t0,3)
    log_shell_run('shell',cmd,r.returncode,r.stdout,r.stderr,str(reg.workdir),dur)
    return {'cmd':cmd,'returncode':r.returncode,'stdout':r.stdout[:8000],'stderr':r.stderr[:4000],'duration_s':dur}
_GIT_SAFE_CMDS={'status','log','diff','branch','blame','show','ls-files','remote','config','rev-parse','describe','tag','shortlog','reflog'}
def _skill_git(args,ctx,reg):
    cmd=(args.get('cmd') or '').strip()
    if not cmd:return {'error':'missing cmd. Allowed: '+','.join(sorted(_GIT_SAFE_CMDS))}
    head=cmd.split()[0].lower()
    if head not in _GIT_SAFE_CMDS:return {'error':f'cmd {head!r} not in safe allowlist (read-only git ops only). Allowed: '+','.join(sorted(_GIT_SAFE_CMDS))}
    parts=['git']+cmd.split()
    if args.get('file'):parts.append(str(args['file']))
    if args.get('n') and head in ('log','shortlog','reflog'):parts[2:2]=['-n',str(int(args['n']))]
    try:
        from amni.serve.shell_audit import log_shell_run
        t0=time.time();r=subprocess.run(parts,capture_output=True,text=True,timeout=20,cwd=str(reg.workdir));dur=round(time.time()-t0,3)
        log_shell_run('git',' '.join(parts),r.returncode,r.stdout,r.stderr,str(reg.workdir),dur)
        return {'cmd':' '.join(parts),'returncode':r.returncode,'stdout':r.stdout[:6000],'stderr':r.stderr[:1500],'duration_s':dur}
    except FileNotFoundError:return {'error':'git not installed or not in PATH'}
    except subprocess.TimeoutExpired:return {'error':'git command exceeded 20s timeout'}
    except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
def _apply_unified_diff(content:str,diff_text:str):
    lines=content.splitlines(keepends=True);out=lines[:];hunks=[]
    cur=None
    for ln in diff_text.splitlines():
        if ln.startswith('@@'):
            m=re.match(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@',ln)
            if not m:continue
            cur={'old_start':int(m.group(1)),'old_count':int(m.group(2) or 1),'new_start':int(m.group(3)),'new_count':int(m.group(4) or 1),'lines':[]}
            hunks.append(cur)
        elif cur is not None and (ln.startswith(' ') or ln.startswith('+') or ln.startswith('-')):cur['lines'].append(ln)
    if not hunks:return None,'no @@ hunks found in diff'
    offset=0
    for h in hunks:
        old_lines=[(l[1:]+'\n' if not l[1:].endswith('\n') else l[1:]) for l in h['lines'] if l.startswith((' ','-'))]
        new_lines=[(l[1:]+'\n' if not l[1:].endswith('\n') else l[1:]) for l in h['lines'] if l.startswith((' ','+'))]
        start=h['old_start']-1+offset
        actual=out[start:start+len(old_lines)]
        if ''.join(actual).rstrip()!=''.join(old_lines).rstrip():return None,f"hunk at line {h['old_start']} doesn't match — file content drifted"
        out[start:start+len(old_lines)]=new_lines
        offset+=len(new_lines)-len(old_lines)
    return ''.join(out),None
_FORMATTERS={'.py':[('ruff format','ruff'),('black','black')],'.rs':[('rustfmt','rustfmt')],'.js':[('prettier --write','prettier')],'.jsx':[('prettier --write','prettier')],'.ts':[('prettier --write','prettier')],'.tsx':[('prettier --write','prettier')],'.go':[('gofmt -w','gofmt')],'.json':[('prettier --write','prettier')],'.html':[('prettier --write','prettier')],'.css':[('prettier --write','prettier')]}
_CODE_EXTS=('.py','.rs','.js','.jsx','.ts','.tsx','.mjs','.go','.cpp','.cc','.c','.h','.hpp','.java','.kt','.rb','.php','.swift','.cs','.scala')
def _skill_rename_symbol(args,ctx,reg):
    old=(args.get('old') or '').strip();new=(args.get('new') or '').strip()
    if not old or not new:return {'error':'missing old/new'}
    if not re.match(r'^[A-Za-z_]\w*$',old):return {'error':f'old name {old!r} not a valid identifier'}
    if not re.match(r'^[A-Za-z_]\w*$',new):return {'error':f'new name {new!r} not a valid identifier'}
    root=Path(reg.workdir);glob=args.get('glob');dry_run=bool(args.get('dry_run'))
    if glob:targets=[p for p in root.glob(glob) if p.is_file()]
    else:
        exts=set(args.get('exts') or _CODE_EXTS)
        targets=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts and '.venv' not in p.parts and 'node_modules' not in p.parts and '.git' not in p.parts and '__pycache__' not in p.parts]
    if len(targets)>1000:return {'error':f'too many files to scan ({len(targets)}); pass glob= or exts= to narrow'}
    pat=re.compile(r'\b'+re.escape(old)+r'\b')
    files_changed=[];total_replacements=0
    for fp in targets:
        try:src=fp.read_text(encoding='utf-8',errors='replace')
        except Exception:continue
        if old not in src:continue
        new_src,n=pat.subn(new,src)
        if n>0:
            files_changed.append({'path':str(fp.relative_to(root)),'replacements':n})
            total_replacements+=n
            if not dry_run:
                try:fp.write_text(new_src,encoding='utf-8')
                except Exception as e:files_changed[-1]['write_error']=str(e)[:200]
    return {'old':old,'new':new,'files_scanned':len(targets),'files_changed':len(files_changed),'total_replacements':total_replacements,'changes':files_changed[:30],'dry_run':dry_run}
def _skill_prune_sessions(args,ctx,reg):
    older_than_days=int(args.get('older_than_days',30));keep_n=int(args.get('keep_n',50));dry_run=bool(args.get('dry_run',False))
    conv_root=Path(reg.workdir)/'experiences'/'conversations'
    if not conv_root.exists():
        for cand in (Path.cwd()/'experiences'/'conversations',Path('experiences/conversations')):
            if cand.exists():conv_root=cand;break
    if not conv_root.exists():return {'error':'no conversations directory found'}
    files=sorted(conv_root.glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
    cutoff=time.time()-(older_than_days*86400)
    keep=set(str(p) for p in files[:keep_n])
    candidates=[p for p in files[keep_n:] if p.stat().st_mtime<cutoff and str(p) not in keep]
    freed=sum(p.stat().st_size for p in candidates);deleted=[]
    if not dry_run:
        for p in candidates:
            try:p.unlink();deleted.append(p.name)
            except Exception:pass
    return {'total_sessions':len(files),'kept_recent':min(keep_n,len(files)),'candidates_for_prune':len(candidates),'deleted':len(deleted),'freed_bytes':freed,'freed_mb':round(freed/(1024*1024),2),'dry_run':dry_run,'older_than_days':older_than_days,'keep_n':keep_n,'samples':[p.name for p in candidates[:5]]}
def _skill_export_session(args,ctx,reg):
    """Dump a conversation session as markdown. Args: {session_id, out_path?, format?}"""
    import json as _j
    sid=args.get('session_id')
    if not sid:return {'error':'missing session_id'}
    fmt=(args.get('format') or 'markdown').lower()
    conv_root=Path(reg.workdir)/'experiences'/'conversations'
    if not conv_root.exists():
        for cand in (Path.cwd()/'experiences'/'conversations',Path('experiences/conversations')):
            if cand.exists():conv_root=cand;break
    fp=conv_root/f'{sid}.jsonl'
    if not fp.exists():return {'error':f'session {sid!r} not found at {fp}'}
    try:lines=fp.read_text(encoding='utf-8').strip().splitlines()
    except Exception as e:return {'error':f'read failed: {e}'}
    turns=[]
    for ln in lines:
        try:turns.append(_j.loads(ln))
        except Exception:continue
    if fmt=='json':
        out_text=_j.dumps(turns,indent=2,ensure_ascii=False)
    elif fmt=='text':
        out_text='\n\n'.join(f"[{t.get('role','?')}] {t.get('content','')}" for t in turns)
    else:
        lines_md=[f"# Conversation `{sid}`",f"_{len(turns)} turns from {fp.name}_\n"]
        for t in turns:
            role=t.get('role','?');content=(t.get('content') or '').strip()
            meta_bits=[f"tier={t['tier']}"] if t.get('tier') else []
            if t.get('persona'):meta_bits.append(f"persona={t['persona']}")
            if t.get('is_private'):meta_bits.append('private')
            if t.get('blocked'):meta_bits.append('blocked')
            meta=(' · '.join(meta_bits))
            if role=='user':lines_md.append(f"\n## 👤 user");lines_md.append(content)
            elif role=='assistant':lines_md.append(f"\n## 🤖 {t.get('persona','assistant')}"+(f" · *{meta}*" if meta else ''));lines_md.append(content)
            else:lines_md.append(f"\n## {role}\n{content}")
        out_text='\n'.join(lines_md)
    out_path=args.get('out_path')
    if out_path:
        gate_res=_gate_path({'path':out_path},ctx,reg)
        if gate_res and gate_res.get('error'):return gate_res
        Path(out_path).write_text(out_text,encoding='utf-8')
        return {'session_id':sid,'turns':len(turns),'format':fmt,'path':out_path,'bytes':len(out_text)}
    return {'session_id':sid,'turns':len(turns),'format':fmt,'content':out_text[:8000],'truncated':len(out_text)>8000}
def _skill_parse_error(args,ctx,reg):
    """Parse a compiler/runtime error message and identify the root cause + likely fix."""
    txt=args.get('text') or args.get('error','')
    if not txt:return {'error':'missing text/error'}
    result={'language':None,'kind':None,'file':None,'line':None,'message':None,'likely_cause':None,'suggested_fix':None}
    pat_py=re.search(r'File "([^"]+)", line (\d+)(?:, in (\w+))?\s*\n[^\n]*\n\s*(\w+(?:Error|Exception|Warning))\s*:\s*(.+?)(?:\n|$)',txt,re.DOTALL)
    if pat_py:
        result.update({'language':'Python','file':pat_py.group(1),'line':int(pat_py.group(2)),'function':pat_py.group(3),'kind':pat_py.group(4),'message':pat_py.group(5).strip()})
    else:
        m=re.search(r'(\w+(?:Error|Exception))\s*:\s*(.+?)(?:\n|$)',txt)
        if m:result.update({'language':'Python','kind':m.group(1),'message':m.group(2).strip()})
    if not result['language']:
        m=re.search(r'error\[(E\d+)\]:\s*(.+?)\n.*?-->\s*([^\s:]+):(\d+):(\d+)',txt,re.DOTALL)
        if m:result.update({'language':'Rust','kind':m.group(1),'message':m.group(2).strip(),'file':m.group(3),'line':int(m.group(4)),'col':int(m.group(5))})
    if not result['language']:
        m=re.search(r'(\w+(?:Error)?):\s*(.+?)\n\s*at\s+[^\s]+\s+\(([^:)]+):(\d+):(\d+)\)',txt)
        if m:result.update({'language':'JavaScript','kind':m.group(1),'message':m.group(2).strip(),'file':m.group(3),'line':int(m.group(4)),'col':int(m.group(5))})
    if not result['language']:
        m=re.search(r'([^:]+\.go):(\d+):(\d+):\s*(.+?)(?:\n|$)',txt)
        if m:result.update({'language':'Go','file':m.group(1),'line':int(m.group(2)),'col':int(m.group(3)),'message':m.group(4).strip()})
    _CAUSE_PATTERNS={
        r'NameError.*not defined':('undefined name','add an import or check spelling'),
        r'AttributeError.*has no attribute':('wrong attribute or wrong object type','check the object type or the attribute name'),
        r'TypeError.*missing.*required.*argument':('missing function argument','add the missing argument to the call'),
        r'TypeError.*takes.*positional argument':('argument count mismatch','check the function signature'),
        r'ImportError|ModuleNotFoundError':('missing module','pip install the package or fix the import path'),
        r'IndentationError|TabError':('Python indentation','fix indentation (4 spaces, no tabs)'),
        r'SyntaxError.*invalid syntax':('Python syntax error','check for unmatched brackets, missing colons, or stray characters'),
        r'IndexError.*out of range':('list/array index out of bounds','add a length check before indexing'),
        r'KeyError':('dict key missing','use .get(key, default) or check membership first'),
        r'ZeroDivisionError':('division by zero','add a check that the divisor is non-zero'),
        r'RecursionError':('infinite recursion','add a base case or increase sys.setrecursionlimit'),
        r'cannot find function|cannot find type|cannot find macro|unresolved import':('Rust missing import','add use statement for the missing symbol'),
        r'mismatched types':('Rust type mismatch','check the function signature or add an explicit conversion'),
        r'borrow.*cannot be|borrowed.*moved':('Rust borrow checker','clone the value or restructure to avoid concurrent borrows'),
        r'cannot move out of':('Rust ownership move','use a reference, clone, or restructure to avoid move-after-borrow'),
        r'TS\d{4}':('TypeScript type error','check types and interface declarations'),
        r'cannot read propert(?:y|ies) of (?:null|undefined)':('JS null/undefined access','add a null check before accessing the property'),
        r'undefined: ':('Go undefined symbol','import the package or check spelling'),
    }
    for pat,(cause,fix) in _CAUSE_PATTERNS.items():
        if re.search(pat,txt,re.IGNORECASE):result['likely_cause']=cause;result['suggested_fix']=fix;break
    return {k:v for k,v in result.items() if v is not None}
def _skill_auto_import(args,ctx,reg):
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=Path(path)
    if not p.exists():return {'error':f'file not found: {path}'}
    if p.suffix.lower()!='.py':return {'error':'only .py supported currently'}
    src=p.read_text(encoding='utf-8')
    try:tree=ast.parse(src)
    except SyntaxError as e:return {'error':f'parse error at line {e.lineno}: {e.msg}'}
    existing_imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for n in node.names:existing_imports.add(n.asname or n.name)
        elif isinstance(node,ast.ImportFrom):
            for n in node.names:existing_imports.add(n.asname or n.name)
    referenced_names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):referenced_names.add(node.id)
        elif isinstance(node,ast.Attribute):
            base=node
            while isinstance(base,ast.Attribute):base=base.value
            if isinstance(base,ast.Name):referenced_names.add(base.id)
    _BUILTINS={'print','len','range','str','int','float','bool','list','dict','tuple','set','frozenset','type','isinstance','issubclass','object','None','True','False','self','cls','enumerate','zip','map','filter','sorted','reversed','min','max','sum','abs','round','pow','open','input','iter','next','any','all','hash','id','repr','format','vars','dir','getattr','setattr','hasattr','delattr','globals','locals','super','property','staticmethod','classmethod','Exception','ValueError','TypeError','KeyError','IndexError','RuntimeError','OSError','FileNotFoundError','ImportError','AttributeError','NotImplementedError','StopIteration','ZeroDivisionError','NameError','Warning','DeprecationWarning'}
    _STDLIB_HINTS={'os':'os','sys':'sys','re':'re','json':'json','time':'time','math':'math','random':'random','datetime':'datetime','Path':'from pathlib import Path','defaultdict':'from collections import defaultdict','Counter':'from collections import Counter','OrderedDict':'from collections import OrderedDict','deque':'from collections import deque','dataclass':'from dataclasses import dataclass','field':'from dataclasses import field','asdict':'from dataclasses import asdict','partial':'from functools import partial','wraps':'from functools import wraps','lru_cache':'from functools import lru_cache','cache':'from functools import cache','reduce':'from functools import reduce','chain':'from itertools import chain','combinations':'from itertools import combinations','permutations':'from itertools import permutations','product':'from itertools import product','islice':'from itertools import islice','Optional':'from typing import Optional','List':'from typing import List','Dict':'from typing import Dict','Tuple':'from typing import Tuple','Any':'from typing import Any','Union':'from typing import Union','Callable':'from typing import Callable','Iterator':'from typing import Iterator','Sequence':'from typing import Sequence','Mapping':'from typing import Mapping'}
    undefined=referenced_names-existing_imports-_BUILTINS
    suggestions=[]
    for name in sorted(undefined):
        if name in _STDLIB_HINTS:suggestions.append({'name':name,'import':_STDLIB_HINTS[name]})
    return {'path':str(p),'undefined_references':sorted(undefined),'suggested_imports':suggestions,'note':'Apply manually via file_write or code_edit. AST-based detection only catches stdlib hints currently.'}
def _skill_format_code(args,ctx,reg):
    import shutil as _sh
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=Path(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    ext=p.suffix.lower()
    if ext not in _FORMATTERS:return {'skipped':True,'reason':f'no formatter for {ext}'}
    for tool_cmd,binary in _FORMATTERS[ext]:
        if _sh.which(binary):
            try:
                r=subprocess.run(tool_cmd.split()+[str(p)],capture_output=True,text=True,timeout=15,cwd=str(reg.workdir))
                return {'path':str(p),'formatter':binary,'returncode':r.returncode,'stdout':r.stdout[:500],'stderr':r.stderr[:500]}
            except Exception as e:return {'error':f'{type(e).__name__}: {e}','formatter':binary}
    return {'skipped':True,'reason':f'no formatter installed for {ext} (tried: {[b for _,b in _FORMATTERS[ext]]})'}
def _skill_tts(args,ctx,reg):
    try:from amni.voice import speak,tts_backend,list_voices
    except Exception as e:return {'error':f'voice module import failed: {e}'}
    text=args.get('text','')
    if args.get('list_voices'):return {'backend':tts_backend(),'voices':list_voices()[:30]}
    if not text:return {'error':'missing text'}
    _voice=args.get('voice');_persona_key=None
    _ps=ctx.get('personas') or (getattr(ctx.get('agent'),'personas',None)) or (getattr(ctx.get('adam'),'personas',None))
    try:
        if _ps is not None:
            _sid=args.get('session_id') or args.get('sid')
            _name=None
            if _sid and hasattr(_ps,'session_persona'):_name=_ps.session_persona(_sid)
            if not _name and hasattr(_ps,'_default'):_name=_ps._default
            _cur=_ps.get(_name) if _name else None
            if _cur:
                _persona_key=(_cur.name or '').lower()
                if not _voice and hasattr(_cur,'tts_voice'):_voice=_cur.tts_voice
    except Exception as _pe:print(f'[tts] persona lookup: {_pe}',flush=True)
    audio=speak(text,backend=args.get('backend'),voice=_voice,persona=_persona_key)
    if not audio:return {'error':'TTS produced no audio','backend':tts_backend()}
    out_path=args.get('out_path')
    if out_path:
        gate_res=_gate_path({'path':out_path},ctx,reg)
        if gate_res and gate_res.get('error'):return gate_res
        Path(out_path).write_bytes(audio)
        return {'path':out_path,'bytes':len(audio),'backend':tts_backend(),'voice_used':_voice}
    import base64
    return {'audio_base64':base64.b64encode(audio).decode('ascii'),'bytes':len(audio),'backend':tts_backend(),'mime':'audio/wav','voice_used':_voice}
def _skill_stt(args,ctx,reg):
    try:from amni.voice import transcribe,stt_backend
    except Exception as e:return {'error':f'voice module import failed: {e}'}
    audio=None
    if args.get('path'):
        gate_res=_gate_path({'path':args['path']},ctx,reg)
        if gate_res and gate_res.get('error'):return gate_res
        try:audio=Path(args['path']).read_bytes()
        except Exception as e:return {'error':f'read failed: {e}'}
    elif args.get('audio_base64'):
        import base64
        try:audio=base64.b64decode(args['audio_base64'])
        except Exception as e:return {'error':f'base64 decode failed: {e}'}
    if not audio:return {'error':'missing path or audio_base64','backend':stt_backend()}
    return transcribe(audio,backend=args.get('backend'),model_size=args.get('model_size','base'))
def _skill_symbols(args,ctx,reg):
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=Path(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    src=p.read_text(encoding='utf-8',errors='replace')
    ext=p.suffix.lower();out={'path':str(p),'lang':None,'functions':[],'classes':[],'imports':[]}
    if ext=='.py':
        out['lang']='Python'
        try:
            tree=ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef):
                    out['functions'].append({'name':node.name,'line':node.lineno,'args':[a.arg for a in node.args.args]})
                elif isinstance(node,ast.ClassDef):
                    methods=[n.name for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    out['classes'].append({'name':node.name,'line':node.lineno,'methods':methods[:20]})
                elif isinstance(node,ast.Import):
                    out['imports'].extend(n.name for n in node.names)
                elif isinstance(node,ast.ImportFrom):
                    out['imports'].append(f"{node.module}.{','.join(n.name for n in node.names)}")
        except SyntaxError as e:return {'error':f'parse error at line {e.lineno}: {e.msg}'}
    elif ext=='.rs':
        out['lang']='Rust'
        for m in re.finditer(r'^\s*(?:pub\s+)?fn\s+(\w+)\s*[<(]',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r'^\s*(?:pub\s+)?struct\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'struct'})
        for m in re.finditer(r'^\s*(?:pub\s+)?enum\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'enum'})
        for m in re.finditer(r'^\s*(?:pub\s+)?trait\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'trait'})
        for m in re.finditer(r'^\s*use\s+([\w:{}*,\s]+);',src,re.MULTILINE):
            out['imports'].append(m.group(1).strip())
    elif ext in ('.js','.ts','.jsx','.tsx','.mjs'):
        out['lang']='JavaScript/TypeScript'
        for m in re.finditer(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r'^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln,'kind':'arrow'})
        for m in re.finditer(r'^\s*(?:export\s+)?class\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r"^\s*import\s+(.+?)\s+from\s+['\"](.+?)['\"]",src,re.MULTILINE):
            out['imports'].append(f"{m.group(1).strip()} from {m.group(2)}")
    else:return {'lang':None,'note':f'no symbol extractor for {ext} — use file_read instead'}
    return out
def _skill_project_info(args,ctx,reg):
    p=Path(reg.workdir);info={'workdir':str(p),'top_files':[],'dependencies':{},'languages':set(),'git':{}}
    try:
        for f in sorted(p.iterdir()):
            if f.is_file() and not f.name.startswith('.') and len(info['top_files'])<25:info['top_files'].append(f.name)
    except Exception:pass
    _LANG_EXT={'.py':'Python','.rs':'Rust','.js':'JavaScript','.ts':'TypeScript','.tsx':'TypeScript','.jsx':'JavaScript','.go':'Go','.cpp':'C++','.cc':'C++','.c':'C','.h':'C/C++','.hpp':'C++','.java':'Java','.kt':'Kotlin','.rb':'Ruby','.php':'PHP','.swift':'Swift','.cs':'C#','.scala':'Scala','.sh':'Shell','.html':'HTML','.css':'CSS','.sql':'SQL'}
    for f in p.rglob('*'):
        if f.is_file():
            s=f.suffix.lower()
            if s in _LANG_EXT:info['languages'].add(_LANG_EXT[s])
        if len(info['languages'])>=8:break
    info['languages']=sorted(info['languages'])
    for dep_file,key in (('Cargo.toml','cargo'),('package.json','npm'),('pyproject.toml','python'),('requirements.txt','pip'),('go.mod','go'),('Gemfile','ruby'),('pom.xml','maven'),('build.gradle','gradle'),('CMakeLists.txt','cmake'),('Makefile','make')):
        if (p/dep_file).exists():
            try:info['dependencies'][key]=(p/dep_file).read_text(encoding='utf-8',errors='replace')[:1500]
            except Exception:pass
    try:
        r=subprocess.run(['git','rev-parse','--abbrev-ref','HEAD'],capture_output=True,text=True,timeout=4,cwd=str(p))
        if r.returncode==0:info['git']['branch']=r.stdout.strip()
        r2=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,timeout=4,cwd=str(p))
        if r2.returncode==0:
            ch=r2.stdout.strip().splitlines();info['git']['dirty_files']=len(ch);info['git']['changes_preview']=ch[:10]
    except Exception:pass
    return info
def _gate_diff(args,ctx,reg):return _gate_path(args,ctx,reg)
def _skill_code_diff(args,ctx,reg):
    path=args.get('path');diff=args.get('diff') or args.get('patch')
    if not path:return {'error':'missing path'}
    if not diff:return {'error':'missing diff/patch (unified diff format with @@ hunks)'}
    p=Path(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    try:src=p.read_text(encoding='utf-8')
    except Exception as e:return {'error':f'read failed: {e}'}
    new,err=_apply_unified_diff(src,diff)
    if err:return {'error':err,'path':str(p)}
    if args.get('dry_run'):return {'path':str(p),'dry_run':True,'preview_first_200':new[:200],'old_bytes':len(src),'new_bytes':len(new)}
    p.write_text(new,encoding='utf-8')
    return {'path':str(p),'old_bytes':len(src),'new_bytes':len(new),'applied':True}
def _detect_test_runner(workdir):
    p=Path(workdir)
    if (p/'Cargo.toml').exists():return ('cargo test --quiet','rust')
    if (p/'package.json').exists():
        try:
            import json as _j
            d=_j.loads((p/'package.json').read_text(encoding='utf-8'))
            if 'scripts' in d and 'test' in d['scripts']:return ('npm test --silent','js')
        except Exception:pass
    if (p/'pyproject.toml').exists() or (p/'pytest.ini').exists() or (p/'tox.ini').exists() or any(p.glob('test_*.py')) or any(p.glob('tests')):return ('pytest -x --tb=short -q','python')
    if (p/'go.mod').exists():return ('go test ./...','go')
    if (p/'Makefile').exists():
        try:
            mk=(p/'Makefile').read_text(encoding='utf-8',errors='replace')
            if re.search(r'^test\s*:',mk,re.MULTILINE):return ('make test','make')
        except Exception:pass
    return (None,None)
def _skill_test_run(args,ctx,reg):
    explicit=args.get('cmd');timeout=int(args.get('timeout',60))
    if explicit:cmd=explicit;flavor='custom'
    else:
        cmd,flavor=_detect_test_runner(reg.workdir)
        if not cmd:return {'error':'no test runner detected. Pass cmd= explicitly. Tried: Cargo.toml/package.json/pyproject.toml/go.mod/Makefile'}
    try:
        r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
        passed=r.returncode==0
        out=(r.stdout or '')[-3000:];err=(r.stderr or '')[-1500:]
        return {'cmd':cmd,'flavor':flavor,'passed':passed,'returncode':r.returncode,'stdout':out,'stderr':err}
    except subprocess.TimeoutExpired:return {'error':f'tests exceeded {timeout}s timeout','cmd':cmd,'flavor':flavor,'passed':False}
    except Exception as e:return {'error':f'{type(e).__name__}: {e}','cmd':cmd,'flavor':flavor,'passed':False}
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
    reg.register('find',_skill_find,desc=f'Fast substring/regex search across workdir text files. Skips binary + noise dirs (.git/.venv/__pycache__/etc). Args: {{query, regex?, case_sensitive?, glob?, max_hits?, max_chars?}}',schema={'query':'str','regex':'bool?','case_sensitive':'bool?','glob':'str?','max_hits':'int?','max_chars':'int?'})
    reg.register('file_write',_skill_file_write,gate=_gate_path,desc=f'Write/overwrite a UTF-8 text file within {scope}. Args: {{path, content}}',schema={'path':'str','content':'str'})
    reg.register('code_edit',_skill_code_edit,gate=_gate_code_edit,desc=f'Find-and-replace edit in a file within {scope}; .py edits ast-validated. Args: {{path, find, replace, count?}}',schema={'path':'str','find':'str','replace':'str','count':'int?'})
    reg.register('shell',_skill_shell,gate=_gate_shell,desc=f'Run a read-only allowlisted shell command from primary root. Scope: {scope}. Args: {{cmd, timeout?}}',schema={'cmd':'str','timeout':'int?'})
    reg.register('git',_skill_git,desc=f'Read-only git in {scope}. Args: {{cmd, file?, n?}}. cmd one of: status, log, diff, branch, blame, show, ls-files, remote, config, rev-parse, describe, tag, shortlog, reflog. Mutation ops (add/commit/push/etc) refused.',schema={'cmd':'str','file':'str?','n':'int?'})
    reg.register('test_run',_skill_test_run,desc=f'Auto-detect + run project tests in {scope}. Detects cargo/pytest/npm/go/make. Args: {{cmd?, timeout?}}. Returns passed:bool + stdout/stderr.',schema={'cmd':'str?','timeout':'int?'})
    reg.register('code_diff',_skill_code_diff,gate=_gate_diff,desc=f'Apply a unified diff (@@ hunks) to a file in {scope}. Safer than full-file rewrite. Args: {{path, diff, dry_run?}}.',schema={'path':'str','diff':'str','dry_run':'bool?'})
    reg.register('project_info',_skill_project_info,desc=f'Summarize the current workspace ({scope}): top files, detected languages, dependency manifests, git branch + dirty status. No args.',schema={})
    reg.register('format_code',_skill_format_code,gate=_gate_path,desc=f'Run the canonical formatter for a file in {scope} (.py:ruff/black, .rs:rustfmt, .js/.ts/.jsx/.tsx/.json/.html/.css:prettier, .go:gofmt). Args: {{path}}.',schema={'path':'str'})
    reg.register('symbols',_skill_symbols,gate=_gate_path,desc=f'Extract functions/classes/imports from a code file in {scope}. AST-based for Python; regex for Rust/JS/TS. Args: {{path}}.',schema={'path':'str'})
    reg.register('rename_symbol',_skill_rename_symbol,desc=f'Rename a symbol across all code files in {scope} (word-boundary regex). Args: {{old, new, glob?, exts?, dry_run?}}. Use dry_run first to preview.',schema={'old':'str','new':'str','glob':'str?','exts':'list?','dry_run':'bool?'})
    reg.register('auto_import',_skill_auto_import,gate=_gate_path,desc=f'Detect undefined names in a Python file and suggest stdlib imports. Args: {{path}}.',schema={'path':'str'})
    reg.register('parse_error',_skill_parse_error,desc='Parse a Python/Rust/JS/TS/Go compiler or runtime error/stack-trace. Returns {language, kind, file, line, message, likely_cause, suggested_fix}. Args: {text}.',schema={'text':'str'})
    reg.register('export_session',_skill_export_session,desc='Dump a session as markdown/text/json. Args: {session_id, out_path?, format?}. format = markdown (default) | text | json.',schema={'session_id':'str','out_path':'str?','format':'str?'})
    reg.register('prune_sessions',_skill_prune_sessions,desc='Delete old session jsonl files. Keeps N most-recent; only deletes >older_than_days old. Args: {older_than_days?=30, keep_n?=50, dry_run?=false}.',schema={'older_than_days':'int?','keep_n':'int?','dry_run':'bool?'})
    reg.register('tts',_skill_tts,desc='Text-to-speech. Returns WAV audio (base64) or writes to out_path. Backends auto-detect: piper > pyttsx3 (Windows SAPI / espeak). Args: {text, out_path?, backend?, voice?, list_voices?}.',schema={'text':'str','out_path':'str?','backend':'str?','voice':'str?','list_voices':'bool?'})
    reg.register('stt',_skill_stt,desc='Speech-to-text. Accepts path (workdir-scoped WAV) or audio_base64. Backends: faster-whisper (recommended) > vosk. Args: {path? | audio_base64?, model_size?, backend?}.',schema={'path':'str?','audio_base64':'str?','model_size':'str?','backend':'str?'})
    reg.register('run_python',_skill_run_python,desc='Execute a Python snippet in a sandboxed subprocess (workdir-confined, timeout-bounded). Rejects dangerous ops (network, fs-mutation, subprocess, exec/eval). Returns stdout/stderr/returncode. Args: {code, timeout?}',schema={'code':'str','timeout':'int?'})
    reg.register('scan',_skill_scan,gate=_gate_path,desc=f'Walk path (file or dir + glob), chunk text, teach each chunk to Adam. Args: {{path, glob?, max_files?, max_chars_per_file?, distill?, only_text?}}',schema={'path':'str','glob':'str?','max_files':'int?','max_chars_per_file':'int?','distill':'bool?','only_text':'bool?'})
    def _skill_chain(args,ctx,reg_):
        steps=args.get('steps') or [];max_steps=int(args.get('max_steps',8))
        if not isinstance(steps,list) or not steps:return {'error':'chain needs steps: list of {skill, args} dicts'}
        if len(steps)>max_steps:return {'error':f'chain capped at {max_steps} steps (got {len(steps)})'}
        results=[];prev=None;ok=True
        for i,step in enumerate(steps):
            if not isinstance(step,dict):results.append({'step':i,'error':'step must be a dict','ok':False});ok=False;break
            name=(step.get('skill') or '').strip();sargs=step.get('args') or {}
            if not name:results.append({'step':i,'error':'step missing "skill" name','ok':False});ok=False;break
            if name=='chain':results.append({'step':i,'error':'nested chain not allowed (prevent recursion)','ok':False});ok=False;break
            try:
                if isinstance(sargs,dict) and prev is not None:
                    sargs={k:(json.dumps(prev,default=str)[:4000] if v=='$prev' else (str(prev)[:4000] if v=='$prev_str' else v)) for k,v in sargs.items()}
            except Exception:pass
            r=reg_.call(name,sargs,ctx=ctx)
            d=r.to_dict() if hasattr(r,'to_dict') else {'ok':bool(r and not getattr(r,'error',None)),'output':r}
            results.append({'step':i,'skill':name,'args':sargs,**d})
            if not d.get('ok',False):
                if step.get('continue_on_error'):ok=False;continue
                else:ok=False;break
            prev=d.get('output')
        return {'ok':ok,'n_steps':len(results),'results':results,'final':results[-1].get('output') if results else None}
    reg.register('chain',_skill_chain,desc='Run a sequence of skills in order. Args: {steps:[{skill, args, continue_on_error?},...], max_steps?=8}. Use "$prev" / "$prev_str" in args to inject previous step output. Nested chain disallowed.',schema={'steps':'list','max_steps':'int?'})
    def _skill_self_inspect(args,ctx,reg_):
        """Read Adam's own source for self-reflection. Returns summary digest the LLM can reason over to identify hotspots / propose improvements."""
        from pathlib import Path as _P
        repo_root=_P(__file__).resolve().parents[2]
        subsystem=(args.get('subsystem') or 'amni/serve').strip().lstrip('/')
        max_files=int(args.get('max_files',12));max_chars=int(args.get('max_chars_per_file',2400))
        base=(repo_root/subsystem).resolve()
        try:base.relative_to(repo_root.resolve())
        except Exception:return {'error':f'subsystem must be inside repo root; got {subsystem}'}
        if not base.exists():return {'error':f'subsystem path not found: {subsystem}'}
        targets=[]
        if base.is_file():targets=[base]
        else:
            for p in sorted(base.rglob('*.py')):
                if p.name.startswith('_') and p.name!='__init__.py':continue
                if '__pycache__' in p.parts:continue
                targets.append(p)
                if len(targets)>=max_files:break
        snippets=[]
        for p in targets:
            try:
                txt=p.read_text(encoding='utf-8',errors='ignore')
                lines=txt.splitlines();line_count=len(lines)
                if len(txt)>max_chars:txt=txt[:max_chars]+f'\n... ({line_count} total lines, truncated)'
                snippets.append({'path':str(p.relative_to(repo_root)),'lines':line_count,'bytes':p.stat().st_size,'preview':txt})
            except Exception:continue
        return {'subsystem':subsystem,'files_inspected':len(snippets),'snippets':snippets,'hint':'Read the snippets, then propose specific improvements via the self_improvement skill (action=propose). Reference file paths + line numbers in your rationale.'}
    reg.register('self_inspect',_skill_self_inspect,desc='Read Adam\'s own source files for self-reflection. Args: {subsystem?=\'amni/serve\', max_files?=12, max_chars_per_file?=2400}. Scope-checked to repo root.',schema={'subsystem':'str?','max_files':'int?','max_chars_per_file':'int?'})
    def _skill_self_improvement(args,ctx,reg_):
        """Record + manage self-improvement proposals. Actions: propose | list | get | transition | stats. The actual code changes flow through file_write+code_edit (with v6.10.16 verify + v6.10.19 auto-pytest). This skill is the notebook, not the robot arm."""
        from amni.serve import self_improvement as _si
        action=(args.get('action') or 'list').strip().lower()
        if action=='propose':
            return _si.propose(title=args.get('title',''),rationale=args.get('rationale',''),planned_change=args.get('planned_change',''),files_touched=args.get('files_touched',[]),category=args.get('category','enhancement'),author=args.get('author','adam'))
        if action=='list':return {'proposals':_si.list_proposals(status=args.get('status'),category=args.get('category'),limit=int(args.get('limit',50)),include_history=bool(args.get('include_history')))}
        if action=='get':
            pid=args.get('id');
            if not pid:return {'error':'need id for action=get'}
            p=_si.get_proposal(pid);return p if p else {'error':f'no proposal {pid!r}'}
        if action=='transition':
            pid=args.get('id');new_status=args.get('status');
            if not pid or not new_status:return {'error':'need id + status for transition'}
            return _si.transition(pid,new_status,notes=args.get('notes',''),author=args.get('author','adam'))
        if action=='stats':return _si.stats()
        return {'error':f'unknown action {action!r}; valid: propose|list|get|transition|stats'}
    reg.register('self_improvement',_skill_self_improvement,desc='Record and query Adam\'s self-improvement proposals. Actions: propose (title, rationale, planned_change, files_touched?, category?) | list (status?, category?, limit?, include_history?) | get (id) | transition (id, status: proposed|attempted|validated|deployed|declined|reverted, notes?) | stats. Proposals are an append-only audit log.',schema={'action':'str','title':'str?','rationale':'str?','planned_change':'str?','files_touched':'list?','category':'str?','id':'str?','status':'str?','notes':'str?','limit':'int?','include_history':'bool?'})
    def _skill_venv(args,ctx,reg_):
        """Manage sandboxed Python virtual environments under <workdir>/.adam-venvs/. Actions: list | create | install | run | remove."""
        from amni.serve import venv_manager as _vm
        action=(args.get('action') or '').strip().lower()
        wd=reg_.workdir
        if action=='list':return {'venvs':_vm.list_venvs(wd)}
        name=(args.get('name') or '').strip()
        if action=='create':return _vm.create(wd,name)
        if action=='install':return _vm.install(wd,name,args.get('packages') or [])
        if action=='run':return _vm.run(wd,name,args.get('cmd') or '',timeout=int(args.get('timeout',300)))
        if action=='remove':return _vm.remove(wd,name)
        return {'error':f'unknown action {action!r}; valid: list|create|install|run|remove'}
    reg.register('venv',_skill_venv,desc='Sandboxed Python venv management for Adam\'s experiments. Actions: list | create (name) | install (name, packages: list) | run (name, cmd, timeout?) | remove (name). All venvs live under <workdir>/.adam-venvs/; name must match [a-z0-9_-]{1,32}; cap of 8 concurrent venvs; pip install validates package specs; run refuses obviously-destructive cmds.',schema={'action':'str','name':'str?','packages':'list?','cmd':'str?','timeout':'int?'})
    def _skill_self_reflect(args,ctx,reg_):
        """Run / inspect Adam's periodic self-reflection cycle. Actions: status | run | enable | disable."""
        from amni.serve import self_reflection as _sr
        action=(args.get('action') or 'status').strip().lower()
        if action=='status':return _sr.status()
        if action=='run':return _sr.run_cycle(force=bool(args.get('force',False)),dry_run=bool(args.get('dry_run',False)),notify=bool(args.get('notify',True)))
        if action=='enable':return _sr.set_enabled(True)
        if action=='disable':return _sr.set_enabled(False)
        return {'error':f'unknown action {action!r}; valid: status|run|enable|disable'}
    reg.register('self_reflect',_skill_self_reflect,desc='Adam\'s daily self-reflection cycle. Rotates through subsystems (amni/serve, amni/storage, amni/agent, amni/skills, amni/cli, scripts), scans for heuristic signals (TODOs, large files, missing tests, missing docstrings), drops up to 3 proposals/cycle into the self_improvement log. Actions: status | run (force?, dry_run?, notify?) | enable | disable. Cap: one cycle per ~20h unless force=True.',schema={'action':'str?','force':'bool?','dry_run':'bool?','notify':'bool?'})
    def _skill_proposal_attempt(args,ctx,reg_):
        """Auto-attempt a low-risk (category=documentation) self-improvement proposal via deterministic handler. NEVER deploys — human approval required."""
        from amni.serve import proposal_attempter as _pa
        action=(args.get('action') or 'attempt').strip().lower()
        if action=='handlers':return {'handlers':_pa.list_handlers()}
        if action=='attempt':
            pid=(args.get('id') or '').strip()
            if not pid:return {'error':'id required for action=attempt'}
            return _pa.attempt(pid,dry_run=bool(args.get('dry_run',False)),notify=bool(args.get('notify',True)))
        if action=='attempt_next':return _pa.attempt_next_eligible(max_attempts=int(args.get('max',1)),dry_run=bool(args.get('dry_run',False)))
        return {'error':f'unknown action {action!r}; valid: handlers|attempt|attempt_next'}
    reg.register('proposal_attempt',_skill_proposal_attempt,desc='Auto-attempt LOW-RISK (category=documentation) self-improvement proposals via deterministic handlers. Pipeline: backup -> apply -> ast.parse + sha256 readback + sibling pytest -> transition state. NEVER marks deployed (human approval required). Actions: handlers (list known) | attempt (id, dry_run?) | attempt_next (max?, dry_run?).',schema={'action':'str?','id':'str?','dry_run':'bool?','notify':'bool?','max':'int?'})
    def _skill_metrics_snapshot(args,ctx,reg_):
        """Adam's daily metrics snapshot for trend tracking. Actions: status | snapshot | history | trend | enable | disable."""
        from amni.serve import metrics_snapshot as _ms
        action=(args.get('action') or 'status').strip().lower()
        if action=='status':return _ms.status()
        if action=='snapshot':return _ms.snapshot(force=bool(args.get('force',False)),notify=bool(args.get('notify',False)))
        if action=='collect':return {'snapshot':_ms.collect()}
        if action=='history':return {'history':_ms.history(limit=int(args.get('limit',30)))}
        if action=='trend':return _ms.trend(days=int(args.get('days',7)))
        if action=='enable':return _ms.set_enabled(True)
        if action=='disable':return _ms.set_enabled(False)
        return {'error':f'unknown action {action!r}; valid: status|snapshot|collect|history|trend|enable|disable'}
    reg.register('metrics_snapshot',_skill_metrics_snapshot,desc='Daily metric snapshots of Adam\'s own behavior (skill latency, daemon throughput, coach streak, verification pass rate, proposal mix, queue depth). One row/day to data/metrics_snapshots.jsonl. Actions: status | snapshot (force?, notify?) | collect (read-only sample) | history (limit?) | trend (days?) | enable | disable.',schema={'action':'str?','force':'bool?','notify':'bool?','days':'int?','limit':'int?'})
    try:
        from amni.serve import widgets as _w
        def _skill_weather(args,ctx,reg_):
            d=_w.fetch_weather(location=args.get('location',''),lat=args.get('lat'),lon=args.get('lon'))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('weather',d,title=f"Weather — {d.get('location','?')}",icon='🌤')
            return d
        def _skill_system_stats(args,ctx,reg_):
            d=_w.fetch_system_stats()
            d['widget']=_w.make_widget_envelope('system',d,title='System',icon='⚙')
            return d
        def _skill_time_card(args,ctx,reg_):
            d=_w.fetch_time_card(tz_name=args.get('tz'))
            d['widget']=_w.make_widget_envelope('time',d,title='Time',icon='🕐')
            return d
        reg.register('weather',_skill_weather,desc='Current weather + forecast for a location via Open-Meteo (no API key). Emits a weather widget. Args: {location?:str, lat?:float, lon?:float}',schema={'location':'str?','lat':'float?','lon':'float?'})
        reg.register('system_stats',_skill_system_stats,desc='CPU/memory/disk/GPU snapshot via psutil + torch. Emits a system widget.',schema={})
        reg.register('time_card',_skill_time_card,desc='Time + timezone + weekday as a time widget. Args: {tz?:str like America/New_York}',schema={'tz':'str?'})
        def _skill_news(args,ctx,reg_):
            d=_w.fetch_news(query=args.get('query',''),n=int(args.get('n',6)))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('news',d,title=f"News — {d.get('query','top')}",icon='📰')
            return d
        def _skill_stock(args,ctx,reg_):
            d=_w.fetch_stock(symbols=args.get('symbols','') or args.get('symbol',''))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('stock',d,title=f"Stock — {d.get('symbols','?')}",icon='📈')
            return d
        def _skill_file_preview(args,ctx,reg_):
            d=_w.fetch_file_preview(args.get('path',''),max_lines=int(args.get('max_lines',40)),max_chars=int(args.get('max_chars',2400)))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('file',d,title=f"File — {(d.get('path') or '').split('/')[-1].split(chr(92))[-1]}",icon='📄')
            return d
        def _skill_disk(args,ctx,reg_):
            d=_w.fetch_disk()
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('disk',d,title='Disk usage',icon='💾')
            return d
        def _skill_git_status(args,ctx,reg_):
            d=_w.fetch_git_status(workdir=args.get('workdir'))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('git',d,title=f"Git — {d.get('branch','?')}",icon='⎇')
            return d
        reg.register('news',_skill_news,desc='Top news headlines for a topic via DuckDuckGo news. Emits a news widget. Args: {query?:str, n?:int=6}',schema={'query':'str?','n':'int?'})
        reg.register('stock',_skill_stock,desc='Stock quote(s) via Yahoo Finance. Emits a stock widget. Args: {symbols:str e.g. "AAPL,MSFT,GOOG"}',schema={'symbols':'str','symbol':'str?'})
        reg.register('file_preview',_skill_file_preview,gate=_gate_path,desc=f'Read first N lines of a text file in {scope} and render as a file widget. Args: {{path, max_lines?=40, max_chars?=2400}}',schema={'path':'str','max_lines':'int?','max_chars':'int?'})
        reg.register('disk_widget',_skill_disk,desc='Per-partition disk usage via psutil. Emits a disk widget.',schema={})
        reg.register('git_status',_skill_git_status,desc='git branch + dirty count + recent commits + remote + ahead/behind. Emits a git widget. Args: {workdir?:str}',schema={'workdir':'str?'})
    except Exception as _we:print(f'[skills] widget skills register failed: {_we}',flush=True)
    try:
        from amni.serve.coach import coach_skill as _coach_skill
        reg.register('coach',_coach_skill,desc='Socratic coaching/tutor mode. Actions: start <topic> [seed_question?] | ask | answer <text> | hint | skip | summary | status. Tracks per-topic mastery in coach_atlas. Args: {action, topic?, session_id?, answer?, difficulty?, seed_question?, seed_model_answer?, seed_hint?}',schema={'action':'str','topic':'str?','session_id':'str?','answer':'str?','difficulty':'int?','seed_question':'str?','seed_model_answer':'str?','seed_hint':'str?'})
    except Exception as _ce:print(f'[skills] coach skill register failed: {_ce}',flush=True)
    try:
        from amni.serve.scheduler import schedule_loop_skill as _sched_skill
        reg.register('schedule_loop',_sched_skill,desc='Adam-driven recurring jobs. Actions: add (kind, payload, cadence_s, label?, start_in_s?) | list | get <id> | cancel <id> | enable <id> | disable <id> | runs <id> | run_now <id> | stats. Kinds: skill (payload={name,args}), prompt (payload={text,system?}), webpoll (payload={url|query}). Args: {action, kind?, payload?, cadence_s?, id?, label?, start_in_s?}',schema={'action':'str','kind':'str?','payload':'dict?','cadence_s':'int?','id':'str?','label':'str?','start_in_s':'int?'})
    except Exception as _se:print(f'[skills] schedule_loop skill register failed: {_se}',flush=True)
    try:
        from amni.serve import ingest as _ingest
        _ingest.register(reg)
    except Exception as _ie:print(f'[skills] ingest skills register failed: {_ie}',flush=True)
    try:
        from amni.serve.learning_daemon import learning_daemon_skill as _ld_skill
        reg.register('learning_daemon',_ld_skill,desc='Inspect/control Adam\'s 24/7 self-improvement daemon. Actions: stats | curiosity_tick | sleep_pass | repetition_pass | pause | resume | queue_topic <topic> | atlas_verified | atlas_debated. Args: {action, topic?, limit?}',schema={'action':'str','topic':'str?','limit':'int?'})
    except Exception as _le:print(f'[skills] learning_daemon skill register failed: {_le}',flush=True)
    try:
        from amni.serve.kg_query import kg_query_skill as _kg_skill
        reg.register('kg_query',_kg_skill,desc='Query Adam\'s knowledge graph (SPO triples). Actions: stats | neighbors <subject> | out <subject> | in <subject> | predicate <p> | path <from> <to> [max_hops] | search <q> | add s,p,o | forget [s|p|o]. Args: {action, subject?, q?, predicate?, p?, a?, b?, from?, to?, max_hops?, s?, o?, object?, source?, confidence?, limit?, direction?}',schema={'action':'str','subject':'str?','q':'str?','predicate':'str?','p':'str?','s':'str?','o':'str?','a':'str?','b':'str?','from':'str?','to':'str?','max_hops':'int?','limit':'int?','direction':'str?','source':'str?','confidence':'float?'})
    except Exception as _kge:print(f'[skills] kg_query skill register failed: {_kge}',flush=True)
    try:
        from amni.serve.vision import describe_image_skill as _vis_skill
        reg.register('describe_image',_vis_skill,desc='Describe an image via BLIP. Provide image_base64 OR path. Optional question for VQA mode ("what color is the dog?"). Args: {image_base64?, path?, question?}',schema={'image_base64':'str?','path':'str?','question':'str?'})
    except Exception as _ve:print(f'[skills] describe_image skill register failed: {_ve}',flush=True)
    try:
        from amni.storage.file_watcher import watch_skill as _watch_skill
        reg.register('watch',_watch_skill,desc='File/folder change watcher. Actions: add (path, glob?, recursive?=true, label?, on_change_skill?, on_change_args?, coalesce_s?=2.0) | list | get <id> | cancel <id> | enable <id> | disable <id> | events <id> | tick | stats. Args: {action, path?, glob?, recursive?, label?, on_change_skill?, on_change_args?, coalesce_s?, id?, limit?}',schema={'action':'str','path':'str?','glob':'str?','recursive':'bool?','label':'str?','on_change_skill':'str?','on_change_args':'dict?','coalesce_s':'float?','id':'str?','limit':'int?'})
    except Exception as _we:print(f'[skills] watch skill register failed: {_we}',flush=True)
    if with_agentic:
        try:from amni.serve.agentic import register as _reg_agentic;_reg_agentic(reg)
        except Exception as e:print(f'[skills] agentic register failed: {e}',flush=True)
    return reg
