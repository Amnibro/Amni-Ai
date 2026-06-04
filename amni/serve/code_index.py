"""code_index — train Adam on a codebase: walk a tree, extract per-file language + symbols + summary into a
queryable map so Adam knows WHERE everything is before it edits. Foundation for Adam-as-software-engineer:
locate (this) -> understand (project_info/scan) -> edit (code_edit) -> verify (test_run/edit_verifier) -> iterate.
Index persists to experiences/code_index.json (local). Symbol extraction is regex-based per language — fast, dependency-free, good enough to route the coding loop to the right file."""
import re,json,time,os
from pathlib import Path
from typing import Dict,Any,List,Optional
_EXCLUDE_DIRS={'.git','node_modules','__pycache__','.venv','venv','env','backups','data','models','downloaded_models','bakes','logs','ptex_hf','hf_cache','textures','checkpoints','eval_reports','dist','build','.mypy_cache','.pytest_cache','target','.idea','.vscode','site-packages','.next','out','coverage','__snapshots__'}
_LANG={'.py':'python','.js':'javascript','.jsx':'javascript','.ts':'typescript','.tsx':'typescript','.rs':'rust','.go':'go','.java':'java','.c':'c','.h':'c','.cpp':'cpp','.hpp':'cpp','.cs':'csharp','.rb':'ruby','.php':'php','.swift':'swift','.kt':'kotlin','.scala':'scala','.sh':'shell','.html':'html','.css':'css','.sql':'sql'}
_M=re.MULTILINE
_SYM={
 'python':[re.compile(r'^\s*(?:async\s+)?def\s+(\w+)',_M),re.compile(r'^\s*class\s+(\w+)',_M)],
 'javascript':[re.compile(r'(?:^|\s)function\s+(\w+)',_M),re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function|\()',_M),re.compile(r'(?:^|\s)class\s+(\w+)',_M)],
 'typescript':[re.compile(r'(?:^|\s)function\s+(\w+)',_M),re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function|\()',_M),re.compile(r'(?:^|\s)(?:class|interface|type|enum)\s+(\w+)',_M)],
 'rust':[re.compile(r'(?:^|\s)(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',_M),re.compile(r'(?:^|\s)(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)',_M)],
 'go':[re.compile(r'^\s*func\s+(?:\([^)]*\)\s*)?(\w+)',_M),re.compile(r'^\s*type\s+(\w+)\s+(?:struct|interface)',_M)],
 'java':[re.compile(r'(?:public|private|protected|static|\s)+(?:class|interface|enum)\s+(\w+)',_M),re.compile(r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(',_M)],
 'cpp':[re.compile(r'(?:^|\s)(?:class|struct)\s+(\w+)',_M),re.compile(r'(?:^|\s)[\w:<>*&]+\s+(\w+)\s*\(',_M)],
 'c':[re.compile(r'(?:^|\s)[\w*]+\s+(\w+)\s*\(',_M)],
}
_aliases={'jsx':'javascript','tsx':'typescript','hpp':'cpp','h':'c'}
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _index_path()->Path:
    p=_repo_root()/'experiences'/'code_index.json';p.parent.mkdir(parents=True,exist_ok=True);return p
def _ptex_path()->Path:
    p=_repo_root()/'experiences'/'code_map_ptex';p.parent.mkdir(parents=True,exist_ok=True);return p
def _build_ptex_map(files:Dict[str,Any],encoder=None,lut=None)->Dict[str,Any]:
    """The map IS ptex: each file -> a SemanticPTEXLUT cell keyed by its descriptor, so semantic 'where is X' queries
    land on the right file by Reffelt-style cell address. Best-effort; needs an encoder (Adam's MiniLM) — skipped cleanly otherwise."""
    if lut is None:
        if encoder is None:return {'ptex_built':False,'reason':'no encoder (boot Adam to build the PTEX map)'}
        try:
            from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
            lut=SemanticPTEXLUT(grid=64,pca_dim=8,encoder=encoder)
        except Exception as e:return {'ptex_built':False,'reason':f'lut unavailable: {e}'}
    n=0
    for path,info in files.items():
        desc=f"{path} :: {info.get('summary','')} :: {','.join((info.get('symbols') or [])[:20])}"
        try:lut.add(desc,path);n+=1
        except Exception:pass
    if n==0:return {'ptex_built':False,'reason':'no files'}
    try:lut.fit();lut.save(str(_ptex_path()))
    except Exception as e:return {'ptex_built':False,'reason':f'fit/save failed: {e}','attempted':n}
    return {'ptex_built':True,'ptex_path':str(_ptex_path()),'cells':n}
def semantic_query(q:str,encoder=None,k:int=5)->Dict[str,Any]:
    try:from amni.inference.semantic_ptex_lut import SemanticPTEXLUT
    except Exception as e:return {'error':f'lut unavailable: {e}'}
    p=_ptex_path()
    if not Path(str(p)+'.json').exists():return {'error':'no ptex map yet — run build with an encoder'}
    try:
        lut=SemanticPTEXLUT.load(str(p),encoder=encoder)
        hit=lut.lookup_soft(q,k=k,margin='auto')
    except Exception as e:return {'error':f'semantic query failed: {e}'}
    return {'query':q,'path':hit}
def _summary(text:str,lang:str)->str:
    for ln in text.splitlines()[:40]:
        s=ln.strip()
        if not s:continue
        if s.startswith('"""') or s.startswith("'''"):return s.strip('"\' ')[:160]
        if s.startswith('//') or s.startswith('#') or s.startswith('/*') or s.startswith('*'):
            c=s.lstrip('/#*  ').strip()
            if len(c)>8:return c[:160]
    return ''
def _extract_symbols(text:str,lang:str,cap:int=80):
    pats=_SYM.get(lang) or []
    out=[];seen=set();lines={}
    for pat in pats:
        for m in pat.finditer(text):
            sym=m.group(1)
            if sym and sym not in seen and not sym.startswith('_'):
                seen.add(sym);out.append(sym);lines[sym]=text.count('\n',0,m.start())+1
                if len(out)>=cap:return out,lines
    return out,lines
def build_index(root:Optional[str]=None,max_files:int=6000,max_bytes:int=500000,ptex:bool=True,encoder=None)->Dict[str,Any]:
    base=Path(root).expanduser() if root else _repo_root()
    if not base.exists():return {'error':f'root not found: {base}'}
    files={};langs={};n_sym=0;scanned=0;skipped=0;t0=time.time()
    for dirpath,dirnames,filenames in os.walk(base):
        dirnames[:]=[d for d in dirnames if d not in _EXCLUDE_DIRS and not d.startswith('.')]
        for fn in filenames:
            ext=os.path.splitext(fn)[1].lower()
            lang=_LANG.get(ext)
            if lang is None:continue
            fp=Path(dirpath)/fn
            try:
                if fp.stat().st_size>max_bytes:skipped+=1;continue
                text=fp.read_text(encoding='utf-8',errors='ignore')
            except Exception:skipped+=1;continue
            rel=(str(fp.relative_to(base)) if str(fp).startswith(str(base)) else str(fp)).replace('\\','/')
            lang=_aliases.get(lang,lang)
            syms,slines=_extract_symbols(text,lang)
            files[rel]={'lang':lang,'lines':text.count('\n')+1,'symbols':syms,'sym_lines':slines,'summary':_summary(text,lang),'bytes':fp.stat().st_size}
            langs[lang]=langs.get(lang,0)+1;n_sym+=len(syms);scanned+=1
            if scanned>=max_files:break
        if scanned>=max_files:break
    idx={'root':str(base),'built_at':time.time(),'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'n_files':len(files),'n_symbols':n_sym,'languages':dict(sorted(langs.items(),key=lambda kv:-kv[1])),'skipped':skipped,'build_s':round(time.time()-t0,2),'files':files}
    try:_index_path().write_text(json.dumps(idx),encoding='utf-8')
    except Exception as e:return {**{k:idx[k] for k in idx if k!='files'},'save_error':str(e)}
    out={k:idx[k] for k in idx if k!='files'}
    if ptex:out['ptex']=_build_ptex_map(files,encoder=encoder)
    return out
def _load()->Optional[Dict[str,Any]]:
    p=_index_path()
    if not p.exists():return None
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return None
def query(term:str,limit:int=25)->Dict[str,Any]:
    idx=_load()
    if idx is None:return {'error':'no index yet — run build first','built':False}
    term=(term or '').strip()
    if not term:return {'error':'query required'}
    tl=term.lower();sym_hits=[];file_hits=[]
    for path,info in (idx.get('files') or {}).items():
        if tl in path.lower():file_hits.append({'path':path,'lang':info.get('lang'),'lines':info.get('lines'),'summary':info.get('summary','')})
        for s in info.get('symbols') or []:
            if tl in s.lower():sym_hits.append({'symbol':s,'path':path,'line':(info.get('sym_lines') or {}).get(s),'lang':info.get('lang')})
    sym_hits.sort(key=lambda h:(h['symbol'].lower()!=tl,len(h['symbol'])))
    return {'query':term,'symbols':sym_hits[:limit],'files':file_hits[:limit],'n_symbol_hits':len(sym_hits),'n_file_hits':len(file_hits),'root':idx.get('root')}
def file_info(path:str)->Dict[str,Any]:
    idx=_load()
    if idx is None:return {'error':'no index yet'}
    files=idx.get('files') or {}
    if path in files:return {'path':path,**files[path]}
    matches=[p for p in files if p.endswith(path) or path in p]
    if len(matches)==1:return {'path':matches[0],**files[matches[0]]}
    return {'error':f'no exact match for {path!r}','candidates':matches[:10]}
def stats()->Dict[str,Any]:
    idx=_load()
    if idx is None:return {'built':False}
    return {'built':True,'root':idx.get('root'),'iso':idx.get('iso'),'n_files':idx.get('n_files'),'n_symbols':idx.get('n_symbols'),'languages':idx.get('languages'),'skipped':idx.get('skipped'),'build_s':idx.get('build_s')}
