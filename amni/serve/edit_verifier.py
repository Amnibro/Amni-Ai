"""edit_verifier — verify that file_write / code_edit changes are actually functional.
Per-extension checks (Python ast+compile, JSON parse, YAML/TOML parse, HTML/JS/CSS bracket balance).
Also verifies disk content matches what we wrote (sha + len) — catches IO corruption.
When auto-verification can't be done (no extension match) OR user testing is genuinely needed (UI, network, side effects), appends a note to needs_testing.jsonl so the user can pick it up via GET /memory/needs-testing."""
import json,ast,hashlib,time,re
from pathlib import Path
from typing import Dict,Any,List,Optional
def _data_dir()->Path:
    base=Path(__file__).resolve().parents[2]/'data';base.mkdir(parents=True,exist_ok=True);return base
def _needs_testing_path()->Path:return _data_dir()/'needs_testing.jsonl'
def _verification_log_path()->Path:return _data_dir()/'verification_log.jsonl'
def _bracket_balance(text:str,pairs=('()','[]','{}'))->List[str]:
    issues=[];text_no_strings=re.sub(r"(?:'[^'\n]*'|\"[^\"\n]*\"|`[^`]*`)",'""',text);text_no_comments=re.sub(r"//[^\n]*|/\*[\s\S]*?\*/|<!--[\s\S]*?-->|#[^\n]*",'',text_no_strings)
    for o,cl in pairs:
        no=text_no_comments.count(o);nc=text_no_comments.count(cl)
        if no!=nc:issues.append(f'unbalanced {o}{cl}: {no} open vs {nc} close')
    return issues
def _verify_py(path:Path,content:str)->Dict[str,Any]:
    issues=[]
    try:ast.parse(content)
    except SyntaxError as e:issues.append(f'syntax error: line {e.lineno}: {e.msg}')
    try:compile(content,str(path),'exec')
    except Exception as e:
        msg=f'{type(e).__name__}: {e}';
        if msg not in issues:issues.append('compile failed: '+msg)
    sibling_test=path.parent/f'test_{path.stem}.py';tests_dir_test=path.parent.parent/'tests'/f'test_{path.stem}.py'
    suggested_tests=[str(p) for p in (sibling_test,tests_dir_test) if p.exists()]
    return {'checks':['ast.parse','compile'],'issues':issues,'suggested_tests':suggested_tests,'verified':len(issues)==0}
def _verify_json(path:Path,content:str)->Dict[str,Any]:
    try:json.loads(content);return {'checks':['json.loads'],'issues':[],'verified':True}
    except json.JSONDecodeError as e:return {'checks':['json.loads'],'issues':[f'invalid JSON at line {e.lineno} col {e.colno}: {e.msg}'],'verified':False}
def _verify_yaml(path:Path,content:str)->Dict[str,Any]:
    try:
        import yaml;yaml.safe_load(content);return {'checks':['yaml.safe_load'],'issues':[],'verified':True}
    except ImportError:return {'checks':[],'issues':[],'verified':None,'reason':'PyYAML not installed; cannot verify'}
    except Exception as e:return {'checks':['yaml.safe_load'],'issues':[f'yaml parse: {e}'],'verified':False}
def _verify_toml(path:Path,content:str)->Dict[str,Any]:
    try:
        try:import tomllib as _t
        except ImportError:import tomli as _t
        _t.loads(content);return {'checks':['tomllib.loads'],'issues':[],'verified':True}
    except Exception as e:return {'checks':['tomllib.loads'],'issues':[f'toml parse: {e}'],'verified':False}
def _verify_bracket_lang(path:Path,content:str,lang:str)->Dict[str,Any]:
    issues=_bracket_balance(content)
    return {'checks':['bracket_balance'],'issues':issues,'verified':len(issues)==0,'language':lang}
def _verify_disk(path:Path,expected:str)->Dict[str,Any]:
    try:actual=path.read_text(encoding='utf-8')
    except Exception as e:return {'checks':['disk_readback'],'issues':[f'disk readback failed: {e}'],'verified':False}
    if len(actual)!=len(expected):return {'checks':['disk_readback'],'issues':[f'size mismatch: wrote {len(expected)}b, disk has {len(actual)}b'],'verified':False}
    h_exp=hashlib.sha256(expected.encode('utf-8')).hexdigest()[:16];h_act=hashlib.sha256(actual.encode('utf-8')).hexdigest()[:16]
    if h_exp!=h_act:return {'checks':['disk_readback'],'issues':[f'sha256 mismatch: expected {h_exp}, disk {h_act}'],'verified':False}
    return {'checks':['disk_readback'],'issues':[],'verified':True}
_VERIFIERS={'py':_verify_py,'json':_verify_json,'yaml':_verify_yaml,'yml':_verify_yaml,'toml':_verify_toml,'js':lambda p,c:_verify_bracket_lang(p,c,'javascript'),'mjs':lambda p,c:_verify_bracket_lang(p,c,'javascript'),'ts':lambda p,c:_verify_bracket_lang(p,c,'typescript'),'tsx':lambda p,c:_verify_bracket_lang(p,c,'typescript'),'jsx':lambda p,c:_verify_bracket_lang(p,c,'javascript'),'html':lambda p,c:_verify_bracket_lang(p,c,'html'),'htm':lambda p,c:_verify_bracket_lang(p,c,'html'),'css':lambda p,c:_verify_bracket_lang(p,c,'css'),'scss':lambda p,c:_verify_bracket_lang(p,c,'css'),'rs':lambda p,c:_verify_bracket_lang(p,c,'rust'),'go':lambda p,c:_verify_bracket_lang(p,c,'go'),'c':lambda p,c:_verify_bracket_lang(p,c,'c'),'cpp':lambda p,c:_verify_bracket_lang(p,c,'cpp'),'h':lambda p,c:_verify_bracket_lang(p,c,'c'),'hpp':lambda p,c:_verify_bracket_lang(p,c,'cpp')}
_NEEDS_USER_TEST_HINTS={'html':'visual UI render','htm':'visual UI render','css':'visual style render','js':'browser runtime behavior','mjs':'browser runtime behavior','jsx':'react render','tsx':'react render'}
def _log_verification(path:str,result:Dict[str,Any]):
    try:
        entry={'ts':time.time(),'path':path,'verified':result.get('verified'),'issues':result.get('issues',[]),'checks':result.get('checks',[])}
        with open(_verification_log_path(),'a',encoding='utf-8') as f:f.write(json.dumps(entry,default=str)+'\n')
    except Exception:pass
def _queue_needs_testing(path:str,reason:str,checks_done:Optional[List[str]]=None,op:str='edit'):
    try:
        entry={'ts':time.time(),'path':path,'op':op,'reason':reason,'checks_already_done':checks_done or [],'status':'pending'}
        with open(_needs_testing_path(),'a',encoding='utf-8') as f:f.write(json.dumps(entry,default=str)+'\n')
    except Exception:pass
def verify_edit(path:str,written_content:str,op:str='edit')->Dict[str,Any]:
    p=Path(path);ext=p.suffix.lstrip('.').lower()
    result={'path':path,'ext':ext,'op':op}
    disk=_verify_disk(p,written_content)
    result['disk']=disk
    if not disk.get('verified',True):
        result['verified']=False;result['issues']=disk.get('issues',[]);_log_verification(path,result);return result
    verifier=_VERIFIERS.get(ext)
    if verifier is None:
        result['verified']=None;result['reason']=f'no automatic verifier for .{ext}';_queue_needs_testing(path,reason=f'no auto-verifier for .{ext} files',op=op);_log_verification(path,result);return result
    semantic=verifier(p,written_content)
    result.update(semantic)
    if ext in _NEEDS_USER_TEST_HINTS:_queue_needs_testing(path,reason=f'{_NEEDS_USER_TEST_HINTS[ext]} needs human eyes',checks_done=semantic.get('checks',[]),op=op)
    elif semantic.get('suggested_tests'):_queue_needs_testing(path,reason='associated test file detected — please run',checks_done=semantic.get('checks',[])+[f'has_tests:{semantic.get("suggested_tests")}'],op=op)
    _log_verification(path,result);return result
def list_needs_testing(limit:int=50,include_done:bool=False)->List[Dict[str,Any]]:
    p=_needs_testing_path()
    if not p.exists():return []
    out=[]
    try:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            try:obj=json.loads(line)
            except Exception:continue
            if not include_done and obj.get('status')=='done':continue
            out.append(obj)
    except Exception:return []
    return out[-limit:][::-1]
def mark_needs_testing_done(path_substring:str)->int:
    p=_needs_testing_path()
    if not p.exists():return 0
    try:lines=p.read_text(encoding='utf-8').splitlines()
    except Exception:return 0
    n=0;out=[]
    for line in lines:
        line=line.strip()
        if not line:continue
        try:obj=json.loads(line)
        except Exception:out.append(line);continue
        if path_substring in obj.get('path','') and obj.get('status')=='pending':
            obj['status']='done';obj['done_ts']=time.time();n+=1
        out.append(json.dumps(obj,default=str))
    p.write_text('\n'.join(out)+'\n',encoding='utf-8')
    return n
