"""proposal_attempter — Adam's auto-attempt pipeline for low-risk self-improvement proposals (v6.10.59).
Step 3 of the self-improvement trajectory: ONLY operates on category='documentation' proposals whose
planned_change matches a deterministic handler. Workflow:
  1. Pick handler by matching keywords in proposal.planned_change
  2. Snapshot affected files to backups/<name>.v<ts>.bak
  3. Apply deterministic transformation
  4. Verify: ast.parse(new_source) + sha256 disk readback + sibling pytest auto-run if any
  5. Transition: proposed -> attempted, then attempted -> validated | reverted (NEVER 'deployed')
Human approval is required to promote validated -> deployed. This module DOES NOT call LLMs; every
transformation is mechanical and inspectable. Higher-risk categories (refactor, experiment, etc.) are
rejected outright."""
import ast,hashlib,json,shutil,time,re
from pathlib import Path
from typing import Dict,Any,List,Optional,Callable
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _backups_dir()->Path:
    p=_repo_root()/'backups';p.mkdir(parents=True,exist_ok=True);return p
def _sha256(text:str)->str:return hashlib.sha256(text.encode('utf-8','ignore')).hexdigest()
def _file_has_top_docstring(src:str)->bool:
    try:
        mod=ast.parse(src);body=mod.body
        if not body:return False
        first=body[0]
        return isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str)
    except SyntaxError:return False
def _docstring_for(path:Path)->str:return f'"""{path.stem} — Amni-Ai module (docstring auto-added by self-reflection cycle; replace with intent description)."""\n'
def _handle_module_docstrings(proposal:Dict[str,Any],dry_run:bool)->Dict[str,Any]:
    """Adds a single-line module docstring to flagged .py files that lack one. Idempotent."""
    files=[f for f in (proposal.get('files_touched') or []) if isinstance(f,str) and f.endswith('.py')]
    if not files:return {'ok':False,'reason':'no .py files in files_touched'}
    changes=[];errors=[];backups=[]
    for rel in files:
        p=_repo_root()/rel
        if not p.exists():errors.append({'path':rel,'reason':'file missing'});continue
        try:src=p.read_text(encoding='utf-8')
        except Exception as e:errors.append({'path':rel,'reason':f'read failed: {e}'});continue
        if _file_has_top_docstring(src):continue
        new_src=_docstring_for(p)+src
        try:ast.parse(new_src)
        except SyntaxError as e:errors.append({'path':rel,'reason':f'new source would not parse: {e}'});continue
        changes.append({'path':rel,'before_sha':_sha256(src),'after_sha':_sha256(new_src),'lines_added':1})
        if dry_run:continue
        ts=int(time.time());bak=_backups_dir()/f'{p.stem}.v{ts}.bak'
        try:shutil.copy2(p,bak);backups.append({'path':rel,'backup':str(bak.relative_to(_repo_root())).replace('\\','/')})
        except Exception as e:errors.append({'path':rel,'reason':f'backup failed: {e}'});continue
        try:p.write_text(new_src,encoding='utf-8')
        except Exception as e:errors.append({'path':rel,'reason':f'write failed: {e}'});continue
        try:
            disk=p.read_text(encoding='utf-8');disk_sha=_sha256(disk)
            if disk_sha!=_sha256(new_src):errors.append({'path':rel,'reason':'sha256 readback mismatch'})
        except Exception as e:errors.append({'path':rel,'reason':f'readback failed: {e}'})
    return {'ok':len(errors)==0 and len(changes)>0,'changes':changes,'errors':errors,'backups':backups,'handler':'module_docstrings'}
_HANDLERS:List[Dict[str,Any]]=[{'name':'module_docstrings','keywords':['module docstring','docstring','add docstring'],'category':'documentation','fn':_handle_module_docstrings}]
def _match_handler(proposal:Dict[str,Any])->Optional[Dict[str,Any]]:
    cat=(proposal.get('category') or '').lower()
    pc=(proposal.get('planned_change') or '').lower()+' '+(proposal.get('title') or '').lower()
    for h in _HANDLERS:
        if h['category']!=cat:continue
        if any(kw in pc for kw in h['keywords']):return h
    return None
def _revert_backups(backups:List[Dict[str,str]])->List[Dict[str,Any]]:
    out=[]
    for b in backups:
        try:
            src=_repo_root()/b['backup'];dst=_repo_root()/b['path']
            if src.exists():shutil.copy2(src,dst);out.append({'path':b['path'],'reverted':True})
            else:out.append({'path':b['path'],'reverted':False,'reason':'backup missing'})
        except Exception as e:out.append({'path':b['path'],'reverted':False,'reason':str(e)})
    return out
def _run_sibling_tests(touched:List[str],timeout:int=30)->Dict[str,Any]:
    try:from amni.serve.test_runner import run_sibling_pytest
    except Exception as e:return {'ran':False,'reason':f'test_runner unavailable: {e}'}
    results=[]
    for rel in touched:
        try:r=run_sibling_pytest(rel,timeout=timeout);results.append({'path':rel,**r})
        except Exception as e:results.append({'path':rel,'error':str(e)})
    all_ok=all(r.get('result')!='fail' for r in results if 'result' in r)
    return {'ran':True,'results':results,'all_ok':all_ok}
def attempt(proposal_id:str,dry_run:bool=False,notify:bool=True)->Dict[str,Any]:
    """Apply a deterministic handler for the proposal if its category+keywords match. Auto-transitions state."""
    try:from amni.serve.self_improvement import get_proposal,transition
    except Exception as e:return {'ok':False,'error':f'self_improvement import failed: {e}'}
    p=get_proposal(proposal_id)
    if p is None:return {'ok':False,'error':f'unknown proposal {proposal_id!r}'}
    if p.get('status')!='proposed':return {'ok':False,'error':f'proposal status is {p.get("status")!r}; only "proposed" can be auto-attempted'}
    h=_match_handler(p)
    if h is None:return {'ok':False,'error':'no automated handler matches this proposal','category':p.get('category'),'reason':'auto-attempt declines: requires human review'}
    if not dry_run:
        try:transition(proposal_id,'attempted',notes=f'auto-attempt via handler={h["name"]}',author='proposal_attempter')
        except Exception as e:return {'ok':False,'error':f'pre-transition failed: {e}'}
    try:res=h['fn'](p,dry_run=dry_run)
    except Exception as e:
        if not dry_run:
            try:transition(proposal_id,'reverted',notes=f'handler {h["name"]} threw: {e}',author='proposal_attempter')
            except Exception:pass
        return {'ok':False,'error':f'handler {h["name"]} raised: {type(e).__name__}: {e}'}
    if dry_run:return {'ok':res.get('ok',False),'dry_run':True,'handler':h['name'],'preview':res}
    if not res.get('ok'):
        rev=_revert_backups(res.get('backups') or [])
        try:transition(proposal_id,'reverted',notes=f'handler reported errors: {json.dumps(res.get("errors") or [])[:600]}',author='proposal_attempter')
        except Exception:pass
        return {'ok':False,'attempted':True,'reverted':True,'handler':h['name'],'reverts':rev,'errors':res.get('errors')}
    touched_paths=[c['path'] for c in (res.get('changes') or [])]
    tests=_run_sibling_tests(touched_paths) if touched_paths else {'ran':False}
    if tests.get('ran') and tests.get('all_ok') is False:
        rev=_revert_backups(res.get('backups') or [])
        try:transition(proposal_id,'reverted',notes=f'sibling tests failed: {json.dumps(tests.get("results") or [])[:600]}',author='proposal_attempter')
        except Exception:pass
        return {'ok':False,'attempted':True,'reverted':True,'handler':h['name'],'reverts':rev,'tests':tests}
    try:transition(proposal_id,'validated',notes=f'handler={h["name"]} applied {len(touched_paths)} change(s); awaiting human review for deploy',author='proposal_attempter')
    except Exception as e:return {'ok':False,'error':f'post-transition failed: {e}'}
    if notify:
        try:from amni.serve.notifications import push_notification
        except Exception:push_notification=None
        if push_notification:
            try:push_notification(kind='proposal_validated',title=f'Auto-attempted proposal validated',body=f'{h["name"]} applied {len(touched_paths)} change(s) — review for deploy approval',meta={'proposal_id':proposal_id,'files':touched_paths,'tests':tests})
            except Exception:pass
    return {'ok':True,'attempted':True,'validated':True,'handler':h['name'],'changes':res.get('changes'),'tests':tests,'backups':res.get('backups'),'awaiting':'human approval to mark deployed'}
def attempt_next_eligible(max_attempts:int=1,dry_run:bool=False)->Dict[str,Any]:
    """Walks the proposal log for category=documentation proposals in 'proposed' state with a matching handler."""
    try:from amni.serve.self_improvement import list_proposals
    except Exception as e:return {'ok':False,'error':f'self_improvement import failed: {e}'}
    pending=list_proposals(status='proposed',category='documentation',limit=20)
    attempted=[];skipped=[]
    for p in pending:
        if _match_handler(p) is None:skipped.append({'id':p.get('id'),'reason':'no handler'});continue
        r=attempt(p.get('id'),dry_run=dry_run);attempted.append({'id':p.get('id'),'result':r})
        if len(attempted)>=max_attempts:break
    return {'ok':True,'n_attempted':len(attempted),'n_skipped':len(skipped),'attempted':attempted,'skipped':skipped,'dry_run':dry_run}
def list_handlers()->List[Dict[str,Any]]:return [{'name':h['name'],'category':h['category'],'keywords':h['keywords']} for h in _HANDLERS]
