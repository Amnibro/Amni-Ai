"""self_reflection — Adam's daily self-improvement reflection cycle (v6.10.58).
Step 2 of the self-improvement trajectory after v6.10.56 scaffold + v6.10.57 sandboxes:
  - Rotates through subsystems (amni/serve, amni/storage, amni/agent, amni/skills, amni/cli, scripts)
  - Once per ~24h (cap), scans the next subsystem for heuristic improvement signals
  - Drops 1-3 proposals via self_improvement.propose() (capped daily so the log doesn't get spammed)
  - Emits a notification toast for human review
  - State persisted to data/self_reflection_state.json
Intentionally heuristic, NOT LLM-driven. Adam reads its own code via signals (TODOs, missing
docstrings, missing sibling tests, large modules) and proposes. The HUMAN decides what to attempt."""
import json,re,time,os
from pathlib import Path
from typing import Dict,Any,List,Optional
_SUBSYSTEMS=['amni/serve','amni/storage','amni/agent','amni/skills','amni/cli','scripts']
_MIN_INTERVAL_S=20*3600
_MAX_PROPOSALS_PER_CYCLE=3
_LARGE_FILE_LINES=600
_TODO_RE=re.compile(r'(?:^|\s)(?:#|//)\s*(TODO|FIXME|XXX|HACK)\b',re.IGNORECASE)
def _repo_root()->Path:return Path(__file__).resolve().parents[2]
def _data_dir()->Path:
    p=_repo_root()/'data';p.mkdir(parents=True,exist_ok=True);return p
def _state_path()->Path:return _data_dir()/'self_reflection_state.json'
def _load_state()->Dict[str,Any]:
    p=_state_path()
    if not p.exists():return {'last_run_ts':0.0,'last_subsystem':None,'cycle_count':0,'enabled':True,'history':[]}
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return {'last_run_ts':0.0,'last_subsystem':None,'cycle_count':0,'enabled':True,'history':[]}
def _save_state(state:Dict[str,Any])->None:
    try:_state_path().write_text(json.dumps(state,indent=2,default=str),encoding='utf-8')
    except Exception as e:print(f'[self_reflection] state save failed: {e}',flush=True)
def _pick_next_subsystem(state:Dict[str,Any])->str:
    last=state.get('last_subsystem')
    if last not in _SUBSYSTEMS:return _SUBSYSTEMS[0]
    return _SUBSYSTEMS[(_SUBSYSTEMS.index(last)+1)%len(_SUBSYSTEMS)]
def _scan_subsystem(rel_path:str)->Dict[str,Any]:
    root=_repo_root()/rel_path
    if not root.exists():return {'subsystem':rel_path,'exists':False,'files':[],'todos':[],'large':[],'no_docstring':[],'no_sibling_test':[]}
    files=[];todos=[];large=[];no_docstring=[];no_sibling_test=[]
    for p in sorted(root.rglob('*.py')):
        rel=p.relative_to(_repo_root());name=p.name
        if name.startswith('_') and name!='__init__.py':continue
        if '__pycache__' in p.parts:continue
        try:src=p.read_text(encoding='utf-8',errors='ignore')
        except Exception:continue
        lines=src.split('\n');nlines=len(lines)
        files.append({'path':str(rel).replace(os.sep,'/'),'lines':nlines})
        td=sum(1 for ln in lines if _TODO_RE.search(ln))
        if td>0:todos.append({'path':str(rel).replace(os.sep,'/'),'count':td})
        if nlines>=_LARGE_FILE_LINES:large.append({'path':str(rel).replace(os.sep,'/'),'lines':nlines})
        stripped=src.lstrip()
        if not (stripped.startswith('"""') or stripped.startswith("'''")):
            if name!='__init__.py' and nlines>30:no_docstring.append({'path':str(rel).replace(os.sep,'/'),'lines':nlines})
        if name not in ('__init__.py',) and not name.startswith('test_'):
            sibling=_repo_root()/'tests'/f'test_{p.stem}.py'
            globbed=list((_repo_root()/'tests').glob(f'test_{p.stem}_*.py')) if (_repo_root()/'tests').exists() else []
            if not sibling.exists() and not globbed and nlines>50:
                no_sibling_test.append({'path':str(rel).replace(os.sep,'/'),'lines':nlines})
    return {'subsystem':rel_path,'exists':True,'files':files[:200],'todos':todos[:50],'large':large[:20],'no_docstring':no_docstring[:30],'no_sibling_test':no_sibling_test[:50]}
def _draft_proposals(scan:Dict[str,Any])->List[Dict[str,Any]]:
    drafts=[];sub=scan.get('subsystem','?')
    todos=scan.get('todos') or []
    if todos:
        top=sorted(todos,key=lambda t:-int(t.get('count') or 0))[:5]
        files=[t['path'] for t in top]
        drafts.append({'title':f'Address open TODO/FIXME markers in {sub}','rationale':f"Self-reflection found {sum(t['count'] for t in todos)} TODO/FIXME/XXX/HACK markers across {len(todos)} files in {sub}. Top offenders: {', '.join(files)}. These accumulate and signal deferred decisions; cleaning them up reduces unknowns about subsystem health.",'planned_change':'Read each flagged file, decide per marker: (a) implement now, (b) convert to a self_improvement proposal with concrete scope, (c) delete if obsolete. Prefer (a) for trivial cases, (b) for non-trivial scope.','files_touched':files,'category':'refactor'})
    large=scan.get('large') or []
    if large:
        top=sorted(large,key=lambda t:-int(t.get('lines') or 0))[:3]
        files=[t['path'] for t in top]
        top_str=', '.join('{} ({} lines)'.format(t['path'],t['lines']) for t in top)
        drafts.append({'title':f'Investigate splitting large modules in {sub}','rationale':f"Self-reflection found {len(large)} files at >={_LARGE_FILE_LINES} lines in {sub}. Largest: {top_str}. Big modules become harder to reason about; consider whether a cohesive subset can be extracted.",'planned_change':'For each large file, identify cohesive groupings (e.g. separate classes, related helpers). Extract to a sibling module only if the grouping is genuinely standalone — avoid mechanical splits that just shuffle imports.','files_touched':files,'category':'refactor'})
    nst=scan.get('no_sibling_test') or []
    if nst:
        top=sorted(nst,key=lambda t:-int(t.get('lines') or 0))[:5]
        files=[t['path'] for t in top]
        drafts.append({'title':f'Add tests for under-tested modules in {sub}','rationale':f"Self-reflection found {len(nst)} substantial files in {sub} with no sibling test_*.py. Top by size: {', '.join(files[:3])}. Untested code is a verification gap — v6.10.19 auto-pytest can\\'t protect what doesn\\'t exist.",'planned_change':'Prioritize by recent-edit frequency (git log --since=30d). For each, write at least one happy-path + one boundary-condition test. Reuse existing test patterns from tests/test_*_v6_*.py.','files_touched':files,'category':'experiment'})
    nds=scan.get('no_docstring') or []
    if nds:
        top=sorted(nds,key=lambda t:-int(t.get('lines') or 0))[:5]
        files=[t['path'] for t in top]
        drafts.append({'title':f'Add module docstrings in {sub}','rationale':f"Self-reflection found {len(nds)} files in {sub} with no top-of-module docstring. A 1-2 sentence purpose line helps future readers (and Adam itself) place the module in context.",'planned_change':'For each flagged file, add a single-line """summary""" describing what the module exists for. No multi-paragraph rambling — purpose + role only.','files_touched':files,'category':'documentation'})
    if not drafts:drafts.append({'title':f'{sub} looks clean — no immediate signals','rationale':f'Self-reflection scan of {sub} found no TODOs, no oversize files, no untested modules, no missing docstrings. Recording this for trend-tracking.','planned_change':'No code change. This proposal is a heartbeat for trend visibility; mark "declined" with note "clean cycle" to close.','files_touched':[],'category':'documentation'})
    return drafts[:_MAX_PROPOSALS_PER_CYCLE]
def status()->Dict[str,Any]:
    s=_load_state();now=time.time();last=float(s.get('last_run_ts') or 0)
    return {'enabled':bool(s.get('enabled',True)),'last_run_ts':last,'last_run_iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(last)) if last else '','last_subsystem':s.get('last_subsystem'),'next_subsystem':_pick_next_subsystem(s),'cycle_count':int(s.get('cycle_count') or 0),'seconds_until_eligible':max(0,int(_MIN_INTERVAL_S-(now-last))),'min_interval_s':_MIN_INTERVAL_S,'subsystems':list(_SUBSYSTEMS),'recent':list(s.get('history') or [])[-10:]}
def should_run_now(force:bool=False)->bool:
    s=_load_state()
    if not s.get('enabled',True) and not force:return False
    if force:return True
    return (time.time()-float(s.get('last_run_ts') or 0))>=_MIN_INTERVAL_S
def run_cycle(force:bool=False,dry_run:bool=False,notify:bool=True)->Dict[str,Any]:
    if not should_run_now(force=force):
        st=status();return {'ran':False,'reason':f"too soon (next eligible in {st['seconds_until_eligible']}s)",'status':st}
    s=_load_state();sub=_pick_next_subsystem(s)
    scan=_scan_subsystem(sub);drafts=_draft_proposals(scan)
    proposed_ids=[]
    if not dry_run:
        try:from amni.serve.self_improvement import propose as _propose
        except Exception as e:return {'ran':False,'error':f'self_improvement import failed: {e}'}
        for d in drafts:
            try:
                rec=_propose(title=d['title'],rationale=d['rationale'],planned_change=d['planned_change'],files_touched=d.get('files_touched') or [],category=d.get('category','enhancement'),author='self-reflection')
                if rec and rec.get('id'):proposed_ids.append(rec['id'])
            except Exception as e:print(f'[self_reflection] propose failed: {e}',flush=True)
    if not dry_run:
        s['last_run_ts']=time.time();s['last_subsystem']=sub;s['cycle_count']=int(s.get('cycle_count') or 0)+1
        hist=list(s.get('history') or []);hist.append({'ts':s['last_run_ts'],'subsystem':sub,'n_proposals':len(proposed_ids),'proposal_ids':proposed_ids});s['history']=hist[-50:]
        _save_state(s)
    if notify and not dry_run and proposed_ids:
        try:from amni.serve.notifications import push_notification
        except Exception:push_notification=None
        if push_notification:
            try:push_notification(kind='self_reflection',title=f'Adam reflected on {sub}',body=f'{len(proposed_ids)} new self-improvement proposal{("s" if len(proposed_ids)!=1 else "")} dropped for review',meta={'subsystem':sub,'proposal_ids':proposed_ids,'cycle':s['cycle_count']})
            except Exception as e:print(f'[self_reflection] notify failed: {e}',flush=True)
    return {'ran':True,'dry_run':dry_run,'subsystem':sub,'scan_summary':{'files':len(scan.get('files') or []),'todos':len(scan.get('todos') or []),'large':len(scan.get('large') or []),'no_docstring':len(scan.get('no_docstring') or []),'no_sibling_test':len(scan.get('no_sibling_test') or [])},'drafts':drafts,'proposed_ids':proposed_ids,'next_subsystem':_pick_next_subsystem({'last_subsystem':sub})}
def set_enabled(enabled:bool)->Dict[str,Any]:
    s=_load_state();s['enabled']=bool(enabled);_save_state(s);return status()
