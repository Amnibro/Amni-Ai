"""Pre-push lint — fails if any TRACKED .py file imports a module that resolves to an UNTRACKED file on disk.
Catches the ghost-import class of bug (e.g. the v6.9.4 incident where amni/storage/{conversation_notes,local_profile}.py existed on the publisher machine but were never `git add`-ed, so every fresh clone exploded on `python -m amni.cli serve`).
Usage:
  python scripts/check_publish_health.py             # exit 0 if clean, 1 if ghost imports found
  python scripts/check_publish_health.py --fix       # `git add` any ghost-imported file that exists locally
Install as a git pre-push hook by linking it:
  cp scripts/check_publish_health.py .git/hooks/pre-push   # POSIX
  copy scripts/check_publish_health.py .git/hooks/pre-push  # Windows (forward slashes work in PowerShell/CMD)"""
import os,sys,ast,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PKG='amni'
def _tracked()->set:
    r=subprocess.run(['git','-C',str(ROOT),'ls-files','*.py'],capture_output=True,text=True,timeout=20)
    return {p.strip() for p in r.stdout.splitlines() if p.strip()}
def _resolve_import(dotted:str)->list:
    parts=dotted.split('.')
    if not parts or parts[0]!=PKG:return []
    head='/'.join(parts)
    return [f'{head}.py',f'{head}/__init__.py']
def _resolves_to_tracked(dotted:str,tracked:set)->tuple:
    cands=_resolve_import(dotted)
    for c in cands:
        if c in tracked:return (True,c,True)
    for c in cands:
        if (ROOT/c).exists():return (True,c,False)
    return (False,cands[0] if cands else dotted,False)
def _imports(py_path:Path)->list:
    out=[]
    try:tree=ast.parse(py_path.read_text(encoding='utf-8',errors='ignore'))
    except Exception:return out
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            mod=node.module or ''
            if node.level!=0:continue
            if mod==PKG:
                for alias in node.names:out.append((f'{PKG}.{alias.name}',node.lineno))
            elif mod.startswith(PKG+'.'):
                out.append((mod,node.lineno))
                for alias in node.names:
                    nm=alias.name
                    if nm and nm!='*':out.append((f'{mod}.{nm}',node.lineno))
        elif isinstance(node,ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG+'.') or alias.name==PKG:out.append((alias.name,node.lineno))
    return out
def _is_submodule_path(dotted:str)->bool:
    cands=_resolve_import(dotted)
    return any((ROOT/c).exists() for c in cands)
def scan(do_fix:bool=False)->int:
    tracked=_tracked()
    ghosts=[];unresolved=[]
    for rel in sorted(tracked):
        if not rel.endswith('.py'):continue
        for dotted,lineno in _imports(ROOT/rel):
            ok,resolved,is_tracked=_resolves_to_tracked(dotted,tracked)
            if ok and not is_tracked:ghosts.append((rel,lineno,dotted,resolved))
            elif not ok and _is_submodule_path(dotted):ghosts.append((rel,lineno,dotted,_resolve_import(dotted)[0]))
    if not ghosts and not unresolved:
        print(f'[publish_health] OK — {len(tracked)} tracked .py files, all amni.* imports resolve to tracked files.');return 0
    if ghosts:
        print(f'[publish_health] FAIL — {len(ghosts)} ghost import(s): tracked code imports modules that exist locally but are NOT in git:')
        for rel,lineno,dotted,resolved in ghosts:print(f'  {rel}:{lineno}  import {dotted}  -> {resolved} (UNTRACKED)')
    if unresolved:
        print(f'[publish_health] FAIL — {len(unresolved)} unresolved import(s): tracked code imports modules that do NOT exist on disk:')
        for rel,lineno,dotted in unresolved:print(f'  {rel}:{lineno}  import {dotted}  -> NO MATCHING FILE')
    if do_fix and ghosts:
        print('\n[publish_health] --fix: staging ghost-imported files for commit')
        added=set()
        for _,_,_,resolved in ghosts:
            if resolved not in added and (ROOT/resolved).exists():
                r=subprocess.run(['git','-C',str(ROOT),'add',resolved],capture_output=True,text=True)
                ok='OK' if r.returncode==0 else f'FAIL ({r.stderr.strip()})'
                print(f'  git add {resolved}  -> {ok}');added.add(resolved)
        print(f'\n[publish_health] {len(added)} file(s) staged. Review with `git status`, then commit.')
        return 0 if len(added)==len(set(g[3] for g in ghosts)) else 1
    if ghosts:print('\n[publish_health] Run with --fix to auto-stage these files, or delete the imports if the modules are intentionally private.')
    return 1
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--fix',action='store_true',help='Stage any ghost-imported file that exists locally (`git add`)')
    args=ap.parse_args()
    sys.exit(scan(do_fix=args.fix))
