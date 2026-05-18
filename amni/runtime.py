"""Reffelt runtime — iter29 public-source compatibility shim.
Pre-iter29 (iter21 era) this module downloaded an encrypted .reffelt.enc blob from
example.com on first launch. Under iter29 the full Reffelt source ships in the
public git clone under CC BY-NC 4.0, so there is nothing to fetch — the public clone
IS the runtime. This module is kept so that older error messages and tutorials that
say `from amni.runtime import fetch; fetch(license_key='...')` still work and report
honestly instead of raising NotImplementedError.

Compatibility surface (all callable but no-op when public source is importable):
  is_ready()        -> bool   True when amni.compute.reffelt4 is importable.
  status()          -> dict   Inspectable state for diagnostics.
  fetch(...)        -> dict   Returns status() on success, raises RuntimeNotReadyError with
                              the underlying ImportError if the public source is broken.
  load()            -> None   No-op when modules are importable. Raises with diagnostic if not.
  require_runtime() -> None   Guard for callers that depend on Reffelt internals."""
import os,sys,json
from pathlib import Path
from typing import Optional,Dict,Any
RUNTIME_DIR=Path(os.environ.get('AMNI_RUNTIME_DIR',str(Path.home()/'.amni_runtime')))
EXPECTED_VERSION=os.environ.get('AMNI_RUNTIME_VERSION','v6.8')
_PROBE_MODULES=('amni.compute.reffelt4','amni.compute.ternary5','amni.inference.streaming_linear')
class RuntimeNotReadyError(RuntimeError):
    """Raised when the Reffelt public-source modules cannot be imported."""
def _probe_imports()->Dict[str,Any]:
    out={'ok':True,'modules':{},'first_error':None}
    for m in _PROBE_MODULES:
        try:__import__(m);out['modules'][m]='ok'
        except Exception as e:out['modules'][m]=f'{type(e).__name__}: {e}';out['ok']=False;out['first_error']=out['first_error'] or {'module':m,'type':type(e).__name__,'msg':str(e)}
    return out
def is_ready()->bool:return _probe_imports()['ok']
def status()->Dict[str,Any]:
    p=_probe_imports()
    return {'ready':p['ok'],'mode':'public-source-iter29','expected_version':EXPECTED_VERSION,'runtime_dir':str(RUNTIME_DIR),'python':sys.version.split()[0],'modules':p['modules'],'first_error':p['first_error'],'note':'Under iter29 the public clone IS the runtime — no blob fetch is needed. If ready=False, the Reffelt source modules failed to import; see first_error.'}
def fetch(license_key:Optional[str]=None,force:bool=False,verbose:bool=True)->Dict[str,Any]:
    """No-op under iter29 — returns success when the public-source modules import cleanly.
    Kept for compatibility with older docs and error messages that say `from amni.runtime import fetch; fetch(...)`."""
    s=status()
    if s['ready']:
        if verbose:
            print('[runtime] Adam runtime is ready (iter29 public-source mode — no fetch needed).',flush=True)
            print(f'[runtime] python={s["python"]}  expected_version={s["expected_version"]}',flush=True)
            for m,v in s['modules'].items():print(f'[runtime]   {m}: {v}',flush=True)
        return s
    e=s['first_error']
    raise RuntimeNotReadyError(f'Reffelt public-source modules failed to import (likely cause: wrong Python version for the prebuilt amni_kernels .pyd, or a missing dep).\n  First error: {e["module"]} -> {e["type"]}: {e["msg"]}\n  Fix:\n    1. Re-run `python install.py` from the repo root to repair the venv (auto-detects GPU vendor).\n    2. If the error is from `amni_kernels`, rebuild it with `cd amni_kernels && pip install maturin && maturin develop --release`.\n    3. If still stuck, file an issue at https://github.com/Amnibro/Amni-Ai/issues with the output of `python -c "from amni.runtime import status;import json;print(json.dumps(status(),indent=2))"`.')
def load()->None:
    """No-op under iter29 — the public source modules import normally via Python's import system.
    Raises RuntimeNotReadyError with diagnostic when the public-source modules don't import."""
    s=status()
    if s['ready']:return
    e=s['first_error']
    raise RuntimeNotReadyError(f'Reffelt source failed to import: {e["module"]} -> {e["type"]}: {e["msg"]}. Run `python -c "from amni.runtime import fetch; fetch()"` for full diagnostic + fix steps.')
def require_runtime():
    """Guard for callers that depend on Reffelt internals. Raises if public source not importable."""
    s=status()
    if not s['ready']:e=s['first_error'];raise RuntimeNotReadyError(f'{e["module"]}: {e["type"]}: {e["msg"]}')
