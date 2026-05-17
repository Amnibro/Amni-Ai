"""Reffelt runtime fetcher — first-launch installer for the encrypted GF(17) engine.
The public GitHub repo intentionally omits the Reffelt internals (amni.compute, amni.core,
amni.training, amni.model, amni_kernels). On first launch, this module fetches an encrypted
blob from amni-scient.com that contains the compiled Rust kernels + Python decode/encode
helpers. The blob is validated by signature, decrypted, and installed into ~/.amni_runtime/.
Without the blob, importing inference will raise RuntimeNotReadyError with a fix-up command."""
import os,sys,hashlib,json
from pathlib import Path
from typing import Optional,Dict,Any
REMOTE_BLOB_URL=os.environ.get('AMNI_RUNTIME_URL','https://amni-scient.com/adam/runtime/latest.reffelt.enc')
REMOTE_SIG_URL=os.environ.get('AMNI_RUNTIME_SIG_URL','https://amni-scient.com/adam/runtime/latest.reffelt.sig')
RUNTIME_DIR=Path(os.environ.get('AMNI_RUNTIME_DIR',str(Path.home()/'.amni_runtime')))
EXPECTED_VERSION=os.environ.get('AMNI_RUNTIME_VERSION','v6.8')
class RuntimeNotReadyError(RuntimeError):
    """Raised when Adam's Reffelt runtime blob has not been fetched + installed."""
def _blob_path()->Path:return RUNTIME_DIR/f'{EXPECTED_VERSION}.reffelt.enc'
def _sig_path()->Path:return RUNTIME_DIR/f'{EXPECTED_VERSION}.reffelt.sig'
def _manifest_path()->Path:return RUNTIME_DIR/'manifest.json'
def is_ready()->bool:
    return _blob_path().exists() and _blob_path().stat().st_size>0 and _manifest_path().exists()
def status()->Dict[str,Any]:
    out={'ready':is_ready(),'blob_path':str(_blob_path()),'sig_path':str(_sig_path()),'manifest_path':str(_manifest_path()),'remote_url':REMOTE_BLOB_URL,'expected_version':EXPECTED_VERSION,'runtime_dir':str(RUNTIME_DIR)}
    if _manifest_path().exists():
        try:out['manifest']=json.loads(_manifest_path().read_text())
        except Exception:pass
    return out
def fetch(license_key:Optional[str]=None,force:bool=False,verbose:bool=True)->Dict[str,Any]:
    """Download + verify + install the Reffelt runtime blob. license_key is required by
    the server to enforce CC BY-NC distribution; non-commercial users can request one free."""
    if is_ready() and not force:
        if verbose:print(f'[runtime] already installed at {_blob_path()}',flush=True)
        return status()
    raise NotImplementedError('Runtime fetcher pipeline not yet wired. Build steps (Anthony to do): (1) compile amni_kernels Rust to .pyd (2) bundle amni/compute/* + amni/core/* + amni/training/* + amni/model/* as tarball (3) age-encrypt with public key (4) host as https://amni-scient.com/adam/runtime/<version>.reffelt.enc + .sig. For now, run from a private full clone with Reffelt source present.')
def load()->None:
    """Decrypt the installed blob in-memory and inject Reffelt modules into sys.modules.
    Idempotent — calling twice is a no-op."""
    if 'amni.compute' in sys.modules:return
    if not is_ready():raise RuntimeNotReadyError(f'Reffelt runtime not installed. Run:\n  python -c "from amni.runtime import fetch; fetch(license_key=\'YOUR_KEY\')"\n\nFree non-commercial license keys at https://amni-scient.com/adam/get-key')
    raise NotImplementedError('Runtime loader stub — public skeleton repo. The actual decrypt + sys.modules injection ships with the runtime blob itself, never with the public source.')
def require_runtime():
    """Decorator-style guard. Call at top of any public function that depends on Reffelt internals."""
    if 'amni.compute' not in sys.modules:
        try:load()
        except NotImplementedError as e:raise RuntimeNotReadyError(str(e))
