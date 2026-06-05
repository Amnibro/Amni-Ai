"""integrity_gate — launch-time tamper gate for Adam's immutable core (the 5 Laws + safety/integrity code).
DESIGN (the maintainer 2026-06-04): the unadulterated core + its functions are sealed in amni-crypt as TWO files
held SERVER-SIDE (example.com). At launch Adam fingerprints its LIVE core and compares to the
server-held canonical (polled); on a CONFIRMED MISMATCH it HALTS — it will not run a mutilated Adam.
amni-crypt is the maintainer's proprietary format (the obscurity barrier — "no one else understands it"); this
module is the framework + a pluggable `_amni_crypt_open` hook to wire it (kept OUT of the public repo).
OPT-IN via AMNI_INTEGRITY_GATE=1 so default behavior is unchanged until the server canonical + amni-crypt
are in place. Fails CLOSED on a confirmed mismatch, fails OPEN if the canonical is merely unreachable
(so offline use still works) unless AMNI_INTEGRITY_REQUIRE_CANONICAL=1. Re-seal after a legitimate law edit.
This EXTENDS the always-on AsimovLayer GF17-Merkle self-check + source_integrity sha256 manifest + ptex_scan."""
import os,sys,hashlib,json
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
_CORE_FILES=('amni/a1/asimov.py','amni/inference/asimov.py','amni/learning/integrity.py','amni/serve/code_safety.py','amni/serve/pii_egress.py','amni/serve/source_integrity.py','amni/serve/ptex_scan.py','amni/serve/integrity_gate.py')
def _sha(rel):
    f=_ROOT/rel
    try:return hashlib.sha256(f.read_bytes()).hexdigest()
    except Exception:return None
def live_core_fingerprint():
    parts={rel:_sha(rel) for rel in _CORE_FILES}
    try:
        from amni.a1 import asimov as _az
        for nm in ('_AXIOMS','_HARM_KEYWORDS','_JAILBREAK_PATTERNS','_DIVINE_DENIAL','_COMMANDMENT_VIOLATIONS','_EXPLOIT_KEYWORDS'):
            v=getattr(_az,nm,None)
            if v is not None:parts['az:'+nm]=hashlib.sha256(repr(sorted(v) if isinstance(v,(set,frozenset)) else v).encode()).hexdigest()
    except Exception as e:parts['az:ERROR']=str(e)[:80]
    blob=json.dumps(parts,sort_keys=True)
    return {'files':parts,'root':hashlib.sha256(blob.encode()).hexdigest()}
def _amni_crypt_open(blob:bytes)->bytes:
    """Decrypt an amni-crypted (.acrypt) canonical via the PRIVATE amni-crypt container — never bundled in this public repo.
    Resolves `import amni_crypt`, else AMNI_CRYPT_PATH (path to acrypt_container.py). Key/passphrase from env."""
    key=os.environ.get('AMNI_CRYPT_KEY','');key=key.encode() if key else None
    passphrase=os.environ.get('AMNI_CRYPT_PASSPHRASE','')
    try:import amni_crypt as _ac
    except Exception:
        _p=os.environ.get('AMNI_CRYPT_PATH','')
        if not _p:raise RuntimeError('amni-crypt unavailable (pip install amni_crypt, or set AMNI_CRYPT_PATH to acrypt_container.py)')
        import importlib.util as _ilu
        _spec=_ilu.spec_from_file_location('amni_crypt_ext',_p);_ac=_ilu.module_from_spec(_spec);_spec.loader.exec_module(_ac)
    res=_ac.unpack_acrypt(blob,key=key,passphrase=passphrase)
    if isinstance(res,(tuple,list)) and res:res=res[0]
    return res if isinstance(res,(bytes,bytearray)) else (res.encode() if isinstance(res,str) else bytes(res))
def fetch_canonical():
    """Poll the server for the amni-crypted canonical (core.amc + functions.amc), decrypt via amni-crypt, return the expected root."""
    base=os.environ.get('AMNI_INTEGRITY_SERVER','https://example.com/amni-ai/seal').rstrip('/')
    import urllib.request
    parts={}
    for name in ('core','functions'):
        try:
            with urllib.request.urlopen(base+'/'+name+'.amc',timeout=8) as r:parts[name]=_amni_crypt_open(r.read())
        except Exception as e:return {'error':f'{name}: {type(e).__name__}: {str(e)[:80]}'}
    try:
        core=json.loads(parts['core'].decode('utf-8'))
        return {'root':core.get('root'),'sealed_at':core.get('sealed_at')}
    except Exception as e:return {'error':f'parse: {e}'}
def verify():
    live=live_core_fingerprint();canon=fetch_canonical()
    if canon.get('error'):return {'ok':None,'reason':'canonical unavailable: '+canon['error'],'live_root':live['root']}
    ok=bool(canon.get('root')) and canon.get('root')==live['root']
    return {'ok':ok,'live_root':live['root'],'canonical_root':canon.get('root'),'sealed_at':canon.get('sealed_at')}
def _halt(msg):
    sys.stderr.write('\n[SECURITY] Adam immutable-core integrity FAILED — '+msg+'\nRefusing to run a mutilated Adam. (Re-seal via the sealing tool after a legitimate law edit.)\n')
    sys.stderr.flush();os._exit(99)
def gate(adam=None):
    """Launch gate. OPT-IN via AMNI_INTEGRITY_GATE=1. Confirmed core mismatch -> HALT. Also runs the PTEX bad-actor scan (HALT on high-severity if strict)."""
    if os.environ.get('AMNI_INTEGRITY_GATE','0')!='1':return {'enabled':False}
    r=verify()
    if r.get('ok') is False:_halt(f'live core root {r.get("live_root","?")[:12]} != sealed canonical {str(r.get("canonical_root"))[:12]}.')
    if r.get('ok') is None and os.environ.get('AMNI_INTEGRITY_REQUIRE_CANONICAL','0')=='1':_halt('sealed canonical unreachable and AMNI_INTEGRITY_REQUIRE_CANONICAL=1 ('+str(r.get('reason'))+').')
    if adam is not None:
        try:
            from amni.serve.ptex_scan import scan_adam
            ps=scan_adam(adam);r['ptex']={'lessons':ps.get('lessons_scanned'),'findings':len(ps.get('findings',[])),'high':ps.get('n_high')}
            if ps.get('n_high') and os.environ.get('AMNI_INTEGRITY_STRICT','0')=='1':_halt(f'{ps["n_high"]} high-severity injection payload(s) found in the lesson/PTEX substrate.')
        except Exception as e:r['ptex']={'error':str(e)[:80]}
    return r
if __name__=='__main__':
    print(json.dumps({'live':live_core_fingerprint()['root'],'verify':verify()},indent=2)[:2000])
