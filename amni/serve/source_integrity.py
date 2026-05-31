"""source_integrity — tamper-DETECTION for the law + security-enforcement SOURCE files. The S45/S46 write-protection
gates only block the *skill* write path (file_write/code_edit/code_diff); a raw filesystem edit, a different write
path, or an in-place patch bypasses them. This records a sha256 manifest of the protected sources and verifies it, so
ANY change between recordings is detected — extending `learning/integrity.py` (weight tensors) to the source. Records
on first run, re-record after a legitimate edit. Wire `verify_source_integrity()` into a startup/periodic check; a
mismatch is a security event (or a not-yet-re-recorded legit edit). Cheap: sha256 of ~6 small files, no model."""
import hashlib,json,time
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
_PROTECTED=('amni/inference/asimov.py','amni/a1/asimov.py','amni/learning/integrity.py','amni/serve/code_safety.py','amni/serve/federated.py','amni/serve/pii_egress.py')
def _manifest_path():
    p=_ROOT/'data'/'source_integrity.json';p.parent.mkdir(parents=True,exist_ok=True);return p
def _sha(rel):
    f=_ROOT/rel
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
def record_source_integrity(path=None,files=_PROTECTED):
    entries={rel:s for rel in files for s in (_sha(rel),) if s is not None}
    out={'schema':1,'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'root':str(_ROOT),'n':len(entries),'entries':entries}
    (Path(path) if path else _manifest_path()).write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
def verify_source_integrity(path=None,auto_record=True):
    mp=Path(path) if path else _manifest_path()
    if not mp.exists():
        if auto_record:rec=record_source_integrity(mp);return {'ok':True,'recorded':True,'n':rec['n'],'mismatches':[],'missing':[]}
        return {'ok':False,'reason':'no manifest','mismatches':[],'missing':[]}
    rec=json.loads(mp.read_text(encoding='utf-8'))
    mismatches=[];missing=[]
    for rel,sha in (rec.get('entries') or {}).items():
        cur=_sha(rel)
        if cur is None:missing.append(rel)
        elif cur!=sha:mismatches.append(rel)
    ok=not mismatches and not missing
    return {'ok':ok,'n':len(rec.get('entries') or {}),'mismatches':mismatches,'missing':missing,'recorded_at':rec.get('recorded_at'),'verified_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
