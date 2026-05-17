"""Bake integrity protection for the foundational tiers.

The AsimovLayer (amni/a1/asimov.py) sha256-checks the AXIOM TEXT at runtime via
`_AXIOM_INTEGRITY`. This module extends the same protection to the WEIGHTS of the
asimov / foundation / ascension tiers — so direct tampering with `tensors/*.gf17`
files (bypassing LearningWriter, e.g. via filesystem edit or malicious patch)
gets caught.

Two operations:

    record_immutable_integrity(bake_dir, out_path=None)
        Scan every tier-protected tensor (asimov, foundation, ascension), compute
        sha256 of its .gf17 file content, write to <bake_dir>/tensors_integrity.json
        with metadata (tensor_name, tier, gf17_path, sha256, recorded_at).

    verify_immutable_integrity(bake_dir, manifest_path=None)
        Recompute hashes for every tensor in the integrity manifest, compare to
        recorded hash. Returns dict {ok: bool, n_total: N, n_passed: M, mismatches: [...]}.
        A mismatch indicates either:
            (a) Recorded after legitimate change without re-recording (workflow bug)
            (b) Direct file tampering bypassing LearningWriter (security event)
            (c) Bit-rot on disk (storage failure)

Workflow:
    1. After baking: `record_immutable_integrity` → tensors_integrity.json
    2. On every `LearningWriter.__init__` (optional, env-gated): `verify_immutable_integrity`
    3. Standalone: `adam1 verify_integrity --bake bakes/...`

If integrity verification fails, raise IntegrityError. The asimov axiom text already
fail-fast asserts at AsimovLayer.__init__; this gives the same fail-fast for tier weights.
"""
import hashlib,json,time
from pathlib import Path
from typing import Dict,List,Optional
class IntegrityError(Exception):pass
_PROTECTED_TIERS=('asimov','commandments','ascension','foundation')
def _sha256_file(p,chunk=1<<20):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b:break
            h.update(b)
    return h.hexdigest()
def record_immutable_integrity(bake_dir,out_path=None,protected_tiers=_PROTECTED_TIERS):
    bake=Path(bake_dir)
    manifest=json.loads((bake/'manifest.json').read_text(encoding='utf-8'))
    out_path=Path(out_path) if out_path else bake/'tensors_integrity.json'
    entries=[]
    for name,info in sorted(manifest['tensors'].items()):
        tier=info.get('tier')
        if tier is None:tier='asimov' if info.get('asimov_immutable') else 'wisdom'
        if tier not in protected_tiers:continue
        gf17_path=info.get('gf17_path')
        if not gf17_path:continue
        full=bake/gf17_path
        if not full.exists():continue
        sha=_sha256_file(full)
        entries.append({'tensor_name':name,'tier':tier,'gf17_path':gf17_path,'sha256':sha,'gf17_bytes':int(info.get('gf17_bytes',0))})
    out={'schema_version':1,'bake':str(bake.resolve()),'protected_tiers':list(protected_tiers),'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'n_entries':len(entries),'entries':entries}
    out_path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
def verify_immutable_integrity(bake_dir,manifest_path=None,raise_on_mismatch=True):
    bake=Path(bake_dir)
    manifest_path=Path(manifest_path) if manifest_path else bake/'tensors_integrity.json'
    if not manifest_path.exists():
        if raise_on_mismatch:raise IntegrityError(f'no integrity manifest at {manifest_path} — run record_immutable_integrity first')
        return {'ok':False,'reason':'no manifest','n_total':0,'n_passed':0,'mismatches':[]}
    rec=json.loads(manifest_path.read_text(encoding='utf-8'))
    mismatches=[];missing=[];n_passed=0
    for e in rec['entries']:
        full=bake/e['gf17_path']
        if not full.exists():
            missing.append({'tensor_name':e['tensor_name'],'gf17_path':e['gf17_path'],'reason':'file missing'})
            continue
        actual=_sha256_file(full)
        if actual==e['sha256']:n_passed+=1
        else:mismatches.append({'tensor_name':e['tensor_name'],'tier':e['tier'],'gf17_path':e['gf17_path'],'expected_sha256':e['sha256'],'actual_sha256':actual})
    n_total=len(rec['entries'])
    ok=(len(mismatches)==0 and len(missing)==0)
    result={'ok':ok,'n_total':n_total,'n_passed':n_passed,'n_mismatches':len(mismatches),'n_missing':len(missing),'mismatches':mismatches,'missing':missing,'recorded_at':rec.get('recorded_at'),'verified_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if not ok and raise_on_mismatch:
        first_few=mismatches[:3]+missing[:3]
        raise IntegrityError(f'integrity check FAILED: {len(mismatches)} mismatches, {len(missing)} missing. First: {first_few}')
    return result
